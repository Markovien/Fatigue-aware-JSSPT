"""CP-SAT model of the (fatigue-aware) JSSPT -- revised version.

Changes w.r.t. the submitted cp_sat_solver_jsspt_hf.py
  * hard fatigue bound F(C_ij) <= F_max (cfg.hard_bound); legacy start-threshold kept
  * rest-duration table defined for every level above F_min (bug fix)
  * idle recovery acts on waiting time without rest; a (triggered) rest starts right
    after the machine predecessor (dominant position).  The submitted code decayed
    fatigue over the whole gap *including* the rest and computed the rest duration
    from that over-decayed level (optimistic; it only matters when rests occur
    together with idle recovery)
  * rounding mode "nearest" (legacy) or "conservative" (upper-bounding durations/fatigue)
  * redundant NoOverlap constraints, robot symmetry breaking, optional solution hint
  * solver statistics: status, objective, best bound, gap, wall time, seed
  * full schedule export (start/end, F^pre, F^S, F^C, rests, robots) for continuous replay
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

from ortools.sat.python import cp_model

from .fatigue import FatigueConfig, FatigueTables


def _transport_tasks(inst):
    """Transport task list: (job, op index j | n_i for the final return, pickup, drop, dur)."""
    sig, tasks, of_op, of_ret = inst["sigma"], [], {}, {}
    for i, job in enumerate(inst["jobs"]):
        prev = 0
        for j, (k, _) in enumerate(job):
            of_op[(i, j)] = len(tasks)
            tasks.append((i, j, prev, k, sig[prev][k]))
            prev = k
        if inst["last_ttask"]:
            of_ret[i] = len(tasks)
            tasks.append((i, len(job), prev, 0, sig[prev][0]))
    return tasks, of_op, of_ret


def solve(inst: dict, cfg: FatigueConfig, time_limit: float = 3600.0, num_workers: int = 8,
          seed: int = 0, redundant: bool = True, symmetry: bool = True,
          hint: Optional[dict] = None, ub: Optional[int] = None, log: bool = False) -> dict:
    t_build = time.time()
    tabs = FatigueTables.build(inst, cfg)
    bad = tabs.check_feasible(inst)
    if bad:
        return dict(status="INSTANCE_INFEASIBLE", cmax=None, bound=None, gap=None,
                    wall_time=0.0, message="; ".join(bad[:3]))
    M, R, L = inst["M"], inst["R"], cfg.L
    jobs, sig = inst["jobs"], inst["sigma"]
    use_f = cfg.use_fatigue
    ops = [(i, j) for i, job in enumerate(jobs) for j in range(len(job))]
    mach = {o: jobs[o[0]][o[1]][0] for o in ops}
    tau = {o: jobs[o[0]][o[1]][1] for o in ops}
    tables = {o: tabs.op_tables(mach[o], tau[o], inst["kappa"][o]) for o in ops}
    tasks, of_op, of_ret = _transport_tasks(inst)
    T = len(tasks)
    max_sig = max(max(r) for r in sig)
    rmax = {k: max(tabs.rest[k]) for k in range(1, M + 1)}
    horizon = (sum(max(tables[o][0]) for o in ops) + (sum(rmax[mach[o]] for o in ops) if use_f else 0)
               + 2 * T * max_sig + 10)
    if ub is not None:
        horizon = min(horizon, int(ub))

    md = cp_model.CpModel()
    S, C, P, I = {}, {}, {}, {}
    fS, fC, fpre, rest, drest, idle = {}, {}, {}, {}, {}, {}
    for o in ops:
        Ptab, Gtab = tables[o]
        S[o] = md.NewIntVar(0, horizon, f"S{o}")
        C[o] = md.NewIntVar(0, horizon, f"C{o}")
        if use_f:
            fS[o] = md.NewIntVar(0, L, f"fS{o}")
            fC[o] = md.NewIntVar(0, L, f"fC{o}")
            P[o] = md.NewIntVar(min(Ptab), max(Ptab), f"P{o}")
            md.AddElement(fS[o], Ptab, P[o])
            md.AddElement(fS[o], Gtab, fC[o])
            if cfg.hard_bound:
                md.Add(fC[o] <= tabs.fmax[mach[o]])
        else:
            P[o] = md.NewConstant(tau[o])
        md.Add(C[o] == S[o] + P[o])
        I[o] = md.NewIntervalVar(S[o], P[o], C[o], f"I{o}")

    # ------------------------------------------------------------------ machines
    on_m = {k: [o for o in ops if mach[o] == k] for k in range(1, M + 1)}
    arcs = {}
    SRC = (-1, -1)
    for k, mops in on_m.items():
        if not mops:
            continue
        if not use_f:
            md.AddNoOverlap([I[o] for o in mops])
            continue
        if redundant:
            md.AddNoOverlap([I[o] for o in mops])
        pred = {(SRC, b): md.NewBoolVar(f"src{b}") for b in mops}
        arcs[k] = pred
        for a in mops:
            for b in mops:
                if a != b:
                    pred[(a, b)] = md.NewBoolVar(f"p{a}{b}")
        sink = {a: md.NewBoolVar(f"snk{a}") for a in mops}
        md.AddExactlyOne([pred[(SRC, b)] for b in mops])
        md.AddExactlyOne(list(sink.values()))
        for b in mops:
            md.AddExactlyOne([pred[(SRC, b)]] + [pred[(a, b)] for a in mops if a != b])
        for a in mops:
            md.AddExactlyOne([pred[(a, b)] for b in mops if b != a] + [sink[a]])
        Rtab = tabs.rest[k]
        idle_on = cfg.recover_on_idle and tabs.mu_scaled[k] > 0
        for b in mops:
            Ptab, Gtab = tables[b]
            pend = md.NewIntVar(0, horizon, f"pend{b}")
            pfend = md.NewIntVar(0, L, f"pfend{b}")
            md.Add(pend == 0).OnlyEnforceIf(pred[(SRC, b)])
            md.Add(pfend == tabs.f0[k]).OnlyEnforceIf(pred[(SRC, b)])
            for a in mops:
                if a != b:
                    md.Add(pend == C[a]).OnlyEnforceIf(pred[(a, b)])
                    md.Add(pfend == fC[a]).OnlyEnforceIf(pred[(a, b)])
            rest[b] = md.NewBoolVar(f"rest{b}")
            drest[b] = md.NewIntVar(0, max(Rtab), f"d{b}")
            thr = tabs.fmax[k] + 1 if cfg.hard_bound else L - cfg.eps_exhaust_int

            def trig_lit(fvar, tag):
                ge = md.NewBoolVar(f"ge{tag}{b}")
                md.Add(fvar >= tabs.fmax[k]).OnlyEnforceIf(ge)
                md.Add(fvar <= tabs.fmax[k] - 1).OnlyEnforceIf(ge.Not())
                g = md.NewIntVar(0, L, f"g{tag}{b}")
                md.AddElement(fvar, Gtab, g)
                ov = md.NewBoolVar(f"ov{tag}{b}")
                md.Add(g >= thr).OnlyEnforceIf(ov)
                md.Add(g <= thr - 1).OnlyEnforceIf(ov.Not())
                tr = md.NewBoolVar(f"tr{tag}{b}")
                md.AddBoolOr([ge, ov]).OnlyEnforceIf(tr)
                md.AddBoolAnd([ge.Not(), ov.Not()]).OnlyEnforceIf(tr.Not())
                return tr

            # rest (if any) starts right after the machine predecessor: this dominates any
            # later rest position (more idle decay before a rest never shortens rest + idle
            # when mu_idle <= mu, and the rest ends at F_min anyway)
            trig_c = trig_lit(pfend, "c")              # trigger at predecessor completion
            md.AddImplication(rest[b], trig_c)        # rest only when triggered
            rreq = md.NewIntVar(0, max(Rtab), f"rreq{b}")
            md.AddElement(pfend, Rtab, rreq)
            md.Add(drest[b] == rreq).OnlyEnforceIf(rest[b])
            md.Add(drest[b] == 0).OnlyEnforceIf(rest[b].Not())
            md.Add(S[b] >= pend + drest[b])
            if idle_on:
                # waiting without rest: idle decay over the whole gap S_b - C_a
                gap = md.NewIntVar(0, horizon, f"gap{b}")
                md.Add(gap == S[b] - pend)
                idle[b] = gap
                lnv = md.NewIntVar(min(tabs.ln_table), 0, f"ln{b}")
                md.AddElement(pfend, tabs.ln_table, lnv)
                idx = md.NewIntVar(-horizon * tabs.mu_scaled[k] - tabs.OFFSET, tabs.OFFSET, f"ix{b}")
                md.Add(idx == lnv - gap * tabs.mu_scaled[k] + tabs.OFFSET)
                idxc = md.NewIntVar(0, len(tabs.exp_table) - 1, f"ixc{b}")
                md.AddMaxEquality(idxc, [idx, md.NewConstant(0)])
                val = md.NewIntVar(0, L, f"dec{b}")
                md.AddElement(idxc, tabs.exp_table, val)
                vmin = md.NewIntVar(0, L, f"vmin{b}")
                md.AddMinEquality(vmin, [val, pfend])          # decay never increases fatigue
                fdec = md.NewIntVar(0, L, f"fdec{b}")
                md.AddMaxEquality(fdec, [vmin, md.NewConstant(tabs.fmin[k])])
                trig_w = trig_lit(fdec, "w")
                md.AddImplication(rest[b].Not(), trig_w.Not())   # no rest => safe after waiting
                fpre[b] = md.NewIntVar(0, L, f"fpre{b}")
                md.Add(fpre[b] == pfend).OnlyEnforceIf(rest[b])
                md.Add(fpre[b] == fdec).OnlyEnforceIf(rest[b].Not())
            else:
                fpre[b] = pfend
                md.AddImplication(trig_c, rest[b])               # rest iff triggered
            md.Add(fS[b] == tabs.fmin[k]).OnlyEnforceIf(rest[b])
            md.Add(fS[b] == fpre[b]).OnlyEnforceIf(rest[b].Not())

    # ------------------------------------------------------------------ transport
    tS, tE, asg = [], [], []
    for t, (i, j, pu, dr, dur) in enumerate(tasks):
        s = md.NewIntVar(0, horizon, f"TS{t}")
        e = md.NewIntVar(0, horizon, f"TE{t}")
        md.Add(e == s + dur)
        tS.append(s)
        tE.append(e)
        asg.append([md.NewBoolVar(f"a{r}_{t}") for r in range(R)])
        md.AddExactlyOne(asg[t])
        if j < len(jobs[i]):
            md.Add(S[(i, j)] >= e)
        if j > 0:
            md.Add(s >= C[(i, j - 1)])
        for r in range(R):   # robots start at Z
            md.Add(s >= sig[0][pu]).OnlyEnforceIf(asg[t][r])
    if symmetry and R > 1 and T > 0:
        md.Add(asg[0][0] == 1)
    for r in range(R):
        if redundant:
            md.AddNoOverlap([md.NewOptionalIntervalVar(tS[t], tasks[t][4], tE[t], asg[t][r], f"TI{r}_{t}")
                             for t in range(T)])
        for t1 in range(T):
            for t2 in range(t1 + 1, T):
                z = md.NewBoolVar("")
                md.AddBoolAnd([asg[t1][r], asg[t2][r]]).OnlyEnforceIf(z)
                md.AddBoolOr([asg[t1][r].Not(), asg[t2][r].Not()]).OnlyEnforceIf(z.Not())
                y = md.NewBoolVar("")
                e12 = sig[tasks[t1][3]][tasks[t2][2]]
                e21 = sig[tasks[t2][3]][tasks[t1][2]]
                md.Add(tS[t2] >= tE[t1] + e12).OnlyEnforceIf([z, y])
                md.Add(tS[t1] >= tE[t2] + e21).OnlyEnforceIf([z, y.Not()])

    cmax = md.NewIntVar(0, horizon, "Cmax")
    ends = [tE[of_ret[i]] for i in range(len(jobs))] if inst["last_ttask"] else \
        [C[(i, len(job) - 1)] for i, job in enumerate(jobs)]
    md.AddMaxEquality(cmax, ends)
    md.Minimize(cmax)

    hint_status = None
    if hint:   # a feasible schedule (e.g. from the decoder): complete it into a full hint
        fixes = []
        for o, rec in hint["ops"].items():
            o = tuple(o)
            if o in S:
                fixes.append((S[o], int(rec["start"])))
                if o in rest:
                    fixes.append((rest[o], int(bool(rec.get("rest")))))
        for k, seq in hint["machine_seq"].items():
            if int(k) in arcs:
                seq = [tuple(o) for o in seq]
                prevs = {b_: (seq[q - 1] if q > 0 else SRC) for q, b_ in enumerate(seq)}
                for (a_, b_), lit in arcs[int(k)].items():
                    fixes.append((lit, int(prevs.get(b_) == a_)))
        relabel = list(range(R))          # symmetry breaking: task 0 on robot 0
        r0 = next((d["robot"] for d in hint["transports"] if d["task"] == 0), 0)
        if symmetry and r0 != 0:
            relabel[0], relabel[r0] = r0, 0
        for d in hint["transports"]:
            t = d["task"]
            if t < T:
                fixes.append((tS[t], int(d["start"])))
                for rr in range(R):
                    fixes.append((asg[t][rr], int(rr == relabel[d["robot"]])))
        m2 = md.clone()
        for v, val in fixes:
            m2.Add(m2.get_int_var_from_proto_index(v.Index()) == val)
        s2 = cp_model.CpSolver()
        s2.parameters.max_time_in_seconds = 30.0
        s2.parameters.num_workers = 1
        st2 = s2.Solve(m2)
        if st2 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            sol = s2.ResponseProto().solution
            hp = md.Proto().solution_hint
            hp.vars.extend(range(len(sol)))
            hp.values.extend(sol)
            hint_status = "complete"
        else:   # fall back to a partial hint
            for v, val in fixes:
                md.AddHint(v, val)
            hint_status = "partial:" + s2.StatusName(st2)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_workers = int(num_workers)
    solver.parameters.random_seed = int(seed)
    solver.parameters.log_search_progress = bool(log)
    build_time = time.time() - t_build
    t0 = time.time()
    st = solver.Solve(md)
    wall = time.time() - t0
    name = solver.StatusName(st)
    res = dict(status=name, wall_time=round(wall, 3), build_time=round(build_time, 3),
               user_time=round(solver.UserTime(), 3), conflicts=solver.NumConflicts(),
               branches=solver.NumBranches(), horizon=horizon, seed=seed,
               num_workers=num_workers, time_limit=time_limit, hint=hint_status)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        res.update(cmax=None, bound=None, gap=None)
        return res
    cm = int(solver.Value(cmax))
    bd = int(round(solver.BestObjectiveBound()))
    res.update(cmax=cm, bound=bd, gap=round((cm - bd) / cm, 6) if cm else 0.0)

    # ------------------------------------------------------------------ export
    sched = dict(cmax=cm, ops={}, transports=[], machine_seq={}, robot_seq={})
    for o in ops:
        rec = dict(machine=mach[o], start=solver.Value(S[o]), end=solver.Value(C[o]), dur=solver.Value(P[o]))
        if use_f:
            rec.update(f_pre=solver.Value(fpre[o]) / L, f_start=solver.Value(fS[o]) / L,
                       f_end=solver.Value(fC[o]) / L, rest=bool(solver.Value(rest[o])),
                       rest_dur=solver.Value(drest[o]), idle=solver.Value(idle[o]) if o in idle else None)
        sched["ops"][o] = rec
    for k, mops in on_m.items():
        sched["machine_seq"][k] = sorted(mops, key=lambda o: sched["ops"][o]["start"])
    for t, (i, j, pu, dr, dur) in enumerate(tasks):
        r = next(rr for rr in range(R) if solver.Value(asg[t][rr]))
        sched["transports"].append(dict(task=t, job=i, j=j, pickup=pu, drop=dr, robot=r,
                                        start=solver.Value(tS[t]), end=solver.Value(tE[t])))
    for r in range(R):
        sched["robot_seq"][r] = [d["task"] for d in sorted(
            (d for d in sched["transports"] if d["robot"] == r), key=lambda d: d["start"])]
    res["schedule"] = sched
    return res
