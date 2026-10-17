"""Schedule validation, fatigue-aware simulation (SIM1/SIM2) and continuous replay.

replay() re-times a schedule with FIXED machine sequences, robot assignments and
robot sequences.  Activities are processed in topological order of the
precedence graph (job route T_i1 -> O_i1 -> ... -> T_i,n+1, machine order,
robot order); a delay therefore propagates to the job successors, to the
following operations on the same machine and to the following tasks of the
same robot (including the empty move to the next pickup).

  durations   "discrete"   : lookup tables of the CP-SAT model (integer time)
              "continuous" : exact laws, real-valued time (no rounding at all)
  rest_policy "planned"    : keep the rests of the schedule (continuous check of a plan)
              "none"       : never rest (SIM1, passive fatigue tracking)
              "reactive"   : rest iff triggered (SIM2, a-posteriori correction)
  rest_start  "after_pred" : a reactive rest starts right after the previous operation
              "at_need"    : it starts when the operation is due (planned start)
  timing      "keep"       : start >= planned start (right-shift only, as in the paper)
              "left"       : earliest start (left-shift allowed)
  nominal     True         : nominal durations (legacy SIM1 of the submitted code)
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Dict, List

from .cpsat_model import _transport_tasks
from .fatigue import (FatigueConfig, FatigueTables, accumulate_cont, idle_decay_cont, rest_cont,
                      tau_cont, remaining_workable_time)

TOL = 1e-9


def _graph(inst, sched):
    tasks, of_op, of_ret = _transport_tasks(inst)
    nodes, succ, indeg = [], defaultdict(list), defaultdict(int)

    def edge(a, b):
        succ[a].append(b)
        indeg[b] += 1

    for i, job in enumerate(inst["jobs"]):
        for j in range(len(job)):
            nodes += [("T", of_op[(i, j)]), ("O", (i, j))]
            edge(("T", of_op[(i, j)]), ("O", (i, j)))
            if j > 0:
                edge(("O", (i, j - 1)), ("T", of_op[(i, j)]))
        if inst["last_ttask"]:
            nodes.append(("T", of_ret[i]))
            edge(("O", (i, len(job) - 1)), ("T", of_ret[i]))
    mpred, rpred = {}, {}
    for k, seq in sched["machine_seq"].items():
        seq = [tuple(o) for o in seq]
        for a, b in zip(seq, seq[1:]):
            edge(("O", a), ("O", b))
        for idx, o in enumerate(seq):
            mpred[o] = seq[idx - 1] if idx > 0 else None
    for r, seq in sched["robot_seq"].items():
        for a, b in zip(seq, seq[1:]):
            edge(("T", a), ("T", b))
        for idx, t in enumerate(seq):
            rpred[t] = (int(r), seq[idx - 1] if idx > 0 else None)
    order, q = [], deque([n for n in nodes if indeg[n] == 0])
    while q:
        n = q.popleft()
        order.append(n)
        for m in succ[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                q.append(m)
    if len(order) != len(nodes):
        raise ValueError("cyclic precedence graph (inconsistent sequences)")
    return tasks, of_op, order, mpred, rpred


def replay(inst: dict, sched: dict, cfg: FatigueConfig, durations: str = "discrete",
           rest_policy: str = "planned", timing: str = "keep", rest_start: str = "after_pred",
           nominal: bool = False) -> dict:
    tabs = FatigueTables.build(inst, cfg)
    L = cfg.L
    tasks, of_op, order, mpred, rpred = _graph(inst, sched)
    fat = inst["fatigue"]
    sig = inst["sigma"]
    keep = timing == "keep"
    cont = durations == "continuous"
    plan_ops = {tuple(o) if not isinstance(o, tuple) else o: v for o, v in sched["ops"].items()}
    plan_t = {d["task"]: d for d in sched["transports"]}
    tS, tE, oS, oE, oF = {}, {}, {}, {}, {}
    out_ops = {}
    viol_end = viol_start = n_rest = 0
    rest_time = excess_time = 0.0
    Fpeak = 0.0
    for kind, ref in order:
        if kind == "T":
            i, j, pu, dr, dur = tasks[ref]
            r, prev = rpred[ref]
            rel = oE[(i, j - 1)] if j > 0 else 0.0
            ready = (tE[prev] + sig[tasks[prev][3]][pu]) if prev is not None else sig[0][pu]
            s = max(rel, ready)
            if keep:
                s = max(s, plan_t[ref]["start"])
            tS[ref], tE[ref] = s, s + dur
            continue
        o = ref
        i, j = o
        k, tau = inst["jobs"][i][j]
        kap = inst["kappa"][o]
        a = mpred[o]
        Ca = oE[a] if a is not None else 0.0
        if cont:
            FC = oF[a] if a is not None else fat["F0"][k]
        else:
            FC = oF[a] if a is not None else tabs.f0[k]
        arrival = tE[of_op[o]]
        plan = plan_ops.get(o, {})
        base = max(arrival, Ca)
        if keep and plan:
            base = max(base, plan["start"])
        Fmax, Fmin, lam, mu = fat["Fmax"][k], fat["Fmin"][k], fat["lam"][k], fat["mu"][k]
        mu_i = mu * cfg.idle_factor if cfg.recover_on_idle else 0.0

        def decay(F, gap):
            if cont:
                return idle_decay_cont(F, mu_i, gap, Fmin)
            return tabs.decay(k, F, int(round(gap)))

        def dur_and_end(FS):
            if cont:
                p = tau if (nominal or not cfg.use_fatigue) else tau_cont(tau, kap, FS)
                return p, accumulate_cont(FS, lam, p)
            P, G = tabs.op_tables(k, tau, kap)
            p = tau if nominal else P[FS]
            Fe = accumulate_cont(FS / L, lam, p)
            return p, (G[FS] if not nominal else int(round(L * Fe)))

        def trig(F):
            if not cont:
                return tabs.triggered(k, F, tabs.op_tables(k, tau, kap)[1])
            p, Fe = dur_and_end(F)
            Fv, Fev = (F, Fe) if cont else (F / L, Fe / L)
            if Fv >= Fmax - TOL:
                return True
            if cfg.hard_bound:
                return Fev > Fmax + (TOL if cont else 0.5 / L)
            return Fev >= 1.0 - (1e-3 if cont else cfg.eps_exhaust_int / L)

        if rest_policy == "planned":
            do_rest = bool(plan.get("rest", False))
        elif rest_policy == "none" or not cfg.use_fatigue:
            do_rest = False
        else:
            do_rest = trig(decay(FC, base - Ca))
        if do_rest:
            if rest_policy == "planned":
                rs = Ca
            elif rest_start == "at_need":
                rs = base if not keep else max(Ca, plan.get("start", base))
            else:
                rs = Ca
            Frs = decay(FC, rs - Ca)
            if cont:
                d = rest_cont(Frs, mu, Fmin)
            else:
                d = tabs.rest[k][Frs]
            s = max(arrival, rs + d)
            if keep and plan:
                s = max(s, plan["start"])
            FS = Fmin if cont else tabs.fmin[k]
            n_rest += 1
            rest_time += d
            Fpre = Frs
        else:
            s = base
            FS = decay(FC, s - Ca)
            d = 0
            Fpre = FS
        p, Fe = dur_and_end(FS)
        oS[o], oE[o], oF[o] = s, s + p, Fe
        FSv, Fev = (FS, Fe) if cont else (FS / L, Fe / L)
        Fpeak = max(Fpeak, Fev)
        if FSv >= Fmax - TOL:
            viol_start += 1
        if Fev > Fmax + (TOL if cont else 0.5 / L):
            viol_end += 1
            excess_time += p - remaining_workable_time(FSv, lam, Fmax)
        out_ops[o] = dict(machine=k, start=s, end=s + p, dur=p, f_pre=Fpre if cont else Fpre / L,
                          f_start=FSv, f_end=Fev, rest=do_rest, rest_dur=d)
    if inst["last_ttask"]:
        cmax = max(tE[t] for t in range(len(tasks)) if tasks[t][1] == len(inst["jobs"][tasks[t][0]]))
    else:
        cmax = max(oE.values())
    new = dict(cmax=cmax, ops=out_ops, machine_seq=sched["machine_seq"], robot_seq=sched["robot_seq"],
               transports=[dict(plan_t[t], start=tS[t], end=tE[t]) for t in sorted(tS)])
    return dict(cmax=cmax, viol_end=viol_end, viol_start=viol_start, peak_fatigue=round(Fpeak, 6),
                time_above_fmax=round(excess_time, 4), n_rest=n_rest, rest_time=round(rest_time, 4),
                schedule=new)


def validate(inst: dict, sched: dict, cfg: FatigueConfig) -> List[str]:
    """Check a (discrete) schedule against every constraint of the CP-SAT model."""
    tabs = FatigueTables.build(inst, cfg)
    L, sig, errs = cfg.L, inst["sigma"], []
    tasks, of_op, of_ret = _transport_tasks(inst)
    ops = {tuple(o): v for o, v in sched["ops"].items()}
    tr = {d["task"]: d for d in sched["transports"]}
    for t, (i, j, pu, dr, dur) in enumerate(tasks):
        d = tr[t]
        if d["end"] - d["start"] != dur:
            errs.append(f"T{t} duration")
        if j > 0 and d["start"] < ops[(i, j - 1)]["end"]:
            errs.append(f"T{t} starts before O({i},{j-1}) ends")
        if j < len(inst["jobs"][i]) and ops[(i, j)]["start"] < d["end"]:
            errs.append(f"O({i},{j}) starts before its transport")
    for r, seq in sched["robot_seq"].items():
        prev = None
        for t in seq:
            pu = tasks[t][2]
            need = (tr[prev]["end"] + sig[tasks[prev][3]][pu]) if prev is not None else sig[0][pu]
            if tr[t]["start"] < need:
                errs.append(f"robot {r}: T{t} too early ({tr[t]['start']} < {need})")
            prev = t
    for k, seq in sched["machine_seq"].items():
        prev = None
        for o in seq:
            o = tuple(o)
            rec = ops[o]
            i, j = o
            P, G = tabs.op_tables(k, inst["jobs"][i][j][1], inst["kappa"][o])
            Ca = ops[prev]["end"] if prev else 0
            if cfg.use_fatigue:
                fC = int(round(ops[prev]["f_end"] * L)) if prev else tabs.f0[k]
                fpre, fS, fe = (int(round(rec[x] * L)) for x in ("f_pre", "f_start", "f_end"))
                d = rec["rest_dur"]
                if rec["start"] < Ca + d:
                    errs.append(f"O{o} overlaps machine predecessor/rest")
                if rec["rest"]:
                    if fpre != fC or not tabs.triggered(k, fC, G):
                        errs.append(f"O{o} rest not triggered at predecessor completion")
                else:
                    if fpre != tabs.decay(k, fC, rec["start"] - Ca):
                        errs.append(f"O{o} f_pre {fpre} != decay of {fC}")
                    if tabs.triggered(k, fpre, G):
                        errs.append(f"O{o} unsafe without rest")
                if rec["rest"] and d != tabs.rest[k][fpre]:
                    errs.append(f"O{o} rest duration")
                if fS != (tabs.fmin[k] if rec["rest"] else fpre):
                    errs.append(f"O{o} f_start")
                if rec["dur"] != P[fS] or fe != G[fS]:
                    errs.append(f"O{o} duration/fatigue tables")
                if cfg.hard_bound and fe > tabs.fmax[k]:
                    errs.append(f"O{o} exceeds F_max")
            else:
                if rec["start"] < Ca:
                    errs.append(f"O{o} overlaps machine predecessor")
            if rec["end"] - rec["start"] != rec["dur"]:
                errs.append(f"O{o} end-start != dur")
            prev = o
    ends = [tr[of_ret[i]]["end"] for i in of_ret] if inst["last_ttask"] else [v["end"] for v in ops.values()]
    if max(ends) != sched["cmax"]:
        errs.append(f"cmax {sched['cmax']} != {max(ends)}")
    return errs
