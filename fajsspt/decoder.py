"""Shared schedule decoder for GA and VNS (and for constructive heuristics).

A solution is an operation sequence encoded as a permutation with repetition of
job indices (the j-th occurrence of job i is O_ij).  The decoder builds a
semi-active schedule with *exactly* the semantics of the CP-SAT model
(same lookup tables, triggers, rest and idle-recovery rules), so objective
values are directly comparable with CP-SAT.

Options (each one can be switched off to reproduce the submitted decoder):
  robot_rule      "earliest_delivery" (submitted) | "earliest_start" (revision:
                  robot that lets the *operation* start earliest; ties -> just-in-time)
  robot_insertion insert transport tasks into idle gaps of robot timelines
  returns_order   "index" (submitted) | "completion" (final returns in order of job completion)
  smart_wait      with idle recovery: compare 'wait until idle decay makes the
                  operation safe' with 'rest now' and take the earlier start
Critical path: every activity records which predecessor fixed its start
(job, robot, machine, rest or idle-wait); `critical_path()` walks back from C_max.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .cpsat_model import _transport_tasks
from .fatigue import FatigueConfig, FatigueTables

BIG = 10 ** 15


class Decoder:
    def __init__(self, inst: dict, cfg: FatigueConfig, robot_rule: str = "earliest_start",
                 robot_insertion: bool = True, returns_order: str = "completion",
                 smart_wait: bool = True):
        self.inst, self.cfg = inst, cfg
        self.tabs = FatigueTables.build(inst, cfg)
        bad = self.tabs.check_feasible(inst)
        if bad:
            raise ValueError("instance infeasible under hard bound: " + bad[0])
        self.robot_rule, self.robot_insertion = robot_rule, robot_insertion
        self.returns_order, self.smart_wait = returns_order, smart_wait
        self.jobs = inst["jobs"]
        self.J, self.M, self.R = len(self.jobs), inst["M"], inst["R"]
        self.sig = inst["sigma"]
        self.nops = [len(j) for j in self.jobs]
        self.N = sum(self.nops)
        self.op_tab = {}
        for i, job in enumerate(self.jobs):
            for j, (k, tau) in enumerate(job):
                self.op_tab[(i, j)] = self.tabs.op_tables(k, tau, inst["kappa"][(i, j)])
        self.tasks, self.task_of_op, self.task_of_ret = _transport_tasks(inst)
        self.base_seq = [i for i in range(self.J) for _ in range(self.nops[i])]
        self.n_evals = 0

    # ------------------------------------------------------------------ state
    def new_state(self):
        st = dict(
            job_next=[0] * self.J, job_ready=[0] * self.J, job_loc=[0] * self.J,
            m_free=[0] * (self.M + 1), m_f=list(self.tabs.f0), m_last=[None] * (self.M + 1),
            robots=[[(0, 0, 0, 0, None)] for _ in range(self.R)],  # (start,end,pickup,drop,task)
            ops={}, tasks={}, rec=[])
        return st

    @staticmethod
    def copy_state(st):
        return dict(job_next=st["job_next"][:], job_ready=st["job_ready"][:], job_loc=st["job_loc"][:],
                    m_free=st["m_free"][:], m_f=st["m_f"][:], m_last=st["m_last"][:],
                    robots=[r[:] for r in st["robots"]], ops=dict(st["ops"]), tasks=dict(st["tasks"]),
                    rec=st["rec"][:])

    # ------------------------------------------------------------------ robots
    def _robot_options(self, st, rel, pu, dr, dur):
        """For each robot: (fin, est, robot, gap index, prev task id)."""
        sig, out = self.sig, []
        for r, tl in enumerate(st["robots"]):
            best = None
            rng = range(len(tl)) if self.robot_insertion else (len(tl) - 1,)
            for g in rng:
                prev = tl[g]
                est = max(rel, prev[1] + sig[prev[3]][pu])
                fin = est + dur
                if g + 1 < len(tl):
                    nxt = tl[g + 1]
                    if fin + sig[dr][nxt[2]] > nxt[0]:
                        continue
                if best is None or fin < best[0]:
                    best = (fin, est, r, g, prev[4])
            out.append(best)
        return out

    def _commit_robot(self, st, opt, pu, dr, task):
        fin, est, r, g, prev_task = opt
        st["robots"][r].insert(g + 1, (est, fin, pu, dr, task))
        return r

    # ------------------------------------------------------------------ operation timing
    def _op_timing(self, st, o, k, arrival):
        """Returns (start, rest, rest_dur, f_pre, f_S, cause_kind)."""
        tabs = self.tabs
        Ca, fC = st["m_free"][k], st["m_f"][k]
        base = arrival if arrival > Ca else Ca
        if not self.cfg.use_fatigue:
            return base, False, 0, 0, 0, ("T" if arrival >= Ca else "M")
        P, G = self.op_tab[o]
        idle_on = self.cfg.recover_on_idle and tabs.mu_scaled[k] > 0
        trig0 = tabs.triggered(k, fC, G)
        if not idle_on:
            if trig0:
                d = tabs.rest[k][fC]
                s = max(arrival, Ca + d)
                return s, True, d, fC, tabs.fmin[k], ("T" if arrival >= Ca + d else "rest")
            return base, False, 0, fC, fC, ("T" if arrival >= Ca else "M")
        # --- idle recovery ---
        # option A: no rest; idle decay over the gap must make the operation safe
        gap0 = base - Ca
        fp0 = tabs.decay(k, fC, gap0)
        sA = fpA = None
        if not tabs.triggered(k, fp0, G):
            sA, fpA = base, fp0
        elif self.smart_wait:          # wait until idle decay makes it safe (binary search)
            lo, hi = gap0, max(gap0 + 1, 1)
            while tabs.triggered(k, tabs.decay(k, fC, hi), G):
                lo, hi = hi, hi * 2
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if tabs.triggered(k, tabs.decay(k, fC, mid), G):
                    lo = mid
                else:
                    hi = mid
            sA, fpA = Ca + hi, tabs.decay(k, fC, hi)
        # option B: rest right after the predecessor (only if triggered there)
        if trig0:
            d = tabs.rest[k][fC]
            sB = max(arrival, Ca + d)
            if sA is None or sB <= sA:
                return sB, True, d, fC, tabs.fmin[k], ("T" if arrival >= Ca + d else "rest")
        return sA, False, 0, fpA, fpA, ("T" if sA == arrival and arrival >= Ca else ("M" if sA == Ca else "wait"))

    # ------------------------------------------------------------------ one gene
    def step(self, st, i):
        j = st["job_next"][i]
        o = (i, j)
        k, _ = self.jobs[i][j]
        pu = st["job_loc"][i]
        t = self.task_of_op[o]
        dur = self.sig[pu][k]
        rel = st["job_ready"][i]
        opts = [x for x in self._robot_options(st, rel, pu, k, dur) if x is not None]
        if self.robot_rule == "earliest_delivery":
            opt = min(opts, key=lambda x: (x[0], x[2]))
            timing = self._op_timing(st, o, k, opt[0])
        else:
            best = None
            for x in opts:
                tm = self._op_timing(st, o, k, x[0])
                key = (tm[0], -x[1], x[2])
                if best is None or key < best[0]:
                    best = (key, x, tm)
            _, opt, timing = best
        r = self._commit_robot(st, opt, pu, k, t)
        fin, est = opt[0], opt[1]
        tcause = ("J", (i, j - 1)) if (j > 0 and est == rel) else ("R", opt[4])
        st["tasks"][t] = (est, fin, r, tcause)
        s, rest, d, fpre, fS, ck = timing
        P, G = self.op_tab[o]
        if self.cfg.use_fatigue:
            p, fc = P[fS], G[fS]
        else:
            p, fc = self.jobs[i][j][1], 0
        c = s + p
        ocause = ("T", t) if ck == "T" else (ck, st["m_last"][k])
        st["ops"][o] = (s, c, p, fpre, fS, fc, rest, d, ocause)
        st["m_free"][k], st["m_f"][k], st["m_last"][k] = c, fc, o
        st["job_ready"][i], st["job_loc"][i] = c, k
        st["job_next"][i] = j + 1
        return o

    def finish(self, st):
        """Schedule final returns; return C_max."""
        if not self.inst["last_ttask"]:
            return max(st["job_ready"]) if self.J else 0
        order = range(self.J) if self.returns_order == "index" else \
            sorted(range(self.J), key=lambda i: (st["job_ready"][i], i))
        cmax = 0
        for i in order:
            t = self.task_of_ret[i]
            pu = st["job_loc"][i]
            dur = self.sig[pu][0]
            rel = st["job_ready"][i]
            opts = [x for x in self._robot_options(st, rel, pu, 0, dur) if x is not None]
            opt = min(opts, key=lambda x: (x[0], x[2]))
            r = self._commit_robot(st, opt, pu, 0, t)
            tcause = ("J", (i, self.nops[i] - 1)) if opt[1] == rel else ("R", opt[4])
            st["tasks"][t] = (opt[1], opt[0], r, tcause)
            cmax = max(cmax, opt[0])
        return cmax

    def decode(self, seq: List[int], record: bool = False):
        self.n_evals += 1
        st = self.new_state()
        for i in seq:
            self.step(st, i)
        cmax = self.finish(st)
        if record:
            return cmax, st
        return cmax

    # ------------------------------------------------------------------ analysis
    def critical_path(self, st, cmax) -> List[tuple]:
        """List of activities ('op', o, cause) / ('task', t, cause) from the end back."""
        chain = []
        if self.inst["last_ttask"]:
            t = max(self.task_of_ret.values(), key=lambda tt: (st["tasks"][tt][1], -tt))
            node = ("task", t)
        else:
            o = max(st["ops"], key=lambda oo: st["ops"][oo][1])
            node = ("op", o)
        guard = 0
        while node is not None and guard < 10 * (self.N + len(self.tasks)):
            guard += 1
            if node[0] == "task":
                s, e, r, cause = st["tasks"][node[1]]
                chain.append(("task", node[1], cause))
                kind, ref = cause
                if kind == "J" and ref is not None:
                    node = ("op", ref)
                elif kind == "R" and ref is not None:
                    node = ("task", ref)
                else:
                    node = None
            else:
                rec = st["ops"][node[1]]
                cause = rec[8]
                chain.append(("op", node[1], cause))
                kind, ref = cause
                if kind == "T":
                    node = ("task", ref)
                elif ref is not None:
                    node = ("op", ref)
                else:
                    node = None
        return chain

    def to_schedule(self, st, cmax) -> dict:
        L = self.cfg.L
        sched = dict(cmax=cmax, ops={}, transports=[], machine_seq={}, robot_seq={})
        for o, (s, c, p, fpre, fS, fc, rest, d, _) in st["ops"].items():
            sched["ops"][o] = dict(machine=self.jobs[o[0]][o[1]][0], start=s, end=c, dur=p,
                                   f_pre=fpre / L, f_start=fS / L, f_end=fc / L, rest=rest, rest_dur=d)
        for k in range(1, self.M + 1):
            sched["machine_seq"][k] = sorted([o for o in st["ops"] if self.jobs[o[0]][o[1]][0] == k],
                                             key=lambda o: st["ops"][o][0])
        for t, (s, e, r, _) in sorted(st["tasks"].items()):
            i, j, pu, dr, dur = self.tasks[t]
            sched["transports"].append(dict(task=t, job=i, j=j, pickup=pu, drop=dr, robot=r, start=s, end=e))
        for r, tl in enumerate(st["robots"]):
            sched["robot_seq"][r] = [x[4] for x in tl[1:]]
        return sched


def sequence_from_schedule(sched: dict) -> List[int]:
    """Operation sequence (by start time) of any schedule, e.g. a CP-SAT solution."""
    ops = sorted(sched["ops"].items(), key=lambda kv: (kv[1]["start"], kv[0]))
    return [o[0] for o, _ in ops]
