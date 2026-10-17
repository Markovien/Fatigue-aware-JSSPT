"""GA and VNS for the FaJSSPT, with switchable revision improvements.

Both algorithms keep the encoding of the submitted paper (operation sequence as a
permutation with repetition) and its overall loop.  Improvements (all optional):
  I1 decoder: robot chosen to let the operation start earliest + insertion of
     transports into robot idle gaps + returns in completion order    (Decoder options)
  I2 critical-path neighbourhood including rest blocks ("fatigue-critical moves")
  I3 seeded initial solutions (dispatching rules MWKR, MOR, SPT, LPT and a
     fatigue-aware earliest-start greedy rule, randomised)
  I4 VNS: acceptance of equal-makespan moves + restarts after stagnation
  I5 GA: elitism, insert mutation with decreasing rate, memetic critical local search
  I6 evaluation cache
Legacy presets (DEC_LEGACY, VNS_LEGACY, GA_LEGACY) reproduce the behaviour of the submitted code
(random start, OX + swap mutation without elitism; VNS with k_max = 50, strength 2,
3 random candidates per local-search step and strict descent).
Every run is reproducible through its seed; the budget is wall-clock time
(identical for GA and VNS), optionally capped by a number of evaluations.
"""
from __future__ import annotations

import random
import time
from typing import Dict, List, Optional, Tuple

from .decoder import Decoder


# =============================================================================
# common machinery
# =============================================================================
class _Run:
    def __init__(self, dec: Decoder, seed: int, time_limit: float, max_evals: Optional[int],
                 cache: bool):
        self.dec, self.rng = dec, random.Random(seed)
        self.t0, self.tl, self.max_evals = time.time(), time_limit, max_evals
        self.cache = {} if cache else None
        self.evals = 0
        self.best, self.best_seq, self.t_best = float("inf"), None, 0.0
        self.trace = []

    def elapsed(self):
        return time.time() - self.t0

    def done(self):
        return self.elapsed() >= self.tl or (self.max_evals is not None and self.evals >= self.max_evals)

    def ev(self, seq: List[int]) -> int:
        if self.cache is not None:
            key = tuple(seq)
            v = self.cache.get(key)
            if v is not None:
                return v
        self.evals += 1
        v = self.dec.decode(seq)
        if self.cache is not None:
            if len(self.cache) > 200000:
                self.cache.clear()
            self.cache[key] = v
        if v < self.best:
            self.best, self.best_seq, self.t_best = v, list(seq), self.elapsed()
            self.trace.append((round(self.t_best, 3), v))
        return v

    def result(self, extra=None):
        out = dict(cmax=self.best, time_to_best=round(self.t_best, 3), total_time=round(self.elapsed(), 3),
                   evals=self.evals, seq=self.best_seq, trace=self.trace)
        out.update(extra or {})
        return out


# --- random moves (as in the submitted code) -------------------------------------
def mv_swap(seq, rng):
    s = seq[:]
    a, b = rng.sample(range(len(s)), 2)
    s[a], s[b] = s[b], s[a]
    return s


def mv_insert(seq, rng):
    s = seq[:]
    a, b = rng.sample(range(len(s)), 2)
    s.insert(b, s.pop(a))
    return s


def mv_invert(seq, rng):
    s = seq[:]
    a, b = sorted(rng.sample(range(len(s)), 2))
    s[a:b + 1] = reversed(s[a:b + 1])
    return s


RANDOM_MOVES = (mv_swap, mv_insert, mv_invert)


# --- critical-path moves ------------------------------------------------------------
def _positions(seq, J):
    cnt, pos = [0] * J, {}
    for p, i in enumerate(seq):
        pos[(i, cnt[i])] = p
        cnt[i] += 1
    return pos


def _move_gene(seq, frm, to):
    s = seq[:]
    g = s.pop(frm)
    if to > frm:
        to -= 1
    s.insert(to, g)
    return s


def critical_moves(dec: Decoder, seq: List[int], st, cmax) -> List[List[int]]:
    """Moves on the critical chain: machine-adjacent reorderings (incl. rest and idle-wait
    blocks) and robot-adjacent reorderings.  Job order is always preserved."""
    chain = dec.critical_path(st, cmax)
    pos = _positions(seq, dec.J)
    nops = dec.nops
    out, seen = [], set()

    def before(b, a):          # put gene of op b right before gene of op a
        pb, pa = pos[b], pos[a]
        if pb <= pa:
            return
        if b[1] > 0 and pos[(b[0], b[1] - 1)] >= pa:
            return
        s = _move_gene(seq, pb, pa)
        key = tuple(s)
        if key not in seen:
            seen.add(key)
            out.append(s)

    def after(a, b):           # put gene of op a right after gene of op b
        pa, pb = pos[a], pos[b]
        if pa >= pb:
            return
        if a[1] + 1 < nops[a[0]] and pos[(a[0], a[1] + 1)] <= pb:
            return
        s = _move_gene(seq, pa, pb + 1)
        key = tuple(s)
        if key not in seen:
            seen.add(key)
            out.append(s)

    msucc = None
    for kind, ref, cause in chain:
        ck, cref = cause
        if kind == "op" and ck in ("M", "rest", "wait") and cref is not None:
            a, b = cref, ref
            before(b, a)
            after(a, b)
            if ck in ("rest", "wait"):           # fatigue-critical block
                if msucc is None:
                    msucc = {}
                    for k in range(1, dec.M + 1):
                        ops_k = sorted((o for o in st["ops"] if dec.jobs[o[0]][o[1]][0] == k),
                                       key=lambda o: st["ops"][o][0])
                        for x, y in zip(ops_k, ops_k[1:]):
                            msucc[x] = y
                c = msucc.get(b)
                if c is not None:
                    before(c, b)        # let the successor go first
                    after(b, c)
                # move the rest-triggering operation one machine position earlier
                p = [x for x, y in msucc.items() if y == a]
                if p:
                    before(b, p[0])
        elif kind == "task" and ck == "R" and cref is not None:
            i, j = dec.tasks[ref][0], dec.tasks[ref][1]
            ip, jp = dec.tasks[cref][0], dec.tasks[cref][1]
            if j < nops[i] and jp < nops[ip] and i != ip:
                before((i, j), (ip, jp))
    return out


def critical_ls(run: _Run, seq, cmax, max_steps=30, plateau=8):
    dec, rng = run.dec, run.rng
    cur, cur_c = seq, cmax
    flat = plateau
    for _ in range(max_steps):
        if run.done():
            break
        c, st = dec.decode(cur, record=True)
        moves = critical_moves(dec, cur, st, c)
        rng.shuffle(moves)
        eq, improved = [], False
        for m in moves:
            if run.done():
                break
            v = run.ev(m)
            if v < cur_c:
                cur, cur_c, improved = m, v, True
                break
            if v == cur_c:
                eq.append(m)
        if improved:
            flat = plateau
            continue
        if eq and flat > 0:
            cur = rng.choice(eq)
            flat -= 1
            continue
        break
    return cur, cur_c


def random_ls(run: _Run, seq, cmax, max_steps=50):
    """First-improvement local search of the submitted VNS (3 random candidates per step)."""
    cur, cur_c = seq, cmax
    for _ in range(max_steps):
        if run.done():
            break
        improved = False
        for mv in RANDOM_MOVES:
            cand = mv(cur, run.rng)
            v = run.ev(cand)
            if v < cur_c:
                cur, cur_c, improved = cand, v, True
                break
        if not improved:
            break
    return cur, cur_c


# --- constructive seeds -------------------------------------------------------------------
def rule_sequence(dec: Decoder, rule: str, rng: random.Random, noise: float = 0.0) -> List[int]:
    J, jobs = dec.J, dec.jobs
    nxt = [0] * J
    seq = []
    if rule == "EST":                          # fatigue-aware earliest-start greedy
        st = dec.new_state()
        for _ in range(dec.N):
            best = None
            for i in range(J):
                if nxt[i] >= dec.nops[i]:
                    continue
                s2 = dec.copy_state(st)
                o = dec.step(s2, i)
                key = s2["ops"][o][0] + (rng.random() * noise * 10 if noise else 0)
                if best is None or key < best[0]:
                    best = (key, i, s2)
            _, i, st = best
            nxt[i] += 1
            seq.append(i)
        return seq
    rem = [sum(t for _, t in job) for job in jobs]
    for _ in range(dec.N):
        cand = [i for i in range(J) if nxt[i] < dec.nops[i]]
        def pr(i):
            t = jobs[i][nxt[i]][1]
            v = {"MWKR": -rem[i], "MOR": -(dec.nops[i] - nxt[i]), "SPT": t, "LPT": -t}[rule]
            return v + (rng.random() * noise * abs(v) if noise else 0)
        i = min(cand, key=pr)
        rem[i] -= jobs[i][nxt[i]][1]
        nxt[i] += 1
        seq.append(i)
    return seq


def seeds(dec: Decoder, rng: random.Random, n: int) -> List[List[int]]:
    out = [rule_sequence(dec, r, rng) for r in ("EST", "MWKR", "MOR", "SPT", "LPT")]
    while len(out) < n:
        out.append(rule_sequence(dec, rng.choice(("EST", "MWKR", "MOR")), rng, noise=0.3))
    return out[:n]


# =============================================================================
# VNS
# =============================================================================
def vns(dec: Decoder, seed: int = 0, time_limit: float = 10.0, max_evals: Optional[int] = None,
        ls: str = "critical", seeded: bool = True, accept_equal: bool = True,
        restart_after: Optional[int] = 300, k_max: int = 6, cache: bool = True,
        legacy_shake: bool = False, label: str = "VNS") -> dict:
    legacy = legacy_shake
    run = _Run(dec, seed, time_limit, max_evals, cache)
    rng = run.rng
    if seeded:
        init = min(seeds(dec, rng, 5), key=run.ev)
    else:
        init = dec.base_seq[:]
        rng.shuffle(init)
    cur, cur_c = init, run.ev(init)
    since_best, it = 0, 0

    def shake(s, k):
        if legacy:   # submitted code: 2 perturbations, type by k
            mv = RANDOM_MOVES[min(k, 3) - 1]
            for _ in range(2):
                s = mv(s, rng)
            return s
        mv = RANDOM_MOVES[(k - 1) % 3]
        for _ in range(1 + (k - 1) // 3):
            s = mv(s, rng)
        return s

    while not run.done():
        k = 1
        while k <= k_max and not run.done():
            it += 1
            s1 = shake(cur, k)
            c1 = run.ev(s1)
            if ls == "critical":
                s1, c1 = critical_ls(run, s1, c1)
                s1, c1 = random_ls(run, s1, c1, max_steps=10)
            else:
                s1, c1 = random_ls(run, s1, c1)
            old_best = run.best
            if c1 < cur_c:
                cur, cur_c, k = s1, c1, 1
            elif accept_equal and c1 == cur_c:
                cur, k = s1, k + 1
            else:
                k += 1
            since_best = 0 if run.best < old_best else since_best + 1
            if restart_after and since_best >= restart_after:
                base = run.best_seq if rng.random() < 0.5 else rule_sequence(dec, "EST", rng, noise=0.5)
                cur = base
                for _ in range(max(3, dec.N // 5)):
                    cur = mv_insert(cur, rng)
                cur_c, since_best, k = run.ev(cur), 0, 1
    return run.result(dict(algorithm=label, iterations=it))


# =============================================================================
# GA
# =============================================================================
def _ox(p1, p2, rng, J, nops):
    """Order crossover adapted to repeated job indices (as in the submitted code)."""
    n = len(p1)
    a, b = sorted(rng.sample(range(n), 2))

    def child(pa, pb):
        c = [None] * n
        used = [0] * J
        for q in range(a, b + 1):
            c[q] = pa[q]
            used[pa[q]] += 1
        pos = (b + 1) % n
        for g in pb:
            if used[g] < nops[g]:
                while c[pos] is not None:
                    pos = (pos + 1) % n
                c[pos] = g
                used[g] += 1
        return c
    return child(p1, p2), child(p2, p1)


def ga(dec: Decoder, seed: int = 0, time_limit: float = 10.0, max_evals: Optional[int] = None,
       pop_size: int = 50, generations: Optional[int] = None,
       pc: float = 0.9, pm: float = 0.2, elite: int = 2, seeded: bool = True,
       memetic: bool = True, mutation: str = "insert", cache: bool = True,
       decreasing_pm: bool = True, label: str = "GA") -> dict:
    run = _Run(dec, seed, time_limit, max_evals, cache)
    rng = run.rng
    J, nops = dec.J, dec.nops
    pop = seeds(dec, rng, max(1, pop_size // 10)) if seeded else []
    while len(pop) < pop_size:
        s = dec.base_seq[:]
        rng.shuffle(s)
        pop.append(s)
    fit = [run.ev(s) for s in pop]
    gen = 0
    while not run.done() and (generations is None or gen < generations):
        frac = min(1.0, run.elapsed() / max(run.tl, 1e-9))
        pm_g = pm * (1.0 - 0.75 * frac) if decreasing_pm else pm          # decreasing mutation rate
        order = sorted(range(len(pop)), key=lambda q: fit[q])
        new = [pop[q][:] for q in order[:elite]]

        def tour():
            best = None
            for _ in range(3):
                q = rng.randrange(len(pop))
                if best is None or fit[q] < fit[best]:
                    best = q
            return pop[best]
        while len(new) < pop_size:
            p1, p2 = tour(), tour()
            c1, c2 = _ox(p1, p2, rng, J, nops) if rng.random() < pc else (p1[:], p2[:])
            for c in (c1, c2):
                if rng.random() < pm_g:
                    c[:] = mv_swap(c, rng) if mutation == "swap" else mv_insert(c, rng)
                if len(new) < pop_size:
                    new.append(c)
        if run.done():
            break
        nfit = [run.ev(s) for s in new]
        if memetic:
            q = min(range(len(new)), key=lambda x: nfit[x])
            new[q], nfit[q] = critical_ls(run, new[q], nfit[q], max_steps=10, plateau=3)
        pop, fit = new, nfit
        gen += 1
    return run.result(dict(algorithm=label, generations=gen))


# =============================================================================
# presets
# =============================================================================
DEC_NEW = dict(robot_rule="earliest_start", robot_insertion=True, returns_order="completion", smart_wait=True)
DEC_LEGACY = dict(robot_rule="earliest_delivery", robot_insertion=False, returns_order="index", smart_wait=False)
VNS_LEGACY = dict(ls="random", seeded=False, accept_equal=False, restart_after=None, k_max=50,
                  cache=False, legacy_shake=True)
GA_LEGACY = dict(elite=0, seeded=False, memetic=False, mutation="swap", cache=False, decreasing_pm=False)

# ablation ladder (E8): decoder (I1), critical neighbourhood (I2), seeding (I3), rest (I4-I6)
VARIANTS = {
    "legacy":   (DEC_LEGACY, VNS_LEGACY, GA_LEGACY),
    "I1":       (DEC_NEW, VNS_LEGACY, GA_LEGACY),
    "I1+I2":    (DEC_NEW, dict(VNS_LEGACY, ls="critical"), dict(GA_LEGACY, memetic=True)),
    "I1+I3":    (DEC_NEW, dict(VNS_LEGACY, seeded=True), dict(GA_LEGACY, seeded=True)),
    "I1+I2+I3": (DEC_NEW, dict(VNS_LEGACY, ls="critical", seeded=True),
                 dict(GA_LEGACY, memetic=True, seeded=True)),
    "full":     (DEC_NEW, {}, {}),
}


def run_variant(dec_factory, alg: str, variant: str, seed: int, time_limit: float, max_evals=None):
    dkw, vkw, gkw = VARIANTS[variant]
    dec = dec_factory(**dkw)
    if alg == "VNS":
        return vns(dec, seed=seed, time_limit=time_limit, max_evals=max_evals, label=f"VNS[{variant}]", **vkw), dec
    return ga(dec, seed=seed, time_limit=time_limit, max_evals=max_evals, label=f"GA[{variant}]", **gkw), dec
