"""Fatigue model: continuous laws, discretised lookup tables, triggers.

Notation (revised manuscript):
  accumulation   F(t+d) = 1 - (1 - F(t)) * exp(-lambda_k * d)
  recovery       F(t+d) = F(t) * exp(-mu_k * d)                (explicit rest)
  idle recovery  F(t+d) = max(F_min, F(t) * exp(-mu_idle * d)), mu_idle = rho * mu_k
  processing     tau'   = tau * (1 + kappa_ij * ln(1 + F(S)))
                 kappa = delta * lambda in the legacy code (delta = (0.3/lambda) ln 2),
                 i.e. kappa = 0.3 ln 2 = 0.2079 for the base case.
  rest duration  psi(F) = (1/mu_k) ln(F / F_min)
  hard bound     F(C_ij) <= F_max  (fatigue is monotone during work, hence the
                 bound holds over the whole operation)

Rest policy ("rest only when triggered"): a rest before O_ij is taken
if and only if the operation cannot be processed safely from the
current (pre-rest) fatigue level F^pre, i.e.
     trigger  <=>  F^pre >= F_max  OR  F^C(F^pre) > F_max        (hard bound)
     trigger  <=>  F^pre >= F_max  OR  F^C(F^pre) >= 1 - eps      (legacy, soft)
A rest always brings the operator back to F_min.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

OpKey = Tuple[int, int]  # (job index i, operation index j), 0-based


@dataclass
class FatigueConfig:
    """Options shared by CP-SAT, the decoder and the replay simulator."""
    use_fatigue: bool = True
    hard_bound: bool = True          # False = legacy start-threshold semantics
    L: int = 1000                    # fatigue grid resolution
    rounding: str = "nearest"        # "nearest" (legacy) | "conservative"
    recover_on_idle: bool = False
    idle_factor: float = 0.0         # rho in mu_idle = rho * mu
    K: int = 1000                    # log-domain precision for idle decay
    eps_exhaust_int: int = 1         # legacy soft trigger: f_C >= L - eps

    def label(self) -> str:
        return (f"fat={int(self.use_fatigue)} hard={int(self.hard_bound)} L={self.L} "
                f"round={self.rounding} idle={self.idle_factor if self.recover_on_idle else 0}")


def round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def _ceil(x: float) -> int:
    return int(math.ceil(x - 1e-9))


# ---------------------------------------------------------------------------
# Continuous laws
# ---------------------------------------------------------------------------
def tau_cont(tau: float, kappa: float, F: float) -> float:
    return tau * (1.0 + kappa * math.log(1.0 + max(F, 0.0)))


def accumulate_cont(F: float, lam: float, d: float) -> float:
    return 1.0 - (1.0 - F) * math.exp(-lam * d)


def recover_cont(F: float, mu: float, d: float) -> float:
    return F * math.exp(-mu * d)


def rest_cont(F: float, mu: float, Fmin: float) -> float:
    if F <= Fmin:
        return 0.0
    return math.log(F / Fmin) / mu


def idle_decay_cont(F: float, mu_idle: float, d: float, Fmin: float) -> float:
    if mu_idle <= 0 or d <= 0:
        return F
    return max(Fmin, F * math.exp(-mu_idle * d))


def remaining_workable_time(F: float, lam: float, Fmax: float) -> float:
    """theta_k(F): time an operator can work from F before reaching F_max."""
    if F >= Fmax:
        return 0.0
    return math.log((1.0 - F) / (1.0 - Fmax)) / lam


# ---------------------------------------------------------------------------
# Discretised tables (shared by CP-SAT and the decoder)
# ---------------------------------------------------------------------------
@dataclass
class FatigueTables:
    cfg: FatigueConfig
    M: int
    Fmin: List[float]
    Fmax: List[float]
    lam: List[float]
    mu: List[float]
    F0: List[float]
    fmin: List[int] = field(default_factory=list)
    fmax: List[int] = field(default_factory=list)
    f0: List[int] = field(default_factory=list)
    rest: Dict[int, List[int]] = field(default_factory=dict)
    _op: Dict[Tuple[int, int, float], Tuple[List[int], List[int]]] = field(default_factory=dict)
    ln_table: List[int] = field(default_factory=list)
    exp_table: List[int] = field(default_factory=list)
    OFFSET: int = 0
    mu_scaled: List[int] = field(default_factory=list)

    @classmethod
    def build(cls, inst: dict, cfg: FatigueConfig) -> "FatigueTables":
        fat = inst["fatigue"]
        M = inst["M"]
        t = cls(cfg=cfg, M=M, Fmin=fat["Fmin"], Fmax=fat["Fmax"], lam=fat["lam"],
                mu=fat["mu"], F0=fat["F0"])
        L = cfg.L
        cons = cfg.rounding == "conservative"
        t.fmin = [0] * (M + 1)
        t.fmax = [0] * (M + 1)
        t.f0 = [0] * (M + 1)
        for k in range(1, M + 1):
            if cons:
                t.fmin[k] = _ceil(L * t.Fmin[k])
                t.fmax[k] = int(math.floor(L * t.Fmax[k] + 1e-9))
                t.f0[k] = _ceil(L * t.F0[k])
            else:  # legacy behaviour of the submitted code
                t.fmin[k] = int(math.floor(L * t.Fmin[k] + 1e-9))
                t.fmax[k] = _ceil(L * t.Fmax[k])
                t.f0[k] = int(math.floor(L * t.F0[k] + 1e-9))
            t.rest[k] = t._build_rest_table(k)
        # idle-decay tables (same construction as the submitted CP-SAT code)
        K = cfg.K
        # only log-levels between ln(F_min) and 0 matter (below: clamped to F_min); a
        # compact table keeps CP-SAT presolve fast (the submitted code used 25*K entries)
        t.OFFSET = -(int(math.floor(K * math.log(min(t.Fmin[1:])))) - 2)
        size = t.OFFSET + 1
        if cons:   # less recovery than reality
            t.ln_table = [_ceil(K * math.log(max(1, f) / L)) for f in range(L + 1)]
            t.exp_table = [min(L, _ceil(L * math.exp((i - t.OFFSET) / K))) for i in range(size)]
            t.mu_scaled = [0] + [int(math.floor(t.mu[k] * cfg.idle_factor * K + 1e-9))
                                 for k in range(1, M + 1)]
        else:
            t.ln_table = [round(K * math.log(max(1, f) / L)) for f in range(L + 1)]
            t.exp_table = [min(L, round(L * math.exp((i - t.OFFSET) / K))) for i in range(size)]
            t.mu_scaled = [0] + [int(round(t.mu[k] * cfg.idle_factor * K)) for k in range(1, M + 1)]
        return t

    # -- rest duration from integer level f back to F_min (always rounded up) --
    def _build_rest_table(self, k: int) -> List[int]:
        L = self.cfg.L
        tab = [0] * (L + 1)
        for f in range(L + 1):
            F = f / L
            # FIX (revision): defined for every level above F_min, not only >= F_max,
            # otherwise a rest triggered by the completion rule would be a free reset.
            if F > self.Fmin[k] + 1e-12:
                tab[f] = _ceil(math.log(F / self.Fmin[k]) / self.mu[k])
        return tab

    def op_tables(self, k: int, tau: int, kappa: float) -> Tuple[List[int], List[int]]:
        """P[f] (realised duration) and G[f] (fatigue level at completion)."""
        key = (k, tau, kappa)
        if key in self._op:
            return self._op[key]
        L = self.cfg.L
        cons = self.cfg.rounding == "conservative"
        P, G = [0] * (L + 1), [0] * (L + 1)
        for f in range(L + 1):
            F = f / L
            tp = tau_cont(tau, kappa, F) if self.cfg.use_fatigue else float(tau)
            p = max(1, _ceil(tp)) if cons else max(1, round_half_up(tp))
            Fc = accumulate_cont(F, self.lam[k], p)
            g = _ceil(L * Fc) if cons else int(round(L * Fc))
            P[f], G[f] = p, max(0, min(L, g))
        self._op[key] = (P, G)
        return P, G

    def decay(self, k: int, f: int, gap: int) -> int:
        """Integer idle decay, identical to the CP-SAT element construction."""
        if not self.cfg.recover_on_idle or gap <= 0 or self.mu_scaled[k] <= 0:
            return f
        idx = self.ln_table[f] - gap * self.mu_scaled[k] + self.OFFSET
        if idx < 0:
            idx = 0
        # never above the current level (log/exp round trip), never below F_min
        return max(min(self.exp_table[idx], f), self.fmin[k])

    def triggered(self, k: int, f_pre: int, G: List[int]) -> bool:
        if f_pre >= self.fmax[k]:
            return True
        if self.cfg.hard_bound:
            return G[f_pre] > self.fmax[k]
        return G[f_pre] >= self.cfg.L - self.cfg.eps_exhaust_int

    def check_feasible(self, inst: dict) -> List[str]:
        """Every operation must be processable from F_min (else instance infeasible)."""
        bad = []
        if not (self.cfg.use_fatigue and self.cfg.hard_bound):
            return bad
        for i, job in enumerate(inst["jobs"]):
            for j, (k, tau) in enumerate(job):
                P, G = self.op_tables(k, tau, inst["kappa"][(i, j)])
                if G[self.fmin[k]] > self.fmax[k]:
                    bad.append(f"O({i},{j}) on M{k}: tau={tau} cannot be processed from F_min")
        return bad
