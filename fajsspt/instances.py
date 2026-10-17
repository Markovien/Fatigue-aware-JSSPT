"""Instance construction, loading and transformation.

Normalised instance (dict):
  id, M, R, last_ttask (1 = jobs return to Z), sigma ((M+1)x(M+1), index 0 = Z),
  jobs: list of jobs, each a list of (machine 1..M, nominal duration),
  fatigue: {Fmin, Fmax, lam, mu, F0}: lists indexed 1..M (index 0 unused),
  kappa: {(i, j): fatigue sensitivity of O_ij},
  minutes_per_unit: physical meaning of one time unit,
  meta: free information.
Rates lambda/mu are expressed *per time unit of the instance*.
"""
from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional

DATA = Path(__file__).parent / "data"

# Base fatigue parameters (per minute), as in the submitted manuscript.
BASE = dict(lam=0.04, mu=0.08, Fmin=0.20, Fmax=0.80, kappa=0.3 * math.log(2.0))

_BU = None


def _bu():
    global _BU
    if _BU is None:
        _BU = json.loads((DATA / "bu_data.json").read_text())
    return _BU


def _fatigue_block(M, lam, mu, Fmin, Fmax, F0=None):
    rep = lambda v: [None] + [float(v)] * M if not isinstance(v, (list, tuple)) else [None] + [float(x) for x in v]
    blk = dict(lam=rep(lam), mu=rep(mu), Fmin=rep(Fmin), Fmax=rep(Fmax))
    blk["F0"] = list(blk["Fmin"]) if F0 is None else rep(F0)
    return blk


def make_instance(iid, sigma, jobs, R, last_ttask=1, lam=BASE["lam"], mu=BASE["mu"],
                  Fmin=BASE["Fmin"], Fmax=BASE["Fmax"], kappa=BASE["kappa"],
                  minutes_per_unit=1.0, meta=None) -> dict:
    M = len(sigma) - 1
    kap = {(i, j): float(kappa) for i, job in enumerate(jobs) for j in range(len(job))}
    return dict(id=iid, M=M, R=int(R), last_ttask=int(last_ttask),
                sigma=[[int(x) for x in row] for row in sigma],
                jobs=[[(int(k), int(t)) for k, t in job] for job in jobs],
                fatigue=_fatigue_block(M, lam, mu, Fmin, Fmax),
                kappa=kap, minutes_per_unit=float(minutes_per_unit), meta=meta or {})


# ---------------------------------------------------------------------------
# Bilge & Ulusoy (1995)
# ---------------------------------------------------------------------------
def bu_name(jobset: int, layout: int) -> str:
    return f"EX{jobset}{layout}"


def bu_adapted(jobset: int, layout: int, **fat) -> dict:
    """Adapted BU instance of the manuscript (AIVs: symmetric times, return to Z)."""
    d = _bu()
    return make_instance(bu_name(jobset, layout), d["sigma_adapted"][str(layout)],
                         d["jobsets"][str(jobset)], R=2, last_ttask=1,
                         meta=dict(source="Adapted Bilge & Ulusoy (1995)", jobset=jobset, layout=layout),
                         **fat)


def bu_original(jobset: int, layout: int) -> dict:
    """Original BU instance (AGVs: asymmetric times, no return to Z), fatigue off."""
    d = _bu()
    return make_instance(bu_name(jobset, layout), d["sigma_original"][str(layout)],
                         d["jobsets"][str(jobset)], R=2, last_ttask=0,
                         meta=dict(source="Bilge & Ulusoy (1995), original", jobset=jobset, layout=layout))


def bu_all(kind="adapted", **fat) -> List[dict]:
    """40 instances in the order of the manuscript tables (layout-major)."""
    out = []
    for layout in (1, 2, 3, 4):
        for js in range(1, 11):
            out.append(bu_adapted(js, layout, **fat) if kind == "adapted" else bu_original(js, layout))
    return out


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------
def with_params(inst: dict, suffix: str = "", **p) -> dict:
    """Override fatigue parameters (uniform across operators)."""
    out = copy.deepcopy(inst)
    M = out["M"]
    for key in ("lam", "mu", "Fmin", "Fmax"):
        if key in p:
            out["fatigue"][key] = [None] + [float(p[key])] * M
    if "Fmin" in p:
        out["fatigue"]["F0"] = list(out["fatigue"]["Fmin"])
    if "kappa" in p:
        out["kappa"] = {k: float(p["kappa"]) for k in out["kappa"]}
    if suffix:
        out["id"] = inst["id"] + suffix
    return out


HETERO_RANGES = dict(Fmin=(0.05, 0.20), Fmax=(0.75, 0.90), lam=(0.01, 0.05), mu=(0.05, 0.10))


def heterogeneous(inst: dict, seed: int, ranges=HETERO_RANGES, feasible=None) -> dict:
    """Operator-specific parameters drawn uniformly (ranges of the authors' generator)."""
    rng = random.Random(seed)
    for _ in range(200):
        out = copy.deepcopy(inst)
        M = out["M"]
        for key, (lo, hi) in ranges.items():
            out["fatigue"][key] = [None] + [round(rng.uniform(lo, hi), 3) for _ in range(M)]
        out["fatigue"]["F0"] = list(out["fatigue"]["Fmin"])
        out["id"] = f"{inst['id']}_H{seed}"
        out["meta"] = dict(out.get("meta", {}), hetero_seed=seed)
        if feasible is None or feasible(out):
            return out
    raise RuntimeError("could not draw a feasible heterogeneous instance")


def time_scale(inst: dict, s: int) -> dict:
    """Finer time resolution: durations x s, rates / s (same physical system)."""
    out = copy.deepcopy(inst)
    out["sigma"] = [[x * s for x in row] for row in inst["sigma"]]
    out["jobs"] = [[(k, t * s) for k, t in job] for job in inst["jobs"]]
    for key in ("lam", "mu"):
        out["fatigue"][key] = [None] + [v / s for v in inst["fatigue"][key][1:]]
    out["minutes_per_unit"] = inst["minutes_per_unit"] / s
    out["meta"] = dict(inst.get("meta", {}), time_scale=s)
    out["id"] = f"{inst['id']}_x{s}"
    return out


def no_fatigue(inst: dict) -> dict:
    out = copy.deepcopy(inst)
    out["meta"] = dict(out.get("meta", {}), fatigue="off")
    return out


# ---------------------------------------------------------------------------
# Larger instances: JSPLIB job data + transport layout
# ---------------------------------------------------------------------------
def parse_jsplib(path: Path) -> List[List[tuple]]:
    rows = [ln.split() for ln in Path(path).read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]
    n, m = int(rows[0][0]), int(rows[0][1])
    jobs = []
    for r in rows[1:1 + n]:
        v = list(map(int, r))
        jobs.append([(v[2 * q] + 1, v[2 * q + 1]) for q in range(m)])
    return jobs


def generated_layout(M: int, seed: int, tmin: int = 2, tmax: int = 10) -> List[List[int]]:
    """Symmetric, metric travel times in [tmin, tmax] from random 2-D positions
    (Manhattan distance), depot Z at index 0.  Used when the published layouts
    of Homayouni & Fontes are not supplied as CSV."""
    rng = random.Random(seed)
    pts = [(rng.uniform(0, 1), rng.uniform(0, 1)) for _ in range(M + 1)]
    D = [[abs(a[0] - b[0]) + abs(a[1] - b[1]) for b in pts] for a in pts]
    off = [D[a][b] for a in range(M + 1) for b in range(M + 1) if a != b]
    lo, hi = min(off), max(off)
    S = [[0] * (M + 1) for _ in range(M + 1)]
    for a in range(M + 1):
        for b in range(M + 1):
            if a != b:
                S[a][b] = int(round(tmin + (D[a][b] - lo) / (hi - lo) * (tmax - tmin)))
    return S


def read_matrix_csv(path: Path) -> List[List[int]]:
    rows = [ln for ln in Path(path).read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    return [[int(float(x)) for x in ln.replace(";", ",").split(",")] for ln in rows]


DN_LAYOUT = DATA / "dn_layout_8m.csv"   # Deroussi & Norre (2010), Homayouni & Fontes (2021) Table A2


def jsplib_instance(name: str, jsplib_dir: Path, R: int, travel_csv: Optional[Path] = None,
                    layout_seed: int = 1, target_max_op_minutes: float = 22.0,
                    travel_scale: Optional[float] = None, layout: str = "dn",
                    layout_max_op: float = 44.0) -> dict:
    """JSPLIB job data (e.g. ft06, la01-la20) with robots and fatigue.

    Time unit: chosen so that the longest operation lasts `target_max_op_minutes`
    (22 min = longest BU operation), i.e. rates lambda, mu are rescaled per unit.
    Travel times: CSV (published layout, depot first) or generated in [2,10]; they are
    multiplied by `travel_scale` (default max_tau/22) to keep the BU travel/processing ratio.
    """
    jobs = parse_jsplib(Path(jsplib_dir) / name)
    M = max(k for job in jobs for k, _ in job)
    max_tau = max(t for job in jobs for _, t in job)
    u = target_max_op_minutes / max_tau
    if travel_csv:
        sigma, src = read_matrix_csv(travel_csv), str(travel_csv)
    elif layout == "dn":
        full = read_matrix_csv(DN_LAYOUT)
        if M + 1 > len(full):
            raise ValueError(f"{name}: {M} machines but the DN layout has {len(full) - 1}")
        sigma, src = [row[:M + 1] for row in full[:M + 1]], "Deroussi-Norre 8-machine layout (LU, M1..M%d)" % M
    else:
        sigma, src = generated_layout(M, layout_seed), f"generated(seed={layout_seed})"
    if len(sigma) != M + 1:
        raise ValueError(f"travel matrix must be {(M + 1)}x{(M + 1)} (depot first)")
    # keep the travel/processing ratio of the layout's own instances (DN: longest op = 44)
    ts = max(1.0, max_tau / (layout_max_op if layout == "dn" and not travel_csv else 22.0)) \
        if travel_scale is None else travel_scale
    sigma = [[int(round(x * ts)) for x in row] for row in sigma]
    inst = make_instance(f"{name}_R{R}", sigma, jobs, R=R, last_ttask=1,
                         lam=BASE["lam"] * u, mu=BASE["mu"] * u, minutes_per_unit=u,
                         meta=dict(source=f"JSPLIB {name}", travel=src, travel_scale=ts))
    return inst


# ---------------------------------------------------------------------------
# Deroussi & Norre (2010) instances (fjsp1-fjsp10), as printed in Homayouni & Fontes (2021)
# ---------------------------------------------------------------------------
def dn_instance(jobset: int, R: int, minutes_per_unit: float = 0.5, scale_mu: bool = True) -> dict:
    """Job-shop version of the Deroussi-Norre FJSPT instance `fjsp<jobset>`.

    Each job of Table A1 offers two machine-disjoint routes (M1/M2, M3/M4, M5/M6, M7/M8 with
    equal times); every route becomes a job of its own, J_a on the odd machines and J_b on the
    even machines, which doubles the number of jobs.  Travel times: Table A2 (eight machines,
    LU = depot).  One time unit = 0.5 min (DN processing times are about twice the BU ones), so
    the longest operation (44 units) lasts 22 min, as in BU.
    """
    dn = json.loads((DATA / "dn_jobsets.json").read_text())[str(jobset)]
    jobs, names = [], []
    for q, ops in enumerate(dn, start=1):
        jobs.append([(a, t) for a, b, t in ops])
        jobs.append([(b, t) for a, b, t in ops])
        names += [f"J{q}a", f"J{q}b"]
    # scale_mu=True : lambda and mu both halved  (= time unit of 0.5 min, same dynamics as BU)
    # scale_mu=False: only lambda halved (recovery relatively twice as fast as in BU)
    lam = BASE["lam"] * minutes_per_unit
    mu = BASE["mu"] * minutes_per_unit if scale_mu else BASE["mu"]
    return make_instance(f"DN{jobset}_R{R}", read_matrix_csv(DN_LAYOUT), jobs, R=R, last_ttask=1,
                         lam=lam, mu=mu, minutes_per_unit=minutes_per_unit,
                         meta=dict(source=f"Deroussi & Norre (2010) fjsp{jobset}, routes split into jobs",
                                   job_names=names, travel="Deroussi-Norre 8-machine layout (Table A2)"))


# ---------------------------------------------------------------------------
# (De)serialisation
# ---------------------------------------------------------------------------
def to_json(inst: dict) -> dict:
    d = dict(inst)
    d["kappa"] = {f"{i},{j}": v for (i, j), v in inst["kappa"].items()}
    d["jobs"] = [[list(op) for op in job] for job in inst["jobs"]]
    return d


def from_json(d: dict) -> dict:
    if "machines_nb" in d:            # legacy JSON written by the original generators
        return from_legacy_json(d)
    d = dict(d)
    d["kappa"] = {tuple(map(int, k.split(","))): v for k, v in d["kappa"].items()}
    d["jobs"] = [[tuple(op) for op in job] for job in d["jobs"]]
    return d


def from_legacy_json(d: dict) -> dict:
    """Read instances produced by adapted_/original_bu_instances_generator.py.
    Legacy delta is defined w.r.t. tau(1 + delta*lambda*ln(1+F)), so kappa = delta*lambda."""
    M = d["machines_nb"]
    jobs = [[(int(op["machine_index"]), int(op["duration"])) for op in job["ops"]] for job in d["jobs"]]
    meta = d.get("meta", {})
    fat = meta.get("fatigue")
    iid = d["instance_id"].split("_S")[0].replace("_", "")
    last = 1 if "scenario" in d else 0
    if not fat:
        return make_instance(iid, d["sigma"], jobs, R=d.get("robots_nb", 2), last_ttask=last, meta=meta)
    inst = make_instance(iid, d["sigma"], jobs, R=d.get("robots_nb", 2), last_ttask=last,
                         lam=fat["lambda"], mu=fat["mu"], Fmin=fat["F_min"], Fmax=fat["F_max"], meta=meta)
    lam = inst["fatigue"]["lam"]
    dm = meta.get("delta", {})
    for i, job in enumerate(d["jobs"]):
        for j, op in enumerate(job["ops"]):
            k = int(op["machine_index"])
            delta = float(dm["value"]) if dm.get("mode") == "uniform" else float(op.get("delta", 0.0))
            inst["kappa"][(i, j)] = delta * lam[k]
    return inst


def save(inst: dict, path: Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(to_json(inst), indent=1))


def load(path: Path) -> dict:
    return from_json(json.loads(Path(path).read_text()))


def n_ops(inst: dict) -> int:
    return sum(len(j) for j in inst["jobs"])
