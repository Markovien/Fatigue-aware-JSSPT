"""Experiment runner for the CAIE revision (E0-E8).

    python experiments.py <exp> [<exp> ...] [options]   exp in: e0 ... e8, all
    python experiments.py e1 e2 e7                 runs E1, then E2, then E7 (queued, never in parallel)
    python experiments.py e2 --quick               smoke test (3 instances, short limits)

All results go to ./results (CSV, one row per run, written immediately).
Re-running a command skips the runs already in the CSV (safe to interrupt).
CP-SAT runs are cached in results/cpsat_runs.csv and shared by all experiments
(e.g. OPT1 of E1 is reused by E4/E5).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fajsspt import instances as I                                      # noqa: E402
from fajsspt.cpsat_model import solve                                     # noqa: E402
from fajsspt.decoder import Decoder                                       # noqa: E402
from fajsspt.fatigue import FatigueConfig, FatigueTables                  # noqa: E402
from fajsspt.metaheuristics import run_variant                            # noqa: E402
from fajsspt.replay import replay, validate                               # noqa: E402
from fajsspt.serial import CsvLog, load_sched, save_sched                 # noqa: E402
from fajsspt.warm import warm_start                                        # noqa: E402

RES = Path("results")
SUBSET10 = ["EX11", "EX22", "EX33", "EX44", "EX51", "EX62", "EX73", "EX84", "EX91", "EX102"]
LAYOUT2 = [f"EX{j}2" for j in range(1, 11)]
QUICK = ["EX11", "EX52", "EX82"]
RHOS = [0.0, 0.25]          # beyond 0.25 idle recovery removes all rests (authors' decision)
SENS = {"lam": [0.02, 0.03, 0.05, 0.06], "mu": [0.04, 0.06, 0.12, 0.16],
        "kappa": [0.0, 0.10, 0.30, 0.40], "Fmax": [0.70, 0.75, 0.85, 0.90], "Fmin": [0.05, 0.10, 0.30]}
# JSPLIB job data on the published 8-machine layout of Deroussi & Norre (2010):
# ft06 (6x6), la01-05 (10x5), la06-10 (15x5), la11-15 (20x5)
LARGE = ([("ft06", 2)] + [(f"la{q:02d}", 2) for q in range(1, 6)] + [(f"la{q:02d}", 3) for q in range(6, 16)])


# =============================================================================
# helpers
# =============================================================================
def cfg_for(rho=0.0, hard=True, L=1000, rounding="nearest", fatigue=True):
    return FatigueConfig(use_fatigue=fatigue, hard_bound=hard, L=L, rounding=rounding,
                         recover_on_idle=rho > 0, idle_factor=float(rho))


def cfg_tag(c: FatigueConfig) -> str:
    if not c.use_fatigue:
        return "nofat"
    return f"h{int(c.hard_bound)}_L{c.L}_{c.rounding[0]}_r{c.idle_factor if c.recover_on_idle else 0}"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def select(args, default="all"):
    sel = args.instances or default
    pool = {i["id"]: i for i in I.bu_all("adapted")}
    if args.quick:
        sel = ",".join(QUICK)
    if sel == "all":
        return list(pool.values())
    if sel == "subset10":
        return [pool[n] for n in SUBSET10]
    if sel == "layout2":
        return [pool[n] for n in LAYOUT2]
    return [pool[n.strip()] for n in sel.split(",")]


def sched_stats(s: dict, cfg: FatigueConfig) -> dict:
    if not cfg.use_fatigue:
        return {}
    ops = s["ops"].values()
    return dict(n_rest=sum(1 for o in ops if o.get("rest")), rest_time=sum(o.get("rest_dur", 0) for o in ops),
                peak_fatigue=round(max(o["f_end"] for o in ops), 4))


_CP = None


def run_cp(inst, cfg, args, time_limit=None, tag=""):
    """CP-SAT with a global cache (results/cpsat_runs.csv + schedules/cp/*.json)."""
    global _CP
    if _CP is None:
        _CP = CsvLog(RES / "cpsat_runs.csv", ["instance", "cfg", "time_limit", "seed"])
    tl = float(time_limit if time_limit is not None else args.cp_time)
    key = dict(instance=inst["id"] + tag, cfg=cfg_tag(cfg), time_limit=tl, seed=args.cp_seed)
    fpath = RES / "schedules" / "cp" / f"{inst['id']}{tag}__{cfg_tag(cfg)}__{int(tl)}.json"
    row = _CP.get(**key)
    if row is not None:
        sched = load_sched(fpath) if fpath.exists() else None
        return row, sched
    log(f"CP-SAT {inst['id']}{tag} [{cfg_tag(cfg)}] limit={tl:.0f}s")
    hint, ub, ws = None, None, None
    if args.cp_warm_start:
        hint, ub = warm_start(inst, cfg)
        ws = ub
    r = solve(inst, cfg, time_limit=tl, num_workers=args.cp_workers, seed=args.cp_seed, hint=hint, ub=ub)
    sched = r.pop("schedule", None)
    row = dict(key, status=r["status"], cmax=r["cmax"], bound=r["bound"], gap=r["gap"],
               wall_time=r["wall_time"], build_time=r.get("build_time"), conflicts=r.get("conflicts"),
               workers=args.cp_workers, warm_start=ws)
    if sched is not None:
        errs = validate(inst, sched, cfg)
        row["validation_errors"] = len(errs)
        row.update(sched_stats(sched, cfg))
        save_sched(fpath, sched)
    log(f"   -> {row['status']} Cmax={row['cmax']} bound={row['bound']} t={row['wall_time']}s")
    _CP.add(row)
    return _CP.get(**key), sched


def mh_job(spec):
    """One metaheuristic run (top-level function, works with Windows 'spawn')."""
    inst, cfg = spec["inst"], FatigueConfig(**spec["cfg"])
    factory = lambda **kw: Decoder(inst, cfg, **kw)                     # noqa: E731
    r, dec = run_variant(factory, spec["alg"], spec["variant"], spec["seed"], spec["budget"])
    c, st = dec.decode(r["seq"], record=True)
    sched = dec.to_schedule(st, c)
    row = {k: spec[k] for k in spec["key_cols"]}
    row.update(cmax=r["cmax"], time_to_best=r["time_to_best"], total_time=r["total_time"],
               evals=r["evals"], budget=spec["budget"], validation_errors=len(validate(inst, sched, cfg)))
    row.update(sched_stats(sched, cfg))
    if spec.get("continuous"):
        rc = replay(inst, sched, cfg, durations="continuous", rest_policy="planned", timing="keep")
        row.update(cont_cmax=round(rc["cmax"], 4), cont_viol_end=rc["viol_end"])
    return row


def run_mh(specs, csv_path, key_cols, workers):
    logf = CsvLog(csv_path, key_cols)
    todo = [s for s in specs if not logf.has(**{k: s[k] for k in key_cols})]
    for s in todo:
        s["key_cols"] = key_cols
    if not todo:
        return logf
    est = sum(s["budget"] for s in todo) / max(1, workers)
    log(f"{len(todo)} metaheuristic runs on {workers} worker(s), about {est / 3600:.1f} h")
    if workers <= 1:
        for s in todo:
            logf.add(mh_job(s))
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            for n, row in enumerate(pool.imap_unordered(mh_job, todo), 1):
                logf.add(row)
                if n % 20 == 0:
                    log(f"   {n}/{len(todo)} done")
    return logf


def cfg_dict(c: FatigueConfig):
    return dict(use_fatigue=c.use_fatigue, hard_bound=c.hard_bound, L=c.L, rounding=c.rounding,
                recover_on_idle=c.recover_on_idle, idle_factor=c.idle_factor)


def sims(inst, sched1, cfgH):
    """SIM1 / SIM2 variants from a fatigue-agnostic schedule (discrete time, hard bound)."""
    out = {}
    # SIM1: exact reproduction of OPT1 (nominal durations, no rest) with fatigue tracking
    # SIM2: rests enforced when triggered, fatigue-dependent durations, delays propagated
    for name, kw in [("SIM1", dict(rest_policy="none", nominal=True)),
                     ("SIM2", dict(rest_policy="reactive", rest_start="after_pred")),
                     ("SIM2atneed", dict(rest_policy="reactive", rest_start="at_need")),
                     ("SIM2left", dict(rest_policy="reactive", timing="left"))]:
        kw.setdefault("timing", "keep")
        r = replay(inst, sched1, cfgH, durations="discrete", **kw)
        out[f"{name}_cmax"] = r["cmax"]
        if name.startswith("SIM1"):
            out.update({f"{name}_viol_end": r["viol_end"], f"{name}_viol_start": r["viol_start"],
                        f"{name}_peak": r["peak_fatigue"], f"{name}_time_above": r["time_above_fmax"]})
        else:
            out.update({f"{name}_nrest": r["n_rest"], f"{name}_rest_time": r["rest_time"]})
    return out


# =============================================================================
# experiments
# =============================================================================
def e0(args):
    """Regression: legacy semantics must reproduce the submitted Table 3 (Scenario 1, CP-SAT)."""
    import csv
    ref = {r["instance"]: r for r in csv.DictReader(open(Path(__file__).parent / "fajsspt/data/paper_reference.csv"))}
    out = CsvLog(RES / "E0_regression.csv", ["instance"])
    for inst in select(args, default="subset10"):
        if out.has(instance=inst["id"]):
            continue
        r0, _ = run_cp(inst, cfg_for(fatigue=False), args)
        r1, _ = run_cp(inst, cfg_for(hard=False), args)
        out.add(dict(instance=inst["id"], OPT1=r0["cmax"], OPT1_paper=ref[inst["id"]]["OPT1"],
                     OPT2_legacy=r1["cmax"], OPT2_paper=ref[inst["id"]]["S1_CP_cmax"],
                     status=r1["status"], match=int(str(r1["cmax"]) == ref[inst["id"]]["S1_CP_cmax"])))


def e1(args):
    """Experiment 1: OPT1 -> SIM1 / SIM2 -> OPT2 under the hard bound."""
    out = CsvLog(RES / "E1_simulation.csv", ["instance"])
    cfgH = cfg_for()
    for inst in select(args):
        if out.has(instance=inst["id"]):
            continue
        r1, s1 = run_cp(inst, cfg_for(fatigue=False), args)
        r2, _ = run_cp(inst, cfgH, args)
        row = dict(instance=inst["id"], OPT1=r1["cmax"], OPT1_status=r1["status"])
        row.update(sims(inst, s1, cfgH))
        row.update(OPT2=r2["cmax"], OPT2_status=r2["status"], OPT2_bound=r2["bound"],
                   OPT2_nrest=r2.get("n_rest"), OPT2_rest_time=r2.get("rest_time"), OPT2_peak=r2.get("peak_fatigue"))
        out.add(row)


def e2(args):
    """Experiment 2: four idle-recovery settings, CP-SAT + GA/VNS (repeated runs)."""
    insts = select(args)
    out = CsvLog(RES / "E2_cpsat.csv", ["instance", "rho"])
    for rho in args.rhos:
        for inst in insts:
            if out.has(instance=inst["id"], rho=rho):
                continue
            r, _ = run_cp(inst, cfg_for(rho), args)
            out.add(dict(instance=inst["id"], rho=rho, **{k: r.get(k) for k in
                    ("status", "cmax", "bound", "gap", "wall_time", "n_rest", "rest_time", "peak_fatigue",
                     "validation_errors")}))
    specs = [dict(inst=inst, cfg=cfg_dict(cfg_for(rho)), alg=alg, variant="full", seed=s, budget=args.mh_budget,
                  instance=inst["id"], rho=rho, continuous=True)
             for rho in args.rhos for inst in insts for alg in ("GA", "VNS") for s in range(args.seeds)]
    run_mh(specs, RES / "E2_metaheuristics.csv", ["instance", "rho", "alg", "seed", "variant"], args.workers)


def e3(args):
    """Discretisation: fatigue grid L, time resolution (x5, x10), rounding; continuous replay."""
    configs = ([(1000, 1, "nearest")] + [(L, 1, "nearest") for L in (100, 250, 500, 2000)]
               + [(1000, 1, "conservative"), (1000, 5, "nearest"), (1000, 10, "nearest"),
                  (1000, 10, "conservative")])
    if args.quick:
        configs = [(1000, 1, "nearest"), (250, 1, "nearest"), (1000, 1, "conservative"), (1000, 5, "nearest")]
    out = CsvLog(RES / "E3_discretisation.csv", ["instance", "rho", "L", "scale", "rounding"])
    for rho in args.disc_rhos:
        for inst in select(args):
            for L, s, rnd in configs:
                if out.has(instance=inst["id"], rho=rho, L=L, scale=s, rounding=rnd):
                    continue
                ins = I.time_scale(inst, s) if s > 1 else inst
                cfg = cfg_for(rho, L=L, rounding=rnd)
                r, sched = run_cp(ins, cfg, args, time_limit=args.cp_time_short)
                row = dict(instance=inst["id"], rho=rho, L=L, scale=s, rounding=rnd, status=r["status"],
                           cmax=float(r["cmax"]) / s if r["cmax"] not in (None, "", "None") else None,
                           bound=float(r["bound"]) / s if r["bound"] not in (None, "", "None") else None,
                           wall_time=r["wall_time"])
                if sched is not None:
                    for tm in ("keep", "left"):
                        rc = replay(ins, sched, cfg, durations="continuous", rest_policy="planned", timing=tm)
                        row.update({f"cont_{tm}_cmax": round(rc["cmax"] / s, 4), f"cont_{tm}_viol_end": rc["viol_end"],
                                    f"cont_{tm}_peak": rc["peak_fatigue"]})
                    rr = replay(ins, sched, cfg, durations="continuous", rest_policy="reactive", timing="keep")
                    row.update(cont_reactive_cmax=round(rr["cmax"] / s, 4), cont_reactive_viol_end=rr["viol_end"])
                out.add(row)


def _sens_instances(args):
    return select(args, default="layout2")


def e4(args):
    """One-at-a-time sensitivity on lambda, mu, kappa (delta), F_max, F_min."""
    out = CsvLog(RES / "E4_sensitivity.csv", ["instance", "param", "level", "rho"])
    for inst in _sens_instances(args):
        _, s1 = run_cp(inst, cfg_for(fatigue=False), args)
        levels = [("base", None)] + [(p, v) for p, vs in SENS.items() for v in vs]
        for p, v in levels:
            ins = inst if p == "base" else I.with_params(inst, suffix=f"_{p}{v}", **{p: v})
            for rho in args.sens_rhos:
                if out.has(instance=inst["id"], param=p, level=v, rho=rho):
                    continue
                cfg = cfg_for(rho)
                bad = FatigueTables.build(ins, cfg).check_feasible(ins)
                row = dict(instance=inst["id"], param=p, level=v, rho=rho, feasible=int(not bad))
                if not bad:
                    r, _ = run_cp(ins, cfg, args, time_limit=args.cp_time_sens)
                    row.update(OPT2=r["cmax"], status=r["status"], n_rest=r.get("n_rest"),
                               rest_time=r.get("rest_time"))
                    if rho == 0:
                        row.update(sims(ins, s1, cfg))
                out.add(row)


def e5(args):
    """Heterogeneous operators: 5 random draws per instance."""
    out = CsvLog(RES / "E5_heterogeneous.csv", ["instance", "draw"])
    cfgH = cfg_for()
    feas = lambda x: not FatigueTables.build(x, cfgH).check_feasible(x)   # noqa: E731
    specs = []
    for inst in select(args):
        _, s1 = run_cp(inst, cfg_for(fatigue=False), args)
        for d in range(1, args.draws + 1):
            ins = I.heterogeneous(inst, seed=d, feasible=feas)
            if not out.has(instance=inst["id"], draw=d):
                r, _ = run_cp(ins, cfgH, args, time_limit=args.cp_time_short)
                row = dict(instance=inst["id"], draw=d, OPT2=r["cmax"], status=r["status"], bound=r["bound"],
                           n_rest=r.get("n_rest"), rest_time=r.get("rest_time"),
                           params=str({k: v[1:] for k, v in ins["fatigue"].items() if k != "F0"}))
                row.update(sims(ins, s1, cfgH))
                out.add(row)
            specs += [dict(inst=ins, cfg=cfg_dict(cfgH), alg="VNS", variant="full", seed=s, budget=args.mh_budget,
                           instance=inst["id"], draw=d) for s in range(args.seeds_hetero)]
    run_mh(specs, RES / "E5_vns.csv", ["instance", "draw", "alg", "seed", "variant"], args.workers)


def large_instances(args):
    """E6 instances. Default: the 10 Deroussi-Norre instances (routes split into jobs:
    10-16 jobs, 26-42 operations, 8 machines, Table A2 travel times) with 3 and 4 robots."""
    if args.large_set == "dn":
        sets = range(1, 11) if not args.quick else (5,)
        return [I.dn_instance(js, R, scale_mu=args.dn_scaling == "both") for R in args.large_robots for js in sets]
    out = []
    names = LARGE if not args.quick else [("ft06", 2), ("la01", 2)]
    for name, R in names:
        jobs = I.parse_jsplib(Path(args.jsplib_dir) / name)
        M = max(k for job in jobs for k, _ in job)
        csvp = Path(args.travel_dir) / f"M{M}.csv" if args.travel_dir else None
        out.append(I.jsplib_instance(name, args.jsplib_dir, R, travel_csv=csvp if csvp and csvp.exists() else None,
                                     layout_seed=M, layout=args.large_layout))
    return out


def e6(args):
    """Larger instances (JSPLIB job data + robots + fatigue)."""
    out = CsvLog(RES / "E6_large_cpsat.csv", ["instance", "rho"])
    insts = large_instances(args)
    specs = []
    for inst in insts:
        I.save(inst, RES / "instances" / f"{inst['id']}.json")
        r1, s1 = run_cp(inst, cfg_for(fatigue=False), args, time_limit=args.cp_time_large)
        for rho in args.large_rhos:
            cfg = cfg_for(rho)
            if not out.has(instance=inst["id"], rho=rho):
                bad = FatigueTables.build(inst, cfg).check_feasible(inst)
                row = dict(instance=inst["id"], rho=rho, n_jobs=len(inst["jobs"]), M=inst["M"], R=inst["R"],
                           n_ops=I.n_ops(inst), minutes_per_unit=round(inst["minutes_per_unit"], 4),
                           travel=inst["meta"]["travel"], OPT1=r1["cmax"], OPT1_status=r1["status"],
                           OPT1_bound=r1["bound"], feasible=int(not bad))
                if not bad:
                    r2, _ = run_cp(inst, cfg, args, time_limit=args.cp_time_large)
                    row.update(OPT2=r2["cmax"], status=r2["status"], bound=r2["bound"], gap=r2["gap"],
                               wall_time=r2["wall_time"], n_rest=r2.get("n_rest"))
                    if rho == 0 and s1 is not None:
                        row.update(sims(inst, s1, cfg))
                out.add(row)
            budget = max(args.mh_budget, args.mh_budget_per_op * I.n_ops(inst))
            specs += [dict(inst=inst, cfg=cfg_dict(cfg), alg=alg, variant="full", seed=s, budget=budget,
                           instance=inst["id"], rho=rho) for alg in ("GA", "VNS") for s in range(args.seeds_large)]
    run_mh(specs, RES / "E6_large_metaheuristics.csv", ["instance", "rho", "alg", "seed", "variant"], args.workers)


def e7(args):
    """Classical (fatigue-free) original BU benchmark vs published values."""
    insts = I.bu_all("original")
    if args.quick:
        insts = insts[:3]
    elif args.instances and args.instances != "all":
        keep = set(args.instances.split(","))
        insts = [i for i in insts if i["id"] in keep]
    cfg = cfg_for(fatigue=False)
    out = CsvLog(RES / "E7_original_bu_cpsat.csv", ["instance"])
    for inst in insts:
        if not out.has(instance=inst["id"]):
            r, _ = run_cp(inst, cfg, args, tag="_orig")
            out.add(dict(instance=inst["id"], cmax=r["cmax"], bound=r["bound"], status=r["status"],
                         wall_time=r["wall_time"]))
    specs = [dict(inst=inst, cfg=cfg_dict(cfg), alg=alg, variant=v, seed=s, budget=args.mh_budget,
                  instance=inst["id"]) for inst in insts for alg in ("GA", "VNS") for v in ("full", "legacy")
             for s in range(args.seeds_bu)]
    run_mh(specs, RES / "E7_original_bu_metaheuristics.csv", ["instance", "alg", "variant", "seed"], args.workers)


def e8(args):
    """Ablation of the GA/VNS improvements."""
    specs = [dict(inst=inst, cfg=cfg_dict(cfg_for(rho)), alg=alg, variant=v, seed=s, budget=args.mh_budget,
                  instance=inst["id"], rho=rho)
             for rho in args.ablation_rhos for inst in select(args)
             for alg in ("GA", "VNS") for v in ("legacy", "I1", "I1+I2", "I1+I3", "I1+I2+I3", "full")
             for s in range(args.seeds_ablation)]
    run_mh(specs, RES / "E8_ablation.csv", ["instance", "rho", "alg", "variant", "seed"], args.workers)


EXPS = dict(e0=e0, e1=e1, e2=e2, e3=e3, e4=e4, e5=e5, e6=e6, e7=e7, e8=e8)
PRIORITY = ["e0", "e1", "e2", "e7", "e4", "e3", "e8", "e5", "e6"]   # order used by "all"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp", nargs="+", choices=list(EXPS) + ["all"],
                    help="one or several experiments, executed one after the other (queue)")
    ap.add_argument("--quick", action="store_true", help="smoke test: 3 instances, short limits")
    ap.add_argument("--instances", default=None, help="'all', 'subset10' or comma list (EX11,EX82,...)")
    ap.add_argument("--results", default="results")
    ap.add_argument("--cp-time", type=float, default=3600, help="CP-SAT limit (E1, E2, E7)")
    ap.add_argument("--cp-time-short", type=float, default=600, help="CP-SAT limit (E3, E4, E5)")
    ap.add_argument("--cp-time-sens", type=float, default=300, help="CP-SAT limit (E4)")
    ap.add_argument("--cp-time-large", type=float, default=1800, help="CP-SAT limit (E6)")
    ap.add_argument("--no-cp-warm-start", dest="cp_warm_start", action="store_false",
                    help="do not give CP-SAT the best dispatching-rule schedule as hint/upper bound")
    ap.add_argument("--cp-workers", type=int, default=os.cpu_count() or 8,
                    help="CP-SAT threads (default: all logical cores)")
    ap.add_argument("--cp-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                    help="parallel single-threaded GA/VNS runs (default: physical cores). CP-SAT and "
                         "metaheuristic phases never overlap")
    ap.add_argument("--mh-budget", type=float, default=15.0, help="wall-clock seconds per GA/VNS run")
    ap.add_argument("--mh-budget-per-op", type=float, default=0.5, help="E6: budget = max(mh-budget, x * #ops)")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seeds-large", type=int, default=10)
    ap.add_argument("--seeds-ablation", type=int, default=3)
    ap.add_argument("--seeds-bu", type=int, default=10, help="E7 seeds")
    ap.add_argument("--seeds-hetero", type=int, default=5, help="E5 seeds")
    ap.add_argument("--disc-rhos", type=float, nargs="+", default=[0.0], help="E3 idle factors")
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--rhos", type=float, nargs="+", default=RHOS)
    ap.add_argument("--sens-rhos", type=float, nargs="+", default=[0.0, 0.25])
    ap.add_argument("--large-rhos", type=float, nargs="+", default=[0.0])
    ap.add_argument("--ablation-rhos", type=float, nargs="+", default=[0.0, 0.25])
    ap.add_argument("--jsplib-dir", default=str(Path(__file__).parent / "fajsspt/data/jsplib"))
    ap.add_argument("--travel-dir", default=None, help="optional folder with M5.csv, M6.csv ... (own layouts)")
    ap.add_argument("--large-set", choices=["dn", "jsplib"], default="dn",
                    help="E6 instances: Deroussi-Norre fjsp1-10 with split routes (default) or JSPLIB")
    ap.add_argument("--large-robots", type=int, nargs="+", default=[3, 4], help="E6 fleet sizes (DN set)")
    ap.add_argument("--dn-scaling", choices=["both", "lambda"], default="both",
                    help="DN instances: halve lambda and mu (default) or lambda only")
    ap.add_argument("--large-layout", choices=["dn", "generated"], default="dn",
                    help="E6 travel times: Deroussi-Norre published layout (default) or generated")
    args = ap.parse_args()
    global RES
    RES = Path(args.results)
    if args.quick:
        args.cp_time = args.cp_time_short = args.cp_time_large = args.cp_time_sens = 60
        args.seeds = args.seeds_large = args.seeds_ablation = args.seeds_bu = args.seeds_hetero = 2
        args.disc_rhos = [0.0, 0.25]
        args.draws, args.mh_budget, args.mh_budget_per_op = 1, 2.0, 0.02
        args.rhos, args.sens_rhos, args.ablation_rhos = [0.0, 0.25], [0.0], [0.0]
        args.large_robots = [3]
        RES = Path(args.results) / "quick"
        SENS.update({k: v[:1] for k, v in SENS.items()})
    RES.mkdir(parents=True, exist_ok=True)
    queue = PRIORITY if "all" in args.exp else args.exp
    for name in queue:
        log(f"===== {name.upper()} =====")
        EXPS[name](args)
    log("done")


if __name__ == "__main__":
    main()
