# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 10:56:42 2026

@author: Kader SANOGO

cp_sat_solver_jsspt_hf.py

CP-SAT solver for JSSPT-HF with:
- fatigue discretization (L integer states)
- lookup tables for:
    * processing-time inflation tau'(F_start)
    * fatigue accumulation F_end = g(F_start, tau'(F_start))
    * rest time needed to recover from fatigue F to Fmin
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from typing import Dict, List, Tuple, Optional

from ortools.sat.python import cp_model
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# ---------------------------------------------------------------------------
# Helpers for KPIs computing
# ---------------------------------------------------------------------------

def kpi_makespan(machine_sched, robot_sched):
    end_times = [e for (_, _, _, e) in machine_sched] + \
                [e for (_, _, _, e) in robot_sched]
    return max(end_times)

def kpi_machine_utilization(machine_sched, Cmax, M):
    busy = defaultdict(float)
    for m, _, s, e in machine_sched:
        busy[m] += (e - s)
    U = np.array([busy[m]/Cmax for m in range(1, M+1)])
    return U.mean(), U

def kpi_robot_utilization(robot_sched, Cmax, R):
    busy = defaultdict(float)
    for r, _, s, e in robot_sched:
        busy[r] += (e - s)
    U = np.array([busy[r]/Cmax for r in range(1, R+1)])
    return U.mean(), U

def kpi_fatigue_violations_cp(fatigue_log, Fmax_per_machine):
    """
    Count processing intervals where Fs >= Fmax(machine).
    fatigue_log: (m, s, e, Fs, Fe, is_rest) with Fs in [0,1]
    Fmax_per_machine: dict m -> float threshold
    """
    viol = 0
    for (m, _, _, Fs, _, seg_type) in fatigue_log:
        if seg_type != 0:
            continue
        if Fs >= Fmax_per_machine[m]:
            viol += 1
    return viol


def kpi_max_fatigue(fatigue_log):
    """
    Maximum fatigue observed over all work and rest intervals.
    """
    if not fatigue_log:
        return 0.0
    maxF = max(max(Fs, Fe) for (_, _, _, Fs, Fe, _) in fatigue_log)
    return maxF

def kpi_fatigue_accumulation_growth(fatigue_log):
    """
    Sum of (F_end - F_start) over processing intervals (not rest).
    """
    total = 0.0
    for (_, _, _, Fs, Fe, seg_type) in fatigue_log:
        if seg_type == 0:  # only count work intervals
            total += max(0.0, Fe - Fs)
    return total

def kpi_num_rest_breaks(fatigue_log):
    """
    Count number of explicit rest intervals.
    """
    return sum(1 for (_, _, _, _, _, seg_type) in fatigue_log if seg_type == 1)

def kpi_total_rest(fatigue_log):
    return sum((e - s) for (_, s, e, Fs, Fe, seg_type) in fatigue_log if seg_type in [1, 2])

def kpi_fatigue_accumulation(fatigue_log):
    return sum(Fs * (e - s) for (_, s, e, Fs, Fe, seg_type) in fatigue_log if seg_type == 0)


# ---------------------------------------------------------------------------
# Instance loading
# ---------------------------------------------------------------------------

def extract_fatigue_and_delta(instance: Dict):
    """Read fatigue + delta modes from the instance meta block."""
    M = instance["machines_nb"]
    meta = instance.get("meta", {})
    fat = meta.get("fatigue", {})
    delta_meta = meta.get("delta", {})

    mode = fat.get("mode", "uniform")
    if mode == "uniform":
        Fmin = [None] + [float(fat["F_min"])] * M
        Fmax = [None] + [float(fat["F_max"])] * M
        lam  = [None] + [float(fat["lambda"])] * M
        mu   = [None] + [float(fat["mu"])] * M
    elif mode == "per_human":
        # lists are length M (in your generator), map to 1..M
        Fmin = [None] + [float(x) for x in fat["F_min"]]
        Fmax = [None] + [float(x) for x in fat["F_max"]]
        lam  = [None] + [float(x) for x in fat["lambda"]]
        mu   = [None] + [float(x) for x in fat["mu"]]
    else:
        raise ValueError(f"Unknown fatigue mode: {mode}")

    # delta: S1 uniform, S2 per-op field
    delta_mode = delta_meta.get("mode", "uniform")
    delta_uniform = None
    if delta_mode == "uniform":
        delta_uniform = float(delta_meta["value"])

    return Fmin, Fmax, lam, mu, delta_mode, delta_uniform


def to_int_fatigue_params(Fmin, Fmax, L=1000):
    """Convert fatigue thresholds to integer grid 0..L."""
    fmin = [None]
    fmax = [None]
    for k in range(1, len(Fmin)):
        fmin.append(int(math.floor(L * Fmin[k])))  # round down
        fmax.append(int(math.ceil(L * Fmax[k])))   # round up
    return fmin, fmax


def round_half_up(x: float) -> int:
    """Standard rounding: fractional part >= .5 rounds up."""
    return int(math.floor(x + 0.5))


def build_rest_time_table(mu_k: float, Fmin_k: float, fmax_int_k: int, L=1000):
    """
    Precompute rest duration (integer time units) needed to recover from F to Fmin.
    """
    if Fmin_k <= 0.0:
        raise ValueError("Fmin_k must be > 0 for logarithmic rest duration.")

    table = [0] * (L + 1)
    for f in range(L + 1):
        if f < fmax_int_k:
            table[f] = 0
            continue

        F = f / L
        if F <= Fmin_k:
            table[f] = 0
            continue

        t = (1.0 / mu_k) * math.log(F / Fmin_k)
        table[f] = int(math.ceil(t))  # conservative
    return table


def build_proc_and_fatigue_tables(lam_k: float, delta_ij: float, tau: int, L=1000):
    """
    For each discrete fatigue state fS in 0..L:
      - compute inflated duration tau'(fS):
            tau' = tau * (1 + delta * lam * ln(1+F))
      - compute fatigue after working tau'(fS):
            F_end = 1 - (1 - F) * exp(-lam * tau')

    Returns:
      proc_table[fS] = tau'(fS) (int)
      fat_table[fS]  = fC (int in 0..L)
    """
    proc_table = [0] * (L + 1)
    fat_table = [0] * (L + 1)

    for f in range(L + 1):
        F = f / L

        infl = 1.0 + delta_ij * lam_k * math.log(1.0 + F)
        tau_p = tau * infl

        tau_p_int = max(1, round_half_up(tau_p))
        proc_table[f] = tau_p_int

        Fp = 1.0 - (1.0 - F) * math.exp(-lam_k * tau_p_int)
        fp = int(round(L * Fp))
        fp = max(0, min(L, fp))
        fat_table[f] = fp

    return proc_table, fat_table


def load_jsspt_instance(path: str) -> Dict:
    """Load one JSON instance and enrich it with fatigue integer parameters."""
    with open(path, "r") as f:
        data = json.load(f)

    L = 1000
    Fmin, Fmax, lam, mu, delta_mode, delta_uniform = extract_fatigue_and_delta(data)
    fmin_int, fmax_int = to_int_fatigue_params(Fmin, Fmax, L=L)

    data["_fatigue"] = {
        "L": L,
        "Fmin": Fmin,   # float, indexed 1..M
        "Fmax": Fmax,   # float, indexed 1..M
        "lambda": lam,  # float, indexed 1..M
        "mu": mu,       # float, indexed 1..M
        "fmin": fmin_int,  # int, indexed 1..M
        "fmax": fmax_int,  # int, indexed 1..M
        "delta_mode": delta_mode,
        "delta_uniform": delta_uniform,
    }
    return data


def sort_files(directory, save_index=False, index_filename="index.json"):
    """Sort BU instance files by subindex then experiment number."""
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(".json")
    ]
    pattern = re.compile(r'EX(\d+)_([0-9]+)_')

    def sort_key(path):
        match = pattern.search(os.path.basename(path))
        if match:
            experiment = int(match.group(1))
            subindex = int(match.group(2))
            return (subindex, experiment)
        return (9999, 9999)

    sorted_files = sorted(files, key=sort_key)

    if save_index:
        index_path = os.path.join(directory, index_filename)
        with open(index_path, "w") as f:
            json.dump(sorted_files, f, indent=4)
        print(f"✔ Index saved: {index_path}")

    return sorted_files


# ---------------------------------------------------------------------------
# CP-SAT solver
# ---------------------------------------------------------------------------

def solve_jsspt_hf_cp(
    instance: Dict,
    last_ttask: int = 1,
    num_robots: Optional[int] = None,
    time_limit: float = 600.0,
    num_workers: int = 8,
    make_gantt: bool = True,
    to_plot: bool = False,
    gantt_path: Optional[str] = None,
    recover_on_idle: Optional[bool] = False,
    idle_recovery_factor: float = 1.0,
    # Safety trigger robustness: unsafe if fc_norest >= (L - eps_exhaust_int)
    eps_exhaust_int: int = 1,
    # If True, prints post-solve fatigue propagation on each machine
    debug_fatigue: bool = False,
) -> Tuple[str, Optional[int], float, Dict]:

    inst_id = instance.get("instance_id", "unknown_instance")
    layout_id = instance.get("layout_id", -1)
    M = instance["machines_nb"]
    sigma = instance["sigma"]  # (M+1)x(M+1), index 0=Z, 1..M=machines
    jobs_data = instance["jobs"]

    # Fatigue details
    fat = instance["_fatigue"]
    L = fat["L"]
    fmin_int = fat["fmin"]
    fmax_int = fat["fmax"]
    lam = fat["lambda"]
    mu = fat["mu"]
    Fmin = fat["Fmin"]

    if num_robots is None:
        R = instance.get("robots_nb", 1)
    else:
        R = num_robots

    # -------------------------
    # Build operation data
    # -------------------------
    J = len(jobs_data)
    op_duration: Dict[Tuple[int, int], int] = {}
    op_machine: Dict[Tuple[int, int], int] = {}
    op_delta: Dict[Tuple[int, int], float] = {}
    job_n_ops: List[int] = []

    total_proc_time_ub = 0

    for i, job in enumerate(jobs_data):
        n_ops = job["n_ops"]
        job_n_ops.append(n_ops)
        for j, op in enumerate(job["ops"]):
            m_idx = int(op["machine_index"])  # 1..M
            tau = int(op["duration"])

            if fat["delta_mode"] == "uniform":
                delt = float(fat["delta_uniform"])
            else:
                delt = float(op.get("delta", 0.0))

            op_machine[(i, j)] = m_idx
            op_duration[(i, j)] = tau
            op_delta[(i, j)] = delt

            ub = tau * (1.0 + delt * lam[m_idx] * math.log(2.0))
            total_proc_time_ub += int(math.ceil(ub))

    # -------------------------
    # Build transport tasks
    # -------------------------
    t_job: List[int] = []
    t_op: List[int] = []
    t_pickup: List[int] = []
    t_drop: List[int] = []
    t_duration: List[int] = []

    trans_of_op: Dict[Tuple[int, int], int] = {}
    trans_of_ret: Dict[int, int] = {}

    for i in range(J):
        n_ops = job_n_ops[i]
        for j in range(n_ops):
            pickup = 0 if j == 0 else op_machine[(i, j - 1)]
            drop = op_machine[(i, j)]
            dur = int(sigma[pickup][drop])

            idx = len(t_job)
            t_job.append(i)
            t_op.append(j)
            t_pickup.append(pickup)
            t_drop.append(drop)
            t_duration.append(dur)
            trans_of_op[(i, j)] = idx

        if last_ttask == 1:
            pickup = op_machine[(i, n_ops - 1)]
            drop = 0
            dur = int(sigma[pickup][drop])

            idx = len(t_job)
            t_job.append(i)
            t_op.append(-1)
            t_pickup.append(pickup)
            t_drop.append(drop)
            t_duration.append(dur)
            trans_of_ret[i] = idx

    T = len(t_job)
    max_sigma = max(max(row) for row in sigma)
    horizon = total_proc_time_ub + T * max_sigma + 10

    # -------------------------
    # CP-SAT model
    # -------------------------
    model = cp_model.CpModel()

    # ----- caching lookup tables -----
    op_tables = {}        # key (m, tau, delta) -> (proc_table, fat_table)
    rest_time_table = {}  # key m -> list length L+1

    def get_op_tables(m: int, tau: int, delta_ij: float):
        key = (m, tau, delta_ij)
        if key not in op_tables:
            op_tables[key] = build_proc_and_fatigue_tables(lam[m], delta_ij, tau, L=L)
        return op_tables[key]

    def get_rest_table(m: int):
        if m not in rest_time_table:
            rest_time_table[m] = build_rest_time_table(mu[m], Fmin[m], fmax_int[m], L=L)
        return rest_time_table[m]

    # PRECISION factor for log/exp space (1000 = 0.001 precision)
    K = 1000
    OFFSET = 20 * K # To keep exp indices positive

    # ln_table[f] = round(K * ln(max(1, f) / L))
    ln_table = [round(K * math.log(max(1, f) / L)) for f in range(L + 1)]

    # exp_table[idx] = round(L * exp((idx - OFFSET) / K))
    # Range: needs to cover ln(1/L) to 0. ln(1/100) approx -4.6, ln(1/1000) approx -6.9.
    # OFFSET + (K * ln(L)) is 0 index. Let's make it broad.
    exp_table_size = 25 * K
    exp_table = [0] * exp_table_size
    for idx in range(exp_table_size):
        val = L * math.exp((idx - OFFSET) / K)
        exp_table[idx] = min(L, round(val))

    # ----- operation variables -----
    op_start: Dict[Tuple[int, int], cp_model.IntVar] = {}
    op_end: Dict[Tuple[int, int], cp_model.IntVar] = {}
    op_interval: Dict[Tuple[int, int], cp_model.IntervalVar] = {}

    op_fstart: Dict[Tuple[int, int], cp_model.IntVar] = {}
    op_fend: Dict[Tuple[int, int], cp_model.IntVar] = {}

    op_rest_before: Dict[Tuple[int, int], cp_model.BoolVar] = {}
    op_rest_dur: Dict[Tuple[int, int], cp_model.IntVar] = {}
    op_pred_end: Dict[Tuple[int, int], cp_model.IntVar] = {}
    op_pred_fend: Dict[Tuple[int, int], cp_model.IntVar] = {}
    f_start_idle: Dict[Tuple[int, int], cp_model.IntVar] = {}

    for i in range(J):
        for j in range(job_n_ops[i]):
            o = (i, j)
            m_idx = op_machine[o]
            tau = op_duration[o]
            delta_ij = op_delta[o]

            start = model.NewIntVar(0, horizon, f"S_{i}_{j}")
            end = model.NewIntVar(0, horizon, f"C_{i}_{j}")

            fS = model.NewIntVar(0, L, f"fS_{i}_{j}")
            fC = model.NewIntVar(0, L, f"fC_{i}_{j}")

            rb = model.NewBoolVar(f"restB_{i}_{j}")
            op_rest_before[o] = rb

            proc_table, fat_table = get_op_tables(m_idx, tau, delta_ij)

            proc_dur = model.NewIntVar(min(proc_table), max(proc_table), f"P_{i}_{j}")
            model.AddElement(fS, proc_table, proc_dur)

            model.Add(end == start + proc_dur)
            model.AddElement(fS, fat_table, fC)

            interval = model.NewIntervalVar(start, proc_dur, end, f"I_{i}_{j}")

            op_start[o] = start
            op_end[o] = end
            op_interval[o] = interval
            op_fstart[o] = fS
            op_fend[o] = fC

    # Prepare for plotting (needed for idle calculation if requested)
    meta = instance.get("meta", {})
    rec_on_idle = bool(meta.get("recover_on_idle", False)) if recover_on_idle is None else bool(recover_on_idle)
    irf = float(meta.get("idle_recovery_factor", 1.0)) if idle_recovery_factor == 1.0 else idle_recovery_factor

    # ---------------------------------------------
    # Machine sequencing + fatigue/rest propagation
    # --------------------------------------------
    ops_on_m = {m: [] for m in range(1, M + 1)}
    for i in range(J):
        for j in range(job_n_ops[i]):
            o = (i, j)
            ops_on_m[op_machine[o]].append(o)

    SRC = (-1, -1)  # dummy source

    # Store the predecessor arcs for debug extraction
    pred_arc_by_m: Dict[int, Dict[Tuple[Tuple[int, int], Tuple[int, int]], cp_model.BoolVar]] = {}

    for m in range(1, M + 1):
        ops = ops_on_m[m]
        n = len(ops)
        if n == 0:
            continue

        if n == 1:
            b = ops[0]
            op_pred_end[b] = model.NewIntVar(0, 0, f"predEnd_single_m{m}_i{b[0]}_j{b[1]}")
            op_pred_fend[b] = model.NewIntVar(fmin_int[m], fmin_int[m], f"predFend_single_m{m}_i{b[0]}_j{b[1]}")
            model.Add(op_fstart[b] == fmin_int[m])
            model.Add(op_rest_before[b] == 0)
            op_rest_dur[b] = model.NewIntVar(0, 0, f"restDur_single_m{m}_i{b[0]}_j{b[1]}")
            f_start_idle[b] = model.NewIntVar(fmin_int[m], fmin_int[m], f"fStartIdle_single_m{m}_i{b[0]}_j{b[1]}")
            model.Add(op_start[b] >= 0)
            continue

        pred: Dict[Tuple[Tuple[int, int], Tuple[int, int]], cp_model.BoolVar] = {}
        pred_arc_by_m[m] = pred

        # SRC -> b arcs
        for b in ops:
            pred[(SRC, b)] = model.NewBoolVar(f"pred_SRC_m{m}_i{b[0]}_j{b[1]}")

        # a -> b arcs
        for a in ops:
            for b in ops:
                if a == b:
                    continue
                pred[(a, b)] = model.NewBoolVar(
                    f"pred_m{m}_a{a[0]}_{a[1]}_b{b[0]}_{b[1]}"
                )

        # a -> SINK
        succ_sink = {a: model.NewBoolVar(f"succ_SINK_m{m}_i{a[0]}_j{a[1]}") for a in ops}

        # ---- chain constraints ----
        model.Add(sum(pred[(SRC, b)] for b in ops) == 1)

        for b in ops:
            model.Add(pred[(SRC, b)] + sum(pred[(a, b)] for a in ops if a != b) == 1)

        for a in ops:
            model.Add(sum(pred[(a, b)] for b in ops if b != a) + succ_sink[a] == 1)

        model.Add(sum(succ_sink[a] for a in ops) == 1)

        # ---- predecessor-derived (end time, fatigue) for each b ----
        pred_end = {}
        pred_fend = {}
        
        for b in ops:
            pred_end[b] = model.NewIntVar(0, horizon, f"predEnd_m{m}_i{b[0]}_j{b[1]}")
            pred_fend[b] = model.NewIntVar(0, L, f"predFend_m{m}_i{b[0]}_j{b[1]}")
            f_start_idle[b] = model.NewIntVar(0, L, f"fStartIdle_m{m}_i{b[0]}_j{b[1]}")

            # Case: First op on machine
            model.Add(pred_end[b] == 0).OnlyEnforceIf(pred[(SRC, b)])
            model.Add(pred_fend[b] == fmin_int[m]).OnlyEnforceIf(pred[(SRC, b)])
            model.Add(f_start_idle[b] == fmin_int[m]).OnlyEnforceIf(pred[(SRC, b)])

            for a in ops:
                if a == b:
                    continue
                lit = pred[(a, b)]
                model.Add(pred_end[b] == op_end[a]).OnlyEnforceIf(lit)
                model.Add(pred_fend[b] == op_fend[a]).OnlyEnforceIf(lit)

            if recover_on_idle:
                # log_f_end_b = ln_table[pred_fend[b]]
                log_f_end_b = model.NewIntVar(-OFFSET, 0, f"logFEnd_m{m}_b{b[0]}_{b[1]}")
                model.AddElement(pred_fend[b], ln_table, log_f_end_b)

                gap_b = model.NewIntVar(0, horizon, f"gap_m{m}_b{b[0]}_{b[1]}")
                model.Add(gap_b == op_start[b] - pred_end[b])

                mu_scaled = int(round(mu[m] * idle_recovery_factor * K))
                decay_b = model.NewIntVar(0, horizon * K, f"decay_m{m}_b{b[0]}_{b[1]}")
                model.Add(decay_b == gap_b * mu_scaled)

                log_f_start_b = model.NewIntVar(-OFFSET - (horizon * K), 0, f"logFStart_m{m}_b{b[0]}_{b[1]}")
                model.Add(log_f_start_b == log_f_end_b - decay_b)

                log_f_start_idx_b = model.NewIntVar(-horizon * K, exp_table_size - 1, f"logFIdx_m{m}_b{b[0]}_{b[1]}")
                model.Add(log_f_start_idx_b == log_f_start_b + OFFSET)

                log_f_start_clamped_b = model.NewIntVar(0, exp_table_size - 1, f"logFClamped_m{m}_b{b[0]}_{b[1]}")
                model.AddMaxEquality(log_f_start_clamped_b, [log_f_start_idx_b, 0])

                val_idle_b = model.NewIntVar(0, L, f"valIdle_m{m}_b{b[0]}_{b[1]}")
                model.AddElement(log_f_start_clamped_b, exp_table, val_idle_b)
                
                # f_start_idle[b] = max(val_idle_b, fmin_int[m])
                model.AddMaxEquality(f_start_idle[b], [val_idle_b, fmin_int[m]])
            else:
                model.Add(f_start_idle[b] == pred_fend[b])

            op_pred_end[b] = pred_end[b]
            op_pred_fend[b] = pred_fend[b]

        # ---- rest duration lookup from propagated fatigue ----
        rest_req = {}
        for b in ops:
            rest_req[b] = model.NewIntVar(0, horizon, f"restReq_m{m}_i{b[0]}_j{b[1]}")
            model.AddElement(f_start_idle[b], get_rest_table(m), rest_req[b])

            op_rest_dur[b] = model.NewIntVar(0, horizon, f"restDur_m{m}_i{b[0]}_j{b[1]}")
            model.Add(op_rest_dur[b] == 0).OnlyEnforceIf(op_rest_before[b].Not())
            model.Add(op_rest_dur[b] == rest_req[b]).OnlyEnforceIf(op_rest_before[b])

        # ---- timing + fatigue start ----
        for b in ops:
            model.Add(op_start[b] >= pred_end[b] + op_rest_dur[b])
            model.Add(op_fstart[b] == fmin_int[m]).OnlyEnforceIf(op_rest_before[b])
            model.Add(op_fstart[b] == f_start_idle[b]).OnlyEnforceIf(op_rest_before[b].Not())

        # ---- mandatory rest triggers (A/B) ----
        erg_lit = {}
        trigB_lit = {}

        for b in ops:
            # Rule A (ergonomics): if decayed fatigue >= Fmax => rest mandatory
            erg = model.NewBoolVar(f"erg_m{m}_i{b[0]}_j{b[1]}")
            erg_lit[b] = erg
            model.Add(f_start_idle[b] >= fmax_int[m]).OnlyEnforceIf(erg)
            model.Add(f_start_idle[b] <= fmax_int[m] - 1).OnlyEnforceIf(erg.Not())
            model.AddImplication(erg, op_rest_before[b])

            # Rule B (safety): f_start_idle < Fmax AND running op without rest would reach exhaustion
            tau_b = op_duration[b]
            delta_b = op_delta[b]
            _, fat_table_b = get_op_tables(m, tau_b, delta_b)

            fc_norest = model.NewIntVar(0, L, f"fCNoRest_m{m}_i{b[0]}_j{b[1]}")
            model.AddElement(f_start_idle[b], fat_table_b, fc_norest)

            # Robust unsafe: >= (L - eps)
            thr = max(0, L - int(eps_exhaust_int))
            unsafe = model.NewBoolVar(f"unsafe_m{m}_i{b[0]}_j{b[1]}")
            model.Add(fc_norest >= thr).OnlyEnforceIf(unsafe)
            model.Add(fc_norest <= thr - 1).OnlyEnforceIf(unsafe.Not())

            low = model.NewBoolVar(f"low_m{m}_i{b[0]}_j{b[1]}")
            model.Add(f_start_idle[b] <= fmax_int[m] - 1).OnlyEnforceIf(low)
            model.Add(f_start_idle[b] >= fmax_int[m]).OnlyEnforceIf(low.Not())

            trigB = model.NewBoolVar(f"trigB_m{m}_i{b[0]}_j{b[1]}")
            trigB_lit[b] = trigB
            model.AddBoolAnd([low, unsafe]).OnlyEnforceIf(trigB)
            model.AddBoolOr([low.Not(), unsafe.Not(), trigB])
            model.AddImplication(trigB, op_rest_before[b])

        # Forbid voluntary/free rest (rest iff triggered)
        for b in ops:
            model.AddBoolOr([erg_lit[b], trigB_lit[b]]).OnlyEnforceIf(op_rest_before[b])
            # Forward direction already ensured by:
            # erg -> rest_before and trigB -> rest_before

    # -------------------------
    # Transport variables
    # -------------------------
    t_start: List[cp_model.IntVar] = []
    t_end: List[cp_model.IntVar] = []
    t_interval: List[cp_model.IntervalVar] = []

    for t in range(T):
        dur = t_duration[t]
        s = model.NewIntVar(0, horizon, f"TS_{t}")
        e = model.NewIntVar(0, horizon, f"TC_{t}")
        model.Add(e == s + dur)
        iv = model.NewIntervalVar(s, dur, e, f"TI_{t}")
        t_start.append(s)
        t_end.append(e)
        t_interval.append(iv)

    # Robot assignment
    robot_assign: List[List[cp_model.BoolVar]] = []
    for r in range(R):
        row = []
        for t in range(T):
            row.append(model.NewBoolVar(f"r{r}_t{t}"))
        robot_assign.append(row)

    for t in range(T):
        model.Add(sum(robot_assign[r][t] for r in range(R)) == 1)

    # -------------------------
    # Precedence: transport vs processing (job routing)
    # -------------------------
    for i in range(J):
        n_ops = job_n_ops[i]
        for j in range(n_ops):
            o = (i, j)
            t_idx = trans_of_op[o]

            model.Add(op_start[o] >= t_end[t_idx])

            # next transport cannot start before this processing ends
            if j > 0:
                model.Add(t_start[t_idx] >= op_end[(i, j - 1)])

        if last_ttask == 1:
            t_ret = trans_of_ret[i]
            model.Add(t_start[t_ret] >= op_end[(i, n_ops - 1)])

    # -------------------------
    # Robots: initial positioning + empty travel sequencing
    # -------------------------
    bigM = horizon

    for r in range(R):
        # Initial positioning
        for t in range(T):
            pickup = t_pickup[t]
            travel0 = int(sigma[0][pickup])
            model.Add(t_start[t] >= travel0 * robot_assign[r][t])

        for t1 in range(T):
            for t2 in range(t1 + 1, T):
                z = model.NewBoolVar(f"both_r{r}_t{t1}_t{t2}")
                model.Add(z <= robot_assign[r][t1])
                model.Add(z <= robot_assign[r][t2])
                model.Add(z >= robot_assign[r][t1] + robot_assign[r][t2] - 1)

                y12 = model.NewBoolVar(f"y_r{r}_t{t1}_t{t2}")
                y21 = model.NewBoolVar(f"y_r{r}_t{t2}_t{t1}")
                model.Add(y12 <= z)
                model.Add(y21 <= z)
                model.Add(y12 + y21 == z)

                drop1 = t_drop[t1]
                pick2 = t_pickup[t2]
                drop2 = t_drop[t2]
                pick1 = t_pickup[t1]
                empty_12 = int(sigma[drop1][pick2])
                empty_21 = int(sigma[drop2][pick1])

                model.Add(t_start[t2] >= t_end[t1] + empty_12 - bigM * (1 - y12))
                model.Add(t_start[t1] >= t_end[t2] + empty_21 - bigM * (1 - y21))

    # -------------------------
    # Makespan
    # -------------------------
    job_completion = []
    for i in range(J):
        if last_ttask == 1:
            job_completion.append(t_end[trans_of_ret[i]])
        else:
            job_completion.append(op_end[(i, job_n_ops[i] - 1)])

    C_max = model.NewIntVar(0, horizon, "Cmax")
    model.AddMaxEquality(C_max, job_completion)
    model.Minimize(C_max)

    # -------------------------
    # Solve
    # -------------------------
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = int(num_workers)

    print(f"Solving instance {inst_id} with {R} robots, last_ttask={last_ttask}...")
    start_time = time.time()
    status = solver.Solve(model)
    cpu_time = time.time() - start_time

    status_map = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.UNKNOWN: "UNKNOWN",
    }
    status_str = status_map.get(status, "UNKNOWN")

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"  Status={status_str}, CPU={cpu_time:.2f}s")
        return status_str, None, cpu_time, {}

    cmax_val = int(solver.Value(C_max))
    print(f"  Status={status_str}, Cmax={cmax_val}, CPU={cpu_time:.2f}s")

    # -------------------------
    # Optional debug: show fatigue propagation per machine
    # -------------------------
    if debug_fatigue:
        print("\n=== FATIGUE PROPAGATION (per machine) ===")
        for m in range(1, M + 1):
            ops = ops_on_m[m]
            if not ops:
                continue

            # Build successor map from chosen predecessor arcs (if chain exists)
            if len(ops) == 1:
                b = ops[0]
                print(f"Machine M{m}: single op {b}  fS={solver.Value(op_fstart[b])} fC={solver.Value(op_fend[b])}")
                continue

            pred = pred_arc_by_m.get(m, {})
            succ = {}
            first = None
            for b in ops:
                if solver.Value(pred[(SRC, b)]) == 1:
                    first = b
                    break
            for a in ops:
                for b in ops:
                    if a == b:
                        continue
                    if solver.Value(pred[(a, b)]) == 1:
                        succ[a] = b

            print(f"\nMachine M{m}:")
            cur = first
            seen = set()
            idx = 0
            while cur is not None and cur not in seen:
                seen.add(cur)
                rb = solver.Value(op_rest_before[cur])
                rd = solver.Value(op_rest_dur[cur])
                pe = solver.Value(op_pred_end[cur])
                pf = solver.Value(op_pred_fend[cur])
                s = solver.Value(op_start[cur])
                e = solver.Value(op_end[cur])
                fs = solver.Value(op_fstart[cur])
                fc = solver.Value(op_fend[cur])
                fidl = solver.Value(f_start_idle[cur])
                print(f"  {idx:02d} op={cur} predF={pf} predEnd={pe} fidle={fidl} restB={rb} restDur={rd} "
                      f"start={s} end={e} fS={fs} fC={fc}")
                cur = succ.get(cur, None)
                idx += 1

    # -------------------------
    # Extract schedule for Gantt
    # -------------------------
    machine_sched: List[List[int]] = []
    rest_sched: List[List[int]] = []  # [machine, start, end, job, op]

    for i in range(J):
        for j in range(job_n_ops[i]):
            o = (i, j)
            m_idx = op_machine[o]
            s = solver.Value(op_start[o])
            e = solver.Value(op_end[o])
            machine_sched.append([m_idx, i + 1, s, e, j + 1])

            if solver.Value(op_rest_before[o]) == 1:
                pe = solver.Value(op_pred_end[o])
                rd = solver.Value(op_rest_dur[o])
                if rd > 0:
                    rest_sched.append([m_idx, pe, pe + rd, i + 1, j + 1])

    robot_sched: List[List[int]] = []
    for t in range(T):
        assigned_robot = None
        for r in range(R):
            if solver.Value(robot_assign[r][t]) == 1:
                assigned_robot = r
                break
        if assigned_robot is None:
            continue

        i = t_job[t]
        op_index = t_op[t]
        op_label = 99 if op_index == -1 else op_index + 1

        s = solver.Value(t_start[t])
        e = solver.Value(t_end[t])
        robot_sched.append([assigned_robot + 1, i + 1, s, e, op_label])

    # Prepare machines_next_start to compute idle gaps
    Fmax_per_machine = {m: fat["Fmax"][m] for m in range(1, M+1)}
    Fmin_per_machine = {m: fat["Fmin"][m] for m in range(1, M+1)}
    machine_next_start = {m: 0.0 for m in range(1, M + 1)}
    # Sort ALL intervals (work and rest) by start time
    all_intervals = []
    for i in range(J):
        for j in range(job_n_ops[i]):
            o = (i, j)
            m_idx = op_machine[o]
            s_val = float(solver.Value(op_start[o]))
            e_val = float(solver.Value(op_end[o]))
            fs = solver.Value(op_fstart[o]) / L
            fe = solver.Value(op_fend[o]) / L
            # Type 0: Work
            all_intervals.append({'m': m_idx, 's': s_val, 'e': e_val, 'fs': fs, 'fe': fe, 'type': 0})

            if solver.Value(op_rest_before[o]) == 1:
                rs = float(solver.Value(op_pred_end[o]))
                rd = float(solver.Value(op_rest_dur[o]))
                if rd > 0:
                    fb = solver.Value(op_pred_fend[o]) / L
                    fa = fmin_int[m_idx] / L
                    # Type 1: Mandatory Rest
                    all_intervals.append({'m': m_idx, 's': rs, 'e': rs + rd, 'fs': fb, 'fe': fa, 'type': 1})

    all_intervals.sort(key=lambda x: x['s'])

    # fatigue_log will store (m, s, e, fs, fe, type)
    # type: 0=Work, 1=Mandatory Rest, 2=Idle Gap
    fatigue_log = []
    machine_used = {m: False for m in range(1, M + 1)}
    machine_last_e = {m: 0.0 for m in range(1, M + 1)}
    machine_last_f = {m: 0.0 for m in range(1, M + 1)}

    for m in range(1, M + 1):
        machine_last_f[m] = Fmin[m]

    for interval in all_intervals:
        m = interval['m']
        # Handle idle gap before this interval
        if machine_used[m] and interval['s'] > machine_last_e[m]:
            s_idle = machine_last_e[m]
            e_idle = interval['s']
            f_start_idle = machine_last_f[m]
            
            if recover_on_idle:
                # F(t) = F_start * exp(-mu * idle_recovery_factor * t)
                f_end_idle = f_start_idle * math.exp(-mu[m] * idle_recovery_factor * (e_idle - s_idle))
            else:
                f_end_idle = f_start_idle
            
            # Type 2: Idle Gap
            fatigue_log.append((m, s_idle, e_idle, f_start_idle, f_end_idle, 2))
            machine_last_f[m] = f_end_idle

        # Add the actual interval
        fatigue_log.append((m, interval['s'], interval['e'], interval['fs'], interval['fe'], interval['type']))
        machine_used[m] = True
        machine_last_e[m] = interval['e']
        machine_last_f[m] = interval['fe']

    fatigue_log.sort(key=lambda x: x[1])

    if make_gantt:
        _plot_gantt(
            inst_id=inst_id,
            layout_id=layout_id,
            cmax_value=cmax_val,
            machine_sched=machine_sched,
            rest_sched=rest_sched,
            robot_sched=robot_sched,
            num_machines=M,
            num_robots=R,
            last_ttask=last_ttask,
            to_plot=to_plot,
            gantt_path=gantt_path,
            fatigue_log=fatigue_log,
            Fmax_per_machine=Fmax_per_machine,
            Fmin_per_machine=Fmin_per_machine,
            lambda_k=lam,
            mu_k=mu,
            irf_val=irf,
            recover_on_idle=recover_on_idle
        )

    # -----------------------------------------------------------------------
    # KPIs calculation
    # -----------------------------------------------------------------------
    machine_sched_kpi = [(m, j, s, e) for (m, j, s, e, op) in machine_sched]
    robot_sched_kpi   = [(r, j, s, e) for (r, j, s, e, op) in robot_sched]

    Cmax = float(cmax_val)

    Uavg_M, U_M = kpi_machine_utilization(machine_sched_kpi, Cmax, M)
    Uavg_R, U_R = kpi_robot_utilization(robot_sched_kpi, Cmax, R)

    Fmax_obs = kpi_max_fatigue(fatigue_log)
    Fgrowth  = kpi_fatigue_accumulation_growth(fatigue_log)
    Nbreaks  = kpi_num_rest_breaks(fatigue_log)
    Trest    = kpi_total_rest(fatigue_log)
    Facc     = kpi_fatigue_accumulation(fatigue_log)

    viol = kpi_fatigue_violations_cp(fatigue_log, Fmax_per_machine)

    KPIs = {
        "makespan":           Cmax,
        "cpu_time":           cpu_time,
        "avg_machine_util":   float(Uavg_M),
        "machine_util":       U_M.tolist(),
        "avg_robot_util":     float(Uavg_R),
        "robot_util":         U_R.tolist(),
        "fatigue_viol":       int(viol),
        "fatigue_max":        float(Fmax_obs),
        "fatigue_acc":        float(Facc),
        "fatigue_growth":     float(Fgrowth),
        "num_rest_breaks":    int(Nbreaks),
        "total_rest_time":    float(Trest),
    }

    return status_str, cmax_val, cpu_time, KPIs

# ---------------------------------------------------------------------------
# Gantt plotting
# ---------------------------------------------------------------------------

def _plot_gantt(
    inst_id: str,
    layout_id: int,
    cmax_value: int,
    machine_sched: List[List[int]],
    rest_sched: List[List[int]],
    robot_sched: List[List[int]],
    num_machines: int,
    num_robots: int,
    last_ttask: int,
    to_plot: bool,
    gantt_path: Optional[str],
    fatigue_log: List[Tuple],
    Fmax_per_machine: Dict[int, float],
    Fmin_per_machine: Dict[int, float],
    lambda_k: List[Optional[float]],
    mu_k: List[Optional[float]],
    irf_val: float,
    recover_on_idle: bool,
):
    fig, (ax, ax_fat) = plt.subplots(2, 1, figsize=(20, 12), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)

    mcolors = [
        "tab:red", "tab:cyan", "tab:green", "tab:orange",
        "yellow", "tab:brown", "magenta", "lime",
        "tomato", "tab:blue", "red", "cyan", "green", "blue",
    ]

    total_lines = num_machines + num_robots
    yticks = [10 * (k + 1) for k in range(total_lines)]
    ax.set_yticks([y + 5 for y in yticks])

    ylabels = [f"M{m+1}" for m in range(num_machines)] + [f"R{r+1}" for r in range(num_robots)]
    ax.set_yticklabels(ylabels, fontsize=16)

    #ax.set_xlabel("Time", fontsize=18, weight="bold")
    ax.tick_params(labelsize=14)

    ax.minorticks_on()
    ax.grid(which="major", linestyle="-", linewidth=0.5, color="red")
    ax.grid(which="minor", linestyle=":", linewidth=0.5, color="black")
    ax.set_xlim(0, int(cmax_value) + 10)

    for rseg in rest_sched:
        machine_id, start, end, job_id, op_idx = rseg
        a = [(start, end - start)]
        ax.broken_barh(a, (yticks[machine_id - 1], 9), facecolors="lightgray")
        ax.text(start + (end - start) / 2 - 1, 4 + yticks[machine_id - 1],
                "Rest", fontsize=10, weight="bold")

    for m in machine_sched:
        machine_id, job_id, start, end, op_idx = m
        a = [(start, end - start)]
        color = mcolors[(job_id - 1) % len(mcolors)]
        text = r"$O_{%d,%d}$" % (job_id, op_idx)
        x = start + (end - start) / 2 - 2
        y = 4 + yticks[machine_id - 1]
        ax.text(x, y, text, fontsize=14, weight="bold")
        ax.broken_barh(a, (yticks[machine_id - 1], 9), facecolors=color)

    for r in robot_sched:
        robot_id, job_id, start, end, op_idx = r
        a = [(start, end - start)]
        color = mcolors[(job_id - 1) % len(mcolors)]
        if op_idx == 99:
            text = r"$T^R_{J%d}$" % job_id
        else:
            text = r"$T_{%d,%d}$" % (job_id, op_idx)
        x = start + (end - start) / 2 - 2
        y = 4 + yticks[num_machines + robot_id - 1]
        ax.text(x, y, text, fontsize=14, weight="bold")
        ax.broken_barh(a, (yticks[num_machines + robot_id - 1], 9), facecolors=color)

    title = f"{inst_id}\nResolution method=CP-SAT (fixed), Cmax={cmax_value}, last_ttask={last_ttask}"
    ax.set_title(title, fontsize=20, weight="bold")

    # --- Fatigue Evolution Subplot ---
    ax_fat.set_ylabel("Fatigue", fontsize=16, weight="bold")
    ax_fat.set_xlabel("Time", fontsize=18, weight="bold")
    ax_fat.set_ylim(0, 1.05)
    ax_fat.grid(True, linestyle='--', alpha=0.7)

    for m in range(1, num_machines + 1):
        m_entries = [e for e in fatigue_log if e[0] == m]
        if not m_entries:
            continue
        
        m_color = mcolors[(m - 1) % len(mcolors)]
        
        all_t = []
        all_f = []
        
        for (m_id, s, e, Fs, Fe, seg_type) in m_entries:
            t_vals = np.linspace(s, e, 50)
            
            if seg_type == 1:
                # Type 1: Mandatory Rest recovery
                mu_m = mu_k[m]
                if mu_m is not None:
                    f_vals = Fs * np.exp(-mu_m * (t_vals - s))
                    f_vals = np.maximum(f_vals, Fmin_per_machine[m])
                else:
                    f_vals = np.full_like(t_vals, Fs)
            elif seg_type == 0:
                # Type 0: Work accumulation
                lam_m = lambda_k[m]
                if lam_m is not None and lam_m > 0:
                    f_vals = 1.0 - (1.0 - Fs) * np.exp(-lam_m * (t_vals - s))
                else:
                    f_vals = np.full_like(t_vals, Fs)
            else:
                # Type 2: Idle Gap
                if recover_on_idle:
                    mu_m = mu_k[m]
                    if mu_m is not None:
                        f_vals = Fs * np.exp(-mu_m * irf_val * (t_vals - s))
                        f_vals = np.maximum(f_vals, Fmin_per_machine[m])
                    else:
                        f_vals = np.full_like(t_vals, Fs)
                else:
                    f_vals = np.full_like(t_vals, Fs)

            # Concatenate to ensure a single continuous line per machine
            if len(all_t) > 0:
                # Add a point to bridge the gap if times are close
                all_t.append(t_vals[0])
                all_f.append(f_vals[0])
                
            all_t.extend(t_vals.tolist())
            all_f.extend(f_vals.tolist())

        ax_fat.plot(all_t, all_f, color=m_color, linewidth=2, alpha=0.8, label=f"M{m}")
        
        # Draw threshold
        ax_fat.axhline(y=Fmax_per_machine[m], color=m_color, linestyle='--', alpha=0.3)
    
    ax_fat.legend(loc='upper right', bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()

    fname = f"{inst_id}_gantt_fatigue_cp.png" if gantt_path is None else gantt_path
    plt.savefig(fname, dpi=300)
    if to_plot:
        plt.show()
    plt.close(fig)
    print(f"Gantt chart saved to: {fname}")

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    inst_path = os.path.join("bu_instances_eval/scenario_1", "EX8_2_S1.json")
    inst_name=os.path.basename(inst_path)[:-7].replace('_', '')
    instance = load_jsspt_instance(inst_path)

    status, cmax, cpu, kpis = solve_jsspt_hf_cp(
        instance,
        last_ttask=1,
        num_robots=2,
        time_limit=3600.0, #i.e., 1 hour
        num_workers=8,
        make_gantt=True,
        to_plot=False,
        gantt_path=None,
        recover_on_idle=True,
        idle_recovery_factor=1.0, # Should ranges from 0.0 to 1.0
        eps_exhaust_int=1,      # treat >= L-1 as exhaustion
        debug_fatigue=False,     # prints propagation chain
    )
    print("Done:", status, cmax, f"{cpu:.2f}s")
    #print(kpis)
