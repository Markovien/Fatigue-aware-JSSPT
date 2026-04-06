# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 15:53:02 2025

@author: Kader SANOGO

meta_ga.py
--------------------

Genetic Algorithm metaheuristic for JSSPT/JSSPT-HF.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

import numpy as np
from collections import defaultdict


# --- Quantization helpers (aligned with the CP-SAT solver) ---
FAT_SCALE = 1000  # 3 decimal digits grid: 0..1000


def round_half_up(x: float) -> int:
    """Standard rounding: fractional part >= .5 rounds up."""
    return int(math.floor(x + 0.5))


def clamp_int(x: int, lo: int, hi: int) -> int:
    return lo if x < lo else hi if x > hi else x


def quantize_fatigue(F: float) -> int:
    """Map fatigue in [0,1] to integer grid 0..FAT_SCALE using Python round()."""
    return clamp_int(int(round(FAT_SCALE * F)), 0, FAT_SCALE)


def to_int_fatigue_params(Fmin, Fmax, L: int = FAT_SCALE):
    """Convert fatigue thresholds to integer grid 0..L (floor for min, ceil for max)."""
    fmin = [None]
    fmax = [None]
    for k in range(1, len(Fmin)):
        fmin.append(int(math.floor(L * Fmin[k])))
        fmax.append(int(math.ceil(L * Fmax[k])))
    return fmin, fmax


#----------------------------------------------------------
# KPIs computing functions
#----------------------------------------------------------

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

def kpi_fatigue_violations(fatigue_log, Fmax):
    return sum(Fs >= Fmax for (_, _, _, Fs, _, rest) in fatigue_log)


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
    for (_, _, _, Fs, Fe, is_rest) in fatigue_log:
        if not is_rest:  # only count work intervals
            total += max(0.0, Fe - Fs)
    return total

def kpi_num_rest_breaks(fatigue_log):
    """
    Count number of explicit rest intervals.
    """
    return sum(1 for (_, _, _, _, _, is_rest) in fatigue_log if is_rest)

def kpi_total_rest(fatigue_log):
    return sum((e - s) for (_, s, e, Fs, Fe, rest) in fatigue_log if rest)

def kpi_fatigue_accumulation(fatigue_log):
    return sum(Fs * (e - s) for (_, s, e, Fs, Fe, rest) in fatigue_log if not rest)


# ---------------------------------------------------------------------------
# Instance loading
# ---------------------------------------------------------------------------

def load_jsspt_instance(path: str) -> Dict:
    """Load a JSSPT JSON instance."""
    with open(path, "r") as f:
        data = json.load(f)
    return data


# ---------------------------------------------------------------------------
# Fatigue & delta parameter extraction (used only if use_fatigue=True)
# ---------------------------------------------------------------------------

def build_fatigue_params(instance: Dict) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Build per-machine arrays (1..M) for:
        F_min_k, F_max_k, lambda_k, mu_k

    Supports:
    - Scenario 1:
        meta["fatigue"]["mode"] == "uniform"
        with scalar F_min, F_max, lambda, mu.
    - Scenario 2:
        meta["fatigue"]["mode"] == "per_human"
        with lists F_min[0..M-1], F_max[0..M-1], lambda[0..M-1], mu[0..M-1].
    """
    M = instance["machines_nb"]
    meta = instance.get("meta", {})
    fat = meta.get("fatigue", {})

    mode = fat.get("mode", "uniform")

    # Allocate 1..M (index 0 unused)
    F_min_k = [0.0] * (M + 1)
    F_max_k = [1.0] * (M + 1)
    lambda_k = [0.0] * (M + 1)
    mu_k = [0.0] * (M + 1)

    if mode == "per_human":
        F_min_list = fat["F_min"]
        F_max_list = fat["F_max"]
        lambda_list = fat["lambda"]
        mu_list = fat["mu"]
        assert len(F_min_list) == M
        assert len(F_max_list) == M
        assert len(lambda_list) == M
        assert len(mu_list) == M
        for k in range(1, M + 1):
            F_min_k[k] = float(F_min_list[k - 1])
            F_max_k[k] = float(F_max_list[k - 1])
            lambda_k[k] = float(lambda_list[k - 1])
            mu_k[k] = float(mu_list[k - 1])
    else:
        # default: uniform
        F_min = float(fat.get("F_min", 0.2))
        F_max = float(fat.get("F_max", 0.8))
        lam = float(fat.get("lambda", 0.04))
        mu = float(fat.get("mu", 0.08))
        for k in range(1, M + 1):
            F_min_k[k] = F_min
            F_max_k[k] = F_max
            lambda_k[k] = lam
            mu_k[k] = mu

    return F_min_k, F_max_k, lambda_k, mu_k


def build_delta_params(instance: Dict) -> Dict[Tuple[int, int, int], float]:
    """
    Build a dict delta_ijk[(i,j,k)] with the fatigue influence coefficient.

    Uses (priority order):
    1) If op has explicit "delta" field in jobs[i]["ops"][j], use it.
    2) Else, use meta["delta"]["mode"]:
        - "uniform": use meta["delta"]["value"] for all operations.
        - "per_op_machine_duration": simple rule based on duration:
             delta = (a / lambda_k) * ln(2) where a is duration-dependent
    """
    jobs_data = instance["jobs"]
    meta = instance.get("meta", {})
    delta_meta = meta.get("delta", {})

    mode = delta_meta.get("mode", "uniform")
    default_val = float(delta_meta.get("value", 0.0))

    delta_ijk: Dict[Tuple[int, int, int], float] = {}

    a,b,c,d=build_fatigue_params(instance) # c is the lambda_k list

    for i, job in enumerate(jobs_data):
        for j, op in enumerate(job["ops"]):
            m_idx = op["machine_index"]  # 1..M
            if "delta" in op:
                d = float(op["delta"])
            else:
                dur = float(op["duration"])
                if mode == "per_op_machine_duration":
                    if dur <= 10:
                        d = (0.1 / c[m_idx]) * math.log(2)
                    elif dur <= 15:
                        d = (0.3 / c[m_idx]) * math.log(2)
                    else:
                        d = (0.5 / c[m_idx]) * math.log(2)
                else:
                    d = default_val
            delta_ijk[(i, j, m_idx)] = d

    return delta_ijk


# ---------------------------------------------------------------------------
# Gantt plotting: baseline (no fatigue)
# ---------------------------------------------------------------------------

def _plot_gantt_baseline(
    inst_id: str,
    layout_id: int,
    cmax_value: float,
    machine_sched: List[List[int]],
    robot_sched: List[List[int]],
    num_machines: int,
    num_robots: int,
    last_ttask: int,
    gantt_path: Optional[str],
    to_plot: bool
):
    """
    Plot a Gantt chart for the baseline JSSPT (no fatigue).

    machine_sched: [M_id, Job_id, start, end, op_idx]
    robot_sched  : [R_id, Job_id, start, end, op_idx (99 for final)]
    """
    fig, ax = plt.subplots(figsize=(20, 8))

    mcolors = [
        "tab:red",
        "tab:cyan",
        "tab:green",
        "tab:orange",
        "yellow",
        "tab:brown",
        "magenta",
        "lime",
        "tomato",
        "tab:blue",
        "red",
        "cyan",
        "green",
        "blue",
    ]

    # Y-axis ticks: machines first, then robots
    total_lines = num_machines + num_robots
    yticks = []
    setticks = []
    for k in range(total_lines):
        yticks.append(10 * (k + 1))
        setticks.append(10 * (k + 1) + 5)
    ax.set_yticks(setticks)

    ylabels = []
    for m in range(num_machines):
        ylabels.append(f"M{m+1}")
    for r in range(num_robots):
        ylabels.append(f"R{r+1}")
    ax.set_yticklabels(ylabels, fontsize=16)

    ax.set_xlabel("Time", fontsize=18, weight="bold")
    ax.tick_params(labelsize=14)

    ax.minorticks_on()
    ax.grid(which="major", linestyle="-", linewidth=0.5, color="red")
    ax.grid(which="minor", linestyle=":", linewidth=0.5, color="black")
    ax.set_xlim(0, int(cmax_value) + 10)

    # Machines
    for m in machine_sched:
        machine_id, job_id, start, end, op_idx = m
        a = [(start, end - start)]
        color = mcolors[(job_id - 1) % len(mcolors)]
        text = r"$O_{%d,%d}$" % (job_id, op_idx)
        x = start + (end - start) / 2 - 2
        y = 4 + yticks[machine_id - 1]
        ax.text(x, y, text, fontsize=14, weight="bold")
        ax.broken_barh(a, (yticks[machine_id - 1], 9), facecolors=color)

    # Robots
    for r in robot_sched:
        robot_id, job_id, start, end, op_idx = r
        a = [(start, end - start)]
        color = mcolors[(job_id - 1) % len(mcolors)]
        if op_idx == 99:
            text = r"$T_{fp%d}$" % job_id
        else:
            text = r"$T_{O_{%d,%d}}$" % (job_id, op_idx)
        x = start + (end - start) / 2 - 2
        y = 4 + yticks[num_machines + robot_id - 1]
        ax.text(x, y, text, fontsize=14, weight="bold")
        ax.broken_barh(a, (yticks[num_machines + robot_id - 1], 9), facecolors=color)

    title = f"{inst_id}\nResolution method=GA (no fatigue), Cmax={int(round(cmax_value))}, last_ttask={last_ttask}"
    ax.set_title(title, fontsize=20, weight="bold")
    plt.tight_layout()

    if gantt_path is None:
        fname = f"{inst_id}_GA_gantt.png"
    else:
        fname = gantt_path
    plt.savefig(fname, dpi=300)
    if to_plot: plt.show()
    plt.close(fig)
    print(f"Gantt chart saved to: {fname}")


# ---------------------------------------------------------------------------
# Gantt plotting: fatigue-aware (with rests)
# ---------------------------------------------------------------------------

def _plot_gantt_hf(
    inst_id: str,
    layout_id: int,
    cmax_value: float,
    machine_sched: List[List[int]],
    machine_rest_sched: List[List[float]],
    robot_sched: List[List[int]],
    num_machines: int,
    num_robots: int,
    last_ttask: int,
    gantt_path: Optional[str],
    to_plot: bool,
    fatigue_log: List[Tuple[int, float, float, float, float, bool]],
    Fmax_per_machine: Dict[int, float],
    Fmin_per_machine: Dict[int, float],
    lambda_k: List[float],
    mu_k: List[float],
    irf: float
):
    """
    Plot Gantt chart with smooth fatigue curves.
    """
    scenario = "S1" if "_S1" in inst_id else ("S2" if "_S2" in inst_id else "?")
    fig, (ax, ax_fat) = plt.subplots(2, 1, figsize=(20, 12), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)

    mcolors = ["tab:red", "tab:cyan", "tab:green", "tab:orange", "yellow",
               "tab:brown", "magenta", "lime", "tomato", "tab:blue",
               "red", "cyan", "green", "blue"]

    total_lines = num_machines + num_robots
    yticks = [10 * (k + 1) for k in range(total_lines)]
    setticks = [y + 5 for y in yticks]
    ax.set_yticks(setticks)

    ylabels = [f"M{m+1}" for m in range(num_machines)] + \
              [f"R{r+1}" for r in range(num_robots)]
    ax.set_yticklabels(ylabels, fontsize=16)
    ax.tick_params(labelsize=14)
    ax.minorticks_on()
    ax.grid(which="major", linestyle="-", linewidth=0.5, color="red")
    ax.grid(which="minor", linestyle=":", linewidth=0.5, color="black")
    ax.set_xlim(0, int(cmax_value) + 10)

    # Rest mapping
    rest_info = {}
    for m_id, rs, re, Fb, Fa in machine_rest_sched:
        rest_info[(m_id, int(rs))] = (Fb, Fa)

    # Unified activities
    activities = []
    for m in machine_sched:
        activities.append((m[0], m[1], m[2], m[3], m[4], False))
    for m_str in machine_rest_sched:
        activities.append((m_str[0], -1, int(m_str[1]), int(m_str[2]), -1, True))
    activities.sort(key=lambda x: x[2])

    for machine_id, job_id, start, end, op_idx, is_rest in activities:
        y = yticks[machine_id - 1]
        width = end - start
        if is_rest:
            ax.broken_barh([(start, width)], (y, 9), facecolors="lightgray", hatch="//", edgecolor="black", linewidth=0.5)
            Fb, Fa = rest_info.get((machine_id, int(start)), (0, 0))
            txt = f"F:{Fb:.2f}→{Fa:.2f}"
            ax.text(start + width/2, y + 4.5, txt, fontsize=12, weight="bold", ha="center", va="center", path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        else:
            color = mcolors[(job_id - 1) % len(mcolors)]
            label = r"$O_{%d,%d}$" % (job_id, op_idx)
            ax.text(start + width/2 - 2, y + 4, label, fontsize=14, weight="bold")
            ax.broken_barh([(start, width)], (y, 9), facecolors=color)

    for r_entry in robot_sched:
        robot_id, job_id, start, end, op_idx = r_entry
        color = mcolors[(job_id - 1) % len(mcolors)]
        txt = r"$T_{fp%d}$" % job_id if op_idx == 99 else r"$T_{O_{%d,%d}}$" % (job_id, op_idx)
        ax.text(start + (end - start)/2 - 2, 4 + yticks[num_machines + robot_id - 1], txt, fontsize=14, weight="bold")
        ax.broken_barh([(start, end - start)], (yticks[num_machines + robot_id - 1], 9), facecolors=color)

    # --- Fatigue Evolution Subplot ---
    ax_fat.set_ylabel("Fatigue", fontsize=16, weight="bold")
    ax_fat.set_xlabel("Time", fontsize=16, weight="bold")
    ax_fat.set_ylim(-0.05, 1.05)
    ax_fat.grid(True, linestyle='--', alpha=0.7)

    for m in range(1, num_machines + 1):
        m_logs = [log for log in fatigue_log if log[0] == m]
        m_logs.sort(key=lambda x: x[1])
        m_color = mcolors[(m - 1) % len(mcolors)]

        # Thresholds
        ax_fat.axhline(Fmax_per_machine[m], color=m_color, linestyle='--', alpha=0.2)
        ax_fat.axhline(Fmin_per_machine[m], color=m_color, linestyle='--', alpha=0.2)

        for i, (m_id, s, e, Fs, Fe, is_recovery) in enumerate(m_logs):
            if e <= s: continue

            # Connect gaps
            if i > 0:
                prev_e = m_logs[i-1][2]
                if s > prev_e:
                    ax_fat.plot([prev_e, s], [m_logs[i-1][4], Fs], color=m_color, linewidth=2, alpha=0.8)

            t_vals = np.linspace(s, e, 20)
            if is_recovery:
                is_machine_rest = any(r[0] == m and abs(r[1] - s) < 1e-3 and abs(r[2] - e) < 1e-3 for r in machine_rest_sched)
                mu_eff = mu_k[m] if is_machine_rest else mu_k[m] * irf
                if mu_eff > 0:
                    f_vals = Fs * np.exp(-mu_eff * (t_vals - s))
                    # Only clamp to Fmin for actual machine rests, not idle recovery
                    if is_machine_rest:
                        f_vals = np.maximum(f_vals, Fmin_per_machine[m])
                else:
                    f_vals = np.full_like(t_vals, Fs)
            else:
                lam = lambda_k[m]
                if lam > 0:
                    f_vals = 1.0 - (1.0 - Fs) * np.exp(-lam * (t_vals - s))
                else:
                    f_vals = np.full_like(t_vals, Fs)

            label = f"M{m}" if i == 0 else None
            ax_fat.plot(t_vals, f_vals, color=m_color, linewidth=2, alpha=0.8, label=label)

    ax_fat.legend(loc='upper right', bbox_to_anchor=(1.05, 1))
    title = f"{inst_id} [{scenario}]\nResolution: GA + Fatigue, Cmax={int(round(cmax_value))}, last_ttask={last_ttask}"
    ax.set_title(title, fontsize=20, weight="bold")
    plt.tight_layout()
    fname = f"{inst_id}_GA_HF_gantt.png" if gantt_path is None else gantt_path
    plt.savefig(fname, dpi=300)
    if to_plot: plt.show()
    plt.close(fig)
    print(f"Gantt chart saved to: {fname}")


# ---------------------------------------------------------------------------
# GA solver
# ---------------------------------------------------------------------------

def solve_jsspt_ga(
    instance: Dict,
    use_fatigue: bool = False,
    last_ttask: int = 1,
    num_robots: Optional[int] = None,
    time_limit: float = 60.0,
    population_size: int = 50,
    generations: int = 500,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.2,
    recover_on_idle: Optional[bool] = None,
    idle_recovery_factor: float = 1.0,
    make_gantt: bool = True,
    to_plot: bool = False,
    gantt_path: Optional[str] = None,
) -> Tuple[str, Optional[float], float, Dict]:
    """
    Unified GA solver for JSSPT / JSSPT-HF.

    Parameters
    ----------
    instance : dict
        JSSPT instance (same JSON schema as RL model).
    use_fatigue : bool, default False
        If False → baseline JSSPT (no fatigue).
        If True  → JSSPT-HF (fatigue-aware).
    last_ttask : int, default 1
        If 1, includes final return-to-Z transports in makespan.
        If 0, Cmax = last operation completion time.
    num_robots : int or None
        If None, uses instance["robots_nb"].
    time_limit : float
        Wall-clock time limit in seconds for GA.
    population_size : int
    generations : int
    crossover_rate : float
    mutation_rate : float
    recover_on_idle : bool or None
        Used only when use_fatigue=True.
        If None, uses instance["meta"]["recover_on_idle"] (default False).
    make_gantt : bool
        Whether to generate a Gantt chart.
    gantt_path : str or None
        Optional explicit path for the Gantt PNG.

    Returns
    -------
    (status_str, C_max, cpu_time)
        C_max is a float (even for baseline mode; effectively an int there).
    """
    start_wall = time.time()

    # -------------------------
    # Basic instance data
    # -------------------------
    inst_id = instance.get("instance_id", "unknown_instance")
    layout_id = instance.get("layout_id", -1)
    M = instance["machines_nb"]
    sigma = instance["sigma"]  # (M+1)x(M+1), index 0=Z, 1..M=machines
    jobs_data = instance["jobs"]
    meta = instance.get("meta", {})

    if num_robots is None:
        R = instance.get("robots_nb", 1)
    else:
        R = num_robots

    # -------------------------
    # Build operation data
    # -------------------------
    J = len(jobs_data)
    op_duration: Dict[Tuple[int, int], float] = {}
    op_machine: Dict[Tuple[int, int], int] = {}
    job_n_ops: List[int] = []
    total_proc_time = 0.0

    for i, job in enumerate(jobs_data):
        n_ops = job["n_ops"]
        job_n_ops.append(n_ops)
        for j, op in enumerate(job["ops"]):
            m_idx = op["machine_index"]  # 1..M
            dur = float(op["duration"])
            op_machine[(i, j)] = m_idx
            op_duration[(i, j)] = dur
            total_proc_time += dur


    # -------------------------
    # Chromosome utilities
    # -------------------------
    # Each job i appears job_n_ops[i] times.
    base_seq: List[int] = []
    for i in range(J):
        base_seq.extend([i] * job_n_ops[i])

    def random_chromosome() -> List[int]:
        chrom = base_seq[:]
        random.shuffle(chrom)
        return chrom

    def order_crossover(p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
        """Order crossover (OX) adapted to sequences with repeated jobs."""
        n = len(p1)
        if n <= 1:
            return p1[:], p2[:]
        a, b = sorted(random.sample(range(n), 2))

        def make_child(pa: List[int], pb: List[int]) -> List[int]:
            child: List[Optional[int]] = [None] * n  # type: ignore
            # copy slice
            for k in range(a, b + 1):
                child[k] = pa[k]
            # count used per job
            used = [0] * J
            for gene in child:
                if gene is not None:
                    used[gene] += 1
            # fill from pb
            pos = (b + 1) % n
            for job_i in pb:
                if used[job_i] < job_n_ops[job_i]:
                    while child[pos] is not None:
                        pos = (pos + 1) % n
                    child[pos] = job_i
                    used[job_i] += 1
            return [int(g) for g in child]  # type: ignore

        c1 = make_child(p1, p2)
        c2 = make_child(p2, p1)
        return c1, c2

    def mutate_swap(chrom: List[int]) -> None:
        """Simple swap mutation."""
        if len(chrom) <= 1:
            return
        i, j = random.sample(range(len(chrom)), 2)
        chrom[i], chrom[j] = chrom[j], chrom[i]

    # -----------------------------------------------------------------------
    # Decoding WITHOUT fatigue (baseline JSSPT)
    # -----------------------------------------------------------------------

    def decode_schedule_baseline(
        seq: List[int]
    ) -> Tuple[float,
               Dict[Tuple[int, int], Tuple[float, float]],
               List[Tuple[int, int, int, float, float]],
               List[List[float]],
               Dict[Tuple[int, int], Tuple[float, float]],
               List[Tuple[int, float, float, float, float]]]:
        """
        Baseline decoding: no fatigue, no rests.

        Returns:
            cmax, op_times, transports, rests(empty)
        """
        # Job state
        job_next_op = [0] * J
        job_last_op_end = [0.0] * J

        # Machine state
        machine_ready = [0.0] * (M + 1)  # index 1..M used

        # Robot state
        robot_time = [0.0] * R
        robot_loc = [0] * R  # 0..M, all start at Z(0)

        # Schedule storage
        op_times: Dict[Tuple[int, int], Tuple[float, float]] = {}
        transports: List[Tuple[int, int, int, float, float]] = []
        rests: List[List[float]] = []  # empty for baseline

        # --- schedule operations and their preceding transports ---
        for job_i in seq:
            j = job_next_op[job_i]
            if j >= job_n_ops[job_i]:
                continue  # this job has no remaining ops

            m_idx = op_machine[(job_i, j)]
            proc_dur = op_duration[(job_i, j)]

            # Preceding transport
            if j == 0:
                pickup = 0  # Z
            else:
                pickup = op_machine[(job_i, j - 1)]
            drop = m_idx
            travel_dur = float(sigma[pickup][drop])

            # Predecessor operation completion (only for j>0)
            pred_job_time = job_last_op_end[job_i] if j > 0 else 0.0

            # Choose best robot (earliest completion)
            best_robot = 0
            best_t_start = 0.0
            best_t_end = float("inf")

            for r in range(R):
                # time to reach pickup from last location of this robot
                empty_dur = float(sigma[robot_loc[r]][pickup])
                ready_from_prev = robot_time[r] + empty_dur
                t_start_r = max(pred_job_time, ready_from_prev)
                t_end_r = t_start_r + travel_dur
                if t_end_r < best_t_end:
                    best_t_end = t_end_r
                    best_t_start = t_start_r
                    best_robot = r

            # Assign to best_robot
            r = best_robot
            t_start = best_t_start
            t_end = best_t_end
            robot_time[r] = t_end
            robot_loc[r] = drop
            transports.append((r, job_i, j, t_start, t_end))

            # Operation on its machine
            m_ready = machine_ready[m_idx]
            op_start = max(t_end, m_ready)
            op_end = op_start + proc_dur
            machine_ready[m_idx] = op_end
            job_last_op_end[job_i] = op_end
            op_times[(job_i, j)] = (op_start, op_end)

            job_next_op[job_i] += 1

        # --- schedule final returns, if requested ---
        final_end_times: List[float] = []
        if last_ttask == 1:
            for job_i in range(J):
                # pickup: last machine
                last_op_idx = job_n_ops[job_i] - 1
                pickup = op_machine[(job_i, last_op_idx)]
                drop = 0  # Z
                travel_dur = float(sigma[pickup][drop])
                pred_job_time = job_last_op_end[job_i]

                best_robot = 0
                best_t_start = 0.0
                best_t_end = float("inf")

                for r in range(R):
                    empty_dur = float(sigma[robot_loc[r]][pickup])
                    ready_from_prev = robot_time[r] + empty_dur
                    t_start_r = max(pred_job_time, ready_from_prev)
                    t_end_r = t_start_r + travel_dur
                    if t_end_r < best_t_end:
                        best_t_end = t_end_r
                        best_t_start = t_start_r
                        best_robot = r

                r = best_robot
                t_start = best_t_start
                t_end = best_t_end
                robot_time[r] = t_end
                robot_loc[r] = drop
                transports.append((r, job_i, -1, t_start, t_end))
                final_end_times.append(t_end)

        # Makespan
        if last_ttask == 1 and final_end_times:
            cmax = max(final_end_times)
        else:
            cmax = max(job_last_op_end) if job_last_op_end else 0.0

        return float(cmax), op_times, transports, rests, {}, []

    # -----------------------------------------------------------------------
    # Decoding WITH fatigue (JSSPT-HF) – only if use_fatigue=True
    # -----------------------------------------------------------------------

    if use_fatigue:
        # Fatigue params & delta
        F_min_k, F_max_k, lambda_k, mu_k = build_fatigue_params(instance)
        delta_ijk = build_delta_params(instance)

        # Recover-on-idle flag
        if recover_on_idle is None:
            recover_on_idle_flag = bool(meta.get("recover_on_idle", False))
        else:
            recover_on_idle_flag = bool(recover_on_idle)

        def decode_schedule_hf(
            seq: List[int],
        ) -> Tuple[float,
                   Dict[Tuple[int, int], Tuple[float, float]],
                   List[Tuple[int, int, int, float, float]],
                   List[List[float]],
                   Dict[Tuple[int, int], Tuple[float, float]],
                   List[Tuple[int, float, float, float, float]]]:
            """
            Decode a chromosome into a schedule with fatigue & rests.

            Returns
            -------
            cmax : float
            op_times : dict (i,j) -> (start, end)
            transports : list of (robot_id, job_i, op_idx or -1 for final, start, end)
            machine_rests : list of [machine_id, start, end, F_before, F_after]
            """
            # Job state
            job_next_op = [0] * J
            job_last_op_end = [0] * J  # integer time

            # Machine state (integer time, integer fatigue grid)
            machine_time = [0] * (M + 1)  # last event time on each machine (int)
            fmin_int, fmax_int = to_int_fatigue_params(F_min_k, F_max_k, L=FAT_SCALE)
            f_k = [0] * (M + 1)
            for k in range(1, M + 1):
                f_k[k] = fmin_int[k]  # start at (floored) F_min on the grid
            machine_used = [False] * (M + 1)

            # Robot state (integer time)
            robot_time = [0] * R
            robot_loc = [0] * R  # 0..M, all start at Z

            # Schedules
            op_times: Dict[Tuple[int, int], Tuple[float, float]] = {}
            op_fatigue: Dict[Tuple[int, int], Tuple[float, float]] = {}
            transports: List[Tuple[int, int, int, float, float]] = []
            machine_rests: List[List[float]] = []
            idle_rests: List[Tuple[int, float, float, float, float]] = []

            # Helper: forced rest to F_min
            def rest_to_Fmin(k: int, current_time: int) -> Tuple[int, int]:
                """Forced rest until fatigue returns to F_min (integerized like CP-SAT)."""
                f = f_k[k]
                fmin = fmin_int[k]
                mu = mu_k[k]
                if f <= fmin or mu <= 0:
                    return current_time, f

                F = f / FAT_SCALE
                F_min = F_min_k[k]
                # T_rest = (1/μ) ln(F / F_min)  -> integer time via ceil
                T_rest = (1.0 / mu) * math.log(F / F_min)
                if T_rest < 0:
                    T_rest = 0.0
                T_rest_int = int(math.ceil(T_rest))

                start_rest = current_time
                end_rest = current_time + T_rest_int
                machine_rests.append([k, start_rest, end_rest, F, fmin / FAT_SCALE])
                return end_rest, fmin

            # --- schedule all operations (transport + processing) ---
            for job_i in seq:
                j = job_next_op[job_i]
                if j >= job_n_ops[job_i]:
                    continue  # all ops already scheduled for this job

                # Operation info
                m_idx = op_machine[(job_i, j)]
                base_tau = op_duration[(job_i, j)]

                # 1) Schedule transport (same as baseline GA)
                if j == 0:
                    pickup = 0  # Z
                else:
                    pickup = op_machine[(job_i, j - 1)]
                drop = m_idx
                travel_dur = int(sigma[pickup][drop])

                pred_job_time = job_last_op_end[job_i] if j > 0 else 0

                best_robot = 0
                best_t_start = 0
                best_t_end = 10**18

                for r in range(R):
                    empty_dur = int(sigma[robot_loc[r]][pickup])
                    ready_from_prev = robot_time[r] + empty_dur
                    t_start_r = max(pred_job_time, ready_from_prev)
                    t_end_r = t_start_r + travel_dur
                    if t_end_r < best_t_end:
                        best_t_end = t_end_r
                        best_t_start = t_start_r
                        best_robot = r

                r = best_robot
                tT_start = best_t_start
                tT_end = best_t_end
                robot_time[r] = tT_end
                robot_loc[r] = drop
                transports.append((r, job_i, j, tT_start, tT_end))

                # 2) Schedule processing with fatigue
                k = m_idx
                lam = lambda_k[k]
                mu = mu_k[k]
                fmax = fmax_int[k]
                fmin = fmin_int[k]

                earliest_from_job = job_last_op_end[job_i]
                t_candidate = max(tT_end, earliest_from_job, machine_time[k])
                irf = idle_recovery_factor

                # If we allow idle recovery, treat waiting as rest (only if machine already used)
                if recover_on_idle_flag and t_candidate > machine_time[k] and machine_used[k]:
                    idle = t_candidate - machine_time[k]
                    if idle > 0 and mu > 0:
                        F_before_idle = f_k[k] / FAT_SCALE
                        F_idle = F_before_idle * math.exp(-mu * irf * idle)
                        f_k[k] = quantize_fatigue(F_idle)
                        idle_rests.append((k, machine_time[k], t_candidate, F_before_idle, f_k[k] / FAT_SCALE))
                machine_used[k] = True

                # Otherwise, fatigue remains unchanged if idle (event-based).
                if t_candidate > machine_time[k]:
                    machine_time[k] = t_candidate

                t_start = machine_time[k]
                f = f_k[k]

                # We may need a rest before starting if:
                #   - F > Fmax
                #   - or predicted F_after >= 1
                # We'll allow at most a couple of rest iterations.
                iter_guard = 0
                while True:
                    iter_guard += 1
                    if iter_guard > 3:
                        # Safety: avoid infinite cycle; accept and cap fatigue.
                        break

                    # If above fatigue threshold, rest to F_min first
                    if f > fmax:
                        t_start, f = rest_to_Fmin(k, t_start)
                        machine_time[k] = t_start
                        f_k[k] = f

                    # Compute effective processing time at fatigue f (grid-aligned)
                    delta_val = delta_ijk.get((job_i, j, k), 0.0)
                    F = f / FAT_SCALE
                    if F < -0.999999:
                        F = -0.999999  # numeric safety
                    tau_prime = base_tau * (1.0 + delta_val * lam * math.log(1.0 + F))
                    if tau_prime < 0:
                        tau_prime = 0.0
                    tau_prime_int = max(1, round_half_up(tau_prime))

                    # Fatigue after operation (use integerized tau_prime_int like CP-SAT)
                    if lam > 0 and tau_prime_int > 0:
                        F_after_raw = 1.0 - (1.0 - F) * math.exp(-lam * tau_prime_int)
                    else:
                        F_after_raw = F
                    f_after = quantize_fatigue(F_after_raw)

                    # Check the second condition: f_after < FAT_SCALE
                    if f_after >= FAT_SCALE:
                        # Need a pre-rest, if possible.
                        if f > fmin:
                            t_start, f = rest_to_Fmin(k, t_start)
                            machine_time[k] = t_start
                            f_k[k] = f
                            continue
                        else:
                            # Already at F_min but still drives to 1; cap below 1 to avoid issues.
                            f_after = FAT_SCALE - 1

                    # If we reach here, start is legal under our heuristic
                    break

                # Finalize operation (integer time)
                op_start = t_start
                op_end = op_start + tau_prime_int

                machine_time[k] = op_end
                f_k[k] = f_after
                job_last_op_end[job_i] = op_end
                op_times[(job_i, j)] = (op_start, op_end)
                op_fatigue[(job_i, j)] = (f / FAT_SCALE, f_after / FAT_SCALE)

                job_next_op[job_i] += 1

            # 3) Schedule final returns, if requested (robots only)
            final_end_times: List[float] = []
            if last_ttask == 1:
                for job_i in range(J):
                    last_op_idx = job_n_ops[job_i] - 1
                    pickup = op_machine[(job_i, last_op_idx)]
                    drop = 0  # Z
                    travel_dur = int(sigma[pickup][drop])
                    pred_job_time = job_last_op_end[job_i]

                    best_robot = 0
                    best_t_start = 0.0
                    best_t_end = float("inf")

                    for r in range(R):
                        empty_dur = int(sigma[robot_loc[r]][pickup])
                        ready_from_prev = robot_time[r] + empty_dur
                        t_start_r = max(pred_job_time, ready_from_prev)
                        t_end_r = t_start_r + travel_dur
                        if t_end_r < best_t_end:
                            best_t_end = t_end_r
                            best_t_start = t_start_r
                            best_robot = r

                    r = best_robot
                    t_start = best_t_start
                    t_end = best_t_end
                    robot_time[r] = t_end
                    robot_loc[r] = drop
                    transports.append((r, job_i, -1, t_start, t_end))
                    final_end_times.append(t_end)

            # Makespan
            if last_ttask == 1 and final_end_times:
                cmax = max(final_end_times)
            else:
                cmax = max(job_last_op_end) if job_last_op_end else 0.0

            return cmax, op_times, transports, machine_rests, op_fatigue, idle_rests

    # -----------------------------------------------------------------------
    # Unified evaluation wrapper
    # -----------------------------------------------------------------------

    if use_fatigue:
        def evaluate(seq: List[int]):
            return decode_schedule_hf(seq)  # type: ignore
    else:
        def evaluate(seq: List[int]):
            return decode_schedule_baseline(seq)

    # -----------------------------------------------------------------------
    # GA main loop
    # -----------------------------------------------------------------------

    population: List[List[int]] = [random_chromosome() for _ in range(population_size)]
    fitness: List[float] = []
    best_cmax = float("inf")
    best_seq: Optional[List[int]] = None
    best_op_times: Dict[Tuple[int, int], Tuple[float, float]] = {}
    best_transports: List[Tuple[int, int, int, float, float]] = []
    best_rests: List[List[float]] = []
    best_op_fatigue: Dict[Tuple[int, int], Tuple[float, float]] = {}
    best_idle_rests: List[Tuple[int, float, float, float, float]] = []

    # Evaluate initial population
    for chrom in population:
        #cmax, op_times, transports, rests = evaluate(chrom)
        cmax, op_times, transports, rests, op_fatigue, idle_rests = evaluate(chrom)
        fitness.append(cmax)
        if cmax < best_cmax:
            best_cmax = cmax
            best_seq = chrom[:]
            best_op_times = op_times
            best_transports = transports
            best_rests = rests
            best_op_fatigue = op_fatigue
            best_idle_rests = idle_rests

    def tournament_select(k: int = 3) -> List[int]:
        """Tournament selection."""
        best_idx = None
        best_fit = float("inf")
        for _ in range(k):
            idx = random.randrange(len(population))
            if fitness[idx] < best_fit:
                best_fit = fitness[idx]
                best_idx = idx
        return population[best_idx][:]  # type: ignore

    gen = 0
    while gen < generations and (time.time() - start_wall) < time_limit:
        new_population: List[List[int]] = []

        while len(new_population) < population_size:
            p1 = tournament_select()
            p2 = tournament_select()
            # Crossover
            if random.random() < crossover_rate:
                c1, c2 = order_crossover(p1, p2)
            else:
                c1, c2 = p1[:], p2[:]
            # Mutation
            if random.random() < mutation_rate:
                mutate_swap(c1)
            if random.random() < mutation_rate:
                mutate_swap(c2)
            new_population.append(c1)
            if len(new_population) < population_size:
                new_population.append(c2)

        population = new_population
        fitness = []

        for chrom in population:
            #cmax, op_times, transports, rests = evaluate(chrom)
            cmax, op_times, transports, rests, op_fatigue, idle_rests = evaluate(chrom)
            fitness.append(cmax)
            if cmax < best_cmax:
                best_cmax = cmax
                best_seq = chrom[:]
                best_op_times = op_times
                best_transports = transports
                best_rests = rests
                best_op_fatigue = op_fatigue
                best_idle_rests = idle_rests

        gen += 1

    cpu_time = time.time() - start_wall
    status_str = "HEURISTIC"

    if best_seq is None:
        return "NO_SOLUTION", None, cpu_time, {}

    mode_label = "GA-HF" if use_fatigue else "GA"
    if use_fatigue:
        print(
            f"{mode_label} finished: best Cmax={best_cmax:.2f}, "
            f"generations={gen}, time={cpu_time:.2f}s"
        )
    else:
        print(
            f"{mode_label} finished: best Cmax={int(round(best_cmax))}, "
            f"generations={gen}, time={cpu_time:.2f}s"
        )

    # -------------------------
    # Build schedules for Gantt
    # -------------------------
    machine_sched: List[List[int]] = []
    for (i, j), (s, e) in best_op_times.items():
        m_idx = op_machine[(i, j)]
        machine_sched.append([m_idx, i + 1, int(round(s)), int(round(e)), j + 1])

    robot_sched: List[List[int]] = []
    for (r, job_i, op_idx, s, e) in best_transports:
        if op_idx == -1:
            label = 99
        else:
            label = op_idx + 1
        robot_sched.append([r + 1, job_i + 1, int(round(s)), int(round(e)), label])

    # Convert rests to int for plotting (HF only)
    machine_rest_sched_int: List[List[float]] = []
    if use_fatigue:
        for (m_id, rs, re, Fb, Fa) in best_rests:
            machine_rest_sched_int.append([m_id, int(round(rs)), int(round(re)), Fb, Fa])

    # -------------------------
    # Gantt chart
    # -------------------------
    if make_gantt:
        if use_fatigue:
            # Prepare fatigue log for plotting
            F_min_k, F_max_k, lambda_k, mu_k = build_fatigue_params(instance)
            fatigue_log_plt = []
            for (i, j), (s, e) in best_op_times.items():
                m_idx_plt = op_machine[(i, j)]
                Fs_plt, Fe_plt = best_op_fatigue[(i, j)]
                fatigue_log_plt.append((m_idx_plt, s, e, Fs_plt, Fe_plt, False))
            for (m_plt, s_plt, e_plt, Fb_plt, Fa_plt) in machine_rest_sched_int:
                fatigue_log_plt.append((m_plt, s_plt, e_plt, Fb_plt, Fa_plt, True))
            for (m_plt, s_plt, e_plt, Fb_plt, Fa_plt) in best_idle_rests:
                fatigue_log_plt.append((m_plt, s_plt, e_plt, Fb_plt, Fa_plt, True))
            fatigue_log_plt.sort(key=lambda x: x[1])

            _plot_gantt_hf(
                inst_id=inst_id,
                layout_id=layout_id,
                cmax_value=best_cmax,
                machine_sched=machine_sched,
                machine_rest_sched=machine_rest_sched_int,
                robot_sched=robot_sched,
                num_machines=M,
                num_robots=R,
                last_ttask=last_ttask,
                gantt_path=gantt_path,
                to_plot=to_plot,
                fatigue_log=fatigue_log_plt,
                Fmax_per_machine={k: F_max_k[k] for k in range(1, M+1)},
                Fmin_per_machine={k: F_min_k[k] for k in range(1, M+1)},
                lambda_k=lambda_k,
                mu_k=mu_k,
                irf=idle_recovery_factor
            )
        else:
            _plot_gantt_baseline(
                inst_id=inst_id,
                layout_id=layout_id,
                cmax_value=best_cmax,
                machine_sched=machine_sched,
                robot_sched=robot_sched,
                num_machines=M,
                num_robots=R,
                last_ttask=last_ttask,
                gantt_path=gantt_path,
                to_plot=to_plot
            )

    # -----------------------------------------------------------------------
    # KPI calculation
    # -----------------------------------------------------------------------
    # Convert machine_sched and robot_sched to the format expected by KPI functions
    machine_sched_kpi = [(m, j, s, e) for (m, j, s, e, op) in machine_sched]
    robot_sched_kpi   = [(r, j, s, e) for (r, j, s, e, op) in robot_sched]

    Cmax = best_cmax

    # Machine & robot utilization
    Uavg_M, U_M = kpi_machine_utilization(machine_sched_kpi, Cmax, M)
    Uavg_R, U_R = kpi_robot_utilization(robot_sched_kpi, Cmax, R)

    # Fatigue KPIs only in HF mode
    if use_fatigue:
        # Build complete fatigue_log
        fatigue_log = []

        # Processing intervals
        for (i, j), (s, e) in best_op_times.items():
            m_idx = op_machine[(i, j)]
            Fs, Fe = best_op_fatigue[(i, j)]
            fatigue_log.append((m_idx, s, e, Fs, Fe, False))

        # Rest intervals
        for (m, s, e, Fb, Fa) in machine_rest_sched_int:
            fatigue_log.append((m, s, e, Fb, Fa, True))
        for (m, s, e, Fb, Fa) in best_idle_rests:
            fatigue_log.append((m, s, e, Fb, Fa, True))

        # sort the fatigue_log by start time
        fatigue_log.sort(key=lambda x: x[1])

        # Compute KPIs
        Fmax_obs = kpi_max_fatigue(fatigue_log)
        Fgrowth  = kpi_fatigue_accumulation_growth(fatigue_log)
        Nbreaks  = kpi_num_rest_breaks(fatigue_log)
        viol = kpi_fatigue_violations(fatigue_log, max(F_max_k))
        Trest = kpi_total_rest(fatigue_log)
        Facc  = kpi_fatigue_accumulation(fatigue_log)   # exposure
    else:
        Fmax_obs = 0
        Fgrowth  = 0
        Nbreaks  = 0
        viol = 0
        Trest = 0.0
        Facc  = 0.0

    KPIs = {
        "makespan":       Cmax,
        "cpu_time":       cpu_time,
        "avg_machine_util": Uavg_M,
        "machine_util":     U_M.tolist(),
        "avg_robot_util":   Uavg_R,
        "robot_util":       U_R.tolist(),
        "fatigue_viol":     viol,
        "fatigue_max":       Fmax_obs,
        "fatigue_acc":      Facc,
        "fatigue_growth":    Fgrowth,
        "num_rest_breaks":   Nbreaks,
        "total_rest_time":  Trest,
    }

    # Return KPIs together with solver output
    return status_str, float(best_cmax), cpu_time, KPIs
    #return status_str, float(best_cmax), cpu_time


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified GA solver for JSSPT (no fatigue) and JSSPT-HF (with fatigue)."
    )

    parser.add_argument(
        "--instance_path",
        type=str,
        required=True,
        #default=os.path.join("bu_instances_eval", "scenario_1", "EX1_1_S1.json"),
        help="Path to JSON instance file.",
    )
    parser.add_argument(
        "--last_ttask",
        type=int,
        choices=[0, 1],
        default=1,
        help="1: include final return-to-Z transports in Cmax; 0: only operation completion.",
    )
    parser.add_argument(
        "--num_robots",
        type=int,
        default=None,
        help="Number of robots; if omitted, uses instance['robots_nb'].",
    )
    parser.add_argument(
        "--time_limit",
        type=float,
        default=60.0,
        help="Wall-clock time limit in seconds.",
    )
    parser.add_argument(
        "--population_size",
        type=int,
        default=50,
        help="GA population size.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=500,
        help="Maximum number of GA generations.",
    )
    parser.add_argument(
        "--crossover_rate",
        type=float,
        default=0.9,
        help="Crossover probability.",
    )
    parser.add_argument(
        "--mutation_rate",
        type=float,
        default=0.2,
        help="Mutation probability.",
    )
    parser.add_argument(
        "--fatigue",
        type=int,
        choices=[0, 1],
        default=0,
        help="0: no fatigue (JSSPT), 1: enable fatigue (JSSPT-HF). Default=0.",
    )
    parser.add_argument(
        "--recover_on_idle",
        type=int,
        choices=[0, 1],
        default=None,
        help="If fatigue is enabled: 1 to treat idle as rest, 0 to disable. "
             "If omitted, uses instance['meta']['recover_on_idle'] (default False).",
    )
    parser.add_argument(
        "--idle_recovery_factor",
        type=float,
        default=1.0,
        help="Multiplier for fatigue recovery during idle periods (default=1.0).",
    )
    parser.add_argument(
        "--no_gantt",
        action="store_true",
        help="Disable Gantt chart plotting.",
    )
    parser.add_argument(
        "--gantt_path",
        type=str,
        default=None,
        help="Optional explicit path for Gantt PNG.",
    )

    return parser.parse_args()

"""
if __name__ == "__main__":

  # Uncomment this block to use the script in CLI mode
    args = parse_args()

    instance = load_jsspt_instance(args.instance_path)

    use_fatigue = bool(args.fatigue)
    if args.recover_on_idle is None:
        recover_flag = None
    else:
        recover_flag = bool(args.recover_on_idle)


    status, cmax, cpu = solve_jsspt_ga(
        instance=instance,
        use_fatigue=use_fatigue,
        last_ttask=args.last_ttask,
        num_robots=args.num_robots,
        time_limit=args.time_limit,
        population_size=args.population_size,
        generations=args.generations,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        recover_on_idle=recover_flag,
        idle_recovery_factor=args.idle_recovery_factor,
        make_gantt=not args.no_gantt,
        gantt_path=args.gantt_path,
    )"""

us = 1
status, cmax, cpu, kpis = solve_jsspt_ga(
        instance=load_jsspt_instance(os.path.join("bu_instances_eval", "scenario_1", "EX2_3_S1.json")),
        use_fatigue=us,
        last_ttask=1,
        num_robots=None,
        time_limit=120,
        population_size=50,
        generations=500,
        crossover_rate=0.9,
        mutation_rate=0.2,
        recover_on_idle=False,
        idle_recovery_factor=1.0,
        make_gantt=False,
        to_plot=False,
        gantt_path=None,
    )

mode_name = "GA-HF" if us else "GA"
print(f"Done ({mode_name}):", status, cmax, f"{cpu:.2f}s")

print("\n=== KPIs ===")
for k, v in kpis.items():
    print(f"{k}: {v}")
