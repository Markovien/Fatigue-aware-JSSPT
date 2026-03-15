# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 09:41:33 2026

@author: Kader SANOGO

meta_vns_scheduler.py
---------------------

Variable Neighborhood Search (VNS) metaheuristic for JSSPT/JSSPT-HF.

- Uses VNS algorithm for optimization.
- Neighborhoods: Swap, Insert, Invert.
- Shaking and First-Improvement Local Search.

Usage examples:
--------------
# Baseline JSSPT (no fatigue):
python meta_vns_scheduler.py --instance_path bu_instances_eval/scenario_1/EX5_2_S1.json

# JSSPT-HF (with fatigue):
python meta_vns_scheduler.py --instance_path bu_instances_eval/scenario_1/EX1_1_S1.json --fatigue 1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

#----------------------------------------------------------
# Helper functions: sort_files & evaluate_and_export_excel
#----------------------------------------------------------

def sort_files(directory, save_index=False, index_filename="index.json"):
    """
    Lists and sorts files in a directory according to:
      - First: subindex (_1, _2, _3, _4, ...)
      - Second: experiment number (EX1..EX10..)
    """
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
            subindex   = int(match.group(2))
            return (subindex, experiment)
        return (9999, 9999)

    sorted_files = sorted(files, key=sort_key)
    if save_index:
        index_path = os.path.join(directory, index_filename)
        with open(index_path, "w") as f:
            json.dump(sorted_files, f, indent=4)
        print(f"✔ Index saved: {index_path}")

    return sorted_files

def evaluate_and_export_excel(root, excel_path="VNS_results.xlsx"):
    files = sort_files(root)
    results = []

    for inst_path in files:
        instance = load_jsspt_instance(inst_path)
        # Solve using VNS
        status, cmax, bct, cpu, kpis = solve_jsspt_vns(
                instance=instance,
                use_fatigue=1,
                last_ttask=1,
                num_robots=None,
                time_limit=60,
                k_max=15,                  # VNS param
                max_no_improv=500,        # VNS param (optional stop criteria)
                recover_on_idle=None,
                idle_recovery_factor=1.0,
                make_gantt=True,
                gantt_path=None,
                to_plot=False
            )

        results.append({
            "instance": os.path.basename(inst_path)[:-7].replace('_', ''),
            "cmax": cmax,
            "status": status,
            "cpu": bct,
            **kpis
        })

        inst_name = os.path.basename(inst_path)[:-7].replace('_', '')
        #print("Done:", status, cmax, f"{cpu:.2f}s")
        print(f"Instance: {inst_name} | Best Cmax: {cmax:.2f} | Best Cmax time: {bct:.2f}s | Time limit: {cpu:.2f}s")
        print("=== KPIs ===")
        for k, v in kpis.items():
            print(f"{k}: {v}")
        print("=== End KPIs ===\n")

    df = pd.DataFrame(results)
    df.to_excel(excel_path, index=False)
    print(f"\n📌 Results exported to: {excel_path}")

#----------------------------------------------------------
# KPIs computing functions (Identical to GA)
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
    if not fatigue_log:
        return 0.0
    maxF = max(max(Fs, Fe) for (_, _, _, Fs, Fe, _) in fatigue_log)
    return maxF

def kpi_fatigue_accumulation_growth(fatigue_log):
    total = 0.0
    for (_, _, _, Fs, Fe, is_rest) in fatigue_log:
        if not is_rest:
            total += max(0.0, Fe - Fs)
    return total

def kpi_num_rest_breaks(fatigue_log):
    return sum(1 for (_, _, _, _, _, is_rest) in fatigue_log if is_rest)

def kpi_total_rest(fatigue_log):
    return sum((e - s) for (_, s, e, Fs, Fe, rest) in fatigue_log if rest)

def kpi_fatigue_accumulation(fatigue_log):
    return sum(Fs * (e - s) for (_, s, e, Fs, Fe, rest) in fatigue_log if not rest)

def kpi_tradeoff(Cmax, Facc, alpha=0.5):
    return alpha*Cmax + (1-alpha)*Facc

# ---------------------------------------------------------------------------
# Instance loading & Params
# ---------------------------------------------------------------------------

def load_jsspt_instance(path: str) -> Dict:
    with open(path, "r") as f:
        data = json.load(f)
    return data

def build_fatigue_params(instance: Dict) -> Tuple[List[float], List[float], List[float], List[float]]:
    M = instance["machines_nb"]
    meta = instance.get("meta", {})
    fat = meta.get("fatigue", {})
    mode = fat.get("mode", "uniform")

    F_min_k = [0.0] * (M + 1)
    F_max_k = [1.0] * (M + 1)
    lambda_k = [0.0] * (M + 1)
    mu_k = [0.0] * (M + 1)

    if mode == "per_human":
        F_min_list = fat["F_min"]
        F_max_list = fat["F_max"]
        lambda_list = fat["lambda"]
        mu_list = fat["mu"]
        for k in range(1, M + 1):
            F_min_k[k] = float(F_min_list[k - 1])
            F_max_k[k] = float(F_max_list[k - 1])
            lambda_k[k] = float(lambda_list[k - 1])
            mu_k[k] = float(mu_list[k - 1])
    else:
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
    jobs_data = instance["jobs"]
    meta = instance.get("meta", {})
    delta_meta = meta.get("delta", {})
    mode = delta_meta.get("mode", "uniform")
    default_val = float(delta_meta.get("value", 0.0))

    delta_ijk: Dict[Tuple[int, int, int], float] = {}
    for i, job in enumerate(jobs_data):
        for j, op in enumerate(job["ops"]):
            m_idx = op["machine_index"]
            if "delta" in op:
                d = float(op["delta"])
            else:
                dur = float(op["duration"])
                if mode == "per_op_duration":
                    d = 0.1 if dur <= 10 else (0.3 if dur <= 15 else 0.5)
                else:
                    d = default_val
            delta_ijk[(i, j, m_idx)] = d
    return delta_ijk

# ---------------------------------------------------------------------------
# Gantt Plotting Functions (Identical to GA)
# ---------------------------------------------------------------------------

def _plot_gantt_baseline(inst_id, layout_id, cmax_value, machine_sched, robot_sched,
                         num_machines, num_robots, last_ttask, gantt_path, to_plot):
    fig, ax = plt.subplots(figsize=(20, 8))
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
    ax.set_xlabel("Time", fontsize=18, weight="bold")
    ax.tick_params(labelsize=14)
    ax.minorticks_on()
    ax.grid(which="major", linestyle="-", linewidth=0.5, color="red")
    ax.grid(which="minor", linestyle=":", linewidth=0.5, color="black")
    ax.set_xlim(0, int(cmax_value) + 10)

    # Machines
    for m_entry in machine_sched:
        machine_id, job_id, start, end, op_idx = m_entry
        color = mcolors[(job_id - 1) % len(mcolors)]
        text = r"$O_{%d,%d}$" % (job_id, op_idx)
        x = start + (end - start) / 2 - 2
        y = 4 + yticks[machine_id - 1]
        ax.text(x, y, text, fontsize=14, weight="bold")
        ax.broken_barh([(start, end - start)], (yticks[machine_id - 1], 9), facecolors=color)

    # Robots
    for r_entry in robot_sched:
        robot_id, job_id, start, end, op_idx = r_entry
        color = mcolors[(job_id - 1) % len(mcolors)]
        text = r"$T_{fp%d}$" % job_id if op_idx == 99 else r"$T_{O_{%d,%d}}$" % (job_id, op_idx)
        x = start + (end - start) / 2 - 2
        y = 4 + yticks[num_machines + robot_id - 1]
        ax.text(x, y, text, fontsize=14, weight="bold")
        ax.broken_barh([(start, end - start)], (yticks[num_machines + robot_id - 1], 9), facecolors=color)

    title = f"{inst_id}\nResolution method=VNS (no fatigue), Cmax={int(round(cmax_value))}, last_ttask={last_ttask}"
    ax.set_title(title, fontsize=20, weight="bold")
    plt.tight_layout()
    fname = f"{inst_id}_VNS_gantt.png" if gantt_path is None else gantt_path
    if to_plot:
      plt.show()
    plt.savefig(fname, dpi=300)
    plt.close(fig)
    print(f"Gantt chart saved to: {fname}")

def _plot_gantt_hf(inst_id, layout_id, cmax_value, machine_sched, machine_rest_sched,
                   robot_sched, num_machines, num_robots, last_ttask, gantt_path, to_plot,
                   fatigue_log, Fmax_per_machine, Fmin_per_machine, lambda_k, mu_k, irf):
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

    # Plot smooth curves per machine
    for m in range(1, num_machines + 1):
        m_logs = [log for log in fatigue_log if log[0] == m]
        m_logs.sort(key=lambda x: x[1])
        m_color = mcolors[(m - 1) % len(mcolors)]
        
        # Thresholds
        ax_fat.axhline(Fmax_per_machine[m], color=m_color, linestyle='--', alpha=0.2)
        ax_fat.axhline(Fmin_per_machine[m], color=m_color, linestyle='--', alpha=0.2)

        for i, (m_id, s, e, Fs, Fe, is_recovery) in enumerate(m_logs):
            if e <= s: continue
            
            # Connect gaps with constant fatigue line
            if i > 0:
                prev_e = m_logs[i-1][2]
                if s > prev_e:
                    ax_fat.plot([prev_e, s], [m_logs[i-1][4], Fs], color=m_color, linewidth=2, alpha=0.8)

            t_vals = np.linspace(s, e, 20)
            if is_recovery:
                # Recovery: detect if idle or machine_rest by checking machine_rest_sched
                is_machine_rest = any(r[0] == m and abs(r[1] - s) < 1e-3 and abs(r[2] - e) < 1e-3 for r in machine_rest_sched)
                mu_eff = mu_k[m] if is_machine_rest else mu_k[m] * irf
                if mu_eff > 0:
                    # Decay formula
                    f_vals = Fs * np.exp(-mu_eff * (t_vals - s))
                    # Only clamp to Fmin for actual machine rests, not idle recovery
                    is_machine_rest = any(r[0] == m and abs(r[1] - s) < 1e-3 and abs(r[2] - e) < 1e-3 for r in machine_rest_sched)
                    if is_machine_rest:
                        f_vals = np.maximum(f_vals, Fmin_per_machine[m])
                else:
                    f_vals = np.full_like(t_vals, Fs)
            else:
                # Work: F(t) = 1 - (1 - Fs) * exp(-lam * (t - s))
                lam = lambda_k[m]
                if lam > 0:
                    f_vals = 1.0 - (1.0 - Fs) * np.exp(-lam * (t_vals - s))
                else:
                    f_vals = np.full_like(t_vals, Fs)
            
            # Add label only for the first segment of each machine to avoid duplicate legend entries
            label = f"M{m}" if i == 0 else None
            ax_fat.plot(t_vals, f_vals, color=m_color, linewidth=2, alpha=0.8, label=label)

    ax_fat.legend(loc='upper right', bbox_to_anchor=(1.05, 1))

    title = f"{inst_id} [{scenario}]\nResolution: VNS + Fatigue, Cmax={int(round(cmax_value))}, last_ttask={last_ttask}"
    ax.set_title(title, fontsize=20, weight="bold")
    plt.tight_layout()
    fname = f"{inst_id}_VNS_HF_gantt.png" if gantt_path is None else gantt_path
    if to_plot:
      plt.show()
    plt.savefig(fname, dpi=300)
    plt.close(fig)
    print(f"Gantt chart saved to: {fname}")


# ---------------------------------------------------------------------------
# VNS Solver
# ---------------------------------------------------------------------------

def solve_jsspt_vns(
    instance: Dict,
    use_fatigue: bool = False,
    last_ttask: int = 1,
    num_robots: Optional[int] = None,
    time_limit: float = 60.0,
    k_max: int = 10,       # Number of neighborhoods
    max_no_improv: int = 1000,
    recover_on_idle: Optional[bool] = False,
    idle_recovery_factor: float = 1.0,
    make_gantt: bool = True,
    to_plot: bool = False,
    gantt_path: Optional[str] = None,
) -> Tuple[str, Optional[float], float, float, Dict]:

    start_wall = time.time()

    # --- Instance Setup ---
    inst_id = instance.get("instance_id", "unknown_instance")
    layout_id = instance.get("layout_id", -1)
    M = instance["machines_nb"]
    sigma = instance["sigma"]
    jobs_data = instance["jobs"]
    meta = instance.get("meta", {})
    R = num_robots if num_robots is not None else instance.get("robots_nb", 1)

    # Build operation lookups
    J = len(jobs_data)
    op_duration: Dict[Tuple[int, int], float] = {}
    op_machine: Dict[Tuple[int, int], int] = {}
    job_n_ops: List[int] = []
    for i, job in enumerate(jobs_data):
        n_ops = job["n_ops"]
        job_n_ops.append(n_ops)
        for j, op in enumerate(job["ops"]):
            m_idx = op["machine_index"]
            dur = float(op["duration"])
            op_machine[(i, j)] = m_idx
            op_duration[(i, j)] = dur

    # Base sequence for chromosomes
    base_seq: List[int] = []
    for i in range(J):
        base_seq.extend([i] * job_n_ops[i])

    # --- Solution Representation ---
    def random_chromosome() -> List[int]:
        chrom = base_seq[:]
        random.shuffle(chrom)
        return chrom

    # --- Evaluation Functions (same as GA) ---
    # Baseline (No Fatigue)
    def decode_schedule_baseline(seq: List[int]):
        job_next_op = [0] * J
        job_last_op_end = [0.0] * J
        machine_ready = [0.0] * (M + 1)
        robot_time = [0.0] * R
        robot_loc = [0] * R

        op_times = {}
        transports = []
        rests = [] # Empty

        # Decoding loop
        for job_i in seq:
            j = job_next_op[job_i]
            if j >= job_n_ops[job_i]: continue

            m_idx = op_machine[(job_i, j)]
            proc_dur = op_duration[(job_i, j)]

            # Transport
            pickup = 0 if j == 0 else op_machine[(job_i, j - 1)]
            drop = m_idx
            travel_dur = float(sigma[pickup][drop])
            pred_job_time = job_last_op_end[job_i] if j > 0 else 0.0

            best_robot = 0
            best_t_end = float("inf")
            best_t_start = 0.0

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
            transports.append((r, job_i, j, t_start, t_end))

            # Machine Proc
            op_start = max(t_end, machine_ready[m_idx])
            op_end = op_start + proc_dur
            machine_ready[m_idx] = op_end
            job_last_op_end[job_i] = op_end
            op_times[(job_i, j)] = (op_start, op_end)
            job_next_op[job_i] += 1

        # Final transports
        final_end_times = []
        if last_ttask == 1:
            for job_i in range(J):
                last_op_idx = job_n_ops[job_i] - 1
                pickup = op_machine[(job_i, last_op_idx)]
                drop = 0
                travel_dur = float(sigma[pickup][drop])
                pred_job_time = job_last_op_end[job_i]

                best_t_end = float("inf")
                best_start = 0.0
                best_robot = 0
                for r in range(R):
                    empty_dur = float(sigma[robot_loc[r]][pickup])
                    ready = robot_time[r] + empty_dur
                    t_s = max(pred_job_time, ready)
                    t_e = t_s + travel_dur
                    if t_e < best_t_end:
                        best_t_end = t_e
                        best_start = t_s
                        best_robot = r

                transports.append((best_robot, job_i, -1, best_start, best_t_end))
                final_end_times.append(best_t_end)
                robot_time[best_robot] = best_t_end
                robot_loc[best_robot] = drop

        if last_ttask == 1 and final_end_times:
            cmax = max(final_end_times)
        else:
            cmax = max(job_last_op_end) if job_last_op_end else 0.0

        return float(cmax), op_times, transports, rests, {}, []

    # Fatigue (HF)
    if use_fatigue:
        F_min_k, F_max_k, lambda_k, mu_k = build_fatigue_params(instance)
        delta_ijk = build_delta_params(instance)
        rec_on_idle = bool(meta.get("recover_on_idle", False)) if recover_on_idle is None else bool(recover_on_idle)
        irf = float(meta.get("idle_recovery_factor", 1.0)) if idle_recovery_factor == 1.0 else idle_recovery_factor

        def decode_schedule_hf(seq: List[int]):
            job_next_op = [0]*J
            job_last_op_end = [0.0]*J
            machine_time = [0.0]*(M+1)
            F_k = list(F_min_k) # Copy initial fatigue
            robot_time = [0.0]*R
            robot_loc = [0]*R
            machine_used = [False] * (M + 1)

            op_times = {}
            op_fatigue = {}
            transports = []
            machine_rests = []
            idle_rests = []

            def rest_to_Fmin(k, current_time):
                F = F_k[k]
                Fmin = F_min_k[k]
                mu = mu_k[k]
                if F <= Fmin or mu <= 0: return current_time, F
                T_rest = (1.0/mu) * math.log(F/Fmin)
                if T_rest < 0: T_rest = 0.0
                end_rest = current_time + T_rest
                machine_rests.append([k, current_time, end_rest, F, Fmin])
                return end_rest, Fmin

            for job_i in seq:
                j = job_next_op[job_i]
                if j >= job_n_ops[job_i]: continue

                m_idx = op_machine[(job_i, j)]
                base_tau = op_duration[(job_i, j)]

                # Transport logic
                pickup = 0 if j == 0 else op_machine[(job_i, j-1)]
                drop = m_idx
                travel_dur = float(sigma[pickup][drop])
                pred_job_time = job_last_op_end[job_i] if j > 0 else 0.0

                best_robot = 0
                best_t_end = float("inf")
                best_t_start = 0.0
                for r in range(R):
                    empty_dur = float(sigma[robot_loc[r]][pickup])
                    ready = robot_time[r] + empty_dur
                    ts = max(pred_job_time, ready)
                    te = ts + travel_dur
                    if te < best_t_end:
                         best_t_end = te
                         best_t_start = ts
                         best_robot = r

                r = best_robot
                tT_start = best_t_start
                tT_end = best_t_end
                robot_time[r] = tT_end
                robot_loc[r] = drop
                transports.append((r, job_i, j, tT_start, tT_end))

                # Processing with Fatigue
                k = m_idx
                lam = lambda_k[k]
                mu = mu_k[k]
                Fmax = F_max_k[k]
                Fmin = F_min_k[k]

                earliest = job_last_op_end[job_i]
                t_candidate = max(tT_end, earliest, machine_time[k])

                if rec_on_idle and t_candidate > machine_time[k] and machine_used[k]:
                    idle = t_candidate - machine_time[k]
                    if idle > 0 and mu > 0:
                        F_before_idle = F_k[k]
                        F_idle = F_before_idle * math.exp(-mu * irf * idle)
                        F_k[k] = F_idle
                        idle_rests.append((k, machine_time[k], t_candidate, F_before_idle, F_idle))
                    machine_time[k] = t_candidate
                elif t_candidate > machine_time[k]:
                    machine_time[k] = t_candidate
                
                machine_used[k] = True

                t_start = machine_time[k]
                F = F_k[k]

                iter_guard = 0
                while True:
                    iter_guard += 1
                    if iter_guard > 3: break

                    if F > Fmax + 1e-9:
                        t_start, F = rest_to_Fmin(k, t_start)
                        machine_time[k] = t_start
                        F_k[k] = F

                    d_val = delta_ijk.get((job_i, j, k), 0.0)
                    if F < -0.9999: F = -0.9999
                    tau_prime = base_tau * (1.0 + d_val * lam * math.log(1.0 + F))
                    if tau_prime < 0: tau_prime = 0.0

                    if lam > 0 and tau_prime > 0:
                        F_after = 1.0 - (1.0 - F) * math.exp(-lam * tau_prime)
                    else:
                        F_after = F

                    if F_after >= 1.0 - 1e-9:
                        if F > Fmin + 1e-9:
                            t_start, F = rest_to_Fmin(k, t_start)
                            machine_time[k] = t_start
                            F_k[k] = F
                            continue
                        else:
                            F_after = 1.0 - 1e-9
                    break

                op_start = t_start
                op_end = op_start + tau_prime
                machine_time[k] = op_end
                F_k[k] = F_after
                job_last_op_end[job_i] = op_end
                op_times[(job_i, j)] = (op_start, op_end)
                op_fatigue[(job_i, j)] = (F, F_after)
                job_next_op[job_i] += 1

            # Final Returns
            final_end_times = []
            if last_ttask == 1:
                for job_i in range(J):
                    last_op_idx = job_n_ops[job_i] - 1
                    pickup = op_machine[(job_i, last_op_idx)]
                    drop = 0
                    travel = float(sigma[pickup][drop])
                    pred = job_last_op_end[job_i]

                    best_end = float("inf")
                    best_start = 0.0
                    best_r = 0
                    for r in range(R):
                         ready = robot_time[r] + float(sigma[robot_loc[r]][pickup])
                         ts = max(pred, ready)
                         te = ts + travel
                         if te < best_end:
                             best_end = te
                             best_start = ts
                             best_r = r

                    r = best_r
                    transports.append((r, job_i, -1, best_start, best_end))
                    final_end_times.append(best_end)
                    robot_time[r] = best_end
                    robot_loc[r] = drop

            if last_ttask == 1 and final_end_times:
                cmax = max(final_end_times)
            else:
                cmax = max(job_last_op_end) if job_last_op_end else 0.0

            return cmax, op_times, transports, machine_rests, op_fatigue, idle_rests

    # --- Wrapper ---
    if use_fatigue:
        def evaluate(seq): return decode_schedule_hf(seq)
    else:
        def evaluate(seq): return decode_schedule_baseline(seq)

    # --- Neighborhood Functions ---
    def neighbor_swap(seq: List[int]) -> List[int]:
        n = len(seq)
        if n < 2: return seq[:]
        i, j = random.sample(range(n), 2)
        new_seq = seq[:]
        new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
        return new_seq

    def neighbor_insert(seq: List[int]) -> List[int]:
        n = len(seq)
        if n < 2: return seq[:]
        # Pick element at i, move to j
        i, j = random.sample(range(n), 2)
        new_seq = seq[:]
        val = new_seq.pop(i)
        new_seq.insert(j, val)
        return new_seq

    def neighbor_invert(seq: List[int]) -> List[int]:
        n = len(seq)
        if n < 2: return seq[:]
        i, j = sorted(random.sample(range(n), 2))
        # Reverse slice [i:j+1]
        new_seq = seq[:]
        new_seq[i:j+1] = reversed(new_seq[i:j+1])
        return new_seq

    def shake(seq: List[int], k: int) -> List[int]:
        """Apply perturbation based on k.
           k=1: Perform N random swaps (N small, e.g. 2)
           k=2: Perform N random inserts
           k=3: Perform N random inverts
        """
        new_seq = seq[:]
        perturbations = 2 # Strength of shaking

        if k == 1:
            for _ in range(perturbations): new_seq = neighbor_swap(new_seq)
        elif k == 2:
           for _ in range(perturbations): new_seq = neighbor_insert(new_seq)
        else: # k >= 3
           for _ in range(perturbations): new_seq = neighbor_invert(new_seq)

        return new_seq

    def local_search(seq: List[int], current_cmax: float) -> Tuple[List[int], float]:
        """First improvement local search."""
        # Try a limited number of random neighbors of each type
        # Or better: systematic scan is hard with duplicates, so random sampling

        best_local_seq = seq[:]
        best_local_cmax = current_cmax

        improved = True
        step = 0
        MAX_STEPS = 50 # Limit local search steps per VNS iteration

        while improved and step < MAX_STEPS:
            improved = False
            step += 1

            # Try a random move from each neighborhood type
            candidates = []
            candidates.append(neighbor_swap(best_local_seq))
            candidates.append(neighbor_insert(best_local_seq))
            candidates.append(neighbor_invert(best_local_seq))

            for cand in candidates:
                res = evaluate(cand)
                c_cmax = res[0]
                if c_cmax < best_local_cmax:
                    best_local_cmax = c_cmax
                    best_local_seq = cand
                    improved = True
                    break # First improvement

        return best_local_seq, best_local_cmax

    # --- VNS Main Loop ---

    # Initial Solution
    current_seq = random_chromosome()
    res = evaluate(current_seq)
    current_cmax = res[0]

    best_seq = current_seq[:]
    best_cmax = current_cmax
    best_cmax_time = 0.0
    best_res = res

    no_improv_count = 0

    print(f"Start VNS: Initial Cmax={best_cmax:.2f}")

    while (time.time() - start_wall) < time_limit:
        k = 1
        while k <= k_max:
            # 1. Shaking
            shaken_seq = shake(current_seq, k)
            res_shaken = evaluate(shaken_seq)
            shaken_cmax = res_shaken[0]

            # 2. Local Search
            # Pass shaken_cmax to optimize starting point
            ls_seq, ls_cmax = local_search(shaken_seq, shaken_cmax)

            # 3. Move Acceptance (Descent)
            if ls_cmax < current_cmax:
                current_seq = ls_seq
                current_cmax = ls_cmax
                k = 1 # Reset
                no_improv_count = 0

                # Update global best
                if current_cmax < best_cmax:
                    best_cmax = current_cmax
                    best_cmax_time = time.time() - start_wall
                    best_seq = current_seq[:]
                    # Evaluate fully to get all details
                    best_res = evaluate(best_seq)
                    print(f"New Best: {best_cmax:.2f} (Time: {time.time()-start_wall:.2f}s)")
            else:
                k += 1 # Next neighborhood
                no_improv_count += 1

        if no_improv_count > max_no_improv:
            # Restart if stuck too long?
            # Or just continue random walking via shaking
            pass

    cpu_time = time.time() - start_wall
    status_str = "HEURISTIC"

    if best_seq is None:
        return "NO_SOLUTION", None, 0.0, cpu_time, {}

    mode_label = "VNS-HF" if use_fatigue else "VNS"
    print(f"{mode_label} finished: best Cmax={best_cmax:.2f}, obtained after {best_cmax_time:.2f}s")

    # Unpack best results
    cmax, op_times, transports, rests, op_fatigue, idle_rests = best_res

    # -------------------------
    # Build schedules for Gantt
    # -------------------------
    machine_sched = []
    for (i, j), (s, e) in op_times.items():
        m_idx = op_machine[(i, j)]
        machine_sched.append([m_idx, i + 1, int(round(s)), int(round(e)), j + 1])

    robot_sched = []
    for (r, job_i, op_idx, s, e) in transports:
        label = 99 if op_idx == -1 else op_idx + 1
        robot_sched.append([r + 1, job_i + 1, int(round(s)), int(round(e)), label])

    machine_rest_sched_int = []
    if use_fatigue and rests:
        for (m_id, rs, re, Fb, Fa) in rests:
            machine_rest_sched_int.append([m_id, int(round(rs)), int(round(re)), Fb, Fa])

    # -------------------------
    # Gantt chart
    # -------------------------
    if make_gantt:
        if use_fatigue:
            # Prepare fatigue log for plotting
            fatigue_log_plt = []
            for (i, j), (s, e) in op_times.items():
                m_idx_plt = op_machine[(i, j)]
                Fs_plt, Fe_plt = op_fatigue[(i, j)]
                fatigue_log_plt.append((m_idx_plt, s, e, Fs_plt, Fe_plt, False))
            for (m_plt, s_plt, e_plt, Fb_plt, Fa_plt) in machine_rest_sched_int:
                fatigue_log_plt.append((m_plt, s_plt, e_plt, Fb_plt, Fa_plt, True))
            for (m_plt, s_plt, e_plt, Fb_plt, Fa_plt) in idle_rests:
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
                irf=irf
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

    # -------------------------
    # KPIs
    # -------------------------
    machine_sched_kpi = [(m, j, s, e) for (m, j, s, e, op) in machine_sched]
    robot_sched_kpi   = [(r, j, s, e) for (r, j, s, e, op) in robot_sched]
    Uavg_M, U_M = kpi_machine_utilization(machine_sched_kpi, best_cmax, M)
    Uavg_R, U_R = kpi_robot_utilization(robot_sched_kpi, best_cmax, R)

    if use_fatigue:
        fatigue_log = []
        for (i, j), (s, e) in op_times.items():
            m_idx = op_machine[(i, j)]
            Fs, Fe = op_fatigue[(i, j)]
            fatigue_log.append((m_idx, s, e, Fs, Fe, False))
        for (m, s, e, Fb, Fa) in machine_rest_sched_int:
            fatigue_log.append((m, s, e, Fb, Fa, True))
        fatigue_log.sort(key=lambda x: x[1])

        Fmax_obs = kpi_max_fatigue(fatigue_log)
        Fgrowth  = kpi_fatigue_accumulation_growth(fatigue_log)
        Nbreaks  = kpi_num_rest_breaks(fatigue_log)
        viol = kpi_fatigue_violations(fatigue_log, max(F_max_k))
        Trest = kpi_total_rest(fatigue_log)
        Facc  = kpi_fatigue_accumulation(fatigue_log)
        trade = kpi_tradeoff(best_cmax, Facc, alpha=0.5)
    else:
        Fmax_obs = Fgrowth = Nbreaks = viol = Trest = Facc = 0
        trade = best_cmax

    KPIs = {
        "makespan":       best_cmax,
        "cpu_time":       best_cmax_time,
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
        "tradeoff":         trade,
    }

    return status_str, float(best_cmax), best_cmax_time, cpu_time, KPIs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VNS solver for JSSPT (no fatigue) and JSSPT-HF (with fatigue)."
    )
    parser.add_argument("--instance_path", type=str, required=True, help="Path to JSON instance file.")
    parser.add_argument("--last_ttask", type=int, choices=[0, 1], default=1, help="1: include final return.")
    parser.add_argument("--num_robots", type=int, default=None, help="Number of robots.")
    parser.add_argument("--time_limit", type=float, default=60.0, help="Wall-clock time limit.")

    # VNS specific
    parser.add_argument("--k_max", type=int, default=3, help="Max neighborhood size for shaking.")
    parser.add_argument("--max_no_improv", type=int, default=500, help="Max iterations without improvement.")

    parser.add_argument("--fatigue", type=int, choices=[0, 1], default=1, help="Enable fatigue.")
    parser.add_argument("--recover_on_idle", type=int, choices=[0, 1], default=None, help="Idle recovery.")
    parser.add_argument("--no_gantt", action="store_true", help="Disable Gantt.")
    parser.add_argument("--idle_recovery_factor", type=float, default=1.0, help="Recovery factor during idle time ([0,1]).")
    parser.add_argument("--gantt_path", type=str, default=None, help="Explicit path for Gantt PNG.")

    return parser.parse_args()

"""if __name__ == "__main__":
    args = parse_args()

    instance = load_jsspt_instance(args.instance_path)
    use_fatigue = bool(args.fatigue)
    recover_flag = bool(args.recover_on_idle) if args.recover_on_idle is not None else None

    # Solve
    status, cmax, cpu, kpis = solve_jsspt_vns(
        instance=instance,
        use_fatigue=use_fatigue,
        last_ttask=args.last_ttask,
        num_robots=args.num_robots,
        time_limit=args.time_limit,
        k_max=args.k_max,
        max_no_improv=args.max_no_improv,
        recover_on_idle=recover_flag,
        make_gantt=not args.no_gantt,
        gantt_path=args.gantt_path,
    )"""

inst_path = os.path.join("bu_instances_eval", "scenario_1", "EX6_1_S1.json")
instance = load_jsspt_instance(inst_path)
use_fatigue = True
recover_flag = True

# Solve
status, cmax, bct, cpu, kpis = solve_jsspt_vns(
    instance=instance,
    use_fatigue=use_fatigue,
    last_ttask=1,
    num_robots=None,
    time_limit=30,
    k_max=50,
    max_no_improv=500,
    recover_on_idle=recover_flag,
    idle_recovery_factor=0.5,
    make_gantt=True,
    gantt_path=None,
    to_plot=True
)

mode_name = "VNS-HF" if use_fatigue else "VNS"
print(f"Done ({mode_name}):", status)#, cmax, f"{cpu:.2f}s")
print(f"Instance: {os.path.basename(inst_path)[:-7].replace('_', '')} | Best Cmax: {cmax:.2f} | Best Cmax time: {bct:.2f}s | Time limit: {cpu:.2f}s")
print("\n=== KPIs ===")
for k, v in kpis.items():
    print(f"{k}: {v}")
