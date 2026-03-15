# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 16:02:08 2025

@author: Kader SANOGO

adapted_bu_instances_generator.py
---------------------------
Generate extended Bilge & Ulusoy (1995)-style instances with
fatigue and difficulty parameters for two evaluation scenarios:
- Scenario 1 (S1): Uniform fatigue parameters for all workers and
  a uniform δ for all operations.
- Scenario 2 (S2): Worker-specific fatigue parameters and an
  operation-specific δ based on processing duration. This scenario is
  planned for future work.

The generated instances (layouts, job sets, and fatigue parameters)
are converted into JSON.
Outputs:
    bu_instances_eval/
        scenario_1/
            EX1_1.json
            EX1_2.json
            ...
        scenario_2/
            EX1_1.json
            EX1_2.json
            ...
        bu_eval_index_s1.json
        bu_eval_index_s2.json

Each output JSON includes:
- instance_id, layout_id, layout_name
- robots_nb, machines_nb
- locations with (approximate) 2D coordinates
- symmetric sigma matrix
- jobs list
- metadata block
"""

import json
import os
import math, random
from datetime import datetime, UTC

# ---------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------
ROOT_DIR = "bu_instances_eval"
SCENARIO_1_DIR = os.path.join(ROOT_DIR, "scenario_1")
SCENARIO_2_DIR = os.path.join(ROOT_DIR, "scenario_2")
os.makedirs(SCENARIO_1_DIR, exist_ok=True)
os.makedirs(SCENARIO_2_DIR, exist_ok=True)

# -------------------------------------------------------
# Layout coordinates (Approximation)
# Note: these coordinates are only for 2D representation.
#       Only the transportation times matter.
# -------------------------------------------------------
BU_LAYOUTS = {
    1: {
        "layout_name": "layout_1",
        "locations": [
            {"id": "Z", "index": 0, "pos": [0.0, 0.0]},
            {"id": "M1", "index": 1, "pos": [-3.0, 3.0]},
            {"id": "M2", "index": 2, "pos": [-1.0, 3.0]},
            {"id": "M3", "index": 3, "pos": [1.0, 3.0]},
            {"id": "M4", "index": 4, "pos": [3.0, 3.0]},
        ],
    },
    2: {
        "layout_name": "layout_2",
        "locations": [
            {"id": "Z", "index": 0, "pos": [0.0, 0.0]},
            {"id": "M1", "index": 1, "pos": [3.0, 1.0]},
            {"id": "M2", "index": 2, "pos": [5.0, 1.0]},
            {"id": "M3", "index": 3, "pos": [5.0, -1.0]},
            {"id": "M4", "index": 4, "pos": [3.0, -1.0]},
        ],
    },
    3: {
        "layout_name": "layout_3",
        "locations": [
            {"id": "Z", "index": 0, "pos": [0.0, 0.0]},
            {"id": "M1", "index": 1, "pos": [-2.0, 0.0]},
            {"id": "M2", "index": 2, "pos": [-2.5, 1.5]},
            {"id": "M3", "index": 3, "pos": [2.5, 1.5]},
            {"id": "M4", "index": 4, "pos": [2.0, 0.0]},
        ],
    },
    4: {
        "layout_name": "layout_4",
        "locations": [
            {"id": "Z", "index": 0, "pos": [0.0, 0.0]},
            {"id": "M1", "index": 1, "pos": [3.0, 1.0]},
            {"id": "M2", "index": 2, "pos": [8.0, 1.0]},
            {"id": "M3", "index": 3, "pos": [5.0, -3.0]},
            {"id": "M4", "index": 4, "pos": [8.0, -4.0]},
        ],
    },
}

# ----------------------------------------------
# Symmetric travel-time matrices for each layout
# ----------------------------------------------
BU_SIGMAS = {
    1: [[0, 6, 8, 8, 6],
        [6, 0, 6, 8, 10],
        [8, 6, 0, 6, 8],
        [8, 8, 6, 0, 6],
        [6, 10, 8, 6, 0]],

    2: [[0, 4, 6, 6, 4],
        [4, 0, 2, 4, 2],
        [6, 2, 0, 2, 4],
        [6, 4, 2, 0, 2],
        [4, 2, 4, 2, 0]],

    3: [[0, 2, 4, 4, 2],
        [2, 0, 2, 6, 4],
        [4, 2, 0, 6, 6],
        [4, 6, 6, 0, 2],
        [2, 4, 6, 2, 0]],

    4: [[0, 4, 8, 10, 14],
        [4, 0, 4, 6, 10],
        [8, 4, 0, 6, 6],
        [10, 6, 6, 0, 6],
        [14, 10, 6, 6, 0]],
}

# -------------------------------------------------------------------
# Bilge & Ulusoy (1995) job sets
# Each list within the rows refers to machine_id and processing_time
# -------------------------------------------------------------------
BU_JOBSETS = {
    "1": [# [(Machine_id, duration)]
        [[(1, 8)], [(2, 16)], [(4, 12)]],
        [[(1, 20)], [(3, 10)], [(2, 18)]],
        [[(3, 12)], [(4, 8)], [(1, 15)]],
        [[(4, 14)], [(2, 18)]],
        [[(3, 10)], [(1, 15)]],
    ],
    "2": [
        [[(1, 10)], [(4, 18)]],
        [[(2, 10)], [(4, 18)]],
        [[(1, 10)], [(3, 20)]],
        [[(2, 10)], [(3, 15)], [(4, 12)]],
        [[(1, 10)], [(2, 15)], [(4, 12)]],
        [[(1, 10)], [(2, 15)], [(3, 12)]],
    ],
    "3": [
        [[(1, 16)], [(3, 15)]],
        [[(2, 18)], [(4, 15)]],
        [[(1, 20)], [(2, 10)]],
        [[(3, 15)], [(4, 10)]],
        [[(1, 8)], [(2, 10)], [(3, 15)], [(4, 17)]],
        [[(2, 10)], [(3, 15)], [(4, 8)], [(1, 15)]],
    ],
    "4": [
        [[(4, 11)], [(1, 10)], [(2, 7)]],
        [[(3, 12)], [(2, 10)], [(4, 8)]],
        [[(2, 7)], [(3, 10)], [(1, 9)], [(3, 8)]],
        [[(2, 7)], [(4, 8)], [(1, 12)], [(2, 6)]],
        [[(1, 9)], [(2, 7)], [(4, 8)], [(2, 10)], [(3, 8)]],
    ],
    "5": [
        [[(1, 6)], [(2, 12)], [(4, 9)]],
        [[(1, 18)], [(3, 6)], [(2, 15)]],
        [[(3, 9)], [(4, 3)], [(1, 12)]],
        [[(4, 6)], [(2, 15)]],
        [[(3, 3)], [(1, 9)]],
    ],
    "6": [
        [[(1, 9)], [(2, 11)], [(4, 7)]],
        [[(1, 19)], [(2, 20)], [(4, 13)]],
        [[(2, 14)], [(3, 20)], [(4, 9)]],
        [[(2, 14)], [(3, 20)], [(4, 9)]],
        [[(1, 11)], [(3, 16)], [(4, 8)]],
        [[(1, 10)], [(3, 12)], [(4, 10)]],
    ],
    "7": [
        [[(1, 6)], [(4, 6)]],
        [[(2, 11)], [(4, 9)]],
        [[(2, 9)], [(4, 7)]],
        [[(3, 16)], [(4, 7)]],
        [[(1, 9)], [(3, 18)]],
        [[(2, 13)], [(3, 19)], [(4, 6)]],
        [[(1, 10)], [(2, 9)], [(3, 13)]],
        [[(1, 11)], [(2, 9)], [(4, 8)]],
    ],
    "8": [
        [[(2, 12)], [(3, 21)], [(4, 11)]],
        [[(2, 12)], [(3, 21)], [(4, 11)]],
        [[(2, 12)], [(3, 21)], [(4, 11)]],
        [[(2, 12)], [(3, 21)], [(4, 11)]],
        [[(1, 10)], [(2, 14)], [(3, 18)], [(4, 9)]],
        [[(1, 10)], [(2, 14)], [(3, 18)], [(4, 9)]],
    ],
    "9": [
        [[(3, 9)], [(1, 12)], [(2, 9)], [(4, 6)]],
        [[(3, 16)], [(2, 11)], [(4, 9)]],
        [[(1, 21)], [(2, 18)], [(4, 7)]],
        [[(2, 20)], [(3, 22)], [(4, 11)]],
        [[(3, 14)], [(1, 16)], [(2, 13)], [(4, 9)]],
    ],
    "10": [
        [[(1, 11)], [(3, 19)], [(2, 16)], [(4, 13)]],
        [[(2, 21)], [(3, 16)], [(4, 14)]],
        [[(3, 8)], [(2, 10)], [(1, 14)], [(4, 9)]],
        [[(2, 13)], [(3, 20)], [(4, 10)]],
        [[(1, 9)], [(3, 16)], [(4, 18)]],
        [[(2, 19)], [(1, 21)], [(3, 11)], [(4, 15)]],
    ],
}

# ---------------------------------------------------------------------
# Scenario 1: uniform fatigue & uniform delta
# ---------------------------------------------------------------------
S1_F_MIN = 0.20
S1_F_MAX = 0.80
S1_LAMBDA = 0.04
S1_MU = 0.08
S1_DELTA = (0.30 /  S1_LAMBDA) * math.log(2)
# The current δ value means that the processing time may increase by up to 30%
# (0.3) compared to its nominal value when fatigue reaches its maximum level (ln(2)).

# ---------------------------------------------------------------------
# Scenario 2: worker-specific fatigue & δ based on operation duration
# K = 4 humans/machines
# ---------------------------------------------------------------------
def compute_s2_fatigue_params(num_machines: int = 4):
    """
    Compute arrays F_min, F_max, lambda, mu per human k using
    a linear interpolation scheme.
    """
    F_min_list = []
    F_max_list = []
    lambda_list = []
    mu_list = []

    for i in range(num_machines):
        F_min_k = round(random.uniform(0.05, 0.2), 3)
        F_max_k = round(random.uniform(0.75, 0.9), 3)
        fatigue_lambda_k = round(random.uniform(0.01, 0.05), 3)
        mu_k = round(random.uniform(0.05, 0.1), 3)

        F_min_list.append(F_min_k)
        F_max_list.append(F_max_k)
        lambda_list.append(fatigue_lambda_k)
        mu_list.append(mu_k)

    return F_min_list, F_max_list, lambda_list, mu_list


def compute_s2_delta_from_duration(duration: int) -> float:
    """
    Duration-based rule:
      δ = 0.1 if dur ≤ 10
      δ = 0.3 if 10 < dur ≤ 15
      δ = 0.5 if dur > 15
    """
    if duration <= 10:
        return 0.1
    elif duration <= 15:
        return 0.3
    else:
        return 0.5

# ---------------------------------------------------------------------
# Jobset conversion
# ---------------------------------------------------------------------
def convert_jobset(jobset, with_delta=False, scenario="S1", fatigue_lambda_list=None):
    """
    Convert a BU jobset into the 'jobs' list for our JSON schema.

    If with_delta=True and scenario=="S2", each operation includes a "delta" field
    calculated as delta = (a(dur) / lambda_k) * ln(2).
    """
    jobs = []
    for j_idx, job_ops in enumerate(jobset, start=1):
        ops = []
        for o_idx, alt_ops in enumerate(job_ops, start=1):
            # Each alt_ops is a list of alternatives; here there is exactly one.
            m, dur = alt_ops[0]
            op_dict = {
                "op_id": f"O{j_idx}_{o_idx}",
                "op_index": o_idx,
                "machine_id": f"M{m}",
                "machine_index": m,
                "duration": dur,
            }
            if with_delta:
                if scenario == "S1":
                    # Scenario 1 uses a uniform delta defined in the meta block,
                    pass
                elif scenario == "S2" and fatigue_lambda_list is not None:
                    # delta = (a(dur) / lambda_k) * ln(2)
                    # machine_index m is 1-based (M1..M4), so k = m-1
                    k = m - 1
                    lam_k = fatigue_lambda_list[k]
                    a_dur = compute_s2_delta_from_duration(dur)
                    delta_val = (a_dur / lam_k) * math.log(2)
                    op_dict["delta"] = round(float(delta_val), 3)
            ops.append(op_dict)

        jobs.append({
            "job_id": f"J{j_idx}",
            "job_index": j_idx,
            "n_ops": len(ops),
            "ops": ops,
        })
    return jobs


# ---------------------------------------------------------------------
# Instance creation for both scenarios
# ---------------------------------------------------------------------
def create_bu_instance_s1(layout_id: int, jobset_id: str, jobset):
    """
    Scenario 1: uniform fatigue & uniform delta.
    """
    layout = BU_LAYOUTS[layout_id]
    sigma = BU_SIGMAS[layout_id]
    jobs = convert_jobset(jobset, with_delta=False, scenario="S1")

    instance = {
        "instance_id": f"EX{jobset_id}_{layout_id}_S1",
        "scenario": "S1",
        "layout_id": layout_id,
        "layout_name": layout["layout_name"],
        "robots_nb": 2,
        "machines_nb": 4,
        "locations": layout["locations"],
        "sigma": sigma,
        "jobs": jobs,
        "meta": {
            "author": "Kader Sanogo",
            "source": "Adapted Bilge & Ulusoy (1995)",
            "description": f"Scenario 1 (uniform fatigue + uniform delta) for jobset {jobset_id} on layout {layout_id}",
            "symmetric_sigma": True,
            "scenario": "S1",
            "recover_on_idle": False,
            "fatigue": {
                "mode": "uniform",
                "F_min": S1_F_MIN,
                "F_max": S1_F_MAX,
                "lambda": S1_LAMBDA,
                "mu": S1_MU,
            },
            "delta": {
                "mode": "uniform",
                "value": S1_DELTA,
            },
            "created": datetime.now(UTC).isoformat(),
        },
    }

    fname = os.path.join(SCENARIO_1_DIR, f"EX{jobset_id}_{layout_id}_S1.json")
    with open(fname, "w") as f:
        json.dump(instance, f, indent=2)
    return fname


def create_bu_instance_s2(layout_id: int, jobset_id: str, jobset):
    """
    Scenario 2: worker-specific fatigue & per-op delta based on duration and machine lambda.
    """
    layout = BU_LAYOUTS[layout_id]
    sigma = BU_SIGMAS[layout_id]

    # Compute fatigue params first to pass fatigue_lambda_list to convert_jobset
    F_min_list, F_max_list, lambda_list, mu_list = compute_s2_fatigue_params(num_machines=4)

    jobs = convert_jobset(jobset, with_delta=True, scenario="S2", fatigue_lambda_list=lambda_list)

    instance = {
        "instance_id": f"EX{jobset_id}_{layout_id}_S2",
        "scenario": "S2",
        "layout_id": layout_id,
        "layout_name": layout["layout_name"],
        "robots_nb": 2,
        "machines_nb": 4,
        "locations": layout["locations"],
        "sigma": sigma,
        "jobs": jobs,
        "meta": {
            "author": "Kader Sanogo",
            "source": "Adapted Bilge & Ulusoy (1995)",
            "description": (
                f"Scenario 2 (worker-specific fatigue + per-op machine-duration-based delta) "
                f"for jobset {jobset_id} on layout {layout_id}"
            ),
            "symmetric_sigma": True,
            "scenario": "S2",
            "recover_on_idle": False,
            "fatigue": {
                "mode": "per_human",
                "F_min": F_min_list,
                "F_max": F_max_list,
                "lambda": lambda_list,
                "mu": mu_list,
            },
            "delta": {
                "mode": "per_op_machine_duration",
                "rule": "delta = (a(dur) / lambda_k) * ln(2) where a(dur) is duration-dependent factor",
            },
            "created": datetime.now(UTC).isoformat(),
        },
    }

    fname = os.path.join(SCENARIO_2_DIR, f"EX{jobset_id}_{layout_id}_S2.json")
    with open(fname, "w") as f:
        json.dump(instance, f, indent=2)
    return fname


# ---------------------------------------------------------------------
# Main: generate all instances and index files
# ---------------------------------------------------------------------
def main():
    index_s1 = []
    index_s2 = []

    for layout_id in sorted(BU_LAYOUTS.keys()):
        for jobset_id, jobset in BU_JOBSETS.items():
            f1 = create_bu_instance_s1(layout_id, jobset_id, jobset)
            f2 = create_bu_instance_s2(layout_id, jobset_id, jobset)
            index_s1.append(f1)
            index_s2.append(f2)

    # Create index files
    index_s1_path = os.path.join(ROOT_DIR, "bu_eval_index_s1.json")
    index_s2_path = os.path.join(ROOT_DIR, "bu_eval_index_s2.json")

    with open(index_s1_path, "w") as f:
        json.dump(sorted(index_s1), f, indent=2)
    with open(index_s2_path, "w") as f:
        json.dump(sorted(index_s2), f, indent=2)

    print("✅ Extended BU instances generated.")
    print(f"  Scenario 1: {len(index_s1)} files → {SCENARIO_1_DIR}")
    print(f"  Scenario 2: {len(index_s2)} files → {SCENARIO_2_DIR}")
    print(f"  Index S1: {index_s1_path}")
    print(f"  Index S2: {index_s2_path}")


if __name__ == "__main__":
    main()