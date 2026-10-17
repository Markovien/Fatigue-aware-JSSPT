# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:46:24 2026

@author: Kader SANOGO

original_bu_instances_generator.py
----------------------------------
"""

import os, json

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

# ---------------------------------------------------
# Bilge & Ulusoy travel-time matrices for each layout
# ---------------------------------------------------
BU_SIGMAS = {
    1: [[0, 6, 8, 10, 12],
        [12, 0, 6, 8, 10],
        [10, 6, 0, 6, 8 ],
        [8, 8, 6, 0, 6  ],
        [6, 10, 8, 6, 0 ]],

    2: [[0, 4, 6, 8, 6  ],
        [6, 0, 2, 4, 2  ],
        [8, 12, 0, 2, 4 ],
        [6, 10, 12, 0, 2],
        [4, 8, 10, 12, 0]],

    3: [[0, 2, 4, 10, 12],
        [12, 0, 2, 8, 10],
        [10, 12, 0, 6, 8],
        [4, 6, 8, 0, 2  ],
        [2, 4, 6, 12, 0 ]],

    4: [[0, 4, 8, 10, 14 ],
        [18, 0, 4, 6, 10 ],
        [20, 14, 0, 8, 6 ],
        [12, 8, 6, 0, 6  ],
        [14, 14, 12, 6, 0]],
}

# ------------------------------------------------------------------
# Bilge & Ulusoy job sets
# Each list within the rows refers to machine_id and processing_time
# ------------------------------------------------------------------
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
# Jobset conversion
# ---------------------------------------------------------------------
def convert_jobset(jobset):
    """
    Convert a BU jobset into the 'jobs' list for our JSON schema.
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
            # The `op_dict` was created but not appended to the `ops` list
            ops.append(op_dict) # FIX: Add this line to append op_dict to ops list

        jobs.append({
            "job_id": f"J{j_idx}",
            "job_index": j_idx,
            "n_ops": len(ops),
            "ops": ops,
        })
    return jobs

ROOT_DIR = "original_bu_instances"

# ---------------------------------------------------------------------
# Instance creation
# ---------------------------------------------------------------------
def create_bu_instance(layout_id: int, jobset_id: str, jobset):
    layout = BU_LAYOUTS[layout_id]
    sigma = BU_SIGMAS[layout_id]
    jobs = convert_jobset(jobset)

    instance = {
        "instance_id": f"EX{jobset_id}{layout_id}",
        "layout_id": layout_id,
        "layout_name": layout["layout_name"],
        "robots_nb": 2,
        "machines_nb": 4,
        "sigma": sigma,
        "jobs": jobs,
        "meta": {
            "source": "Bilge & Ulusoy (1995)",
            "description": f"Original benchmark instance for jobset {jobset_id} on layout {layout_id}",
        },
    }

    fname = os.path.join(ROOT_DIR, f"EX{jobset_id}{layout_id}.json")
    with open(fname, "w") as f:
        json.dump(instance, f, indent=2)
    return fname

# ---------------------------------------------------------------------
# Main: generate all instances and index files
# ---------------------------------------------------------------------
def main():
    # Ensure the root directory exists
    os.makedirs(ROOT_DIR, exist_ok=True)
    index = []

    for layout_id in sorted(BU_LAYOUTS.keys()):
        for jobset_id, jobset in BU_JOBSETS.items():
            f1 = create_bu_instance(layout_id, jobset_id, jobset)
            index.append(f1)

    # Create index files
    index_path = os.path.join(ROOT_DIR, "index.json")

    with open(index_path, "w") as f:
        json.dump(sorted(index), f, indent=2)

    print("✅ Original BU instances generated.")
    print(f"{len(index)} files → {ROOT_DIR}")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()