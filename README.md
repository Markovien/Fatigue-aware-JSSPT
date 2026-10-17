# Fatigue-Aware Job Shop Scheduling with Transportation (FaJSSPT)

This repository contains the implementation and the experimental pipeline of the paper
*A Fatigue-aware Scheduling Framework for Transportation Robots and Human Operators in Industry 5.0*.
The FaJSSPT jointly schedules human-operated machines and transportation robots while operator
fatigue accumulates during work, recovers during rest (and, optionally, during machine idle time),
inflates processing times, and must remain below an admissible level $F^{\max}$ throughout every
operation. Rests are taken only when they are required.

All solution methods share one fatigue model and one set of lookup tables, so their makespans
are directly comparable. Every schedule produced by any method can be checked by an independent
validator and replayed with the exact continuous fatigue laws.

## Repository structure

| Path | Content |
|---|---|
| `fajsspt/fatigue.py` | Fatigue laws, discretized lookup tables (processing time, completion fatigue, rest duration, idle decay), rest trigger, and hard bound |
| `fajsspt/instances.py` | Instances: adapted and original Bilge & Ulusoy (1995), Deroussi & Norre (2010) instances (routes split into jobs), JSPLIB instances; parameter, heterogeneity, and time-scaling transformations |
| `fajsspt/cpsat_model.py` | CP-SAT model (Google OR-Tools) with bounds/gaps reporting, warm start, and schedule export |
| `fajsspt/decoder.py` | Fatigue-aware decoder shared by GA and VNS (robot choice with insertion, rest-or-wait decision, critical path including rest blocks) |
| `fajsspt/metaheuristics.py` | GA and VNS with critical-path neighborhood, seeding, plateau moves/restarts, memetic step |
| `fajsspt/replay.py` | SIM1/SIM2 simulation of fixed schedules, continuous-time replay, schedule validator |
| `fajsspt/warm.py` | Warm start of CP-SAT from dispatching rules |
| `fajsspt/data/` | Benchmark data (job sets, travel times), published values of the original benchmark |
| `experiments.py` | Experiments E0–E8 of the paper (queued and resumable) |
| `analyze.py` | Summary statistics (`results/summary.md`), including Wilcoxon tests |
| `make_tables.py`, `make_figures.py`| Tables and figures of the paper, generated from `results/` |
| `tests/self_check.py` | Cross-check of CP-SAT, decoder, validator, and replay (about 2 min) |

## Installation

```bash
python -m pip install -r requirements.txt
python tests/self_check.py          # must end with "SELF-CHECK PASSED"
```

## Reproducing the experiments

```bash
python experiments.py all --quick   # smoke test (~25 min), results/quick/
python experiments.py all           # full campaign (several days on a laptop)
python analyze.py                   # results/summary.md
python make_tables.py --results results --out paper/
python make_figures.py --results results --instance EX103 --out paper/
```

Experiments: E0 regression with the submitted results; E1 fatigue-agnostic schedules under fatigue
(OPT1, SIM1, SIM2, OPT2); E2 CP-SAT, GA, and VNS with and without idle recovery (20 runs, equal
15 s budgets); E3 discretization and time resolution with continuous replay; E4 sensitivity to
the fatigue parameters; E5 heterogeneous operators; E6 larger instances (Deroussi & Norre, 3 and
4 robots); E7 original Bilge & Ulusoy instances versus the literature; E8 ablation of the
algorithmic components. Run `python experiments.py --help` for all options (time limits,
number of runs, workers). Interrupted runs resume where they stopped.

## Citation

If you use this repository, please cite: `<upcoming>`

```
@article{ksanogo-fajsspt,
  title={A Fatigue-aware Scheduling Framework for Transportation Robots and Human Operators in Industry 5.0},
  author={Kader Sanogo and Malek Masmoudi and Sameh T. Al-Shihabi and Ali Cheaitou and Atidel B. Hadj-Alouane},
  journal={Computers \& Industrial Engineering},
  year={},
  doi={}
}
```

## Data sources

* Bilge, Ü., Ulusoy, G. (1995). A time window approach to simultaneous scheduling of machines and
  material handling system in an FMS. *Operations Research* 43(6), 1058–1070.
* Deroussi, L., Norre, S. (2010), instances as reported in Homayouni, S.M., Fontes, D.B.M.M.,
  Gonçalves, J.F. (2023). *International Transactions in Operational Research* 30(2), 688–716.
* JSPLIB (https://github.com/tamy0612/JSPLIB), OR-Library instances ft06 and la01–la15.
