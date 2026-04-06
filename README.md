# Fatigue-Aware Job Shop Scheduling with Transportation (FaJSSPT)
This repository contains the implementation of the solution approaches developed for the **Fatigue-aware Job Shop Scheduling Problem with Transportation (FaJSSPT)**. The project focuses on integrating production scheduling, transportation coordination, and operator fatigue dynamics within a unified decision-making framework.

## Repository Structure
- cp_sat_solver_jsspt.py is our constraint programming model implemented with CP-SAT (Google OR-Tools) to solve the classical JSSPT without fatigue.
- cp_sat_solver_jsspt_hf.py extends cp_sat_solver_jsspt.py by including operator fatigue.
- meta_ga.py and meta_vns.py are the metaheuristic algorithms developed to address the problem. They can solve both JSSPT and FaJSSPT.
- adapted_bu_instances_generator.py generates the instances employed during experiments.
- original_bu_instances_generator.py generates the instances as proposed by Bilge and Ulusoy (1995).
- schedule_simulator.py is designed to simulate the schedules derived from our solution approaches.

## Citation
If you use this repository, please cite as: <upcoming>

### BibTeX entry
@article{ksanogo-fajsspt,
  title={A Fatigue-aware Integrated Scheduling Framework for Automated Transportation and Human-Operated Production},
  author={Kader Sanogo and Malek Masmoudi and Sameh T. Al-Shihabi and Ali Cheaitou and Atidel B. Hadj-Alouane},
  journal={},
  year={},
  doi={}
}
