"""Gantt charts with operator fatigue trajectories for the revised manuscript.

    python make_figures.py --results results --instance EX82 --out figures
Produces: <inst>_SIM1.png, <inst>_SIM2.png (from OPT1), <inst>_OPT2_r0.png, <inst>_OPT2_r0.25.png
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.patches import Patch                # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from fajsspt import instances as I                  # noqa: E402
from fajsspt.fatigue import FatigueConfig           # noqa: E402
from fajsspt.replay import replay                   # noqa: E402
from fajsspt.serial import load_sched               # noqa: E402

JOB_COLORS = plt.get_cmap("tab10").colors
MACH_COLORS = ["#2a6fdb", "#e0782b", "#2e9d5b", "#b8336a"]   # M1..M4 fatigue lines


def fatigue_curve(inst, sched, k, rho, n=30):
    """Piecewise fatigue trajectory of operator k: processing (accumulation), rest
    (recovery to F_min right after the predecessor), idle (constant or idle decay)."""
    fat = inst["fatigue"]
    lam, mu, Fmin = fat["lam"][k], fat["mu"][k], fat["Fmin"][k]
    ops = sorted((o for o in sched["ops"].values() if o["machine"] == k), key=lambda o: o["start"])
    t, F = 0.0, fat["F0"][k]
    xs, ys = [0.0], [F]

    def seg(t0, t1, f):
        for q in range(1, n + 1):
            x = t0 + (t1 - t0) * q / n
            xs.append(x)
            ys.append(f(x - t0))

    for o in ops:
        if o.get("rest") and o.get("rest_dur", 0) > 0:
            F0 = F
            d = math.log(max(F0, Fmin) / Fmin) / mu
            seg(t, t + d, lambda dt: max(Fmin, F0 * math.exp(-mu * dt)))
            t, F = t + d, Fmin
        if o["start"] > t:
            F0 = F
            mi = mu * rho
            seg(t, o["start"], lambda dt: max(Fmin, F0 * math.exp(-mi * dt)) if mi > 0 else F0)
            F = ys[-1]
        F0 = o["f_start"]
        seg(o["start"], o["end"], lambda dt: 1 - (1 - F0) * math.exp(-lam * dt))
        t, F = o["end"], ys[-1]
    return xs, ys


def plot(inst, sched, title, path, rho=0.0):
    M, R = inst["M"], inst["R"]
    fig, (ax, axf) = plt.subplots(2, 1, figsize=(13, 7.2), gridspec_kw=dict(height_ratios=[2.2, 1]), sharex=True)
    rows = {f"M{k}": k - 1 for k in range(1, M + 1)}
    rows.update({f"R{r + 1}": M + r for r in range(R)})
    for (i, j), o in sched["ops"].items():
        y = rows[f"M{o['machine']}"]
        ax.broken_barh([(o["start"], o["dur"])], (y - 0.4, 0.8), facecolors=JOB_COLORS[i % 10],
                       edgecolor="white", linewidth=1.5)
        ax.text(o["start"] + o["dur"] / 2, y, f"$O_{{{i + 1},{j + 1}}}$", ha="center", va="center", fontsize=8)
        if o.get("rest") and o.get("rest_dur", 0) > 0:
            prev_end = max([p["end"] for p in sched["ops"].values()
                            if p["machine"] == o["machine"] and p["end"] <= o["start"]] or [0])
            ax.broken_barh([(prev_end, o["rest_dur"])], (y - 0.4, 0.8), facecolors="#d9d9d9",
                           edgecolor="#555555", hatch="///", linewidth=0.8)
    for d in sched["transports"]:
        y = rows[f"R{d['robot'] + 1}"]
        ax.broken_barh([(d["start"], d["end"] - d["start"])], (y - 0.4, 0.8), facecolors=JOB_COLORS[d["job"] % 10],
                       edgecolor="white", linewidth=1.5)
    ax.set_yticks(list(rows.values()), list(rows.keys()))
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.set_title(title, fontsize=12)
    ax.legend(handles=[Patch(facecolor="#d9d9d9", edgecolor="#555555", hatch="///", label="Rest")],
              loc="upper right", fontsize=9, frameon=False)
    for k in range(1, M + 1):
        xs, ys = fatigue_curve(inst, sched, k, rho)
        axf.plot(xs, ys, color=MACH_COLORS[(k - 1) % 4], linewidth=2, label=f"$H_{k}$ (M{k})")
    fmax = inst["fatigue"]["Fmax"][1]
    axf.axhline(fmax, color="#c0392b", linestyle="--", linewidth=1.2)
    axf.text(0.5, fmax + 0.015, "$F^{\\max}$", color="#c0392b", fontsize=9)
    axf.set_ylim(0, 1.02)
    axf.set_ylabel("Fatigue")
    axf.set_xlabel("Time")
    axf.grid(color="#eeeeee", linewidth=0.6)
    axf.legend(loc="lower right", ncol=M, fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--instance", default="EX82")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = a.instance
    inst = I.bu_adapted(int(name[2:-1]), int(name[-1]))
    cp = Path(a.results) / "schedules" / "cp"
    s1 = load_sched(cp / f"{name}__nofat__3600.json")
    cfg = FatigueConfig()
    sim1 = replay(inst, s1, cfg, durations="discrete", rest_policy="none", timing="keep", nominal=True)
    sim2 = replay(inst, s1, cfg, durations="discrete", rest_policy="reactive", timing="keep")
    plot(inst, sim1["schedule"], f"{name} - SIM1: OPT1 schedule with fatigue tracking ($C_{{\\max}}$={sim1['cmax']:.0f})",
         out / f"{name}_SIM1.png")
    plot(inst, sim2["schedule"], f"{name} - SIM2: reactive rests ($C_{{\\max}}$={sim2['cmax']:.0f})",
         out / f"{name}_SIM2.png")
    for rho in (0, 0.25):
        s = load_sched(cp / f"{name}__h1_L1000_n_r{rho}__3600.json")
        plot(inst, s, f"{name} - OPT2, $\\mu_{{idle}}={rho}\\mu$ ($C_{{\\max}}$={s['cmax']})",
             out / f"{name}_OPT2_r{rho}.png", rho=rho)
    print("figures written to", out)


if __name__ == "__main__":
    main()
