"""LaTeX tables and key numbers, computed from results/.

    python make_tables.py --results results --out <manuscript dir>
Writes tab_*.tex (wrapped in the revenv environment) and key_numbers.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ORDER = [f"EX{j}{l}" for l in (1, 2, 3, 4) for j in range(1, 11)]
DN = [f"DN{j}_R{r}" for r in (3, 4) for j in range(1, 11)]


def pct(a, b):
    return 100.0 * (np.asarray(a, float) - np.asarray(b, float)) / np.asarray(b, float)


def b(v, opt):
    return f"\\textbf{{{int(v)}}}" if opt else f"{int(v)}"


def f1(x):
    return f"{x:.1f}"


def f2(x):
    return f"{x:.2f}"


def wrap(body, caption, label, landscape=False, size=1.0, note=None):
    rb = f"\\resizebox{{{size}\\textwidth}}{{!}}{{%\n{body}}}" if size else body
    s = (f"\\begin{{table}}[!htbp]\n\\centering\n\\begin{{revenv}}\n\\caption{{{caption}}}\n\\label{{{label}}}\n"
         f"{rb}\n" + (f"\\\\[2pt]{{\\footnotesize {note}}}\n" if note else "") + "\\end{revenv}\n\\end{table}\n")
    if landscape:
        s = s.replace("\\begin{table}[!htbp]", "\\begin{sidewaystable}").replace("\\end{table}", "\\end{sidewaystable}")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    R, out = Path(a.results), Path(a.out)
    K = {}

    # ------------------------------------------------------------------ E1
    e1 = pd.read_csv(R / "E1_simulation.csv").set_index("instance").loc[ORDER]
    e1["gap"] = pct(e1.SIM2_cmax, e1.OPT2)
    rows = []
    half = [ORDER[:20], ORDER[20:]]
    for q in range(20):
        cells = []
        for h in half:
            n = h[q]
            r = e1.loc[n]
            cells += [n, b(r.OPT1, r.OPT1_status == "OPTIMAL"), str(int(r.SIM1_viol_end)), f2(r.SIM1_peak),
                      str(int(r.SIM2_cmax)), str(int(r.SIM2_nrest)), b(r.OPT2, r.OPT2_status == "OPTIMAL"),
                      str(int(r.OPT2_nrest)), f1(r.gap)]
        rows.append(" & ".join(cells[:9]) + " & & " + " & ".join(cells[9:]) + " \\\\")
    hdr1 = ("& & \\multicolumn{2}{c}{SIM1} & \\multicolumn{2}{c}{SIM2} & \\multicolumn{2}{c}{OPT2} & & & "
            "& & \\multicolumn{2}{c}{SIM1} & \\multicolumn{2}{c}{SIM2} & \\multicolumn{2}{c}{OPT2} & \\\\\n"
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){13-14}\\cmidrule(lr){15-16}\\cmidrule(lr){17-18}\n")
    cols = "Inst. & OPT1 & \\#exc. & $F^{peak}$ & $C_{\\max}$ & \\#rest & $C_{\\max}$ & \\#rest & Gap"
    body = ("\\begin{tabular}{lrrrrrrrrcl rrrrrrrr}\n\\toprule\n" + hdr1 + cols + " & & " + cols + " \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\midrule\n")
    avg = e1[["OPT1", "SIM1_viol_end", "SIM1_peak", "SIM2_cmax", "SIM2_nrest", "OPT2", "OPT2_nrest", "gap"]].mean()
    body += ("Average & " + " & ".join([f1(avg.OPT1), f1(avg.SIM1_viol_end), f2(avg.SIM1_peak), f1(avg.SIM2_cmax),
                                        f1(avg.SIM2_nrest), f1(avg.OPT2), f1(avg.OPT2_nrest), f2(avg.gap)])
             + " & & & & & & & & & & \\\\\n\\bottomrule\n\\end{tabular}\n")
    (out / "tab_e1.tex").write_text(wrap(
        body, "Experiment~1: fatigue-agnostic schedules (OPT1) replayed with fatigue tracking (SIM1) and with "
              "reactive rests (SIM2), versus fatigue-aware optimisation (OPT2). \\#exc.: operations completed above "
              "$F^{\\max}$; $F^{peak}$: highest fatigue level reached; Gap: (SIM2$-$OPT2)/OPT2 in \\%. Bold: proven optimal.",
        "tab:opt_vs_sim"))
    K["E1"] = dict(n=len(e1), viol_inst=int((e1.SIM1_viol_end > 0).sum()), viol_mean=e1.SIM1_viol_end.mean(),
                   viol_max=int(e1.SIM1_viol_end.max()), start_viol_inst=int((e1.SIM1_viol_start > 0).sum()),
                   peak_mean=e1.SIM1_peak.mean(), peak90=int((e1.SIM1_peak >= .9).sum()),
                   peak95=int((e1.SIM1_peak >= .95).sum()), peak99=int((e1.SIM1_peak >= .99).sum()),
                   time_above_mean=e1.SIM1_time_above.mean(),
                   gap_mean=e1.gap.mean(), gap_median=e1.gap.median(), gap_max=e1.gap.max(),
                   gap_ratio=pct(e1.SIM2_cmax.mean(), e1.OPT2.mean()).item(),
                   sim2_eq=int((e1.SIM2_cmax == e1.OPT2).sum()),
                   gap_left=pct(e1.SIM2left_cmax, e1.OPT2).mean(), gap_atneed=pct(e1.SIM2atneed_cmax, e1.OPT2).mean(),
                   price=pct(e1.OPT2, e1.OPT1).mean(), opt1_unproven=list(e1[e1.OPT1_status != "OPTIMAL"].index),
                   opt2_opt=int((e1.OPT2_status == "OPTIMAL").sum()), rest_mean=e1.OPT2_nrest.mean(),
                   rest_min=int(e1.OPT2_nrest.min()), rest_max=int(e1.OPT2_nrest.max()),
                   rest_time_mean=e1.OPT2_rest_time.mean(), sim2_rest_mean=e1.SIM2_nrest.mean())

    # ------------------------------------------------------------------ E2
    cp = pd.read_csv(R / "E2_cpsat.csv")
    mh = pd.read_csv(R / "E2_metaheuristics.csv")
    ref = cp.set_index(["instance", "rho"])["cmax"]
    mh["ref"] = [ref[(i, r)] for i, r in zip(mh.instance, mh.rho)]
    mh["rpd"] = pct(mh.cmax, mh.ref)
    ag = mh.groupby(["instance", "rho", "alg"]).agg(best=("cmax", "min"), mean=("cmax", "mean"), sd=("cmax", "std"),
                                                     ttb=("time_to_best", "mean"))
    rows = []
    for n in ORDER:
        cells = [n]
        for rho in (0.0, 0.25):
            c = cp[(cp.instance == n) & (cp.rho == rho)].iloc[0]
            cells += [b(c.cmax, c.status == "OPTIMAL") + ("" if c.status == "OPTIMAL" else f" ({100 * c.gap:.1f})"),
                      f1(c.wall_time)]
            for alg in ("GA", "VNS"):
                g = ag.loc[(n, rho, alg)]
                cells += [str(int(g.best)), f1(g["mean"]), f1(g.sd), f1(g.ttb)]
        rows.append(" & ".join(cells) + " \\\\")
    sub = "$C_{\\max}$ & Time & Best & Mean & SD & TTB & Best & Mean & SD & TTB"
    body = ("\\begin{tabular}{l" + "rrrrrrrrrr" * 2 + "}\n\\toprule\n"
            "& \\multicolumn{10}{c}{$\\mu_{idle}=0$} & \\multicolumn{10}{c}{$\\mu_{idle}=0.25\\mu$} \\\\\n"
            "\\cmidrule(lr){2-11}\\cmidrule(lr){12-21}\n"
            "& \\multicolumn{2}{c}{CP-SAT} & \\multicolumn{4}{c}{GA} & \\multicolumn{4}{c}{VNS} & "
            "\\multicolumn{2}{c}{CP-SAT} & \\multicolumn{4}{c}{GA} & \\multicolumn{4}{c}{VNS} \\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-7}\\cmidrule(lr){8-11}\\cmidrule(lr){12-13}\\cmidrule(lr){14-17}\\cmidrule(lr){18-21}\n"
            "Instance & " + sub + " & " + sub + " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\midrule\n")
    avgc = ["Average"]
    for rho in (0.0, 0.25):
        c = cp[cp.rho == rho]
        avgc += [f2(c.cmax.mean()), f1(c.wall_time.mean())]
        for alg in ("GA", "VNS"):
            g = ag.xs(rho, level="rho").xs(alg, level="alg")
            avgc += [f2(g.best.mean()), f2(g["mean"].mean()), f2(g.sd.mean()), f1(g.ttb.mean())]
    body += " & ".join(avgc) + " \\\\\n\\bottomrule\n\\end{tabular}\n"
    (out / "tab_e2.tex").write_text(wrap(
        body, "Experiment~2: CP-SAT (bold: proven optimal; otherwise best value and optimality gap in \\%) and "
              "GA/VNS over 20 independent runs with an identical 15~s wall-clock budget (best, mean and standard "
              "deviation of $C_{\\max}$; TTB: mean time-to-best in s) for the two idle-recovery settings.",
        "tab:exp_res", landscape=True, size=1.0))
    K["E2"] = {}
    base = cp[cp.rho == 0].set_index("instance").cmax
    for rho in (0.0, 0.25):
        c = cp[cp.rho == rho]
        d = dict(avg=c.cmax.mean(), optimal=int((c.status == "OPTIMAL").sum()), time_mean=c.wall_time.mean(),
                 time_median=c.wall_time.median(), rests=c.n_rest.mean(), rest_time=c.rest_time.mean(),
                 nonopt={r.instance: dict(cmax=int(r.cmax), bound=int(r.bound), gap=100 * r.gap)
                         for r in c[c.status != "OPTIMAL"].itertuples()})
        for alg in ("GA", "VNS"):
            m = mh[(mh.rho == rho) & (mh.alg == alg)]
            g = ag.xs(rho, level="rho").xs(alg, level="alg")
            d[alg] = dict(arpd=m.rpd.mean(), arpd_best=pct(g.best.values, base.reindex(g.index).values).mean()
                          if rho == 0 else pct(g.best.values, c.set_index("instance").cmax.reindex(g.index).values).mean(),
                          hit=100 * (m.rpd <= 1e-9).mean(), sd=g.sd.mean(), ttb=m.time_to_best.mean(),
                          evals=m.evals.mean(), best_hits=int((g.best.values <= c.set_index("instance").cmax.reindex(g.index).values).sum()))
        pv = ag.xs(rho, level="rho")["mean"].unstack()
        dd = pv.VNS - pv.GA
        st = wilcoxon(pv.VNS, pv.GA, zero_method="wilcox")
        d["wilcoxon"] = dict(vns_better=int((dd < 0).sum()), ties=int((dd == 0).sum()), worse=int((dd > 0).sum()),
                             W=float(st.statistic), p=float(st.pvalue))
        if rho > 0:
            cur = c.set_index("instance").cmax.reindex(base.index)
            red = 100 * (base - cur) / base
            d["idle_red"] = dict(mean=red.mean(), median=red.median(), min=red.min(), max=red.max(),
                                 ratio=100 * (base.mean() - cur.mean()) / base.mean(), n_zero=int((red == 0).sum()),
                                 n_pos=int((red > 0).sum()), argmax=red.idxmax())
        K["E2"][str(rho)] = d
    K["E2"]["cp_opt_total"] = int((cp.status == "OPTIMAL").sum())
    # summary table
    rows = []
    for rho in (0.0, 0.25):
        d = K["E2"][str(rho)]
        for alg in ("GA", "VNS"):
            x = d[alg]
            w = d["wilcoxon"]
            rows.append(f"{rho:g}$\\mu$ & {alg} & {f2(x['arpd'])} & {f1(x['hit'])} & {x['best_hits']}/40 & {f2(x['sd'])} & "
                        f"{f2(x['ttb'])} & {x['evals'] / 1000:.1f}k & " +
                        (f"\\multirow{{2}}{{*}}{{{w['vns_better']}/{w['ties']}/{w['worse']}; $p={w['p']:.1e}$}}" if alg == "GA" else "")
                        + " \\\\")
    body = ("\\begin{tabular}{llrrrrrrc}\n\\toprule\n$\\mu_{idle}$ & Algorithm & ARPD (\\%) & Hit (\\%) & Best = CP-SAT "
            "& Mean SD & TTB (s) & Evaluations & VNS vs GA: W/T/L; Wilcoxon \\\\\n\\midrule\n" + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n")
    (out / "tab_e2sum.tex").write_text(wrap(
        body, "Summary of the metaheuristics (20 runs $\\times$ 40 instances per setting, 15~s each). ARPD: average "
              "relative percentage deviation of a run from CP-SAT; Hit: share of runs reaching the CP-SAT value; "
              "W/T/L: instances on which the mean makespan of VNS is better/equal/worse than that of GA; "
              "two-sided Wilcoxon signed-rank test on the 40 per-instance means.", "tab:mh_summary", size=1.0))

    # ------------------------------------------------------------------ E7
    c7 = pd.read_csv(R / "E7_original_bu_cpsat.csv").set_index("instance")
    m7 = pd.read_csv(R / "E7_original_bu_metaheuristics.csv")
    pub = pd.read_csv(Path(__file__).parent / "fajsspt/data/published_bu_original.csv", comment="#").set_index("instance")
    refs = [("Bilge_and_Ulusoy_1995_cmax", "\\citet{bilge1995}"), ("Abderrahim_et_al_2020_cmax", "\\citet{moussa2020}"),
            ("Andy_Ham_2020_cmax", "\\citet{ham2021}"), ("Dalila_et_al_2023_cmax", "\\citet{FONTES20231140}"),
            ("Jiachen_et_al_2023_cmax", "\\citet{s23094526}"), ("Gayon_et_al_2025_cmax", "\\citet{Gayon2025}")]
    a7 = m7[m7.variant == "full"].groupby(["instance", "alg"]).cmax.agg(["min", "mean"])
    rows = []
    for n in ORDER:
        bk = pub.loc[n, "best_known"]
        cells = [n] + [str(int(pub.loc[n, c])) for c, _ in refs]
        cells += [b(c7.loc[n, "cmax"], c7.loc[n, "status"] == "OPTIMAL"), f1(c7.loc[n, "wall_time"])]
        for alg in ("GA", "VNS"):
            cells += [str(int(a7.loc[(n, alg), "min"])), f1(a7.loc[(n, alg), "mean"])]
        rows.append(" & ".join(cells) + " \\\\")
    avg = ["Average"] + [f2(pub.loc[ORDER, c].mean()) for c, _ in refs] + [f2(c7.loc[ORDER, "cmax"].mean()),
                                                                          f1(c7.loc[ORDER, "wall_time"].mean())]
    for alg in ("GA", "VNS"):
        g = a7.xs(alg, level="alg").loc[ORDER]
        avg += [f2(g["min"].mean()), f2(g["mean"].mean())]
    body = ("\\begin{tabular}{l" + "r" * 6 + "rrrrrr}\n\\toprule\n& \\multicolumn{6}{c}{Literature} & "
            "\\multicolumn{6}{c}{This work} \\\\\n\\cmidrule(lr){2-7}\\cmidrule(lr){8-13}\n"
            "Instance & " + " & ".join(r for _, r in refs) + " & CP-SAT & Time (s) & GA best & GA mean & VNS best & VNS mean"
            " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\midrule\n" + " & ".join(avg) + " \\\\\n\\bottomrule\n\\end{tabular}\n")
    (out / "tab_e7.tex").write_text(wrap(
        body, "Original instances of \\citet{bilge1995} (no fatigue, AGVs, no return to the L/U station): values "
              "reported in the literature and obtained by the proposed approaches (GA/VNS: 10 runs of 15~s; bold: "
              "proven optimal).", "tab:bu_original", landscape=True, size=1.0))
    bk = pub.loc[ORDER, "best_known"]
    K["E7"] = dict(cp_opt=int((c7.status == "OPTIMAL").sum()), cp_eq=int((c7.loc[ORDER, "cmax"] == bk).sum()),
                   cp_time_mean=c7.wall_time.mean(), cp_time_max=c7.wall_time.max())
    for v in ("full", "legacy"):
        for alg in ("GA", "VNS"):
            g = m7[(m7.variant == v) & (m7.alg == alg)].groupby("instance").cmax.agg(["min", "mean"]).loc[ORDER]
            K["E7"][f"{alg}_{v}"] = dict(hits=int((g["min"] == bk).sum()), arpd_best=pct(g["min"], bk).mean(),
                                          arpd_mean=pct(g["mean"], bk).mean())
    for c, _ in refs:
        K["E7"][c] = dict(hits=int((pub.loc[ORDER, c] == bk).sum()), arpd=pct(pub.loc[ORDER, c], bk).mean())

    # ------------------------------------------------------------------ E6
    c6 = pd.read_csv(R / "E6_large_cpsat.csv").set_index("instance").loc[DN]
    m6 = pd.read_csv(R / "E6_large_metaheuristics.csv")
    m6["rpd"] = pct(m6.cmax, m6.instance.map(c6.OPT2))
    a6 = m6.groupby(["instance", "alg"]).agg(best=("cmax", "min"), mean=("cmax", "mean"), sd=("cmax", "std"),
                                             ttb=("time_to_best", "mean"))
    rows = []
    for n in DN:
        r = c6.loc[n]
        cells = [n.replace("_R", " / "), f"{int(r.n_jobs)}/{int(r.n_ops)}",
                 b(r.OPT1, r.OPT1_status == "OPTIMAL"), str(int(r.SIM1_viol_end)), str(int(r.SIM2_cmax)),
                 b(r.OPT2, r.status == "OPTIMAL") + ("" if r.status == "OPTIMAL" else f" ({100 * r.gap:.1f})"),
                 f1(r.wall_time)]
        for alg in ("GA", "VNS"):
            g = a6.loc[(n, alg)]
            cells += [str(int(g.best)), f1(g["mean"]), f1(g.sd)]
        rows.append(" & ".join(cells) + " \\\\")
    body = ("\\begin{tabular}{lcrrrrrrrrrrr}\n\\toprule\n& & & \\multicolumn{2}{c}{Simulation} & \\multicolumn{2}{c}{CP-SAT (OPT2)} & "
            "\\multicolumn{3}{c}{GA} & \\multicolumn{3}{c}{VNS} \\\\\n"
            "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-10}\\cmidrule(lr){11-13}\n"
            "Instance / $|\\mathcal{V}|$ & Jobs/Ops & OPT1 & \\#exc. & SIM2 & $C_{\\max}$ & Time (s) & Best & Mean & SD & Best & Mean & SD"
            " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    (out / "tab_e6.tex").write_text(wrap(
        body, "Larger instances derived from \\citet{DEROUSSI2010} (each alternative route is a job; eight machines; "
              "travel times of \\citet{HOMAYOUNI2023}, Table A2) with three and four robots. CP-SAT limit: 1800~s; GA/VNS: "
              "10 runs with a budget of $\\max(15, 0.5\\,|\\mathcal{O}|)$~s. Bold: proven optimal; in brackets: gap in \\%.",
        "tab:large", size=1.0))
    K["E6"] = dict(opt2_opt=int((c6.status == "OPTIMAL").sum()), opt1_opt=int((c6.OPT1_status == "OPTIMAL").sum()),
                   nonopt={n: dict(cmax=int(c6.loc[n, "OPT2"]), gap=100 * c6.loc[n, "gap"]) for n in DN
                           if c6.loc[n, "status"] != "OPTIMAL"},
                   viol_inst=int((c6.SIM1_viol_end > 0).sum()), sim2_gap=pct(c6.SIM2_cmax, c6.OPT2).mean(),
                   price=pct(c6.OPT2, c6.OPT1).mean(), cp_time_mean=c6.wall_time.mean(),
                   R3=c6[c6.R == 3].OPT2.mean(), R4=c6[c6.R == 4].OPT2.mean())
    for alg in ("GA", "VNS"):
        m = m6[m6.alg == alg]
        K["E6"][alg] = dict(arpd=m.rpd.mean(), hit=100 * (m.rpd <= 0).mean(), ttb=m.time_to_best.mean(),
                            better_than_cp=int((a6.xs(alg, level="alg").best.reindex(DN) < c6.OPT2).sum()))

    # ------------------------------------------------------------------ E3
    e3 = pd.read_csv(R / "E3_discretisation.csv")
    basec = e3[(e3.L == 1000) & (e3.scale == 1) & (e3.rounding == "nearest")].set_index("instance").cmax
    rows, K["E3"] = [], {}
    for (L, s, rnd), g in e3.groupby(["L", "scale", "rounding"]):
        bb = basec.reindex(g.instance).values
        x = dict(opt=int((g.status == "OPTIMAL").sum()), avg=g.cmax.mean(), dev_base=np.nanmean(pct(g.cmax.values, bb)),
                 cont=np.nanmean(pct(g.cont_keep_cmax, g.cmax)), cont_max=np.nanmax(pct(g.cont_keep_cmax, g.cmax)),
                 viol=int((g.cont_keep_viol_end > 0).sum()), peak=g.cont_keep_peak.max(), time=g.wall_time.mean())
        K["E3"][f"L{L}_s{s}_{rnd}"] = x
        rows.append((s, L, rnd, f"{L} & $\\times${s} & {rnd} & {x['opt']}/40 & {f2(x['avg'])} & {x['dev_base']:+.2f} & "
                                f"{x['cont']:+.3f} & {x['cont_max']:+.2f} & {x['viol']} & {x['peak']:.3f} & {f1(x['time'])} \\\\"))
    rows.sort(key=lambda t: (t[0], t[2] != "nearest", t[1]))
    body = ("\\begin{tabular}{rrlrrrrrrrr}\n\\toprule\n$L$ & Time res. & Rounding & Optimal & Avg. $C_{\\max}$ & "
            "vs ref. (\\%) & Cont. dev. (\\%) & Max dev. (\\%) & \\#sched. $>F^{\\max}$ & Max $F$ & Time (s) \\\\\n\\midrule\n"
            + "\n".join(r[3] for r in rows) + "\n\\bottomrule\n\\end{tabular}\n")
    (out / "tab_e3.tex").write_text(wrap(
        body, "Effect of the fatigue grid $L$, of the time resolution (durations $\\times s$, rates $/s$; makespans "
              "reported in original units) and of the rounding rule (40 instances, $\\mu_{idle}=0$, 600~s). Every "
              "CP-SAT schedule is replayed in continuous time with the exact laws; Cont. dev.: change of $C_{\\max}$; "
              "\\#sched. $>F^{\\max}$: schedules whose continuous replay exceeds $F^{\\max}$. Reference: $L=1000$, "
              "$\\times1$, nearest.", "tab:discretisation", size=1.0))

    # ------------------------------------------------------------------ E4
    e4 = pd.read_csv(R / "E4_sensitivity.csv")
    e1o = pd.read_csv(R / "E1_simulation.csv").set_index("instance").OPT1
    r0 = e4[e4.rho == 0].copy()
    r1 = e4[e4.rho == 0.25].set_index(["instance", "param", "level"]).OPT2
    names = {"lam": "$\\lambda$", "mu": "$\\mu$", "kappa": "$\\delta$", "Fmax": "$F^{\\max}$", "Fmin": "$F^{\\min}$"}
    base_vals = {"lam": 0.04, "mu": 0.08, "kappa": 0.208, "Fmax": 0.80, "Fmin": 0.20}
    rows, K["E4"] = [], {}
    for p in ["lam", "mu", "kappa", "Fmax", "Fmin"]:
        lv = sorted(set(r0[r0.param == p].level.astype(float)) | {base_vals[p]})
        for v in lv:
            g = r0[r0.param == "base"] if abs(v - base_vals[p]) < 1e-6 else r0[(r0.param == p) & (r0.level.astype(float) == v)]
            key = list(zip(g.instance, g.param, g.level))
            v1 = r1.reindex(key).values
            x = dict(OPT2=g.OPT2.mean(), price=pct(g.OPT2.values, e1o.reindex(g.instance).values).mean(),
                     rests=g.n_rest.mean(), viol=int((g.SIM1_viol_end > 0).sum()), sim2=pct(g.SIM2_cmax, g.OPT2).mean(),
                     idle=np.nanmean(100 * (g.OPT2.values - v1) / g.OPT2.values))
            K["E4"][f"{p}={v}"] = x
            mark = "$^{*}$" if abs(v - base_vals[p]) < 1e-6 else ""
            rows.append(f"{names[p]} & {v:g}{mark} & {f1(x['OPT2'])} & {x['price']:+.1f} & {f1(x['rests'])} & "
                        f"{x['viol']}/10 & {f2(x['sim2'])} & {f2(x['idle'])} \\\\")
        rows.append("\\midrule")
    body = ("\\begin{tabular}{lrrrrrrr}\n\\toprule\nParameter & Value & OPT2 & OPT2 vs OPT1 (\\%) & \\#rests & "
            "SIM1 $>F^{\\max}$ & SIM2 vs OPT2 (\\%) & Idle gain (\\%) \\\\\n\\midrule\n" + "\n".join(rows[:-1])
            + "\n\\bottomrule\n\\end{tabular}\n")
    (out / "tab_e4.tex").write_text(wrap(
        body, "One-at-a-time sensitivity of the main indicators to the fatigue parameters (instances of layout~2, "
              "averages over 10 instances; CP-SAT limit 300~s; $^{*}$: reference value). Idle gain: makespan "
              "reduction obtained with $\\mu_{idle}=0.25\\mu$.", "tab:sensitivity", size=0.9))

    # ------------------------------------------------------------------ E5
    e5 = pd.read_csv(R / "E5_heterogeneous.csv")
    v5 = pd.read_csv(R / "E5_vns.csv")
    ref5 = e5.set_index(["instance", "draw"]).OPT2
    v5["rpd"] = pct(v5.cmax, [ref5[(i, d)] for i, d in zip(v5.instance, v5.draw)])
    K["E5"] = dict(n=len(e5), opt=int((e5.status == "OPTIMAL").sum()), viol=int((e5.SIM1_viol_end > 0).sum()),
                   sim2=pct(e5.SIM2_cmax, e5.OPT2).mean(), sim2_med=float(np.median(pct(e5.SIM2_cmax, e5.OPT2))),
                   sim2_max=float(np.max(pct(e5.SIM2_cmax, e5.OPT2))),
                   rests=e5.n_rest.mean(), vns_arpd=v5.rpd.mean(), vns_hit=100 * (v5.rpd <= 1e-9).mean())

    # ------------------------------------------------------------------ E8
    e8 = pd.read_csv(R / "E8_ablation.csv")
    e8["ref"] = [ref[(i, r)] for i, r in zip(e8.instance, e8.rho)]
    e8["rpd"] = pct(e8.cmax, e8.ref)
    labels = {"legacy": "Submitted version", "I1": "+ I1 (decoder)", "I1+I2": "+ I1 + I2 (critical moves)",
              "I1+I3": "+ I1 + I3 (seeding)", "I1+I2+I3": "+ I1 + I2 + I3", "full": "Full (I1--I6)"}
    rows, K["E8"] = [], {}
    for v in labels:
        cells = [labels[v]]
        for rho in (0.0, 0.25):
            for alg in ("GA", "VNS"):
                g = e8[(e8.variant == v) & (e8.rho == rho) & (e8.alg == alg)]
                K["E8"][f"{alg}_{v}_{rho}"] = dict(arpd=g.rpd.mean(), hit=100 * (g.rpd <= 1e-9).mean(), evals=g.evals.mean())
                cells += [f2(g.rpd.mean()), f1(100 * (g.rpd <= 1e-9).mean())]
        rows.append(" & ".join(cells) + " \\\\")
    body = ("\\begin{tabular}{lrrrrrrrr}\n\\toprule\n& \\multicolumn{4}{c}{$\\mu_{idle}=0$} & \\multicolumn{4}{c}{$\\mu_{idle}=0.25\\mu$} \\\\\n"
            "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n& \\multicolumn{2}{c}{GA} & \\multicolumn{2}{c}{VNS} & "
            "\\multicolumn{2}{c}{GA} & \\multicolumn{2}{c}{VNS} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\n"
            "Variant & ARPD & Hit & ARPD & Hit & ARPD & Hit & ARPD & Hit \\\\\n\\midrule\n" + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n")
    (out / "tab_e8.tex").write_text(wrap(
        body, "Ablation of the algorithmic components (40 instances, 3 runs of 15~s per instance, setting and variant; "
              "ARPD and Hit in \\% with respect to CP-SAT).", "tab:ablation", size=0.85))

    (out / "key_numbers.json").write_text(json.dumps(K, indent=1, default=float))
    print(json.dumps(K, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
