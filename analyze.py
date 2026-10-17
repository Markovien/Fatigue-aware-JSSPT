"""Summarise the experiment CSVs into results/summary.md and LaTeX tables.

    python analyze.py                 (reads ./results)
    python analyze.py --results results/quick
Each section is skipped if its CSV does not exist yet.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except Exception:      # pragma: no cover
    wilcoxon = None

ORDER = [f"EX{j}{l}" for l in (1, 2, 3, 4) for j in range(1, 11)]
OUT = []


def md(s=""):
    OUT.append(s)


def read(res, name):
    p = Path(res) / name
    return pd.read_csv(p) if p.exists() else None


def pct(a, b):
    return 100.0 * (a - b) / b


def order(df, col="instance"):
    df = df.copy()
    df["_o"] = df[col].map({n: q for q, n in enumerate(ORDER)}).fillna(999)
    return df.sort_values(["_o", col]).drop(columns="_o")


def table_md(df, floatfmt=".2f"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f"{v:{floatfmt}}" if isinstance(v, (float, np.floating)) and not pd.isna(v) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    md("\n".join(lines))
    md()


def write_tex(res, name, df, caption, label):
    tdir = Path(res) / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    tex = df.to_latex(index=False, escape=False, na_rep="--", float_format=lambda x: f"{x:.2f}")
    (tdir / f"{name}.tex").write_text(
        "\\begin{table}[ht]\n\\centering\n\\resizebox{\\textwidth}{!}{%\n" + tex +
        f"}}\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}\n")


def wil(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    w, t, l = int((d < 0).sum()), int((d == 0).sum()), int((d > 0).sum())
    if wilcoxon is None or (d != 0).sum() < 5:
        return w, t, l, None, None
    st = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
    return w, t, l, float(st.statistic), float(st.pvalue)


# =============================================================================
def e0(res):
    df = read(res, "E0_regression.csv")
    if df is None:
        return
    md("## E0 - Regression (legacy semantics vs submitted Table 3)")
    md(f"{int(df['match'].sum())}/{len(df)} instances reproduce the submitted CP-SAT value.")
    bad = df[df["match"] == 0]
    if len(bad):
        table_md(bad)
    md()


def e1(res):
    df = read(res, "E1_simulation.csv")
    if df is None:
        return
    df = order(df)
    md("## E1 - Fatigue-agnostic schedules under fatigue (hard bound)")
    n = len(df)
    v = df["SIM1_viol_end"] > 0
    md(f"- Instances whose OPT1 schedule exceeds F_max at least once (SIM1): **{v.sum()}/{n} ({100 * v.mean():.1f}%)**; "
       f"operations above F_max: mean {df['SIM1_viol_end'].mean():.2f}, max {df['SIM1_viol_end'].max()}")
    md(f"- Peak fatigue in SIM1: mean {df['SIM1_peak'].mean():.3f}; >=0.90: {(df['SIM1_peak'] >= .9).sum()}, "
       f">=0.95: {(df['SIM1_peak'] >= .95).sum()}, >=0.99: {(df['SIM1_peak'] >= .99).sum()} instances")
    md(f"- SIM1 reproduces OPT1 (same Cmax) in {(df['SIM1_cmax'] == df['OPT1']).sum()}/{n} instances")
    d2 = pct(df["SIM2_cmax"], df["OPT2"])
    md(f"- SIM2 (reactive rests, right-shift) vs OPT2: mean of per-instance gaps **{d2.mean():.2f}%** "
       f"(median {d2.median():.2f}%, max {d2.max():.2f}%), ratio of averages "
       f"{pct(df['SIM2_cmax'].mean(), df['OPT2'].mean()):.2f}%, SIM2 = OPT2 in {(df['SIM2_cmax'] == df['OPT2']).sum()}/{n}")
    md(f"- Variants: SIM2 with left-shift {pct(df['SIM2left_cmax'], df['OPT2']).mean():.2f}%, "
       f"rest at need {pct(df['SIM2atneed_cmax'], df['OPT2']).mean():.2f}%")
    md(f"- OPT2 vs OPT1 (price of fatigue safety): mean +{pct(df['OPT2'], df['OPT1']).mean():.2f}%")
    md(f"- OPT2 optimal: {(df['OPT2_status'] == 'OPTIMAL').sum()}/{n}")
    md()
    t = pd.DataFrame(dict(Instance=df["instance"], OPT1=df["OPT1"], SIM1=df["SIM1_cmax"],
                          **{"#viol": df["SIM1_viol_end"], "Fpeak": df["SIM1_peak"]},
                          SIM2=df["SIM2_cmax"], **{"#rest": df["SIM2_nrest"]},
                          OPT2=[f"\\textbf{{{c}}}" if s == "OPTIMAL" else str(c) for c, s in zip(df["OPT2"], df["OPT2_status"])],
                          **{"#rest*": df["OPT2_nrest"], "Gap (\\%)": d2.round(2)}))
    write_tex(res, "E1_table", t, "Fatigue-agnostic (OPT1) schedules simulated with fatigue (SIM1: no rest; "
              "SIM2: reactive rests) vs fatigue-aware optimisation (OPT2). Bold: proven optimal.", "tab:opt_vs_sim")


def e2(res):
    cp = read(res, "E2_cpsat.csv")
    mh = read(res, "E2_metaheuristics.csv")
    if cp is None:
        return
    md("## E2 - Solution approaches and idle-time recovery")
    rhos = sorted(cp["rho"].unique())
    rows = []
    for rho in rhos:
        c = cp[cp["rho"] == rho]
        nonopt = c[c["status"] != "OPTIMAL"]
        rows.append(dict(rho=rho, n=len(c), avg_cmax=c["cmax"].mean(), optimal=int((c["status"] == "OPTIMAL").sum()),
                         nonopt=", ".join(f"{r.instance}({r.cmax}/LB {r.bound}, gap {100 * r.gap:.1f}%)"
                                          for r in nonopt.itertuples()),
                         avg_rests=c["n_rest"].mean(), avg_rest_time=c["rest_time"].mean()))
    table_md(pd.DataFrame(rows))
    tot = len(cp)
    md(f"CP-SAT proves optimality in {int((cp['status'] == 'OPTIMAL').sum())}/{tot} runs "
       f"({100 * (cp['status'] == 'OPTIMAL').mean():.2f}%).")
    base = cp[cp["rho"] == 0].set_index("instance")["cmax"]
    for rho in rhos:
        if rho == 0:
            continue
        cur = cp[cp["rho"] == rho].set_index("instance")["cmax"]
        red = 100 * (base - cur.reindex(base.index)) / base
        md(f"- Idle recovery rho={rho}: per-instance reduction mean {red.mean():.2f}%, median {red.median():.2f}%, "
           f"min {red.min():.2f}%, max {red.max():.2f}% ; ratio of averages "
           f"{100 * (base.mean() - cur.mean()) / base.mean():.2f}% ; increases: {list(red[red < 0].index)}")
    md()
    if mh is None:
        return
    ref = cp.set_index(["instance", "rho"])["cmax"]
    mh["ref"] = [ref.get((i, r), np.nan) for i, r in zip(mh["instance"], mh["rho"])]
    mh["rpd"] = pct(mh["cmax"], mh["ref"])
    agg = mh.groupby(["rho", "alg", "instance"]).agg(best=("cmax", "min"), mean=("cmax", "mean"), sd=("cmax", "std"),
                                                     ttb=("time_to_best", "mean"), rpd=("rpd", "mean"),
                                                     hit=("rpd", lambda x: (x <= 1e-9).mean())).reset_index()
    md("### Metaheuristics (repeated runs, equal wall-clock budget)")
    srows = []
    for (rho, alg), g in agg.groupby(["rho", "alg"]):
        srows.append(dict(rho=rho, alg=alg, avg_best=g["best"].mean(), avg_mean=g["mean"].mean(),
                          avg_sd=g["sd"].mean(), ARPD_mean=g["rpd"].mean(), hit_rate=100 * g["hit"].mean(),
                          ttb=g["ttb"].mean(), budget=mh["budget"].iloc[0]))
    table_md(pd.DataFrame(srows))
    md("Wilcoxon signed-rank test on per-instance mean makespan (VNS vs GA):")
    for rho in sorted(agg["rho"].unique()):
        a = agg[(agg.rho == rho)].pivot(index="instance", columns="alg", values="mean").dropna()
        if {"GA", "VNS"} <= set(a.columns):
            w, t, l, s, p = wil(a["VNS"], a["GA"])
            md(f"- rho={rho}: VNS better {w}, ties {t}, worse {l}; W={s}, p={p}")
    md()
    # LaTeX: wide table
    piv = []
    for inst in ORDER:
        if inst not in set(cp["instance"]):
            continue
        r = {"Instance": inst}
        for rho in rhos:
            c = cp[(cp.instance == inst) & (cp.rho == rho)].iloc[0]
            r[f"CP$_{{{rho}}}$"] = f"\\textbf{{{c.cmax}}}" if c.status == "OPTIMAL" else f"{c.cmax} ({100 * c.gap:.1f})"
            for alg in ("GA", "VNS"):
                g = agg[(agg.instance == inst) & (agg.rho == rho) & (agg.alg == alg)]
                if len(g):
                    g = g.iloc[0]
                    r[f"{alg}$_{{{rho}}}$"] = f"{int(g['best'])}/{g['mean']:.1f}({g['sd']:.1f})"
        piv.append(r)
    write_tex(res, "E2_table", pd.DataFrame(piv),
              "CP-SAT (bold: optimal; otherwise best value and gap in \\%) and GA/VNS (best/mean(sd) over repeated runs) "
              "for each idle-recovery factor $\\rho$ ($\\mu_{idle}=\\rho\\mu$).", "tab:exp_res")


def e3(res):
    df = read(res, "E3_discretisation.csv")
    if df is None:
        return
    md("## E3 - Discretisation and time resolution (continuous replay of CP-SAT schedules)")
    base = df[(df.L == 1000) & (df.scale == 1) & (df.rounding == "nearest")].set_index(["instance", "rho"])["cmax"]
    rows = []
    for (rho, L, s, rnd), g in df.groupby(["rho", "L", "scale", "rounding"]):
        b = base.reindex(list(zip(g.instance, g.rho))).values
        rows.append(dict(rho=rho, L=L, scale=s, rounding=rnd, n=len(g), optimal=int((g.status == "OPTIMAL").sum()),
                         avg_cmax=g.cmax.mean(), dev_vs_base=np.nanmean(pct(g.cmax.values, b)),
                         cont_dev=np.nanmean(pct(g.cont_keep_cmax, g.cmax)),
                         cont_dev_max=np.nanmax(pct(g.cont_keep_cmax, g.cmax)),
                         sched_with_viol=int((g.cont_keep_viol_end > 0).sum()),
                         peak_max=g.cont_keep_peak.max(), cont_left_dev=np.nanmean(pct(g.cont_left_cmax, g.cmax)),
                         avg_time=g.wall_time.mean()))
    table_md(pd.DataFrame(rows), ".3f")


def e4(res):
    df = read(res, "E4_sensitivity.csv")
    if df is None:
        return
    md("## E4 - Sensitivity to fatigue parameters (one at a time)")
    r0 = df[df.rho == 0].copy()
    rhos_idle = sorted(r for r in df.rho.unique() if r > 0)
    r1 = df[df.rho == (rhos_idle[0] if rhos_idle else -1)].set_index(["instance", "param", "level"])["OPT2"]
    e1 = read(res, "E1_simulation.csv")
    opt1 = e1.set_index("instance")["OPT1"] if e1 is not None else None
    rows = []
    for (p, lv), g in r0.groupby(["param", "level"], dropna=False):
        g = g[g.feasible == 1]
        if not len(g):
            rows.append(dict(param=p, level=lv, feasible=0))
            continue
        row = dict(param=p, level=lv, n=len(g), OPT2=g.OPT2.mean(), rests=g.n_rest.mean(),
                   SIM1_viol_inst=int((g.SIM1_viol_end > 0).sum()), SIM2_vs_OPT2=pct(g.SIM2_cmax, g.OPT2).mean())
        if opt1 is not None:
            row["OPT2_vs_OPT1"] = pct(g.OPT2.values, opt1.reindex(g.instance).values).mean()
        if len(r1):
            v1 = r1.reindex(list(zip(g.instance, g.param, g.level))).values
            row[f"idle_gain_rho{rhos_idle[0]}"] = np.nanmean(100 * (g.OPT2.values - v1) / g.OPT2.values)
        rows.append(row)
    table_md(pd.DataFrame(rows))


def e5(res):
    df = read(res, "E5_heterogeneous.csv")
    if df is None:
        return
    md("## E5 - Heterogeneous operators")
    md(f"- {len(df)} instance-draws; OPT2 optimal in {(df.status == 'OPTIMAL').sum()}")
    md(f"- SIM1 instances above F_max: {(df.SIM1_viol_end > 0).sum()}/{len(df)}; SIM2 vs OPT2 mean "
       f"{pct(df.SIM2_cmax, df.OPT2).mean():.2f}% (median {pct(df.SIM2_cmax, df.OPT2).median():.2f}%)")
    v = read(res, "E5_vns.csv")
    if v is not None:
        ref = df.set_index(["instance", "draw"])["OPT2"]
        v["rpd"] = pct(v.cmax, [ref.get((i, d), np.nan) for i, d in zip(v.instance, v.draw)])
        md(f"- VNS ARPD vs CP-SAT: {v.rpd.mean():.2f}% (hit rate {100 * (v.rpd <= 1e-9).mean():.1f}%)")
    md()


def e6(res):
    cp = read(res, "E6_large_cpsat.csv")
    if cp is None:
        return
    md("## E6 - Larger instances")
    mh = read(res, "E6_large_metaheuristics.csv")
    rows = []
    for r in cp.itertuples():
        row = dict(instance=r.instance, rho=r.rho, size=f"{r.n_jobs}x{r.M}", R=r.R, OPT1=r.OPT1,
                   OPT1_status=r.OPT1_status, OPT2=getattr(r, "OPT2", np.nan), status=getattr(r, "status", ""),
                   gap=100 * getattr(r, "gap", np.nan) if not pd.isna(getattr(r, "gap", np.nan)) else np.nan,
                   SIM2=getattr(r, "SIM2_cmax", np.nan))
        if mh is not None:
            for alg in ("GA", "VNS"):
                g = mh[(mh.instance == r.instance) & (mh.rho == r.rho) & (mh.alg == alg)]
                if len(g):
                    row[f"{alg}_best"], row[f"{alg}_mean"], row[f"{alg}_sd"] = g.cmax.min(), g.cmax.mean(), g.cmax.std()
        rows.append(row)
    t = pd.DataFrame(rows)
    table_md(t)
    write_tex(res, "E6_table", t, "Larger instances (JSPLIB job data with transport robots and fatigue).", "tab:large")


def e7(res):
    cp = read(res, "E7_original_bu_cpsat.csv")
    if cp is None:
        return
    md("## E7 - Original Bilge & Ulusoy benchmark (no fatigue) vs literature")
    pub = pd.read_csv(Path(__file__).parent / "fajsspt/data/published_bu_original.csv", comment="#")
    refs = [c for c in pub.columns if c.endswith("_cmax")]
    df = order(pub[["instance"] + refs + ["best_known"]].merge(
        cp.rename(columns={"cmax": "CPSAT_rev", "wall_time": "CPSAT_rev_time", "status": "CPSAT_rev_status"})
        [["instance", "CPSAT_rev", "CPSAT_rev_time", "CPSAT_rev_status", "bound"]], on="instance", how="right"))
    mh = read(res, "E7_original_bu_metaheuristics.csv")
    if mh is not None:
        for (alg, v), g in mh.groupby(["alg", "variant"]):
            a = g.groupby("instance").agg(best=("cmax", "min"), mean=("cmax", "mean"), sd=("cmax", "std"),
                                          ttb=("time_to_best", "mean"))
            for k in a.columns:
                df[f"{alg}[{v}]_{k}"] = df.instance.map(a[k])
    bk = df["best_known"]
    md(f"- Revised CP-SAT: optimal {int((df.CPSAT_rev_status == 'OPTIMAL').sum())}/{len(df)}; equal to best known "
       f"{int((df.CPSAT_rev == bk).sum())}, better {int((df.CPSAT_rev < bk).sum())}, worse {int((df.CPSAT_rev > bk).sum())}")
    for c in [c for c in df.columns if c.endswith("_best")]:
        alg = c[:-5]
        md(f"- {alg}: best = best known in {int((df[c] == bk).sum())}/{len(df)}; ARPD(best) "
           f"{pct(df[c], bk).mean():.2f}%, ARPD(mean) {pct(df[alg + '_mean'], bk).mean():.2f}%")
    for r in refs:
        md(f"- {r[:-5]}: ARPD vs best known {pct(df[r], bk).mean():.2f}%, hits {int((df[r] == bk).sum())}")
    avg = {c: df[c].mean() for c in df.columns if c != "instance" and pd.api.types.is_numeric_dtype(df[c])}
    table_md(pd.concat([df, pd.DataFrame([dict(instance="Average", **avg)])], ignore_index=True))
    show = ["instance"] + refs + ["CPSAT_rev"] + [c for c in df.columns if c.endswith("_best") or c.endswith("_mean")]
    t = df[show].copy()
    t = pd.concat([t, pd.DataFrame([dict(instance="Average", **{c: t[c].mean() for c in show[1:]})])], ignore_index=True)
    write_tex(res, "E7_table", t, "Original Bilge \\& Ulusoy (1995) instances (no fatigue): literature, revised CP-SAT "
              "and GA/VNS (best and mean over repeated runs).", "tab:bu_original")


def e8(res):
    df = read(res, "E8_ablation.csv")
    if df is None:
        return
    md("## E8 - Ablation of GA/VNS improvements")
    cp = read(res, "E2_cpsat.csv")
    if cp is not None:
        ref = cp.set_index(["instance", "rho"])["cmax"]
        df["ref"] = [ref.get((i, r), np.nan) for i, r in zip(df.instance, df.rho)]
    else:
        df["ref"] = df.groupby(["instance", "rho"]).cmax.transform("min")
    df["rpd"] = pct(df.cmax, df.ref)
    t = df.groupby(["rho", "alg", "variant"]).agg(ARPD=("rpd", "mean"), hit=("rpd", lambda x: 100 * (x <= 1e-9).mean()),
                                                 evals=("evals", "mean")).reset_index()
    vo = {v: q for q, v in enumerate(["legacy", "I1", "I1+I2", "I1+I3", "I1+I2+I3", "full"])}
    t = t.sort_values(["rho", "alg"], kind="stable").sort_values(["rho", "alg"], key=lambda s: s, kind="stable")
    t["_v"] = t.variant.map(vo)
    table_md(t.sort_values(["rho", "alg", "_v"]).drop(columns="_v"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    md(f"# Revision experiments - summary ({a.results})\n")
    for f in (e0, e1, e2, e3, e4, e5, e6, e7, e8):
        try:
            f(a.results)
        except Exception as ex:          # keep going, report the problem
            md(f"**{f.__name__} could not be summarised: {ex!r}**\n")
    out = Path(a.results) / "summary.md"
    out.write_text("\n".join(OUT))
    print("\n".join(OUT))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
