"""Figure 1 of the revised manuscript: fatigue dynamics under the hard bound
(work from F_min, rest triggered because the next operation would exceed F_max)."""
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

lam, mu, Fmin, Fmax = 0.04, 0.08, 0.2, 0.8
acc = lambda F, t: 1 - (1 - F) * math.exp(-lam * t)
fig, ax = plt.subplots(figsize=(12, 6))
RED, GREEN, GREY = "#d62728", "#2ca02c", "#555555"
# operation 1 and 2 (work), idle, rest, operation 3
segs, t, F = [], 0.0, Fmin
for d in (14, 12):                                    # two operations
    xs = [t + d * q / 60 for q in range(61)]
    ax.plot(xs, [acc(F, x - t) for x in xs], color=RED, lw=3)
    t, F = t + d, acc(F, d)
t_idle_end = t                                       # next operation would start here
# predicted fatigue of the next operation without rest (exceeds Fmax)
xs = [t_idle_end + 18 * q / 60 for q in range(61)]
ax.plot(xs, [acc(F, x - t_idle_end) for x in xs], color=RED, lw=1.8, ls="--", alpha=0.55)
ax.annotate("next operation\nwould exceed $F^{\\max}$", xy=(t_idle_end + 15, acc(F, 15)),
            xytext=(t_idle_end + 13, 0.95), fontsize=13, color="#7f1d1d",
            arrowprops=dict(arrowstyle="->", color="#7f1d1d"))
# rest from the end of the predecessor (dominant position)
d_rest = math.log(F / Fmin) / mu
xs = [t + d_rest * q / 60 for q in range(61)]
ax.plot(xs, [F * math.exp(-mu * (x - t)) for x in xs], color=GREEN, lw=3)
t2 = t + d_rest
xs = [t2 + 18 * q / 60 for q in range(61)]
ax.plot(xs, [acc(Fmin, x - t2) for x in xs], color=RED, lw=3)
tend = t2 + 18
for y, lab in ((Fmax, "$F^{\\max}$"), (Fmin, "$F^{\\min}$")):
    ax.axhline(y, color="#666666", ls="--", lw=1.6)
    ax.text(-1.0, y, lab, ha="right", va="center", fontsize=17)
for x in (26, t, t2):
    ax.axvline(x, color="#999999", ls=":", lw=1.2)
ymid = 0.06
for (a, b, lab) in ((0, 26, "Work (2 operations)"), (t, t2, "Rest"), (t2, tend, "Work")):
    ax.annotate("", xy=(a + 0.6, ymid), xytext=(b - 0.6, ymid), arrowprops=dict(arrowstyle="<->", lw=1.8))
    ax.text((a + b) / 2, ymid + 0.035, lab, ha="center", fontsize=15)
ax.set_xlim(0, tend + 1); ax.set_ylim(0, 1.05)
ax.set_yticks([0, 1]); ax.set_xticks([])
ax.set_xlabel("Time", fontsize=18); ax.set_ylabel("Fatigue level", fontsize=18)
ax.tick_params(labelsize=15)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig("Fig.1.png", dpi=250)
print("Fig.1.png written")
