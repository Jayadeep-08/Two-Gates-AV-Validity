"""Figure: the two-agent chain experiment (replaces fig6_chain, which was built on v1 data).

(a) following gap by chain position against the isolated-decision baseline  -> the null
(b) chain-minus-isolated difference by inter-decision spacing               -> the dose-response
(c) ego entry speed by chain position                                       -> the regime control

Reads tiv_revision/chain_numbers.json (produced by chain_numbers.py).
Run:  py -3.7 tiv_revision\\scripts\\fig_chain.py
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HD = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("RPI_OUT") or os.path.abspath(os.path.join(HD, ".."))
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
N = json.load(open(os.path.join(ROOT, "chain_numbers.json")))

CHAIN, ISO = "#0072B2", "#D55E00"      # validated pair, all checks pass
INK, MUTED, GRID = "#1a1a1a", "#595959", "#d8d8d8"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "STIX Two Text", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.edgecolor": MUTED, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.dpi": 400, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.16, 2.55))
fig.subplots_adjust(wspace=0.42, bottom=0.30)
for ax in (a1, a2, a3):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

# ---------------- (a) gap by position, against the isolated baseline
prof = N["profile"]
x = [p["pos"] + 1 for p in prof]
a1.axhspan(N["iso_lo"], N["iso_hi"], color=ISO, alpha=0.15, lw=0, zorder=1)
a1.axhline(N["iso_gap"], color=ISO, ls="--", lw=1.4, zorder=2)
a1.errorbar(x, [p["mean"] for p in prof],
            yerr=[[p["mean"] - p["lo"] for p in prof], [p["hi"] - p["mean"] for p in prof]],
            fmt="o", ms=3.6, lw=1.4, capsize=2, color=CHAIN,
            mec="white", mew=0.5, zorder=3)
a1.plot(x, [p["mean"] for p in prof], color=CHAIN, lw=1.3, zorder=2)
a1.set_xlim(0.4, 8.6); a1.set_xticks(range(1, 9))
a1.set_ylim(5.5, 20.5)
a1.set_xlabel("decision in chain")
a1.set_ylabel("following gap after decision (m)")
a1.set_title("(a)  gap by chain position", loc="left", pad=5)

# ---------------- (b) dose-response
sp = [s["spacing_s"] for s in N["by_spacing"]]
df = [s["diff"] for s in N["by_spacing"]]
a2.axhline(0, color=MUTED, lw=0.7, zorder=1)
b = N["ols_beta"]
xs = [0.75, 4.85]
a2.plot(xs, [b[2] + b[3] * v for v in xs], color=CHAIN, lw=1.2, ls="-", alpha=0.55, zorder=2)
a2.errorbar(sp, df,
            yerr=[[d - s["lo"] for d, s in zip(df, N["by_spacing"])],
                  [s["hi"] - d for d, s in zip(df, N["by_spacing"])]],
            fmt="o", ms=4.0, lw=1.4, capsize=2.5, color=CHAIN,
            mec="white", mew=0.6, zorder=3)
a2.set_xlim(0.55, 5.05)
a2.set_xticks(sp); a2.set_xticklabels([f"{v:.1f}" for v in sp])
a2.set_ylim(-1.80, 2.20)
a2.set_xlabel("inter-decision spacing (s)")
a2.set_ylabel("chain $-$ isolated gap (m)")
a2.set_title("(b)  dependence on spacing", loc="left", pad=5)
a2.text(0.62, 2.03, f"interaction {N['interaction']:+.2f} m/s",
        fontsize=5.9, color=MUTED, ha="left", va="center")
a2.text(0.62, 1.76, f"95% CI [{N['interaction_lo']:+.2f}, {N['interaction_hi']:+.2f}]",
        fontsize=5.9, color=MUTED, ha="left", va="center")
a2.text(4.95, -1.62, "accumulation lies below zero",
        fontsize=5.9, color=MUTED, ha="right", va="center", style="italic")

# ---------------- (c) entry-speed regime control
a3.errorbar(x, [p["entry_mean"] for p in prof], yerr=[p["entry_sd"] for p in prof],
            fmt="s", ms=3.2, lw=1.2, capsize=2, color=CHAIN, mec="white", mew=0.5, zorder=3)
a3.set_xlim(0.4, 8.6); a3.set_xticks(range(1, 9))
a3.set_ylim(10.92, 11.82)
a3.set_yticks([11.0, 11.2, 11.4, 11.6])
a3.set_xlabel("decision in chain")
a3.set_ylabel("ego entry speed (m/s)")
a3.set_title("(c)  regime control", loc="left", pad=5)
a3.text(0.75, 11.76, f"spread of means {N['entry_spread']:.3f} m/s",
        fontsize=6.2, color=MUTED, ha="left", va="center")

handles = [Line2D([], [], color=CHAIN, marker="o", ms=3.6, lw=1.3, mec="white", mew=0.5,
                  label="CHAIN (mean, 95% CI)"),
           Line2D([], [], color=ISO, ls="--", lw=1.4, label="ISO-LATE (mean)"),
           Patch(facecolor=ISO, alpha=0.15, label="ISO-LATE 95% CI")]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 0.0), handlelength=2.4, columnspacing=2.0)

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, "fig_chain." + ext))
print("wrote figures/fig_chain.pdf and .png")
