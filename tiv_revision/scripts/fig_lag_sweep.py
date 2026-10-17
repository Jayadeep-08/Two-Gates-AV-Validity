"""Figure: lag-sweep decay of the severity-risk association. IEEE double-column (7.16in).

Reads tiv_revision/numbers.json (produced by table1.py).
Run:  py -3.7 tiv_revision\\scripts\\fig_lag_sweep.py
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HD = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("RPI_OUT") or os.path.abspath(os.path.join(HD, ".."))
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
lag = json.load(open(os.path.join(ROOT, "numbers.json")))["lag"]

# validated categorical palette (all checks pass, --pairs all, light surface)
# secondary encoding: solid = scripted treatment, broken = no independent treatment
STYLE = {
    "C": dict(c="#0072B2", ls="-",  m="o", lw=1.6, label="C  emergency braking (treatment)"),
    "A": dict(c="#009E73", ls="-",  m="s", lw=1.6, label="A  lane change (treatment)"),
    "B": dict(c="#E69F00", ls=":",  m="^", lw=1.6, label="B  intersection yield-cross (no treatment)"),
    "D": dict(c="#D55E00", ls="--", m="D", lw=2.0, label="D  open-road lane keep (no decision)"),
}
INK, MUTED, GRID = "#1a1a1a", "#595959", "#d8d8d8"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "STIX Two Text", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.edgecolor": MUTED, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.dpi": 400, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.16, 2.80))
fig.subplots_adjust(wspace=0.26, bottom=0.265)

for ax in (axL, axR):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.xaxis.grid(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.axvspan(5.5, 7.4, color="#000000", alpha=0.045, lw=0, zorder=0)
    ax.set_xlim(-0.25, 7.4)
    ax.set_xticks(range(8))
    ax.set_xlabel("decision lag $k$")

# ---------------- panel (a): raw association
axL.axhline(0, color=MUTED, lw=0.7, zorder=1)
for key in ("C", "A", "B", "D"):
    s = lag[key]; st = STYLE[key]
    ks = [p["k"] for p in s]
    axL.fill_between(ks, [p["lo"] for p in s], [p["hi"] for p in s],
                     color=st["c"], alpha=0.13, lw=0, zorder=2)
    axL.plot(ks, [p["r"] for p in s], color=st["c"], ls=st["ls"], lw=st["lw"],
             marker=st["m"], ms=3.4, mec="white", mew=0.5, zorder=3)
axL.set_ylim(-0.72, 1.0)
axL.set_yticks([-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])
axL.set_ylabel(r"$r$ (severity at $t$,  $\rho$ at $t{+}k$)")
axL.set_title("(a)  association by lag, with 95% CI", loc="left", pad=5)
axL.text(6.45, -0.655, "$\\leq$2 pairs/episode", fontsize=5.8, color=MUTED,
         ha="center", style="italic")

# ---------------- panel (b): decay profile
for key in ("C", "A", "D"):
    s = lag[key]; st = STYLE[key]
    ks = [p["k"] for p in s]; ys = [p["retention"] for p in s]
    axR.plot(ks, ys, color=st["c"], ls=st["ls"], lw=st["lw"],
             marker=st["m"], ms=3.4, mec="white", mew=0.5, zorder=3)
    # label at k=5, the last point outside the reduced-n band, where the
    # three curves are well separated
    y5 = [p["retention"] for p in s if p["k"] == 5][0]
    axR.annotate(key, xy=(5, y5), xytext=(0, 5.5), textcoords="offset points",
                 color=st["c"], fontsize=8, fontweight="bold",
                 ha="center", va="bottom", zorder=4)
axR.set_ylim(-0.03, 1.09)
axR.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
axR.set_ylabel(r"$|r(k)| \,/\, |r(0)|$")
axR.set_title("(b)  decay profile, normalised to $k{=}0$", loc="left", pad=5)
axR.text(0.18, 0.10, "the no-decision control decays\nmore slowly than either treatment",
         fontsize=6.6, color=MUTED, ha="left", va="center", linespacing=1.45)

handles = [Line2D([], [], color=STYLE[k]["c"], ls=STYLE[k]["ls"], lw=STYLE[k]["lw"],
                  marker=STYLE[k]["m"], ms=3.4, mec="white", mew=0.5,
                  label=STYLE[k]["label"]) for k in ("C", "A", "B", "D")]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, 0.005), handlelength=2.6, columnspacing=1.8,
           labelspacing=0.3)

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, "fig_lag_sweep." + ext))
print("wrote figures/fig_lag_sweep.pdf and .png")
