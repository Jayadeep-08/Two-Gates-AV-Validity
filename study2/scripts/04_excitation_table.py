"""STEP 2 - excitation table for highD. Descriptive only; no model is fitted.

Pre-registration: 05_PREREGISTRATION_DRAFT.md B.4 (Step 2) and Amendment F. It describes
the calibration populations before any fit:
    site groups : location 1 (primary), location 5 (replication), other (2,3,4,6; descriptive),
                  all highD pooled (sensitivity)
    sets        : car-led >= 10 s, E10,0.5, E10,1.0 (primary), E10,1.5
Nothing here is an outcome and nothing here selects anything. Its job is to state
what information each population carries, so that later differences in identifiability
can be read against it - together with the Step 4 synthetic bridge, which is what
actually separates information from behaviour (Amendment F.3).

Reads   derived/highd_episodes_defE/??_index.csv  (definition E == C for highD)
Writes  code/results/step2_excitation_highd.json   aggregates only
        code/results/step2_excitation_highd.md     human-readable tables
        code/results/figures/step2_*.png           aggregate plots (if matplotlib)

Usage (from AINN_ReWORK/scripts):   python 04_excitation_table.py
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

from _common import DERIVED, RESULTS

SETS = {"carled_10s": 0.0, "E10,0.5": 0.5, "E10,1.0": 1.0, "E10,1.5": 1.5}
GROUPS = {"loc1_primary": [1], "loc5_replication": [5], "other_2346": [2, 3, 4, 6],
          "all_pooled": [1, 2, 3, 4, 5, 6]}
QUANT = ["sd_vL", "sd_aL", "sd_dv", "range_dv", "sd_gap", "mean_gap", "min_gap",
         "sd_thw", "mean_thw", "mean_vF", "min_vF", "max_vF", "sd_aF", "min_aF", "min_aL",
         "dur_s"]


def load(d):
    fs = [f for f in sorted(glob.glob(os.path.join(d, "??_index.csv"))) if os.path.getsize(f) > 1]
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


def in_set(df, x):
    return df[(df["leader_class"] == "car") & (df["dur_s"] >= 10) & (df["sd_vL"] >= x)]


def describe(df):
    if df.empty:
        return {"episodes": 0}
    out = {"episodes": int(len(df)),
           "drivers": int(df[["rec", "follower"]].drop_duplicates().shape[0]),
           "recordings": int(df["rec"].nunique()),
           "hours": round(float(df["dur_s"].sum() / 3600), 3),
           "frac_leader_min_acc_le_-2": round(float((df["min_aL"] <= -2).mean()), 4),
           "frac_leader_min_acc_le_-3": round(float((df["min_aL"] <= -3).mean()), 4),
           "frac_follower_min_acc_le_-2": round(float((df["min_aF"] <= -2).mean()), 4),
           "frac_min_vF_lt_15": round(float((df["min_vF"] < 15).mean()), 4),
           "frac_min_vF_lt_8": round(float((df["min_vF"] < 8).mean()), 4)}
    for q in QUANT:
        out[q] = {str(p): round(float(df[q].quantile(p)), 3) for p in (0.1, 0.5, 0.9)}
    return out


def main():
    d = os.path.join(DERIVED, "highd_episodes_defE")
    ep = load(d)
    res = {"dataset": "highD", "definition": "E (== C for highD)",
           "note": "descriptive only; no fit; B.4 Step 2, Amendment F", "table": {}}
    for g, locs in GROUPS.items():
        sub = ep[ep["location"].isin(locs)]
        res["table"][g] = {k: describe(in_set(sub, x)) for k, x in SETS.items()}
    # driver multiplicity: episodes per driver in the primary set
    e10 = in_set(ep[ep["location"] == 1], 1.0)
    res["loc1_E10_1.0_episodes_per_driver"] = e10.groupby(["rec", "follower"]).size().value_counts().sort_index().to_dict()
    # speed regime overlap between the two sites (information vs behaviour caveat)
    l1, l5 = in_set(ep[ep["location"] == 1], 1.0), in_set(ep[ep["location"] == 5], 1.0)
    bins = [0, 10, 15, 20, 25, 30, 35, 50]
    res["mean_vF_distribution_E10_1.0"] = {
        "bins_m_s": bins,
        "loc1": np.round(np.histogram(l1["mean_vF"], bins)[0] / max(len(l1), 1), 4).tolist(),
        "loc5": np.round(np.histogram(l5["mean_vF"], bins)[0] / max(len(l5), 1), 4).tolist()}

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "step2_excitation_highd.json"), "w") as f:
        json.dump(res, f, indent=2)
    md = render(res)
    with open(os.path.join(RESULTS, "step2_excitation_highd.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    try:
        plots(ep)
    except ImportError:
        print("matplotlib not installed - plots skipped")
    return 0


def render(res):
    rows = [("episodes", lambda t: t["episodes"]), ("drivers", lambda t: t.get("drivers")),
            ("recordings", lambda t: t.get("recordings")), ("hours", lambda t: t.get("hours"))]
    med = ["sd_vL", "sd_aL", "sd_dv", "sd_gap", "sd_thw", "mean_gap", "mean_thw", "mean_vF",
           "min_vF", "dur_s"]
    L = ["# Step 2 - highD excitation table (descriptive, no fit)", "",
         "Median [p10, p90] per episode. Site groups per Amendment F.", ""]
    for g, t in res["table"].items():
        L += [f"## {g}", "", "| | " + " | ".join(SETS) + " |", "|---" * (len(SETS) + 1) + "|"]
        for name, f in rows:
            L.append(f"| {name} | " + " | ".join(str(f(t[k])) for k in SETS) + " |")
        for q in med:
            L.append(f"| {q} | " + " | ".join(
                f"{t[k][q]['0.5']} [{t[k][q]['0.1']}, {t[k][q]['0.9']}]" if t[k]["episodes"] else "-"
                for k in SETS) + " |")
        for q in ("frac_leader_min_acc_le_-2", "frac_leader_min_acc_le_-3",
                  "frac_follower_min_acc_le_-2", "frac_min_vF_lt_15", "frac_min_vF_lt_8"):
            L.append(f"| {q} | " + " | ".join(str(t[k].get(q, "-")) for k in SETS) + " |")
        L.append("")
    L.append(f"Episodes per driver, loc 1 E10,1.0: {res['loc1_E10_1.0_episodes_per_driver']}")
    m = res["mean_vF_distribution_E10_1.0"]
    L.append(f"mean_vF share by bin {m['bins_m_s']} (m/s): loc1 {m['loc1']} | loc5 {m['loc5']}")
    return "\n".join(L) + "\n"


def plots(ep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = os.path.join(RESULTS, "figures")
    os.makedirs(fd, exist_ok=True)
    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    vars_ = [("sd_vL", "leader speed SD (m/s)"), ("sd_aL", "leader accel SD (m/s²)"),
             ("sd_dv", "relative speed SD (m/s)"), ("mean_vF", "mean follower speed (m/s)"),
             ("mean_gap", "mean gap (m)"), ("sd_thw", "time-headway SD (s)")]
    base = ep[(ep["leader_class"] == "car") & (ep["dur_s"] >= 10)]
    for a, (v, lab) in zip(ax.ravel(), vars_):
        for g, locs in (("loc 1", [1]), ("loc 5", [5]), ("other", [2, 3, 4, 6])):
            x = base[base["location"].isin(locs)][v]
            a.hist(x, bins=60, range=(x.quantile(.005), x.quantile(.995)) if len(x) else None,
                   density=True, histtype="step", label=g)
        a.set(xlabel=lab)
    ax[0, 0].legend()
    fig.suptitle("highD car-led >= 10 s episodes by site (density)")
    fig.tight_layout(); fig.savefig(os.path.join(fd, "step2_excitation_by_site.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
