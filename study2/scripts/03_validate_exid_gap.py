"""STEP 1c - validate the seam-corrected exiD gap (Amendment E). Measurement only.

No model is fitted here. This script reads the extraction outputs only:
  derived/exid_episodes_defE            seam-corrected gap, F9 tau 0.3 (primary)
  derived/exid_episodes_defE_f9tau0.2   seam-corrected gap, F9 tau 0.2 (sensitivity)
  derived/exid_episodes_defC[_f9tau0.2] raw leadDHW (legacy arm)
  derived/exid_episodes_defD            traveledDistance reconstruction (FAILED arm, optional)
  derived/highd_episodes_defE           highD reference
and evaluates the decision criteria E-A ... E-F frozen in 05_PREREGISTRATION_DRAFT.md E.2
(thresholds copied below, not chosen here). The Amendment D validator is archived in
_archive/03_validate_exid_gap_defD.py.

Outputs
  code/results/step1c_exid_gap_validation_defE.json    machine-readable, aggregates only
  code/results/step1c_exid_gap_validation_defE.txt     human-readable report
  code/results/figures/step1c_E_*.png                  aggregate plots (if matplotlib)
  derived/exid_gap_audit_defE.csv                      episode-level audit (LOCAL ONLY)
  derived/figures_local/step1c_E_examples.png          example trajectories (LOCAL ONLY)

Usage (from AINN_ReWORK/scripts):   python 03_validate_exid_gap.py
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

from _common import DERIVED, RESULTS, STEP_EDGES

# ---- E.2 thresholds, frozen 2026-09-26 11:02 UTC before any E diagnostic was computed --
EA_EXCEED, EA_P99 = 0.001, 0.10     # non-seam raw leadDHW: P(|eps|>0.3), p99 |eps| (m)
EB_EXCEED = 0.001                   # seam frames: P(|eps_traveledDistance| > 0.3 m)
EC_MAX_LOSS, EC_MAX_GRADIENT = 0.02, 0.01
ED_SPREAD, ED_SHARE, ED_EXCL_GRADIENT = 0.30, 0.50, 0.05
EE_BIG_LOCATIONS, EE_DROP_FAILS = (0, 1, 4), 2
SETS = {"carled_10s": 0.0, "E10,0.5": 0.5, "E10,1.0": 1.0, "E10,1.5": 1.5}


def folder(name):
    return os.path.join(DERIVED, name)


def load_index(d, suffix=""):
    fs = [f for f in sorted(glob.glob(os.path.join(d, f"??_index{suffix}.csv")))
          if os.path.getsize(f) > 1]
    if not fs:
        raise FileNotFoundError(f"no index files in {d} - run the extraction first")
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


def in_set(df, x):
    return df[(df["leader_class"] == "car") & (df["dur_s"] >= 10) & (df["sd_vL"] >= x)]


def hours(df):
    return float(df["dur_s"].sum() / 3600)


def loss(kept, ref):
    b = hours(ref)
    return round(1 - hours(kept) / b, 4) if b > 0 else None


def f9_losses(prim, no9):
    return {k: {"hours_primary": round(hours(in_set(prim, x)), 3),
                "hours_without_F9": round(hours(in_set(no9, x)), 3),
                "loss": loss(in_set(prim, x), in_set(no9, x))} for k, x in SETS.items()}


def f9_loss_by_location(prim, no9, x=1.0):
    return {int(L): loss(in_set(prim[prim["location"] == L], x), in_set(g, x))
            for L, g in no9.groupby("location")}


def load_stepdiag(d):
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "??_stepdiag.json")))]
    if not rows:
        raise FileNotFoundError(f"no stepdiag files in {d}")
    return rows


def step_summary(diags, key, location=None):
    """|step| quantiles from fixed-edge histograms (upper bin edges, 1 mm resolution below
    0.5 m, so conservative) plus exact exceedance counts."""
    ds = [d for d in diags if (location is None or d["location"] == location) and key in d["hist"]]
    if not ds:
        return {"n": 0}
    h = np.sum([d["hist"][key] for d in ds], axis=0)
    n = int(h.sum())
    if n == 0:
        return {"n": 0}
    c = np.cumsum(h) / n
    q = lambda p: round(float(STEP_EDGES[1:][np.searchsorted(c, p)]), 4)
    gt = {t: int(sum(d[f"n_{key}_gt{t:g}"] for d in ds)) for t in (0.2, 0.3)}
    return {"n": n, "median_ub": q(0.5), "p95_ub": q(0.95), "p99_ub": q(0.99),
            "max": round(float(max(d[f"max_{key}"] for d in ds)), 4),
            "frac_gt0.2": gt[0.2] / n, "frac_gt0.3": gt[0.3] / n}


def overlap_fraction(cand, prim):
    """Fraction of each candidate episode's frames lying inside primary episodes of the
    same recording and pair."""
    def parse(df):
        k = df["episode"].str.rsplit("_", n=3, expand=True)
        return df.assign(start=k[3].astype(int)).assign(end=lambda d: d["start"] + d["n_frames"])
    c, p = parse(cand), parse(prim)
    m = c.reset_index().merge(p[["rec", "follower", "leader", "start", "end"]],
                              on=["rec", "follower", "leader"], how="left", suffixes=("", "_p"))
    ov = (np.minimum(m["end"], m["end_p"]) - np.maximum(m["start"], m["start_p"])).clip(lower=0)
    got = ov.fillna(0).groupby(m["index"]).sum()
    return (got.reindex(c.index).fillna(0) / c["n_frames"]).to_numpy()


def main():
    dE, dE2 = folder("exid_episodes_defE"), folder("exid_episodes_defE_f9tau0.2")
    dC, dC2 = folder("exid_episodes_defC"), folder("exid_episodes_defC_f9tau0.2")
    dD, dH = folder("exid_episodes_defD"), folder("highd_episodes_defE")

    pE, nE, pE2 = load_index(dE), load_index(dE, "_noF9"), load_index(dE2)
    pC, nC, pC2 = load_index(dC), load_index(dC, "_noF9"), load_index(dC2)
    pH, nH = load_index(dH), load_index(dH, "_noF9")
    sdE, sdH = load_stepdiag(dE), load_stepdiag(dH)
    locs = sorted(int(x) for x in nE["location"].unique())
    res = {"amendment": "E", "thresholds": {
        "EA_exceed": EA_EXCEED, "EA_p99_m": EA_P99, "EB_exceed": EB_EXCEED,
        "EC_max_loss": EC_MAX_LOSS, "EC_max_gradient": EC_MAX_GRADIENT,
        "ED_spread_m": ED_SPREAD, "ED_share": ED_SHARE, "ED_excl_gradient": ED_EXCL_GRADIENT,
        "EE_big_locations": list(EE_BIG_LOCATIONS), "EE_drop_fails": EE_DROP_FAILS}}

    # ---- E-A leadDHW away from seams ----------------------------------------------------
    def eA(loc):
        s = step_summary(sdE, "ref_noseam", loc)
        s["pass"] = bool(s["n"] and s["frac_gt0.3"] <= EA_EXCEED and s["p99_ub"] <= EA_P99)
        return s
    EA = {"pooled": eA(None), "by_location": {L: eA(L) for L in locs}}
    res["EA_leadDHW_off_seam"] = EA

    # ---- E-B seam step is a reference change ----------------------------------------------
    def eB(loc):
        s = step_summary(sdE, "td_seam", loc)
        s["pass"] = bool(s["n"] and s["frac_gt0.3"] <= EB_EXCEED)
        s["raw_leadDHW_seam"] = step_summary(sdE, "ref_seam", loc)
        return s
    EB = {"pooled": eB(None), "by_location": {L: eB(L) for L in locs}}
    res["EB_seam_reference_change"] = EB

    # ---- E-C excitation neutrality ---------------------------------------------------------
    lossE = f9_losses(pE, nE)
    grad = lossE["E10,1.5"]["loss"] - lossE["carled_10s"]["loss"]
    locE = f9_loss_by_location(pE, nE)
    EC = {"seam_corrected": lossE, "raw_legacy_defC": f9_losses(pC, nC), "highD": f9_losses(pH, nH),
          "gradient_E10_1.5_minus_carled": round(grad, 4),
          "E10_1.0_loss_by_location": locE,
          "raw_legacy_E10_1.0_loss_by_location": f9_loss_by_location(pC, nC),
          "pass": bool(all(v["loss"] is not None and v["loss"] <= EC_MAX_LOSS for v in lossE.values())
                       and grad <= EC_MAX_GRADIENT)}
    res["EC_excitation_neutrality"] = EC

    # ---- E-E location robustness (before E-D: the replication set is decided by location) --
    per = {}
    for L in locs:
        okC = locE.get(L) is not None and locE[L] <= EC_MAX_LOSS
        per[L] = {"EA": EA["by_location"][L]["pass"], "EB": EB["by_location"][L]["pass"],
                  "EC_E10_1.0": bool(okC)}
        per[L]["pass"] = all(per[L].values())
    fails = [L for L in locs if not per[L]["pass"]]
    drop = sum(L in fails for L in EE_BIG_LOCATIONS) >= EE_DROP_FAILS
    res["EE_location_robustness"] = {"per_location": per, "failing_locations": fails,
                                     "drop_exiD_replication": bool(drop)}

    # ---- E-D anchor spread ------------------------------------------------------------------
    keep = pE[~pE["location"].isin(fails)]
    e10 = in_set(keep, 1.0)
    flagged = e10["run_anchor_spread"] > ED_SPREAD
    share = hours(e10[flagged]) / hours(e10) if len(e10) else float("nan")
    excl = {k: loss(in_set(keep[keep["run_anchor_spread"] <= ED_SPREAD], x), in_set(keep, x))
            for k, x in SETS.items()}
    excl_grad = excl["E10,1.5"] - excl["carled_10s"]
    ED = {"flagged_share_E10_1.0_hours": round(share, 4),
          "flagged_share_by_location": {int(L): round(hours(g[g["run_anchor_spread"] > ED_SPREAD]) / hours(g), 4)
                                        for L, g in in_set(pE, 1.0).groupby("location")},
          "anchor_spread_q_primary": pE["run_anchor_spread"].quantile([.5, .9, .99]).round(4).to_dict(),
          "anchor_spread_q_E10_1.0": in_set(pE, 1.0)["run_anchor_spread"].quantile([.5, .9, .99]).round(4).to_dict(),
          "nseams_q": pE["run_nseams"].quantile([.5, .9, .99]).to_dict(),
          "hour_loss_if_flagged_excluded": excl,
          "exclusion_gradient_E10_1.5_minus_carled": round(excl_grad, 4)}
    if share > ED_SHARE:
        ED["replication_primary_set"] = "unflagged runs only"
        ED["excitation_biased_label"] = bool(excl_grad > ED_EXCL_GRADIENT)
    else:
        ED["replication_primary_set"] = "all runs (flagged runs excluded in sensitivity analysis)"
        ED["excitation_biased_label"] = False
    res["ED_anchor_spread"] = ED
    res["EF_no_hidden_selection"] = ("procedural: thresholds frozen in 05_PREREGISTRATION_DRAFT.md "
                                     "Amendment E (11:02 UTC) before this script was first run; no "
                                     "fitting code has been run on real data (DIARY Entries 18-20)")

    # ---- decision -----------------------------------------------------------------------------
    dataset_fail = [k for k, ok in (("E-A", EA["pooled"]["pass"]), ("E-B", EB["pooled"]["pass"]),
                                    ("E-C", EC["pass"])) if not ok]
    if drop or dataset_fail:
        decision = (f"exiD replication DROPPED (dataset-level failures {dataset_fail}, "
                    f"failing locations {fails}); report as a measurement finding. highD primary proceeds.")
    else:
        decision = ("PASS - freeze Amendment E. exiD replication set: locations "
                    f"{[L for L in locs if L not in fails]}"
                    + (f" (excluded: {fails})" if fails else "")
                    + f"; {ED['replication_primary_set']}"
                    + ("; LABEL: excitation-biased" if ED["excitation_biased_label"] else ""))
    res["verdict"] = {"dataset_level_failures": dataset_fail, "failing_locations": fails,
                      "drop": bool(drop or dataset_fail), "decision": decision}

    # ---- descriptive arm comparison -----------------------------------------------------------
    arms = {}
    for name, p, n, p2 in (("exiD_raw_leadDHW_defC", pC, nC, pC2),
                           ("exiD_seam_corrected_defE", pE, nE, pE2),
                           ("highD_defE", pH, nH, None)):
        arms[name] = {"episodes": int(len(p)), "hours": round(hours(p), 3),
                      "F9_marginal_loss_hours": loss(p, n),
                      "E10_1.0": {"episodes": int(len(in_set(p, 1.0))), "hours": round(hours(in_set(p, 1.0)), 3)},
                      "sd_vL_q_carled_10s": in_set(p, 0.0)["sd_vL"].quantile([.25, .5, .75, .9]).round(3).to_dict()}
        if p2 is not None:
            arms[name]["tau0.2_hours"] = round(hours(p2), 3)
            arms[name]["tau0.2_vs_0.3_loss_E10_1.0"] = loss(in_set(p2, 1.0), in_set(p, 1.0))
    if os.path.isdir(dD):
        pD, nD = load_index(dD), load_index(dD, "_noF9")
        arms["exiD_traveledDistance_defD_FAILED"] = {
            "episodes": int(len(pD)), "hours": round(hours(pD), 3), "F9_marginal_loss_hours": loss(pD, nD),
            "E10_1.0": {"episodes": int(len(in_set(pD, 1.0))), "hours": round(hours(in_set(pD, 1.0)), 3)}}
    arms["exiD_raw_leadDHW_defC"]["step_all_frames"] = step_summary(sdE, "ref_noseam")
    arms["exiD_raw_leadDHW_defC"]["step_seam_frames"] = step_summary(sdE, "ref_seam")
    arms["exiD_seam_corrected_defE"]["step_all_frames"] = step_summary(sdE, "used_noseam")
    arms["highD_defE"]["step_all_frames"] = step_summary(sdH, "used_noseam")
    flips = [d["f4_flip"] for d in sdE]
    arms["exiD_seam_corrected_defE"]["F4_flip_vs_raw"] = {k: int(sum(f[k] for f in flips)) for k in flips[0]}
    res["arm_comparison"] = arms
    res["step_note"] = ("Non-seam |step| of the seam-corrected gap equals raw leadDHW's by construction; "
                        "at seam frames it is zero by construction. Neither is evidence; E-A and E-B are. "
                        "Quantiles are upper bin edges, 1 mm resolution.")

    # ---- audit table and excitation of retained vs rejected -------------------------------------
    nE = nE.copy()
    nE["retained_frac"] = overlap_fraction(nE, pE)
    nC = nC.copy()
    nC["retained_frac"] = overlap_fraction(nC, pC)
    status = lambda f: np.where(f >= 0.999, "retained", np.where(f <= 0.001, "rejected", "partial"))
    nE["F9_status"] = status(nE["retained_frac"])
    nC["F9_status"] = status(nC["retained_frac"])
    res["excitation_retained_vs_rejected"] = {
        name: {st: {"episodes": int(len(g)), "hours": round(hours(g), 3),
                    "sd_vL_median": round(float(g["sd_vL"].median()), 3)}
               for st, g in df.groupby("F9_status")}
        for name, df in (("seam_corrected", nE), ("raw_legacy", nC))}
    cols = ["rec", "episode", "location", "dur_s", "run_K_median", "run_nseams", "run_anchor_spread",
            "run_abs_seam_sum", "max_gap_step_m", "p95_gap_step_m", "max_raw_step_m", "gap_check_m",
            "sd_vL", "retained_frac", "F9_status"]
    audit = nE[cols].copy()
    audit["F4"] = "pass"                      # candidates satisfy F4 by construction
    audit["anchor_flag"] = audit["run_anchor_spread"] > ED_SPREAD
    audit["location_excluded"] = audit["location"].isin(fails)
    for k, x in SETS.items():
        audit[f"in_{k}"] = audit.index.isin(in_set(nE, x).index)
    audit.to_csv(os.path.join(DERIVED, "exid_gap_audit_defE.csv"), index=False)

    os.makedirs(RESULTS, exist_ok=True)
    jpath = os.path.join(RESULTS, "step1c_exid_gap_validation_defE.json")
    with open(jpath, "w") as f:
        json.dump(res, f, indent=2, default=str)
    report = render_report(res)
    with open(jpath.replace(".json", ".txt"), "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    try:
        make_plots(res, pE, nE, nC, sdE, sdH, dE)
    except ImportError:
        print("matplotlib not installed - plots skipped")
    return 0


def render_report(res):
    EA, EB, EC = res["EA_leadDHW_off_seam"], res["EB_seam_reference_change"], res["EC_excitation_neutrality"]
    ED, EE, v = res["ED_anchor_spread"], res["EE_location_robustness"], res["verdict"]
    ok = lambda b: "PASS" if b else "FAIL"
    L = ["AMENDMENT E - exiD seam-corrected gap validation (thresholds frozen in E.2)", ""]
    p = EA["pooled"]
    L.append(f"E-A leadDHW off-seam: P(|eps|>0.3) {p['frac_gt0.3']:.4%} (<= {EA_EXCEED:.1%}), "
             f"p99 {p['p99_ub']} m (<= {EA_P99}) -> {ok(p['pass'])}")
    p = EB["pooled"]
    L.append(f"E-B seam frames, traveledDistance step > 0.3 m: {p['frac_gt0.3']:.4%} (<= {EB_EXCEED:.1%}); "
             f"raw leadDHW at the same frames {p['raw_leadDHW_seam']['frac_gt0.3']:.2%} -> {ok(p['pass'])}")
    L.append("E-C F9 hour loss (seam-corrected | raw legacy):")
    for k in SETS:
        L.append(f"    {k:11s} {EC['seam_corrected'][k]['loss']:.2%} | {EC['raw_legacy_defC'][k]['loss']:.2%}")
    L.append(f"  gradient {EC['gradient_E10_1.5_minus_carled']:+.2%} -> {ok(EC['pass'])}")
    L.append("Per location (E-A | E-B | E-C E10,1.0 loss):")
    for loc, e in EE["per_location"].items():
        L.append(f"    loc {loc}: {ok(e['EA'])} (p99 {EA['by_location'][loc]['p99_ub']}, "
                 f">0.3 {EA['by_location'][loc]['frac_gt0.3']:.3%}) | {ok(e['EB'])} "
                 f"({EB['by_location'][loc]['frac_gt0.3']:.3%}) | {ok(e['EC_E10_1.0'])} "
                 f"({EC['E10_1.0_loss_by_location'].get(loc)})")
    L.append(f"E-E failing locations {EE['failing_locations']}; drop replication: {EE['drop_exiD_replication']}")
    L.append(f"E-D anchor spread > {ED_SPREAD} m: {ED['flagged_share_E10_1.0_hours']:.1%} of E10,1.0 hours "
             f"(threshold {ED_SHARE:.0%}); by location {ED['flagged_share_by_location']}")
    L.append(f"    exclusion loss gradient {ED['exclusion_gradient_E10_1.5_minus_carled']:+.2%}; "
             f"replication primary set: {ED['replication_primary_set']}; "
             f"excitation-biased label: {ED['excitation_biased_label']}")
    L += ["", f"DECISION: {v['decision']}", "", "Arm comparison:"]
    for k, a in res["arm_comparison"].items():
        s = a.get("step_all_frames", {})
        L.append(f"  {k}: {a['episodes']} episodes / {a['hours']} h; F9 loss {a['F9_marginal_loss_hours']}; "
                 f"E10,1.0 {a['E10_1.0']['episodes']} / {a['E10_1.0']['hours']} h"
                 + (f"; |step| p99 {s.get('p99_ub')} max {s.get('max')} >0.3 {s.get('frac_gt0.3', 0):.4%}" if s else ""))
    L.append("  " + res["step_note"])
    return "\n".join(L) + "\n"


def make_plots(res, pE, nE, nC, sdE, sdH, dE):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = os.path.join(RESULTS, "figures")
    os.makedirs(fd, exist_ok=True)

    # 1 step CCDF
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = STEP_EDGES[1:-1]
    for lab, diags, key in (("raw leadDHW, seam frames", sdE, "ref_seam"),
                            ("raw leadDHW, non-seam frames", sdE, "ref_noseam"),
                            ("traveledDistance diff., seam frames", sdE, "td_seam"),
                            ("seam-corrected, all non-seam frames", sdE, "used_noseam"),
                            ("highD", sdH, "used_noseam")):
        h = np.sum([d["hist"][key] for d in diags], axis=0)
        if h.sum():
            ax.loglog(x, 1 - np.cumsum(h)[:-1] / h.sum(), label=lab)
    for t in (0.2, 0.3):
        ax.axvline(t, ls=":", c="grey")
    ax.set(xlabel="|dg + dv dt| (m)", ylabel="P(|step| > x)", title="Frame-level step residual (exiD E)",
           ylim=(1e-7, 1))
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(os.path.join(fd, "step1c_E_step_ccdf.png"), dpi=130); plt.close(fig)

    # 2 anchor spread
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.logspace(-3, 1, 60)
    e10 = in_set(pE, 1.0)
    ax[0].hist(e10["run_anchor_spread"].clip(1e-3), bins=bins, weights=e10["dur_s"] / 3600)
    ax[0].axvline(ED_SPREAD, ls="--", c="k"); ax[0].set_xscale("log")
    ax[0].set(xlabel="anchor spread of run (m)", ylabel="E10,1.0 hours", title="E-D: level ambiguity")
    fl = res["ED_anchor_spread"]["flagged_share_by_location"]
    ax[1].bar([f"loc {k}" for k in fl], list(fl.values()))
    ax[1].axhline(ED_SHARE, ls="--", c="k")
    ax[1].set(ylabel="flagged share of E10,1.0 hours", title="flagged by location")
    fig.tight_layout(); fig.savefig(os.path.join(fd, "step1c_E_anchor_spread.png"), dpi=130); plt.close(fig)

    # 3 F9 loss by location
    EC = res["EC_excitation_neutrality"]
    locs = sorted(EC["E10_1.0_loss_by_location"])
    raw = [EC["raw_legacy_E10_1.0_loss_by_location"].get(L) or 0 for L in locs]
    new = [EC["E10_1.0_loss_by_location"].get(L) or 0 for L in locs]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    w = 0.38
    ax.bar(np.arange(len(locs)) - w / 2, raw, w, label="raw leadDHW (def C)")
    ax.bar(np.arange(len(locs)) + w / 2, new, w, label="seam-corrected (def E)")
    ax.axhline(EC_MAX_LOSS, ls="--", c="k")
    ax.set_xticks(range(len(locs)), [f"loc {L}" for L in locs])
    ax.set(ylabel="F9 hour loss in E10,1.0", title="F9 loss by location"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(fd, "step1c_E_f9_loss_by_location.png"), dpi=130); plt.close(fig)

    # 4 excitation retained vs rejected
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    bins = np.linspace(0, 5, 51)
    for a, (name, df) in zip(ax, (("raw leadDHW (def C)", nC), ("seam-corrected (def E)", nE))):
        for lab, m in (("retained", df["retained_frac"] >= 0.999), ("rejected/partial", df["retained_frac"] < 0.999)):
            a.hist(df.loc[m, "sd_vL"], bins=bins, weights=df.loc[m, "dur_s"] / 3600, alpha=.6, label=lab)
        a.set(xlabel="leader speed SD (m/s)", title=name); a.legend()
    ax[0].set_ylabel("hours (candidates without F9)")
    fig.tight_layout(); fig.savefig(os.path.join(fd, "step1c_E_excitation_retained_vs_rejected.png"), dpi=130)
    plt.close(fig)

    # 5 example trajectories - real data, LOCAL ONLY
    flc = os.path.join(DERIVED, "figures_local")
    os.makedirs(flc, exist_ok=True)
    s = pE.sort_values("run_anchor_spread")
    pick = s.iloc[[len(s) // 2, int(len(s) * .9), len(s) - 1]]
    fig, ax = plt.subplots(3, 1, figsize=(9, 8))
    for a, (_, e) in zip(ax, pick.iterrows()):
        arr = np.load(os.path.join(dE, f"{int(e['rec']):02d}_episodes.npz"))[e["episode"]]
        t = (arr[0] - arr[0][0]) / 25
        a.plot(t, arr[8], label="leadDHW (raw)", alpha=.7)
        a.plot(t, arr[1], label="seam-corrected g")
        a.set(title=f"{e['episode']} (loc {e['location']}, anchor spread {e['run_anchor_spread']:.2f} m)",
              ylabel="gap (m)")
        a.legend(fontsize=8)
    ax[-1].set_xlabel("t (s)")
    fig.tight_layout(); fig.savefig(os.path.join(flc, "step1c_E_examples.png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
