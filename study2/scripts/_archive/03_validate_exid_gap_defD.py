"""STEP 1c - validate the reconstructed exiD gap (Amendment D). Measurement only.

No model is fitted here. This script reads the extraction outputs only:
  derived/exid_episodes_defD            reconstructed gap, F9 tau 0.3 (primary)
  derived/exid_episodes_defD_f9tau0.2   reconstructed gap, F9 tau 0.2 (sensitivity)
  derived/exid_episodes_defC            raw leadDHW (legacy arm, definition C)
  derived/exid_episodes_defC_f9tau0.2
  derived/highd_episodes_defD           highD reference
and evaluates the decision criteria A-F frozen in 05_PREREGISTRATION_DRAFT.md D.4
(thresholds copied below, not chosen here).

Outputs
  code/results/step1c_exid_gap_validation_defD.json    machine-readable, aggregates only
  code/results/step1c_exid_gap_validation_defD.txt     human-readable report
  code/results/figures/step1c_*.png                    aggregate plots (if matplotlib)
  derived/exid_gap_audit_defD.csv                      episode-level audit (LOCAL ONLY)
  derived/figures_local/step1c_examples.png            example trajectories (LOCAL ONLY)

Usage (from AINN_ReWORK/scripts):   python 03_validate_exid_gap.py
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

from _common import DERIVED, RESULTS, STEP_EDGES

# ---- D.4 thresholds, frozen 2026-09-26 before any D diagnostic was computed ----------
A_P90, A_P99 = 0.10, 0.30          # m, within-seam robust range of r
B_SPREAD, B_STOP_SHARE = 0.30, 0.50
C_SEAM_EXCEED = 0.001              # P(|eps_new| > 0.3 m | seam frame)
D_MAX_LOSS, D_MAX_GRADIENT = 0.02, 0.01
E_BIG_LOCATIONS, E_STOP_FAILS = (0, 1, 4), 2
SETS = {"carled_10s": 0.0, "E10,0.5": 0.5, "E10,1.0": 1.0, "E10,1.5": 1.5}


def folder(name):
    return os.path.join(DERIVED, name)


def load_index(d, suffix=""):
    fs = sorted(glob.glob(os.path.join(d, f"??_index{suffix}.csv")))
    fs = [f for f in fs if os.path.getsize(f) > 1]
    if not fs:
        raise FileNotFoundError(f"no index files in {d} - run the extraction first")
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


def in_set(df, x):
    return df[(df["leader_class"] == "car") & (df["dur_s"] >= 10) & (df["sd_vL"] >= x)]


def hours(df):
    return float(df["dur_s"].sum() / 3600)


def f9_losses(prim, no9):
    out = {}
    for name, x in SETS.items():
        a, b = hours(in_set(prim, x)), hours(in_set(no9, x))
        out[name] = {"hours_primary": round(a, 3), "hours_without_F9": round(b, 3),
                     "loss": round(1 - a / b, 4) if b > 0 else None}
    return out


def f9_losses_by_location(prim, no9, x=1.0):
    out = {}
    for loc, g in no9.groupby("location"):
        b = hours(in_set(g, x))
        a = hours(in_set(prim[prim["location"] == loc], x))
        out[int(loc)] = round(1 - a / b, 4) if b > 0 else None
    return out


def load_stepdiag(d):
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "??_stepdiag.json")))]
    if not rows:
        raise FileNotFoundError(f"no stepdiag files in {d}")
    return rows


def step_summary(diags, key, location=None):
    """Quantiles of |step| from the fixed-edge histograms (resolution 1 mm below 0.5 m)
    plus exact exceedance counts."""
    ds = [d for d in diags if location is None or d["location"] == location]
    h = np.sum([d["hist"][key] for d in ds], axis=0) if ds else np.zeros(len(STEP_EDGES) - 1)
    n = int(h.sum())
    if n == 0:
        return {"n": 0}
    c = np.cumsum(h) / n
    q = lambda p: float(STEP_EDGES[1:][np.searchsorted(c, p)])   # upper bin edge
    gt = {t: int(sum(d[f"n_{key}_gt{t:g}"] for d in ds)) for t in (0.2, 0.3)}
    return {"n": n, "median_ub": q(0.5), "p95_ub": q(0.95), "p99_ub": q(0.99),
            "max": float(max(d[f"max_{key}"] for d in ds)),
            "frac_gt0.2": gt[0.2] / n, "frac_gt0.3": gt[0.3] / n}


def wquantile(x, w, p):
    o = np.argsort(x)
    x, w = np.asarray(x)[o], np.asarray(w, dtype=float)[o]
    c = np.cumsum(w) / w.sum()
    return float(x[np.searchsorted(c, p)])


def overlap_fraction(cand, prim):
    """For each candidate episode (without-F9 variant), the fraction of its frames that
    lie inside primary episodes of the same recording and pair."""
    def parse(df):
        k = df["episode"].str.rsplit("_", n=3, expand=True)
        return df.assign(start=k[3].astype(int)).assign(end=lambda d: d["start"] + d["n_frames"])
    c, p = parse(cand), parse(prim)
    m = c.reset_index().merge(p[["rec", "follower", "leader", "start", "end"]],
                              on=["rec", "follower", "leader"], how="left",
                              suffixes=("", "_p"))
    ov = (np.minimum(m["end"], m["end_p"]) - np.maximum(m["start"], m["start_p"])).clip(lower=0)
    got = ov.fillna(0).groupby(m["index"]).sum()
    return (got.reindex(c.index).fillna(0) / c["n_frames"]).to_numpy()


def main():
    dD, dD2 = folder("exid_episodes_defD"), folder("exid_episodes_defD_f9tau0.2")
    dC, dC2 = folder("exid_episodes_defC"), folder("exid_episodes_defC_f9tau0.2")
    dH = folder("highd_episodes_defD")

    pD, nD = load_index(dD), load_index(dD, "_noF9")
    pC, nC = load_index(dC), load_index(dC, "_noF9")
    pH, nH = load_index(dH), load_index(dH, "_noF9")
    seg = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(dD, "??_seamseg.csv")))
                     if os.path.getsize(f) > 1], ignore_index=True)
    sdD, sdH = load_stepdiag(dD), load_stepdiag(dH)
    locs = sorted(int(x) for x in nD["location"].unique())
    res = {"amendment": "D", "thresholds": {
        "A_p90_m": A_P90, "A_p99_m": A_P99, "B_spread_m": B_SPREAD,
        "B_stop_share": B_STOP_SHARE, "C_seam_exceed": C_SEAM_EXCEED,
        "D_max_loss": D_MAX_LOSS, "D_max_gradient": D_MAX_GRADIENT,
        "E_big_locations": list(E_BIG_LOCATIONS), "E_stop_fails": E_STOP_FAILS}}

    # ---- A. within-seam agreement ------------------------------------------------------
    sA = seg[(seg["n"] >= 50) & seg["run_in_primary"]]
    def critA(s):
        if s.empty:
            return {"n_segments": 0, "pass": False}
        p90, p99 = wquantile(s["robust_range"], s["n"], .90), wquantile(s["robust_range"], s["n"], .99)
        return {"n_segments": int(len(s)), "hours": round(s["n"].sum() / 25 / 3600, 3),
                "p50_m": round(wquantile(s["robust_range"], s["n"], .5), 4),
                "p90_m": round(p90, 4), "p99_m": round(p99, 4),
                "pass": bool(p90 <= A_P90 and p99 <= A_P99)}
    A = critA(sA)
    A["by_location"] = {L: critA(sA[sA["location"] == L]) for L in locs}
    lenbins = pd.cut(sA["n"] / 25, [2, 5, 10, 20, 40, np.inf])
    A["p90_by_segment_length_s"] = {str(k): round(wquantile(g["robust_range"], g["n"], .9), 4)
                                    for k, g in sA.groupby(lenbins, observed=True)}
    res["A_within_seam_agreement"] = A

    # ---- B. anchor spread ---------------------------------------------------------------
    e10 = in_set(pD, 1.0)
    flag = e10["run_anchor_spread"] > B_SPREAD
    share = hours(e10[flag]) / hours(e10) if len(e10) else float("nan")
    B = {"episode_anchor_spread_q": pD["run_anchor_spread"].quantile([.5, .9, .99]).round(4).to_dict(),
         "episode_run_mad_q": pD["run_mad"].quantile([.5, .9, .99]).round(4).to_dict(),
         "episode_run_iqr_q": pD["run_iqr"].quantile([.5, .9, .99]).round(4).to_dict(),
         "episode_nseams_q": pD["run_nseams"].quantile([.5, .9, .99]).to_dict(),
         "flagged_share_E10_1.0_hours": round(share, 4),
         "flagged_share_by_location": {int(L): round(hours(g[g["run_anchor_spread"] > B_SPREAD]) / hours(g), 4)
                                       for L, g in e10.groupby("location")},
         "stop": bool(share > B_STOP_SHARE)}
    res["B_anchor_spread"] = B

    # ---- C. seam artefact ----------------------------------------------------------------
    C = {"used_seam": step_summary(sdD, "used_seam"), "used_noseam": step_summary(sdD, "used_noseam"),
         "raw_seam": step_summary(sdD, "ref_seam"), "raw_noseam": step_summary(sdD, "ref_noseam")}
    C["pass"] = bool(C["used_seam"].get("frac_gt0.3", 1.0) <= C_SEAM_EXCEED)
    res["C_seam_artefact"] = C

    # ---- D. excitation neutrality --------------------------------------------------------
    lossD = f9_losses(pD, nD)
    grad = lossD["E10,1.5"]["loss"] - lossD["carled_10s"]["loss"]
    D = {"reconstructed": lossD, "raw_legacy_defC": f9_losses(pC, nC), "highD": f9_losses(pH, nH),
         "gradient_E10_1.5_minus_carled": round(grad, 4),
         "pass": bool(all(v["loss"] is not None and v["loss"] <= D_MAX_LOSS for v in lossD.values())
                      and grad <= D_MAX_GRADIENT)}
    res["D_excitation_neutrality"] = D

    # ---- E. location robustness ----------------------------------------------------------
    lD = f9_losses_by_location(pD, nD)
    E = {"per_location": {}}
    for L in locs:
        a = A["by_location"][L]
        okA = bool(a.get("pass", False)) if a["n_segments"] else False
        okA90 = bool(a["n_segments"] and a["p90_m"] <= A_P90)
        okD = lD.get(L) is not None and lD[L] <= D_MAX_LOSS
        E["per_location"][L] = {"A_p90_pass": okA90, "D_E10_1.0_loss": lD.get(L),
                                "D_pass": bool(okD), "pass": bool(okA90 and okD)}
    fails = [L for L in locs if not E["per_location"][L]["pass"]]
    E["failing_locations"] = fails
    E["stop"] = bool(sum(L in fails for L in E_BIG_LOCATIONS) >= E_STOP_FAILS)
    E["raw_legacy_F9_loss_E10_1.0_by_location"] = f9_losses_by_location(pC, nC)
    res["E_location_robustness"] = E

    res["F_no_hidden_selection"] = ("procedural: thresholds frozen in 05_PREREGISTRATION_DRAFT.md "
                                    "Amendment D before this script was first run; no fitting code "
                                    "has been run on real data (DIARY Entries 18-19)")

    # ---- descriptive three-arm comparison ---------------------------------------------------
    pD2 = load_index(dD2)
    pC2 = load_index(dC2)
    arms = {}
    for name, p, n, p2 in (("exiD_raw_leadDHW_defC", pC, nC, pC2),
                           ("exiD_reconstructed_defD", pD, nD, pD2),
                           ("highD_defD", pH, nH, None)):
        arms[name] = {"episodes": int(len(p)), "hours": round(hours(p), 3),
                      "F9_marginal_loss_hours": round(1 - hours(p) / hours(n), 4),
                      "E10_1.0": {"episodes": int(len(in_set(p, 1.0))), "hours": round(hours(in_set(p, 1.0)), 3)},
                      "sd_vL_q_E10_0.0": in_set(p, 0.0)["sd_vL"].quantile([.25, .5, .75, .9]).round(3).to_dict()}
        if p2 is not None:
            arms[name]["tau0.2_hours"] = round(hours(p2), 3)
            arms[name]["tau0.2_vs_0.3_loss_E10_1.0"] = round(1 - hours(in_set(p2, 1.0)) / hours(in_set(p, 1.0)), 4)
    arms["exiD_raw_leadDHW_defC"]["step_all_frames"] = step_summary(sdD, "ref_noseam")
    arms["exiD_reconstructed_defD"]["step_all_frames"] = step_summary(sdD, "used_noseam")
    arms["highD_defD"]["step_all_frames"] = step_summary(sdH, "used_noseam")
    arms["highD_defD"]["step_dhw_all_frames"] = step_summary(sdH, "ref_noseam")
    flips = [d["f4_flip"] for d in sdD]
    arms["exiD_reconstructed_defD"]["F4_flip_vs_raw"] = {
        k: int(sum(f[k] for f in flips)) for k in flips[0]}
    res["three_arm_comparison"] = arms
    res["step_note"] = ("|step| for the reconstructed gap and for highD is small largely by "
                        "construction (shared smoothing); reported descriptively, not as evidence "
                        "of validity (D.3). Quantiles are upper bin edges, 1 mm resolution.")

    # ---- audit table + excitation of retained vs rejected ------------------------------------
    nD = nD.copy()
    nD["retained_frac"] = overlap_fraction(nD, pD)
    nD["F9_status"] = np.where(nD["retained_frac"] >= 0.999, "retained",
                               np.where(nD["retained_frac"] <= 0.001, "rejected", "partial"))
    nC = nC.copy()
    nC["retained_frac"] = overlap_fraction(nC, pC)
    ex = {}
    for name, df in (("reconstructed", nD), ("raw_legacy", nC)):
        ex[name] = {st: {"episodes": int(len(g)), "hours": round(hours(g), 3),
                         "sd_vL_median": round(float(g["sd_vL"].median()), 3) if len(g) else None}
                    for st, g in df.assign(s=np.where(df["retained_frac"] >= 0.999, "retained",
                                           np.where(df["retained_frac"] <= 0.001, "rejected", "partial"))
                                           ).groupby("s")}
    res["excitation_retained_vs_rejected"] = ex
    audit_cols = ["rec", "episode", "location", "dur_s", "run_C", "run_mad", "run_iqr",
                  "run_nseams", "run_anchor_spread", "run_within_drift", "max_gap_step_m",
                  "p95_gap_step_m", "max_raw_step_m", "gap_check_m", "sd_vL",
                  "retained_frac", "F9_status"]
    audit = nD[audit_cols].copy()
    audit["F4"] = "pass"                  # candidates satisfy F4 by construction
    for name, x in SETS.items():
        audit[f"in_{name}"] = audit.index.isin(in_set(nD, x).index)
    audit.to_csv(os.path.join(DERIVED, "exid_gap_audit_defD.csv"), index=False)

    # ---- verdict ------------------------------------------------------------------------
    stop = (not A["pass"]) or (not C["pass"]) or (not D["pass"]) or B["stop"] or E["stop"]
    res["verdict"] = {"A": A["pass"], "B_stop": B["stop"], "C": C["pass"], "D": D["pass"],
                      "E_stop": E["stop"], "E_failing_locations": fails,
                      "decision": "STOP - report and re-plan" if stop else
                      ("PASS - freeze Amendment D" + (f"; exclude locations {fails} from primary exiD set" if fails else ""))}

    os.makedirs(RESULTS, exist_ok=True)
    jpath = os.path.join(RESULTS, "step1c_exid_gap_validation_defD.json")
    with open(jpath, "w") as f:
        json.dump(res, f, indent=2, default=str)
    report = render_report(res)
    with open(jpath.replace(".json", ".txt"), "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    try:
        make_plots(res, sA, pD, nD, nC, sdD, sdH, seg, dD)
    except ImportError:
        print("matplotlib not installed - plots skipped")
    return 0


def render_report(res):
    v, A, B, C, D, E = (res["verdict"], res["A_within_seam_agreement"], res["B_anchor_spread"],
                        res["C_seam_artefact"], res["D_excitation_neutrality"],
                        res["E_location_robustness"])
    L = ["AMENDMENT D - exiD reconstructed-gap validation (thresholds frozen in D.4)", ""]
    L.append(f"A within-seam agreement: p90 {A['p90_m']} m (<= {A_P90}), p99 {A['p99_m']} m "
             f"(<= {A_P99}) over {A['n_segments']} segments -> {'PASS' if A['pass'] else 'FAIL'}")
    L.append(f"B anchor spread > {B_SPREAD} m: {B['flagged_share_E10_1.0_hours']:.1%} of E10,1.0 hours "
             f"(stop if > {B_STOP_SHARE:.0%}) -> {'STOP' if B['stop'] else 'ok'}")
    L.append(f"C seam frames with |eps| > 0.3 m: reconstructed {C['used_seam'].get('frac_gt0.3', 0):.4%} "
             f"vs raw {C['raw_seam'].get('frac_gt0.3', 0):.4%} (<= {C_SEAM_EXCEED:.1%}) -> "
             f"{'PASS' if C['pass'] else 'FAIL'}")
    L.append("D F9 hour loss (reconstructed | raw legacy):")
    for k in SETS:
        L.append(f"    {k:11s} {D['reconstructed'][k]['loss']:.2%} | {D['raw_legacy_defC'][k]['loss']:.2%}")
    L.append(f"  gradient {D['gradient_E10_1.5_minus_carled']:+.2%} -> {'PASS' if D['pass'] else 'FAIL'}")
    L.append("E per location (A p90 pass, E10,1.0 F9 loss):")
    for loc, e in E["per_location"].items():
        L.append(f"    loc {loc}: A {'ok' if e['A_p90_pass'] else 'FAIL'}, loss {e['D_E10_1.0_loss']}, "
                 f"{'pass' if e['pass'] else 'FAIL'}")
    L.append(f"  failing {E['failing_locations']} -> {'STOP' if E['stop'] else 'ok'}")
    L += ["", f"DECISION: {v['decision']}", "", "Three-arm comparison:"]
    for k, a in res["three_arm_comparison"].items():
        s = a["step_all_frames"]
        L.append(f"  {k}: {a['episodes']} episodes / {a['hours']} h; F9 loss {a['F9_marginal_loss_hours']:.2%}; "
                 f"E10,1.0 {a['E10_1.0']['episodes']} / {a['E10_1.0']['hours']} h; |step| median "
                 f"{s.get('median_ub')} p95 {s.get('p95_ub')} p99 {s.get('p99_ub')} max {s.get('max'):.3f} "
                 f">0.3 {s.get('frac_gt0.3', 0):.4%}")
    L.append("  " + res["step_note"])
    return "\n".join(L) + "\n"


def make_plots(res, sA, pD, nD, nC, sdD, sdH, seg, dD):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = os.path.join(RESULTS, "figures")
    os.makedirs(fd, exist_ok=True)

    # 1 offset stability
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.logspace(-3, 1, 60)
    ax[0].hist(sA["robust_range"].clip(1e-3), bins=bins, weights=sA["n"] / 25 / 3600)
    ax[0].axvline(A_P90, ls="--", c="k"); ax[0].set_xscale("log")
    ax[0].set(xlabel="within-seam robust range of r (m)", ylabel="hours", title="A: within-seam drift")
    ax[1].hist(pD["run_anchor_spread"].clip(1e-3), bins=bins)
    ax[1].axvline(B_SPREAD, ls="--", c="k"); ax[1].set_xscale("log")
    ax[1].set(xlabel="anchor spread of run (m)", ylabel="primary episodes", title="B: anchor spread")
    fig.tight_layout(); fig.savefig(os.path.join(fd, "step1c_offset_stability.png"), dpi=130); plt.close(fig)

    # 2 dynamic consistency: CCDF of |step|
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = STEP_EDGES[1:-1]
    for lab, diags, key in (("exiD raw leadDHW, seam", sdD, "ref_seam"),
                            ("exiD raw leadDHW, non-seam", sdD, "ref_noseam"),
                            ("exiD reconstructed, seam", sdD, "used_seam"),
                            ("exiD reconstructed, non-seam", sdD, "used_noseam"),
                            ("highD", sdH, "used_noseam")):
        h = np.sum([d["hist"][key] for d in diags], axis=0)
        if h.sum():
            ax.loglog(x, 1 - np.cumsum(h)[:-1] / h.sum(), label=lab)
    for t in (0.2, 0.3):
        ax.axvline(t, ls=":", c="grey")
    ax.set(xlabel="|dg + dv dt| (m)", ylabel="P(|step| > x)", title="Frame-level step residual", ylim=(1e-7, 1))
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(fd, "step1c_step_ccdf.png"), dpi=130); plt.close(fig)

    # 3 F9 loss by location, raw vs reconstructed
    E = res["E_location_robustness"]
    locs = list(E["per_location"].keys())
    raw = [E["raw_legacy_F9_loss_E10_1.0_by_location"].get(L) or 0 for L in locs]
    new = [E["per_location"][L]["D_E10_1.0_loss"] or 0 for L in locs]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    w = 0.38
    ax.bar(np.arange(len(locs)) - w / 2, raw, w, label="raw leadDHW (def C)")
    ax.bar(np.arange(len(locs)) + w / 2, new, w, label="reconstructed (def D)")
    ax.axhline(D_MAX_LOSS, ls="--", c="k")
    ax.set_xticks(range(len(locs)), [f"loc {L}" for L in locs])
    ax.set(ylabel="F9 hour loss in E10,1.0", title="F9 loss by location"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(fd, "step1c_f9_loss_by_location.png"), dpi=130); plt.close(fig)

    # 4 excitation retained vs rejected
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    bins = np.linspace(0, 5, 51)
    for a, (name, df) in zip(ax, (("raw leadDHW (def C)", nC), ("reconstructed (def D)", nD))):
        for lab, m in (("retained", df["retained_frac"] >= 0.999), ("rejected/partial", df["retained_frac"] < 0.999)):
            a.hist(df.loc[m, "sd_vL"], bins=bins, weights=df.loc[m, "dur_s"] / 3600, alpha=.6, label=lab)
        a.set(xlabel="leader speed SD (m/s)", title=name); a.legend()
    ax[0].set_ylabel("hours (candidates without F9)")
    fig.tight_layout(); fig.savefig(os.path.join(fd, "step1c_excitation_retained_vs_rejected.png"), dpi=130)
    plt.close(fig)

    # 5 example trajectories - real data, LOCAL ONLY
    fl = os.path.join(DERIVED, "figures_local")
    os.makedirs(fl, exist_ok=True)
    pick = pD.sort_values("run_anchor_spread").iloc[[len(pD) // 2, int(len(pD) * .9), len(pD) - 1]]
    fig, ax = plt.subplots(3, 1, figsize=(9, 8))
    for a, (_, e) in zip(ax, pick.iterrows()):
        arr = np.load(os.path.join(dD, f"{int(e['rec']):02d}_episodes.npz"))[e["episode"]]
        t = (arr[0] - arr[0][0]) / 25
        a.plot(t, arr[1], label="reconstructed g")
        a.plot(t, arr[8], label="leadDHW", alpha=.7)
        a.set(title=f"{e['episode']} (loc {e['location']}, anchor spread {e['run_anchor_spread']:.2f} m)",
              ylabel="gap (m)")
        a.legend(fontsize=8)
    ax[-1].set_xlabel("t (s)")
    fig.tight_layout(); fig.savefig(os.path.join(fl, "step1c_examples.png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
