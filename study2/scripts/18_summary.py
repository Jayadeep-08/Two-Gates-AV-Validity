"""Step 7c - the decomposition that explains the Step 7 result, and the paper's tables (K.12).

Step 7 and 7b together say something that needs one more number to be stated honestly:

  * the LEVEL of the verdict moves a lot across the confidence region (W_epi up to 0.0760 at
    the 3.0 s threshold, 28% of the value itself);
  * the CONTRAST between two ego policies is resolved at every theta and never changes sign.

The obvious explanation is that calibration uncertainty is largely COMMON MODE: moving theta
moves both policies' rates together, because both ego policies drive through the same
background traffic. Obvious is not measured, so this script measures it: for each threshold it
reports the range of the baseline level, the range of the aggressive level, and the range of
their difference, all from the same runs. If the difference's range is much smaller than
either level's range, the common-mode account is right and the paper can make it.

It also takes the ego's minimum-TTC distribution at theta_hat and at the two extremes of the
3.0 s search, which is what shows WHERE in the TTC distribution theta actually acts. (The
first explanation I reached for - that sensitivity follows the density of runs near the
threshold - is wrong: the 1.0-1.5 s and 2.5-3.0 s bands carry the same density and differ
sixfold in width.)

No new theta, no new seeds: the same 13 points and the same phi/crn base as stage 8 and
17_phi_grid.py, plus two TTC snapshots. About two minutes.

Run after 17_phi_grid.py:  python 18_summary.py
"""
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("v7", os.path.join(HERE, "16_verdict.py"))
V = importlib.util.module_from_spec(spec)
sys.argv = [sys.argv[0]]
spec.loader.exec_module(V)
vd = V.vd
KEYS = [f"c{t}" for t in vd.TTC_GRID] + ["crash"]


def levels(S, cr, tt):
    d = {f"c{t}": float(np.mean((tt < t) | (cr > 0))) for t in vd.TTC_GRID}
    d["crash"] = float(cr.mean())
    return d


def main():
    res = os.path.join(V.OUT, "step7.json")
    out = json.load(open(res))
    gridf = os.path.join(V.OUT, "step7_phigrid.json")
    if not os.path.exists(gridf):
        sys.exit("STOP - run 17_phi_grid.py first")
    grid = json.load(open(gridf))["grid"]
    fit = json.load(open(V.STEP6))["fits"]["loc1"]
    gen = json.load(open(V.GEN))
    th = np.array(fit["theta_hat"], float)
    S = V.Suite(th, V.N_INST, gen)
    reg = V.Region(fit, "primary_robust")

    pts = {"theta_hat": th}
    for nm, e in out["final"]["primary_robust"].items():
        if reg.contains(e["theta"]):
            pts[nm] = np.array(e["theta"], float)

    lv, ttc = {}, {}
    for nm, t2 in pts.items():
        d = {}
        for pol, tau in (("baseline", V.TAU_BASE), ("aggressive", V.TAU_AGGR)):
            cr, tt, _, _ = S.run(t2, V.N_FINAL, V.sim_base("phi/crn"), tau=tau)
            d[pol] = levels(S, cr, tt)
            if pol == "baseline" and nm in ("theta_hat", "c3.0/min", "c3.0/max"):
                ttc[nm] = tt.ravel().copy()
        lv[nm] = d
    print(f"[1] {len(pts)} theta points evaluated for both policies", flush=True)

    # the 2.0 s contrast must match 17_phi_grid, which already matches stage 8
    chk = lv["theta_hat"]["baseline"]["c2.0"] - lv["theta_hat"]["aggressive"]["c2.0"]
    if abs(chk - grid["c2.0"]["delta_theta_hat"]) > 1e-9:
        sys.exit(f"STOP - contrast at theta_hat does not reproduce 17_phi_grid "
                 f"({chk} vs {grid['c2.0']['delta_theta_hat']}); discard this output")
    print("[2] reproduces 17_phi_grid at 2.0 s", flush=True)

    dec = {}
    for k in KEYS:
        b = [lv[nm]["baseline"][k] for nm in pts]
        a = [lv[nm]["aggressive"][k] for nm in pts]
        d = [lv[nm]["baseline"][k] - lv[nm]["aggressive"][k] for nm in pts]
        rb, ra, rd = max(b) - min(b), max(a) - min(a), max(d) - min(d)
        dec[k] = {"range_baseline": rb, "range_aggressive": ra, "range_contrast": rd,
                  "common_mode_factor": (rb / rd) if rd > 0 else None,
                  "level_baseline_at_theta_hat": lv["theta_hat"]["baseline"][k],
                  "delta_at_theta_hat": d[list(pts).index("theta_hat")],
                  "W_epi_search": out.get("interval", {}).get("primary_robust", {})
                                     .get(k, {}).get("width")}
        print(f"[3] {k}: level range {rb:.4f} (baseline) / {ra:.4f} (aggressive), "
              f"contrast range {rd:.4f}, common-mode factor "
              f"{dec[k]['common_mode_factor'] if rd > 0 else float('nan'):.1f}", flush=True)

    cdf = {}
    gridpts = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]
    for nm, arr in ttc.items():
        cdf[nm] = {str(t): float(np.mean(arr < t)) for t in gridpts}

    V.save({"decomposition": dec, "ttc_cdf": cdf, "n_theta_points": len(pts)},
           os.path.join(V.OUT, "step7_summary.json"))

    L = ["# Step 7c - why the level moves and the decision does not (K.12)", "",
         "All columns from the same runs: the 13 region theta, both ego policies, one seed "
         "base. The 2.0 s contrast reproduces stage 8.", "",
         "| outcome | V(theta_hat) | range of the LEVEL over the region | range of the "
         "CONTRAST | common-mode factor | Phi | seed-only null |",
         "|---|---|---|---|---|---|---|"]
    for k in KEYS:
        d, g = dec[k], grid[k]
        L.append(f"| {k} | {d['level_baseline_at_theta_hat']:.4f} | {d['range_baseline']:.4f} "
                 f"| {d['range_contrast']:.4f} | "
                 f"{('%.1fx' % d['common_mode_factor']) if d['common_mode_factor'] else 'n/a'} "
                 f"| {g['n_flipped']}/{S.M} | {g['n_flipped_null']}/{S.M} |")
    L += ["", "**The common-mode factor is the whole result.** It is the range of the verdict "
          "level across the confidence region divided by the range of the policy contrast "
          "across the same region. A factor well above 1 means calibration uncertainty moves "
          "both policies together, so it damages an absolute safety claim far more than a "
          "comparative one.", "",
          "## Where theta acts in the TTC distribution", "",
          "| TTC (s) | fraction of runs below, at theta_hat | at the 3.0 s minimum | "
          "at the 3.0 s maximum |", "|---|---|---|---|"]
    for t in gridpts:
        row = [cdf.get(n, {}).get(str(t)) for n in ("theta_hat", "c3.0/min", "c3.0/max")]
        L.append(f"| {t} | " + " | ".join(f"{v:.4f}" if v is not None else "n/a"
                                          for v in row) + " |")
    L += ["", "The gap between the last two columns at each row is how far calibration "
          "uncertainty slides the TTC distribution there. Sensitivity is NOT simply the "
          "density of runs near a threshold: the 1.0-1.5 s and 2.5-3.0 s bands carry the same "
          "density and their epistemic widths differ sixfold."]
    open(os.path.join(V.OUT, "step7c_summary.md"), "w", encoding="utf-8").write(
        "\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
