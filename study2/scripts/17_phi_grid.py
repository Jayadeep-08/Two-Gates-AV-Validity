"""Step 7b - the policy contrast across the DECLARED THRESHOLD GRID (Amendment K.11).

Stage 8 of 16_verdict.py computes the verdict-flip statistic and the policy contrast at the
2.0 s threshold only. That was an implementation choice and it was never pre-registered: K.5
declares the outcome as a CURVE over the whole grid (1.0, 1.5, 2.0, 2.5, 3.0 s) precisely so
that no single threshold is tuned. The run showed why that matters - the epistemic width is
0.0069 at 2.0 s and 0.0760 at 3.0 s, an eleven-fold difference - so a flip statistic computed
at one threshold says nothing about the others.

This script recomputes Phi, its seed-only null and the aggregate policy contrast at EVERY
threshold on the declared grid, and at the collision rate. It reports all of them. It adds no
new theta, no new seeds and no new arm: the same 13 theta points, the same policies, the same
seed bases as stage 8, so the 2.0 s column MUST reproduce the main run exactly - that
reproduction is asserted, and a mismatch means this script is wired wrongly and its numbers
are to be discarded.

It also records the distribution of the ego's minimum TTC at theta_hat, which is what makes
the epistemic width threshold-dependent in the first place.

Run after 16_verdict.py:  python 17_phi_grid.py        (about 2 minutes)
"""
import importlib.util
import json
import os
import sys

import numpy as np
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("v7", os.path.join(HERE, "16_verdict.py"))
V = importlib.util.module_from_spec(spec)
sys.argv = [sys.argv[0]]                 # 16_verdict parses argv at main(); never call it
spec.loader.exec_module(V)
vd = V.vd


def per_instance_all(S, cr, tt):
    """Per-instance rates for every declared threshold, plus the collision rate."""
    out = {f"c{t}": S.per_instance(cr, tt, t) for t in vd.TTC_GRID}
    out["crash"] = cr.mean(axis=1)
    return out


def main():
    res = os.path.join(V.OUT, "step7.json")
    if not os.path.exists(res):
        sys.exit(f"STOP - {res} not found; run 16_verdict.py first")
    out = json.load(open(res))
    if "phi" not in out:
        sys.exit("STOP - step7.json has no phi block, so the main run did not reach stage 8")
    fit = json.load(open(V.STEP6))["fits"]["loc1"]
    gen = json.load(open(V.GEN))
    th = np.array(fit["theta_hat"], float)
    S = V.Suite(th, V.N_INST, gen)
    reg = V.Region(fit, "primary_robust")

    # the SAME theta points stage 8 used, selected the same way
    pts = {"theta_hat": th}
    for nm, e in out["final"]["primary_robust"].items():
        if reg.contains(e["theta"]):
            pts[nm] = np.array(e["theta"], float)
    if len(pts) != out["phi"]["n_theta_points"]:
        sys.exit(f"STOP - rebuilt {len(pts)} theta points against stage 8's "
                 f"{out['phi']['n_theta_points']}; the region or the results file has changed")
    print(f"[0] {len(pts)} theta points, {S.M} instances, {V.N_FINAL} seeds", flush=True)

    per, per0 = {}, {}
    for nm, t2 in pts.items():
        d = {}
        for pol, tau in (("baseline", V.TAU_BASE), ("aggressive", V.TAU_AGGR)):
            cr, tt, _, _ = S.run(t2, V.N_FINAL, V.sim_base("phi/crn"), tau=tau)
            d[pol] = per_instance_all(S, cr, tt)
            if nm == "theta_hat" and pol == "baseline":
                ttc_hat = tt.ravel().copy()
        per[nm] = d
    print("[1] region points evaluated", flush=True)
    for i in range(len(pts)):
        d0 = {}
        for pol, tau in (("baseline", V.TAU_BASE), ("aggressive", V.TAU_AGGR)):
            cr, tt, _, _ = S.run(th, V.N_FINAL, V.sim_base(f"phinull/crn/{i}"), tau=tau)
            d0[pol] = per_instance_all(S, cr, tt)
        per0[f"null{i}"] = d0
    print("[2] seed-only null evaluated", flush=True)

    rng = np.random.default_rng(V.seed("phi/boot"))
    bidx = rng.integers(0, S.M, (V.B_INST, S.M))
    keys = [f"c{t}" for t in vd.TTC_GRID] + ["crash"]
    grid = {}
    for k in keys:
        ap = np.zeros(S.M, dtype=bool); an = np.zeros(S.M, dtype=bool)
        for nm in pts:
            s_ = np.sign(per[nm]["baseline"][k] - per[nm]["aggressive"][k])
            ap |= s_ > 0; an |= s_ < 0
        flips = int((ap & an).sum())
        np_ = np.zeros(S.M, dtype=bool); nn_ = np.zeros(S.M, dtype=bool)
        for d0 in per0.values():
            s_ = np.sign(d0["baseline"][k] - d0["aggressive"][k])
            np_ |= s_ > 0; nn_ |= s_ < 0
        fnull = int((np_ & nn_).sum())
        delta = {}
        for nm in pts:
            dd = per[nm]["baseline"][k] - per[nm]["aggressive"][k]
            bs = dd[bidx].mean(1)
            q = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
            delta[nm] = {"delta": float(dd.mean()), "ci": q,
                         "excludes_zero": bool(q[0] > 0 or q[1] < 0),
                         "sign": int(np.sign(dd.mean()))}
        s0 = delta["theta_hat"]
        resolved = sorted(n for n, d in delta.items() if d["excludes_zero"])
        rev = [n for n, d in delta.items()
               if n != "theta_hat" and d["sign"] == -s0["sign"] and d["sign"] != 0]
        gaps = [abs(d["delta"]) for d in delta.values()]
        grid[k] = {"n_flipped": flips, "phi": flips / S.M,
                   "ci": list(binomtest(flips, S.M).proportion_ci(0.95)),
                   "n_flipped_null": fnull, "phi_null": fnull / S.M,
                   "ci_null": list(binomtest(fnull, S.M).proportion_ci(0.95)),
                   "delta_theta_hat": s0["delta"], "ci_theta_hat": s0["ci"],
                   "n_resolved": len(resolved), "n_points": len(delta),
                   "resolved_at_theta_hat": s0["excludes_zero"],
                   "sign_reversed_at": rev,
                   "policy_gap_min": min(gaps), "policy_gap_max": max(gaps),
                   "W_epi": (out.get("interval", {}).get("primary_robust", {})
                             .get(k, {}).get("width")),
                   "contrast": delta}
        w = grid[k]["W_epi"]
        grid[k]["gap_over_W_epi"] = (min(gaps) / w) if w else None
        print(f"[3] {k}: Phi {flips}/{S.M} (null {fnull}/{S.M}), contrast at theta_hat "
              f"{s0['delta']:+.4f}, resolved at {len(resolved)}/{len(delta)} points", flush=True)

    # the 2.0 s column must reproduce the main run exactly
    ref, got = out["phi"], grid["c2.0"]
    bad = []
    for a, b in (("n_flipped", "n_flipped"), ("n_flipped_null", "n_flipped_null")):
        if ref.get(a) != got[b]:
            bad.append(f"{a}: main run {ref.get(a)} vs here {got[b]}")
    if abs(ref["contrast"]["theta_hat"]["delta"] - got["delta_theta_hat"]) > 1e-12:
        bad.append(f"contrast at theta_hat: {ref['contrast']['theta_hat']['delta']} vs "
                   f"{got['delta_theta_hat']}")
    if bad:
        sys.exit("STOP - the 2.0 s column does not reproduce stage 8, so this script is wired "
                 "wrongly and its numbers must be discarded: " + "; ".join(bad))
    print("[4] 2.0 s column reproduces stage 8 exactly", flush=True)

    q = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90]
    ttc = {"quantiles": {str(p): float(np.quantile(ttc_hat, p)) for p in q},
           "fraction_below": {str(t): float(np.mean(ttc_hat < t)) for t in vd.TTC_GRID},
           "note": "ego minimum TTC at theta_hat over all runs; the density near a threshold "
                   "is what makes the epistemic width threshold-dependent"}
    res2 = {"grid": grid, "ttc_at_theta_hat": ttc, "n_instances": S.M,
            "n_seeds": V.N_FINAL, "n_theta_points": len(pts)}
    V.save(res2, os.path.join(V.OUT, "step7_phigrid.json"))

    L = ["# Step 7b - the policy contrast across the declared threshold grid (K.11)", "",
         f"The same {len(pts)} theta points, policies and seed bases as stage 8; only the "
         f"threshold varies. The 2.0 s column reproduces the main run exactly, which is "
         f"asserted in the script.", "",
         "| threshold | W_epi | Phi | seed-only null | contrast at theta_hat | resolved |"
         " smallest gap | smallest gap / W_epi |", "|---|---|---|---|---|---|---|---|"]
    for k in keys:
        g = grid[k]
        L.append(f"| {k} | {('%.4f' % g['W_epi']) if g['W_epi'] else 'n/a'} "
                 f"| {g['n_flipped']}/{S.M} | {g['n_flipped_null']}/{S.M} "
                 f"| {g['delta_theta_hat']:+.4f} [{g['ci_theta_hat'][0]:+.4f}, "
                 f"{g['ci_theta_hat'][1]:+.4f}] "
                 f"| {g['n_resolved']}/{g['n_points']} | {g['policy_gap_min']:.4f} "
                 f"| {('%.1f' % g['gap_over_W_epi']) if g['gap_over_W_epi'] else 'n/a'} |")
    L += ["", "**How to read the last column.** The policy comparison can only be overturned "
          "by calibration uncertainty where the gap between the two policies is comparable "
          "with the width the region induces. A ratio far above 1 means the comparison is "
          "safe from calibration uncertainty at that threshold; a ratio near or below 1 means "
          "it is not.", "",
          "## Ego minimum TTC at theta_hat", "", "| quantile | TTC (s) |", "|---|---|"]
    for p in q:
        L.append(f"| {int(p*100)}% | {ttc['quantiles'][str(p)]:.2f} |")
    L += ["", "| threshold | fraction of runs below it |", "|---|---|"]
    for t in vd.TTC_GRID:
        L.append(f"| {t} | {ttc['fraction_below'][str(t)]:.4f} |")
    open(os.path.join(V.OUT, "step7b_phigrid.md"), "w", encoding="utf-8").write(
        "\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
