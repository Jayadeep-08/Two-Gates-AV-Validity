"""STEP 6 - the real B-lite calibration and its confidence region (Amendment J.12).

Stages (each checkpointed; rerun the same command to resume):
  0  build the real episode sets -> derived/step6_sets.npz (LOCAL ONLY: real trajectories)
       loc1     all location-1 E10,1.0 episodes           (primary, 5,691)
       loc1_ff  location-1 with min v_F >= 15 m/s         (regime-matched control, 3,912)
       loc5     all location-5 E10,1.0 episodes           (replication site, 518)
  1  primary fit on loc1: B-lite, per = {T}, generic + 2 random starts (the S4-validated
     procedure), then a driver bootstrap of B = 199 warm-started refits in parallel workers
  2  region: leave-one-out threshold q (J.8), then the three J.12 factors
       c = 5.254 primary (robust across the S4 arms), 1.546 (fitted generator), 1 (plain)
  3  replication: the same on loc5 and loc1_ff
  4  contrast: option A's pooled theta_KL on loc1
  5  report -> code/results/step6/{fit_<set>.json, step6.json, step6.md} (aggregates only)

Usage (from AINN_ReWORK/scripts):
  python 15_fit_real.py                  # everything (resumable)
  python 15_fit_real.py --workers 8      # default: cores // 4 workers, cores // workers threads
  python 15_fit_real.py --only loc1      # one set only
  python 15_fit_real.py --summary        # rebuild the report from saved results
Keep the laptop plugged in with sleep off. Expect roughly 3-5 h in total.
"""

import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import hashlib
import json
import math
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
import multiprocessing

import numpy as np
import pandas as pd

from _common import ARRAY_ROWS, DERIVED, RESULTS, SRC  # noqa: F401

import bridge as br                                     # noqa: E402

OUT = os.path.join(RESULTS, "step6")
SETS_FILE = os.path.join(DERIVED, "step6_sets.npz")
GEN_FILE = os.path.join(RESULTS, "step5_generator", "generator.json")
PROBE_FILE = os.path.join(RESULTS, "step5_probe", "probe.json")
RIDX = {k: i for i, k in enumerate(ARRAY_ROWS)}
B_BOOT, LEVEL = 199, 0.95
C_PRIMARY, C_GENERATOR, C_PLAIN = 5.254, 1.546, 1.0     # J.12
SETS = ("loc1", "loc1_ff", "loc5")
EXPECT = {"loc1": 5691, "loc1_ff": 3912, "loc5": 518}
PN = list(br.PARAM_NAMES)


def seed(label):
    return zlib.crc32(label.encode("utf-8"))


# ---------------------------------------------------------------------------- data
def build_sets():
    """Real episodes for the three sets. LICENCE: written under derived/, never shared."""
    d = os.path.join(DERIVED, "highd_episodes_defE")
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "??_index.csv"))):
        if os.path.getsize(f) <= 1:
            continue
        ix = pd.read_csv(f)
        ix = ix[(ix["leader_class"] == "car") & (ix["dur_s"] >= 10) & (ix["sd_vL"] >= 1.0)
                & ix["location"].isin([1, 5])]
        for _, e in ix.iterrows():
            rows.append((f, e["episode"], int(e["location"]), float(e["min_vF"])))
    ul, vl, gap, v0, lens, meta, cache = [], [], [], [], [], [], {}
    for f, ep, loc, minvf in rows:
        if f not in cache:
            cache = {f: np.load(f.replace("_index.csv", "_episodes.npz"))}
        a = cache[f][ep]
        ul.append(a[RIDX["u_rL"]]); vl.append(a[RIDX["vL"]]); gap.append(a[RIDX["gap"]])
        v0.append(a[RIDX["vF"], 0]); lens.append(a.shape[1]); meta.append((loc, minvf))
    off = np.r_[0, np.cumsum(lens)].astype(np.int64)
    meta = np.array(meta)
    idx = {"loc1": np.flatnonzero(meta[:, 0] == 1),
           "loc1_ff": np.flatnonzero((meta[:, 0] == 1) & (meta[:, 1] >= 15.0)),
           "loc5": np.flatnonzero(meta[:, 0] == 5)}
    np.savez(SETS_FILE, ul=np.concatenate(ul), vl=np.concatenate(vl), off=off,
             gap=np.concatenate(gap), v0=np.array(v0),
             **{f"idx_{k}": v for k, v in idx.items()})
    return {k: len(v) for k, v in idx.items()}


def load_set(name):
    z = np.load(SETS_FILE)
    idx = z[f"idx_{name}"]
    off0 = z["off"]
    lens = off0[idx + 1] - off0[idx]
    off = np.r_[0, np.cumsum(lens)].astype(np.int64)
    sel = np.concatenate([np.arange(off0[i], off0[i + 1]) for i in idx])
    return br.BliteSample(z["ul"][sel], z["vl"][sel], off, z["v0"][idx], z["gap"][sel], per=(1,))


# ---------------------------------------------------------------------------- worker
_S = None


def _init_worker(name, threads):
    global _S
    import numba
    numba.set_num_threads(threads)
    _S = load_set(name)


def boot_one(name, b, theta_hat):
    """One bootstrap refit: resample episodes, warm-started 'cheap' fit (the S4 procedure)."""
    t0 = time.time()
    rng = np.random.default_rng(seed(f"real/{name}/boot/{b}"))
    sb = _S.subset(rng.integers(0, _S.M, _S.M))
    fb = br.fit_blite(sb, np.random.default_rng(seed(f"real/{name}/boot/{b}/starts")),
                      n_random=0, starts=(np.asarray(theta_hat, float),))
    return {"b": b, "theta_typ": fb["theta_typ"].tolist(), "L": fb["L"],
            "shared_at_bound": fb["shared_at_bound"], "seconds": time.time() - t0}


# ---------------------------------------------------------------------------- fitting
def read_boot(name):
    fn = os.path.join(OUT, f"boot_{name}.jsonl")
    recs = {}
    if os.path.exists(fn):
        for line in open(fn, encoding="utf-8"):
            try:
                r = json.loads(line)
                recs[r["b"]] = r
            except (json.JSONDecodeError, KeyError):
                continue
    return recs


def append_boot(name, rec):
    fn = os.path.join(OUT, f"boot_{name}.jsonl")
    lead = ""
    if os.path.exists(fn) and os.path.getsize(fn) > 0:
        with open(fn, "rb") as fh:
            fh.seek(-1, os.SEEK_END)
            lead = "" if fh.read(1) == b"\n" else "\n"
    with open(fn, "a", encoding="utf-8") as fh:
        fh.write(lead + json.dumps(rec, default=float) + "\n")


def fit_set(name, workers, threads):
    """Stages 1-3 for one set. Returns the aggregate record (also cached on disk)."""
    res_file = os.path.join(OUT, f"fit_{name}.json")
    ptfile = os.path.join(OUT, f"point_{name}.json")
    if os.path.exists(res_file):
        return json.load(open(res_file))
    if os.path.exists(ptfile):
        pt = json.load(open(ptfile))
        S = load_set(name)
        if (S.M, S.N) != (pt["M"], pt["N"]):
            sys.exit(f"STOP - {name}: the episode set changed since the saved fit "
                     f"({S.M}/{S.N} now, {pt['M']}/{pt['N']} then)")
        del S
    else:
        S = load_set(name)
        if S.M != EXPECT[name]:
            sys.exit(f"STOP - set {name} has {S.M} episodes, {EXPECT[name]} expected")
        t0 = time.time()
        f = br.fit_blite(S, np.random.default_rng(seed(f"real/{name}/fit")), n_random=2)
        r = br.blite_residuals(S, f)
        ep_rms = f["ep_rms"]
        T = f["P"][:, 0]
        pt = {"set": name, "M": int(S.M), "N": int(S.N),
              "theta_hat": f["theta_typ"].tolist(), "L": f["L"], "rms": f["rms"],
              "shared_idx": f["shared_idx"], "shared_at_bound": f["shared_at_bound"],
              "T_at_bound": f["per_at_bound"]["T"],
              "T_q10_25_50_75_90": np.quantile(T, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(),
              "T_log_mean": float(np.mean(np.log(T))), "T_log_sd": float(np.std(np.log(T), ddof=1)),
              "ep_rms_median": float(np.median(ep_rms)), "ep_rms_p90": float(np.quantile(ep_rms, 0.9)),
              "resid_rms": float(np.sqrt(np.mean(r ** 2))), "opt_message": f["opt_message"],
              "seconds": time.time() - t0}
        json.dump(pt, open(ptfile, "w"), indent=2, default=float)
        np.savez(os.path.join(DERIVED, f"step6_perdriver_{name}.npz"), T=T)   # LOCAL ONLY
        print(f"[1:{name}] fit {pt['seconds'] / 60:.1f} min  theta_hat "
              f"{np.round(pt['theta_hat'], 4).tolist()}  rms {pt['rms']:.3f} m  "
              f"shared at bound {pt['shared_at_bound']}", flush=True)
        del S

    recs = read_boot(name)
    todo = [b for b in range(B_BOOT) if b not in recs]
    if todo:
        print(f"[2:{name}] {len(todo)} of {B_BOOT} bootstrap refits on {workers} x {threads} threads",
              flush=True)
        t0, n, failed = time.time(), 0, []
        try:
            with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                     initargs=(name, threads),
                                     mp_context=multiprocessing.get_context("spawn")) as ex:
                futs = {ex.submit(boot_one, name, b, pt["theta_hat"]): b for b in todo}
                for fu in as_completed(futs):
                    try:
                        rec = fu.result()
                    except BrokenProcessPool:
                        raise
                    except Exception as e:
                        failed.append({"b": futs[fu], "error": repr(e)})
                        print(f"[2:{name}] FAILED b={futs[fu]}: {e!r}", flush=True)
                        continue
                    append_boot(name, rec)
                    n += 1
                    if n % 20 == 0 or n == len(todo):
                        el = time.time() - t0
                        print(f"[2:{name}] {n}/{len(todo)}, {el / 60:.0f} min, about "
                              f"{el / n * (len(todo) - n) / 60:.0f} min left", flush=True)
        except BrokenProcessPool as e:
            sys.exit(f"worker pool broke ({e!r}); finished refits are saved - rerun to resume")
        if failed:
            json.dump(failed, open(os.path.join(OUT, f"failed_{name}.json"), "w"), indent=2)
            sys.exit(f"STOP - {len(failed)} bootstrap refit(s) failed for {name}; the rest are saved")
        recs = read_boot(name)
    if len(recs) != B_BOOT:
        sys.exit(f"STOP - {name}: {len(recs)} of {B_BOOT} bootstrap refits")

    th_b = np.array([recs[b]["theta_typ"] for b in sorted(recs)])
    th = np.array(pt["theta_hat"], float)
    reg = br.ellipsoid_region(th, th_b, LEVEL)
    S_cov = np.asarray(reg["cov"], float)
    sd_b = th_b.std(0, ddof=1)
    rank = int(np.linalg.matrix_rank(S_cov))
    cond = float(np.linalg.cond(S_cov))
    out = dict(pt)
    out.update({
        "B": B_BOOT, "boot_sd": sd_b.tolist(), "q_loo": reg["q"], "q_in_sample": reg["q_emp"],
        "flat_directions": reg["flat_names"], "cov_rank": rank, "cov_cond": cond,
        "boot_shared_at_bound_frac": float(np.mean([bool(recs[b]["shared_at_bound"]) for b in recs])),
        "pct_ci_95": np.quantile(th_b, [0.025, 0.975], axis=0).tolist(),
        "cov": S_cov.tolist(),
        "regions": {}})
    for tag, c in (("primary_robust", C_PRIMARY), ("generator_only", C_GENERATOR), ("plain", C_PLAIN)):
        half = np.sqrt(c * reg["q"] * np.clip(np.diag(S_cov), 0, None))
        lo = np.maximum(th - half, br.LO)
        hi = np.minimum(th + half, br.HI)
        out["regions"][tag] = {"c": c, "radius_factor": math.sqrt(c),
                               "marginal_half_width": half.tolist(),
                               "box_lo": lo.tolist(), "box_hi": hi.tolist(),
                               "at_parameter_bound": [PN[j] for j in range(br.K)
                                                      if th[j] - half[j] < br.LO[j]
                                                      or th[j] + half[j] > br.HI[j]]}
    json.dump(out, open(res_file, "w"), indent=2, default=float)
    print(f"[3:{name}] region done; boot sd {np.round(sd_b, 4).tolist()}", flush=True)
    return out


def option_a(name="loc1"):
    """Stage 4: option A's pooled theta_KL on the same set (contrast, Amendment I.2)."""
    fn = os.path.join(OUT, f"optionA_{name}.json")
    if os.path.exists(fn):
        return json.load(open(fn))
    S = load_set(name)
    SA = br.Sample(S.ul, S.vl, S.off, S.v0, S.s_obs)
    SA.init_mode = "mse"
    t0 = time.time()
    fa = br.fit(SA, np.random.default_rng(seed(f"real/{name}/optionA")), n_random=2)
    res = {"theta_kl": np.asarray(fa["theta"], float).tolist(), "rms": float(np.exp(-fa["ll"] / SA.M)),
           "at_bound": fa["at_bound"], "seconds": time.time() - t0}
    json.dump(res, open(fn, "w"), indent=2, default=float)
    print(f"[4] option A {res['seconds'] / 60:.1f} min  theta_KL "
          f"{np.round(res['theta_kl'], 4).tolist()}  rms {res['rms']:.3f} m", flush=True)
    return res


# ---------------------------------------------------------------------------- report
def region_dict(fit, tag):
    """The J.12 region as bridge.in_region expects it: centre, bootstrap covariance, c*q."""
    r = fit["regions"][tag]
    flat = np.array([n in fit["flat_directions"] for n in PN])
    return {"center": np.array(fit["theta_hat"], float), "cov": np.array(fit["cov"], float),
            "q": r["c"] * fit["q_loo"], "flat": flat, "flat_names": fit["flat_directions"]}


def box_overlap(a, b):
    """Do the axis-aligned marginal boxes overlap? Only a NECESSARY condition for the regions
    to intersect - a False here is conclusive, a True is not (S5 review of J.12)."""
    lo1, hi1 = np.array(a["box_lo"]), np.array(a["box_hi"])
    lo2, hi2 = np.array(b["box_lo"]), np.array(b["box_hi"])
    per = [bool(hi1[j] >= lo2[j] and hi2[j] >= lo1[j]) for j in range(br.K)]
    return {"per_parameter": per, "all": bool(all(per)),
            "disjoint_parameters": [PN[j] for j in range(br.K) if not per[j]]}


def ellipsoid_intersect(r1, r2):
    """g = min over theta in the parameter box of max(d1/q1, d2/q2); the two regions intersect
    iff g <= 1. Convex in theta, so a local minimum is global. Returns None when either region
    is degenerate (a flat bootstrap direction), where the quadratic form is not a region."""
    if r1["flat_names"] or r2["flat_names"]:
        return {"g": None, "intersect": None,
                "note": f"degenerate region: flat directions {r1['flat_names']} / {r2['flat_names']}"}
    from scipy.optimize import minimize
    P1, P2 = np.linalg.pinv(r1["cov"]), np.linalg.pinv(r2["cov"])
    c1, c2 = r1["center"], r2["center"]

    def g(t):
        d1, d2 = t - c1, t - c2
        return max(float(d1 @ P1 @ d1) / r1["q"], float(d2 @ P2 @ d2) / r2["q"])

    best = None
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        t0 = np.clip((1 - w) * c1 + w * c2, br.LO, br.HI)
        r = minimize(g, t0, method="Powell", bounds=list(zip(br.LO, br.HI)),
                     options={"xtol": 1e-10, "ftol": 1e-12, "maxiter": 20000})
        if best is None or r.fun < best[0]:
            best = (float(r.fun), r.x.copy())
    return {"g": best[0], "intersect": bool(best[0] <= 1.0), "witness": best[1].tolist()}


def replication(fits, tag):
    """Amendment F, R3: cross-site containment both ways, plus a real intersection test."""
    a, b = fits["loc5"], fits["loc1_ff"]
    ra, rb = region_dict(a, tag), region_dict(b, tag)
    out = {"box_overlap": box_overlap(a["regions"][tag], b["regions"][tag]),
           "loc5_theta_in_loc1ff_region": bool(br.in_region(rb, a["theta_hat"])),
           "loc1ff_theta_in_loc5_region": bool(br.in_region(ra, b["theta_hat"]))}
    out.update({"ellipsoid": ellipsoid_intersect(ra, rb)})
    return out


def clipped_coords(fit, tag):
    """Parameters whose marginal interval is cut by the parameter bounds, and those whose
    interval fills the whole allowed range (where an overlap statement carries no information)."""
    r = fit["regions"][tag]
    th, half = np.array(fit["theta_hat"]), np.array(r["marginal_half_width"])
    lo, hi = np.array(r["box_lo"]), np.array(r["box_hi"])
    frac = (hi - lo) / (br.HI - br.LO)
    return {"clipped": [PN[j] for j in range(br.K)
                        if th[j] - half[j] < br.LO[j] - 1e-12 or th[j] + half[j] > br.HI[j] + 1e-12],
            "saturated": [PN[j] for j in range(br.K) if frac[j] > 0.999],
            "width_fraction_of_bounds": frac.tolist()}


def summarise():
    fits = {n: json.load(open(os.path.join(OUT, f"fit_{n}.json")))
            for n in SETS if os.path.exists(os.path.join(OUT, f"fit_{n}.json"))}
    if "loc1" not in fits:
        sys.exit("STOP - the primary fit is not finished; nothing to report")
    oa = (json.load(open(os.path.join(OUT, "optionA_loc1.json")))
          if os.path.exists(os.path.join(OUT, "optionA_loc1.json")) else None)
    probe = json.load(open(PROBE_FILE)) if os.path.exists(PROBE_FILE) else None
    if probe and probe.get("j1_selected") not in (None, "T"):
        sys.exit(f"STOP - the probe selected per = {probe['j1_selected']}, this script assumes T")
    p = fits["loc1"]
    th = np.array(p["theta_hat"])
    res = {"note": "Step 6 (Amendment J.12): real B-lite calibration and its region. "
                   "Aggregates only; real trajectories stay in derived/.",
           "fits": fits, "optionA": oa,
           "probe_theta_typ": (probe["fits"]["T"]["theta_typ"] if probe else None)}
    if probe and not probe.get("episodes"):
        print("WARNING - probe.json has no episode count; the probe comparison is skipped", flush=True)
        probe = None
    if probe:
        # the probe is a subset of these same episodes, so the SE of the DIFFERENCE is
        # boot_sd * sqrt(M/m - 1), not boot_sd (S5 review of J.12)
        diff = th - np.array(probe["fits"]["T"]["theta_typ"])
        se_d = np.array(p["boot_sd"]) * math.sqrt(max(p["M"] / probe["episodes"] - 1.0, 1e-12))
        res["probe_diff"] = diff.tolist()
        res["probe_diff_in_se_of_difference"] = (diff / se_d).tolist()
    res["clipping"] = {n: {tag: clipped_coords(f, tag) for tag in f["regions"]}
                       for n, f in fits.items()}
    if "loc5" in fits and "loc1_ff" in fits:
        res["replication"] = {tag: replication(fits, tag) for tag in fits["loc5"]["regions"]}
    json.dump(res, open(os.path.join(OUT, "step6.json"), "w"), indent=2, default=float)
    f4 = lambda v: "[" + ", ".join(f"{x:.4f}" for x in v) + "]"
    L = ["# Step 6 - real calibration on highD (Amendment J.12)", "",
         f"**theta_hat (loc1, {p['M']} episodes) = {f4(p['theta_hat'])}**  (v0, T, s0, a, b)",
         f"- residual RMS {p['resid_rms']:.3f} m (sample-weighted); episode-weighted "
         f"{p['rms']:.3f} m; per-episode RMS median {p['ep_rms_median']:.3f} m, "
         f"p90 {p['ep_rms_p90']:.3f} m",
         f"- per-driver T: q10-q90 {f4(p['T_q10_25_50_75_90'][:1] + p['T_q10_25_50_75_90'][-1:])}, "
         f"{p['T_at_bound'] * 100:.1f}% at a bound",
         f"- bootstrap (B = {p['B']}) SD {f4(p['boot_sd'])}; shared at bound: "
         f"{p['shared_at_bound'] or 'none'}; flat directions: {p['flat_directions'] or 'none'}"]
    if probe and "probe_diff_in_se_of_difference" in res:
        L.append(f"- vs the S1 probe ({probe['episodes']} of these episodes), in SDs of the "
                 f"difference: "
                 f"{f4(res['probe_diff_in_se_of_difference'])} (falsifier: any |value| > 3)")
    if p["cov_rank"] < br.K:
        L.append(f"- **the bootstrap covariance has rank {p['cov_rank']} of {br.K}**: the region "
                 f"is unbounded in some direction and the coverage calibration does not cover it")
    L += ["", "## 95% region for the typical driver: marginal projections (bounding box)", "",
          "| basis | c | radius | v0 | T | s0 | a | b |", "|---|---|---|---|---|---|---|---|"]
    flat = p["flat_directions"]
    for tag, r in p["regions"].items():
        lo, hi = r["box_lo"], r["box_hi"]
        cells = " | ".join("degenerate" if PN[j] in flat else f"{lo[j]:.2f}-{hi[j]:.2f}"
                           for j in range(br.K))
        L.append(f"| {tag} | {r['c']:.3f} | x{r['radius_factor']:.2f} | {cells} |")
    L += ["", "These are the axis-aligned projections of the ellipsoidal region, so each single "
          "interval is wider than the region itself; S7 propagates the ellipsoid (centre, `cov` "
          "and `q_loo` are in step6.json), not this box.",
          "The primary row is the J.12 robust calibration (at least 95% coverage in every S4 arm). "
          "`generator_only` uses the fitted generator alone, calibrated at N = 1,000; `plain` is "
          "the uncalibrated bootstrap region, which S4 measured at 0.77-0.85 coverage."]
    if flat:
        L.append(f"**Degenerate directions: {flat}.** Those parameters are not identified by the "
                 f"bootstrap; the calibration factor was never validated for them (J.8).")
    cl = res["clipping"]["loc1"]["primary_robust"]
    L.append(f"- cut by the parameter bounds: {cl['clipped'] or 'none'}; filling the whole allowed "
             f"range: {cl['saturated'] or 'none'}")
    L.append(f"- shared parameters at a bound in the point fit: {p['shared_at_bound'] or 'none'}; "
             f"in {p['boot_shared_at_bound_frac'] * 100:.0f}% of bootstrap refits")
    L.append("")
    if "replication" in res:
        L += ["## Replication: location 5 vs regime-matched location-1 free flow", ""]
        for n in ("loc1_ff", "loc5"):
            q = fits[n]
            L.append(f"- **{n}** ({q['M']} episodes): theta_hat {f4(q['theta_hat'])}, "
                     f"rms {q['resid_rms']:.3f} m, shared at bound {q['shared_at_bound'] or 'none'}")
        for tag, ov in res["replication"].items():
            e = ov["ellipsoid"]
            verdict = ("not testable (degenerate region)" if e["intersect"] is None
                       else f"**{e['intersect']}** (g = {e['g']:.3f}, intersect iff g <= 1)")
            L.append(f"- {tag}: regions intersect: {verdict}; "
                     f"loc5 estimate inside the loc1_ff region: **{ov['loc5_theta_in_loc1ff_region']}**, "
                     f"and the reverse: **{ov['loc1ff_theta_in_loc5_region']}**")
            sat = res["clipping"]["loc5"][tag]["saturated"]
            if sat:
                L.append(f"  - location 5 fills the whole allowed range for {sat}, so an overlap "
                         f"there carries no information")
        L.append("Amendment F's R2 (subsampling loc1_ff to n = 518) is not computed here; it needs "
                 "its own bootstrap run and is recorded as outstanding.")
        L.append("")
    if oa:
        L += ["## Contrast: option A (pooled theta_KL)", "",
              f"- theta_KL {f4(oa['theta_kl'])}, RMS {oa['rms']:.3f} m, at bound "
              f"{oa['at_bound'] or 'none'}", ""]
    open(os.path.join(OUT, "step6.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


def fingerprint():
    """Code and data this run depends on (J.8 rule: bridge, idm, _common, the runner, the set)."""
    h = hashlib.md5()
    here = os.path.dirname(os.path.abspath(__file__))
    for fn in (os.path.join(SRC, "bridge.py"), os.path.join(SRC, "idm.py"),
               os.path.join(here, "_common.py"), os.path.abspath(__file__), SETS_FILE):
        h.update(open(fn, "rb").read())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ncpu = os.cpu_count() or 4
    ap.add_argument("--workers", type=int, default=max(1, ncpu // 4))
    ap.add_argument("--set-md5", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--only", choices=SETS)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.workers < 1:
        sys.exit("STOP - --workers must be at least 1")
    threads = max(1, ncpu // a.workers)
    os.makedirs(OUT, exist_ok=True)
    if a.summary:
        fpf = os.path.join(OUT, "fingerprint.txt")
        if os.path.exists(fpf) and os.path.exists(SETS_FILE) and open(fpf).read().strip() != fingerprint():
            print("WARNING - these saved results come from different code or a different episode "
                  "set; the report below is of stale results", flush=True)
        summarise()
        return 0
    g = json.load(open(GEN_FILE))
    if not g.get("gate", {}).get("passed", False):
        sys.exit("STOP - the S3 gate did not pass; S6 may not run")
    if not os.path.exists(SETS_FILE):
        n = build_sets()
        print(f"[0] sets built: {n}", flush=True)
    fpf = os.path.join(OUT, "fingerprint.txt")
    fp = fingerprint()
    if os.path.exists(fpf) and open(fpf).read().strip() != fp:
        sys.exit("STOP - saved Step 6 results come from different code, _common.py or episode set; "
                 "move code/results/step6 aside and start again")
    open(fpf, "w").write(fp + "\n")
    for name in ([a.only] if a.only else SETS):
        fit_set(name, a.workers, threads)
    if not a.only or a.only == "loc1":
        option_a()
    if a.only and not os.path.exists(os.path.join(OUT, "fit_loc1.json")):
        print(f"[5] {a.only} done; the report needs the primary loc1 fit, so it is not written yet.",
              flush=True)
        return 0
    summarise()
    return 0


if __name__ == "__main__":
    sys.exit(main())
