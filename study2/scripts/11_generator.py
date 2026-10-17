"""STEP 5-S2/S3 - build the synthetic generator from the S1 probe and run the parity gate
(Amendments J.3 and J.4).

  0  Reproducibility: refit B-lite {T} on the 300 probe episodes exactly as 10_probe.py did,
     recompute the S1 metrics with bridge.s1_metrics, and require equality with probe.json
     (1e-6, or the stored 4-decimal rounding). Stops otherwise.
  1  T model: OLS of log T_hat on log initial headway (bridge.t_model_fit); per-episode
     per-episode F4 slack (bridge.dhw_add_from_real, J.5).
  2  Tuning: 225-point grid (sigma_eta, tau, omega, kappa) x 2 replicates on the probe traces,
     common random numbers across grid points, full fit_blite refit, F4-conditioned
     generator. Score = max over gated metrics of d_m computed on the replicate MEAN.
     Checkpointed: rows are appended to grid_partial.csv and skipped on restart.
  2b Refinement (J.4): the 10 best points re-scored on 6 fresh replicates each (common
     random numbers); the best 6-replicate score is chosen. Reduces the winner's curse of
     picking the minimum of 225 two-replicate scores. Checkpointed (refine_partial.csv).
  3  S3 parity gate (J.4): 39 fresh replicates at the best point, full refit. Calibrated
     rank test: Z = max_m |m - mean_others| / sd_others for the real probe and for every
     replicate against the other 39; p = #{Z_j >= Z_real} / 40; PASS iff p > 0.05.

Writes code/results/step5_generator/{generator.json, grid.csv, grid_partial.csv,
generator.md} (aggregates only).
Usage (from AINN_ReWORK/scripts):  python 11_generator.py
Keep the laptop awake (plugged in, sleep off): the tuning step takes 1.5-3 h.
"""

import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # BLAS single-threaded; numba prange uses all cores

import glob
import hashlib
import importlib.util
import itertools
import json
import math
import sys
import time
import zlib

import numpy as np
import pandas as pd

from _common import ARRAY_ROWS, DERIVED, RESULTS, SRC  # noqa: F401

import bridge as br                                     # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "probe10", os.path.join(os.path.dirname(os.path.abspath(__file__)), "10_probe.py"))
probe10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe10)

GRID = {"sig_eta": (0.10, 0.20, 0.30, 0.45, 0.60), "tau": (0.5, 1.0, 2.0, 4.0, 8.0),
        "omega": (0.0, 0.5, 0.8), "kappa": (0.6, 0.8, 1.0)}
N_REP_TUNE, N_REFINE_TOP, N_REP_REFINE, N_REP_GATE, GATE_ALPHA = 2, 10, 6, 39, 0.05
LOG125 = math.log(1.25)
# J.4 gated metrics and their TUNING scales (the gate itself is scale-free)
RATIO_METRICS = ("ep_rms_median", "ep_rms_p90", "one_minus_acf_1s", "slope",
                 "T_iqr", "T_median", "T_sres")
ABS_METRICS = {"acf_5s": 0.10, "T_at_bound": 0.05, "diseq_q10": 0.10, "diseq_q50": 0.10,
               "diseq_q90": 0.10, "T_beta": 0.10}    # acf_5s absolute (J.5): it can be near 0
METRICS = RATIO_METRICS + tuple(ABS_METRICS)
RIDX = {k: i for i, k in enumerate(ARRAY_ROWS)}


def seed(label):
    return zlib.crc32(label.encode("utf-8"))


def load_probe_vf(n):
    """probe10.load_probe plus the follower speed series (for the F4 stand-in). Same
    selection code; equality with probe10.load_probe is asserted by the caller."""
    d = os.path.join(DERIVED, "highd_episodes_defE")
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "??_index.csv"))):
        if os.path.getsize(f) <= 1:
            continue
        ix = pd.read_csv(f)
        ix = ix[(ix["leader_class"] == "car") & (ix["dur_s"] >= 10) & (ix["sd_vL"] >= 1.0)
                & (ix["location"] == 1)]
        rows += [(f, ep) for ep in ix["episode"]]
    pick = np.sort(np.random.default_rng(probe10.PROBE_SEED).choice(len(rows), n, replace=False))
    vf, gap, gref, cache = [], [], [], {}
    for i in pick:
        f, ep = rows[i]
        if f not in cache:
            cache = {f: np.load(f.replace("_index.csv", "_episodes.npz"))}
        a = cache[f][ep]
        vf.append(a[RIDX["vF"]]); gap.append(a[RIDX["gap"]]); gref.append(a[RIDX["gap_ref"]])
    return np.concatenate(vf), np.concatenate(gap), np.concatenate(gref)


def gated(m, fit, gap0, v0):
    """The J.4 gated metrics from an s1_metrics dict and its fit."""
    tm = br.t_model_fit(fit["P"][:, 0], gap0, v0)
    return {"ep_rms_median": m["ep_rms_median"], "ep_rms_p90": m["ep_rms_p90"],
            "one_minus_acf_1s": 1.0 - m["acf_ep_mean"][1], "acf_5s": m["acf_ep_mean"][3],
            "slope": m["slope_ep_median"], "T_iqr": m["T_iqr"],
            "T_median": m["T_q10_25_50_75_90"][2], "T_sres": tm["s_res"],
            "T_at_bound": m["T_at_bound"], "diseq_q10": m["start_diseq_q10_50_90"][0],
            "diseq_q50": m["start_diseq_q10_50_90"][1], "diseq_q90": m["start_diseq_q10_50_90"][2],
            "T_beta": tm["beta"]}


def distances(syn, real):
    """Tuning distances (J.4): |log ratio| / log 1.25, or |diff| / tolerance. NaN -> inf."""
    d = {}
    for k in RATIO_METRICS:
        r = syn[k] / real[k] if real[k] != 0 else float("nan")
        d[k] = abs(math.log(r)) / LOG125 if (r > 0 and math.isfinite(r)) else float("inf")
    for k, tol in ABS_METRICS.items():
        v = abs(syn[k] - real[k]) / tol
        d[k] = v if math.isfinite(v) else float("inf")
    return d


def zmax(x, others):
    """max_m |x_m - mean(others_m)| / sd(others_m); 0/0 -> 0, x/0 -> inf."""
    mu = others.mean(axis=0)
    sd = others.std(axis=0, ddof=1)
    dev = np.abs(x - mu)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sd > 0, dev / np.where(sd > 0, sd, 1.0), np.where(dev > 0, np.inf, 0.0))
    return float(np.max(z)), z


def check_repro(m, f, pj):
    """Step 0: recomputed real metrics must equal probe.json."""
    bad = []

    def cmp(name, a, b, tol):
        a, b = np.atleast_1d(np.asarray(a, float)), np.atleast_1d(np.asarray(b, float))
        if a.shape != b.shape or not np.all(np.isfinite(a)) or np.max(np.abs(a - b)) > tol:
            bad.append(f"{name}: recomputed {a.tolist()} vs probe.json {b.tolist()}")

    r4 = 5e-5 + 1e-6
    if pj["j1_selected"] != "T":
        bad.append(f"probe.json selected {pj['j1_selected']}, this script assumes T")
    cmp("theta_typ", np.round(f["theta_typ"], 4), pj["fits"]["T"]["theta_typ"], 1e-6)
    cmp("rms", f["rms"], pj["fits"]["T"]["rms"], 1e-6)
    cmp("ep_rms_median", m["ep_rms_median"], pj["ep_rms_median"], 1e-6)
    cmp("ep_rms_p90", m["ep_rms_p90"], pj["ep_rms_p90"], 1e-6)
    cmp("acf", m["acf"], pj["acf"], r4)
    cmp("slope", m["abs_resid_slope_m_per_s"], pj["abs_resid_slope_m_per_s"], 1e-6)
    cmp("diseq", m["start_diseq_q10_50_90"], pj["start_diseq_q10_50_90"], r4)
    cmp("T quantiles", m["T_q10_25_50_75_90"], pj["per_driver"]["T"]["q10_25_50_75_90"], r4)
    cmp("T iqr", m["T_iqr"], pj["per_driver"]["T"]["iqr"], 1e-6)
    cmp("T at bound", m["T_at_bound"], pj["per_driver"]["T"]["at_bound"], 1e-9)
    return bad


def synth(ul, vl, off, gap0, v0, dhw_add, shared_theta, tmod, knobs, label):
    s_syn, thetas, _, info = br.gen_synthetic(ul, vl, off, gap0, v0, shared_theta, tmod,
                                              knobs["sig_eta"], knobs["tau"], knobs["omega"],
                                              knobs["kappa"], seed(label), dhw_add)
    S = br.BliteSample(ul, vl, off, v0, s_syn, per=(1,))
    f = br.fit_blite(S, np.random.default_rng(seed(label + "/fit")), n_random=2)
    m = gated(br.s1_metrics(S, f), f, s_syn[off[:-1]], v0)
    true_typ = np.array(shared_theta, float)
    true_typ[1] = np.median(thetas[:, 1])
    return m, f, true_typ, info


def main():
    out = os.path.join(RESULTS, "step5_generator")
    os.makedirs(out, exist_ok=True)
    pj = json.load(open(os.path.join(RESULTS, "step5_probe", "probe.json")))
    ul, vl, off, v0, gap, n_pop = probe10.load_probe(pj["episodes"])
    vf_all, gap_chk, gap_ref = load_probe_vf(pj["episodes"])
    if gap_chk.shape != gap.shape or not np.array_equal(gap_chk, gap):
        print("STOP - F4 loader does not select the same episodes as 10_probe.load_probe")
        return 1
    dmax = float(np.max(np.abs(gap - gap_ref)))
    if dmax > 0.011:          # J.5 premise: gap == highD bumper-to-bumper dhw (DIARY Entry 10)
        print(f"STOP - max |gap - dhw| = {dmax:.4f} m > 0.011 m; the F4 premise does not hold")
        return 1
    gap0 = gap[off[:-1]]
    dhw_add = br.dhw_add_from_real(gap, vf_all, off)
    print(f"probe: {len(v0)} of {n_pop} episodes; max |gap - dhw| {dmax:.4f} m; F4 slack: "
          f"{int(np.sum(dhw_add < -1e-5))} episodes loosened, min {dhw_add.min():.3f} m", flush=True)

    # 0  reproducibility
    S = br.BliteSample(ul, vl, off, v0, gap, per=(1,))
    t0 = time.time()
    f = br.fit_blite(S, np.random.default_rng(probe10.PROBE_SEED), n_random=2)
    m_real_full = br.s1_metrics(S, f)
    bad = check_repro(m_real_full, f, pj)
    if bad:
        print("STOP - real metrics do not reproduce probe.json:\n  " + "\n  ".join(bad))
        return 1
    real = gated(m_real_full, f, gap0, v0)
    nonpos = [k for k in RATIO_METRICS if not (real[k] > 0 and math.isfinite(real[k]))]
    if nonpos:
        print(f"STOP - real value of ratio metric(s) {nonpos} is not positive: "
              f"{ {k: real[k] for k in nonpos} }; J.4's ratio scale is undefined")
        return 1
    print(f"[0] reproducibility OK ({time.time() - t0:.0f} s); real gated metrics "
          f"{json.dumps({k: round(v, 4) for k, v in real.items()})}", flush=True)

    # 1  T model
    tmod = br.t_model_fit(f["P"][:, 0], gap0, v0)
    shared_theta = np.array(f["theta_typ"], float)
    print(f"[1] T model {tmod}", flush=True)

    # checkpoint fingerprint: code + real inputs; a mismatch refuses to resume
    here = os.path.dirname(os.path.abspath(__file__))
    fp = hashlib.md5()
    for fn in (os.path.join(SRC, "bridge.py"), os.path.join(here, "11_generator.py")):
        fp.update(open(fn, "rb").read())
    fp.update(json.dumps({"real": real, "tmod": tmod, "grid": GRID}, sort_keys=True).encode())
    fp = fp.hexdigest()
    part = os.path.join(out, "grid_partial.csv")
    rpart = os.path.join(out, "refine_partial.csv")
    fpfile = os.path.join(out, "checkpoint_fingerprint.txt")
    if os.path.exists(part) or os.path.exists(rpart):
        old = open(fpfile).read().strip() if os.path.exists(fpfile) else ""
        if old != fp:
            print("STOP - checkpoint files come from different code or inputs. Move "
                  "grid_partial.csv and refine_partial.csv out of code/results/step5_generator "
                  "and start again.")
            return 1
        print("[2] resuming from checkpoint", flush=True)
    open(fpfile, "w").write(fp + "\n")

    def read_part(fn):
        if not os.path.exists(fn):
            return pd.DataFrame()
        d = pd.read_csv(fn, on_bad_lines="skip")
        d = d.dropna(subset=["label", "score"])
        d.to_csv(fn, index=False)                 # drop a truncated last line, if any
        return d

    # 2  tuning grid (checkpointed)
    done = read_part(part)
    done_labels = set(done["label"]) if len(done) else set()
    t0 = time.time()
    pts = list(itertools.product(*GRID.values()))
    for i, vals in enumerate(pts):
        knobs = dict(zip(GRID.keys(), vals))
        lab = "/".join(f"{k}={v:g}" for k, v in knobs.items())
        if lab in done_labels:
            continue
        ms, infos = [], []
        for rep in range(N_REP_TUNE):
            m, _, _, info = synth(ul, vl, off, gap0, v0, dhw_add, shared_theta, tmod, knobs,
                                  f"tune/rep{rep}")
            ms.append(m); infos.append(info)
        mean = {k: float(np.mean([mm[k] for mm in ms])) for k in METRICS}
        row = dict(label=lab, **knobs, score=float(max(distances(mean, real).values())), **mean,
                   attempts_mean=float(np.mean([x["attempts_mean"] for x in infos])),
                   accept_first_try=float(np.mean([x["accept_first_try"] for x in infos])),
                   f4_violating_final=int(sum(x["f4_violating_final"] for x in infos)))
        pd.DataFrame([row]).to_csv(part, mode="a", header=not os.path.exists(part), index=False)
        if (i + 1) % 5 == 0:
            print(f"[2] {i + 1}/{len(pts)} points, {time.time() - t0:.0f} s this session", flush=True)
    grid = read_part(part).drop_duplicates("label").sort_values("score")
    grid.to_csv(os.path.join(out, "grid.csv"), index=False)
    if not np.isfinite(grid["score"].iloc[0]):
        print("STOP - every grid point has an infinite or undefined score; see grid.csv")
        return 1
    print(grid.head(N_REFINE_TOP)[["label", "score", "attempts_mean", "accept_first_try",
                                   "f4_violating_final"]].to_string(index=False), flush=True)

    # 2b refinement of the top points on fresh replicates
    top = set(grid.head(N_REFINE_TOP)["label"])
    rd = read_part(rpart)
    rdone = set(rd["label"]) if len(rd) else set()
    t0 = time.time()
    for _, g in grid.head(N_REFINE_TOP).iterrows():
        if g["label"] in rdone:
            continue
        knobs = {k: float(g[k]) for k in GRID}
        ms = [synth(ul, vl, off, gap0, v0, dhw_add, shared_theta, tmod, knobs, f"refine/rep{rep}")[0]
              for rep in range(N_REP_REFINE)]
        mean = {k: float(np.mean([mm[k] for mm in ms])) for k in METRICS}
        sd = {f"sd_{k}": float(np.std([mm[k] for mm in ms], ddof=1)) for k in METRICS}
        row = dict(label=g["label"], **knobs, tune_score=float(g["score"]),
                   score=float(max(distances(mean, real).values())), **mean, **sd)
        pd.DataFrame([row]).to_csv(rpart, mode="a", header=not os.path.exists(rpart), index=False)
        print(f"[2b] {g['label']}: 2-rep score {g['score']:.3f} -> 6-rep score {row['score']:.3f} "
              f"({time.time() - t0:.0f} s)", flush=True)
    refine = read_part(rpart).drop_duplicates("label")
    refine = refine[refine["label"].isin(top)].sort_values("score")
    refine.to_csv(os.path.join(out, "refine.csv"), index=False)
    if not np.isfinite(refine["score"].iloc[0]):
        print("STOP - every refined point has an infinite or undefined score; see refine.csv")
        return 1
    best = {k: float(refine.iloc[0][k]) for k in GRID}
    print(f"[2b] best point {best}, 6-replicate score {refine.iloc[0]['score']:.3f}", flush=True)

    # 3  parity gate: calibrated rank test on 39 fresh replicates, full refit
    t0 = time.time()
    gm, typs = [], []
    for rep in range(N_REP_GATE):
        m, fg, true_typ, info = synth(ul, vl, off, gap0, v0, dhw_add, shared_theta, tmod, best,
                                      f"gate/rep{rep}")
        gm.append(m)
        typs.append({"true": true_typ.tolist(), "fitted": fg["theta_typ"].tolist(),
                     "shared_at_bound": fg["shared_at_bound"], **info})
        if (rep + 1) % 5 == 0:
            print(f"[3] gate replicate {rep + 1}/{N_REP_GATE}: {time.time() - t0:.0f} s", flush=True)
    A = np.array([[m[k] for k in METRICS] for m in gm])
    x_real = np.array([real[k] for k in METRICS])
    if not (np.isfinite(A).all() and np.isfinite(x_real).all()):
        bad_m = [k for i, k in enumerate(METRICS) if not np.isfinite(A[:, i]).all()]
        print(f"STOP - non-finite gate metric(s) {bad_m}; gate not evaluated")
        return 1
    Z_real, z_real = zmax(x_real, A)
    Z_rep = [zmax(A[j], np.vstack([np.delete(A, j, axis=0), x_real]))[0] for j in range(len(A))]
    p = (1 + sum(z >= Z_real for z in Z_rep)) / (len(A) + 1)
    passed = p > GATE_ALPHA
    lo, hi = np.quantile(A, 0.05, axis=0), np.quantile(A, 0.95, axis=0)
    med = np.median(A, axis=0)
    dj = distances(dict(zip(METRICS, med)), real)
    table = {k: {"real": real[k], "syn_median": float(med[i]), "syn_q05": float(lo[i]),
                 "syn_q95": float(hi[i]), "z": float(z_real[i]),
                 "rank_p": float((1 + np.sum(np.abs(A[:, i] - A[:, i].mean()) >= abs(x_real[i] - A[:, i].mean())))
                                 / (len(A) + 1)),
                 "J_tolerance_d_on_median": float(dj[k])} for i, k in enumerate(METRICS)}
    res = {"note": "S2/S3 (Amendments J.3, J.4). Generator design and parity gate; not results.",
           "real_metrics": real, "t_model": tmod, "shared_theta": shared_theta.tolist(),
           "dhw_add_q10_50_90": np.quantile(dhw_add, [0.1, 0.5, 0.9]).tolist(),
           "grid": {k: list(v) for k, v in GRID.items()}, "best": best,
           "tuning_score": float(grid.iloc[0]["score"]),
           "refined_score": float(refine.iloc[0]["score"]),
           "gate": {"Z_real": Z_real, "p": p, "alpha": GATE_ALPHA, "passed": bool(passed),
                    "replicates": N_REP_GATE, "metrics": table},
           "gate_replicate_metrics": gm, "gate_typical_driver_true_vs_fitted": typs}
    json.dump(res, open(os.path.join(out, "generator.json"), "w"), indent=2,
              default=float)
    L = ["# S2/S3 generator and parity gate (Amendments J.3, J.4)", "",
         f"- T model: {json.dumps(tmod)}", f"- best point: {json.dumps(best)}",
         f"- best 2-replicate tuning score: {res['tuning_score']:.3f}; chosen point's 6-replicate "
         f"score: {res['refined_score']:.3f}",
         f"- **gate: Z_real = {Z_real:.2f}, p = {p:.3f} -> {'PASSED' if passed else 'FAILED'}**", "",
         "| metric | real | synthetic median | synthetic 5-95% | z | J-tolerance d (info) |",
         "|---|---|---|---|---|---|"]
    for k, t in table.items():
        L.append(f"| {k} | {t['real']:.4f} | {t['syn_median']:.4f} | {t['syn_q05']:.4f} - {t['syn_q95']:.4f} | "
                 f"{t['z']:.2f} | {t['J_tolerance_d_on_median']:.2f} |")
    L += ["", "Typical driver, true vs fitted (informational, not gated; S4 measures this properly):"]
    tru = np.array([t["true"] for t in typs]); fit_ = np.array([t["fitted"] for t in typs])
    L += [f"- true (median over replicates) {np.round(np.median(tru, 0), 3).tolist()}",
          f"- fitted median {np.round(np.median(fit_, 0), 3).tolist()}, "
          f"q05 {np.round(np.quantile(fit_, 0.05, 0), 3).tolist()}, q95 {np.round(np.quantile(fit_, 0.95, 0), 3).tolist()}",
          f"- replicates with a shared parameter at a bound: {sum(bool(t['shared_at_bound']) for t in typs)}",
          f"- F4 conditioning in the gate replicates: mean attempts "
          f"{np.mean([t['attempts_mean'] for t in typs]):.2f}, accepted first try "
          f"{np.mean([t['accept_first_try'] for t in typs]):.3f}, episodes still violating "
          f"{sum(t['f4_violating_final'] for t in typs)} of {N_REP_GATE * len(v0)}"]
    open(os.path.join(out, "generator.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
