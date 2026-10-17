"""STEP 5-S4b - diagnostic of the pseudo-true value theta* for generator G0 (Amendment J.10).

Question (from J.9 / D5.4): the B-lite estimates at N = 1,000 and 5,691 agree with each other
(a ~ 1.24, b ~ 4.20) but differ from theta* (a 1.296, b 4.452), estimated by two fits on 4 copies
of the 5,691 location-1 traces. Is that
  O1  an OPTIMISER effect  - some fits stopped short on the flat v0-a-b ridge,
  O2  a SAMPLE-SIZE effect - the estimator still drifts between 5,691 and 22,764 drivers, or
  O3  neither              - the gap is within Monte Carlo noise?

S4 datasets are regenerated exactly from the S4 seeds (checked in step 0).

  0  reproduction: regenerate V-5691 rep 0 and 1 and pseudo-true half 0, standard-fit, and
     require equality with the stored S4 values (<= 1e-9); stop if code/pool/generator differ
     from the S4 run (fingerprint)
  1  R1 (optimiser) on the S4 datasets themselves: strong refits of V-5691 reps 0-5, both
     pseudo-true halves and G0-1000 reps 0-9 (stored criterion value re-checked for each)
  2  FRESH datasets, which did not produce the S4 gap (S5 review of J.10): 20 x k = 1
     (standard fits) and 6 x k = 4 (standard + strong fits), for R2 and theta*_final
  3  classification by J.10 (revised); theta*_final from the fresh data; all G0-based S4
     decisions re-scored against theta*_final from the stored bootstrap vectors

A "strong" fit = the standard start set (generic + 2 random, seeded by the dataset id) + 3
more random starts + warm starts at the dataset's own standard solution, theta*_S4, the V-5691
median and the G0-1000 median; every start's optimum is recorded.

Writes code/results/step5_diag/{records.jsonl, diag.json, diag.md} (synthetic aggregates only).
Usage (from AINN_ReWORK/scripts):  python 13_theta_star_diag.py      (resumable)
"""

import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import hashlib
import importlib.util
import json
import math
import sys
import time

import numpy as np
from scipy.optimize import minimize

from _common import RESULTS  # noqa: F401

import bridge as br                                     # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "s4", os.path.join(os.path.dirname(os.path.abspath(__file__)), "12_bridge_blite.py"))
s4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s4)

OUT = os.path.join(RESULTS, "step5_diag")
S4 = s4.OUT
V_REUSE, K1_FRESH, K4_FRESH, G1000_REPS = 6, 20, 6, 10       # J.10 (revised)
V_EXPECTED, G1K_EXPECTED, G1K_N, PT_K = 30, 60, 1000, 4
EXTRA_RANDOM = 3
OPT = {"maxiter": 300, "ftol": 1e-12, "gtol": 1e-9, "eps": 1e-6}     # = fit_blite
REL_TOL, MOVE_SD = 1e-7, 0.25                                        # J.10 rule R1


def seed(label):
    return s4.seed(label)


def multi_start(S, starts, rng, n_random):
    """fit_blite's exact procedure, but over an explicit start list, recording every start."""
    sh = S.shared
    lo, span = br.LO[sh], br.HI[sh] - br.LO[sh]
    to_sh = lambda u: lo + np.clip(u, 0, 1) * span
    us = [("generic", (br.GENERIC_START[sh] - lo) / span)]
    us += [(f"random{i}", rng.uniform(0.15, 0.85, len(sh))) for i in range(n_random)]
    us += [(name, (np.asarray(st, float)[sh] - lo) / span) for name, st in starts]
    rows, best = [], None
    for name, u0 in us:
        r = minimize(lambda u: S.criterion(to_sh(u)), np.clip(u0, 0, 1), method="L-BFGS-B",
                     bounds=[(0.0, 1.0)] * len(sh), options=OPT)
        rows.append({"start": name, "L": float(r.fun), "shared": to_sh(r.x).tolist(),
                     "message": str(r.message)})
        if best is None or r.fun < best[0]:
            best = (float(r.fun), r.x.copy())
    L = S.criterion(to_sh(best[1]))
    typ = np.array(br.GENERIC_START, dtype=float)
    typ[sh] = to_sh(best[1])
    typ[1] = float(np.median(S.P[:, 0]))
    return {"L": float(L), "theta_typ": typ.tolist(), "starts": rows}


def crit_at(S, theta):
    """Criterion at the shared values of a full 5-vector; also returns the implied theta_typ."""
    L = S.criterion(np.asarray(theta, float)[S.shared])
    typ = np.array(theta, float)
    typ[1] = float(np.median(S.P[:, 0]))
    return float(L), typ.tolist()


def read_records():
    fn = os.path.join(OUT, "records.jsonl")
    recs = {}
    if os.path.exists(fn):
        for line in open(fn, encoding="utf-8"):
            try:
                r = json.loads(line)
                recs[r["id"]] = r
            except (json.JSONDecodeError, KeyError):
                continue
    return recs


def append(rec):
    fn = os.path.join(OUT, "records.jsonl")
    lead = ""
    if os.path.exists(fn) and os.path.getsize(fn) > 0:
        with open(fn, "rb") as fh:
            fh.seek(-1, os.SEEK_END)
            lead = "" if fh.read(1) == b"\n" else "\n"
    with open(fn, "a", encoding="utf-8") as fh:
        fh.write(lead + json.dumps(rec, default=float) + "\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    g = json.load(open(s4.GEN_FILE))
    gen = {"shared_theta": g["shared_theta"], "t_model": g["t_model"], "best": g["best"]}
    arm = s4.arms_from_generator(g)["G0"]
    pt = json.load(open(os.path.join(S4, "pseudo_true.json")))
    V = s4.read_cell("V-5691")
    G1k = s4.read_cell("G0-1000")
    if len(V) != V_EXPECTED or len(G1k) != G1K_EXPECTED:
        sys.exit("STOP - S4 cells V-5691 / G0-1000 are not complete")
    th_star = np.array(pt["G0"]["theta_star"])
    v_med = np.median([r["theta_hat"] for r in V], 0)
    g1k_med = np.median([r["theta_hat"] for r in G1k], 0)
    sd_v = np.std([r["theta_hat"] for r in V], 0, ddof=1)
    P = s4.load_pool()
    M = len(P["v0"])
    if M != s4.N_POOL:
        sys.exit("STOP - pool size")
    fp4 = open(os.path.join(S4, "fingerprint.txt")).read().strip()
    if s4.fingerprint(gen) != fp4:
        sys.exit("STOP - code, pool or generator differ from the S4 run; datasets would not be the S4 ones")
    h = hashlib.md5()                       # the data-producing code only (summary excluded)
    own = open(os.path.abspath(__file__), "rb").read()
    h.update(own[:own.index(b"def summarise(")])
    for fn in (s4.__file__, os.path.join(s4.SRC, "bridge.py")):
        h.update(open(fn, "rb").read())
    fpf = os.path.join(OUT, "fingerprint.txt")
    recs = read_records()
    if "--summary" in sys.argv:
        summarise(recs, th_star, v_med, sd_v, pt)
        return 0
    if os.path.exists(fpf) and open(fpf).read().strip() != h.hexdigest():
        sys.exit("STOP - saved diagnostic records come from different code; move code/results/step5_diag aside")
    open(fpf, "w").write(h.hexdigest() + "\n")
    warm = [("theta_star_S4", th_star), ("V5691_median", v_med), ("G01000_median", g1k_med)]
    recs = read_records()
    t_all = time.time()

    def dataset(kind, idx, label):
        S, _, info = s4.generate(P, idx, arm, gen, label)
        return S, info

    def do(rid, kind, k, idx, gen_label, std_fit_label, std_known=None, check=None,
           stored_L=None, strong_fit=True):
        if rid in recs:
            return recs[rid]
        t0 = time.time()
        S, info = dataset(kind, idx, gen_label)
        assert list(S.per) == [1]
        err = None
        if std_known is None:
            f = br.fit_blite(S, np.random.default_rng(seed(std_fit_label)), n_random=2)
            std = {"L": f["L"], "theta_typ": f["theta_typ"].tolist()}
        else:
            L, _ = crit_at(S, std_known)
            std = {"L": L, "theta_typ": list(std_known)}
            if stored_L is not None and abs(L - stored_L) > 1e-12 * abs(stored_L):
                sys.exit(f"STOP - {rid}: regenerated dataset gives criterion {L!r} at the stored "
                         f"S4 estimate, S4 stored {stored_L!r}; the dataset is not the S4 one")
        if check is not None:
            err = float(np.max(np.abs(np.array(std["theta_typ"]) - np.array(check))))
            if err > 1e-9:
                sys.exit(f"STOP - {rid}: regenerated fit differs from the stored S4 value by {err:.3g}")
        strong = (multi_start(S, [("own_standard", std["theta_typ"])] + warm,
                              np.random.default_rng(seed(rid + "/strong")), 2 + EXTRA_RANDOM)
                  if strong_fit else None)
        cross = {name: crit_at(S, th)[0] for name, th in warm}
        rec = {"id": rid, "kind": kind, "k": k, "N": int(len(idx)), "standard": std, "strong": strong,
               "cross_L": cross, "repro_err": err, "f4": info, "seconds": time.time() - t0}
        append(rec)
        recs[rid] = rec
        msg = f"[{rid}] {rec['seconds'] / 60:.1f} min  std a,b {np.round(np.array(std['theta_typ'])[[3, 4]], 4).tolist()}"
        if strong:
            msg += (f"  strong a,b {np.round(np.array(strong['theta_typ'])[[3, 4]], 4).tolist()}"
                    f"  dL/L {(std['L'] - strong['L']) / std['L']:.2e}")
        done_n[0] += 1
        print(msg + f"   [{done_n[0]}/{n_total} this session, {(time.time() - t_all) / 60:.0f} min]", flush=True)
        return rec

    one = np.arange(M)
    plan = (["k1_V0", "k1_V1", "k4_pt0"] + [f"k1_V{r}" for r in range(2, V_REUSE)] + ["k4_pt1"]
            + [f"g1000_{r}" for r in range(G1000_REPS)] + [f"f1_{i}" for i in range(K1_FRESH)]
            + [f"f4_{i}" for i in range(K4_FRESH)])
    n_total = sum(1 for rid in plan if rid not in recs)
    done_n = [0]
    print(f"[plan] {len(plan)} datasets, {n_total} still to do (about 4 h in total on the laptop)", flush=True)
    # 0 reproduction checks: regenerate + standard refit must equal the stored S4 values
    for r in (0, 1):
        do(f"k1_V{r}", "k1", 1, one, f"V-5691/rep{r}/gen", f"V-5691/rep{r}/fit",
           check=V[r]["theta_hat"])
    do("k4_pt0", "k4", 4, np.tile(one, PT_K), "pt/G0/half0", "pt/G0/half0/fit",
       check=pt["G0"]["halves"][0])
    print("[0] reproduction OK: regenerated datasets and fits equal the S4 values (<= 1e-9)", flush=True)
    # 1 R1 on the S4 datasets themselves (strong refits; standard fits known exactly)
    for r in range(2, V_REUSE):
        do(f"k1_V{r}", "k1", 1, one, f"V-5691/rep{r}/gen", None, std_known=V[r]["theta_hat"],
           stored_L=V[r]["L"])
    do("k4_pt1", "k4", 4, np.tile(one, PT_K), "pt/G0/half1", None, std_known=pt["G0"]["halves"][1])
    for r in range(G1000_REPS):
        lab = f"G0-1000/rep{r}"
        idx = np.sort(np.random.default_rng(seed(lab)).choice(M, G1K_N, replace=False))
        do(f"g1000_{r}", "g1000", None, idx, lab + "/gen", None, std_known=G1k[r]["theta_hat"],
           stored_L=G1k[r]["L"])
    # 2 FRESH datasets for R2 and theta*_final (none of them produced the S4 gap)
    for i in range(K1_FRESH):
        do(f"f1_{i}", "f1", 1, one, f"diag/k1/{i}", f"diag/k1/{i}/fit", strong_fit=False)
    for i in range(K4_FRESH):
        do(f"f4_{i}", "f4", 4, np.tile(one, PT_K), f"diag/k4/{i}", f"diag/k4/{i}/fit")

    summarise(recs, th_star, v_med, sd_v, pt)
    print(f"total {(time.time() - t_all) / 60:.0f} min this session", flush=True)
    return 0


def summarise(recs, th_star, v_med, sd_v, pt):
    rs = list(recs.values())
    A, B = 3, 4
    # R1 optimiser (J.10): strong fit better by > REL_TOL and moves a or b by > MOVE_SD * SD_V/sqrt(k)
    moved = []
    for r in rs:
        r["_dL"], r["_dth"] = None, None
        if not r["strong"]:
            continue
        k = r["k"] or (G1K_N / s4.N_POOL)
        sd_k = sd_v / math.sqrt(k)
        dL = (r["standard"]["L"] - r["strong"]["L"]) / r["standard"]["L"]
        dth = (np.array(r["strong"]["theta_typ"]) - np.array(r["standard"]["theta_typ"])) / sd_k
        r["_dL"], r["_dth"] = dL, dth.tolist()
        if dL > REL_TOL and max(abs(dth[A]), abs(dth[B])) > MOVE_SD:
            moved.append(r["id"])
    by = lambda kind: [r for r in rs if r["kind"] == kind]
    k4_all = by("k4") + by("f4")
    o1 = (any(r["id"] in moved for r in k4_all) or sum(r["id"] in moved for r in by("k1")) >= 2
          or sum(r["id"] in moved for r in by("g1000")) >= 2)
    # flat ridge: strong moved a or b by > MOVE_SD sd_k but gained <= REL_TOL (J.10, reported)
    flat = [r["id"] for r in rs if r["strong"] and r["_dL"] <= REL_TOL
            and max(abs(r["_dth"][A]), abs(r["_dth"][B])) > MOVE_SD]
    # R2 sample size (J.10, FRESH data only): STANDARD fits at k = 1 and at k = 4 (same procedure
    # on both sides; S5 re-review); strong k = 4 version reported as a sensitivity
    f1 = np.array([r["standard"]["theta_typ"] for r in by("f1")])
    f4 = np.array([r["standard"]["theta_typ"] for r in by("f4")])
    f4s_ = np.array([r["strong"]["theta_typ"] for r in by("f4")])
    med1, med4 = np.median(f1, 0), np.median(f4, 0)
    se = np.sqrt(1.57 * sd_v ** 2 / len(f1) + 1.57 * (sd_v / 2) ** 2 / len(f4))
    z = (med1 - med4) / se
    z_strong = (med1 - np.median(f4s_, 0)) / se
    sign = np.sign(v_med - th_star)                     # direction of the S4 gap
    o2 = bool((not o1) and (z[A] * sign[A] > 2 or z[B] * sign[B] > 2))
    verdict = "O1 optimiser" if o1 else ("O2 sample size" if o2 else "O3 within noise")
    # theta*_final from FRESH data only (J.10): O1 -> mean of strong fresh k = 4 fits;
    # O2 -> mean of standard fresh k = 4 fits; O3 -> k-weighted mean of fresh standard fits
    if verdict.startswith("O3"):
        th_new = (f1.sum(0) + 4 * f4.sum(0)) / (len(f1) + 4 * len(f4))
        mc_new = sd_v / math.sqrt(len(f1) + 4 * len(f4))
        th_alt = (f1.sum(0) + 4 * f4s_.sum(0)) / (len(f1) + 4 * len(f4s_))
    elif verdict.startswith("O2"):
        th_new, th_alt = f4.mean(0), f4s_.mean(0)
        mc_new = sd_v / math.sqrt(4 * len(f4))
    else:
        th_new, th_alt = f4s_.mean(0), f4.mean(0)
        mc_new = sd_v / math.sqrt(4 * len(f4))
    th_all_k4 = np.mean([r["strong"]["theta_typ"] for r in k4_all], 0)
    per_ds = []
    for r in rs:
        d = {k: r.get(k) for k in ("id", "kind", "k", "N", "_dL", "_dth", "repro_err", "seconds")}
        ref = (r["strong"] or r["standard"])["L"]
        d["cross_rel_excess"] = {n: (v - ref) / ref for n, v in r["cross_L"].items()}
        per_ds.append(d)
    res = {"verdict": verdict, "rule_R1_moved": moved, "flat_ridge_datasets": flat,
           "R2_z_f1_minus_f4": z.tolist(), "R2_z_sensitivity_strong_f4": z_strong.tolist(),
           "theta_star_final_alternative_fit_type": th_alt.tolist(),
           "R2_median_f1_standard": med1.tolist(), "R2_median_f4_standard": med4.tolist(),
           "theta_star_S4": th_star.tolist(), "theta_star_final": th_new.tolist(),
           "theta_star_final_mc_model": mc_new.tolist(),
           "info_mean_strong_all_k4_incl_pt_halves": th_all_k4.tolist(),
           "V5691_median_standard": v_med.tolist(), "sd_V": sd_v.tolist(), "per_dataset": per_ds}
    res["rescore"] = rescore(th_new, mc_new, sd_v)
    json.dump(res, open(os.path.join(OUT, "diag.json"), "w"), indent=2, default=float)
    f4s = lambda v: "[" + ", ".join(f"{x:.4f}" for x in v) + "]"
    L = ["# theta* diagnostic (Amendment J.10)", "", f"**Verdict: {verdict}**", "",
         f"- theta*_S4 = {f4s(th_star)}", f"- V-5691 median (standard fits) = {f4s(v_med)}",
         f"- fresh k=1 median (standard fits, n={len(f1)}) = {f4s(med1)}",
         f"- fresh k=4 median (standard fits, n={len(f4)}) = {f4s(med4)}",
         f"- R2 z (fresh k1 - fresh k4, standard fits) per parameter: {f4s(z)}",
         f"- sensitivity, R2 z with strong k=4 fits: {f4s(z_strong)}",
         f"- R1 datasets where the strong fit moved a or b by > {MOVE_SD} SD: {moved or 'none'}",
         f"- flat-ridge datasets (moved > {MOVE_SD} SD with gain <= {REL_TOL:g}): {flat or 'none'}"
         + ("  -> theta*_final depends on the start points; both versions below" if flat else ""),
         f"- theta*_final alternative (other fit type): {f4s(th_alt)}",
         f"- (information) mean strong k=4 incl. the S4 pt halves: {f4s(th_all_k4)}", "",
         "| dataset | N | strong better by (rel) | move / SD (v0, T, s0, a, b) | criterion excess at theta*_S4 / V median |",
         "|---|---|---|---|---|"]
    for r, d in zip(rs, per_ds):
        ce = d["cross_rel_excess"]
        mv = f"{r['_dL']:.2e} | {f4s(r['_dth'])}" if r["strong"] else "- | -"
        L.append(f"| {r['id']} | {r['N']} | {mv} | {ce['theta_star_S4']:.2e} / {ce['V5691_median']:.2e} |")
    rsc = res["rescore"]
    L += ["", "## S4 decisions re-scored against theta*_final (J.10; the J.9 values stay reported)", "",
          f"- theta*_final = {f4s(th_new)} (model MC error {f4s(mc_new)})",
          f"- G0-300 coverage {rsc['G0-300']['coverage']:.3f} {f4s(rsc['G0-300']['ci'])} (J.9: 0.710)",
          f"- G0-1000 coverage {rsc['G0-1000']['coverage']:.3f} {f4s(rsc['G0-1000']['ci'])} (J.9: 0.800)",
          f"- D5.2 validated: {rsc['D5.2']['validated']}; calibration factor c = {rsc['D5.2'].get('c')} (J.9: 2.283)",
          f"- D5.3 ({'coverage' if rsc['D5.2']['validated'] else 'calibrated coverage'}): {json.dumps(rsc['D5.3'])}",
          f"- D5.4: bias/SD {f4s(rsc['D5.4']['bias_over_sd'])}, CI lower {f4s(rsc['D5.4']['ci_lower'])}, "
          f"CI upper {f4s(rsc['D5.4']['ci_upper'])}, flag {rsc['D5.4']['flag']}"]
    open(os.path.join(OUT, "diag.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


def rescore(th_new, mc_new, sd_v):
    """J.10: recompute the G0-based S4 decisions against theta*_final from the stored bootstrap
    vectors (no refitting): G0 coverage, D5.2 (and its calibration factor c), D5.3 with that c
    (G1-G3 keep their own theta*), D5.4. The J.9 values remain reported next to these."""
    out = {}
    for cell in ("G0-300", "G0-1000"):
        recs = s4.read_cell(cell)
        cov, ratios = [], []
        for r in recs:
            reg = br.ellipsoid_region(np.array(r["theta_hat"]), np.array(r["theta_b"]), 0.95)
            if abs(reg["q"] - r["q"]) > 1e-9 * max(1.0, abs(r["q"])):
                sys.exit(f"STOP - {cell} rep {r['rep']}: recomputed q {reg['q']} != stored {r['q']}")
            d = s4.mahal(reg, th_new)
            cov.append(bool(d <= reg["q"]))
            ratios.append(d / reg["q"] if reg["q"] > 0 else float("inf"))
        k = int(sum(cov))
        out[cell] = {"coverage": k / len(recs), "ci": list(s4.wilson(k, len(recs))), "ratios": ratios}
    ok = [not (out[c]["ci"][1] < 0.95) for c in ("G0-300", "G0-1000")]
    d52 = {"validated": bool(all(ok))}
    if not all(ok):
        rr = np.sort(np.array(out["G0-1000"]["ratios"], float))
        d52["c"] = float(rr[int(math.ceil(0.95 * len(rr))) - 1])
    out["D5.2"] = d52
    d53 = {}
    for cell in ("G1-300", "G2-300", "G3-300"):
        recs = s4.read_cell(cell)
        rr = np.array([r["d_star"] / r["q"] if r["q"] > 0 else float("inf") for r in recs])
        if d52["validated"]:
            k = int(np.sum(rr <= 1.0))
            ci = s4.wilson(k, len(recs))
            d53[cell] = {"coverage": k / len(recs), "holds": bool(not (ci[1] < 0.95))}
        elif np.isfinite(d52["c"]):
            cov = float(np.mean(rr <= d52["c"]))
            d53[cell] = {"calibrated_coverage": cov, "holds": bool(cov >= 0.90)}
        else:
            d53[cell] = {"holds": False, "note": "calibration impossible"}
    out["D5.3"] = d53
    H = np.array([r["theta_hat"] for r in s4.read_cell("V-5691")])
    bias = np.median(H, 0) - th_new
    se = np.sqrt(1.57 * sd_v ** 2 / len(H) + mc_new ** 2)
    out["D5.4"] = {"bias_over_sd": (np.abs(bias) / sd_v).tolist(),
                   "ci_lower": ((np.abs(bias) - 2 * se) / sd_v).tolist(),
                   "ci_upper": ((np.abs(bias) + 2 * se) / sd_v).tolist(),
                   "flag": bool(np.any((np.abs(bias) - 2 * se) / sd_v > 0.5))}
    return out


if __name__ == "__main__":
    sys.exit(main())
