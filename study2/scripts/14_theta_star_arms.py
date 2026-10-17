"""STEP 5-S4c - pseudo-true theta* for the sensitivity arms G1, G2, G3 (Amendment J.11).

Why: J.10 showed that a theta* built from two k = 4 datasets is too noisy. For G0 it was off by
1.27 SD_V in a (claimed Monte Carlo error 0.56 SD_V), and correcting it raised the measured
coverage. G1-G3 still carry two-dataset theta* values with the same weakness, and D5.3 is scored
against them. This script re-estimates theta* for those three arms the same way, from FRESH
datasets, and re-scores D5.3.

Per arm (seeds `arms/{arm}/k1/{i}` and `arms/{arm}/k4/{i}`, none of them used before):
  - K1 = 12 fresh datasets of 1 x 5,691 drivers, standard fit;
  - K4 = 4 fresh datasets of 4 x 5,691 drivers, standard fit;
  - a strong fit on k4 dataset 0 only, as an optimiser spot check (J.10's R1 found no optimiser
    effect anywhere in G0, so the full strong-fit sweep is not repeated);
  - theta*_arm = k-weighted mean of the 16 standard fits (28 units of data), as J.10's O3 rule;
  - Monte Carlo error = SD of the 12 k = 1 fits / sqrt(28).

Then D5.3 is re-scored: for every replicate of G1-300, G2-300 and G3-300 the Mahalanobis distance
to theta*_arm is recomputed from the STORED bootstrap vectors (no refitting), and compared with
the calibration factor c from the J.10 re-scoring of G0-1000. Raw coverage is reported too.

Writes code/results/step5_arms/{records.jsonl, arms.json, arms.md} (synthetic aggregates only).
Usage (from AINN_ReWORK/scripts):  python 14_theta_star_arms.py            (resumable, ~2-3 h)
                                   python 14_theta_star_arms.py --summary  (re-summarise only)
Keep the laptop plugged in with sleep off.
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

from _common import RESULTS  # noqa: F401

import bridge as br                                     # noqa: E402

_here = os.path.dirname(os.path.abspath(__file__))


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_here, fn))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s4 = _load("s4", "12_bridge_blite.py")
d13 = _load("d13", "13_theta_star_diag.py")

OUT = os.path.join(RESULTS, "step5_arms")
DIAG = os.path.join(RESULTS, "step5_diag")
ARMS = ("G1", "G2", "G3")
K1, K4 = 12, 4
SPOT_STRONG = 0                      # the k = 4 dataset index that also gets a strong fit
MOVE_SD = 0.25                       # J.10 R1 move threshold, for the optimiser spot check


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
    arms = s4.arms_from_generator(g)
    for fn in (os.path.join(DIAG, "diag.json"), os.path.join(s4.OUT, "pseudo_true.json")):
        if not os.path.exists(fn):
            sys.exit(f"STOP - {fn} is missing; run the earlier steps first")
    diag = json.load(open(os.path.join(DIAG, "diag.json")))
    if "rescore" not in diag or "D5.2" not in diag["rescore"]:
        sys.exit("STOP - step5_diag/diag.json has no re-scored D5.2; run 13_theta_star_diag.py first")
    d52 = diag["rescore"]["D5.2"]
    c = None if d52["validated"] else float(d52["c"])
    if c is not None and not math.isfinite(c):
        sys.exit("STOP - the calibration factor is not finite (flat bootstrap directions); "
                 "D5.3 cannot be re-scored")
    pt = json.load(open(os.path.join(s4.OUT, "pseudo_true.json")))
    if s4.fingerprint(gen) != open(os.path.join(s4.OUT, "fingerprint.txt")).read().strip():
        sys.exit("STOP - code, pool or generator differ from the S4 run")
    planned = {cid: R for cid, _a, _N, R, _B in s4.CELLS}
    for arm in ARMS:                                  # the cells D5.3 is re-scored from
        cell = f"{arm}-300"
        cr = s4.read_cell(cell)
        if len(cr) != planned[cell]:
            sys.exit(f"STOP - S4 cell {cell} has {len(cr)} of {planned[cell]} replicates; "
                     f"D5.3 cannot be re-scored")
        if "theta_b" not in cr[0]:
            sys.exit(f"STOP - S4 cell {cell} has no stored bootstrap vectors")
    recs = read_records()
    h = hashlib.md5()
    own = open(os.path.abspath(__file__), "rb").read()
    h.update(own[:own.index(b"def summarise(")])
    for fn in (s4.__file__, os.path.join(s4.SRC, "bridge.py")):
        h.update(open(fn, "rb").read())
    if "--summary" in sys.argv:
        summarise(recs, pt, c, diag)
        return 0
    fpf = os.path.join(OUT, "fingerprint.txt")
    if os.path.exists(fpf) and open(fpf).read().strip() != h.hexdigest():
        sys.exit("STOP - saved records come from different code; move code/results/step5_arms aside")
    open(fpf, "w").write(h.hexdigest() + "\n")

    P = s4.load_pool()
    M = len(P["v0"])
    if M != s4.N_POOL:
        sys.exit(f"STOP - pool has {M} episodes, {s4.N_POOL} expected")
    one = np.arange(M)
    plan = [(arm, k, i) for arm in ARMS for k, n in ((1, K1), (4, K4)) for i in range(n)]
    todo = [p for p in plan if f"{p[0]}_k{p[1]}_{p[2]}" not in recs]
    print(f"[plan] {len(plan)} datasets, {len(todo)} still to do (about 2-3 h in total)", flush=True)
    t0, done = time.time(), 0
    for arm, k, i in plan:
        rid = f"{arm}_k{k}_{i}"
        if rid in recs:
            continue
        lab = f"arms/{arm}/k{k}/{i}"
        t1 = time.time()
        try:
            S, _, info = s4.generate(P, np.tile(one, k), arms[arm], gen, lab)
            assert list(S.per) == [1]
            f = br.fit_blite(S, np.random.default_rng(s4.seed(lab + "/fit")), n_random=2)
        except Exception as e:
            sys.exit(f"STOP - {rid} failed ({e!r}); completed datasets are saved, rerun to resume")
        rec = {"id": rid, "arm": arm, "k": k, "i": i, "N": int(k * M),
               "standard": {"L": f["L"], "theta_typ": f["theta_typ"].tolist()},
               "shared_at_bound": f["shared_at_bound"], "f4": info, "strong": None}
        if k == 4 and i == SPOT_STRONG:                 # optimiser spot check
            warm = [("own_standard", f["theta_typ"].tolist()),
                    ("theta_star_S4", pt[arm]["theta_star"])]
            rec["strong"] = d13.multi_start(S, warm, np.random.default_rng(s4.seed(rid + "/strong")),
                                            2 + d13.EXTRA_RANDOM)
        rec["seconds"] = time.time() - t1
        append(rec)
        recs[rid] = rec
        done += 1
        msg = (f"[{rid}] {rec['seconds'] / 60:.1f} min  a,b "
               f"{np.round(np.array(rec['standard']['theta_typ'])[[3, 4]], 4).tolist()}")
        if rec["strong"]:
            msg += f"  strong dL/L {(rec['standard']['L'] - rec['strong']['L']) / rec['standard']['L']:.2e}"
        el = time.time() - t0
        print(msg + f"   [{done}/{len(todo)} this session, {el / 60:.0f} min, about "
                    f"{el / done * (len(todo) - done) / 60:.0f} min left]", flush=True)
        del S, f
    summarise(recs, pt, c, diag)
    print(f"total {(time.time() - t0) / 60:.0f} min this session", flush=True)
    return 0


def summarise(recs, pt, c, diag):
    out, rows = {}, []
    for arm in ARMS:
        r1 = [r for r in recs.values() if r["arm"] == arm and r["k"] == 1]
        r4 = [r for r in recs.values() if r["arm"] == arm and r["k"] == 4]
        if len(r1) < K1 or len(r4) < K4:
            out[arm] = {"holds": False, "note": f"incomplete: {len(r1)}/{K1} k=1, {len(r4)}/{K4} k=4"}
            print(f"[warn] {arm}: {out[arm]['note']}; not scored", flush=True)
            continue
        A1 = np.array([r["standard"]["theta_typ"] for r in r1])
        A4 = np.array([r["standard"]["theta_typ"] for r in r4])
        units = len(A1) + 4 * len(A4)
        th_new = (A1.sum(0) + 4 * A4.sum(0)) / units
        sd1 = A1.std(0, ddof=1)
        mc = sd1 / math.sqrt(units)
        sd1g = np.where(sd1 > 0, sd1, np.nan)           # a direction pinned at a bound
        spot = next((r for r in r4 if r["strong"]), None)
        cell = f"{arm}-300"
        cr = s4.read_cell(cell)
        H = np.array([r["theta_hat"] for r in cr])
        sd300 = H.std(0, ddof=1)
        sd300 = np.where(sd300 > 0, sd300, np.nan)      # a degenerate direction: report NaN
        th_old = np.array(pt[arm]["theta_star"])
        cov, cal, ratios, old_ratios = [], [], [], []
        for r in cr:
            old_ratios.append(r["d_star"] / r["q"] if r["q"] > 0 else float("inf"))
            reg = br.ellipsoid_region(np.array(r["theta_hat"]), np.array(r["theta_b"]), 0.95)
            if abs(reg["q"] - r["q"]) > 1e-9 * max(1.0, abs(r["q"])):
                sys.exit(f"STOP - {cell} rep {r['rep']}: recomputed q != stored q")
            d = s4.mahal(reg, th_new)
            ratio = d / reg["q"] if reg["q"] > 0 else float("inf")
            ratios.append(ratio)
            cov.append(bool(ratio <= 1.0))
            cal.append(bool(ratio <= c) if c is not None else bool(ratio <= 1.0))
        k = int(sum(cov))
        out[arm] = {"theta_star_S4": th_old.tolist(), "theta_star_new": th_new.tolist(),
                    "mc_error": mc.tolist(), "shift_in_sd300": ((th_new - th_old) / sd300).tolist(),
                    "k1_fits": len(A1), "k4_fits": len(A4), "units": units,
                    "sd_k1": sd1.tolist(), "sd300": sd300.tolist(),
                    "spot_strong_rel_gain": (None if not spot else
                                             (spot["standard"]["L"] - spot["strong"]["L"]) / spot["standard"]["L"]),
                    "spot_strong_move_sd": (None if not spot else
                                            ((np.array(spot["strong"]["theta_typ"])
                                              - np.array(spot["standard"]["theta_typ"])) / (sd1g / 2)).tolist()),
                    "coverage_raw": k / len(cr), "coverage_raw_ci": list(s4.wilson(k, len(cr))),
                    "ratios": ratios, "ratios_J10": old_ratios,
                    "flips_up": int(sum((o > (c or 1.0)) and (n <= (c or 1.0))
                                        for o, n in zip(old_ratios, ratios))),
                    "flips_down": int(sum((o <= (c or 1.0)) and (n > (c or 1.0))
                                          for o, n in zip(old_ratios, ratios))),
                    "f4_accept_first_try": float(np.mean([x["f4"]["accept_first_try"]
                                                          for x in r1 + r4])),
                    "f4_violating": int(sum(x["f4"]["f4_violating_final"] for x in r1 + r4)),
                    "calibrated_coverage": float(np.mean(cal)),
                    "calibrated_coverage_J10": diag.get("rescore", {}).get("D5.3", {}).get(cell, {}),
                    "shared_at_bound_reps": int(sum(bool(r["shared_at_bound"]) for r in cr)),
                    "holds": bool(np.mean(cal) >= 0.90) if c is not None
                    else bool(not (s4.wilson(k, len(cr))[1] < 0.95))}
        rows.append((arm, out[arm]))
    res = {"note": "S4c (Amendment J.11): theta* for G1-G3 re-estimated from fresh data; D5.3 re-scored.",
           "calibration_factor_c": c, "arms": out,
           "D5.3_final": {a: {"calibrated_coverage": v.get("calibrated_coverage"),
                              "holds": v["holds"], "note": v.get("note")} for a, v in out.items()},
           "claim_holds_in_all_arms": (bool(len(out) == len(ARMS) and all(out[a]["holds"] for a in ARMS))
                                       if out else None)}
    json.dump(res, open(os.path.join(OUT, "arms.json"), "w"), indent=2, default=float)
    f4 = lambda v: "[" + ", ".join(f"{x:.4f}" for x in v) + "]"
    L = ["# theta* for the sensitivity arms, and D5.3 re-scored (Amendment J.11)", "",
         f"- calibration factor c (from the J.10 re-scoring of G0-1000): {c}", "",
         "| arm | theta*_S4 | theta*_new | MC error | shift / SD(N=300) |", "|---|---|---|---|---|"]
    for a, v in rows:
        L.append(f"| {a} | {f4(v['theta_star_S4'])} | {f4(v['theta_star_new'])} | {f4(v['mc_error'])} "
                 f"| {f4(v['shift_in_sd300'])} |")
    L += ["", "| arm | raw coverage [CI] | calibrated coverage | J.10 value | flips +/- | holds | "
          "optimiser spot check | F4 first try |", "|---|---|---|---|---|---|---|---|"]
    for a, v in rows:
        oldd = v["calibrated_coverage_J10"]
        old = oldd.get("calibrated_coverage", oldd.get("coverage"))
        sg, mv = v["spot_strong_rel_gain"], v["spot_strong_move_sd"]
        mab = None if mv is None else float(np.nanmax(np.abs(np.array(mv)[[3, 4]])))
        ridge = (mab is not None and sg is not None and sg <= d13.REL_TOL and mab > MOVE_SD)
        L.append(f"| {a} | {v['coverage_raw']:.3f} [{v['coverage_raw_ci'][0]:.3f}, {v['coverage_raw_ci'][1]:.3f}] "
                 f"| {v['calibrated_coverage']:.3f} | {'-' if old is None else f'{old:.3f}'} "
                 f"| +{v['flips_up']} / -{v['flips_down']} | {'yes' if v['holds'] else 'NO'} "
                 f"| gain {'-' if sg is None else f'{sg:.1e}'}, max |move a,b| "
                 f"{'-' if mab is None else f'{mab:.2f}'} SD{', RIDGE' if ridge else ''} "
                 f"| {v['f4_accept_first_try']:.3f} |")
    for a, v in out.items():
        if v.get("note"):
            L.append(f"| {a} | {v['note']} | - | - | - | NO | - | - |")
    L += ["", f"**D5.3 claim holds in all three arms: {res['claim_holds_in_all_arms']}**"]
    open(os.path.join(OUT, "arms.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    sys.exit(main())
