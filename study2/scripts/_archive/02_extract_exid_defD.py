"""STEP 1b - extract car-following episodes from exiD.   Episode definition D.

highD's rules do NOT transfer directly; exiD's representation differs. Conventions
measured on recording 00 (DIARY Entry 13), not assumed:
  - leadDHW is the LONGITUDINAL bumper-to-bumper gap: measured along the follower's
    heading (centre separation minus half lengths) it agrees to 0.012 m median.
    Euclidean centre distance does not agree, because it includes lateral offset.
  - leadDV = v_follower - v_leader;  leadTHW = leadDHW / lonVelocity
  - leadId = -1 when there is no leader
  - lonVelocity >= 0 and equals |velocity|; heading is in DEGREES
  - laneChange is a 0/1 event at the crossing frame
  - laneletId is NOT a lane-continuity signal (only 21% of leader rows share it)
  - 13% of follower lane changes occur without a leadId change within +-1 s, so the
    laneChange flag of BOTH vehicles is used in addition to leader continuity

EPISODE DEFINITION D (05_PREREGISTRATION_DRAFT.md Amendments B, C and D)
  F1  follower class car
  F2  same leader (leadId >= 0) for the whole episode
  F3  no laneChange event by the follower or the leader at any retained frame
  F4  following regime: g > 0 and 0 < g / v_F <= 5.0 s at every frame (g = reconstructed gap)
  F5  contiguous frames
  F6  1.0 s trimmed from each end of every segment
  F7  >= 8.0 s after trimming  (analysis sets E10,x additionally require >= 10 s)
  F8  lateral-path validity: at every frame, |lateral centre separation IN THE LANE
      FRAME| < (w_leader + w_follower) / 2, using each vehicle's own width.
      Lane frame = |latLaneCenterOffset_L - latLaneCenterOffset_F| (Amendment C).
  F9  kinematic gap consistency on the reconstructed gap g: for consecutive frames of
      the same pair, |dg + dv*dt| <= 0.3 m (0.2 m sensitivity). Unchanged rule; under D
      it is expected to be a near-no-op check, as in highD.

RECONSTRUCTED GAP (Amendment D, DIARY Entries 18-19)
  leadDHW is computed per lanelet and shifts level by 0.3-1 m at lanelet seams; under
  definition C, F9 therefore removed exiD data selectively by excitation. Under D:
    run    = maximal consecutive frames, same follower-leader pair, both present, no
             laneChange event of either vehicle
    r(t)   = leadDHW(t) - (D_L(t) - D_F(t)),   D = traveledDistance
    C_run  = median of r over the run (frames with leadDHW > 0)
    g(t)   = D_L(t) - D_F(t) + C_run
  The offset belongs to the pair's run, not the episode: traveledDistance origins are
  fixed per vehicle, and episode boundaries depend on F4, which depends on g.
  g is a DERIVED measurement, not ground truth: its level comes from leadDHW (and
  inherits its seam ambiguity), its time variation from traveledDistance.
  leadDHW is kept per frame as array row `gap_ref`.

  Run-level diagnostics carried into every episode (index columns run_*):
    run_C, run_mad, run_iqr, run_nseams,
    run_within_drift   max over seam segments >= 2 s of the robust range (p97.5-p2.5) of r
    run_anchor_spread  max - min of the seam-segment medians of r (segments >= 1 s)
  Per recording also: <rec>_seamseg.csv (one row per seam segment) and
  <rec>_stepdiag.json (frame-level |step| histograms of g and of leadDHW, split by
  seam / non-seam frames, and the F4 flip count raw vs reconstructed).

  F2-F4, F8 and F9 form ONE per-frame validity mask; any invalid frame ends a segment and
  the shared segments_to_episodes() applies F5-F7 - the same code as for highD.

LATERAL SEPARATION IN exiD  (Amendment C, DIARY Entries 16-17)
  Measured in the LANE's frame:  lat = | latLaneCenterOffset_L - latLaneCenterOffset_F |
  This is the same quantity as highD's |y_L - y_F|, which is lane-relative because
  highD's lanes are straight and parallel. Definition B measured lateral separation
  perpendicular to the follower's heading; on curves a same-lane leader then appears
  displaced by ~gap^2/(2R), and F8 discarded 28-43% of frames at the curved locations
  5 and 6 for that reason alone (correlation with the pure-curvature prediction
  0.95-0.97). The heading-frame value is still computed, as `lat_heading`, for the
  diagnostic column max_lat_heading_over_halfw only; it plays no part in F8.
  The lane-frame measure relies on exiD's own lane assignment (leadId, via lanelet
  topology) to establish that both offsets refer to the same lane - the same
  assumption highD makes through precedingId / laneId.

ALONG-TRACK COORDINATE
  u_fF = traveledDistance_F + length_F / 2 ;  u_rL = u_fF + g
  (the informative Step 3 check is follower velocity-vs-position drift.)

DIAGNOSTIC OUTPUTS: <rec>_index_noF8.csv and <rec>_index_noF9.csv (index only, one rule
removed each) - the marginal loss of each rule. Output folders carry a _manifest.json;
a run refuses to write into a folder produced under a different definition.

Usage (from AINN_ReWORK/scripts):
  python 02_extract_exid.py                  # all 93 recordings, parallel, resumable
  python 02_extract_exid.py --f9-tau 0.2     # F9 sensitivity extraction (separate folder)
  python 02_extract_exid.py --rec 0-1 --workers 1
  python 02_extract_exid.py --summarise
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from _common import (DEF_VERSION, EXID_DATA, F9_TAU, FPS, THW_MAX,
                     check_or_write_manifest, default_workers, emit_all_variants,
                     gap_consistency_step, manifest, output_dir, pending_recordings,
                     summarise_extraction, write_stepdiag)

DATASET = "exid"
COLS = ["frame", "trackId", "xCenter", "yCenter", "heading", "width", "length",
        "lonVelocity", "lonAcceleration", "traveledDistance", "laneChange",
        "leadDHW", "leadTHW", "leadId", "latLaneCenterOffset", "laneletId"]
SEAM_MIN_ANCHOR = 25        # frames (1 s)  - seam segments used for the anchor spread
SEAM_MIN_DRIFT = 50         # frames (2 s)  - seam segments used for within-seam drift


def extract_recording(rec, f9_tau=F9_TAU):
    tag = f"{rec:02d}"
    tr = pd.read_csv(os.path.join(EXID_DATA, f"{tag}_tracks.csv"), usecols=COLS)
    tm = pd.read_csv(os.path.join(EXID_DATA, f"{tag}_tracksMeta.csv")).set_index("trackId")
    rm = pd.read_csv(os.path.join(EXID_DATA, f"{tag}_recordingMeta.csv")).iloc[0]
    if int(rm["frameRate"]) != FPS:
        raise RuntimeError(f"rec {rec}: frameRate {rm['frameRate']} != {FPS}")

    tr["cls"] = tr["trackId"].map(tm["class"]).astype(str).str.lower()
    lead = tr[["frame", "trackId", "xCenter", "yCenter", "width", "length",
               "lonVelocity", "lonAcceleration", "laneChange", "cls",
               "latLaneCenterOffset", "traveledDistance", "laneletId"]].rename(columns={
        "trackId": "leadId", "xCenter": "xL", "yCenter": "yL", "width": "widL",
        "length": "lenL", "lonVelocity": "vL", "lonAcceleration": "aL",
        "laneChange": "lcL", "cls": "clsL", "latLaneCenterOffset": "offL",
        "traveledDistance": "tdL", "laneletId": "llL"})
    tr = tr.merge(lead, on=["frame", "leadId"], how="left")
    tr = tr.sort_values(["trackId", "frame"]).reset_index(drop=True)
    n = len(tr)

    h = np.deg2rad(tr["heading"].to_numpy())
    dx = (tr["xL"] - tr["xCenter"]).to_numpy()
    dy = (tr["yL"] - tr["yCenter"]).to_numpy()
    lat_heading = np.abs(-np.sin(h) * dx + np.cos(h) * dy)      # diagnostic only
    lat = np.abs((tr["offL"] - tr["latLaneCenterOffset"]).to_numpy())  # F8 (lane frame)
    half_w = (tr["width"] + tr["widL"]).to_numpy() / 2.0

    breaks = ((tr["trackId"] != tr["trackId"].shift(1))
              | (tr["frame"] != tr["frame"].shift(1) + 1)
              | (tr["leadId"] != tr["leadId"].shift(1))).to_numpy()
    present = ((tr["leadId"] >= 0) & tr["xL"].notna() & tr["tdL"].notna()).to_numpy()
    lc = ((tr["laneChange"] == 1) | (tr["lcL"].fillna(0) == 1)).to_numpy()

    # ---- runs and the reconstructed gap (Amendment D.1) --------------------------
    run_start = breaks | lc | np.r_[True, lc[:-1]] | ~present | np.r_[True, ~present[:-1]]
    run_id = np.cumsum(run_start)
    dD = (tr["tdL"] - tr["traveledDistance"]).to_numpy()
    lead_dhw = tr["leadDHW"].to_numpy()
    r = lead_dhw - dD
    anchor = present & ~lc & (lead_dhw > 0) & np.isfinite(r)
    rs = pd.Series(np.where(anchor, r, np.nan))
    C_run = rs.groupby(run_id).transform("median").to_numpy()
    gap = dD + C_run                                    # NaN where the run has no anchor

    # ---- seam segments and run-level diagnostics (Amendment D.3) ------------------
    llF, llL = tr["laneletId"].to_numpy(), tr["llL"].to_numpy()
    seam = np.r_[False, (llF[1:] != llF[:-1]) | (llL[1:] != llL[:-1])] & ~run_start
    seg_id = np.cumsum(run_start | seam)
    sd = pd.DataFrame({"run": run_id, "seg": seg_id, "r": rs})
    sd = sd[anchor]
    g = sd.groupby("seg")
    seg = pd.DataFrame({"run": g["run"].first(), "n": g["r"].size(),
                        "median": g["r"].median(), "q025": g["r"].quantile(0.025),
                        "q975": g["r"].quantile(0.975), "min": g["r"].min(),
                        "max": g["r"].max()})
    seg["robust_range"] = seg["q975"] - seg["q025"]
    seg["full_range"] = seg["max"] - seg["min"]
    gr = sd.groupby("run")["r"]
    med = gr.transform("median")
    run = pd.DataFrame({"run_C": gr.median(),
                        "run_mad": (sd["r"] - med).abs().groupby(sd["run"]).median(),
                        "run_iqr": gr.quantile(0.75) - gr.quantile(0.25),
                        "run_nseams": seg.groupby("run").size() - 1})
    sa = seg[seg["n"] >= SEAM_MIN_ANCHOR].groupby("run")["median"]
    run["run_anchor_spread"] = (sa.max() - sa.min()).reindex(run.index).fillna(0.0)
    run["run_within_drift"] = (seg[seg["n"] >= SEAM_MIN_DRIFT].groupby("run")
                               ["robust_range"].max().reindex(run.index))

    # ---- rules -------------------------------------------------------------------
    vF = tr["lonVelocity"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        thw = np.where(vF > 0, gap / vF, np.inf)
        f4_new = (gap > 0) & (thw > 0) & (thw <= THW_MAX)
        f4_raw = ((tr["leadDHW"] > 0) & (tr["leadTHW"] > 0)
                  & (tr["leadTHW"] <= THW_MAX)).to_numpy()
        f8 = lat < half_w                                  # NaN (no leader) -> False
    core = present & (tr["cls"] == "car").to_numpy() & ~lc & np.isfinite(gap)
    base = core & f4_new

    u_fF = (tr["traveledDistance"] + tr["length"] / 2).to_numpy()
    cols = {"frame": tr["frame"].to_numpy(dtype=float), "gap": gap,
            "vF": vF, "vL": tr["vL"].to_numpy(),
            "aF": tr["lonAcceleration"].to_numpy(), "aL": tr["aL"].to_numpy(),
            "u_fF": u_fF, "u_rL": u_fF + gap, "thw": thw,
            "dhw": lead_dhw,             # gap_check_m = max |g - leadDHW| (seam shifts)
            "gap_ref": lead_dhw,
            "lat": lat, "half_w": half_w, "lat_heading": lat_heading}
    for k in run.columns:
        cols[k] = run[k].reindex(run_id).to_numpy()
    ids = {"follower": tr["trackId"].to_numpy(), "leader": tr["leadId"].to_numpy(),
           "leader_class": tr["clsL"].fillna("none").to_numpy()}
    meta = {"dataset": "exiD", "rec": rec, "location": int(rm["locationId"]),
            "prefix": f"eX{tag}"}

    cols["step"] = gap_consistency_step(cols["gap"], cols["vF"], cols["vL"], ~breaks)
    cols["step_raw"] = gap_consistency_step(lead_dhw, cols["vF"], cols["vL"], ~breaks)
    f9 = np.abs(cols["step"]) <= f9_tau

    out = output_dir(DATASET, f9_tau)
    evaluable = ~breaks & base & np.r_[False, base[:-1]]
    flip = core & (f4_new != f4_raw)
    write_stepdiag(out, tag, meta, evaluable, seam,
                   {"used": cols["step"], "ref": cols["step_raw"]},
                   f4_flip={"n_core_frames": int(core.sum()),
                            "n_flip": int(flip.sum()),
                            "n_raw_only": int((core & f4_raw & ~f4_new).sum()),
                            "n_new_only": int((core & f4_new & ~f4_raw).sum())})
    n = emit_all_variants(base, f8, f9, breaks, cols, ids, meta, out, tag)

    prim = pd.read_csv(os.path.join(out, f"{tag}_index.csv")) \
        if os.path.getsize(os.path.join(out, f"{tag}_index.csv")) > 1 else pd.DataFrame()
    prim_runs = set()
    if not prim.empty:                    # runs contributing >= 1 primary episode
        rows = pd.DataFrame({"f": ids["follower"], "l": ids["leader"],
                             "fr": cols["frame"].astype(int), "run": run_id})
        key = prim["episode"].str.rsplit("_", n=3, expand=True)
        starts = pd.DataFrame({"f": key[1].astype(int), "l": key[2].astype(int),
                               "fr": key[3].astype(int)})
        prim_runs = set(starts.merge(rows, on=["f", "l", "fr"])["run"])
    seg.insert(0, "location", meta["location"])
    seg.insert(0, "rec", rec)
    seg["run_in_primary"] = seg["run"].isin(prim_runs)
    seg.to_csv(os.path.join(out, f"{tag}_seamseg.csv"), index=False)
    return (rec,) + n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", type=str, default="0-92")
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--f9-tau", type=float, default=F9_TAU)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--summarise", action="store_true")
    a = ap.parse_args()
    out = output_dir(DATASET, a.f9_tau)
    results_name = f"step1_exid_def{DEF_VERSION}_f9tau{a.f9_tau:g}.json"
    if a.summarise:
        return summarise_extraction(out, "exiD", a.f9_tau, results_name)

    check_or_write_manifest(out, manifest(DATASET, a.f9_tau))
    lo, _, hi = a.rec.partition("-")
    todo = pending_recordings(range(int(lo), int(hi or lo) + 1), out, a.force)
    print(f"exiD (definition {DEF_VERSION}, F9 tau {a.f9_tau} m): {len(todo)} "
          f"recordings to do, {a.workers} workers -> {out}", flush=True)
    t0 = time.time()
    report = lambda r, n, n8, n9, n89: print(
        f"rec {r:02d}: {n} episodes | without F8: {n8} | without F9: {n9} | "
        f"without both: {n89}", flush=True)
    if a.workers == 1:
        for r in todo:
            report(*extract_recording(r, a.f9_tau))
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(extract_recording, r, a.f9_tau) for r in todo]
            for fu in as_completed(futs):
                report(*fu.result())
    print(f"finished in {time.time() - t0:.0f} s", flush=True)
    summarise_extraction(out, "exiD", a.f9_tau, results_name)


if __name__ == "__main__":
    sys.exit(main())
