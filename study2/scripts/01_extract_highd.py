"""STEP 1a - extract car-following episodes from highD.   Episode definition E.

highD's rules are identical under definitions C, D and E (D and E change only the exiD
gap). The E extraction must reproduce C exactly - a regression check on the shared code.

EPISODE DEFINITION (05_PREREGISTRATION_DRAFT.md Amendments B, C, E)
  F1 follower class Car
  F2 same leader (precedingId > 0) for the whole episode
  F3 follower does not change lane (laneId constant). A leader changing lane changes
     precedingId, which ends the episode (preceding vehicle is in the follower's lane
     in 100% of matched rows - DIARY Entry 10)
  F4 following regime: dhw > 0 and 0 < thw <= 5.0 s at every frame
  F5 contiguous frames
  F6 1.0 s trimmed from each end of every segment
  F7 >= 8.0 s after trimming  (analysis sets E10,x additionally require >= 10 s)
  F8 lateral-path validity: at every frame, |lateral centre separation| <
     (w_leader + w_follower) / 2, using each vehicle's own width. No fixed offset.
  F9 kinematic gap consistency: for consecutive frames of the same follower-leader
     pair, |dg + dv*dt| <= 0.3 m, dv = v_follower - v_leader at the earlier frame,
     dt = 0.04 s. Violating frames are excluded, never repaired. 0.2 m = sensitivity.

  F2-F4, F8 and F9 enter ONE per-frame validity mask. Any invalid frame ends a segment;
  F6 then trims 1 s from both sides of the break, and F7 is applied to every
  resulting segment independently. So an F8 violation splits an episode exactly as a
  leader change does, with the same transient trimming either side.

LATERAL SEPARATION IN highD
  highD has no heading column. The road is straight and aligned with x, so the
  follower-frame lateral separation is |y_centre_L - y_centre_F|, with y_centre =
  y + height/2 and vehicle width = 'height' (highD's 'width' is length along x).
  The approximation error is (1 - cos(yaw)), measured and reported in DIARY Entry 14.

DIAGNOSTIC OUTPUTS FOR F8 AND F9 (and <rec>_stepdiag.json: frame-level |step| histograms
of the gap used and of highD's own dhw - the highD reference for Amendment D.3)
  Each recording also writes <rec>_index_noF8.csv and <rec>_index_noF9.csv: the same
  extraction with exactly one rule removed, index rows only. Primary vs diagnostic is
  that rule's MARGINAL loss, reported by dataset and location.
  Every output folder has a _manifest.json of the definition; a run refuses to write
  into a folder produced under a different definition.

COORDINATES (measured on recording 01, DIARY Entry 10)
  dir 2: xVelocity > 0, u = x ;  dir 1: xVelocity < 0, u = -x
  follower front u_fF = x + width (dir 2) | -x (dir 1)
  leader rear    u_rL = x_L       (dir 2) | -(x_L + w_L)
  gap = u_rL - u_fF == dhw to within 0.010 m

Usage (from AINN_ReWORK/scripts):
  python 01_extract_highd.py                 # all 60 recordings, parallel, resumable
  python 01_extract_highd.py --f9-tau 0.2    # F9 sensitivity extraction (separate folder)
  python 01_extract_highd.py --rec 1-3 --workers 1
  python 01_extract_highd.py --summarise
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from _common import (DEF_VERSION, F9_TAU, FPS, HIGHD_DATA, THW_MAX,
                     check_or_write_manifest, default_workers, emit_all_variants,
                     gap_consistency_step, manifest, output_dir, pending_recordings,
                     summarise_extraction, write_stepdiag)

DATASET = "highd"
COLS = ["frame", "id", "x", "y", "width", "height", "xVelocity", "yVelocity",
        "xAcceleration", "dhw", "thw", "precedingId", "laneId"]


def extract_recording(rec, f9_tau=F9_TAU):
    tag = f"{rec:02d}"
    tr = pd.read_csv(os.path.join(HIGHD_DATA, f"{tag}_tracks.csv"), usecols=COLS)
    tm = pd.read_csv(os.path.join(HIGHD_DATA, f"{tag}_tracksMeta.csv")).set_index("id")
    rm = pd.read_csv(os.path.join(HIGHD_DATA, f"{tag}_recordingMeta.csv")).iloc[0]
    if int(rm["frameRate"]) != FPS:
        raise RuntimeError(f"rec {rec}: frameRate {rm['frameRate']} != {FPS}")

    tr["dir"] = tr["id"].map(tm["drivingDirection"]).astype(int)
    tr["cls"] = tr["id"].map(tm["class"])
    tr["yc"] = tr["y"] + tr["height"] / 2.0

    lead = tr[["frame", "id", "x", "width", "height", "yc", "xVelocity",
               "xAcceleration", "cls"]].rename(columns={
        "id": "precedingId", "x": "xL", "width": "lenL", "height": "widL", "yc": "ycL",
        "xVelocity": "vxL", "xAcceleration": "axL", "cls": "clsL"})
    tr = tr.merge(lead, on=["frame", "precedingId"], how="left")
    tr = tr.sort_values(["id", "frame"]).reset_index(drop=True)

    lat = (tr["yc"] - tr["ycL"]).abs()
    half_sum_w = (tr["height"] + tr["widL"]) / 2.0
    base = ((tr["precedingId"] > 0) & tr["xL"].notna() & (tr["dhw"] > 0)
            & (tr["thw"] > 0) & (tr["thw"] <= THW_MAX) & (tr["cls"] == "Car"))
    f8 = lat < half_sum_w

    d = tr["dir"].to_numpy()
    sgn = np.where(d == 2, 1.0, -1.0)
    x, w = tr["x"].to_numpy(), tr["width"].to_numpy()
    xL, lenL = tr["xL"].to_numpy(), tr["lenL"].to_numpy()
    cols = {
        "frame": tr["frame"].to_numpy(dtype=float),
        "u_fF": np.where(d == 2, x + w, -x),
        "u_rL": np.where(d == 2, xL, -(xL + lenL)),
        "vF": sgn * tr["xVelocity"].to_numpy(), "vL": sgn * tr["vxL"].to_numpy(),
        "aF": sgn * tr["xAcceleration"].to_numpy(), "aL": sgn * tr["axL"].to_numpy(),
        "thw": tr["thw"].to_numpy(), "dhw": tr["dhw"].to_numpy(),
        "lat": lat.to_numpy(), "half_w": half_sum_w.to_numpy(),
        "yaw": np.abs(np.arctan2(tr["yVelocity"].to_numpy(),
                                 np.abs(tr["xVelocity"].to_numpy()))),
    }
    cols["gap"] = cols["u_rL"] - cols["u_fF"]
    ids = {"follower": tr["id"].to_numpy(), "leader": tr["precedingId"].to_numpy(),
           "leader_class": tr["clsL"].astype(str).str.lower().to_numpy()}
    breaks = (tr["id"] != tr["id"].shift(1)) | (tr["frame"] != tr["frame"].shift(1) + 1) \
        | (tr["precedingId"] != tr["precedingId"].shift(1)) \
        | (tr["laneId"] != tr["laneId"].shift(1))
    meta = {"dataset": "highD", "rec": rec, "location": int(rm["locationId"]),
            "prefix": f"hD{tag}"}
    breaks = breaks.to_numpy()
    cols["gap_ref"] = cols["dhw"]
    cols["step"] = gap_consistency_step(cols["gap"], cols["vF"], cols["vL"], ~breaks)
    cols["step_raw"] = gap_consistency_step(cols["gap_ref"], cols["vF"], cols["vL"], ~breaks)
    f9 = np.abs(cols["step"]) <= f9_tau

    out = output_dir(DATASET, f9_tau)
    b = base.to_numpy()
    evaluable = ~breaks & b & np.r_[False, b[:-1]]
    write_stepdiag(out, tag, meta, evaluable, np.zeros(len(b), bool),
                   {"used": cols["step"], "ref": cols["step_raw"]})
    n = emit_all_variants(b, f8.to_numpy(), f9, breaks, cols, ids, meta, out, tag)
    return (rec,) + n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", type=str, default="1-60")
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--f9-tau", type=float, default=F9_TAU)
    ap.add_argument("--force", action="store_true", help="redo recordings already done")
    ap.add_argument("--summarise", action="store_true")
    a = ap.parse_args()
    out = output_dir(DATASET, a.f9_tau)
    results_name = f"step1_highd_def{DEF_VERSION}_f9tau{a.f9_tau:g}.json"
    if a.summarise:
        return summarise_extraction(out, "highD", a.f9_tau, results_name)

    check_or_write_manifest(out, manifest(DATASET, a.f9_tau))
    lo, _, hi = a.rec.partition("-")
    todo = pending_recordings(range(int(lo), int(hi or lo) + 1), out, a.force)
    print(f"highD (definition {DEF_VERSION}, F9 tau {a.f9_tau} m): {len(todo)} "
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
    summarise_extraction(out, "highD", a.f9_tau, results_name)


if __name__ == "__main__":
    sys.exit(main())
