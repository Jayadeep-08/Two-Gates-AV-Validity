"""Shared paths, the frozen episode definition, and helpers for every script in
scripts/.

Every runnable script lives in scripts/ and imports from here, so paths and the
episode definition are defined exactly once. Library code (the IDM, the likelihoods)
stays in code/src/.

LICENCE: anything written under DERIVED contains real trajectories from highD / exiD
and is a "modified version" under the leveLXData licence. It never leaves this
machine. Only aggregate statistics go to RESULTS.
"""

import json
import os
import sys

import numpy as np

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPTS)                        # AINN_ReWORK/
DATASETS = os.path.join(ROOT, "Datasets")
HIGHD_DATA = os.path.join(DATASETS, "highD-dataset-v1.0", "data")
EXID_DATA = os.path.join(DATASETS, "exiD-dataset-v2.1", "data")
DERIVED = os.path.join(ROOT, "derived")                # LOCAL ONLY - never share
RESULTS = os.path.join(ROOT, "code", "results")        # aggregates only - shareable
SRC = os.path.join(ROOT, "code", "src")

if SRC not in sys.path:
    sys.path.insert(0, SRC)

FPS = 25
DT = 1.0 / FPS

# ------------------------------------------------------------------------------------
# EPISODE DEFINITION - frozen in 05_PREREGISTRATION_DRAFT.md Amendments B, C, D and E.
# Changing any value here changes the analysis population and must be recorded as an
# amendment, not edited silently. Every output folder carries a manifest of these
# values, and a run refuses to write into a folder whose manifest differs - so
# episodes from different definitions can never be mixed, even by the resume logic.
# ------------------------------------------------------------------------------------
DEF_VERSION = "E"                   # C = B with F8 in the lane frame (Amendment C)
                                    # D = exiD gap from traveledDistance - FAILED, withdrawn
                                    # E = exiD gap = seam-corrected leadDHW (Amendment E)
EXID_GAP = ("leadDHW - K + median_run(K); K = cumulative sum over lanelet-seam frames of the run "
            "of (d leadDHW + (vF - vL) dt); non-seam frames unmodified; run = consecutive frames, "
            "same pair, no laneChange")
THW_MAX = 5.0                       # F4  s, following regime
TRIM_FRAMES = int(1.0 * FPS)        # F6  trimmed from each side of every break
MIN_FRAMES = int(8.0 * FPS)         # F7  extraction floor; analysis sets use >= 10 s
F9_TAU = 0.3                        # F9  m, kinematic gap consistency (primary)
F9_TAU_SENSITIVITY = 0.2            # F9  m, pre-declared sensitivity value

# Array rows stored per episode, identical order for both datasets.
# gap_ref = the dataset's own precomputed gap (highD dhw, exiD leadDHW), kept for audit.
ARRAY_ROWS = ("frame", "gap", "vF", "vL", "aF", "aL", "u_fF", "u_rL", "gap_ref")

# Fixed histogram edges (m) for frame-level |step| diagnostics (Amendment D.3).
STEP_EDGES = np.r_[np.arange(0.0, 0.5, 0.001), np.arange(0.5, 5.0, 0.01), 5.0, np.inf]


def manifest(dataset, f9_tau):
    return {"definition": DEF_VERSION, "dataset": dataset, "fps": FPS,
            "thw_max": THW_MAX, "trim_frames": TRIM_FRAMES, "min_frames": MIN_FRAMES,
            "f8": "|lateral centre separation in the LANE frame| < (w_L + w_F)/2 (highD |y_L-y_F|; exiD |latLaneCenterOffset_L - latLaneCenterOffset_F|)",
            "f9_tau_m": f9_tau, "array_rows": list(ARRAY_ROWS),
            "gap": (EXID_GAP if dataset == "exid" else "u_rL - u_fF from x positions")}


def output_dir(dataset, f9_tau):
    """derived/<dataset>_episodes_def<V>            primary (F9 tau = 0.3 m)
       derived/<dataset>_episodes_def<V>_f9tau0.2   sensitivity"""
    suffix = "" if f9_tau == F9_TAU else f"_f9tau{f9_tau:g}"
    return os.path.join(DERIVED, f"{dataset}_episodes_def{DEF_VERSION}{suffix}")


def check_or_write_manifest(out_dir, man):
    """Refuse to write into a folder produced under a different definition."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "_manifest.json")
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        if old != man:
            raise RuntimeError(
                f"{out_dir} was produced under a different definition:\n"
                f"  existing: {old}\n  this run: {man}\n"
                "Refusing to mix. Move that folder aside and re-run.")
    else:
        existing = [f for f in os.listdir(out_dir) if f.endswith("_index.csv")]
        if existing:
            raise RuntimeError(f"{out_dir} has episode files but no manifest - "
                               "provenance unknown. Move it aside and re-run.")
        with open(path, "w") as f:
            json.dump(man, f, indent=2)


def default_workers():
    n = os.cpu_count() or 1
    return max(1, min(8, n - 2))


def pending_recordings(all_ids, out_dir, force=False):
    """Recordings whose outputs are missing or empty - makes extraction resumable.
    Safe only because check_or_write_manifest() guarantees the folder holds a single
    definition."""
    if force:
        return list(all_ids)
    todo = []
    for r in all_ids:
        tag = f"{r:02d}"
        need = [f"{tag}_index.csv", f"{tag}_episodes.npz", f"{tag}_index_noF8.csv",
                f"{tag}_index_noF9.csv", f"{tag}_index_noF8F9.csv", f"{tag}_stepdiag.json"]
        if not all(os.path.exists(os.path.join(out_dir, n))
                   and os.path.getsize(os.path.join(out_dir, n)) > 0 for n in need):
            todo.append(r)
    return todo


def gap_consistency_step(gap, vF, vL, same_pair):
    """F9 statistic for each row i, relative to row i-1 of the SAME follower-leader
    pair at the previous frame:

        step_i = (g_i - g_{i-1}) + (vF_{i-1} - vL_{i-1}) * dt

    Zero where row i starts a new pair or skips a frame. Large |step| means the gap
    and the velocities disagree - F9 detects the inconsistency, not which variable is
    wrong; either corrupts the calibration."""
    step = np.zeros(len(gap))
    step[1:] = (gap[1:] - gap[:-1]) + (vF[:-1] - vL[:-1]) * DT
    step[~np.asarray(same_pair, dtype=bool)] = 0.0
    return np.nan_to_num(step, nan=0.0)


def write_stepdiag(out_dir, tag, meta, evaluable, seam, steps, f4_flip=None):
    """Frame-level |step| histograms (Amendment D.3), split by seam / non-seam frame.

    evaluable : bool per row - this row and the previous row are the same pair,
                consecutive, and both satisfy the base rules (F1-F4); F8/F9 NOT applied,
                so the histogram describes the measurement, not the filtered data
    seam      : bool per row - a lanelet change of either vehicle at this row (exiD);
                all False for highD
    steps     : {name: per-row step array}, e.g. {"new": ..., "raw": ...}
    """
    evaluable = np.asarray(evaluable, dtype=bool)
    seam = np.asarray(seam, dtype=bool)
    out = {"rec": meta["rec"], "location": meta["location"], "dataset": meta["dataset"],
           "n_evaluable": int(evaluable.sum()),
           "n_seam": int((evaluable & seam).sum()), "hist": {}}
    for name, st in steps.items():
        a = np.abs(np.nan_to_num(np.asarray(st, dtype=float), nan=np.inf))
        for part, m in (("seam", evaluable & seam), ("noseam", evaluable & ~seam)):
            out["hist"][f"{name}_{part}"] = np.histogram(a[m], bins=STEP_EDGES)[0].tolist()
            out[f"max_{name}_{part}"] = float(a[m].max()) if m.any() else 0.0
            for t in (0.2, 0.3):                       # exact exceedance counts
                out[f"n_{name}_{part}_gt{t:g}"] = int((a[m] > t).sum())
    if f4_flip is not None:
        out["f4_flip"] = f4_flip
    with open(os.path.join(out_dir, f"{tag}_stepdiag.json"), "w") as f:
        json.dump(out, f)
    return out


def segments_to_episodes(valid, breaks, cols, ids, meta, with_arrays=True):
    """Turn a per-frame validity mask into episodes. Shared by both datasets so that
    F5-F7 are applied by literally the same code.

    valid  : bool per row - F2-F4, F8 and F9 already combined by the caller
    breaks : bool per row - True where the row starts a new track / leader / lane or
             skips a frame (structural breaks, independent of validity)
    cols   : per-row float arrays: frame, gap, vF, vL, aF, aL, u_fF, u_rL, thw, dhw
             (reference gap for the consistency check), lat, half_w, step, opt. yaw
    ids    : per-row arrays: follower, leader, leader_class
    Rows must be sorted by (track, frame).

    Every invalid frame ends a segment (F5). F6 trims TRIM_FRAMES from both ends of
    every segment, including ends created by F8 or F9. F7 is applied to each
    resulting segment independently.
    """
    valid = np.asarray(valid, dtype=bool)
    boundary = np.asarray(breaks, dtype=bool) | np.r_[True, valid[1:] != valid[:-1]]
    seg = np.cumsum(boundary)
    rows = np.flatnonzero(valid)
    episodes, arrays = [], {}
    if rows.size == 0:
        return episodes, arrays
    s = seg[rows]
    starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]
    ends = np.r_[starts[1:], rows.size]
    for i0, i1 in zip(starts, ends):
        if i1 - i0 < MIN_FRAMES + 2 * TRIM_FRAMES:
            continue
        r = rows[i0 + TRIM_FRAMES:i1 - TRIM_FRAMES]
        g = {k: v[r] for k, v in cols.items()}
        a0 = r[0]
        ep = f"{meta['prefix']}_{int(ids['follower'][a0])}_{int(ids['leader'][a0])}_{int(g['frame'][0])}"
        row = {"dataset": meta["dataset"], "episode": ep, "rec": meta["rec"],
               "location": meta["location"], "follower": int(ids["follower"][a0]),
               "leader": int(ids["leader"][a0]), "follower_class": "car",
               "leader_class": str(ids["leader_class"][a0]),
               "n_frames": int(r.size), "dur_s": r.size / FPS,
               "gap_check_m": float(np.max(np.abs(g["gap"] - g["dhw"]))),
               "max_lat_offset_m": float(np.max(g["lat"])),
               "max_lat_over_halfw": float(np.max(g["lat"] / g["half_w"])),
               "max_gap_step_m": float(np.max(np.abs(g["step"]))),
               "p95_gap_step_m": float(np.quantile(np.abs(g["step"]), 0.95))}
        if "step_raw" in g:                            # step of the dataset's own gap
            row["max_raw_step_m"] = float(np.max(np.abs(g["step_raw"])))
        for k in g:                                    # run-level constants (exiD, D.3)
            if k.startswith("run_"):
                row[k] = float(g[k][0])
        if "yaw" in g:
            row["max_yaw_rad"] = float(np.max(g["yaw"]))
        if "lat_heading" in g:                         # exiD diagnostic, not a criterion
            row["max_lat_heading_over_halfw"] = float(np.max(g["lat_heading"] / g["half_w"]))
        row.update(episode_stats(g["gap"], g["vF"], g["vL"], g["aF"], g["aL"], g["thw"]))
        episodes.append(row)
        if with_arrays:
            arrays[ep] = np.stack([g[k] for k in ARRAY_ROWS])
    return episodes, arrays


def emit_all_variants(base, f8, f9, breaks, cols, ids, meta, out_dir, tag):
    """Write the primary extraction (base & F8 & F9, with trajectories) and three
    index-only diagnostics: without F8, without F9, and without both. The first two
    differ from the primary by exactly one rule, so primary-vs-diagnostic is that rule's
    MARGINAL loss, independent of rule order. The third gives the JOINT loss; joint vs
    the sum of marginals shows how much the two rules overlap."""
    import pandas as pd
    eps, arrays = segments_to_episodes(base & f8 & f9, breaks, cols, ids, meta, True)
    no8, _ = segments_to_episodes(base & f9, breaks, cols, ids, meta, False)
    no9, _ = segments_to_episodes(base & f8, breaks, cols, ids, meta, False)
    no89, _ = segments_to_episodes(base, breaks, cols, ids, meta, False)
    np.savez_compressed(os.path.join(out_dir, f"{tag}_episodes.npz"), **arrays)
    pd.DataFrame(eps).to_csv(os.path.join(out_dir, f"{tag}_index.csv"), index=False)
    pd.DataFrame(no8).to_csv(os.path.join(out_dir, f"{tag}_index_noF8.csv"), index=False)
    pd.DataFrame(no9).to_csv(os.path.join(out_dir, f"{tag}_index_noF9.csv"), index=False)
    pd.DataFrame(no89).to_csv(os.path.join(out_dir, f"{tag}_index_noF8F9.csv"), index=False)
    return len(eps), len(no8), len(no9), len(no89)


def summarise_extraction(out_dir, dataset, f9_tau, results_name):
    """Totals, integrity checks and the per-location loss table for one dataset."""
    import glob
    import pandas as pd
    read = lambda f: pd.read_csv(f) if os.path.getsize(f) > 1 else pd.DataFrame()
    files = sorted(glob.glob(os.path.join(out_dir, "??_index.csv")))
    main = pd.concat([read(f) for f in files], ignore_index=True)
    no8 = pd.concat([read(f.replace("_index.csv", "_index_noF8.csv")) for f in files],
                    ignore_index=True)
    no9 = pd.concat([read(f.replace("_index.csv", "_index_noF9.csv")) for f in files],
                    ignore_index=True)
    no89 = pd.concat([read(f.replace("_index.csv", "_index_noF8F9.csv")) for f in files],
                     ignore_index=True)
    bad = []
    for f in files:
        ids = set(read(f)["episode"]) if os.path.getsize(f) > 1 else set()
        if set(np.load(f.replace("_index.csv", "_episodes.npz")).files) != ids:
            bad.append(os.path.basename(f)[:2])

    def by_loc(df):
        if df.empty:
            return {}
        g = df.groupby("location")
        return {int(k): {"episodes": int(len(v)), "hours": round(v["dur_s"].sum() / 3600, 3)}
                for k, v in g}

    out = {"dataset": dataset, "definition": DEF_VERSION, "f9_tau_m": f9_tau,
           "recordings_found": len(files), "integrity_failures": bad,
           "episodes": {"primary": int(len(main)), "without_F8": int(len(no8)),
                        "without_F9": int(len(no9)), "without_F8_F9": int(len(no89))},
           "hours": {"primary": round(main["dur_s"].sum() / 3600, 3),
                     "without_F8": round(no8["dur_s"].sum() / 3600, 3),
                     "without_F9": round(no9["dur_s"].sum() / 3600, 3),
                     "without_F8_F9": round(no89["dur_s"].sum() / 3600, 3)},
           "by_location": {"primary": by_loc(main), "without_F8": by_loc(no8),
                           "without_F9": by_loc(no9), "without_F8_F9": by_loc(no89)},
           "checks": {
               "retained_F8_worst_ratio": float(main["max_lat_over_halfw"].max()),
               "retained_F9_worst_step_m": float(main["max_gap_step_m"].max()),
               "gap_check_quantiles_m": main["gap_check_m"].quantile(
                   [.5, .9, .99, 1.0]).round(3).to_dict()}}
    if "max_yaw_rad" in main:
        out["checks"]["yaw_deg_p99_of_episode_max"] = round(
            float(np.degrees(main["max_yaw_rad"].quantile(.99))), 3)
    out["checks"]["pass"] = (not bad and out["checks"]["retained_F8_worst_ratio"] < 1.0
                             and out["checks"]["retained_F9_worst_step_m"] <= f9_tau)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, results_name), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps(out, indent=2, default=str))
    return out


def episode_stats(gap, vF, vL, aF, aL, thw):
    """Per-episode descriptors, computed identically for both datasets. Used for the
    analysis-set definitions and the Step 2 excitation table. None of these is an
    outcome variable of the study."""
    dv = vF - vL
    return {
        "mean_vF": float(vF.mean()), "min_vF": float(vF.min()),
        "max_vF": float(vF.max()), "range_vF": float(np.ptp(vF)),
        "sd_vL": float(vL.std()),
        "sd_aL": float(aL.std()), "min_aL": float(aL.min()),
        "sd_aF": float(aF.std()), "min_aF": float(aF.min()), "max_aF": float(aF.max()),
        "sd_dv": float(dv.std()), "range_dv": float(np.ptp(dv)),
        "sd_gap": float(gap.std()), "mean_gap": float(gap.mean()),
        "min_gap": float(gap.min()),
        "sd_thw": float(thw.std()), "mean_thw": float(thw.mean()),
    }
