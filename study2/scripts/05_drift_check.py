"""STEP 3 - kinematic drift check on highD episodes. Descriptive only; no model is fitted.

Pre-registration: 05_PREREGISTRATION_DRAFT.md B.4 (Step 3). The simulator in Steps 4-5
integrates follower acceleration -> speed -> position and compares the resulting gap with
the observed gap, which highD computes from POSITIONS. Two things can make even a perfect
car-following model miss the observed gap:

  (1) data inconsistency - observed positions are not the integral of observed speeds,
      or observed speeds are not the integral of observed accelerations;
  (2) integrator discretisation - the ballistic update at 25 Hz.

This script measures both, per episode, before any fit:

  dF   follower position vs integral of its own speed      u_fF(t)-u_fF(0) - int vF dt
  dL   leader  position vs integral of its own speed       u_rL(t)-u_rL(0) - int vL dt
  dvF  follower speed vs integral of its own acceleration  vF(t)-vF(0) - int aF dt
  dvL  leader  speed vs integral of its own acceleration
  RF   REPLAY: the observed follower acceleration fed through the simulator's ballistic
       update (v += a dt; x += v dt + a dt^2/2) from the observed initial state,
       compared with the observed follower position. This is the gap error a model that
       reproduced the follower's acceleration EXACTLY would still make.
  RFv  the same replay, but driven by observed speed (x += v dt): isolates the
       acceleration channel.

Decision already fixed in B.4: the leader enters the simulator through its observed
POSITION u_rL(t), never through its integrated speed, so dL never enters the simulated
gap. dL is reported to show what a speed-driven leader would have cost.
Reported per episode as max |.| over the episode and |.| at 10 s, for the Amendment F
site groups and the analysis sets. No threshold is applied here; how the replay error
enters the likelihood is decided in Step 4 on synthetic data, before any real fit.

Reads   derived/highd_episodes_defE/??_episodes.npz, ??_index.csv
Writes  code/results/step3_drift_highd.json / .md

Usage (from AINN_ReWORK/scripts):   python 05_drift_check.py
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

from _common import ARRAY_ROWS, DERIVED, DT, RESULTS

R = {k: i for i, k in enumerate(ARRAY_ROWS)}
GROUPS = {"loc1_primary": [1], "loc5_replication": [5], "other_2346": [2, 3, 4, 6]}
N10 = int(round(10 / DT))


def cumtrapz(y):
    out = np.zeros_like(y)
    out[1:] = np.cumsum((y[1:] + y[:-1]) * 0.5 * DT)
    return out


def replay(x0, v0, acc):
    """Ballistic update exactly as in code/src/idm.py simulate_follower."""
    n = len(acc)
    x, v = np.empty(n), np.empty(n)
    x[0], v[0] = x0, v0
    for t in range(n - 1):
        x[t + 1] = x[t] + v[t] * DT + 0.5 * acc[t] * DT * DT
        v[t + 1] = v[t] + acc[t] * DT
    return x, v


def episode_metrics(a):
    uF, uL, vF, vL, aF, aL = (a[R[k]] for k in ("u_fF", "u_rL", "vF", "vL", "aF", "aL"))
    dF = (uF - uF[0]) - cumtrapz(vF)
    dL = (uL - uL[0]) - cumtrapz(vL)
    dvF = (vF - vF[0]) - cumtrapz(aF)
    dvL = (vL - vL[0]) - cumtrapz(aL)
    xr, _ = replay(uF[0], vF[0], aF)
    RF = xr - uF
    RFv = np.r_[uF[0], uF[0] + np.cumsum(vF[:-1] * DT)] - uF
    m = {}
    for k, s in (("dF", dF), ("dL", dL), ("dvF", dvF), ("dvL", dvL), ("RF", RF), ("RFv", RFv)):
        m[f"{k}_max"] = float(np.max(np.abs(s)))
        m[f"{k}_10s"] = float(abs(s[min(N10, len(s) - 1)]))
    return m


def main():
    d = os.path.join(DERIVED, "highd_episodes_defE")
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "??_index.csv"))):
        if os.path.getsize(f) <= 1:
            continue
        ix = pd.read_csv(f)
        ix = ix[(ix["leader_class"] == "car") & (ix["dur_s"] >= 10)]
        if ix.empty:
            continue
        arrs = np.load(f.replace("_index.csv", "_episodes.npz"))
        for _, e in ix.iterrows():
            m = episode_metrics(arrs[e["episode"]])
            m.update(episode=e["episode"], location=int(e["location"]), sd_vL=float(e["sd_vL"]),
                     dur_s=float(e["dur_s"]))
            rows.append(m)
        print(f"{os.path.basename(f)[:2]}: {len(ix)} episodes", flush=True)
    df = pd.DataFrame(rows)
    keys = ["dF", "dL", "dvF", "dvL", "RF", "RFv"]
    res = {"dataset": "highD", "note": "descriptive only; B.4 Step 3", "units": "m (dF, dL, RF, RFv), m/s (dvF, dvL)",
           "table": {}}
    for g, locs in GROUPS.items():
        for sname, x in (("carled_10s", 0.0), ("E10,1.0", 1.0)):
            sub = df[df["location"].isin(locs) & (df["sd_vL"] >= x)]
            if sub.empty:
                continue
            res["table"][f"{g}|{sname}"] = {"episodes": int(len(sub)), **{
                f"{k}_{w}": {str(p): round(float(sub[f'{k}_{w}'].quantile(p)), 4) for p in (0.5, 0.9, 0.99)}
                for k in keys for w in ("10s", "max")}}
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "step3_drift_highd.json"), "w") as f:
        json.dump(res, f, indent=2)
    L = ["# Step 3 - highD kinematic drift (descriptive, no fit)", "",
         "|.| at 10 s into the episode; median / p90 / p99 over episodes.", "",
         "| group | n | " + " | ".join(keys) + " |", "|---" * (len(keys) + 2) + "|"]
    for g, t in res["table"].items():
        L.append(f"| {g} | {t['episodes']} | " + " | ".join(
            f"{t[k + '_10s']['0.5']} / {t[k + '_10s']['0.9']} / {t[k + '_10s']['0.99']}" for k in keys) + " |")
    L += ["", "max |.| over the whole episode; median / p90 / p99.", "",
          "| group | n | " + " | ".join(keys) + " |", "|---" * (len(keys) + 2) + "|"]
    for g, t in res["table"].items():
        L.append(f"| {g} | {t['episodes']} | " + " | ".join(
            f"{t[k + '_max']['0.5']} / {t[k + '_max']['0.9']} / {t[k + '_max']['0.99']}" for k in keys) + " |")
    md = "\n".join(L) + "\n"
    with open(os.path.join(RESULTS, "step3_drift_highd.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
