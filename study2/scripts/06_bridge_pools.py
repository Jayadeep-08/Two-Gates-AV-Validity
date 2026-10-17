"""STEP 4a - build the real-leader-trace pools for the synthetic bridge (Amendment G).

Pools (highD, definition E == C, analysis set E10,1.0, car-led):
  loc1     all location-1 episodes                         (primary population, F.1)
  loc1_ff  location-1 episodes with min v_F >= 15 m/s      (free-flow stratum, F.6)
  loc5     all location-5 episodes                         (replication site, F.2)

Per episode the pool keeps only what the bridge needs:
  ul   leader rear position u_rL(t)   - drives the simulated follower (Step 3)
  vl   leader speed v_L(t)            - IDM approach term
  vF0  observed follower initial speed - synthetic initial state
  rf   replay drift RF(t) = x_replay(t) - u_fF(t): the observed follower acceleration
       fed through the simulator's integrator, minus the observed position (Step 3).
       Added to the synthetic gap in the drift arm (G, arm D).

LICENCE: the pool contains real highD trajectories. It is written under derived/ and
never leaves this machine.

Usage (from AINN_ReWORK/scripts):   python 06_bridge_pools.py
"""

import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

from _common import ARRAY_ROWS, DERIVED, DT, RESULTS

R = {k: i for i, k in enumerate(ARRAY_ROWS)}
SRC = os.path.join(DERIVED, "highd_episodes_defE")
OUT = os.path.join(DERIVED, "bridge_pool_defE.npz")


def replay_drift(a):
    uF, vF, aF = a[R["u_fF"]], a[R["vF"]], a[R["aF"]]
    n = len(aF)
    x = np.empty(n)
    x[0], v = uF[0], vF[0]
    for t in range(n - 1):
        x[t + 1] = x[t] + v * DT + 0.5 * aF[t] * DT * DT
        v = v + aF[t] * DT
    return x - uF


def main():
    rows, ul, vl, rf, lens = [], [], [], [], []
    for f in sorted(glob.glob(os.path.join(SRC, "??_index.csv"))):
        if os.path.getsize(f) <= 1:
            continue
        ix = pd.read_csv(f)
        ix = ix[(ix["leader_class"] == "car") & (ix["dur_s"] >= 10) & (ix["sd_vL"] >= 1.0)
                & ix["location"].isin([1, 5])]
        if ix.empty:
            continue
        arrs = np.load(f.replace("_index.csv", "_episodes.npz"))
        for _, e in ix.iterrows():
            a = arrs[e["episode"]]
            ul.append(a[R["u_rL"]]); vl.append(a[R["vL"]]); rf.append(replay_drift(a))
            lens.append(a.shape[1])
            rows.append({"episode": e["episode"], "location": int(e["location"]),
                         "vF0": float(a[R["vF"], 0]), "min_vF": float(e["min_vF"]),
                         "n": int(a.shape[1])})
    meta = pd.DataFrame(rows)
    off = np.r_[0, np.cumsum(lens)].astype(np.int64)
    pools = {"loc1": np.flatnonzero(meta["location"] == 1),
             "loc1_ff": np.flatnonzero((meta["location"] == 1) & (meta["min_vF"] >= 15.0)),
             "loc5": np.flatnonzero(meta["location"] == 5)}
    np.savez(OUT, ul=np.concatenate(ul), vl=np.concatenate(vl), rf=np.concatenate(rf), off=off,
             vF0=meta["vF0"].to_numpy(), episode=meta["episode"].to_numpy().astype(str),
             **{f"pool_{k}": v for k, v in pools.items()})
    h = hashlib.md5(open(OUT, "rb").read()).hexdigest()
    summ = {"file": os.path.relpath(OUT, os.path.dirname(DERIVED)), "md5": h,
            "episodes": int(len(meta)), "steps": int(off[-1]),
            "pools": {k: {"episodes": int(len(v)), "mean_steps": round(float(meta["n"].iloc[v].mean()), 1)}
                      for k, v in pools.items()},
            "rf_episode_max_abs_q": {str(q): round(float(np.quantile(
                [np.abs(rf[i]).max() for i in range(len(rf))], q)), 4) for q in (0.5, 0.9, 0.99)}}
    os.makedirs(os.path.join(RESULTS, "step4_bridge"), exist_ok=True)
    with open(os.path.join(RESULTS, "step4_bridge", "pools.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print(json.dumps(summ, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
