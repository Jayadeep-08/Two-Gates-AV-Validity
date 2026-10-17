"""Two-agent chain experiment, sweep v2 — every number used in the section.

Analysis plan fixed before outcomes were inspected (DIARY entry 2):
  primary  = episode-level mean follower_gap_after at position 7, CHAIN vs ISO_LATE,
             valid gaps only, at the 2.9 s spacing; bootstrap CI + permutation test.
  controls = entry-speed spread by position, survivorship to position 7,
             valid-gap rate by arm, severity-vs-validity association.
  secondary= spacing x arm interaction on the same outcome.

READ-ONLY on chain_risk/data/. Writes only under tiv_revision/.
Run:  py -3.7 tiv_revision\\scripts\\chain_numbers.py
"""
import json, os
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("RPI_OUT") or os.path.abspath(os.path.join(HERE, ".."))
DATA = os.environ.get("RPI_CHAIN_DATA") or os.path.join(ROOT, "..", "chain_risk", "data")
os.makedirs(os.path.join(ROOT, "tables"), exist_ok=True)
RNG = np.random.default_rng(42)
TICK_S = {0: 0.90, 12: 1.50, 40: 2.90, 76: 4.70}
PRIMARY = 40

truthy = lambda s: s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "yes"])


def boot_diff(a, b, n=10000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a[RNG.integers(0, len(a), (n, len(a)))].mean(1) - b[RNG.integers(0, len(b), (n, len(b)))].mean(1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def perm_diff(a, b, n=10000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = abs(a.mean() - b.mean()); pool = np.concatenate([a, b]); na = len(a)
    c = sum(abs((p := RNG.permutation(pool))[:na].mean() - p[na:].mean()) >= obs for _ in range(n))
    return float((c + 1) / (n + 1))


f = pd.read_csv(os.path.join(DATA, "chain_position_hazard_sweep_v2.csv"))
f["valid"] = truthy(f.follower_gap_valid)
f["is_chain"] = f.arm.str.upper().str.contains("CHAIN")
P = f[f.setting_cruise_ticks == PRIMARY]
N = {}

# ---- design / coverage
N["n_settings"] = int(f.setting_cruise_ticks.nunique())
N["n_ep_chain"] = int(P[P.is_chain].episode_id.nunique())
N["n_ep_iso"] = int(P[~P.is_chain].episode_id.nunique())
N["valid_pct"] = float(100 * P.valid.mean())
N["present_pct"] = float(100 * truthy(P.follower_present).mean())
N["spacing_s"] = TICK_S[PRIMARY]

# ---- control 1: entry-speed regime (GATE 5)
g = P[P.is_chain].groupby("position").entry_speed.mean()
N["entry_lo"], N["entry_hi"] = float(g.min()), float(g.max())
N["entry_spread"] = float(g.max() - g.min())

# ---- control 2: survivorship
reach = P[P.is_chain].groupby("episode_id").position.max()
N["reach_pos7"] = int((reach == 7).sum())
N["reach_pct"] = float(100 * (reach == 7).mean())

# ---- control 3: validity not selected on treatment
rpb = {}
for s in sorted(f.setting_cruise_ticks.unique()):
    sub = f[f.setting_cruise_ticks == s]
    rpb[int(s)] = float(stats.pointbiserialr(sub.valid.astype(int), sub.severity)[0])
N["validity_severity_r"] = rpb
N["validity_severity_rmax"] = float(max(abs(v) for v in rpb.values()))
N["valid_pct_chain7"] = float(100 * P[P.is_chain & (P.position == 7)].valid.mean())
N["valid_pct_iso"] = float(100 * P[~P.is_chain].valid.mean())

# ---- primary outcome
c7 = P[P.is_chain & (P.position == 7) & P.valid].groupby("episode_id").follower_gap_after.mean()
i7 = P[~P.is_chain & P.valid].groupby("episode_id").follower_gap_after.mean()
N["chain_gap"], N["chain_n"], N["chain_sd"] = float(c7.mean()), int(len(c7)), float(c7.std())
N["iso_gap"], N["iso_n"], N["iso_sd"] = float(i7.mean()), int(len(i7)), float(i7.std())
N["diff"] = N["chain_gap"] - N["iso_gap"]
N["diff_lo"], N["diff_hi"] = boot_diff(c7.values, i7.values)
N["diff_permp"] = perm_diff(c7.values, i7.values)

# how large an accumulation would this design have seen?
# accumulation = chain gap SMALLER than isolated = negative difference,
# so the interval's lower bound is the bound on accumulation.
N["accum_bound"] = float(abs(min(0.0, N["diff_lo"])))
sp = np.sqrt(((N["chain_n"] - 1) * N["chain_sd"] ** 2 + (N["iso_n"] - 1) * N["iso_sd"] ** 2)
             / (N["chain_n"] + N["iso_n"] - 2))
N["mde80"] = float((1.96 + 0.8416) * sp * np.sqrt(1 / N["chain_n"] + 1 / N["iso_n"]))

# ---- per-position profile (gap, and the entry-speed control)
def boot_mean_ci(x, n=10000):
    x = np.asarray(x, float)
    m = x[RNG.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


N["iso_lo"], N["iso_hi"] = boot_mean_ci(i7.values)
N["profile"] = []
for p in range(8):
    s = P[P.is_chain & (P.position == p) & P.valid].groupby("episode_id").follower_gap_after.mean()
    lo_, hi_ = boot_mean_ci(s.values)
    e = P[P.is_chain & (P.position == p)].entry_speed
    N["profile"].append({"pos": p, "n": int(len(s)), "mean": float(s.mean()),
                         "sd": float(s.std()), "lo": lo_, "hi": hi_,
                         "entry_mean": float(e.mean()), "entry_sd": float(e.std())})

# ---- secondary: spacing x arm interaction
rows = []
for s in sorted(f.setting_cruise_ticks.unique()):
    sub = f[(f.setting_cruise_ticks == s) & f.valid]
    for v in sub[sub.is_chain & (sub.position == 7)].groupby("episode_id").follower_gap_after.mean():
        rows.append((TICK_S[int(s)], 1, v))
    for v in sub[~sub.is_chain].groupby("episode_id").follower_gap_after.mean():
        rows.append((TICK_S[int(s)], 0, v))
d = pd.DataFrame(rows, columns=["spacing", "chain", "gap"])
X = np.column_stack([np.ones(len(d)), d.spacing, d.chain, d.spacing * d.chain])
beta, *_ = np.linalg.lstsq(X, d.gap.values, rcond=None)
bs = np.empty((4000, 4))
for i in range(4000):
    p = RNG.integers(0, len(d), len(d))
    bs[i], *_ = np.linalg.lstsq(X[p], d.gap.values[p], rcond=None)
lo, hi = np.nanpercentile(bs[:, 3], [2.5, 97.5])
N["interaction"], N["interaction_lo"], N["interaction_hi"] = float(beta[3]), float(lo), float(hi)
N["ols_beta"] = [float(b) for b in beta]
N["n_episode_means"] = int(len(d))
N["by_spacing"] = []
for s in sorted(TICK_S):
    sub = d[d.spacing == TICK_S[s]]
    a, b = sub[sub.chain == 1].gap.values, sub[sub.chain == 0].gap.values
    l2, h2 = boot_diff(a, b, 4000)
    N["by_spacing"].append({"spacing_s": TICK_S[s], "diff": float(a.mean() - b.mean()),
                            "lo": l2, "hi": h2, "n_chain": len(a), "n_iso": len(b)})

# ---- the decomposition the data do NOT support
v = P[P.is_chain & P.valid].copy()
N["eq_gap"] = float(v.follower_gap_before.mean())
N["mean_applied"] = float((v.follower_gap_before - v.follower_gap_after).mean())
v = v.sort_values(["episode_id", "position"]); nx = v.shift(-1)
m = (nx.episode_id == v.episode_id) & (nx.position == v.position + 1)
app = (v.follower_gap_before - v.follower_gap_after)[m].values
still = (v.follower_gap_before[m].values - nx.follower_gap_before[m].values)
k = app > 1.0
N["persistence"] = float(still[k].sum() / app[k].sum())
N["persist_n"] = int(k.sum())
N["amplification"] = float(1 / (1 - N["persistence"]))
N["predicted_chain_gap"] = float(N["eq_gap"] - N["amplification"] * (N["eq_gap"] - N["iso_gap"]))

json.dump(N, open(os.path.join(ROOT, "chain_numbers.json"), "w"), indent=1, default=float)

M = {
    "chGapChain": f"{N['chain_gap']:.2f}", "chGapIso": f"{N['iso_gap']:.2f}",
    "chNChain": N["chain_n"], "chNIso": N["iso_n"],
    "chDiff": f"{N['diff']:+.2f}".replace("+", "{+}"),
    "chDiffCI": f"[{N['diff_lo']:+.2f},\\;{N['diff_hi']:+.2f}]".replace("+", "{+}").replace("-", "{-}"),
    "chDiffP": f"{N['diff_permp']:.2f}",
    "chEntryLo": f"{N['entry_lo']:.2f}", "chEntryHi": f"{N['entry_hi']:.2f}",
    "chEntrySpread": f"{N['entry_spread']:.3f}",
    "chReach": N["reach_pos7"], "chEpChain": N["n_ep_chain"], "chEpIso": N["n_ep_iso"],
    "chValidPct": f"{N['valid_pct']:.0f}", "chSpacing": f"{N['spacing_s']:.1f}",
    "chInter": f"{N['interaction']:+.2f}".replace("+", "{+}"),
    "chInterCI": f"[{N['interaction_lo']:+.2f},\\;{N['interaction_hi']:+.2f}]".replace("+", "{+}").replace("-", "{-}"),
    "chAccumBound": f"{N['accum_bound']:.2f}", "chMDE": f"{N['mde80']:.1f}",
    "chPersist": f"{N['persistence']:.2f}", "chAmp": f"{N['amplification']:.2f}",
    "chPredGap": f"{N['predicted_chain_gap']:.2f}",
    "chValidSevR": f"{N['validity_severity_rmax']:.2f}",
}
with open(os.path.join(ROOT, "tables", "chain_macros.tex"), "w") as fh:
    fh.write("% GENERATED by tiv_revision/scripts/chain_numbers.py -- do not hand-edit.\n")
    for k2, v2 in M.items():
        fh.write("\\newcommand{\\%s}{%s}\n" % (k2, v2))

for k2 in ("chain_gap", "iso_gap", "diff", "diff_lo", "diff_hi", "diff_permp",
           "entry_spread", "reach_pct", "interaction", "interaction_lo", "interaction_hi",
           "persistence", "amplification", "predicted_chain_gap", "valid_pct"):
    print(f"  {k2:<22} {N[k2]}")
print("\nwrote chain_numbers.json and tables/chain_macros.tex")
