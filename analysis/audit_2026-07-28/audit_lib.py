"""Independent audit harness. Written from scratch; does not import project code."""
import numpy as np, pandas as pd
from scipy import stats

D = "/sessions/confident-vigilant-allen/mnt/RPI_AINN/data/"
RNG = 12345  # deliberately NOT the project's seed 42


def load(name):
    df = pd.read_csv(D + name)
    n0 = len(df)
    df = df.dropna(how="all")
    if len(df) != n0:
        print(f"  [!] {name}: dropped {n0-len(df)} all-NaN rows")
    return df


def ep_agg(df, sev, out, ep="episode_id"):
    """Episode-level means of severity and outcome."""
    g = df.groupby(ep)[[sev, out]].mean().dropna()
    return g[sev].values, g[out].values


def perm_p(x, y, n=10000, seed=RNG):
    """Two-sided permutation test on |r| by shuffling y against x."""
    r0 = np.corrcoef(x, y)[0, 1]
    rs = np.random.default_rng(seed)
    cnt = 0
    yy = y.copy()
    for _ in range(n):
        rs.shuffle(yy)
        if abs(np.corrcoef(x, yy)[0, 1]) >= abs(r0) - 1e-15:
            cnt += 1
    return r0, (cnt + 1) / (n + 1)


def boot_ci(x, y, n=10000, seed=RNG, alpha=0.05):
    rs = np.random.default_rng(seed)
    m = len(x)
    out = np.empty(n)
    for i in range(n):
        idx = rs.integers(0, m, m)
        xa, ya = x[idx], y[idx]
        if xa.std() == 0 or ya.std() == 0:
            out[i] = np.nan
        else:
            out[i] = np.corrcoef(xa, ya)[0, 1]
    out = out[~np.isnan(out)]
    return np.percentile(out, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def full(label, x, y, npermute=10000):
    r, p = perm_p(x, y, npermute)
    lo, hi = boot_ci(x, y)
    pear_p = stats.pearsonr(x, y)[1]
    print(f"{label:52s} n={len(x):4d}  r={r:+.4f}  permp={p:.5f}  "
          f"CI[{lo:+.4f},{hi:+.4f}]  pear_p={pear_p:.3g}")
    return dict(label=label, n=len(x), r=r, perm_p=p, lo=lo, hi=hi, pear_p=pear_p)


def desc(df, cols):
    for c in cols:
        if c not in df.columns:
            print(f"  {c:24s} MISSING")
            continue
        v = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(v) == 0:
            print(f"  {c:24s} all-NaN")
            continue
        # pinning diagnostics
        top = v.value_counts(normalize=True).iloc[0]
        topval = v.value_counts(normalize=True).index[0]
        print(f"  {c:24s} n={len(v):5d} mean={v.mean():+.4f} std={v.std():.4f} "
              f"min={v.min():+.4f} max={v.max():+.4f} "
              f"modefrac={top:.3f}@{topval:.4f} "
              f"frac>=.99max={(v >= 0.99*v.max()).mean():.3f}")
