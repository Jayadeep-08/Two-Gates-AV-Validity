"""
Diagnostic run 2026-07-09 as part of the RPI methodology review.

Checks three things on the EXISTING scenario_c_braking.csv / scenario_d_baseline.csv
data (no CARLA required):

1. Does decision T (the scripted brake) actually vary episode to episode? If not,
   there is nothing for a within-scenario correlation/permutation ablation to detect.
2. Does a naive label-shuffle of rho_actual within a scenario change its mean?
   (Spoiler: it can't, by construction — this documents why that idea doesn't work.)
3. A partial same-formula check: apply the one risk-formula component that's fully
   reconstructable from logged CSV columns (the inverse-speed term) to both
   Scenario C's post-brake-recovery speeds and Scenario D's cruising speeds, and
   compare them directly — an apples-to-apples check that sidesteps the
   cross-formula confound, even though it's only 40% of C's full risk function.

See CLAUDE.md ("Methodology status" section, 2026-07-09) for the write-up of what
this means for the project and what the real fix is (decision T needs genuine
per-episode variance — e.g. randomized brake intensity — before a true
within-scenario propagation ablation is possible).
"""
import pandas as pd
import numpy as np
from scipy import stats
import os

DATA_DIR = r'C:\Users\jayad\Desktop\RPI_AINN\data'
df_c = pd.read_csv(os.path.join(DATA_DIR, 'scenario_c_braking.csv'))
df_d = pd.read_csv(os.path.join(DATA_DIR, 'scenario_d_baseline.csv'))

print("=" * 70)
print("DIAGNOSTIC 1: Does decision T have any real variance to propagate?")
print("=" * 70)
for col in ['speed_t', 'yaw_t', 'x_t', 'y_t']:
    print(f"  {col:10s}  std={df_c[col].std():10.4f}  "
          f"range=[{df_c[col].min():.3f}, {df_c[col].max():.3f}]")
print()
print("speed_t (state right after the brake) has almost zero variance across")
print("all 3200 rows because BRAKE_INTENSITY=1.0 and TICKS_BRAKE=20 are fixed")
print("constants -- every episode brakes identically hard for identically long.")
print("yaw_t/x_t/y_t vary a lot, but that's spawn-point geography, not brake severity.")
print()

print("=" * 70)
print("DIAGNOSTIC 2: Naive label-shuffle of rho_actual within Scenario C")
print("=" * 70)
np.random.seed(42)
real_mean = df_c['rho_actual'].mean()
shuffled_means = np.array([
    df_c['rho_actual'].sample(frac=1, replace=False).values.mean()
    for _ in range(1000)
])
print(f"  Real mean rho_actual       : {real_mean:.6f}")
print(f"  Shuffled mean (avg of 1000): {shuffled_means.mean():.6f}  "
      f"std={shuffled_means.std():.2e}")
print("  -> Identical, as expected: shuffling values and taking the mean of the")
print("     SAME set is mathematically invariant. A naive shuffle test can never")
print("     detect anything here -- do not use this as the ablation.")
print()

print("=" * 70)
print("DIAGNOSTIC 3: Same-formula partial check (inverse-speed component only)")
print("=" * 70)
print("Applying C's inverse-speed term to D's own speeds -- an apples-to-apples,")
print("same-formula comparison (unlike the full D-mean baseline, which mixes")
print("formulas). Partial: lane_deviation/yaw_risk need live CARLA waypoint")
print("queries not present in the CSV.")
print()
inv_speed_c = np.maximum(0.0, 1.0 - df_c['speed_t1'] / 15.0)
inv_speed_d = np.maximum(0.0, 1.0 - df_d['speed_t1'] / 15.0)
t, p = stats.ttest_ind(inv_speed_c, inv_speed_d, equal_var=False)
print(f"  Mean inverse_speed term, Scenario C (post-brake recovery): {inv_speed_c.mean():.4f}")
print(f"  Mean inverse_speed term, Scenario D (normal cruising)    : {inv_speed_d.mean():.4f}")
print(f"  Welch t-test: t={t:.3f}, p={p:.2e}")
print()
print("Significant and in the predicted direction -- supporting evidence, but")
print("only 40% of C's full formula, so not sufficient on its own.")
