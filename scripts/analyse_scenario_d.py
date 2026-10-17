import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_FILE   = r'C:\Users\jayad\Desktop\RPI_AINN\data\scenario_d_baseline.csv'
OUTPUT_DIR  = r'C:\Users\jayad\Desktop\RPI_AINN\analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("RPI ANALYSIS — Scenario D (Open Road Baseline)")
print("=" * 60)

# ── Step 1: Load data ─────────────────────────────────────────────────────────
df = pd.read_csv(DATA_FILE)
print(f"\nLoaded {len(df)} decision pairs from {df['episode_id'].nunique()} episodes")

# ── Step 2: Estimate rho_baseline ─────────────────────────────────────────────
# The independent-decision baseline is the mean risk across ALL transitions,
# ignoring which episode or prior decision produced them.
# This is the stationary distribution estimate — what you'd predict
# if you assumed each decision's risk was independent of the prior one.
rho_baseline_value = df['rho_actual'].mean()
df['rho_baseline'] = rho_baseline_value

print(f"\nStep 2 — Baseline risk estimate")
print(f"  rho_baseline (stationary mean) = {rho_baseline_value:.6f}")
print(f"  This is what independent-decision methods would predict for every step")

# ── Step 3: Compute RPI per decision pair ─────────────────────────────────────
df['RPI_pair'] = df['rho_actual'] - df['rho_baseline']

print(f"\nStep 3 — RPI per decision pair")
print(f"  Mean RPI_pair : {df['RPI_pair'].mean():.6f}")
print(f"  Std  RPI_pair : {df['RPI_pair'].std():.6f}")
print(f"  Min  RPI_pair : {df['RPI_pair'].min():.6f}")
print(f"  Max  RPI_pair : {df['RPI_pair'].max():.6f}")

# ── Step 4: Aggregate RPI per episode (chain-level) ───────────────────────────
episode_stats = df.groupby('episode_id').agg(
    RPI_chain     = ('RPI_pair',   'mean'),
    rho_actual_mean = ('rho_actual', 'mean'),
    rho_baseline_mean = ('rho_baseline', 'mean'),
    n_pairs       = ('RPI_pair',   'count')
).reset_index()

print(f"\nStep 4 — RPI chain (per episode aggregate)")
print(f"  Mean RPI_chain : {episode_stats['RPI_chain'].mean():.6f}")
print(f"  Std  RPI_chain : {episode_stats['RPI_chain'].std():.6f}")

# ── Step 5: Paired t-test ─────────────────────────────────────────────────────
# H0: E[RPI_pair] = 0  (independent-decision assumption holds)
# H1: E[RPI_pair] != 0 (carry-forward risk exists)
t_stat, p_value = stats.ttest_1samp(df['RPI_pair'], popmean=0)

print(f"\nStep 5 — Paired t-test (H0: mean RPI = 0)")
print(f"  t-statistic : {t_stat:.4f}")
print(f"  p-value     : {p_value:.6f}")
if p_value < 0.05:
    print(f"  Result      : SIGNIFICANT (p < 0.05) — RPI != 0")
else:
    print(f"  Result      : NOT significant (p >= 0.05) — consistent with baseline")
print(f"  Interpretation: For Scenario D this SHOULD be non-significant")
print(f"  (confirms our control group behaves as expected)")

# ── Step 6: Bootstrap confidence interval for mean RPI ────────────────────────
np.random.seed(42)
n_boot        = 10000
boot_means    = np.array([
    np.random.choice(df['RPI_pair'], size=len(df), replace=True).mean()
    for _ in range(n_boot)
])
ci_lower = np.percentile(boot_means, 2.5)
ci_upper = np.percentile(boot_means, 97.5)

print(f"\nStep 6 — Bootstrap 95% CI for mean RPI_pair (n={n_boot} resamples)")
print(f"  Mean RPI : {df['RPI_pair'].mean():.6f}")
print(f"  95% CI   : [{ci_lower:.6f}, {ci_upper:.6f}]")
print(f"  CI includes zero: {ci_lower <= 0 <= ci_upper}")

# ── Step 7: Underestimation gap ───────────────────────────────────────────────
# Gap = how much rho_actual deviates from rho_baseline on average, as a %
gap_pct = (df['RPI_pair'].mean() / rho_baseline_value) * 100
print(f"\nStep 7 — Underestimation gap")
print(f"  Gap = {gap_pct:.2f}%")
print(f"  For Scenario D this should be close to 0%")
print(f"  (by definition — baseline IS the mean of actual, so gap = 0 here)")
print(f"  The meaningful gap will appear when we compare across scenarios")

# ── Step 8: Save updated CSV with RPI filled in ───────────────────────────────
out_csv = os.path.join(OUTPUT_DIR, 'scenario_d_analysed.csv')
df.to_csv(out_csv, index=False)
print(f"\nStep 8 — Saved analysed data to: {out_csv}")

# ── Step 9: Figures ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Scenario D — Open Road Baseline', fontsize=13, fontweight='bold')

# Fig 1: rho_actual distribution
axes[0].hist(df['rho_actual'], bins=30, color='steelblue', edgecolor='white', alpha=0.85)
axes[0].axvline(rho_baseline_value, color='red', linestyle='--',
                linewidth=1.5, label=f'Baseline={rho_baseline_value:.3f}')
axes[0].set_title('Risk score distribution (rho_actual)')
axes[0].set_xlabel('Risk score')
axes[0].set_ylabel('Count')
axes[0].legend()

# Fig 2: RPI_pair distribution
axes[1].hist(df['RPI_pair'], bins=30, color='coral', edgecolor='white', alpha=0.85)
axes[1].axvline(0, color='black', linestyle='-', linewidth=1, label='Zero line')
axes[1].axvline(df['RPI_pair'].mean(), color='red', linestyle='--',
                linewidth=1.5, label=f'Mean={df["RPI_pair"].mean():.4f}')
axes[1].set_title('RPI per decision pair')
axes[1].set_xlabel('RPI_pair value')
axes[1].set_ylabel('Count')
axes[1].legend()

# Fig 3: RPI_chain per episode
axes[2].plot(episode_stats['episode_id'], episode_stats['RPI_chain'],
             color='purple', alpha=0.6, linewidth=0.8)
axes[2].axhline(0, color='black', linestyle='--', linewidth=1)
axes[2].axhline(episode_stats['RPI_chain'].mean(), color='red',
                linestyle='--', linewidth=1.5,
                label=f'Mean={episode_stats["RPI_chain"].mean():.4f}')
axes[2].set_title('RPI chain per episode')
axes[2].set_xlabel('Episode ID')
axes[2].set_ylabel('RPI_chain')
axes[2].legend()

plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, 'scenario_d_figures.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Figures saved to: {fig_path}")

print("\n" + "=" * 60)
print("SCENARIO D SUMMARY")
print("=" * 60)
print(f"  Episodes          : {df['episode_id'].nunique()}")
print(f"  Decision pairs    : {len(df)}")
print(f"  Mean rho_actual   : {df['rho_actual'].mean():.4f}")
print(f"  Mean RPI_pair     : {df['RPI_pair'].mean():.6f}")
print(f"  95% CI            : [{ci_lower:.6f}, {ci_upper:.6f}]")
print(f"  p-value (H0=0)    : {p_value:.6f}")
print(f"  Underestimation   : {gap_pct:.2f}%")
print(f"  Control group OK  : {abs(df['RPI_pair'].mean()) < 0.01}")