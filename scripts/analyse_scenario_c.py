import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

DATA_FILE_D  = r'C:\Users\jayad\Desktop\RPI_AINN\data\scenario_d_baseline.csv'
DATA_FILE_C  = r'C:\Users\jayad\Desktop\RPI_AINN\data\scenario_c_braking.csv'
OUTPUT_DIR   = r'C:\Users\jayad\Desktop\RPI_AINN\analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("RPI ANALYSIS — Scenario C (Emergency Braking)")
print("=" * 60)

# ── Load both datasets ────────────────────────────────────────────────────────
df_d = pd.read_csv(DATA_FILE_D)
df_c = pd.read_csv(DATA_FILE_C)
print(f"\nScenario D: {len(df_d)} pairs from {df_d['episode_id'].nunique()} episodes")
print(f"Scenario C: {len(df_c)} pairs from {df_c['episode_id'].nunique()} episodes")

# ── Step 1: Estimate rho_baseline for Scenario C ─────────────────────────────
# The independent-decision baseline for Scenario C is estimated two ways:
# (A) Stationary mean of Scenario C itself — what you'd predict ignoring sequence
# (B) Cross-scenario baseline using Scenario D mean — what a point-in-time
#     method trained on normal driving would predict
rho_baseline_C_self   = df_c['rho_actual'].mean()
rho_baseline_D        = df_d['rho_actual'].mean()

df_c['rho_baseline']  = rho_baseline_C_self   # primary baseline
df_c['RPI_pair']      = df_c['rho_actual'] - df_c['rho_baseline']

# Cross-scenario gap — the headline number for the paper
# This is what an independent-decision method calibrated on normal driving
# would predict vs what actually happens during braking
cross_gap_pct = ((rho_baseline_C_self - rho_baseline_D) / rho_baseline_D) * 100

print(f"\nStep 1 — Baseline estimates")
print(f"  rho_baseline (Scenario C self)     = {rho_baseline_C_self:.6f}")
print(f"  rho_baseline (Scenario D, normal)  = {rho_baseline_D:.6f}")
print(f"  Cross-scenario gap                 = {cross_gap_pct:.1f}%")
print(f"  Interpretation: braking scenarios have {cross_gap_pct:.1f}% higher")
print(f"  risk than normal driving baselines predict")

# ── Step 2: RPI pair stats ────────────────────────────────────────────────────
print(f"\nStep 2 — RPI per decision pair (Scenario C)")
print(f"  Mean RPI_pair : {df_c['RPI_pair'].mean():.6f}")
print(f"  Std  RPI_pair : {df_c['RPI_pair'].std():.6f}")
print(f"  Min  RPI_pair : {df_c['RPI_pair'].min():.6f}")
print(f"  Max  RPI_pair : {df_c['RPI_pair'].max():.6f}")

# ── Step 3: RPI chain per episode ─────────────────────────────────────────────
ep_c = df_c.groupby('episode_id').agg(
    RPI_chain       =('RPI_pair',    'mean'),
    rho_actual_mean =('rho_actual',  'mean'),
    speed_t_mean    =('speed_t',     'mean'),
    speed_t1_mean   =('speed_t1',    'mean'),
).reset_index()

print(f"\nStep 3 — RPI chain per episode")
print(f"  Mean RPI_chain : {ep_c['RPI_chain'].mean():.6f}")
print(f"  Std  RPI_chain : {ep_c['RPI_chain'].std():.6f}")

# ── Step 4: Significance test ─────────────────────────────────────────────────
# Test 1: Is mean RPI_pair significantly different from zero?
t1, p1 = stats.ttest_1samp(df_c['RPI_pair'], popmean=0)

# Test 2: Is Scenario C risk significantly higher than Scenario D?
# This is the core cross-scenario claim
t2, p2 = stats.ttest_ind(df_c['rho_actual'], df_d['rho_actual'])

print(f"\nStep 4 — Significance tests")
print(f"  Test 1: H0: mean RPI_pair = 0")
print(f"    t={t1:.4f}, p={p1:.8f}")
print(f"    Result: {'SIGNIFICANT' if p1 < 0.05 else 'not significant'}")
print(f"  Test 2: H0: rho_actual(C) = rho_actual(D)")
print(f"    t={t2:.4f}, p={p2:.8f}")
print(f"    Result: {'SIGNIFICANT' if p2 < 0.05 else 'not significant'}")

# ── Step 5: Bootstrap CI ──────────────────────────────────────────────────────
np.random.seed(42)
n_boot     = 10000
boot_means = np.array([
    np.random.choice(df_c['RPI_pair'], size=len(df_c), replace=True).mean()
    for _ in range(n_boot)
])
ci_lo = np.percentile(boot_means, 2.5)
ci_hi = np.percentile(boot_means, 97.5)

print(f"\nStep 5 — Bootstrap 95% CI for mean RPI_pair")
print(f"  Mean RPI : {df_c['RPI_pair'].mean():.6f}")
print(f"  95% CI   : [{ci_lo:.6f}, {ci_hi:.6f}]")
print(f"  CI includes zero: {ci_lo <= 0 <= ci_hi}")

# ── Step 6: Underestimation gap ───────────────────────────────────────────────
# Primary gap: within Scenario C, how much does sequential risk
# deviate from the independent-decision prediction?
gap_self_pct = (df_c['RPI_pair'].mean() / rho_baseline_C_self) * 100

print(f"\nStep 6 — Underestimation gap")
print(f"  Within-scenario gap  : {gap_self_pct:.2f}%")
print(f"  Cross-scenario gap   : {cross_gap_pct:.1f}%")
print(f"  (Cross-scenario = what a method trained on normal driving")
print(f"   would underestimate braking-chain risk by)")

# ── Step 7: Save ──────────────────────────────────────────────────────────────
out_csv = os.path.join(OUTPUT_DIR, 'scenario_c_analysed.csv')
df_c.to_csv(out_csv, index=False)
print(f"\nStep 7 — Saved to: {out_csv}")

# ── Step 8: Figures ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Scenario C — Emergency Braking Chain', fontsize=13, fontweight='bold')

# Row 1: Scenario C standalone
axes[0,0].hist(df_c['rho_actual'], bins=40, color='crimson',
               edgecolor='white', alpha=0.85)
axes[0,0].axvline(rho_baseline_C_self, color='black', linestyle='--',
                  linewidth=1.5, label=f'Baseline={rho_baseline_C_self:.3f}')
axes[0,0].set_title('Risk score distribution (Scenario C)')
axes[0,0].set_xlabel('Risk score')
axes[0,0].set_ylabel('Count')
axes[0,0].legend()

axes[0,1].hist(df_c['RPI_pair'], bins=40, color='darkorange',
               edgecolor='white', alpha=0.85)
axes[0,1].axvline(0, color='black', linewidth=1)
axes[0,1].axvline(df_c['RPI_pair'].mean(), color='red', linestyle='--',
                  linewidth=1.5, label=f'Mean={df_c["RPI_pair"].mean():.4f}')
axes[0,1].set_title('RPI per decision pair (Scenario C)')
axes[0,1].set_xlabel('RPI_pair value')
axes[0,1].set_ylabel('Count')
axes[0,1].legend()

axes[0,2].plot(ep_c['episode_id'], ep_c['RPI_chain'],
               color='darkred', alpha=0.5, linewidth=0.7)
axes[0,2].axhline(0, color='black', linestyle='--', linewidth=1)
axes[0,2].axhline(ep_c['RPI_chain'].mean(), color='red',
                  linestyle='--', linewidth=1.5,
                  label=f'Mean={ep_c["RPI_chain"].mean():.4f}')
axes[0,2].set_title('RPI chain per episode')
axes[0,2].set_xlabel('Episode ID')
axes[0,2].set_ylabel('RPI_chain')
axes[0,2].legend()

# Row 2: Cross-scenario comparison — this is the paper's key figure
ep_d = df_d.groupby('episode_id').agg(
    RPI_chain      =('RPI_pair',   'mean'),
    rho_actual_mean=('rho_actual', 'mean'),
).reset_index()
# Need to compute RPI_pair for D too
df_d['rho_baseline'] = df_d['rho_actual'].mean()
df_d['RPI_pair']     = df_d['rho_actual'] - df_d['rho_baseline']
ep_d = df_d.groupby('episode_id').agg(
    RPI_chain      =('RPI_pair',   'mean'),
    rho_actual_mean=('rho_actual', 'mean'),
).reset_index()

# Fig 2-1: Side-by-side rho_actual distributions
axes[1,0].hist(df_d['rho_actual'], bins=30, color='steelblue',
               edgecolor='white', alpha=0.7, label='D: baseline', density=True)
axes[1,0].hist(df_c['rho_actual'], bins=30, color='crimson',
               edgecolor='white', alpha=0.7, label='C: braking', density=True)
axes[1,0].set_title('Risk score: Scenario D vs C')
axes[1,0].set_xlabel('Risk score')
axes[1,0].set_ylabel('Density')
axes[1,0].legend()

# Fig 2-2: Mean risk comparison bar chart with CI
means  = [df_d['rho_actual'].mean(), df_c['rho_actual'].mean()]
sems   = [df_d['rho_actual'].sem(),  df_c['rho_actual'].sem()]
colors = ['steelblue', 'crimson']
labels = ['D: baseline', 'C: braking']
bars   = axes[1,1].bar(labels, means, color=colors, alpha=0.85,
                        edgecolor='white', width=0.5)
axes[1,1].errorbar(labels, means, yerr=[1.96*s for s in sems],
                   fmt='none', color='black', capsize=6, linewidth=1.5)
axes[1,1].set_title('Mean risk score comparison (95% CI)')
axes[1,1].set_ylabel('Mean rho_actual')
for bar, val in zip(bars, means):
    axes[1,1].text(bar.get_x() + bar.get_width()/2, val + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)

# Fig 2-3: Speed at T vs risk at T+1 — shows the carry-forward mechanism
axes[1,2].scatter(df_c['speed_t'], df_c['rho_actual'],
                  alpha=0.15, color='crimson', s=8)
axes[1,2].set_title('Speed at T vs Risk at T+1\n(carry-forward mechanism)')
axes[1,2].set_xlabel('Speed at decision T (m/s)')
axes[1,2].set_ylabel('Risk at T+1 (rho_actual)')

# Correlation
corr = df_c['speed_t'].corr(df_c['rho_actual'])
axes[1,2].text(0.05, 0.92, f'r = {corr:.3f}',
               transform=axes[1,2].transAxes, fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, 'scenario_c_figures.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Figures saved to: {fig_path}")

print("\n" + "=" * 60)
print("SCENARIO C SUMMARY")
print("=" * 60)
print(f"  Episodes           : {df_c['episode_id'].nunique()}")
print(f"  Decision pairs     : {len(df_c)}")
print(f"  Mean rho_actual    : {df_c['rho_actual'].mean():.4f}  "
      f"(D was {df_d['rho_actual'].mean():.4f})")
print(f"  Mean RPI_pair      : {df_c['RPI_pair'].mean():.6f}")
print(f"  95% CI             : [{ci_lo:.6f}, {ci_hi:.6f}]")
print(f"  p-value (H0=0)     : {p1:.8f}")
print(f"  Cross-scenario gap : {cross_gap_pct:.1f}%")
print(f"  CI includes zero   : {ci_lo <= 0 <= ci_hi}")