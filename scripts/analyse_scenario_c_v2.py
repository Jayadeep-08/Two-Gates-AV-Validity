import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

DATA_FILE_D = r'C:\Users\jayad\Desktop\RPI_AINN\data\scenario_d_baseline.csv'
DATA_FILE_C = r'C:\Users\jayad\Desktop\RPI_AINN\data\scenario_c_braking.csv'
OUTPUT_DIR  = r'C:\Users\jayad\Desktop\RPI_AINN\analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("RPI ANALYSIS v2 — Scenario C (Emergency Braking)")
print("=" * 60)

df_d = pd.read_csv(DATA_FILE_D)
df_c = pd.read_csv(DATA_FILE_C)

# ── Step 1: Define rho_baseline correctly ─────────────────────────────────────
# rho_baseline for each T+1 state in Scenario C is the expected risk
# at a state with similar speed_t1, estimated from Scenario D.
#
# Method: for each decision pair in Scenario C, find all Scenario D pairs
# where speed_t1 is within a tolerance band, and take their mean rho_actual.
# This gives the "what would risk be here if no braking had happened" estimate.

print("\nStep 1 — Computing speed-matched baseline from Scenario D...")

SPEED_TOLERANCE = 1.0   # m/s — match pairs within this speed band

baselines = []
for idx, row in df_c.iterrows():
    speed_t1  = row['speed_t1']
    # Find Scenario D pairs with similar T+1 speed
    matched   = df_d[
        (df_d['speed_t1'] >= speed_t1 - SPEED_TOLERANCE) &
        (df_d['speed_t1'] <= speed_t1 + SPEED_TOLERANCE)
    ]
    if len(matched) >= 5:
        # Enough matched samples — use their mean as baseline
        baselines.append(matched['rho_actual'].mean())
    else:
        # Not enough matches — fall back to global Scenario D mean
        baselines.append(df_d['rho_actual'].mean())

df_c['rho_baseline'] = baselines
df_c['RPI_pair']     = df_c['rho_actual'] - df_c['rho_baseline']

n_matched = sum(1 for b in baselines if b != df_d['rho_actual'].mean())
print(f"  Matched {n_matched}/{len(df_c)} pairs by speed band (±{SPEED_TOLERANCE} m/s)")
print(f"  Mean rho_actual   (C) : {df_c['rho_actual'].mean():.6f}")
print(f"  Mean rho_baseline (D) : {df_c['rho_baseline'].mean():.6f}")

# ── Step 2: RPI pair stats ────────────────────────────────────────────────────
print(f"\nStep 2 — RPI per decision pair")
print(f"  Mean RPI_pair : {df_c['RPI_pair'].mean():.6f}")
print(f"  Std  RPI_pair : {df_c['RPI_pair'].std():.6f}")
print(f"  Min  RPI_pair : {df_c['RPI_pair'].min():.6f}")
print(f"  Max  RPI_pair : {df_c['RPI_pair'].max():.6f}")
print(f"  % pairs with RPI > 0 : {(df_c['RPI_pair'] > 0).mean()*100:.1f}%")

# ── Step 3: RPI chain per episode ─────────────────────────────────────────────
ep_c = df_c.groupby('episode_id').agg(
    RPI_chain       =('RPI_pair',   'mean'),
    rho_actual_mean =('rho_actual', 'mean'),
    rho_base_mean   =('rho_baseline','mean'),
).reset_index()

print(f"\nStep 3 — RPI chain per episode")
print(f"  Mean RPI_chain : {ep_c['RPI_chain'].mean():.6f}")
print(f"  Std  RPI_chain : {ep_c['RPI_chain'].std():.6f}")
print(f"  Min  RPI_chain : {ep_c['RPI_chain'].min():.6f}")
print(f"  Max  RPI_chain : {ep_c['RPI_chain'].max():.6f}")

# ── Step 4: Significance tests ────────────────────────────────────────────────
# Primary test: is mean RPI_pair significantly > 0?
t1, p1 = stats.ttest_1samp(df_c['RPI_pair'], popmean=0)

# One-sided p-value (we predict RPI > 0, not just != 0)
p1_one_sided = p1 / 2 if t1 > 0 else 1 - p1 / 2

# Chain-level test: is mean RPI_chain significantly > 0?
t2, p2 = stats.ttest_1samp(ep_c['RPI_chain'], popmean=0)
p2_one_sided = p2 / 2 if t2 > 0 else 1 - p2 / 2

print(f"\nStep 4 — Significance tests (H0: RPI = 0, H1: RPI > 0)")
print(f"  Pair-level:  t={t1:.4f}, p(two-sided)={p1:.6f}, "
      f"p(one-sided)={p1_one_sided:.6f}")
print(f"  Chain-level: t={t2:.4f}, p(two-sided)={p2:.6f}, "
      f"p(one-sided)={p2_one_sided:.6f}")
sig1 = "SIGNIFICANT" if p1_one_sided < 0.05 else "not significant"
sig2 = "SIGNIFICANT" if p2_one_sided < 0.05 else "not significant"
print(f"  Pair-level result  : {sig1}")
print(f"  Chain-level result : {sig2}")

# ── Step 5: Bootstrap CI ──────────────────────────────────────────────────────
np.random.seed(42)
n_boot     = 10000
boot_means = np.array([
    np.random.choice(df_c['RPI_pair'], size=len(df_c), replace=True).mean()
    for _ in range(n_boot)
])
ci_lo = np.percentile(boot_means, 2.5)
ci_hi = np.percentile(boot_means, 97.5)

print(f"\nStep 5 — Bootstrap 95% CI")
print(f"  Mean RPI_pair : {df_c['RPI_pair'].mean():.6f}")
print(f"  95% CI        : [{ci_lo:.6f}, {ci_hi:.6f}]")
print(f"  CI above zero : {ci_lo > 0}")

# ── Step 6: Underestimation gap ───────────────────────────────────────────────
gap_abs = df_c['RPI_pair'].mean()
gap_pct = (gap_abs / df_c['rho_baseline'].mean()) * 100

print(f"\nStep 6 — Underestimation gap")
print(f"  Absolute gap : {gap_abs:.6f} risk units")
print(f"  Relative gap : {gap_pct:.1f}%")
print(f"  Interpretation: independent-decision methods underestimate")
print(f"  post-braking risk by {gap_pct:.1f}% on average")

# ── Step 7: Save ──────────────────────────────────────────────────────────────
out_csv = os.path.join(OUTPUT_DIR, 'scenario_c_analysed_v2.csv')
df_c.to_csv(out_csv, index=False)
print(f"\nStep 7 — Saved to: {out_csv}")

# ── Step 8: Figures ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Scenario C — Emergency Braking Chain (v2: speed-matched baseline)',
             fontsize=12, fontweight='bold')

# 1: rho_actual vs rho_baseline per pair
axes[0,0].scatter(range(len(df_c)), df_c['rho_actual'],
                  alpha=0.15, s=4, color='crimson', label='rho_actual')
axes[0,0].scatter(range(len(df_c)), df_c['rho_baseline'],
                  alpha=0.15, s=4, color='steelblue', label='rho_baseline')
axes[0,0].set_title('Actual vs baseline risk per pair')
axes[0,0].set_xlabel('Decision pair index')
axes[0,0].set_ylabel('Risk score')
axes[0,0].legend(markerscale=3)

# 2: RPI_pair distribution
axes[0,1].hist(df_c['RPI_pair'], bins=50, color='darkorange',
               edgecolor='white', alpha=0.85)
axes[0,1].axvline(0, color='black', linewidth=1, label='Zero')
axes[0,1].axvline(df_c['RPI_pair'].mean(), color='red',
                  linestyle='--', linewidth=2,
                  label=f'Mean={df_c["RPI_pair"].mean():.4f}')
axes[0,1].axvline(ci_lo, color='gray', linestyle=':', linewidth=1)
axes[0,1].axvline(ci_hi, color='gray', linestyle=':', linewidth=1,
                  label=f'95% CI')
axes[0,1].set_title('RPI per decision pair')
axes[0,1].set_xlabel('RPI_pair value')
axes[0,1].set_ylabel('Count')
axes[0,1].legend(fontsize=9)

# 3: RPI chain per episode
axes[0,2].plot(ep_c['episode_id'], ep_c['RPI_chain'],
               color='darkred', alpha=0.4, linewidth=0.6)
axes[0,2].axhline(0, color='black', linestyle='--', linewidth=1)
axes[0,2].axhline(ep_c['RPI_chain'].mean(), color='red',
                  linestyle='--', linewidth=2,
                  label=f'Mean={ep_c["RPI_chain"].mean():.4f}')
axes[0,2].set_title('RPI chain per episode')
axes[0,2].set_xlabel('Episode ID')
axes[0,2].set_ylabel('RPI_chain')
axes[0,2].legend()

# 4: D vs C risk distributions
axes[1,0].hist(df_d['rho_actual'], bins=30, color='steelblue',
               edgecolor='white', alpha=0.7, label='D: baseline', density=True)
axes[1,0].hist(df_c['rho_actual'], bins=30, color='crimson',
               edgecolor='white', alpha=0.7, label='C: braking', density=True)
axes[1,0].set_title('Risk distributions: D vs C')
axes[1,0].set_xlabel('Risk score')
axes[1,0].set_ylabel('Density')
axes[1,0].legend()

# 5: Mean comparison bar chart
means  = [df_d['rho_actual'].mean(),
           df_c['rho_baseline'].mean(),
           df_c['rho_actual'].mean()]
sems   = [df_d['rho_actual'].sem(),
           df_c['rho_baseline'].sem(),
           df_c['rho_actual'].sem()]
labels = ['D: normal\n(actual)', 'C: baseline\n(predicted)', 'C: braking\n(actual)']
colors = ['steelblue', 'lightcoral', 'crimson']
bars   = axes[1,1].bar(labels, means, color=colors,
                        edgecolor='white', alpha=0.9, width=0.5)
axes[1,1].errorbar(labels, means, yerr=[1.96*s for s in sems],
                   fmt='none', color='black', capsize=5, linewidth=1.5)
for bar, val in zip(bars, means):
    axes[1,1].text(bar.get_x() + bar.get_width()/2,
                   val + 0.01, f'{val:.3f}',
                   ha='center', va='bottom', fontsize=9)
axes[1,1].set_title('Risk: normal vs predicted vs actual braking')
axes[1,1].set_ylabel('Mean risk score')

# Annotate the gap
axes[1,1].annotate('',
    xy=(2, means[2]), xytext=(1, means[1]),
    arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
axes[1,1].text(1.5, (means[1]+means[2])/2 + 0.02,
               f'RPI\n{gap_pct:.0f}%',
               ha='center', fontsize=9, fontweight='bold')

# 6: Speed at T vs RPI_pair — shows the mechanism
axes[1,2].scatter(df_c['speed_t'], df_c['RPI_pair'],
                  alpha=0.2, color='darkred', s=8)
axes[1,2].axhline(0, color='black', linestyle='--', linewidth=1)
corr = df_c['speed_t'].corr(df_c['RPI_pair'])
axes[1,2].set_title(f'Speed at T vs RPI_pair\n(r={corr:.3f})')
axes[1,2].set_xlabel('Speed at decision T (m/s)')
axes[1,2].set_ylabel('RPI_pair')

plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, 'scenario_c_figures_v2.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Figures saved to: {fig_path}")

print("\n" + "=" * 60)
print("SCENARIO C FINAL SUMMARY")
print("=" * 60)
print(f"  Episodes              : {df_c['episode_id'].nunique()}")
print(f"  Decision pairs        : {len(df_c)}")
print(f"  Mean rho_actual       : {df_c['rho_actual'].mean():.4f}")
print(f"  Mean rho_baseline     : {df_c['rho_baseline'].mean():.4f}")
print(f"  Mean RPI_pair         : {df_c['RPI_pair'].mean():.6f}")
print(f"  95% CI                : [{ci_lo:.6f}, {ci_hi:.6f}]")
print(f"  p-value (one-sided)   : {p1_one_sided:.6f}")
print(f"  Underestimation gap   : {gap_pct:.1f}%")
print(f"  CI strictly above 0   : {ci_lo > 0}")