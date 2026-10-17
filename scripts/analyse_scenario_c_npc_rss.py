"""
Episode-level ablation for the Scenario C NPC-traffic / RSS-margin variant.

Purpose (see CLAUDE.md "RSS baseline" and "NPC-traffic scope" decision entries,
2026-07-14, and scenario_c_npc_rss.py's own module docstring): decision-T brake
severity is already shown, on the banked no-NPC scenario_c_braking.csv, to
predict rho_actual at T+1 (episode-level r = +0.770, see analyse_unified.py /
unified_table1.csv). RSS (Shalev-Shwartz et al., 2017) is memoryless BY
CONSTRUCTION -- it scores each instant's inter-vehicle geometry with no
knowledge of past decisions. If severity ALSO predicts the RSS safe-following
margin to a trailing vehicle at T+1, carry-forward is demonstrated on a
second, externally-defined, industry-standard measure, not just this
project's own hand-built rho formula. That is the whole point of this file.

METHODOLOGY -- deliberately identical to analyse_unified.py, not a variant:
  - Statistical unit is the EPISODE everywhere a p-value is attached. Pair-level
    r is reported as descriptive-only, exactly as unified_table1.csv does --
    pairs inside one episode share a vehicle/run and are not independent
    samples (see CLAUDE.md "Statistical unit of analysis").
  - permutation_p() and bootstrap_ci_r() below are the SAME functions,
    same N_PERM=10000, N_BOOT=10000, same seed=42, as analyse_unified.py.
    Copied rather than imported so this file stays a standalone, runnable
    script like every other analyse_*.py in this project, but the numbers
    are directly comparable to Table I -- a different permutation scheme
    here would make "does the RSS ablation agree with the rho ablation"
    an apples-to-oranges question.
  - Episode-level severity/margin pairs are built ONLY from pairs where the
    RSS margin actually exists (rss_t1_<side>_present == True). This matches
    the collection script's own gate language ("episode-level aggregation
    tolerates missing pairs") -- missing pairs are excluded, never zero-filled
    or paired against a severity value whose outcome is unknown. An episode
    with zero covered pairs contributes no row to that side's table; it is
    not scored as a null/zero result.

WHAT THIS FILE DELIBERATELY DOES NOT DO:
  - No mechanism/covariance decomposition. RSS's margin isn't a weighted sum
    of logged components the way rho_actual is (see analyse_unified.py's
    covariance_attribution) -- it's a single physics-derived quantity. There
    is nothing to decompose.
  - No blind pooling of front and rear. Front coverage (37/96 episodes at
    full scale) is materially weaker than rear (68/96) and is not the
    scenario's core question (Scenario C's hypothesis is about risk to a
    TRAILING vehicle, i.e. rear). Front is reported honestly as secondary/
    exploratory, never presented with the same weight as rear.
  - No claim that this run's rho_actual-vs-severity number IS the banked
    Table I number. It's a separate dataset (NPC traffic, different episode
    count) -- this file recomputes it on scenario_c_npc_rss.csv as an
    explicit REPLICATION CHECK, to verify NPC traffic didn't change the core
    finding, and reports it as its own row.

Run:  py -3.7 scripts\\analyse_scenario_c_npc_rss.py   (no CARLA needed)
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR     = r'C:\Users\jayad\Desktop\RPI_AINN\data'
ANALYSIS_DIR = r'C:\Users\jayad\Desktop\RPI_AINN\analysis'
SRC_FILE     = 'scenario_c_npc_rss.csv'
N_PERM       = 10000
N_BOOT       = 10000
SEED         = 42

SEVERITY_NAME = 'brake intensity x duration'


def severity_of(df):
    return df['brake_intensity_t'] * df['brake_duration_ticks_t']


# ── Statistical machinery -- byte-identical to analyse_unified.py, on purpose ──

def permutation_p(x, y, rng, n=N_PERM):
    obs = stats.pearsonr(x, y)[0]
    null = np.empty(n)
    xv, yv = np.asarray(x), np.asarray(y)
    for i in range(n):
        null[i] = stats.pearsonr(rng.permutation(xv), yv)[0]
    return obs, float(np.mean(np.abs(null) >= abs(obs)))


def bootstrap_ci_r(x, y, rng, n=N_BOOT):
    xv, yv = np.asarray(x), np.asarray(y)
    idx_all = np.arange(len(xv))
    rs = np.empty(n)
    for i in range(n):
        idx = rng.choice(idx_all, size=len(idx_all), replace=True)
        if np.std(xv[idx]) == 0 or np.std(yv[idx]) == 0:
            rs[i] = 0.0
        else:
            rs[i] = stats.pearsonr(xv[idx], yv[idx])[0]
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def load():
    df = pd.read_csv(os.path.join(DATA_DIR, SRC_FILE))
    n_raw = len(df)
    df = df.dropna(subset=['rho_actual'])
    if len(df) != n_raw:
        print(f"  NOTE: dropped {n_raw - len(df)} NaN row(s) from {SRC_FILE}")
    return df


def coverage_report(df, side):
    """Recompute the collection script's own Gate 2 / Gate 4 numbers directly
    from the CSV, so this analysis file is self-documenting and doesn't
    require cross-referencing the original run's console output."""
    present_col = f'rss_t1_{side}_present'
    n = len(df)
    present = df[present_col].astype(str) == 'True'
    pres_pct = present.mean()
    per_ep = df.assign(_present=present).groupby('episode_id')['_present'].sum()
    n_eps = len(per_ep)
    zero_eps = int((per_ep == 0).sum())
    total_hits = int(per_ep.sum())
    max_share = (per_ep.max() / total_hits) if total_hits else 0.0
    return {
        'side': side, 'pairs': n, 'present_pct': round(100 * pres_pct, 1),
        'episodes': n_eps, 'zero_coverage_episodes': zero_eps,
        'episodes_with_coverage': n_eps - zero_eps,
        'busiest_episode_share_pct': round(100 * max_share, 1),
        'total_hits': total_hits,
    }


def episode_table_rho(df):
    """Replication check: standard severity vs rho_actual ablation, on ALL
    pairs of this NPC dataset (rho_actual always exists, no coverage gating
    needed) -- same shape as analyse_unified.py's episode_table()."""
    tmp = df.copy()
    tmp['_sev'] = severity_of(tmp)
    return tmp.groupby('episode_id').agg(sev=('_sev', 'mean'),
                                         rho=('rho_actual', 'mean'),
                                         n_pairs=('_sev', 'size'))


def episode_table_margin(df, side):
    """Severity vs RSS margin, restricted to pairs where the margin actually
    exists. Both columns are averaged over the SAME covered subset per
    episode, so the episode-level point stays an honest average of matched
    (severity, margin) pairs -- never a severity mean over uncovered pairs
    paired against a margin mean over a different, smaller subset."""
    present_col = f'rss_t1_{side}_present'
    margin_col  = f'rss_t1_{side}_margin'
    covered = df[df[present_col].astype(str) == 'True'].copy()
    covered = covered.dropna(subset=[margin_col])
    covered['_sev'] = severity_of(covered)
    ep = covered.groupby('episode_id').agg(sev=('_sev', 'mean'),
                                           margin=(margin_col, 'mean'),
                                           n_covered=('_sev', 'size'))
    return ep


def run_ablation(ep, ycol, rng, label):
    pair_note = f"n={len(ep)} episodes"
    x, y = ep['sev'].values, ep[ycol].values
    r, perm_p = permutation_p(x, y, rng)
    _, analytic_p = stats.pearsonr(x, y)
    lo, hi = bootstrap_ci_r(x, y, rng)
    print(f"\n{label}  ({pair_note})")
    print(f"  episode-level r = {r:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  analytic p = {analytic_p:.3g} | permutation p "
          f"{'< %.4f' % (1.0 / N_PERM) if perm_p == 0 else '= %.5f' % perm_p}")
    return {
        'label': label, 'episodes': len(ep), 'episode_r': round(r, 4),
        'r_CI95_lo': round(lo, 4), 'r_CI95_hi': round(hi, 4),
        'analytic_p': analytic_p,
        'permutation_p': perm_p if perm_p > 0 else 1.0 / N_PERM,
    }, (r, lo, hi)


def main():
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    df = load()
    n_episodes_total = df['episode_id'].nunique()
    n_pairs_total = len(df)
    n_crashes = int((df['outcome'] == 'crash').sum())
    print("=" * 78)
    print(f"SCENARIO C — NPC traffic + RSS margin variant")
    print(f"  {SRC_FILE}: {n_episodes_total} episodes, {n_pairs_total} pairs, "
          f"{n_crashes} crashes ({100*n_crashes/max(n_pairs_total,1):.1f}%)")
    print("=" * 78)

    # ── Coverage, recomputed independently of the original run's console log ──
    print("\n--- Coverage (recomputed from CSV) ---")
    cov_rows = []
    for side in ('front', 'rear'):
        c = coverage_report(df, side)
        cov_rows.append(c)
        print(f"  {side}: {c['present_pct']}% of pairs present | "
              f"{c['episodes_with_coverage']}/{c['episodes']} episodes have "
              f"coverage | busiest episode = {c['busiest_episode_share_pct']}% "
              f"of all {side} hits")
    pd.DataFrame(cov_rows).to_csv(
        os.path.join(ANALYSIS_DIR, 'scenario_c_npc_rss_coverage.csv'), index=False)

    results = []
    scatter = {}

    # ── Replication check: does severity still predict rho_actual with NPCs? ──
    print("\n" + "=" * 78)
    print("REPLICATION CHECK — severity vs rho_actual, same test as banked C")
    print("(banked, no-NPC scenario_c_braking.csv: episode r = +0.770, see")
    print(" unified_table1.csv. This is a DIFFERENT dataset — NPC traffic, 96")
    print(" vs 400 episodes — not the same number restated. It should land in")
    print(" the same neighborhood if NPCs didn't disturb the core decision.)")
    print("=" * 78)
    ep_rho = episode_table_rho(df)
    row, _ = run_ablation(ep_rho, 'rho', rng, 'severity -> rho_actual (replication)')
    row['n_pairs_used'] = int(ep_rho['n_pairs'].sum())
    results.append(row)
    scatter['rho'] = (ep_rho['sev'], ep_rho['rho'], row['episode_r'], 'rho_actual (replication)')

    # ── PRIMARY: severity vs RSS rear margin ──────────────────────────────────
    print("\n" + "=" * 78)
    print("PRIMARY — severity vs RSS rear-margin at T+1 (risk to a TRAILING")
    print("vehicle — this is Scenario C's actual hypothesis: does hard braking")
    print("endanger whoever is behind you). Negative r is the physically")
    print("expected sign: harder/longer braking -> smaller (worse) safe margin.")
    print("=" * 78)
    ep_rear = episode_table_margin(df, 'rear')
    row, _ = run_ablation(ep_rear, 'margin', rng, 'severity -> RSS rear_margin_t1 (PRIMARY)')
    row['n_pairs_used'] = int(ep_rear['n_covered'].sum())
    results.append(row)
    scatter['rear'] = (ep_rear['sev'], ep_rear['margin'], row['episode_r'], 'RSS rear margin (primary)')

    # ── SECONDARY / exploratory: severity vs RSS front margin ────────────────
    print("\n" + "=" * 78)
    print("SECONDARY / EXPLORATORY — severity vs RSS front-margin at T+1 (risk")
    print("relative to a LEADING vehicle). Materially lower, more clumped")
    print("coverage than rear (see coverage report above) — this is NOT")
    print("Scenario C's core hypothesis and should not be given equal weight")
    print("to the rear result in the paper. Reported for completeness only.")
    print("=" * 78)
    ep_front = episode_table_margin(df, 'front')
    row, _ = run_ablation(ep_front, 'margin', rng, 'severity -> RSS front_margin_t1 (secondary)')
    row['n_pairs_used'] = int(ep_front['n_covered'].sum())
    results.append(row)
    scatter['front'] = (ep_front['sev'], ep_front['margin'], row['episode_r'], 'RSS front margin (secondary)')

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(ANALYSIS_DIR, 'scenario_c_npc_rss_results.csv'), index=False)
    ep_rho.to_csv(os.path.join(ANALYSIS_DIR, 'scenario_c_npc_rss_episodes_rho.csv'))
    ep_rear.to_csv(os.path.join(ANALYSIS_DIR, 'scenario_c_npc_rss_episodes_rear.csv'))
    ep_front.to_csv(os.path.join(ANALYSIS_DIR, 'scenario_c_npc_rss_episodes_front.csv'))

    # ── Interpretation: does the RSS result agree with the primary rho result? ──
    print("\n" + "=" * 78)
    print("INTERPRETATION")
    print("=" * 78)
    r_rho = results_df.loc[results_df['label'].str.contains('replication'), 'episode_r'].iloc[0]
    r_rear = results_df.loc[results_df['label'].str.contains('PRIMARY'), 'episode_r'].iloc[0]
    p_rear = results_df.loc[results_df['label'].str.contains('PRIMARY'), 'permutation_p'].iloc[0]
    print(f"  rho_actual replication : r = {r_rho:+.4f} (banked no-NPC C: +0.770)")
    print(f"  RSS rear margin (primary): r = {r_rear:+.4f}, permutation p "
          f"{'< 1e-4' if p_rear <= 1.0/N_PERM else '= %.4f' % p_rear}")
    if r_rear < 0 and p_rear < 0.05:
        print("  -> Same conclusion as the primary rho-based result, on an")
        print("     independent, externally-defined (RSS) safety measure:")
        print("     harder/longer braking predicts a WORSE safe-following")
        print("     margin for a trailing vehicle at T+1. Carry-forward risk")
        print("     is not an artifact of this project's own risk formula.")
    elif r_rear < 0:
        print("  -> Sign agrees with the hypothesis (worse margin under harder")
        print("     braking) but did NOT clear conventional significance at")
        print("     this coverage level (n episodes above). Report as")
        print("     directionally consistent, not as a confirmed result --")
        print("     do not round p up to significant in the paper.")
    else:
        print("  -> Sign does NOT match the hypothesis direction. Do not")
        print("     paper this as confirmatory — investigate before writing")
        print("     anything about RSS agreement.")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    labels = [results_df['label'].iloc[i].split(' (')[0].split(' -> ')[-1] for i in range(len(results_df))]
    rs   = results_df['episode_r'].values
    los  = results_df['episode_r'].values - results_df['r_CI95_lo'].values
    his  = results_df['r_CI95_hi'].values - results_df['episode_r'].values
    colors = ['#2e7d32' if v > 0 else '#c62828' for v in rs]
    ax.bar(labels, rs, yerr=[los, his], capsize=6, color=colors)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_ylabel('episode-level r (severity vs outcome at T+1)')
    ax.set_title('Scenario C + NPC traffic: severity ablation\n(bootstrap 95% CI)')
    plt.setp(ax.get_xticklabels(), rotation=20, ha='right')

    for ax, key in zip([axes[0, 1], axes[1, 0]], ['rear', 'front']):
        x, y, r, label = scatter[key]
        ax.scatter(x, y, s=14, alpha=0.6)
        if len(x) > 1:
            z = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, np.polyval(z, xs), 'r-', lw=1.5)
        ax.axhline(0, color='gray', lw=0.6, ls=':')
        ax.set_title(f'{label}\n(episode means, r={r:+.2f}, n={len(x)})')
        ax.set_xlabel('mean decision-T severity (covered pairs only)')
        ax.set_ylabel('mean RSS margin at T+1 (m)')

    ax = axes[1, 1]
    cov_df = pd.DataFrame(cov_rows)
    ax.bar(cov_df['side'], cov_df['present_pct'], color='#546e7a', label='pair-level present %')
    ax.bar(cov_df['side'], cov_df['episodes_with_coverage'] / cov_df['episodes'] * 100,
           color='#90a4ae', alpha=0.6, width=0.4, label='episodes w/ any coverage %')
    ax.axhline(30, color='k', lw=0.8, ls='--', label='~30% workable bar')
    ax.set_ylabel('%')
    ax.set_title('RSS coverage: pair-level vs episode-level')
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(ANALYSIS_DIR, 'scenario_c_npc_rss_figures.png')
    plt.savefig(fig_path, dpi=150)

    print(f"\nOutputs written to {ANALYSIS_DIR}:")
    for f_ in ['scenario_c_npc_rss_coverage.csv', 'scenario_c_npc_rss_results.csv',
               'scenario_c_npc_rss_episodes_rho.csv', 'scenario_c_npc_rss_episodes_rear.csv',
               'scenario_c_npc_rss_episodes_front.csv', 'scenario_c_npc_rss_figures.png']:
        print("  " + f_)


if __name__ == '__main__':
    main()
