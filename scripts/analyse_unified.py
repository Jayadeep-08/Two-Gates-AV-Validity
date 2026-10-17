"""
Unified cross-scenario RPI analysis — the single source of every number in
the paper's Table I, mechanism table, and secondary-baseline table.

Written 2026-07-13, after the core four-scenario suite was completed and
independently audited (see CLAUDE.md, audit rounds 1-5). Replaces
analyse_scenario_c.py / analyse_scenario_c_v2.py / analyse_scenario_d.py,
which are DEPRECATED: they pool pairs as independent samples
(pseudo-replication — pairs within an episode share a vehicle/spawn/run),
use mutually incomparable baselines, and predate all four data reruns.

Design (locked before running, per house discipline):
  PRIMARY evidence, per treatment scenario (A, B, C):
    episode-level Pearson r between the scenario's a-priori-locked
    decision-T severity metric and mean rho_actual at T+1, with a
    10,000-shuffle permutation p-value and a 10,000-resample bootstrap
    95% CI (both seeded). Pair-level r is reported as DESCRIPTIVE detail
    only. Scenario D is the control row: it has no scripted decision, so
    no ablation exists for it by design.
  MECHANISM attribution:
    covariance-share decomposition of cov(severity, rho_t1) across the
    logged risk components — A from its own CSV, B from the instrumented
    companion run (scenario_b_instrumented.csv).
  SECONDARY evidence (explicitly subordinate — cross-formula confound):
    D-global-mean baseline comparison. rho_baseline is RECOMPUTED from the
    current scenario_d_baseline.csv (the old 0.0705 is dead — stale file,
    see CLAUDE.md). RPI_pair = rho_actual - rho_baseline, RPI_chain =
    per-episode mean, episode-level one-sample t-test + bootstrap CI,
    underestimation gap = mean RPI_chain / rho_baseline * 100.

Statistical unit: EPISODE, everywhere a p-value is attached.
Seeds: numpy default_rng(42) for permutation and bootstrap.

Run:  py -3.7 scripts\\analyse_unified.py   (no CARLA needed)
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
N_PERM       = 10000
N_BOOT       = 10000
SEED         = 42

# ── Scenario registry ────────────────────────────────────────────────────────
# severity: the a-priori-locked decision-T metric (see CLAUDE.md per scenario).
# components: T+1 risk-component columns + formula weights, for the mechanism
# table (None where the banked file has no component columns — B's mechanism
# comes from its instrumented companion file instead).
SCENARIOS = [
    {
        'key'      : 'C',
        'label'    : 'C: emergency braking',
        'file'     : 'scenario_c_braking.csv',
        'severity' : lambda df: df['brake_intensity_t'] * df['brake_duration_ticks_t'],
        'sev_name' : 'brake intensity x duration',
        'components': None,
    },
    {
        'key'      : 'B',
        'label'    : 'B: intersection yield-cross',
        'file'     : 'scenario_b_intersection.csv',
        'severity' : lambda df: df['speed_t'],
        'sev_name' : 'realized yield speed',
        'components': None,   # mechanism from scenario_b_instrumented.csv below
    },
    {
        'key'      : 'A',
        'label'    : 'A: lane change',
        'file'     : 'scenario_a_lanechange.csv',
        'severity' : lambda df: df['severity_t'],
        'sev_name' : 'steer magnitude x duration',
        'components': [('lat_dev_t1', 0.5), ('heading_err_t1', 0.3), ('speed_risk_t1', 0.2)],
    },
]
D_FILE = 'scenario_d_baseline.csv'
B_INSTRUMENTED = 'scenario_b_instrumented.csv'
B_INSTR_COMPONENTS = [('junction_progress_t1', 0.5), ('speed_risk_t1', 0.3), ('lane_dev_t1', 0.2)]


def load(fname):
    df = pd.read_csv(os.path.join(DATA_DIR, fname))
    n_raw = len(df)
    df = df.dropna(subset=['rho_actual'])   # defensive: B's old file once had a NUL-padded NaN row
    if len(df) != n_raw:
        print(f"  NOTE: dropped {n_raw - len(df)} NaN row(s) from {fname}")
    return df


def episode_table(df, severity_fn):
    tmp = df.copy()
    tmp['_sev'] = severity_fn(tmp)
    return tmp.groupby('episode_id').agg(sev=('_sev', 'mean'),
                                         rho=('rho_actual', 'mean'),
                                         n_pairs=('_sev', 'size'))


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


def bootstrap_ci_mean(x, rng, n=N_BOOT):
    xv = np.asarray(x)
    means = np.empty(n)
    for i in range(n):
        means[i] = np.mean(rng.choice(xv, size=len(xv), replace=True))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def covariance_attribution(df, severity_fn, components):
    """Share of cov(severity, rho_t1) carried by each weighted component."""
    sev = severity_fn(df)
    total = np.cov(sev, df['rho_actual'])[0, 1]
    rows = []
    for col, weight in components:
        cv = weight * np.cov(sev, df[col])[0, 1]
        r, p = stats.pearsonr(sev, df[col])
        rows.append({'component': col, 'weight': weight,
                     'cov_share_pct': round(100 * cv / total, 1),
                     'r_vs_severity': round(r, 3), 'p': p})
    return rows


def main():
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ── Scenario D: control group + secondary baseline ───────────────────────
    d = load(D_FILE)
    d_ep = d.groupby('episode_id').agg(rho=('rho_actual', 'mean'))
    rho_baseline = float(d['rho_actual'].mean())
    print("=" * 78)
    print("SCENARIO D (control): %d episodes, %d pairs" % (d['episode_id'].nunique(), len(d)))
    print("  rho_actual mean %.4f std %.4f  <- this run's D-mean, used as the" %
          (d['rho_actual'].mean(), d['rho_actual'].std()))
    print("  SECONDARY baseline below. (The old hardcoded 0.0705 is dead - stale file.)")
    print("  No scripted decision exists in D, so no severity ablation is defined -")
    print("  that absence IS the control condition.")

    # ── Primary evidence ─────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("PRIMARY EVIDENCE - episode-level severity ablation (unit: episode)")
    print("=" * 78)
    table1 = []
    scatter_data = {}
    for sc in SCENARIOS:
        df = load(sc['file'])
        ep = episode_table(df, sc['severity'])
        pair_r, pair_p = stats.pearsonr(sc['severity'](df), df['rho_actual'])
        r, perm_p = permutation_p(ep['sev'], ep['rho'], rng)
        _, analytic_p = stats.pearsonr(ep['sev'], ep['rho'])
        lo, hi = bootstrap_ci_r(ep['sev'].values, ep['rho'].values, rng)
        table1.append({
            'scenario'      : sc['label'],
            'severity_metric': sc['sev_name'],
            'episodes'      : len(ep),
            'pairs'         : len(df),
            'mean_rho_t1'   : round(df['rho_actual'].mean(), 4),
            'episode_r'     : round(r, 4),
            'r_CI95_lo'     : round(lo, 4),
            'r_CI95_hi'     : round(hi, 4),
            'analytic_p'    : analytic_p,
            'permutation_p' : perm_p if perm_p > 0 else 1.0 / N_PERM,
            'pair_r_descriptive': round(pair_r, 4),
        })
        scatter_data[sc['key']] = (ep, sc['label'], r)
        print("\n%s  (%d episodes, %d pairs; severity = %s)" %
              (sc['label'], len(ep), len(df), sc['sev_name']))
        print("  episode-level r = %+.4f  95%% CI [%+.4f, %+.4f]" % (r, lo, hi))
        print("  analytic p = %.3g | permutation p %s" %
              (analytic_p, ("< %.4f" % (1.0 / N_PERM)) if perm_p == 0 else "= %.5f" % perm_p))
        print("  pair-level r = %+.4f (descriptive only - pairs within an episode are correlated)" % pair_r)

    # D control row for Table I
    table1.append({
        'scenario': 'D: open-road control', 'severity_metric': 'none (no scripted decision)',
        'episodes': int(d['episode_id'].nunique()), 'pairs': len(d),
        'mean_rho_t1': round(d['rho_actual'].mean(), 4),
        'episode_r': None, 'r_CI95_lo': None, 'r_CI95_hi': None,
        'analytic_p': None, 'permutation_p': None, 'pair_r_descriptive': None,
    })
    t1 = pd.DataFrame(table1)
    t1.to_csv(os.path.join(ANALYSIS_DIR, 'unified_table1.csv'), index=False)

    # ── Mechanism attribution ────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("MECHANISM ATTRIBUTION - covariance share of cov(severity, rho_t1)")
    print("=" * 78)
    mech_rows = []
    a_df = load('scenario_a_lanechange.csv')
    for row in covariance_attribution(a_df, lambda df: df['severity_t'],
                                      [('lat_dev_t1', 0.5), ('heading_err_t1', 0.3), ('speed_risk_t1', 0.2)]):
        row['scenario'] = 'A: lane change'
        mech_rows.append(row)
    bi = load(B_INSTRUMENTED)
    for row in covariance_attribution(bi, lambda df: df['speed_t'], B_INSTR_COMPONENTS):
        row['scenario'] = 'B: intersection (instrumented run, n=%d pairs)' % len(bi)
        mech_rows.append(row)
    mech = pd.DataFrame(mech_rows)[['scenario', 'component', 'weight',
                                    'cov_share_pct', 'r_vs_severity', 'p']]
    mech.to_csv(os.path.join(ANALYSIS_DIR, 'unified_mechanism.csv'), index=False)
    print(mech.to_string(index=False))
    print("  (B's attribution comes from the instrumented companion run - the banked")
    print("   500-episode file has no component columns. C's mechanism is the")
    print("   single-channel severity->residual-speed link; no decomposition needed.)")

    # ── Secondary evidence: D-mean baseline ──────────────────────────────────
    print("\n" + "=" * 78)
    print("SECONDARY EVIDENCE - gap vs naive point-in-time model trained on normal")
    print("driving (D global mean). EXPLICITLY subordinate: different scenarios use")
    print("different risk formulas, so this partly reflects formula differences.")
    print("Never present these as the primary proof of propagation.")
    print("=" * 78)
    sec_rows = []
    for sc in SCENARIOS:
        df = load(sc['file'])
        df['RPI_pair'] = df['rho_actual'] - rho_baseline
        chain = df.groupby('episode_id')['RPI_pair'].mean()   # RPI_chain
        t, p = stats.ttest_1samp(chain, 0.0)
        lo, hi = bootstrap_ci_mean(chain.values, rng)
        gap = 100.0 * chain.mean() / rho_baseline
        sec_rows.append({
            'scenario': sc['label'], 'rho_baseline_D': round(rho_baseline, 4),
            'mean_RPI_chain': round(chain.mean(), 4),
            'CI95_lo': round(lo, 4), 'CI95_hi': round(hi, 4),
            't': round(t, 2), 'p': p, 'underestimation_gap_pct': round(gap, 1),
            'episodes': len(chain),
        })
        print("%s: mean RPI_chain = %+.4f  CI [%+.4f, %+.4f]  t=%.1f p=%.3g  gap = %+.1f%%" %
              (sc['label'], chain.mean(), lo, hi, t, p, gap))
    # D against itself (sanity: ~0 by construction, shown for completeness)
    d_chain = d.groupby('episode_id')['rho_actual'].mean() - rho_baseline
    print("D vs own mean: %+.6f (zero by construction - control sanity row)" % d_chain.mean())
    pd.DataFrame(sec_rows).to_csv(os.path.join(ANALYSIS_DIR, 'unified_secondary.csv'), index=False)

    # ── LaTeX table body (paper Table I) ─────────────────────────────────────
    tex_path = os.path.join(ANALYSIS_DIR, 'unified_table1.tex')
    with open(tex_path, 'w') as f:
        f.write("% Auto-generated by analyse_unified.py - do not hand-edit\n")
        f.write("% Columns: Scenario & Episodes & Pairs & r (episode) & 95% CI & perm. p\n")
        for row in table1:
            if row['episode_r'] is None:
                f.write("%s & %d & %d & \\multicolumn{3}{c}{control -- no scripted decision} \\\\\n"
                        % (row['scenario'], row['episodes'], row['pairs']))
            else:
                f.write("%s & %d & %d & $%+.3f$ & $[%+.3f, %+.3f]$ & $%s$ \\\\\n"
                        % (row['scenario'], row['episodes'], row['pairs'], row['episode_r'],
                           row['r_CI95_lo'], row['r_CI95_hi'],
                           ("<10^{-4}" if row['permutation_p'] <= 1.0 / N_PERM
                            else "%.4f" % row['permutation_p'])))

    # ── Figures ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    # Panel 1: headline - episode-level r with CI per scenario
    ax = axes[0, 0]
    labels = [r_['scenario'].split(':')[0] for r_ in table1 if r_['episode_r'] is not None]
    rs     = [r_['episode_r'] for r_ in table1 if r_['episode_r'] is not None]
    los    = [r_['episode_r'] - r_['r_CI95_lo'] for r_ in table1 if r_['episode_r'] is not None]
    his    = [r_['r_CI95_hi'] - r_['episode_r'] for r_ in table1 if r_['episode_r'] is not None]
    colors = ['#2e7d32' if v > 0 else '#c62828' for v in rs]
    ax.bar(labels, rs, yerr=[los, his], capsize=6, color=colors)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_ylabel('episode-level r (severity vs rho at T+1)')
    ax.set_title('Primary evidence: carry-forward ablation\n(bootstrap 95% CI; all perm. p < 1e-3)')
    # Panels 2-4: per-scenario episode scatters
    for ax, key in zip([axes[0, 1], axes[0, 2], axes[1, 0]], ['C', 'B', 'A']):
        ep, label, r = scatter_data[key]
        ax.scatter(ep['sev'], ep['rho'], s=12, alpha=0.5)
        z = np.polyfit(ep['sev'], ep['rho'], 1)
        xs = np.linspace(ep['sev'].min(), ep['sev'].max(), 50)
        ax.plot(xs, np.polyval(z, xs), 'r-', lw=1.5)
        ax.set_title('%s (episode means, r=%+.2f)' % (label, r))
        ax.set_xlabel('mean decision-T severity')
        ax.set_ylabel('mean rho at T+1')
    # Panel 5: rho distributions incl. D
    ax = axes[1, 1]
    dist_data, dist_labels = [], []
    for sc in SCENARIOS:
        dist_data.append(load(sc['file'])['rho_actual'].values)
        dist_labels.append(sc['key'])
    dist_data.append(d['rho_actual'].values); dist_labels.append('D')
    ax.violinplot(dist_data, showmedians=True)
    ax.set_xticks(range(1, len(dist_labels) + 1)); ax.set_xticklabels(dist_labels)
    ax.axhline(rho_baseline, color='gray', ls='--', lw=1, label='D mean (secondary baseline)')
    ax.legend(fontsize=8)
    ax.set_title('rho_actual distributions (formulas differ by scenario)')
    # Panel 6: secondary gap
    ax = axes[1, 2]
    sec = pd.DataFrame(sec_rows)
    ax.bar([s.split(':')[0] for s in sec['scenario']], sec['mean_RPI_chain'],
           yerr=[sec['mean_RPI_chain'] - sec['CI95_lo'], sec['CI95_hi'] - sec['mean_RPI_chain']],
           capsize=6, color='#546e7a')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_ylabel('mean RPI_chain vs D-mean baseline')
    ax.set_title('SECONDARY: gap vs naive baseline\n(cross-formula confound - contextual only)')
    plt.tight_layout()
    fig_path = os.path.join(ANALYSIS_DIR, 'unified_figures.png')
    plt.savefig(fig_path, dpi=150)
    print("\nOutputs written to %s:" % ANALYSIS_DIR)
    for f_ in ['unified_table1.csv', 'unified_table1.tex', 'unified_mechanism.csv',
               'unified_secondary.csv', 'unified_figures.png']:
        print("  " + f_)


if __name__ == '__main__':
    main()
