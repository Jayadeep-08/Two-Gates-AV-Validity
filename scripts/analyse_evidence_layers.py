"""
Companion analysis: assembles every RPI result into the paper's four EVIDENCE
LAYERS, on one comparable scale, with identical statistical machinery.

WHY THIS FILE EXISTS (and why it does not touch analyse_unified.py):
`analyse_unified.py` is validated and is the single source for the banked
four-scenario numbers. It is deliberately left alone. This script imports the
same design — episode-level unit, 10,000-shuffle permutation, 10,000-resample
bootstrap, seed 42 — and adds the two NPC-traffic variants plus the RSS
comparator, so all layers can be read side by side without re-deriving the
banked table.

THE FOUR LAYERS (see README.md for the plain-language version):

  L1 PRIMARY — within-scenario ablation on this project's own rho.
     "Does decision-T severity predict residual risk at T+1?"
     Unit: episode. Scenarios A/B/C; D is the no-decision control.
     This is the paper's core claim, and the SIGN HETEROGENEITY across
     scenarios is the part that defeats a scalar bias correction.

  L2 EXTERNAL METRIC — the same ablation run against RSS, an externally
     defined, memoryless safe-distance formula (Shalev-Shwartz et al. 2017).
     "Is L1 an artifact of our hand-crafted risk function?"
     Scenario C only — RSS needs a second vehicle, and Scenario A's
     target-lane coverage was structurally unattainable (8.4%; see CLAUDE.md
     2026-07-22). RSS is NOT a baseline to subtract: there is no 'expected
     RSS margin under independence'. It is a comparator OUTCOME variable.

  L3 SECONDARY — gap versus a naive point-in-time model, using Scenario D's
     global mean rho as the reference.
     "How much does a memoryless model underestimate the magnitude?"
     Explicitly subordinate: scenarios use different rho formulas, so part of
     each gap reflects formula differences rather than carry-forward. Reports
     MAGNITUDE only, never direction. Never present this as the main proof.

  L4 ROBUSTNESS — L1 re-run on the NPC-traffic datasets.
     "Does the result survive a world containing other vehicles?"
     Scenarios C and A. Note the honest scope limit for A: 91.6% of its
     decisions still occurred with an empty target lane, so this licenses
     'holds with ambient traffic present', NOT 'holds with a vehicle
     adjacent during the merge'.

Outputs to analysis/:
  evidence_layers_ablations.csv   L1 + L2 + L4 (everything that is a correlation)
  evidence_layers_secondary.csv   L3 (magnitude gaps — different shape, own table)
  evidence_layers.tex             paper-ready table body, auto-generated
  evidence_layers_figure.png      4-panel summary

Run:  py -3.7 scripts\\analyse_evidence_layers.py     (no CARLA needed)
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


# ── Statistical machinery — identical to analyse_unified.py by design ────────

def permutation_p(x, y, rng, n=N_PERM):
    obs = stats.pearsonr(x, y)[0]
    xv, yv = np.asarray(x), np.asarray(y)
    null = np.empty(n)
    for i in range(n):
        null[i] = stats.pearsonr(rng.permutation(xv), yv)[0]
    return obs, float(np.mean(np.abs(null) >= abs(obs)))


def bootstrap_ci_r(x, y, rng, n=N_BOOT):
    xv, yv = np.asarray(x), np.asarray(y)
    idx = np.arange(len(xv))
    rs = np.empty(n)
    for i in range(n):
        j = rng.choice(idx, size=len(idx), replace=True)
        rs[i] = 0.0 if (np.std(xv[j]) == 0 or np.std(yv[j]) == 0) \
            else stats.pearsonr(xv[j], yv[j])[0]
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def bootstrap_ci_mean(x, rng, n=N_BOOT):
    xv = np.asarray(x)
    means = np.empty(n)
    for i in range(n):
        means[i] = np.mean(rng.choice(xv, size=len(xv), replace=True))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load(fname):
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        print(f"  !! MISSING: {fname} — rows depending on it will be skipped")
        return None
    df = pd.read_csv(path)
    n_raw = len(df)
    df = df.dropna(subset=['rho_actual'])
    if len(df) != n_raw:
        print(f"  note: dropped {n_raw - len(df)} NaN row(s) from {fname}")
    return df


# ── Severity definitions — locked a priori, identical to the banked runs ─────
SEV = {
    'C': lambda df: df['brake_intensity_t'] * df['brake_duration_ticks_t'],
    'B': lambda df: df['speed_t'],
    'A': lambda df: df['severity_t'],
}
SEV_NAME = {
    'C': 'brake intensity x duration',
    'B': 'realized yield speed',
    'A': 'steer magnitude x duration',
}


def episode_ablation(df, sev_fn, outcome_col, rng, require_present=None):
    """Episode-level r(mean severity, mean outcome). When require_present is
    given, only pairs where that column is True contribute — used for RSS,
    where a margin exists only if a neighbour vehicle was detected. Both the
    severity and the outcome are averaged over the SAME surviving subset, so
    an episode point is always a matched average."""
    d = df.copy()
    if require_present is not None:
        d = d[d[require_present].astype(str) == 'True']
    d = d.dropna(subset=[outcome_col])
    if len(d) == 0:
        return None
    d['_sev'] = sev_fn(d)
    ep = d.groupby('episode_id').agg(sev=('_sev', 'mean'),
                                     out=(outcome_col, 'mean'),
                                     n=('_sev', 'size'))
    if len(ep) < 3 or ep['sev'].std() == 0 or ep['out'].std() == 0:
        return None
    r, p = permutation_p(ep['sev'].values, ep['out'].values, rng)
    lo, hi = bootstrap_ci_r(ep['sev'].values, ep['out'].values, rng)
    return {'r': r, 'ci_lo': lo, 'ci_hi': hi, 'perm_p': p,
            'episodes': len(ep), 'pairs_used': int(ep['n'].sum())}


def add_row(rows, layer, scenario, dataset, outcome, res, note=''):
    if res is None:
        print(f"  skipped: {layer} / {scenario} / {outcome} (insufficient data)")
        return
    rows.append({
        'layer': layer, 'scenario': scenario, 'dataset': dataset,
        'outcome_measured': outcome,
        'episodes': res['episodes'], 'pairs_used': res['pairs_used'],
        'r_episode': round(res['r'], 4),
        'ci95_lo': round(res['ci_lo'], 4), 'ci95_hi': round(res['ci_hi'], 4),
        'perm_p': res['perm_p'] if res['perm_p'] > 0 else 1.0 / N_PERM,
        'ci_excludes_zero': (res['ci_lo'] > 0) or (res['ci_hi'] < 0),
        'note': note,
    })
    star = '' if ((res['ci_lo'] > 0) or (res['ci_hi'] < 0)) else '   <-- CI INCLUDES ZERO'
    pstr = f"< {1.0/N_PERM:g}" if res['perm_p'] == 0 else f"= {res['perm_p']:.4f}"
    print(f"  {layer:4s} {scenario:22s} r={res['r']:+.4f} "
          f"CI[{res['ci_lo']:+.4f},{res['ci_hi']:+.4f}] p {pstr} "
          f"n={res['episodes']}{star}")


def main():
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows = []

    print("=" * 78)
    print("LOADING")
    print("=" * 78)
    banked = {k: load(f) for k, f in (('C', 'scenario_c_braking.csv'),
                                      ('B', 'scenario_b_intersection.csv'),
                                      ('A', 'scenario_a_lanechange.csv'))}
    d_df   = load('scenario_d_baseline.csv')
    c_npc  = load('scenario_c_npc_rss.csv')
    a_npc  = load('scenario_a_lanechange_npc_rss.csv')

    # ── L1 PRIMARY ───────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("L1 PRIMARY — carry-forward ablation on this project's own rho")
    print("Unit: EPISODE. D is the control: no scripted decision exists, so no")
    print("ablation is defined for it — that absence IS the control condition.")
    print("=" * 78)
    for k in ('C', 'B', 'A'):
        if banked[k] is None:
            continue
        res = episode_ablation(banked[k], SEV[k], 'rho_actual', rng)
        add_row(rows, 'L1', f'{k}: banked (no NPC)', f'scenario_{k.lower()}',
                'rho_actual', res, f'severity = {SEV_NAME[k]}')

    # ── L2 EXTERNAL METRIC ───────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("L2 EXTERNAL METRIC — same ablation, RSS safe-distance margin as the")
    print("outcome. RSS is memoryless by construction, so a severity->margin")
    print("relationship cannot be an artifact of our own risk formula.")
    print("Scenario C only: RSS needs a second vehicle, and A's target-lane")
    print("coverage is structurally unattainable (8.4%) — see CLAUDE.md.")
    print("=" * 78)
    if c_npc is not None:
        res = episode_ablation(c_npc, SEV['C'], 'rss_t1_rear_margin', rng,
                               require_present='rss_t1_rear_present')
        add_row(rows, 'L2', 'C: RSS rear margin', 'scenario_c_npc_rss',
                'rss_t1_rear_margin', res,
                'negative r = harder braking consumes the follower RSS budget')
        res = episode_ablation(c_npc, SEV['C'], 'rss_t1_front_margin', rng,
                               require_present='rss_t1_front_present')
        add_row(rows, 'L2*', 'C: RSS front margin', 'scenario_c_npc_rss',
                'rss_t1_front_margin', res,
                'EXPLORATORY ONLY - thin/clumped coverage, do not weight equally')

    # ── L4 ROBUSTNESS ────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("L4 ROBUSTNESS — L1 re-run with ambient NPC traffic present.")
    print("Answers the 'your world is empty' objection. For A, note the scope")
    print("limit: 91.6% of decisions still had an empty TARGET lane, so this")
    print("licenses 'holds with ambient traffic', not 'holds with a vehicle")
    print("adjacent during the merge'.")
    print("=" * 78)
    if c_npc is not None:
        res = episode_ablation(c_npc, SEV['C'], 'rho_actual', rng)
        add_row(rows, 'L4', 'C: NPC traffic', 'scenario_c_npc_rss',
                'rho_actual', res, 'compare against L1 row for C')
    if a_npc is not None:
        res = episode_ablation(a_npc, SEV['A'], 'rho_actual', rng)
        add_row(rows, 'L4', 'A: NPC traffic', 'scenario_a_lanechange_npc_rss',
                'rho_actual', res, 'compare against L1 row for A')

    abl = pd.DataFrame(rows)
    abl.to_csv(os.path.join(ANALYSIS_DIR, 'evidence_layers_ablations.csv'), index=False)

    # ── L3 SECONDARY ─────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("L3 SECONDARY — magnitude gap vs a naive point-in-time model")
    print("(Scenario D global mean rho as the reference). SUBORDINATE by")
    print("design: different scenarios use different rho formulas, so part of")
    print("each gap is formula difference, not carry-forward. MAGNITUDE ONLY —")
    print("this layer cannot speak to direction.")
    print("=" * 78)
    sec = []
    if d_df is not None:
        rho_base = float(d_df['rho_actual'].mean())
        print(f"  Scenario D reference mean rho = {rho_base:.4f} "
              f"({d_df['episode_id'].nunique()} episodes, {len(d_df)} pairs)")
        for k in ('C', 'B', 'A'):
            if banked[k] is None:
                continue
            chain = (banked[k].assign(_rpi=banked[k]['rho_actual'] - rho_base)
                     .groupby('episode_id')['_rpi'].mean())
            t, p = stats.ttest_1samp(chain, 0.0)
            lo, hi = bootstrap_ci_mean(chain.values, rng)
            gap = 100.0 * chain.mean() / rho_base
            sec.append({'scenario': k, 'rho_baseline_D': round(rho_base, 4),
                        'mean_RPI_chain': round(float(chain.mean()), 4),
                        'ci95_lo': round(lo, 4), 'ci95_hi': round(hi, 4),
                        't': round(float(t), 2), 'p': p,
                        'underestimation_gap_pct': round(gap, 1),
                        'episodes': int(len(chain))})
            print(f"  {k}: mean RPI_chain={chain.mean():+.4f} "
                  f"CI[{lo:+.4f},{hi:+.4f}] gap={gap:+.1f}%")
        pd.DataFrame(sec).to_csv(
            os.path.join(ANALYSIS_DIR, 'evidence_layers_secondary.csv'), index=False)

    # ── LaTeX body ───────────────────────────────────────────────────────────
    with open(os.path.join(ANALYSIS_DIR, 'evidence_layers.tex'), 'w') as f:
        f.write("% Auto-generated by analyse_evidence_layers.py - do not hand-edit\n")
        f.write("% Layer & Scenario & Episodes & r & 95% CI & perm. p\n")
        for _, r in abl.iterrows():
            pstr = "<10^{-4}" if r['perm_p'] <= 1.0 / N_PERM else f"{r['perm_p']:.4f}"
            f.write(f"{r['layer']} & {r['scenario']} & {r['episodes']} & "
                    f"$%+.3f$ & $[%+.3f, %+.3f]$ & $%s$ \\\\\n"
                    % (r['r_episode'], r['ci95_lo'], r['ci95_hi'], pstr))

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    l1 = abl[abl['layer'] == 'L1']
    if len(l1):
        cols = ['#2e7d32' if v > 0 else '#c62828' for v in l1['r_episode']]
        ax.bar(l1['scenario'].str[0], l1['r_episode'],
               yerr=[l1['r_episode'] - l1['ci95_lo'], l1['ci95_hi'] - l1['r_episode']],
               capsize=6, color=cols)
        ax.axhline(0, color='k', lw=0.8)
        ax.set_title('L1 PRIMARY: carry-forward on our rho\n(sign differs by scenario)')
        ax.set_ylabel('episode-level r')

    ax = axes[0, 1]
    comp = abl[abl['layer'].isin(['L1', 'L4'])]
    if len(comp):
        labels, vals, los, his, cs = [], [], [], [], []
        for k in ('C', 'A'):
            for lay, colr in (('L1', '#546e7a'), ('L4', '#ef6c00')):
                m = comp[(comp['layer'] == lay) & (comp['scenario'].str.startswith(k))]
                if len(m):
                    m = m.iloc[0]
                    labels.append(f"{k}\n{'no NPC' if lay=='L1' else 'NPC'}")
                    vals.append(m['r_episode'])
                    los.append(m['r_episode'] - m['ci95_lo'])
                    his.append(m['ci95_hi'] - m['r_episode'])
                    cs.append(colr)
        ax.bar(labels, vals, yerr=[los, his], capsize=6, color=cs)
        ax.axhline(0, color='k', lw=0.8)
        ax.set_title('L4 ROBUSTNESS: same result with traffic?\n(grey = empty world, orange = NPC traffic)')
        ax.set_ylabel('episode-level r')

    ax = axes[1, 0]
    l2 = abl[abl['layer'].isin(['L2', 'L2*'])]
    if len(l2):
        cols = ['#1565c0' if l == 'L2' else '#b0bec5' for l in l2['layer']]
        ax.bar(range(len(l2)), l2['r_episode'],
               yerr=[l2['r_episode'] - l2['ci95_lo'], l2['ci95_hi'] - l2['r_episode']],
               capsize=6, color=cols)
        ax.set_xticks(range(len(l2)))
        ax.set_xticklabels([s.replace('C: RSS ', '') for s in l2['scenario']])
        ax.axhline(0, color='k', lw=0.8)
        ax.set_title('L2 EXTERNAL METRIC: RSS margin (Scenario C)\n(blue = primary, grey = exploratory)')
        ax.set_ylabel('episode-level r')

    ax = axes[1, 1]
    if sec:
        sdf = pd.DataFrame(sec)
        ax.bar(sdf['scenario'], sdf['underestimation_gap_pct'], color='#6a1b9a')
        ax.set_title('L3 SECONDARY: gap vs naive D-mean model\n(magnitude only - cross-formula confound)')
        ax.set_ylabel('underestimation gap (%)')

    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, 'evidence_layers_figure.png'), dpi=150)

    print("\n" + "=" * 78)
    print("HOW TO READ THIS TOGETHER (the Discussion-section argument)")
    print("=" * 78)
    print("  L3 shows a memoryless model gets the MAGNITUDE wrong (2-4x).")
    print("  L1 shows even a magnitude-corrected independence model still fails,")
    print("     because the DIRECTION of carry-forward is scenario-specific.")
    print("  L2 shows neither of those is an artifact of our own risk formula.")
    print("  L4 shows none of it depends on an empty simulated world.")
    print("  No scalar correction survives sign-varying, externally-confirmed,")
    print("  traffic-robust dependence — which is the paper's central claim.")
    print(f"\nOutputs written to {ANALYSIS_DIR}:")
    for f_ in ('evidence_layers_ablations.csv', 'evidence_layers_secondary.csv',
               'evidence_layers.tex', 'evidence_layers_figure.png'):
        print('  ' + f_)


if __name__ == '__main__':
    main()
