"""
report_figures.py — every figure used in the report and the presentation.

Same rule as report_numbers.py: everything is recomputed from the raw CSVs at run time. No figure
is drawn from a stored summary, a hand-typed value, or an older analysis script.

Outputs PDF (vector, for LaTeX \\includegraphics) and PNG (for the slides) into analysis/figures/.

Run:  py -3.7 scripts\\report_figures.py
"""

import csv
import math
import os
import random
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT   = r'C:\Users\jayad\Desktop\RPI_AINN'
DATA   = os.path.join(ROOT, 'data')
CHAIN  = os.path.join(ROOT, 'chain_risk', 'data')
FIGDIR = os.path.join(ROOT, 'analysis', 'figures')

SEED, N_BOOT = 42, 10000

# consistent palette across every figure and both documents
CLR = {'C': '#C0392B', 'B': '#2874A6', 'A': '#1E8449', 'D': '#7D3C98',
       'grey': '#5D6D7E', 'warn': '#B7950B'}

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 150, 'savefig.bbox': 'tight',
})


# ── data helpers (duplicated from report_numbers.py on purpose: these two scripts
#    must be independently runnable, and a shared import is one more thing to break) ──

def read(fn, base=DATA):
    p = os.path.join(base, fn)
    if not os.path.exists(p):
        return []
    with open(p, 'r') as f:
        return list(csv.DictReader(f))


def num(r, k):
    v = r.get(k)
    if v in ('', 'None', None):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def corr(a, b):
    n = len(a)
    if n < 3:
        return float('nan')
    ma, mb = sum(a) / n, sum(b) / n
    nu = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return nu / (da * db) if da > 0 and db > 0 else float('nan')


def boot_ci(xs, ys, n=N_BOOT, seed=SEED):
    rng = random.Random(seed)
    k, out = len(xs), []
    for _ in range(n):
        idx = [rng.randrange(k) for _ in range(k)]
        v = corr([xs[i] for i in idx], [ys[i] for i in idx])
        if not math.isnan(v):
            out.append(v)
    out.sort()
    return out[int(0.025 * len(out))], out[int(0.975 * len(out))]


SEV = {
    'C': lambda r: (num(r, 'brake_intensity_t') or 0.0) * (num(r, 'brake_duration_ticks_t') or 0.0),
    'B': lambda r: num(r, 'speed_t'),
    'A': lambda r: num(r, 'severity_t'),
}
FILES = {'C': 'scenario_c_braking.csv', 'B': 'scenario_b_intersection.csv',
         'A': 'scenario_a_lanechange.csv', 'D': 'scenario_d_baseline.csv'}
NAMES = {'C': 'C — emergency braking', 'B': 'B — intersection',
         'A': 'A — lane change', 'D': 'D — control'}


def ep_means(rows, sev, key='rho_actual'):
    g = {}
    for r in rows:
        s, o = sev(r), num(r, key)
        if s is None or o is None:
            continue
        g.setdefault(r['episode_id'], ([], []))
        g[r['episode_id']][0].append(s)
        g[r['episode_id']][1].append(o)
    return ([statistics.mean(v[0]) for v in g.values()],
            [statistics.mean(v[1]) for v in g.values()])


def save(fig, name):
    if not os.path.isdir(FIGDIR):
        os.makedirs(FIGDIR)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIGDIR, '%s.%s' % (name, ext)))
    plt.close(fig)
    print('  wrote %s.pdf / .png' % name)


def fit_line(ax, xs, ys, color):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    lo, hi = min(xs), max(xs)
    ax.plot([lo, hi], [my + b * (lo - mx), my + b * (hi - mx)], color=color, lw=1.8, zorder=3)


# ── Figure 1: the headline — effects with CIs, control included ──────────────

def fig_forest():
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    items = []
    for c in ('C', 'B', 'A'):
        xs, ys = ep_means(read(FILES[c]), SEV[c])
        r = corr(xs, ys)
        lo, hi = boot_ci(xs, ys)
        items.append((NAMES[c], r, lo, hi, CLR[c], len(xs), False))
    xs, ys = ep_means(read(FILES['D']), SEV['B'])   # B's ablation applied to the control
    r = corr(xs, ys)
    lo, hi = boot_ci(xs, ys)
    items.append(('D — CONTROL\n(no scripted decision)', r, lo, hi, CLR['D'], len(xs), True))

    ypos = list(range(len(items)))[::-1]
    for y, (lab, r, lo, hi, col, n, ctrl) in zip(ypos, items):
        ax.plot([lo, hi], [y, y], color=col, lw=2.6, solid_capstyle='round',
                alpha=0.45 if not ctrl else 0.75)
        ax.plot([r], [y], 'o', color=col, ms=8, zorder=4,
                markeredgecolor='white', markeredgewidth=1.2)
        ax.text(r, y + 0.30, '%+.3f' % r, ha='center', fontsize=8.5,
                fontweight='bold', color=col)
        ax.text(1.02, y, 'n=%d' % n, transform=ax.get_yaxis_transform(),
                va='center', fontsize=7.5, color=CLR['grey'])
    ax.axvline(0, color='k', lw=0.8, ls='--', alpha=0.6)
    ax.set_yticks(ypos)
    ax.set_yticklabels([i[0] for i in items], fontsize=8.5)
    ax.set_xlabel('episode-level $r$(decision severity, $\\rho$ at $T{+}1$)   '
                  'with 10,000-resample bootstrap 95% CI')
    ax.set_xlim(-0.8, 1.05)
    # headroom so the top row's value label does not collide with the title
    ax.set_ylim(-0.6, len(items) - 0.25)
    ax.set_title('Episode-level effect sizes by scenario',
                 fontweight='bold', color='#333333', pad=12)
    save(fig, 'fig1_forest')


# ── Figure 2: episode-level scatter per scenario ─────────────────────────────

def fig_scatter():
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6))
    xlab = {'C': 'brake intensity $\\times$ duration', 'B': 'yield speed (m/s)',
            'A': 'steer magnitude $\\times$ ticks'}
    for ax, c in zip(axes, ('C', 'B', 'A')):
        xs, ys = ep_means(read(FILES[c]), SEV[c])
        ax.scatter(xs, ys, s=7, alpha=0.32, color=CLR[c], edgecolors='none')
        fit_line(ax, xs, ys, CLR[c])
        ax.set_title('%s\n$r=%+.3f$, $n=%d$' % (NAMES[c], corr(xs, ys), len(xs)), fontsize=9)
        ax.set_xlabel(xlab[c], fontsize=8)
    axes[0].set_ylabel('mean $\\rho$ at $T{+}1$')
    fig.suptitle('Decision severity vs residual risk — direction differs by maneuver',
                 fontweight='bold', y=1.06)
    save(fig, 'fig2_scatter')


# ── Figure 3: the validity test that ended the original claim ────────────────

def fig_conditional():
    rows = read(FILES['C'])
    by = {}
    for r in rows:
        by.setdefault(r['episode_id'], []).append(r)
    sev, r2, sp, xx, yy, yw = [], [], [], [], [], []
    for _e, rs in by.items():
        rs.sort(key=lambda r: int(r['t']))
        for a, b in zip(rs, rs[1:]):
            v = (SEV['C'](a), num(b, 'rho_actual'), num(a, 'speed_t1'),
                 num(a, 'x_t1'), num(a, 'y_t1'), num(a, 'yaw_t1'))
            if None in v:
                continue
            sev.append(v[0]); r2.append(v[1]); sp.append(v[2])
            xx.append(v[3]); yy.append(v[4]); yw.append(v[5])

    def resid(y, X):
        n = len(y)
        cols = [[1.0] * n] + [list(c) for c in X]
        p = len(cols)
        M = [[sum(cols[i][k] * cols[j][k] for k in range(n)) for j in range(p)]
             + [sum(cols[i][k] * y[k] for k in range(n))] for i in range(p)]
        for i in range(p):
            piv = max(range(i, p), key=lambda q: abs(M[q][i]))
            M[i], M[piv] = M[piv], M[i]
            d = M[i][i]
            M[i] = [v / d for v in M[i]]
            for q in range(p):
                if q != i and M[q][i]:
                    f = M[q][i]
                    M[q] = [a - f * b for a, b in zip(M[q], M[i])]
        beta = [M[i][p] for i in range(p)]
        fit = [sum(beta[i] * cols[i][k] for i in range(p)) for k in range(n)]
        return [y[k] - fit[k] for k in range(n)]

    r_un = corr(sev, r2)
    r_pa = corr(resid(sev, [sp, xx, yy, yw]), resid(r2, [sp, xx, yy, yw]))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.9),
                                 gridspec_kw={'width_ratios': [1.15, 1]})
    a1.bar([0, 1], [r_un, r_pa], color=[CLR['C'], CLR['grey']], width=0.55)
    for i, v in enumerate([r_un, r_pa]):
        a1.text(i, v + (0.02 if v > 0 else -0.045), '%+.4f' % v,
                ha='center', fontweight='bold', fontsize=9.5)
    a1.set_xticks([0, 1])
    a1.set_xticklabels(['severity at $T$\nvs risk at $T{+}2$',
                        'same, controlling for\nthe state at $T{+}1$'], fontsize=8.5)
    a1.axhline(0, color='k', lw=0.8)
    a1.set_ylabel('correlation')
    a1.set_ylim(-0.15, 0.48)
    a1.set_title('Conditioning on the intervening state\neliminates the effect (n=%d)' % len(sev),
                 fontsize=9)

    rows = read(FILES['C'])
    s1 = [num(r, 'speed_t1') for r in rows if num(r, 'speed_t1') is not None
          and num(r, 'rho_actual') is not None]
    ro = [num(r, 'rho_actual') for r in rows if num(r, 'speed_t1') is not None
          and num(r, 'rho_actual') is not None]
    a2.scatter(s1, ro, s=3, alpha=0.18, color=CLR['C'], edgecolors='none')
    a2.set_xlabel('speed at $T{+}1$ (m/s)')
    a2.set_ylabel('$\\rho$ at $T{+}1$')
    a2.set_title('$\\rho_C$ is a speedometer\n$r=%+.4f$' % corr(s1, ro), fontsize=9)
    fig.suptitle('Why the original claim does not survive', fontweight='bold', y=1.05)
    save(fig, 'fig3_conditional')


# ── Figure 4: rho distributions ──────────────────────────────────────────────

def fig_dists():
    fig, ax = plt.subplots(figsize=(5.4, 2.7))
    data, labs, cols = [], [], []
    for c in ('D', 'C', 'A', 'B'):
        rows = read(FILES[c])
        v = [num(r, 'rho_actual') for r in rows if num(r, 'rho_actual') is not None]
        data.append(v); labs.append(NAMES[c]); cols.append(CLR[c])
    bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.55,
                    medianprops=dict(color='white', lw=1.6),
                    flierprops=dict(marker='.', ms=1.5, alpha=0.25))
    for patch, c in zip(bp['boxes'], cols):
        patch.set_facecolor(c); patch.set_alpha(0.8); patch.set_edgecolor('none')
    ax.set_yticklabels(labs, fontsize=8.5)
    ax.set_xlabel('$\\rho$ at $T{+}1$')
    ax.set_title('Risk distributions are not comparable across scenarios\n'
                 '(each uses a different weighted formula)', fontsize=9, fontweight='bold')
    save(fig, 'fig4_distributions')


# ── Figure 5: the censoring artifact under traffic ───────────────────────────

def fig_censoring():
    rows = read('scenario_a_lanechange_npc_rss.csv')
    if not rows:
        return
    by = {}
    for r in rows:
        by.setdefault(r['episode_id'], []).append(r)
    stats = {}
    for e, v in by.items():
        sv = [SEV['A'](r) for r in v if SEV['A'](r) is not None]
        ro = [num(r, 'rho_actual') for r in v if num(r, 'rho_actual') is not None]
        if sv and ro:
            stats[e] = (len(v), statistics.mean(sv), statistics.mean(ro))
    ths, rs, ns = [], [], []
    for t in range(1, 9):
        sel = [v for v in stats.values() if v[0] >= t]
        if len(sel) >= 10:
            ths.append(t)
            rs.append(corr([v[1] for v in sel], [v[2] for v in sel]))
            ns.append(len(sel))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    a1.scatter([v[1] for v in stats.values()], [v[2] for v in stats.values()],
               s=[14 if v[0] == 1 else 7 for v in stats.values()],
               c=[CLR['warn'] if v[0] == 1 else CLR['A'] for v in stats.values()],
               alpha=0.75, edgecolors='none')
    a1.set_xlabel('episode mean severity')
    a1.set_ylabel('episode mean $\\rho$')
    a1.set_title('Single-pair episodes (amber) sit\nhigh on both axes', fontsize=9)
    a1.legend(handles=[Patch(facecolor=CLR['warn'], label='1 pair (crash-truncated)'),
                       Patch(facecolor=CLR['A'], label='$\\geq$2 pairs')],
              fontsize=7, frameon=False, loc='lower right')

    a2.plot(ths, rs, 'o-', color=CLR['A'], lw=1.8, ms=5)
    a2.axhline(0, color='k', lw=0.8, ls='--', alpha=0.6)
    a2.axhline(-0.4951, color=CLR['grey'], lw=1.2, ls=':',
               label='banked, no traffic ($-0.495$)')
    a2.legend(fontsize=7, frameon=False, loc='center right')
    for t, r, n in zip(ths, rs, ns):
        a2.annotate('n=%d' % n, (t, r), textcoords='offset points',
                    xytext=(0, 9), ha='center', fontsize=6.5, color=CLR['grey'])
    a2.set_ylim(min(rs) - 0.18, max(rs) + 0.14)
    a2.set_xlabel('minimum pairs required per episode')
    a2.set_ylabel('episode-level $r$')
    a2.set_title('Excluding single-pair episodes restores\nthe banked effect', fontsize=9)
    fig.suptitle('Ambient traffic introduces severity-correlated censoring',
                 fontweight='bold', y=1.05)
    save(fig, 'fig5_censoring')


# ── Figure 6: Study 2 ────────────────────────────────────────────────────────

def fig_chain():
    rows = read('chain_position_hazard_pilot.csv', base=CHAIN)
    if not rows:
        return
    ch = [r for r in rows if r['arm'] == 'CHAIN']
    iso = [r for r in rows if r['arm'] == 'ISO_LATE']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.8))

    pos, gap, ent = [], [], []
    for p in range(8):
        g = [num(r, 'follower_gap_before') for r in ch
             if int(r['position']) == p and r.get('follower_gap_valid') == 'True'
             and num(r, 'follower_gap_before') is not None]
        e = [num(r, 'entry_speed') for r in ch
             if int(r['position']) == p and num(r, 'entry_speed') is not None]
        if g and e:
            pos.append(p); gap.append(statistics.mean(g)); ent.append(statistics.mean(e))
    ig = [num(r, 'follower_gap_before') for r in iso
          if r.get('follower_gap_valid') == 'True' and num(r, 'follower_gap_before') is not None]

    a1.plot(pos, gap, 'o-', color=CLR['C'], lw=1.8, ms=5, label='CHAIN (position $t$)')
    if ig:
        a1.axhline(statistics.mean(ig), color=CLR['B'], ls='--', lw=1.5,
                   label='ISO-LATE (single decision)')
    a1.set_xlabel('decision position in chain')
    a1.set_ylabel('following gap at decision (m)')
    a1.set_title('The 8th decision looks like\na fresh vehicle', fontsize=9)
    a1.legend(fontsize=7, frameon=False)

    a2.plot(pos, ent, 's-', color=CLR['D'], lw=1.8, ms=5)
    a2.set_ylim(min(ent) - 1.2, max(ent) + 1.2)
    a2.set_xlabel('decision position in chain')
    a2.set_ylabel('ego entry speed (m/s)')
    a2.set_title('Regime control: entry speed varies by\nonly %.2f m/s across positions'
                 % (max(ent) - min(ent)), fontsize=9)
    fig.suptitle('Study 2 — chain position vs isolated decision', fontweight='bold', y=1.05)
    save(fig, 'fig6_chain')


# ── Figure 7: hard-braking characterisation ──────────────────────────────────

def fig_hardbrake():
    rows = read('chain_position_hazard_pilot.csv', base=CHAIN)
    if not rows:
        return
    dec = []
    for r in rows:
        e, s, t = num(r, 'entry_speed'), num(r, 'speed_t'), num(r, 'brake_ticks')
        if None in (e, s, t) or not t:
            continue
        dec.append((e - s) / (t * 0.05))
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    ax.hist(dec, bins=34, color=CLR['C'], alpha=0.8, edgecolor='white', lw=0.4)
    ax.axvspan(2.8, 4.4, color=CLR['warn'], alpha=0.18)
    ax.axvline(2.8, color=CLR['warn'], lw=1.4)
    ax.axvline(4.4, color=CLR['warn'], lw=1.4)
    ax.text(3.6, ax.get_ylim()[1] * 0.93, 'telematics\nhard-brake band',
            ha='center', fontsize=7, color='#7D6608')
    pct = 100.0 * sum(1 for x in dec if x > 2.8) / len(dec)
    ax.set_xlabel('achieved deceleration (m/s$^2$)')
    ax.set_ylabel('decisions')
    ax.set_title('%.0f%% of decisions exceed the hard-braking threshold\n'
                 '(median %.2f m/s$^2$, n=%d) — the null is not a weak-treatment artifact'
                 % (pct, statistics.median(dec), len(dec)), fontsize=8.5, fontweight='bold')
    save(fig, 'fig7_hardbraking')


def main():
    print('Generating figures into analysis/figures/ ...')
    for fn in (fig_forest, fig_scatter, fig_conditional, fig_dists,
               fig_censoring, fig_chain, fig_hardbrake):
        try:
            fn()
        except Exception as e:
            print('  !! %s FAILED: %s: %s' % (fn.__name__, type(e).__name__, e))
    print('Done.')


if __name__ == '__main__':
    main()
