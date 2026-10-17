"""
report_numbers.py — SINGLE SOURCE OF TRUTH for every number cited in the report and presentation.

Rule adopted 2026-09-21: if a figure is not in this script's output, it does not appear in the
report or the slides. No number is typed from memory, from CLAUDE.md, or from an older analysis
script. This exists because two separate results in this project were published internally and
later withdrawn, and because a documented 678-pair dataset was found overwritten by a 97-row run.

Everything is recomputed from the raw CSVs in data/ and chain_risk/data/ on every invocation.

Run:  py -3.7 scripts\\report_numbers.py
Out:  analysis\\report_numbers.txt   (human-readable, paste-checkable)
      analysis\\report_numbers.csv   (key,value — for scripted insertion into LaTeX)
"""

import csv
import math
import os
import random
import statistics
import sys
from collections import OrderedDict

ROOT     = r'C:\Users\jayad\Desktop\RPI_AINN'
DATA     = os.path.join(ROOT, 'data')
CHAIN    = os.path.join(ROOT, 'chain_risk', 'data')
ANALYSIS = os.path.join(ROOT, 'analysis')

N_PERM = 10000
N_BOOT = 10000
SEED   = 42

OUT = []          # text lines
KV  = OrderedDict()   # key -> value for LaTeX macro generation


# ── helpers ──────────────────────────────────────────────────────────────────

def say(s=''):
    print(s)
    OUT.append(s)


def kv(key, val, note=''):
    KV[key] = val
    if note:
        say('    %-34s %s   (%s)' % (key, val, note))
    else:
        say('    %-34s %s' % (key, val))


def read(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return list(csv.DictReader(f))


def num(row, key):
    v = row.get(key)
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
    numr = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return numr / (da * db) if da > 0 and db > 0 else float('nan')


def episode_means(rows, sev_fn, out_key='rho_actual'):
    """Collapse pairs to one (severity, outcome) point per episode.

    Episode is the unit of analysis throughout. Pairs within an episode share a vehicle, a spawn
    point and a road segment, so pooling them as independent samples is pseudo-replication and
    understates p-values. Every significance claim in the report uses this function.
    """
    g = {}
    for r in rows:
        s, o = sev_fn(r), num(r, out_key)
        if s is None or o is None:
            continue
        g.setdefault(r['episode_id'], ([], []))
        g[r['episode_id']][0].append(s)
        g[r['episode_id']][1].append(o)
    xs = [statistics.mean(v[0]) for v in g.values()]
    ys = [statistics.mean(v[1]) for v in g.values()]
    return xs, ys


def perm_test(xs, ys, n=N_PERM, seed=SEED):
    r0 = corr(xs, ys)
    rng = random.Random(seed)
    hits = 0
    for _ in range(n):
        sh = ys[:]
        rng.shuffle(sh)
        if abs(corr(xs, sh)) >= abs(r0):
            hits += 1
    return r0, (hits + 1.0) / (n + 1.0)


def bootstrap_ci(xs, ys, n=N_BOOT, seed=SEED):
    rng = random.Random(seed)
    k = len(xs)
    out = []
    for _ in range(n):
        idx = [rng.randrange(k) for _ in range(k)]
        out.append(corr([xs[i] for i in idx], [ys[i] for i in idx]))
    out = [v for v in out if not math.isnan(v)]
    out.sort()
    return out[int(0.025 * len(out))], out[int(0.975 * len(out))]


def ols_resid(y, X):
    """Residualise y on columns X (list of lists) with an intercept, via normal equations."""
    n = len(y)
    cols = [[1.0] * n] + [list(c) for c in X]
    p = len(cols)
    XtX = [[sum(cols[i][k] * cols[j][k] for k in range(n)) for j in range(p)] for i in range(p)]
    Xty = [sum(cols[i][k] * y[k] for k in range(n)) for i in range(p)]
    # gaussian elimination
    M = [XtX[i][:] + [Xty[i]] for i in range(p)]
    for i in range(p):
        piv = max(range(i, p), key=lambda r: abs(M[r][i]))
        if abs(M[piv][i]) < 1e-12:
            return None
        M[i], M[piv] = M[piv], M[i]
        d = M[i][i]
        M[i] = [v / d for v in M[i]]
        for r in range(p):
            if r != i and M[r][i] != 0:
                f = M[r][i]
                M[r] = [a - f * b for a, b in zip(M[r], M[i])]
    beta = [M[i][p] for i in range(p)]
    fit = [sum(beta[i] * cols[i][k] for i in range(p)) for k in range(n)]
    return [y[k] - fit[k] for k in range(n)]


def partial_corr(y, x, controls):
    ry = ols_resid(y, controls)
    rx = ols_resid(x, controls)
    if ry is None or rx is None:
        return float('nan')
    return corr(rx, ry)


def hdr(t):
    say()
    say('=' * 78)
    say(t)
    say('=' * 78)


# ── severity definitions (locked a priori, see PREREGISTRATION / CLAUDE.md) ───

SEV = {
    'C': lambda r: (num(r, 'brake_intensity_t') or 0.0) * (num(r, 'brake_duration_ticks_t') or 0.0),
    'B': lambda r: num(r, 'speed_t'),
    'A': lambda r: num(r, 'severity_t'),
}


# ── Section 1: primary ablations ─────────────────────────────────────────────

def primary():
    hdr('1. PRIMARY ABLATION  —  episode-level r(decision severity, rho at T+1)')
    say('   Unit of analysis: episode. Permutation null: %d shuffles. Bootstrap CI: %d resamples, '
        'seed %d.' % (N_PERM, N_BOOT, SEED))
    jobs = [
        ('C', 'Scenario C  emergency braking', 'scenario_c_braking.csv',      'sevC'),
        ('B', 'Scenario B  intersection',      'scenario_b_intersection.csv', 'sevB'),
        ('A', 'Scenario A  lane change',       'scenario_a_lanechange.csv',   'sevA'),
    ]
    for code, label, fn, tag in jobs:
        rows = read(os.path.join(DATA, fn))
        say()
        say('  %s   [%s]' % (label, fn))
        if not rows:
            say('    FILE MISSING')
            continue
        xs, ys = episode_means(rows, SEV[code])
        r0, p = perm_test(xs, ys)
        lo, hi = bootstrap_ci(xs, ys)
        kv('%s_r' % tag, '%+.4f' % r0)
        kv('%s_p' % tag, ('%.4f' % p) if p > 1.0 / N_PERM else '<1e-4')
        kv('%s_ci' % tag, '[%+.3f, %+.3f]' % (lo, hi))
        kv('%s_nep' % tag, '%d' % len(xs))
        kv('%s_npairs' % tag, '%d' % len(rows))
        rho = [num(r, 'rho_actual') for r in rows if num(r, 'rho_actual') is not None]
        kv('%s_rho_mean' % tag, '%.4f' % statistics.mean(rho))
        kv('%s_rho_sd' % tag, '%.4f' % statistics.pstdev(rho))

    # Scenario D — the control, tested properly
    say()
    say('  Scenario D  open-road control   [scenario_d_baseline.csv]')
    d = read(os.path.join(DATA, 'scenario_d_baseline.csv'))
    if d:
        rho = [num(r, 'rho_actual') for r in d if num(r, 'rho_actual') is not None]
        kv('sevD_npairs', '%d' % len(d))
        kv('sevD_nep', '%d' % len(set(r['episode_id'] for r in d)))
        kv('sevD_rho_mean', '%.4f' % statistics.mean(rho))
        kv('sevD_rho_sd', '%.4f' % statistics.pstdev(rho))
        # D has no scripted decision. Apply B's ablation (speed_t) to it anyway — this is the
        # test that was never run during the original study and that invalidated the control.
        xs, ys = episode_means(d, SEV['B'])
        r0, p = perm_test(xs, ys)
        lo, hi = bootstrap_ci(xs, ys)
        kv('sevD_r_applied', '%+.4f' % r0, "B's ablation applied to the CONTROL")
        kv('sevD_p_applied', ('%.4f' % p) if p > 1.0 / N_PERM else '<1e-4')
        kv('sevD_ci_applied', '[%+.3f, %+.3f]' % (lo, hi))
        kv('sevD_nep_applied', '%d' % len(xs))
        sp0 = [num(r, 'speed_t') for r in d if num(r, 'speed_t') is not None
               and num(r, 'speed_t1') is not None]
        sp1 = [num(r, 'speed_t1') for r in d if num(r, 'speed_t') is not None
               and num(r, 'speed_t1') is not None]
        kv('sevD_speed_autocorr', '%+.4f' % corr(sp0, sp1), 'the mechanism behind it')


# ── Section 2: the Markov / validity tests ───────────────────────────────────

def markov():
    hdr('2. VALIDITY  —  does the effect survive conditioning on the intervening state?')
    say('   If risk at T+1 is a function of the state at T+1, and the decision sets that state,')
    say('   then r(severity, rho) is guaranteed. The test is whether severity at T still predicts')
    say('   risk at T+2 AFTER controlling for the full state at T+1.')

    rows = read(os.path.join(DATA, 'scenario_c_braking.csv'))
    if not rows:
        say('   scenario_c_braking.csv MISSING')
        return

    by_ep = {}
    for r in rows:
        by_ep.setdefault(r['episode_id'], []).append(r)

    sev, rho2, s_sp, s_x, s_y, s_yaw = [], [], [], [], [], []
    for _e, rs in by_ep.items():
        rs.sort(key=lambda r: int(r['t']))
        for a, b in zip(rs, rs[1:]):
            sv = SEV['C'](a)
            r2 = num(b, 'rho_actual')
            sp = num(a, 'speed_t1')
            xx, yy, yw = num(a, 'x_t1'), num(a, 'y_t1'), num(a, 'yaw_t1')
            if None in (sv, r2, sp, xx, yy, yw):
                continue
            sev.append(sv); rho2.append(r2)
            s_sp.append(sp); s_x.append(xx); s_y.append(yy); s_yaw.append(yw)

    kv('cond_n', '%d' % len(sev), 'sequential T->T+2 cycle pairs')
    kv('cond_r_uncond', '%+.4f' % corr(sev, rho2), 'severity at T vs risk at T+2')
    kv('cond_r_partial', '%+.4f' % partial_corr(rho2, sev, [s_sp, s_x, s_y, s_yaw]),
       'same, controlling for state at T+1')

    # ── CAUTION, and the reason this block is labelled the way it is ─────────
    # These control for the state logged at T. In scenarios C and A that state is sampled AFTER the
    # decision executes, so it is a POST-TREATMENT mediator, not a pre-treatment covariate.
    # Conditioning on a mediator biases the estimate toward zero; it does not make it conservative.
    # This project previously reported such a figure as "the honest conservative number" and had to
    # retract it. They are retained only as a demonstration of that error and must NEVER be cited in
    # the report as the effect size. The valid mediation test is cond_r_partial above, which
    # conditions on the INTERVENING state between two separate decisions.
    for code, fn, tag in (('C', 'scenario_c_braking.csv', 'incC'),
                          ('B', 'scenario_b_intersection.csv', 'incB'),
                          ('A', 'scenario_a_lanechange.csv', 'incA')):
        rs = read(os.path.join(DATA, fn))
        if not rs:
            continue
        y, x, c1, c2 = [], [], [], []
        for r in rs:
            sv, ro = SEV[code](r), num(r, 'rho_actual')
            sp, yw = num(r, 'speed_t'), num(r, 'yaw_t')
            if None in (sv, ro, sp, yw):
                continue
            y.append(ro); x.append(sv); c1.append(sp); c2.append(yw)
        if len(y) > 20:
            pc = partial_corr(y, x, [c1, c2])
            if math.isnan(pc):
                # Not a numerical failure — a finding. Scenario B's "severity" IS speed_t, so it is
                # perfectly collinear with the state control and the partial correlation is
                # undefined. B never had a treatment variable independent of the state it was
                # supposed to be predicting; it was structurally identical to the D control.
                kv('%s_partial_BIASED' % tag, 'undefined',
                   'severity is collinear with the state control — no independent treatment')
            else:
                kv('%s_partial_BIASED' % tag, '%+.4f' % pc,
                   'POST-TREATMENT conditioning — biased, do NOT cite as an effect size')

    # C: is rho effectively a speedometer?
    rs = [r for r in rows if num(r, 'speed_t1') is not None and num(r, 'rho_actual') is not None]
    kv('C_rho_vs_speed', '%+.4f' % corr([num(r, 'speed_t1') for r in rs],
                                        [num(r, 'rho_actual') for r in rs]),
       'r(speed at T+1, rho) — rho_C is dominated by inverse_speed')


# ── Section 3: robustness under NPC traffic + formula-free outcome ───────────

def npc():
    hdr('3. ROBUSTNESS  —  replication under ambient traffic, and a formula-free outcome')

    a = read(os.path.join(DATA, 'scenario_a_lanechange_npc_rss.csv'))
    say()
    say('  Scenario A under traffic   [scenario_a_lanechange_npc_rss.csv]')
    if a:
        kv('npcA_npairs', '%d' % len(a))
        by_ep = {}
        for r in a:
            by_ep.setdefault(r['episode_id'], []).append(r)
        kv('npcA_nep', '%d' % len(by_ep))
        kv('npcA_ncrash_ep', '%d' % sum(1 for v in by_ep.values()
                                        if any(r.get('outcome') == 'crash' for r in v)))

        # Pair level first: this is the estimator that does NOT depend on episode length, so it is
        # the one that speaks directly to whether the banked effect replicates.
        pr = [(SEV['A'](r), num(r, 'rho_actual')) for r in a]
        pr = [(s, o) for s, o in pr if s is not None and o is not None]
        kv('npcA_r_pair', '%+.4f' % corr([s for s, _ in pr], [o for _, o in pr]),
           'pair level — compare with the banked pair-level value')
        bp = read(os.path.join(DATA, 'scenario_a_lanechange.csv')) or []
        bpr = [(SEV['A'](r), num(r, 'rho_actual')) for r in bp]
        bpr = [(s, o) for s, o in bpr if s is not None and o is not None]
        if bpr:
            kv('bankA_r_pair', '%+.4f' % corr([s for s, _ in bpr], [o for _, o in bpr]),
               'banked, pair level')

        # ── Outcome-dependent censoring, and why the naive episode estimator flips ──
        # Ambient traffic raises the crash rate, crashes terminate episodes, and crashes follow
        # HIGH-severity lane changes. A crash on the first decision leaves an episode contributing
        # one point at (high severity, high rho). Those points are not episode means in any
        # meaningful sense — there is nothing to average — and they dominate the correlation.
        lens = {}
        for e, v in by_ep.items():
            sv = [SEV['A'](r) for r in v if SEV['A'](r) is not None]
            ro = [num(r, 'rho_actual') for r in v if num(r, 'rho_actual') is not None]
            if sv and ro:
                lens[e] = (len(v), statistics.mean(sv), statistics.mean(ro))
        for thr in (1, 2, 3):
            sel = [v for v in lens.values() if v[0] >= thr]
            if len(sel) < 10:
                continue
            xs = [v[1] for v in sel]
            ys = [v[2] for v in sel]
            r0, p = perm_test(xs, ys)
            lo, hi = bootstrap_ci(xs, ys)
            kv('npcA_r_min%d' % thr, '%+.4f' % r0, 'episodes with >=%d pairs, n=%d' % (thr, len(sel)))
            kv('npcA_p_min%d' % thr, ('%.4f' % p) if p > 1.0 / N_PERM else '<1e-4')
            kv('npcA_ci_min%d' % thr, '[%+.3f, %+.3f]' % (lo, hi))
        one = [v for v in lens.values() if v[0] == 1]
        many = [v for v in lens.values() if v[0] >= 7]
        if one and many:
            kv('npcA_1pair_n', '%d' % len(one))
            kv('npcA_1pair_sev', '%.2f' % statistics.mean([v[1] for v in one]))
            kv('npcA_1pair_rho', '%.3f' % statistics.mean([v[2] for v in one]))
            kv('npcA_full_sev', '%.2f' % statistics.mean([v[1] for v in many]),
               'episodes with >=7 pairs')
            kv('npcA_full_rho', '%.3f' % statistics.mean([v[2] for v in many]))

    c = read(os.path.join(DATA, 'scenario_c_npc_rss.csv'))
    say()
    say('  Scenario C under traffic — RAW REAR GAP (no risk formula involved)   '
        '[scenario_c_npc_rss.csv]')
    if c:
        kv('npcC_npairs', '%d' % len(c))
        kv('npcC_nep', '%d' % len(set(r['episode_id'] for r in c)))
        xs, ys = episode_means(c, SEV['C'])
        if len(xs) >= 5:
            r0, p = perm_test(xs, ys)
            kv('npcC_r_rho', '%+.4f' % r0, 'severity vs rho (formula-based)')
            kv('npcC_p_rho', ('%.4f' % p) if p > 1.0 / N_PERM else '<1e-4')
        # formula-free: does the brake decision erode the FOLLOWER's gap at T+1?
        have = [r for r in c if r.get('rss_t1_rear_present') in ('True', 'true', '1')]
        xs, ys = episode_means(have, SEV['C'], out_key='rss_t1_rear_gap')
        kv('npcC_rear_n', '%d' % len(have), 'pairs with a rear vehicle present at T+1')
        if len(xs) >= 5:
            r0, p = perm_test(xs, ys)
            lo, hi = bootstrap_ci(xs, ys)
            kv('npcC_rear_r', '%+.4f' % r0, 'severity vs RAW rear gap at T+1')
            kv('npcC_rear_p', ('%.4f' % p) if p > 1.0 / N_PERM else '<1e-4')
            kv('npcC_rear_ci', '[%+.3f, %+.3f]' % (lo, hi))
            kv('npcC_rear_nep', '%d' % len(xs))


# ── Section 4: chain_risk (Study 2) ──────────────────────────────────────────

#: Primary Study 2 condition. cruise=40 is the configuration whose null was first reported; the
#: report cites THIS setting, never a pooled figure across settings.
CHAIN_PRIMARY_SETTING = '40'
#: Newest run first. The script falls back down the list so the report still builds on older data.
CHAIN_FILES = ['chain_position_hazard_sweep_v2.csv',
               'chain_position_hazard_sweep_v2_pilot.csv',
               'chain_position_hazard_sweep_v1_2026-09-21.csv',
               'chain_position_hazard_pilot.csv']


def chain_study():
    hdr('4. STUDY 2  —  chain position vs isolated decision (two-agent design)')

    rows, src = None, None
    for fn in CHAIN_FILES:
        rows = read(os.path.join(CHAIN, fn))
        if rows:
            src = fn
            break
    if not rows:
        say('   no Study 2 data file found; tried: %s' % ', '.join(CHAIN_FILES))
        return
    say('   source: %s' % src)

    # ── Pool ONLY within a sweep setting ─────────────────────────────────────
    # The inter-decision interval is an experimental condition, not noise. Averaging across
    # settings mixes four different experiments: it previously produced an entry-speed "spread" of
    # 0.06 m/s that was an averaging coincidence rather than the regime control it was reported as,
    # and per-setting the true value is 0.11-0.24 m/s. Everything below is computed within one
    # setting, and the setting is named in the output.
    settings = sorted(set(r.get('setting_cruise_ticks', '') for r in rows))
    if settings and settings != ['']:
        say('   sweep settings present: %s' % ', '.join(s for s in settings if s))
        sel = CHAIN_PRIMARY_SETTING if CHAIN_PRIMARY_SETTING in settings else settings[0]
        rows = [r for r in rows if r.get('setting_cruise_ticks') == sel]
        kv('ch_setting', sel, 'cruise ticks; ALL Study 2 values below are this setting only')
    else:
        kv('ch_setting', 'single-condition run')

    ch = [r for r in rows if r['arm'] == 'CHAIN']
    iso = [r for r in rows if r['arm'] == 'ISO_LATE']
    kv('ch_ndec', '%d' % len(rows))
    kv('ch_nchain', '%d' % len(ch))
    kv('ch_niso', '%d' % len(iso))

    ent = {}
    for r in ch:
        e = num(r, 'entry_speed')
        if e is not None:
            ent.setdefault(int(r['position']), []).append(e)
    if ent:
        means = [statistics.mean(v) for _k, v in sorted(ent.items())]
        kv('ch_entry_lo', '%.2f' % min(means))
        kv('ch_entry_hi', '%.2f' % max(means))
        kv('ch_entry_spread', '%.2f' % (max(means) - min(means)), 'm/s across 8 positions')

    def vgap(rs, pos=None):
        out = []
        for r in rs:
            if r.get('follower_gap_valid') != 'True':
                continue
            if pos is not None and int(r['position']) != pos:
                continue
            g = num(r, 'follower_gap_before')
            if g is not None:
                out.append(g)
        return out

    c7, ig = vgap(ch, 7), vgap(iso)
    if c7 and ig:
        kv('ch_t7_gap', '%.2f' % statistics.mean(c7))
        kv('ch_t7_n', '%d' % len(c7))
        kv('ch_iso_gap', '%.2f' % statistics.mean(ig))
        kv('ch_iso_n', '%d' % len(ig))
        diff = statistics.mean(ig) - statistics.mean(c7)
        kv('ch_gap_diff', '%+.2f' % diff, 'metres (ISO-LATE minus CHAIN t=7)')
        # Bootstrap CI on the difference. A difference without an interval is not reportable:
        # the earlier single-run value had no interval and could not be defended when challenged.
        rng = random.Random(SEED)
        boots = []
        for _ in range(N_BOOT):
            a = [c7[rng.randrange(len(c7))] for _ in range(len(c7))]
            b = [ig[rng.randrange(len(ig))] for _ in range(len(ig))]
            boots.append(statistics.mean(b) - statistics.mean(a))
        boots.sort()
        lo = boots[int(0.025 * N_BOOT)]
        hi = boots[int(0.975 * N_BOOT)]
        kv('ch_gap_ci', '[%+.2f, %+.2f]' % (lo, hi))
        kv('ch_gap_sig', 'excludes zero' if (lo > 0 or hi < 0) else 'includes zero')

    # Disturbance persistence into the following decision. The report previously stated "89% of a
    # braking disturbance is erased" as hand-typed prose carried over from an earlier run — the one
    # kind of value this pipeline exists to prevent. It is computed here, within the primary
    # setting, so it moves with the data like everything else.
    by_ep = {}
    for r in ch:
        by_ep.setdefault(r['episode_id'], []).append(r)
    applied, residual = [], []
    for _e, rs in by_ep.items():
        rs.sort(key=lambda r: int(r['position']))
        for a, b in zip(rs, rs[1:]):
            if a.get('follower_gap_valid') != 'True':
                continue
            gb, ga = num(a, 'follower_gap_before'), num(a, 'follower_gap_after')
            gn = num(b, 'follower_gap_before')
            if None in (gb, ga, gn):
                continue
            applied.append(gb - ga)
            residual.append(gb - gn)
    if applied and statistics.mean(applied) > 1.0:
        frac = 1.0 - statistics.mean(residual) / statistics.mean(applied)
        kv('ch_recovery_pct', '%.0f' % (100.0 * frac),
           '%% of the disturbance erased before the next decision (n=%d pairs)' % len(applied))
        kv('ch_recovery_n', '%d' % len(applied))
    elif applied:
        say('    !! persistence not computed: mean applied disturbance is %.2f m, too near zero '
            'for a ratio to be meaningful' % statistics.mean(applied))

    dec = []
    for r in rows:
        e, s, t = num(r, 'entry_speed'), num(r, 'speed_t'), num(r, 'brake_ticks')
        if None in (e, s, t) or not t:
            continue
        dec.append((e - s) / (t * 0.05))
    if dec:
        dec.sort()
        kv('hb_n', '%d' % len(dec))
        kv('hb_median', '%.2f' % statistics.median(dec), 'm/s^2')
        kv('hb_max', '%.2f' % dec[-1], 'm/s^2')
        kv('hb_frac_28', '%.1f' % (100.0 * sum(1 for x in dec if x > 2.8) / len(dec)),
           '%% above 2.8 m/s^2 (telematics hard-brake threshold)')
        kv('hb_frac_44', '%.1f' % (100.0 * sum(1 for x in dec if x > 4.4) / len(dec)),
           '%% above 4.4 m/s^2')

    # ── Secondary: the spacing comparison, reported per setting, never pooled ──
    all_rows = read(os.path.join(CHAIN, src))
    sets = sorted(set(r.get('setting_cruise_ticks', '') for r in all_rows) - set(['']),
                  key=lambda x: int(x))
    if len(sets) > 1:
        say()
        say('   Per-setting comparison (secondary; each row is an independent condition):')
        say('     cruise   CHAIN t=7 (n)      ISO-LATE (n)      difference')
        for st in sets:
            sub = [r for r in all_rows if r.get('setting_cruise_ticks') == st]
            a = vgap([r for r in sub if r['arm'] == 'CHAIN'], 7)
            b = vgap([r for r in sub if r['arm'] == 'ISO_LATE'])
            if a and b:
                say('     %6s   %6.2f m (%2d)     %6.2f m (%2d)     %+.2f m'
                    % (st, statistics.mean(a), len(a), statistics.mean(b), len(b),
                       statistics.mean(b) - statistics.mean(a)))


# ── LaTeX macro export ───────────────────────────────────────────────────────
# Explicit key -> macro map rather than an automatic transform. LaTeX control sequences cannot
# contain digits or underscores, and an automatic mangler would silently produce collisions
# (npcA_r_min1 and npcA_r_min2 both reduce to the same name under naive stripping). Explicit is
# longer but it cannot fail quietly, which is the whole point of this pipeline.
MACROS = [
    ('sevC_r', 'sevCr'), ('sevC_p', 'sevCp'), ('sevC_ci', 'sevCci'),
    ('sevC_nep', 'sevCnep'), ('sevC_npairs', 'sevCnpairs'),
    ('sevB_r', 'sevBr'), ('sevB_p', 'sevBp'), ('sevB_ci', 'sevBci'),
    ('sevB_nep', 'sevBnep'), ('sevB_npairs', 'sevBnpairs'),
    ('sevA_r', 'sevAr'), ('sevA_p', 'sevAp'), ('sevA_ci', 'sevAci'),
    ('sevA_nep', 'sevAnep'), ('sevA_npairs', 'sevAnpairs'),
    ('sevD_npairs', 'sevDnpairs'), ('sevD_nep', 'sevDnep'),
    ('sevD_r_applied', 'sevDrapplied'), ('sevD_p_applied', 'sevDpapplied'),
    ('sevD_ci_applied', 'sevDciapplied'), ('sevD_nep_applied', 'sevDnepapplied'),
    ('sevD_speed_autocorr', 'sevDspeedautocorr'),
    ('cond_n', 'condn'), ('cond_r_uncond', 'condruncond'), ('cond_r_partial', 'condrpartial'),
    ('C_rho_vs_speed', 'Crhovsspeed'),
    ('npcC_rear_r', 'npcCrearr'), ('npcC_rear_p', 'npcCrearp'),
    ('npcC_rear_ci', 'npcCrearci'), ('npcC_rear_nep', 'npcCrearnep'),
    ('npcC_rear_n', 'npcCrearn'),
    ('npcA_npairs', 'npcAnpairs'), ('npcA_nep', 'npcAnep'),
    ('npcA_r_pair', 'npcArpair'), ('bankA_r_pair', 'bankArpair'),
    ('npcA_r_min1', 'npcArminone'), ('npcA_r_min3', 'npcArminthree'),
    ('npcA_ci_min3', 'npcAciminthree'),
    ('npcA_1pair_n', 'npcAonepairn'), ('npcA_1pair_sev', 'npcAonepairsev'),
    ('npcA_1pair_rho', 'npcAonepairrho'),
    ('npcA_full_sev', 'npcAfullsev'), ('npcA_full_rho', 'npcAfullrho'),
    ('ch_entry_spread', 'chentryspread'), ('ch_entry_lo', 'chentrylo'),
    ('ch_entry_hi', 'chentryhi'), ('ch_t7_gap', 'chtsevengap'),
    ('ch_iso_gap', 'chisogap'), ('ch_ndec', 'chndec'),
    ('ch_setting', 'chsetting'), ('ch_t7_n', 'chtsevenn'), ('ch_iso_n', 'chison'),
    ('ch_gap_diff', 'chgapdiff'), ('ch_gap_ci', 'chgapci'),
    ('ch_recovery_pct', 'chrecoverypct'), ('ch_recovery_n', 'chrecoveryn'),
    ('hb_n', 'hbn'), ('hb_median', 'hbmedian'), ('hb_max', 'hbmax'),
    ('hb_frac_28', 'hbfracc'), ('hb_frac_44', 'hbfracd'),
]


def latex_escape(v):
    """Typeset a computed value so it works in BOTH text and math mode.

    \\ensuremath is essential here: the report writes both "$r=\\sevCr$" and bare "\\sevCr" in
    running text. A macro that carried its own $ delimiters would produce nested math in the first
    case and fail to compile — which is exactly what happened on the first build.
    """
    v = str(v)
    if v == '<1e-4':
        return '\\ensuremath{<\\!10^{-4}}'
    body = v.replace('+', '{+}').replace('-', '{-}')
    # signed scalar, e.g. +0.7697
    if v and v[0] in '+-' and v[1:].replace('.', '').isdigit():
        return '\\ensuremath{%s}' % body
    # confidence interval, e.g. [+0.729, +0.808]
    if v.startswith('[') and v.endswith(']'):
        return '\\ensuremath{%s}' % body.replace(', ', ',\\;')
    return v


def write_latex_macros():
    path = os.path.join(ANALYSIS, 'numbers.tex')
    missing = []
    with open(path, 'w') as f:
        f.write('%% numbers.tex — GENERATED by scripts/report_numbers.py. DO NOT EDIT BY HAND.\n')
        f.write('%% Every value is recomputed from the raw CSVs. Re-run the script to refresh.\n\n')
        for key, macro in MACROS:
            if key not in KV:
                missing.append(key)
                f.write('\\newcommand{\\%s}{\\textbf{??%s??}}\n' % (macro, macro))
            else:
                f.write('\\newcommand{\\%s}{%s}\n' % (macro, latex_escape(KV[key])))
    say()
    say('Wrote analysis\\numbers.tex  (%d macros)' % len(MACROS))
    if missing:
        say('   !! %d macro(s) had no computed value and will render as ??name?? in the PDF:'
            % len(missing))
        for m in missing:
            say('      %s' % m)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    if not os.path.isdir(ANALYSIS):
        os.makedirs(ANALYSIS)
    say('REPORT NUMBERS — regenerated from raw CSVs')
    say('Every figure below is recomputed from data/ and chain_risk/data/ at run time.')
    say('If a number is not in this file, it must not appear in the report or the slides.')

    primary()
    markov()
    npc()
    chain_study()

    hdr('PROVENANCE')
    for fn in sorted(os.listdir(DATA)):
        if fn.endswith('.csv') and not fn.endswith('_ARCHIVED.csv'):
            p = os.path.join(DATA, fn)
            say('   %-52s %8d bytes' % (fn, os.path.getsize(p)))

    write_latex_macros()

    with open(os.path.join(ANALYSIS, 'report_numbers.txt'), 'w') as f:
        f.write('\n'.join(OUT) + '\n')
    with open(os.path.join(ANALYSIS, 'report_numbers.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['key', 'value'])
        for k, v in KV.items():
            w.writerow([k, v])

    say()
    say('Wrote analysis\\report_numbers.txt and analysis\\report_numbers.csv  (%d values)' % len(KV))


if __name__ == '__main__':
    sys.exit(main())
