"""
split_sweep.py — split a sweep CSV into one standalone file per experimental condition.

Run this straight after chain_position_hazard.py finishes.

WHY THIS EXISTS
The sweep writes every inter-decision-spacing condition into a single CSV, distinguished only by
the setting_cruise_ticks column. That is correct for analysis but poor for provenance: the
condition that reproduces the original single-run experiment (cruise=40) is then not a file anyone
can point at, cite, archive, or hand to someone else. An earlier run of this study was lost because
output filenames did not distinguish runs; this script applies the same lesson one level down, to
conditions within a run.

Inputs are never modified. Outputs are written alongside with a _cruiseNN suffix, and the script
refuses to overwrite an existing file.

Run:  py -3.7 chain_risk\\scripts\\split_sweep.py
"""

import csv
import os
import sys

DATA = r'C:\Users\jayad\Desktop\RPI_AINN\chain_risk\data'

#: Newest first. The first file present is split.
CANDIDATES = [
    'chain_position_hazard_sweep_v2.csv',
    'chain_position_hazard_sweep_v2_pilot.csv',
    'chain_position_hazard_sweep_v1_2026-09-21.csv',
]

#: Condition that reproduces the original single-run experiment, labelled in the output.
PRIMARY = '40'


def main():
    src = None
    for name in CANDIDATES:
        if os.path.exists(os.path.join(DATA, name)):
            src = name
            break
    if src is None:
        print('No sweep file found. Looked for:')
        for c in CANDIDATES:
            print('   %s' % c)
        return 1

    path = os.path.join(DATA, src)
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)

    if 'setting_cruise_ticks' not in (fields or []):
        print('%s has no setting_cruise_ticks column — single-condition file, nothing to split.'
              % src)
        return 0

    print('Source: %s  (%d rows)\n' % (src, len(rows)))

    groups = {}
    for r in rows:
        groups.setdefault(r['setting_cruise_ticks'], []).append(r)

    stem = src[:-4]
    written = 0
    for setting in sorted(groups, key=lambda x: int(x)):
        sub = groups[setting]
        out = os.path.join(DATA, '%s_cruise%s.csv' % (stem, setting))
        label = '  <-- primary condition (reproduces the original run)' \
            if setting == PRIMARY else ''
        if os.path.exists(out):
            print('  SKIP   %s already exists, not overwriting%s'
                  % (os.path.basename(out), label))
            continue
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in sub:
                w.writerow(r)
            f.flush()
            os.fsync(f.fileno())
        chain = sum(1 for r in sub if r.get('arm') == 'CHAIN')
        iso = sum(1 for r in sub if r.get('arm') == 'ISO_LATE')
        interval = (8 + int(setting) + 10) * 0.05
        print('  WROTE  %s' % os.path.basename(out))
        print('         cruise=%s ticks, spacing ~%.2f s, %d rows (%d CHAIN / %d ISO-LATE)%s'
              % (setting, interval, len(sub), chain, iso, label))
        written += 1

    total = sum(len(v) for v in groups.values())
    print('\n%d file(s) written. Row count check: %d split == %d source  %s'
          % (written, total, len(rows), 'OK' if total == len(rows) else 'MISMATCH'))
    print('Source file unchanged.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
