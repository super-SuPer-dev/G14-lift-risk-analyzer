"""Offline test of a hand-height round counter (optionally confirmed by the model) on 22 clips.

A round = one excursion of the hand height V away from its standing baseline:
  enter when |V - V0| > hi for `k` frames in a row, leave when |V - V0| < lo for `k` frames.
The excursion is counted only if the model voted a handling class in at least `m` of its frames.
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path(__file__).parent / 'work'
HANDLING = {'lift', 'carry/hold', 'lower/place'}
clips, expect = {}, {}
for p in sorted((WORK / 'out').glob('G14_*_batch.csv')):
    name = p.stem.replace('_batch', '')
    clips[name] = pd.read_csv(p)
    expect[name] = 0 if 'NEG' in name else 1
for name, n in [('VID_long', 8), ('VID_fast', 7)]:
    clips[name] = pd.read_csv(WORK / 'new_out' / f'{name}_frames.csv')
    expect[name] = n


def count(table, hi, lo, k, m, classes=HANDLING):
    v = table['V_cm'].to_numpy()
    handling = table['label'].isin(classes).to_numpy()
    measured = v[np.isfinite(v)]
    if len(measured) < 15:
        return 0, []
    v0 = float(np.median(measured[:15]))
    active, above, below, seen, counted, rounds, start = False, 0, 0, 0, False, [], 0
    for i, value in enumerate(v):
        if np.isfinite(value):
            far = abs(value - v0)
            above = above + 1 if far > hi else 0
            below = below + 1 if far < lo else 0
        if not active and above >= k:
            active, seen, counted, start = True, 0, False, i
        if active:
            seen += int(handling[i])
            if not counted and seen >= m:
                counted = True
                rounds.append(start)
            if below >= k:
                active = False
    return len(rounds), rounds


rows = []
for hi, k, m in itertools.product([6, 8, 10, 12, 15, 18], [3, 5, 8, 12], [0, 1, 3, 5, 8, 12, 20]):
    lo = hi / 2
    got = {n: count(t, hi, lo, k, m)[0] for n, t in clips.items()}
    rows.append({'hi': hi, 'lo': lo, 'k': k, 'm': m,
                 'correct': sum(got[n] == expect[n] for n in clips),
                 'neg_ok': sum(got[n] == 0 for n in clips if expect[n] == 0),
                 'single_ok': sum(got[n] == 1 for n in clips if expect[n] == 1),
                 'long': got['VID_long'], 'fast': got['VID_fast']})
table = pd.DataFrame(rows).sort_values(['correct', 'hi'], ascending=[False, True])
table.to_csv(WORK / 'out' / 'geo_counter_sweep.csv', index=False)
print(table.head(15).to_string(index=False))
print('\nไม่ใช้โมเดลเลย (m=0) ดีที่สุด:')
print(table[table.m == 0].head(3).to_string(index=False))

best = table.iloc[0]
print('\nรายคลิปด้วยค่าดีที่สุด:', dict(best))
for n, t in clips.items():
    got, starts = count(t, best.hi, best.lo, int(best.k), int(best.m))
    flag = '' if got == expect[n] else '  <-- ผิด'
    print(f'{n:34s} expect {expect[n]} got {got}{flag}')
