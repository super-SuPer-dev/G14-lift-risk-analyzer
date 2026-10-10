"""Search a twist rule that holds across groups: per-clip features from the per-frame CSVs of explore_twist.py."""
import itertools
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / 'work' / 'out'
rows = []
for path in sorted(OUT.glob('*_twist.csv')):
    stem = path.stem[:-6]
    d = pd.read_csv(path)
    b = pd.read_csv(OUT / f'{stem}_batch.csv')
    d = d.merge(b[['frame_number', 'round']].assign(frame=lambda x: x['frame_number'] - 1), on='frame', how='left')
    med = lambda c: d[c].rolling(5, center=True, min_periods=1).median()
    base = d['az_sh'].dropna().head(10).median()
    dev = ((med('az_sh') - base + 90) % 180 - 90).abs()          # shoulder line vs its standing direction, 0..90
    inr = d['round'] > 0
    a = np.radians(med('az_sh')[inr].dropna().to_numpy())
    rng = np.nan
    if len(a) >= 5:
        mean = math.atan2(np.sin(a).mean(), np.cos(a).mean())
        w = np.degrees((a - mean + np.pi) % (2 * np.pi) - np.pi)
        rng = np.percentile(w, 95) - np.percentile(w, 5)
    rows.append({'clip': stem, 'group': stem[:3], 'subject': stem[:7], 'neg': 'NEG' in stem,
                 'T1': bool(re.search(r'_T1_', stem)),
                 'A_shhip_rmax': med('sh_hip')[inr].max(), 'B_sh_rrange': rng,
                 'C_shfloor_rmax': dev[inr].max(), 'C_shfloor_rp90': dev[inr].quantile(0.9), 'rounds': int(d['round'].max())})
t = pd.DataFrame(rows)
t.to_csv(OUT / 'twist_rule_features.csv', index=False)
lift = t[~t['neg'] & (t['rounds'] > 0)].reset_index(drop=True)
print(len(lift), 'clips with a counted round | T1', int(lift['T1'].sum()))
pd.set_option('display.width', 250)
print(lift.drop(columns=['neg', 'group']).round(0).to_string(index=False))


def fit(df, cols):
    best = (-1, None)
    grids = [np.unique(np.round(df[c].dropna(), 0)) for c in cols]
    for thr in itertools.product(*grids):
        pred = np.zeros(len(df), bool)
        for c, x in zip(cols, thr):
            pred |= (df[c] > x).to_numpy()
        acc = (pred == df['T1'].to_numpy()).mean()
        if acc > best[0]:
            best = (acc, thr)
    return best


def apply(df, cols, thr):
    pred = np.zeros(len(df), bool)
    for c, x in zip(cols, thr):
        pred |= (df[c] > x).to_numpy()
    return pred == df['T1'].to_numpy()


for cols in [['A_shhip_rmax'], ['B_sh_rrange'], ['C_shfloor_rmax'], ['C_shfloor_rp90'],
             ['A_shhip_rmax', 'B_sh_rrange'], ['A_shhip_rmax', 'C_shfloor_rmax'], ['A_shhip_rmax', 'C_shfloor_rp90']]:
    acc, thr = fit(lift, cols)
    line = f"{' OR '.join(cols):38s} all: thr {thr} acc {acc:.2f}"
    for key in ['group', 'subject']:
        ok = []
        for g in lift[key].unique() if key == 'subject' else ['G14', 'other']:
            test = lift[key] == g if key == 'subject' else ((lift['group'] == 'G14') == (g == 'G14'))
            _, th = fit(lift[~test], cols)
            ok += list(apply(lift[test], cols, th))
        line += f" | leave-{key}-out {np.mean(ok):.2f} ({int(np.sum(ok))}/{len(ok)})"
    print(line)
