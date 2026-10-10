"""Compare candidate twist measures on every coded clip (G14 + other groups) to find one that generalises."""
import math
import os
import re
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

WORK = Path(__file__).parent / 'work'
nb = nbformat.read(WORK / 'G14_LiftRisk_Realtime.ipynb', as_version=4)
ns = {'__name__': 'nb'}
os.chdir(WORK)
for cell in nb.cells:
    if cell.cell_type != 'code':
        continue
    if cell.source.startswith('stem = Path(VIDEO_NAME).stem'):
        break
    exec(compile(cell.source, '<nb>', 'exec'), ns)
unit = ns['unit']


def signed(a, b, axis):
    a = unit(a - (a @ axis) * axis)
    b = unit(b - (b @ axis) * axis)
    return math.degrees(math.atan2(np.cross(a, b) @ axis, a @ b))


def wrap(d):
    return abs((d + 180) % 360 - 180)


def line_diff(d):                       # lines have no direction: 0..90
    d = wrap(d)
    return min(d, 180 - d)


def per_frame(path, height_cm=170.0):
    data = np.load(path)
    fps, width, height = float(data['fps']), int(data['width']), int(data['height'])
    ns['FPS'] = fps
    smoother = ns['PoseSmoother'](width, height)
    n_calib = int(round(ns['CALIB_SECONDS'] * fps))
    calib, samples, pose, rows, base = None, [], None, [], {}
    for i in range(len(data['xy'])):
        xy_s, world_s, observed = smoother.update(data['xy'][i], data['world'][i], data['conf'][i])
        if calib is None:
            samples.append({'xy': xy_s, 'world': world_s, 'observed': observed})
            if len(samples) == n_calib:
                calib = ns['calibrate'](samples, height_cm, width, height)
                stand, p = [], None
                for s in samples:                      # baseline of the new angles while standing still
                    P = s['world'] * 100 * calib['world_scale']
                    p = ns['solve_camera_pose'](P, s['xy'], s['observed'], calib['K'], p)
                    if p is not None:
                        C = ns['to_camera_3d'](s['xy'], P, p, calib['K'])
                        up = calib['up']
                        stand.append({'sh_feet': signed(C[28] - C[27], C[12] - C[11], up),
                                      'hip_feet': signed(C[28] - C[27], C[24] - C[23], up)})
                base = pd.DataFrame(stand).median().to_dict()
            continue
        P = world_s * 100 * calib['world_scale']
        pose = ns['solve_camera_pose'](P, xy_s, observed, calib['K'], pose)
        row = {'frame': i, 'sh_hip': ns['twist_from_baseline'](world_s, calib)}
        if pose is not None:
            C = ns['to_camera_3d'](xy_s, P, pose, calib['K'])
            up = calib['up']
            row['sh_feet'] = wrap(signed(C[28] - C[27], C[12] - C[11], up) - base['sh_feet'])
            row['hip_feet'] = wrap(signed(C[28] - C[27], C[24] - C[23], up) - base['hip_feet'])
            e1 = unit(np.cross(up, [0.0, 0.0, 1.0]))            # floor-fixed horizontal axes (camera does not move)
            e2 = np.cross(up, e1)
            az = lambda v: math.degrees(math.atan2(v @ e2, v @ e1))
            row['az_sh'] = az(C[12] - C[11])
            row['az_hip'] = az(C[24] - C[23])
            row['az_feet'] = az(C[28] - C[27])
            hand = np.nanmean([ns['mean_point'](C, ns['LEFT_HAND']), ns['mean_point'](C, ns['RIGHT_HAND'])], axis=0)
            dh = hand - calib['floor']
            dh = dh - (dh @ up) * up
            if np.isfinite(dh).all() and np.linalg.norm(dh) >= 15:
                row['az_hand'] = az(dh)
            m = ns['measure_hands'](C, xy_s, calib)
            if m is not None and m['H'] <= ns['MAX_REACH_FRACTION'] * height_cm:
                row['H'] = m['H']
                row['V'] = m['V']
                if m['H'] >= 15:
                    row['asym_hip'] = math.degrees(math.atan2(abs(m['side']), abs(m['front'])))
                    feet = C[28] - C[27]
                    feet = unit(feet - (feet @ up) * up)
                    d = (np.nanmean([ns['mean_point'](C, ns['LEFT_HAND']), ns['mean_point'](C, ns['RIGHT_HAND'])], axis=0) - m['body']['O'])
                    d = d - (d @ up) * up
                    side = abs(d @ feet)
                    row['asym_feet'] = math.degrees(math.atan2(side, math.sqrt(max(d @ d - side ** 2, 1e-9))))
        rows.append(row)
    return pd.DataFrame(rows)


out = []
for cache in ['cache', 'cache_other']:
    for path in sorted((WORK / cache).glob('*.npz')):
        code = re.search(r'_(T[01])_', path.stem)
        if not code:
            continue
        d = per_frame(path)
        d.to_csv(WORK / 'out' / f'{path.stem}_twist.csv', index=False)
        b = pd.read_csv(WORK / 'out' / f'{path.stem}_batch.csv')
        d = d.merge(b[['frame_number', 'round']].assign(frame=lambda x: x['frame_number'] - 1), on='frame', how='left')
        row = {'clip': path.stem, 'group': path.stem[:3], 'neg': 'NEG' in path.stem, 'T1': code.group(1) == 'T1'}
        for col in ['sh_hip', 'sh_feet', 'hip_feet', 'asym_hip', 'asym_feet']:
            if col not in d:
                continue
            s = d[col].rolling(5, center=True, min_periods=1).median()
            row[f'{col}_max'] = d[col].max()
            row[f'{col}_p95'] = s.quantile(0.95)
            row[f'{col}_p80'] = s.quantile(0.80)
            inr = s[d['round'] > 0]
            row[f'{col}_rmax'] = inr.max() if len(inr) else np.nan
            row[f'{col}_rp90'] = inr.quantile(0.90) if len(inr) else np.nan
        for col in ['az_sh', 'az_hip', 'az_feet', 'az_hand']:      # how far the direction swings (circular range)
            if col not in d:
                continue
            for tag, part in [('all', d), ('r', d[d['round'] > 0])]:
                a = np.radians(part[col].rolling(5, center=True, min_periods=1).median().dropna().to_numpy())
                if len(a) < 5:
                    row[f'{col}_{tag}range'] = np.nan
                    continue
                mean = math.atan2(np.sin(a).mean(), np.cos(a).mean())
                dev = np.degrees((a - mean + np.pi) % (2 * np.pi) - np.pi)
                row[f'{col}_{tag}range'] = np.percentile(dev, 95) - np.percentile(dev, 5)
        out.append(row)
        print(path.stem, flush=True)
table = pd.DataFrame(out)
table.to_csv(WORK / 'out' / 'twist_explore.csv', index=False)

lift = table[~table['neg']]
print(len(lift), 'transfer clips | T1', int(lift['T1'].sum()))


def auc(x, y):
    ok = x.notna()
    x, y = x[ok].to_numpy(), y[ok].to_numpy()
    pos, neg = x[y], x[~y]
    return np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])


def best_thr(x, y):
    cands = np.unique(np.round(x.dropna(), 1))
    acc = [((x > t) == y).mean() for t in cands]
    return cands[int(np.argmax(acc))], max(acc)


res = []
for col in [c for c in table.columns if c.endswith(('range', '_rmax', '_rp90'))]:
    g14, oth = lift[lift['group'] == 'G14'], lift[lift['group'] != 'G14']
    t_all, a_all = best_thr(lift[col], lift['T1'])
    t14, _ = best_thr(g14[col], g14['T1'])
    to, _ = best_thr(oth[col], oth['T1'])
    res.append({'measure': col, 'auc_all': auc(lift[col], lift['T1']), 'auc_g14': auc(g14[col], g14['T1']),
                'auc_other': auc(oth[col], oth['T1']), 'thr_all': t_all, 'acc_all': a_all,
                'thr_g14': t14, 'g14thr_on_other': ((oth[col] > t14) == oth['T1']).mean(),
                'thr_other': to, 'otherthr_on_g14': ((g14[col] > to) == g14['T1']).mean()})
pd.set_option('display.width', 250)
print(pd.DataFrame(res).round(2).sort_values('auc_all', ascending=False).to_string(index=False))
