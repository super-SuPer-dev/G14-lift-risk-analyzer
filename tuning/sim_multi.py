"""Simulate multi-lift clips by joining cached keypoints of one lifter's four single-lift clips.

This is NOT the real test the assignment asks for (the joins are not real motion); it only tells us
early how the round counter behaves when lifts come back to back, and which RELEASE_FRAMES is safe.
"""
import os
from collections import Counter, deque
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

WORK = Path(__file__).parent / 'work'
os.chdir(WORK)
nb = nbformat.read(WORK / 'G14_LiftRisk_Realtime.ipynb', as_version=4)
ns = {'__name__': 'nb'}
for cell in nb.cells:
    if cell.cell_type != 'code':
        continue
    if cell.source.startswith('stem = Path(VIDEO_NAME).stem'):
        break
    exec(compile(cell.source, '<nb>', 'exec'), ns)

HANDLING = {'lift', 'carry/hold', 'lower/place'}
ORDER = ['BK-WS', 'KW-AS', 'AS-WS', 'WS-KW']


def handling_range(stem):
    labels = pd.read_csv(WORK / 'out' / f'{stem}_batch.csv')['label']
    frames = np.flatnonzero(labels.isin(HANDLING).to_numpy())
    return int(frames[0]), int(frames[-1])


def build(subject, gap):
    # gap = idle frames kept on each side of a lift (None = keep the whole clips)
    parts = {'xy': [], 'world': [], 'conf': []}
    for k, code in enumerate(ORDER):
        path = next((WORK / 'cache').glob(f'G14_{subject}_{code}_*.npz'))
        data = np.load(path)
        n = len(data['xy'])
        a, b = 0, n - 1
        if gap is not None:
            first, last = handling_range(path.stem)
            a = 0 if k == 0 else max(0, first - gap)          # the first clip keeps its calibration part
            b = min(n - 1, last + gap)
        for key in parts:
            parts[key].append(data[key][a:b + 1])
        meta = (float(data['fps']), int(data['width']), int(data['height']))
    return {key: np.concatenate(value) for key, value in parts.items()}, meta


def raw_labels(seq, meta):
    fps, width, height = meta
    ns['FPS'] = fps
    smoother = ns['PoseSmoother'](width, height)
    features = ns['FeatureStream'](ns['SCALE_MEAN'], ns['SCALE_STD'])
    buffer, labels = deque(maxlen=ns['WINDOW']), []
    for i in range(len(seq['xy'])):
        xy_s, _, observed = smoother.update(seq['xy'][i], seq['world'][i], seq['conf'][i])
        buffer.append(features.update(xy_s, observed, seq['conf'][i]))
        if len(buffer) == ns['WINDOW']:
            probs = ns['predict_window'](ns['MODEL'], np.stack(buffer))
            labels.append(ns['CLASS_NAMES'][int(probs.argmax())])
        else:
            labels.append(None)
    return labels


def count(labels, release, min_frames=20, smooth=9):
    voter = deque(maxlen=smooth)
    counter = ns['RoundCounter'](min_frames, release)
    for raw in labels:
        if raw is None:
            counter.update(False)
            continue
        voter.append(raw)
        counter.update(Counter(voter).most_common(1)[0][0] in HANDLING)
    return counter.count


RELEASES = [0, 5, 10, 15, 20, 30, 45, 60]
rows = []
for subject in ['S01', 'S02', 'S03', 'S04']:
    for gap in [None, 45, 20, 5]:
        seq, meta = build(subject, gap)
        labels = raw_labels(seq, meta)
        row = {'subject': subject, 'idle_between': 'เต็มคลิป' if gap is None else f'{2 * gap} เฟรม ({2 * gap / 30:.1f} วิ)',
               'frames': len(labels), 'expect': 4}
        for release in RELEASES:
            row[f'rel{release}'] = count(labels, release)
        rows.append(row)
        print(row, flush=True)

table = pd.DataFrame(rows)
table.to_csv(WORK / 'out' / 'sim_multi_lift.csv', index=False)
print()
print(table.to_string(index=False))
print('\nจำนวนชุดที่นับได้ 4 รอบพอดี (จาก', len(table), 'ชุด):')
print({f'release {r}': int((table[f'rel{r}'] == 4).sum()) for r in RELEASES})
for name, part in table.groupby('idle_between', sort=False):
    print(name, {f'rel{r}': int((part[f'rel{r}'] == 4).sum()) for r in RELEASES})
