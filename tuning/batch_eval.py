"""Run the notebook's pipeline (no drawing) on every cached clip and compare with what was really done.

The functions come straight from the notebook (its definition cells are exec'd), so this checks the
same code the group submits. Expected rounds: G14 clips from the filename code (transfer = 1, NEG = 0),
the two new multi-lift clips from the group's own count.
"""
import os
import re
import sys
from collections import deque
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

WORK = Path(__file__).parent / 'work'
HEIGHTS = {'VID_long': 167.0, 'VID_fast': 167.0}      # real height where the group told us
S03_HEIGHT = 167.0                                     # G14 S03 is the same person as in the two new clips
DEFAULT_HEIGHT = 170.0
CACHE_NAME = sys.argv[1] if len(sys.argv) > 1 else 'cache'          # e.g. cache_other for other groups' clips
OUT_NAME = sys.argv[2] if len(sys.argv) > 2 else 'batch_eval.csv'
EXPECT_ROUNDS = {'VID_long': 8, 'VID_fast': 7}

nb = nbformat.read(WORK / 'G14_LiftRisk_Realtime.ipynb', as_version=4)
ns = {'__name__': 'nb'}
os.chdir(WORK)
for cell in nb.cells:
    if cell.cell_type != 'code':
        continue
    if cell.source.startswith('stem = Path(VIDEO_NAME).stem'):      # stop before the run cells
        break
    exec(compile(cell.source, '<nb>', 'exec'), ns)
for cell in nb.cells:                                              # lift_report is defined after the run cells
    if cell.cell_type == 'code' and 'def lift_report' in cell.source:
        exec(compile(cell.source.split('lifts = lift_report(results)')[0], '<nb>', 'exec'), ns)

ZONE_CODE = {'AS': 'Above Shoulder', 'WS': 'Waist-Shoulder', 'KW': 'Knee-Waist', 'BK': 'Below Knee'}


def analyze(path, height_cm):
    data = np.load(path)
    fps, width, height = float(data['fps']), int(data['width']), int(data['height'])
    ns['FPS'] = fps
    smoother = ns['PoseSmoother'](width, height)
    features = ns['FeatureStream'](ns['SCALE_MEAN'], ns['SCALE_STD'])
    buffer = deque(maxlen=ns['WINDOW'])
    voter = ns['MajorityVote'](ns['SMOOTH_FRAMES'])
    counter = ns['HeightRoundCounter']()
    twist_recent = deque(maxlen=ns['TWIST_SMOOTH_FRAMES'])
    n_calib = int(round(ns['CALIB_SECONDS'] * fps))
    calib, samples, pose, records = None, [], None, []

    def tag(finished):
        if finished is not None:
            for past in records[finished[0]:finished[1] + 1]:
                past['round'] = counter.count

    for i in range(len(data['xy'])):
        xy_s, world_s, observed = smoother.update(data['xy'][i], data['world'][i], data['conf'][i])
        row = {'frame_number': i + 1, 'time_s': (i + 1) / fps, 'round': 0}
        buffer.append(features.update(xy_s, observed, data['conf'][i]))
        label = None
        if len(buffer) == ns['WINDOW']:
            probs = ns['predict_window'](ns['MODEL'], np.stack(buffer))
            row['raw_label'] = ns['CLASS_NAMES'][int(probs.argmax())]
            label = voter.update(row['raw_label'])
        row['label'] = label
        row['is_lift'] = label in ns['LIFT_CLASSES']
        hand_height = None
        if calib is None:
            samples.append({'xy': xy_s, 'world': world_s, 'observed': observed})
            if len(samples) == n_calib:
                calib = ns['calibrate'](samples, height_cm, width, height)
        else:
            P = world_s * 100 * calib['world_scale']
            pose = ns['solve_camera_pose'](P, xy_s, observed, calib['K'], pose)
            row['twist_deg'] = ns['twist_from_baseline'](world_s, calib)
            twist_recent.append(row['twist_deg'])
            row['twist_smooth_deg'] = float(np.nanmedian(twist_recent))
            if pose is not None:
                C = ns['to_camera_3d'](xy_s, P, pose, calib['K'])
                row['orientation_deg'] = ns['orientation_angle'](C)
                row['shoulder_dir_deg'] = ns['shoulder_direction'](C, calib['up'])
                m = ns['measure_hands'](C, xy_s, calib)
                if m is not None and m['H'] <= ns['MAX_REACH_FRACTION'] * height_cm:
                    hand_height = m['V']
                    row.update({'H_cm': m['H'], 'V_cm': m['V'], 'zone': ns['ROW_NAMES'][m['row']],
                                'reach_class': ns['COL_NAMES'][m['col']], 'limit_kg': m['limit_kg']})
        tag(counter.update(hand_height, row['is_lift']))
        row['lift_count'] = counter.count
        records.append(row)
    tag(counter.finish())
    return pd.DataFrame(records), calib, counter.count


rows = []
for path in sorted((WORK / CACHE_NAME).glob('*.npz')):
    stem = path.stem
    results, calib, count = analyze(path, S03_HEIGHT if stem.startswith('G14_S03') else HEIGHTS.get(stem, DEFAULT_HEIGHT))
    results.to_csv(WORK / 'out' / f'{stem}_batch.csv', index=False)
    lifts = ns['lift_report'](results)
    expect = EXPECT_ROUNDS.get(stem, 0 if 'NEG' in stem else 1)
    twist_code = re.search(r'_(T[01])_', stem)
    found = re.search(r'_([A-Z]{2})-([A-Z]{2})_', stem)
    seconds = results['zone'].value_counts() / float(np.load(path)['fps'])
    visited = set(seconds[seconds >= 0.2].index)
    row = {'clip': stem.replace('G14_', ''), 'expect_rounds': expect, 'rounds': count,
           'count_ok': count == expect, 'report_rows': len(lifts),
           'twist_code': twist_code.group(1) if twist_code else '',
           'twist_max': lifts['twist_max_deg'].max() if len(lifts) else np.nan,
           'swing_max': lifts['shoulder_swing_deg'].max() if len(lifts) else np.nan,
           'twist_flag': bool(lifts['twist_flag'].any()) if len(lifts) else False,
           'orient_med': round(results['orientation_deg'].median(), 0),
           'H_max': round(results['H_cm'].max(), 1), 'limit_min': results['limit_kg'].min()}
    if twist_code:
        row['twist_ok'] = row['twist_flag'] == (row['twist_code'] == 'T1')
    if found:
        row['from_visited'] = ZONE_CODE[found.group(1)] in visited
        row['to_visited'] = ZONE_CODE[found.group(2)] in visited
    rows.append(row)
    print(row, flush=True)

table = pd.DataFrame(rows)
table.to_csv(WORK / 'out' / OUT_NAME, index=False)
pd.set_option('display.width', 250)
print()
print(table.to_string(index=False))
old = table[table['twist_code'] != '']
lifted = old[~old['clip'].str.contains('NEG')]
t1 = lifted[lifted['twist_code'] == 'T1']
t0 = lifted[lifted['twist_code'] == 'T0']
print(f"\nคลิปที่มีการยก: มุมบิดตรงรหัส {int(lifted['twist_ok'].sum())}/{len(lifted)} | จับ T1 ได้ {int(t1['twist_flag'].sum())}/{len(t1)}"
      f" | เตือนผิดใน T0 {int(t0['twist_flag'].sum())}/{len(t0)}")
print(f"นับรอบถูก {int(table['count_ok'].sum())}/{len(table)} คลิป | มุมบิดตรงรหัส (รวม NEG) {int(old['twist_ok'].sum())}/{len(old)}"
      f" | ผ่านโซนต้นทาง {int(old['from_visited'].sum())}/{int(old['from_visited'].notna().sum())}"
      f" | ผ่านโซนปลายทาง {int(old['to_visited'].sum())}/{int(old['to_visited'].notna().sum())}")
