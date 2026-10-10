"""Sweep the notebook's MODIFY parameters over cached keypoints and score them against anthropometry.

Reference fractions of standing height (Drillis & Contini): knee 0.285, shoulder 0.818, wrist 0.485.
These ratios do not depend on the height we type in (everything scales linearly), so they are a fair
check even before the group gives us their real heights.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import wlimit_core as w

WORK = Path(__file__).parent / 'work'
CACHE = WORK / 'cache'
HEIGHT_CM = 170.0                      # ratios are invariant to this
REF = {'knee': 0.285, 'shoulder': 0.818, 'wrist': 0.485}


def analyze(path, fov=70.0, calib_seconds=1.5, waist_fraction=0.30, nose_above_foot=0.905,
            world_window=5, history_frames=3):
    data = np.load(path)
    xy_raw, world_raw, conf = data['xy'], data['world'], data['conf']
    fps, width, height = float(data['fps']), int(data['width']), int(data['height'])
    w.FPS, w.CAMERA_FOV_DEG, w.CALIB_SECONDS = fps, fov, calib_seconds
    w.WAIST_FRACTION, w.NOSE_ABOVE_FOOT = waist_fraction, nose_above_foot
    w.WORLD_WINDOW, w.HISTORY_FRAMES = world_window, history_frames

    smoother = w.PoseSmoother(width, height)
    frames = []
    for i in range(len(xy_raw)):
        xy_s, world_s, observed = smoother.update(xy_raw[i], world_raw[i], conf[i])
        frames.append({'xy': xy_s, 'world': world_s, 'observed': observed})

    n_calib = int(round(calib_seconds * fps))
    calib = w.calibrate(frames[:n_calib], HEIGHT_CM, width, height)

    pose, rows = None, []
    for i, f in enumerate(frames):
        P = f['world'] * 100 * calib['world_scale']
        pose = w.solve_camera_pose(P, f['xy'], f['observed'], calib['K'], pose)
        row = {'frame': i + 1, 'twist': w.twist_from_baseline(f['world'], calib)}
        if pose is not None:
            C = w.to_camera_3d(f['xy'], P, pose, calib['K'])
            row['orient'] = w.orientation_angle(C)
            row['pnp_z'] = pose['t'][2]
            m = w.measure_hands(C, f['xy'], calib)
            if m is not None:
                row.update({'H': m['H'], 'V': m['V'], 'limit': m['limit_kg'],
                            'zone': w.ROW_NAMES[m['row']], 'reach': w.COL_NAMES[m['col']]})
        rows.append(row)
    table = pd.DataFrame(rows)
    still = table.iloc[n_calib:n_calib + 15]
    return {
        'clip': Path(path).stem,
        'cm_per_px': calib['cm_per_px'],
        'cam_dist_cm': calib['camera_dist_cm'],
        'pnp_z_cm': table['pnp_z'].median() if 'pnp_z' in table else np.nan,
        'knee_ratio': calib['knee_cm'] / HEIGHT_CM,
        'waist_ratio': calib['waist_cm'] / HEIGHT_CM,
        'shoulder_ratio': calib['shoulder_cm'] / HEIGHT_CM,
        'V_still_ratio': still['V'].median() / HEIGHT_CM if 'V' in still else np.nan,
        'H_still_cm': still['H'].median() if 'H' in still else np.nan,
        'orient_med': table['orient'].median() if 'orient' in table else np.nan,
        'twist_max': table['twist'].max(),
        'limit_min': table['limit'].min() if 'limit' in table else np.nan,
    }, table


def score(row):
    # how far the calibration is from the anthropometric reference (lower is better)
    return (abs(row['knee_ratio'] - REF['knee']) + abs(row['shoulder_ratio'] - REF['shoulder'])
            + abs(row['V_still_ratio'] - REF['wrist']))


if __name__ == '__main__':
    clips = sorted(CACHE.glob('*.npz'))
    if len(sys.argv) > 1:
        clips = [c for c in clips if any(k in c.stem for k in sys.argv[1:])]
    results = []
    for fov in [50, 60, 70, 80, 90, 110]:
        for clip in clips:
            row, _ = analyze(clip, fov=fov)
            row['fov'] = fov
            row['score'] = score(row)
            results.append(row)
            print(f"fov {fov:3d} | {row['clip'][:28]:28s} knee {row['knee_ratio']:.3f} "
                  f"shoulder {row['shoulder_ratio']:.3f} wrist {row['V_still_ratio']:.3f} "
                  f"camdist {row['cam_dist_cm']:5.0f} pnpZ {row['pnp_z_cm']:5.0f} "
                  f"H_still {row['H_still_cm']:5.1f} score {row['score']:.3f}", flush=True)
    table = pd.DataFrame(results)
    table.to_csv(WORK / 'out' / 'fov_sweep.csv', index=False)
    print()
    print(table.groupby('fov')[['knee_ratio', 'shoulder_ratio', 'V_still_ratio', 'H_still_cm',
                                'cam_dist_cm', 'pnp_z_cm', 'score']].median().round(3).to_string())
