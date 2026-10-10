"""Extract raw MediaPipe keypoints once per clip and cache them, so parameter sweeps are instant."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import wlimit_core as w

WORK = Path(__file__).parent / 'work'
w.POSE_MODEL_PATH = WORK / 'models' / 'pose_landmarker_lite.task'
CACHE = WORK / (sys.argv[2] if len(sys.argv) > 2 else 'cache')
CACHE.mkdir(parents=True, exist_ok=True)

for clip in sorted(Path(sys.argv[1]).glob('*.mp4')):
    out = CACHE / (clip.stem + '.npz')
    if out.is_file():
        print('skip', clip.stem, flush=True)
        continue
    cap, info = w.open_video(clip)
    w.FPS = info['fps']
    xy_all, world_all, conf_all = [], [], []
    with w.create_pose_detector() as detector:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            xy, world, conf = w.extract_keypoints(detector, frame, i)
            xy_all.append(xy)
            world_all.append(world)
            conf_all.append(conf)
            i += 1
    cap.release()
    np.savez_compressed(out, xy=np.array(xy_all), world=np.array(world_all), conf=np.array(conf_all),
                        fps=info['fps'], width=info['width'], height=info['height'])
    print('cached', clip.stem, len(xy_all), 'frames', flush=True)
print('done')
