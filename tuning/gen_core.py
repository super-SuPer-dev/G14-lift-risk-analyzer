"""Generate app/liftrisk_core.py from the group's notebook so the app runs the exact same code."""
import sys
from pathlib import Path

import nbformat

NB, DST = Path(sys.argv[1]), Path(sys.argv[2])
nb = nbformat.read(NB, as_version=4)

HEAD = '''"""Lift-risk pipeline for the app.

GENERATED from G14_LiftRisk_Realtime.ipynb by tuning/gen_core.py - edit the notebook, not this file.
"""
import math, re, time, shutil, subprocess, warnings
from pathlib import Path
from collections import deque, Counter

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import torch
from torch import nn
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "model"
CKPT_NAME = "best_model.pt"
POSE_MODEL_PATH = BASE / "models" / "pose_landmarker_lite.task"   # the app points this at an ASCII temp copy
FPS = 30.0
DRAW_SCENE_TABLE = True      # live mode turns the perspective table off to keep up with the camera

'''
TAIL = '''

def analyze_video(video_path, out_path, height_cm, progress=None, max_frames=None, on_frame=None):
    # run the whole pipeline on one clip: annotated video + per-frame table + per-round table
    global FPS
    cap, info = open_video(video_path)
    cap.release()
    FPS = info["fps"]
    results, calib = run_pipeline(video_path, out_path, height_cm, max_frames=max_frames,
                                  progress=progress, on_frame=on_frame)
    return results, calib, lift_report(results), info


def analyze_stream(cap, info, out_path, height_cm, on_frame=None, max_frames=None):
    # same pipeline fed from an already-open capture (webcam); on_frame may return False to stop
    global FPS
    FPS = info["fps"]
    results, calib = run_pipeline((cap, info), out_path, height_cm, max_frames=max_frames, on_frame=on_frame)
    return results, calib, (lift_report(results) if calib is not None and len(results) else pd.DataFrame())
'''
SKIP_STARTS = ('import subprocess, sys', 'IN_COLAB = ', 'POSE_MODEL_URL = ')
CUTS = ['\n\ncap, VIDEO_INFO = open_video(VIDEO_PATH)', '\nprint("โมเดล:", CHECKPOINT']

parts = [HEAD]
for cell in nb.cells:
    if cell.cell_type != 'code':
        continue
    source = cell.source
    if source.startswith('stem = Path(VIDEO_NAME).stem'):
        break
    if source.startswith(SKIP_STARTS):
        continue
    for cut in CUTS:
        if cut in source:
            source = source[:source.index(cut)]
    parts.append(source.rstrip() + '\n\n')
report = next(c.source for c in nb.cells if c.cell_type == 'code' and 'def lift_report' in c.source)
parts.append(report.split('lifts = lift_report(results)')[0].rstrip() + '\n')
code = '\n'.join(parts) + TAIL

# progress callback instead of printing every 50 frames
old_sig = 'def run_pipeline(video_path, out_path, height_cm, max_frames=None):'
assert old_sig in code
code = code.replace(old_sig, 'def run_pipeline(video_path, out_path, height_cm, max_frames=None, progress=None, on_frame=None):')
for old_text, new_text in [
    ('    cap, info = open_video(video_path)\n    writer =',
     '    cap, info = video_path if isinstance(video_path, tuple) else open_video(video_path)\n    writer ='),
    ('                if m is not None:\n                    img = draw_weight_table(img, calib, m)',
     '                if m is not None and DRAW_SCENE_TABLE:\n                    img = draw_weight_table(img, calib, m)'),
    ('            records.append(row)\n            i += 1\n',
     '            records.append(row)\n            i += 1\n'
     '            if on_frame is not None and on_frame(img, row) is False:\n                break\n'),
]:
    assert old_text in code, old_text
    code = code.replace(old_text, new_text)
old_print = '''            if i % 50 == 0:
                print(f"  {i} เฟรม · ยกไปแล้ว {counter.count} ครั้ง · {i / (time.time() - t0):.1f} เฟรม/วินาที")'''
assert old_print in code
code = code.replace(old_print, '''            if progress is not None and i % 5 == 0:
                progress(i, info["frames"], counter.count)''')
code = code.replace('    print(f"เสร็จ {i} เฟรม ใน {time.time() - t0:.1f} วินาที · นับการยกได้ {counter.count} ครั้ง")\n', '')
code = '\n'.join(line for line in code.splitlines()
                 if not line.startswith(('print(', 'assert count_rounds'))) + '\n'
compile(code, str(DST), 'exec')
DST.write_text(code, encoding='utf-8')
print('wrote', DST, len(code.splitlines()), 'lines')
