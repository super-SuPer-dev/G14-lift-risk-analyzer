"""Offline tuning on the 20 cached G14 clips: lift counter, twist threshold, zone visits, H outliers."""
import itertools
import re
from collections import Counter, deque
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / 'work' / 'out'
ZONE_CODE = {'AS': 'Above Shoulder', 'WS': 'Waist-Shoulder', 'KW': 'Knee-Waist', 'BK': 'Below Knee'}
clips = {p.stem.replace('_batch', ''): pd.read_csv(p) for p in sorted(OUT.glob('G14_*_batch.csv'))}
expect = {name: 0 if 'NEG' in name else 1 for name in clips}


def count_lifts(raw_labels, lift_classes, smooth, min_frames, release):
    # majority vote -> LIFT state -> count runs; a run only ends after `release` NO-LIFT frames
    buf, run, gap, counted, count = deque(maxlen=smooth), 0, 0, False, 0
    for raw in raw_labels:
        if not isinstance(raw, str):
            continue
        buf.append(raw)
        lifting = Counter(buf).most_common(1)[0][0] in lift_classes
        if lifting:
            run += 1
            gap = 0
        else:
            gap += 1
            if gap > release:
                run, counted = 0, False
        if run >= min_frames and not counted:
            count += 1
            counted = True
    return count


print('== 1) ตัวนับรอบ: จำนวนคลิปที่นับถูก (จาก 20) ==')
rows = []
for classes, smooth, min_frames, release in itertools.product(
        [('lift',), ('lift', 'carry/hold'), ('lift', 'carry/hold', 'lower/place')],
        [9, 15, 21, 31], [6, 10, 15, 20], [0, 15, 30, 45, 60]):
    counts = {n: count_lifts(d['raw_label'], set(classes), smooth, min_frames, release) for n, d in clips.items()}
    correct = sum(counts[n] == expect[n] for n in clips)
    over = sum(counts[n] > expect[n] for n in clips)
    under = sum(counts[n] < expect[n] for n in clips)
    rows.append({'classes': '+'.join(c.split('/')[0] for c in classes), 'smooth': smooth,
                 'min_frames': min_frames, 'release': release, 'correct': correct,
                 'over': over, 'under': under})
sweep = pd.DataFrame(rows).sort_values(['correct', 'release'], ascending=[False, True])
sweep.to_csv(OUT / 'counter_sweep.csv', index=False)
print(sweep.head(12).to_string(index=False))
print('ค่าตั้งต้นของ demo (lift, smooth 9, min 6, release 0):',
      sweep[(sweep.classes == 'lift') & (sweep.smooth == 9) & (sweep.min_frames == 6) & (sweep.release == 0)]
      [['correct', 'over', 'under']].to_dict('records'))
print(sweep.groupby('classes')['correct'].max().to_dict(), '| by release:',
      sweep[sweep.classes == 'lift'].groupby('release')['correct'].max().to_dict())

print('\n== 2) เกณฑ์มุมบิด ==')
twist = pd.DataFrame([{'clip': n, 'code': re.search(r'_(T[01])_', n).group(1),
                       'max': d['twist_deg'].max(), 'p95': d['twist_deg'].quantile(0.95)}
                      for n, d in clips.items()])
print(twist.sort_values('max').round(1).to_string(index=False))
for column in ['max', 'p95']:
    best = max(((sum((twist[column] > t) == (twist['code'] == 'T1')), t) for t in range(10, 61)),
               key=lambda x: (x[0], -abs(x[1] - 45)))
    print(f'ใช้ {column}: เกณฑ์ 45° ถูก {sum((twist[column] > 45) == (twist["code"] == "T1"))}/20 '
          f'| เกณฑ์ดีที่สุด {best[1]}° ถูก {best[0]}/20')

print('\n== 3) โซนที่มือผ่าน เทียบรหัสต้นทาง/ปลายทาง (เฉพาะคลิปยก 16 คลิป) ==')
zone_rows = []
for n, d in clips.items():
    found = re.search(r'_([A-Z]{2})-([A-Z]{2})_', n)
    if not found:
        continue
    seconds = d['zone'].value_counts() / 30
    visited = set(seconds[seconds >= 0.2].index)            # stayed at least 0.2 s
    origin, dest = ZONE_CODE[found.group(1)], ZONE_CODE[found.group(2)]
    zone_rows.append({'clip': n.replace('G14_', ''), 'from': origin, 'to': dest,
                      'from_visited': origin in visited, 'to_visited': dest in visited,
                      'V_min': d['V_cm'].min(), 'V_max': d['V_cm'].max(),
                      'visited': ', '.join(z.split()[0] for z in seconds[seconds >= 0.2].index)})
zones = pd.DataFrame(zone_rows)
print(zones.round(0).to_string(index=False))
print('ผ่านโซนต้นทาง', zones['from_visited'].sum(), '/ 16 | ผ่านโซนปลายทาง', zones['to_visited'].sum(), '/ 16')
print(zones.groupby('from')['from_visited'].agg(['sum', 'count']).to_string())

print('\n== 4) ค่า H ที่เกินช่วงแขน (> 80 ซม.) ==')
for n, d in clips.items():
    bad = d[d['H_cm'] > 80]
    if len(bad):
        print(f"{n}: {len(bad)} เฟรม (สูงสุด {d['H_cm'].max():.0f} ซม.) | มุมหันมัธยฐานช่วงนั้น "
              f"{bad['orientation_deg'].median():.0f}° | H มัธยฐานทั้งคลิป {d['H_cm'].median():.0f}")
