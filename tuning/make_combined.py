"""Build G14_LiftRisk_Realtime.ipynb: the teacher's weight-limit geometry + our Workshop12 model.

Code from the teacher's notebooks is copied verbatim (minus their demo/plot blocks) so the group
version stays recognisable; the new parts are the streaming features for our model, the lift
counter wired to it, the merged pipeline and the per-lift report.
"""
import re
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

WLIMIT_NB = Path(r'D:/_Downloads/Workshop_LiftingWeightLimit_Realtime_Colab.ipynb')
COUNTER_DUMP = Path(__file__).parent / 'liftcounter_dump.txt'
DST = Path(sys.argv[1])

wl = nbformat.read(WLIMIT_NB, as_version=4)


def wl_cell(index, cut=None, start=None):
    source = wl.cells[index].source
    if start is not None:
        source = source[source.index(start):]
    if cut is not None:
        source = source[:source.index(cut)]
    return source.rstrip()


dump = COUNTER_DUMP.read_text(encoding='utf-8')
parts = re.split(r'^--- \[(\d+)\] (\w+) outputs=\d+$', dump, flags=re.M)
counter_cells = {int(parts[i]): parts[i + 2].split('>>> OUT:')[0].rstrip()
                 for i in range(1, len(parts), 3)}


def counter_cell(index, cut=None, start=None):
    source = counter_cells[index]
    if start is not None:
        source = source[source.index(start):]
    if cut is not None:
        source = source[:source.index(cut)]
    return source.strip('\n').rstrip()


md, code = new_markdown_cell, new_code_cell
cells = []

cells.append(md(r'''# 🏋️ G14 · วิเคราะห์ความเสี่ยงการยกแบบครบวงจร (มุมบิด + ระยะเอื้อม + นับรอบยก)

**รายวิชา CP413705 การฝึกปฏิบัติงานปัญญาประดิษฐ์ 3 (AI Workshop III)** · กลุ่ม G14

Notebook นี้รวมงานสามส่วนที่เคยแยกกันให้อยู่ในลูปเดียว ทำงานทีละเฟรมแบบที่ระบบจริงต้องทำ

| ส่วน | มาจาก | ได้อะไร |
|---|---|---|
| โมเดลเชิงเรขาคณิต | `Workshop_LiftingWeightLimit_Realtime_Colab` ของอาจารย์ | มุมหันเทียบกล้อง · **มุมบิดตัว** · **ระยะเอื้อม H / ความสูงมือ V** · ตารางน้ำหนักสูงสุด 3×4 |
| โมเดลที่เราเทรนเอง | `G14_Workshop12` (Transformer, 166 features, window 60) | กิจกรรมรายเฟรม 5 คลาส → **นับรอบการยก** |
| รายงานผล | ส่วนที่เขียนเพิ่มใน notebook นี้ | ตารางสรุป **รายรอบการยก** + ท่าที่แย่ที่สุดของแต่ละรอบ |

```
 วิดีโอ ─► ① อ่านเฟรม ─► ② keypoint ─► ③ ลดการสั่น ─┬─► ⑤ สเกล + พิกัด 3 มิติ ─► ⑥ มุมหัน ─► ⑦ มุมบิด ─► ⑧ H/V + ตารางน้ำหนัก ─┐
                                                   │                                                                      ├─► ⑪ วิดีโอ + CSV + สรุปรายรอบ
                                                   └─► ⑨ features 166 ค่า ─► Transformer (60 เฟรม) ─► ⑩ LIFT/NO LIFT + นับรอบ ─┘
```

### สิ่งที่ต้องเตรียม
| รายการ | หมายเหตุ |
|---|---|
| วิดีโอการยก `.mp4` | กล้องตั้งนิ่ง · ผู้ยก **ยืนตรงนิ่ง 2 วินาทีแรก** (ใช้ calibrate) |
| ส่วนสูงจริงของผู้ยก (ซม.) | กรอกในขั้นที่ 0.2 — มีผลโดยตรงกับค่า H/V เป็นเซนติเมตร |
| โฟลเดอร์โมเดล Workshop12 | `best_model.pt` · `features.csv` · `class_map.csv` · `scaler.csv` · `settings.csv` |
| `pose_landmarker_lite.task` | ดาวน์โหลดอัตโนมัติถ้าไม่มี |

> ⚠️ ต้องใช้ **OpenCV 4.x** — บน OpenCV 5 ฟังก์ชัน `cv2.putText` ไม่รับภาพ float32 แล้ว ตอนวาดตารางจะ error
> (Colab ใช้ 4.14 จึงไม่มีปัญหา ถ้ารันบนเครื่องตัวเองให้ `pip install "opencv-contrib-python<5"`)'''))

cells.append(md('---\n## ขั้นที่ 0 · เตรียมสภาพแวดล้อม\n\n### 0.1 ติดตั้งและนำเข้าไลบรารี'))
cells.append(code(r'''import subprocess, sys

for module, package in [("mediapipe", "mediapipe"), ("torch", "torch")]:
    try:
        __import__(module)
    except ImportError:
        print(f"กำลังติดตั้ง {package} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

import math, re, time, shutil, base64, urllib.request, warnings
from pathlib import Path
from collections import deque, Counter

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mediapipe as mp
import torch
from torch import nn
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from IPython.display import HTML, display

warnings.filterwarnings("ignore", category=RuntimeWarning)
print("OpenCV", cv2.__version__, "| MediaPipe", mp.__version__, "| PyTorch", torch.__version__)
if int(cv2.__version__.split(".")[0]) >= 5:
    print("⚠️ OpenCV 5 ขึ้นไปจะ error ตอนวาดตาราง — ให้ลดเป็น 4.x")'''))

cells.append(md(r'''### 0.2 ตั้งค่า 👉 MODIFY
- **บน Colab** วางไฟล์ไว้ที่ `/content/lift_demo` (อัปโหลดในขั้น 0.3) · **บนเครื่องตัวเอง** ใช้โฟลเดอร์ที่ notebook วางอยู่
- `SUBJECT_HEIGHT_CM` คือส่วนสูงจริงของผู้ยกในคลิป ค่านี้กำหนดสเกลเป็นเซนติเมตรของทั้งระบบ
- ค่าพารามิเตอร์ของกลุ่มอยู่ในขั้นที่ ⑤ (ตั้งจากการทดลองในหัวข้อ "ปรับค่าตั้งต้น" ท้าย notebook)'''))
cells.append(code(r'''IN_COLAB = "google.colab" in sys.modules
BASE = Path("/content/lift_demo") if IN_COLAB else Path.cwd()   # 👉 MODIFY — โฟลเดอร์ทำงาน
BASE.mkdir(parents=True, exist_ok=True)

VIDEO_NAME        = "VID_long.mp4"   # 👉 MODIFY — ชื่อไฟล์วิดีโอ
SUBJECT_HEIGHT_CM = 167.0                             # 👉 MODIFY — ส่วนสูงจริงของผู้ยก (ซม.)
MAX_FRAMES        = None                              # 👉 MODIFY — เช่น 150 เพื่อทดลองเร็ว ๆ · None = ทั้งคลิป
MODEL_DIR         = BASE / "model"                    # 👉 MODIFY — โฟลเดอร์โมเดล Workshop12 ของกลุ่ม
CKPT_NAME         = "best_model.pt"

VIDEO_PATH      = BASE / VIDEO_NAME
POSE_MODEL_PATH = BASE / "models" / "pose_landmarker_lite.task"
OUTPUT_DIR      = BASE / "liftrisk_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_FILES = [CKPT_NAME, "features.csv", "class_map.csv", "scaler.csv", "settings.csv"]'''))

cells.append(md('### 0.3 เตรียมไฟล์\nโมเดล MediaPipe ดาวน์โหลดอัตโนมัติ · บน Colab จะขึ้นปุ่มให้อัปโหลดวิดีโอและไฟล์โมเดลที่ยังไม่มี'))
cells.append(code(r'''POSE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                  "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")

if not POSE_MODEL_PATH.is_file():
    POSE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("กำลังดาวน์โหลดโมเดล MediaPipe ...")
    urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)


def upload_into(folder, what):
    from google.colab import files
    folder.mkdir(parents=True, exist_ok=True)
    print(f"📤 เลือกไฟล์ {what}")
    for name, data in files.upload().items():
        (folder / name).write_bytes(data)


missing_model = [f for f in MODEL_FILES if not (MODEL_DIR / f).is_file()]
if IN_COLAB and not VIDEO_PATH.is_file():
    upload_into(BASE, f"วิดีโอ ({VIDEO_NAME})")
if IN_COLAB and missing_model:
    upload_into(MODEL_DIR, "โมเดล Workshop12: " + ", ".join(missing_model))

for p in [VIDEO_PATH, POSE_MODEL_PATH] + [MODEL_DIR / f for f in MODEL_FILES]:
    print("✅" if p.is_file() else "❌ ไม่พบ", p)
assert VIDEO_PATH.is_file(), "ไม่พบวิดีโอ — แก้ VIDEO_NAME ในขั้น 0.2"
assert all((MODEL_DIR / f).is_file() for f in MODEL_FILES), "ไฟล์โมเดลยังไม่ครบ"'''))

# ---------------------------------------------------------------- steps 1-4 (teacher's code)
cells.append(md(wl.cells[7].source))
cells.append(code(wl_cell(8, cut='cap, VIDEO_INFO = open_video') + '''

cap, VIDEO_INFO = open_video(VIDEO_PATH)
cap.release()
FPS, WIDTH, HEIGHT = VIDEO_INFO["fps"], VIDEO_INFO["width"], VIDEO_INFO["height"]
print(VIDEO_INFO, f"→ ยาว {VIDEO_INFO['frames'] / FPS:.1f} วินาที")'''))

cells.append(md(wl.cells[9].source))
cells.append(code(wl_cell(10, cut='# ทดลองกับเฟรมแรก')))

cells.append(md(wl.cells[11].source))
cells.append(code(wl_cell(12)))

cells.append(md(wl.cells[15].source))
cells.append(code(wl_cell(16, cut='k = len(demo["frame"])')))

# ---------------------------------------------------------------- step 5 geometry
cells.append(md(wl.cells[17].source))
cells.append(code(wl_cell(18, cut='CALIB_FRAMES = int(round(')))
cells.append(code(wl_cell(20, start='def draw_body_frame', cut='k = CALIB_FRAMES - 1')))

cells.append(md(wl.cells[21].source))
cells.append(code(wl_cell(22, cut='orient_demo = [')))

cells.append(md(wl.cells[24].source))
cells.append(code(wl_cell(25, cut='twist_demo = [')))
cells.append(md(r'''> 👉 **ค่าที่กลุ่มปรับ:** มุมบิดของ demo วัดแนวไหล่เทียบแนวสะโพก และเตือนเมื่อเกิน 45°
> เมื่อลองกับ 35 คลิปที่มีการยก (กลุ่มเรา 16 คลิป + กลุ่มอื่น 19 คลิป) พบว่าคนบิดตัวสองแบบ
> - **บิดเฉพาะช่วงบน** สะโพกอยู่กับที่ → มุมไหล่เทียบสะโพกสูง (คลิปของกลุ่มเรา)
> - **หมุนทั้งลำตัว** สะโพกหมุนตามไหล่ → มุมไหล่เทียบสะโพกต่ำ (9–18°) แต่แนวไหล่หมุนไปมากเมื่อเทียบกับพื้น (คลิปของกลุ่มอื่น)
>
> จึงเตือนบิดตัว **เป็นรายรอบ** เมื่อเข้าข้อใดข้อหนึ่ง
> 1. มุมไหล่เทียบสะโพกสูงสุดในรอบ (กรองมัธยฐาน 5 เฟรม) เกิน **17°**
> 2. แนวไหล่แกว่งเทียบกับพื้นภายในรอบ (เปอร์เซ็นไทล์ 5–95) เกิน **50°**
>
> ผล: แยกคลิป T0/T1 ถูก **29 จาก 35** · จับคลิปบิดตัวได้ **11 จาก 12** · เตือนผิด 5 จาก 23 คลิปที่ไม่บิด (เกณฑ์เดิม 27° จับได้ 4 จาก 13)
> ค่าทั้งสองปรับจาก 35 คลิปนี้เอง เมื่อทดสอบแบบกันผู้ยกออกทีละคนได้ 27 จาก 35 (ดูขั้น ⑭)'''))
cells.append(code(r'''TWIST_WARN_DEG = 17.0       # 👉 MODIFY — มุมไหล่เทียบสะโพกสูงสุดในรอบ (ค่าตั้งต้นของ demo คือ 45)
SWING_WARN_DEG = 50.0       # 👉 MODIFY — แนวไหล่แกว่งเทียบกับพื้นภายในหนึ่งรอบ
TWIST_SMOOTH_FRAMES = 5     # กรองมัธยฐานก่อนเทียบเกณฑ์ กันค่ากระโดดเฟรมเดียว


def shoulder_direction(C, up):
    # ทิศของแนวไหล่บนพื้น (องศา) ในกรอบที่ยึดกับพื้น — กล้องไม่ขยับ จึงเทียบข้ามเฟรมได้
    e1 = unit(np.cross(up, [0.0, 0.0, 1.0]))
    e2 = np.cross(up, e1)
    line = C[12] - C[11]
    return math.degrees(math.atan2(line @ e2, line @ e1))


def swing_range(angles_deg):
    # ทิศแกว่งกว้างแค่ไหน: ช่วงเปอร์เซ็นไทล์ 5–95 รอบค่าเฉลี่ยเชิงวงกลม
    smooth = pd.Series(angles_deg, dtype=float).rolling(TWIST_SMOOTH_FRAMES, center=True, min_periods=1).median()
    a = np.radians(smooth.dropna().to_numpy())
    if len(a) < 5:
        return float("nan")
    mean = math.atan2(np.sin(a).mean(), np.cos(a).mean())
    spread = np.degrees((a - mean + np.pi) % (2 * np.pi) - np.pi)
    return float(np.percentile(spread, 95) - np.percentile(spread, 5))'''))

cells.append(md(wl.cells[26].source))
cells.append(code(wl_cell(27, cut='measures = [measure_hands')))

cells.append(md(wl.cells[28].source))
cells.append(code(wl_cell(29, cut='# ลำดับการวาด')))

# ---------------------------------------------------------------- step 9: our Workshop12 model
W12_NB = Path(sys.argv[2]) if len(sys.argv) > 2 else None
w12 = nbformat.read(W12_NB, as_version=4)
model_classes = next(c.source for c in w12.cells if c.cell_type == 'code' and 'class ClassifierHead' in c.source)

cells.append(md(r'''---
## ขั้นที่ ⑨ · กิจกรรมรายเฟรมจากโมเดล Workshop12 ของกลุ่ม

โมเดลที่เลือกไว้ใน Workshop12 คือ **Transformer** ที่รับ **60 เฟรมล่าสุด × 166 features** แล้วทำนายกิจกรรมของ **เฟรมสุดท้าย** เป็น 5 คลาส
(`idle`, `reach/grasp`, `lift`, `lower/place`, `carry/hold`)

### 9.1 features ต้องสร้างให้เหมือนตอนเทรน **ทุกประการ**
Workshop11 สร้าง features จากพิกัดที่ผ่าน Kalman + moving average มาแล้ว (ขั้นที่ ③ ของ notebook นี้) ประกอบด้วย

| กลุ่ม | จำนวน | วิธีคำนวณ |
|---|---|---|
| base | 132 | 33 จุด × (x, y, confidence, valid) · x/y ลบกึ่งกลางสะโพกแล้วหารความยาวลำตัว |
| velocity | 20 | 10 จุดสำคัญ × (vx, vy) = ระยะที่เคลื่อนจากเฟรมก่อนหน้า หารความยาวลำตัว |
| angle | 14 | มุมศอก/สะโพก/เข่า ซ้าย-ขวา + มุมลำตัว และความเร็วเชิงมุมของทั้ง 7 ค่า |

ชื่อและลำดับคอลัมน์อ่านจาก `features.csv` · ค่าเฉลี่ย/ส่วนเบี่ยงเบนอ่านจาก `scaler.csv` (ของชุด train เท่านั้น)
**ห้ามคำนวณ mean/std ใหม่จากคลิปที่กำลังวิเคราะห์**

คลาส `FeatureStream` ด้านล่างทำงานแบบทีละเฟรม โดยจำพิกัดและมุมของเฟรมก่อนหน้าไว้คำนวณความเร็ว — ใช้เฉพาะอดีต จึงใช้กับงานเรียลไทม์ได้'''))
cells.append(code(r'''FEATURE_TABLE = pd.read_csv(MODEL_DIR / "features.csv").sort_values("order")
FEATURE_COLUMNS = FEATURE_TABLE["feature"].tolist()
CLASS_NAMES = (pd.read_csv(MODEL_DIR / "class_map.csv").sort_values("label_id")["label"]
               .str.replace("\\/", "/", regex=False).tolist())
W11_SETTINGS = pd.read_csv(MODEL_DIR / "settings.csv", dtype=str).set_index("setting")["value"]

VELOCITY_POINTS = [int(p) for p in W11_SETTINGS["velocity_points"].split()]
JOINT_ANGLES = {}
for part in W11_SETTINGS["angles"].split("|"):
    name, _, points = part.strip().partition(":")
    if points != "vertical":
        JOINT_ANGLES[name] = tuple(int(p) for p in points.split("-"))
ANGLE_NAMES = list(JOINT_ANGLES) + ["trunk"]


def load_scaler(model_dir, feature_columns):
    # mean/std ของชุด train เรียงตาม feature_columns (คอลัมน์ที่ไม่ scale ใช้ mean=0, std=1)
    table = pd.read_csv(model_dir / "scaler.csv").set_index("feature").reindex(feature_columns)
    scaled = table["scaled"].astype(str).str.lower().eq("true").to_numpy()
    mean = np.where(scaled, table["mean"].to_numpy(float), 0.0)
    std = np.where(scaled, table["std"].to_numpy(float), 1.0)
    return mean, std


def joint_angle_px(xy, a, b, c):
    # มุมที่จุด b ระหว่างเส้น b→a และ b→c หน่วยองศา (คำนวณบนพิกัดพิกเซลเหมือน Workshop11)
    v1, v2 = xy[a] - xy[b], xy[c] - xy[b]
    cosine = float(v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return math.degrees(math.acos(min(1.0, max(-1.0, cosine))))


class FeatureStream:
    # ⑨ สร้าง features 166 ค่าของหนึ่งเฟรม ให้ตรงกับ Workshop11

    def __init__(self, mean, std):
        self.mean, self.std = mean, std
        self.previous_xy = None
        self.previous_angles = None

    def update(self, xy_s, observed, conf):
        values = {}
        hip = np.nanmean(xy_s[[23, 24]], axis=0)
        shoulder = np.nanmean(xy_s[[11, 12]], axis=0)
        torso = float(np.linalg.norm(shoulder - hip))
        anchor_ok = np.isfinite(torso) and torso > 5

        for p in range(33):                                          # base 132
            valid = anchor_ok and bool(np.isfinite(xy_s[p]).all())
            prefix = f"kp{p:02d}"
            values[prefix + "_x"] = (xy_s[p, 0] - hip[0]) / torso if valid else np.nan
            values[prefix + "_y"] = (xy_s[p, 1] - hip[1]) / torso if valid else np.nan
            values[prefix + "_confidence"] = float(np.clip(np.nan_to_num(conf[p]), 0, 1)) if observed[p] else 0.0
            values[prefix + "_valid"] = float(valid)

        for p in VELOCITY_POINTS:                                    # velocity 20
            prefix = f"kp{p:02d}"
            if self.previous_xy is None or not anchor_ok:
                values[prefix + "_vx"] = values[prefix + "_vy"] = np.nan
            else:
                change = xy_s[p] - self.previous_xy[p]
                values[prefix + "_vx"] = change[0] / torso
                values[prefix + "_vy"] = change[1] / torso

        angles = {name: joint_angle_px(xy_s, *points) for name, points in JOINT_ANGLES.items()}
        trunk = shoulder - hip
        angles["trunk"] = math.degrees(math.atan2(abs(trunk[0]), -trunk[1]))
        for name in ANGLE_NAMES:                                     # angle 14
            values[name + "_angle"] = angles[name]
            previous = self.previous_angles[name] if self.previous_angles else np.nan
            values[name + "_angle_vel"] = angles[name] - previous

        self.previous_xy, self.previous_angles = xy_s.copy(), angles
        raw = np.array([values[column] for column in FEATURE_COLUMNS], dtype=float)
        return np.nan_to_num((raw - self.mean) / self.std, nan=0.0).astype(np.float32)   # NaN → 0 เหมือนตอนเทรน


SCALE_MEAN, SCALE_STD = load_scaler(MODEL_DIR, FEATURE_COLUMNS)
print("features:", len(FEATURE_COLUMNS), "| คลาส:", CLASS_NAMES)
print("จุดที่คิดความเร็ว:", VELOCITY_POINTS)
print("มุมข้อต่อ:", JOINT_ANGLES)'''))

cells.append(md(r'''### 9.2 โหลดโมเดลที่เทรนไว้
โครงสร้างโมเดลคัดลอกมาจาก `G14_Workshop12` ตรง ๆ และ `build_model()` สร้างโมเดลจาก `config` ที่เก็บไว้ใน checkpoint
จึงโหลดได้ทั้ง LSTM, GRU และ Transformer โดยไม่ต้องแก้โค้ด'''))
cells.append(code(model_classes))
cells.append(code(r'''MODEL, CHECKPOINT = load_trained_model(MODEL_DIR / CKPT_NAME, torch.device("cpu"))
CONFIG = CHECKPOINT["config"]
WINDOW = int(CONFIG["window"])
if CHECKPOINT["feature_columns"] != FEATURE_COLUMNS:
    raise ValueError("ลำดับ features ใน checkpoint ไม่ตรงกับ features.csv")
if CHECKPOINT["class_names"] != CLASS_NAMES:
    raise ValueError("ชื่อคลาสใน checkpoint ไม่ตรงกับ class_map.csv")


@torch.no_grad()
def predict_window(model, window_features):
    # ⑨ ทำนายจากหน้าต่าง (WINDOW, 166) → ความน่าจะเป็นของแต่ละคลาส
    x = torch.from_numpy(np.asarray(window_features, dtype=np.float32)).unsqueeze(0)
    return torch.softmax(model(x), dim=1)[0].numpy()


print("โมเดล:", CHECKPOINT["experiment"], "|", CONFIG["model"], "| window", WINDOW, "เฟรม",
      "| features", CONFIG["input_size"], "| epoch", CHECKPOINT["best_epoch"])
print("warm-up:", WINDOW - 1, "เฟรมแรกยังทำนายไม่ได้ (", (WINDOW - 1) / FPS, "วินาที )")'''))

# ---------------------------------------------------------------- step 10: lift counting
cells.append(md(r"""---
## ขั้นที่ ⑩ · แปลงผลทำนายเป็นสถานะ LIFT และนับรอบการยก

ผลรายเฟรมสลับไปมาได้ จึงใช้ **majority vote** จาก `SMOOTH_FRAMES` เฟรมล่าสุด (ย้อนหลังเท่านั้น)
แล้วถือว่าเป็น **LIFT** เมื่อคลาสอยู่ใน `LIFT_CLASSES`

เซลล์แรกคือ `MajorityVote`, `LiftCounter` และ `draw_counter` จาก demo ของอาจารย์ (เก็บไว้เทียบ)
เซลล์ถัดไปคือ **ตัวนับที่กลุ่มปรับ** หลังทดลองกับคลิปของกลุ่มทั้ง 20 คลิป"""))
cells.append(code('LIFT_CLASSES  = {"lift"}    # ค่าตั้งต้นของ demo (เซลล์ถัดไปจะตั้งค่าของกลุ่มทับ)\n'
                  + counter_cell(29, start='SMOOTH_FRAMES', cut='def draw_prediction') + '\n\n'
                  + counter_cell(32, cut='# ทดสอบกับลำดับจำลอง')))
cells.append(md(r"""### 10.1 ตัวนับรอบของกลุ่ม: ใช้ความสูงมือเป็นหลัก ให้โมเดลยืนยัน 👉 MODIFY

ตัวนับของ demo นับจากป้ายของโมเดลอย่างเดียว เมื่อทดสอบกับคลิปยกหลายรอบที่ถ่ายใหม่พบว่า
- คลิปยกปกติ 8 รอบ นับได้ 10 (ป้ายกะพริบหลังวางของ)
- คลิปยกเร็ว 7 รอบ นับได้ 4 (โมเดลแทบไม่ตอบว่ากำลังยกในรอบที่ยกเร็ว)

แต่ **ความสูงมือ V จากขั้นที่ ⑧ เป็นสัญญาณที่สะอาดมาก** ทุกรอบเห็นเป็นลูกคลื่นชัดเจน เราจึงเปลี่ยนเป็น

1. จำ **ระดับมือขณะยืนนิ่ง** `V0` (มัธยฐานของ 15 เฟรมแรกที่วัดได้หลัง calibrate)
2. **เริ่มรอบ** เมื่อมือห่างจาก `V0` เกิน `HEIGHT_HI_CM` ติดกัน `HEIGHT_HOLD_FRAMES` เฟรม (ยกขึ้นหรือก้มลงก็นับ)
3. **จบรอบ** เมื่อมือกลับมาใกล้ `V0` ไม่เกิน `HEIGHT_LO_CM` ติดกัน `HEIGHT_HOLD_FRAMES` เฟรม
4. **นับเป็นรอบการยก** ก็ต่อเมื่อช่วงนั้นยาวอย่างน้อย `MIN_ROUND_FRAMES` เฟรม และโมเดลตอบว่ามีของในมือ
   (`LIFT_CLASSES`) อย่างน้อย `HANDLING_FRACTION` ของช่วง — ข้อนี้กันไม่ให้นับการเดินหรือการก้มตัวเปล่า ๆ

ผลกับ 22 คลิป (20 คลิปเดิมของกลุ่ม + 2 คลิปยกหลายรอบ): นับถูก **22 จาก 22**
> ⚠️ ค่าเหล่านี้ปรับจากทั้ง 22 คลิปรวมคลิปทดสอบด้วย จึงยังไม่มีคลิปที่ระบบ "ไม่เคยเห็น" มายืนยัน
> ตัวนับนับเมื่อ **จบรอบ** (มือกลับมาที่ระดับเดิม) ตัวเลขบนจอจึงขึ้นช้ากว่าการยกจริงเล็กน้อย"""))
cells.append(code(r"""LIFT_CLASSES       = {"lift", "carry/hold", "lower/place"}   # 👉 MODIFY — คลาสที่ถือว่ามีของอยู่ในมือ
SMOOTH_FRAMES      = 9       # 👉 MODIFY — จำนวนเฟรมที่ใช้โหวตป้ายของโมเดล
HEIGHT_HI_CM       = 12.0    # 👉 MODIFY — มือห่างจากระดับยืนนิ่งเกินเท่านี้ = เริ่มรอบ
HEIGHT_LO_CM       = 6.0     # 👉 MODIFY — มือกลับมาใกล้ระดับยืนนิ่งไม่เกินเท่านี้ = จบรอบ
HEIGHT_HOLD_FRAMES = 8       # 👉 MODIFY — ต้องเกิน/ต่ำกว่าเกณฑ์ติดกันกี่เฟรม
MIN_ROUND_FRAMES   = 20      # 👉 MODIFY — รอบต้องยาวอย่างน้อยกี่เฟรม
HANDLING_FRACTION  = 0.20    # 👉 MODIFY — โมเดลต้องตอบว่ามีของในมืออย่างน้อยสัดส่วนนี้ของรอบ
BASELINE_FRAMES    = 15      # จำนวนเฟรมที่ใช้หาระดับมือขณะยืนนิ่ง


class HeightRoundCounter:
    # ⑩ นับรอบการยกจากความสูงมือ V (ซม.) โดยให้ป้ายของโมเดลยืนยัน ทำงานทีละเฟรม

    def __init__(self):
        self.count = 0
        self.v0 = None                    # ระดับมือขณะยืนนิ่ง
        self.baseline = []
        self.active = False               # กำลังอยู่ในช่วงที่มือออกจากระดับยืนนิ่ง
        self.above = self.below = 0
        self.frame = -1
        self.start = 0
        self.recent = deque(maxlen=HEIGHT_HOLD_FRAMES)
        self.flags = []                   # โมเดลบอกว่ามีของในมือหรือไม่ ของแต่ละเฟรมในช่วงนี้

    def _close(self, end):
        # จบช่วง: คืน (เฟรมเริ่ม, เฟรมจบ) ถ้านับเป็นรอบการยก มิฉะนั้นคืน None
        self.active = False
        flags = self.flags[:end - self.start + 1]
        if len(flags) >= MIN_ROUND_FRAMES and np.mean(flags) >= HANDLING_FRACTION:
            self.count += 1
            return self.start, end
        return None

    def update(self, v_cm, handling):
        # v_cm = ความสูงมือของเฟรมนี้ (None ถ้าวัดไม่ได้) · handling = โมเดลบอกว่ามีของในมือ
        self.frame += 1
        self.recent.append(bool(handling))
        if v_cm is not None and np.isfinite(v_cm):
            if self.v0 is None:
                self.baseline.append(v_cm)
                if len(self.baseline) == BASELINE_FRAMES:
                    self.v0 = float(np.median(self.baseline))
                return None
            far = abs(v_cm - self.v0)
            self.above = self.above + 1 if far > HEIGHT_HI_CM else 0
            self.below = self.below + 1 if far < HEIGHT_LO_CM else 0
        if self.v0 is None:
            return None
        if self.active:
            self.flags.append(bool(handling))
            if self.below >= HEIGHT_HOLD_FRAMES:
                return self._close(self.frame - HEIGHT_HOLD_FRAMES + 1)
        elif self.above >= HEIGHT_HOLD_FRAMES:
            self.active = True
            self.start = self.frame - HEIGHT_HOLD_FRAMES + 1
            self.flags = list(self.recent)
        return None

    def finish(self):
        # เรียกเมื่อวิดีโอจบ เผื่อมีรอบที่ยังไม่ทันกลับมาระดับเดิม
        return self._close(self.frame) if self.active else None


# ทดสอบ: ยืนนิ่ง 85 ซม. → ยกขึ้น 110 ซม. พร้อมของ → กลับ → ก้มมือเปล่า → กลับ  (ควรนับ 1 รอบ)
test = HeightRoundCounter()
heights = [85] * 30 + [110] * 40 + [85] * 30 + [60] * 40 + [85] * 30
holding = [False] * 30 + [True] * 40 + [False] * 100
rounds = [r for v, h in zip(heights, holding) if (r := test.update(v, h)) is not None]
print("นับได้", test.count, "รอบ ที่เฟรม", rounds)
assert test.count == 1"""))

# ---------------------------------------------------------------- step 11: merged pipeline
cells.append(md(r'''---
## ขั้นที่ ⑪ · รวมทุกขั้นในลูปเดียว

หนึ่งรอบของลูป = หนึ่งเฟรม ทำตามลำดับนี้

| ขั้น | ทำอะไร |
|---|---|
| ① ② ③ | อ่านเฟรม → keypoint → Kalman + moving average |
| ⑤ | ช่วง `CALIB_SECONDS` แรกเก็บไว้ calibrate (ขึ้นข้อความ CALIBRATING) จากนั้นหาท่าทางกล้องและพิกัด 3 มิติ |
| ⑥ ⑦ ⑧ | มุมหัน · มุมบิด · H/V + ช่องในตารางน้ำหนัก |
| ⑨ ⑩ | features 166 ค่า → Transformer → majority vote → สถานะ LIFT → นับรอบ |
| ④ + วาด | ตาราง (ล่างสุด) → โครงกระดูก → ป้ายน้ำหนัก → แผงข้อมูล → กล่องนับรอบ |

**ข้อสังเกตจากการทดลอง:** เมื่อผู้ยกหันหน้าเข้ากล้อง (มุมหันต่ำ) ตารางจะเกือบหันสันเข้ากล้องจนอ่านไม่ออก
เราจึงเพิ่ม **ตาราง 2 มิติมุมขวาล่าง** ที่อ่านได้เสมอ และไฮไลต์ช่องปัจจุบันเหมือนกัน'''))
cells.append(code(wl_cell(31, cut='def run_pipeline')))
cells.append(code(r'''MAX_REACH_FRACTION = 0.50   # 👉 MODIFY — H ที่เกินสัดส่วนนี้ของส่วนสูงถือว่าวัดพลาด (แขนเอื้อมไม่ถึง)
MINI_TABLE_WIDTH = 330     # 👉 MODIFY — ความกว้างของตาราง 2 มิติมุมจอ (พิกเซลที่ภาพกว้าง 720)


def draw_mini_table(img, calib, m):
    # ตาราง 3×4 แบบ 2 มิติมุมขวาล่าง อ่านได้แม้ตารางในฉากจะหันสันเข้ากล้อง
    s = ui_scale(img)
    width = int(MINI_TABLE_WIDTH * s)
    cell_w, cell_h = width // 5, int(26 * s)          # ป้ายชื่อแถวกว้าง 2 ช่อง + 3 คอลัมน์
    x0, y0 = img.shape[1] - width - int(10 * s), img.shape[0] - cell_h * 5 - int(10 * s)
    font = cv2.FONT_HERSHEY_SIMPLEX
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + width, y0 + cell_h * 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, dst=img)

    for c, name in enumerate(COL_NAMES):
        cv2.putText(img, name, (x0 + cell_w * (c + 2) + int(4 * s), y0 + int(18 * s)),
                    font, 0.4 * s, (255, 255, 255), 1, cv2.LINE_AA)
    for r, row_name in enumerate(ROW_NAMES):
        y = y0 + cell_h * (r + 1)
        cv2.putText(img, row_name, (x0 + int(4 * s), y + int(18 * s)),
                    font, 0.38 * s, (255, 255, 255), 1, cv2.LINE_AA)
        for c in range(3):
            x = x0 + cell_w * (c + 2)
            kg = WEIGHT_LIMITS_KG[r][c]
            here = m is not None and (r, c) == (m["row"], m["col"])
            cv2.rectangle(img, (x, y), (x + cell_w, y + cell_h),
                          (0, 230, 255) if here else cell_color(kg), -1)
            cv2.rectangle(img, (x, y), (x + cell_w, y + cell_h), (255, 255, 255), 1)
            cv2.putText(img, str(kg), (x + int(12 * s), y + int(18 * s)), font, 0.5 * s,
                        (0, 0, 0) if here else (255, 255, 255), 2 if here else 1, cv2.LINE_AA)
    return img


def run_pipeline(video_path, out_path, height_cm, max_frames=None):
    cap, info = open_video(video_path)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), info["fps"],
                             (info["width"], info["height"]))
    smoother = PoseSmoother(info["width"], info["height"])
    features = FeatureStream(SCALE_MEAN, SCALE_STD)
    buffer = deque(maxlen=WINDOW)
    voter = MajorityVote(SMOOTH_FRAMES)
    counter = HeightRoundCounter()
    twist_recent = deque(maxlen=TWIST_SMOOTH_FRAMES)
    n_calib = int(round(CALIB_SECONDS * info["fps"]))
    calib, samples, pose, records = None, [], None, []
    WHITE, YELLOW, RED, GREEN = (255, 255, 255), (0, 230, 255), (80, 80, 255), (0, 220, 120)
    t0 = time.time()

    with create_pose_detector() as detector:
        i = 0
        while max_frames is None or i < max_frames:
            ok, frame = cap.read()                                             # ①
            if not ok:
                break
            xy, world, conf = extract_keypoints(detector, frame, i)            # ②
            xy_s, world_s, observed = smoother.update(xy, world, conf)         # ③
            img = frame.copy()
            row = {"frame_number": i + 1, "time_s": (i + 1) / info["fps"]}

            buffer.append(features.update(xy_s, observed, conf))               # ⑨ features ทุกเฟรม
            label, probs = None, None
            if len(buffer) == WINDOW:
                probs = predict_window(MODEL, np.stack(buffer))                # ⑨ โมเดล
                raw_label = CLASS_NAMES[int(probs.argmax())]
                label = voter.update(raw_label)                                # ⑩ majority vote
                row["raw_label"] = raw_label
                row["label"] = label
                row.update({f"p_{name}": float(p) for name, p in zip(CLASS_NAMES, probs)})
            is_lift = label in LIFT_CLASSES
            row["is_lift"] = is_lift
            row["round"] = 0
            hand_height = None

            if calib is None:                                                  # ⑤ ช่วงยืนนิ่ง
                samples.append({"xy": xy_s, "world": world_s, "observed": observed})
                if len(samples) == n_calib:
                    calib = calibrate(samples, height_cm, info["width"], info["height"])
                img = draw_skeleton(img, xy_s, observed)
                img = draw_info_panel(img, [(f"CALIBRATING... stand still {len(samples)}/{n_calib}", YELLOW)])
            else:
                P = world_s * 100 * calib["world_scale"]                       # ⑤ พิกัด 3 มิติ
                pose = solve_camera_pose(P, xy_s, observed, calib["K"], pose)
                m = None
                if pose is not None:
                    C = to_camera_3d(xy_s, P, pose, calib["K"])
                    row["orientation_deg"] = orientation_angle(C)              # ⑥
                    row["shoulder_dir_deg"] = shoulder_direction(C, calib["up"])   # ⑦ ทิศแนวไหล่เทียบพื้น
                    m = measure_hands(C, xy_s, calib)                          # ⑧
                    if m is not None and m["H"] > MAX_REACH_FRACTION * height_cm:
                        m = None                                               # ไกลเกินช่วงแขน → ไม่น่าเชื่อถือ
                row["twist_deg"] = twist_from_baseline(world_s, calib)         # ⑦
                twist_recent.append(row["twist_deg"])
                row["twist_smooth_deg"] = float(np.nanmedian(twist_recent))
                hand_height = m["V"] if m is not None else None

                if m is not None:
                    img = draw_weight_table(img, calib, m)                     # ⑧ ตารางในฉาก
                img = draw_skeleton(img, xy_s, observed)                       # ④
                lines = [(f"scale   {calib['cm_per_px']:.3f} cm/px", WHITE),
                         (f"orient  {row.get('orientation_deg', float('nan')):.0f} deg to camera", WHITE),
                         (f"twist   {row['twist_smooth_deg']:.0f} deg",
                          RED if row["twist_smooth_deg"] > TWIST_WARN_DEG else WHITE)]
                if m is not None:
                    img = draw_hand_label(img, calib, m)                       # ⑧ ป้ายน้ำหนัก
                    lines += [(f"hands   H {m['H']:.0f} cm | V {m['V']:.0f} cm", WHITE),
                              (f"zone    {ROW_NAMES[m['row']]} | {COL_NAMES[m['col']]}", WHITE),
                              (f"LIMIT   {m['limit_kg']} kg", YELLOW)]
                    row.update({"H_cm": m["H"], "V_cm": m["V"], "zone": ROW_NAMES[m["row"]],
                                "reach_class": COL_NAMES[m["col"]], "limit_kg": m["limit_kg"]})
                lines.append((f"model   {label if label else 'warming up...'}",
                              GREEN if is_lift else WHITE))
                img = draw_info_panel(img, lines)
                img = draw_mini_table(img, calib, m)
            finished = counter.update(hand_height, is_lift)                    # ⑩ นับรอบจากความสูงมือ
            if finished is not None:                                           # รอบเพิ่งจบ → ใส่เลขรอบย้อนหลัง
                for past in records[finished[0]:finished[1] + 1]:
                    past["round"] = counter.count
            row["lift_count"] = counter.count
            img = draw_counter(img, counter.count, finished is not None)       # ⑩ กล่องนับรอบ
            writer.write(img)                                                  # ⑪
            records.append(row)
            i += 1
            if i % 50 == 0:
                print(f"  {i} เฟรม · ยกไปแล้ว {counter.count} ครั้ง · {i / (time.time() - t0):.1f} เฟรม/วินาที")

    finished = counter.finish()                                                # รอบที่ค้างตอนวิดีโอจบ
    if finished is not None:
        for past in records[finished[0]:finished[1] + 1]:
            past["round"] = counter.count
    cap.release()
    writer.release()
    print(f"เสร็จ {i} เฟรม ใน {time.time() - t0:.1f} วินาที · นับการยกได้ {counter.count} ครั้ง")
    return pd.DataFrame(records), calib'''))

cells.append(md('### 11.1 รันกับคลิปจริง'))
cells.append(code(r'''stem = Path(VIDEO_NAME).stem
RAW_OUT = OUTPUT_DIR / f"{stem}_liftrisk_mp4v.mp4"
results, CALIB_RUN = run_pipeline(VIDEO_PATH, RAW_OUT, SUBJECT_HEIGHT_CM, max_frames=MAX_FRAMES)
results.to_csv(OUTPUT_DIR / f"{stem}_liftrisk.csv", index=False)
display(results.dropna(subset=["limit_kg"]).iloc[::30][
    ["frame_number", "time_s", "label", "is_lift", "lift_count", "twist_deg", "twist_smooth_deg", "shoulder_dir_deg", "H_cm", "V_cm",
     "zone", "reach_class", "limit_kg"]].round(1).head(12))'''))

cells.append(md('### 11.2 แปลงเป็น H.264 และแสดงผล\n`cv2.VideoWriter` ใช้ codec `mp4v` ที่เบราว์เซอร์เล่นไม่ได้ จึงแปลงด้วย `ffmpeg` ถ้ามี'))
cells.append(code(r'''FINAL_OUT = OUTPUT_DIR / f"{stem}_liftrisk.mp4"
PREVIEW = OUTPUT_DIR / f"{stem}_liftrisk_preview.mp4"

if shutil.which("ffmpeg"):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(RAW_OUT),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(FINAL_OUT)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(FINAL_OUT), "-vf", "scale=360:-2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "28", str(PREVIEW)], check=True)
    RAW_OUT.unlink()
    print("บันทึกวิดีโอ:", FINAL_OUT, f"({FINAL_OUT.stat().st_size / 1e6:.1f} MB)")
    display(HTML(f'<video width="360" controls><source src="data:video/mp4;base64,'
                 f'{base64.b64encode(PREVIEW.read_bytes()).decode()}" type="video/mp4"></video>'))
else:
    FINAL_OUT = RAW_OUT
    print("ไม่พบ ffmpeg — ได้ไฟล์ mp4v:", FINAL_OUT, "(เปิดด้วย VLC)")

if IN_COLAB:
    from google.colab import files
    files.download(str(FINAL_OUT))
    files.download(str(OUTPUT_DIR / f"{stem}_liftrisk.csv"))'''))

cells.append(code(r'''# ภาพนิ่ง 3 จังหวะ: ระหว่าง calibrate · เฟรมที่มือยื่นไกลที่สุด · เฟรมที่น้ำหนักที่ยอมรับได้ต่ำที่สุด
valid = results.dropna(subset=["limit_kg"])
frame_ids = [int(round(CALIB_SECONDS * FPS)) // 2,
             int(valid.loc[valid["H_cm"].idxmax(), "frame_number"]),
             int(valid.loc[valid["limit_kg"].idxmin(), "frame_number"])]
show_images([read_frame(FINAL_OUT, f) for f in frame_ids],
            [f"frame {f}" for f in frame_ids])'''))

# ---------------------------------------------------------------- step 12: per-lift report
cells.append(md(r'''---
## ขั้นที่ ⑫ · สรุปผลรายรอบการยก

คำถามที่อาจารย์ทิ้งไว้ท้าย notebook demo คือ *"ถ้ารวมสองระบบเข้าด้วยกัน ควรรายงานน้ำหนักสูงสุด ณ เฟรมใดของการยกหนึ่งครั้ง"*

เราตอบว่า **เฟรมที่น้ำหนักที่ยอมรับได้ต่ำที่สุดในรอบนั้น** เพราะเป็นท่าที่เสี่ยงที่สุดของรอบ
ถ้าของที่ยกหนักเกินค่านี้ แปลว่ามีช่วงหนึ่งของการยกที่เกินเกณฑ์แล้ว แม้ท่าเริ่มต้นจะดูปลอดภัย

ตารางด้านล่างให้หนึ่งแถวต่อหนึ่งรอบการยก พร้อมท่าที่แย่ที่สุดและมุมบิดสูงสุดของรอบนั้น'''))
cells.append(code(r'''def lift_report(results):
    # ⑫ หนึ่งแถวต่อหนึ่งรอบการยก (ใช้หมายเลขรอบจาก HeightRoundCounter)
    report = []
    for number in sorted(n for n in results["round"].unique() if n > 0):
        part = results[results["round"] == number]
        measured = part.dropna(subset=["limit_kg"])
        if measured.empty:
            continue
        worst = measured.loc[measured["limit_kg"].idxmin()]
        report.append({
            "lift": number,
            "start_s": round(part["time_s"].iloc[0], 2),
            "end_s": round(part["time_s"].iloc[-1], 2),
            "duration_s": round(part["time_s"].iloc[-1] - part["time_s"].iloc[0], 2),
            "V_start_cm": round(measured["V_cm"].iloc[0], 1),
            "V_end_cm": round(measured["V_cm"].iloc[-1], 1),
            "V_min_cm": round(measured["V_cm"].min(), 1),
            "V_max_cm": round(measured["V_cm"].max(), 1),
            "H_max_cm": round(measured["H_cm"].max(), 1),
            "twist_max_deg": round(part["twist_smooth_deg"].max(), 1),
            "shoulder_swing_deg": round(swing_range(part["shoulder_dir_deg"]), 1),
            "worst_frame": int(worst["frame_number"]),
            "worst_zone": f'{worst["zone"]} | {worst["reach_class"]}',
            "limit_kg": int(worst["limit_kg"]),
        })
    report = pd.DataFrame(report)
    if len(report):                                    # ⑦ เตือนบิดตัวเมื่อเข้าข้อใดข้อหนึ่ง
        report["twist_flag"] = ((report["twist_max_deg"] > TWIST_WARN_DEG)
                                | (report["shoulder_swing_deg"] > SWING_WARN_DEG))
    return report


lifts = lift_report(results)
lifts.to_csv(OUTPUT_DIR / f"{stem}_lifts.csv", index=False)
print(f"นับรอบการยกได้ {len(lifts)} รอบ")
display(lifts)
if len(lifts):
    worst = lifts.loc[lifts["limit_kg"].idxmin()]
    print(f"รอบที่เสี่ยงที่สุด: รอบ {worst['lift']} ที่เวลา {worst['start_s']}–{worst['end_s']} วินาที "
          f"· ท่าแย่สุดที่เฟรม {worst['worst_frame']} ({worst['worst_zone']}) "
          f"→ ยกได้ไม่เกิน {worst['limit_kg']} kg · บิดตัวสูงสุด {worst['twist_max_deg']}°")
    print(f"รอบที่เตือนบิดตัว: {int(lifts['twist_flag'].sum())} จาก {len(lifts)} รอบ "
          f"(ไหล่เทียบสะโพก > {TWIST_WARN_DEG:.0f}° หรือแนวไหล่แกว่ง > {SWING_WARN_DEG:.0f}°)")'''))

cells.append(md('### 12.1 เส้นเวลา\nแถบสีเขียว = รอบการยกที่นับได้ · เส้นประ = เกณฑ์ (เข่า/เอว/ไหล่ · 7″/12″ · มุมบิด 17°)'))
cells.append(code(r'''t = results["time_s"]
fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
blue, grey = "#1a73e8", "#9aa0a6"

axes[0].plot(t, results["V_cm"], color=blue, lw=2)
for name in ["knee", "waist", "shoulder"]:
    axes[0].axhline(CALIB_RUN[name + "_cm"], color=grey, ls="--", lw=1)
    axes[0].text(t.iloc[-1], CALIB_RUN[name + "_cm"], " " + name, va="center", fontsize=8)
axes[0].set_ylabel("V (cm)")

axes[1].plot(t, results["H_cm"], color=blue, lw=2)
for cm, name in zip(REACH_LIMITS_CM, ['7"', '12"']):
    axes[1].axhline(cm, color=grey, ls="--", lw=1)
    axes[1].text(t.iloc[-1], cm, " " + name, va="center", fontsize=8)
axes[1].set_ylabel("H (cm)")

axes[2].plot(t, results["twist_smooth_deg"], color=blue, lw=2, label="twist")
axes[2].plot(t, results["orientation_deg"], color="#e8710a", lw=1.5, label="orientation to camera")
axes[2].axhline(TWIST_WARN_DEG, color=grey, ls="--", lw=1)
axes[2].set_ylabel("angle (deg)")
axes[2].legend(frameon=False, loc="upper left", ncol=2)

axes[3].step(t, results["limit_kg"], where="post", color=blue, lw=2)
axes[3].set_ylim(0, 46)
axes[3].set_ylabel("max weight (kg)")
axes[3].set_xlabel("time (s)")

for ax in axes:                                   # ช่วง calibrate + ช่วงที่กำลังยก
    ax.axvspan(0, CALIB_SECONDS, color="#e8eaed", zorder=0)
    for _, lift in lifts.iterrows():
        ax.axvspan(lift["start_s"], lift["end_s"], color="#34a853", alpha=0.15, zorder=0)
    ax.grid(alpha=0.25)
plt.tight_layout()
plt.show()'''))

cells.append(md(r'''### 12.2 เวลาที่มืออยู่ในแต่ละช่องของตาราง
ช่องที่ใช้เวลามากและน้ำหนักต่ำ คือจุดที่ควรปรับสถานีงานก่อน'''))
cells.append(code(r'''seconds = valid.groupby(["zone", "reach_class"]).size() / FPS
zone_table = seconds.unstack("reach_class").reindex(index=ROW_NAMES, columns=COL_NAMES).fillna(0).round(2)
print("เวลาที่กึ่งกลางมืออยู่ในแต่ละช่อง (วินาที)")
display(zone_table)
print(f"น้ำหนักที่ยอมรับได้ต่ำสุดตลอดคลิป = {valid['limit_kg'].min():.0f} kg")
if len(lifts):
    print(f"มุมบิดตัวสูงสุดในรอบการยก = {lifts['twist_max_deg'].max():.0f}° (เกณฑ์ {TWIST_WARN_DEG:.0f}°) · "
          f"แนวไหล่แกว่งมากสุด = {lifts['shoulder_swing_deg'].max():.0f}° (เกณฑ์ {SWING_WARN_DEG:.0f}°)")
print(f"จำนวนรอบการยกที่นับได้ = {len(lifts)} รอบ")'''))

# ---------------------------------------------------------------- step 13: calibration check
cells.append(md(r'''---
## ขั้นที่ ⑬ · ตรวจความถูกต้องของค่าที่ calibrate (ใช้ตอนปรับค่าตั้งต้น)

ค่าที่ calibrate ได้เทียบกับ **สัดส่วนร่างกายมาตรฐาน** (Drillis & Contini) ซึ่งคิดเป็นสัดส่วนของส่วนสูง

| จุด | สัดส่วนมาตรฐาน |
|---|---|
| ความสูงเข่า | 0.285 |
| ความสูงไหล่ | 0.818 |
| ความสูงข้อมือขณะยืนปล่อยแขน | 0.485 |

> 🔑 สัดส่วนเหล่านี้ **ไม่ขึ้นกับส่วนสูงที่กรอก** เพราะทุกค่าสเกลเป็นเส้นตรงตาม `SUBJECT_HEIGHT_CM`
> จึงใช้ตรวจว่าการ calibrate เพี้ยนหรือไม่ได้ แม้ยังไม่รู้ส่วนสูงจริง
> ส่วนค่าที่ **ขึ้นกับส่วนสูงจริง** คือ H และ V เป็นเซนติเมตร ซึ่งมีผลกับการแบ่งคอลัมน์ Near/Mild/Extended'''))
cells.append(code(r'''REFERENCE = {"knee": 0.285, "shoulder": 0.818, "wrist": 0.485}
n_calib = int(round(CALIB_SECONDS * FPS))
still = valid[valid["frame_number"].between(n_calib + 1, n_calib + 15)]

check = pd.DataFrame([
    {"ค่า": "ความสูงเข่า", "วัดได้ (ซม.)": CALIB_RUN["knee_cm"],
     "สัดส่วนที่วัดได้": CALIB_RUN["knee_cm"] / SUBJECT_HEIGHT_CM, "สัดส่วนมาตรฐาน": REFERENCE["knee"]},
    {"ค่า": "ความสูงไหล่", "วัดได้ (ซม.)": CALIB_RUN["shoulder_cm"],
     "สัดส่วนที่วัดได้": CALIB_RUN["shoulder_cm"] / SUBJECT_HEIGHT_CM, "สัดส่วนมาตรฐาน": REFERENCE["shoulder"]},
    {"ค่า": "ความสูงมือขณะยืนนิ่ง", "วัดได้ (ซม.)": still["V_cm"].median(),
     "สัดส่วนที่วัดได้": still["V_cm"].median() / SUBJECT_HEIGHT_CM, "สัดส่วนมาตรฐาน": REFERENCE["wrist"]},
])
check["ต่างจากมาตรฐาน"] = (check["สัดส่วนที่วัดได้"] - check["สัดส่วนมาตรฐาน"]).round(3)
display(check.round(3))
print(f"ระยะกล้อง: จาก cm/px = {CALIB_RUN['camera_dist_cm']:.0f} ซม.")
print(f"H ขณะยืนนิ่ง = {still['H_cm'].median():.1f} ซม. (ยืนปล่อยแขนควรได้ประมาณ 10–20 ซม.)")
print(f"มุมหันเทียบกล้อง (มัธยฐาน) = {results['orientation_deg'].median():.0f}° "
      f"(ถ่ายด้านข้างตามโปรโตคอลควรเข้าใกล้ 90°)")'''))

cells.append(md(r"""---
## ขั้นที่ ⑭ · ผลการปรับค่าตั้งต้น การทดสอบกับคลิปยกหลายรอบ และคลิปของกลุ่มอื่น

ข้อมูลที่ใช้: **20 คลิปเดิมของกลุ่ม** (ผู้ยก 4 คน · คำตอบมาจากรหัสในชื่อไฟล์) และ **2 คลิปยกหลายรอบที่ถ่ายใหม่**
(ผู้ยกสูง 167 ซม. · ถ่ายด้านข้าง · นับรอบด้วยตาได้ 8 และ 7 รอบ ตรงกับจำนวนลูกคลื่นในกราฟความสูงมือ)
ทดลองโดยสกัด keypoint ของทุกคลิปเก็บไว้ครั้งเดียว แล้วรันขั้น ③–⑫ ซ้ำด้วยค่าต่าง ๆ

### 14.1 ตัวนับรอบ: เทียบสามแบบ (ถูกกี่คลิปจาก 22)
| ตัวนับ | ถูก | คลิปยกปกติ (จริง 8) | คลิปยกเร็ว (จริง 7) | คลิปที่ไม่มีการยก 4 คลิป |
|---|---|---|---|---|
| ป้ายของโมเดลอย่างเดียว (แบบ demo ปรับค่าแล้ว) | 18 | 10 | 4 | ถูก 3 |
| ความสูงมืออย่างเดียว | 20 | 8 | 7 | ถูก 2 (นับการเดินและการก้มเป็นการยก) |
| **ความสูงมือ + โมเดลยืนยัน (ที่ใช้)** | **22** | **8** | **7** | **ถูก 4** |

ค่าตั้งต้นของ demo (นับเฉพาะ `lift` ต่อเนื่อง 6 เฟรม) นับถูกแค่ 10 จาก 20 คลิปเดิม

### 14.2 ค่าอื่นที่ปรับ
| ค่า | ค่าตั้งต้นของ demo | ค่าของกลุ่ม | ผล |
|---|---|---|---|
| การเตือนบิดตัว | ไหล่เทียบสะโพก > 45° | รายรอบ: ไหล่เทียบสะโพก > 17° **หรือ** แนวไหล่แกว่งเทียบพื้น > 50° | จับคลิป T1 ได้ 1 → **11 จาก 13** (ดู 14.5) |
| `MAX_REACH_FRACTION` | ไม่มี | 0.50 | ตัดเฟรมที่ H เกินช่วงแขน (เจอใน 3 คลิป สูงสุด 127 ซม.) |
| ตาราง 2 มิติมุมจอ | ไม่มี | เพิ่ม | อ่านค่าได้แม้ตารางในฉากหันสันเข้ากล้อง |

### 14.3 ค่าที่ทดลองแล้ว **ไม่ต้องปรับ**
| ค่า | สิ่งที่ลอง | สิ่งที่พบ |
|---|---|---|
| `CAMERA_FOV_DEG` | 50, 60, 70, 80, 90, 110 | สัดส่วนเข่า/ไหล่/ข้อมือแทบไม่เปลี่ยน ค่า FOV เปลี่ยนแค่ระยะกล้องที่คำนวณได้ ที่ 70° คลิปใหม่คำนวณได้ 2.2 ม. อยู่ในช่วง 2–3 ม. ที่กลุ่มประมาณไว้ จึงคงค่าเดิม |
| การแบ่งแถวตามความสูงมือ | เทียบกับรหัสโซนในชื่อไฟล์ | มือผ่านโซนต้นทาง **16 จาก 16** คลิป และโซนปลายทาง **14 จาก 16** คลิป |

### 14.4 ตรวจกับสัดส่วนร่างกายมาตรฐาน
| จุด | 20 คลิปเดิม (มัธยฐาน) | คลิปใหม่ (ส่วนสูงจริง 167 ซม.) | มาตรฐาน |
|---|---|---|---|
| ความสูงเข่า / ส่วนสูง | 0.267 | 0.29 (49 ซม.) | 0.285 |
| ความสูงไหล่ / ส่วนสูง | 0.799 | 0.83 (139 ซม.) | 0.818 |
| ความสูงมือขณะยืนนิ่ง / ส่วนสูง | 0.491 | 0.50 (84 ซม.) | 0.485 |

คลิปใหม่ถ่ายด้านข้างตรงโปรโตคอล (มุมหันมัธยฐาน 85°) สัดส่วนที่วัดได้จึงใกล้มาตรฐานกว่าคลิปเดิม

### 14.5 ทดสอบกับคลิปของกลุ่มอื่น
คลิปที่มีวิดีโอใน Drive รวม: **G01 20 คลิป** (ผู้ยก 4 คน) และ **G20 5 คลิป** (ผู้ยก 1 คน) ใช้ส่วนสูงสมมติ 170 ซม.
ค่าของตัวนับและการแบ่งโซนไม่เคยปรับจากคลิปเหล่านี้ ส่วนโมเดลเคยเห็น 5 คลิป (G01 S02 อยู่ในชุด train) · ผลรายคลิปอยู่ใน `tuning/batch_eval_other.csv`

| สิ่งที่ตรวจ | ผล |
|---|---|
| จำนวนรอบ | ถูก **21 จาก 25** คลิป |
| มือผ่านโซนต้นทาง / ปลายทาง ตามรหัส | **20 จาก 20** / **18 จาก 20** |
| เกณฑ์มุมบิดเดิม 27° จับคลิป T1 | **1 จาก 9** (วัดได้ 9–31°) → ต้องเปลี่ยนวิธีวัด ดูด้านล่าง |

คลิปที่นับรอบผิด 4 คลิป
| คลิป | จริง → นับได้ | สาเหตุ |
|---|---|---|
| `G01_S01_NEG-WALK` | 0 → 1 | ความสูงมือที่วัดได้เปลี่ยนจาก 105 เป็น 153 ซม. ขณะเดิน และโมเดลตอบ `lower/place` |
| `G01_S05_NEG-TWIST` | 0 → 1 | ยกมือขึ้นขณะบิดตัว (V 86 → 113 ซม.) และโมเดลตอบว่ามีของในมือบางช่วง |
| `G01_S04_BK-WS_T1` | 1 → 2 | ก้มหยิบจากพื้นแล้วยกขึ้นชั้น มือผ่านระดับยืนนิ่งกลางรอบนานพอจนถูกตัดเป็นสองรอบ |
| `G01_S05_WS-AS_T1` | 1 → 0 | ความสูงมือเห็นชัด (90 → 168 ซม.) แต่โมเดลตอบว่ามีของในมือเพียง 6% ของคลิป ต่ำกว่าเกณฑ์ 20% |

**การเตือนบิดตัว: จากเกณฑ์เดียวเป็นกฎสองข้อ**

เกณฑ์ 27° ที่ได้จากคลิปของกลุ่มเราใช้กับกลุ่มอื่นไม่ได้ เพราะผู้ยกของกลุ่มอื่นบิดแบบหมุนทั้งลำตัว สะโพกหมุนตามไหล่
มุมไหล่เทียบสะโพกจึงต่ำ (9–18°) แม้จะบิดจริง เราลองตัววัด 9 แบบกับ 36 คลิปที่มีการยก (`tuning/twist_measures.csv`)
แบบที่แยกคลิปของกลุ่มอื่นได้ดีที่สุดคือ **แนวไหล่แกว่งเทียบกับพื้นภายในรอบ** จึงใช้คู่กับมุมไหล่เทียบสะโพก

| กฎ | กลุ่มเรา (16) | กลุ่มอื่น (20) | รวม (36) | จับ T1 | เตือนผิดใน T0 |
|---|---|---|---|---|---|
| demo: ไหล่เทียบสะโพก > 45° | 13 | 11 | 24 | 1 จาก 13 | 0 จาก 23 |
| ไหล่เทียบสะโพก > 27° | 15 | 12 | 27 | 4 จาก 13 | 0 จาก 23 |
| **ไหล่เทียบสะโพก > 17° หรือ แนวไหล่แกว่ง > 50° (ที่ใช้)** | 12 | 17 | **29** | **11 จาก 13** | 5 จาก 23 |

- กฎใหม่ถูกเพิ่มแค่ 2 คลิปเมื่อนับรวม สิ่งที่เปลี่ยนจริงคือ **จับการบิดตัวได้เกือบหมด แลกกับการเตือนผิด 5 คลิป** ซึ่งเหมาะกับงานคัดกรองมากกว่าการปล่อยผ่าน
- คะแนนของกลุ่มเราลดจาก 15 เป็น 12 เพราะเตือนผิด 4 คลิป โดย 3 คลิปถ่ายเกือบหน้าตรง (มุมหัน 7–31°) ซึ่งทิศของแนวไหล่วัดได้ไม่นิ่ง
- คลิป T1 ที่ยังพลาด 2 คลิป: `G20_S05_WS-AS` (วัดได้ 15° และ 27°) และ `G01_S05_WS-AS` ที่ตัวนับไม่พบรอบจึงไม่มีการตรวจ
- ค่า 17° และ 50° ปรับจาก 35 คลิปที่นับรอบได้ชุดนี้เอง เมื่อปรับโดยกันผู้ยกออกทีละคน (9 คน) ได้ถูก 27 จาก 35
- ค่า 17° อยู่ชิดขอบ: คลิป T0 วัดได้สูงสุด 17.3° ส่วนคลิป T1 ที่ต้องอาศัยข้อนี้วัดได้ 17.8° ทั้งสองคลิป
- การเตือนคิดเมื่อจบรอบ ตัวเลขบนวิดีโอขณะเล่นแสดงเฉพาะมุมไหล่เทียบสะโพก

### 14.6 สิ่งที่ยังไม่ดี
- **ตัวนับพลาด 4 จาก 25 คลิปของกลุ่มอื่น** (ตาราง 14.5) ส่วน 22 จาก 22 เป็นคลิปชุดที่ใช้ปรับค่าเอง
- **การเตือนบิดตัวยังเตือนผิด 5 จาก 23 คลิปที่ไม่บิด** และค่าปรับจากคลิปชุดเดียวกับที่ใช้วัดผล ยังไม่มีคลิปที่ไม่เคยเห็นมายืนยัน
- **ป้ายของโมเดลยังอ่อนในคลิปยกเร็ว** บางรอบโมเดลตอบว่ามีของในมือเพียง 26–39% ของรอบ ใกล้เกณฑ์ 20% ที่ตั้งไว้
- **ตัวนับรู้ผลเมื่อจบรอบ** และถ้ามือไม่กลับมาใกล้ระดับยืนนิ่งระหว่างสองรอบ จะถูกนับรวมเป็นรอบเดียว
- **15 จาก 20 คลิปเดิมยังใช้ส่วนสูงสมมติ 170 ซม.** (5 คลิปของ S03 ใช้ส่วนสูงจริง 167 ซม. ผลรวมไม่เปลี่ยน) และมุมกล้องเอียง (มุมหันมัธยฐาน S03 31–52° · S04 7–36°)
- **ไม่ได้แปะเทปที่ 18 และ 30 ซม.** จึงยังไม่ได้ตรวจความแม่นของ H เป็นเซนติเมตรกับของจริง"""))

cells.append(md(r'''---
## 🧭 สรุป

| ขั้น | ฟังก์ชันหลัก | ได้อะไร |
|---|---|---|
| ①–④ | `open_video` · `extract_keypoints` · `PoseSmoother` · `draw_skeleton` | keypoint ที่นิ่งแล้ว |
| ⑤ | `calibrate` · `solve_camera_pose` · `to_camera_3d` · `body_frame` | ซม./พิกเซล · พิกัด 3 มิติ · ระดับเข่า/เอว/ไหล่ |
| ⑥ ⑦ | `orientation_angle` · `twist_from_baseline` | มุมหันกล้อง · **มุมบิดตัว** |
| ⑧ | `measure_hands` · `draw_weight_table` · `draw_mini_table` | **H / V แบบ NIOSH** · ตารางน้ำหนัก 3×4 |
| ⑨ ⑩ | `FeatureStream` · `predict_window` · `MajorityVote` · `HeightRoundCounter` | กิจกรรมรายเฟรม · **จำนวนรอบการยก** |
| ⑪ ⑫ | `run_pipeline` · `lift_report` | วิดีโอ + CSV รายเฟรม + **ตารางสรุปรายรอบ** |

### ข้อจำกัดที่ต้องบอกผู้ใช้
- ความลึกมาจากโครงร่าง 3 มิติของ MediaPipe กับค่า FOV ที่สมมติ จึงยังไม่ใช่การวัดจริง ควรตรวจกับกล่องที่ระยะ 7″/12″ ก่อนใช้งานจริง
- ตารางน้ำหนักเป็นเกณฑ์ **ท่าทาง** ไม่ได้บอกว่าของที่ยกหนักเท่าไร ระบบจึงบอกได้แค่ "ท่านี้ยกได้ไม่เกินกี่กิโล"
- โมเดลกิจกรรมมี macro-F1 ประมาณ 0.43 บนชุด test ตัวนับรอบจึงใช้ความสูงมือเป็นหลัก และใช้โมเดลแค่ยืนยันว่ามีของในมือ
- ต้องยืนนิ่งช่วงต้นคลิปเพื่อ calibrate ถ้าขยับระหว่างนั้นค่าสเกลจะเพี้ยนทั้งคลิป'''))

nbformat.write(new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'}, 'colab': {'provenance': []}}), DST)
print('written:', DST, len(cells), 'cells')
