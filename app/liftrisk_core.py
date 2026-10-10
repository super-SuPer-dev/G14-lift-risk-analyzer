"""Lift-risk pipeline for the app.

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


def open_video(path):
    # เปิดวิดีโอ คืน (cap, ข้อมูลวิดีโอ)
    cap = cv2.VideoCapture(str(path))
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    return cap, info


def read_frame(path, frame_number):
    # อ่านเฟรมที่ต้องการ (นับจาก 1)
    cap, _ = open_video(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
    ok, frame = cap.read()
    cap.release()
    return frame


def show_images(images, titles, width=4):
    # แสดงภาพ BGR หลายภาพเรียงกัน
    fig, axes = plt.subplots(1, len(images), figsize=(width * len(images), width * 1.6))
    axes = np.atleast_1d(axes)
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))     # BGR → RGB ก่อนแสดง
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def create_pose_detector():
    # สร้างตัวตรวจจับท่าทาง (สร้างครั้งเดียว ใช้ได้ทั้งคลิป)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1)
    return mp_vision.PoseLandmarker.create_from_options(options)


def extract_keypoints(detector, frame, frame_index):
    # ② คืน xy (33, 2) พิกเซล · world (33, 3) เมตร · conf (33,)   (เป็น NaN ถ้าไม่พบคน)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    timestamp_ms = int(frame_index * 1000 / FPS)
    result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)

    xy = np.full((33, 2), np.nan)
    world = np.full((33, 3), np.nan)
    conf = np.full(33, np.nan)
    if result.pose_landmarks:
        for i, lm in enumerate(result.pose_landmarks[0]):
            xy[i] = [lm.x * w, lm.y * h]
            conf[i] = min(lm.visibility, lm.presence)
        for i, lm in enumerate(result.pose_world_landmarks[0]):
            world[i] = [lm.x, lm.y, lm.z]
    return xy, world, conf


MIN_CONFIDENCE     = 0.5   # ค่าวัดที่ confidence ต่ำกว่านี้ไม่ใช้
MAX_PREDICT_FRAMES = 5     # ใช้ค่าทายแทนได้ติดกันกี่เฟรม
HISTORY_FRAMES     = 3     # moving average = เฟรมนี้ + 3 เฟรมก่อนหน้า
WORLD_WINDOW       = 5     # 👉 MODIFY — จำนวนเฟรมที่เฉลี่ยโครงร่าง 3 มิติ


def create_kalman(x, y):
    # Kalman filter แบบความเร็วคงที่  สถานะ = [x, y, vx, vy]  ค่าวัด = [x, y]
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    acc = np.array([[0.5, 0], [0, 0.5], [1, 0], [0, 1]], np.float32)
    kf.processNoiseCov = 16.0 * acc @ acc.T                     # การเคลื่อนที่ไม่แน่นอนแค่ไหน
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 25.0  # ค่าวัดคลาดเคลื่อนประมาณ 5 px
    kf.errorCovPost = np.diag([25, 25, 400, 400]).astype(np.float32)
    kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
    return kf


class PointTracker:
    # ติดตาม + ลดการสั่นของ keypoint 1 จุด

    def __init__(self, width, height):
        self.width, self.height = width, height
        self.gate = max(25.0, 0.04 * math.hypot(width, height))   # ระยะกระโดดสูงสุดที่ยอมรับ (px)
        self.kalman = None       # None = ยังไม่ได้เริ่มติดตาม
        self.missing = 0         # จำนวนเฟรมติดกันที่ไม่มีค่าวัดที่ดี
        self.history = []        # ตำแหน่งล่าสุด สำหรับ moving average

    def update(self, x, y, conf):
        # คืน (ตำแหน่งที่กรองแล้ว, วัดได้จริงในเฟรมนี้หรือไม่)
        good = conf >= MIN_CONFIDENCE and 0 <= x < self.width and 0 <= y < self.height   # NaN → False
        point, observed = None, False

        if self.kalman is None:                          # ยังไม่ได้ติดตาม → เริ่มเมื่อค่าวัดดี
            if good:
                self.kalman = create_kalman(x, y)
                point, observed = np.array([x, y]), True
        else:
            predicted = self.kalman.predict()[:2, 0]     # (ก-1) predict
            jump = math.hypot(x - predicted[0], y - predicted[1]) if good else 0
            if good and jump <= self.gate:               # (ก-2) correct ด้วยค่าวัด
                self.kalman.correct(np.array([[x], [y]], np.float32))
                point, observed = np.array([x, y]), True
                self.missing = 0
            else:                                        # ไม่มีค่าวัดที่ดี → ใช้ค่าทาย
                self.missing += 1
                inside = 0 <= predicted[0] < self.width and 0 <= predicted[1] < self.height
                if self.missing <= MAX_PREDICT_FRAMES and inside:
                    point = predicted
                else:                                    # หายนานเกินไป → เริ่มใหม่
                    self.kalman, self.missing, self.history = None, 0, []

        if point is None:
            return np.array([np.nan, np.nan]), False
        self.history = (self.history + [point])[-(HISTORY_FRAMES + 1):]   # (ข) moving average
        return np.mean(self.history, axis=0), observed


class PoseSmoother:
    # PointTracker 33 ตัว (สำหรับภาพ) + ค่าเฉลี่ยย้อนหลังของโครงร่าง 3 มิติ

    def __init__(self, width, height):
        self.trackers = [PointTracker(width, height) for _ in range(33)]
        self.world_history = deque(maxlen=WORLD_WINDOW)

    def update(self, xy, world, conf):
        xy_s = np.full((33, 2), np.nan)
        observed = np.zeros(33, dtype=bool)
        for i in range(33):
            xy_s[i], observed[i] = self.trackers[i].update(xy[i, 0], xy[i, 1], conf[i])
        self.world_history.append(world)
        world_s = np.nanmean(np.array(self.world_history), axis=0)    # (ค)
        return xy_s, world_s, observed


SKELETON = [(11, 12), (11, 23), (12, 24), (23, 24),                  # ลำตัว
            (11, 13), (13, 15), (15, 17), (15, 19),                  # แขนซ้าย
            (12, 14), (14, 16), (16, 18), (16, 20),                  # แขนขวา
            (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),        # ขาซ้าย
            (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),        # ขาขวา
            (0, 11), (0, 12)]                                        # ศีรษะ


def pt(p):
    # แปลงพิกัดทศนิยมเป็นจุดจำนวนเต็มสำหรับ OpenCV
    return (int(round(p[0])), int(round(p[1])))


def ui_scale(img):
    # ขนาดตัวอักษร/เส้น เทียบกับภาพกว้าง 720 px
    return img.shape[1] / 720


def draw_skeleton(img, xy_s, observed):
    # ④ วาดโครงกระดูกลงบนภาพ
    s = ui_scale(img)
    for a, b in SKELETON:
        if np.isfinite(xy_s[a]).all() and np.isfinite(xy_s[b]).all():
            cv2.line(img, pt(xy_s[a]), pt(xy_s[b]), (255, 255, 255), max(1, int(2 * s)), cv2.LINE_AA)
    for a, b in SKELETON:
        for p in (a, b):
            if np.isfinite(xy_s[p]).all():
                color = (0, 200, 0) if observed[p] else (0, 140, 255)
                cv2.circle(img, pt(xy_s[p]), max(2, int(4 * s)), color, -1, cv2.LINE_AA)
    return img


NOSE_ABOVE_FOOT = 0.905     # พื้น→จมูก / ส่วนสูง
CALIB_SECONDS   = 1.5       # 👉 MODIFY — ช่วงยืนนิ่งต้นคลิป (วินาที)
CAMERA_FOV_DEG  = 70.0      # 👉 MODIFY — มุมมองกล้องตามด้านยาวของภาพ (มือถือทั่วไป 65–75°)
WAIST_FRACTION  = 0.30      # 👉 MODIFY — ตำแหน่งเอวระหว่างสะโพกกับไหล่
FEET = [29, 30, 31, 32]     # ส้นเท้า + ปลายเท้า (ใช้หาระดับพื้น)
LOWER = [27, 28, 29, 30, 31, 32]   # ข้อเท้า + ส้นเท้า + ปลายเท้า (ใช้หาแกนลำตัว เท้า→สะโพก)
PNP_POINTS = [0, 11, 12, 13, 14, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]


def unit(v):
    # ทำเวกเตอร์ให้ยาว 1
    return v / np.linalg.norm(v)


def mean_point(points, ids):
    # จุดเฉลี่ยของหลาย landmark (ข้ามค่า NaN)
    return np.nanmean(points[ids], axis=0)


# ---------------------------------------------------------------- 5.1
def calc_cm_per_px(xy_frames, height_cm):
    # ซม. ต่อพิกเซล จากระยะพื้น→จมูกในภาพ
    distances = []
    for xy in xy_frames:
        floor_y = np.nanmax(xy[FEET, 1])
        distances.append(floor_y - xy[0, 1])
    return height_cm * NOSE_ABOVE_FOOT / np.nanmedian(distances)


def camera_matrix(width, height):
    # เมทริกซ์กล้องรูเข็ม K จาก FOV
    f = (max(width, height) / 2) / math.tan(math.radians(CAMERA_FOV_DEG) / 2)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]])


# ---------------------------------------------------------------- 5.2
def calc_world_scale(world_frames, height_cm):
    # ตัวคูณที่ทำให้โครงร่าง 3 มิติสูง (พื้น→จมูก) เท่าผู้ยกจริง
    heights = []
    for world in world_frames:
        axis = unit(mean_point(world, [23, 24]) - mean_point(world, LOWER))  # เท้า→สะโพก
        heights.append(np.nanmax((world[0] - world[FEET]) @ axis) * 100)    # พื้น→จมูก (ซม.)
    return height_cm * NOSE_ABOVE_FOOT / np.nanmedian(heights)


def solve_camera_pose(P, xy, observed, K, previous=None):
    # หา R, t ที่ทำให้จุด 3 มิติ P ฉายลงภาพแล้วตรงกับ xy มากที่สุด
    ids = [i for i in PNP_POINTS if observed[i]]
    if len(ids) < 6:
        return None
    if previous is None:
        ok, rvec, tvec = cv2.solvePnP(P[ids], xy[ids], K, None)
    else:                                                    # ใช้ผลเฟรมก่อนเป็นจุดเริ่ม → นิ่งและเร็ว
        ok, rvec, tvec = cv2.solvePnP(P[ids], xy[ids], K, None,
                                      previous["rvec"].copy(), previous["tvec"].copy(), True)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return {"R": R, "t": tvec.ravel(), "rvec": rvec, "tvec": tvec}


def to_camera_3d(xy, P, pose, K):
    # ตำแหน่งในภาพ (u, v) + ความลึก Z จากโครงร่าง → พิกัดกล้อง 3 มิติ (33, 3) ซม.
    P_cam = P @ pose["R"].T + pose["t"]                      # โครงร่างในพิกัดกล้อง
    Z = P_cam[:, 2]
    X = (xy[:, 0] - K[0, 2]) * Z / K[0, 0]
    Y = (xy[:, 1] - K[1, 2]) * Z / K[1, 1]
    C = np.column_stack([X, Y, Z])
    missing = np.isnan(xy[:, 0])
    C[missing] = P_cam[missing]                              # จุดที่ไม่เห็นในภาพ → ใช้โครงร่างแทน
    return C


def project(points, K):
    # จุดในพิกัดกล้อง (…, 3) → ตำแหน่งในภาพ (…, 2) พิกเซล
    points = np.asarray(points, dtype=float)
    u = K[0, 0] * points[..., 0] / points[..., 2] + K[0, 2]
    v = K[1, 1] * points[..., 1] / points[..., 2] + K[1, 2]
    return np.stack([u, v], axis=-1)


# ---------------------------------------------------------------- 5.3
def body_frame(C, up, floor=None):
    # กรอบพิกัดร่างกาย: O, up, side, front  (floor = จุดบนพื้นจาก calibrate)
    ankle = mean_point(C, [27, 28])
    side = C[24] - C[23]
    side = unit(side - (side @ up) * up)                     # เส้นสะโพกในแนวราบ
    front = unit(np.cross(side, up))                         # ตั้งฉากกับระนาบสะโพก
    toe, heel = mean_point(C, [31, 32]), mean_point(C, [29, 30])
    if (toe - heel) @ front < 0:                             # ให้ front ชี้ไปทางปลายเท้า
        front = -front
    if floor is None:
        drop = np.nanmin((C[FEET] - ankle) @ up)             # พื้น = จุดเท้าที่ต่ำที่สุด
    else:
        drop = (floor - ankle) @ up
    O = ankle + up * drop                                    # กึ่งกลางข้อเท้า ที่ระดับพื้น
    if not (np.isfinite(O).all() and np.isfinite(front).all()):
        return None
    return {"O": O, "up": up, "side": side, "front": front}


def body_coords(p, body):
    # จุด 3 มิติ → (ด้านหน้า, ความสูง, ด้านข้าง) ซม. เทียบกับ O
    d = p - body["O"]
    return d @ body["front"], d @ body["up"], d @ body["side"]


def twist_angle(world):
    # มุม (มีเครื่องหมาย) ระหว่างเส้นไหล่กับเส้นสะโพก รอบแกนลำตัว — ใช้ในขั้นที่ ⑦
    axis = unit(mean_point(world, [23, 24]) - mean_point(world, LOWER))
    hip = world[24] - world[23]
    shoulder = world[12] - world[11]
    hip = unit(hip - (hip @ axis) * axis)
    shoulder = unit(shoulder - (shoulder @ axis) * axis)
    return math.degrees(math.atan2(np.cross(hip, shoulder) @ axis, hip @ shoulder))


# ---------------------------------------------------------------- รวมทุกอย่าง
def calibrate(samples, height_cm, width, height):
    # ⑤ samples = รายการ {"xy", "world", "observed"} ของช่วงยืนนิ่ง → dict calib
    calib = {"height_cm": height_cm}
    calib["cm_per_px"] = calc_cm_per_px([s["xy"] for s in samples], height_cm)               # 5.1
    calib["K"] = camera_matrix(width, height)
    calib["camera_dist_cm"] = calib["K"][0, 0] * calib["cm_per_px"]
    calib["world_scale"] = calc_world_scale([s["world"] for s in samples], height_cm)          # 5.2

    C_list, axes, pose = [], [], None                                                        # พิกัด 3 มิติของแต่ละเฟรม
    for s in samples:
        P = s["world"] * 100 * calib["world_scale"]
        pose = solve_camera_pose(P, s["xy"], s["observed"], calib["K"], pose)
        if pose is None:
            continue
        C = to_camera_3d(s["xy"], P, pose, calib["K"])
        C_list.append(C)
        axes.append(unit(mean_point(C, [23, 24]) - mean_point(C, LOWER)))
    calib["up"] = unit(np.mean(axes, axis=0))                                                # 5.3 แนวดิ่ง
    floors = []                                                                              # จุดบนพื้น
    for C in C_list:
        body = body_frame(C, calib["up"])
        if body is not None:
            floors.append(body["O"])
    calib["floor"] = np.median(floors, axis=0)

    knee, hip, shoulder = [], [], []                                                         # ระดับเข่า/สะโพก/ไหล่
    for C in C_list:
        body = body_frame(C, calib["up"], calib["floor"])
        if body is None:
            continue
        knee.append(body_coords(mean_point(C, [25, 26]), body)[1])
        hip.append(body_coords(mean_point(C, [23, 24]), body)[1])
        shoulder.append(body_coords(mean_point(C, [11, 12]), body)[1])
    calib["knee_cm"] = np.median(knee)
    calib["shoulder_cm"] = np.median(shoulder)
    calib["waist_cm"] = np.median(hip) + WAIST_FRACTION * (calib["shoulder_cm"] - np.median(hip))
    calib["twist_baseline"] = np.nanmedian([twist_angle(s["world"]) for s in samples])       # ใช้ในขั้นที่ ⑦
    return calib


def draw_body_frame(img, C, calib):
    # วาด O · แนวดิ่ง · ลูกศร front · ระดับเข่า/เอว/ไหล่
    body = body_frame(C, calib["up"], calib["floor"])
    K, s = calib["K"], ui_scale(img)
    O, up, side, front = body["O"], body["up"], body["side"], body["front"]
    cv2.line(img, pt(project(O, K)), pt(project(O + up * calib["shoulder_cm"], K)), (255, 255, 0), 2)
    cv2.arrowedLine(img, pt(project(O, K)), pt(project(O + front * 40, K)), (255, 0, 255), 3, tipLength=0.25)
    cv2.circle(img, pt(project(O, K)), int(7 * s), (255, 0, 255), -1)
    cv2.putText(img, "O", pt(project(O, K) + 10 * s), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, (255, 0, 255), 2)
    for name, color in [("knee", (0, 215, 255)), ("waist", (0, 165, 255)), ("shoulder", (60, 60, 255))]:
        level = O + up * calib[name + "_cm"]
        a, b = project(level - side * 25, K), project(level + side * 25, K)
        cv2.line(img, pt(a), pt(b), color, 2, cv2.LINE_AA)
        cv2.putText(img, f"{name} {calib[name + '_cm']:.0f} cm", pt((max(a[0], b[0]) + 5, a[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, color, 2)
    return img


def orientation_angle(C):
    # ⑥ มุมระหว่างเส้นสะโพกกับระนาบกล้อง (0 = หันเข้ากล้อง, 90 = หันข้าง)
    d = C[24] - C[23]
    yaw = abs(math.degrees(math.atan2(d[2], d[0])))    # 0…180
    return min(yaw, 180 - yaw)                         # พับเป็น 0…90


TWIST_WARN_DEG = 45.0     # 👉 MODIFY


def twist_from_baseline(world, calib):
    # ⑦ มุมบิดตัว (องศา, ค่าบวก) เทียบกับท่ายืนตรง
    diff = twist_angle(world) - calib["twist_baseline"]
    diff = (diff + 180) % 360 - 180                    # ให้อยู่ในช่วง −180…180
    return abs(diff)


TWIST_WARN_DEG = 17.0       # 👉 MODIFY — มุมไหล่เทียบสะโพกสูงสุดในรอบ (ค่าตั้งต้นของ demo คือ 45)
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
    return float(np.percentile(spread, 95) - np.percentile(spread, 5))


REACH_LIMITS_CM = [7 * 2.54, 12 * 2.54]                        # 17.78, 30.48 ซม.
ROW_NAMES = ["Above Shoulder", "Waist-Shoulder", "Knee-Waist", "Below Knee"]
COL_NAMES = ["Near", "Mild", "Extended"]
WEIGHT_LIMITS_KG = [[29, 18, 14],      # Above Shoulder
                    [32, 23, 18],      # Waist to Shoulder
                    [41, 25, 18],      # Knee to Waist  (สูงสุด)
                    [31, 23, 16]]      # Below Knee
LEFT_HAND, RIGHT_HAND = [15, 17, 19], [16, 18, 20]


def table_row(v_cm, calib):
    # แถวของตารางจากความสูงมือ
    if v_cm >= calib["shoulder_cm"]:
        return 0
    if v_cm >= calib["waist_cm"]:
        return 1
    if v_cm >= calib["knee_cm"]:
        return 2
    return 3


def table_col(h_cm):
    # คอลัมน์ของตารางจากระยะ H
    if h_cm < REACH_LIMITS_CM[0]:
        return 0
    if h_cm < REACH_LIMITS_CM[1]:
        return 1
    return 2


def measure_hands(C, xy, calib):
    # ⑧-1 คืน H, V, ช่องในตาราง และน้ำหนักสูงสุด (None ถ้าวัดไม่ได้)
    body = body_frame(C, calib["up"], calib["floor"])
    if body is None:
        return None
    hands_3d = np.nanmean([mean_point(C, LEFT_HAND), mean_point(C, RIGHT_HAND)], axis=0)
    hands_px = np.nanmean([mean_point(xy, LEFT_HAND), mean_point(xy, RIGHT_HAND)], axis=0)
    if np.isnan(hands_3d).any():
        return None
    if np.isnan(hands_px).any():                            # มือหลุดจากภาพ → ฉายจาก 3 มิติแทน
        hands_px = project(hands_3d, calib["K"])

    front, v_cm, side = body_coords(hands_3d, body)
    h_cm = math.hypot(front, side)                          # NIOSH H
    row, col = table_row(v_cm, calib), table_col(h_cm)
    return {"H": h_cm, "V": v_cm, "front": front, "side": side,
            "row": row, "col": col, "limit_kg": WEIGHT_LIMITS_KG[row][col],
            "hands_px": hands_px, "body": body,
            "direction": -1 if front < -REACH_LIMITS_CM[0] else 1}   # มือหลังข้อเท้าชัดเจน → ตารางยื่นไปด้านหลัง


TABLE_LENGTH_CM = 55.0     # 👉 MODIFY — ความยาวตารางด้านหน้า (ซม.)
ABOVE_MARGIN_CM = 30.0     # ความสูงของแถว Above Shoulder เหนือไหล่
HEADER_CM       = 10.0     # แถบหัวคอลัมน์
PX_PER_CM       = 8        # ความละเอียดของภาพตาราง
CELL_ALPHA      = 0.45     # 👉 MODIFY — ความทึบของช่อง


def cell_color(kg):
    # สีเดียวไล่ความเข้ม: น้ำหนักมาก (ปลอดภัยกว่า) = ครีมอ่อน · น้อย = แดงเข้ม
    t = (41 - kg) / (41 - 14)
    light, dark = np.array([160, 225, 255]), np.array([20, 20, 170])
    return tuple(int(c) for c in light + (dark - light) * t)


def make_table_image(calib, highlight, axis_on_left):
    # ⑧-2 วาดตารางแบน: คืน (ภาพสี, ภาพความทึบ 0–1)
    top_cm = calib["shoulder_cm"] + ABOVE_MARGIN_CM + HEADER_CM
    w, h = int(TABLE_LENGTH_CM * PX_PER_CM), int(top_cm * PX_PER_CM)
    color = np.zeros((h, w, 3), np.uint8)
    alpha = np.zeros((h, w), np.float32)

    col_edges = [0, REACH_LIMITS_CM[0], REACH_LIMITS_CM[1], TABLE_LENGTH_CM]            # ซม. จากลำตัว
    row_edges = [calib["shoulder_cm"] + ABOVE_MARGIN_CM, calib["shoulder_cm"],
                 calib["waist_cm"], calib["knee_cm"], 0]                               # ซม. จากพื้น (บน→ล่าง)
    font = cv2.FONT_HERSHEY_SIMPLEX

    def to_x(cm):                     # ซม. จากลำตัว → x บนภาพตาราง
        x = int(cm * PX_PER_CM)
        return x if axis_on_left else w - x

    def to_y(cm):                     # ซม. จากพื้น → y บนภาพตาราง
        return int((top_cm - cm) * PX_PER_CM)

    def write(text, pos, size, thick, rgb=(255, 255, 255)):
        cv2.putText(color, text, pos, font, size, rgb, thick, cv2.LINE_AA)
        cv2.putText(alpha, text, pos, font, size, 1.0, thick, cv2.LINE_AA)

    for r in range(4):
        for c in range(3):
            x0, x1 = sorted([to_x(col_edges[c]), to_x(col_edges[c + 1])])
            y0, y1 = to_y(row_edges[r]), to_y(row_edges[r + 1])
            kg = WEIGHT_LIMITS_KG[r][c]
            is_hand_cell = (r, c) == highlight
            cv2.rectangle(color, (x0, y0), (x1, y1), (0, 230, 255) if is_hand_cell else cell_color(kg), -1)
            alpha[y0:y1, x0:x1] = 0.85 if is_hand_cell else CELL_ALPHA
            cv2.rectangle(color, (x0, y0), (x1, y1), (255, 255, 255), 2)
            cv2.rectangle(alpha, (x0, y0), (x1, y1), 1.0, 2)
            write(str(kg), ((x0 + x1) // 2 - 22, (y0 + y1) // 2 + 15), 1.6, 4,
                  (0, 0, 0) if is_hand_cell else (255, 255, 255))
        x_label = 8 if not axis_on_left else w - 190
        write(ROW_NAMES[r], (x_label, to_y(row_edges[r]) + 26), 0.7, 2)
    for c in range(3):
        x0, x1 = sorted([to_x(col_edges[c]), to_x(col_edges[c + 1])])
        write(COL_NAMES[c], (x0 + 8, to_y(row_edges[0]) - 14), 0.7, 2)
    write("MAX WEIGHT (kg)", (w // 2 - 120, 30), 0.8, 2)
    return color, alpha


def draw_weight_table(img, calib, m):
    # ⑧-3 ฉายมุมตาราง 3 มิติลงภาพ แล้ว warpPerspective ภาพตารางลงไป
    body, K = m["body"], calib["K"]
    O, up, front = body["O"], body["up"], body["front"] * m["direction"]
    top_cm = calib["shoulder_cm"] + ABOVE_MARGIN_CM + HEADER_CM
    corners_3d = np.array([O,                                               # ล่าง-ลำตัว
                           O + front * TABLE_LENGTH_CM,                     # ล่าง-ปลาย
                           O + front * TABLE_LENGTH_CM + up * top_cm,       # บน-ปลาย
                           O + up * top_cm])                                # บน-ลำตัว
    if (corners_3d[:, 2] < 10).any():                                       # อยู่หลังกล้อง → ไม่วาด
        return img
    corners_px = project(corners_3d, K).astype(np.float32)
    if abs(cv2.contourArea(corners_px)) < 0.002 * img.shape[0] * img.shape[1]:
        return img                                                          # แผ่นหันสันเข้ากล้อง → ไม่วาด

    axis_on_left = corners_px[1, 0] >= corners_px[0, 0]                     # ตารางยื่นไปทางขวาของภาพ?
    table, alpha = make_table_image(calib, (m["row"], m["col"]), axis_on_left)
    th, tw = alpha.shape
    x_body, x_end = (0, tw) if axis_on_left else (tw, 0)
    table_corners = np.float32([[x_body, th], [x_end, th], [x_end, 0], [x_body, 0]])

    M = cv2.getPerspectiveTransform(table_corners, corners_px)              # เมทริกซ์ 3×3
    size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(table, M, size)
    warped_alpha = cv2.warpPerspective(alpha, M, size)[:, :, None]
    img[:] = (img * (1 - warped_alpha) + warped * warped_alpha).astype(np.uint8)
    return img


def draw_hand_label(img, calib, m):
    # ⑧-4 จุด (H, V) บนตาราง + ป้ายน้ำหนักสูงสุด
    body, K, s = m["body"], calib["K"], ui_scale(img)
    O, up, front = body["O"], body["up"], body["front"] * m["direction"]
    on_table = project(O + front * m["H"] + up * m["V"], K)                 # (H, V) บนแผ่นตาราง
    hands = m["hands_px"]
    cv2.line(img, pt(hands), pt(on_table), (255, 255, 255), max(1, int(2 * s)), cv2.LINE_AA)
    cv2.circle(img, pt(on_table), int(7 * s), (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, pt(hands), int(16 * s), (0, 230, 255), max(2, int(3 * s)), cv2.LINE_AA)

    # ป้ายวางที่ขอบปลายตาราง ระดับเดียวกับมือ
    anchor = project(O + front * (TABLE_LENGTH_CM + 3) + up * m["V"], K)
    box_w, box_h = int(230 * s), int(80 * s)
    x0 = anchor[0] if anchor[0] >= project(O, K)[0] else anchor[0] - box_w
    x0 = int(np.clip(x0, 5, img.shape[1] - box_w - 5))
    y0 = int(np.clip(anchor[1] - box_h / 2, 5, img.shape[0] - box_h - 5))
    cv2.line(img, pt(on_table), (x0 + box_w // 2, y0 + box_h // 2), (0, 230, 255), max(2, int(3 * s)))
    cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), (0, 230, 255), -1)
    cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f"MAX {m['limit_kg']} kg", (x0 + int(10 * s), y0 + int(35 * s)), font, 1.0 * s,
                (0, 0, 0), max(2, int(3 * s)), cv2.LINE_AA)
    cv2.putText(img, f"{ROW_NAMES[m['row']]} | {COL_NAMES[m['col']]}", (x0 + int(10 * s), y0 + int(55 * s)),
                font, 0.45 * s, (0, 0, 0), max(1, int(1.5 * s)), cv2.LINE_AA)
    cv2.putText(img, f"H {m['H']:.0f} cm   V {m['V']:.0f} cm", (x0 + int(10 * s), y0 + int(72 * s)),
                font, 0.45 * s, (0, 0, 0), max(1, int(1.5 * s)), cv2.LINE_AA)
    return img


FEATURE_TABLE = pd.read_csv(MODEL_DIR / "features.csv").sort_values("order")
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


class ClassifierHead(nn.Module):
    def __init__(self, input_size, num_classes, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.hidden = nn.Linear(input_size, 32)
        self.relu = nn.ReLU()
        self.output = nn.Linear(32, num_classes)

    def forward(self, x):
        return self.output(self.relu(self.hidden(self.dropout(x))))


class RecurrentClassifier(nn.Module):
    # LSTM หรือ GRU
    def __init__(self, cell, input_size, hidden_size, num_classes, dropout, num_layers=1):
        super().__init__()
        rnn_class = nn.LSTM if cell == 'lstm' else nn.GRU
        between_layers = dropout if num_layers > 1 else 0.0
        self.rnn = rnn_class(input_size, hidden_size, num_layers=num_layers,
                             batch_first=True, dropout=between_layers)
        self.head = ClassifierHead(hidden_size, num_classes, dropout)

    def forward(self, x):
        sequence_output, _ = self.rnn(x)
        return self.head(sequence_output[:, -1, :])


class TransformerClassifier(nn.Module):
    def __init__(self, input_size, d_model, num_classes, dropout, num_layers, heads, window):
        super().__init__()
        self.input = nn.Linear(input_size, d_model)
        self.position = nn.Parameter(torch.randn(1, window, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=2 * d_model,
                                           dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.head = ClassifierHead(d_model, num_classes, dropout)

    def forward(self, x):
        x = self.input(x) + self.position[:, :x.shape[1], :]
        x = self.encoder(x)
        return self.head(x[:, -1, :])


def build_model(config):
    if config['model'] in ('lstm', 'gru'):
        return RecurrentClassifier(config['model'], config['input_size'], config['hidden_size'],
                                   config['num_classes'], config['dropout'], config['num_layers'])
    if config['model'] == 'transformer':
        return TransformerClassifier(config['input_size'], config['hidden_size'], config['num_classes'],
                                     config['dropout'], config['num_layers'], config['heads'],
                                     config['window'])
    raise ValueError('ไม่รู้จักโมเดล ' + config['model'])


def load_trained_model(path, device):
    # โหลดไฟล์ .pt ที่สร้างจาก workshop นี้
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    loaded_model = build_model(checkpoint['config']).to(device)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()
    return loaded_model, checkpoint


MODEL, CHECKPOINT = load_trained_model(MODEL_DIR / CKPT_NAME, torch.device("cpu"))
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


LIFT_CLASSES  = {"lift"}    # ค่าตั้งต้นของ demo (เซลล์ถัดไปจะตั้งค่าของกลุ่มทับ)
SMOOTH_FRAMES = 9          # 👉 MODIFY — จำนวนเฟรมที่ใช้โหวต (เลขคี่)


class MajorityVote:
    # ทำให้ label นิ่ง: คืนคลาสที่พบบ่อยที่สุดใน n เฟรมล่าสุด
    def __init__(self, n):
        self.buf = deque(maxlen=n)

    def update(self, label):
        self.buf.append(label)
        return Counter(self.buf).most_common(1)[0][0]

MIN_LIFT_FRAMES = 6   # 👉 MODIFY — LIFT ต้องต่อเนื่องอย่างน้อยกี่เฟรมจึงนับ (6 เฟรม = 0.2 วินาที)


class LiftCounter:
    # ⑧ นับจำนวนครั้งที่ยกแบบทีละเฟรม
    def __init__(self, min_frames):
        self.min_frames = min_frames
        self.count = 0          # จำนวนครั้งที่ยกทั้งหมด
        self.run = 0            # จำนวนเฟรม LIFT ติดกันล่าสุด
        self.counted = False    # ช่วง LIFT ปัจจุบันถูกนับไปแล้วหรือยัง

    def update(self, is_lift):
        if is_lift:
            self.run += 1
        else:
            self.run, self.counted = 0, False                 # จบช่วง LIFT → พร้อมนับครั้งถัดไป
        if self.run >= self.min_frames and not self.counted:
            self.count += 1
            self.counted = True
            return True                                       # เพิ่งนับครั้งใหม่ในเฟรมนี้
        return False


def draw_counter(img, count, just_counted=False):
    # วาดกล่องจำนวนครั้งที่ยกที่มุมขวาบน (กะพริบสีเหลืองในเฟรมที่เพิ่งนับ)
    h, w = img.shape[:2]
    s = w / 720
    color = (0, 220, 255) if just_counted else (50, 50, 50)
    x1 = w - int(250 * s)
    cv2.rectangle(img, (x1, int(15 * s)), (w - int(15 * s), int(95 * s)), color, -1)
    cv2.putText(img, f"LIFTS: {count}", (x1 + int(15 * s), int(70 * s)), cv2.FONT_HERSHEY_SIMPLEX,
                1.3 * s, (255, 255, 255), max(2, int(3 * s)), cv2.LINE_AA)
    return img


LIFT_CLASSES       = {"lift", "carry/hold", "lower/place"}   # 👉 MODIFY — คลาสที่ถือว่ามีของอยู่ในมือ
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
assert test.count == 1


def draw_info_panel(img, lines):
    # แผงข้อความมุมซ้ายบน: lines = [(ข้อความ, สี), ...]
    s = ui_scale(img)
    line_h = int(26 * s)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (int(360 * s), int(15 * s) + line_h * len(lines)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, dst=img)
    for i, (text, color) in enumerate(lines):
        cv2.putText(img, text, (int(10 * s), int(28 * s) + i * line_h), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55 * s, color, max(1, int(1.5 * s)), cv2.LINE_AA)
    return img


MAX_REACH_FRACTION = 0.50   # 👉 MODIFY — H ที่เกินสัดส่วนนี้ของส่วนสูงถือว่าวัดพลาด (แขนเอื้อมไม่ถึง)
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


def run_pipeline(video_path, out_path, height_cm, max_frames=None, progress=None, on_frame=None):
    cap, info = video_path if isinstance(video_path, tuple) else open_video(video_path)
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

                if m is not None and DRAW_SCENE_TABLE:
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
            if on_frame is not None and on_frame(img, row) is False:
                break
            if progress is not None and i % 5 == 0:
                progress(i, info["frames"], counter.count)

    finished = counter.finish()                                                # รอบที่ค้างตอนวิดีโอจบ
    if finished is not None:
        for past in records[finished[0]:finished[1] + 1]:
            past["round"] = counter.count
    cap.release()
    writer.release()
    return pd.DataFrame(records), calib


def lift_report(results):
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
