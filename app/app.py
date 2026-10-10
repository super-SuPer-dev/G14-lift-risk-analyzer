"""G14 Lift Risk Analyzer - web app prototype (Streamlit).

Run:  streamlit run app.py
The pipeline itself lives in liftrisk_core.py, generated from G14_LiftRisk_Realtime.ipynb.
"""
import hashlib
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

import liftrisk_core as core

APP_DIR = Path(__file__).resolve().parent
# MediaPipe/OpenCV cannot open files under a non-ASCII path, so all work happens in the temp folder
WORK_DIR = Path(tempfile.gettempdir()) / "g14_liftrisk_app"
WORK_DIR.mkdir(parents=True, exist_ok=True)
POSE_MODEL = WORK_DIR / "pose_landmarker_lite.task"
if not POSE_MODEL.is_file():
    shutil.copy2(APP_DIR / "models" / "pose_landmarker_lite.task", POSE_MODEL)
core.POSE_MODEL_PATH = POSE_MODEL

st.set_page_config(page_title="G14 Lift Risk Analyzer", page_icon="🏋️", layout="wide")
st.title("🏋️ Lift Risk Analyzer")
st.caption("กลุ่ม G14 · AI Workshop III — วิเคราะห์ท่ายกจากวิดีโอ: นับรอบการยก · ระยะเอื้อม · มุมบิดตัว · น้ำหนักสูงสุดที่ยอมรับได้")

with st.sidebar:
    st.header("1. ข้อมูลเข้า")
    samples = sorted((APP_DIR / "samples").glob("*.mp4"))
    source_kind = st.radio("แหล่งวิดีโอ", ["อัปโหลดไฟล์", "คลิปตัวอย่าง", "กล้องสด"], horizontal=True)
    video_bytes, video_name = None, None
    live = source_kind == "กล้องสด"
    if source_kind == "อัปโหลดไฟล์":
        uploaded = st.file_uploader("วิดีโอการยก (.mp4 / .mov)", type=["mp4", "mov"])
        if uploaded is not None:
            video_bytes, video_name = uploaded.getvalue(), uploaded.name
    elif source_kind == "คลิปตัวอย่าง" and samples:
        sample = st.selectbox("เลือกคลิป", samples, format_func=lambda p: p.name)
        video_bytes, video_name = sample.read_bytes(), sample.name
    elif live:
        camera_index = st.text_input("กล้อง (หมายเลข หรือ URL ของกล้อง IP)", "0",
                                     help="0 = กล้องตัวแรกของเครื่องที่รันแอป · ใส่ URL เช่น http://192.168.1.5:8080/video เพื่อใช้มือถือเป็นกล้อง")
        prepare_s = st.slider("เวลาเดินไปยืนประจำที่ (วินาที)", 0, 15, 5)
        live_s = st.slider("ระยะเวลาวิเคราะห์ (วินาที)", 10, 180, 30)
    height_cm = st.number_input("ส่วนสูงของผู้ยก (ซม.)", 120.0, 220.0, 170.0, 1.0)
    load_kg = st.number_input("น้ำหนักของที่ยก (กก.)", 0.0, 100.0, 10.0, 0.5,
                              help="ใช้เทียบกับน้ำหนักสูงสุดที่ยอมรับได้ของแต่ละรอบ")
    with st.expander("ตั้งค่าขั้นสูง"):
        core.TWIST_WARN_DEG = st.slider("เกณฑ์มุมไหล่เทียบสะโพก (องศา)", 10, 60, int(core.TWIST_WARN_DEG),
                                        help="เตือนบิดตัวเมื่อมุมสูงสุดในรอบเกินค่านี้")
        core.SWING_WARN_DEG = st.slider("เกณฑ์แนวไหล่หมุนเทียบกับพื้นในหนึ่งรอบ (องศา)", 20, 120, int(core.SWING_WARN_DEG),
                                        help="จับการบิดแบบหมุนทั้งลำตัว ซึ่งมุมไหล่เทียบสะโพกจับไม่ได้")
        core.HEIGHT_HI_CM = float(st.slider("มือต้องขยับจากระดับยืนนิ่งกี่ ซม. จึงเริ่มรอบ", 6, 25,
                                            int(core.HEIGHT_HI_CM)))
        core.HEIGHT_LO_CM = core.HEIGHT_HI_CM / 2
        scene_table = st.checkbox("วาดตารางเอียงในฉาก (ช้าลงราว 2.5 เท่า)", value=not live,
                                  help="ปิดไว้จะเหลือตาราง 2 มิติมุมจอ และเร็วพอสำหรับกล้องสด")
    run = st.button("▶ เริ่มกล้องสด" if live else "▶ วิเคราะห์", type="primary",
                    disabled=(not live and video_bytes is None), use_container_width=True)
    if live:
        st.button("■ หยุด", use_container_width=True)        # any click reruns the script and ends the loop
    st.divider()
    st.markdown("**วิธีถ่าย**\n- ยืนตรงนิ่ง 2 วินาทีแรก\n- กล้องตั้งนิ่ง ถ่ายด้านข้าง เห็นเต็มตัวถึงเท้า\n- วางของแล้วกลับมายืนตรงก่อนยกรอบถัดไป")

core.DRAW_SCENE_TABLE = scene_table


def to_h264(source, target, speed=1.0):
    # browsers cannot play OpenCV's mp4v, so re-encode when ffmpeg is available
    # speed < 1 slows the recording down (live runs are written at a nominal 30 fps)
    if shutil.which("ffmpeg") is None:
        return source
    command = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source)]
    if abs(speed - 1.0) > 0.02:
        command += ["-vf", f"setpts={1 / speed:.4f}*PTS"]
    subprocess.run(command + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", str(target)], check=True)
    return target


class Viewer:
    # shows the annotated frame while the pipeline runs, and measures the real processing speed
    def __init__(self, every=2, height=560, stop_after_s=None):
        self.slot = st.empty()
        self.status = st.empty()
        self.every, self.height, self.stop_after_s = every, height, stop_after_s
        self.times = []

    def __call__(self, img, row):
        self.times.append(time.time())
        n = len(self.times)
        elapsed = self.times[-1] - self.times[0]
        if n % self.every == 0:
            scale = self.height / img.shape[0]
            small = cv2.resize(img, (int(img.shape[1] * scale), self.height))
            self.slot.image(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            fps = (n - 1) / elapsed if elapsed > 0 else 0
            left = f" · เหลือ {max(0, self.stop_after_s - elapsed):.0f} วินาที" if self.stop_after_s else ""
            self.status.caption(f"เฟรม {n} · {fps:.1f} เฟรม/วินาที · นับได้ {row['lift_count']} รอบ{left}")
        return not (self.stop_after_s and elapsed >= self.stop_after_s)

    def fps(self):
        return (len(self.times) - 1) / (self.times[-1] - self.times[0]) if len(self.times) > 1 else 0.0

    def clear(self):
        self.slot.empty()
        self.status.empty()


def analyze_file(data, name, height):
    folder = WORK_DIR / hashlib.md5(data).hexdigest()[:12]
    folder.mkdir(exist_ok=True)
    source = folder / ("input" + Path(name).suffix.lower())
    source.write_bytes(data)
    viewer = Viewer(every=3)
    results, calib, lifts, info = core.analyze_video(source, folder / "raw.mp4", height, on_frame=viewer)
    viewer.status.caption("กำลังแปลงวิดีโอ...")
    final = to_h264(folder / "raw.mp4", folder / "result.mp4")
    speed = viewer.fps()
    viewer.clear()
    return {"results": results, "calib": calib, "lifts": lifts, "fps": info["fps"], "video": final,
            "name": name, "speed": speed, "source_fps": info["fps"]}


def analyze_camera(index, height, prepare, duration):
    index = str(index).strip()
    cap = cv2.VideoCapture(int(index) if index.isdigit() else index)
    if not cap.isOpened():
        raise RuntimeError(f"เปิดกล้อง {index} ไม่ได้")
    folder = WORK_DIR / time.strftime("live_%Y%m%d_%H%M%S")
    folder.mkdir(exist_ok=True)
    slot, note = st.empty(), st.empty()
    try:
        start = time.time()
        while time.time() - start < prepare:                      # time to walk into position
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("อ่านภาพจากกล้องไม่ได้")
            scale = 560 / frame.shape[0]
            slot.image(cv2.cvtColor(cv2.resize(frame, (int(frame.shape[1] * scale), 560)), cv2.COLOR_BGR2RGB))
            note.warning(f"ไปยืนให้เห็นเต็มตัว แล้วยืนตรงนิ่ง — เริ่มใน {prepare - (time.time() - start):.0f} วินาที")
        slot.empty()
        note.empty()
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("อ่านภาพจากกล้องไม่ได้")
        info = {"fps": 30.0, "width": frame.shape[1], "height": frame.shape[0], "frames": 0}
        viewer = Viewer(every=2, stop_after_s=duration)
        results, calib, lifts = core.analyze_stream(cap, info, folder / "raw.mp4", height, on_frame=viewer)
    finally:
        cap.release()
        slot.empty()
        note.empty()
    if calib is None or results.empty:
        raise RuntimeError("ยังไม่ทัน calibrate — ต้องยืนนิ่งให้เห็นเต็มตัวอย่างน้อย 2 วินาทีแรก")
    speed = viewer.fps()
    results["time_s"] = [t - viewer.times[0] for t in viewer.times][:len(results)]   # real clock, not frame/30
    lifts = core.lift_report(results)
    viewer.status.caption("กำลังแปลงวิดีโอ...")
    final = to_h264(folder / "raw.mp4", folder / "result.mp4", speed=speed / 30.0)
    viewer.clear()
    return {"results": results, "calib": calib, "lifts": lifts, "fps": speed, "video": final,
            "name": "กล้องสด " + time.strftime("%H:%M:%S"), "speed": speed, "source_fps": None}


if run:
    st.session_state.pop("analysis", None)
    try:
        if live:
            st.session_state["analysis"] = analyze_camera(camera_index, height_cm, prepare_s, live_s)
        else:
            st.session_state["analysis"] = analyze_file(video_bytes, video_name, height_cm)
    except Exception as error:                                       # show the reason instead of a stack trace
        st.error(f"วิเคราะห์ไม่สำเร็จ: {error}\n\nตรวจว่าเห็นผู้ยกเต็มตัวและยืนนิ่งช่วงต้น")

analysis = st.session_state.get("analysis")
if analysis is None:
    st.info("เลือกวิดีโอหรือกล้องสดทางซ้าย กรอกส่วนสูงและน้ำหนักของที่ยก แล้วกดปุ่มเริ่ม")
    st.markdown("""
| ระบบบอกอะไร | คำนวณจาก |
|---|---|
| จำนวนรอบการยก | ความสูงมือที่ขยับออกจากระดับยืนนิ่ง โดยให้โมเดล Transformer ของกลุ่มยืนยันว่ามีของในมือ |
| ระยะเอื้อม H และความสูงมือ V | พิกัด 3 มิติจาก MediaPipe ปรับสเกลด้วยส่วนสูงผู้ยก (นิยามแบบ NIOSH) |
| มุมบิดตัว | มุมระหว่างแนวไหล่กับแนวสะโพก เทียบกับท่ายืนตรงต้นคลิป |
| น้ำหนักสูงสุดที่ยอมรับได้ | ตาราง 3×4 ตามโซนความสูงมือและระยะเอื้อม |
""")
    st.stop()

results, calib, lifts = analysis["results"], analysis["calib"], analysis["lifts"]
measured = results.dropna(subset=["limit_kg"])
if measured.empty:
    st.error("วัดตำแหน่งมือไม่ได้เลยในคลิปนี้ — ตรวจว่ายืนนิ่งช่วงต้นคลิปและเห็นเต็มตัว")
    st.stop()

lifts = lifts.copy()
if len(lifts):
    lifts["เกินเกณฑ์"] = load_kg > lifts["limit_kg"]
    lifts["บิดตัว"] = ((lifts["twist_max_deg"] > core.TWIST_WARN_DEG)
                       | (lifts["shoulder_swing_deg"] > core.SWING_WARN_DEG))
worst_limit = int(lifts["limit_kg"].min()) if len(lifts) else int(measured["limit_kg"].min())
over = int(lifts["เกินเกณฑ์"].sum()) if len(lifts) else 0
twisted = int(lifts["บิดตัว"].sum()) if len(lifts) else 0

st.subheader(f"ผลการวิเคราะห์ · {analysis['name']}")
realtime = analysis["speed"] / (analysis["source_fps"] or 30.0)
st.caption(f"ความเร็วประมวลผล {analysis['speed']:.1f} เฟรม/วินาที"
           + (f" = {realtime:.1f} เท่าของเวลาจริง" if analysis["source_fps"] else " (ภาพสดจากกล้อง)"))
c1, c2, c3, c4 = st.columns(4)
c1.metric("รอบการยกที่นับได้", f"{len(lifts)} รอบ")
c2.metric("น้ำหนักที่ยอมรับได้ต่ำสุด", f"{worst_limit} กก.", f"ของที่ยก {load_kg:g} กก.",
          delta_color="inverse" if load_kg > worst_limit else "normal")
twist_peak = lifts["twist_max_deg"].max() if len(lifts) else results["twist_smooth_deg"].max()
c3.metric("มุมบิดตัวสูงสุดในรอบ", f"{twist_peak:.0f}°", f"เกณฑ์ {core.TWIST_WARN_DEG:.0f}°", delta_color="off")
c4.metric("ระยะเอื้อมไกลสุด", f"{measured['H_cm'].max():.0f} ซม.")

if len(lifts) == 0:
    st.warning("ไม่พบรอบการยกในคลิปนี้")
elif over or twisted:
    st.error(f"⚠️ มีความเสี่ยง: น้ำหนักเกินเกณฑ์ {over} จาก {len(lifts)} รอบ · บิดตัวเกินเกณฑ์ {twisted} จาก {len(lifts)} รอบ")
else:
    st.success(f"✅ ทั้ง {len(lifts)} รอบ น้ำหนักไม่เกินเกณฑ์และไม่บิดตัวเกินเกณฑ์")

left, right = st.columns([2, 3])
with left:
    st.video(str(analysis["video"]))
    st.download_button("⬇ วิดีโอผลลัพธ์", Path(analysis["video"]).read_bytes(), "liftrisk_result.mp4", "video/mp4")
with right:
    st.markdown("**สรุปรายรอบการยก** (ท่าที่แย่ที่สุดของแต่ละรอบ)")
    if len(lifts):
        table = lifts.rename(columns={
            "lift": "รอบ", "start_s": "เริ่ม (วิ)", "end_s": "จบ (วิ)", "duration_s": "นาน (วิ)",
            "V_start_cm": "มือเริ่ม (ซม.)", "V_end_cm": "มือจบ (ซม.)", "V_min_cm": "มือต่ำสุด (ซม.)", "V_max_cm": "มือสูงสุด (ซม.)", "H_max_cm": "เอื้อมไกลสุด (ซม.)",
            "twist_max_deg": "ไหล่เทียบสะโพก (°)", "shoulder_swing_deg": "แนวไหล่หมุน (°)", "worst_zone": "โซนที่แย่ที่สุด", "limit_kg": "ยกได้ไม่เกิน (กก.)"})
        st.dataframe(table.drop(columns=["worst_frame", "twist_flag"]), hide_index=True, use_container_width=True)
    st.markdown("**เวลาที่มืออยู่ในแต่ละช่องของตาราง (วินาที)**")
    seconds = measured.groupby(["zone", "reach_class"]).size() / max(1e-6, analysis["fps"])
    zone_table = (seconds.unstack("reach_class").reindex(index=core.ROW_NAMES, columns=core.COL_NAMES)
                  .fillna(0).round(1))
    st.dataframe(zone_table, use_container_width=True)

st.markdown("**เส้นเวลา**")
timeline = results.set_index("time_s")
t1, t2, t3 = st.tabs(["ความสูงมือ V / ระยะเอื้อม H", "มุมบิดตัว", "น้ำหนักที่ยอมรับได้"])
t1.line_chart(timeline[["V_cm", "H_cm"]])
t2.line_chart(timeline[["twist_smooth_deg"]])
t3.line_chart(timeline[["limit_kg"]])

d1, d2, d3 = st.columns(3)
d1.download_button("⬇ CSV รายเฟรม", results.to_csv(index=False).encode("utf-8-sig"), "liftrisk_frames.csv", "text/csv")
d2.download_button("⬇ CSV รายรอบ", lifts.to_csv(index=False).encode("utf-8-sig"), "liftrisk_rounds.csv", "text/csv")
with d3.popover("ค่าที่ calibrate ได้"):
    st.write({"ซม./พิกเซล": round(calib["cm_per_px"], 3), "ระยะกล้อง (ซม.)": round(calib["camera_dist_cm"]),
              "ระดับเข่า (ซม.)": round(calib["knee_cm"], 1), "ระดับเอว (ซม.)": round(calib["waist_cm"], 1),
              "ระดับไหล่ (ซม.)": round(calib["shoulder_cm"], 1)})

st.caption("ข้อจำกัด: ความลึกประมาณจากกล้องตัวเดียว · เกณฑ์น้ำหนักเป็นเกณฑ์ท่าทาง ไม่ได้ชั่งของจริง · "
           "ตัวนับรอบใช้ความสูงมือและให้โมเดลยืนยัน ถ้ามือไม่กลับมาระดับยืนนิ่งระหว่างสองรอบจะถูกนับรวมกัน")
