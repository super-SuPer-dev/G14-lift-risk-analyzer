"""Write the three slides that carry diagrams: geometry (side + top view), multi (real hand-height chart), others (two twist styles)."""
import math
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
SLIDES = HERE / 'deck' / 'project' / 'slides'
OUTPUT = Path(sys.argv[1])                      # liftrisk_output folder with the real CSVs

INK, PAPER, ACCENT, AMBER, GREY, LINE, MUTED = '#152238', '#F6F4EE', '#C2570C', '#F2A65A', '#9AA3AF', '#DAD6CB', '#4A5568'
BODY = "font-family:'IBM Plex Sans Thai', Tahoma, sans-serif"
HEAD = "font-family:'Kanit', Tahoma, sans-serif"
CARD = f'background:#FFFFFF;border:1px solid {LINE};border-radius:20px'


def section(sid, kicker, title, body, footer, gap=36):
    return f'''<section id="{sid}" data-transition="fade" style="background:{PAPER};color:{INK};{BODY};padding:128px 128px 160px;display:flex;flex-direction:column;gap:{gap}px">
{body.strip()}
<p style="position:absolute;left:128px;bottom:64px;font-size:24px;color:#6B7280">{footer}</p>
<p style="position:absolute;right:128px;bottom:64px;font-size:24px;color:#6B7280">0</p>
<aside></aside>
</section>
'''.replace('{KICKER}', kicker).replace('{TITLE}', title)


def header(kicker, title, width=''):
    return f'''<div style="display:flex;flex-direction:column;gap:16px{width}">
<p style="font-size:30px;font-weight:500;color:{ACCENT}">{kicker}</p>
<h2 style="{HEAD};font-size:64px;font-weight:600;line-height:1.2">{title}</h2>
</div>'''


def label(text, left, top, color=INK, weight=400, size=24, width=None):
    w = f';width:{width}px' if width else ''
    return f'<p style="position:absolute;left:{left}px;top:{top}px;font-size:{size}px;font-weight:{weight};line-height:1.4;color:{color}{w}">{text}</p>'


def arrow(x1, y1, x2, y2, color=ACCENT, width=5, both=True):
    # dimension line with triangular heads
    angle = math.atan2(y2 - y1, x2 - x1)
    out = [f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"/>']
    for (x, y, a) in ([(x2, y2, angle)] + ([(x1, y1, angle + math.pi)] if both else [])):
        p1 = (x - 18 * math.cos(a - 0.45), y - 18 * math.sin(a - 0.45))
        p2 = (x - 18 * math.cos(a + 0.45), y - 18 * math.sin(a + 0.45))
        out.append(f'<polygon points="{x:.1f},{y:.1f} {p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}" fill="{color}"/>')
    return ''.join(out)


# ------------------------------------------------------------------ slide 4: geometry
side = f'''<svg aria-label="ภาพด้านข้างของคนถือกล่อง แสดงความสูงมือ V จากพื้น ระยะเอื้อม H จากข้อเท้าถึงมือ และเส้นระดับเข่า เอว ไหล่" width="780" height="490" viewBox="0 0 780 490">
<line x1="40" y1="115" x2="540" y2="115" stroke="{GREY}" stroke-width="2" stroke-dasharray="10 8"/>
<line x1="40" y1="215" x2="540" y2="215" stroke="{GREY}" stroke-width="2" stroke-dasharray="10 8"/>
<line x1="40" y1="305" x2="540" y2="305" stroke="{GREY}" stroke-width="2" stroke-dasharray="10 8"/>
<line x1="40" y1="400" x2="740" y2="400" stroke="{INK}" stroke-width="4"/>
<rect x="415" y="158" width="90" height="64" rx="6" fill="{AMBER}" stroke="{INK}" stroke-width="4"/>
<polyline points="300,400 250,400 262,305 245,215 285,115 350,185 440,192" fill="none" stroke="{INK}" stroke-width="14" stroke-linecap="round" stroke-linejoin="round"/>
<circle cx="305" cy="68" r="28" fill="{INK}"/>
<line x1="250" y1="408" x2="250" y2="446" stroke="{ACCENT}" stroke-width="2" stroke-dasharray="6 6"/>
<line x1="460" y1="226" x2="460" y2="446" stroke="{ACCENT}" stroke-width="2" stroke-dasharray="6 6"/>
<line x1="510" y1="190" x2="585" y2="190" stroke="{ACCENT}" stroke-width="2" stroke-dasharray="6 6"/>
{arrow(250, 436, 460, 436)}
{arrow(570, 400, 570, 190)}
</svg>'''
top = f'''<svg aria-label="ภาพมองจากด้านบน แนวไหล่หมุนไปจากแนวสะโพก มุมระหว่างสองแนวคือมุมบิดตัว" width="420" height="360" viewBox="0 0 420 360">
<line x1="100" y1="230" x2="280" y2="230" stroke="{INK}" stroke-width="26" stroke-linecap="round"/>
<line x1="83.5" y1="304.6" x2="296.5" y2="155.4" stroke="{ACCENT}" stroke-width="22" stroke-linecap="round"/>
<circle cx="190" cy="230" r="34" fill="{PAPER}" stroke="{INK}" stroke-width="6"/>
<path d="M 312 230 A 122 122 0 0 0 289.9 160" fill="none" stroke="{INK}" stroke-width="4"/>
</svg>'''
geometry = section('geometry', '', '', f'''
{header('เส้นทางที่ 1 · เรขาคณิต 3 มิติ', 'วัดท่ายกด้วยสูตร ไม่มีการเทรน')}
<div style="display:flex;align-items:stretch;gap:24px">
<div style="position:relative;width:780px;height:490px;{CARD}">
{side}
{label('มองจากด้านข้าง', 28, 14, MUTED)}
{label('ไหล่', 36, 79, MUTED)}
{label('เอว', 36, 179, MUTED)}
{label('เข่า', 36, 269, MUTED)}
{label('<b>V</b> ความสูงมือ จากพื้น', 592, 258, ACCENT, size=26, width=170)}
{label('<b>H</b> ระยะเอื้อม จากข้อเท้าถึงมือ', 250, 446, ACCENT, size=26)}
</div>
<div style="position:relative;width:420px;height:490px;{CARD}">
{top}
{label('มองจากด้านบน', 28, 14, MUTED)}
{label('แนวไหล่', 222, 96, ACCENT, 600, 26)}
{label('แนวสะโพก', 200, 252, INK, 600, 26)}
{label('มุมบิด', 320, 176, INK, 600, 26)}
{label('มุมบิดตัวดูสองอย่าง: ไหล่เทียบสะโพก และแนวไหล่ที่หมุนเทียบกับพื้น', 28, 376, MUTED, size=24, width=364)}
</div>
<div style="flex:1;display:flex;flex-direction:column;gap:18px;background:{INK};padding:32px;border-radius:20px">
<p style="font-size:24px;font-weight:600;color:{AMBER}">รับอะไร</p>
<p style="font-size:26px;line-height:1.4;color:{PAPER}">จุดร่างกาย 33 จุด และส่วนสูงของผู้ยก</p>
<p style="font-size:24px;font-weight:600;color:{AMBER}">ทำอย่างไร</p>
<p style="font-size:26px;line-height:1.4;color:{PAPER}">ยืนตรง 1.5 วินาทีแรก เพื่อแปลงหน่วยเป็นเซนติเมตร แล้วคำนวณตำแหน่งกล้องทุกเฟรม</p>
<p style="font-size:24px;font-weight:600;color:{AMBER}">ได้อะไร</p>
<p style="font-size:26px;line-height:1.4;color:{PAPER}">V, H และมุมบิดตัว ของทุกเฟรม</p>
</div>
</div>
<p style="font-size:30px;line-height:1.5;color:{MUTED}">ความสูงมือ 4 โซน × ระยะเอื้อม 3 ช่วง (แบ่งที่ 18 และ 30 ซม.) → <b>น้ำหนักสูงสุดของท่านั้น</b></p>
''', 'สูตรจาก notebook ของอาจารย์ · กลุ่มเปลี่ยนวิธีเตือนบิดตัว และตัดเฟรมที่ระยะเอื้อมเกินครึ่งของส่วนสูง')
(SLIDES / 'geometry.html').write_text(geometry, encoding='utf-8', newline='\n')

# ------------------------------------------------------------------ slide 8: multi-lift chart from the real clip
frames = pd.read_csv(OUTPUT / 'VID_long_liftrisk.csv')
lifts = pd.read_csv(OUTPUT / 'VID_long_lifts.csv')
v = frames['V_cm'].rolling(5, center=True, min_periods=1).median()
baseline = float(frames['V_cm'].dropna().head(15).median())
W, H = 1130, 300
X0, X1, Y0, Y1 = 30, 1100, 56, 262
T1, VLO, VHI = float(frames['time_s'].iloc[-1]), 70.0, 135.0
fx = lambda t: X0 + (X1 - X0) * t / T1
fy = lambda cm: Y1 - (Y1 - Y0) * (cm - VLO) / (VHI - VLO)
points = ' '.join(f'{fx(t):.1f},{fy(cm):.1f}' for t, cm in zip(frames['time_s'][::3], v[::3]) if cm == cm)
bands = ''.join(f'<rect x="{fx(r.start_s):.1f}" y="{Y0}" width="{fx(r.end_s) - fx(r.start_s):.1f}" height="{Y1 - Y0}" fill="{AMBER}" opacity="0.4"/>'
                for r in lifts.itertuples())
chart = f'''<svg aria-label="กราฟความสูงมือตลอดคลิปยกปกติ มีแปดช่วงที่มือสูงขึ้นจากระดับยืนนิ่ง ตรงกับแปดรอบที่ระบบนับได้" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
{bands}
<line x1="{X0}" y1="{fy(baseline):.1f}" x2="{X1}" y2="{fy(baseline):.1f}" stroke="{GREY}" stroke-width="2" stroke-dasharray="10 8"/>
<line x1="{X0}" y1="{fy(baseline + 12):.1f}" x2="{X1}" y2="{fy(baseline + 12):.1f}" stroke="{ACCENT}" stroke-width="2" stroke-dasharray="4 8"/>
<polyline points="{points}" fill="none" stroke="{INK}" stroke-width="4" stroke-linejoin="round"/>
</svg>'''
numbers = ''.join(label(str(r.lift), round(fx((r.start_s + r.end_s) / 2)) - 7, 14, ACCENT, 600) for r in lifts.itertuples())
multi = section('multi', '', '', f'''
<img src="/_blob/ebe4e8a1b226799171d5ba5db4eca405" alt="เฟรมจากคลิปยกหลายรอบ: ผู้ยกหันข้างให้กล้อง ถือกล่องระดับอก ตารางน้ำหนักไฮไลต์ช่อง 23 กิโลกรัม ตัวนับขึ้น 2 รอบ" style="position:absolute;right:128px;top:128px;width:450px;height:800px;object-fit:cover;border-radius:20px">
{header('ทดสอบการยกหลายรอบ', 'เปลี่ยนตัวนับ เป็นดูความสูงมือ', ';width:1130px')}
<table style="font-size:26px;color:{INK};background:#FFFFFF;border:1px solid {LINE};border-radius:20px;width:1130px">
<tr style="background:{INK}"><th style="width:15%;text-align:left;color:{PAPER};font-weight:600">ตัวนับ</th><th style="width:53%;text-align:left;color:{PAPER};font-weight:600">นับหนึ่งรอบเมื่อ</th><th style="width:16%;text-align:left;color:{PAPER};font-weight:600">คลิปปกติ ยกจริง 8</th><th style="width:16%;text-align:left;color:{PAPER};font-weight:600">คลิปเร็ว ยกจริง 7</th></tr>
<tr><td>เดิม</td><td>โมเดลตอบว่ากำลังยก ถือ หรือวาง แล้วเลิกตอบ</td><td>10 รอบ</td><td>4 รอบ</td></tr>
<tr><td><b>ใหม่</b></td><td>มือออกจากระดับยืนนิ่งเกิน 12 ซม. แล้วกลับมา และโมเดลยืนยันว่ามีของในมืออย่างน้อย 20% ของช่วงนั้น</td><td><b>8 รอบ</b></td><td><b>7 รอบ</b></td></tr>
</table>
<div style="position:relative;width:{W}px;height:{H}px;{CARD}">
{chart}
{numbers}
{label('ความสูงมือ คลิปยกปกติ', 30, 262, MUTED)}
{label('แถบสี = รอบที่นับได้ · เส้นประ = ระดับยืนนิ่ง และเกณฑ์ +12 ซม.', 440, 262, MUTED)}
</div>
''', 'ผู้ยกสูง 167 ซม. · ถ่ายด้านข้าง · ทั้งสองคลิปทุกรอบยกได้ไม่เกิน 18 กก. และบิดตัวไม่เกิน 7°', gap=28)
(SLIDES / 'multi.html').write_text(multi, encoding='utf-8', newline='\n')

# ------------------------------------------------------------------ slide 9: two ways of twisting, seen from above
def body_top(hip_deg, shoulder_deg, floor_arc):
    cx, cy = 160, 150
    end = lambda r, deg: (cx + r * math.cos(math.radians(deg)), cy - r * math.sin(math.radians(deg)))
    arc_from = 0 if floor_arc else hip_deg
    radius = 126 if floor_arc else 100
    a, b = end(radius, arc_from), end(radius, shoulder_deg)
    ghost = (f'<line x1="{cx - 130}" y1="{cy}" x2="{cx + 140}" y2="{cy}" stroke="{GREY}" stroke-width="3" stroke-dasharray="10 8"/>'
             if floor_arc else '')
    return f'''<rect x="{cx - 52}" y="{cy - 44}" width="30" height="64" rx="14" fill="{LINE}"/>
<rect x="{cx + 22}" y="{cy - 44}" width="30" height="64" rx="14" fill="{LINE}"/>
{ghost}
<line x1="{cx - 78}" y1="{cy}" x2="{cx + 78}" y2="{cy}" stroke="{INK}" stroke-width="24" stroke-linecap="round" transform="rotate({-hip_deg} {cx} {cy})"/>
<line x1="{cx - 112}" y1="{cy}" x2="{cx + 112}" y2="{cy}" stroke="{ACCENT}" stroke-width="20" stroke-linecap="round" transform="rotate({-shoulder_deg} {cx} {cy})"/>
<circle cx="{cx}" cy="{cy}" r="27" fill="{PAPER}" stroke="{INK}" stroke-width="6"/>
<path d="M {a[0]:.1f} {a[1]:.1f} A {radius} {radius} 0 0 0 {b[0]:.1f} {b[1]:.1f}" fill="none" stroke="{INK if not floor_arc else ACCENT}" stroke-width="5"/>'''


def panel(title, text, rule, alt, hip, shoulder, floor_arc):
    return f'''<div style="flex:1;display:flex;align-items:center;gap:28px;padding:20px 32px;{CARD}">
<svg aria-label="{alt}" width="320" height="290" viewBox="0 0 320 290">
{body_top(hip, shoulder, floor_arc)}
</svg>
<div style="flex:1;display:flex;flex-direction:column;gap:12px">
<h3 style="font-size:34px;font-weight:600;line-height:1.3">{title}</h3>
<p style="font-size:26px;line-height:1.45;color:{MUTED}">{text}</p>
<p style="font-size:26px;font-weight:600;line-height:1.45;color:{ACCENT}">{rule}</p>
</div>
</div>'''


def stat(value, text, dark=False, color=INK):
    back = f'background:{INK}' if dark else CARD
    return f'''<div style="flex:1;display:flex;flex-direction:column;gap:6px;{back};padding:26px 32px;border-radius:20px">
<p style="{HEAD};font-size:64px;font-weight:600;line-height:1.1;color:{AMBER if dark else color}">{value}</p>
<p style="font-size:26px;line-height:1.4;color:{'#C9D3E0' if dark else MUTED}">{text}</p>
</div>'''


others = section('others', '', '', f'''
{header('ทดสอบกับคลิปทดสอบเพิ่มเติม', 'นับรอบและโซนใช้ได้ มุมบิดตัวต้องเปลี่ยนวิธีวัด')}
<div style="display:flex;gap:24px">
{stat('21 / 25', 'คลิปที่นับรอบถูก', dark=True)}
{stat('20 / 20', 'โซนความสูงมือต้นทางถูก')}
{stat('1 → 7 / 9', 'คลิปบิดตัวที่จับได้ หลังใช้กฎสองข้อ', color=ACCENT)}
</div>
<div style="display:flex;gap:24px">
{panel('บิดเฉพาะช่วงบน', 'สะโพกอยู่กับที่ ไหล่หมุนไป มุมไหล่เทียบสะโพกจึงสูง', 'ข้อ 1: ไหล่เทียบสะโพกเกิน 17°',
       'มองจากด้านบน สะโพกอยู่แนวเดิม ไหล่หมุนไปสี่สิบองศา', 0, 40, False)}
{panel('หมุนทั้งลำตัว', 'สะโพกหมุนตามไหล่ มุมไหล่เทียบสะโพกได้แค่ 9–18° แม้จะบิดจริง', 'ข้อ 2: แนวไหล่หมุนเทียบพื้นเกิน 50°',
       'มองจากด้านบน สะโพกและไหล่หมุนไปด้วยกันจากแนวเดิมที่เป็นเส้นประ', 40, 52, True)}
</div>
''', 'คลิปทดสอบ 25 คลิปที่กลุ่มหามาเพิ่ม · ตัวนับและโซนไม่ปรับค่าเพิ่ม · กฎบิดตัวปรับโดยใช้คลิปชุดนี้ด้วย · สีส้ม = แนวไหล่ สีเข้ม = แนวสะโพก', gap=28)
(SLIDES / 'others.html').write_text(others, encoding='utf-8', newline='\n')
print('wrote geometry, multi, others')
