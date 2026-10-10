"""Second batch of slides: calculation of V/H, twist + weight table, defaults and sources, demo, limits; plus small edits."""
import math
import re
import shutil
from pathlib import Path

HERE = Path(__file__).parent
SLIDES = HERE / 'deck' / 'project' / 'slides'
ASSETS = HERE / 'artifact-files' / '9ef22c93-143c-4fb9-ad28-bb923d12db4d'
for name, blob in [('result_167.png', 'b04d6c7c01d935e067a407bf470825cd'), ('demo_clip.mp4', 'acce233cd23ae69ecf17f2793204a0ad'),
                   ('demo_still.png', 'cd5bedfa8bc7415cdbb3476bf1a6f20a'), ('app_now.png', '73fdab005252411599013380dc437f3b')]:
    shutil.copy(HERE / 'deckcheck' / name, ASSETS / (blob + Path(name).suffix))

INK, PAPER, ACCENT, AMBER, GREY, LINE, MUTED = '#152238', '#F6F4EE', '#C2570C', '#F2A65A', '#9AA3AF', '#DAD6CB', '#4A5568'
BODY = "font-family:'IBM Plex Sans Thai', Tahoma, sans-serif"
HEAD = "font-family:'Kanit', Tahoma, sans-serif"
CARD = f'background:#FFFFFF;border:1px solid {LINE};border-radius:20px'
TEACHER = 'ไฟล์ตัวอย่างของอาจารย์'


def section(sid, body, footer, gap=32, dark=False):
    back, fore, foot = (INK, PAPER, '#8FA0B8') if dark else (PAPER, INK, '#6B7280')
    return f'''<section id="{sid}" data-transition="fade" style="background:{back};color:{fore};{BODY};padding:128px 128px 160px;display:flex;flex-direction:column;gap:{gap}px">
{body.strip()}
<p style="position:absolute;left:128px;bottom:64px;font-size:24px;color:{foot}">{footer}</p>
<p style="position:absolute;right:128px;bottom:64px;font-size:24px;color:{foot}">0</p>
<aside></aside>
</section>
'''


def header(kicker, title, width='', dark=False):
    return f'''<div style="display:flex;flex-direction:column;gap:16px{width}">
<p style="font-size:30px;font-weight:500;color:{AMBER if dark else ACCENT}">{kicker}</p>
<h2 style="{HEAD};font-size:64px;font-weight:600;line-height:1.2">{title}</h2>
</div>'''


def label(text, left, top, color=INK, weight=400, size=24, width=None):
    w = f';width:{width}px' if width else ''
    return f'<p style="position:absolute;left:{left}px;top:{top}px;font-size:{size}px;font-weight:{weight};line-height:1.4;color:{color}{w}">{text}</p>'


def arrow(x1, y1, x2, y2, color=ACCENT, width=5):
    angle = math.atan2(y2 - y1, x2 - x1)
    out = [f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"/>']
    for (x, y, a) in [(x2, y2, angle), (x1, y1, angle + math.pi)]:
        p1 = (x - 18 * math.cos(a - 0.45), y - 18 * math.sin(a - 0.45))
        p2 = (x - 18 * math.cos(a + 0.45), y - 18 * math.sin(a + 0.45))
        out.append(f'<polygon points="{x:.1f},{y:.1f} {p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}" fill="{color}"/>')
    return ''.join(out)


def step(number, title, formula, note):
    return f'''<div style="display:flex;gap:20px">
<p style="{HEAD};font-size:44px;font-weight:600;line-height:1.1;color:{ACCENT};width:44px">{number}</p>
<div style="flex:1;display:flex;flex-direction:column;gap:6px">
<h3 style="font-size:28px;font-weight:600;line-height:1.3">{title}</h3>
<p style="font-size:26px;font-weight:600;line-height:1.4;color:{INK};background:{PAPER};padding:8px 16px;border-radius:10px">{formula}</p>
<p style="font-size:24px;line-height:1.4;color:{MUTED}">{note}</p>
</div>
</div>'''


# ------------------------------------------------------------------ slide 4: how V and H are computed
side = f'''<svg aria-label="ภาพด้านข้างของคนถือกล่อง จุดอ้างอิง O อยู่ที่กึ่งกลางข้อเท้าบนพื้น V คือความสูงมือจากพื้น H คือระยะแนวราบจาก O ถึงมือ" width="700" height="560" viewBox="0 0 700 560">
<line x1="30" y1="135" x2="500" y2="135" stroke="{GREY}" stroke-width="2" stroke-dasharray="10 8"/>
<line x1="30" y1="245" x2="500" y2="245" stroke="{GREY}" stroke-width="2" stroke-dasharray="10 8"/>
<line x1="30" y1="345" x2="500" y2="345" stroke="{GREY}" stroke-width="2" stroke-dasharray="10 8"/>
<line x1="30" y1="450" x2="670" y2="450" stroke="{INK}" stroke-width="4"/>
<rect x="385" y="180" width="90" height="64" rx="6" fill="{AMBER}" stroke="{INK}" stroke-width="4"/>
<polyline points="270,450 220,450 232,345 215,245 255,135 320,207 410,214" fill="none" stroke="{INK}" stroke-width="14" stroke-linecap="round" stroke-linejoin="round"/>
<circle cx="275" cy="86" r="28" fill="{INK}"/>
<line x1="220" y1="458" x2="220" y2="500" stroke="{ACCENT}" stroke-width="2" stroke-dasharray="6 6"/>
<line x1="430" y1="248" x2="430" y2="500" stroke="{ACCENT}" stroke-width="2" stroke-dasharray="6 6"/>
<line x1="480" y1="212" x2="555" y2="212" stroke="{ACCENT}" stroke-width="2" stroke-dasharray="6 6"/>
{arrow(220, 490, 430, 490)}
{arrow(540, 450, 540, 212)}
<circle cx="220" cy="450" r="11" fill="{ACCENT}" stroke="#FFFFFF" stroke-width="3"/>
</svg>'''
geometry = section('geometry', f'''
{header('การคำนวณ 1 · ความสูงมือและระยะเอื้อม', 'จากจุดในภาพ เป็นระยะจริงหน่วยเซนติเมตร')}
<div style="display:flex;align-items:stretch;gap:24px">
<div style="position:relative;width:700px;height:560px;{CARD}">
{side}
{label('ไหล่', 28, 99, MUTED)}
{label('เอว', 28, 209, MUTED)}
{label('เข่า', 28, 309, MUTED)}
{label('<b>O</b>', 176, 404, ACCENT, size=28)}
{label('<b>V</b> ความสูงมือ จากพื้น', 560, 290, ACCENT, size=26, width=130)}
{label('<b>H</b> ระยะเอื้อม จาก O ถึงมือ', 220, 500, ACCENT, size=26)}
</div>
<div style="flex:1;display:flex;flex-direction:column;gap:22px;padding:32px;{CARD}">
{step(1, 'แปลงพิกเซลเป็นเซนติเมตร', 'ซม. ต่อพิกเซล = (ส่วนสูง × 0.905) ÷ ระยะพื้นถึงจมูกในภาพ',
      'วัดตอนยืนตรง 1.5 วินาทีแรก · 0.905 คือสัดส่วนความสูงจมูกต่อส่วนสูง')}
{step(2, 'หาพิกัด 3 มิติของทุกจุด', 'หาตำแหน่งกล้องที่ทำให้โครงร่าง 3 มิติฉายลงภาพแล้วตรงกับจุดในภาพ',
      'ใช้ solvePnP ของ OpenCV · ระยะโฟกัสคิดจากมุมมองกล้อง 70°')}
{step(3, 'วัดมือเทียบกับจุด O', 'V = ระยะแนวดิ่งจาก O ถึงมือ · H = √(ด้านหน้า² + ด้านข้าง²)',
      'O = กึ่งกลางข้อเท้าที่ระดับพื้น · มือ = ค่าเฉลี่ยข้อมือและโคนนิ้วของสองมือ')}
</div>
</div>
''', f'นิยาม V และ H ตาม NIOSH Lifting Equation · สูตรจาก{TEACHER}', gap=28)
(SLIDES / 'geometry.html').write_text(geometry, encoding='utf-8', newline='\n')

# ------------------------------------------------------------------ slide 5: twist and the weight table
top = f'''<svg aria-label="ภาพมองจากด้านบน แนวไหล่หมุนไปจากแนวสะโพก มุมระหว่างสองแนวคือมุมบิดตัว" width="330" height="250" viewBox="0 0 330 250">
<line x1="75" y1="130" x2="255" y2="130" stroke="{INK}" stroke-width="26" stroke-linecap="round"/>
<line x1="58.5" y1="204.6" x2="271.5" y2="55.4" stroke="{ACCENT}" stroke-width="22" stroke-linecap="round"/>
<circle cx="165" cy="130" r="34" fill="{PAPER}" stroke="{INK}" stroke-width="6"/>
<path d="M 287 130 A 122 122 0 0 0 264.9 60" fill="none" stroke="{INK}" stroke-width="4"/>
</svg>'''


def kg_row(name, cells, strong=None):
    tds = ''.join(f'<td>{"<b>" + str(v) + "</b>" if i == strong else v}</td>' for i, v in enumerate(cells))
    return f'<tr><td>{name}</td>{tds}</tr>'


weight = section('weight', f'''
{header('การคำนวณ 2 · มุมบิดตัวและน้ำหนักสูงสุด', 'เทียบมุมกับเกณฑ์ และอ่านน้ำหนักจากตาราง')}
<div style="display:flex;align-items:stretch;gap:24px">
<div style="width:760px;display:flex;flex-direction:column;gap:14px;padding:28px 32px;{CARD}">
<div style="display:flex;align-items:center;gap:16px">
<div style="position:relative;width:330px;height:250px">
{top}
{label('แนวไหล่', 196, 8, ACCENT, 600)}
{label('แนวสะโพก', 196, 150, INK, 600)}
</div>
<p style="flex:1;font-size:24px;line-height:1.45;color:{MUTED}">มองจากด้านบน มุมระหว่างแนวไหล่กับแนวสะโพก คือการบิดของลำตัว</p>
</div>
<p style="font-size:26px;line-height:1.4"><b>ข้อ 1</b> มุมไหล่เทียบสะโพก − ค่าตอนยืนตรง <b>เกิน 17°</b></p>
<p style="font-size:26px;line-height:1.4"><b>ข้อ 2</b> แนวไหล่หมุนเทียบกับพื้นภายในหนึ่งรอบ <b>เกิน 50°</b></p>
<p style="font-size:26px;font-weight:600;line-height:1.4;color:{ACCENT}">เข้าข้อใดข้อหนึ่ง = เตือนบิดตัวในรอบนั้น</p>
</div>
<div style="flex:1;display:flex;flex-direction:column;gap:14px;padding:28px 32px;{CARD}">
<h3 style="font-size:28px;font-weight:600;line-height:1.3">น้ำหนักสูงสุดที่ยอมรับได้ (กก.)</h3>
<table style="font-size:26px;color:{INK};background:#FFFFFF;border:1px solid {LINE};border-radius:14px">
<tr style="background:{INK}"><th style="width:34%;text-align:left;color:{PAPER};font-weight:600">ความสูงมือ V</th><th style="width:22%;text-align:left;color:{PAPER};font-weight:600">H ไม่ถึง 18</th><th style="width:22%;text-align:left;color:{PAPER};font-weight:600">18 ถึง 30</th><th style="width:22%;text-align:left;color:{PAPER};font-weight:600">เกิน 30 ซม.</th></tr>
{kg_row('เหนือไหล่', [29, 18, 14])}
{kg_row('เอวถึงไหล่', [32, 23, 18])}
{kg_row('เข่าถึงเอว', [41, 25, 18])}
{kg_row('ต่ำกว่าเข่า', [31, 23, 16])}
</table>
<p style="font-size:24px;line-height:1.45;color:{MUTED}">แถว: เทียบ V กับระดับเข่า เอว ไหล่ ของผู้ยกเองที่วัดตอนยืนตรง โดยเอว = สะโพก + 0.30 × (ไหล่ − สะโพก)</p>
<p style="font-size:24px;line-height:1.45;color:{MUTED}">คอลัมน์: แบ่งค่า H ที่ 7 และ 12 นิ้ว (18 และ 30 ซม.)</p>
</div>
</div>
''', f'ตารางจาก{TEACHER} ตัวเลขตรงกับ Lifting Calculator ของรัฐวอชิงตัน (หน่วยปอนด์ แปลงเป็นกิโลกรัม) · ข้อ 2 กลุ่มเพิ่มเอง', gap=28)
(SLIDES / 'weight.html').write_text(weight, encoding='utf-8', newline='\n')

# ------------------------------------------------------------------ slide 7: defaults and where they come from
ROWS = [
    ('สัดส่วนพื้นถึงจมูกต่อส่วนสูง', '0.905', TEACHER, 'คงเดิม'),
    ('ช่วงยืนตรงเพื่อตั้งสเกล', '1.5 วินาที', TEACHER, 'คงเดิม'),
    ('มุมมองกล้อง', '70°', 'กล้องมือถือทั่วไป 65–75°', 'คงเดิม ลอง 50–110° ผลแทบไม่เปลี่ยน'),
    ('ตำแหน่งเอว ระหว่างสะโพกกับไหล่', '0.30', TEACHER, 'คงเดิม โซนต้นทางถูก 16 จาก 16 คลิป'),
    ('ระยะเอื้อมที่ยอมรับ', 'ไม่จำกัด', TEACHER, '<b>ตัดเฟรมที่เกิน 50% ของส่วนสูง</b>'),
    ('เกณฑ์เตือนบิดตัว', '45°', TEACHER, '<b>17° หรือแนวไหล่หมุน 50°</b> จับได้ 1 → 11 จาก 13'),
    ('ตัวนับรอบ', 'ป้าย "ยก" ติดกัน 6 เฟรม', TEACHER, '<b>มือห่างระดับยืนนิ่ง 12 ซม. + โมเดลยืนยัน 20%</b> ถูก 10 จาก 20 → 22 จาก 22'),
    ('โมเดลกิจกรรม', 'LSTM 30 เฟรม 132 ค่า', 'ค่าเดิมของ Workshop 12', '<b>Transformer 60 เฟรม 166 ค่า</b> คะแนน 0.35 → 0.42'),
]
rows = ''.join(f'<tr><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td></tr>' for a, b, c, d in ROWS)
defaults = section('tuning', f'''
{header('ค่าตั้งต้นและที่มา', 'คงไว้ 4 ค่า ปรับ 4 ค่า จากการทดลองกับคลิปจริง')}
<table style="font-size:24px;color:{INK};background:#FFFFFF;border:1px solid {LINE};border-radius:20px">
<tr style="background:{INK}"><th style="width:25%;text-align:left;color:{PAPER};font-weight:600">ค่า</th><th style="width:19%;text-align:left;color:{PAPER};font-weight:600">ค่าตั้งต้น</th><th style="width:20%;text-align:left;color:{PAPER};font-weight:600">ที่มา</th><th style="width:36%;text-align:left;color:{PAPER};font-weight:600">ค่าที่กลุ่มใช้ และผล</th></tr>
{rows}
</table>
''', 'ทดลองกับ 22 คลิปของกลุ่ม (20 คลิปเดิม + 2 คลิปยกหลายรอบ) · เกณฑ์บิดตัวใช้ 36 คลิป รวมของกลุ่มอื่น · คะแนนโมเดล = macro-F1 ชุดตรวจสอบ', gap=28)
(SLIDES / 'tuning.html').write_text(defaults, encoding='utf-8', newline='\n')

# ------------------------------------------------------------------ slide 12: demo
def demo_step(number, title, text):
    return f'''<div style="display:flex;gap:24px;padding:26px 32px;{CARD}">
<p style="{HEAD};font-size:56px;font-weight:600;line-height:1;color:{ACCENT};width:52px">{number}</p>
<div style="flex:1;display:flex;flex-direction:column;gap:6px">
<h3 style="font-size:32px;font-weight:600;line-height:1.3">{title}</h3>
<p style="font-size:26px;line-height:1.45;color:{MUTED}">{text}</p>
</div>
</div>'''


demo = section('demo', f'''
<img src="/_blob/cd5bedfa8bc7415cdbb3476bf1a6f20a" data-video="/_blob/acce233cd23ae69ecf17f2793204a0ad" alt="คลิปผลลัพธ์ของคลิปยกเร็ว: โครงร่างของผู้ยก ตารางน้ำหนัก และตัวนับรอบที่เพิ่มขึ้นทีละรอบ" style="position:absolute;right:128px;top:128px;width:450px;height:800px;object-fit:cover;border-radius:20px">
{header('สาธิตการใช้งาน', 'จากคลิปหนึ่งคลิป ถึงผลสรุปรายรอบ', ';width:1130px')}
<div style="display:flex;flex-direction:column;gap:20px;width:1130px">
{demo_step(1, 'เลือกคลิป กรอกส่วนสูงและน้ำหนักของที่ยก', 'ใช้ไฟล์วิดีโอ หรือกล้องสดของเครื่องก็ได้')}
{demo_step(2, 'กดวิเคราะห์ แล้วดูภาพขณะประมวลผล', 'เห็นโครงร่าง ตารางน้ำหนัก และตัวนับที่ขึ้นเมื่อจบแต่ละรอบ')}
{demo_step(3, 'อ่านผลสรุป หนึ่งแถวต่อหนึ่งรอบ', 'ท่าที่แย่ที่สุดของรอบ น้ำหนักสูงสุด และแถบเตือนเมื่อน้ำหนักหรือการบิดตัวเกินเกณฑ์')}
</div>
''', 'คลิปด้านขวา: ผลลัพธ์จริงของคลิปยกเร็ว 7 รอบ · ความเร็ว 37 เฟรมต่อวินาทีเมื่อปิดตารางเอียงในฉาก (CPU)', gap=32)
(SLIDES / 'demo.html').write_text(demo, encoding='utf-8', newline='\n')

# ------------------------------------------------------------------ slide 13: limits (no next steps)
def limit(title, text):
    return f'''<div style="flex:1;display:flex;flex-direction:column;gap:14px;background:#1E3050;padding:40px;border-radius:20px">
<h3 style="font-size:36px;font-weight:600;line-height:1.3">{title}</h3>
<p style="font-size:28px;line-height:1.5;color:#C9D3E0">{text}</p>
</div>'''


limits = section('limits', f'''
{header('ข้อจำกัด', 'ใช้คัดกรองท่ายกได้ แต่ยังไม่ใช่เครื่องวัด', dark=True)}
<div style="display:flex;gap:32px">
{limit('ความลึกมาจากกล้องตัวเดียว', 'ระยะเอื้อมเพี้ยนเมื่อผู้ยกหันเข้ากล้อง และยังไม่ได้เทียบกับระยะที่วัดจริงบนพื้น')}
{limit('ตัวนับรอบพลาด 4 จาก 25 คลิปของกลุ่มอื่น', 'นับการเดินเป็นการยก ตัดหนึ่งรอบเป็นสองเมื่อมือผ่านระดับเดิมกลางรอบ และข้ามรอบที่โมเดลไม่ยืนยัน')}
{limit('การเตือนบิดตัวยังเตือนผิด', 'จับคลิปบิดตัวได้ 11 จาก 13 แต่เตือนผิด 5 จาก 23 คลิปที่ไม่บิด และค่าปรับจากคลิปชุดเดียวกันนี้')}
</div>
<div style="display:flex;align-items:center;gap:32px;background:{AMBER};padding:32px 40px;border-radius:20px">
<h3 style="font-size:36px;font-weight:600;line-height:1.3;color:{INK}">โค้ดและผลการทดลอง</h3>
<p style="flex:1;font-size:32px;font-weight:600;line-height:1.4;color:{INK}">github.com/super-SuPer-dev/G14-lift-risk-analyzer</p>
</div>
''', 'G14 · Lift Risk Analyzer', gap=48, dark=True)
(SLIDES / 'limits.html').write_text(limits, encoding='utf-8', newline='\n')


# ------------------------------------------------------------------ small edits to existing slides
def edit(name, pairs):
    path = SLIDES / name
    text = path.read_text(encoding='utf-8')
    for old, new in pairs:
        if old not in text and new not in text:
            raise SystemExit(f'{name}: not found: {old[:60]}')
        text = text.replace(old, new)
    path.write_text(text, encoding='utf-8', newline='\n')


edit('result.html', [('>178 → 88</p>', '>175 → 86</p>'), ('ใช้ส่วนสูง 170 ซม.', 'ผู้ยกสูง 167 ซม.'),
                     ('/_blob/1a98b55a61ca2164ea544961cbb3cdf2', '/_blob/b04d6c7c01d935e067a407bf470825cd')])
edit('app.html', [('เว็บแอปที่ใช้โค้ดชุดเดียวกับ notebook', 'เว็บแอปที่รันบนเครื่องตัวเอง (local)'),
                  ('/_blob/47161be9c962694987188e9ba0891c60', '/_blob/73fdab005252411599013380dc437f3b')])
edit('model.html', [
    ('<h3 style="font-size:30px;font-weight:600;line-height:1.3">แม่นแค่ไหน</h3>', '<h3 style="font-size:30px;font-weight:600;line-height:1.3">พารามิเตอร์ได้จากไหน</h3>'),
    ('ชุดทดสอบถูก 45% (macro-F1 0.43) ดีที่สุดจาก 5 การทดลอง LSTM, GRU, Transformer',
     'ทดลอง 5 แบบ (LSTM, GRU, Transformer) เลือกแบบที่คะแนนชุดตรวจสอบสูงสุด: ขนาด 64, 2 ชั้น, dropout 0.3, อัตราเรียนรู้ 0.001'),
    ('325 คลิปของทุกกลุ่ม แบ่งเทรนและทดสอบตามตัวบุคคล เพิ่มสำเนากลับซ้ายขวาในชุดเทรน',
     '325 คลิปของทุกกลุ่ม แบ่งตามตัวบุคคล เพิ่มสำเนากลับซ้ายขวา · ชุดทดสอบถูก 45% (macro-F1 0.43)'),
    ('<h3 style="font-size:30px;font-weight:600;line-height:1.3">เทรนจากอะไร</h3>', '<h3 style="font-size:30px;font-weight:600;line-height:1.3">เทรนจากอะไร แม่นแค่ไหน</h3>'),
])
for name in ['pipeline.html', 'others.html', 'multi.html', 'goal.html', 'cover.html']:
    path = SLIDES / name
    text = path.read_text(encoding='utf-8')
    body, aside = text.split('<aside>')
    if 'notebook' in body:
        print('still says notebook:', name, re.findall(r'[^<>]{0,30}notebook[^<>]{0,30}', body))
print('slides written')
