"""Round 6: start from the slides as edited online, add a few topic icons, and colour wrong/right and kept/changed values.

From here on the slide files in deck/project/slides are edited directly (the generator scripts no longer match them).
"""
import shutil
from pathlib import Path

HERE = Path(__file__).parent
LIVE = HERE / 'artifact-files' / '9ef22c93-143c-4fb9-ad28-bb923d12db4d' / 'project'
SLIDES = HERE / 'deck' / 'project' / 'slides'
for path in (LIVE / 'slides').glob('*.html'):
    shutil.copy(path, SLIDES / path.name)
shutil.copy(LIVE / 'deck.json', HERE / 'deck' / 'project' / 'deck.json')

WRONG, RIGHT, ACCENT, INK, AMBER, KEPT, TINT = '#B4281E', '#0F766E', '#C2570C', '#152238', '#F2A65A', '#6B7280', '#FBEBD6'


def icon(kind, size, color, label):
    shapes = {
        'weight': '<circle cx="24" cy="12" r="6"/><path d="M13 19h22l6 21a3 3 0 0 1-3 4H10a3 3 0 0 1-3-4z"/>',
        'cycle': '<path d="M40 24a16 16 0 0 1-28 10.6"/><path d="M8 24a16 16 0 0 1 28-10.6"/><path d="M36 5v9h-9"/><path d="M12 43v-9h9"/>',
        'posture': '<circle cx="17" cy="8" r="5"/><path d="M17 14l-3 15 5 15"/><path d="M16 20l13 6"/><rect x="29" y="19" width="13" height="12" rx="2"/>',
        'warning': f'<path d="M24 6L44 41H4z"/><path d="M24 19v11"/><circle cx="24" cy="35" r="1.6" fill="{color}"/>',
        'code': '<path d="M18 13L7 24l11 11"/><path d="M30 13l11 11-11 11"/>',
    }[kind]
    return (f'<svg aria-label="{label}" width="{size}" height="{size}" viewBox="0 0 48 48" fill="none" stroke="{color}" '
            f'stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round">{shapes}</svg>')


def edit(name, pairs):
    path = SLIDES / name
    text = path.read_text(encoding='utf-8')
    for old, new, count in pairs:
        assert text.count(old) == count, (name, text.count(old), old[:70])
        text = text.replace(old, new)
    path.write_text(text, encoding='utf-8', newline='\n')


# ---- goal: one icon per question
NUM = '<p style="font-family:\'Kanit\', Tahoma, sans-serif;font-size:112px;font-weight:600;line-height:1;color:#C2570C">{n}</p>'
edit('goal.html', [
    (NUM.format(n=n), '<div style="display:flex;justify-content:space-between;align-items:center">' + NUM.format(n=n)
     + icon(kind, 96, INK, label) + '</div>', 1)
    for n, kind, label in [(1, 'cycle', 'ลูกศรวน หมายถึงรอบการยก'), (2, 'posture', 'คนถือกล่อง หมายถึงท่ายก'),
                           (3, 'weight', 'ตุ้มน้ำหนัก หมายถึงน้ำหนักสูงสุด')]])

# ---- weight: icon beside the table title
edit('weight.html', [
    ('<h3 style="font-size:28px;font-weight:600;line-height:1.3">น้ำหนักสูงสุดที่ยอมรับได้ (กก.)</h3>',
     '<div style="display:flex;align-items:center;gap:14px">' + icon('weight', 44, ACCENT, 'ตุ้มน้ำหนัก')
     + '<h3 style="font-size:28px;font-weight:600;line-height:1.3">น้ำหนักสูงสุดที่ยอมรับได้ (กก.)</h3></div>', 1)])

# ---- limits: warning icon on each limit, code icon on the repository band
edit('limits.html', [
    ('<h3 style="font-size:36px;font-weight:600;line-height:1.3">', icon('warning', 52, AMBER, 'เครื่องหมายเตือน')
     + '\n<h3 style="font-size:36px;font-weight:600;line-height:1.3">', 3),
    ('<h3 style="font-size:36px;font-weight:600;line-height:1.3;color:#152238">โค้ดและผลการทดลอง</h3>',
     icon('code', 52, INK, 'สัญลักษณ์โค้ด') + '\n<h3 style="font-size:36px;font-weight:600;line-height:1.3;color:#152238">โค้ดและผลการทดลอง</h3>', 1)])

# ---- result: the correct count in the "right" colour
edit('result.html', [
    ('font-size:88px;font-weight:600;line-height:1.1">1 รอบ</p>', f'font-size:88px;font-weight:600;line-height:1.1;color:{RIGHT}">1 รอบ</p>', 1)])

# ---- multi (edited online): wrong counts red, right counts green
edit('multi.html', [
    ('<td style="color:#c2570c">10 รอบ</td><td style="color:#c2570c">4 รอบ</td>',
     f'<td style="color:{WRONG}"><b>10 รอบ</b></td><td style="color:{WRONG}"><b>4 รอบ</b></td>', 1),
    ('<td>8 รอบ</td><td>7 รอบ</td>', f'<td style="color:{RIGHT}"><b>8 รอบ</b></td><td style="color:{RIGHT}"><b>7 รอบ</b></td>', 1),
    ('ทั้งสองคลิปทุกรอบยกได้ไม่เกิน 18 กก. และบิดตัวไม่เกิน 7°', 'ตัวเลขแดง = นับผิด เขียว = นับถูก · ทุกรอบยกได้ไม่เกิน 18 กก. บิดตัวไม่เกิน 7°', 1)])

# ---- others (edited online): before red, after green
edit('others.html', [
    ('color:#c2570c">1 / 9 → <span style="color:#1e3050">7 / 9</span>', f'color:{WRONG}">1 / 9 → <span style="color:{RIGHT}">7 / 9</span>', 1)])

# ---- tuning: kept values grey, changed rows tinted, before/after numbers red/green
span = lambda color, text: f'<span style="color:{color}">{text}</span>'
edit('tuning.html', [
    ('<td>คงเดิม</td>', f'<td style="color:{KEPT}">คงเดิม</td>', 3),
    ('<td>คงเดิม ลอง 50–110° ผลแทบไม่เปลี่ยน</td>', f'<td style="color:{KEPT}">คงเดิม ลอง 50–110° ผลแทบไม่เปลี่ยน</td>', 1),
    ('<td>คงเดิม โซนต้นทางถูก 16 จาก 16 คลิป</td>', f'<td style="color:{KEPT}">คงเดิม โซนต้นทางถูก 16 จาก 16 คลิป</td>', 1),
    ('<tr><td>ระยะเอื้อมที่ยอมรับ</td>', f'<tr style="background:{TINT}"><td>ระยะเอื้อมที่ยอมรับ</td>', 1),
    ('<tr><td>เกณฑ์เตือนบิดตัว</td>', f'<tr style="background:{TINT}"><td>เกณฑ์เตือนบิดตัว</td>', 1),
    ('<tr><td>ตัวนับรอบ</td>', f'<tr style="background:{TINT}"><td>ตัวนับรอบ</td>', 1),
    ('<tr><td>โมเดลกิจกรรม</td>', f'<tr style="background:{TINT}"><td>โมเดลกิจกรรม</td>', 1),
    ('จับได้ 1 → 11 จาก 13', f'จับได้ {span(WRONG, "1")} → {span(RIGHT, "11")} จาก 13', 1),
    ('ถูก 10 จาก 20 → 22 จาก 22', f'ถูก {span(WRONG, "10 จาก 20")} → {span(RIGHT, "22 จาก 22")}', 1),
    ('คะแนน 0.35 → 0.42', f'คะแนน {span(WRONG, "0.35")} → {span(RIGHT, "0.42")}', 1),
    ('ทดลองกับ 22 คลิปของกลุ่ม (20 คลิปเดิม + 2 คลิปยกหลายรอบ) · เกณฑ์บิดตัวใช้ 36 คลิป รวมคลิปทดสอบเพิ่มเติม · คะแนนโมเดล = macro-F1 ชุดตรวจสอบ',
     'แถวสีส้มอ่อน = ค่าที่ปรับ · ตัวเลขแดง = ก่อนปรับ เขียว = หลังปรับ · ทดลองกับ 22 คลิปของกลุ่มและคลิปทดสอบเพิ่มเติม · คะแนนโมเดล = macro-F1', 1)])
print('round 6 applied')
