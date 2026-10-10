"""Assemble the slide files into one standalone HTML (images beside it) for offline viewing, PDF export and overflow checks."""
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).parent
DECK = HERE / 'deck' / 'project'
ASSETS = HERE / 'artifact-files' / '9ef22c93-143c-4fb9-ad28-bb923d12db4d'
OUT = Path(sys.argv[1])
CHECK = len(sys.argv) > 2 and sys.argv[2] == 'check'
(OUT / 'img').mkdir(parents=True, exist_ok=True)

deck = json.loads((DECK / 'deck.json').read_text(encoding='utf-8'))
sections = []
for sid in deck['order']:
    html = (DECK / 'slides' / f'{sid}.html').read_text(encoding='utf-8')
    for blob in re.findall(r'/_blob/([0-9a-f]{32})', html):
        source = next(ASSETS.glob(blob + '.*'))
        shutil.copy(source, OUT / 'img' / source.name)
        html = html.replace(f'/_blob/{blob}', f'img/{source.name}')
    sections.append(html.strip())

fonts = ''.join(f'<link rel="stylesheet" href="{face["href"]}">\n' for face in deck['faces'].values())
css = '''
* { box-sizing: border-box; margin: 0; }
html, body { background: #2b2b2b; }
section { position: relative; width: 1920px; height: 1080px; overflow: hidden; break-after: page; }
h1 { font-size: 96px; font-weight: 600; line-height: 1.1; }
h2 { font-size: 64px; font-weight: 600; line-height: 1.15; }
h3 { font-size: 44px; font-weight: 600; line-height: 1.2; }
p { font-size: 32px; font-weight: 400; line-height: 1.4; }
img { display: block; }
aside { display: none; }
table { width: 100%; border-collapse: separate; border-spacing: 0; overflow: hidden; }
th, td { padding: 0.35em 0.6em; text-align: left; vertical-align: top; border-top: 1px solid #DAD6CB; }
tr:first-child th, tr:first-child td { border-top: none; }
x-connector { display: block; flex: none; position: relative; height: 24px; }
x-connector::before { content: ""; position: absolute; left: 0; right: 10px; top: 10px; height: 4px; background: currentColor; }
x-connector::after { content: ""; position: absolute; right: 0; top: 2px; border-left: 16px solid currentColor;
                     border-top: 10px solid transparent; border-bottom: 10px solid transparent; }
@page { size: 1920px 1080px; margin: 0; }
@media screen { section { margin: 0 0 24px; } }
@media print { html, body { zoom: 1 !important; background: none; } }
'''
fit = '''<script>
if (!matchMedia('print').matches) {
  const fit = () => { document.body.style.zoom = Math.min(1, innerWidth / 1920); };
  addEventListener('resize', fit); fit();
}
</script>'''
check = '''<script>
document.fonts.ready.then(() => setTimeout(() => {
  const report = [];
  document.querySelectorAll('section').forEach((s, i) => {
    const box = s.getBoundingClientRect(), issues = [];
    const footer = s.style.padding.includes('160px');
    const limit = box.top + (footer ? 920 : 952);
    s.querySelectorAll('*').forEach(e => {
      if (e.tagName === 'ASIDE' || e.closest('aside')) return;
      const r = e.getBoundingClientRect();
      if (!r.width && !r.height) return;
      const pinned = getComputedStyle(e).position === 'absolute';
      const text = (e.textContent || e.alt || '').trim().slice(0, 40);
      if (!pinned && !e.closest('[style*="position:absolute"]') && r.bottom > limit + 1)
        issues.push(`below content area by ${Math.round(r.bottom - limit)}px: <${e.tagName.toLowerCase()}> ${text}`);
      if (r.right > box.left + 1792 + 1 || r.left < box.left + 127)
        issues.push(`outside side margin (${Math.round(r.left - box.left)}..${Math.round(r.right - box.left)}): <${e.tagName.toLowerCase()}> ${text}`);
      if (r.bottom > box.bottom + 1) issues.push(`off canvas bottom: <${e.tagName.toLowerCase()}> ${text}`);
      if (e.scrollWidth > e.clientWidth + 1 && e.children.length === 0)
        issues.push(`text wider than its box by ${e.scrollWidth - e.clientWidth}px: ${text}`);
      if (pinned && e.tagName === 'P') {
        s.querySelectorAll('p').forEach(o => {
          if (o === e || getComputedStyle(o).position !== 'absolute') return;
          const q = o.getBoundingClientRect();
          if (r.left < q.right && q.left < r.right && r.top < q.bottom && q.top < r.bottom && e.compareDocumentPosition(o) & 4)
            issues.push(`pinned texts overlap: "${text}" and "${o.textContent.trim().slice(0, 30)}"`);
        });
      }
    });
    const flow = [...s.children].filter(e => getComputedStyle(e).position !== 'absolute' && e.tagName !== 'ASIDE');
    const bottom = Math.max(...flow.map(e => e.getBoundingClientRect().bottom)) - box.top;
    report.push({slide: i + 1, id: s.id, content_bottom: Math.round(bottom), limit: footer ? 920 : 952, issues: [...new Set(issues)]});
  });
  const pre = document.createElement('pre'); pre.id = 'report'; pre.textContent = JSON.stringify(report);
  document.body.appendChild(pre);
}, 1500));
</script>'''
css = css.replace('@media screen { section { margin: 0 0 24px; } }', '') if CHECK else css
page = (f'<!doctype html>\n<html lang="th">\n<head>\n<meta charset="utf-8">\n<title>{deck["title"]}</title>\n{fonts}'
        f'<style>{css}</style>\n</head>\n<body>\n' + '\n'.join(sections) + '\n' + (check if CHECK else fit) + '\n</body>\n</html>\n')
name = 'check.html' if CHECK else 'G14_slides.html'
(OUT / name).write_text(page, encoding='utf-8', newline='\n')
print('wrote', OUT / name, len(sections), 'slides')
