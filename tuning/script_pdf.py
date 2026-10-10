"""presentation_script.md -> PDF (A4) via HTML and headless Chrome."""
import subprocess
import sys
from pathlib import Path

import markdown

src, out = Path(sys.argv[1]), Path(sys.argv[2])
work = Path(__file__).parent / 'deckcheck'
body = markdown.markdown(src.read_text(encoding='utf-8'), extensions=['tables', 'nl2br'])
html = f'''<!doctype html><html lang="th"><head><meta charset="utf-8"><title>สคริปต์วิดีโอนำเสนอ G14</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@400;600&family=Kanit:wght@600&display=swap">
<style>
@page {{ size: A4; margin: 16mm 15mm; }}
body {{ font-family: 'IBM Plex Sans Thai', Tahoma, sans-serif; font-size: 11.5pt; line-height: 1.6; color: #152238; }}
h1 {{ font-family: 'Kanit', sans-serif; font-size: 20pt; margin: 0 0 8pt; }}
h2 {{ font-family: 'Kanit', sans-serif; font-size: 13.5pt; margin: 16pt 0 4pt; color: #C2570C; break-after: avoid; }}
p {{ margin: 4pt 0; }}
table {{ border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 10pt; }}
th, td {{ border: 0.5pt solid #C9C4B6; padding: 3pt 6pt; text-align: left; vertical-align: top; }}
th {{ background: #152238; color: #F6F4EE; font-weight: 600; }}
tr {{ break-inside: avoid; }}
code {{ font-size: 9pt; }}
hr {{ border: none; border-top: 0.5pt solid #C9C4B6; margin: 12pt 0; }}
a {{ color: #C2570C; }}
</style></head><body>{body}</body></html>'''
page = work / 'script.html'
page.write_text(html, encoding='utf-8')
pdf = work / 'script.pdf'
subprocess.run([r'C:\Program Files\Google\Chrome\Application\chrome.exe', '--headless=new', '--disable-gpu', '--no-pdf-header-footer',
                '--virtual-time-budget=15000', f'--print-to-pdf={pdf}', page.as_uri()], capture_output=True)
out.write_bytes(pdf.read_bytes())
print('wrote', out.name, pdf.stat().st_size)
