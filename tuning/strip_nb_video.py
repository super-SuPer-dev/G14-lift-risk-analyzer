"""git clean filter: drop embedded base64 video previews from the notebook before it is committed (stdin -> stdout)."""
import json
import sys

sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8', newline='\n')
nb = json.load(sys.stdin)
for cell in nb.get('cells', []):
    kept = []
    for output in cell.get('outputs', []):
        html = ''.join(output.get('data', {}).get('text/html', []))
        if 'video/mp4;base64' in html:
            kept.append({'output_type': 'stream', 'name': 'stdout',
                         'text': ['(วิดีโอตัวอย่างถูกตัดออกจากไฟล์ที่เผยแพร่ รัน notebook เองเพื่อดู)\n']})
        else:
            kept.append(output)
    if 'outputs' in cell:
        cell['outputs'] = kept
json.dump(nb, sys.stdout, ensure_ascii=False, indent=1)
sys.stdout.write('\n')
