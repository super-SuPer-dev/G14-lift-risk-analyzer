"""Record a demo of the running Streamlit app (sample clip, 167 cm) as numbered JPEG frames via headless Chrome."""
import asyncio
import base64
import json
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets

CHROME = r'C:\Program Files\Google\Chrome\Application\chrome.exe'
OUT = Path(__file__).parent / 'deckcheck' / 'rec'
PORT = 9334
STEP = 0.25


async def main():
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    profile = tempfile.mkdtemp(prefix='g14chrome')
    proc = subprocess.Popen([CHROME, '--headless=new', '--disable-gpu', f'--remote-debugging-port={PORT}',
                             f'--user-data-dir={profile}', '--window-size=1500,964', 'http://localhost:8501'])
    try:
        for _ in range(40):
            try:
                tabs = json.loads(urllib.request.urlopen(f'http://127.0.0.1:{PORT}/json').read())
                page = next(t for t in tabs if t['type'] == 'page')
                break
            except Exception:
                time.sleep(0.5)
        async with websockets.connect(page['webSocketDebuggerUrl'], max_size=None) as ws:
            counter, frame = 0, 0

            async def send(method, **params):
                nonlocal counter
                counter += 1
                await ws.send(json.dumps({'id': counter, 'method': method, 'params': params}))
                while True:
                    message = json.loads(await ws.recv())
                    if message.get('id') == counter:
                        return message.get('result', {})

            async def js(code):
                result = await send('Runtime.evaluate', expression=code, returnByValue=True, awaitPromise=True)
                return result.get('result', {}).get('value')

            async def record(seconds):
                nonlocal frame
                end = time.time() + seconds
                while time.time() < end:
                    started = time.time()
                    shot = await send('Page.captureScreenshot', format='jpeg', quality=85)
                    (OUT / f'{frame:04d}.jpg').write_bytes(base64.b64decode(shot['data']))
                    frame += 1
                    await asyncio.sleep(max(0.0, STEP - (time.time() - started)))

            await send('Emulation.setDeviceMetricsOverride', width=1500, height=964, deviceScaleFactor=1, mobile=False)
            await asyncio.sleep(6)
            await record(1.5)
            await js("""(() => { const r = [...document.querySelectorAll('label')].find(l => l.innerText.includes('คลิปตัวอย่าง')); if (r) r.click(); })()""")
            await record(2)
            await js("""(() => { const i = document.querySelector('input[aria-label^="ส่วนสูง"]'); if (i) { i.focus(); i.select(); } })()""")
            await send('Input.insertText', text='167')
            for kind in ('keyDown', 'keyUp'):
                await send('Input.dispatchKeyEvent', type=kind, key='Enter', code='Enter', windowsVirtualKeyCode=13)
            await record(2)
            await js("""(() => { const b = [...document.querySelectorAll('button')].find(b => b.innerText.includes('วิเคราะห์')); if (b) b.click(); })()""")
            for _ in range(240):
                await record(STEP)
                if await js("document.body.innerText.includes('ผลการวิเคราะห์')"):
                    break
            await record(4)
            for _ in range(3):
                print('scroll', await js("""(() => { const m = document.querySelector('[data-testid="stMain"]') || document.querySelector('section.main') || document.scrollingElement; m.scrollBy({top: 330, behavior: 'smooth'}); return m.scrollTop; })()"""))
                await record(2.5)
            print('frames', frame)
    finally:
        proc.terminate()


asyncio.run(main())
