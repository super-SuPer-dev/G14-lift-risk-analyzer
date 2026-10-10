"""Capture the running Streamlit app (sample clip analysed) with headless Chrome over the DevTools protocol."""
import asyncio
import base64
import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets

CHROME = r'C:\Program Files\Google\Chrome\Application\chrome.exe'
OUT = Path(__file__).parent / 'deckcheck' / 'app_now.png'
PORT = 9333


async def main():
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
            counter = 0

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

            await send('Emulation.setDeviceMetricsOverride', width=1500, height=964, deviceScaleFactor=1, mobile=False)
            await asyncio.sleep(6)
            print('radio', await js("""(() => { const r = [...document.querySelectorAll('label')].find(l => l.innerText.includes('คลิปตัวอย่าง')); if (r) r.click(); return !!r; })()"""))
            await asyncio.sleep(3)
            print('height', await js("""(() => { const i = document.querySelector('input[aria-label^="ส่วนสูง"]'); if (!i) return false; i.focus(); i.select(); return true; })()"""))
            await send('Input.insertText', text='167')
            for kind in ('keyDown', 'keyUp'):
                await send('Input.dispatchKeyEvent', type=kind, key='Enter', code='Enter', windowsVirtualKeyCode=13)
            await asyncio.sleep(3)
            print('run', await js("""(() => { const b = [...document.querySelectorAll('button')].find(b => b.innerText.includes('วิเคราะห์')); if (b) b.click(); return !!b; })()"""))
            for _ in range(60):
                await asyncio.sleep(2)
                if await js("document.body.innerText.includes('ผลการวิเคราะห์')"):
                    break
            await asyncio.sleep(5)
            print('text', (await js("document.body.innerText") or '')[-400:].replace('\n', ' | ')[:300])
            shot = await send('Page.captureScreenshot', format='png')
            OUT.write_bytes(base64.b64decode(shot['data']))
            print('saved', OUT)
    finally:
        proc.terminate()


asyncio.run(main())
