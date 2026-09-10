#!/usr/bin/env python3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json, subprocess

ROOT = Path(__file__).parent
SCRIPT = ROOT / "switch-profile.sh"
PAGE = '''<!doctype html><html><head><meta charset="utf-8"><link rel="icon" href="data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%20100%20100'%3E%3Ctext%20y='.9em'%20font-size='90'%3E🔌%3C/text%3E%3C/svg%3E"><title>llama-swap profiles</title><style>html,body{height:100%;margin:0;overflow:hidden;font:16px system-ui;background:#101416;color:#eef2f6}header{height:56px;display:flex;gap:12px;align-items:center;padding:0 18px;background:#182129}label{font-weight:600}select,button{font:inherit;padding:6px 10px}#status{color:#b8c6d1}iframe{display:block;width:100%;height:calc(100vh - 56px);border:0;background:white}</style></head><body><header><label for="profile">Profile</label><select id="profile"><option value="27b">Gemma + Qwen 3.8 27B MTP</option><option value="dflash2">Gemma + Qwen 3.8 27B DFlash2</option><option value="flash">Gemma + Qwen 3.8 Flash-Next</option><option value="q6">Gemma + Qwen 3.8 27B Q6</option></select><button id="switch">Switch</button><span id="status" role="status"></span></header><iframe title="llama-swap" src="https://marvin.akita-betelgeuse.ts.net:8033/"></iframe><script>const s=document.querySelector('#status'),p=document.querySelector('#profile');document.querySelector('#switch').onclick=async()=>{s.textContent='Switching; this can take several minutes…';try{const r=await fetch('/switch',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({profile:p.value})});const x=await r.json();if(!r.ok)throw Error(x.error);s.textContent='Active: '+x.profile}catch(e){s.textContent='Switch failed: '+e.message}};</script></body></html>'''
class App(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != '/': self.send_error(404); return
        self.send_response(200); self.send_header('content-type','text/html; charset=utf-8'); self.end_headers(); self.wfile.write(PAGE.encode())
    def do_POST(self):
        if self.path != '/switch': self.send_error(404); return
        try:
            profile=json.loads(self.rfile.read(int(self.headers.get('content-length', '0'))))['profile']
            commands = {'27b': [str(SCRIPT), '27b'], 'dflash2': [str(SCRIPT), 'dflash2'], 'flash': [str(SCRIPT), 'flash'], 'q6': [str(SCRIPT), 'q6']}
            if profile not in commands: raise ValueError('invalid profile')
            subprocess.run(commands[profile],check=True,capture_output=True,text=True,timeout=1900)
            payload={'profile':profile}
            code=200
        except Exception as e: payload={'error':str(e)}; code=500
        body=json.dumps(payload).encode(); self.send_response(code); self.send_header('content-type','application/json'); self.send_header('content-length',str(len(body))); self.end_headers(); self.wfile.write(body)
if __name__ == '__main__': ThreadingHTTPServer(('127.0.0.1',8041),App).serve_forever()
