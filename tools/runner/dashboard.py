"""Read-only status page for a run, plus optional public tunnel (cloudflared / ngrok).

Security: the page only *shows* state (no commands can be run through it) and
requires a random token in the URL. Bound to 127.0.0.1 unless you choose
otherwise; a tunnel makes it reachable from the internet, so only use one on a
machine where you are allowed to (ask the owner / IT).
"""

import json
import re
import secrets
import shutil
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>AgriVision run</title>
<style>
:root{--bg:#f6f8f7;--card:#fff;--ink:#17241e;--muted:#5a6b62;--line:#d6e2db;--ok:#1f7a4f;--bad:#b8432c;--run:#d99a1e}
@media(prefers-color-scheme:dark){:root{--bg:#0f1714;--card:#17221d;--ink:#e6efe9;--muted:#9db0a5;--line:#2a3a32}}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,Segoe UI,Arial,sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px}h1{font-size:20px;margin:4px 0 12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:12px}
.k{color:var(--muted);font-size:12px}.v{font-size:18px;font-weight:600}
table{border-collapse:collapse;width:100%}td,th{padding:4px 6px;border-bottom:1px solid var(--line);text-align:left;font-size:13px}
.pill{padding:1px 8px;border-radius:9px;font-size:12px;color:#fff}
.done{background:var(--ok)}.failed{background:var(--bad)}.running{background:var(--run)}.pending,.skipped{background:#8a978f}
pre{white-space:pre-wrap;word-break:break-all;max-height:420px;overflow:auto;font:12px/1.4 Consolas,monospace;margin:0}
svg{width:100%;height:160px}.muted{color:var(--muted)}
</style></head><body><main>
<h1>AgriVision run <span id="name" class="muted"></span></h1>
<div class="grid" id="tiles"></div>
<div class="card"><b>Steps</b><table id="steps"></table></div>
<div class="card"><b>Training</b> <span class="muted">(validation mIoU per epoch)</span><div id="chart"></div><table id="epochs"></table></div>
<div class="card"><b>Alerts</b><pre id="alerts" class="muted">none</pre></div>
<div class="card"><b>Live log</b> <span class="muted" id="since"></span><pre id="log"></pre></div>
<p class="muted">Read-only view · refreshes every 10 s · <a id="dl" href="#">download results (zip)</a> · <span id="updated"></span></p>
</main><script>
const T=new URLSearchParams(location.search).get("t")||"";
const esc=s=>String(s??"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const f=(v,d=4)=>v==null||v===""||isNaN(+v)?"-":(+v).toFixed(d);
async function load(){
 try{
  const r=await fetch("api/state?t="+encodeURIComponent(T));if(!r.ok)throw r.status;const s=await r.json();
  document.getElementById("name").textContent=s.name+" · "+s.host;
  const g=(s.gpu||[])[0];
  const tiles=[["Status",s.status],["Target",s.target],["Started",s.started],
   ["GPU",g?`${f(g["util_%"],0)}% · ${f(g.mem_used_mb/1024,1)}/${f(g.mem_total_mb/1024,1)} GB · ${f(g.temp_c,0)}°C`:"n/a"],
   ["Disk free",(s.disk_free_gb??"-")+" GB"]];
  document.getElementById("tiles").innerHTML=tiles.map(([k,v])=>`<div class="card"><div class="k">${k}</div><div class="v">${esc(v)}</div></div>`).join("");
  document.getElementById("steps").innerHTML="<tr><th>step</th><th>status</th><th>started</th><th>ended</th></tr>"+
   s.steps.map(x=>`<tr><td>${esc(x.name)}</td><td><span class="pill ${x.status}">${x.status}</span></td><td>${esc(x.started)}</td><td>${esc(x.ended)}</td></tr>`+
   (x.error?`<tr><td colspan=4><pre>${esc(x.error)}</pre></td></tr>`:"")).join("");
  const runs=Object.entries(s.epochs||{});
  document.getElementById("epochs").innerHTML="<tr><th>run</th><th>epoch</th><th>loss</th><th>mIoU</th><th>drydown</th><th>nutrient</th><th>weed</th></tr>"+
   runs.map(([n,r])=>`<tr><td>${esc(n)}</td><td>${esc(r.epoch)}</td><td>${f(r.train_loss)}</td><td>${f(r.val_miou)}</td><td>${f(r.iou_drydown,3)}</td><td>${f(r.iou_nutrient_deficiency,3)}</td><td>${f(r.iou_weed_cluster,3)}</td></tr>`).join("");
  const h=s.history||{};const series=Object.entries(h).filter(([,v])=>v.length);
  if(series.length){const all=series.flatMap(([,v])=>v);const mx=Math.max(...all.map(p=>p[0]));const top=Math.max(0.05,...all.map(p=>p[1]));
   const cols=["#1f7a4f","#d99a1e","#3a6fb0","#b8432c","#7a4b9c"];
   document.getElementById("chart").innerHTML=`<svg viewBox="0 0 600 160" preserveAspectRatio="none">`+series.map(([n,v],i)=>
    `<polyline fill="none" stroke="${cols[i%5]}" stroke-width="2" points="${v.map(p=>`${((p[0]-1)/Math.max(mx-1,1))*590+5},${150-(p[1]/top)*140}`).join(" ")}"/>`+
    `<text x="8" y="${14+i*14}" fill="${cols[i%5]}" font-size="12">${esc(n)}: ${f(v[v.length-1][1])}</text>`).join("")+`</svg>`;}
  document.getElementById("alerts").textContent=(s.alerts||[]).slice(-10).join("\n")||"none";
  const log=document.getElementById("log");const atEnd=log.scrollTop+log.clientHeight>=log.scrollHeight-20;
  log.textContent=(s.log_tail||[]).join("\n");if(atEnd)log.scrollTop=log.scrollHeight;
  document.getElementById("since").textContent=`(last output ${s.seconds_since_output}s ago)`;
  document.getElementById("updated").textContent="updated "+new Date().toLocaleTimeString();
 }catch(e){document.getElementById("updated").textContent="cannot reach runner ("+e+")";}
}
document.getElementById("dl").href="results.zip?t="+encodeURIComponent(T);
load();setInterval(load,10000);
</script></body></html>"""


class Dashboard:
    def __init__(self, state, monitor=None, host="127.0.0.1", port=8765, token=None):
        self.state, self.monitor = state, monitor
        self.host, self.port = host, int(port)
        self.token = token or secrets.token_urlsafe(12)
        self.server = None

    def _payload(self):
        snap = self.state.snapshot()
        from .monitor import read_history
        runs = self.state.dir.parent.parent / "outputs" / "runs"  # workspace/outputs/runs
        history = {}
        ws = Path(snap.get("workspace") or "")
        for base in {runs, ws / "outputs" / "runs"}:
            for hist in base.glob("*/history.csv") if base.exists() else []:
                pts = [(float(r["epoch"]), float(r["val_miou"])) for r in read_history(hist) if r.get("val_miou")]
                history[hist.parent.name] = pts
        snap["history"] = history
        return snap

    def start(self):
        dash = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                url = urllib.parse.urlparse(self.path)
                q = urllib.parse.parse_qs(url.query)
                if q.get("t", [""])[0] != dash.token:
                    self.send_response(403)
                    self.end_headers()
                    self.wfile.write(b"forbidden: add ?t=<token> from the notification")
                    return
                headers = {}
                if url.path.endswith("/api/state"):
                    body, ctype = json.dumps(dash._payload()).encode(), "application/json"
                elif url.path.endswith("/results.zip"):
                    from .results import zip_bytes
                    body, ctype = zip_bytes(dash.state.data.get("workspace") or dash.state.dir.parent.parent), "application/zip"
                    headers["Content-Disposition"] = f'attachment; filename="results_{time.strftime("%Y%m%d-%H%M%S")}.zip"'
                else:
                    body, ctype = PAGE.encode(), "text/html; charset=utf-8"
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                for k, v in headers.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer((self.host, self.port), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return f"http://{'127.0.0.1' if self.host in ('0.0.0.0', '') else self.host}:{self.port}/?t={self.token}"

    def stop(self):
        if self.server:
            self.server.shutdown()


class Tunnel:
    """Expose the local dashboard: 'cloudflared' (no account) or 'ngrok' (free account + authtoken)."""

    def __init__(self, kind, port, exe=None):
        self.kind, self.port, self.proc, self.url = kind, int(port), None, None
        self.exe = exe

    def start(self, timeout=60):
        exe = self.exe or shutil.which(self.kind) or next(
            (str(p) for p in (Path.cwd() / f"{self.kind}.exe", Path.cwd() / self.kind) if p.is_file()), None)
        if not exe or not Path(exe).is_file() and not shutil.which(exe):
            raise FileNotFoundError(f"{self.kind} not found: put it on PATH, in this folder, or pass --tunnel-exe <path>")
        if self.kind == "cloudflared":
            self.proc = subprocess.Popen([exe, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{self.port}"],
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            found = {}

            def reader():
                for line in self.proc.stdout:
                    m = re.search(r"https://[-a-z0-9]+\.trycloudflare\.com", line)
                    if m and "url" not in found:
                        found["url"] = m.group(0)
            threading.Thread(target=reader, daemon=True).start()
            t0 = time.time()
            while time.time() - t0 < timeout and "url" not in found:
                time.sleep(0.5)
            self.url = found.get("url")
        elif self.kind == "ngrok":
            self.proc = subprocess.Popen([exe, "http", str(self.port), "--log", "stdout"],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            t0 = time.time()
            while time.time() - t0 < timeout and not self.url:
                time.sleep(1)
                try:
                    with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=5) as r:
                        tunnels = json.loads(r.read()).get("tunnels", [])
                    self.url = next((t["public_url"] for t in tunnels if t["public_url"].startswith("https")), None)
                except OSError:
                    pass
        else:
            raise ValueError("tunnel must be 'cloudflared' or 'ngrok'")
        if not self.url:
            self.stop()
            raise RuntimeError(f"{self.kind} did not report a public URL within {timeout}s")
        return self.url

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
