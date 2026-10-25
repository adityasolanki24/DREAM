"""Authenticated phone interface to the task scheduler: submit, watch, reprioritise, cancel.

Standard library only (http.server). Every request needs ``Authorization: Bearer <token>``,
compared in constant time; the token is generated outside Git and never logged. The server
binds to 127.0.0.1 unless a host is given explicitly: reach it from a phone through a private
tunnel or VPN, never by exposing a robot-control endpoint to the public internet.

Robot control goes only through ``TaskScheduler`` (queue, safe preemption, cancellation at a
safe boundary): nothing here commands motion directly. Notifications are an event log the
page polls (``GET /events?after=n``).
"""

from __future__ import annotations

import hmac
import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PAGE = """<!doctype html><html><head>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>DREAM</title><style>body{font:16px system-ui;margin:16px;max-width:640px}
input,button,select{font:inherit;padding:8px;margin:4px 0}textarea{width:100%;font:inherit}
.t{border:1px solid #8886;border-radius:8px;padding:8px;margin:8px 0}</style></head><body>
<h2>DREAM tasks</h2><input id=k type=password placeholder="access token" style="width:100%">
<textarea id=x rows=2 placeholder="Bring me my water bottle in the bedroom"></textarea>
<select id=p><option value=0>normal</option><option value=5>urgent</option></select>
<button onclick=send()>Send</button><div id=s></div><h3>Tasks</h3><div id=l></div>
<h3>Events</h3><div id=e></div><script>
const E=v=>String(v).replace(/[&<>"']/g,c=>'&#'+c.charCodeAt(0)+';');
const K=document.getElementById('k');K.value=sessionStorage.getItem('k')||'';
const H=()=>({'Authorization':'Bearer '+K.value,'Content-Type':'application/json'});
async function api(m,u,b){sessionStorage.setItem('k',K.value);
const r=await fetch(u,{method:m,headers:H(),body:b?JSON.stringify(b):undefined});return r.json()}
async function send(){const t=document.getElementById('x').value;
await api('POST','/tasks',{text:t,priority:+document.getElementById('p').value});load()}
async function act(id,a){await api('POST','/tasks/'+id+'/'+a,{});load()}
let after=0;async function load(){try{const s=await api('GET','/status');
document.getElementById('s').textContent='running: '+(s.running||'idle');
const ts=await api('GET','/tasks');document.getElementById('l').innerHTML=ts.reverse().map(t=>
`<div class=t><b>${E(t.state)}</b> p${E(t.priority)}: ${E(t.text)}<br>
<small>${E(t.result||'')}</small><br><button onclick="act('${E(t.id)}','cancel')">cancel</button>
<button onclick="act('${E(t.id)}','resume')">resume</button></div>`).join('');
const ev=await api('GET','/events?after='+after);ev.forEach(x=>{after=x.n;
const d=document.createElement('div');d.textContent=x.kind+' '+(x.text||'');
document.getElementById('e').prepend(d)})
}catch(err){document.getElementById('s').textContent='not connected'}}
setInterval(load,2000);load()</script></body></html>"""


def token_file() -> Path:
    import os

    root = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".config")) / "DREAM"
    return root / "phone_token"


def load_or_create_token(path: Path | None = None) -> str:
    path = path or token_file()
    if path.is_file():
        return path.read_text().strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    path.write_text(token)
    return token


class PhoneAPI:
    """Wire a scheduler (and optionally a memory query) to an authenticated HTTP server."""

    def __init__(self, scheduler, token: str, status=None, memory_query=None) -> None:
        if len(token) < 24:
            raise ValueError("token too short")
        self.scheduler, self.token = scheduler, token
        self.status = status or (lambda: {})
        self.memory_query = memory_query
        self.events: list[dict] = []
        self._lock = threading.Lock()

    def notify(self, kind: str, data: dict) -> None:
        """Scheduler notification sink; also what the page shows as completion notices."""
        with self._lock:
            self.events.append({"n": len(self.events) + 1, "kind": kind, **data})

    def handle(self, method: str, url: str, auth: str | None, body: bytes) -> tuple[int, object]:
        """Pure request handler (tested without sockets)."""
        parts = urlparse(url)
        if method == "GET" and parts.path == "/":
            return 200, PAGE
        if not auth or not hmac.compare_digest(auth.encode(), f"Bearer {self.token}".encode()):
            return 401, {"error": "unauthorised"}
        try:
            data = json.loads(body or b"{}")
        except json.JSONDecodeError:
            return 400, {"error": "invalid JSON"}
        path = parts.path.rstrip("/").split("/")[1:]
        s = self.scheduler
        try:
            if method == "GET" and path == ["status"]:
                return 200, {"running": s.running, **self.status()}
            if method == "GET" and path == ["tasks"]:
                return 200, s.tasks()
            if method == "GET" and path == ["events"]:
                after = int(parse_qs(parts.query).get("after", ["0"])[0])
                with self._lock:
                    return 200, [e for e in self.events if e["n"] > after]
            if method == "GET" and path == ["memory"] and self.memory_query:
                label = parse_qs(parts.query).get("label", [""])[0]
                return 200, self.memory_query(label)
            if method == "POST" and path == ["tasks"]:
                text = str(data.get("text", "")).strip()
                if not text or len(text) > 500:
                    return 400, {"error": "task text required (<= 500 characters)"}
                return 201, {"id": s.submit(text, int(data.get("priority", 0)))}
            if method == "POST" and len(path) == 3 and path[0] == "tasks":
                tid, action = path[1], path[2]
                if action == "cancel":
                    return 200, {"result": s.cancel(tid)}
                if action == "resume":
                    s.resume(tid)
                    return 200, {"result": "resume allowed"}
                if action == "priority":
                    s.set_priority(tid, int(data["priority"]))
                    return 200, {"result": "priority set"}
        except ValueError as error:  # e.g. a request the robot cannot represent
            return 422, {"error": str(error)}
        except KeyError:
            return 404, {"error": "unknown task"}
        return 404, {"error": "not found"}

    def serve(self, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
        api = self

        class Handler(BaseHTTPRequestHandler):
            def _reply(self, method):
                n = int(self.headers.get("Content-Length") or 0)
                code, out = api.handle(method, self.path, self.headers.get("Authorization"),
                                       self.rfile.read(n) if n else b"")  # fmt: skip
                raw = out.encode() if isinstance(out, str) else json.dumps(out).encode()
                self.send_response(code)
                kind = "text/html" if isinstance(out, str) else "application/json"
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):  # noqa: N802
                self._reply("GET")

            def do_POST(self):  # noqa: N802
                self._reply("POST")

            def log_message(self, *args):  # requests are not logged: headers carry the token
                pass

        server = ThreadingHTTPServer((host, port), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server
