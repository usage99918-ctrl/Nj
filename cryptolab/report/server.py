"""Token-protected dashboard server (standard library only).

  python -m cryptolab serve [--host 127.0.0.1] [--port 8787]

* Every request needs the access token (printed at start; stored in data/run/dashboard_token):
  open http://HOST:PORT/?token=...  -- a cookie keeps you signed in.
* Read-only except ONE emergency action: POST /kill-switch engages the paper-trading kill switch.
  Releasing it is deliberately only possible from the host (`cryptolab kill-switch off`).
* Default bind is 127.0.0.1 (this computer only). For phone access use an encrypted private
  network such as Tailscale, or a Cloudflare Tunnel, instead of exposing the port publicly;
  see docs/INSTALL.md.
"""
from __future__ import annotations

import hmac
import json
import logging
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

log = logging.getLogger(__name__)


def get_token(cfg) -> str:
    p = cfg.path("data_dir") / "run" / "dashboard_token"
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text(secrets.token_urlsafe(24))
    return p.read_text().strip()


def serve(cfg, host: str = "127.0.0.1", port: int = 8787, cache_s: int = 30) -> None:
    from cryptolab.live.paper import kill_switch_file
    from cryptolab.live.supervisor import record_event
    from cryptolab.live.supervisor import status as sup_status
    from cryptolab.report.dashboard import build_all
    from cryptolab.report.metrics import continuity, experiment_summary
    from cryptolab.storage.db import open_store

    token = get_token(cfg)
    lock = threading.Lock()
    cache = {"t": 0.0, "html": b""}

    def page() -> bytes:
        with lock:
            if time.time() - cache["t"] > cache_s:
                build_all(cfg)
                html = (cfg.path("reports_dir") / "dashboard.html").read_text(encoding="utf-8")
                html = html.replace("</header>", '</header><form method="post" action="/kill-switch" '
                                    'onsubmit="return window.confirm(\'Engage the paper-trading kill switch?\')">'
                                    '<button class="pill pill-crit" type="submit">Emergency stop paper trading</button></form>', 1)
                html = html.replace("<head>", '<head>\n<meta http-equiv="refresh" content="60">', 1)
                cache.update(t=time.time(), html=html.encode("utf-8"))
            return cache["html"]

    class H(BaseHTTPRequestHandler):
        def _authed(self) -> bool:
            q = parse_qs(urlparse(self.path).query).get("token", [""])[0]
            c = self.headers.get("Cookie", "")
            ck = next((x.split("=", 1)[1] for x in c.split("; ") if x.startswith("cl_token=")), "")
            return hmac.compare_digest(q, token) or hmac.compare_digest(ck, token)

        def _send(self, code: int, body: bytes, ctype: str = "text/html; charset=utf-8", cookie: bool = False):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            if cookie:
                self.send_header("Set-Cookie", f"cl_token={token}; HttpOnly; SameSite=Strict; Path=/; Max-Age=2592000")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self._authed():
                return self._send(401, b"Access token required: open /?token=YOUR_TOKEN", "text/plain")
            path = urlparse(self.path).path
            set_cookie = "token=" in self.path
            if path == "/":
                return self._send(200, page(), cookie=set_cookie)
            if path == "/api/status":
                store = open_store(cfg)
                body = {"supervisor": sup_status(cfg), "continuity": continuity(store, cfg),
                        "experiments": {k: {x: v.get(x) for x in ("forecasts", "scored", "gaps", "accuracy")}
                                        for k, v in ((k, experiment_summary(store, e["name"]))
                                                     for k, e in cfg["experiments"].items())},
                        "kill_switch": kill_switch_file(cfg).exists()}
                return self._send(200, json.dumps(body, default=str).encode(), "application/json")
            return self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if not self._authed():
                return self._send(401, b"unauthorized", "text/plain")
            if urlparse(self.path).path == "/kill-switch":
                kill_switch_file(cfg).write_text(f"engaged from dashboard {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
                record_event(cfg, "kill_switch_requested_from_dashboard", client=self.client_address[0])
                cache["t"] = 0
                self.send_response(303)
                self.send_header("Location", "/")
                self.end_headers()
                return
            return self._send(404, b"not found", "text/plain")

        def log_message(self, fmt, *args):
            log.info("dashboard %s %s", self.client_address[0], fmt % args)

    srv = ThreadingHTTPServer((host, port), H)
    print(f"Dashboard: http://{host}:{port}/?token={token}")
    if host not in ("127.0.0.1", "localhost"):
        print("WARNING: listening beyond this computer. Prefer Tailscale/Cloudflare Tunnel; never port-forward without TLS.")
    srv.serve_forever()
