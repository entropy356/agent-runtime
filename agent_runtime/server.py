"""Loopback dashboard with serialized operations and an optional fixed clock."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import sqlite3
import threading


def serve(runtime, port=8765, interval=10):
    if not math.isfinite(interval) or interval < 0.25:
        raise ValueError("Clock interval must be finite and at least 0.25 seconds")
    lock, stopped = threading.RLock(), threading.Event()
    clock = {"running": False, "interval": interval, "error": None}
    page = Path(__file__).with_name("index.html").read_bytes()

    def ticker():
        while not stopped.wait(interval):
            with lock:
                if clock["running"]:
                    try:
                        runtime.step()
                    except Exception:
                        clock.update(running=False, error="自动运行失败，已暂停；请手动运行查看错误。")

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, value):
            body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def local_request(self):
            # Bind loopback and reject browser cross-origin writes / DNS rebinding.
            host = self.headers.get("Host", "")
            valid = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            origin = self.headers.get("Origin")
            return host in valid and (origin is None or origin == "http://" + host)

        def do_GET(self):
            if not self.local_request():
                self.respond(403, {"error": "Local same-origin requests only"})
                return
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(page)
            elif self.path == "/api/state":
                with lock:
                    self.respond(200, {**runtime.inspect(), "clock": dict(clock)})
            else:
                self.respond(404, {"error": "Not found"})

        def do_POST(self):
            if not self.local_request():
                self.respond(403, {"error": "Local same-origin requests only"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16000:
                    raise ValueError("Invalid request size")
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    raise ValueError("Request must be an object")
                with lock:
                    if self.path == "/api/step":
                        enabled = data.get("enabled", True)
                        if type(enabled) is not bool:
                            raise ValueError("enabled must be boolean")
                        result = runtime.step(data.get("text"), enabled)
                    elif self.path == "/api/seed":
                        result = runtime.seed()
                    elif self.path == "/api/save":
                        runtime.save(data.get("name"))
                        result = {"ok": True}
                    elif self.path == "/api/restore":
                        runtime.restore(data.get("name"))
                        clock["running"] = False
                        result = {"ok": True}
                    elif self.path == "/api/compare":
                        # Pausing the scheduler keeps the workspace fixed while inspecting results.
                        clock["running"] = False
                        result = runtime.compare(data.get("text"), data.get("baseline"), data.get("a"), data.get("b"))
                    elif self.path == "/api/clock":
                        if type(data.get("running")) is not bool:
                            raise ValueError("running must be boolean")
                        clock.update(running=data["running"], error=None)
                        result = dict(clock)
                    else:
                        self.respond(404, {"error": "Not found"})
                        return
                self.respond(200, result)
            except (ValueError, TypeError, sqlite3.IntegrityError) as error:
                self.respond(400, {"error": str(error)})
            except Exception:
                self.respond(500, {"error": "Operation failed; inspect the local configuration and retry"})

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=ticker, daemon=True)
    thread.start()
    print(f"Agent Runtime [{runtime.provider.mode}] http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stopped.set()
        server.server_close()
