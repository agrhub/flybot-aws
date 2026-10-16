"""Local stand-in for the vision Lambda (docs/CONTRACTS.md section 1, local development).

    uv run python -m sentinel.lambda_local            # POST http://localhost:9001/invoke, loopback only
    uv run python -m sentinel.lambda_local --port 9001 --host 127.0.0.1

Same JSON in and out as `Lambda.Invoke`; errors come back as HTTP 400 / 404 / 503 / 500 with the contract's
`{"v": 1, "error": {...}}` body. `GET /health` reports the OpenCV version and the detector. No auth: it binds
127.0.0.1 only (use --host 0.0.0.0 on purpose, e.g. inside docker compose, never on a public host).

Env: the handler's (DETECTOR, DETECTOR_ONNX, EVIDENCE_DIR, PHOTOS_BUCKET, ANTHROPIC_API_KEY, SENTINEL_PLANNER...).
`uv run --env-file .env python -m sentinel.lambda_local` loads a local .env.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from sentinel import handler as h

MAX_BODY = 6 * 1024 * 1024  # Lambda's own payload limit: photos travel by reference


class Handler(BaseHTTPRequestHandler):
    server_version = "flybot-vision-local/1"

    def _send(self, status: int, body: dict) -> None:
        raw = json.dumps(body, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", ""):
            try:
                from sentinel import require_opencv5
                from sentinel.vision.detector import model_info

                self._send(200, {"ok": True, "opencv": require_opencv5(), "model": model_info(),
                                 "planner": os.environ.get("SENTINEL_PLANNER", "rule"),
                                 "llm": bool(os.environ.get("ANTHROPIC_API_KEY"))})
            except Exception as e:  # noqa: BLE001
                self._send(503, {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]})
            return
        self._send(404, {"v": 1, "error": {"code": "bad_request", "message": "POST /invoke"}})

    def do_POST(self):
        if self.path.rstrip("/") != "/invoke":
            self._send(404, {"v": 1, "error": {"code": "bad_request", "message": "POST /invoke"}})
            return
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            self._send(413, {"v": 1, "error": {"code": "bad_request", "message": "body over 6 MB: send photos by reference"}})
            return
        raw = self.rfile.read(n)
        try:
            self._send(200, h.analyze(raw))
        except h.VisionError as e:
            self._send(e.status, e.payload())
        except Exception as e:  # noqa: BLE001
            self._send(500, h.Internal(f"{type(e).__name__}: {e}"[:300]).payload())

    def log_message(self, fmt, *args):  # one JSON line per request, like the Lambda
        print(json.dumps({"msg": "http", "client": self.client_address[0], "line": fmt % args}))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "9001")))
    a = ap.parse_args(argv)
    if a.host not in ("127.0.0.1", "localhost", "::1") and not os.environ.get("LAMBDA_LOCAL_ALLOW_REMOTE"):
        print("refusing a non-loopback host without LAMBDA_LOCAL_ALLOW_REMOTE=1 (no auth on this server)", file=sys.stderr)
        return 2
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"vision lambda local: POST http://{a.host}:{a.port}/invoke  (GET /health)  detector={os.environ.get('DETECTOR', 'yolov7')}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
