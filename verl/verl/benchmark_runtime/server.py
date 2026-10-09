"""Run by file path in a benchmark venv; does not import the verl package."""

import argparse
import importlib.util
import json
import math
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import RLock

SESSION_OPERATIONS = frozenset({"parse_calls", "execute", "malformed_response", "finish_turn", "outcome"})
MAX_REQUEST_BYTES = 32 * 1024 * 1024


class Runtime:
    def __init__(self, adapter, session_ttl_s=3600):
        if not math.isfinite(session_ttl_s) or session_ttl_s <= 0:
            raise ValueError("session_ttl_s must be positive and finite")
        self.adapter = adapter
        self.session_ttl_s = session_ttl_s
        self.sessions = {}
        self.lock = RLock()

    def reap_expired(self):
        now = time.monotonic()
        with self.lock:
            for session_id, record in list(self.sessions.items()):
                if now - record["last_seen"] > self.session_ttl_s and record["lock"].acquire(blocking=False):
                    try:
                        record["session"].close()
                        del self.sessions[session_id]
                    finally:
                        record["lock"].release()

    def dispatch(self, method, params):
        self.reap_expired()
        if method == "health":
            return {
                **self.adapter.health(),
                "backend": self.adapter.name,
                "python": sys.executable,
                "active_sessions": len(self.sessions),
            }
        if method == "create_session":
            session_id = params["session_id"]
            if not isinstance(session_id, str) or not session_id:
                raise ValueError("session_id must be a non-empty string")
            with self.lock:
                if session_id in self.sessions:
                    raise ValueError("Duplicate session_id")
                session = self.adapter.create_session(params["context"])
                self.sessions[session_id] = {"session": session, "lock": RLock(), "last_seen": time.monotonic()}
            return {"tool_schemas": session.tool_schemas}
        if method == "close_session":
            with self.lock:
                record = self.sessions.pop(params["session_id"], None)
            if record is not None:
                with record["lock"]:
                    record["session"].close()
            return None
        if method == "session_call":
            operation = params["operation"]
            if operation not in SESSION_OPERATIONS:
                raise ValueError(f"Unsupported session operation: {operation}")
            with self.lock:
                record = self.sessions[params["session_id"]]
            with record["lock"]:
                try:
                    value = getattr(record["session"], operation)(**params.get("arguments", {}))
                    return {"value": value, "tool_schemas": record["session"].tool_schemas}
                finally:
                    record["last_seen"] = time.monotonic()
        if method == "decode_responses":
            return self.adapter.decode_responses(**params)
        if method == "score_outcomes":
            return self.adapter.score_outcomes(**params)
        raise ValueError(f"Unsupported runtime method: {method}")

    def close(self):
        with self.lock:
            records, self.sessions = list(self.sessions.values()), {}
        for record in records:
            with record["lock"]:
                record["session"].close()


class RequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/rpc":
            self.send_error(404)
            return
        request_id = None
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_REQUEST_BYTES:
                raise ValueError("Invalid request size")
            request = json.loads(self.rfile.read(length))
            request_id = request.get("id")
            if request.get("version") != 1 or not isinstance(request.get("params"), dict):
                raise ValueError("Expected protocol version 1 and an object of params")
            result = self.server.runtime.dispatch(request["method"], request["params"])
            response = {"version": 1, "id": request_id, "result": result}
            encoded = json.dumps(response, allow_nan=False).encode("utf-8")
        except Exception as exc:
            traceback.print_exc()
            encoded = json.dumps(
                {"version": 1, "id": request_id, "error": {"type": type(exc).__name__, "message": str(exc)}}
            ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        try:
            self.wfile.write(encoded)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # A disconnected caller's abandoned session is reclaimed by the TTL.

    def log_message(self, format, *args):
        pass


def load_adapter(path):
    spec = importlib.util.spec_from_file_location("_benchmark_adapter", Path(path).resolve())
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Adapter()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, help="Python file defining Adapter; loaded only in this process")
    parser.add_argument("--host", default="127.0.0.1", choices=("127.0.0.1", "localhost"))
    parser.add_argument("--port", type=int, default=1054)
    parser.add_argument("--session-ttl-s", type=float, default=3600)
    args = parser.parse_args()
    runtime = Runtime(load_adapter(args.adapter), args.session_ttl_s)
    # Check adapter imports before advertising a healthy runtime.
    print(json.dumps(runtime.dispatch("health", {})), flush=True)
    with ThreadingHTTPServer((args.host, args.port), RequestHandler) as server:
        server.runtime = runtime
        server.timeout = 1
        print(f"Benchmark runtime listening on http://{args.host}:{server.server_port}", flush=True)
        try:
            while True:
                server.handle_request()
                runtime.reap_expired()
        except KeyboardInterrupt:
            pass
        finally:
            runtime.close()


if __name__ == "__main__":
    main()
