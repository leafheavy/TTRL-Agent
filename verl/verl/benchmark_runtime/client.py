"""Standard-library JSON client; never imports a benchmark or GPU library."""

import asyncio
import json
import math
import warnings
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener
from uuid import uuid4


class BenchmarkRuntimeError(RuntimeError):
    """Transport or remote adapter failure; never convert it into a task reward."""


class RuntimeClient:
    def __init__(self, endpoint, timeout_s=120):
        if not isinstance(endpoint, str) or not endpoint:
            raise ValueError(
                "Set benchmark_runtime.endpoint and start the runtime in the benchmark's own virtual environment"
            )
        parsed = urlsplit(endpoint)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError("benchmark_runtime.endpoint must be an HTTP(S) URL without a query or fragment")
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
            raise ValueError("benchmark_runtime.timeout_s must be a positive number")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("benchmark_runtime.timeout_s must be a positive finite number")
        self.endpoint = endpoint.rstrip("/")
        self.timeout_s = timeout_s

    @classmethod
    def from_config(cls, config):
        config = config or {}
        return cls(config.get("endpoint"), config.get("timeout_s", 120))

    def call(self, method, **params):
        request_id = uuid4().hex
        payload = json.dumps(
            {"version": 1, "id": request_id, "method": method, "params": params}, allow_nan=False
        ).encode("utf-8")
        request = Request(self.endpoint + "/rpc", data=payload, headers={"Content-Type": "application/json"})
        try:
            # Local runtime traffic must not inherit HTTP_PROXY from the training job.
            with build_opener(ProxyHandler({})).open(request, timeout=self.timeout_s) as response:
                result = json.loads(response.read())
        except (URLError, OSError, ValueError) as exc:
            raise BenchmarkRuntimeError(f"Benchmark runtime {method} failed at {self.endpoint}: {exc}") from exc
        if not isinstance(result, dict) or result.get("id") != request_id or result.get("version") != 1:
            raise BenchmarkRuntimeError("Benchmark runtime returned an incompatible response")
        if "error" in result:
            error = result["error"]
            raise BenchmarkRuntimeError(f"Benchmark runtime {method}: {error['type']}: {error['message']}")
        if "result" not in result:
            raise BenchmarkRuntimeError("Benchmark runtime response has no result")
        return result["result"]

    def health(self, expected_backend=None):
        result = self.call("health")
        if expected_backend is not None and result.get("backend") != expected_backend:
            raise BenchmarkRuntimeError(f"Expected {expected_backend!r} runtime, got {result.get('backend')!r}")
        return result


class RemoteSession:
    """One remote environment per rollout; only JSON values cross the boundary."""

    def __init__(self, context, runtime_config=None):
        self.client = RuntimeClient.from_config(runtime_config)
        self.session_id = uuid4().hex
        self.closed = False
        try:
            result = self.client.call("create_session", session_id=self.session_id, context=context)
            self.tool_schemas = result["tool_schemas"]
        except BaseException:
            self.close()
            raise

    def _call(self, method, **params):
        if self.closed:
            raise BenchmarkRuntimeError("Benchmark session is already closed")
        result = self.client.call("session_call", session_id=self.session_id, operation=method, arguments=params)
        self.tool_schemas = result["tool_schemas"]
        return result["value"]

    def parse_calls(self, response):
        return self._call("parse_calls", response=response)

    def execute(self, calls):
        return self._call("execute", calls=calls)

    def malformed_response(self, reason=None):
        return self._call("malformed_response", **({"reason": reason} if reason is not None else {}))

    def finish_turn(self, reply):
        return self._call("finish_turn", reply=reply)

    def outcome(self, termination):
        return self._call("outcome", termination=termination)

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                self.client.call("close_session", session_id=self.session_id)
            except Exception as exc:
                # Cleanup must not replace an earlier generation/cancellation failure.
                warnings.warn(f"Benchmark session cleanup failed; server TTL will reclaim it: {exc}", stacklevel=2)


async def open_remote_session(context, runtime_config=None, session_class=RemoteSession):
    """Create without blocking generation; clean up even if creation is cancelled."""
    loop = asyncio.get_running_loop()
    pending = loop.run_in_executor(None, session_class, context, runtime_config)
    try:
        return await asyncio.shield(pending)
    except BaseException:

        def release(future):
            try:
                session = future.result()
            except BaseException:
                return
            loop.run_in_executor(None, session.close)

        pending.add_done_callback(release)
        raise


def main():
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Check a benchmark runtime without importing benchmark libraries")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--wait-s", type=float, default=0)
    args = parser.parse_args()
    deadline = time.monotonic() + args.wait_s
    client = RuntimeClient(args.endpoint, timeout_s=2)
    while True:
        try:
            print(json.dumps(client.health(args.backend), ensure_ascii=False))
            return
        except BenchmarkRuntimeError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(min(1, max(0, deadline - time.monotonic())))


if __name__ == "__main__":
    main()
