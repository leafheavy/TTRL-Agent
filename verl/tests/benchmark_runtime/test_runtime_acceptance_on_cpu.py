"""Verify runtime HTTP acceptance and idle TTL cleanup without BFCL or GPUs."""

import importlib.util
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


server_module = load("runtime_acceptance_server", "verl/benchmark_runtime/server.py")
client_module = load("runtime_acceptance_client", "verl/benchmark_runtime/client.py")


class Session:
    def __init__(self, value):
        self.value = value
        self.tool_schemas = [{"name": f"tool_{value}"}]
        self.closed = threading.Event()

    def execute(self, calls):
        self.value += calls.get("increment", 1)
        return self.value

    def close(self):
        self.closed.set()


class Adapter:
    name = "acceptance_fixture"

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def health(self):
        return {}

    def create_session(self, context):
        if context.get("block"):
            self.entered.set()
            if not self.release.wait(timeout=8):
                raise RuntimeError("Test did not release the slow session initializer")
        return Session(context.get("value", 0))


class CountingServer(server_module.BenchmarkHTTPServer):
    cleanup_interval_s = 0.01

    def __init__(self, runtime, expected=0):
        self.accepted = 0
        self.expected = expected
        self.all_accepted = threading.Event()
        super().__init__(("127.0.0.1", 0), server_module.RequestHandler)
        self.runtime = runtime

    def process_request(self, request, client_address):
        # This executes on the listening thread before dispatching the handler.
        # It proves acceptance can advance while handlers wait on Runtime.lock.
        self.accepted += 1
        if self.accepted >= self.expected:
            self.all_accepted.set()
        return super().process_request(request, client_address)


class ObservedRuntime(server_module.Runtime):
    def __init__(self, adapter, **kwargs):
        super().__init__(adapter, **kwargs)
        self.cleanup_observed = threading.Event()
        self.cleanup_thread = None

    def reap_expired(self):
        if threading.current_thread() is not threading.main_thread():
            self.cleanup_thread = threading.current_thread()
            self.cleanup_observed.set()
        return super().reap_expired()


class TestRuntimeAcceptance(unittest.TestCase):
    @contextmanager
    def running(self, runtime, expected=0):
        server = CountingServer(runtime, expected)
        listener = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
        listener.start()
        rpc = client_module.RuntimeClient(f"http://127.0.0.1:{server.server_port}", timeout_s=8)
        try:
            yield server, rpc
        finally:
            server.shutdown()
            listener.join(timeout=3)
            server.server_close()
            self.assertFalse(listener.is_alive(), "The HTTP accept thread did not stop")
            self.assertIsNotNone(server._reaper_thread)
            self.assertFalse(server._reaper_thread.is_alive(), "The TTL cleanup thread survived shutdown")

    def test_slow_session_creation_does_not_block_accepting_32_real_http_requests(self):
        adapter = Adapter()
        runtime = server_module.Runtime(adapter)
        with self.running(runtime, expected=32) as (server, rpc):
            with ThreadPoolExecutor(max_workers=32) as executor:
                first = executor.submit(
                    rpc.call, "create_session", session_id="s0", context={"value": 0, "block": True}
                )
                try:
                    self.assertTrue(adapter.entered.wait(timeout=3), "The first create never acquired the runtime lock")
                    pending = [
                        executor.submit(rpc.call, "create_session", session_id=f"s{index}", context={"value": index})
                        for index in range(1, 32)
                    ]
                    # The initializer is still blocked, so none of these extra
                    # requests can complete Runtime.dispatch. They must still
                    # all be accepted by the independent listener.
                    self.assertTrue(server.all_accepted.wait(timeout=3), "Session creation blocked HTTP acceptance")
                    self.assertFalse(adapter.release.is_set())
                    self.assertFalse(first.done())
                    self.assertTrue(all(not future.done() for future in pending))
                finally:
                    adapter.release.set()
                results = [first.result(timeout=3)] + [future.result(timeout=3) for future in pending]
            self.assertEqual(len(runtime.sessions), 32)
            for index, result in enumerate(results):
                self.assertEqual(result["tool_schemas"], [{"name": f"tool_{index}"}])
            # Removing acceptance contention must not merge independent tool
            # environments or bypass the runtime's existing session locks.
            for session_id, expected in (("s0", 1), ("s1", 2), ("s0", 2)):
                result = rpc.call(
                    "session_call", session_id=session_id, operation="execute", arguments={"calls": {}}
                )
                self.assertEqual(result["value"], expected)
            runtime.close()

    def test_expired_session_is_reaped_without_any_incoming_request(self):
        runtime = ObservedRuntime(Adapter(), session_ttl_s=0.1)
        runtime.dispatch("create_session", {"session_id": "expired", "context": {"value": 7}})
        session = runtime.sessions["expired"]["session"]
        runtime.sessions["expired"]["last_seen"] = time.monotonic() - 1
        with self.running(runtime) as (server, _):
            self.assertTrue(session.closed.wait(timeout=3), "Idle expiry required another HTTP request")
            self.assertTrue(runtime.cleanup_observed.is_set())
            self.assertIs(runtime.cleanup_thread, server._reaper_thread)
            self.assertEqual(server.accepted, 0)
            self.assertNotIn("expired", runtime.sessions)
        self.assertFalse(runtime.cleanup_thread.is_alive())

    def test_idle_maintenance_keeps_unexpired_session_usable(self):
        runtime = ObservedRuntime(Adapter(), session_ttl_s=60)
        runtime.dispatch("create_session", {"session_id": "active", "context": {"value": 10}})
        session = runtime.sessions["active"]["session"]
        with self.running(runtime) as (_, rpc):
            self.assertTrue(runtime.cleanup_observed.wait(timeout=3))
            self.assertFalse(session.closed.is_set())
            self.assertIn("active", runtime.sessions)
            result = rpc.call(
                "session_call", session_id="active", operation="execute", arguments={"calls": {"increment": 3}}
            )
            self.assertEqual(result["value"], 13)
            runtime.close()


if __name__ == "__main__":
    unittest.main()
