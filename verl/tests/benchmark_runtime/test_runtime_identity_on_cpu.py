"""Check runtime process identity over HTTP and during a conflicting launch."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = ROOT / "verl/benchmark_runtime"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


server = load("identity_server", RUNTIME_DIR / "server.py")
client = load("identity_client", RUNTIME_DIR / "client.py")


class Adapter:
    name = "identity_fixture"

    def health(self):
        return {}


class ExclusiveServer(server.BenchmarkHTTPServer):
    # Ensure the conflicting-launch test also has exclusive ownership on Windows.
    allow_reuse_address = False


class LegacyRuntime(server.Runtime):
    def dispatch(self, method, params):
        result = super().dispatch(method, params)
        if method == "health":
            for key in ("pid", "runtime_id", "started_at_unix_s", "server_sha256"):
                result.pop(key, None)
        return result


class TestRuntimeIdentity(unittest.TestCase):
    @contextmanager
    def running(self, runtime=None):
        runtime = runtime or server.Runtime(Adapter())
        httpd = ExclusiveServer(("127.0.0.1", 0), server.RequestHandler)
        httpd.runtime = runtime
        listener = threading.Thread(target=lambda: httpd.serve_forever(poll_interval=0.01), daemon=True)
        listener.start()
        rpc = client.RuntimeClient(f"http://127.0.0.1:{httpd.server_port}", timeout_s=2)
        try:
            yield runtime, httpd, rpc
        finally:
            httpd.shutdown()
            listener.join(timeout=3)
            httpd.server_close()
            runtime.close()
            self.assertFalse(listener.is_alive())

    def test_http_health_identifies_the_process_and_loaded_source_without_creating_sessions(self):
        before = time.time()
        with self.running() as (_, _, rpc):
            expected_hash = hashlib.sha256((RUNTIME_DIR / "server.py").read_bytes()).hexdigest()
            first = rpc.health("identity_fixture", os.getpid(), expected_hash)
            second = rpc.health("identity_fixture", os.getpid(), expected_hash)
            self.assertEqual(first, second)
            self.assertEqual(first["active_sessions"], 0)
            self.assertEqual(first["python"], sys.executable)
            self.assertLessEqual(before, first["started_at_unix_s"])
            self.assertLessEqual(first["started_at_unix_s"], time.time())
            self.assertEqual(len(first["runtime_id"]), 32)
            self.assertNotEqual(first["runtime_id"], server.Runtime(Adapter()).identity["runtime_id"])

    def test_http_health_rejects_wrong_process_or_loaded_source(self):
        with self.running() as (_, _, rpc):
            with self.assertRaisesRegex(client.RuntimeIdentityError, "expected pid="):
                rpc.health("identity_fixture", expected_pid=os.getpid() + 1)
            with self.assertRaisesRegex(client.RuntimeIdentityError, "expected server_sha256="):
                rpc.health("identity_fixture", expected_server_sha256="0" * 64)

    def test_old_servers_remain_compatible_only_when_identity_is_not_required(self):
        with self.running(LegacyRuntime(Adapter())) as (_, _, rpc):
            health = rpc.health("identity_fixture")
            self.assertNotIn("pid", health)
            with self.assertRaisesRegex(client.RuntimeIdentityError, "got None"):
                rpc.health("identity_fixture", expected_pid=os.getpid())
            with self.assertRaisesRegex(client.RuntimeIdentityError, "got None"):
                rpc.health("identity_fixture", expected_server_sha256=server.SERVER_SHA256)

    def test_loaded_hash_does_not_change_when_the_source_on_disk_is_updated(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "server_copy.py"
            source.write_bytes((RUNTIME_DIR / "server.py").read_bytes())
            loaded = load("identity_server_copy", source)
            runtime = loaded.Runtime(Adapter())
            expected_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            source.write_bytes(source.read_bytes() + b"\n# Updated after this process loaded its code.\n")
            self.assertNotEqual(expected_hash, hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(runtime.dispatch("health", {})["server_sha256"], expected_hash)

    def test_old_listener_cannot_validate_a_still_running_new_child(self):
        with self.running() as (_, httpd, rpc), tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            entered, release = directory / "entered", directory / "release"
            adapter_path = directory / "adapter.py"
            adapter_path.write_text(
                "import time\nfrom pathlib import Path\n"
                "class Adapter:\n"
                "    name = 'identity_fixture'\n"
                "    def health(self):\n"
                f"        Path({str(entered)!r}).touch()\n"
                f"        release = Path({str(release)!r})\n"
                "        deadline = time.monotonic() + 10\n"
                "        while not release.exists():\n"
                "            if time.monotonic() >= deadline:\n"
                "                raise RuntimeError('test did not release adapter initialization')\n"
                "            time.sleep(0.01)\n"
                "        return {}\n",
                encoding="utf-8",
            )
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    str(RUNTIME_DIR / "server.py"),
                    "--adapter",
                    str(adapter_path),
                    "--port",
                    str(httpd.server_port),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                deadline = time.monotonic() + 5
                while not entered.exists() and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(entered.exists(), "Child did not reach adapter initialization")
                self.assertIsNone(process.poll())
                # This is the old recipe's false positive: a healthy port and
                # a live new child can coexist before the child attempts bind.
                self.assertEqual(rpc.health("identity_fixture")["pid"], os.getpid())
                probe = subprocess.run(
                    [
                        sys.executable,
                        str(RUNTIME_DIR / "client.py"),
                        "--endpoint",
                        rpc.endpoint,
                        "--backend",
                        "identity_fixture",
                        "--wait-s",
                        "30",
                        "--expect-pid",
                        str(process.pid),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                self.assertNotEqual(probe.returncode, 0)
                self.assertIn("RuntimeIdentityError", probe.stderr)
                self.assertIn(f"expected pid={process.pid}", probe.stderr)
                self.assertEqual(probe.stdout, "")
                self.assertIsNone(process.poll(), "Child must still be alive during the identity check")
                release.touch()
                output, _ = process.communicate(timeout=5)
                self.assertNotEqual(process.returncode, 0)
                self.assertIn("OSError", output)
                self.assertNotIn("Benchmark runtime listening", output)
                self.assertFalse(any(line.startswith('{"') for line in output.splitlines()), output)
            finally:
                release.touch()
                if process.poll() is None:
                    process.terminate()
                process.communicate(timeout=5)

    def test_cli_can_require_the_actual_child_pid_and_loaded_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter_path = Path(directory) / "adapter.py"
            adapter_path.write_text(
                "class Adapter:\n    name = 'identity_fixture'\n    def health(self):\n        return {}\n",
                encoding="utf-8",
            )
            process = subprocess.Popen(
                [sys.executable, "-u", str(RUNTIME_DIR / "server.py"), "--adapter", str(adapter_path), "--port", "0"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                # The server writes these only after a successful bind.
                ready = json.loads(process.stdout.readline())
                listening = process.stdout.readline().strip()
                self.assertEqual(ready["pid"], process.pid)
                self.assertEqual(ready["active_sessions"], 0)
                endpoint = listening.removeprefix("Benchmark runtime listening on ")
                result = subprocess.run(
                    [
                        sys.executable,
                        str(RUNTIME_DIR / "client.py"),
                        "--endpoint",
                        endpoint,
                        "--backend",
                        "identity_fixture",
                        "--expect-pid",
                        str(process.pid),
                        "--expect-server-sha256",
                        ready["server_sha256"],
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout), ready)
            finally:
                if process.poll() is None:
                    process.terminate()
                process.communicate(timeout=5)


if __name__ == "__main__":
    unittest.main()
