"""Check CLI resource teardown without loading Ray, CUDA, or the training stack."""

import ast
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import psutil

SOURCE = Path(__file__).resolve().parents[2] / "verl/trainer/main_ppo.py"


def load_teardown(ray=None, process_api=None):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("main", "_shutdown_ray")]
    for node in functions:
        node.decorator_list = []
    namespace = {
        "ray": ray if ray is not None else Mock(),
        "psutil": process_api if process_api is not None else psutil,
        "time": time,
        "os": os,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])),
                 str(SOURCE), "exec"), namespace)
    return namespace


class TestMainRayTeardown(unittest.TestCase):
    def test_successful_cli_waits_for_teardown_before_returning(self):
        namespace = load_teardown()
        order = []
        config = object()
        namespace["run_ppo"] = Mock(side_effect=lambda received: order.append(("run", received)))
        namespace["_shutdown_ray"] = Mock(side_effect=lambda: order.append(("shutdown", None)))
        namespace["main"](config)
        self.assertEqual(order, [("run", config), ("shutdown", None)])

    def test_failed_cli_still_runs_teardown_and_keeps_training_error(self):
        namespace = load_teardown()
        error = ValueError("second category failed during engine initialization")
        namespace["run_ppo"] = Mock(side_effect=error)
        namespace["_shutdown_ray"] = Mock()
        with self.assertRaises(ValueError) as raised:
            namespace["main"](object())
        self.assertIs(raised.exception, error)
        namespace["_shutdown_ray"].assert_called_once_with()

    def test_teardown_failure_is_not_reported_as_a_successful_category(self):
        namespace = load_teardown()
        namespace["run_ppo"] = Mock()
        namespace["_shutdown_ray"] = Mock(side_effect=RuntimeError("worker is still alive"))
        with self.assertRaisesRegex(RuntimeError, "worker is still alive"):
            namespace["main"](object())


class TestShutdownRayWorkers(unittest.TestCase):
    def test_snapshot_precedes_shutdown_and_only_survivors_receive_signals(self):
        for shutdown_fails in (False, True):
            with self.subTest(shutdown_fails=shutdown_fails):
                self.check_shutdown_order(shutdown_fails)

    def check_shutdown_order(self, shutdown_fails):
        order = []
        children = []
        for pid in (101, 102, 103):
            child = Mock(pid=pid)
            child.status.return_value = psutil.STATUS_RUNNING
            child.terminate.side_effect = lambda pid=pid: order.append(("terminate", pid))
            child.kill.side_effect = lambda pid=pid: order.append(("kill", pid))
            children.append(child)
        parent = Mock()
        parent.children.side_effect = lambda **kwargs: order.append("snapshot") or children
        remaining = iter([children[1:], children[2:], []])

        def wait(processes, timeout):
            order.append(("wait", [item.pid for item in processes]))
            return [], next(remaining)

        process_api = SimpleNamespace(
            Process=Mock(return_value=parent), wait_procs=Mock(side_effect=wait),
            NoSuchProcess=psutil.NoSuchProcess, STATUS_ZOMBIE=psutil.STATUS_ZOMBIE,
        )
        shutdown_error = RuntimeError("Ray shutdown failed")

        def shutdown():
            order.append("shutdown")
            if shutdown_fails:
                raise shutdown_error

        ray = SimpleNamespace(shutdown=Mock(side_effect=shutdown))
        namespace = load_teardown(ray, process_api)
        if shutdown_fails:
            with self.assertRaises(RuntimeError) as raised:
                namespace["_shutdown_ray"]()
            self.assertIs(raised.exception, shutdown_error)
        else:
            namespace["_shutdown_ray"]()
        parent.children.assert_called_once_with(recursive=True)
        ray.shutdown.assert_called_once_with()
        self.assertEqual(order, [
            "snapshot", "shutdown", ("wait", [101, 102, 103]),
            ("terminate", 102), ("terminate", 103), ("wait", [102, 103]),
            ("kill", 103), ("wait", [103]),
        ])

    def test_dead_workers_do_not_block_but_live_survivors_fail_the_category(self):
        for state in ("gone", "zombie", "gone_before_terminate", "running"):
            with self.subTest(state=state):
                child = Mock(pid=12345)
                if state == "gone":
                    child.status.side_effect = psutil.NoSuchProcess(child.pid)
                else:
                    child.status.return_value = (psutil.STATUS_ZOMBIE if state == "zombie"
                                                 else psutil.STATUS_RUNNING)
                parent = Mock()
                parent.children.return_value = [child]
                process_api = SimpleNamespace(
                    Process=Mock(return_value=parent),
                    wait_procs=Mock(side_effect=lambda processes, timeout: ([], list(processes))),
                    NoSuchProcess=psutil.NoSuchProcess, STATUS_ZOMBIE=psutil.STATUS_ZOMBIE,
                )
                if state == "gone_before_terminate":
                    child.terminate.side_effect = psutil.NoSuchProcess(child.pid)
                    process_api.wait_procs.side_effect = [([], [child]), ([child], [])]
                namespace = load_teardown(process_api=process_api)
                if state == "running":
                    with self.assertRaisesRegex(RuntimeError, "did not exit.*12345"):
                        namespace["_shutdown_ray"]()
                    child.terminate.assert_called_once_with()
                    child.kill.assert_called_once_with()
                else:
                    namespace["_shutdown_ray"]()
                    if state == "gone_before_terminate":
                        child.terminate.assert_called_once_with()
                    else:
                        child.terminate.assert_not_called()
                    child.kill.assert_not_called()


class TestRealDescendantTeardown(unittest.TestCase):
    def test_stubborn_child_is_gone_before_driver_exits_and_foreign_process_survives(self):
        # The worker has its own session on Linux, as Ray/vLLM subprocesses may.
        # The foreign process is a sibling, so the driver must never signal it.
        worker_code = """
import os, signal, sys, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(sys.argv[1]).write_text(str(os.getpid()))
while True:
    time.sleep(0.05)
"""
        driver_code = """
import ast, json, os, psutil, subprocess, sys, time
from pathlib import Path
from types import SimpleNamespace

source, ready_path, worker_code = sys.argv[1:]
tree = ast.parse(Path(source).read_text(encoding='utf-8'))
function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == '_shutdown_ray')
namespace = {'ray': SimpleNamespace(shutdown=lambda: None), 'psutil': psutil,
             'time': time, 'os': os}
exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
             source, 'exec'), namespace)
wait_procs = psutil.wait_procs
psutil.wait_procs = lambda processes, timeout=None, **kwargs: wait_procs(
    processes, timeout=min(timeout, 0.25) if timeout is not None else 0.25, **kwargs)
worker = subprocess.Popen([sys.executable, '-u', '-c', worker_code, ready_path],
                          start_new_session=os.name != 'nt')
deadline = time.monotonic() + 10
while not Path(ready_path).exists():
    if worker.poll() is not None or time.monotonic() >= deadline:
        raise RuntimeError('test worker did not become ready')
    time.sleep(0.02)
try:
    namespace['_shutdown_ray']()
    print(json.dumps({'pid': worker.pid, 'returncode': worker.poll()}), flush=True)
finally:
    if worker.poll() is None:
        worker.kill()
    worker.wait(timeout=5)
"""
        foreign = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                                   start_new_session=os.name != "nt")
        try:
            with tempfile.TemporaryDirectory(prefix="multi turn cleanup ") as temporary:
                ready = Path(temporary) / "worker.pid"
                try:
                    completed = subprocess.run(
                        [sys.executable, "-u", "-c", driver_code, str(SOURCE), str(ready), worker_code],
                        capture_output=True, text=True,
                        encoding="utf-8", errors="replace", timeout=20,
                    )
                    self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                    result = json.loads(completed.stdout.strip().splitlines()[-1])
                    self.assertIsNotNone(result["returncode"], "Driver returned before its worker exited")
                    self.assertFalse(psutil.pid_exists(result["pid"]), "Worker survived category completion")
                    self.assertIsNone(foreign.poll(), "Cleanup killed a process owned by another task")
                finally:
                    # Also reap this test's worker if a regression times out its driver.
                    if ready.exists():
                        try:
                            worker = psutil.Process(int(ready.read_text()))
                            worker.kill()
                            worker.wait(timeout=5)
                        except psutil.NoSuchProcess:
                            pass
        finally:
            foreign.terminate()
            foreign.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
