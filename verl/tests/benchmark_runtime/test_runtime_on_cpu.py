"""Exercise actual HTTP/process/venv boundaries without installing BFCL or Ray."""

import asyncio
import builtins
import importlib.util
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import venv
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = ROOT / "verl/benchmark_runtime"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


client = load("runtime_client_test", RUNTIME_DIR / "client.py")
server = load("runtime_server_test", RUNTIME_DIR / "server.py")


class TestOperationalRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.core = load("failure_modes_rules_test", RUNTIME_DIR / "failure_modes.py")
        cls.rules = load("operational_rules_test", RUNTIME_DIR / "bfcl_diagnostics.py")

    def diagnostics(self):
        return self.core.FailureDiagnostics(self.rules.BFCLFailureRules(["VehicleControlAPI"]), enabled=True)

    def test_vehicle_start_preconditions_and_explicit_recovery(self):
        call = {"name": "startEngine", "arguments": {"ignitionMode": "START"}}
        cases = [
            ("Brake pedal needs to be pressed when starting the engine.", "brake_not_pressed"),
            ("Must press the brake fully before starting the engine.", "brake_not_fully_pressed"),
            (
                "All doors must be locked before starting the engine. Here are the unlocked doors: driver, rear_left",
                "doors_not_locked",
            ),
        ]
        for error, rule in cases:
            with self.subTest(rule=rule):
                diagnostics = self.diagnostics()
                diagnostics.observe(call, json.dumps({"error": error}))
                event = diagnostics.snapshot("completed", True)[0]
                self.assertEqual(event["type"], "precondition_violation")
                self.assertEqual(event["rule"], rule)
                diagnostics.observe(call, {"engineState": "running", "fuelLevel": 50, "batteryVoltage": 12.5})
                events = diagnostics.snapshot("completed", True)
                self.assertEqual(len(events), 1)
                self.assertEqual(events[0]["recovery"], "recovered")

    def test_vehicle_cruise_requires_a_corresponding_success(self):
        diagnostics = self.diagnostics()
        call = {
            "name": "setCruiseControl",
            "arguments": {"speed": 60, "activate": True, "distanceToNextVehicle": 50},
        }
        diagnostics.observe(call, {"error": "Start the engine before activating the cruise control."})
        diagnostics.observe(call, {"cruiseStatus": "active", "currentSpeed": 40, "distanceToNextVehicle": 50})
        event = diagnostics.snapshot("response_length", False)[0]
        self.assertEqual(event["rule"], "engine_not_started")
        self.assertEqual(event["recovery"], "unknown")
        diagnostics.observe(call, {"cruiseStatus": "active", "currentSpeed": 60, "distanceToNextVehicle": 50})
        self.assertEqual(diagnostics.snapshot("completed", True)[0]["recovery"], "recovered")

    def test_vehicle_ambiguous_exceptions_and_partial_results_are_ignored(self):
        diagnostics = self.diagnostics()
        call = {"name": "startEngine", "arguments": {"ignitionMode": "START"}}
        for observation in (
            "Error during execution: connection timed out",
            {"error": "Service unavailable"},
            {"error": "All doors must be locked before starting the engine. Here are the unlocked doors: unknown"},
            {"engineState": "running"},
            None,
        ):
            diagnostics.observe(call, observation)
        self.assertEqual(diagnostics.snapshot("completed", True), [])


class CounterSession:
    def __init__(self, context):
        self.tool_schemas = []
        self.value = context.get("value", 0)
        self.closed = False

    def execute(self, calls):
        self.value += 1
        time.sleep(calls.get("delay", 0))
        if calls.get("fail"):
            raise ValueError("fixture tool failed")
        return self.value

    def close(self):
        self.closed = True


class CounterAdapter:
    name = "counter_benchmark"

    def health(self):
        return {}

    def create_session(self, context):
        return CounterSession(context)


class TestGenericProtocol(unittest.TestCase):
    def setUp(self):
        log_errors = patch.object(server.traceback, "print_exc")
        log_errors.start()
        self.addCleanup(log_errors.stop)
        self.runtime = server.Runtime(CounterAdapter(), session_ttl_s=60)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.RequestHandler)
        self.httpd.runtime = self.runtime
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.config = {"endpoint": f"http://127.0.0.1:{self.httpd.server_port}", "timeout_s": 2}
        self.rpc = client.RuntimeClient.from_config(self.config)

    def tearDown(self):
        self.doCleanups()
        self.httpd.shutdown()
        self.thread.join(timeout=3)
        self.httpd.server_close()
        self.runtime.close()

    def test_another_benchmark_and_parallel_session_isolation(self):
        self.assertEqual(self.rpc.health("counter_benchmark")["backend"], "counter_benchmark")
        sessions = [client.RemoteSession({"value": i * 10}, self.config) for i in range(6)]
        try:
            with ThreadPoolExecutor(max_workers=6) as pool:
                values = list(pool.map(lambda session: session.execute({}), sessions))
            self.assertEqual(values, [1, 11, 21, 31, 41, 51])
            with self.assertRaises(client.BenchmarkRuntimeError):
                self.rpc.health("bfcl")
        finally:
            for session in sessions:
                session.close()
                session.close()
        self.assertEqual(self.runtime.sessions, {})

    def test_tool_errors_and_timeouts_are_not_rewards_or_retries(self):
        session = client.RemoteSession({}, self.config)
        self.addCleanup(session.close)
        with self.assertRaisesRegex(client.BenchmarkRuntimeError, "fixture tool failed"):
            session.execute({"fail": True})
        record = self.runtime.sessions[session.session_id]
        self.assertEqual(record["session"].value, 1)
        session.client.timeout_s = 0.02
        with self.assertRaises(client.BenchmarkRuntimeError):
            session.execute({"delay": 0.1})
        time.sleep(0.15)
        self.assertEqual(record["session"].value, 2)
        session.client.timeout_s = 2

    def test_abandoned_session_expires_and_private_operations_are_rejected(self):
        session = client.RemoteSession({}, self.config)
        record = self.runtime.sessions[session.session_id]
        with self.assertRaises(client.BenchmarkRuntimeError):
            session._call("__getattribute__", name="value")
        record["last_seen"] -= 61
        self.rpc.health()
        self.assertTrue(record["session"].closed)
        self.assertNotIn(session.session_id, self.runtime.sessions)
        session.close()

    def test_missing_endpoint_cannot_fall_back_to_local_benchmark_imports(self):
        with self.assertRaisesRegex(ValueError, "benchmark_runtime.endpoint"):
            client.RemoteSession({})

    def test_cancellation_during_creation_closes_the_eventual_session(self):
        closed = threading.Event()
        started = threading.Event()

        class SlowSession:
            def __init__(self, context, config):
                started.set()
                time.sleep(0.1)

            def close(self):
                closed.set()

        async def cancel():
            task = asyncio.create_task(client.open_remote_session({}, {}, SlowSession))
            while not started.is_set():
                await asyncio.sleep(0.005)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(await asyncio.get_running_loop().run_in_executor(None, closed.wait, 3))

        asyncio.run(cancel())


FAKE_FILES = {
    "__init__.py": """import os
with open(os.environ["BFCL_AUDIT"], "a") as audit:
    audit.write("bfcl_import\\n")
""",
    "model_handler/local_inference/qwen_fc.py": """import json, re
class QwenFCHandler:
    @staticmethod
    def _extract_tool_calls(text):
        calls = []
        for payload in re.findall(r"<tool_call>\\n(.*?)\\n</tool_call>", text, re.S):
            try: calls.append(json.loads(payload))
            except ValueError: pass
        return calls
""",
    "model_handler/utils.py": """def convert_to_function_call(calls):
    return [name + "(" + ", ".join(key + "=" + repr(value) for key, value in args.items()) + ")"
            for call in calls for name, args in call.items()]
""",
    "eval_checker/multi_turn_eval/multi_turn_utils.py": """import ast, json
class FileSystem:
    def __init__(self): self.root, self.cwd = {}, "/"
def execute_multi_turn_func_call(calls, initial_config, classes, model, task, **kwargs):
    key = model + "_" + task + "_GorillaFileSystem_instance"
    if key not in globals(): globals()[key] = FileSystem()
    fs = globals()[key]
    results = []

    def target(path):
        parts = path.strip("/").split("/")
        parent = fs.root
        for part in parts[:-1]: parent = parent[part]
        return parent, parts[-1]

    for call in calls:
        node = ast.parse(call, mode="eval").body
        args = {item.arg: ast.literal_eval(item.value) for item in node.keywords}
        if node.func.id == "mkdir":
            parent, name = target(args["dir_name"])
            parent[name] = {}; result = None
        elif node.func.id == "rmdir":
            path = args["dir_name"]
            parent, name = target(path)
            if name not in parent: result = {"error": f"rmdir: cannot remove '{path}': No such file or directory"}
            elif parent[name]: result = {"error": f"rmdir: cannot remove '{path}': Directory not empty"}
            else:
                del parent[name]
                result = {"result": f"'{path}' removed"}
        elif node.func.id == "cd": fs.cwd = "/" + args["folder"]; result = None
        elif node.func.id == "pwd": result = fs.cwd
        else: raise ValueError("unknown fixture function")
        results.append(json.dumps(result))
    return results, {"GorillaFileSystem": fs}
""",
    "constants/eval_config.py": """from pathlib import Path
PROMPT_PATH = Path(__file__).parents[1] / "prompts"
POSSIBLE_ANSWER_PATH = Path(__file__).parents[1] / "answers"
""",
    "utils.py": """import json, os
from pathlib import Path
def find_file_by_category(category, directory): return Path(directory) / (category + ".json")
def load_file(path, use_lock=False):
    if Path(path).parent.name == "answers":
        with open(os.environ["BFCL_AUDIT"], "a") as audit: audit.write("oracle_read\\n")
    return json.loads(Path(path).read_text())
""",
    "eval_checker/multi_turn_eval/multi_turn_checker.py": """import os
def multi_turn_checker(trace, ground_truth, entry, category, model):
    with open(os.environ["BFCL_AUDIT"], "a") as audit: audit.write("checker_call\\n")
    flattened = [[call for step in turn for call in step] for turn in trace]
    return {"valid": flattened == ground_truth}
""",
}


def context(**overrides):
    functions = [
        {
            "name": "mkdir",
            "parameters": {"type": "object", "properties": {"dir_name": {"type": "string"}}, "required": ["dir_name"]},
        },
        {"name": "pwd", "parameters": {"type": "object", "properties": {}}},
    ]
    return {
        "functions": functions,
        "involved_classes": ["GorillaFileSystem"],
        "initial_config": {},
        "consensus_mode": "observations",
        **overrides,
    }


@unittest.skipUnless(importlib.util.find_spec("jsonschema"), "BFCL adapter fixture requires jsonschema")
class TestSeparateEnvironments(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="benchmark_runtime_test_")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        for name in ("model_env", "bfcl_env"):
            venv.EnvBuilder(with_pip=False, system_site_packages=(name == "bfcl_env")).create(cls.directory / name)
        executable = "Scripts/python.exe" if os.name == "nt" else "bin/python"
        cls.model_python = cls.directory / "model_env" / executable
        cls.bfcl_python = cls.directory / "bfcl_env" / executable
        site = subprocess.check_output(
            [str(cls.bfcl_python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"], text=True
        ).strip()
        package = Path(site) / "bfcl_eval"
        for relative, content in FAKE_FILES.items():
            target = package / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            parent = target.parent
            while parent != package.parent:
                (parent / "__init__.py").touch(exist_ok=True)
                parent = parent.parent
            target.write_text(content, encoding="utf-8")
        entry = {"id": "multi_turn_base_0", "initial_config": {}, "involved_classes": ["GorillaFileSystem"]}
        for folder, rows in (("prompts", [entry]), ("answers", [{"ground_truth": [["mkdir(dir_name='A')"]]}])):
            (package / folder).mkdir()
            (package / folder / "multi_turn_base.json").write_text(json.dumps(rows), encoding="utf-8")
        cls.audit = cls.directory / "audit.log"
        environment = {
            **os.environ,
            "BFCL_AUDIT": str(cls.audit),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        cls.process = subprocess.Popen(
            [
                str(cls.bfcl_python),
                str(RUNTIME_DIR / "server.py"),
                "--adapter",
                str(RUNTIME_DIR / "bfcl_adapter.py"),
                "--port",
                "0",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            env=environment,
        )
        cls.addClassCleanup(cls.stop_server)
        ready = queue.Queue()
        cls.logs = []

        def read_output():
            for line in cls.process.stdout:
                cls.logs.append(line)
                if line.startswith("Benchmark runtime listening on "):
                    ready.put(line.strip().split()[-1])

        threading.Thread(target=read_output, daemon=True).start()
        try:
            endpoint = ready.get(timeout=30)
        except queue.Empty:
            raise AssertionError("BFCL fixture server failed to start: " + "".join(cls.logs)) from None
        cls.config = {"endpoint": endpoint, "timeout_s": 3}
        cls.rpc = client.RuntimeClient.from_config(cls.config)

    @classmethod
    def stop_server(cls):
        cls.process.terminate()
        cls.process.wait(timeout=10)
        cls.process.stdout.close()

    def session(self, **overrides):
        session = client.RemoteSession(context(**overrides), self.config)
        self.addCleanup(session.close)
        return session

    def diagnostic_session(self, **overrides):
        before = self.audit.read_text()

        def check_no_oracle_access():
            after = self.audit.read_text()
            for operation in ("oracle_read", "checker_call"):
                self.assertEqual(before.count(operation), after.count(operation))

        self.addCleanup(check_no_oracle_access)
        functions = context()["functions"] + [
            {
                "name": "rmdir",
                "parameters": {
                    "type": "object",
                    "properties": {"dir_name": {"type": "string"}},
                    "required": ["dir_name"],
                },
            },
            {
                "name": "cd",
                "parameters": {
                    "type": "object",
                    "properties": {"folder": {"type": "string"}},
                    "required": ["folder"],
                },
            },
        ]
        if "failure_mode_detection" not in overrides:
            overrides["failure_mode_detection"] = {"enabled": True}
        elif overrides["failure_mode_detection"] is None:
            overrides.pop("failure_mode_detection")
        return self.session(functions=functions, **overrides)

    @staticmethod
    def execute_call(session, name, **arguments):
        response = "<tool_call>" + json.dumps({"name": name, "arguments": arguments}) + "</tool_call>"
        return session.execute(session.parse_calls(response))

    def nonempty_directory(self, session):
        self.execute_call(session, "mkdir", dir_name="A")
        self.execute_call(session, "mkdir", dir_name="A/B")

    def test_diagnostics_switch_preserves_validation_when_disabled(self):
        for settings in (None, {"enabled": False}, {"enabled": True}):
            with self.subTest(settings=settings):
                session = self.diagnostic_session(failure_mode_detection=settings)
                for response in (
                    "<tool_call>{broken}</tool_call>",
                    '<tool_call>{"name":"mkdir","arguments":{}}</tool_call>',
                ):
                    self.assertIsNone(session.parse_calls(response))
                    session.malformed_response()
                self.nonempty_directory(session)
                self.execute_call(session, "rmdir", dir_name="A")
                session.finish_turn("done")
                outcome = session.outcome("completed")
                enabled = settings is not None and settings["enabled"]
                self.assertIs(outcome["failure_mode_detection_enabled"], enabled)
                self.assertEqual(outcome["parse_errors"], 2)
                self.assertEqual(outcome["num_calls"], 3)
                self.assertEqual(
                    [event["type"] for event in outcome["failure_modes"]],
                    ["format_failure", "schema_failure", "precondition_violation"] if enabled else [],
                )

    def test_diagnostics_distinguish_format_and_schema_without_double_counting(self):
        cases = [
            ("<tool_call>{broken}</tool_call>", "format_failure"),
            ('<tool_call>{"name":"mkdir","arguments":{}}</tool_call>', "schema_failure"),
            ('<tool_call>{"name":"mkdir","arguments":{"dir_name":3}}</tool_call>', "schema_failure"),
        ]
        for response, kind in cases:
            with self.subTest(kind=kind, response=response):
                session = self.diagnostic_session()
                self.assertIsNone(session.parse_calls(response))
                session.malformed_response()
                self.execute_call(session, "pwd")
                session.finish_turn("done")
                outcome = session.outcome("completed")
                self.assertEqual(outcome["parse_errors"], 1)
                self.assertEqual([event["type"] for event in outcome["failure_modes"]], [kind])
                self.assertEqual(outcome["failure_modes"][0]["recovery"], "unknown")

    def test_diagnostics_accept_natural_language_and_explicit_no_call(self):
        for response in ("Please clarify the destination.", "done", "[]"):
            with self.subTest(response=response):
                session = self.diagnostic_session()
                self.assertEqual(session.parse_calls(response), [])
                session.finish_turn(response)
                self.assertEqual(session.outcome("completed")["failure_modes"], [])

    def test_diagnostics_reject_the_entire_response_with_a_malformed_call(self):
        session = self.diagnostic_session()
        response = (
            '<tool_call>{"name":"mkdir","arguments":{"dir_name":"A"}}</tool_call>' "<tool_call>{broken}</tool_call>"
        )
        self.assertIsNone(session.parse_calls(response))
        session.malformed_response()
        session.finish_turn("done")
        outcome = session.outcome("completed")
        self.assertEqual(outcome["num_calls"], 0)
        self.assertEqual([event["type"] for event in outcome["failure_modes"]], ["format_failure"])

    def test_diagnostics_recover_only_the_corresponding_operation(self):
        session = self.diagnostic_session()
        self.nonempty_directory(session)
        self.execute_call(session, "rmdir", dir_name="A")
        self.execute_call(session, "pwd")
        self.assertEqual(session.outcome("response_length")["failure_modes"][0]["recovery"], "unknown")
        self.execute_call(session, "rmdir", dir_name="A/B")
        self.assertEqual(session.outcome("response_length")["failure_modes"][0]["recovery"], "unknown")
        self.execute_call(session, "rmdir", dir_name="A")
        session.finish_turn("done")
        events = session.outcome("completed")["failure_modes"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "precondition_violation")
        self.assertEqual(events[0]["rule"], "directory_not_empty")
        self.assertEqual(events[0]["recovery"], "recovered")
        self.assertEqual(events[0]["recovered_by"]["call"], {"name": "rmdir", "arguments": {"dir_name": "A"}})

    def test_diagnostics_completed_and_truncated_trajectories_have_distinct_status(self):
        session = self.diagnostic_session()
        self.nonempty_directory(session)
        self.execute_call(session, "rmdir", dir_name="A")
        self.execute_call(session, "pwd")
        session.finish_turn("done")
        self.assertEqual(session.outcome("completed")["failure_modes"][0]["recovery"], "unrecovered")
        self.assertEqual(session.outcome("response_length")["failure_modes"][0]["recovery"], "unknown")

    def test_diagnostics_keep_unknown_after_potential_state_changes(self):
        for error_kind in ("precondition_violation", "format_failure"):
            with self.subTest(error_kind=error_kind):
                session = self.diagnostic_session()
                self.nonempty_directory(session)
                if error_kind == "precondition_violation":
                    self.execute_call(session, "rmdir", dir_name="A")
                else:
                    self.assertIsNone(session.parse_calls('{"name":"rmdir","arguments":{"dir_name":"A"}}'))
                    session.malformed_response()
                self.execute_call(session, "mkdir", dir_name="C")
                session.finish_turn("done")
                event = session.outcome("completed")["failure_modes"][0]
                self.assertEqual(event["type"], error_kind)
                self.assertEqual(event["recovery"], "unknown")

    def test_diagnostics_can_recover_identifiable_framing_errors(self):
        session = self.diagnostic_session()
        self.execute_call(session, "mkdir", dir_name="A")
        self.assertIsNone(session.parse_calls('{"name":"rmdir","arguments":{"dir_name":"A"}}'))
        session.malformed_response()
        self.execute_call(session, "rmdir", dir_name="A")
        session.finish_turn("done")
        event = session.outcome("completed")["failure_modes"][0]
        self.assertEqual(event["type"], "format_failure")
        self.assertEqual(event["recovery"], "recovered")

    def test_diagnostics_do_not_link_retries_across_user_turns(self):
        for retry in (False, True):
            with self.subTest(retry=retry):
                session = self.diagnostic_session(future_user_turns=[[{"role": "user", "content": "Next task"}]])
                self.nonempty_directory(session)
                self.execute_call(session, "rmdir", dir_name="A")
                session.finish_turn("Need another step")
                if retry:
                    self.execute_call(session, "rmdir", dir_name="A/B")
                    self.execute_call(session, "rmdir", dir_name="A")
                session.finish_turn("done")
                event = session.outcome("completed")["failure_modes"][0]
                self.assertEqual(event["turn"], 0)
                self.assertEqual(event["recovery"], "unknown")

    def test_diagnostics_abstain_after_navigation_and_on_unrecognized_errors(self):
        session = self.diagnostic_session()
        self.execute_call(session, "rmdir", dir_name="missing")
        self.assertEqual(session.outcome("response_length")["failure_modes"], [])
        self.nonempty_directory(session)
        self.execute_call(session, "rmdir", dir_name="A")
        self.execute_call(session, "cd", folder=".")
        self.execute_call(session, "rmdir", dir_name="A/B")
        self.execute_call(session, "rmdir", dir_name="A")
        session.finish_turn("done")
        self.assertEqual(session.outcome("completed")["failure_modes"][0]["recovery"], "unknown")

    def test_model_environment_has_no_bfcl_but_can_use_the_adapter(self):
        probe = """import importlib.util, json, sys
assert importlib.util.find_spec("bfcl_eval") is None
spec = importlib.util.spec_from_file_location("client", sys.argv[1])
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
print(json.dumps(module.RuntimeClient(sys.argv[2]).health("bfcl")))
assert "bfcl_eval" not in sys.modules and "jsonschema" not in sys.modules
"""
        output = subprocess.check_output(
            [str(self.model_python), "-c", probe, str(RUNTIME_DIR / "client.py"), self.config["endpoint"]],
            text=True,
            encoding="utf-8",
        )
        health = json.loads(output)
        self.assertEqual(Path(health["python"]).resolve(), self.bfcl_python.resolve())
        self.assertIn("bfcl_env", health["bfcl_eval"])

    def test_multiturn_state_isolation_and_no_oracle_reads(self):
        before = self.audit.read_text()
        first, second = self.session(), self.session()
        calls = first.parse_calls('<tool_call>{"name":"mkdir","arguments":{"dir_name":"A"}}</tool_call>')
        first.execute(calls)
        first.execute([{"name": "pwd", "arguments": {}}])
        other = second.execute([{"name": "pwd", "arguments": {}}])
        self.assertEqual(other[0]["content"], '"/"')
        first.finish_turn("done")
        self.assertTrue(first.outcome("completed")["valid"])
        self.assertEqual(first.outcome("completed")["num_calls"], 2)
        self.assertEqual(before.count("oracle_read"), self.audit.read_text().count("oracle_read"))

    def test_delayed_tools_and_repair_cross_the_boundary(self):
        original = context()
        session = self.session(
            functions=original["functions"][:1],
            future_user_turns=[[{"role": "user", "content": "Now query the directory"}]],
            function_schedule={"1": original["functions"][1:]},
        )
        response = '<tool_call>{"name":"pwd","arguments":{}}</tool_call>'
        self.assertIsNone(session.parse_calls(response))
        session.malformed_response()
        self.assertIn("Additional tools", session.finish_turn("clarify")[0]["content"])
        self.assertEqual(len(session.tool_schemas), 2)
        session.execute(session.parse_calls(response))
        session.finish_turn("done")
        self.assertTrue(session.outcome("completed")["valid"])

    def test_supervised_checker_and_references_stay_in_bfcl_environment(self):
        outcomes = []
        for name in ("A", "B"):
            session = self.session(record_execution_trace=True)
            session.execute([{"name": "mkdir", "arguments": {"dir_name": name}}])
            session.finish_turn("done")
            outcomes.append(session.outcome("completed"))
        item = {"index": "multi_turn_base_0", "bfcl": context()}
        scores = self.rpc.call("score_outcomes", items=[item, item], outcomes=outcomes)
        self.assertEqual(scores, [1.0, 0.0])
        self.assertIn("oracle_read", self.audit.read_text())
        self.assertIn("checker_call", self.audit.read_text())

    def test_training_tensors_and_rewards_work_with_benchmark_imports_blocked(self):
        try:
            import numpy as np
            import torch
        except ImportError:
            self.skipTest("Tensor bridge test requires the training-side numpy and torch dependencies")
        consensus = load("test_consensus", ROOT / "verl/utils/reward_score/ttrl_consensus.py")
        modules = {
            name: ModuleType(name)
            for name in ("verl", "verl.benchmark_runtime", "verl.utils", "verl.utils.reward_score")
        }
        modules["verl.benchmark_runtime.client"] = client
        modules["verl.utils.reward_score.ttrl_consensus"] = consensus
        original_import = builtins.__import__

        def restricted_import(name, *args, **kwargs):
            if name.split(".")[0] in ("bfcl_eval", "jsonschema"):
                raise AssertionError("Benchmark dependency imported into the trainer: " + name)
            return original_import(name, *args, **kwargs)

        with patch.dict(sys.modules, modules), patch("builtins.__import__", side_effect=restricted_import):
            training = load("test_training_bfcl", ROOT / "verl/utils/reward_score/ttrl_bfcl.py")
            self.assertEqual(training.RuntimeClient.from_config(self.config).health("bfcl")["backend"], "bfcl")
            calls = self.rpc.call(
                "decode_responses",
                context=context(),
                responses=['<tool_call>{"name":"pwd","arguments":{}}</tool_call>', "<tool_call>{broken}</tool_call>"],
            )
            self.assertEqual(calls, [[{"name": "pwd", "arguments": {}}], None])
            outcomes = []
            for name in ("A", "B"):
                session = training.BFCLSession(
                    context(record_execution_trace=True, consensus_mode="calls"), self.config
                )
                try:
                    session.execute([{"name": "mkdir", "arguments": {"dir_name": name}}])
                    session.finish_turn("done")
                    outcomes.append(session.outcome("completed"))
                finally:
                    session.close()
            data = ModuleType("batch")
            data.batch = {
                "responses": torch.zeros(2, 3, dtype=torch.long),
                "prompts": torch.zeros(2, 2),
                "response_mask": torch.tensor([[1, 0, 1], [1, 0, 1]]),
                "attention_mask": torch.ones(2, 5),
            }
            item = {"index": "multi_turn_base_0", "bfcl": context(consensus_mode="calls")}
            data.non_tensor_batch = {
                "bfcl_outcome": np.array(outcomes, dtype=object),
                "extra_info": np.array([item, item], dtype=object),
            }
            batch_type = type("Batch", (), {"__len__": lambda self: 2})
            batch = batch_type()
            batch.batch, batch.non_tensor_batch = data.batch, data.non_tensor_batch
            rewards = training.BFCLSupervisedReward(runtime_config=self.config)(batch)
            self.assertEqual(rewards.tolist(), [[0, 0, 1], [0, 0, 0]])

            class MiniBatch:
                def __init__(self, count, tensors, metadata):
                    self.count, self.batch, self.non_tensor_batch = count, tensors, metadata

                def __len__(self):
                    return self.count

                def __getitem__(self, row):
                    return SimpleNamespace(
                        batch={key: value[row] for key, value in self.batch.items()},
                        non_tensor_batch={key: value[row] for key, value in self.non_tensor_batch.items()},
                    )

            class PoisonOracle(dict):
                def get(self, *args):
                    raise AssertionError("Unlabeled TTRL accessed an oracle")

                __getitem__ = get

            prompts = MiniBatch(1, {}, {"extra_info": np.array([item], dtype=object), "reward_model": [PoisonOracle()]})
            generated = MiniBatch(
                3,
                {
                    "responses": torch.zeros(3, 3, dtype=torch.long),
                    "prompts": torch.zeros(3, 2),
                    "response_mask": torch.tensor([[1, 0, 1]] * 3),
                    "attention_mask": torch.ones(3, 5),
                },
                {"bfcl_outcome": np.array([outcomes[0], outcomes[0], outcomes[1]], dtype=object)},
            )
            training.apply_bfcl_rewards(prompts, generated, 3, tokenizer=None)
            self.assertEqual(generated.batch["ttrl_bfcl_scores"].tolist(), [[0, 0, 1], [0, 0, 1], [0, 0, 0]])
            self.assertAlmostEqual(prompts.non_tensor_batch["majority_ratio_list"][0], 2 / 3)

    def test_actual_agent_loop_uses_rpc_and_cleans_up_generation_failure_and_cancellation(self):
        try:
            # Keep native tensor modules outside the temporary sys.modules namespace.
            importlib.import_module("numpy")
            importlib.import_module("torch")
        except ImportError:
            self.skipTest("Actual AgentLoop bridge requires training-side numpy and torch")
        consensus = load("loop_test_consensus", ROOT / "verl/utils/reward_score/ttrl_consensus.py")
        modules = {
            name: ModuleType(name)
            for name in (
                "verl",
                "verl.benchmark_runtime",
                "verl.utils",
                "verl.utils.reward_score",
                "verl.experimental",
                "verl.experimental.agent_loop",
                "verl.tools",
                "verl.tools.utils",
                "verl.experimental.agent_loop.agent_loop",
                "verl.tools.utils.tool_registry",
                "verl.utils.debug",
                "pydantic",
            )
        }
        modules["verl.benchmark_runtime.client"] = client
        modules["verl.utils.reward_score.ttrl_consensus"] = consensus
        modules["regex"] = re
        modules["pydantic"].BaseModel = object
        modules["verl.tools.utils.tool_registry"].initialize_tools_from_config = lambda _: []

        @contextmanager
        def timer(*args):
            yield

        modules["verl.utils.debug"].simple_timer = timer

        class Base:
            _class_initialized = False

            def __init__(self, config, server_manager, tokenizer):
                self.config, self.server_manager, self.tokenizer = config, server_manager, tokenizer
                self.loop = asyncio.get_running_loop()
                self.init_class(config, tokenizer)

        base_module = modules["verl.experimental.agent_loop.agent_loop"]
        base_module.AgentLoopBase, base_module.AgentLoopOutput = Base, SimpleNamespace

        class Config(dict):
            __getattr__ = dict.__getitem__

        config = Config(
            benchmark_runtime=self.config,
            ttrl={"enable": True, "reward_mode": "bfcl"},
            actor_rollout_ref=Config(
                rollout=Config(
                    prompt_length=8192,
                    response_length=8192,
                    multi_turn=Config(
                        max_user_turns=16,
                        max_assistant_turns=16,
                        max_parallel_calls=8,
                        max_tool_response_length=4096,
                        tool_response_truncate_side="right",
                        tool_config_path=None,
                        format="hermes",
                    ),
                )
            ),
        )

        class Tokenizer:
            eos_token = "<END>"

            def apply_chat_template(self, messages, **kwargs):
                text = "<SYS>" + json.dumps(messages) + json.dumps(kwargs.get("tools", []))
                return list(map(ord, text))

            def decode(self, tokens, **kwargs):
                return "".join(map(chr, tokens))

        async def exercise(loop_class):
            baseline = self.rpc.health()["active_sessions"]
            for enabled, behavior in (
                (None, "complete"),
                (False, "complete"),
                (True, "complete"),
                (False, "fail"),
                (False, "cancel"),
            ):
                if enabled is None:
                    config.pop("failure_mode_detection", None)
                else:
                    config["failure_mode_detection"] = {"enabled": enabled}
                reached_second_turn = asyncio.Event()

                class Model:
                    def __init__(self, behavior, reached_second_turn):
                        self.behavior, self.reached_second_turn = behavior, reached_second_turn
                        self.count = 0

                    async def generate(self, **kwargs):
                        self.count += 1
                        if self.count == 1 and self.behavior == "complete":
                            response = '<tool_call>{"name":"mkdir","arguments":{}}</tool_call><END>'
                        elif self.count == 1 or (self.count == 2 and self.behavior == "complete"):
                            response = '<tool_call>{"name":"pwd","arguments":{}}</tool_call><END>'
                        elif self.behavior == "fail":
                            raise RuntimeError("generation fixture failure")
                        elif self.behavior == "cancel":
                            self.reached_second_turn.set()
                            await asyncio.Event().wait()
                        else:
                            response = "done<END>"
                        return list(map(ord, response))

                loop_class._class_initialized = False
                loop = loop_class(config, Model(behavior, reached_second_turn), Tokenizer())
                session_context = context(failure_mode_detection={"enabled": not bool(enabled)})
                original_context = json.loads(json.dumps(session_context))
                task = asyncio.create_task(loop.run([{"role": "user", "content": "task"}], {}, session_context))
                if behavior == "cancel":
                    await asyncio.wait_for(reached_second_turn.wait(), timeout=5)
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                elif behavior == "fail":
                    with self.assertRaisesRegex(RuntimeError, "generation fixture failure"):
                        await task
                else:
                    output = await task
                    outcome = output.extra_fields["bfcl_outcome"]
                    self.assertTrue(outcome["valid"])
                    self.assertIs(outcome["failure_mode_detection_enabled"], bool(enabled))
                    self.assertIs(output.extra_fields["failure_mode_detection_enabled"], bool(enabled))
                    self.assertEqual(output.extra_fields["failure_modes"], outcome["failure_modes"])
                    self.assertEqual(
                        [event["type"] for event in output.extra_fields["failure_modes"]],
                        ["schema_failure"] if enabled else [],
                    )
                    self.assertIn(0, output.response_mask)
                    self.assertIn(1, output.response_mask)
                self.assertEqual(session_context, original_context)
                self.assertEqual(self.rpc.health()["active_sessions"], baseline)

        original_import = builtins.__import__

        def restricted_import(name, *args, **kwargs):
            if name.split(".")[0] in ("bfcl_eval", "jsonschema"):
                raise AssertionError("Benchmark dependency imported into AgentLoop: " + name)
            return original_import(name, *args, **kwargs)

        with patch.dict(sys.modules, modules), patch("builtins.__import__", side_effect=restricted_import):
            modules["verl.utils.reward_score.ttrl_bfcl"] = load(
                "loop_training_bfcl", ROOT / "verl/utils/reward_score/ttrl_bfcl.py"
            )
            sys.modules["verl.utils.reward_score.ttrl_bfcl"] = modules["verl.utils.reward_score.ttrl_bfcl"]
            tool_loop = load("actual_tool_loop", ROOT / "verl/experimental/agent_loop/tool_agent_loop.py")
            asyncio.run(exercise(tool_loop.ToolAgentLoop))


if __name__ == "__main__":
    unittest.main()
