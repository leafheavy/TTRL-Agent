"""Exercise reusable diagnostics with a benchmark unrelated to BFCL."""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

PATH = Path(__file__).resolve().parents[2] / "verl/benchmark_runtime/failure_modes.py"
SPEC = importlib.util.spec_from_file_location("generic_failure_modes_test", PATH)
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)


class JobRules(core.FailureRules):
    """A toy job API supplies only public operation contracts."""

    def identity(self, call):
        if call and call.get("name") == "submit" and set(call.get("arguments", {})) == {"job"}:
            return call["arguments"]["job"]
        return None

    def preserves_evidence(self, call):
        return call.get("name") == "status"

    def observe(self, call, result):
        if self.identity(call) is None:
            return None, False
        if result == {"error": "Job is not ready"}:
            return {"kind": "precondition_violation", "rule": "job_not_ready", "evidence": result["error"]}, False
        return None, result == {"accepted": True}


def submit(job="A"):
    return {"name": "submit", "arguments": {"job": job}}


class TestReusableFailureModes(unittest.TestCase):
    def diagnostics(self):
        return core.FailureDiagnostics(JobRules(), enabled=True)

    def test_public_contracts_and_read_only_observations_preserve_evidence(self):
        diagnostics = self.diagnostics()
        diagnostics.observe(submit(), {"error": "Job is not ready"})
        diagnostics.observe({"name": "status", "arguments": {}}, {"pending": 1})
        self.assertEqual(diagnostics.snapshot("response_length", False)[0]["recovery"], "unknown")
        event = diagnostics.snapshot("completed", True)[0]
        self.assertEqual(event["type"], "precondition_violation")
        self.assertEqual(event["recovery"], "unrecovered")
        event["evidence"] = "modified copy"
        self.assertEqual(diagnostics.snapshot("completed", True)[0]["evidence"], "Job is not ready")

    def test_only_a_corresponding_success_confirms_recovery(self):
        diagnostics = self.diagnostics()
        diagnostics.observe(submit(), {"error": "Job is not ready"})
        diagnostics.observe(submit("B"), {"accepted": True})
        self.assertEqual(diagnostics.snapshot("completed", True)[0]["recovery"], "unknown")
        diagnostics.observe(submit(), {"accepted": True})
        event = diagnostics.snapshot("completed", True)[0]
        self.assertEqual(event["recovery"], "recovered")
        self.assertEqual(event["rule"], "job_not_ready")
        self.assertEqual(event["recovered_by"], {"call": submit(), "turn": 0})

    def test_unknown_failures_and_cross_turn_retries_remain_unknown(self):
        diagnostics = self.diagnostics()
        diagnostics.record("format_failure", "broken_envelope", "invalid JSON")
        diagnostics.observe(submit(), {"error": "Job is not ready"}, turn=0)
        diagnostics.observe(submit(), {"accepted": True}, turn=1)
        events = diagnostics.snapshot("completed", True, turn=1)
        self.assertEqual([event["recovery"] for event in events], ["unknown", "unknown"])

    def test_disabled_diagnostics_do_not_evaluate_rules(self):
        class ForbiddenRules(core.FailureRules):
            def identity(self, call):
                raise AssertionError("Disabled detection evaluated identity")

            def preserves_evidence(self, call):
                raise AssertionError("Disabled detection evaluated evidence")

            def observe(self, call, result):
                raise AssertionError("Disabled detection evaluated observation")

        for settings in ({}, {"enabled": False}):
            with self.subTest(settings=settings):
                diagnostics = core.FailureDiagnostics(ForbiddenRules(), **settings)
                diagnostics.record("format_failure", "broken_envelope", "invalid JSON", submit())
                diagnostics.observe(submit(), {"error": "Job is not ready"})
                self.assertEqual(diagnostics.snapshot("completed", True), [])
        for enabled in (1, 0, "true", "false", None):
            with self.subTest(enabled=enabled):
                with self.assertRaisesRegex(ValueError, "bool"):
                    core.FailureDiagnostics(enabled=enabled)

    def test_generic_records_need_no_benchmark_rules(self):
        diagnostics = core.FailureDiagnostics(enabled=True)
        diagnostics.record("format_failure", "broken_envelope", "invalid JSON")
        diagnostics.observe(submit(), {"accepted": True})
        event = diagnostics.snapshot("completed", True)[0]
        self.assertEqual(event["type"], "format_failure")
        self.assertEqual(event["recovery"], "unknown")


class TestGenericCallValidation(unittest.TestCase):
    class Validator:
        def iter_errors(self, arguments):
            if not isinstance(arguments.get("job"), str):
                yield SimpleNamespace(absolute_path=["job"], validator="type", message="job must be a string")

        def is_valid(self, arguments):
            return not list(self.iter_errors(arguments))

    def test_valid_calls_and_explicit_no_calls(self):
        validators = {"submit": self.Validator()}
        for calls in ([], [submit()], [submit(), submit("B")]):
            with self.subTest(calls=calls):
                self.assertEqual(core.validate_calls(calls, validators), (calls, []))

    def test_invalid_calls_reject_the_whole_response(self):
        cases = [
            ([submit(), {"name": "submit", "arguments": {"job": 2}}], "schema_failure"),
            ([submit(), {"name": "missing", "arguments": {}}], "schema_failure"),
            ([submit(), {"name": "submit", "arguments": []}], "format_failure"),
            ([submit(), {"name": "submit", "arguments": {"job": float("nan")}}], "format_failure"),
        ]
        for calls, kind in cases:
            for collect in (True, False):
                with self.subTest(kind=kind, calls=calls, collect=collect):
                    validated, failures = core.validate_calls(
                        calls, {"submit": self.Validator()}, collect_failures=collect
                    )
                    self.assertIsNone(validated)
                    self.assertEqual([failure["kind"] for failure in failures], [kind] if collect else [])


if __name__ == "__main__":
    unittest.main()
