"""Exercise the production TaskRunner capture gates without Ray or model imports."""

import ast
import copy
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call, patch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "verl/trainer/main_ppo.py"


def runner_training_tail(namespace):
    """Load the real post-preflight run body, leaving heavy setup out of scope."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    runner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TaskRunner")
    method = next(node for node in runner.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    start = next(
        index for index, node in enumerate(method.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "gradient_record" for target in node.targets)
    )
    function = ast.FunctionDef(
        name="run_training_tail",
        args=ast.arguments(
            posonlyargs=[], args=[ast.arg(arg="config"), ast.arg(arg="trainer")],
            kwonlyargs=[], kw_defaults=[], defaults=[],
        ),
        body=copy.deepcopy(method.body[start:]), decorator_list=[],
    )
    code = ast.Module(body=[function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(code), str(SOURCE), "exec"), namespace)
    return namespace["run_training_tail"]


class TestGradientCaptureGate(unittest.TestCase):
    def setUp(self):
        self.calls = Mock()
        self.prepare = Mock()
        self.calls.attach_mock(self.prepare, "prepare")
        self.trainer = Mock()
        for method in ("init_workers", "fit", "serve_after_training"):
            self.calls.attach_mock(getattr(self.trainer, method), method)
        self.calls.attach_mock(self.trainer.actor_rollout_wg.complete_gradient_capture, "complete")
        self.document = {"captured": "configuration"}
        self.omega = SimpleNamespace(to_container=Mock(return_value=self.document))
        self.run = runner_training_tail({"OmegaConf": self.omega})
        module = ModuleType("verl.utils.gradient_replay")
        module.prepare_replay_directory = self.prepare
        modules = patch.dict(sys.modules, {module.__name__: module})
        modules.start()
        self.addCleanup(modules.stop)

    def config(self, record):
        return SimpleNamespace(actor_rollout_ref=SimpleNamespace(actor={"gradient_record": record}))

    def test_master_disabled_ignores_a_stale_two_stage_enabled_flag(self):
        self.run(self.config({"enabled": False, "two_stage": {"enabled": True}}), self.trainer)
        self.prepare.assert_not_called()
        self.omega.to_container.assert_not_called()
        self.trainer.actor_rollout_wg.complete_gradient_capture.assert_not_called()
        self.assertEqual(self.calls.mock_calls, [call.init_workers(), call.fit(), call.serve_after_training()])

    def test_both_flags_enabled_prepare_and_complete_around_training(self):
        config = self.config({"enabled": True, "two_stage": {"enabled": True, "replay_dir": "archive"}})
        self.run(config, self.trainer)
        self.omega.to_container.assert_called_once_with(config, resolve=True)
        self.assertEqual(self.calls.mock_calls, [
            call.prepare("archive", self.document), call.init_workers(), call.fit(),
            call.complete(), call.serve_after_training(),
        ])

    def test_missing_or_disabled_stage_flag_keeps_ordinary_training(self):
        for record in ({}, {"enabled": True}, {"enabled": True, "two_stage": {"enabled": False}}):
            with self.subTest(record=record):
                self.calls.reset_mock()
                self.run(self.config(record), self.trainer)
                self.prepare.assert_not_called()
                self.trainer.actor_rollout_wg.complete_gradient_capture.assert_not_called()
                self.assertEqual(self.calls.mock_calls, [call.init_workers(), call.fit(), call.serve_after_training()])

    def test_failed_training_never_marks_capture_complete(self):
        config = self.config({"enabled": True, "two_stage": {"enabled": True, "replay_dir": "archive"}})
        self.trainer.fit.side_effect = RuntimeError("training failed")
        with self.assertRaisesRegex(RuntimeError, "training failed"):
            self.run(config, self.trainer)
        self.prepare.assert_called_once_with("archive", self.document)
        self.trainer.actor_rollout_wg.complete_gradient_capture.assert_not_called()
        self.trainer.serve_after_training.assert_not_called()


if __name__ == "__main__":
    unittest.main()
