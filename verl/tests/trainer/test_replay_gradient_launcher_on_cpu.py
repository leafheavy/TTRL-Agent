"""Phase-two launcher checks without importing Ray or loading an LLM."""

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("replay_gradient_launcher", ROOT / "verl/trainer/replay_gradient.py")
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class Config(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def config_tree(value):
    if isinstance(value, dict):
        return Config({key: config_tree(item) for key, item in value.items()})
    if isinstance(value, list):
        return [config_tree(item) for item in value]
    return value


def configuration():
    return {
        "trainer": {"n_gpus_per_node": 2, "nnodes": 1, "device": "cuda"},
        "ray_init": {"num_cpus": None},
        "actor_rollout_ref": {
            "actor": {
                "strategy": "fsdp", "ulysses_sequence_parallel_size": 1,
                "gradient_record": {"two_stage": {"enabled": True}},
            },
            "rollout": {"n": 8, "name": "vllm"},
        },
    }


class TestReplayGradientLauncher(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.archive = self.root / "archive"
        self.archive.mkdir()
        self.config = configuration()
        self.write_config()
        self.reader = Mock()
        archive_module = ModuleType("verl.utils.gradient_replay")
        archive_module.GradientReplayReader = self.reader
        modules = patch.dict(sys.modules, {archive_module.__name__: archive_module})
        modules.start()
        self.addCleanup(modules.stop)

    def write_config(self, *, world_size=2):
        (self.archive / "run_config.json").write_text(
            json.dumps({"version": 1, "run_id": "test-run", "world_size": world_size, "run_config": self.config}),
            encoding="utf-8",
        )

    def test_metadata_preflight_requires_complete_archive_before_model_loading(self):
        loaded, world_size = launcher.load_replay_configuration(self.archive)
        self.assertEqual(loaded, self.config)
        self.assertEqual(world_size, 2)
        self.reader.assert_called_once_with(self.archive.resolve(), rank=0, world_size=2)
        self.reader.side_effect = ValueError("Incomplete rank archive")
        with self.assertRaisesRegex(ValueError, "Incomplete rank archive"):
            launcher.load_replay_configuration(self.archive)

    def test_invalid_world_size_and_unsupported_configuration_fail_preflight(self):
        for world_size in (True, 0, -1, "2", 1):
            with self.subTest(world_size=world_size):
                self.write_config(world_size=world_size)
                with self.assertRaisesRegex(ValueError, "world.size"):
                    launcher.load_replay_configuration(self.archive)
        cases = (
            ("strategy", "fsdp2", "FSDP1"),
            ("ulysses_sequence_parallel_size", 2, "ulysses"),
            ("gradient_record", {"two_stage": {"enabled": False}}, "two-stage"),
        )
        for key, value, error in cases:
            with self.subTest(key=key):
                self.config = configuration()
                self.config["actor_rollout_ref"]["actor"][key] = value
                self.write_config()
                with self.assertRaisesRegex(ValueError, error):
                    launcher.load_replay_configuration(self.archive)
        self.reader.assert_not_called()

    def test_output_must_not_overlap_archive_or_overwrite_existing_results(self):
        for directory in (self.archive, self.archive / "results", self.root):
            with self.subTest(directory=directory):
                with self.assertRaisesRegex(ValueError, "overlap"):
                    launcher.prepare_output_directory(directory, self.archive)
        output = self.root / "output"
        self.assertEqual(launcher.prepare_output_directory(output, self.archive), output)
        sentinel = output / "keep.json"
        sentinel.write_text("original", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            launcher.prepare_output_directory(output, self.archive)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "original")

    def runtime_modules(self, *, available_gpus=2):
        ray = ModuleType("ray")
        ray.is_initialized = Mock(return_value=False)
        ray.init = Mock()
        ray.available_resources = Mock(return_value={"GPU": available_gpus})
        ray.remote = Mock(side_effect=lambda cls: cls)
        ray.kill = Mock()
        ray.shutdown = Mock()
        ray.util = SimpleNamespace(remove_placement_group=Mock())
        omega = ModuleType("omegaconf")
        omega.OmegaConf = SimpleNamespace(create=config_tree)
        worker_type = type("ActorOnlyWorker", (), {})
        worker_module = ModuleType("verl.workers.fsdp_workers")
        worker_module.ActorRolloutRefWorker = worker_type
        pool = SimpleNamespace(pgs=["owned-placement-group"])
        worker_group = SimpleNamespace(
            init_model=Mock(),
            replay_gradient_diagnostics=Mock(return_value=[{"rank": 0, "updates": 3}, {"rank": 1, "updates": 3}]),
            workers=["owned-worker-0", "owned-worker-1"],
        )
        controller = ModuleType("verl.single_controller.ray")
        controller.RayResourcePool = Mock(return_value=pool)
        controller.RayWorkerGroup = Mock(return_value=worker_group)
        controller.RayClassWithInitArgs = Mock(side_effect=lambda **kwargs: kwargs)
        return {
            ray.__name__: ray, omega.__name__: omega, worker_module.__name__: worker_module,
            controller.__name__: controller,
        }, ray, controller, worker_group, worker_type

    def test_replay_constructs_actor_only_and_marks_completion_after_all_ranks(self):
        modules, ray, controller, workers, worker_type = self.runtime_modules()
        original = copy.deepcopy(self.config)
        output = self.root / "stage2"
        with patch.dict(sys.modules, modules):
            result = launcher.run_replay(self.archive, output)
        self.assertEqual(result, [{"rank": 0, "updates": 3}, {"rank": 1, "updates": 3}])
        arguments = controller.RayClassWithInitArgs.call_args.kwargs
        self.assertEqual(arguments["role"], "actor")
        self.assertIs(arguments["cls"], worker_type)
        self.assertEqual(arguments["config"], self.config["actor_rollout_ref"])
        arguments["config"]["actor"]["strategy"] = "modified-worker-copy"
        self.assertEqual(self.config, original)
        controller.RayResourcePool.assert_called_once_with(
            process_on_nodes=[2], use_gpu=True, max_colocate_count=1
        )
        workers.init_model.assert_called_once()
        workers.replay_gradient_diagnostics.assert_called_once_with(str(self.archive.resolve()), str(output.resolve()))
        completion = json.loads((output / "complete.json").read_text(encoding="utf-8"))
        self.assertEqual(completion["world_size"], 2)
        self.assertEqual(completion["ranks"], result)
        self.assertEqual(ray.kill.call_count, 2)
        ray.shutdown.assert_called_once()

    def test_insufficient_gpus_or_failed_reference_never_publish_completion(self):
        modules, ray, controller, _, _ = self.runtime_modules(available_gpus=1)
        output = self.root / "not-enough-gpus"
        with patch.dict(sys.modules, modules), self.assertRaisesRegex(RuntimeError, "same 2 GPU"):
            launcher.run_replay(self.archive, output)
        controller.RayWorkerGroup.assert_not_called()
        self.assertFalse((output / "complete.json").exists())
        ray.shutdown.assert_called_once()

        modules, ray, _, workers, _ = self.runtime_modules()
        workers.replay_gradient_diagnostics.side_effect = ValueError("supervised reference mismatch")
        output = self.root / "mismatched-reference"
        with patch.dict(sys.modules, modules), self.assertRaisesRegex(ValueError, "reference mismatch"):
            launcher.run_replay(self.archive, output)
        self.assertFalse((output / "complete.json").exists())
        self.assertEqual(ray.kill.call_count, 2)
        ray.shutdown.assert_called_once()


if __name__ == "__main__":
    unittest.main()
