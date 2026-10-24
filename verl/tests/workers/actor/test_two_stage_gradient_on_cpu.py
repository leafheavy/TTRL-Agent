"""Exercise two-stage trajectory capture/replay with real CPU AdamW and dropout.

The production actor bodies are loaded without importing Ray, BFCL, or vLLM.
Archive corruption tests intentionally fail before any result can be merged.
"""

import ast
import copy
import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]
HELPER_SPEC = importlib.util.spec_from_file_location(
    "two_stage_cpu_actor_helpers", Path(__file__).with_name("test_gradient_diagnostics_on_cpu.py")
)
helpers = importlib.util.module_from_spec(HELPER_SPEC)
HELPER_SPEC.loader.exec_module(helpers)
diagnostics = helpers.diagnostics


def load_actor_methods():
    """Include new replay hooks without copying their implementation into tests."""
    source = ROOT / "verl/workers/actor/dp_actor.py"
    actor_node = next(
        node for node in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(node, ast.ClassDef) and node.name == "DataParallelPPOActor"
    )
    methods = {
        node.name for node in actor_node.body
        if isinstance(node, ast.FunctionDef) and node.name != "__init__"
    }
    namespace = dict(helpers.Actor.update_policy.__globals__)
    namespace.update({"np": helpers.np, "sys": sys, "Path": Path})
    helpers.source_definitions(source, {"DataParallelPPOActor"}, namespace, methods=methods)
    helpers.Actor = namespace["DataParallelPPOActor"]


load_actor_methods()


class TestTwoStageGradientReplay(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        modules = patch.dict(sys.modules, {"verl.utils.gradient_diagnostics": diagnostics})
        modules.start()
        self.addCleanup(modules.stop)
        spec = importlib.util.spec_from_file_location(
            "verl.utils.gradient_replay", ROOT / "verl/utils/gradient_replay.py"
        )
        self.replay = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.replay)
        replay_module = patch.dict(sys.modules, {"verl.utils.gradient_replay": self.replay})
        replay_module.start()
        self.addCleanup(replay_module.stop)
        for name, value in (("get_rank", 0), ("get_world_size", 1)):
            mocked = patch.object(torch.distributed, name, return_value=value)
            mocked.start()
            self.addCleanup(mocked.stop)

    def actor(self, name, *, capture=False, online=False, interval=1):
        actor = helpers.make_actor(True, self.root / name, mode="paired")
        actor.gradient_record_config.update(
            interval=interval,
            single_step={"enabled": False},
            trajectory={"enabled": True},
            baselines={
                "enabled": online,
                "seed": 23,
                "shuffled_labels": {"enabled": True},
                "random_direction": {"enabled": False},
            },
            two_stage={"enabled": capture, "replay_dir": str(self.root / "archive")},
        )
        return actor

    def prepare(self, directory=None):
        directory = directory or self.root / "archive"
        self.replay.prepare_replay_directory(
            directory, {"trainer": {"n_gpus_per_node": 1, "nnodes": 1}}
        )
        return directory

    def capture(self, *, complete=True, minibatches=False):
        self.prepare()
        actor = self.actor("stage_one", capture=True, interval=10)
        if minibatches:
            actor.config.ppo_mini_batch_size = 2
            actor.config.ppo_micro_batch_size_per_gpu = 1
            actor.config.ppo_epochs = 2
        torch.manual_seed(113)
        actor.update_policy(helpers.TestPairedRandomBaselines.baseline_batch())
        if complete:
            actor.complete_gradient_capture()
        return actor

    def reader(self):
        return self.replay.GradientReplayReader(self.root / "archive", rank=0, world_size=1)

    def assert_state_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_state_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for one, two in zip(left, right):
                self.assert_state_equal(one, two)
        else:
            self.assertEqual(left, right)

    def test_two_stages_match_online_three_branch_adamw_paths_and_preserve_ttrl(self):
        self.prepare()
        online = self.actor("online", online=True)
        capture = self.actor("stage_one", capture=True, interval=10)
        ordinary = helpers.make_actor(False, self.root / "ordinary")
        for actor in (online, capture, ordinary):
            # Exercise a nonempty optimizer origin, not just fresh Adam moments.
            original_enabled = actor.gradient_record_config["enabled"]
            actor.gradient_record_config["enabled"] = False
            torch.manual_seed(73)
            actor.update_policy(helpers.sample_batch())
            actor.gradient_record_config["enabled"] = original_enabled
            actor.config.ppo_mini_batch_size = 2
            actor.config.ppo_micro_batch_size_per_gpu = 1
            actor.config.ppo_epochs = 2
        for step, learning_rate in ((1, 0.003), (2, 0.008), (3, 0.005)):
            for actor in (online, capture, ordinary):
                actor.actor_optimizer.param_groups[0]["lr"] = learning_rate
            torch.manual_seed(200 + step)
            ordinary_metrics = ordinary.update_policy(helpers.sample_batch(step))
            ordinary_rng = torch.random.get_rng_state().clone()
            for actor in (online, capture):
                batch = helpers.TestPairedRandomBaselines.baseline_batch(step)
                batch.meta_info["gradient_record"]["final_step"] = step == 3
                torch.manual_seed(200 + step)
                actual_metrics = actor.update_policy(batch)
                self.assertEqual(actual_metrics, ordinary_metrics)
                self.assert_state_equal(actor.actor_module.state_dict(), ordinary.actor_module.state_dict())
                self.assert_state_equal(actor.actor_optimizer.state_dict(), ordinary.actor_optimizer.state_dict())
                self.assertTrue(torch.equal(torch.random.get_rng_state(), ordinary_rng))
        capture.complete_gradient_capture()
        frames = list(self.reader().frames())
        self.assertEqual(len(frames), 12)
        # Output interval does not discard training batches needed for replay.
        self.assertEqual(len(list((self.root / "stage_one").glob("trajectory_*.json"))), 8)

        baseline = self.actor("replay_actor")
        baseline.config.ppo_mini_batch_size = 2
        baseline.config.ppo_micro_batch_size_per_gpu = 1
        baseline.config.ppo_epochs = 2
        output = self.root / "stage_two"
        with patch.object(baseline, "_backward_micro_batches", wraps=baseline._backward_micro_batches) as backward:
            baseline.replay_gradient_diagnostics(self.root / "archive", output)
        # Exactly S and shuffle passes per replayed minibatch; no TTRL pass.
        self.assertEqual(backward.call_count, 24)
        online_records = {
            path.name: json.loads(path.read_text(encoding="utf-8"))
            for path in (self.root / "online").glob("trajectory_*.json")
        }
        actual_paths = sorted(output.glob("trajectory_*.json"))
        self.assertEqual(len(actual_paths), 12)
        for path in actual_paths:
            actual = json.loads(path.read_text(encoding="utf-8"))
            expected = online_records[path.name]
            self.assertEqual(actual["endpoint"], expected["endpoint"])
            self.assertEqual(actual["path"], expected["path"])
            self.assertEqual(
                actual["baselines"]["shuffled_labels"]["endpoint"],
                expected["baselines"]["shuffled_labels"]["endpoint"],
            )
            self.assertEqual(
                actual["baselines"]["shuffled_labels"]["path"],
                expected["baselines"]["shuffled_labels"]["path"],
            )
            self.assertEqual(set(actual["baselines"]), {"shuffled_labels"})
        for parameter, expected in zip(
            baseline.actor_module.parameters(), online.optimization_trajectory.shuffled_parameters
        ):
            torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
        expected_optimizer = diagnostics._restore_state(online.optimization_trajectory.shuffled_optimizer_state)
        self.assert_state_equal(
            list(baseline.actor_optimizer.state.values()), list(expected_optimizer.values())
        )
        # A verified prefix is not a completed experiment. The CLI's all-rank
        # completion marker is required before exporting the combined curves.
        with self.assertRaisesRegex(ValueError, "complete.json"):
            diagnostics.compare_paired_run(output, self.root / "export")
        document = json.loads((self.root / "archive" / "run_config.json").read_text(encoding="utf-8"))
        completion = {
            "stage": 2, "run_id": document["run_id"], "world_size": 1,
            "ranks": [{"rank": 0, "updates": 12}],
        }
        (output / "complete.json").write_text(json.dumps(completion), encoding="utf-8")
        csv_path = diagnostics.compare_paired_run(output, self.root / "export")
        with csv_path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 12)
        self.assertIn("shuffled_labels_path_cosine", rows[0])
        self.assertNotIn("random_update_path_cosine", rows[0])

    def test_unfinished_capture_is_not_replayable(self):
        self.capture(complete=False)
        with self.assertRaises(ValueError):
            list(self.reader().frames())

    def test_world_size_change_is_rejected(self):
        self.capture()
        with self.assertRaises(ValueError):
            self.replay.GradientReplayReader(self.root / "archive", rank=0, world_size=2)

    def test_missing_frame_is_rejected(self):
        self.capture(minibatches=True)
        next((self.root / "archive" / "rank_00000").glob("frame_*.pt")).unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            list(self.reader().frames())

    def test_modified_frame_is_rejected(self):
        self.capture()
        path = next((self.root / "archive" / "rank_00000").glob("frame_*.pt"))
        with path.open("ab") as stream:
            stream.write(b"not-the-recorded-frame")
        with self.assertRaises(ValueError):
            list(self.reader().frames())

    def test_reordered_frame_manifest_is_rejected(self):
        self.capture(minibatches=True)
        path = self.root / "archive" / "rank_00000" / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["frames"].reverse()
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(ValueError):
            list(self.reader().frames())

    def test_modified_initial_state_is_rejected(self):
        self.capture()
        path = self.root / "archive" / "rank_00000" / "initial.pt"
        with path.open("ab") as stream:
            stream.write(b"wrong-initial-state")
        actor = self.actor("stage_two_actor")
        with self.assertRaises(ValueError):
            self.reader().restore_initial(actor.actor_module, actor.actor_optimizer)

    def test_mismatched_supervised_reference_cannot_publish_combined_results(self):
        self.capture()
        actor = self.actor("stage_two_actor")
        original_step = actor._optimizer_step

        def different_update():
            result = original_step()
            with torch.no_grad():
                next(actor.actor_module.parameters()).add_(0.01)
            return result

        output = self.root / "mismatched_output"
        with patch.object(actor, "_optimizer_step", side_effect=different_update):
            with self.assertRaisesRegex(ValueError, "[Ss]upervised|reference|mismatch"):
                actor.replay_gradient_diagnostics(self.root / "archive", output)
        self.assertFalse(list(output.glob("trajectory_*.json")))

    def test_reference_verification_is_required_before_merge(self):
        self.capture()
        reader = self.reader()
        frame = next(reader.frames())
        with self.assertRaises(ValueError):
            reader.combine(frame, copy.deepcopy(frame["result"]))


if __name__ == "__main__":
    unittest.main()
