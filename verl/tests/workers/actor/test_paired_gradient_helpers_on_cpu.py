"""CPU checks for full-coordinate paired gradients without binary snapshots."""

import copy
import csv
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("paired_gradient_helpers", ROOT / "verl/utils/gradient_diagnostics.py")
diagnostics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostics)


class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def configuration(mode="paired"):
    return Config(
        actor_rollout_ref=Config(
            actor=Config(
                gradient_record=Config(enabled=True, mode=mode, interval=10, output_dir="unused"),
                strategy="fsdp",
                policy_loss=Config(loss_mode="vanilla"),
                fsdp_config=Config(fsdp_size=-1, use_orig_params=False),
            ),
            rollout=Config(mode="async", multi_turn=Config(enable=True), n=2),
        ),
        algorithm=Config(adv_estimator="grpo", use_kl_in_reward=False),
        trainer=Config(n_gpus_per_node=2, nnodes=1),
        data=Config(return_raw_chat=True),
        ttrl=Config(enable=True, reward_mode="bfcl", share_rollouts=True),
        bfcl_supervised=Config(enabled=False),
    )


def model():
    module = nn.Module()
    module.first = nn.Parameter(torch.zeros(3, dtype=torch.float64))
    module.second = nn.Parameter(torch.zeros(2, dtype=torch.float64))
    return module


def batch():
    tensors = {name: torch.zeros(2, 2) for name in diagnostics.BATCH_KEYS}
    tensors.update(
        diagnostic_supervised_advantages=torch.tensor([[1.0, 1.0], [-1.0, -1.0]]),
        advantages=torch.tensor([[-1.0, -1.0], [1.0, 1.0]]),
    )
    return tensors


def metadata():
    return {"step": 1, "epoch": 0, "minibatch": 0, "rng_fingerprint": "same_rng", "objective": {"loss": "policy"}}


class TestPairedGradientHelpers(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_chunked_statistics_include_all_coordinates_and_none_gradients(self):
        module = model()
        module.first.grad = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        with patch.object(diagnostics, "CHUNK_SIZE", 2):
            cache = diagnostics.capture_policy_gradient(module)
            module.first.grad = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64)
            module.second.grad = torch.tensor([7.0, 8.0], dtype=torch.float64)
            statistics = diagnostics.paired_gradient_statistics(module, cache)
        torch.testing.assert_close(statistics, torch.tensor([32.0, 14.0, 190.0], dtype=torch.float64), rtol=0, atol=0)
        self.assertIsNone(cache["parameters"][1]["gradient"])
        self.assertEqual(cache["numel"], 5)

    def test_cpu_cache_is_independent_and_preserves_float64_precision(self):
        module = model()
        value = 1 + 2**-40
        module.first.grad = torch.tensor([value, 0.0, 0.0], dtype=torch.float64)
        cache = diagnostics.capture_policy_gradient(module)
        module.first.grad.zero_()
        saved = cache["parameters"][0]["gradient"]
        self.assertEqual(saved.dtype, torch.float64)
        self.assertEqual(saved.device.type, "cpu")
        self.assertEqual(saved[0].item(), value)
        self.assertEqual(diagnostics.paired_gradient_statistics(module, cache)[1].item(), value**2)

    def test_nonfinite_gradients_and_changed_parameters_fail(self):
        module = model()
        module.first.grad = torch.tensor([float("nan"), 0.0, 0.0], dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            diagnostics.capture_policy_gradient(module)
        module.first.grad.zero_()
        cache = diagnostics.capture_policy_gradient(module)
        module.first.grad[0] = float("inf")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            diagnostics.paired_gradient_statistics(module, cache)
        module.first.grad.zero_()
        with torch.no_grad():
            module.first[0] = 1.0
        with self.assertRaisesRegex(ValueError, "unchanged"):
            diagnostics.paired_gradient_statistics(module, cache)

    def test_changed_trainable_layout_and_frozen_parameters_fail(self):
        module = model()
        cache = diagnostics.capture_policy_gradient(module)
        module.second.requires_grad_(False)
        with self.assertRaisesRegex(ValueError, "layouts"):
            diagnostics.paired_gradient_statistics(module, cache)
        cache = diagnostics.capture_policy_gradient(module)
        with torch.no_grad():
            module.second[0] = 1.0
        with self.assertRaisesRegex(ValueError, "unchanged"):
            diagnostics.paired_gradient_statistics(module, cache)

    def save(self, statistics, *, rank=0, world_size=1):
        return diagnostics.save_paired_gradient_statistics(
            torch.tensor(statistics, dtype=torch.float64),
            batch(),
            metadata(),
            self.root / "paired",
            cache=diagnostics.capture_policy_gradient(model()),
            rank=rank,
            world_size=world_size,
        )

    def test_paired_record_is_small_json_with_audit_metadata_and_no_binary(self):
        result = self.save([0.0, 4.0, 9.0])
        self.assertEqual(result["angle_deg"], 90)
        self.assertEqual(result["supervised_norm"], 2)
        self.assertEqual(result["unsupervised_norm"], 3)
        paths = list((self.root / "paired").iterdir())
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0].suffix, ".json")
        self.assertLess(paths[0].stat().st_size, 5000)
        self.assertEqual(json.loads(paths[0].read_text()), result)
        fingerprints = result["rank_fingerprints"][0]
        self.assertNotEqual(
            fingerprints["supervised_advantage_fingerprint"], fingerprints["unsupervised_advantage_fingerprint"]
        )
        with self.assertRaises(FileExistsError):
            self.save([0.0, 4.0, 9.0])

    def test_global_statistics_sum_shards_instead_of_averaging_cosines(self):
        def reduce(statistics, op):
            statistics.add_(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64))

        def gather(values, local):
            values[:] = [local, {**local, "rank": 1, "numel": 1}]

        with patch.object(torch.distributed, "all_reduce", side_effect=reduce) as reduction:
            with patch.object(torch.distributed, "all_gather_object", side_effect=gather):
                result = self.save([10.0, 100.0, 1.0], world_size=2)
        self.assertEqual(reduction.call_args.args[0].numel(), 3)
        self.assertEqual(result["dot_product"], 9)
        self.assertAlmostEqual(result["cosine"], 9 / math.sqrt(202))
        self.assertEqual(result["world_size"], 2)
        self.assertEqual(result["numel"], 6)

    def test_nonzero_rank_returns_statistics_without_writing_files(self):
        def gather(values, local):
            values[:] = [{**local, "rank": 0}, local]

        with patch.object(torch.distributed, "all_reduce"):
            with patch.object(torch.distributed, "all_gather_object", side_effect=gather):
                result = self.save([1.0, 1.0, 1.0], rank=1, world_size=2)
        self.assertEqual(result["cosine"], 1)
        self.assertFalse((self.root / "paired").exists())

    def test_zero_gradient_direction_is_explicitly_undefined(self):
        for values, status in (
            ([0, 0, 0], "both_zero"),
            ([0, 0, 1], "supervised_zero"),
            ([0, 1, 0], "unsupervised_zero"),
        ):
            result = diagnostics._comparison_statistics(*values)
            self.assertEqual(result["status"], status)
            self.assertIsNone(result["cosine"])
            self.assertIsNone(result["angle_deg"])

    def test_paired_export_reuses_existing_csv_and_plot(self):
        self.save([0.0, 1.0, 1.0])
        destination = self.root / "export"
        path = diagnostics.compare_paired_run(self.root / "paired", destination)
        self.assertIn("90.0", path.read_text())
        self.assertGreater((destination / "gradient_comparison.png").stat().st_size, 1000)
        with self.assertRaisesRegex(ValueError, "No paired"):
            diagnostics.compare_paired_run(self.root / "missing", destination)


class TestPairedGradientConfiguration(unittest.TestCase):
    def test_full_mode_remains_default_and_does_not_require_ttrl(self):
        config = configuration()
        config.actor_rollout_ref.actor.gradient_record.pop("mode")
        config.ttrl.enable = False
        diagnostics.validate_gradient_recording(config)
        self.assertFalse(diagnostics.paired_recording_due(config.actor_rollout_ref.actor.gradient_record, 1))

    def test_paired_interval_and_enabled_flags(self):
        record = configuration().actor_rollout_ref.actor.gradient_record
        diagnostics.validate_gradient_recording(configuration())
        self.assertTrue(diagnostics.paired_recording_due(record, 1))
        self.assertTrue(diagnostics.paired_recording_due(record, 10))
        self.assertFalse(diagnostics.paired_recording_due(record, 2))
        record.enabled = False
        self.assertFalse(diagnostics.paired_recording_due(record, 1))

    def test_paired_requires_shared_bfcl_training_and_supported_rollouts(self):
        changes = (
            ("ttrl", "enable", False),
            ("ttrl", "reward_mode", "math"),
            ("ttrl", "share_rollouts", False),
            ("bfcl_supervised", "enabled", True),
            ("data", "return_raw_chat", False),
        )
        for section, key, value in changes:
            with self.subTest(section=section, key=key):
                config = configuration()
                config[section][key] = value
                with self.assertRaises(ValueError):
                    diagnostics.validate_gradient_recording(config)
        for key, value in (("n", 1), ("n", True), ("mode", "sync")):
            config = configuration()
            config.actor_rollout_ref.rollout[key] = value
            with self.assertRaises(ValueError):
                diagnostics.validate_gradient_recording(config)
        config = configuration()
        config.actor_rollout_ref.rollout.multi_turn.enable = False
        with self.assertRaises(ValueError):
            diagnostics.validate_gradient_recording(config)
        config = configuration("unknown")
        with self.assertRaisesRegex(ValueError, "mode"):
            diagnostics.validate_gradient_recording(config)


class TestPairedBaselineHelpers(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_gaussian_covers_missing_gradient_coordinates_and_preserves_rng(self):
        module = model()
        module.first.grad = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        cache = diagnostics.capture_policy_gradient(module)
        state = torch.random.get_rng_state().clone()
        first = diagnostics.random_gradient_statistics(cache, seed=17)
        repeated = diagnostics.random_gradient_statistics(cache, seed=17)
        other_rank = diagnostics.random_gradient_statistics(cache, seed=17, rank=1)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), state))
        torch.testing.assert_close(first, repeated, rtol=0, atol=0)
        self.assertFalse(torch.equal(first, other_rank))
        self.assertEqual(first.dtype, torch.float64)
        self.assertEqual(first.shape, (3,))
        self.assertEqual(first[1].item(), 14)
        self.assertIsNone(cache["parameters"][1]["gradient"])
        restricted = {**cache, "parameters": cache["parameters"][:1], "numel": 3}
        restricted_statistics = diagnostics.random_gradient_statistics(restricted, seed=17)
        torch.testing.assert_close(first[:2], restricted_statistics[:2], rtol=0, atol=0)
        self.assertGreater(first[2].item(), restricted_statistics[2].item())
        module.second.grad = torch.zeros(2, dtype=torch.float64)
        explicit_zero = diagnostics.random_gradient_statistics(diagnostics.capture_policy_gradient(module), seed=17)
        torch.testing.assert_close(first, explicit_zero, rtol=0, atol=0)
        with patch.object(diagnostics, "CHUNK_SIZE", 1):
            rechunked = diagnostics.random_gradient_statistics(diagnostics.capture_policy_gradient(module), seed=17)
        torch.testing.assert_close(first, rechunked, rtol=0, atol=0)

    def test_gaussian_remains_nonzero_when_supervised_gradient_is_zero(self):
        cache = diagnostics.capture_policy_gradient(model())
        values = diagnostics.random_gradient_statistics(cache, seed=19)
        self.assertEqual(values[0].item(), 0)
        self.assertEqual(values[1].item(), 0)
        self.assertGreater(values[2].item(), 0)
        result = diagnostics.save_paired_gradient_statistics(
            torch.zeros(3, dtype=torch.float64), batch(), metadata(), self.root / "zero",
            cache=cache, baselines={"random_direction": values, "shuffled_labels": torch.zeros(3)},
        )
        self.assertEqual(result["baselines"]["random_direction"]["status"], "supervised_zero")
        self.assertEqual(result["baselines"]["random_direction"]["unsupervised_norm"], 1)
        self.assertEqual(result["baselines"]["shuffled_labels"]["status"], "both_zero")
        for baseline in result["baselines"].values():
            self.assertIsNone(baseline["cosine"])
            self.assertIsNone(baseline["angle_deg"])

    def test_baselines_reduce_global_statistics_before_normalizing_gaussian(self):
        remotes = iter(([-1, 4, 1], [-1, 4, 11], [2, 4, 5]))

        def reduce(values, op):
            values.add_(torch.tensor(next(remotes), dtype=torch.float64))

        def gather(values, local):
            values[:] = [local, {**local, "rank": 1, "numel": 7}]

        baselines = {
            "random_direction": torch.tensor([3, 9, 25], dtype=torch.float64),
            "shuffled_labels": torch.tensor([-4, 9, 4], dtype=torch.float64),
        }
        baseline_metadata = {"seed": 41, "draws_per_update": 1}
        with patch.object(torch.distributed, "all_reduce", side_effect=reduce) as reduction:
            with patch.object(torch.distributed, "all_gather_object", side_effect=gather):
                result = diagnostics.save_paired_gradient_statistics(
                    torch.tensor([6, 9, 4], dtype=torch.float64), batch(), metadata(), self.root / "distributed",
                    cache=diagnostics.capture_policy_gradient(model()), world_size=2,
                    baselines=baselines, baseline_metadata=baseline_metadata,
                )
        self.assertEqual(reduction.call_count, 3)
        self.assertEqual(result["dot_product"], 5)
        self.assertEqual(result["numel"], 12)
        random = result["baselines"]["random_direction"]
        self.assertAlmostEqual(random["dot_product"], 1 / 3)
        self.assertEqual(random["supervised_squared_norm"], 13)
        self.assertEqual(random["unsupervised_squared_norm"], 1)
        self.assertAlmostEqual(random["cosine"], 1 / (3 * math.sqrt(13)))
        shuffled = result["baselines"]["shuffled_labels"]
        self.assertEqual(shuffled["dot_product"], -2)
        self.assertAlmostEqual(shuffled["cosine"], -2 / math.sqrt(117))
        self.assertEqual(result["baseline_metadata"], baseline_metadata)
        torch.testing.assert_close(baselines["random_direction"], torch.tensor([3, 9, 25], dtype=torch.float64))

    def test_export_mixes_old_records_with_new_baseline_columns(self):
        directory = self.root / "mixed"
        cache = diagnostics.capture_policy_gradient(model())
        diagnostics.save_paired_gradient_statistics(
            torch.tensor([0, 4, 4], dtype=torch.float64), batch(), metadata(), directory, cache=cache
        )
        diagnostics.save_paired_gradient_statistics(
            torch.tensor([2, 4, 4], dtype=torch.float64), batch(), {**metadata(), "step": 2}, directory,
            cache=cache,
            baselines={
                "random_direction": torch.tensor([2, 4, 16], dtype=torch.float64),
                "shuffled_labels": torch.tensor([-2, 4, 4], dtype=torch.float64),
            },
            baseline_metadata={"seed": 17},
        )
        destination = self.root / "export"
        csv_path = diagnostics.compare_paired_run(directory, destination)
        with csv_path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["random_direction_cosine"], "")
        self.assertEqual(rows[0]["shuffled_labels_cosine"], "")
        self.assertAlmostEqual(float(rows[1]["cosine"]), 0.5)
        self.assertAlmostEqual(float(rows[1]["random_direction_cosine"]), 0.25)
        self.assertAlmostEqual(float(rows[1]["shuffled_labels_cosine"]), -0.5)
        self.assertEqual(rows[1]["random_direction_unsupervised_norm"], "1.0")
        self.assertEqual(rows[1]["baseline_seed"], "17")
        self.assertGreater((destination / "gradient_comparison.png").stat().st_size, 1000)


class TestTrajectoryExportAndConfiguration(unittest.TestCase):
    def test_existing_paired_export_handles_trajectory_only_records_and_null_directions(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "trajectory"
            module = nn.Linear(2, 1, bias=False, dtype=torch.float64)
            with torch.no_grad():
                module.weight.zero_()
            optimizer = torch.optim.AdamW(module.parameters())
            trajectory = diagnostics.OptimizationTrajectory(module, optimizer)
            records = []
            for step, supervised_delta, unsupervised_delta in (
                (1, [0, 0], [1, 0]),
                (2, [1, 1], [-1, 0]),
            ):
                def supervised_update(supervised_delta=supervised_delta):
                    with torch.no_grad():
                        module.weight.add_(torch.tensor([supervised_delta], dtype=torch.float64))

                trajectory.supervised_step(module, optimizer, supervised_update)
                with torch.no_grad():
                    module.weight.add_(torch.tensor([unsupervised_delta], dtype=torch.float64))
                record = trajectory.finish_step(module, {"step": step, "epoch": 0, "minibatch": 0})
                diagnostics.save_trajectory_statistics(record, directory)
                records.append(record)
            self.assertIsNone(records[0]["endpoint"]["cosine"])
            self.assertIsNone(records[0]["path"]["cosine"])
            self.assertIsNone(records[1]["endpoint"]["cosine"])
            self.assertAlmostEqual(records[1]["path"]["cosine"], -0.5)
            saved = sorted(directory.glob("trajectory_step_*.json"))
            self.assertEqual(len(saved), 2)
            self.assertIsNone(json.loads(saved[0].read_text(encoding="utf-8"))["endpoint"]["cosine"])
            self.assertFalse(list(directory.glob("step_*.json")))
            destination = Path(temporary) / "export"
            path = diagnostics.compare_paired_run(directory, destination)
            self.assertEqual(path.name, "trajectory_comparison.csv")
            with path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["endpoint_cosine"], "")
            self.assertEqual(rows[0]["path_cosine"], "")
            self.assertEqual(rows[0]["endpoint_status"], "supervised_zero")
            self.assertEqual(rows[1]["endpoint_cosine"], "")
            self.assertEqual(rows[1]["endpoint_status"], "unsupervised_zero")
            self.assertAlmostEqual(float(rows[1]["path_cosine"]), -0.5)
            self.assertEqual(float(rows[1]["endpoint_dot_product"]), 0)
            self.assertEqual(float(rows[1]["path_dot_product"]), -1)
            self.assertEqual(rows[1]["updates"], "2")
            self.assertGreater((destination / "trajectory_comparison.png").stat().st_size, 1000)

    def test_trajectory_requires_paired_fsdp1_while_ordinary_paired_fsdp2_remains_supported(self):
        config = configuration()
        config.actor_rollout_ref.actor.strategy = "fsdp2"
        diagnostics.validate_gradient_recording(config)
        config.actor_rollout_ref.actor.gradient_record["trajectory"] = {"enabled": False}
        diagnostics.validate_gradient_recording(config)
        for mode, strategy in (("paired", "fsdp2"), ("full", "fsdp"), ("full", "fsdp2")):
            with self.subTest(mode=mode, strategy=strategy):
                config = configuration(mode)
                config.actor_rollout_ref.actor.strategy = strategy
                config.actor_rollout_ref.actor.gradient_record["trajectory"] = {"enabled": True}
                with self.assertRaisesRegex(ValueError, "mode=paired and strategy=fsdp"):
                    diagnostics.validate_gradient_recording(config)
        config = configuration()
        config.actor_rollout_ref.actor.gradient_record["trajectory"] = {"enabled": True}
        diagnostics.validate_gradient_recording(config)


class TestTrajectoryRandomBaselines(unittest.TestCase):
    @staticmethod
    def flatten(module):
        return torch.cat([parameter.detach().reshape(-1).double() for parameter in module.parameters()])

    @staticmethod
    def random_displacement(trajectory):
        return torch.cat([value.reshape(-1).double() for value in trajectory.random_displacement])

    def assert_state_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_state_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_state_equal(a, b)
        else:
            self.assertEqual(left, right)

    @staticmethod
    def optimizer(module):
        return torch.optim.AdamW(module.parameters(), lr=0.02, betas=(0.7, 0.85), weight_decay=0.13)

    @staticmethod
    def update(module, optimizer, target):
        inputs = torch.tensor([[1.0, -0.5], [0.2, 0.8], [-0.7, 1.1]], dtype=torch.float64)
        optimizer.zero_grad()
        module.num_forwards.add_(1)
        loss = (module(inputs) - target).square().mean()
        loss += 0.07 * sum(parameter.square().sum() for parameter in module.parameters())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), 0.15)
        optimizer.step()

    def assert_comparison(self, actual, left, right):
        expected = diagnostics._comparison_statistics(
            torch.dot(left, right).item(), torch.dot(left, left).item(), torch.dot(right, right).item()
        )
        self.assertEqual(actual["status"], expected["status"])
        for key, value in expected.items():
            if isinstance(value, float):
                self.assertAlmostEqual(actual[key], value, places=12)
            else:
                self.assertEqual(actual[key], value)

    def test_baseline_selection_preserves_legacy_defaults_and_master_switch(self):
        for enabled, shuffled, random in (
            (False, True, True), (True, True, True), (True, True, False),
            (True, False, True), (True, False, False),
        ):
            with self.subTest(enabled=enabled, shuffled=shuffled, random=random):
                config = {
                    "enabled": enabled,
                    "shuffled_labels": {"enabled": shuffled},
                    "random_direction": {"enabled": random},
                }
                self.assertEqual(diagnostics.baseline_enabled(config, "shuffled_labels"), enabled and shuffled)
                self.assertEqual(diagnostics.baseline_enabled(config, "random_direction"), enabled and random)
        self.assertTrue(diagnostics.baseline_enabled({"enabled": True}, "shuffled_labels"))
        self.assertTrue(diagnostics.baseline_enabled({"enabled": True}, "random_direction"))
        self.assertFalse(diagnostics.baseline_enabled({}, "shuffled_labels"))

    def test_shuffled_only_matches_full_baseline_path_without_allocating_random_displacement(self):
        torch.manual_seed(31)
        original = nn.Linear(2, 1, dtype=torch.float64)
        original.register_buffer("num_forwards", torch.zeros((), dtype=torch.long))
        original_optimizer = self.optimizer(original)
        self.update(original, original_optimizer, torch.zeros(3, 1, dtype=torch.float64))
        branches = []
        for random_enabled in (True, False):
            module = copy.deepcopy(original)
            optimizer = self.optimizer(module)
            optimizer.load_state_dict(copy.deepcopy(original_optimizer.state_dict()))
            trajectory = diagnostics.OptimizationTrajectory(
                module, optimizer, baselines=True, random_direction=random_enabled, seed=17
            )
            branches.append((module, optimizer, trajectory))
        shuffled_only = branches[1][2]
        self.assertTrue(shuffled_only.shuffled_labels_enabled)
        self.assertFalse(shuffled_only.random_direction_enabled)
        self.assertFalse(hasattr(shuffled_only, "random_displacement"))
        with patch.object(shuffled_only, "_random_chunks", side_effect=AssertionError("random baseline ran")):
            for step in (1, 2):
                records = []
                for module, optimizer, trajectory in branches:
                    trajectory.supervised_step(
                        module, optimizer,
                        lambda: self.update(module, optimizer, torch.ones(3, 1, dtype=torch.float64)),
                        shuffled_callback=lambda: self.update(
                            module, optimizer, -torch.ones(3, 1, dtype=torch.float64)
                        ),
                    )
                    self.update(module, optimizer, torch.full((3, 1), 0.3, dtype=torch.float64))
                    records.append(trajectory.finish_step(module, {**metadata(), "step": step}))
                self.assertEqual(set(records[1]["baselines"]), {"shuffled_labels"})
                self.assertEqual(records[0]["baselines"]["shuffled_labels"], records[1]["baselines"]["shuffled_labels"])
                for comparison in ("endpoint", "path"):
                    self.assertEqual(records[0][comparison], records[1][comparison])
                self.assert_state_equal(branches[0][0].state_dict(), branches[1][0].state_dict())
                self.assert_state_equal(branches[0][1].state_dict(), branches[1][1].state_dict())
                self.assert_state_equal(branches[0][2].shuffled_parameters, shuffled_only.shuffled_parameters)
                self.assert_state_equal(
                    list(diagnostics._restore_state(branches[0][2].shuffled_optimizer_state).values()),
                    list(diagnostics._restore_state(shuffled_only.shuffled_optimizer_state).values()),
                )

    def test_random_only_requires_no_shuffled_optimizer_branch_or_callback(self):
        module = nn.Linear(2, 1, dtype=torch.float64)
        module.register_buffer("num_forwards", torch.zeros((), dtype=torch.long))
        optimizer = self.optimizer(module)
        trajectory = diagnostics.OptimizationTrajectory(
            module, optimizer, baselines=True, shuffled_labels=False, seed=17
        )
        self.assertFalse(trajectory.shuffled_labels_enabled)
        self.assertTrue(trajectory.random_direction_enabled)
        for name in ("shuffled_parameters", "shuffled_buffers", "shuffled_optimizer_state"):
            self.assertFalse(hasattr(trajectory, name), name)
        trajectory.supervised_step(
            module, optimizer, lambda: self.update(module, optimizer, torch.ones(3, 1, dtype=torch.float64))
        )
        self.update(module, optimizer, torch.zeros(3, 1, dtype=torch.float64))
        record = trajectory.finish_step(module, metadata())
        self.assertEqual(set(record["baselines"]), {"random_update"})
        self.assertGreater(record["baselines"]["random_update"]["target_step_norm"], 0)

    def test_three_optimizer_paths_match_adamw_oracles_and_random_actual_increments(self):
        torch.manual_seed(31)
        module = nn.Sequential(nn.Linear(2, 3, dtype=torch.float64), nn.Dropout(0.35),
                               nn.Linear(3, 1, dtype=torch.float64))
        module.register_buffer("num_forwards", torch.zeros((), dtype=torch.long))
        optimizer = self.optimizer(module)
        self.update(module, optimizer, torch.zeros(3, 1, dtype=torch.float64))
        origin = self.flatten(module)
        references = [copy.deepcopy(module) for _ in range(3)]
        reference_optimizers = [self.optimizer(item) for item in references]
        for reference_optimizer in reference_optimizers:
            reference_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        supervised, shuffled, unsupervised = references
        supervised_optimizer, shuffled_optimizer, unsupervised_optimizer = reference_optimizers
        trajectory = diagnostics.OptimizationTrajectory(module, optimizer, baselines=True, seed=17)
        self.assertIs(trajectory.supervised_parameters, trajectory.initial_parameters)
        self.assertIs(trajectory.shuffled_parameters, trajectory.initial_parameters)
        self.assertIs(trajectory.supervised_optimizer_state, trajectory.shuffled_optimizer_state)
        increments = {name: [] for name in ("supervised", "shuffled_labels", "unsupervised", "random_update")}

        for step, learning_rate in enumerate((0.02, 0.007, 0.011), start=1):
            for item in (optimizer, *reference_optimizers):
                item.param_groups[0]["lr"] = learning_rate
            targets = [torch.tensor([[0.9], [-0.4], [0.3]], dtype=torch.float64) * step,
                       torch.tensor([[-0.4], [0.3], [0.9]], dtype=torch.float64) * step,
                       torch.tensor([[-0.7], [0.5], [-0.1]], dtype=torch.float64) / step]
            supervised_target, shuffled_target, unsupervised_target = targets
            before = [self.flatten(item) for item in references]
            random_before = self.random_displacement(trajectory)
            torch.manual_seed(100 + step)
            rng = torch.random.get_rng_state().clone()
            for reference, reference_optimizer, target in zip(references[:2], reference_optimizers[:2], targets[:2]):
                with torch.random.fork_rng(devices=[]):
                    self.update(reference, reference_optimizer, target)
            live_state = copy.deepcopy(module.state_dict())
            live_optimizer_state = copy.deepcopy(optimizer.state_dict())

            def supervised_update(target=supervised_target):
                self.update(module, optimizer, target)
                self.assert_state_equal(module.state_dict(), supervised.state_dict())
                self.assert_state_equal(optimizer.state_dict(), supervised_optimizer.state_dict())

            def shuffled_update(target=shuffled_target):
                self.update(module, optimizer, target)
                self.assert_state_equal(module.state_dict(), shuffled.state_dict())
                self.assert_state_equal(optimizer.state_dict(), shuffled_optimizer.state_dict())

            with patch.object(trajectory, "_parameters", wraps=trajectory._parameters) as snapshot:
                trajectory.supervised_step(module, optimizer, supervised_update, shuffled_callback=shuffled_update)
            self.assertEqual(snapshot.call_count, 3)  # One live snapshot, one new snapshot per shadow branch.
            self.assert_state_equal(module.state_dict(), live_state)
            self.assert_state_equal(optimizer.state_dict(), live_optimizer_state)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
            self.assertTrue(all(parameter.grad is None for parameter in module.parameters()))
            self.update(module, optimizer, unsupervised_target)
            with torch.random.fork_rng(devices=[]):
                torch.random.set_rng_state(rng)
                self.update(unsupervised, unsupervised_optimizer, unsupervised_target)
            self.assert_state_equal(module.state_dict(), unsupervised.state_dict())
            self.assert_state_equal(optimizer.state_dict(), unsupervised_optimizer.state_dict())
            rng_after_real_update = torch.random.get_rng_state().clone()
            record = trajectory.finish_step(module, {"step": step, "epoch": 0, "minibatch": 0})
            self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_after_real_update))
            self.assertEqual(record["version"], 2)
            self.assertEqual(record["updates"], step)
            for name, reference, previous in zip(
                ("supervised", "shuffled_labels", "unsupervised"), references, before
            ):
                increments[name].append(self.flatten(reference) - previous)
            random_after = self.random_displacement(trajectory)
            increments["random_update"].append(random_after - random_before)
            for name, endpoint, comparison in (
                ("unsupervised", self.flatten(unsupervised) - origin, record),
                ("shuffled_labels", self.flatten(shuffled) - origin, record["baselines"]["shuffled_labels"]),
                ("random_update", random_after, record["baselines"]["random_update"]),
            ):
                self.assert_comparison(comparison["endpoint"], self.flatten(supervised) - origin, endpoint)
                self.assert_comparison(comparison["path"], torch.cat(increments["supervised"]),
                                       torch.cat(increments[name]))
            random = record["baselines"]["random_update"]
            self.assertAlmostEqual(random["target_step_norm"], increments["unsupervised"][-1].norm().item())
            self.assertAlmostEqual(random["applied_step_norm"], increments["random_update"][-1].norm().item())
            self.assertLess(random["step_norm_relative_error"], 1e-5)
            self.assertIn("without_optimizer", random["update_rule"])

    def test_second_shadow_failure_restores_live_state_rng_lr_and_both_shadow_histories(self):
        module = nn.Linear(2, 1, dtype=torch.float64)
        module.register_buffer("num_forwards", torch.zeros((), dtype=torch.long))
        optimizer = self.optimizer(module)
        self.update(module, optimizer, torch.zeros(3, 1, dtype=torch.float64))
        trajectory = diagnostics.OptimizationTrajectory(module, optimizer, baselines=True)
        original = copy.deepcopy(module.state_dict())
        optimizer_before = copy.deepcopy(optimizer.state_dict())
        supervised_before = trajectory.supervised_parameters
        shuffled_before = trajectory.shuffled_parameters
        rng = torch.random.get_rng_state().clone()

        def supervised_update():
            self.update(module, optimizer, torch.ones(3, 1, dtype=torch.float64))

        def failing_shuffle():
            self.update(module, optimizer, -torch.ones(3, 1, dtype=torch.float64))
            optimizer.param_groups[0]["lr"] = 7
            torch.rand(4)
            raise RuntimeError("shuffled update failed")

        with self.assertRaisesRegex(RuntimeError, "shuffled update failed"):
            trajectory.supervised_step(module, optimizer, supervised_update, shuffled_callback=failing_shuffle)
        self.assert_state_equal(module.state_dict(), original)
        self.assert_state_equal(optimizer.state_dict(), optimizer_before)
        self.assertIs(trajectory.supervised_parameters, supervised_before)
        self.assertIs(trajectory.shuffled_parameters, shuffled_before)
        self.assertIsNone(trajectory.pending)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
        self.assertTrue(all(parameter.grad is None for parameter in module.parameters()))

    def test_global_random_norm_and_all_baseline_statistics_reduce_shards(self):
        def statistics(left, right):
            return torch.tensor([torch.dot(left, right), torch.dot(left, left), torch.dot(right, right)],
                                dtype=torch.float64)

        scale = math.sqrt(25 / 10)
        supervised = [torch.tensor([1., 0.]), torch.tensor([0., 2.])]
        unsupervised = [torch.tensor([3., 0.]), torch.tensor([0., 4.])]
        shuffled = [torch.tensor([0., 1.]), torch.tensor([2., 0.])]
        gaussian = [torch.tensor([1., 2.]), torch.tensor([2., -1.])]
        random = [(value.double() * scale).float().double() for value in gaussian]
        local_statistics = []
        for rank in range(2):
            pieces = []
            for target in (unsupervised[rank].double(), shuffled[rank].double(), random[rank]):
                values = statistics(supervised[rank].double(), target)
                pieces.extend((values, values))
            local_statistics.append(torch.cat((*pieces, random[rank].square().sum().reshape(1))))
        expected = local_statistics[0] + local_statistics[1]
        for rank in range(2):
            module = nn.Linear(2, 1, bias=False, dtype=torch.float64)
            with torch.no_grad():
                module.weight.zero_()
            optimizer = torch.optim.AdamW(module.parameters())
            trajectory = diagnostics.OptimizationTrajectory(module, optimizer, baselines=True)

            def add(values, module=module):
                with torch.no_grad():
                    module.weight.add_(values.double().reshape(1, 2))

            trajectory.supervised_step(module, optimizer, lambda rank=rank: add(supervised[rank]),
                                       shuffled_callback=lambda rank=rank: add(shuffled[rank]))
            add(unsupervised[rank])

            def reduce(values, op, rank=rank):
                if values.numel() == 2:
                    self.assertAlmostEqual(values[0].item(), (9, 16)[rank])
                    values.copy_(torch.tensor([25, 10], dtype=torch.float64))
                else:
                    self.assertEqual(values.numel(), 19)
                    torch.testing.assert_close(values, local_statistics[rank], rtol=0, atol=0)
                    values.add_(local_statistics[1 - rank])

            def random_chunks(key, current_rank, rank=rank):
                self.assertEqual(current_rank, rank)
                return iter(((0, 0, gaussian[rank].double()),))

            with patch.object(trajectory, "_random_chunks", side_effect=random_chunks):
                with patch.object(torch.distributed, "all_reduce", side_effect=reduce) as reduction:
                    record = trajectory.finish_step(module, metadata(), rank=rank, world_size=2)
            self.assertEqual(reduction.call_count, 2)
            for index, comparison in enumerate((record, record["baselines"]["shuffled_labels"],
                                                 record["baselines"]["random_update"])):
                for offset, name in ((0, "endpoint"), (3, "path")):
                    actual = comparison[name]
                    values = expected[6 * index + offset:6 * index + offset + 3]
                    self.assertEqual(actual, diagnostics._comparison_statistics(*values.tolist()))
            self.assertEqual(record["baselines"]["random_update"]["target_step_norm"], 5)

    def test_random_rng_is_reproducible_rank_step_separated_and_chunk_independent(self):
        module = model()
        trajectory = diagnostics.OptimizationTrajectory(module, torch.optim.AdamW(module.parameters()),
                                                       baselines=True, seed=17)
        rng = torch.random.get_rng_state().clone()
        first = list(trajectory._random_chunks("step1", 0))
        with patch.object(diagnostics, "CHUNK_SIZE", 1):
            repeated = list(trajectory._random_chunks("step1", 0))
        other_rank = list(trajectory._random_chunks("step1", 1))
        other_step = list(trajectory._random_chunks("step2", 0))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
        for a, b, c, d in zip(first, repeated, other_rank, other_step):
            self.assertTrue(torch.equal(a[2], b[2]))
            self.assertFalse(torch.equal(a[2], c[2]))
            self.assertFalse(torch.equal(a[2], d[2]))

    def test_zero_actual_update_has_zero_random_step_and_undefined_direction(self):
        module = nn.Linear(2, 1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            module.weight.zero_()
        optimizer = torch.optim.AdamW(module.parameters())
        trajectory = diagnostics.OptimizationTrajectory(module, optimizer, baselines=True)
        trajectory.supervised_step(module, optimizer, lambda: None, shuffled_callback=lambda: None)
        record = trajectory.finish_step(module, metadata())
        random = record["baselines"]["random_update"]
        self.assertEqual(random["target_step_norm"], 0)
        self.assertEqual(random["applied_step_norm"], 0)
        self.assertEqual(random["step_norm_relative_error"], 0)
        for comparison in (record, *record["baselines"].values()):
            for name in ("endpoint", "path"):
                self.assertEqual(comparison[name]["status"], "both_zero")
                self.assertIsNone(comparison[name]["cosine"])

    def test_tiny_random_updates_report_fp32_rounding_and_use_realized_path_increments(self):
        module = nn.Linear(1, 1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            module.weight.zero_()
        optimizer = torch.optim.AdamW(module.parameters())
        trajectory = diagnostics.OptimizationTrajectory(module, optimizer, baselines=True)

        def add_one():
            with torch.no_grad():
                module.weight.add_(1)

        trajectory.supervised_step(module, optimizer, add_one, shuffled_callback=add_one)
        add_one()
        first = trajectory.finish_step(module, metadata())
        random_before = self.random_displacement(trajectory)
        trajectory.supervised_step(module, optimizer, lambda: None, shuffled_callback=lambda: None)
        with torch.no_grad():
            module.weight.add_(2 ** -40)
        final = trajectory.finish_step(module, {**metadata(), "step": 2})
        random = final["baselines"]["random_update"]
        self.assertGreater(random["target_step_norm"], 0)
        self.assertEqual(random["applied_step_norm"], 0)
        self.assertEqual(random["step_norm_relative_error"], 1)
        self.assertTrue(torch.equal(self.random_displacement(trajectory), random_before))
        self.assertEqual(random["path"], first["baselines"]["random_update"]["path"])

    def test_resumed_segment_and_old_new_exports_keep_legacy_columns(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "trajectory"
            module = nn.Linear(2, 1, bias=False, dtype=torch.float64)
            optimizer = torch.optim.AdamW(module.parameters())

            def add(values):
                with torch.no_grad():
                    module.weight.add_(torch.tensor([values], dtype=torch.float64))

            old = diagnostics.OptimizationTrajectory(module, optimizer)
            old.supervised_step(module, optimizer, lambda: add([1, 0]))
            add([0, 1])
            old_record = old.finish_step(module, metadata())
            diagnostics.save_trajectory_statistics(old_record, directory)
            resumed = diagnostics.OptimizationTrajectory(module, optimizer, baselines=True, seed=19)
            resumed.supervised_step(module, optimizer, lambda: add([0, 2]),
                                    shuffled_callback=lambda: add([2, 0]))
            add([1, 0])
            new_record = resumed.finish_step(module, {**metadata(), "step": 200})
            diagnostics.save_trajectory_statistics(new_record, directory)
            self.assertEqual(new_record["updates"], 1)
            self.assertEqual(new_record["start_update"], "step_000200_epoch_000_minibatch_0000")
            self.assertNotEqual(new_record["origin_parameter_fingerprint_local"],
                                old_record["origin_parameter_fingerprint_local"])
            self.assertEqual(old_record["version"], 1)
            self.assertNotIn("baselines", old_record)
            self.assertAlmostEqual(new_record["endpoint"]["supervised_norm"], 2)
            self.assertAlmostEqual(new_record["endpoint"]["unsupervised_norm"], 1)
            output = Path(temporary) / "export"
            path = diagnostics.compare_paired_run(directory, output)
            with path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["shuffled_labels_endpoint_cosine"], "")
            self.assertEqual(rows[0]["random_update_endpoint_cosine"], "")
            self.assertEqual(rows[0]["start_update"], "step_000001_epoch_000_minibatch_0000")
            self.assertEqual(rows[1]["baseline_seed"], "19")
            self.assertEqual(rows[1]["updates"], "1")
            for name in ("endpoint", "path"):
                self.assertIn(f"{name}_dot_product", rows[1])
                self.assertIn(f"shuffled_labels_{name}_dot_product", rows[1])
                self.assertIn(f"random_update_{name}_dot_product", rows[1])
            self.assertGreater((output / "trajectory_comparison.png").stat().st_size, 1000)

    def test_single_step_switch_keeps_legacy_default_and_rejects_noop_paired_configuration(self):
        config = configuration()
        record = config.actor_rollout_ref.actor.gradient_record
        self.assertTrue(diagnostics.paired_recording_due(record, 1))
        record["single_step"] = {"enabled": False}
        self.assertFalse(diagnostics.paired_recording_due(record, 1))
        with self.assertRaisesRegex(ValueError, "trajectory.enabled=True or single_step.enabled=True"):
            diagnostics.validate_gradient_recording(config)
        record["trajectory"] = {"enabled": True}
        diagnostics.validate_gradient_recording(config)
        record["mode"] = "full"
        record["trajectory"]["enabled"] = False
        diagnostics.validate_gradient_recording(config)
        self.assertTrue(diagnostics.recording_due(record, 1))


if __name__ == "__main__":
    unittest.main()
