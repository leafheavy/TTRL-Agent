"""CPU checks for full-coordinate paired gradients without binary snapshots."""

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


if __name__ == "__main__":
    unittest.main()
