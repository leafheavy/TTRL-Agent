"""CPU regression checks for exact policy gradients and recorder transparency.

Service imports are omitted, but actor/GRPO function bodies are loaded directly
from production source and exercised with real PyTorch backward/AdamW.
"""

import ast
import hashlib
import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("gradient_diagnostics", ROOT / "verl/utils/gradient_diagnostics.py")
diagnostics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostics)


class Config(dict):
    __getattr__ = dict.__getitem__


class Tensors(dict):
    def __len__(self):
        return next(iter(self.values())).shape[0]

    def split(self, size):
        return [
            Tensors({key: value[start : start + size] for key, value in self.items()})
            for start in range(0, len(self), size)
        ]

    def to(self, device):
        return Tensors({key: value.to(device) for key, value in self.items()})


class Batch:
    def __init__(self, tensors, metadata=None):
        self.batch, self.non_tensor_batch = Tensors(tensors), {}
        self.meta_info = metadata or {}

    def select(self, batch_keys):
        return Batch({key: self.batch[key] for key in batch_keys}, self.meta_info)


def source_definitions(path, names, namespace, methods=None):
    nodes = []
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            node.decorator_list = []
            if methods is not None and isinstance(node, ast.ClassDef):
                node.body = [
                    method for method in node.body if isinstance(method, ast.FunctionDef) and method.name in methods
                ]
                for method in node.body:
                    method.decorator_list = []
            nodes.append(node)
    tree = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)


def actor_class():
    functional = {"torch": torch}
    source_definitions(ROOT / "verl/utils/torch_functional.py", {"masked_mean"}, functional)
    core = {"torch": torch, "verl_F": SimpleNamespace(masked_mean=functional["masked_mean"])}
    source_definitions(ROOT / "verl/trainer/ppo/core_algos.py", {"agg_loss", "compute_policy_loss", "kl_penalty"}, core)

    def append(metrics, values):
        for key, value in values.items():
            metrics.setdefault(key, []).append(value)

    namespace = {
        **core,
        "hashlib": hashlib,
        "DataProto": Batch,
        "BasePPOActor": object,
        "get_device_id": lambda: "cpu",
        "append_to_dict": append,
        "FSDP": FSDP,
        "FSDPModule": type("FSDP2Stub", (), {}),
    }
    source_definitions(
        ROOT / "verl/workers/actor/dp_actor.py",
        {"DataParallelPPOActor"},
        namespace,
        methods={"update_policy", "_backward_micro_batches", "_record_policy_gradient", "_optimizer_step"},
    )
    return namespace["DataParallelPPOActor"]


Actor = actor_class()


def sample_batch(step=1):
    inputs = torch.tensor([[1.0, 0.0, 2.0], [0.0, 2.0, 1.0], [2.0, 1.0, 0.0], [1.0, 2.0, 2.0]])
    return Batch(
        {
            "input_ids": inputs,
            "responses": torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0]]),
            "attention_mask": torch.ones(4, 3),
            "position_ids": torch.zeros(4, 3, dtype=torch.long),
            "response_mask": torch.tensor([[1.0, 0.0], [1.0, 1.0], [1.0, 0.0], [1.0, 1.0]]),
            "old_log_probs": torch.full((4, 2), -0.7),
            "advantages": torch.tensor([[1.0, 0.0], [-1.0, -1.0], [1.0, 0.0], [-1.0, -1.0]]),
            "ref_log_prob": torch.full((4, 2), -0.3),
        },
        {"temperature": 1.0, "gradient_record": {"step": step, "norm_adv_by_std": True, "rollout_n": 2}},
    )


def make_actor(enabled, directory, *, regularizers=True):
    torch.manual_seed(42)
    module = nn.Sequential(nn.Linear(3, 5), nn.Dropout(0.25), nn.Linear(5, 2))
    actor = Actor()
    actor.actor_module = module
    actor.actor_optimizer = torch.optim.AdamW(module.parameters(), lr=0.01)
    actor.device_name = "cpu"
    actor.ulysses_sequence_parallel_size = 1
    actor.gradient_record_config = {"enabled": enabled, "output_dir": str(directory), "interval": 1}
    actor.config = Config(
        ppo_mini_batch_size=4,
        ppo_micro_batch_size_per_gpu=2,
        ppo_epochs=1,
        use_dynamic_bsz=False,
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=3.0,
        policy_loss=Config(loss_mode="vanilla"),
        loss_agg_mode="token-mean",
        grad_clip=0.02,
        use_kl_loss=regularizers,
        kl_loss_type="mse",
        kl_loss_coef=0.4,
        entropy_coeff=0.3 if regularizers else 0.0,
    )

    def forward(micro_batch, temperature, calculate_entropy):
        log_probs = torch.log_softmax(module(micro_batch["input_ids"]) / temperature, dim=-1)
        entropy = -(log_probs.exp() * log_probs).sum(-1, keepdim=True).expand(-1, 2)
        return entropy if calculate_entropy else None, log_probs.gather(1, micro_batch["responses"])

    actor._forward_micro_batch = forward
    return actor


class TestPolicyGradientRecording(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.modules = patch.dict(sys.modules, {"verl.utils.gradient_diagnostics": diagnostics})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        for name, value in (("get_rank", 0), ("get_world_size", 1)):
            mocked = patch.object(torch.distributed, name, return_value=value)
            mocked.start()
            self.addCleanup(mocked.stop)

    def test_recording_keeps_adamw_update_rng_and_metrics_unchanged(self):
        ordinary = make_actor(False, self.root / "off")
        recorded = make_actor(True, self.root / "on")
        torch.manual_seed(101)
        ordinary_metrics = ordinary.update_policy(sample_batch())
        rng_after = torch.random.get_rng_state()
        torch.manual_seed(101)
        recorded_metrics = recorded.update_policy(sample_batch())
        for a, b in zip(ordinary.actor_module.parameters(), recorded.actor_module.parameters()):
            self.assertTrue(torch.equal(a, b))
        for a, b in zip(ordinary.actor_optimizer.state.values(), recorded.actor_optimizer.state.values()):
            for key in a:
                self.assertTrue(torch.equal(a[key], b[key]))
        self.assertTrue(torch.equal(rng_after, torch.random.get_rng_state()))
        self.assertEqual(ordinary_metrics, recorded_metrics)
        self.assertFalse((self.root / "off").exists())

    def test_saved_gradient_excludes_kl_entropy_and_gradient_clipping(self):
        reference = make_actor(False, self.root / "reference")
        reference.gradient_accumulation = 2
        reference.actor_module.train()
        torch.manual_seed(101)
        reference._backward_micro_batches(sample_batch().batch.split(2), 1.0, None, policy_only=True)
        expected = np.concatenate(
            [parameter.grad.flatten().numpy() for parameter in reference.actor_module.parameters()]
        )
        actor = make_actor(True, self.root / "recorded")
        torch.manual_seed(101)
        actor.update_policy(sample_batch())
        path = next((self.root / "recorded").glob("*/rank_00000.bin"))
        np.testing.assert_array_equal(np.fromfile(path, dtype="<f4"), expected)
        self.assertGreater(np.linalg.norm(expected), actor.config.grad_clip)

    def test_positive_q_weight_scales_policy_gradient_without_rotating_it(self):
        a = make_actor(True, self.root / "plain", regularizers=False)
        b = make_actor(True, self.root / "weighted", regularizers=False)
        weighted = sample_batch()
        weighted.batch["advantages"] *= 0.3
        torch.manual_seed(101)
        a.update_policy(sample_batch())
        torch.manual_seed(101)
        b.update_policy(weighted)
        result = diagnostics.compare_snapshot(
            next((self.root / "plain").iterdir()), next((self.root / "weighted").iterdir())
        )
        self.assertAlmostEqual(result["cosine"], 1, places=6)
        self.assertAlmostEqual(result["unsupervised_norm"] / result["supervised_norm"], 0.3, places=6)
        self.assertEqual(result["comparison"], "matched_conditions")

    def test_disabled_and_interval_skip_do_not_run_extra_backward(self):
        actor = make_actor(True, self.root / "skipped")
        actor.gradient_record_config["interval"] = 10
        actor.update_policy(sample_batch(step=2))
        self.assertFalse((self.root / "skipped").exists())


class TestGradientComparison(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def record(self, directory, values, *, rank=0, world_size=1, batch=None):
        module = nn.Linear(len(values), 1, bias=False)
        with torch.no_grad():
            module.weight.zero_()
        module.weight.grad = torch.tensor([values], dtype=torch.float32)
        path = diagnostics.save_policy_gradient(
            module,
            (batch or sample_batch()).batch,
            {"step": 1, "epoch": 0, "minibatch": 0, "objective": {}},
            self.root / directory,
            rank=rank,
            world_size=world_size,
        )
        return path.parent

    def test_same_orthogonal_opposite_and_zero(self):
        reference = self.record("reference", [1, 0])
        for name, values, cosine, angle in (
            ("same", [2, 0], 1, 0),
            ("orthogonal", [0, 1], 0, 90),
            ("opposite", [-1, 0], -1, 180),
        ):
            result = diagnostics.compare_snapshot(reference, self.record(name, values))
            self.assertAlmostEqual(result["cosine"], cosine)
            self.assertAlmostEqual(result["angle_deg"], angle)
        zero = self.record("zero", [0, 0])
        result = diagnostics.compare_snapshot(reference, zero)
        self.assertIsNone(result["cosine"])
        self.assertIsNone(result["angle_deg"])
        self.assertEqual(result["status"], "unsupervised_zero")
        self.assertEqual(diagnostics.compare_snapshot(zero, zero)["status"], "both_zero")

    def test_global_dot_product_is_not_average_rank_cosine(self):
        reference = self.record("reference", [10], rank=0, world_size=2)
        self.record("reference", [1], rank=1, world_size=2)
        target = self.record("target", [1], rank=0, world_size=2)
        self.record("target", [-1], rank=1, world_size=2)
        result = diagnostics.compare_snapshot(reference, target)
        self.assertAlmostEqual(result["cosine"], 9 / math.sqrt(202))
        self.assertAlmostEqual(result["supervised_norm"], math.sqrt(101))

    def test_missing_rank_and_incompatible_coordinates_fail(self):
        a = self.record("missing", [1], world_size=2)
        with self.assertRaisesRegex(ValueError, "rank"):
            diagnostics.compare_snapshot(a, a)
        b = self.record("one", [1])
        c = self.record("two", [1, 2])
        with self.assertRaisesRegex(ValueError, "layouts"):
            diagnostics.compare_snapshot(b, c)

    def test_different_rollouts_are_explicitly_marked(self):
        a = self.record("a", [1, 0])
        changed = sample_batch()
        changed.batch["input_ids"][0, 0] += 1
        b = self.record("b", [1, 0], batch=changed)
        self.assertEqual(diagnostics.compare_snapshot(a, b)["comparison"], "different_rollouts")

    def test_cli_outputs_csv_and_plot(self):
        self.record("a", [1, 0])
        self.record("b", [0.8, 0.6])
        output = self.root / "diagnosis"
        csv_path = diagnostics.compare_runs(self.root / "a", self.root / "b", output)
        self.assertIn("angle_deg", csv_path.read_text(encoding="utf-8"))
        self.assertGreater((output / "gradient_comparison.png").stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
