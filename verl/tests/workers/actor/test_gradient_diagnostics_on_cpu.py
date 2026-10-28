"""CPU regression checks for exact policy gradients and recorder transparency.

Service imports are omitted, but actor/GRPO function bodies are loaded directly
from production source and exercised with real PyTorch backward/AdamW.
"""

import ast
import copy
import hashlib
import importlib.util
import json
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


class TestActorOptimizerForeach(unittest.TestCase):
    @staticmethod
    def optimizer(module, config):
        """Execute the worker's actual validation and AdamW construction."""
        path = ROOT / "verl/workers/fsdp_workers.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        worker = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ActorRolloutRefWorker")
        method = next(node for node in worker.body if getattr(node, "name", None) == "_build_model_optimizer")
        assignment = next(
            node for node in method.body
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "optimizer_foreach" for target in node.targets
            )
        )
        validation = next(
            node for node in method.body
            if isinstance(node, ast.If) and any(
                isinstance(item, ast.Name) and item.id == "optimizer_foreach" for item in ast.walk(node.test)
            )
        )
        construction = next(
            node for node in ast.walk(method)
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "actor_optimizer" for target in node.targets
            ) and isinstance(node.value, ast.Call)
        )
        namespace = {
            "role": "actor", "optim_config": Config(lr=0.01, **config),
            "actor_module_fsdp": module, "optim": torch.optim,
        }
        exec(compile(ast.Module(body=[assignment, validation, construction], type_ignores=[]), str(path), "exec"), namespace)
        return namespace["actor_optimizer"]

    def test_worker_forwards_auto_true_and_false_to_adamw(self):
        for config, expected in (({}, None), ({"foreach": None}, None), ({"foreach": True}, True), ({"foreach": False}, False)):
            with self.subTest(config=config):
                optimizer = self.optimizer(nn.Linear(2, 1), config)
                self.assertIs(optimizer.defaults["foreach"], expected)

    def test_invalid_foreach_fails_before_optimizer_construction(self):
        for value in ("False", "auto", 0, 1, [], {}):
            with self.subTest(value=value), patch.object(torch.optim, "AdamW") as construction:
                with self.assertRaisesRegex(ValueError, "actor.optim.foreach must be a boolean or null"):
                    self.optimizer(nn.Linear(2, 1), {"foreach": value})
                construction.assert_not_called()

    def test_false_avoids_multi_tensor_path_and_preserves_adamw_updates(self):
        torch.manual_seed(197)
        low_memory = nn.Sequential(nn.Linear(3, 5), nn.Linear(5, 2)).double()
        reference = copy.deepcopy(low_memory)
        actual_optimizer = self.optimizer(low_memory, {"foreach": False, "weight_decay": 0.13})
        reference_optimizer = self.optimizer(reference, {"foreach": True, "weight_decay": 0.13})
        adamw = importlib.import_module("torch.optim.adamw")
        for step in range(3):
            for parameter, expected_parameter in zip(low_memory.parameters(), reference.parameters()):
                gradient = torch.linspace(-0.7, 0.9, parameter.numel(), dtype=parameter.dtype).reshape_as(parameter)
                parameter.grad = gradient + step * 0.03
                expected_parameter.grad = parameter.grad.clone()
            reference_optimizer.step()
            with patch.object(adamw, "_multi_tensor_adamw", side_effect=AssertionError("Unexpected foreach allocation")):
                actual_optimizer.step()
            for parameter, expected_parameter in zip(low_memory.parameters(), reference.parameters()):
                torch.testing.assert_close(parameter, expected_parameter, rtol=1e-12, atol=1e-12)
                for key in ("step", "exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(
                        actual_optimizer.state[parameter][key], reference_optimizer.state[expected_parameter][key],
                        rtol=1e-12, atol=1e-12,
                    )


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
        methods={
            "update_policy",
            "_backward_micro_batches",
            "_record_policy_gradient",
            "_record_paired_policy_gradient",
            "_advance_supervised_trajectory",
            "_finish_optimization_trajectory",
            "_optimizer_step",
        },
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


def make_actor(enabled, directory, *, regularizers=True, mode=None):
    torch.manual_seed(42)
    module = nn.Sequential(nn.Linear(3, 5), nn.Dropout(0.25), nn.Linear(5, 2))
    actor = Actor()
    actor.actor_module = module
    actor.actor_optimizer = torch.optim.AdamW(module.parameters(), lr=0.01)
    actor.device_name = "cpu"
    actor.ulysses_sequence_parallel_size = 1
    actor.gradient_record_config = {"enabled": enabled, "output_dir": str(directory), "interval": 1}
    if mode is not None:
        actor.gradient_record_config["mode"] = mode
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


def paired_batch(step=1):
    batch = sample_batch(step)
    batch.batch["diagnostic_supervised_advantages"] = torch.tensor(
        [[1.0, 0.0], [-1.0, -1.0], [-1.0, 0.0], [1.0, 1.0]]
    )
    return batch


def manual_policy_gradient(batch, *, supervised=False):
    actor = make_actor(False, "unused")
    actor.actor_module.train()
    actor.gradient_accumulation = 2
    if supervised:
        batch = Batch({**batch.batch, "advantages": batch.batch["diagnostic_supervised_advantages"]})
    torch.manual_seed(101)
    actor._backward_micro_batches(batch.batch.split(2), 1.0, None, policy_only=True)
    return torch.cat([parameter.grad.flatten().double() for parameter in actor.actor_module.parameters()])


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

    def test_paired_statistics_match_manual_policy_gradients_with_identical_dropout(self):
        batch = paired_batch()
        supervised = manual_policy_gradient(batch, supervised=True)
        unsupervised = manual_policy_gradient(batch)
        expected_dot = torch.dot(supervised, unsupervised).item()
        expected_supervised_sq = torch.dot(supervised, supervised).item()
        expected_unsupervised_sq = torch.dot(unsupervised, unsupervised).item()
        actor = make_actor(True, self.root / "paired", mode="paired")
        torch.manual_seed(101)
        actor.update_policy(batch)
        files = list((self.root / "paired").glob("*.json"))
        self.assertEqual(len(files), 1)
        result = json.loads(files[0].read_text(encoding="utf-8"))
        self.assertEqual(result["mode"], "paired")
        self.assertEqual(result["accumulation_dtype"], "float64")
        self.assertEqual(result["comparison"], "matched_conditions")
        self.assertEqual(result["numel"], supervised.numel())
        self.assertAlmostEqual(result["dot_product"], expected_dot, places=14)
        self.assertAlmostEqual(result["supervised_squared_norm"], expected_supervised_sq, places=14)
        self.assertAlmostEqual(result["unsupervised_squared_norm"], expected_unsupervised_sq, places=14)
        self.assertAlmostEqual(
            result["cosine"], expected_dot / math.sqrt(expected_supervised_sq * expected_unsupervised_sq)
        )
        self.assertGreater(result["unsupervised_norm"], actor.config.grad_clip)
        self.assertFalse(list((self.root / "paired").rglob("*.bin")))

    def test_paired_recording_preserves_training_update_optimizer_rng_and_metrics(self):
        ordinary = make_actor(False, self.root / "off")
        recorded = make_actor(True, self.root / "paired", mode="paired")
        batch = paired_batch()
        original_advantages = batch.batch["advantages"].clone()
        torch.manual_seed(101)
        ordinary_metrics = ordinary.update_policy(sample_batch())
        rng_after = torch.random.get_rng_state()
        torch.manual_seed(101)
        recorded_metrics = recorded.update_policy(batch)
        for ordinary_parameter, recorded_parameter in zip(
            ordinary.actor_module.parameters(), recorded.actor_module.parameters()
        ):
            self.assertTrue(torch.equal(ordinary_parameter, recorded_parameter))
        for ordinary_state, recorded_state in zip(
            ordinary.actor_optimizer.state.values(), recorded.actor_optimizer.state.values()
        ):
            self.assertEqual(ordinary_state.keys(), recorded_state.keys())
            for key in ordinary_state:
                self.assertTrue(torch.equal(ordinary_state[key], recorded_state[key]))
        self.assertTrue(torch.equal(rng_after, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(original_advantages, batch.batch["advantages"]))
        self.assertEqual(ordinary_metrics, recorded_metrics)

    def test_paired_disabled_or_interval_skip_needs_no_supervised_advantages(self):
        for enabled, step in ((False, 1), (True, 2)):
            with self.subTest(enabled=enabled, step=step):
                actor = make_actor(enabled, self.root / f"skip_{enabled}", mode="paired")
                actor.gradient_record_config["interval"] = 10
                with patch.object(actor, "_backward_micro_batches", wraps=actor._backward_micro_batches) as backward:
                    actor.update_policy(sample_batch(step))
                self.assertEqual(backward.call_count, 1)
                self.assertFalse(Path(actor.gradient_record_config["output_dir"]).exists())

    def test_paired_positive_q_weight_changes_norm_not_direction(self):
        batch = paired_batch()
        batch.batch["diagnostic_supervised_advantages"] = batch.batch["advantages"].clone()
        batch.batch["advantages"] *= 0.3
        actor = make_actor(True, self.root / "paired_weighted", mode="paired", regularizers=False)
        torch.manual_seed(101)
        actor.update_policy(batch)
        result = json.loads(next((self.root / "paired_weighted").glob("*.json")).read_text(encoding="utf-8"))
        self.assertAlmostEqual(result["cosine"], 1, places=6)
        self.assertAlmostEqual(result["unsupervised_norm"] / result["supervised_norm"], 0.3, places=6)

    def test_selected_paired_step_requires_supervised_advantages_before_backward(self):
        actor = make_actor(True, self.root / "missing", mode="paired")
        with patch.object(actor, "_backward_micro_batches", wraps=actor._backward_micro_batches) as backward:
            with self.assertRaisesRegex(ValueError, "supervised advantages"):
                actor.update_policy(sample_batch())
        backward.assert_not_called()
        self.assertFalse((self.root / "missing").exists())

    def test_trajectory_preserves_three_training_updates_moments_rng_and_metrics(self):
        ordinary = make_actor(False, self.root / "ordinary")
        recorded = make_actor(False, self.root / "trajectory", mode="paired")
        for actor in (ordinary, recorded):
            torch.manual_seed(100)
            actor.update_policy(sample_batch())
        recorded.gradient_record_config.update(
            enabled=True, interval=3, trajectory={"enabled": True}
        )
        verifier = TestOptimizationTrajectory()
        for step, learning_rate in enumerate((0.01, 0.003, 0.008), start=1):
            ordinary.actor_optimizer.param_groups[0]["lr"] = learning_rate
            recorded.actor_optimizer.param_groups[0]["lr"] = learning_rate
            torch.manual_seed(200 + step)
            expected_metrics = ordinary.update_policy(sample_batch(step))
            expected_rng = torch.random.get_rng_state()
            torch.manual_seed(200 + step)
            actual_metrics = recorded.update_policy(paired_batch(step))
            verifier.assert_state_equal(recorded.actor_module.state_dict(), ordinary.actor_module.state_dict())
            verifier.assert_state_equal(recorded.actor_optimizer.state_dict(), ordinary.actor_optimizer.state_dict())
            self.assertTrue(torch.equal(torch.random.get_rng_state(), expected_rng))
            self.assertEqual(actual_metrics, expected_metrics)
        records = sorted((self.root / "trajectory").glob("trajectory_step_*.json"))
        self.assertEqual(len(records), 2)
        final_record = json.loads(records[-1].read_text(encoding="utf-8"))
        self.assertEqual(final_record["updates"], 3)
        self.assertEqual(final_record["start_update"], "step_000001_epoch_000_minibatch_0000")
        self.assertEqual(final_record["end_update"], "step_000003_epoch_000_minibatch_0000")

    def test_trajectory_failure_restores_real_training_state_and_rng(self):
        actor = make_actor(False, self.root / "trajectory_failure", mode="paired")
        torch.manual_seed(100)
        actor.update_policy(sample_batch())
        actor.gradient_record_config.update(enabled=True, interval=10, trajectory={"enabled": True})
        original_parameters = copy.deepcopy(actor.actor_module.state_dict())
        original_optimizer = copy.deepcopy(actor.actor_optimizer.state_dict())
        original_step = actor._optimizer_step

        def failing_optimizer_step():
            original_step()
            raise RuntimeError("shadow optimizer failed after updating")

        torch.manual_seed(205)
        original_rng = torch.random.get_rng_state()
        with patch.object(actor, "_optimizer_step", side_effect=failing_optimizer_step):
            with self.assertRaisesRegex(RuntimeError, "shadow optimizer failed"):
                actor.update_policy(paired_batch(step=2))
        verifier = TestOptimizationTrajectory()
        verifier.assert_state_equal(actor.actor_module.state_dict(), original_parameters)
        verifier.assert_state_equal(actor.actor_optimizer.state_dict(), original_optimizer)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), original_rng))
        self.assertTrue(all(parameter.grad is None for parameter in actor.actor_module.parameters()))
        self.assertFalse((self.root / "trajectory_failure").exists())

    def test_final_step_writes_trajectory_outside_gradient_recording_interval(self):
        actor = make_actor(True, self.root / "trajectory_final", mode="paired")
        actor.gradient_record_config.update(interval=10, trajectory={"enabled": True})
        batch = paired_batch(step=2)
        batch.meta_info["gradient_record"]["final_step"] = True
        actor.update_policy(batch)
        files = list((self.root / "trajectory_final").glob("*.json"))
        self.assertEqual(len(files), 1)
        self.assertTrue(files[0].name.startswith("trajectory_step_000002"))
        record = json.loads(files[0].read_text(encoding="utf-8"))
        self.assertEqual(record["updates"], 1)
        self.assertEqual(record["start_update"], "step_000002_epoch_000_minibatch_0000")
        self.assertEqual(record["end_update"], "step_000002_epoch_000_minibatch_0000")

    def test_paired_failure_restores_rng_and_clears_disposable_gradients(self):
        actor = make_actor(True, self.root / "failed", mode="paired")
        parameters = [parameter.detach().clone() for parameter in actor.actor_module.parameters()]
        torch.manual_seed(101)
        rng_before = torch.random.get_rng_state()
        with patch.object(diagnostics, "paired_gradient_statistics", side_effect=ValueError("diagnostic failed")):
            with self.assertRaisesRegex(ValueError, "diagnostic failed"):
                actor.update_policy(paired_batch())
        for parameter, original in zip(actor.actor_module.parameters(), parameters):
            self.assertTrue(torch.equal(parameter, original))
            self.assertIsNone(parameter.grad)
        self.assertFalse(actor.actor_optimizer.state)
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
        self.assertFalse((self.root / "failed").exists())


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


class TestOptimizationTrajectory(unittest.TestCase):
    """Check genuine independent optimization against two ordinary AdamW models."""

    @staticmethod
    def parameters(module):
        return torch.cat([parameter.detach().reshape(-1).double() for parameter in module.parameters()])

    def assert_state_equal(self, actual, expected):
        if isinstance(actual, torch.Tensor):
            self.assertTrue(torch.equal(actual, expected))
        elif isinstance(actual, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in actual:
                self.assert_state_equal(actual[key], expected[key])
        elif isinstance(actual, (tuple, list)):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.assert_state_equal(left, right)
        else:
            self.assertEqual(actual, expected)

    @staticmethod
    def model():
        module = nn.Linear(2, 2, bias=True, dtype=torch.float64)
        with torch.no_grad():
            module.weight.copy_(torch.tensor([[0.6, -0.4], [0.2, 0.3]], dtype=torch.float64))
            module.bias.copy_(torch.tensor([0.1, -0.2], dtype=torch.float64))
        module.register_buffer("forward_count", torch.zeros((), dtype=torch.int64))
        return module

    @staticmethod
    def optimizer(module):
        return torch.optim.AdamW(module.parameters(), lr=0.02, betas=(0.7, 0.85), weight_decay=0.13, eps=1e-7)

    @staticmethod
    def optimization_step(module, optimizer, target):
        inputs = torch.tensor([[1.0, -0.5], [0.2, 0.8], [-0.7, 1.1]], dtype=torch.float64)
        optimizer.zero_grad()
        module.forward_count.add_(1)
        prediction = module(inputs)
        regularizer = 0.07 * sum(parameter.square().sum() for parameter in module.parameters())
        loss = (prediction - target).square().mean() + regularizer
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), 0.15)
        optimizer.step()

    def test_three_adamw_steps_match_independent_models_and_endpoint_path_formulas(self):
        module = self.model()
        optimizer = self.optimizer(module)
        warmup_target = torch.tensor([[0.4, 0.3], [0.1, -0.2], [-0.5, 0.2]], dtype=torch.float64)
        self.optimization_step(module, optimizer, warmup_target)
        self.assertTrue(optimizer.state)
        initial = self.parameters(module)

        supervised = copy.deepcopy(module)
        supervised_optimizer = self.optimizer(supervised)
        supervised_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        unsupervised = copy.deepcopy(module)
        unsupervised_optimizer = self.optimizer(unsupervised)
        unsupervised_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        trajectory = diagnostics.OptimizationTrajectory(module, optimizer)
        supervised_increments, unsupervised_increments = [], []

        for index, learning_rate in enumerate((0.02, 0.007, 0.011), start=1):
            for current_optimizer in (optimizer, supervised_optimizer, unsupervised_optimizer):
                current_optimizer.param_groups[0]["lr"] = learning_rate
            supervised_target = torch.tensor([[0.9, -0.1], [-0.4, 0.7], [0.3, -0.8]], dtype=torch.float64) * index
            unsupervised_target = torch.tensor([[-0.7, 0.8], [0.5, -0.2], [-0.1, 0.6]], dtype=torch.float64) / index
            supervised_before = self.parameters(supervised)
            unsupervised_before = self.parameters(unsupervised)
            actual_before = copy.deepcopy(module.state_dict())
            optimizer_before = copy.deepcopy(optimizer.state_dict())
            self.optimization_step(supervised, supervised_optimizer, supervised_target)

            def supervised_update(supervised_target=supervised_target):
                self.optimization_step(module, optimizer, supervised_target)
                torch.testing.assert_close(self.parameters(module), self.parameters(supervised), rtol=0, atol=0)
                self.assert_state_equal(optimizer.state_dict(), supervised_optimizer.state_dict())
                self.assertEqual(module.forward_count.item(), supervised.forward_count.item())

            trajectory.supervised_step(module, optimizer, supervised_update)
            self.assert_state_equal(module.state_dict(), actual_before)
            self.assert_state_equal(optimizer.state_dict(), optimizer_before)
            self.assertTrue(all(parameter.grad is None for parameter in module.parameters()))

            self.optimization_step(module, optimizer, unsupervised_target)
            self.optimization_step(unsupervised, unsupervised_optimizer, unsupervised_target)
            self.assert_state_equal(module.state_dict(), unsupervised.state_dict())
            self.assert_state_equal(optimizer.state_dict(), unsupervised_optimizer.state_dict())
            supervised_increments.append(self.parameters(supervised) - supervised_before)
            unsupervised_increments.append(self.parameters(unsupervised) - unsupervised_before)
            record = trajectory.finish_step(module, {"step": index, "epoch": 0, "minibatch": 0})

            endpoint_left = self.parameters(supervised) - initial
            endpoint_right = self.parameters(unsupervised) - initial
            path_left, path_right = torch.cat(supervised_increments), torch.cat(unsupervised_increments)
            for field, left, right in (("endpoint", endpoint_left, endpoint_right), ("path", path_left, path_right)):
                expected = diagnostics._comparison_statistics(
                    torch.dot(left, right).item(), torch.dot(left, left).item(), torch.dot(right, right).item()
                )
                self.assertEqual(record[field]["status"], expected["status"])
                for statistic in (
                    "dot_product", "supervised_squared_norm", "unsupervised_squared_norm", "cosine", "angle_deg"
                ):
                    self.assertAlmostEqual(record[field][statistic], expected[statistic], places=13)
            self.assertEqual(record["updates"], index)

        self.assertFalse(torch.equal(self.parameters(supervised), self.parameters(unsupervised)))

    def test_supervised_failure_restores_parameters_buffers_and_adamw_state(self):
        module = self.model()
        optimizer = self.optimizer(module)
        target = torch.zeros(3, 2, dtype=torch.float64)
        self.optimization_step(module, optimizer, target)
        trajectory = diagnostics.OptimizationTrajectory(module, optimizer)
        original_parameters = copy.deepcopy(module.state_dict())
        original_optimizer = copy.deepcopy(optimizer.state_dict())

        def failing_update():
            self.optimization_step(module, optimizer, target + 3)
            raise RuntimeError("supervised branch failed")

        with self.assertRaisesRegex(RuntimeError, "supervised branch failed"):
            trajectory.supervised_step(module, optimizer, failing_update)
        self.assert_state_equal(module.state_dict(), original_parameters)
        self.assert_state_equal(optimizer.state_dict(), original_optimizer)
        self.assertTrue(all(parameter.grad is None for parameter in module.parameters()))

    def test_global_endpoint_and_path_statistics_sum_coordinate_shards(self):
        module = nn.Linear(2, 1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            module.weight.zero_()
        optimizer = torch.optim.AdamW(module.parameters())
        trajectory = diagnostics.OptimizationTrajectory(module, optimizer)
        remote_statistics = (
            torch.tensor([0, 2, 2, 0, 2, 2], dtype=torch.float64),
            torch.tensor([3, 5, 2, 0, 3, 6], dtype=torch.float64),
        )
        for step, (supervised_delta, unsupervised_delta) in enumerate(
            (([1, 0], [0, 2]), ([0, 3], [4, 0])), start=1
        ):
            def shadow_update(supervised_delta=supervised_delta):
                with torch.no_grad():
                    module.weight.add_(torch.tensor([supervised_delta], dtype=torch.float64))

            trajectory.supervised_step(module, optimizer, shadow_update)
            with torch.no_grad():
                module.weight.add_(torch.tensor([unsupervised_delta], dtype=torch.float64))

            def reduction(statistics, op, step=step):
                statistics.add_(remote_statistics[step - 1])

            with patch.object(torch.distributed, "all_reduce", side_effect=reduction) as reduce:
                record = trajectory.finish_step(
                    module, {"step": step, "epoch": 0, "minibatch": 0}, world_size=2
                )
            self.assertEqual(reduce.call_args.args[0].numel(), 6)
        expected_endpoint = diagnostics._comparison_statistics(13, 15, 22)
        expected_path = diagnostics._comparison_statistics(0, 13, 26)
        self.assertEqual(record["endpoint"], expected_endpoint)
        self.assertEqual(record["path"], expected_path)

    def test_zero_learning_rate_keeps_both_trajectory_directions_undefined(self):
        module = self.model()
        optimizer = self.optimizer(module)
        optimizer.param_groups[0]["lr"] = 0
        trajectory = diagnostics.OptimizationTrajectory(module, optimizer)
        trajectory.supervised_step(
            module, optimizer,
            lambda: self.optimization_step(module, optimizer, torch.ones(3, 2, dtype=torch.float64)),
        )
        self.optimization_step(module, optimizer, torch.zeros(3, 2, dtype=torch.float64))
        record = trajectory.finish_step(module, {"step": 1, "epoch": 0, "minibatch": 0})
        for field in ("endpoint", "path"):
            self.assertEqual(record[field]["status"], "both_zero")
            self.assertIsNone(record[field]["cosine"])
            self.assertIsNone(record[field]["angle_deg"])


class TestPairedRandomBaselines(unittest.TestCase):
    def setUp(self):
        TestPolicyGradientRecording.setUp(self)

    @staticmethod
    def baseline_batch(step=1):
        batch = paired_batch(step)
        batch.batch["diagnostic_shuffled_advantages"] = torch.tensor(
            [[-1.0, 0.0], [1.0, 1.0], [1.0, 0.0], [-1.0, -1.0]]
        )
        batch.meta_info["gradient_baselines"] = {
            "seed": 29, "shuffle_scope": "within_prompt_uid", "changed_label_count": 4,
        }
        return batch

    def assert_training_state_equal(self, actual, expected):
        def equal(left, right):
            if isinstance(left, torch.Tensor):
                self.assertTrue(torch.equal(left, right))
            elif isinstance(left, dict):
                self.assertEqual(left.keys(), right.keys())
                for key in left:
                    equal(left[key], right[key])
            elif isinstance(left, (list, tuple)):
                self.assertEqual(len(left), len(right))
                for one, two in zip(left, right):
                    equal(one, two)
            else:
                self.assertEqual(left, right)

        equal(actual.actor_module.state_dict(), expected.actor_module.state_dict())
        equal(actual.actor_optimizer.state_dict(), expected.actor_optimizer.state_dict())

    def test_two_baselines_match_manual_gradients_and_record_auditable_seeds(self):
        batch = self.baseline_batch()
        supervised = manual_policy_gradient(batch, supervised=True)
        shuffled_batch = Batch({**batch.batch, "advantages": batch.batch["diagnostic_shuffled_advantages"]})
        shuffled = manual_policy_gradient(shuffled_batch)
        expected_shuffled = diagnostics._comparison_statistics(
            torch.dot(supervised, shuffled).item(),
            torch.dot(supervised, supervised).item(),
            torch.dot(shuffled, shuffled).item(),
        )
        reference = make_actor(False, "unused")
        reference.actor_module.train()
        reference.gradient_accumulation = 2
        torch.manual_seed(101)
        reference._backward_micro_batches(
            batch.batch.split(2), 1.0, None, policy_only=True, advantages_key="diagnostic_supervised_advantages"
        )
        seed = diagnostics.diagnostic_seed(23, 1, 0, 0)
        random_values = diagnostics.random_gradient_statistics(
            diagnostics.capture_policy_gradient(reference.actor_module), seed=seed
        )
        expected_random = diagnostics._comparison_statistics(
            random_values[0].item() / math.sqrt(random_values[2].item()), random_values[1].item(), 1
        )
        actor = make_actor(True, self.root / "baselines", mode="paired")
        actor.gradient_record_config.update(
            baselines={"enabled": True, "seed": 23}, single_step={"enabled": True}
        )
        torch.manual_seed(101)
        with patch.object(actor, "_backward_micro_batches", wraps=actor._backward_micro_batches) as backward:
            actor.update_policy(batch)
        self.assertEqual(backward.call_count, 4)
        paths = list((self.root / "baselines").glob("*.json"))
        self.assertEqual(len(paths), 1)
        result = json.loads(paths[0].read_text(encoding="utf-8"))
        self.assertEqual(set(result["baselines"]), {"random_direction", "shuffled_labels"})
        for name, expected in (("random_direction", expected_random), ("shuffled_labels", expected_shuffled)):
            actual = result["baselines"][name]
            self.assertEqual(actual["status"], expected["status"])
            for field in ("dot_product", "supervised_squared_norm", "unsupervised_squared_norm", "cosine"):
                self.assertAlmostEqual(actual[field], expected[field], places=14)
        self.assertEqual(result["baseline_metadata"]["seed"], seed)
        self.assertEqual(result["baseline_metadata"]["draws_per_update"], 1)
        self.assertEqual(result["baseline_metadata"]["shuffled_labels"], batch.meta_info["gradient_baselines"])
        self.assertFalse(list((self.root / "baselines").rglob("*.bin")))

    def test_baselines_preserve_adamw_moments_parameters_dropout_rng_and_metrics(self):
        ordinary = make_actor(False, self.root / "off")
        recorded = make_actor(False, self.root / "on", mode="paired")
        for actor in (ordinary, recorded):
            torch.manual_seed(73)
            actor.update_policy(sample_batch())
        recorded.gradient_record_config.update(
            enabled=True, baselines={"enabled": True, "seed": 23}, trajectory={"enabled": True}
        )
        for step in (1, 2):
            batch = self.baseline_batch(step)
            originals = {key: value.clone() for key, value in batch.batch.items()}
            torch.manual_seed(100 + step)
            expected_metrics = ordinary.update_policy(sample_batch(step))
            expected_rng = torch.random.get_rng_state().clone()
            torch.manual_seed(100 + step)
            actual_metrics = recorded.update_policy(batch)
            self.assert_training_state_equal(recorded, ordinary)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), expected_rng))
            self.assertEqual(actual_metrics, expected_metrics)
            for key, value in originals.items():
                self.assertTrue(torch.equal(batch.batch[key], value), key)
            self.assertTrue(all(parameter.grad is None for parameter in recorded.actor_module.parameters()))
        self.assertEqual(recorded.optimization_trajectory.updates, 2)
        self.assertEqual(len(list((self.root / "on").glob("trajectory_step_*.json"))), 2)
        self.assertFalse((self.root / "off").exists())

    def test_zero_supervised_gradient_has_undefined_baseline_directions(self):
        actor = make_actor(True, self.root / "zero", mode="paired")
        actor.gradient_record_config["baselines"] = {"enabled": True, "seed": 23}
        batch = self.baseline_batch()
        batch.batch["diagnostic_supervised_advantages"].zero_()
        batch.batch["diagnostic_shuffled_advantages"].zero_()
        torch.manual_seed(101)
        actor.update_policy(batch)
        result = json.loads(next((self.root / "zero").glob("*.json")).read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "supervised_zero")
        self.assertEqual(result["baselines"]["random_direction"]["status"], "supervised_zero")
        self.assertEqual(result["baselines"]["shuffled_labels"]["status"], "both_zero")
        for values in result["baselines"].values():
            self.assertIsNone(values["cosine"])
            self.assertIsNone(values["angle_deg"])

    def test_enabled_baselines_require_shuffled_advantages_before_any_backward(self):
        actor = make_actor(True, self.root / "missing_shuffle", mode="paired")
        actor.gradient_record_config["baselines"] = {"enabled": True, "seed": 23}
        with patch.object(actor, "_backward_micro_batches", wraps=actor._backward_micro_batches) as backward:
            with self.assertRaisesRegex(ValueError, "shuffled advantages"):
                actor.update_policy(paired_batch())
        backward.assert_not_called()
        self.assertFalse((self.root / "missing_shuffle").exists())

    def test_single_baseline_selection_preserves_training_and_skips_disabled_computation(self):
        for shuffled_enabled, random_enabled in ((True, False), (False, True)):
            with self.subTest(shuffled=shuffled_enabled, random=random_enabled):
                directory = self.root / f"single_{shuffled_enabled}_{random_enabled}"
                ordinary = make_actor(False, self.root / "ordinary")
                recorded = make_actor(True, directory, mode="paired")
                recorded.gradient_record_config.update(
                    single_step={"enabled": True},
                    baselines={
                        "enabled": True, "seed": 23,
                        "shuffled_labels": {"enabled": shuffled_enabled},
                        "random_direction": {"enabled": random_enabled},
                    },
                )
                batch = self.baseline_batch() if shuffled_enabled else paired_batch()
                torch.manual_seed(101)
                expected_metrics = ordinary.update_policy(sample_batch())
                expected_rng = torch.random.get_rng_state().clone()
                torch.manual_seed(101)
                with patch.object(
                    diagnostics, "random_gradient_statistics", wraps=diagnostics.random_gradient_statistics
                ) as random:
                    with patch.object(
                        recorded, "_backward_micro_batches", wraps=recorded._backward_micro_batches
                    ) as backward:
                        actual_metrics = recorded.update_policy(batch)
                self.assertEqual(random.call_count, int(random_enabled))
                self.assertEqual(backward.call_count, 3 + int(shuffled_enabled))
                self.assertEqual(actual_metrics, expected_metrics)
                self.assert_training_state_equal(recorded, ordinary)
                self.assertTrue(torch.equal(torch.random.get_rng_state(), expected_rng))
                record = json.loads(next(directory.glob("*.json")).read_text(encoding="utf-8"))
                expected = {"shuffled_labels"} if shuffled_enabled else {"random_direction"}
                self.assertEqual(set(record["baselines"]), expected)

    def test_disabled_baselines_keep_legacy_paired_records_and_need_no_shuffle_tensor(self):
        actor = make_actor(True, self.root / "legacy", mode="paired")
        actor.gradient_record_config["baselines"] = {"enabled": False, "seed": 23}
        torch.manual_seed(101)
        actor.update_policy(paired_batch())
        result = json.loads(next((self.root / "legacy").glob("*.json")).read_text(encoding="utf-8"))
        self.assertNotIn("baselines", result)
        self.assertNotIn("baseline_metadata", result)


class TestTrajectoryBaselineWiring(unittest.TestCase):
    def setUp(self):
        TestPolicyGradientRecording.setUp(self)

    def test_single_step_switch_and_legacy_default_do_not_change_full_recording(self):
        for mode, single_step, enabled, expected_backwards in (
            ("paired", None, True, 3),
            ("paired", True, True, 3),
            ("paired", False, True, 1),
            ("paired", True, False, 1),
            ("full", False, True, 2),
        ):
            with self.subTest(mode=mode, single_step=single_step, enabled=enabled):
                directory = self.root / f"{mode}_{single_step}_{enabled}"
                actor = make_actor(enabled, directory, mode=mode)
                if single_step is not None:
                    actor.gradient_record_config["single_step"] = {"enabled": single_step}
                batch = paired_batch() if mode == "paired" and expected_backwards == 3 else sample_batch()
                torch.manual_seed(101)
                with patch.object(actor, "_backward_micro_batches", wraps=actor._backward_micro_batches) as backward:
                    actor.update_policy(batch)
                self.assertEqual(backward.call_count, expected_backwards)
                self.assertEqual(directory.exists(), expected_backwards > 1)

    def test_shuffled_trajectory_uses_its_own_full_adamw_path_at_every_minibatch(self):
        recorded = make_actor(False, self.root / "trajectory", mode="paired")
        ordinary = make_actor(False, self.root / "ordinary")
        supervised = make_actor(False, self.root / "supervised")
        shuffled = make_actor(False, self.root / "shuffled")
        actors = (recorded, ordinary, supervised, shuffled)
        for actor in actors:
            torch.manual_seed(73)
            actor.update_policy(sample_batch())
            actor.config.ppo_mini_batch_size = 2
            actor.config.ppo_micro_batch_size_per_gpu = 1
            actor.config.ppo_epochs = 2
        recorded.gradient_record_config.update(
            enabled=True, interval=10, single_step={"enabled": False},
            trajectory={"enabled": True}, baselines={
                "enabled": True, "seed": 23,
                "shuffled_labels": {"enabled": True}, "random_direction": {"enabled": False},
            },
        )
        for step, learning_rate in ((2, 0.003), (3, 0.008)):
            batch = TestPairedRandomBaselines.baseline_batch(step)
            batch.meta_info["gradient_record"]["final_step"] = step == 3
            originals = {key: value.clone() for key, value in batch.batch.items()}
            for actor in actors:
                actor.actor_optimizer.param_groups[0]["lr"] = learning_rate
            for reference, advantages_key in (
                (supervised, "diagnostic_supervised_advantages"),
                (shuffled, "diagnostic_shuffled_advantages"),
            ):
                reference_batch = Batch({**batch.batch, "advantages": batch.batch[advantages_key]})
                reference_batch.meta_info["temperature"] = 1.0
                torch.manual_seed(200 + step)
                reference.update_policy(reference_batch)
            torch.manual_seed(200 + step)
            expected_metrics = ordinary.update_policy(sample_batch(step))
            expected_rng = torch.random.get_rng_state().clone()
            torch.manual_seed(200 + step)
            with patch.object(recorded, "_record_policy_gradient", side_effect=AssertionError("legacy snapshot ran")), \
                    patch.object(torch, "save", side_effect=AssertionError("checkpoint or batch archive written")), \
                    patch.object(diagnostics.OptimizationTrajectory, "_random_chunks",
                                 side_effect=AssertionError("disabled random direction ran")):
                with patch.object(
                    recorded, "_backward_micro_batches", wraps=recorded._backward_micro_batches
                ) as backward:
                    actual_metrics = recorded.update_policy(batch)
            self.assertEqual(backward.call_count, 12)
            self.assertTrue(all(not call.kwargs.get("policy_only", False) for call in backward.call_args_list))
            TestPairedRandomBaselines.assert_training_state_equal(self, recorded, ordinary)
            self.assertEqual(actual_metrics, expected_metrics)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), expected_rng))
            for key, value in originals.items():
                self.assertTrue(torch.equal(batch.batch[key], value), key)
            trajectory = recorded.optimization_trajectory
            self.assertFalse(hasattr(trajectory, "random_displacement"))
            self.assertEqual(trajectory.updates, (step - 1) * 4)
            for parameters, optimizer_state, reference in (
                (trajectory.supervised_parameters, trajectory.supervised_optimizer_state, supervised),
                (trajectory.shuffled_parameters, trajectory.shuffled_optimizer_state, shuffled),
            ):
                for value, expected in zip(parameters, reference.actor_module.parameters()):
                    torch.testing.assert_close(value, expected.detach(), rtol=0, atol=0)
                restored = diagnostics._restore_state(optimizer_state)
                self.assertEqual(len(restored), len(reference.actor_optimizer.state))
                for value, expected in zip(restored.values(), reference.actor_optimizer.state.values()):
                    self.assertEqual(value.keys(), expected.keys())
                    for key in value:
                        torch.testing.assert_close(value[key], expected[key], rtol=0, atol=0)
            if step == 2:
                self.assertFalse((self.root / "trajectory").exists())
        records = sorted((self.root / "trajectory").glob("trajectory_step_*.json"))
        self.assertEqual(len(records), 4)
        final = json.loads(records[-1].read_text(encoding="utf-8"))
        self.assertEqual(final["updates"], 8)
        self.assertEqual(final["start_update"], "step_000002_epoch_000_minibatch_0000")
        self.assertEqual(final["end_update"], "step_000003_epoch_001_minibatch_0001")
        self.assertEqual(set(final["baselines"]), {"shuffled_labels"})
        self.assertFalse(list((self.root / "trajectory").glob("step_*.json")))
        self.assertFalse(list((self.root / "trajectory").rglob("*.bin")))
        self.assertTrue(all(path.suffix == ".json" for path in (self.root / "trajectory").rglob("*") if path.is_file()))
        self.assertTrue(any(
            not torch.equal(left, right)
            for left, right in zip(trajectory.supervised_parameters, trajectory.shuffled_parameters)
        ))

    def test_trajectory_without_baselines_does_not_require_shuffle_or_policy_only_passes(self):
        actor = make_actor(True, self.root / "supervised_only", mode="paired")
        actor.gradient_record_config.update(
            single_step={"enabled": False}, trajectory={"enabled": True}, baselines={"enabled": False}
        )
        with patch.object(actor, "_backward_micro_batches", wraps=actor._backward_micro_batches) as backward:
            actor.update_policy(paired_batch())
        self.assertEqual(backward.call_count, 2)
        self.assertTrue(all(not call.kwargs.get("policy_only", False) for call in backward.call_args_list))
        path = next((self.root / "supervised_only").glob("trajectory_step_*.json"))
        record = json.loads(path.read_text(encoding="utf-8"))
        self.assertFalse(record.get("baselines"))
        self.assertFalse(list((self.root / "supervised_only").glob("step_*.json")))

    def test_single_trajectory_baseline_switches_propagate_to_state_and_backward_passes(self):
        for shuffled_enabled, random_enabled in ((True, False), (False, True)):
            with self.subTest(shuffled=shuffled_enabled, random=random_enabled):
                directory = self.root / f"trajectory_{shuffled_enabled}_{random_enabled}"
                actor = make_actor(True, directory, mode="paired")
                actor.gradient_record_config.update(
                    single_step={"enabled": False}, trajectory={"enabled": True},
                    baselines={
                        "enabled": True, "seed": 23,
                        "shuffled_labels": {"enabled": shuffled_enabled},
                        "random_direction": {"enabled": random_enabled},
                    },
                )
                batch = TestPairedRandomBaselines.baseline_batch() if shuffled_enabled else paired_batch()
                with patch.object(actor, "_backward_micro_batches", wraps=actor._backward_micro_batches) as backward:
                    actor.update_policy(batch)
                self.assertEqual(backward.call_count, 2 + int(shuffled_enabled))
                trajectory = actor.optimization_trajectory
                self.assertEqual(trajectory.shuffled_labels_enabled, shuffled_enabled)
                self.assertEqual(trajectory.random_direction_enabled, random_enabled)
                self.assertEqual(hasattr(trajectory, "random_displacement"), random_enabled)
                self.assertEqual(hasattr(trajectory, "shuffled_optimizer_state"), shuffled_enabled)
                record = json.loads(next(directory.glob("trajectory_step_*.json")).read_text(encoding="utf-8"))
                expected = {"shuffled_labels"} if shuffled_enabled else {"random_update"}
                self.assertEqual(set(record["baselines"]), expected)


if __name__ == "__main__":
    unittest.main()
