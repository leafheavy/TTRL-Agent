"""Exercise production trainer hooks without requiring Ray or GPU workers."""

import ast
import importlib.util
import sys
import unittest
import uuid
from collections import defaultdict
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "paired_gradient_diagnostics", ROOT / "verl/utils/gradient_diagnostics.py"
)
diagnostics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostics)


class Config(dict):
    __getattr__ = dict.__getitem__


class Batch:
    def __init__(self, tensors=None, non_tensor_batch=None, rows=4):
        self.batch = tensors or {}
        self.non_tensor_batch = non_tensor_batch or {}
        self.meta_info = {}
        self.rows = rows

    def __len__(self):
        return self.rows


def source_definitions(path, names, namespace, methods=None):
    nodes = []
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            node.decorator_list = []
            if methods is not None and isinstance(node, ast.ClassDef):
                node.bases = []
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


core = {"torch": torch, "np": np, "defaultdict": defaultdict}
source_definitions(ROOT / "verl/trainer/ppo/core_algos.py", {"compute_grpo_outcome_advantage"}, core)
namespace = {"np": np, "uuid": uuid, "core_algos": SimpleNamespace(**core)}
source_definitions(
    ROOT / "verl/trainer/ppo/ray_trainer.py",
    {"RayPPOTrainer"},
    namespace,
    methods={"_paired_gradient_due", "_prepare_paired_gradient_advantages", "_generate_training_rollouts"},
)
Trainer = namespace["RayPPOTrainer"]


class TestPairedGradientTrainer(unittest.TestCase):
    def setUp(self):
        self.rewards = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        self.oracle = Mock(return_value=self.rewards)
        self.bfcl = ModuleType("verl.utils.reward_score.ttrl_bfcl")
        self.bfcl.BFCLSupervisedReward = Mock(return_value=self.oracle)
        self.bfcl.decode_bfcl_context = lambda context: context
        self.bfcl.apply_bfcl_rewards = Mock(side_effect=lambda batch, *_: batch)
        self.ttrl = ModuleType("verl.trainer.ppo.ttrl_utils")
        self.ttrl.apply_ttrl_gt = Mock()
        self.ttrl.select_top_k_per_prompt = Mock()
        modules = patch.dict(
            sys.modules,
            {
                "verl.utils.gradient_diagnostics": diagnostics,
                self.bfcl.__name__: self.bfcl,
                self.ttrl.__name__: self.ttrl,
            },
        )
        modules.start()
        self.addCleanup(modules.stop)

    def trainer(self, *, enabled=True, mode="paired", step=1, warmup=0):
        trainer = Trainer()
        trainer.config = Config(
            actor_rollout_ref=Config(
                actor=Config(gradient_record=Config(enabled=enabled, mode=mode, interval=10, answer_dir="answers")),
                rollout=Config(n=2, name="vllm", multi_turn=Config(enable=True)),
            ),
            trainer=Config(critic_warmup=warmup),
            algorithm=Config(norm_adv_by_std_in_grpo=True),
            ttrl=Config(enable=True, reward_mode="bfcl", share_rollouts=True),
            benchmark_runtime=Config(endpoint="http://runtime"),
        )
        trainer.global_steps = step
        trainer.async_rollout_mode = True
        trainer.tokenizer = object()
        return trainer

    def batch(self):
        mask = torch.tensor([[1.0, 0.0], [1.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        return Batch(
            {
                "response_mask": mask,
                "token_level_rewards": torch.full((4, 2), 0.3),
                "token_level_scores": torch.full((4, 2), 0.3),
                "advantages": 0.2 * torch.tensor([[1, 0], [-1, -1], [1, 0], [-1, -1]]),
                "returns": torch.full((4, 2), 2.0),
            },
            {"uid": np.array(["a", "a", "b", "b"], dtype=object)},
        )

    def test_supervised_advantages_reuse_actual_grpo_and_preserve_training_tensors(self):
        trainer, batch = self.trainer(), self.batch()
        original = {key: tensor.clone() for key, tensor in batch.batch.items()}
        original_objects = dict(batch.batch)
        trainer._prepare_paired_gradient_advantages(batch)
        self.bfcl.BFCLSupervisedReward.assert_called_once_with(
            answer_dir="answers", runtime_config=trainer.config.benchmark_runtime
        )
        self.oracle.assert_called_once_with(batch)
        expected, _ = core["compute_grpo_outcome_advantage"](
            self.rewards, batch.batch["response_mask"], batch.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=True,
        )
        torch.testing.assert_close(batch.batch["diagnostic_supervised_advantages"], expected, rtol=0, atol=0)
        self.assertEqual(set(batch.batch) - set(original), {"diagnostic_supervised_advantages"})
        for key, tensor in original.items():
            self.assertIs(batch.batch[key], original_objects[key])
            self.assertTrue(torch.equal(batch.batch[key], tensor), key)

    def test_supervised_advantages_respect_grpo_normalization_setting(self):
        trainer, batch = self.trainer(), self.batch()
        trainer.config.algorithm["norm_adv_by_std_in_grpo"] = False
        trainer._prepare_paired_gradient_advantages(batch)
        expected = torch.tensor([[-0.5, 0.0], [0.5, 0.5], [0.0, 0.0], [0.0, 0.0]])
        torch.testing.assert_close(batch.batch["diagnostic_supervised_advantages"], expected, rtol=0, atol=0)

    def test_disabled_full_skipped_and_warmup_do_not_use_oracle(self):
        cases = (
            {"enabled": False},
            {"mode": "full"},
            {"step": 2},
            {"step": 1, "warmup": 2},
        )
        for options in cases:
            with self.subTest(options=options):
                trainer, batch = self.trainer(**options), self.batch()
                self.assertFalse(trainer._paired_gradient_due())
                trainer._prepare_paired_gradient_advantages(batch)
                self.assertNotIn("diagnostic_supervised_advantages", batch.batch)
        self.bfcl.BFCLSupervisedReward.assert_not_called()
        self.oracle.assert_not_called()

    def test_first_step_and_interval_are_selected(self):
        for step, expected in ((1, True), (2, False), (9, False), (10, True), (20, True)):
            with self.subTest(step=step):
                self.assertEqual(self.trainer(step=step)._paired_gradient_due(), expected)

    def test_only_selected_rollouts_record_trace_without_forwarding_answers(self):
        for options, trace_expected in (
            ({"step": 1}, True),
            ({"step": 10}, True),
            ({"step": 2}, False),
            ({"enabled": False}, False),
            ({"mode": "full"}, False),
        ):
            with self.subTest(options=options):
                trainer = self.trainer(**options)
                context = {
                    "functions": [], "initial_config": {}, "involved_classes": ["MathAPI"],
                    "ground_truth": "private reference",
                }
                batch = Batch(
                    non_tensor_batch={"extra_info": np.array([{"bfcl": context}] * 2, dtype=object)}, rows=2
                )
                prompts = Batch(
                    non_tensor_batch={"raw_prompt": np.array([[{"role": "user"}]] * 2, dtype=object)}, rows=2
                )
                output = Batch(non_tensor_batch={"bfcl_outcome": np.array([{}] * 4, dtype=object)}, rows=4)
                trainer.async_rollout_manager = SimpleNamespace(generate_sequences=Mock(return_value=output))
                returned_batch, returned_output = trainer._generate_training_rollouts(batch, prompts)
                self.assertIs(returned_batch, batch)
                self.assertIs(returned_output, output)
                trainer.async_rollout_manager.generate_sequences.assert_called_once_with(prompts)
                for prepared in prompts.non_tensor_batch["bfcl_context"]:
                    self.assertEqual(prepared.get("record_execution_trace", False), trace_expected)
                    self.assertNotIn("ground_truth", prepared)
                self.assertEqual(prompts.non_tensor_batch["agent_name"].tolist(), ["tool_agent"] * 2)
        self.bfcl.BFCLSupervisedReward.assert_not_called()
        self.oracle.assert_not_called()


if __name__ == "__main__":
    unittest.main()
