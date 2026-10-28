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

    def trainer(
        self, *, enabled=True, mode="paired", step=1, warmup=0, baselines=False, seed=0,
        trajectory=False, single_step=None,
    ):
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
        if baselines:
            trainer.config.actor_rollout_ref.actor.gradient_record["baselines"] = Config(enabled=True, seed=seed)
        if trajectory:
            trainer.config.actor_rollout_ref.actor.gradient_record["trajectory"] = Config(enabled=True)
        if single_step is not None:
            trainer.config.actor_rollout_ref.actor.gradient_record["single_step"] = Config(enabled=single_step)
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

    def test_shuffled_baseline_reuses_official_labels_and_leaves_training_unchanged(self):
        trainer, batch = self.trainer(baselines=True, seed=17), self.batch()
        originals = {key: value.clone() for key, value in batch.batch.items()}
        original_objects = dict(batch.batch)
        official_rewards = self.rewards.clone()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state().clone()
        trainer._prepare_paired_gradient_advantages(batch)
        self.oracle.assert_called_once_with(batch)
        shuffled, metadata = diagnostics.shuffle_outcome_rewards(
            official_rewards,
            batch.non_tensor_batch["uid"],
            seed=diagnostics.diagnostic_seed(17, "shuffled_labels", 1),
        )
        expected, _ = core["compute_grpo_outcome_advantage"](
            shuffled,
            batch.batch["response_mask"],
            batch.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=True,
        )
        torch.testing.assert_close(batch.batch["diagnostic_shuffled_advantages"], expected, rtol=0, atol=0)
        self.assertEqual(batch.meta_info["gradient_baselines"], metadata)
        self.assertEqual(
            set(batch.batch) - set(originals),
            {"diagnostic_supervised_advantages", "diagnostic_shuffled_advantages"},
        )
        for key, tensor in originals.items():
            self.assertIs(batch.batch[key], original_objects[key])
            torch.testing.assert_close(batch.batch[key], tensor, rtol=0, atol=0)
        torch.testing.assert_close(self.rewards, official_rewards, rtol=0, atol=0)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state))
        state_after = np.random.get_state()
        self.assertEqual(numpy_state[0], state_after[0])
        np.testing.assert_array_equal(numpy_state[1], state_after[1])
        self.assertEqual(numpy_state[2:], state_after[2:])

    def test_shuffled_baseline_respects_grpo_normalization_and_can_be_disabled(self):
        trainer, batch = self.trainer(baselines=True), self.batch()
        trainer.config.algorithm["norm_adv_by_std_in_grpo"] = False
        trainer._prepare_paired_gradient_advantages(batch)
        shuffled, _ = diagnostics.shuffle_outcome_rewards(
            self.rewards, batch.non_tensor_batch["uid"], seed=diagnostics.diagnostic_seed(0, "shuffled_labels", 1)
        )
        expected, _ = core["compute_grpo_outcome_advantage"](
            shuffled, batch.batch["response_mask"], batch.non_tensor_batch["uid"], norm_adv_by_std_in_grpo=False
        )
        torch.testing.assert_close(batch.batch["diagnostic_shuffled_advantages"], expected, rtol=0, atol=0)
        trainer, batch = self.trainer(), self.batch()
        trainer.config.actor_rollout_ref.actor.gradient_record["baselines"] = Config(enabled=False)
        trainer._prepare_paired_gradient_advantages(batch)
        self.assertNotIn("diagnostic_shuffled_advantages", batch.batch)
        self.assertNotIn("gradient_baselines", batch.meta_info)

    def test_online_trajectory_with_disabled_baselines_only_prepares_supervision(self):
        trainer, batch = self.trainer(baselines=False, trajectory=True, single_step=False), self.batch()
        original = {key: value.clone() for key, value in batch.batch.items()}
        with patch.object(diagnostics, "shuffle_outcome_rewards", wraps=diagnostics.shuffle_outcome_rewards) as shuffle:
            trainer._prepare_paired_gradient_advantages(batch)
        self.assertIn("diagnostic_supervised_advantages", batch.batch)
        self.assertNotIn("diagnostic_shuffled_advantages", batch.batch)
        self.assertNotIn("gradient_baselines", batch.meta_info)
        shuffle.assert_not_called()
        for name, value in original.items():
            torch.testing.assert_close(batch.batch[name], value, rtol=0, atol=0)

    def test_individual_baseline_switches_only_prepare_requested_shuffled_labels(self):
        for shuffled_enabled, random_enabled in ((True, False), (False, True), (False, False)):
            with self.subTest(shuffled=shuffled_enabled, random=random_enabled):
                trainer, batch = self.trainer(baselines=True, trajectory=True, single_step=False), self.batch()
                baseline_config = trainer.config.actor_rollout_ref.actor.gradient_record["baselines"]
                baseline_config.update(
                    shuffled_labels=Config(enabled=shuffled_enabled),
                    random_direction=Config(enabled=random_enabled),
                )
                originals = {key: value.clone() for key, value in batch.batch.items()}
                with patch.object(
                    diagnostics, "shuffle_outcome_rewards", wraps=diagnostics.shuffle_outcome_rewards
                ) as shuffle:
                    trainer._prepare_paired_gradient_advantages(batch)
                self.assertEqual(shuffle.call_count, int(shuffled_enabled))
                self.assertIn("diagnostic_supervised_advantages", batch.batch)
                self.assertEqual("diagnostic_shuffled_advantages" in batch.batch, shuffled_enabled)
                self.assertEqual("gradient_baselines" in batch.meta_info, shuffled_enabled)
                for name, original in originals.items():
                    torch.testing.assert_close(batch.batch[name], original, rtol=0, atol=0)

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

    def test_trajectory_scores_every_step_after_warmup_without_changing_export_interval(self):
        for step in (1, 2, 9, 10, 11):
            trainer = self.trainer(step=step, trajectory=True)
            self.assertTrue(trainer._paired_gradient_due())
            record = trainer.config.actor_rollout_ref.actor.gradient_record
            self.assertEqual(diagnostics.recording_due(record, step), step == 1 or step % 10 == 0)
        self.assertFalse(self.trainer(step=2, warmup=3, trajectory=True)._paired_gradient_due())
        self.assertFalse(self.trainer(step=2, enabled=False, trajectory=True)._paired_gradient_due())
        self.assertFalse(self.trainer(step=2, mode="full", trajectory=True)._paired_gradient_due())

    def test_trajectory_prepares_both_labels_when_legacy_single_step_is_disabled(self):
        trainer = self.trainer(step=2, trajectory=True, baselines=True, single_step=False)
        batch = self.batch()
        self.assertTrue(trainer._paired_gradient_due())
        trainer._prepare_paired_gradient_advantages(batch)
        self.oracle.assert_called_once_with(batch)
        self.assertIn("diagnostic_supervised_advantages", batch.batch)
        self.assertIn("diagnostic_shuffled_advantages", batch.batch)
        self.assertIn("gradient_baselines", batch.meta_info)
        self.assertFalse(diagnostics.paired_recording_due(trainer.config.actor_rollout_ref.actor.gradient_record, 2))

    def test_single_step_switch_preserves_legacy_default_and_skips_disabled_oracle(self):
        for step in (1, 10):
            self.assertTrue(self.trainer(step=step)._paired_gradient_due())
            self.assertTrue(self.trainer(step=step, single_step=True)._paired_gradient_due())
            trainer, batch = self.trainer(step=step, single_step=False), self.batch()
            self.assertFalse(trainer._paired_gradient_due())
            trainer._prepare_paired_gradient_advantages(batch)
            self.assertNotIn("diagnostic_supervised_advantages", batch.batch)
        self.oracle.assert_not_called()

    def test_only_selected_rollouts_record_trace_without_forwarding_answers(self):
        for options, trace_expected in (
            ({"step": 1}, True),
            ({"step": 10}, True),
            ({"step": 2}, False),
            ({"step": 2, "trajectory": True}, True),
            ({"step": 2, "trajectory": True, "single_step": False}, True),
            ({"step": 1, "single_step": False}, False),
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


class TestShuffledOutcomeRewards(unittest.TestCase):
    def test_seed_derivation_is_stable_distinguishes_streams_and_fits_torch_seed(self):
        seed = diagnostics.diagnostic_seed(13, "shuffled_labels", 2)
        self.assertEqual(seed, diagnostics.diagnostic_seed(np.int64(13), "shuffled_labels", 2))
        self.assertNotEqual(seed, diagnostics.diagnostic_seed(13, "shuffled_labels", 3))
        self.assertNotEqual(seed, diagnostics.diagnostic_seed(13, "gaussian_direction", 2))
        self.assertGreaterEqual(seed, 0)
        self.assertLess(seed, 1 << 63)
        with self.assertRaisesRegex(ValueError, "integer"):
            diagnostics.diagnostic_seed(True, "shuffled_labels")

    def test_permutation_preserves_each_interleaved_group_and_token_rewards(self):
        rewards = torch.tensor(
            [[0, 1, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=torch.float64
        )
        original = rewards.clone()
        index = np.array(["a", "b", "a", "b", "a", "b"], dtype=object)
        shuffled, metadata = diagnostics.shuffle_outcome_rewards(rewards, index, seed=7)
        repeated, repeated_metadata = diagnostics.shuffle_outcome_rewards(rewards, index, seed=7)
        self.assertEqual(shuffled.shape, (6, 1))
        self.assertEqual(shuffled.dtype, rewards.dtype)
        torch.testing.assert_close(shuffled, repeated, rtol=0, atol=0)
        self.assertEqual(metadata, repeated_metadata)
        for uid in ("a", "b"):
            rows = torch.as_tensor(np.flatnonzero(index == uid))
            actual = shuffled.index_select(0, rows).flatten().sort().values
            expected = rewards.sum(-1).index_select(0, rows).sort().values
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(metadata["group_count"], 2)
        self.assertEqual(metadata["informative_group_count"], 2)
        self.assertEqual(metadata["changed_label_count"], int(torch.count_nonzero(shuffled[:, 0] != rewards.sum(-1))))
        torch.testing.assert_close(rewards, original, rtol=0, atol=0)

    def test_original_masks_are_applied_after_trajectory_label_permutation(self):
        rewards = torch.tensor([[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
        masks = torch.tensor([[1, 0, 0], [1, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=torch.float32)
        index = np.array(["a", "a", "b", "b"], dtype=object)
        shuffled, _ = diagnostics.shuffle_outcome_rewards(rewards, index, seed=3)
        advantages, _ = core["compute_grpo_outcome_advantage"](
            shuffled, masks, index, norm_adv_by_std_in_grpo=False
        )
        for start in (0, 2):
            values = shuffled[start : start + 2, 0]
            expected = (values - values.mean()).unsqueeze(-1) * masks[start : start + 2]
            torch.testing.assert_close(advantages[start : start + 2], expected, rtol=0, atol=0)

    def test_uniform_labels_and_identity_permutations_remain_explicitly_degenerate(self):
        rewards = torch.tensor([[0.0], [0.0], [1.0], [1.0]])
        index = np.array(["a", "a", "b", "b"], dtype=object)
        shuffled, metadata = diagnostics.shuffle_outcome_rewards(rewards, index, seed=5)
        torch.testing.assert_close(shuffled, rewards, rtol=0, atol=0)
        self.assertEqual(metadata["informative_group_count"], 0)
        self.assertEqual(metadata["changed_group_count"], 0)
        self.assertEqual(metadata["changed_label_count"], 0)
        identity_seed = next(
            seed for seed in range(100) if np.array_equal(np.random.default_rng(seed).permutation(2), [0, 1])
        )
        _, metadata = diagnostics.shuffle_outcome_rewards(torch.tensor([[0.0], [1.0]]), ["a", "a"], seed=identity_seed)
        self.assertEqual(metadata["informative_group_count"], 1)
        self.assertEqual(metadata["changed_label_count"], 0)

    def test_invalid_shapes_nonfinite_labels_and_seeds_fail_early(self):
        for rewards, index, seed in (
            (torch.zeros(2), ["a", "a"], 0),
            (torch.zeros(2, 1), ["a"], 0),
            (torch.tensor([[float("nan")]]), ["a"], 0),
            (torch.zeros(1, 1), ["a"], -1),
            (torch.zeros(1, 1), ["a"], True),
        ):
            with self.subTest(rewards=rewards, index=index, seed=seed), self.assertRaises(ValueError):
                diagnostics.shuffle_outcome_rewards(rewards, index, seed=seed)


if __name__ == "__main__":
    unittest.main()
