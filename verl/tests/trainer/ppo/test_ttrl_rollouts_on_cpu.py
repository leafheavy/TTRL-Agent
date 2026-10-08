"""Sampling orchestration tests; workers and math voting are mocked."""

import sys
import unittest
from copy import deepcopy
from types import ModuleType
from unittest.mock import Mock, patch

import numpy as np

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class Config(dict):
    def __getattr__(self, name):
        return self[name]


class Batch:
    def __init__(self, rows, non_tensor_batch=None, meta_info=None):
        self.rows = rows
        self.non_tensor_batch = non_tensor_batch or {}
        self.meta_info = meta_info or {}

    def __len__(self):
        return len(self.rows)

    def repeat(self, repeat_times, interleave):
        assert interleave
        return Batch(
            [row for row in self.rows for _ in range(repeat_times)],
            {key: np.repeat(value, repeat_times, axis=0) for key, value in self.non_tensor_batch.items()},
            deepcopy(self.meta_info),
        )


class TestTTRLRollouts(unittest.TestCase):
    def setUp(self):
        # The module under test only dispatches to the existing voting utility.
        self.utils = ModuleType("verl.trainer.ppo.ttrl_utils")
        self.utils.apply_ttrl_gt = Mock(side_effect=lambda batch, *_: batch)
        self.utils.select_top_k_per_prompt = Mock(
            side_effect=lambda data, n, k: Batch(
                [row for start in range(0, len(data), n) for row in data.rows[start : start + k]]
            )
        )
        self.module_patch = patch.dict(sys.modules, {self.utils.__name__: self.utils})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def make_trainer(self, *, backend="vllm", asynchronous=False, multi_turn=False, enabled=True, shared=True):
        trainer = object.__new__(RayPPOTrainer)
        rollout = Config(
            n=4,
            name=backend,
            mode="async" if asynchronous else "sync",
            multi_turn=Config(enable=multi_turn),
        )
        trainer.config = Config(
            actor_rollout_ref=Config(rollout=rollout),
            algorithm=Config(adv_estimator="grpo"),
            data=Config(return_raw_chat=True),
            ttrl=Config(enable=enabled, share_rollouts=shared, n_votes_per_prompt=4, n_samples_per_prompt=4),
        )
        trainer.async_rollout_mode = asynchronous
        trainer.tokenizer = object()

        def generate(prompts):
            if not asynchronous and backend == "sglang" and multi_turn:
                n = 1  # The sync tool worker receives already repeated rows.
            elif asynchronous:
                n = rollout.n  # The agent manager repeats internally.
            else:
                n = prompts.meta_info.get("kwargs", {}).get("n", rollout.n)
            return Batch([f"{row}:{sample}" for row in prompts.rows for sample in range(n)])

        trainer.actor_rollout_wg = Config(generate_sequences=Mock(side_effect=generate))
        trainer.async_rollout_manager = Config(generate_sequences=Mock(side_effect=generate))
        return trainer

    def run_generation(self, trainer):
        trainer._validate_ttrl_config()
        batch, prompts = Batch(["task-a", "task-b"]), Batch(["task-a", "task-b"])
        if trainer.config.ttrl.get("reward_mode") == "bfcl":
            batch.non_tensor_batch["extra_info"] = np.array([{"bfcl": {"functions": []}}] * 2, dtype=object)
        if trainer.config.ttrl.get("reward_mode") == "bfcl" and trainer.async_rollout_mode:
            prompts.non_tensor_batch["raw_prompt"] = np.array([[{"role": "user", "content": "task"}]] * 2, dtype=object)
        returned_batch, output = trainer._generate_training_rollouts(batch, prompts)
        self.assertIs(returned_batch, batch)
        self.assertEqual(len(batch), 2)  # Prompt metadata is repeated later in fit().
        self.assertEqual(len(set(batch.non_tensor_batch["uid"])), 2)
        return batch, prompts, output

    def test_shared_sync_keeps_the_entire_voting_output(self):
        trainer = self.make_trainer()
        batch, prompts, output = self.run_generation(trainer)
        trainer.actor_rollout_wg.generate_sequences.assert_called_once_with(prompts)
        trainer.async_rollout_manager.generate_sequences.assert_not_called()
        self.utils.apply_ttrl_gt.assert_called_once_with(batch, output, 4, trainer.tokenizer)
        self.utils.select_top_k_per_prompt.assert_not_called()
        self.assertEqual(output.rows, [f"{task}:{i}" for task in batch.rows for i in range(4)])
        self.assertNotIn("kwargs", prompts.meta_info)

    def test_shared_async_uses_the_agent_manager_without_pre_repeating(self):
        for backend in ("vllm", "sglang"):
            with self.subTest(backend=backend):
                trainer = self.make_trainer(backend=backend, asynchronous=True, multi_turn=True)
                batch, prompts, output = self.run_generation(trainer)
                trainer.async_rollout_manager.generate_sequences.assert_called_once_with(prompts)
                trainer.actor_rollout_wg.generate_sequences.assert_not_called()
                self.assertEqual(len(prompts), 2)
                self.assertEqual(len(output), 8)  # B * N, never B * N * N.
                self.assertEqual(len(batch.repeat(4, True)), len(output))
        self.utils.select_top_k_per_prompt.assert_not_called()

    def test_shared_sync_sglang_repeats_only_generation_inputs(self):
        trainer = self.make_trainer(backend="sglang", multi_turn=True)
        batch, prompts, output = self.run_generation(trainer)
        trainer.actor_rollout_wg.generate_sequences.assert_called_once()
        sent = trainer.actor_rollout_wg.generate_sequences.call_args.args[0]
        self.assertEqual(len(prompts), 2)
        self.assertEqual(sent.rows, ["task-a"] * 4 + ["task-b"] * 4)
        self.assertEqual(len(output), 8)
        np.testing.assert_array_equal(sent.non_tensor_batch["uid"], np.repeat(batch.non_tensor_batch["uid"], 4))
        self.utils.apply_ttrl_gt.assert_called_once_with(batch, output, 4, trainer.tokenizer)
        self.utils.select_top_k_per_prompt.assert_not_called()

    def test_legacy_voting_pool_is_still_sampled_once_then_downsampled(self):
        trainer = self.make_trainer(shared=False)
        trainer.config.ttrl["n_samples_per_prompt"] = 2
        trainer.config.actor_rollout_ref.rollout["n"] = 2
        batch, prompts, output = self.run_generation(trainer)
        trainer.actor_rollout_wg.generate_sequences.assert_called_once_with(prompts)
        self.assertEqual(prompts.meta_info["kwargs"]["n"], 4)
        voting_output = self.utils.apply_ttrl_gt.call_args.args[1]
        self.assertEqual(len(voting_output), 8)
        self.utils.select_top_k_per_prompt.assert_called_once_with(voting_output, 4, 2)
        self.assertEqual(output.rows, ["task-a:0", "task-a:1", "task-b:0", "task-b:1"])
        self.assertEqual(len(batch.repeat(2, True)), len(output))

    def test_disabled_ttrl_preserves_grpo_dispatch(self):
        trainer = self.make_trainer(backend="sglang", asynchronous=True, multi_turn=True, enabled=False)
        _, prompts, output = self.run_generation(trainer)
        trainer.async_rollout_manager.generate_sequences.assert_called_once_with(prompts)
        trainer.actor_rollout_wg.generate_sequences.assert_not_called()
        self.utils.apply_ttrl_gt.assert_not_called()
        self.utils.select_top_k_per_prompt.assert_not_called()
        self.assertEqual(len(output), 8)

    def test_counts_can_default_to_rollout_n(self):
        trainer = self.make_trainer()
        trainer.config.ttrl.pop("n_votes_per_prompt")
        trainer.config.ttrl.pop("n_samples_per_prompt")
        _, _, output = self.run_generation(trainer)
        self.assertEqual(len(output), 8)

    def test_invalid_shared_counts_fail_before_generation(self):
        invalid_counts = ((8, 4, 4), (4, 2, 4), (2, 4, 4), (1, 1, 1), (0, 0, 0), (True, 4, 4), (4.5, 4, 4))
        for votes, samples, n in invalid_counts:
            with self.subTest(votes=votes, samples=samples, n=n):
                trainer = self.make_trainer()
                trainer.config.ttrl.update(n_votes_per_prompt=votes, n_samples_per_prompt=samples)
                trainer.config.actor_rollout_ref.rollout["n"] = n
                with self.assertRaises(ValueError):
                    trainer._validate_ttrl_config()
                trainer.actor_rollout_wg.generate_sequences.assert_not_called()
                trainer.async_rollout_manager.generate_sequences.assert_not_called()

    def test_shared_rollouts_require_grpo(self):
        trainer = self.make_trainer()
        trainer.config.algorithm["adv_estimator"] = "remax"
        with self.assertRaisesRegex(ValueError, "adv_estimator=grpo"):
            trainer._validate_ttrl_config()

    def test_legacy_mode_rejects_unsupported_dispatch(self):
        for asynchronous in (True, False):
            with self.subTest(asynchronous=asynchronous):
                trainer = self.make_trainer(backend="sglang", asynchronous=asynchronous, multi_turn=True, shared=False)
                with self.assertRaisesRegex(ValueError, "share_rollouts=True"):
                    trainer._validate_ttrl_config()

    def test_unexpected_rollout_count_fails_before_voting(self):
        trainer = self.make_trainer()
        trainer.actor_rollout_wg.generate_sequences.side_effect = None
        trainer.actor_rollout_wg.generate_sequences.return_value = Batch(["too-few"])
        with self.assertRaisesRegex(ValueError, "Expected 8 TTRL rollouts, got 1"):
            self.run_generation(trainer)
        self.utils.apply_ttrl_gt.assert_not_called()

    def test_bfcl_rewards_reuse_generated_outputs_without_math_labels(self):
        bfcl = ModuleType("verl.utils.reward_score.ttrl_bfcl")
        bfcl.decode_bfcl_context = lambda context: context
        bfcl.apply_bfcl_rewards = Mock(side_effect=lambda batch, *_: batch)
        for asynchronous in (False, True):
            with self.subTest(asynchronous=asynchronous), patch.dict(sys.modules, {bfcl.__name__: bfcl}):
                trainer = self.make_trainer(asynchronous=asynchronous)
                trainer.config.ttrl["reward_mode"] = "bfcl"
                batch, prompts, output = self.run_generation(trainer)
                worker = trainer.async_rollout_manager if asynchronous else trainer.actor_rollout_wg
                worker.generate_sequences.assert_called_once_with(prompts)
                bfcl.apply_bfcl_rewards.assert_called_once_with(batch, output, 4, trainer.tokenizer)
                self.assertEqual(len(output), 8)
                if asynchronous:
                    self.assertEqual(prompts.non_tensor_batch["agent_name"].tolist(), ["single_turn_agent"] * 2)
                    self.assertEqual(len(prompts.non_tensor_batch["bfcl_context"]), 2)
                bfcl.apply_bfcl_rewards.reset_mock()
        self.utils.apply_ttrl_gt.assert_not_called()
        self.utils.select_top_k_per_prompt.assert_not_called()

    def test_bfcl_requires_shared_supported_dispatch_without_external_rewards(self):
        cases = (
            ({"shared": False}, {}, "share_rollouts=True"),
            ({"multi_turn": True}, {}, "rollout.mode=async"),
            ({}, {"reward_model": Config(enable=True)}, "external reward model"),
        )
        for kwargs, overrides, message in cases:
            with self.subTest(message=message):
                trainer = self.make_trainer(**kwargs)
                trainer.config.update(overrides)
                trainer.config.ttrl["reward_mode"] = "bfcl"
                with self.assertRaisesRegex(ValueError, message):
                    trainer._validate_ttrl_config()
                trainer.actor_rollout_wg.generate_sequences.assert_not_called()
                trainer.async_rollout_manager.generate_sequences.assert_not_called()

    def test_bfcl_multi_turn_context_is_forwarded_once_without_reference_fields(self):
        bfcl = ModuleType("verl.utils.reward_score.ttrl_bfcl")
        bfcl.decode_bfcl_context = lambda context: context
        bfcl.apply_bfcl_rewards = Mock(side_effect=lambda batch, *_: batch)
        trainer = self.make_trainer(asynchronous=True, multi_turn=True)
        trainer.config.ttrl["reward_mode"] = "bfcl"
        trainer.config["data"] = Config(return_raw_chat=True)
        trainer._validate_ttrl_config()
        contexts = [
            {"functions": [], "initial_config": {}, "involved_classes": ["MathAPI"], "ground_truth": "hidden"}
        ] * 2
        batch = Batch(["a", "b"], {"extra_info": np.array([{"bfcl": context} for context in contexts], dtype=object)})
        prompts = Batch(["a", "b"], {"raw_prompt": np.array([[{"role": "user", "content": "task"}]] * 2, dtype=object)})
        output = Batch(list(range(8)), {"bfcl_outcome": np.array([{"valid": False, "key": None}] * 8, dtype=object)})
        trainer.async_rollout_manager.generate_sequences = Mock(return_value=output)
        with patch.dict(sys.modules, {bfcl.__name__: bfcl}):
            _, generated = trainer._generate_training_rollouts(batch, prompts)
        self.assertIs(generated, output)
        trainer.async_rollout_manager.generate_sequences.assert_called_once_with(prompts)
        self.assertEqual(len(prompts), 2)
        self.assertEqual(prompts.non_tensor_batch["agent_name"].tolist(), ["tool_agent"] * 2)
        self.assertNotIn("ground_truth", prompts.non_tensor_batch["bfcl_context"][0])
        bfcl.apply_bfcl_rewards.assert_called_once_with(batch, output, 4, trainer.tokenizer)
        self.utils.apply_ttrl_gt.assert_not_called()

    def test_bfcl_multi_turn_requires_raw_messages_and_outcome_records(self):
        trainer = self.make_trainer(asynchronous=True, multi_turn=True)
        trainer.config.ttrl["reward_mode"] = "bfcl"
        trainer.config.data["return_raw_chat"] = False
        with self.assertRaisesRegex(ValueError, "return_raw_chat=True"):
            trainer._validate_ttrl_config()
        trainer.config["data"] = Config(return_raw_chat=True)
        bfcl = ModuleType("verl.utils.reward_score.ttrl_bfcl")
        bfcl.decode_bfcl_context = lambda context: context
        bfcl.apply_bfcl_rewards = Mock()
        with patch.dict(sys.modules, {bfcl.__name__: bfcl}):
            with self.assertRaisesRegex(ValueError, "raw_prompt"):
                trainer._generate_training_rollouts(Batch(["a"]), Batch(["a"]))
            batch = Batch(["a"], {"extra_info": np.array([{"bfcl": {"involved_classes": ["MathAPI"]}}], dtype=object)})
            prompts = Batch(["a"], {"raw_prompt": np.array([[{"role": "user", "content": "task"}]], dtype=object)})
            with self.assertRaisesRegex(ValueError, "require outcomes"):
                trainer._generate_training_rollouts(batch, prompts)
        bfcl.apply_bfcl_rewards.assert_not_called()

    def test_bfcl_consensus_modes_fail_before_worker_generation(self):
        trainer = self.make_trainer()
        trainer.config.ttrl.update(reward_mode="bfcl", consensus=Config(mode="observations"))
        with self.assertRaisesRegex(ValueError, "mode=calls"):
            trainer._validate_ttrl_config()
        trainer.config.ttrl["consensus"] = Config(mode="calls_unordered")
        with self.assertRaisesRegex(ValueError, "mode"):
            trainer._validate_ttrl_config()
        trainer.actor_rollout_wg.generate_sequences.assert_not_called()

    def test_bfcl_final_state_projection_and_call_mode_reach_the_shared_agent_context(self):
        bfcl = ModuleType("verl.utils.reward_score.ttrl_bfcl")
        bfcl.decode_bfcl_context = lambda context: context
        bfcl.apply_bfcl_rewards = Mock(side_effect=lambda batch, *_: batch)
        for mode in ("calls", "state"):
            trainer = self.make_trainer(asynchronous=True, multi_turn=True)
            trainer.config.ttrl.update(reward_mode="bfcl", consensus=Config(
                mode=mode, observable_state={"MathAPI": ["history"], "GorillaFileSystem": ["root"]}))
            context = {"functions": [], "involved_classes": ["MathAPI"], "ground_truth": "hidden"}
            batch = Batch(["a"], {"extra_info": np.array([{"bfcl": context}], dtype=object)})
            prompts = Batch(["a"], {"raw_prompt": np.array([[{"role": "user", "content": "task"}]], dtype=object)})
            trainer.async_rollout_manager.generate_sequences.return_value = Batch(
                list(range(4)), {"bfcl_outcome": np.array([{"valid": False, "key": None}] * 4, dtype=object)})
            trainer.async_rollout_manager.generate_sequences.side_effect = None
            with patch.dict(sys.modules, {bfcl.__name__: bfcl}):
                trainer._validate_ttrl_config()
                trainer._generate_training_rollouts(batch, prompts)
            prepared = prompts.non_tensor_batch["bfcl_context"][0]
            self.assertEqual(prepared["consensus_mode"], mode)
            self.assertNotIn("ground_truth", prepared)
            self.assertEqual(batch.non_tensor_batch["extra_info"][0]["bfcl"], prepared)
            if mode == "state":
                self.assertEqual(prepared["observable_state"], {"MathAPI": ["history"]})

    def test_bfcl_task_type_must_match_the_configured_agent_mode(self):
        bfcl = ModuleType("verl.utils.reward_score.ttrl_bfcl")
        bfcl.decode_bfcl_context = lambda context: context
        trainer = self.make_trainer(asynchronous=True)
        trainer.config.ttrl["reward_mode"] = "bfcl"
        batch = Batch(["a"], {"extra_info": np.array([{"bfcl": {"involved_classes": ["MathAPI"]}}], dtype=object)})
        prompts = Batch(["a"], {"raw_prompt": np.array([[{"role": "user", "content": "task"}]], dtype=object)})
        with patch.dict(sys.modules, {bfcl.__name__: bfcl}), self.assertRaisesRegex(ValueError, "task type"):
            trainer._generate_training_rollouts(batch, prompts)
        trainer.async_rollout_manager.generate_sequences.assert_not_called()

    def test_q_weighting_rejects_kl_in_reward_and_unshared_groups(self):
        utils = ModuleType(self.utils.__name__)
        utils.validate_q_weight_config = Mock()
        for shared, kl, message in ((False, False, "shared GRPO"), (True, True, "use_kl_in_reward=False")):
            with self.subTest(shared=shared, kl=kl), patch.dict(sys.modules, {utils.__name__: utils}):
                trainer = self.make_trainer(shared=shared)
                trainer.config.ttrl["q_weight"] = {"enabled": True}
                trainer.config.algorithm["use_kl_in_reward"] = kl
                with self.assertRaisesRegex(ValueError, message):
                    trainer._validate_ttrl_config()

    def test_unknown_reward_mode_fails_before_generation(self):
        trainer = self.make_trainer()
        trainer.config.ttrl["reward_mode"] = "unknown"
        with self.assertRaisesRegex(ValueError, "Unsupported ttrl.reward_mode"):
            trainer._validate_ttrl_config()
        trainer.actor_rollout_wg.generate_sequences.assert_not_called()


if __name__ == "__main__":
    unittest.main()
