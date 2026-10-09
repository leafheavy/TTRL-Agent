"""Run the trainer's real validation methods without importing Ray or GPU workers."""

import ast
import builtins
import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_validation():
    # Compile the actual method bodies: importing ray_trainer would require the
    # entire training stack, while copying its guards would hide regressions.
    path = ROOT / "verl/trainer/ppo/ray_trainer.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RayPPOTrainer")
    methods = [
        node for node in trainer.body if getattr(node, "name", None) in ("_validate_config", "_validate_ttrl_config")
    ]
    validation_class = ast.ClassDef(name="TrainerValidation", bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[validation_class], type_ignores=[]))
    namespace = {"AdvantageEstimator": SimpleNamespace(GRPO="grpo")}
    exec(compile(module, str(path), "exec"), namespace)

    path = ROOT / "verl/utils/reward_score/ttrl_bfcl.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    validator = next(node for node in tree.body if getattr(node, "name", None) == "validate_bfcl_supervised_config")
    reward_module = ModuleType("verl.utils.reward_score.ttrl_bfcl")
    exec(compile(ast.Module(body=[validator], type_ignores=[]), str(path), "exec"), reward_module.__dict__)
    return namespace["TrainerValidation"], reward_module


class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def smoke_config():
    return Config(
        benchmark_runtime=Config(endpoint="http://127.0.0.1:1054", timeout_s=120),
        bfcl_supervised=Config(enabled=False),
        ttrl=Config(
            enable=True,
            reward_mode="bfcl",
            share_rollouts=True,
            n_samples_per_prompt=2,
            n_votes_per_prompt=2,
            consensus=Config(mode="observations"),
            q_weight=Config(enabled=False),
        ),
        data=Config(train_batch_size=2, return_raw_chat=True),
        actor_rollout_ref=Config(
            actor=Config(
                strategy="fsdp",
                use_dynamic_bsz=False,
                ppo_mini_batch_size=1,
                ppo_micro_batch_size=None,
                ppo_micro_batch_size_per_gpu=1,
                loss_agg_mode="token-mean",
                use_kl_loss=True,
            ),
            ref=Config(log_prob_micro_batch_size=None, log_prob_micro_batch_size_per_gpu=1),
            model=Config(use_remove_padding=True),
            rollout=Config(
                name="vllm",
                mode="async",
                n=2,
                temperature=0.6,
                log_prob_micro_batch_size=None,
                log_prob_micro_batch_size_per_gpu=1,
                val_kwargs=Config(do_sample=True),
                multi_turn=Config(enable=True, tool_config_path=None, interaction_config_path=None),
            ),
        ),
        algorithm=Config(adv_estimator="grpo", use_kl_in_reward=False),
        reward_model=Config(enable=False),
        trainer=Config(n_gpus_per_node=1, nnodes=1, val_before_train=False, test_freq=-1),
    )


class TestTrainerMultiTurnConfig(unittest.TestCase):
    def setUp(self):
        trainer_type, reward_module = load_validation()
        modules = {
            name: ModuleType(name)
            for name in ("verl", "verl.benchmark_runtime", "verl.utils", "verl.utils.reward_score")
        }
        modules["verl.benchmark_runtime.client"] = load_module("config_test_client", "verl/benchmark_runtime/client.py")
        modules["verl.utils.reward_score.ttrl_consensus"] = load_module(
            "config_test_consensus", "verl/utils/reward_score/ttrl_consensus.py"
        )
        modules["verl.utils.reward_score.ttrl_bfcl"] = reward_module
        module_patch = patch.dict(sys.modules, modules)
        module_patch.start()
        self.addCleanup(module_patch.stop)
        original_import = builtins.__import__

        def restricted_import(name, *args, **kwargs):
            if name.split(".")[0] in ("bfcl_eval", "jsonschema"):
                raise AssertionError("Benchmark dependency imported into trainer validation: " + name)
            return original_import(name, *args, **kwargs)

        import_patch = patch("builtins.__import__", side_effect=restricted_import)
        import_patch.start()
        self.addCleanup(import_patch.stop)
        self.trainer = trainer_type()
        self.trainer.config = smoke_config()
        self.trainer.use_critic = False
        self.trainer.use_reference_policy = True
        self.config = self.trainer.config

    def test_smoke_external_tools_need_no_native_yaml(self):
        self.trainer._validate_config()

    def test_supervised_external_tools_need_no_native_yaml(self):
        self.config.ttrl.enable = False
        self.config.bfcl_supervised.enabled = True
        self.trainer._validate_config()

    def test_native_tool_and_interaction_configs_remain_supported(self):
        self.config.ttrl.enable = False
        self.config.benchmark_runtime.endpoint = None
        multi_turn = self.config.actor_rollout_ref.rollout.multi_turn
        for key in ("tool_config_path", "interaction_config_path"):
            with self.subTest(key=key):
                multi_turn[key] = "native_tools.yaml"
                self.trainer._validate_config()
                multi_turn[key] = None

    def test_endpoint_alone_does_not_bypass_native_tool_requirement(self):
        for enabled, reward_mode in ((False, "bfcl"), (True, "math")):
            with self.subTest(enabled=enabled, reward_mode=reward_mode):
                self.config.ttrl.enable = enabled
                self.config.ttrl.reward_mode = reward_mode
                with self.assertRaisesRegex(AssertionError, "tool_config_path or interaction_config_path"):
                    self.trainer._validate_config()

    def test_bfcl_still_requires_valid_runtime_settings(self):
        for runtime in (
            None,
            {},
            {"endpoint": "file:///tmp/runtime"},
            {"endpoint": "http://127.0.0.1:1054", "timeout_s": 0},
        ):
            with self.subTest(runtime=runtime):
                self.config.benchmark_runtime = runtime
                with self.assertRaisesRegex(ValueError, "benchmark_runtime"):
                    self.trainer._validate_config()

    def test_async_mode_and_raw_messages_remain_required(self):
        for supervised in (False, True):
            self.config.ttrl.enable = not supervised
            self.config.bfcl_supervised.enabled = supervised
            for key in ("mode", "return_raw_chat"):
                with self.subTest(supervised=supervised, key=key):
                    target = self.config.actor_rollout_ref.rollout if key == "mode" else self.config.data
                    previous = target[key]
                    target[key] = "sync" if key == "mode" else False
                    with self.assertRaisesRegex(ValueError, "async|return_raw_chat"):
                        self.trainer._validate_config()
                    target[key] = previous

    def test_grpo_guard_remains_for_native_and_bfcl_tools(self):
        self.config.ttrl.enable = False
        self.config.actor_rollout_ref.rollout.multi_turn.tool_config_path = "native_tools.yaml"
        self.config.algorithm.adv_estimator = "gae"
        with self.assertRaisesRegex(AssertionError, "only GRPO"):
            self.trainer._validate_config()
        self.config.ttrl.enable = True
        with self.assertRaisesRegex(ValueError, "adv_estimator=grpo"):
            self.trainer._validate_config()

    def test_single_turn_bfcl_call_consensus_is_unaffected(self):
        self.config.actor_rollout_ref.rollout.multi_turn.enable = False
        self.config.ttrl.consensus.mode = "calls"
        self.trainer._validate_config()


if __name__ == "__main__":
    unittest.main()
