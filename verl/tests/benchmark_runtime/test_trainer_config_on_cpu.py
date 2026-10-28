"""Run the trainer's real validation methods without importing Ray or GPU workers."""

import ast
import asyncio
import builtins
import importlib.util
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from contextlib import asynccontextmanager
from itertools import chain
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

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
        node for node in trainer.body
        if getattr(node, "name", None) in (
            "_validate_config", "_validate_ttrl_config", "_validate_post_train_eval_config", "serve_after_training",
            "_validate_epoch_failure_schedule", "_finish_failure_epoch", "_training_epoch_iterator",
            "_save_checkpoint", "_load_checkpoint",
        )
    ]
    validation_class = ast.ClassDef(name="TrainerValidation", bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[validation_class], type_ignores=[]))
    namespace = {
        "AdvantageEstimator": SimpleNamespace(GRPO="grpo"),
        "Path": Path,
        "shlex": shlex,
        "tempfile": tempfile,
        "time": time,
        "os": os,
        "json": json,
        "chain": chain,
    }
    exec(compile(module, str(path), "exec"), namespace)

    path = ROOT / "verl/utils/reward_score/ttrl_bfcl.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    validator = next(node for node in tree.body if getattr(node, "name", None) == "validate_bfcl_supervised_config")
    reward_module = ModuleType("verl.utils.reward_score.ttrl_bfcl")
    exec(compile(ast.Module(body=[validator], type_ignores=[]), str(path), "exec"), reward_module.__dict__)
    path = ROOT / "verl/trainer/ppo/ttrl_utils.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    weight_validators = [
        node for node in tree.body
        if getattr(node, "name", None) in ("validate_failure_weight_config", "validate_q_weight_config")
        or isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_FAILURE_TYPE_WEIGHTS" for target in node.targets
        )
    ]
    weight_module = ModuleType("verl.trainer.ppo.ttrl_utils")
    weight_module.np = np
    exec(compile(ast.Module(body=weight_validators, type_ignores=[]), str(path), "exec"), weight_module.__dict__)
    return namespace["TrainerValidation"], reward_module, weight_module


class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def load_vllm_serving():
    path = ROOT / "verl/workers/rollout/vllm_rollout/vllm_async_server.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    server = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AsyncvLLMServer")
    method_names = {"__init__", "init_engine", "chat_completion", "completion", "list_models", "_completion_response"}
    serving_class = ast.ClassDef(
        name="Serving", bases=[ast.Name(id="AsyncServerBase", ctx=ast.Load())], keywords=[],
        body=[method for method in server.body if getattr(method, "name", None) in method_names],
        decorator_list=[],
    )

    class Protocol:
        def __init__(self, **fields):
            self.__dict__.update(fields)

        def model_dump(self):
            return vars(self)

    namespace = dict(
        os=os, DictConfig=Config, AsyncServerBase=object, Request=object,
        SamplingParams=lambda: SimpleNamespace(), AsyncEngineArgs=Mock(),
        AsyncLLM=SimpleNamespace(from_vllm_config=Mock(return_value=SimpleNamespace(model_config=object()))),
        copy_to_local=lambda path: path, BaseModelPath=SimpleNamespace, OpenAIServingModels=Mock(),
        OpenAIServingChat=Mock(), OpenAIServingCompletion=Mock(), RequestLogger=Mock(),
        ExternalZeroMQDistributedExecutor=object(), ExternalRayDistributedExecutor=object(),
        JSONResponse=lambda content, status_code=200: SimpleNamespace(content=content, status_code=status_code),
        StreamingResponse=lambda content, media_type: SimpleNamespace(content=content, media_type=media_type),
    )
    protocol_names = (
        "ChatCompletionRequest", "ChatCompletionResponse", "CompletionRequest", "CompletionResponse", "ErrorResponse"
    )
    for name in protocol_names:
        namespace[name] = type(name, (Protocol,), {})
    module = ast.fix_missing_locations(ast.Module(body=[serving_class], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["Serving"], namespace


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
            model=Config(use_remove_padding=True, path="organization/model"),
            rollout=Config(
                name="vllm",
                mode="async",
                free_cache_engine=True,
                agent=Config(num_workers=8),
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
        trainer_type, reward_module, weight_module = load_validation()
        modules = {
            name: ModuleType(name)
            for name in ("verl", "verl.benchmark_runtime", "verl.utils", "verl.utils.reward_score")
        }
        modules["verl.benchmark_runtime.client"] = load_module("config_test_client", "verl/benchmark_runtime/client.py")
        modules["verl.utils.reward_score.ttrl_consensus"] = load_module(
            "config_test_consensus", "verl/utils/reward_score/ttrl_consensus.py"
        )
        modules["verl.utils.reward_score.ttrl_bfcl"] = reward_module
        modules["verl.trainer.ppo.ttrl_utils"] = weight_module
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

    def test_post_training_serving_guard_runs_before_model_initialization(self):
        self.config.trainer.post_train_eval = Config(enabled="true")
        with self.assertRaisesRegex(ValueError, "post_train_eval.enabled.*boolean"):
            self.trainer._validate_config()

    def test_failure_mode_detection_requires_an_explicit_boolean(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                self.config.failure_mode_detection = Config(enabled=enabled)
                self.trainer._validate_config()
        for enabled in (1, 0, "true", "false", None):
            with self.subTest(enabled=enabled):
                self.config.failure_mode_detection.enabled = enabled
                with self.assertRaisesRegex(ValueError, "failure_mode_detection.enabled.*bool"):
                    self.trainer._validate_config()

    def test_failure_advantage_weighting_is_validated_before_model_workers(self):
        self.config.failure_mode_detection = Config(
            enabled=True, advantage_weight=Config(enabled=True),
        )
        self.trainer._validate_config()
        self.config.failure_mode_detection.advantage_weight.min_weight = 0
        with self.assertRaisesRegex(ValueError, "min_weight.*positive"):
            self.trainer._validate_config()
        self.config.failure_mode_detection.advantage_weight.min_weight = 0.5
        self.config.failure_mode_detection.advantage_weight.enabled = "true"
        with self.assertRaisesRegex(ValueError, "advantage_weight.enabled.*boolean"):
            self.trainer._validate_config()

    def test_enabled_failure_weights_require_label_free_shared_bfcl_agent_grpo(self):
        invalids = (
            (("failure_mode_detection", "enabled"), False),
            (("ttrl", "enable"), False),
            (("ttrl", "reward_mode"), "math"),
            (("ttrl", "share_rollouts"), False),
            (("bfcl_supervised", "enabled"), True),
            (("algorithm", "adv_estimator"), "gae"),
            (("algorithm", "use_kl_in_reward"), True),
            (("actor_rollout_ref", "rollout", "multi_turn", "enable"), False),
            (("actor_rollout_ref", "rollout", "mode"), "sync"),
        )
        for path, value in invalids:
            with self.subTest(path=path, value=value):
                config = smoke_config()
                config.failure_mode_detection = Config(enabled=True, advantage_weight=Config(enabled=True))
                target = config
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                self.trainer.config = config
                with self.assertRaisesRegex(ValueError, "Failure advantage weighting requires"):
                    self.trainer._validate_config()

    def test_disabled_failure_weights_preserve_the_native_agent_setting(self):
        self.config.failure_mode_detection = Config(
            enabled=False, advantage_weight=Config(enabled=False, min_weight="unused"),
        )
        self.config.ttrl.enable = False
        self.config.benchmark_runtime.endpoint = None
        self.config.actor_rollout_ref.rollout.multi_turn.tool_config_path = "native_tools.yaml"
        self.trainer._validate_config()

    def test_epoch_frequency_requires_enabled_weighting_and_multiple_epochs(self):
        self.config.failure_mode_detection = Config(enabled=True, advantage_weight=Config(
            enabled=False, epoch_frequency=Config(enabled=True),
        ))
        with self.assertRaisesRegex(ValueError, "epoch_frequency requires"):
            self.trainer._validate_config()
        self.config.failure_mode_detection.advantage_weight.enabled = True
        for epochs in (1, 0, True, "2"):
            with self.subTest(epochs=epochs):
                self.config.trainer.total_epochs = epochs
                with self.assertRaisesRegex(ValueError, "total_epochs >= 2"):
                    self.trainer._validate_config()
        self.config.trainer.total_epochs = 2
        self.trainer._validate_config()

    def test_epoch_frequency_rejects_dropped_cases_and_incomplete_training(self):
        self.config.failure_mode_detection = Config(enabled=True, advantage_weight=Config(
            enabled=True, epoch_frequency=Config(enabled=True),
        ))
        self.config.trainer.total_epochs = 3
        self.trainer.train_dataset = list(range(8))

        class Loader:
            sampler = list(range(8))

            def __len__(self):
                return 4

        self.trainer.train_dataloader = Loader()
        for steps in (None, 8, 12):
            self.config.trainer.total_training_steps = steps
            self.trainer._validate_epoch_failure_schedule()
        for steps in (4, 5, 9, 16, True, "8"):
            with self.subTest(steps=steps):
                self.config.trainer.total_training_steps = steps
                with self.assertRaisesRegex(ValueError, "two complete available epochs"):
                    self.trainer._validate_epoch_failure_schedule()
        self.config.trainer.total_training_steps = None
        self.trainer.train_dataset.append(8)
        with self.assertRaisesRegex(ValueError, "every case once"):
            self.trainer._validate_epoch_failure_schedule()
        self.trainer.train_dataset.pop()
        self.trainer.train_dataloader.sampler = [0, 1]
        with self.assertRaisesRegex(ValueError, "every case once"):
            self.trainer._validate_epoch_failure_schedule()
        class ReplacementSampler(list):
            replacement = True

        self.trainer.train_dataloader.sampler = ReplacementSampler(range(8))
        with self.assertRaisesRegex(ValueError, "every case once"):
            self.trainer._validate_epoch_failure_schedule()

    def test_static_bfcl_training_also_rejects_silently_dropped_cases(self):
        self.trainer.train_dataset = list(range(9))

        class Loader:
            sampler = list(range(9))

            def __len__(self):
                return 4

        self.trainer.train_dataloader = Loader()
        self.assertFalse(self.config.get("failure_mode_detection", {}).get("advantage_weight", {}).get(
            "epoch_frequency", {}
        ).get("enabled", False))
        with self.assertRaisesRegex(ValueError, "9 cases, batch_size=2"):
            self.trainer._validate_epoch_failure_schedule()
        self.config.data.train_batch_size = 3
        self.trainer._validate_epoch_failure_schedule()

    def test_async_agent_worker_count_is_validated_before_model_initialization(self):
        for count in (0, -1, True, 1.5, "8", None):
            with self.subTest(count=count):
                self.config.actor_rollout_ref.rollout.agent.num_workers = count
                with self.assertRaisesRegex(ValueError, "agent.num_workers must be a positive integer"):
                    self.trainer._validate_config()
        for count in (1, 2, 8):
            with self.subTest(count=count):
                self.config.actor_rollout_ref.rollout.agent.num_workers = count
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


class TestEpochFailureLifecycle(unittest.TestCase):
    def setUp(self):
        trainer_type, _, _ = load_validation()
        self.trainer = trainer_type()
        self.trainer.config = smoke_config()
        self.trainer.config.trainer.update(
            total_epochs=2, resume_mode="resume_path", default_hdfs_dir=None, del_local_ckpt_after_load=False,
        )
        self.trainer.use_critic = False
        self.trainer.actor_rollout_wg = SimpleNamespace(save_checkpoint=Mock(), load_checkpoint=Mock())
        self.trainer._failure_epoch_batches = 0
        self.trainer._failure_resume_boundary = False
        self.trainer.epoch_failure_weights = SimpleNamespace(
            enabled=True, completed_epochs=0, total_rollouts=8,
            finish_epoch=Mock(return_value={"failure_epoch_completed": 1}),
            state_dict=Mock(return_value={"completed_epochs": 0, "total_rollouts": 4}),
        )

        def load_failure_state(state):
            self.trainer.epoch_failure_weights.completed_epochs = state["completed_epochs"]

        self.trainer.epoch_failure_weights.load_state_dict = Mock(side_effect=load_failure_state)

    def test_epoch_finishes_only_after_last_batch_and_exhausts_before_saving(self):
        self.trainer.train_dataset = list(range(4))
        self.trainer.train_dataloader = [1, 2]
        iterator = iter(self.trainer.train_dataloader)
        next(iterator)
        self.assertEqual(self.trainer._finish_failure_epoch(iterator), {})
        self.trainer.epoch_failure_weights.finish_epoch.assert_not_called()
        next(iterator)
        self.assertEqual(self.trainer._finish_failure_epoch(iterator), {"train/failure_epoch_completed": 1})
        self.trainer.epoch_failure_weights.finish_epoch.assert_called_once_with()
        self.assertEqual(self.trainer._failure_epoch_batches, 0)
        self.assertEqual(list(iterator), [])

    def test_resume_boundary_retries_one_empty_iterator_without_losing_first_batch(self):
        loader = Mock()
        loader.__iter__ = Mock(side_effect=[iter(()), iter(("first", "second"))])
        self.trainer.train_dataloader = loader
        self.trainer._failure_resume_boundary = True
        self.assertEqual(list(self.trainer._training_epoch_iterator()), ["first", "second"])
        self.assertFalse(self.trainer._failure_resume_boundary)
        self.assertEqual(loader.__iter__.call_count, 2)

    def test_resume_boundary_does_not_retry_a_loader_that_already_starts_next_epoch(self):
        self.trainer.train_dataloader = ["first", "second"]
        self.trainer._failure_resume_boundary = True
        self.assertEqual(list(self.trainer._training_epoch_iterator()), ["first", "second"])

    def test_resume_partial_epoch_does_not_restart_or_drop_remaining_batches(self):
        self.trainer.train_dataloader = ["remaining"]
        self.trainer._failure_epoch_batches = 1
        self.assertEqual(list(self.trainer._training_epoch_iterator()), ["remaining"])

    def test_incomplete_rollout_budget_is_not_committed_as_a_full_epoch(self):
        self.trainer.train_dataloader = ["last"]
        self.trainer.train_dataset = list(range(4))
        self.trainer.epoch_failure_weights.total_rollouts = 7
        iterator = iter(self.trainer.train_dataloader)
        next(iterator)
        with self.assertRaisesRegex(RuntimeError, "rollout counts do not cover every case"):
            self.trainer._finish_failure_epoch(iterator)
        self.trainer.epoch_failure_weights.finish_epoch.assert_not_called()

    def test_missing_dynamic_checkpoint_state_is_rejected_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "global_step_1"
            checkpoint.mkdir()
            self.trainer.config.trainer.update(default_local_dir=directory, resume_from_path=str(checkpoint))
            namespace = self.trainer._load_checkpoint.__globals__
            with patch.dict(namespace, find_latest_ckpt_path=lambda _: None):
                with self.assertRaisesRegex(ValueError, "requires both its saved state and data.pt"):
                    self.trainer._load_checkpoint()
        self.trainer.actor_rollout_wg.load_checkpoint.assert_not_called()

    def test_checkpoint_round_trip_retains_partial_counts_and_restores_epoch_position(self):
        class Loader:
            def __init__(self):
                self.state = {"position": 1}

            def __len__(self):
                return 2

            def state_dict(self):
                return self.state

            def load_state_dict(self, state):
                self.state = state

        loader = Loader()
        self.trainer.train_dataloader = loader
        self.trainer.global_steps = 1
        self.trainer._failure_epoch_batches = 1
        namespace = self.trainer._save_checkpoint.__globals__
        filesystem = ModuleType("verl.utils.fs")
        filesystem.local_mkdir_safe = lambda path: Path(path).mkdir(parents=True, exist_ok=True)
        fake_torch = SimpleNamespace(
            save=lambda state, path: Path(path).write_text(json.dumps(state), encoding="utf-8"),
            load=lambda path, weights_only: json.loads(Path(path).read_text(encoding="utf-8")),
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "global_step_1"
            self.trainer.config.trainer.update(default_local_dir=directory, resume_from_path=str(checkpoint))
            with patch.dict(sys.modules, {"verl.utils.fs": filesystem}), patch.dict(
                namespace, torch=fake_torch, find_latest_ckpt_path=lambda _: None,
            ):
                self.trainer._save_checkpoint()
                self.trainer._failure_epoch_batches = 0
                loader.state = {}
                self.trainer._load_checkpoint()
            self.assertEqual(loader.state, {"position": 1})
            self.assertEqual(self.trainer._failure_epoch_batches, 1)
            self.assertFalse(self.trainer._failure_resume_boundary)
            self.trainer.epoch_failure_weights.load_state_dict.assert_called_once_with(
                {"completed_epochs": 0, "total_rollouts": 4}
            )
            state_path = checkpoint / "epoch_failure_weights.json"
            saved = json.loads(state_path.read_text(encoding="utf-8"))
            saved["rollouts_per_batch"] = 99
            state_path.write_text(json.dumps(saved), encoding="utf-8")
            with patch.dict(namespace, find_latest_ckpt_path=lambda _: None):
                with self.assertRaisesRegex(ValueError, "original data and rollout schedule"):
                    self.trainer._load_checkpoint()

    def test_fit_observes_only_new_training_rollouts_and_finishes_before_checkpoint(self):
        path = ROOT / "verl/trainer/ppo/ray_trainer.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RayPPOTrainer")
        fit = next(node for node in trainer.body if getattr(node, "name", None) == "fit")
        calls = [node for node in ast.walk(fit) if isinstance(node, ast.Call)]
        observes = [node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "observe"]
        self.assertEqual(len(observes), 1)
        by_name = {node.func.attr: node.lineno for node in calls if isinstance(node.func, ast.Attribute)}
        self.assertLess(by_name["_generate_training_rollouts"], observes[0].lineno)
        self.assertLess(observes[0].lineno, by_name["update_actor"])
        self.assertLess(by_name["_finish_failure_epoch"], by_name["_save_checkpoint"])
        self.assertLess(by_name["_finish_failure_epoch"], by_name["log"])


class TestInMemoryOpenAIServing(unittest.TestCase):
    def test_aliases_and_handlers_share_the_existing_engine(self):
        model_path = "/root/models/Qwen/Qwen2.5-3B-Instruct"
        model_name = "Qwen/Qwen2.5-3B-Instruct"
        cases = (
            (False, "updated", None), (True, "updated", None), (True, None, None), (True, model_name, "native.yaml")
        )
        for enabled, alias, tool_config in cases:
            with self.subTest(enabled=enabled, alias=alias, tool_config=tool_config):
                server_type, namespace = load_vllm_serving()
                rollout = Config(
                    max_model_len=32768, prompt_length=8192, response_length=4096, free_cache_engine=True,
                    dtype="bfloat16", enforce_eager=True, gpu_memory_utilization=0.5, disable_log_stats=True,
                    enable_chunked_prefill=True, multi_turn=Config(tool_config_path=tool_config, format="hermes"),
                )
                config = Config(
                    actor_rollout_ref=Config(model=Config(path=model_path), rollout=rollout),
                    trainer=Config(post_train_eval=Config(enabled=enabled, served_model_name=alias)),
                )
                server = server_type(config, 1, 0, "test")
                server._create_engine_config = Mock()
                asyncio.run(server.init_engine())
                paths = namespace["OpenAIServingModels"].call_args.args[2]
                aliases = (alias or model_name, model_name, model_path)
                expected = list(dict.fromkeys(aliases)) if enabled else [model_name]
                self.assertEqual([item.name for item in paths], expected)
                for factory in ("OpenAIServingChat", "OpenAIServingCompletion"):
                    self.assertIs(namespace[factory].call_args.args[0], server.engine)
                    self.assertIs(namespace[factory].call_args.args[2], server.openai_serving_models)
                auto_tools = namespace["OpenAIServingChat"].call_args.kwargs["enable_auto_tools"]
                self.assertEqual(auto_tools, tool_config is not None)

    def test_existing_vllm_handles_prompting_requests_and_model_listing(self):
        server_type, namespace = load_vllm_serving()
        server = object.__new__(server_type)
        completion = AsyncMock(return_value=namespace["CompletionResponse"](choices=[{"text": "ok"}]))
        server.openai_serving_completion = SimpleNamespace(create_completion=completion)
        request = SimpleNamespace(
            json=AsyncMock(return_value={"model": "updated", "prompt": "hello", "max_tokens": 12})
        )
        response = asyncio.run(server.completion(request))
        self.assertEqual(response.content, {"choices": [{"text": "ok"}]})
        self.assertEqual(vars(completion.call_args.args[0]), {"model": "updated", "prompt": "hello", "max_tokens": 12})
        self.assertIs(completion.call_args.args[1], request)
        server.openai_serving_models = SimpleNamespace(
            show_available_models=AsyncMock(return_value=namespace["CompletionResponse"](data=[{"id": "updated"}]))
        )
        self.assertEqual(asyncio.run(server.list_models()).content, {"data": [{"id": "updated"}]})
        error = server._completion_response(namespace["ErrorResponse"](code=400, message="invalid"))
        self.assertEqual(error.status_code, 400)
        stream = object()
        stream_response = server._completion_response(stream)
        self.assertIs(stream_response.content, stream)
        self.assertEqual(stream_response.media_type, "text/event-stream")

    def test_http_routes_expose_both_generation_apis_and_model_listing(self):
        path = ROOT / "verl/workers/rollout/async_server.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        server = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AsyncServerBase")
        method = next(node for node in server.body if getattr(node, "name", None) == "_start_fastapi_server")
        app = SimpleNamespace(router=Mock())
        uvicorn_server = SimpleNamespace(serve=AsyncMock())
        namespace = {
            "asynccontextmanager": asynccontextmanager,
            "fastapi": SimpleNamespace(FastAPI=Mock(return_value=app)),
            "uvicorn": SimpleNamespace(Config=Mock(), Server=Mock(return_value=uvicorn_server)),
            "_get_free_port": lambda: 8123,
        }
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        server = SimpleNamespace(chat_completion=Mock(), completion=Mock(), list_models=Mock())
        asyncio.run(namespace["_start_fastapi_server"](server))
        registered = [
            (call.args[0], call.args[1], call.kwargs["methods"]) for call in app.router.add_api_route.call_args_list
        ]
        self.assertEqual(registered, [
            ("/v1/chat/completions", server.chat_completion, ["POST"]),
            ("/v1/completions", server.completion, ["POST"]),
            ("/v1/models", server.list_models, ["GET"]),
        ])
        uvicorn_server.serve.assert_awaited_once_with()


class TestPostTrainServing(unittest.TestCase):
    def setUp(self):
        trainer_type, _, _ = load_validation()
        self.trainer = trainer_type()
        self.trainer.config = smoke_config()
        self.config = self.trainer.config
        directory = tempfile.TemporaryDirectory(prefix="post train eval ")
        self.addCleanup(directory.cleanup)
        self.output = Path(directory.name) / "connection"
        self.config.trainer.post_train_eval = Config(enabled=True, output_dir=str(self.output))
        self.manager = SimpleNamespace(
            wake_up=Mock(), sleep=Mock(), server_addresses=["127.0.0.1:8123"]
        )
        self.trainer.async_rollout_manager = self.manager

    def test_disabled_serving_has_no_side_effects(self):
        self.config.trainer.post_train_eval.enabled = False
        del self.trainer.async_rollout_manager
        self.trainer._validate_post_train_eval_config()
        self.trainer.serve_after_training()
        self.assertFalse(self.output.exists())

    def test_enabled_requires_an_explicit_boolean(self):
        for enabled in (1, 0, "true", "false", None):
            with self.subTest(enabled=enabled):
                self.config.trainer.post_train_eval.enabled = enabled
                with self.assertRaisesRegex(ValueError, "post_train_eval.enabled.*boolean"):
                    self.trainer._validate_post_train_eval_config()
        self.assertFalse(self.output.exists())

    def test_incompatible_workers_and_validation_only_fail_before_model_loading(self):
        invalids = (
            (("actor_rollout_ref", "rollout", "name"), "hf"),
            (("actor_rollout_ref", "rollout", "mode"), "sync"),
            (("actor_rollout_ref", "rollout", "free_cache_engine"), False),
            (("actor_rollout_ref", "actor", "strategy"), "megatron"),
            (("trainer", "val_only"), True),
        )
        for path, value in invalids:
            with self.subTest(path=path, value=value):
                config = smoke_config()
                config.trainer.post_train_eval = self.config.trainer.post_train_eval
                target = config
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                self.trainer.config = config
                with self.assertRaisesRegex(ValueError, "Post-training serving requires"):
                    self.trainer._validate_post_train_eval_config()
        self.manager.wake_up.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_model_name_and_output_directory_are_validated(self):
        serving = self.config.trainer.post_train_eval
        for name in ("", "   ", 123, True):
            with self.subTest(model_name=name):
                serving.served_model_name = name
                with self.assertRaisesRegex(ValueError, "served_model_name.*nonempty string"):
                    self.trainer._validate_post_train_eval_config()
        serving.served_model_name = None
        for directory in (None, "", "   ", 123, self.output):
            with self.subTest(output_dir=directory):
                serving.output_dir = directory
                with self.assertRaisesRegex(ValueError, "output_dir.*writable directory"):
                    self.trainer._validate_post_train_eval_config()
        self.assertFalse(self.output.exists())

    def test_valid_fsdp_strategies_preflight_the_connection_directory(self):
        for strategy in ("fsdp", "fsdp2"):
            with self.subTest(strategy=strategy):
                self.config.actor_rollout_ref.actor.strategy = strategy
                self.trainer._validate_post_train_eval_config()
                self.assertTrue(self.output.is_dir())
                self.assertEqual(list(self.output.iterdir()), [])
        with patch.object(tempfile, "TemporaryFile", side_effect=PermissionError("read only")):
            with self.assertRaisesRegex(PermissionError, "read only"):
                self.trainer._validate_post_train_eval_config()
        self.manager.wake_up.assert_not_called()

    def test_existing_connection_or_stop_marker_is_preserved_and_rejected(self):
        self.output.mkdir()
        for filename in ("server.env", "stop"):
            with self.subTest(filename=filename):
                marker = self.output / filename
                marker.write_text("existing run", encoding="utf-8")
                for method in (self.trainer._validate_post_train_eval_config, self.trainer.serve_after_training):
                    with self.assertRaisesRegex(FileExistsError, "fresh post-training evaluation directory"):
                        method()
                    self.assertEqual(marker.read_text(encoding="utf-8"), "existing run")
                marker.unlink()
        self.manager.wake_up.assert_not_called()
        self.manager.sleep.assert_not_called()

    def test_final_weights_are_synchronized_before_publishing_and_stop_cleans_up(self):
        self.trainer._validate_post_train_eval_config()
        ready = self.output / "server.env"
        stop = self.output / "stop"
        model_path = "/models/organization/model with 'quotes'"
        self.config.actor_rollout_ref.model.path = model_path
        self.config.trainer.post_train_eval.served_model_name = "final model 'alias'"
        events = []

        def wake_up():
            self.assertFalse(ready.exists())
            events.append("final weights synchronized")

        def wait_for_evaluator(_):
            self.assertEqual(events, ["final weights synchronized"])
            environment = dict(
                shlex.split(line.removeprefix("export "))[0].split("=", 1)
                for line in ready.read_text(encoding="utf-8").splitlines()
            )
            self.assertEqual(environment, {
                "LOCAL_SERVER_ENDPOINT": "127.0.0.1",
                "LOCAL_SERVER_PORT": "8123",
                "REMOTE_OPENAI_BASE_URL": "http://127.0.0.1:8123/v1",
                "REMOTE_OPENAI_API_KEY": "EMPTY",
                "REMOTE_OPENAI_TOKENIZER_PATH": model_path,
                "SERVED_MODEL_NAME": "final model 'alias'",
            })
            self.assertEqual(list(self.output.iterdir()), [ready])
            stop.touch()

        self.manager.wake_up.side_effect = wake_up
        with patch.object(time, "sleep", side_effect=wait_for_evaluator):
            self.trainer.serve_after_training()
        self.manager.wake_up.assert_called_once_with()
        self.manager.sleep.assert_called_once_with()
        self.assertEqual(list(self.output.iterdir()), [stop])

    def test_default_model_name_uses_the_existing_model_path(self):
        self.trainer._validate_post_train_eval_config()

        def stop_evaluation(_):
            environment = (self.output / "server.env").read_text(encoding="utf-8")
            self.assertIn("export SERVED_MODEL_NAME=organization/model\n", environment)
            (self.output / "stop").touch()

        with patch.object(time, "sleep", side_effect=stop_evaluation):
            self.trainer.serve_after_training()
        self.manager.sleep.assert_called_once_with()

    def test_sync_failure_still_returns_workers_to_sleep(self):
        self.trainer._validate_post_train_eval_config()
        self.manager.wake_up.side_effect = RuntimeError("sync failed")
        with self.assertRaisesRegex(RuntimeError, "sync failed"):
            self.trainer.serve_after_training()
        self.manager.sleep.assert_called_once_with()
        self.assertEqual(list(self.output.iterdir()), [])

    def test_publish_failure_cleans_up_temporary_file_and_sleeps(self):
        self.trainer._validate_post_train_eval_config()
        with patch.object(Path, "replace", side_effect=OSError("publish failed")):
            with self.assertRaisesRegex(OSError, "publish failed"):
                self.trainer.serve_after_training()
        self.manager.sleep.assert_called_once_with()
        self.assertEqual(list(self.output.iterdir()), [])

    def test_interrupted_evaluation_removes_connection_file_and_sleeps(self):
        self.trainer._validate_post_train_eval_config()
        with patch.object(time, "sleep", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.trainer.serve_after_training()
        self.manager.sleep.assert_called_once_with()
        self.assertEqual(list(self.output.iterdir()), [])


class TestIndependentMultiTurnLauncher(unittest.TestCase):
    """Exercise the Bash workflow with CPU-only benchmark and trainer processes."""

    categories = (
        "multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context",
    )

    def setUp(self):
        self.bash = "D:/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")
        if not self.bash or not Path(self.bash).is_file():
            self.skipTest("Bash is required for the launcher integration test")
        temporary_parent = ROOT.parent / "tmp" if os.name == "nt" else None
        if temporary_parent is not None:
            temporary_parent.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix="bfcl suite ", dir=temporary_parent)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.suite = self.root / "results with spaces"
        self.events = self.root / "events"
        self.model = self.root / "original model"
        self.model.mkdir()
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.dispatch = self.root / "fake_python.py"
        self.dispatch.write_text(self.fake_python(), encoding="utf-8")
        self.interpreter = self.bin / "fake-python"
        self.interpreter.write_bytes(textwrap.dedent('''\
            #!/usr/bin/env bash
            if [ "${1:-}" = -u ] && [ "${2:-}" = verl/benchmark_runtime/server.py ]; then
              "$REAL_PYTHON" "$FAKE_DISPATCH" runtime_start "$@"
              trap '"$REAL_PYTHON" "$FAKE_DISPATCH" runtime_stop; exit 0' TERM INT
              while true; do sleep 0.1; done
            fi
            exec "$REAL_PYTHON" "$FAKE_DISPATCH" "$@"
        ''').encode("utf-8"))
        self.interpreter.chmod(0o755)
        if os.name == "nt":
            # Git Bash lacks setsid. The mocked trainer also observes an abort
            # marker so Windows can verify cleanup without process-group signals.
            (self.bin / "setsid").write_bytes(b'#!/usr/bin/env bash\nexec "$@"\n')
            (self.bin / "setsid").chmod(0o755)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.runtime_port = sock.getsockname()[1]
        self.env = dict(os.environ, **{
            "PATH": str(self.bin) + os.pathsep + os.environ.get("PATH", ""),
            "REAL_PYTHON": Path(sys.executable).as_posix(),
            "FAKE_DISPATCH": self.dispatch.as_posix(),
            "FAKE_EVENTS": str(self.events),
            "TRAIN_CATEGORIES": "multi_turn",
            "RUN_ROOT": self.bash_path(self.suite),
            "MODEL_PATH": self.model.as_posix(),
            "BFCL_PYTHON": self.bash_path(self.interpreter),
            "TRAIN_PYTHON": self.bash_path(self.interpreter),
            "BFCL_MODEL": "test-model-FC",
            "BENCHMARK_RUNTIME_ENDPOINT": f"http://127.0.0.1:{self.runtime_port}",
            "PYTHONUTF8": "1",
            "MSYS2_ARG_CONV_EXCL": "*",
        })

    @staticmethod
    def bash_path(path):
        value = path.as_posix()
        return "/" + value[0].lower() + value[2:] if os.name == "nt" else value

    @staticmethod
    def fake_python():
        return textwrap.dedent(r'''
            import hashlib
            import json
            import os
            import re
            import runpy
            import shlex
            import sys
            import threading
            import time
            import uuid
            from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
            from pathlib import Path

            def path(value):
                if os.name == "nt" and re.match(r"^/[a-zA-Z]/", value):
                    value = value[1] + ":" + value[2:]
                return Path(value)

            def event(kind, **fields):
                # Separate records avoid concurrent text appends racing on Windows.
                directory = path(os.environ["FAKE_EVENTS"])
                directory.mkdir(exist_ok=True)
                # Python 3.9 on Windows does not share perf_counter's epoch.
                name = f"{time.time_ns():020d}_{os.getpid()}_{uuid.uuid4().hex}.json"
                (directory / name).write_text(json.dumps(dict(kind=kind, **fields)), encoding="utf-8")

            def value(option):
                return args[args.index(option) + 1]

            args = sys.argv[1:]
            project = path(os.environ.get("BFCL_PROJECT_ROOT", os.environ["RUN_ROOT"]))
            category = project.parent.name
            if args[0] in ("runtime_start", "runtime_stop"):
                event(args[0], category=category, arguments=args[1:])
            elif args[0] == "-c":
                if "MODEL_CONFIG_MAPPING" not in args[1]:
                    source = args[1]
                    sys.argv = ["-c", *args[2:]]
                    exec(compile(source, "<launcher expression>", "exec"))
            elif args[0] == "-":
                os.environ["BFCL_PROJECT_ROOT"] = str(project)
                sys.argv = ["-", *args[1:]]
                exec(compile(sys.stdin.read(), "<launcher model probe>", "exec"))
            elif args[0].endswith("benchmark_runtime/client.py"):
                event("health", category=category, arguments=args)
                assert value("--backend") == "bfcl"
                assert int(value("--expect-pid")) > 0
                expected = hashlib.sha256(Path("verl/benchmark_runtime/server.py").read_bytes()).hexdigest()
                assert value("--expect-server-sha256") == expected
                if os.environ.get("FAKE_OCCUPIED"):
                    sys.argv = args
                    runpy.run_path(args[0], run_name="__main__")
            elif args[0] == "data/preprocess.py":
                if "--bfcl-categories" in args:
                    category = value("--bfcl-categories")
                    output = path(value("--output-dir"))
                    output.mkdir(parents=True, exist_ok=True)
                    (output / "train.parquet").touch()
                    (output / "test.parquet").touch()
                    (output / "case_ids.json").write_text(json.dumps({category: [category + "_0"]}))
                    event("preprocess", category=category)
                elif "--bfcl-check-results" in args:
                    assert (project / "result" / "generated.json").is_file()
                    event("check_results", category=category)
                elif "--bfcl-summarize-multi-turn" in args:
                    event("summary", arguments=args)
                    (path(value("--bfcl-summarize-multi-turn")) / "multi_turn_summary.json").write_text("{}")
                else:
                    raise AssertionError(args)
            elif args[:2] == ["-m", "verl.trainer.main_ppo"]:
                overrides = {}
                for argument in args[2:]:
                    if "=" in argument:
                        key, setting = argument.split("=", 1)
                        overrides[key.lstrip("+")] = setting
                category = os.environ["TRAIN_CATEGORIES"]
                event("train", category=category, overrides=overrides,
                      model=os.environ["MODEL_PATH"], eval_categories=os.environ["EVAL_CATEGORIES"])
                if category == os.environ.get("FAKE_FAIL_TRAIN"):
                    event("train_failed", category=category)
                    sys.exit(7)
                output = path(overrides["trainer.post_train_eval.output_dir"])
                output.mkdir(parents=True)
                model = overrides["actor_rollout_ref.model.path"]

                class Handler(BaseHTTPRequestHandler):
                    def log_message(self, *unused):
                        pass

                    def do_GET(self):
                        assert self.path == "/v1/models"
                        body = json.dumps({"data": [{"id": model}]}).encode()
                        self.send_response(200)
                        self.end_headers()
                        self.wfile.write(body)

                    def do_POST(self):
                        assert self.path == "/v1/completions"
                        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                        assert body["model"] == model
                        event("model_probe", category=category)
                        self.send_response(200)
                        self.end_headers()
                        self.wfile.write(b'{"choices": [{"text": "Hello"}]}')

                server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
                worker = threading.Thread(target=server.serve_forever, daemon=True)
                worker.start()
                settings = {
                    "LOCAL_SERVER_ENDPOINT": "127.0.0.1",
                    "REMOTE_OPENAI_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
                    "REMOTE_OPENAI_TOKENIZER_PATH": model,
                }
                connection = "".join(f"export {key}={shlex.quote(setting)}\n" for key, setting in settings.items())
                (output / "server.env").write_text(connection, encoding="utf-8")
                while not (output / "stop").exists() and not (output.parent / "abort_train").exists():
                    time.sleep(0.05)
                server.shutdown()
                server.server_close()
                event("train_stopped", category=category)
            elif args[:3] == ["-m", "bfcl_eval", "generate"]:
                event("generate", category=category, arguments=args)
                if category == os.environ.get("FAKE_FAIL_GENERATE"):
                    (project.parent / "abort_train").touch()
                    sys.exit(9)
                (project / "result").mkdir()
                (project / "result" / "generated.json").write_text("{}")
            elif args[:3] == ["-m", "bfcl_eval", "evaluate"]:
                assert value("--test-category") == category
                event("evaluate", category=category, arguments=args)
                (project / "score").mkdir()
            else:
                raise AssertionError(args)
        ''')

    def launch(self, **extra_env):
        overrides = [
            "ttrl.reward_mode=bfcl", "ttrl.enable=True", "trainer.post_train_eval.enabled=True",
            "actor_rollout_ref.model.path=wrong-model", "trainer.resume_mode=auto",
            "data.train_files=shared-data", "trainer.default_local_dir=shared-checkpoint",
            "actor_rollout_ref.actor.gradient_record.output_dir=shared-gradients",
            "trainer.post_train_eval.output_dir=shared-serving",
            "trainer.rollout_data_dir=shared-rollouts", "trainer.validation_data_dir=shared-validation",
        ]
        result = subprocess.run(
            [self.bash, (ROOT / "examples/ttrl/Qwen2.5/aime.sh").as_posix(), *overrides],
            cwd=ROOT, env=dict(self.env, **extra_env), text=True, encoding="utf-8", errors="replace",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60,
        )
        return result, self.read_events()

    def read_events(self):
        if not self.events.exists():
            return []
        return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(self.events.glob("*.json"))]

    def test_all_categories_use_independent_original_initialization_and_reject_reuse(self):
        result, events = self.launch()
        self.assertEqual(result.returncode, 0, result.stdout)
        trained = [entry for entry in events if entry["kind"] == "train"]
        self.assertEqual([entry["category"] for entry in trained], list(self.categories))
        for entry in trained:
            category = entry["category"]
            overrides = entry["overrides"]
            self.assertEqual(overrides["actor_rollout_ref.model.path"], self.model.as_posix())
            self.assertEqual(overrides["trainer.resume_mode"], "disable")
            self.assertEqual(entry["eval_categories"], category)
            child = self.suite / category
            for key, suffix in (
                ("data.train_files", "data/train.parquet"), ("data.val_files", "data/test.parquet"),
                ("trainer.default_local_dir", "checkpoints"), ("hydra.run.dir", "hydra"),
                ("actor_rollout_ref.actor.gradient_record.output_dir", "gradients"),
                ("trainer.post_train_eval.output_dir", "post_train_eval"),
                ("trainer.rollout_data_dir", "rollouts"), ("trainer.validation_data_dir", "validation"),
            ):
                expected = (child / suffix).resolve()
                actual = overrides[key]
                if os.name == "nt" and re.match(r"^/[a-zA-Z]/", actual):
                    actual = actual[1] + ":" + actual[2:]
                self.assertEqual(Path(actual).resolve(), expected, (category, key))
            for kind in ("preprocess", "health", "model_probe", "generate", "check_results", "evaluate",
                         "train_stopped", "runtime_stop"):
                self.assertEqual(sum(item["kind"] == kind and item["category"] == category for item in events), 1)
            if category != self.categories[-1]:
                stopped = next(i for i, item in enumerate(events)
                               if item["kind"] == "runtime_stop" and item["category"] == category)
                next_category = self.categories[self.categories.index(category) + 1]
                started = next(i for i, item in enumerate(events)
                               if item["kind"] == "preprocess" and item["category"] == next_category)
                self.assertLess(stopped, started)
        self.assertEqual(events[-1]["kind"], "summary")
        self.assertTrue((self.suite / "multi_turn_summary.json").is_file())
        result, repeated = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("fresh RUN_ROOT", result.stdout)
        self.assertEqual(events, repeated)

    def test_training_failure_stops_before_starting_the_next_category(self):
        result, events = self.launch(FAKE_FAIL_TRAIN=self.categories[0])
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual([event["category"] for event in events if event["kind"] == "train"], [self.categories[0]])
        self.assertFalse(any(event["kind"] in ("generate", "summary") for event in events))
        self.assertFalse((self.suite / self.categories[1]).exists())
        self.assertFalse((self.suite / ".multi_turn_suite.lock").exists())

    def test_seven_b_registry_default_follows_original_model_path(self):
        model = self.root / "Qwen2.5-7B-Instruct"
        model.mkdir()
        self.env.pop("BFCL_MODEL")
        result, events = self.launch(MODEL_PATH=model.as_posix(), FAKE_FAIL_TRAIN=self.categories[0])
        self.assertNotEqual(result.returncode, 0, result.stdout)
        trained = [event for event in events if event["kind"] == "train"]
        self.assertEqual(len(trained), 1)
        self.assertEqual(trained[0]["overrides"]["actor_rollout_ref.model.path"], model.as_posix())
        configuration = (self.suite / self.categories[0] / "run.env").read_text(encoding="utf-8")
        self.assertIn("export BFCL_MODEL=Qwen/Qwen2.5-7B-Instruct-FC", configuration)

    def test_evaluation_failure_stops_the_suite_without_publishing_a_summary(self):
        result, events = self.launch(FAKE_FAIL_GENERATE=self.categories[0])
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual([event["category"] for event in events if event["kind"] == "train"], [self.categories[0]])
        self.assertFalse(any(event["kind"] in ("evaluate", "summary") for event in events))
        self.assertFalse((self.suite / self.categories[1]).exists())

    def test_foreign_runtime_identity_failure_leaves_the_unrelated_listener_running(self):
        foreign_code = textwrap.dedent('''
            import json, os, sys
            from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *unused): pass
                def do_POST(self):
                    request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    result = {"version": 1, "id": request["id"], "result": {
                        "backend": "bfcl", "pid": os.getpid(), "server_sha256": "foreign-code"}}
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(json.dumps(result).encode())
            server = ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler)
            print("ready", flush=True)
            server.serve_forever()
        ''')
        foreign = subprocess.Popen(
            [sys.executable, "-u", "-c", foreign_code, str(self.runtime_port)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        try:
            self.assertEqual(foreign.stdout.readline().strip(), "ready")
            result, events = self.launch(FAKE_OCCUPIED="1")
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn("Runtime identity mismatch", result.stdout)
            self.assertIsNone(foreign.poll(), "The launcher terminated an unrelated runtime")
            self.assertFalse(any(event["kind"] in ("train", "summary") for event in events))
        finally:
            foreign.terminate()
            foreign.communicate(timeout=5)


if __name__ == "__main__":
    unittest.main()
