"""Run the trainer's real validation methods without importing Ray or GPU workers."""

import ast
import asyncio
import builtins
import importlib.util
import os
import shlex
import sys
import tempfile
import time
import unittest
from contextlib import asynccontextmanager
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
            "_validate_config", "_validate_ttrl_config", "_validate_post_train_eval_config", "serve_after_training"
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


if __name__ == "__main__":
    unittest.main()
