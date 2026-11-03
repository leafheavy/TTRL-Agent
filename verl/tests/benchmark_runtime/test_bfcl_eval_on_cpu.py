"""Verify no-update configuration parity without loading Ray or a GPU engine."""

import argparse
import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("bfcl_eval_control", ROOT / "verl/utils/bfcl_eval.py")
control = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(control)


class SamplingParams:
    n = 1
    temperature = 1.0
    top_p = 1.0
    top_k = -1
    repetition_penalty = 1.0
    stop_token_ids = None


def config():
    return {
        "data": {"max_prompt_length": 8192, "max_response_length": 4096},
        "actor_rollout_ref": {
            "model": {"path": "/original/model with spaces", "trust_remote_code": False},
            "rollout": {
                "name": "vllm", "mode": "async", "free_cache_engine": True,
                "tensor_model_parallel_size": 2, "dtype": "bfloat16", "enforce_eager": False,
                "gpu_memory_utilization": 0.4, "max_model_len": 32768,
                "max_num_batched_tokens": 12288, "enable_chunked_prefill": True,
                "disable_log_stats": True, "prompt_length": "${data.max_prompt_length}",
                "response_length": "${data.max_response_length}", "n": 8, "temperature": 0.6,
                "top_p": 0.97, "top_k": 40, "repetition_penalty": 1.07, "seed": 123,
            },
        },
        "trainer": {
            "n_gpus_per_node": 2, "nnodes": 1,
            "post_train_eval": {"enabled": True, "served_model_name": "Qwen/test-model"},
        },
    }


class TestBaselineConfiguration(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="bfcl eval ")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.reference = self.root / "training run"
        self.source_config = self.reference / "hydra/.hydra/config.yaml"
        self.source_config.parent.mkdir(parents=True)
        self.source_config.write_text(json.dumps(config()), encoding="utf-8")
        self.case_source = self.reference / "data/case_ids.json"
        self.case_source.parent.mkdir()
        self.cases = {"multi_turn_base": ["multi_turn_base_0", "multi_turn_base_9"]}
        self.case_source.write_text(json.dumps(self.cases), encoding="utf-8")
        (self.reference / "run.env").write_text("export BFCL_MODEL=Qwen/test-model-FC\n", encoding="utf-8")
        self.args = argparse.Namespace(
            reference_run=str(self.reference), output_dir=str(self.root / "baseline"),
            bfcl_model="Qwen/test-model-FC", temperature=0.001, num_threads=1,
            host="127.0.0.1", port=1053,
        )

    def prepare(self):
        return control.prepare_baseline(self.args, SamplingParams)

    def test_saved_config_overrides_and_exact_cases_are_reused(self):
        output = Path(self.args.output_dir)
        output.mkdir()  # mktemp creates the empty directory before B1 starts.
        settings = self.prepare()
        self.assertEqual(settings["tensor_parallel_size"], 2)
        self.assertEqual(settings["max_num_batched_tokens"], 12288)
        self.assertEqual(settings["max_model_len"], 32768)
        self.assertEqual(settings["generation_config"], "auto")
        self.assertTrue(settings["disable_custom_all_reduce"])
        self.assertTrue(settings["enable_prefix_caching"])
        self.assertEqual(settings["override_generation_config"], {
            "n": 8, "logprobs": 0, "repetition_penalty": 1.07,
            "max_new_tokens": 4096, "temperature": 0.6, "top_p": 0.97, "top_k": 40,
        })
        self.assertEqual(settings["model"], "/original/model with spaces")
        self.assertIn(settings["model"], settings["served_model_name"])
        self.assertEqual((output / "test_case_ids_to_generate.json").read_bytes(), self.case_source.read_bytes())
        self.assertEqual((output / "reference_config.yaml").read_bytes(), self.source_config.read_bytes())
        manifest = json.loads((output / "baseline_config.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["reference_check"], "unverified_legacy_reference")
        self.assertEqual(manifest["case_ids_sha256"], control._sha256(self.case_source))
        environment = (output / "baseline.env").read_text(encoding="utf-8")
        self.assertIn("export MODEL_PATH='/original/model with spaces'", environment)
        self.assertIn("export BFCL_EVAL_TEMPERATURE=0.001", environment)
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_recorded_engine_settings_and_versions_must_match(self):
        _, loaded = control._load_reference_config(self.reference)
        record = {
            "model_path": loaded["actor_rollout_ref"]["model"]["path"],
            "engine_settings": control.get_vllm_eval_settings(loaded, SamplingParams),
            "versions": control.get_vllm_runtime_metadata(),
        }
        path = self.reference / "post_train_eval/vllm_eval_config_0.json"
        path.parent.mkdir()
        path.write_text(json.dumps(record), encoding="utf-8")
        self.prepare()
        manifest = json.loads((Path(self.args.output_dir) / "baseline_config.json").read_text())
        self.assertEqual(manifest["reference_check"], "matched")
        self.args.output_dir = str(self.root / "other baseline")
        record["engine_settings"]["tensor_parallel_size"] = 1
        path.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "settings/version differ"):
            self.prepare()

    def test_heldout_case_override_reuses_training_config_and_records_exact_source(self):
        heldout = self.reference / "data/heldout_case_ids.json"
        heldout.write_text(json.dumps({"multi_turn_base": ["multi_turn_base_1"]}), encoding="utf-8")
        self.args.case_ids = str(heldout)
        settings = self.prepare()
        output = Path(self.args.output_dir)
        self.assertEqual(settings["model"], config()["actor_rollout_ref"]["model"]["path"])
        self.assertEqual((output / "test_case_ids_to_generate.json").read_bytes(), heldout.read_bytes())
        manifest = json.loads((output / "baseline_config.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["case_ids_source"], str(heldout.resolve()))
        self.assertEqual(manifest["case_ids_sha256"], control._sha256(heldout))
        self.assertEqual(json.loads(self.case_source.read_text(encoding="utf-8")), self.cases)

    def test_invalid_case_override_fails_before_creating_output(self):
        self.args.case_ids = str(self.root / "missing.json")
        with self.assertRaises(FileNotFoundError):
            self.prepare()
        self.assertFalse(Path(self.args.output_dir).exists())

    def test_wrong_handler_or_duplicate_case_ids_are_rejected(self):
        self.args.bfcl_model = "other-handler-FC"
        with self.assertRaisesRegex(ValueError, "handler differs"):
            self.prepare()
        self.args.bfcl_model = "Qwen/test-model-FC"
        self.case_source.write_text(json.dumps({"multi_turn_base": ["multi_turn_base_0"] * 2}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate case IDs"):
            self.prepare()

    def test_nonreference_parent_and_multiple_replicas_are_rejected(self):
        with self.assertRaisesRegex(FileNotFoundError, "concrete training run/category"):
            control._load_reference_config(self.root)
        source = config()
        source["trainer"]["n_gpus_per_node"] = 4
        self.source_config.write_text(json.dumps(source), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "one inference replica"):
            self.prepare()

    def test_reference_client_settings_must_match_before_output_is_created(self):
        record = self.reference / "eval/bfcl_environment.json"
        record.parent.mkdir()
        record.write_text(json.dumps({"client": {
            "bfcl_model": self.args.bfcl_model, "temperature": 0.3, "num_threads": 1,
        }}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "client settings differ"):
            self.prepare()
        self.assertFalse(Path(self.args.output_dir).exists())


class TestBFCLRuntimeParity(unittest.TestCase):
    def test_baseline_cli_matches_training_backend_environment(self):
        vllm = ModuleType("vllm")
        vllm.SamplingParams = SamplingParams
        settings = {"model": "original-model"}
        with (
            patch.dict(sys.modules, {"vllm": vllm}),
            patch.dict(os.environ, {"VLLM_USE_V1": "0", "VLLM_ATTENTION_BACKEND": "other"}),
            patch.object(control, "prepare_baseline", return_value=settings) as prepare,
            patch.object(control, "run_server") as serve,
        ):
            control.main(["--reference-run", "reference", "--output-dir", "baseline", "--bfcl-model", "test-FC"])
            self.assertEqual(os.environ["VLLM_USE_V1"], "1")
            self.assertNotIn("VLLM_ATTENTION_BACKEND", os.environ)
        self.assertIs(prepare.call_args.args[1], SamplingParams)
        serve.assert_called_once_with(settings)

    def test_fingerprint_matches_and_changed_code_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="bfcl package ") as temporary:
            root = Path(temporary)
            package = root / "bfcl_eval"
            package.mkdir()
            init = package / "__init__.py"
            init.write_text("# original handler package\n", encoding="utf-8")
            data = package / "data/tasks.json"
            data.parent.mkdir()
            data.write_text('{"id": "same_id", "question": "original"}', encoding="utf-8")
            module = ModuleType("bfcl_eval")
            module.__file__ = str(init)
            original = root / "updated.json"
            baseline = root / "baseline.json"
            with patch.dict(sys.modules, {"bfcl_eval": module}), patch.object(control, "version", return_value="1.0"):
                control.record_bfcl_environment(original)
                record = control.record_bfcl_environment(baseline, original)
                self.assertEqual(record["reference_check"], "matched")
                data.write_text('{"id": "same_id", "question": "changed"}', encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "version/code/data/client differs"):
                    control.record_bfcl_environment(baseline, original)
                control.record_bfcl_environment(original)
                init.write_text("# different handler\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "version/code/data/client differs"):
                    control.record_bfcl_environment(baseline, original)
                control.record_bfcl_environment(original)
                with self.assertRaisesRegex(ValueError, "version/code/data/client differs"):
                    control.record_bfcl_environment(baseline, original, temperature=0.3)
                record = control.record_bfcl_environment(baseline, root / "legacy-missing.json")
                self.assertEqual(record["reference_check"], "unverified_legacy_reference")

    def test_server_dispatch_uses_official_parser_validation_and_lifecycle(self):
        modules = {name: ModuleType(name) for name in (
            "uvloop", "vllm", "vllm.entrypoints", "vllm.entrypoints.openai",
            "vllm.entrypoints.openai.api_server", "vllm.entrypoints.openai.cli_args",
            "vllm.entrypoints.utils", "vllm.utils",
        )}
        received = []

        async def serve(args):
            received.append(args)

        modules["uvloop"].run = asyncio.run
        modules["vllm.entrypoints.openai.api_server"].run_server = serve
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", default="incorrect-default")
        modules["vllm.entrypoints.openai.cli_args"].make_arg_parser = Mock(return_value=parser)
        validate = Mock()
        modules["vllm.entrypoints.openai.cli_args"].validate_parsed_serve_args = validate
        setup = Mock()
        modules["vllm.entrypoints.utils"].cli_env_setup = setup
        modules["vllm.utils"].FlexibleArgumentParser = argparse.ArgumentParser
        settings = {"model": "original-model", "tensor_parallel_size": 2, "override_generation_config": {"top_k": 40}}
        with patch.dict(sys.modules, modules):
            control.run_server(settings)
        setup.assert_called_once_with()
        validate.assert_called_once_with(received[0])
        self.assertEqual(vars(received[0]), settings)


if __name__ == "__main__":
    unittest.main()
