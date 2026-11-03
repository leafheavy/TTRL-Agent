"""Verify no-update configuration parity without loading Ray or a GPU engine."""

import argparse
import asyncio
import importlib.util
import io
import json
import os
import subprocess
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

    def test_flat_reference_fingerprint_is_selected_by_case_ids(self):
        for side, cases, temperature in (("train", self.cases, 0.7), ("heldout", {
            "multi_turn_base": ["multi_turn_base_1"],
        }, 0.001)):
            directory = self.reference / f"eval_{side}"
            directory.mkdir()
            (directory / "test_case_ids_to_generate.json").write_text(json.dumps(cases), encoding="utf-8")
            (directory / "bfcl_environment.json").write_text(json.dumps({"client": dict(
                bfcl_model=self.args.bfcl_model, temperature=temperature, num_threads=1,
            )}), encoding="utf-8")
        self.args.case_ids = str(self.reference / "eval_heldout/test_case_ids_to_generate.json")
        self.prepare()
        manifest = json.loads((Path(self.args.output_dir) / "baseline_config.json").read_text(encoding="utf-8"))
        expected = str(self.reference / "eval_heldout/bfcl_environment.json")
        self.assertEqual(manifest["bfcl_reference_environment"], expected)
        self.assertEqual(manifest["bfcl_reference_check"], "matched")
        environment = (Path(self.args.output_dir) / "baseline.env").read_text(encoding="utf-8")
        self.assertIn("export BASELINE_CASE_OVERRIDE=True", environment)
        self.assertIn("export BASELINE_BFCL_REFERENCE_ENV=", environment)


class TestFlatEvaluation(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="flat bfcl ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "cases.json"
        self.cases = {"multi_turn_base": ["multi_turn_base_0", "multi_turn_base_9"]}
        self.source.write_text(json.dumps(self.cases), encoding="utf-8")
        self.args = argparse.Namespace(
            evaluate_case_ids=str(self.source), output_dir=str(self.root / "eval_train"),
            bfcl_model="Qwen/test-model-FC", local_model_path="original-model",
            temperature=0.001, num_threads=1, compare_bfcl_env=None, score_only=False,
        )
        self.commands = []
        self.run_bfcl = control._run_bfcl
        self.utils = ModuleType("bfcl_eval.utils")
        self.utils.get_directory_structure_by_category = lambda category: "multi_turn"
        self.utils.find_file_by_category = lambda category, directory, **_: control._category_file(
            directory, category, "result"
        )
        self.utils.load_file = lambda path, **_: [json.loads(line) for line in path.read_text().splitlines()]
        self.stack = [
            patch.dict(sys.modules, {"bfcl_eval.utils": self.utils}),
            patch.object(control, "record_bfcl_environment", side_effect=self.fingerprint),
            patch.object(control, "_probe_model"),
            patch.object(control, "_run_bfcl", side_effect=self.run_cli),
        ]
        self.mocks = [item.start() for item in self.stack]
        for item in self.stack:
            self.addCleanup(item.stop)

    @staticmethod
    def fingerprint(output, compare=None, **kwargs):
        result = dict(client=kwargs)
        if compare and Path(compare).is_file():
            if json.loads(Path(compare).read_text(encoding="utf-8")) != result:
                raise ValueError("fingerprint differs")
        Path(output).write_text(json.dumps(result), encoding="utf-8")

    def run_cli(self, command, log, environment):
        self.commands.append(command)
        work = Path(environment["BFCL_PROJECT_ROOT"])
        cases = json.loads((work / "test_case_ids_to_generate.json").read_text())
        log.write_text("official CLI output\n", encoding="utf-8")
        kind = "result" if command[0] == "generate" else "score"
        for category, ids in cases.items():
            destination = work / kind / self.args.bfcl_model.replace("/", "_") / "multi_turn"
            destination.mkdir(parents=True, exist_ok=True)
            rows = [{"id": case_id, "result": []} for case_id in ids] if kind == "result" else [
                {"correct_count": 1, "total_count": len(ids), "accuracy": 1 / len(ids)},
                {"id": ids[-1], "valid": False},
            ]
            (destination / f"BFCL_v4_{category}_{kind}.json").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

    def test_fixed_cases_are_generated_once_scored_and_published_without_subdirectories(self):
        summary = control.evaluate_cases(self.args)
        output = Path(self.args.output_dir)
        self.assertEqual([command[0] for command in self.commands], ["generate", "evaluate"])
        generate = self.commands[0]
        self.assertIn("--run-ids", generate)
        self.assertEqual(generate[generate.index("--local-model-path") + 1], "original-model")
        self.assertEqual(summary["bfcl_model"], self.args.bfcl_model)
        self.assertEqual(summary["average_accuracy"], 0.5)
        self.assertEqual(summary["case_ids_sha256"], control._sha256(self.source))
        self.assertEqual(summary["categories"][0]["total_count"], 2)
        self.assertEqual(list(output.iterdir()), list(output.glob("*")))
        self.assertFalse(any(path.is_dir() for path in output.iterdir()))
        raw = output / "BFCL_v4_multi_turn_base_result.json"
        self.assertEqual(
            [json.loads(line)["id"] for line in raw.read_text().splitlines()], self.cases["multi_turn_base"]
        )
        self.assertTrue((output / "BFCL_v4_multi_turn_base_score.json").is_file())
        self.mocks[2].assert_called_once_with("original-model", output)
        with self.assertRaises(FileExistsError):
            control.evaluate_cases(self.args)

    def test_generation_failure_retains_partial_raw_results_and_never_scores_or_summarizes(self):
        def fail(command, log, environment):
            self.run_cli(command, log, environment)
            raise subprocess.CalledProcessError(9, command)

        self.mocks[3].side_effect = fail
        with self.assertRaises(subprocess.CalledProcessError):
            control.evaluate_cases(self.args)
        output = Path(self.args.output_dir)
        self.assertTrue((output / "generate.log").is_file())
        self.assertTrue((output / "BFCL_v4_multi_turn_base_result.json").is_file())
        self.assertFalse((output / "summary.json").exists())
        self.assertFalse(any(path.is_dir() for path in output.iterdir()))
        self.assertEqual([command[0] for command in self.commands], ["generate"])

    def test_partial_duplicate_or_extra_results_cannot_reach_official_scoring(self):
        for ids in (["multi_turn_base_0"], ["multi_turn_base_0"] * 2, ["multi_turn_base_0", "multi_turn_base_1"]):
            with self.subTest(ids=ids):
                self.args.output_dir = str(self.root / str(len(list(self.root.iterdir()))))

                def incomplete(command, log, environment, selected_ids=ids):
                    self.run_cli(command, log, environment)
                    path = control._category_file(Path(environment["BFCL_PROJECT_ROOT"]) / "result",
                                                  "multi_turn_base", "result")
                    path.write_text(
                        "".join(json.dumps({"id": value}) + "\n" for value in selected_ids), encoding="utf-8"
                    )

                self.mocks[3].side_effect = incomplete
                self.commands.clear()
                with self.assertRaises(ValueError):
                    control.evaluate_cases(self.args)
                self.assertEqual([command[0] for command in self.commands], ["generate"])
                self.assertFalse((Path(self.args.output_dir) / "summary.json").exists())

    def test_bad_official_counts_prevent_publishing_a_success_summary(self):
        def wrong_count(command, log, environment):
            self.run_cli(command, log, environment)
            if command[0] == "evaluate":
                path = control._category_file(Path(environment["BFCL_PROJECT_ROOT"]) / "score",
                                              "multi_turn_base", "score")
                path.write_text('{"correct_count": 1, "total_count": 1}\n', encoding="utf-8")

        self.mocks[3].side_effect = wrong_count
        with self.assertRaisesRegex(ValueError, "score counts disagree"):
            control.evaluate_cases(self.args)
        self.assertFalse((Path(self.args.output_dir) / "summary.json").exists())

    def test_score_only_reconstructs_official_layout_without_inference_and_checks_saved_ids(self):
        control.evaluate_cases(self.args)
        self.commands.clear()
        self.mocks[2].reset_mock()
        self.args.score_only = True
        control.evaluate_cases(self.args)
        self.assertEqual([command[0] for command in self.commands], ["evaluate"])
        self.mocks[2].assert_not_called()
        self.assertFalse(any(path.is_dir() for path in Path(self.args.output_dir).iterdir()))
        self.source.write_text(json.dumps({"multi_turn_base": ["multi_turn_base_0"]}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "case IDs differ"):
            control.evaluate_cases(self.args)

    def test_score_only_fingerprint_mismatch_preserves_the_previous_record(self):
        control.evaluate_cases(self.args)
        fingerprint = Path(self.args.output_dir) / "bfcl_environment.json"
        before = fingerprint.read_bytes()
        self.args.score_only = True
        self.args.temperature = 0.6
        self.commands.clear()
        with self.assertRaisesRegex(ValueError, "fingerprint differs"):
            control.evaluate_cases(self.args)
        self.assertEqual(fingerprint.read_bytes(), before)
        self.assertEqual(self.commands, [])

    def test_failed_rescore_invalidates_the_previous_success_summary(self):
        control.evaluate_cases(self.args)
        self.args.score_only = True
        self.mocks[3].side_effect = subprocess.CalledProcessError(7, ["evaluate"])
        with self.assertRaises(subprocess.CalledProcessError):
            control.evaluate_cases(self.args)
        output = Path(self.args.output_dir)
        self.assertFalse((output / "summary.json").exists())
        self.assertTrue((output / "BFCL_v4_multi_turn_base_result.json").is_file())

    def test_cli_evaluation_dispatch_does_not_import_a_gpu_engine(self):
        with patch.object(control, "evaluate_cases") as evaluate, patch.dict(sys.modules, {"vllm": None}):
            control.main(["--evaluate-case-ids", str(self.source), "--output-dir", self.args.output_dir,
                          "--bfcl-model", self.args.bfcl_model, "--local-model-path", "original-model"])
        evaluate.assert_called_once()
        self.assertEqual(evaluate.call_args.args[0].evaluate_case_ids, str(self.source))

    def test_subprocess_log_streaming_uses_current_interpreter_and_rejects_a_swallowed_404(self):
        process = Mock()
        process.__enter__ = Mock(return_value=process)
        process.__exit__ = Mock(return_value=False)
        process.stdout = ["Error code: 404 The model original-model does not exist\n"]
        process.wait.return_value = 0
        process.args = ["fake"]
        with patch.object(control.subprocess, "Popen", return_value=process) as popen:
            with self.assertRaisesRegex(RuntimeError, "404"):
                self.run_bfcl(["generate", "--run-ids"], self.root / "generate.log", {"test": "env"})
        self.assertEqual(popen.call_args.args[0], [sys.executable, "-m", "bfcl_eval", "generate", "--run-ids"])


class TestBFCLRuntimeParity(unittest.TestCase):
    def test_absolute_script_fingerprints_the_installed_package_instead_of_itself(self):
        with tempfile.TemporaryDirectory(prefix="bfcl script import ") as temporary:
            root = Path(temporary)
            package = root / "bfcl_eval"
            package.mkdir()
            (package / "__init__.py").write_text("# independently installed benchmark\n", encoding="utf-8")
            data = package / "data"
            data.mkdir()
            (data / "tasks.json").write_text('{"id":"task_0"}', encoding="utf-8")
            output = root / "fingerprint.json"
            result = subprocess.run([
                sys.executable, str(ROOT / "verl/utils/bfcl_eval.py"), "--record-bfcl-env", str(output),
                "--bfcl-model", "test-FC",
            ], cwd=root, env=dict(os.environ, PYTHONPATH=str(root), PYTHONUTF8="1"),
                capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            recorded = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(recorded["code_sha256"], control._fingerprint_files(package, package.rglob("*.py")))
            self.assertEqual(recorded["data_sha256"], control._fingerprint_files(package, data.rglob("*.json")))

    def test_short_probe_checks_the_actual_model_and_completion_before_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            opener = Mock()
            opener.open.side_effect = [
                io.BytesIO(b'{"data":[{"id":"actual-model"}]}'),
                io.BytesIO(b'{"choices":[{"text":"Hello"}]}'),
            ]
            with patch.object(control, "build_opener", return_value=opener), patch.dict(os.environ, {
                "REMOTE_OPENAI_BASE_URL": "http://127.0.0.1:1234/v1", "REMOTE_OPENAI_API_KEY": "EMPTY",
            }):
                control._probe_model("actual-model", output)
            request = opener.open.call_args_list[1].args[0]
            payload = json.loads(request.data)
            self.assertEqual(payload["model"], "actual-model")
            self.assertEqual(payload["max_tokens"], 8)
            self.assertEqual(payload["n"], 1)
            self.assertTrue(json.loads((output / "inference_probe.json").read_text())["ok"])
            opener.open.side_effect = [io.BytesIO(b'{"data":[{"id":"other-model"}]}')]
            with patch.object(control, "build_opener", return_value=opener), patch.dict(os.environ, {
                "REMOTE_OPENAI_BASE_URL": "http://127.0.0.1:1234/v1",
            }), self.assertRaisesRegex(RuntimeError, "is absent"):
                control._probe_model("actual-model", output)

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
