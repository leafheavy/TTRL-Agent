"""Catch prompt-budget errors through real dataset methods before LLM initialization."""

import ast
import builtins
import copy
import importlib.util
import json
import logging
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("test_data_preflight", ROOT / "verl/utils/dataset/preflight.py")
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


def compile_functions(relative_path, names, namespace):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


class TestPreflightControl(unittest.TestCase):
    def trainer_tail(self):
        # Execute the actual entry-point tail to ensure preflight precedes init_workers.
        path = ROOT / "verl/trainer/main_ppo.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        runner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TaskRunner")
        run = next(node for node in runner.body if getattr(node, "name", None) == "run")
        start = next(
            index
            for index, node in enumerate(run.body)
            if isinstance(node, ast.ImportFrom) and node.module == "verl.utils.dataset.preflight"
        )
        tail = ast.parse("def run_tail(config, train_dataset, val_dataset, trainer): pass").body[0]
        tail.body = run.body[start:]
        namespace = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[tail], type_ignores=[])), str(path), "exec"), namespace)
        modules = {name: ModuleType(name) for name in ("verl", "verl.utils", "verl.utils.dataset")}
        modules["verl.utils.dataset.preflight"] = preflight
        module_patch = patch.dict(sys.modules, modules)
        module_patch.start()
        self.addCleanup(module_patch.stop)
        return namespace["run_tail"]

    def test_preflight_only_exits_the_real_entry_point_before_model_workers(self):
        from unittest.mock import Mock

        trainer = Mock()
        dataset = [{"raw_prompt_ids": [1]}]
        self.trainer_tail()({"trainer": {"preflight_only": True}}, dataset, dataset, trainer)
        trainer.init_workers.assert_not_called()
        trainer.fit.assert_not_called()

    def test_dataset_failure_in_real_entry_point_never_initializes_models(self):
        from unittest.mock import Mock

        class BadDataset:
            def __len__(self):
                return 1

            def __getitem__(self, index):
                raise NotImplementedError("sequence_length=6165 is larger than max_length=4096")

        trainer = Mock()
        config = {"ttrl": {"enable": True, "reward_mode": "bfcl"}}
        with self.assertRaisesRegex(ValueError, "before model initialization.*6165.*4096"):
            self.trainer_tail()(config, BadDataset(), [], trainer)
        trainer.init_workers.assert_not_called()
        trainer.fit.assert_not_called()

    def test_normal_training_does_not_scan_other_benchmarks_implicitly(self):
        class UntouchedDataset:
            def __len__(self):
                raise AssertionError("Dataset should not have been read")

        self.assertFalse(preflight.run_data_preflight({}, UntouchedDataset(), UntouchedDataset()))

    def test_explicit_preflight_only_scans_both_splits_and_returns_exit_signal(self):
        train = [{"raw_prompt_ids": [1, 2, 3], "index": "train_case"}]
        validation = [{"input_ids": [0, 1, 2], "attention_mask": [0, 1, 1]}]
        config = {"trainer": {"preflight_only": True}}
        self.assertTrue(preflight.run_data_preflight(config, train, validation))
        self.assertEqual(preflight.validate_dataset(validation, "validation")["max_prompt_tokens"], 2)

    def test_bfcl_ttrl_and_supervised_scan_without_requesting_early_exit(self):
        dataset = [{"input_ids": [1, 2]}]
        for config in (
            {"ttrl": {"enable": True, "reward_mode": "bfcl"}},
            {"bfcl_supervised": {"enabled": True}},
        ):
            with self.subTest(config=config), patch.object(
                preflight, "validate_dataset", wraps=preflight.validate_dataset
            ) as validate:
                self.assertFalse(preflight.run_data_preflight(config, dataset, dataset))
                self.assertEqual([call.args[1] for call in validate.call_args_list], ["train", "validation"])


class TestActualBFCLPromptBudget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            import torch.nn.functional as functional
        except ImportError:
            raise unittest.SkipTest("Actual dataset regression requires the training-side torch dependency") from None
        cls.torch = torch
        tensor_namespace = {"torch": torch, "F": functional}
        compile_functions(
            "verl/utils/torch_functional.py", {"postprocess_data", "pad_sequence_to_length"}, tensor_namespace
        )
        cls.tensor_helpers = SimpleNamespace(**tensor_namespace)

    def setUp(self):
        path = ROOT / "verl/utils/dataset/rl_dataset.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        dataset_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RLHFDataset")
        methods = [
            node for node in dataset_node.body if getattr(node, "name", None) in ("_build_messages", "__getitem__")
        ]
        dataset_class = ast.ClassDef(name="DatasetUnderTest", bases=[], keywords=[], body=methods, decorator_list=[])
        helper = next(node for node in tree.body if getattr(node, "name", None) == "bfcl_chat_template_kwargs")
        namespace = {
            "verl_F": self.tensor_helpers,
            "compute_position_id_with_mask": lambda mask: mask.cumsum(-1) - 1,
            "logger": logging.getLogger(__name__),
        }
        module = ast.fix_missing_locations(ast.Module(body=[helper, dataset_class], type_ignores=[]))
        exec(compile(module, str(path), "exec"), namespace)

        modules = {
            name: ModuleType(name)
            for name in ("verl", "verl.utils", "verl.utils.reward_score", "verl.utils.reward_score.ttrl_bfcl")
        }
        reward_module = modules["verl.utils.reward_score.ttrl_bfcl"]
        reward_module.json = json
        compile_functions("verl/utils/reward_score/ttrl_bfcl.py", {"decode_bfcl_context"}, reward_module.__dict__)
        module_patch = patch.dict(sys.modules, modules)
        module_patch.start()
        self.addCleanup(module_patch.stop)
        original_import = builtins.__import__

        def restricted_import(name, *args, **kwargs):
            if name.split(".")[0] in ("bfcl_eval", "jsonschema"):
                raise AssertionError("Benchmark library imported by data preflight: " + name)
            return original_import(name, *args, **kwargs)

        import_patch = patch("builtins.__import__", side_effect=restricted_import)
        import_patch.start()
        self.addCleanup(import_patch.stop)
        torch = self.torch

        class Tokenizer:
            pad_token_id = 0

            def apply_chat_template(self, messages, **kwargs):
                self.kwargs = kwargs
                return "x" * (6165 if kwargs.get("tools") else 100)

            def __call__(self, text, **kwargs):
                return {
                    "input_ids": torch.ones(1, len(text), dtype=torch.long),
                    "attention_mask": torch.ones(1, len(text)),
                }

            def encode(self, text, **kwargs):
                return [1] * len(text)

        class Frame(list):
            def __getitem__(self, index):
                return copy.deepcopy(super().__getitem__(index))

        self.dataset = namespace["DatasetUnderTest"]()
        self.dataset.tokenizer = Tokenizer()
        self.dataset.processor = None
        self.dataset.enable_thinking = None
        self.dataset.prompt_key = "prompt"
        self.dataset.image_key = "images"
        self.dataset.video_key = "videos"
        self.dataset.max_prompt_length = 4096
        self.dataset.truncation = "error"
        self.dataset.return_raw_chat = True
        self.dataset.return_full_prompt = False
        self.dataset.need_tools_kwargs = False
        self.dataset.dataframe = Frame(
            {
                "prompt": [{"role": "user", "content": "task"}],
                "extra_info": {"index": f"case_{index}", "bfcl": json.dumps({"functions": [{"name": "pwd"}]})},
            }
            for index in range(2)
        )
        # Keep the actual data access and processing bodies; omit parquet I/O.
        namespace["DatasetUnderTest"].__len__ = lambda dataset: len(dataset.dataframe)

    def test_6165_token_schema_prompt_fails_during_preflight_with_original_cause(self):
        with self.assertRaisesRegex(ValueError, "train.*before model initialization.*6165.*4096") as caught:
            preflight.validate_dataset(self.dataset, "train")
        self.assertIsInstance(caught.exception.__cause__, NotImplementedError)
        self.assertEqual(len(self.dataset), 2)

    def test_8192_budget_keeps_all_tokens_and_selected_cases(self):
        self.dataset.max_prompt_length = 8192
        summary = preflight.validate_dataset(self.dataset, "train")
        self.assertEqual(summary, {"samples": 2, "max_prompt_tokens": 6165, "longest_index": "case_0"})
        row = self.dataset[0]
        self.assertEqual(len(row["input_ids"]), 8192)
        self.assertEqual(len(row["raw_prompt_ids"]), 6165)
        self.assertEqual(int(row["attention_mask"].sum()), 6165)
        self.assertEqual(self.dataset.tokenizer.kwargs["tools"], [{"type": "function", "function": {"name": "pwd"}}])
        self.assertEqual(self.dataset.dataframe[1]["extra_info"]["index"], "case_1")

    def test_validation_split_is_also_checked_before_models(self):
        train = [{"raw_prompt_ids": [1, 2]}]
        config = {"ttrl": {"enable": True, "reward_mode": "bfcl"}}
        with self.assertRaisesRegex(ValueError, "validation.*6165.*4096"):
            preflight.run_data_preflight(config, train, self.dataset)


if __name__ == "__main__":
    unittest.main()
