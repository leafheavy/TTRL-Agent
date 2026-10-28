"""Check exact case reuse, label isolation, and completeness before BFCL scoring."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import datasets
from bfcl_eval import utils as bfcl_utils

from data.preprocess import (
    MULTI_TURN_COLUMNS,
    check_bfcl_result_ids,
    prepare_bfcl_data,
    select_bfcl_inputs,
    summarize_bfcl_multi_turn,
)


class HiddenAnswer:
    def __deepcopy__(self, memo):
        raise AssertionError("Benchmark reference was read")


def entry(category, index, class_name="GorillaFileSystem"):
    return {
        "id": f"{category}_{index}",
        "question": [[{"role": "user", "content": "List the directory"}]],
        "function": [
            {
                "name": "ls",
                "description": "List the directory",
                "parameters": {"type": "dict", "properties": {}, "required": []},
            }
        ],
        "involved_classes": [class_name],
        "initial_config": {class_name: {"location": "initial"}},
        "ground_truth": HiddenAnswer(),
        "possible_answer": HiddenAnswer(),
    }


class TestBFCLPreprocess(unittest.TestCase):
    def setUp(self):
        self.source = {
            category: [entry(category, 0), entry(category, 1, "MathAPI")]
            for category in (
                "multi_turn_base",
                "multi_turn_miss_param",
                "multi_turn_miss_func",
                "multi_turn_long_context",
            )
        }
        self.files = patch.object(bfcl_utils, "find_file_by_category", side_effect=lambda category, _: category)
        self.loads = patch.object(bfcl_utils, "load_file", side_effect=lambda category, **_: self.source[category])

    def select(self, categories="multi_turn", **kwargs):
        with self.files, self.loads:
            return select_bfcl_inputs(categories, **kwargs)

    def test_multiple_categories_use_existing_bfcl_alias_and_never_read_labels(self):
        selected = self.select()
        self.assertEqual(len(selected), 8)
        self.assertEqual({row["id"].rsplit("_", 1)[0] for row in selected}, set(self.source))
        for row in selected:
            self.assertNotIn("ground_truth", row)
            self.assertNotIn("possible_answer", row)

    def test_state_and_explicit_filesystem_selection_share_the_same_cases(self):
        calls = self.select(consensus="calls", fs_only=True)
        state = self.select(consensus="state")
        self.assertEqual([row["id"] for row in calls], [row["id"] for row in state])
        self.assertEqual(len(state), 4)
        for row in state:
            self.assertEqual(row["observable_state"], {"GorillaFileSystem": ["root"]})
        self.assertNotIn("observable_state", self.source["multi_turn_base"][0])

    def test_sample_limit_applies_after_domain_filter(self):
        selected = self.select(fs_only=True, max_samples=2)
        self.assertEqual(len(selected), 2)
        self.assertTrue(all(row["involved_classes"] == ["GorillaFileSystem"] for row in selected))

    def test_invalid_selection_fails(self):
        for kwargs in ({"consensus": "unknown"}, {"max_samples": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.select(**kwargs)
        with self.assertRaises(ValueError):
            self.select("single_turn")
        self.source = {category: [] for category in self.source}
        with self.assertRaises(ValueError):
            self.select()

    def test_parquet_inputs_and_evaluation_manifest_are_identical_at_case_level(self):
        # Keep poisoned references on the raw entries to also check the JSON-file entry point.
        selected = [self.source["multi_turn_base"][0], self.source["multi_turn_miss_param"][1]]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = prepare_bfcl_data(selected, root)
            train = datasets.Dataset.from_parquet(str(root / "train.parquet")).to_list()
            test = datasets.Dataset.from_parquet(str(root / "test.parquet")).to_list()
            self.assertEqual([row["extra_info"]["index"] for row in train], [row["id"] for row in selected])
            self.assertEqual(manifest, json.loads((root / "case_ids.json").read_text(encoding="utf-8")))
            self.assertEqual({case_id for ids in manifest.values() for case_id in ids}, {row["id"] for row in selected})
            for left, right in zip(train, test):
                self.assertEqual(left["extra_info"]["split"], "train")
                self.assertEqual(right["extra_info"]["split"], "test")
                left["extra_info"].pop("split")
                right["extra_info"].pop("split")
                self.assertEqual(left, right)  # Same messages, schemas, initial states and future turns.
                self.assertNotIn("reward_model", left)
            exported = (root / "task_inputs.json").read_text(encoding="utf-8")
            self.assertNotIn("ground_truth", exported)
            self.assertNotIn("possible_answer", exported)

    def test_existing_explicit_validation_input_remains_supported(self):
        train = [self.source["multi_turn_base"][0]]
        val = [self.source["multi_turn_base"][1]]
        with tempfile.TemporaryDirectory() as tmp:
            manifest = prepare_bfcl_data(train, tmp, val)
            self.assertEqual(manifest, {"multi_turn_base": [val[0]["id"]]})
            rows = datasets.Dataset.from_parquet(str(Path(tmp) / "test.parquet")).to_list()
            self.assertEqual(rows[0]["extra_info"]["index"], val[0]["id"])

    def test_duplicate_or_missing_ids_are_rejected_before_writing(self):
        row = self.source["multi_turn_base"][0]
        for rows in ([], [row, row], [{key: value for key, value in row.items() if key != "id"}]):
            with self.subTest(count=len(rows)), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError):
                    prepare_bfcl_data(rows, tmp)
                self.assertFalse(list(Path(tmp).iterdir()))


class TestBFCLResultIDs(unittest.TestCase):
    def test_scoring_requires_exact_ids_and_rejects_partial_or_duplicate_results(self):
        expected = {"multi_turn_base": ["multi_turn_base_0", "multi_turn_base_1"]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / bfcl_utils.get_file_name_by_category("multi_turn_base", is_result_file=True)
            for actual in (
                ["multi_turn_base_1", "multi_turn_base_0"],
                ["multi_turn_base_0"],
                ["multi_turn_base_0", "multi_turn_base_1", "multi_turn_base_2"],
                ["multi_turn_base_0", "multi_turn_base_0", "multi_turn_base_1"],
            ):
                path.write_text(
                    "".join(json.dumps({"id": case_id, "result": []}) + "\n" for case_id in actual), encoding="utf-8"
                )
                if len(actual) == 2 and set(actual) == set(expected["multi_turn_base"]):
                    check_bfcl_result_ids(expected, tmp)
                else:
                    with self.subTest(actual=actual), self.assertRaises(ValueError):
                        check_bfcl_result_ids(expected, tmp)

    def test_missing_category_or_empty_manifest_cannot_be_scored(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                check_bfcl_result_ids({"multi_turn_base": ["multi_turn_base_0"]}, tmp)
            with self.assertRaises(ValueError):
                check_bfcl_result_ids({}, tmp)


class TestBFCLMultiTurnSummary(unittest.TestCase):
    model = "Qwen/Qwen2.5-3B-Instruct-FC"

    @staticmethod
    def write_jsonl(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def create_runs(self, root):
        """Use BFCL's real result/score filenames, directories and JSONL header."""
        runs = {}
        for (category, _), (correct, total) in zip(MULTI_TURN_COLUMNS, ((1, 3), (1, 2), (0, 1), (4, 5))):
            child = root / category
            ids = [f"{category}_{index}" for index in range(total)]
            manifest = child / "data" / "case_ids.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({category: ids}), encoding="utf-8")
            directory = bfcl_utils.get_directory_structure_by_category(category)
            result = child / "eval" / "result" / self.model.replace("/", "_") / directory
            result /= bfcl_utils.get_file_name_by_category(category, is_result_file=True)
            self.write_jsonl(result, [{"id": case_id, "result": []} for case_id in reversed(ids)])
            score = child / "eval" / "score" / self.model.replace("/", "_") / directory
            score /= bfcl_utils.get_file_name_by_category(category, is_score_file=True)
            header = {"accuracy": correct / total, "correct_count": correct, "total_count": total}
            # Official files contain a header followed by failed-case details;
            # loading the whole file as one JSON document would be incorrect.
            self.write_jsonl(score, [header, {"id": ids[-1], "valid": False}])
            runs[category] = {"ids": ids, "manifest": manifest, "result": result, "score": score}
        return runs

    def assert_no_summary(self, root):
        self.assertEqual(list(root.glob("multi_turn_summary.*")), [])

    def test_macro_average_and_table_order_preserve_unrounded_json(self):
        label = "Full | calls\nfrequency"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.create_runs(root)
            summary = summarize_bfcl_multi_turn(root, self.model, label)
            expected = (1 / 3 + 1 / 2 + 0 + 4 / 5) / 4
            self.assertAlmostEqual(summary["average_accuracy"], expected)
            self.assertNotAlmostEqual(summary["average_accuracy"], 6 / 11)  # Not pooled case accuracy.
            self.assertEqual([row["category"] for row in summary["categories"]], [
                "multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context",
            ])
            stored = json.loads((root / "multi_turn_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(stored, summary)
            self.assertEqual(stored["categories"][0]["accuracy"], 1 / 3)
            with (root / "multi_turn_summary.csv").open(encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.reader(handle))
            self.assertEqual(csv_rows[0], [
                "Model", "Average (%)", "Base (%)", "Miss Func (%)", "Miss Param (%)", "Long Context (%)",
            ])
            self.assertEqual(csv_rows[1], [label, "40.83", "33.33", "50.00", "0.00", "80.00"])
            markdown = (root / "multi_turn_summary.md").read_text(encoding="utf-8")
            self.assertIn("| Full \\| calls frequency | 40.83 | 33.33 | 50.00 | 0.00 | 80.00 |", markdown)
            self.assertIn("不是单一模型的 BFCL 总分", markdown)

    def test_missing_category_manifest_or_generated_results_produce_no_summary(self):
        for missing in ("manifest", "result"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                runs = self.create_runs(root)
                runs["multi_turn_long_context"][missing].unlink()
                with self.assertRaises(FileNotFoundError):
                    summarize_bfcl_multi_turn(root, self.model, "TTRL")
                self.assert_no_summary(root)

    def test_missing_duplicate_or_wrong_model_score_produces_no_summary(self):
        for fault in ("missing", "duplicate", "wrong_model"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                score = self.create_runs(root)["multi_turn_long_context"]["score"]
                if fault == "missing":
                    score.unlink()
                elif fault == "duplicate":
                    duplicate = score.parent / "duplicate" / score.name
                    duplicate.parent.mkdir()
                    duplicate.write_bytes(score.read_bytes())
                model = "Other/Model-FC" if fault == "wrong_model" else self.model
                with self.assertRaises(ValueError):
                    summarize_bfcl_multi_turn(root, model, "TTRL")
                self.assert_no_summary(root)

    def test_invalid_or_mixed_category_manifest_is_rejected(self):
        category = "multi_turn_long_context"
        manifests = (
            {category: []},
            {category: [f"{category}_0", f"{category}_0"]},
            {category: ["multi_turn_base_0"]},
            {category: [f"{category}_0"], "multi_turn_base": ["multi_turn_base_0"]},
        )
        for manifest in manifests:
            with self.subTest(manifest=manifest), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                path = self.create_runs(root)[category]["manifest"]
                path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaises(ValueError):
                    summarize_bfcl_multi_turn(root, self.model, "TTRL")
                self.assert_no_summary(root)

    def test_missing_extra_or_duplicate_generated_case_ids_are_rejected(self):
        category = "multi_turn_long_context"
        for fault in ("missing", "extra", "duplicate"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                run = self.create_runs(root)[category]
                ids = run["ids"][:]
                if fault == "missing":
                    ids.pop()
                elif fault == "extra":
                    ids.append(f"{category}_99")
                else:
                    ids[-1] = ids[0]
                self.write_jsonl(run["result"], [{"id": case_id, "result": []} for case_id in ids])
                with self.assertRaises(ValueError):
                    summarize_bfcl_multi_turn(root, self.model, "TTRL")
                self.assert_no_summary(root)

    def test_mismatched_or_invalid_official_counts_are_rejected(self):
        for correct, total in ((4, 4), (-1, 5), (6, 5), (4.0, 5), (True, 5), (4, 5.0)):
            with self.subTest(correct=correct, total=total), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                score = self.create_runs(root)["multi_turn_long_context"]["score"]
                self.write_jsonl(score, [{"accuracy": 0.8, "correct_count": correct, "total_count": total}])
                with self.assertRaises(ValueError):
                    summarize_bfcl_multi_turn(root, self.model, "TTRL")
                self.assert_no_summary(root)


if __name__ == "__main__":
    unittest.main()
