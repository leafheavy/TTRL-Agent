import argparse
import csv
import hashlib
import json
import os
import random
from copy import deepcopy
from pathlib import Path

BFCL_INPUT_FIELDS = (
    "id",
    "question",
    "function",
    "initial_config",
    "involved_classes",
    "missed_function",
    "observable_state",
)

MULTI_TURN_COLUMNS = (
    ("multi_turn_base", "Base (%)"),
    ("multi_turn_miss_func", "Miss Func (%)"),
    ("multi_turn_miss_param", "Miss Param (%)"),
    ("multi_turn_long_context", "Long Context (%)"),
)


def load_bfcl_seal_split(path=None):
    """Validate the bundled SEAL scenario split; never infer it from file order."""
    path = Path(path) if path is not None else Path(__file__).with_name("bfcl_400_split.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    scenarios = list(range(200))
    random.Random(42).shuffle(scenarios)
    if manifest.get("seed") != 42:
        raise ValueError("The SEAL scenario split requires seed 42")
    for key, indices in (("train", scenarios[:100]), ("test", scenarios[100:])):
        ids = manifest.get(key)
        expected = {f"{category}_{index}" for category, _ in MULTI_TURN_COLUMNS for index in indices}
        if not isinstance(ids, list) or len(ids) != 400 or not all(isinstance(item, str) for item in ids):
            raise ValueError(f"SEAL {key} must contain 400 unique official case IDs")
        if len(set(ids)) != len(ids) or set(ids) != expected:
            raise ValueError(f"SEAL {key} differs from the scenario-disjoint seed-42 split")
    return manifest


def select_bfcl_inputs(
    categories, *, consensus="observations", fs_only=False, max_samples=0, split="all", partition="train"
):
    """Select label-free Agent inputs with BFCL's existing category/data helpers."""
    from bfcl_eval.constants.eval_config import PROMPT_PATH
    from bfcl_eval.utils import find_file_by_category, load_file, parse_test_category_argument

    if consensus not in ("calls", "observations", "state") or max_samples < 0:
        raise ValueError("BFCL consensus must be calls/observations/state and max_samples must be non-negative")
    if split not in ("all", "seal") or partition not in ("train", "heldout"):
        raise ValueError("BFCL split must be all/seal and partition must be train/heldout")
    if split == "all" and partition != "train":
        raise ValueError("Heldout selection requires the SEAL split")
    seal = load_bfcl_seal_split() if split == "seal" else None
    selected_ids = set(seal["train" if partition == "train" else "test"]) if seal else None
    selected_categories = parse_test_category_argument(categories.split(","))
    supported = {"multi_turn_base", "multi_turn_miss_param", "multi_turn_miss_func", "multi_turn_long_context"}
    if set(selected_categories) - supported:
        raise ValueError("BFCL Agent input selection supports only the four local multi_turn categories")
    entries = []
    for category in selected_categories:
        category_entries = load_file(find_file_by_category(category, PROMPT_PATH), use_lock=False)
        if seal:
            ids = [entry.get("id") for entry in category_entries]
            expected = {f"{category}_{index}" for index in range(200)}
            if not all(isinstance(case_id, str) for case_id in ids) or len(ids) != len(set(ids)):
                raise ValueError(f"Installed BFCL inputs contain invalid or duplicate IDs for {category}")
            if set(ids) != expected:
                raise ValueError(
                    f"Installed BFCL inputs do not cover the SEAL split for {category}: "
                    f"missing={sorted(expected - set(ids))[:10]}, extra={sorted(set(ids) - expected)[:10]}"
                )
        for entry in category_entries:
            if selected_ids is not None and entry["id"] not in selected_ids:
                continue
            if (fs_only or consensus == "state") and entry.get("involved_classes") != ["GorillaFileSystem"]:
                continue
            row = {key: deepcopy(entry[key]) for key in BFCL_INPUT_FIELDS if key in entry}
            if consensus == "state":
                row["observable_state"] = {"GorillaFileSystem": ["root"]}
            entries.append(row)
    if max_samples:
        entries = entries[:max_samples]
    if not entries:
        raise ValueError("No BFCL Agent inputs remain after category/state selection")
    return entries


def bfcl_case_ids(entries):
    """Build the existing evaluator's category-to-ID mapping with strict IDs."""
    from bfcl_eval.utils import extract_test_category_from_id

    ids, seen = {}, set()
    for entry in entries:
        case_id = entry.get("id")
        if not isinstance(case_id, str) or "_" not in case_id or case_id in seen:
            raise ValueError(f"BFCL inputs require unique official case IDs: {case_id!r}")
        seen.add(case_id)
        ids.setdefault(extract_test_category_from_id(case_id), []).append(case_id)
    return ids


def prepare_bfcl_seal_data(
    categories, output_dir, *, consensus="observations", fs_only=False, max_samples=0, eval_split="train"
):
    """Use SEAL IDs with this project's label-free preprocessing and evaluator."""
    if eval_split not in ("train", "heldout"):
        raise ValueError("BFCL evaluation split must be train/heldout")
    selection = dict(consensus=consensus, fs_only=fs_only, split="seal")
    entries = select_bfcl_inputs(categories, max_samples=max_samples, **selection)
    # A training smoke-test cap must never change the reserved evaluation pool.
    heldout = select_bfcl_inputs(categories, partition="heldout", **selection)
    evaluation = entries if eval_split == "train" else heldout
    case_ids = prepare_bfcl_data(entries, output_dir, None if eval_split == "train" else heldout)
    source_path = Path(__file__).with_name("bfcl_400_split.json")
    source = load_bfcl_seal_split(source_path)
    metadata = {
        "split": "seal", "eval_split": eval_split,
        "setting": "same_case_adaptation" if eval_split == "train" else "heldout_generalization",
        "source": {
            "name": "SEAL bfcl_400_split.json", "path": "data/bfcl_400_split.json",
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "seed": source["seed"], "rule": source["rule"],
        },
        "selection": {
            "categories": list(bfcl_case_ids(entries)), "consensus": consensus,
            "fs_only": fs_only, "max_samples": max_samples,
        },
        "train_case_ids": bfcl_case_ids(entries), "eval_case_ids": bfcl_case_ids(evaluation),
        "heldout_case_ids": bfcl_case_ids(heldout),
    }
    for filename, content in (
        ("train_case_ids.json", metadata["train_case_ids"]),
        ("heldout_case_ids.json", metadata["heldout_case_ids"]),
        ("split_manifest.json", metadata),
    ):
        (Path(output_dir) / filename).write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    return entries, case_ids


def make_bfcl_map_fn(split):
    from bfcl_eval.constants.default_prompts import DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC
    from bfcl_eval.constants.enums import ModelStyle
    from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
    from bfcl_eval.model_handler.utils import convert_to_tool
    from bfcl_eval.utils import populate_test_cases_with_predefined_functions

    def process_fn(example, idx):
        # Only copy task inputs. Benchmark answers/expected results are never read.
        entry = {
            key: deepcopy(example[key]) for key in BFCL_INPUT_FIELDS if key in example and example[key] is not None
        }
        multi_turn = bool(entry.get("involved_classes"))
        if multi_turn and "function" not in entry:
            entry = populate_test_cases_with_predefined_functions([entry])[0]
        tools = convert_to_tool(entry["function"], GORILLA_TO_OPENAPI, ModelStyle.OSSMODEL)
        questions = entry["question"]
        if questions and isinstance(questions[0], dict):
            questions = [questions]
        schedule = {}
        for index, functions in entry.get("missed_function", {}).items():
            if not functions or not isinstance(functions[0], dict):
                raise ValueError("missed_function must contain BFCL's populated function documents")
            released = convert_to_tool(functions, GORILLA_TO_OPENAPI, ModelStyle.OSSMODEL)
            if int(index) == 0:
                tools.extend(released)
            else:
                schedule[str(index)] = released
            questions[int(index)] = [{"role": "user", "content": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC}]
        context = {"functions": tools}
        if multi_turn:
            context.update(
                {
                    "initial_config": entry.get("initial_config", {}),
                    "involved_classes": entry["involved_classes"],
                    "future_user_turns": questions[1:],
                    "function_schedule": schedule,
                    "long_context": "long_context" in entry["id"] or "composite" in entry["id"],
                }
            )
            if "observable_state" in entry:
                context["observable_state"] = entry["observable_state"]
        return {
            "data_source": "bfcl",
            "prompt": questions[0],
            "ability": "tool_use",
            "agent_name": "tool_agent" if multi_turn else "single_turn_agent",
            "extra_info": {
                "split": split,
                "index": entry.get("id", f"bfcl-{idx}"),
                "bfcl": json.dumps(context, ensure_ascii=False),
            },
        }

    return process_fn


def prepare_bfcl_data(entries, output_dir, val_entries=None):
    """Default to the same cases for updates and fresh, post-update BFCL evaluation."""
    import datasets

    if not entries or (val_entries is not None and not val_entries):
        raise ValueError("BFCL update/evaluation inputs must not be empty")
    bfcl_case_ids(entries)
    case_ids = bfcl_case_ids(entries if val_entries is None else val_entries)

    process_fn = make_bfcl_map_fn("train")
    train_rows = [process_fn(entry, idx) for idx, entry in enumerate(entries)]
    if val_entries is None:
        test_rows = deepcopy(train_rows)
        for row in test_rows:
            row["extra_info"]["split"] = "test"
    else:
        process_fn = make_bfcl_map_fn("test")
        test_rows = [process_fn(entry, idx) for idx, entry in enumerate(val_entries)]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in (("train", train_rows), ("test", test_rows)):
        datasets.Dataset.from_list(rows).to_parquet(str(output_dir / f"{split}.parquet"))
    # Keep original inputs and exact evaluation IDs; never export benchmark answers.
    inputs = [{key: entry[key] for key in BFCL_INPUT_FIELDS if key in entry} for entry in entries]
    (output_dir / "task_inputs.json").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in inputs), encoding="utf-8"
    )
    (output_dir / "case_ids.json").write_text(json.dumps(case_ids, ensure_ascii=False, indent=2), encoding="utf-8")
    return case_ids


def check_bfcl_result_ids(case_ids, result_dir):
    """Reject missing/extra/duplicate cases before BFCL's partial evaluator runs."""
    from bfcl_eval.utils import find_file_by_category, load_file

    if not case_ids or not all(case_ids.values()):
        raise ValueError("BFCL evaluation case IDs must not be empty")
    for category, expected in case_ids.items():
        path = find_file_by_category(category, Path(result_dir), is_result_file=True)
        actual = [row["id"] for row in load_file(path, use_lock=False)]
        missing, extra = sorted(set(expected) - set(actual)), sorted(set(actual) - set(expected))
        if missing or extra or len(actual) != len(expected) or len(actual) != len(set(actual)):
            raise ValueError(
                f"BFCL result IDs differ for {category}: missing={missing[:10]}, extra={extra[:10]} "
                "(duplicate IDs are also rejected)"
            )


def _flat_bfcl_summary_protocol(child, category, evaluation, cases):
    """Check each flat evaluation against the recorded, isolated training split."""
    manifest_path = child / "data" / "split_manifest.json"
    if not manifest_path.is_file():
        if evaluation != "train":
            raise ValueError("Heldout evaluation requires a recorded SEAL split")
        expected = json.loads((child / "data" / "case_ids.json").read_text(encoding="utf-8"))
        if cases != expected:
            raise ValueError(f"Evaluation case IDs disagree with training data for {category}")
        return None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_path = Path(__file__).with_name("bfcl_400_split.json")
    source = load_bfcl_seal_split(source_path)
    expected_setting = {"train": "same_case_adaptation", "heldout": "heldout_generalization"}
    recorded_side = manifest.get("eval_split")
    if (
        manifest.get("split") != "seal" or recorded_side not in expected_setting
        or manifest.get("setting") != expected_setting[recorded_side]
        or manifest.get("source", {}).get("sha256") != hashlib.sha256(source_path.read_bytes()).hexdigest()
    ):
        raise ValueError(f"Invalid recorded SEAL split for {category}")
    partitions = {}
    for side, source_key in (("train", "train"), ("heldout", "test")):
        selected = json.loads((child / "data" / f"{side}_case_ids.json").read_text(encoding="utf-8"))
        ids = selected.get(category) if isinstance(selected, dict) else None
        if (
            not isinstance(ids, list) or not ids or set(selected) != {category}
            or not all(isinstance(case_id, str) and case_id.startswith(category + "_") for case_id in ids)
            or len(ids) != len(set(ids)) or not set(ids) <= set(source[source_key])
            or selected != manifest.get(f"{side}_case_ids")
        ):
            raise ValueError(f"Invalid or inconsistent {side} case IDs for {category}")
        partitions[side] = selected
    train_scenarios = {case_id.rsplit("_", 1)[-1] for case_id in partitions["train"][category]}
    heldout_scenarios = {case_id.rsplit("_", 1)[-1] for case_id in partitions["heldout"][category]}
    if train_scenarios & heldout_scenarios:
        raise ValueError(f"Train and heldout scenarios overlap for {category}")
    if manifest.get("eval_case_ids") != partitions[recorded_side] or cases != partitions[evaluation]:
        raise ValueError(f"Split protocol disagrees with evaluation case IDs for {category}")
    return {
        "split": "seal", "eval_split": evaluation, "setting": expected_setting[evaluation],
        "source": manifest["source"],
        "selection": {key: manifest["selection"][key] for key in ("consensus", "fs_only", "max_samples")},
    }


def _collect_bfcl_multi_turn(run_root, model, label, evaluation=None):
    """Report four independently adapted models using official score counts.

    Average is the unweighted mean of the four category accuracies, even when
    filtering produces different category sizes. It is not BFCL Overall Acc.
    """
    run_root = Path(run_root)
    rows = []
    protocols = []
    for category, _ in MULTI_TURN_COLUMNS:
        child = run_root / category
        evaluation_dir = child / "eval" if evaluation is None else child / f"eval_{evaluation}"
        case_path = (child / "data" / "case_ids.json" if evaluation is None
                     else evaluation_dir / "test_case_ids_to_generate.json")
        cases = json.loads(case_path.read_text(encoding="utf-8"))
        if set(cases) != {category} or not isinstance(cases[category], list) or not cases[category]:
            raise ValueError(f"Expected exactly the selected {category} cases in {child}")
        ids = cases[category]
        if not all(isinstance(case_id, str) and case_id.startswith(category + "_") for case_id in ids):
            raise ValueError(f"Invalid case IDs for {category}")
        if len(ids) != len(set(ids)):
            raise ValueError(f"Invalid or duplicate case IDs for {category}")
        manifest_path = child / "data" / "split_manifest.json"
        protocol = None
        if evaluation is not None:
            protocol = _flat_bfcl_summary_protocol(child, category, evaluation, cases)
        elif manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            eval_split = manifest.get("eval_split")
            expected_setting = {"train": "same_case_adaptation", "heldout": "heldout_generalization"}
            if (
                manifest.get("split") != "seal" or eval_split not in expected_setting
                or manifest.get("setting") != expected_setting[eval_split]
                or manifest.get("eval_case_ids") != cases
            ):
                raise ValueError(f"Split protocol disagrees with evaluation case IDs for {category}")
            protocol = {key: manifest[key] for key in ("split", "eval_split", "setting", "source")}
            protocol["selection"] = {
                key: manifest["selection"][key] for key in ("consensus", "fs_only", "max_samples")
            }
        protocols.append(protocol)
        if protocol != protocols[0]:
            raise ValueError("Cannot summarize categories with different or missing split protocols")
        check_bfcl_result_ids(cases, evaluation_dir / "result" if evaluation is None else evaluation_dir)
        # Accept the installed BFCL data version; do not relabel v4 as v3.
        score_dir = evaluation_dir / "score" / model.replace("/", "_") if evaluation is None else evaluation_dir
        paths = list(score_dir.rglob(f"BFCL_v*_{category}_score.json"))
        if len(paths) != 1:
            raise ValueError(f"Expected one official score file for {category}, got {paths}")
        with paths[0].open(encoding="utf-8") as handle:
            header = json.loads(next(line for line in handle if line.strip()))
        correct, total = header["correct_count"], header["total_count"]
        if (
            type(correct) is not int or type(total) is not int
            or total != len(ids) or not 0 <= correct <= total
        ):
            raise ValueError(f"Official score counts disagree with selected cases for {category}: {header}")
        if evaluation is not None:
            completed = json.loads((evaluation_dir / "summary.json").read_text(encoding="utf-8"))
            expected_counts = [(category, correct, total)]
            counts = [(row["category"], row["correct_count"], row["total_count"])
                      for row in completed.get("categories", [])]
            if (
                completed.get("bfcl_model") != model or counts != expected_counts
                or completed.get("case_ids_sha256") != hashlib.sha256(case_path.read_bytes()).hexdigest()
            ):
                raise ValueError(f"Completed evaluation metadata disagrees with official outputs for {category}")
        rows.append({
            "category": category, "correct_count": correct, "total_count": total,
            "accuracy": correct / total, "score_file": str(paths[0]), "run_root": str(child),
        })
    average = sum(row["accuracy"] for row in rows) / len(rows)
    summary = {
        "label": label, "model_registry": model,
        "setting": "independent_per_category_adaptation",
        "average_accuracy": average, "categories": rows,
    }
    if protocols[0] is not None:
        summary["data_protocol"] = protocols[0]
    if evaluation is not None:
        summary["evaluation"] = evaluation
    return summary


def _write_bfcl_multi_turn_summary(run_root, summaries):
    """Publish only after all requested categories and evaluation sides pass checks."""
    summary = summaries[0] if len(summaries) == 1 else {
        "label": summaries[0]["label"], "model_registry": summaries[0]["model_registry"],
        "setting": "independent_per_category_adaptation",
        "evaluations": {item["evaluation"]: item for item in summaries},
    }
    (run_root / "multi_turn_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    separate_evaluations = "evaluation" in summaries[0]
    columns = ["Model"] + (["Evaluation"] if separate_evaluations else [])
    columns += ["Average (%)"] + [title for _, title in MULTI_TURN_COLUMNS]
    values = []
    for item in summaries:
        row = [item["label"]] + ([item["evaluation"]] if separate_evaluations else [])
        row += [f"{100 * item['average_accuracy']:.2f}"]
        values.append(row + [f"{100 * category['accuracy']:.2f}" for category in item["categories"]])
    with (run_root / "multi_turn_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(values)
    # The first cell can contain user-provided names; escape Markdown syntax.
    md_values = [[row[0].replace("|", "\\|").replace("\n", " ")] + row[1:] for row in values]
    table = "\n".join((
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(columns) - 1)) + " |",
        *("| " + " | ".join(row) + " |" for row in md_values),
    ))
    evaluation_note = (
        "在各自场景隔离的留出 case 上冻结评估，不对留出任务进行更新。"
        if summaries[0].get("data_protocol", {}).get("eval_split") == "heldout"
        else "在各自同批 case 上重新生成并使用官方评分。"
    )
    if len(summaries) > 1:
        evaluation_note = "同一最终 actor 分别完成 train 同题评分与 heldout 未见场景评分；两次评测之间不更新参数。"
    note = (
        "各类别从同一原始模型独立训练，" + evaluation_note +
        "Average 是四类准确率的算术平均；这是四个类别专用模型的汇总，不是单一模型的 BFCL 总分。"
    )
    (run_root / "multi_turn_summary.md").write_text(table + "\n\n" + note + "\n", encoding="utf-8")
    print(table)
    print(f"Summary: {run_root / 'multi_turn_summary.md'}")
    return summary


def summarize_bfcl_multi_turn(run_root, model, label, eval_splits=None):
    """Summarize paired flat evaluations, while accepting historical BFCL layouts."""
    run_root = Path(run_root)
    if isinstance(eval_splits, str):
        eval_splits = eval_splits.split(",")
    if eval_splits is None:
        eval_splits = [side for side in ("train", "heldout")
                       if any((run_root / category / f"eval_{side}").exists() for category, _ in MULTI_TURN_COLUMNS)]
    if not eval_splits:
        return _write_bfcl_multi_turn_summary(run_root, [_collect_bfcl_multi_turn(run_root, model, label)])
    if len(set(eval_splits)) != len(eval_splits) or set(eval_splits) - {"train", "heldout"}:
        raise ValueError("Requested evaluation sides must be distinct train/heldout values")
    summaries = [_collect_bfcl_multi_turn(run_root, model, label, side) for side in eval_splits]
    return _write_bfcl_multi_turn_summary(run_root, summaries)


def make_map_fn(split, source=None):
    def process_fn(example, idx):
        if source is None:
            data_source = example.pop("source")
        else:
            data_source = source
        question = example.pop("prompt")
        solution = example.pop("answer")

        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": question,
                }
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {
                "split": split,
                "index": f"{data_source}-{idx}",
            },
        }
        return data

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare the original math data or unlabeled BFCL task inputs")
    bfcl_input = parser.add_mutually_exclusive_group()
    bfcl_input.add_argument("--bfcl-train-json")
    bfcl_input.add_argument(
        "--bfcl-categories", help="BFCL local Agent categories, comma-separated; accepts multi_turn"
    )
    bfcl_input.add_argument("--bfcl-check-results", help="Check generated result IDs against a case_ids.json manifest")
    bfcl_input.add_argument(
        "--bfcl-summarize-multi-turn", metavar="RUN_ROOT", help="Summarize four independent category runs"
    )
    parser.add_argument("--bfcl-model", default="Qwen/Qwen2.5-3B-Instruct-FC", help="Official BFCL model registry key")
    parser.add_argument("--bfcl-run-label", default="TTRL", help="Display label for the multi-turn summary row")
    parser.add_argument("--bfcl-eval-splits", help="Required summary sides, comma-separated: train,heldout")
    parser.add_argument("--bfcl-result-dir", help="BFCL generated result directory, used with --bfcl-check-results")
    parser.add_argument("--bfcl-val-json")
    parser.add_argument("--bfcl-consensus", choices=("calls", "observations", "state"), default="observations")
    parser.add_argument("--bfcl-fs-only", action="store_true", help="Select only GorillaFileSystem tasks")
    parser.add_argument("--bfcl-max-samples", type=int, default=0, help="Limit selected inputs; 0 uses all")
    parser.add_argument(
        "--bfcl-split", choices=("all", "seal"), default="all",
        help="all keeps legacy input selection; seal uses the bundled scenario-disjoint 100/100 split",
    )
    parser.add_argument(
        "--bfcl-eval-split", choices=("train", "heldout"), default="train",
        help="Evaluate the selected training cases, or reserve SEAL training cases and evaluate its heldout side",
    )
    parser.add_argument("--output-dir", default="BFCL-TTRL")
    args = parser.parse_args()
    if args.bfcl_eval_splits and not args.bfcl_summarize_multi_turn:
        parser.error("--bfcl-eval-splits requires --bfcl-summarize-multi-turn")
    if args.bfcl_split == "seal" and (not args.bfcl_categories or args.bfcl_val_json):
        parser.error("--bfcl-split seal requires --bfcl-categories and cannot use --bfcl-val-json")
    if args.bfcl_eval_split == "heldout" and args.bfcl_split != "seal":
        parser.error("--bfcl-eval-split heldout requires --bfcl-split seal")
    if args.bfcl_val_json and not (args.bfcl_train_json or args.bfcl_categories):
        parser.error("--bfcl-val-json requires BFCL input selection")
    if bool(args.bfcl_check_results) != bool(args.bfcl_result_dir):
        parser.error("--bfcl-check-results and --bfcl-result-dir must be used together")
    if args.bfcl_check_results:
        case_ids = json.loads(Path(args.bfcl_check_results).read_text(encoding="utf-8"))
        check_bfcl_result_ids(case_ids, args.bfcl_result_dir)
        print(f"BFCL result IDs match: {sum(map(len, case_ids.values()))} cases")
        raise SystemExit(0)
    if args.bfcl_summarize_multi_turn:
        summarize_bfcl_multi_turn(
            args.bfcl_summarize_multi_turn, args.bfcl_model, args.bfcl_run_label, args.bfcl_eval_splits
        )
        raise SystemExit(0)
    if args.bfcl_train_json or args.bfcl_categories:
        from bfcl_eval.utils import load_file

        if args.bfcl_split == "seal":
            entries, case_ids = prepare_bfcl_seal_data(
                args.bfcl_categories, args.output_dir, consensus=args.bfcl_consensus,
                fs_only=args.bfcl_fs_only, max_samples=args.bfcl_max_samples, eval_split=args.bfcl_eval_split,
            )
            print(f"BFCL SEAL update inputs: {len(entries)}; {args.bfcl_eval_split} evaluation inputs: "
                  f"{sum(map(len, case_ids.values()))}")
            raise SystemExit(0)

        # Load original JSONL before Arrow to preserve heterogeneous schemas/configuration.
        entries = (
            select_bfcl_inputs(
                args.bfcl_categories,
                consensus=args.bfcl_consensus,
                fs_only=args.bfcl_fs_only,
                max_samples=args.bfcl_max_samples,
            )
            if args.bfcl_categories
            else load_file(args.bfcl_train_json, use_lock=False)
        )
        val_entries = load_file(args.bfcl_val_json, use_lock=False) if args.bfcl_val_json else None
        case_ids = prepare_bfcl_data(entries, args.output_dir, val_entries)
        print(f"BFCL update inputs: {len(entries)}; evaluation inputs: {sum(map(len, case_ids.values()))}")
        raise SystemExit(0)

    data_source = "MATH-L5-TTT"

    import datasets

    train_dataset = datasets.load_dataset("json", data_files=os.path.join(data_source, "train.json"), split="train")
    test_dataset = datasets.load_dataset("json", data_files=os.path.join(data_source, "test.json"), split="train")

    train_dataset = train_dataset.map(function=make_map_fn("train", data_source), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test", data_source), with_indices=True)

    train_dataset.to_parquet(os.path.join(data_source, "train.parquet"))
    test_dataset.to_parquet(os.path.join(data_source, "test.parquet"))
