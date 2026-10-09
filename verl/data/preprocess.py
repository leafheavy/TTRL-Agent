import argparse
import json
import os
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


def select_bfcl_inputs(categories, *, consensus="observations", fs_only=False, max_samples=0):
    """Select label-free Agent inputs with BFCL's existing category/data helpers."""
    from bfcl_eval.constants.eval_config import PROMPT_PATH
    from bfcl_eval.utils import find_file_by_category, load_file, parse_test_category_argument

    if consensus not in ("calls", "observations", "state") or max_samples < 0:
        raise ValueError("BFCL consensus must be calls/observations/state and max_samples must be non-negative")
    selected_categories = parse_test_category_argument(categories.split(","))
    supported = {"multi_turn_base", "multi_turn_miss_param", "multi_turn_miss_func", "multi_turn_long_context"}
    if set(selected_categories) - supported:
        raise ValueError("BFCL Agent input selection supports only the four local multi_turn categories")
    entries = []
    for category in selected_categories:
        for entry in load_file(find_file_by_category(category, PROMPT_PATH), use_lock=False):
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
    from bfcl_eval.utils import extract_test_category_from_id

    if not entries or (val_entries is not None and not val_entries):
        raise ValueError("BFCL update/evaluation inputs must not be empty")
    case_ids = {}
    for rows in (entries, entries if val_entries is None else val_entries):
        seen = set()
        ids = {}
        for entry in rows:
            case_id = entry.get("id")
            if not isinstance(case_id, str) or "_" not in case_id or case_id in seen:
                raise ValueError(f"BFCL inputs require unique official case IDs: {case_id!r}")
            seen.add(case_id)
            ids.setdefault(extract_test_category_from_id(case_id), []).append(case_id)
        case_ids = ids

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
    parser.add_argument("--bfcl-result-dir", help="BFCL generated result directory, used with --bfcl-check-results")
    parser.add_argument("--bfcl-val-json")
    parser.add_argument("--bfcl-consensus", choices=("calls", "observations", "state"), default="observations")
    parser.add_argument("--bfcl-fs-only", action="store_true", help="Select only GorillaFileSystem tasks")
    parser.add_argument("--bfcl-max-samples", type=int, default=0, help="Limit selected inputs; 0 uses all")
    parser.add_argument("--output-dir", default="BFCL-TTRL")
    args = parser.parse_args()
    if args.bfcl_val_json and not (args.bfcl_train_json or args.bfcl_categories):
        parser.error("--bfcl-val-json requires BFCL input selection")
    if bool(args.bfcl_check_results) != bool(args.bfcl_result_dir):
        parser.error("--bfcl-check-results and --bfcl-result-dir must be used together")
    if args.bfcl_check_results:
        case_ids = json.loads(Path(args.bfcl_check_results).read_text(encoding="utf-8"))
        check_bfcl_result_ids(case_ids, args.bfcl_result_dir)
        print(f"BFCL result IDs match: {sum(map(len, case_ids.values()))} cases")
        raise SystemExit(0)
    if args.bfcl_train_json or args.bfcl_categories:
        from bfcl_eval.utils import load_file

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
