import argparse
import json
import os
from copy import deepcopy
from pathlib import Path

import datasets


def make_bfcl_map_fn(split):
    from bfcl_eval.constants.default_prompts import DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC
    from bfcl_eval.constants.enums import ModelStyle
    from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
    from bfcl_eval.model_handler.utils import convert_to_tool
    from bfcl_eval.utils import populate_test_cases_with_predefined_functions

    def process_fn(example, idx):
        # Only copy task inputs. Benchmark answers/expected results are never read.
        fields = (
            "id", "question", "function", "initial_config", "involved_classes", "missed_function", "observable_state"
        )
        entry = {key: deepcopy(example[key]) for key in fields if key in example and example[key] is not None}
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
    parser.add_argument("--bfcl-train-json")
    parser.add_argument("--bfcl-val-json")
    parser.add_argument("--output-dir", default="BFCL-TTRL")
    args = parser.parse_args()
    if args.bfcl_train_json:
        from bfcl_eval.utils import load_file

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for split, path in (("train", args.bfcl_train_json), ("test", args.bfcl_val_json or args.bfcl_train_json)):
            # BFCL's JSONL loader preserves heterogeneous schemas/configuration;
            # reading them through Arrow first would insert schema-changing nulls.
            entries = load_file(path, use_lock=False)
            process_fn = make_bfcl_map_fn(split)
            prepared = [process_fn(entry, idx) for idx, entry in enumerate(entries)]
            datasets.Dataset.from_list(prepared).to_parquet(str(output_dir / f"{split}.parquet"))
        raise SystemExit(0)

    data_source = "MATH-L5-TTT"

    train_dataset = datasets.load_dataset("json", data_files=os.path.join(data_source, "train.json"), split="train")
    test_dataset = datasets.load_dataset("json", data_files=os.path.join(data_source, "test.json"), split="train")

    train_dataset = train_dataset.map(function=make_map_fn("train", data_source), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test", data_source), with_indices=True)

    train_dataset.to_parquet(os.path.join(data_source, "train.parquet"))
    test_dataset.to_parquet(os.path.join(data_source, "test.parquet"))
