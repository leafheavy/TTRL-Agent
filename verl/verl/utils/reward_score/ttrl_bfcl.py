"""Consensus rewards from Qwen FC outputs or observable BFCL execution outcomes.

Reuse BFCL's Qwen extractor and TTRL's voting rule. Function schemas must be
the JSON schemas shown to the model, supplied in extra_info.bfcl.functions.
BFCLSession reuses the original executor with a separate namespace per rollout.
Neither reward construction nor the session reads benchmark reference answers.
"""

import json
import keyword
import re
from copy import deepcopy
from functools import lru_cache
from threading import RLock
from uuid import uuid4

import numpy as np
import torch
from jsonschema import Draft7Validator

from verl.utils.reward_score.ttrl_consensus import (
    canonical_json,
    consensus_key,
    validate_consensus_mode,
    vote_consensus,
)


@lru_cache(maxsize=1)
def get_bfcl_decoder():
    try:
        from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
    except ImportError as exc:
        raise ImportError("BFCL mode requires the existing bfcl_eval package in the training environment") from exc
    # This is a static parser; constructing a model handler/API client is unnecessary.
    return QwenFCHandler._extract_tool_calls


def decode_bfcl_context(context):
    """Accept dicts or JSON strings, avoiding Arrow's union of arbitrary schemas."""
    if isinstance(context, str):
        context = json.loads(context)
    if not isinstance(context, dict):
        raise ValueError("extra_info.bfcl must be a context object or its JSON serialization")
    return context


def _function_validators(functions):
    if not isinstance(functions, (list, tuple)):
        raise ValueError("extra_info.bfcl.functions must contain the function schemas shown to the model")
    validators = {}
    for tool in functions:
        function = tool.get("function", tool)
        name = function["name"]
        if not isinstance(name, str) or not name or name in validators:
            raise ValueError(f"Invalid or duplicate function name: {name!r}")
        schema = deepcopy(function["parameters"])
        schema.setdefault("type", "object")
        schema.setdefault("additionalProperties", False)
        Draft7Validator.check_schema(schema)
        if schema["type"] != "object":
            raise ValueError(f"Function {name!r} requires an object parameter schema")
        validators[name] = Draft7Validator(schema)
    return validators


def decode_bfcl_calls(response, validators, decoder):
    """Parse and validate calls; category construction lives in ttrl_consensus."""
    response = response.strip()
    if response == "[]":
        return []  # Explicit no-call result; never infer this from a parser failure.
    matches = list(re.finditer(r"<tool_call>(.*?)</tool_call>", response, re.DOTALL))
    if not matches or response.count("<tool_call>") != len(matches) or response.count("</tool_call>") != len(matches):
        return None
    # BFCL versions differ in whitespace tolerance. Normalize framing only,
    # leaving JSON parsing to the original extractor.
    framed = "\n".join(f"<tool_call>\n{match.group(1).strip()}\n</tool_call>" for match in matches)
    try:
        calls = decoder(framed)
    except (ValueError, TypeError, KeyError):
        return None
    # BFCL can silently discard malformed JSON. Reject the whole response if
    # any of its calls disappeared instead of voting on a successful prefix.
    if not isinstance(calls, list) or len(calls) != len(matches):
        return None
    normalized = []
    for call in calls:
        if not isinstance(call, dict):
            return None
        name, arguments = call.get("name"), call.get("arguments")
        if not isinstance(name, str) or name not in validators or not isinstance(arguments, dict):
            return None
        if not validators[name].is_valid(arguments):
            return None
        normalized.append({"name": name, "arguments": arguments})
    try:
        canonical_json(normalized)  # Reject non-JSON values, including NaN.
        return normalized
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=1)
def get_bfcl_runtime():
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call
    from bfcl_eval.model_handler.utils import convert_to_function_call

    return execute_multi_turn_func_call, convert_to_function_call


def _observable_json(value, seen=None):
    """Serialize explicitly selected public state, omitting private fields/cycles."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if callable(value):
        raise ValueError("Observable state projections must select values, not callable methods")
    seen = set() if seen is None else seen
    if id(value) in seen:
        return {"cycle": type(value).__name__}
    seen.add(id(value))
    try:
        if isinstance(value, dict):
            return {key: _observable_json(item, seen) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_observable_json(item, seen) for item in value]
        if hasattr(value, "__dict__"):
            return {key: _observable_json(item, seen) for key, item in vars(value).items() if not key.startswith("_")}
        raise ValueError(f"Observable state must have a stable JSON representation, got {type(value).__name__}")
    finally:
        seen.remove(id(value))


class BFCLSession:
    """One isolated BFCL simulator for a complete trajectory, with no oracle scoring.

    Only the eight original local multi-turn domains are enabled. State is read
    only through explicitly declared public observable_state attributes.
    """

    LOCAL_CLASSES = {
        "GorillaFileSystem",
        "MathAPI",
        "MessageAPI",
        "TwitterAPI",
        "TicketAPI",
        "TradingBot",
        "TravelAPI",
        "VehicleControlAPI",
    }

    def __init__(self, context):
        context = decode_bfcl_context(context)
        self.validators = _function_validators(context["functions"])
        self.tool_schemas = [
            tool if "function" in tool else {"type": "function", "function": tool}
            for tool in deepcopy(context["functions"])
        ]
        self.initial_config = deepcopy(context.get("initial_config", {}))
        self.classes = list(context["involved_classes"])
        if not self.classes or len(set(self.classes)) != len(self.classes) or set(self.classes) - self.LOCAL_CLASSES:
            raise ValueError("BFCL sessions require distinct classes from the eight local multi-turn domains")
        self.future_user_turns = deepcopy(context.get("future_user_turns", []))
        for messages in self.future_user_turns:
            if not messages or any(message.get("role") != "user" for message in messages):
                raise ValueError("future_user_turns must contain non-empty lists of user messages")
        self.function_schedule = {}
        self.turn_index = 0
        active_names = set(self.validators)
        for index, functions in context.get("function_schedule", {}).items():
            if not str(index).isdigit() or not 1 <= int(index) <= len(self.future_user_turns):
                raise ValueError("function_schedule keys must identify future user turns starting at 1")
            index = str(int(index))
            if index in self.function_schedule:
                raise ValueError("function_schedule cannot repeat a release turn")
            names = set(_function_validators(functions))
            if names & active_names:
                raise ValueError("Scheduled BFCL functions must not duplicate active or scheduled functions")
            active_names.update(names)
            self.function_schedule[index] = deepcopy(functions)
        self.observable_state = deepcopy(context.get("observable_state", {}))
        self.consensus_mode = validate_consensus_mode(context.get("consensus_mode", "observations"))
        if self.consensus_mode == "state" and not self.observable_state:
            raise ValueError("State consensus requires an explicit deployment-observable state projection")
        for class_name, attributes in self.observable_state.items():
            if class_name not in self.classes or not isinstance(attributes, list) or not attributes:
                raise ValueError("observable_state must map involved classes to non-empty public attribute lists")
            if any(not isinstance(attr, str) or attr.startswith("_") or not attr.isidentifier() for attr in attributes):
                raise ValueError("observable_state cannot include private or computed attributes")
        self.name_map = context.get("name_map", {})
        all_names = list(self.validators)
        for functions in self.function_schedule.values():
            all_names.extend(_function_validators(functions))
        for name in all_names:
            executable_name = self.name_map.get(name, name)
            if (
                not executable_name.isidentifier()
                or executable_name.startswith("_")
                or keyword.iskeyword(executable_name)
            ):
                raise ValueError("BFCL executable names must be public Python method identifiers")
        self.execute_fn, self.convert_calls = get_bfcl_runtime()
        self.decoder = get_bfcl_decoder()
        self.namespace = self.execute_fn.__globals__
        self.session_id = uuid4().hex
        self.namespace_keys = [f"ttrl_{self.session_id}_{name}_instance" for name in self.classes]
        self.lock = RLock()
        self.closed = False
        self.instances = {}
        self.long_context = bool(context.get("long_context", False))
        self.calls, self.observations = [], []
        self.explicit_no_call, self.episode_completed = False, False
        self.num_calls, self.parse_errors, self.execution_errors = 0, 0, 0
        self.blocked = False
        try:
            _, self.instances = self._execute([])
            # Validate the projection at initialization, before any policy actions.
            if self.consensus_mode == "state":
                self._state()
        except BaseException:
            self.close()
            raise

    def _execute(self, calls):
        return self.execute_fn(
            calls,
            deepcopy(self.initial_config),
            self.classes,
            "ttrl",
            self.session_id,
            long_context=self.long_context,
            is_evaL_run=False,
        )

    def parse_calls(self, response):
        calls = decode_bfcl_calls(response, self.validators, self.decoder)
        if calls is not None:
            return calls
        if not response.strip() or "<tool_call>" in response or "</tool_call>" in response:
            return None
        if response.lstrip().startswith(("{", "[")):
            return None  # Malformed structured calls are not natural-language termination.
        return []  # Clarification/refusal/final text is a normal terminal behavior.

    def execute(self, calls):
        """Execute calls in order; API-level errors remain visible observations."""
        with self.lock:
            if self.closed:
                raise RuntimeError("BFCL session is already closed")
            decoded = []
            for call in calls:
                name, arguments = call["name"], call["arguments"]
                if name not in self.validators or not self.validators[name].is_valid(arguments):
                    raise ValueError("BFCL execute requires schema-validated calls")
                if any(not key.isidentifier() or keyword.iskeyword(key) for key in arguments):
                    raise ValueError("BFCL keyword parameters must be Python identifiers")
                decoded.append({self.name_map.get(name, name): arguments})
            results, self.instances = self._execute(self.convert_calls(decoded))
            if len(results) != len(calls):
                raise ValueError("BFCL executor returned an unexpected number of observations")
            self.num_calls += len(calls)
            self.calls.extend(deepcopy(calls))
            errors = sum(result.startswith("Error during execution:") for result in results)
            self.execution_errors += errors
            self.blocked = bool(errors)
            for result in results:
                try:
                    observation = json.loads(result)
                except ValueError:
                    observation = result
                self.observations.append(observation)
            return [{"role": "tool", "content": result} for result in results]

    def malformed_response(self, reason="Incomplete or schema-invalid tool call; correct the call format."):
        self.parse_errors += 1
        self.blocked = True
        return [{"role": "tool", "content": f"Error: {reason}"}]

    def _state(self):
        return {
            name: {attr: _observable_json(getattr(self.instances[name], attr)) for attr in attributes}
            for name, attributes in self.observable_state.items()
        }

    def finish_turn(self, reply):
        # Control dialogue as before, but never classify natural-language wording.
        self.explicit_no_call |= reply.strip() == "[]"
        if not self.future_user_turns:
            self.episode_completed = True
            return None
        self.turn_index += 1
        messages = self.future_user_turns.pop(0)
        functions = self.function_schedule.get(str(self.turn_index), self.function_schedule.get(self.turn_index, []))
        if functions:
            new_validators = _function_validators(functions)
            if set(new_validators) & set(self.validators):
                raise ValueError("Scheduled BFCL functions must not duplicate active functions")
            self.validators.update(new_validators)
            tools = [tool if "function" in tool else {"type": "function", "function": tool} for tool in functions]
            self.tool_schemas.extend(tools)
            # Append public tool updates rather than rewriting earlier policy contexts.
            messages[-1]["content"] += (
                "\nAdditional tools are now available. Call them using "
                '<tool_call>{"name":...,"arguments":...}</tool_call>:\n' + json.dumps(tools, ensure_ascii=False)
            )
        return messages

    def outcome(self, termination):
        complete = termination == "completed" and not self.blocked and self.episode_completed
        trace = {
            "calls": self.calls,
            "observations": self.observations,
            "explicit_no_call": self.explicit_no_call,
            "final_state": self._state() if complete and self.consensus_mode == "state" else None,
        }
        key = consensus_key(trace, self.consensus_mode) if complete else None
        return {
            "key": key,
            "valid": key is not None,
            "consensus_mode": self.consensus_mode,
            "termination": termination,
            "num_calls": self.num_calls,
            "parse_errors": self.parse_errors,
            "execution_errors": self.execution_errors,
        }

    def close(self):
        with self.lock:
            for key in self.namespace_keys:
                self.namespace.pop(key, None)
            self.instances.clear()
            self.closed = True


def apply_bfcl_rewards(batch, gen_batch_output, n, tokenizer):
    """Attach a token reward tensor to generated outputs; never read reward_model."""
    if len(gen_batch_output) != len(batch) * n:
        raise ValueError("BFCL rollout count must equal the number of prompts times rollout.n")
    outcomes = gen_batch_output.non_tensor_batch.get("bfcl_outcome")
    decoder = get_bfcl_decoder() if outcomes is None else None
    rewards = torch.zeros_like(gen_batch_output.batch["responses"], dtype=torch.float32)
    extra = {
        key: [] for key in ("bfcl_valid", "bfcl_group_valid", "bfcl_majority_ratio", "bfcl_tied", "bfcl_informative")
    }
    majority_ratios = []
    for prompt_index in range(len(batch)):
        context = decode_bfcl_context(batch[prompt_index].non_tensor_batch.get("extra_info", {}).get("bfcl", {}))
        if "functions" not in context:
            raise ValueError("Each BFCL training prompt requires extra_info.bfcl.functions, without reference answers")
        mode = validate_consensus_mode(
            context.get("consensus_mode", "observations" if context.get("involved_classes") else "calls")
        )
        if outcomes is None and mode != "calls":
            raise ValueError("Observation/state consensus requires execution outcomes from the BFCL AgentLoop")
        validators = _function_validators(context["functions"]) if outcomes is None else None
        keys, reward_positions = [], []
        for offset in range(n):
            row = prompt_index * n + offset
            item = gen_batch_output[row]
            prompt_length = item.batch["prompts"].shape[-1]
            valid_mask = item.batch["attention_mask"][prompt_length:].bool()
            length = int(valid_mask.sum().item())
            action_mask = valid_mask & item.batch.get("response_mask", valid_mask).bool()
            positions = torch.nonzero(action_mask, as_tuple=True)[0]
            reward_positions.append(int(positions[-1].item()) if len(positions) else None)
            if outcomes is None:
                if torch.any(valid_mask & ~action_mask):
                    raise ValueError("BFCL tool observations require outcomes from the BFCL AgentLoop")
                response = tokenizer.decode(item.batch["responses"][:length], skip_special_tokens=False)
                eos_token = getattr(tokenizer, "eos_token", None)
                if eos_token and response.rstrip().endswith(eos_token):
                    response = response.rstrip()[: -len(eos_token)]
                calls = decode_bfcl_calls(response, validators, decoder)
                keys.append(
                    consensus_key({"calls": calls, "explicit_no_call": response.strip() == "[]"}, mode)
                    if calls is not None
                    else None
                )
            else:
                outcome = outcomes[row]
                if not isinstance(outcome, dict) or not isinstance(outcome.get("valid"), bool):
                    raise ValueError("BFCL AgentLoop must return a valid outcome record for every rollout")
                key = outcome.get("key")
                if outcome["valid"] and (not isinstance(key, str) or not key or not len(positions)):
                    raise ValueError("Valid BFCL outcomes require a canonical key and generated action tokens")
                if outcome.get("consensus_mode") != mode:
                    raise ValueError("BFCL outcome consensus mode must match the configured trajectory representation")
                keys.append(key if outcome["valid"] else None)
        result = vote_consensus(keys)
        majority_ratios.append(result["majority_ratio"])
        for offset, (score, position) in enumerate(zip(result["scores"], reward_positions)):
            if position is not None:
                rewards[prompt_index * n + offset, position] = score
        extra["bfcl_valid"].extend(result["valid"])
        for key, value in (
            ("bfcl_group_valid", result["group_valid"]),
            ("bfcl_majority_ratio", result["majority_ratio"]),
            ("bfcl_tied", result["tied"]),
            ("bfcl_informative", result["informative"]),
        ):
            extra[key].extend([value] * n)
    batch.non_tensor_batch["majority_ratio_list"] = np.asarray(majority_ratios, dtype=float)
    gen_batch_output.batch["ttrl_bfcl_scores"] = rewards
    gen_batch_output.non_tensor_batch.update({key: np.asarray(values) for key, values in extra.items()})
    return batch


def bfcl_reward(data, return_dict=False):
    """Adapter for the existing reward loader, including its async execution path."""
    if "ttrl_bfcl_scores" not in data.batch:
        raise ValueError("BFCL rewards must be prepared from the complete shared rollout group before scoring")
    rewards = data.batch["ttrl_bfcl_scores"]
    if return_dict:
        return {
            "reward_tensor": rewards,
            "reward_extra_info": {
                key: value.tolist()
                for key, value in data.non_tensor_batch.items()
                if key.startswith("bfcl_") and key != "bfcl_outcome"
            },
        }
    return rewards


def compute_bfcl_metrics(batch):
    """Label-free diagnostics, not task accuracy or pass@N."""
    metrics = {
        key: float(np.mean(batch.non_tensor_batch[key]))
        for key in ("bfcl_valid", "bfcl_group_valid", "bfcl_majority_ratio", "bfcl_tied", "bfcl_informative")
    }
    metrics["bfcl_consensus_reward"] = batch.batch["ttrl_bfcl_scores"].sum(-1).mean().item()
    prompt_length = batch.batch["prompts"].shape[-1]
    valid = batch.batch["attention_mask"][:, prompt_length:].bool()
    actions = valid & batch.batch["response_mask"].bool()
    metrics["bfcl_generated_tokens"] = actions.sum(-1).float().mean().item()
    metrics["bfcl_feedback_tokens"] = (valid & ~actions).sum(-1).float().mean().item()
    outcomes = batch.non_tensor_batch.get("bfcl_outcome")
    if outcomes is not None:
        for key in ("num_calls", "parse_errors", "execution_errors"):
            metrics[f"bfcl_{key}"] = float(np.mean([outcome[key] for outcome in outcomes]))
        metrics["bfcl_truncated"] = float(np.mean([outcome["termination"] != "completed" for outcome in outcomes]))
    return metrics
