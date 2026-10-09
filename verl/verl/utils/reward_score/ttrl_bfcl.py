"""Training-side BFCL tensors and voting; BFCL library operations use JSON RPC."""

import json

import numpy as np
import torch

from verl.benchmark_runtime.client import RemoteSession, RuntimeClient
from verl.utils.reward_score.ttrl_consensus import consensus_key, validate_consensus_mode, vote_consensus


def decode_bfcl_context(context):
    """Accept dicts or JSON strings, avoiding Arrow's union of arbitrary schemas."""
    if isinstance(context, str):
        context = json.loads(context)
    if not isinstance(context, dict):
        raise ValueError("extra_info.bfcl must be a context object or its JSON serialization")
    return context


class BFCLSession(RemoteSession):
    """BFCL rollout proxy; the simulator lives in the separate BFCL runtime."""

    def __init__(self, context, runtime_config=None):
        super().__init__(decode_bfcl_context(context), runtime_config)


class BFCLSupervisedReward:
    """Keep tensors in the trainer; request official scores from the BFCL process."""

    def __init__(self, answer_dir=None, runtime_config=None):
        self.answer_dir = str(answer_dir) if answer_dir else None
        self.runtime_config = dict(runtime_config or {})

    def score_outcome(self, item, outcome):
        return RuntimeClient.from_config(self.runtime_config).call(
            "score_outcomes", items=[item], outcomes=[outcome], answer_dir=self.answer_dir
        )[0]

    def __call__(self, data, return_dict=False):
        outcomes = data.non_tensor_batch.get("bfcl_outcome")
        if outcomes is None or len(outcomes) != len(data):
            raise ValueError("BFCL supervised reward requires one AgentLoop outcome per rollout")
        rewards = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        prompt_length = data.batch["prompts"].shape[-1]
        rows, positions = [], []
        for row in range(len(data)):
            mask = data.batch["response_mask"][row].bool() & data.batch["attention_mask"][row, prompt_length:].bool()
            indices = torch.nonzero(mask, as_tuple=True)[0]
            if len(indices):
                rows.append(row)
                positions.append(indices[-1])
        if rows:
            scores = RuntimeClient.from_config(self.runtime_config).call(
                "score_outcomes",
                items=[data.non_tensor_batch["extra_info"][row] for row in rows],
                outcomes=[outcomes[row] for row in rows],
                answer_dir=self.answer_dir,
            )
            if len(scores) != len(rows):
                raise ValueError("BFCL runtime returned an unexpected number of supervised scores")
            for row, position, score in zip(rows, positions, scores):
                rewards[row, position] = score
        return {"reward_tensor": rewards, "reward_extra_info": {}} if return_dict else rewards


def validate_bfcl_supervised_config(config):
    """The labeled baseline uses the same multi-turn AgentLoop, with one rollout group."""
    if config.get("ttrl", {}).get("enable", False):
        raise ValueError("BFCL supervised reward and TTRL must be run separately")
    if config.get("ttrl", {}).get("q_weight", {}).get("enabled", False):
        raise ValueError("Disable TTRL q weighting for the BFCL supervised GRPO baseline")
    rollout = config.actor_rollout_ref.rollout
    if not rollout.multi_turn.enable or rollout.get("mode", "sync") != "async":
        raise ValueError("BFCL supervised Agent GRPO requires multi_turn.enable=True and rollout.mode=async")
    if not config.data.get("return_raw_chat", False):
        raise ValueError("BFCL supervised AgentLoop requires data.return_raw_chat=True")
    if (
        config.algorithm.adv_estimator != "grpo"
        or isinstance(rollout.n, bool)
        or not isinstance(rollout.n, int)
        or rollout.n < 2
    ):
        raise ValueError("BFCL supervised training requires GRPO with rollout.n >= 2")
    if config.get("reward_model", {}).get("enable", False):
        raise ValueError("BFCL official pass/fail cannot be replaced by a reward model")
    if config.trainer.get("val_before_train", True) or config.trainer.get("test_freq", -1) > 0:
        raise ValueError(
            "BFCL Agent runs require val_before_train=False and test_freq=-1; evaluate the frozen model with BFCL CLI"
        )


def apply_bfcl_rewards(batch, gen_batch_output, n, tokenizer, runtime_config=None):
    """Attach a token reward tensor to generated outputs; never read reward_model."""
    if len(gen_batch_output) != len(batch) * n:
        raise ValueError("BFCL rollout count must equal the number of prompts times rollout.n")
    outcomes = gen_batch_output.non_tensor_batch.get("bfcl_outcome")
    runtime = RuntimeClient.from_config(runtime_config) if outcomes is None else None
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
                calls = runtime.call("decode_responses", context=context, responses=[response])[0]
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
