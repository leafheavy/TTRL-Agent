# Copyright 2025 TTRL Team (https://arxiv.org/abs/2504.16084)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import Counter
from typing import List

import numpy as np

from verl.utils.reward_score.ttrl_consensus import majority_vote_labels


def select_top_k_per_prompt(data, n_votes_per_prompt, n_samples_per_prompt):
    """
    Select the first k rollouts per prompt, used for TTRL downsampling.
    """
    assert len(data) % n_votes_per_prompt == 0, "data length must be divisible by n_votes_per_prompt"
    num_prompts = len(data) // n_votes_per_prompt

    selected_indices = []
    for i in range(num_prompts):
        start = i * n_votes_per_prompt
        selected_indices.extend(range(start, start + n_samples_per_prompt))

    return data[selected_indices]


# === Ground Truth Manipulation ===


def apply_original_gt(batch):
    """
    Apply the original ground truth to the batch.
    """
    for i in range(len(batch)):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"]["original_gt"]
        data_item.non_tensor_batch["reward_model"]["ground_truth"] = original_gt

    return batch


def apply_ttrl_gt(batch, gen_batch_output, n, tokenizer):
    """
    Apply the majority vote ground truth to the batch.
    """
    assert len(gen_batch_output) % n == 0, "gen_batch_output length must be divisible by n"
    num_prompts = len(gen_batch_output) // n
    assert len(batch) == num_prompts, "batch length must be equal to the number of prompts"

    model_outputs = []  
    for i in range(num_prompts):
        start = i * n
        for j in range(n):
            data_item = gen_batch_output[start + j]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            model_outputs.append(response_str)

    majority_gt_list, majority_ratio_list = _batch_majority_vote(model_outputs, n)
    
    assert len(batch) == len(majority_gt_list), "batch length must be equal to the number of model outputs"
    
    for i in range(num_prompts):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        data_item.non_tensor_batch["reward_model"]["ground_truth"] = majority_gt_list[i]
        data_item.non_tensor_batch["reward_model"]["majority_gt"] = majority_gt_list[i]
        data_item.non_tensor_batch["reward_model"]["original_gt"] = original_gt

    batch.non_tensor_batch["majority_ratio_list"] = np.array(majority_ratio_list, dtype=float)
    return batch


def _batch_majority_vote(model_outputs: List[str], n: int) -> tuple[List[str], List[float]]:
    """
    Used to generate the ground truth for TTRL.
    Input:
        model_outputs: list of str
        n: int
    Output:
        majority_gt_list: list of str
        majority_ratio_list: list of float
    """
    majority_gt_list = []
    majority_ratio_list = []
    assert len(model_outputs) % n == 0
    n_prompts = len(model_outputs) // n
    for i in range(n_prompts):
        prompt_outputs = model_outputs[i * n:(i + 1) * n]
        prompt_majority_gt, prompt_majority_ratio = _majority_vote(prompt_outputs)
        majority_gt_list.append(prompt_majority_gt)
        majority_ratio_list.append(prompt_majority_ratio)
        
    return majority_gt_list, majority_ratio_list


def validate_q_weight_config(config):
    if not config.get("enabled", False):
        return
    if config.get("mode", "uniform") not in ("uniform", "linear", "u_shaped"):
        raise ValueError("ttrl.q_weight.mode must be uniform, linear or u_shaped")
    for key, default in (("power", 2.0), ("floor", 0.1), ("min_q", 0.0)):
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"ttrl.q_weight.{key} must be a finite number")
        if (key == "power" and value <= 0) or (key != "power" and not 0 <= value <= 1):
            raise ValueError(f"Invalid ttrl.q_weight.{key}: {value}")


def apply_ttrl_q_weights(batch, config):
    """Weight already-normalized GRPO policy advantages, preserving raw rewards/KL.

    Group membership uses uid, so DP sequence balancing need not keep groups
    contiguous. Diagnostics use observable rewards, never ground-truth labels.
    """
    import torch

    if not config.get("enabled", False):
        return {}
    validate_q_weight_config(config)
    if "ttrl_q_weight" in batch.batch:
        raise ValueError("TTRL q weights must be applied exactly once per training batch")
    q = np.asarray(batch.non_tensor_batch["majority_ratio_list"], dtype=float)
    if len(q) != len(batch) or not np.all(np.isfinite(q)) or np.any((q < 0) | (q > 1)):
        raise ValueError("TTRL requires one finite majority ratio in [0,1] per rollout")
    groups = {}
    for row, uid in enumerate(batch.non_tensor_batch["uid"]):
        groups.setdefault(uid, []).append(row)
    if sum(map(len, groups.values())) != len(batch):
        raise ValueError("TTRL uid must identify every rollout")
    for rows in groups.values():
        if not np.allclose(q[rows], q[rows[0]], rtol=0, atol=1e-12):
            raise ValueError("All rollouts in a TTRL group must have the same q")
    mode = config.get("mode", "uniform")
    if mode == "uniform":
        weights = np.ones_like(q)
    else:
        value = q if mode == "linear" else np.abs(2 * q - 1)
        floor = config.get("floor", 0.1)
        weights = floor + (1 - floor) * value ** config.get("power", 2.0)
    weights *= q >= config.get("min_q", 0.0)
    if config.get("multiply_valid_rate", False):
        valid = np.asarray(batch.non_tensor_batch["bfcl_valid"], dtype=float)
        if valid.shape != q.shape or np.any((valid < 0) | (valid > 1)) or not np.all(np.isfinite(valid)):
            raise ValueError("Valid-rate weighting requires a finite bfcl_valid flag for every rollout")
        for rows in groups.values():
            weights[rows] *= np.mean(valid[rows])
    advantages = batch.batch["advantages"]
    if advantages.shape[0] != len(q):
        raise ValueError("TTRL advantages and rollout metadata must be aligned")
    weight_tensor = torch.as_tensor(weights, dtype=advantages.dtype, device=advantages.device)
    batch.batch["advantages"] = advantages * weight_tensor.unsqueeze(-1)
    batch.batch["ttrl_q_weight"] = weight_tensor

    scores = batch.batch["token_level_scores"].sum(-1).detach().cpu().numpy()
    group_rows = list(groups.values())
    group_q = np.array([q[rows[0]] for rows in group_rows])
    group_weights = np.array([weights[rows[0]] for rows in group_rows])
    variance = np.array([np.var(scores[rows]) for rows in group_rows])
    metrics = {
        "q_weight_mean": float(group_weights.mean()),
        "q_weight_min": float(group_weights.min()),
        "q_weight_max": float(group_weights.max()),
        "q_zero_variance_groups": float(np.mean(variance == 0)),
        "q_active_groups": float(np.mean((variance > 0) & (group_weights > 0))),
    }
    for label, selected in (
        ("low", group_q < 1 / 3),
        ("middle", (group_q >= 1 / 3) & (group_q < 2 / 3)),
        ("high", group_q >= 2 / 3),
    ):
        metrics[f"q_{label}_groups"] = int(selected.sum())
        if selected.any():
            metrics[f"q_{label}_weight"] = float(group_weights[selected].mean())
            metrics[f"q_{label}_reward_variance"] = float(variance[selected].mean())
    mask = batch.batch["response_mask"].to(advantages.dtype)
    denominator = mask.sum().clamp_min(1)
    metrics["q_adv_abs_before"] = ((advantages.abs() * mask).sum() / denominator).item()
    metrics["q_adv_abs_after"] = ((batch.batch["advantages"].abs() * mask).sum() / denominator).item()
    return metrics


def _majority_vote(model_outputs: List[str]) -> tuple[str, float]:
    from verl.utils.reward_score.ttrl_math import extract_answer, simplify_expression_string

    assert len(model_outputs) > 0
    model_answers = [extract_answer(generated_text) for generated_text in model_outputs]
    model_answers = [simplify_expression_string(answer) if answer is not None else None for answer in model_answers]
    majority_answer, majority_ratio = majority_vote_labels(model_answers)
    return majority_answer if majority_answer is not None else "None", majority_ratio


# === Metrics Computation ===


def compute_ttrl_metrics(batch, n):
    """
    Compute the TTRL metrics.
    """
    assert len(batch) % n == 0, "batch length must be divisible by n"

    # Sort the batch by the ID
    idx = sorted(range(len(batch)), key=lambda x: batch[x].non_tensor_batch["extra_info"]["index"])

    majority_reward = []
    gt_reward = []
    majority_label = []
    gt_label = []

    for i in range(len(batch)):
        data_item = batch[idx[i]]
        majority_reward.append(data_item.batch["token_level_scores"].sum().item())
        gt_reward.append(data_item.batch["token_level_scores_original"].sum().item())
        majority_label.append(data_item.non_tensor_batch["reward_model"]["majority_gt"])
        gt_label.append(data_item.non_tensor_batch["reward_model"]["original_gt"]) 

    ttrl_metrics = _batch_compute_ttrl_metrics(majority_reward, gt_reward, majority_label, gt_label, n=n)
    majority_ratio_list = batch.non_tensor_batch["majority_ratio_list"]
    majority_ratio = sum(majority_ratio_list) / len(majority_ratio_list)
    ttrl_metrics["majority_ratio"] = majority_ratio

    return ttrl_metrics


def _batch_compute_ttrl_metrics(
    majority_reward: List[float],
    gt_reward: List[float],
    majority_label: List[str],
    gt_label: List[str],
    n: int,
):
    """
    Compute the TTRL metrics for batch inputs.
    """
    assert len(majority_reward) == len(gt_reward) == len(majority_label) == len(gt_label)
    assert len(majority_reward) % n == 0
    n_prompts = len(majority_reward) // n
    ttrl_metrics = []
    for i in range(n_prompts):
        prompt_majority_reward = majority_reward[i * n:(i + 1) * n]
        prompt_gt_reward = gt_reward[i * n:(i + 1) * n]
        prompt_majority_label = majority_label[i * n:(i + 1) * n]
        prompt_gt_label = gt_label[i * n:(i + 1) * n]

        assert Counter(prompt_majority_label).most_common(1)[0][1] == n
        assert Counter(prompt_gt_label).most_common(1)[0][1] == n

        prompt_majority_label = prompt_majority_label[0]
        prompt_gt_label = prompt_gt_label[0]

        ttrl_metric = _prompt_compute_ttrl_metrics(
            prompt_majority_reward, prompt_gt_reward, prompt_majority_label, prompt_gt_label
        )
        ttrl_metrics.append(ttrl_metric)

    # Compute the average metrics
    ttrl_metrics = {k: sum(d[k] for d in ttrl_metrics) / len(ttrl_metrics) for k in ttrl_metrics[0]}

    return ttrl_metrics

def _prompt_compute_ttrl_metrics(
    majority_reward: List[float],
    gt_reward: List[float],
    majority_label: str,
    gt_label: str,
    ):    
    from verl.utils.reward_score.ttrl_math import grade

    assert len(majority_reward) == len(gt_reward)

    hit_rate = 1.0 if grade(majority_label, gt_label) else 0.0    
    rewards_hit_rate = 0
    for estimate_reward, true_reward in zip(majority_reward, gt_reward):
        if estimate_reward == true_reward:
            rewards_hit_rate += 1
    rewards_hit_rate = rewards_hit_rate / len(majority_reward)
    
    ttrl_metric = {
        "label_accuracy": hit_rate,
        "reward_accuracy": rewards_hit_rate,
        "majority_voting_reward": sum(majority_reward) / len(majority_reward),
        "ground_truth_reward": sum(gt_reward) / len(gt_reward),
        f"pass@{len(majority_reward)}": 1.0 if sum(gt_reward) >= 1 else 0.0,
    }
    return ttrl_metric
