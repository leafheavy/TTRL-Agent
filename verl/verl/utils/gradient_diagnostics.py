"""Compare independent optimization trajectories and legacy policy gradients.

Run from the existing verl environment::

    python -m verl.utils.gradient_diagnostics \
        --supervised /path/to/supervised/gradients \
        --unsupervised /path/to/ttrl/gradients --output /path/to/diagnosis

Legacy full mode stores each selected optimizer minibatch's rank shards as
float32 binary arrays with JSON metadata. Offline comparisons stream those
arrays rather than gathering the full model or averaging per-rank cosines.

Paired trajectories compare actual optimizer paths against an independent
supervised branch, optionally with shuffled-label and geometric random-update
baselines sharing the same reference and origin. Legacy single-step paired
diagnostics retain exact full-coordinate gradients only in CPU memory.
Export its records with ``--paired /path/to/gradients --output /path/to/diagnosis``.
"""

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import uuid
from pathlib import Path

import numpy as np
import torch

CHUNK_SIZE = 1 << 20
RANDOM_BLOCK_SIZE = 1 << 20
BATCH_KEYS = ("input_ids", "attention_mask", "position_ids", "responses", "response_mask", "old_log_probs")


def diagnostic_seed(seed, *parts):
    """Derive a stable seed without consuming any training random state."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("Gradient diagnostic seed must be an integer")
    payload = json.dumps([int(seed), *parts], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def shuffle_outcome_rewards(rewards, index, *, seed):
    """Shuffle official trajectory labels within each prompt before GRPO.

    GRPO consumes only the sum of token rewards, so returning one reward column
    avoids moving a source trajectory's token positions or response mask. A
    random permutation may leave labels unchanged, especially for uniform
    groups; the metadata exposes that degeneracy instead of forcing a change.
    """
    if rewards.ndim != 2 or len(index) != rewards.shape[0]:
        raise ValueError("Shuffled labels require a reward matrix and one prompt UID per trajectory")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("Shuffled label seed must be a non-negative integer")
    scores = rewards.detach().sum(dim=-1)
    if not torch.isfinite(scores).all():
        raise ValueError("Shuffled labels require finite trajectory rewards")
    shuffled = scores.clone()
    groups = {}
    for row, uid in enumerate(index):
        groups.setdefault(uid, []).append(row)
    rng = np.random.default_rng(int(seed))
    informative_groups = changed_groups = changed_labels = 0
    for rows in groups.values():
        positions = torch.as_tensor(rows, dtype=torch.long, device=scores.device)
        labels = scores.index_select(0, positions)
        permutation = torch.as_tensor(rng.permutation(len(rows)), dtype=torch.long, device=scores.device)
        permuted = labels.index_select(0, permutation)
        shuffled[positions] = permuted
        informative_groups += int(torch.unique(labels).numel() > 1)
        changes = int(torch.count_nonzero(labels != permuted).item())
        changed_groups += int(changes > 0)
        changed_labels += changes
    metadata = {
        "seed": int(seed),
        "label_source": "official_bfcl_outcome_reward",
        "shuffle_scope": "within_prompt_uid",
        "group_count": len(groups),
        "informative_group_count": informative_groups,
        "changed_group_count": changed_groups,
        "changed_label_count": changed_labels,
        "label_count": int(scores.numel()),
    }
    return shuffled.unsqueeze(-1), metadata


def recording_due(config, step):
    return bool(config.get("enabled", False)) and (step == 1 or step % config.get("interval", 10) == 0)


def paired_recording_due(config, step):
    return (
        config.get("mode", "full") == "paired"
        and config.get("single_step", {}).get("enabled", True)
        and recording_due(config, step)
    )


def validate_gradient_recording(config):
    """Fail early instead of silently recording a regularized or unsupported loss."""
    actor = config.actor_rollout_ref.actor
    record = actor.get("gradient_record", {})
    if not record.get("enabled", False):
        return
    interval = record.get("interval", 10)
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
        raise ValueError("actor.gradient_record.interval must be a positive integer")
    if not record.get("output_dir"):
        raise ValueError("actor.gradient_record.output_dir is required when enabled")
    mode = record.get("mode", "full")
    if mode not in ("full", "paired"):
        raise ValueError("actor.gradient_record.mode must be full or paired")
    trajectory = record.get("trajectory", {}).get("enabled", False)
    if trajectory and (mode != "paired" or actor.strategy != "fsdp"):
        raise ValueError("Optimization trajectory diagnostics require mode=paired and strategy=fsdp (FSDP1)")
    if mode == "paired" and not trajectory and not record.get("single_step", {}).get("enabled", True):
        raise ValueError("Paired diagnostics require trajectory.enabled=True or single_step.enabled=True")
    seed = record.get("baselines", {}).get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("gradient_record.baselines.seed must be a non-negative integer")
    if mode == "paired":
        ttrl = config.get("ttrl", {})
        if not ttrl.get("enable", False) or ttrl.get("reward_mode", "math") != "bfcl":
            raise ValueError("Paired gradient diagnostics require ttrl.enable=True and ttrl.reward_mode=bfcl")
        if not ttrl.get("share_rollouts", False):
            raise ValueError("Paired gradient diagnostics require ttrl.share_rollouts=True")
        if config.get("bfcl_supervised", {}).get("enabled", False):
            raise ValueError("Paired gradient diagnostics score supervision separately from the TTRL update")
        rollout = config.actor_rollout_ref.rollout
        if not rollout.multi_turn.enable or rollout.get("mode", "sync") != "async":
            raise ValueError("Paired gradient diagnostics require multi_turn.enable=True and rollout.mode=async")
        if not config.data.get("return_raw_chat", False):
            raise ValueError("Paired gradient diagnostics require data.return_raw_chat=True")
        if isinstance(rollout.n, bool) or not isinstance(rollout.n, int) or rollout.n < 2:
            raise ValueError("Paired gradient diagnostics require rollout.n >= 2")
    if config.algorithm.adv_estimator != "grpo":
        raise ValueError("Policy gradient recording currently requires algorithm.adv_estimator=grpo")
    if config.algorithm.get("use_kl_in_reward", False):
        raise ValueError("Pure policy gradient recording requires algorithm.use_kl_in_reward=False")
    if actor.strategy not in ("fsdp", "fsdp2"):
        raise ValueError("Policy gradient recording currently supports the FSDP actor only")
    if actor.policy_loss.get("loss_mode", "vanilla") != "vanilla":
        raise ValueError("Pure policy gradient recording currently requires policy_loss.loss_mode=vanilla")
    if actor.get("ulysses_sequence_parallel_size", 1) != 1:
        raise ValueError("Policy gradient recording currently requires ulysses_sequence_parallel_size=1")
    world_size = config.trainer.n_gpus_per_node * config.trainer.nnodes
    fsdp_size = actor.fsdp_config.get("fsdp_size", -1)
    if 0 < fsdp_size < world_size:
        raise ValueError("Policy gradient recording requires full FSDP sharding, without replicated shard groups")
    if world_size > 1 and actor.strategy == "fsdp" and actor.fsdp_config.get("use_orig_params", False):
        raise ValueError("Multi-GPU FSDP gradient recording currently requires use_orig_params=False")


def _local(tensor):
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def _cpu_chunks(tensor):
    flat = _local(tensor).detach().reshape(-1)
    for start in range(0, flat.numel(), CHUNK_SIZE):
        yield flat[start : start + CHUNK_SIZE].to("cpu").contiguous()


def _hash_tensor(digest, tensor):
    digest.update(json.dumps([str(tensor.dtype), list(tensor.shape)]).encode())
    for chunk in _cpu_chunks(tensor):
        digest.update(chunk.view(torch.uint8).numpy().tobytes())


def batch_fingerprint(batch):
    digest = hashlib.sha256()
    for name in BATCH_KEYS:
        digest.update(name.encode())
        _hash_tensor(digest, batch[name])
    return digest.hexdigest()


def tensor_fingerprint(tensor):
    digest = hashlib.sha256()
    _hash_tensor(digest, tensor)
    return digest.hexdigest()


def _parameter_layout(name, parameter):
    local = _local(parameter)
    item = {"name": name, "shape": list(parameter.shape), "local_shape": list(local.shape), "numel": local.numel()}
    if hasattr(parameter, "_fqns"):
        # A flat tensor's size alone cannot establish its original parameter order.
        item["flat_names"] = list(parameter._fqns)
        item["flat_shapes"] = [list(shape) for shape in parameter._shapes]
    if hasattr(parameter, "placements"):
        item["placements"] = [str(placement) for placement in parameter.placements]
        item["mesh"] = parameter.device_mesh.mesh.tolist()
    return item


def _parameter_fingerprint(module):
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        digest.update(json.dumps(_parameter_layout(name, parameter), sort_keys=True).encode())
        _hash_tensor(digest, parameter)
    return digest.hexdigest()


def _gradient_chunks(parameter, name):
    gradient = parameter.grad
    if gradient is None:
        return
    gradient = _local(gradient)
    if gradient.shape != _local(parameter).shape:
        raise ValueError(f"Gradient shard shape differs from parameter shard: {name}")
    if gradient.is_sparse or not gradient.is_floating_point():
        raise ValueError(f"Expected a dense real floating-point policy gradient: {name}")
    for chunk in _cpu_chunks(gradient):
        if not torch.isfinite(chunk).all():
            raise ValueError(f"Non-finite pure policy gradient: {name}")
        yield chunk


def capture_policy_gradient(module):
    """Copy one reduced gradient to CPU without narrowing its dtype or writing it.

    Missing gradients represent zero coordinates and need no CPU allocation.
    The parameter fingerprint also checks that the second backward uses exactly
    the same model parameters, including frozen parameters.
    """
    parameters = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        layout = _parameter_layout(name, parameter)
        gradient = None
        if parameter.grad is not None:
            gradient = torch.empty(layout["numel"], dtype=_local(parameter.grad).dtype, device="cpu")
            for start, chunk in zip(range(0, layout["numel"], CHUNK_SIZE), _gradient_chunks(parameter, name)):
                gradient[start : start + chunk.numel()].copy_(chunk)
        parameters.append({"layout": layout, "gradient": gradient})
    return {
        "parameters": parameters,
        "parameter_fingerprint": _parameter_fingerprint(module),
        "numel": sum(item["layout"]["numel"] for item in parameters),
    }


def paired_gradient_statistics(module, cache):
    """Return local [dot, supervised squared norm, unsupervised squared norm].

    ``cache`` is the supervised gradient; the module currently holds the second
    gradient. All original coordinates participate, with float64 accumulation.
    """
    parameters = [(name, parameter) for name, parameter in module.named_parameters() if parameter.requires_grad]
    if len(parameters) != len(cache["parameters"]):
        raise ValueError("Paired gradient parameter layouts differ")
    if _parameter_fingerprint(module) != cache["parameter_fingerprint"]:
        raise ValueError("Paired gradients require unchanged model parameters")
    statistics = torch.zeros(3, dtype=torch.float64)
    for (name, parameter), item in zip(parameters, cache["parameters"]):
        if _parameter_layout(name, parameter) != item["layout"]:
            raise ValueError(f"Paired gradient parameter layouts differ: {name}")
        left = item["gradient"]
        right_chunks = iter(_gradient_chunks(parameter, name))
        for start in range(0, item["layout"]["numel"], CHUNK_SIZE):
            x = left[start : start + CHUNK_SIZE].double() if left is not None else None
            chunk = next(right_chunks, None)
            y = chunk.double() if chunk is not None else None
            if x is not None:
                statistics[1] += torch.dot(x, x)
            if y is not None:
                statistics[2] += torch.dot(y, y)
            if x is not None and y is not None:
                statistics[0] += torch.dot(x, y)
    if not torch.isfinite(statistics).all():
        raise ValueError("Non-finite paired gradient statistics")
    return statistics


def random_gradient_statistics(cache, *, seed, rank=0):
    """An isotropic Gaussian direction over *all* trainable coordinates.

    Independent fixed-size blocks keep this draw separate from training RNG
    and from the chunk size used for gradient IO. Normalize after rank reduce.
    """
    statistics = torch.zeros(3, dtype=torch.float64)
    for index, item in enumerate(cache["parameters"]):
        left = item["gradient"]
        for start in range(0, item["layout"]["numel"], RANDOM_BLOCK_SIZE):
            size = min(RANDOM_BLOCK_SIZE, item["layout"]["numel"] - start)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(diagnostic_seed(seed, "gaussian", rank, index, start // RANDOM_BLOCK_SIZE))
            random = torch.randn(size, generator=generator, dtype=torch.float64)
            statistics[2] += torch.dot(random, random)
            if left is not None:
                values = left[start : start + size].double()
                statistics[0] += torch.dot(values, random)
                statistics[1] += torch.dot(values, values)
    return statistics


def _comparison_statistics(dot, left_sq, right_sq):
    if not all(math.isfinite(value) for value in (dot, left_sq, right_sq)) or min(left_sq, right_sq) < 0:
        raise ValueError("Invalid gradient inner product or squared norms")
    cosine = angle = None
    status = "ok"
    if left_sq == 0 or right_sq == 0:
        status = "both_zero" if left_sq == right_sq == 0 else "supervised_zero" if left_sq == 0 else "unsupervised_zero"
    else:
        cosine = float(np.clip(dot / (math.sqrt(left_sq) * math.sqrt(right_sq)), -1, 1))
        angle = math.degrees(math.acos(cosine))
    return {
        "dot_product": dot,
        "supervised_squared_norm": left_sq,
        "unsupervised_squared_norm": right_sq,
        "cosine": cosine,
        "angle_deg": angle,
        "supervised_norm": math.sqrt(left_sq),
        "unsupervised_norm": math.sqrt(right_sq),
        "status": status,
    }


def save_paired_gradient_statistics(
    statistics, batch, metadata, directory, *, cache, rank=0, world_size=1, device=None,
    baselines=None, baseline_metadata=None,
):
    """Reduce only three float64 scalars and write one small atomic JSON record.

    Every rank must call this function. ``device`` must be the actor's local
    CUDA device when its process group uses NCCL; CPU/Gloo needs no override.
    """
    statistics = statistics.detach().to(device="cpu" if device is None else device, dtype=torch.float64).clone()
    if statistics.shape != (3,) or not torch.isfinite(statistics).all():
        raise ValueError("Expected three finite paired gradient statistics")
    fingerprint = {
        "rank": rank,
        "numel": cache["numel"],
        "parameter_fingerprint": cache["parameter_fingerprint"],
        "batch_fingerprint": batch_fingerprint(batch),
        "rng_fingerprint": metadata.get("rng_fingerprint"),
        "supervised_advantage_fingerprint": tensor_fingerprint(batch["diagnostic_supervised_advantages"]),
        "unsupervised_advantage_fingerprint": tensor_fingerprint(batch["advantages"]),
    }
    fingerprints = [fingerprint]
    if world_size > 1:
        torch.distributed.all_reduce(statistics, op=torch.distributed.ReduceOp.SUM)
        fingerprints = [None] * world_size
        torch.distributed.all_gather_object(fingerprints, fingerprint)
    baseline_results = {}
    for name, values in (baselines or {}).items():
        values = values.detach().to(device="cpu" if device is None else device, dtype=torch.float64).clone()
        if values.shape != (3,) or not torch.isfinite(values).all():
            raise ValueError("Expected three finite baseline gradient statistics")
        if world_size > 1:
            torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
        dot, left_sq, right_sq = values.cpu().tolist()
        if name == "random_direction" and right_sq > 0:
            dot /= math.sqrt(right_sq)
            right_sq = 1.0
        baseline_results[name] = _comparison_statistics(dot, left_sq, right_sq)
    fingerprints.sort(key=lambda item: item["rank"])
    if [item["rank"] for item in fingerprints] != list(range(world_size)):
        raise ValueError("Missing or duplicated paired gradient rank metadata")
    key = f"step_{metadata['step']:06d}_epoch_{metadata['epoch']:03d}_minibatch_{metadata['minibatch']:04d}"
    result = {
        **metadata,
        "version": 2,
        "mode": "paired",
        "update": key,
        "world_size": world_size,
        "gradient": "policy_loss_gradient_before_regularizers_and_grad_clip",
        "accumulation_dtype": "float64",
        "numel": sum(item["numel"] for item in fingerprints),
        "rank_fingerprints": fingerprints,
        "comparison": "matched_conditions",
        **_comparison_statistics(*statistics.cpu().tolist()),
    }
    if baseline_results:
        result["baselines"] = baseline_results
        result["baseline_metadata"] = baseline_metadata or {}
    if rank == 0:
        _write_statistics(Path(directory) / f"{key}.json", result)
    return result


def _write_statistics(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Diagnostic statistics already exist: {path}; use a separate run directory")
    temporary = path.with_suffix(f".json.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class _CPUTensorState:
    """Preserve optimizer tensor devices (including CPU Adam step counters)."""

    def __init__(self, tensor):
        if hasattr(tensor, "placements"):
            raise ValueError("Optimization trajectories currently support FSDP1 tensor states, not DTensor")
        self.device = tensor.device
        self.value = tensor.detach().to("cpu", copy=True)


def _snapshot_state(value):
    if isinstance(value, torch.Tensor):
        return _CPUTensorState(value)
    if isinstance(value, dict):
        result = copy.copy(value)
        for key, item in value.items():
            result[key] = _snapshot_state(item)
        return result
    if isinstance(value, (list, tuple)):
        return type(value)(_snapshot_state(item) for item in value)
    return copy.deepcopy(value)


def _restore_state(value):
    if isinstance(value, _CPUTensorState):
        return value.value.to(value.device, copy=True)
    if isinstance(value, dict):
        result = copy.copy(value)
        for key, item in value.items():
            result[key] = _restore_state(item)
        return result
    if isinstance(value, (list, tuple)):
        return type(value)(_restore_state(item) for item in value)
    return copy.deepcopy(value)


class OptimizationTrajectory:
    """Independent optimizer branches with one shared origin and live snapshot.

    Optional references comprise a shuffled-label optimizer branch and a
    geometric random-update path, whose intended global step L2 matches TTRL.
    The latter has no optimizer: its FP32 displacement is accumulated directly.
    FSDP1 local shards suffice; no model or update vector is gathered.
    """

    def __init__(self, module, optimizer, *, baselines=False, seed=0):
        diagnostic_seed(seed)
        if seed < 0:
            raise ValueError("Optimization trajectory seed must be non-negative")
        self.baselines = bool(baselines)
        self.seed = int(seed)
        self.layouts = [_parameter_layout(name, parameter) for name, parameter in module.named_parameters()
                        if parameter.requires_grad]
        self.initial_parameters = self._parameters(module)
        self.supervised_parameters = self.initial_parameters
        self.supervised_buffers = self._buffers(module)
        self.supervised_optimizer_state = _snapshot_state(optimizer.state)
        self.origin_fingerprint = _parameter_fingerprint(module)
        self.pending = None
        self.updates = 0
        self.path_statistics = torch.zeros(3, dtype=torch.float64)
        self.start_update = None
        if self.baselines:
            # Initial values are immutable CPU snapshots, shared by reference.
            # Each branch receives its own values only after its first update.
            self.shuffled_parameters = self.initial_parameters
            self.shuffled_buffers = self.supervised_buffers
            self.shuffled_optimizer_state = self.supervised_optimizer_state
            self.random_displacement = [torch.zeros_like(value, dtype=torch.float32)
                                        for value in self.initial_parameters]
            self.baseline_path_statistics = {
                name: torch.zeros(3, dtype=torch.float64) for name in ("shuffled_labels", "random_update")
            }

    def _parameters(self, module):
        parameters = [(name, p) for name, p in module.named_parameters() if p.requires_grad]
        if [_parameter_layout(name, p) for name, p in parameters] != self.layouts:
            raise ValueError("Optimization trajectory parameter layouts changed")
        if any(hasattr(p, "placements") for _, p in parameters):
            raise ValueError("Optimization trajectories currently require FSDP1, not DTensor parameters")
        return [_local(p).detach().to("cpu", copy=True) for _, p in parameters]

    @staticmethod
    def _buffers(module):
        return {name: buffer.detach().to("cpu", copy=True) for name, buffer in module.named_buffers()}

    @staticmethod
    def _load(module, parameters, buffers):
        with torch.no_grad():
            live = [p for p in module.parameters() if p.requires_grad]
            if len(live) != len(parameters):
                raise ValueError("Optimization trajectory parameter count changed")
            for parameter, value in zip(live, parameters):
                _local(parameter).copy_(value)
            current_buffers = dict(module.named_buffers())
            if current_buffers.keys() != buffers.keys():
                raise ValueError("Optimization trajectory buffer names changed")
            for name, value in buffers.items():
                current_buffers[name].copy_(value)

    @staticmethod
    def _restore_groups(optimizer, groups):
        for group, original in zip(optimizer.param_groups, groups):
            for key in list(group):
                if key != "params":
                    del group[key]
            group.update(copy.deepcopy(original))

    def supervised_step(
        self, module, optimizer, callback, *, shuffled_callback=None, rank=0, world_size=1, device=None
    ):
        """Advance shadow branches transactionally, preserving live state/RNG.

        Both callbacks start at the same RNG state and current param-group
        settings, including the live scheduler's LR. Parameters, buffers and
        Adam history evolve independently. The live branch is copied only once;
        failure in either callback commits neither shadow branch.
        """
        if self.pending is not None:
            raise ValueError("The preceding real optimizer step has not been recorded")
        if self.baselines != (shuffled_callback is not None):
            raise ValueError("Trajectory baselines require exactly one shuffled-label update callback")
        live_parameters = self._parameters(module)
        live_buffers = self._buffers(module)
        live_state = _snapshot_state(optimizer.state)
        groups = [{key: copy.deepcopy(value) for key, value in group.items() if key != "params"}
                  for group in optimizer.param_groups]
        previous_supervised = self.supervised_parameters
        previous_shuffled = self.shuffled_parameters if self.baselines else None
        branches = [(callback, self.supervised_parameters, self.supervised_buffers, self.supervised_optimizer_state)]
        if self.baselines:
            branches.append((shuffled_callback, self.shuffled_parameters,
                             self.shuffled_buffers, self.shuffled_optimizer_state))
        updated = []
        live_devices = {parameter.device for parameter in module.parameters() if parameter.device.type != "cpu"}
        device_type = next(iter(live_devices)).type if live_devices else "cuda"
        devices = sorted({item.index for item in live_devices})
        try:
            for update, parameters, buffers, state in branches:
                optimizer.zero_grad()
                self._load(module, parameters, buffers)
                optimizer.state.clear()
                optimizer.state = _restore_state(state)
                self._restore_groups(optimizer, groups)
                with torch.random.fork_rng(devices=devices, device_type=device_type):
                    update()
                updated.append((self._parameters(module), self._buffers(module), _snapshot_state(optimizer.state)))
        finally:
            optimizer.zero_grad()
            self._load(module, live_parameters, live_buffers)
            optimizer.state.clear()
            optimizer.state = _restore_state(live_state)
            self._restore_groups(optimizer, groups)
        self.supervised_parameters, self.supervised_buffers, self.supervised_optimizer_state = updated[0]
        if self.baselines:
            self.shuffled_parameters, self.shuffled_buffers, self.shuffled_optimizer_state = updated[1]
        self.pending = (live_parameters, previous_supervised, previous_shuffled)

    def _random_chunks(self, key, rank):
        """Use fixed blocks and rank-separated CPU RNG, independent of IO chunks."""
        for index, layout in enumerate(self.layouts):
            for start in range(0, layout["numel"], RANDOM_BLOCK_SIZE):
                size = min(RANDOM_BLOCK_SIZE, layout["numel"] - start)
                generator = torch.Generator(device="cpu")
                generator.manual_seed(diagnostic_seed(
                    self.seed, "trajectory_random_update", key, rank, index, start // RANDOM_BLOCK_SIZE
                ))
                yield index, start, torch.randn(size, generator=generator, dtype=torch.float64)

    @staticmethod
    def _add_statistics(statistics, left, right):
        statistics += torch.stack((torch.dot(left, right), torch.dot(left, left), torch.dot(right, right)))

    def finish_step(self, module, metadata, *, rank=0, world_size=1, device=None):
        """Compare actual endpoint displacements and concatenated step deltas."""
        if self.pending is None:
            raise ValueError("A supervised optimizer step is required before the real step")
        live_before, supervised_before, shuffled_before = self.pending
        endpoint = torch.zeros(3, dtype=torch.float64)
        increment = torch.zeros(3, dtype=torch.float64)
        baseline_endpoints = {name: torch.zeros(3, dtype=torch.float64)
                              for name in self.baseline_path_statistics} if self.baselines else {}
        baseline_increments = {name: torch.zeros(3, dtype=torch.float64) for name in baseline_endpoints}
        parameters = [(name, p) for name, p in module.named_parameters() if p.requires_grad]
        if [_parameter_layout(name, p) for name, p in parameters] != self.layouts:
            raise ValueError("Optimization trajectory parameter layouts changed")
        for index, (_, parameter) in enumerate(parameters):
            origin = self.initial_parameters[index].reshape(-1)
            supervised = self.supervised_parameters[index].reshape(-1)
            previous_s = supervised_before[index].reshape(-1)
            previous_u = live_before[index].reshape(-1)
            for start, current in zip(range(0, origin.numel(), CHUNK_SIZE), _cpu_chunks(parameter)):
                stop = start + current.numel()
                x = supervised[start:stop].double() - origin[start:stop].double()
                y = current.double() - origin[start:stop].double()
                dx = supervised[start:stop].double() - previous_s[start:stop].double()
                dy = current.double() - previous_u[start:stop].double()
                self._add_statistics(endpoint, x, y)
                self._add_statistics(increment, dx, dy)
                if self.baselines:
                    shuffled = self.shuffled_parameters[index].reshape(-1)[start:stop].double()
                    previous = shuffled_before[index].reshape(-1)[start:stop].double()
                    self._add_statistics(baseline_endpoints["shuffled_labels"], x,
                                         shuffled - origin[start:stop].double())
                    self._add_statistics(baseline_increments["shuffled_labels"], dx, shuffled - previous)
        if not torch.isfinite(endpoint).all() or not torch.isfinite(increment).all():
            raise ValueError("Non-finite optimization trajectory statistics")
        key = f"step_{metadata['step']:06d}_epoch_{metadata['epoch']:03d}_minibatch_{metadata['minibatch']:04d}"
        random_norms = None
        if self.baselines:
            # The first collective defines a single full-coordinate scale.
            # Per-rank normalization would bias the global random direction.
            norms = torch.zeros(2, dtype=torch.float64)
            norms[0] = increment[2]
            for _, _, random in self._random_chunks(key, rank):
                norms[1] += torch.dot(random, random)
            norms = norms.to("cpu" if device is None else device)
            if world_size > 1:
                torch.distributed.all_reduce(norms, op=torch.distributed.ReduceOp.SUM)
            target_sq, gaussian_sq = norms.cpu().tolist()
            if not all(math.isfinite(value) and value >= 0 for value in (target_sq, gaussian_sq)):
                raise ValueError("Non-finite random update normalization")
            scale = math.sqrt(target_sq / gaussian_sq) if gaussian_sq > 0 else 0.0
            if target_sq > 0 and gaussian_sq == 0:
                raise ValueError("A nonzero TTRL update requires a nonzero random direction")
            applied_sq = torch.zeros((), dtype=torch.float64)
            for index, start, random in self._random_chunks(key, rank):
                stop = start + random.numel()
                displacement = self.random_displacement[index].reshape(-1)[start:stop]
                previous = displacement.double()
                displacement.add_((random * scale).float())
                applied = displacement.double() - previous
                supervised = self.supervised_parameters[index].reshape(-1)[start:stop].double()
                origin = self.initial_parameters[index].reshape(-1)[start:stop].double()
                previous_s = supervised_before[index].reshape(-1)[start:stop].double()
                self._add_statistics(baseline_endpoints["random_update"], supervised - origin, displacement.double())
                self._add_statistics(baseline_increments["random_update"], supervised - previous_s, applied)
                applied_sq += torch.dot(applied, applied)
            random_norms = (target_sq, applied_sq)
        self.path_statistics += increment
        self.updates += 1
        if self.start_update is None:
            self.start_update = key
        pieces = [endpoint, self.path_statistics]
        for name in baseline_endpoints:
            self.baseline_path_statistics[name] += baseline_increments[name]
            pieces.extend((baseline_endpoints[name], self.baseline_path_statistics[name]))
        if self.baselines:
            pieces.append(random_norms[1].reshape(1))
        values = torch.cat(pieces)
        if not torch.isfinite(values).all():
            raise ValueError("Non-finite optimization trajectory statistics")
        values = values.to("cpu" if device is None else device)
        if world_size > 1:
            torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
        values = values.cpu().tolist()
        self.pending = None
        result = {
            "version": 2 if self.baselines else 1,
            "mode": "optimization_trajectory",
            "reference": "independent_supervised_grpo_parameters_and_optimizer_state",
            "data": "shared_ttrl_rollouts_and_old_log_probs",
            "update_objective": "training_loss_including_regularizers_and_gradient_clipping",
            "trajectory_sampling": "every_optimizer_minibatch",
            "segment": "since_first_diagnostic_update_in_this_process",
            "update": key,
            **metadata,
            "updates": self.updates,
            "start_update": self.start_update,
            "end_update": key,
            "world_size": world_size,
            "origin_parameter_fingerprint_local": self.origin_fingerprint,
            "endpoint": _comparison_statistics(*values[:3]),
            "path": _comparison_statistics(*values[3:6]),
        }
        # Preserve the legacy top-level U-vs-S fields and reuse the same S
        # reference for both baselines, rather than comparing U with noise.
        if self.baselines:
            result["baselines"] = {}
            for index, name in enumerate(baseline_endpoints):
                offset = 6 + 6 * index
                result["baselines"][name] = {
                    "reference": "same_independent_supervised_optimizer_trajectory",
                    "endpoint": _comparison_statistics(*values[offset:offset + 3]),
                    "path": _comparison_statistics(*values[offset + 3:offset + 6]),
                }
            target = math.sqrt(random_norms[0])
            applied = math.sqrt(values[-1])
            result["baselines"]["shuffled_labels"]["update_rule"] = "independent_training_optimizer_and_history"
            result["baselines"]["random_update"].update({
                "update_rule": "isotropic_gaussian_displacement_without_optimizer",
                "step_norm_target": "ttrl_actual_global_step_l2_before_float32_accumulation_rounding",
                "target_step_norm": target,
                "applied_step_norm": applied,
                "step_norm_relative_error": abs(applied - target) / target if target else 0.0,
                "endpoint_accumulation_dtype": "float32",
                "path_increment": "actual_difference_of_float32_cumulative_displacement",
                "rounding": "float32_accumulation_can_change_step_norm_especially_for_tiny_updates",
            })
            result["baseline_metadata"] = {
                "seed": self.seed, "draws_per_update": 1,
                "reference": "same_independent_supervised_optimizer_trajectory",
                "random_coordinates": "all_trainable_coordinates_with_global_norm_and_rank_separated_cpu_rng",
                "statistics_accumulation_dtype": "float64",
            }
        return result


def save_trajectory_statistics(result, directory, *, rank=0):
    if rank == 0:
        _write_statistics(Path(directory) / f"trajectory_{result['update']}.json", result)


def save_policy_gradient(module, batch, metadata, directory, *, rank=0, world_size=1):
    """Save accumulated, reduced .grad before clipping; None gradients are zero.

    The caller performs a policy-only backward with the existing masks, clipping
    objective and microbatch scaling. This function never changes gradients or
    parameters and stores no rewards, reference answers or optimizer state.
    """
    key = f"step_{metadata['step']:06d}_epoch_{metadata['epoch']:03d}_minibatch_{metadata['minibatch']:04d}"
    destination = Path(directory) / key
    destination.mkdir(parents=True, exist_ok=True)
    stem = destination / f"rank_{rank:05d}"
    binary_path, json_path = stem.with_suffix(".bin"), stem.with_suffix(".json")
    if binary_path.exists() or json_path.exists():
        raise FileExistsError(f"Gradient snapshot already exists: {stem}; use a separate run directory")
    temporary = stem.with_suffix(f".bin.{uuid.uuid4().hex}.tmp")
    json_temporary = temporary.with_suffix(".json.tmp")
    parameters = []
    digest = hashlib.sha256()
    total = 0
    try:
        with temporary.open("xb") as stream:
            for name, parameter in module.named_parameters():
                layout = _parameter_layout(name, parameter)
                digest.update(json.dumps(layout, sort_keys=True).encode())
                _hash_tensor(digest, parameter)
                if not parameter.requires_grad:
                    continue
                layout["offset"] = total
                parameters.append(layout)
                local = _local(parameter)
                gradient = parameter.grad
                if gradient is not None:
                    gradient = _local(gradient)
                    if gradient.shape != local.shape:
                        raise ValueError(f"Gradient shard shape differs from parameter shard: {name}")
                    chunks = (chunk.float().numpy() for chunk in _cpu_chunks(gradient))
                else:
                    chunks = (
                        np.zeros(min(CHUNK_SIZE, local.numel() - start), dtype=np.float32)
                        for start in range(0, local.numel(), CHUNK_SIZE)
                    )
                for array in chunks:
                    if not np.isfinite(array).all():
                        raise ValueError(f"Non-finite pure policy gradient: {key}/{name}")
                    stream.write(array.astype("<f4", copy=False).tobytes())
                total += local.numel()
        info = {
            **metadata,
            "version": 1,
            "rank": rank,
            "world_size": world_size,
            "gradient": "policy_loss_gradient_before_regularizers_and_grad_clip",
            "dtype": "float32_le",
            "parameters": parameters,
            "numel": total,
            "parameter_fingerprint": digest.hexdigest(),
            "batch_fingerprint": batch_fingerprint(batch),
        }
        json_temporary.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, binary_path)
        # Publish metadata last: an interrupted binary write is never a valid record.
        os.replace(json_temporary, json_path)
    finally:
        temporary.unlink(missing_ok=True)
        json_temporary.unlink(missing_ok=True)
    return json_path


def _load_snapshot(path):
    headers = []
    for json_path in sorted(path.glob("rank_*.json")):
        header = json.loads(json_path.read_text(encoding="utf-8"))
        if (
            header.get("version") != 1
            or header.get("gradient") != "policy_loss_gradient_before_regularizers_and_grad_clip"
        ):
            raise ValueError(f"Unsupported gradient record: {json_path}")
        if header.get("dtype") != "float32_le":
            raise ValueError(f"Unsupported gradient dtype: {json_path}")
        binary = json_path.with_suffix(".bin")
        if not binary.is_file() or binary.stat().st_size != header["numel"] * 4:
            raise ValueError(f"Incomplete gradient binary: {binary}")
        offset = 0
        for parameter in header["parameters"]:
            if parameter["offset"] != offset or parameter["numel"] < 0:
                raise ValueError(f"Invalid parameter offsets: {json_path}")
            offset += parameter["numel"]
        if offset != header["numel"]:
            raise ValueError(f"Invalid gradient length: {json_path}")
        headers.append((header, binary))
    if not headers:
        raise ValueError(f"No gradient records in {path}")
    world_size = headers[0][0]["world_size"]
    if [header["rank"] for header, _ in headers] != list(range(world_size)):
        raise ValueError(f"Missing or duplicated rank shards in {path}")
    key = ("step", "epoch", "minibatch", "world_size", "objective")
    if any(any(header[field] != headers[0][0][field] for field in key) for header, _ in headers):
        raise ValueError(f"Inconsistent rank metadata in {path}")
    return headers


def compare_snapshot(supervised, unsupervised):
    left, right = _load_snapshot(Path(supervised)), _load_snapshot(Path(unsupervised))
    if len(left) != len(right):
        raise ValueError("Gradient comparison requires identical world sizes and sharding layouts")
    dot = left_sq = right_sq = 0.0
    differences = set()
    for (a, a_path), (b, b_path) in zip(left, right):
        if any(a[field] != b[field] for field in ("step", "epoch", "minibatch", "rank", "parameters")):
            raise ValueError(f"Gradient coordinate layouts or update keys differ: {a_path} vs {b_path}")
        for field, reason in (
            ("parameter_fingerprint", "different_parameters"),
            ("batch_fingerprint", "different_rollouts"),
            ("objective", "different_objective"),
            ("rng_fingerprint", "different_random_state"),
        ):
            if a.get(field) != b.get(field):
                differences.add(reason)
        with a_path.open("rb") as a_file, b_path.open("rb") as b_file:
            while True:
                x = np.fromfile(a_file, dtype="<f4", count=CHUNK_SIZE).astype(np.float64)
                y = np.fromfile(b_file, dtype="<f4", count=CHUNK_SIZE).astype(np.float64)
                if not x.size:
                    break
                if not np.isfinite(x).all() or not np.isfinite(y).all():
                    raise ValueError(f"Non-finite gradient values: {a_path} or {b_path}")
                dot += float(np.dot(x, y))
                left_sq += float(np.dot(x, x))
                right_sq += float(np.dot(y, y))
    statistics = _comparison_statistics(dot, left_sq, right_sq)
    return {
        "update": Path(supervised).name,
        **{field: statistics[field] for field in ("cosine", "angle_deg", "supervised_norm", "unsupervised_norm")},
        "comparison": "+".join(sorted(differences)) if differences else "matched_conditions",
        "status": statistics["status"],
    }


def compare_runs(supervised, unsupervised, output):
    roots = [Path(supervised), Path(unsupervised)]
    snapshots = [
        {path.name: path for path in root.glob("step_*_epoch_*_minibatch_*") if path.is_dir()} for root in roots
    ]
    common = sorted(snapshots[0].keys() & snapshots[1].keys())
    if not common:
        raise ValueError("The two runs have no matching step/epoch/minibatch snapshots")
    for label, paths in zip(("supervised", "unsupervised"), snapshots):
        unmatched = sorted(paths.keys() - set(common))
        if unmatched:
            print(f"Unpaired {label} snapshots, skipped: {', '.join(unmatched)}")
    rows = [compare_snapshot(snapshots[0][key], snapshots[1][key]) for key in common]
    return _export_comparison(rows, output)


def compare_paired_run(directory, output):
    """Export statistics already computed from paired full-coordinate gradients."""
    rows = []
    for path in sorted(Path(directory).glob("step_*_epoch_*_minibatch_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if (
            record.get("version") != 2
            or record.get("mode") != "paired"
            or record.get("gradient") != "policy_loss_gradient_before_regularizers_and_grad_clip"
        ):
            raise ValueError(f"Unsupported paired gradient record: {path}")
        statistics = _comparison_statistics(
            record["dot_product"], record["supervised_squared_norm"], record["unsupervised_squared_norm"]
        )
        row = {
                "update": record["update"],
                **{
                    field: statistics[field]
                    for field in ("cosine", "angle_deg", "supervised_norm", "unsupervised_norm")
                },
                "comparison": record["comparison"],
                "status": statistics["status"],
            }
        for name, baseline in record.get("baselines", {}).items():
            comparison = _comparison_statistics(
                baseline["dot_product"], baseline["supervised_squared_norm"], baseline["unsupervised_squared_norm"]
            )
            row.update({f"{name}_{field}": comparison[field] for field in comparison})
        if record.get("baseline_metadata"):
            row["baseline_seed"] = record["baseline_metadata"].get("seed")
        rows.append(row)
    trajectory_path = _export_trajectories(directory, output)
    if not rows and trajectory_path is None:
        raise ValueError(f"No paired gradient statistics in {directory}")
    return _export_comparison(rows, output) if rows else trajectory_path


def _export_trajectories(directory, output):
    rows = []
    for path in sorted(Path(directory).glob("trajectory_step_*_epoch_*_minibatch_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("version") not in (1, 2) or record.get("mode") != "optimization_trajectory":
            raise ValueError(f"Unsupported optimization trajectory record: {path}")
        keys = ("update", "step", "epoch", "minibatch", "updates", "start_update", "end_update")
        row = {key: record[key] for key in keys}
        for name in ("endpoint", "path"):
            comparison = record[name]
            statistics = _comparison_statistics(
                comparison["dot_product"], comparison["supervised_squared_norm"],
                comparison["unsupervised_squared_norm"],
            )
            row.update({f"{name}_{field}": statistics[field] for field in statistics})
        for baseline, comparisons in record.get("baselines", {}).items():
            for name in ("endpoint", "path"):
                comparison = comparisons[name]
                statistics = _comparison_statistics(
                    comparison["dot_product"], comparison["supervised_squared_norm"],
                    comparison["unsupervised_squared_norm"],
                )
                row.update({f"{baseline}_{name}_{field}": statistics[field] for field in statistics})
        if record.get("baseline_metadata"):
            row["baseline_seed"] = record["baseline_metadata"].get("seed")
        random = record.get("baselines", {}).get("random_update", {})
        for name in ("target_step_norm", "applied_step_norm", "step_norm_relative_error"):
            if name in random:
                row[f"random_update_{name}"] = random[name]
        rows.append(row)
    if not rows:
        return None
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "trajectory_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    # A resumed run starts a new independently initialized trajectory segment.
    segments = dict.fromkeys(row["start_update"] for row in rows)
    for segment in segments:
        positions = [i for i, row in enumerate(rows) if row["start_update"] == segment]
        suffix = f" ({segment})" if len(segments) > 1 else ""
        targets = (("", "TTRL", "C0"), ("shuffled_labels_", "Shuffled labels", "C1"),
                   ("random_update_", "Random update", "C2"))
        for prefix, label, color in targets:
            for index, name in enumerate(("endpoint", "path")):
                field = f"{prefix}{name}_cosine"
                if not any(field in rows[i] for i in positions):
                    continue
                values = [rows[i].get(field) for i in positions]
                axes[index].plot(
                    positions, [np.nan if value is None else value for value in values], ".-",
                    label=f"{label}{suffix}", color=color,
                )
        norm_fields = (("endpoint_supervised_norm", "Supervised", "C3"),
                       ("endpoint_unsupervised_norm", "TTRL", "C0"),
                       ("shuffled_labels_endpoint_unsupervised_norm", "Shuffled labels", "C1"),
                       ("random_update_endpoint_unsupervised_norm", "Random update", "C2"))
        for field, label, color in norm_fields:
            if not any(field in rows[i] for i in positions):
                continue
            values = [rows[i].get(field) for i in positions]
            axes[2].plot(
                positions, [np.nan if value is None else value for value in values], ".-",
                label=f"{label}{suffix}", color=color,
            )
    for axis, label in zip(axes[:2], ("Endpoint cosine", "Update-path cosine")):
        axis.axhline(0, color="gray", linewidth=0.8)
        axis.set_ylim(-1.05, 1.05)
        axis.set_ylabel(label)
    axes[2].set_ylabel("Endpoint displacement L2 norm")
    for axis in axes:
        axis.legend(fontsize=7)
    ticks = np.unique(np.linspace(0, len(rows) - 1, min(6, len(rows)), dtype=int))
    axes[2].set_xticks(ticks, [str(rows[i]["updates"]) for i in ticks])
    axes[2].set_xlabel("Optimizer minibatches since the segment's common start")
    figure.suptitle("Optimization trajectories against a common supervised reference")
    figure.tight_layout()
    figure.savefig(destination / "trajectory_comparison.png", dpi=160)
    plt.close(figure)
    print(f"Saved optimization trajectories: {csv_path}")
    return csv_path


def _export_comparison(rows, output):
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "gradient_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        cosine = "undefined" if row["cosine"] is None else f"{row['cosine']:.6f}"
        angle = "undefined" if row["angle_deg"] is None else f"{row['angle_deg']:.2f}"
        print(f"{row['update']}: cosine={cosine}, angle={angle}, {row['status']}, {row['comparison']}")
    if any(row["comparison"] != "matched_conditions" for row in rows):
        print(
            "Across-run comparison: model, rollout or objective conditions differ; "
            "this does not isolate reward effects."
        )
    _plot(rows, destination / "gradient_comparison.png")
    return csv_path


def _plot(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positions = np.arange(len(rows))
    figure, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    values = [row["cosine"] if row["cosine"] is not None else np.nan for row in rows]
    axes[0].plot(positions, values, ".-", label="Policy gradient cosine")
    baselines = (("random_direction", "Random direction baseline"), ("shuffled_labels", "Shuffled-label baseline"))
    for name, label in baselines:
        if any(f"{name}_cosine" in row for row in rows):
            values = [row.get(f"{name}_cosine") for row in rows]
            axes[0].plot(positions, [np.nan if value is None else value for value in values], ".--", label=label)
    axes[0].axhline(0, color="gray", linewidth=0.8)
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].set_ylabel("Cosine similarity")
    axes[0].legend()
    for field, label in (("supervised_norm", "Supervised"), ("unsupervised_norm", "Unsupervised")):
        axes[1].plot(positions, [row[field] for row in rows], ".-", label=label)
    axes[1].set_ylabel("Policy gradient L2 norm")
    axes[1].legend()
    ticks = np.unique(np.linspace(0, len(rows) - 1, min(6, len(rows)), dtype=int))
    keys = [rows[index]["update"].split("_") for index in ticks]
    axes[1].set_xticks(
        ticks,
        ["/".join(str(int(key[index])) for index in (1, 3, 5)) for key in keys],
        fontsize=9,
    )
    axes[1].set_xlabel("Recorded update (step / PPO epoch / minibatch)")
    figure.suptitle(
        "Policy gradient comparison"
        + (
            " (across runs)"
            if any(row["comparison"] != "matched_conditions" for row in rows)
            else " (matched conditions)"
        )
    )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", help="Paired run's gradient_record.output_dir")
    parser.add_argument("--supervised", help="Supervised run's gradient_record.output_dir")
    parser.add_argument("--unsupervised", help="TTRL run's gradient_record.output_dir")
    parser.add_argument("--output", required=True, help="Directory for comparison CSV and PNG")
    arguments = parser.parse_args()
    if arguments.paired:
        if arguments.supervised or arguments.unsupervised:
            parser.error("--paired cannot be combined with --supervised or --unsupervised")
        path = compare_paired_run(arguments.paired, arguments.output)
    else:
        if not arguments.supervised or not arguments.unsupervised:
            parser.error("provide --paired, or both --supervised and --unsupervised")
        path = compare_runs(arguments.supervised, arguments.unsupervised, arguments.output)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
