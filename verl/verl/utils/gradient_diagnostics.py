"""Record exact policy-loss gradients and compare two training runs offline.

Run from the existing verl environment::

    python -m verl.utils.gradient_diagnostics \
        --supervised /path/to/supervised/gradients \
        --unsupervised /path/to/ttrl/gradients --output /path/to/diagnosis

Each optimizer minibatch is a snapshot. Rank shards are stored as float32 binary
arrays with JSON metadata; comparisons stream the arrays rather than gathering
the full model or averaging per-rank cosine similarities.
"""

import argparse
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
BATCH_KEYS = ("input_ids", "attention_mask", "position_ids", "responses", "response_mask", "old_log_probs")


def recording_due(config, step):
    return bool(config.get("enabled", False)) and (step == 1 or step % config.get("interval", 10) == 0)


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
    cosine = angle = None
    status = "ok"
    if left_sq == 0 or right_sq == 0:
        status = "both_zero" if left_sq == right_sq == 0 else "supervised_zero" if left_sq == 0 else "unsupervised_zero"
    else:
        cosine = float(np.clip(dot / math.sqrt(left_sq * right_sq), -1, 1))
        angle = math.degrees(math.acos(cosine))
    return {
        "update": Path(supervised).name,
        "cosine": cosine,
        "angle_deg": angle,
        "supervised_norm": math.sqrt(left_sq),
        "unsupervised_norm": math.sqrt(right_sq),
        "comparison": "+".join(sorted(differences)) if differences else "matched_conditions",
        "status": status,
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
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "gradient_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
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
    parser.add_argument("--supervised", required=True, help="Supervised run's gradient_record.output_dir")
    parser.add_argument("--unsupervised", required=True, help="TTRL run's gradient_record.output_dir")
    parser.add_argument("--output", required=True, help="Directory for comparison CSV and PNG")
    arguments = parser.parse_args()
    print(f"Saved: {compare_runs(arguments.supervised, arguments.unsupervised, arguments.output)}")


if __name__ == "__main__":
    main()
