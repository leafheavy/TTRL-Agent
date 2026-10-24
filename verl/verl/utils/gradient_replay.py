"""Bounded-memory, rank-local archives for two-stage gradient diagnostics.

The archive contains actual optimizer minibatches, not newly sampled answers.
Only plain Python values and tensors are serialized, so readers can use
``torch.load(weights_only=True)``. A completed archive and an exactly reproduced
supervised reference are required before an old TTRL comparison is combined
with a new shuffled-label comparison.
"""

import copy
import hashlib
import json
import math
import os
import random
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from verl.utils.gradient_diagnostics import _hash_tensor, _local, _parameter_fingerprint, _parameter_layout

VERSION = 1
OBJECTIVE_KEYS = (
    "ppo_mini_batch_size", "ppo_micro_batch_size_per_gpu", "ppo_epochs", "use_dynamic_bsz",
    "ppo_max_token_len_per_gpu", "clip_ratio", "clip_ratio_low", "clip_ratio_high", "clip_ratio_c",
    "entropy_coeff", "loss_agg_mode", "grad_clip", "use_kl_loss", "kl_loss_type", "kl_loss_coef",
    "ulysses_sequence_parallel_size", "use_remove_padding", "use_fused_kernels",
    "entropy_from_logits_with_chunking", "use_torch_compile", "policy_loss",
)


def _plain(value):
    if hasattr(value, "items"):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError(f"Replay configuration contains unsupported value {type(value).__name__}")


def actor_objective_config(config):
    """Select normalized loss and batching settings without importing OmegaConf."""
    return {key: _plain(config.get(key)) for key in OBJECTIVE_KEYS if key in config}


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path, value, *, replace=False, max_bytes=None):
    path = Path(path)
    if path.exists() and not replace:
        raise FileExistsError(f"Replay file already exists: {path}")
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        encoded = _json_bytes(value)
        if max_bytes is not None and len(encoded) > max_bytes:
            raise ValueError("Replay archive capacity limit exceeded while writing metadata")
        temporary.write_bytes(encoded)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class _LimitedWriter:
    """Stop serialization before a tensor file can exceed its remaining budget."""

    def __init__(self, stream, limit):
        self.stream, self.limit, self.written = stream, limit, 0
        self.exceeded = False

    def write(self, value):
        if self.limit is not None and self.written + len(value) > self.limit:
            self.exceeded = True
            raise OSError("Replay archive capacity limit exceeded")
        count = self.stream.write(value)
        self.written += count
        return count

    def flush(self):
        return self.stream.flush()

    def tell(self):
        return self.stream.tell()


def _atomic_tensor(path, value, *, max_bytes=None):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Replay tensor file already exists: {path}")
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("wb") as stream:
            limited = _LimitedWriter(stream, max_bytes)
            try:
                torch.save(value, limited)
            except Exception as exc:
                if limited.exceeded:
                    raise ValueError(
                        "Replay archive capacity limit exceeded; use a larger archive budget or volume"
                    ) from exc
                raise
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _file_hash(path)


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _runtime():
    return {
        "torch": str(torch.__version__), "cuda": torch.version.cuda, "numpy": str(np.__version__),
        "default_dtype": str(torch.get_default_dtype()),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def prepare_replay_directory(directory, run_config):
    """Reserve a fresh archive before model initialization in stage one."""
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Use a new, empty two-stage replay directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    run_config = _plain(run_config)
    settings = run_config.get("actor_rollout_ref", {}).get("actor", {}).get("gradient_record", {}).get("two_stage", {})
    initial_state = settings.get("initial_state", "pretrained")
    max_archive_gb = settings.get("max_archive_gb", 4.0 if initial_state == "pretrained" else None)
    if initial_state not in ("pretrained", "snapshot"):
        raise ValueError("two_stage.initial_state must be pretrained or snapshot")
    if max_archive_gb is not None and (
        isinstance(max_archive_gb, bool) or not isinstance(max_archive_gb, (int, float))
        or not math.isfinite(max_archive_gb) or max_archive_gb <= 0
    ):
        raise ValueError("two_stage.max_archive_gb must be positive and finite, or null")
    trainer = run_config.get("trainer", {})
    world_size = trainer.get("n_gpus_per_node", 1) * trainer.get("nnodes", 1)
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
        raise ValueError("Replay world_size must be a positive integer")
    document = {
        "version": VERSION, "run_id": uuid.uuid4().hex, "world_size": world_size,
        "runtime": _runtime(), "run_config": run_config,
        "initial_state": initial_state, "max_archive_gb": max_archive_gb,
    }
    _atomic_json(directory / "run_config.json", document)
    return document


def _encode(value):
    """Encode optimizer snapshots without pickling Parameters or state wrappers."""
    if hasattr(value, "value") and isinstance(value.value, torch.Tensor) and hasattr(value, "device"):
        return {"kind": "tensor", "device": torch.device(value.device).type, "value": value.value}
    if isinstance(value, torch.Tensor):
        return {"kind": "tensor", "device": value.device.type, "value": value.detach().to("cpu")}
    if isinstance(value, dict):
        return {"kind": "dict", "items": [(key, _encode(item)) for key, item in value.items()]}
    if isinstance(value, (tuple, list)):
        return {"kind": "tuple" if isinstance(value, tuple) else "list", "items": [_encode(item) for item in value]}
    if value is None or isinstance(value, (str, bool, int, float)):
        return {"kind": "value", "value": value}
    raise ValueError(f"Unsupported replay optimizer state: {type(value).__name__}")


def _decode(value, device):
    kind = value["kind"]
    if kind == "tensor":
        target = torch.device("cpu") if value["device"] == "cpu" else torch.device(device)
        if value["device"] != target.type:
            raise ValueError("Replay optimizer tensor device type changed")
        return value["value"].to(target, copy=True)
    if kind == "dict":
        return {key: _decode(item, device) for key, item in value["items"]}
    if kind in ("tuple", "list"):
        items = [_decode(item, device) for item in value["items"]]
        return tuple(items) if kind == "tuple" else items
    if kind == "value":
        return value["value"]
    raise ValueError(f"Unsupported replay state tag: {kind}")


def _hash_tree(digest, value):
    if hasattr(value, "value") and isinstance(value.value, torch.Tensor) and hasattr(value, "device"):
        digest.update(b"optimizer_tensor:" + torch.device(value.device).type.encode())
        _hash_tensor(digest, value.value)
    elif isinstance(value, torch.Tensor):
        digest.update(b"tensor:")
        _hash_tensor(digest, value)
    elif isinstance(value, dict):
        digest.update(b"dict:")
        for key in sorted(value, key=lambda item: (type(item).__name__, str(item))):
            _hash_tree(digest, key)
            _hash_tree(digest, value[key])
    elif isinstance(value, (tuple, list)):
        digest.update(("tuple:" if isinstance(value, tuple) else "list:").encode())
        for item in value:
            _hash_tree(digest, item)
    else:
        digest.update(_json_bytes(value))


def _tree_hash(value):
    digest = hashlib.sha256()
    _hash_tree(digest, value)
    return digest.hexdigest()


def _optimizer_name(optimizer):
    return type(optimizer).__module__ + "." + type(optimizer).__qualname__


def _groups(optimizer):
    return [_encode({key: value for key, value in group.items() if key != "params"})
            for group in optimizer.param_groups]


def _restore_groups(optimizer, groups):
    if len(optimizer.param_groups) != len(groups):
        raise ValueError("Replay optimizer parameter-group count changed")
    for group, encoded in zip(optimizer.param_groups, groups):
        device = group["params"][0].device if group["params"] else torch.device("cpu")
        values = _decode(encoded, device)
        for key in list(group):
            if key != "params":
                del group[key]
        group.update(values)


def _rng(device_name):
    numpy_state = np.random.get_state()
    result = {
        "torch": torch.random.get_rng_state().clone(), "python": random.getstate(),
        "numpy": [numpy_state[0], torch.from_numpy(numpy_state[1].astype(np.int64)),
                  int(numpy_state[2]), int(numpy_state[3]), float(numpy_state[4])],
        "device_name": str(device_name),
    }
    if device_name != "cpu":
        result["device"] = getattr(torch, device_name).get_rng_state().cpu()
    return result


def restore_rng(state, device_name):
    """Restore a captured optimizer-update RNG point on the rank's current GPU."""
    if state["device_name"] != device_name:
        raise ValueError("Replay device type changed")
    torch.random.set_rng_state(state["torch"])
    random.setstate(state["python"])
    algorithm, keys, position, gaussian, cached = state["numpy"]
    np.random.set_state((algorithm, keys.numpy().astype(np.uint32), position, gaussian, cached))
    if device_name != "cpu":
        getattr(torch, device_name).set_rng_state(state["device"])


def _reference(trajectory, parameter_indices):
    state = []
    for parameter, value in trajectory.supervised_optimizer_state.items():
        try:
            index = parameter_indices[id(parameter)]
        except KeyError as exc:
            raise ValueError("Supervised optimizer contains an unknown parameter") from exc
        state.append((index, value))
    return {
        "parameters": _tree_hash(trajectory.supervised_parameters),
        "buffers": _tree_hash(trajectory.supervised_buffers),
        "optimizer": _tree_hash(sorted(state, key=lambda item: item[0])),
    }


def _update_key(metadata):
    return (f"step_{metadata['step']:06d}_epoch_{metadata['epoch']:03d}"
            f"_minibatch_{metadata['minibatch']:04d}")


class GradientReplayWriter:
    """One writer per FSDP rank; only one minibatch is buffered at a time."""

    def __init__(self, directory, rank, world_size, actor_config):
        self.directory = Path(directory)
        self.document = _load_json(self.directory / "run_config.json")
        if self.document["version"] != VERSION or self.document["world_size"] != world_size:
            raise ValueError("Replay archive version or world_size mismatch")
        if not 0 <= rank < world_size:
            raise ValueError("Replay rank is outside world_size")
        self.rank, self.world_size = rank, world_size
        self.initial_state = self.document.get("initial_state", "snapshot")
        maximum = self.document.get("max_archive_gb")
        self.byte_budget = None if maximum is None else (
            int(maximum * 1_000_000_000) - (self.directory / "run_config.json").stat().st_size
        ) // world_size
        self.rank_directory = self.directory / f"rank_{rank:05d}"
        self.rank_directory.mkdir(exist_ok=False)
        self.manifest = {
            "version": VERSION, "run_id": self.document["run_id"], "rank": rank,
            "world_size": world_size, "actor_config": _plain(actor_config), "frames": [],
        }
        self.pending = None
        self.started = False
        self.completed = False

    def _remaining_bytes(self):
        if self.byte_budget is None:
            return None
        used = sum(path.stat().st_size for path in self.rank_directory.iterdir() if path.is_file())
        return max(0, self.byte_budget - used)

    def start(self, trajectory, module, optimizer):
        if self.started:
            raise ValueError("Replay initial state has already been captured")
        if trajectory.updates or trajectory.pending is not None:
            raise ValueError("Replay must start before the first supervised update")
        if self.initial_state == "pretrained" and any(trajectory.supervised_optimizer_state.values()):
            raise ValueError(
                "Pretrained replay requires an empty initial optimizer; model checkpoints will not be saved"
            )
        parameters = list(module.parameters())
        self.parameter_indices = {id(parameter): index for index, parameter in enumerate(parameters)}
        trainable = [(name, parameter) for name, parameter in module.named_parameters() if parameter.requires_grad]
        state = []
        for parameter, value in trajectory.supervised_optimizer_state.items():
            state.append({"index": self.parameter_indices[id(parameter)], "value": _encode(value)})
        initial = {
            "version": VERSION, "run_id": self.document["run_id"], "rank": self.rank,
            "world_size": self.world_size, "source": self.initial_state,
            "layouts": [_parameter_layout(name, parameter) for name, parameter in trainable],
            "dtypes": [str(parameter.dtype) for _, parameter in trainable],
            "optimizer_name": _optimizer_name(optimizer), "param_groups": _groups(optimizer),
            "group_parameters": [[self.parameter_indices[id(parameter)] for parameter in group["params"]]
                                 for group in optimizer.param_groups],
            "module_fingerprint": _parameter_fingerprint(module),
            "reference": _reference(trajectory, self.parameter_indices),
        }
        if self.initial_state == "pretrained":
            initial["buffer_fingerprint"] = _tree_hash(trajectory.supervised_buffers)
            self.manifest["initial_file"] = "initial_metadata.pt"
        else:
            initial.update(parameters=trajectory.initial_parameters, buffers=trajectory.supervised_buffers,
                           optimizer_state=state)
            self.manifest["initial_file"] = "initial.pt"
        self.manifest["initial_sha256"] = _atomic_tensor(
            self.rank_directory / self.manifest["initial_file"], initial, max_bytes=self._remaining_bytes()
        )
        _atomic_json(self.rank_directory / "manifest.json", self.manifest, max_bytes=self._remaining_bytes())
        self.started = True

    def begin(self, micro_batches, *, metadata, temperature, gradient_accumulation, optimizer, device_name):
        if not self.started or self.completed or self.pending is not None:
            raise ValueError("Replay writer cannot begin an update in its current state")
        if not math.isfinite(float(temperature)) or temperature <= 0:
            raise ValueError("Replay temperature must be positive and finite")
        key = _update_key(metadata)
        if self.manifest["frames"] and key <= self.manifest["frames"][-1]["update"]:
            raise ValueError("Replay optimizer updates must be strictly ordered")
        batches = []
        for micro_batch in micro_batches:
            values = micro_batch.batch if hasattr(micro_batch, "batch") else micro_batch
            if not all(isinstance(value, torch.Tensor) for value in values.values()):
                raise ValueError("Two-stage replay currently supports text tensor minibatches only")
            required = {"input_ids", "responses", "attention_mask", "position_ids", "response_mask",
                        "old_log_probs", "diagnostic_supervised_advantages", "diagnostic_shuffled_advantages"}
            if not required.issubset(values):
                raise ValueError("Replay minibatch is missing supervised/shuffled advantages or rollout tensors")
            batches.append({name: value.detach().to("cpu", copy=True) for name, value in values.items()})
        if not batches:
            raise ValueError("Replay requires at least one microbatch per optimizer update")
        self.pending = {
            "version": VERSION, "run_id": self.document["run_id"], "rank": self.rank,
            "world_size": self.world_size, "sequence": len(self.manifest["frames"]) + 1,
            "update": key, "metadata": _plain(metadata), "temperature": float(temperature),
            "gradient_accumulation": gradient_accumulation, "param_groups": _groups(optimizer),
            "rng": _rng(device_name), "micro_batches": batches,
        }

    def finish(self, trajectory, result):
        if self.pending is None or result.get("update") != self.pending["update"]:
            raise ValueError("Replay result does not match the pending optimizer update")
        if trajectory.pending is not None or trajectory.updates != self.pending["sequence"]:
            raise ValueError("Replay requires a completed, consecutive trajectory update")
        frame = self.pending
        frame["expected_reference"] = _reference(trajectory, self.parameter_indices)
        frame["result"] = _plain(result)
        name = f"frame_{frame['sequence']:08d}.pt"
        digest = _atomic_tensor(self.rank_directory / name, frame, max_bytes=self._remaining_bytes())
        previous = (self.manifest["frames"][-1]["sha256"] if self.manifest["frames"]
                    else self.manifest["initial_sha256"])
        self.manifest["frames"].append({
            "sequence": frame["sequence"], "update": frame["update"], "file": name,
            "sha256": digest, "previous_sha256": previous,
        })
        _atomic_json(self.rank_directory / "manifest.json", self.manifest, replace=True,
                     max_bytes=self._remaining_bytes())
        self.pending = None

    def complete(self):
        if not self.started or self.pending is not None or not self.manifest["frames"] or self.completed:
            raise ValueError("Cannot complete an empty, pending, or already completed replay archive")
        _atomic_json(self.rank_directory / "COMPLETE.json", {
            "run_id": self.document["run_id"], "rank": self.rank,
            "updates": len(self.manifest["frames"]),
            "manifest_sha256": _file_hash(self.rank_directory / "manifest.json"),
        }, max_bytes=self._remaining_bytes())
        self.completed = True


class GradientReplayReader:
    """Validate all rank inventories, then stream this rank's tensor files."""

    def __init__(self, directory, rank, world_size):
        self.directory = Path(directory)
        self.document = _load_json(self.directory / "run_config.json")
        if self.document.get("version") != VERSION or self.document.get("world_size") != world_size:
            raise ValueError("Replay archive version or world_size mismatch")
        if not 0 <= rank < world_size:
            raise ValueError("Replay rank is outside world_size")
        if self.document.get("runtime") != _runtime():
            raise ValueError("Replay runtime/numerical settings differ from stage one")
        self.rank, self.world_size = rank, world_size
        self.rank_directory = self.directory / f"rank_{rank:05d}"
        inventories = []
        for expected_rank in range(world_size):
            location = self.directory / f"rank_{expected_rank:05d}"
            if not (location / "COMPLETE.json").is_file():
                raise ValueError(f"Replay archive is incomplete: rank {expected_rank} has no completion marker")
            manifest = _load_json(location / "manifest.json")
            complete = _load_json(location / "COMPLETE.json")
            if (manifest.get("run_id") != self.document["run_id"] or manifest.get("rank") != expected_rank
                    or manifest.get("world_size") != world_size or complete.get("rank") != expected_rank
                    or complete.get("run_id") != self.document["run_id"]
                    or complete.get("manifest_sha256") != _file_hash(location / "manifest.json")):
                raise ValueError("Replay rank manifest identity or checksum mismatch")
            initial_file = manifest.get("initial_file", "initial.pt")
            if initial_file not in ("initial.pt", "initial_metadata.pt"):
                raise ValueError("Replay initial metadata filename is invalid")
            if not (location / initial_file).is_file():
                raise ValueError("Replay archive is missing its initial optimizer state")
            if _file_hash(location / initial_file) != manifest["initial_sha256"]:
                raise ValueError("Replay initial-state checksum mismatch")
            frames = manifest.get("frames", [])
            if not frames or complete.get("updates") != len(frames):
                raise ValueError("Replay completion marker has an inconsistent update count")
            previous = manifest["initial_sha256"]
            updates = []
            for sequence, entry in enumerate(frames, 1):
                expected_file = f"frame_{sequence:08d}.pt"
                if (entry.get("sequence") != sequence or entry.get("file") != expected_file
                        or entry.get("previous_sha256") != previous
                        or not (location / expected_file).is_file()):
                    raise ValueError("Replay frame order, hash chain, or file inventory is invalid")
                if _file_hash(location / expected_file) != entry["sha256"]:
                    raise ValueError(f"Replay frame checksum mismatch: {entry['update']}")
                updates.append(entry["update"])
                previous = entry["sha256"]
            if updates != sorted(set(updates)):
                raise ValueError("Replay update keys are duplicated or out of order")
            inventories.append(updates)
            if expected_rank == rank:
                self.manifest = manifest
        if any(inventory != inventories[0] for inventory in inventories[1:]):
            raise ValueError("Replay rank update sequences differ")
        self._initial = None
        self._verified = set()
        self._restored = set()

    @property
    def initial(self):
        if self._initial is None:
            path = self.rank_directory / self.manifest.get("initial_file", "initial.pt")
            if _file_hash(path) != self.manifest["initial_sha256"]:
                raise ValueError("Replay initial-state checksum mismatch")
            self._initial = torch.load(path, map_location="cpu", weights_only=True)
            if (self._initial.get("run_id") != self.document["run_id"] or self._initial.get("rank") != self.rank
                    or self._initial.get("world_size") != self.world_size):
                raise ValueError("Replay initial-state identity mismatch")
        return self._initial

    def validate_actor_config(self, config):
        if _plain(config) != self.manifest["actor_config"]:
            raise ValueError("Replay actor objective/batching configuration differs from stage one")

    def restore_initial(self, module, optimizer):
        initial = self.initial
        trainable = [(name, parameter) for name, parameter in module.named_parameters() if parameter.requires_grad]
        if ([_parameter_layout(name, parameter) for name, parameter in trainable] != initial["layouts"]
                or [str(parameter.dtype) for _, parameter in trainable] != initial["dtypes"]):
            raise ValueError("Replay initial parameter layouts or dtypes differ")
        if _optimizer_name(optimizer) != initial["optimizer_name"]:
            raise ValueError("Replay optimizer type differs from stage one")
        parameters = list(module.parameters())
        self.parameter_indices = {id(parameter): index for index, parameter in enumerate(parameters)}
        groups = [[self.parameter_indices[id(parameter)] for parameter in group["params"]]
                  for group in optimizer.param_groups]
        if groups != initial["group_parameters"]:
            raise ValueError("Replay optimizer parameter grouping differs")
        buffers = dict(module.named_buffers())
        source = initial.get("source", "snapshot")
        if source == "pretrained":
            if _tree_hash(buffers) != initial["buffer_fingerprint"]:
                raise ValueError("Pretrained replay initial model buffers do not match stage one")
        elif source == "snapshot":
            if buffers.keys() != initial["buffers"].keys():
                raise ValueError("Replay model buffers differ")
            with torch.no_grad():
                for (_, parameter), value in zip(trainable, initial["parameters"]):
                    _local(parameter).copy_(value)
                for name, buffer in buffers.items():
                    value = initial["buffers"][name]
                    if buffer.shape != value.shape or buffer.dtype != value.dtype:
                        raise ValueError("Replay model buffer shape or dtype differs")
                    buffer.copy_(value)
        else:
            raise ValueError("Replay initial state source is invalid")
        if _parameter_fingerprint(module) != initial["module_fingerprint"]:
            raise ValueError("Replay initial model fingerprint differs, including frozen parameters")
        optimizer.zero_grad()
        optimizer.state = defaultdict(dict)
        for entry in initial.get("optimizer_state", []):
            parameter = parameters[entry["index"]]
            optimizer.state[parameter] = _decode(entry["value"], parameter.device)
        _restore_groups(optimizer, initial["param_groups"])
        # The live model/optimizer now own their copies. Do not keep an extra
        # full CPU checkpoint resident alongside the trajectory's origin.
        self._initial = None

    def frames(self):
        for entry in self.manifest["frames"]:
            path = self.rank_directory / entry["file"]
            if _file_hash(path) != entry["sha256"]:
                raise ValueError(f"Replay frame checksum mismatch: {entry['update']}")
            frame = torch.load(path, map_location="cpu", weights_only=True)
            if (frame.get("run_id") != self.document["run_id"] or frame.get("rank") != self.rank
                    or frame.get("world_size") != self.world_size or frame.get("sequence") != entry["sequence"]
                    or frame.get("update") != entry["update"] or _update_key(frame["metadata"]) != entry["update"]
                    or frame["result"].get("update") != entry["update"]):
                raise ValueError("Replay frame identity, order, or result mismatch")
            frame["_file_sha256"] = entry["sha256"]
            frame["_content_fingerprint"] = _tree_hash(frame)
            yield frame

    @staticmethod
    def _check_frame(frame):
        digest = frame.get("_content_fingerprint")
        content = {key: value for key, value in frame.items() if key != "_content_fingerprint"}
        if digest != _tree_hash(content):
            raise ValueError("Replay frame changed after checksum validation")

    def restore_frame(self, frame, module, optimizer, device_name):
        if not hasattr(self, "parameter_indices"):
            raise ValueError("Restore the initial replay state before processing updates")
        self._check_frame(frame)
        expected_sequence = len(self._restored) + 1
        if frame["sequence"] != expected_sequence:
            raise ValueError("Replay updates must be restored once, in their captured order")
        if expected_sequence > 1 and expected_sequence - 1 not in self._verified:
            raise ValueError("The preceding supervised replay reference has not been verified")
        _restore_groups(optimizer, frame["param_groups"])
        restore_rng(frame["rng"], device_name)
        self._restored.add(frame["sequence"])

    def verify_reference(self, frame, trajectory):
        self._check_frame(frame)
        if frame["sequence"] not in self._restored:
            raise ValueError("Replay frame was not restored before reference verification")
        actual = _reference(trajectory, self.parameter_indices)
        if actual != frame["expected_reference"]:
            differences = [name for name in actual if actual[name] != frame["expected_reference"].get(name)]
            raise ValueError("Supervised replay reference mismatch: " + ", ".join(differences))
        reference_updates = trajectory.updates + int(trajectory.pending is not None)
        if reference_updates != frame["sequence"]:
            raise ValueError("Supervised replay reference update count mismatch")
        self._verified.add(frame["sequence"])
        return actual

    def combine(self, frame, baseline_result):
        self._check_frame(frame)
        if frame["sequence"] not in self._verified:
            raise ValueError("Cannot combine unverified supervised replay references")
        old = frame["result"]
        for name in ("update", "updates", "start_update", "end_update", "world_size"):
            if baseline_result.get(name) != old.get(name):
                raise ValueError(f"Replay comparison differs in {name}")
        if baseline_result.get("origin_parameter_fingerprint_local") != old.get("origin_parameter_fingerprint_local"):
            raise ValueError("Replay comparison origin differs from stage one")
        for name in ("endpoint", "path"):
            if baseline_result[name]["supervised_squared_norm"] != old[name]["supervised_squared_norm"]:
                raise ValueError("Supervised replay trajectory statistics differ from stage one")
        result = copy.deepcopy(old)
        result["version"] = 2
        result["baselines"] = {
            "shuffled_labels": {
                "reference": "same_independent_supervised_optimizer_trajectory",
                "endpoint": copy.deepcopy(baseline_result["endpoint"]),
                "path": copy.deepcopy(baseline_result["path"]),
                "update_rule": "independent_training_optimizer_and_history",
            },
        }
        result["baseline_metadata"] = {
            "reference": "exactly_replayed_stage_one_supervised_trajectory",
            "shuffled_labels": copy.deepcopy(frame["metadata"].get("shuffled_labels", {})),
        }
        result["two_stage"] = {
            "verified": True, "run_id": self.document["run_id"], "rank": self.rank,
            "frame_sha256": frame["_file_sha256"], "reference": frame["expected_reference"],
            "verification": "exact_parameters_buffers_optimizer_and_trajectory_statistics",
        }
        return result
