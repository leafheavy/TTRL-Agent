"""Phase-two supervised/shuffled-label trajectory replay without sampling.

Run after phase one has exited and released its GPU workers::

    python -m verl.trainer.replay_gradient \
        --replay-dir /path/to/run/gradient_replay \
        --output-dir /path/to/run/gradient_stage2

The archive supplies the complete configuration. Changing world size, losses,
batch sizes or precision here would invalidate the paired reference.
"""

import argparse
import copy
import json
from pathlib import Path


def load_replay_configuration(directory):
    """Validate archive metadata before importing Ray or loading model weights."""
    from verl.utils.gradient_replay import GradientReplayReader

    directory = Path(directory).resolve()
    document = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))
    config = document["run_config"]
    world_size = document["world_size"]
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
        raise ValueError("Gradient replay archive has an invalid world_size")
    trainer = config["trainer"]
    recorded_world_size = trainer["n_gpus_per_node"] * trainer["nnodes"]
    if world_size != recorded_world_size:
        raise ValueError("Gradient replay world size does not match the captured configuration")
    actor = config["actor_rollout_ref"]["actor"]
    if actor["strategy"] != "fsdp":
        raise ValueError("Two-stage gradient replay currently requires FSDP1")
    if actor.get("ulysses_sequence_parallel_size", 1) != 1:
        raise ValueError("Two-stage gradient replay requires ulysses_sequence_parallel_size=1")
    if not actor.get("gradient_record", {}).get("two_stage", {}).get("enabled", False):
        raise ValueError("This run was not recorded as a two-stage gradient diagnostic")
    # The reader verifies all rank completion markers, manifests and file
    # hashes. Its initial state and minibatch tensors remain lazily loaded.
    GradientReplayReader(directory, rank=0, world_size=world_size)
    return config, world_size


def prepare_output_directory(directory, replay_directory):
    """Keep stage-two output separate and preserve any existing experiment."""
    directory, replay_directory = Path(directory).resolve(), Path(replay_directory).resolve()
    if (directory == replay_directory or replay_directory in directory.parents
            or directory in replay_directory.parents):
        raise ValueError("Gradient replay output and archive directories must not overlap")
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise FileExistsError(f"Gradient replay output must be a new or empty directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def run_replay(directory, output_dir):
    """Allocate only actor workers, retaining the captured rank layout."""
    config_dict, world_size = load_replay_configuration(directory)
    output = prepare_output_directory(output_dir, directory)

    import ray
    from omegaconf import OmegaConf

    from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
    from verl.workers.fsdp_workers import ActorRolloutRefWorker

    config = OmegaConf.create(config_dict)
    device_name = config.trainer.get("device", "cuda")
    resource_name = {"cuda": "GPU", "npu": "NPU"}.get(device_name)
    if resource_name is None:
        raise ValueError("The FSDP replay launcher requires the captured CUDA or NPU device type")
    owns_ray = not ray.is_initialized()
    workers = None
    pool = None
    try:
        if owns_ray:
            ray.init(
                runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}},
                num_cpus=config.get("ray_init", {}).get("num_cpus"),
            )
        if ray.available_resources().get(resource_name, 0) < world_size:
            raise RuntimeError(
                f"Replay needs the same {world_size} {resource_name} workers as phase one; "
                "finish phase one and release its workers before starting phase two"
            )
        pool = RayResourcePool(
            process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes,
            use_gpu=True,
            max_colocate_count=1,
        )
        workers = RayWorkerGroup(
            resource_pool=pool,
            ray_cls_with_init=RayClassWithInitArgs(
                cls=ray.remote(ActorRolloutRefWorker),
                config=copy.deepcopy(config.actor_rollout_ref),
                role="actor",
            ),
            device_name=device_name,
            ray_wait_register_center_timeout=config.trainer.get("ray_wait_register_center_timeout", 300),
        )
        # role='actor' constructs neither vLLM nor the KL reference model. All
        # reference log probabilities and advantages come from the archive.
        workers.init_model()
        results = workers.replay_gradient_diagnostics(str(Path(directory).resolve()), str(output))
        if (len(results) != world_size or {result.get("rank") for result in results} != set(range(world_size))
                or any(not isinstance(result.get("updates"), int) or result["updates"] < 1 for result in results)
                or len({result["updates"] for result in results}) != 1):
            raise ValueError("Replay workers did not complete the same optimizer update sequence")
        archive = json.loads((Path(directory) / "run_config.json").read_text(encoding="utf-8"))
        completion = {
            "version": 1, "stage": 2, "run_id": archive["run_id"],
            "world_size": world_size, "ranks": results,
        }
        temporary = output / "complete.json.tmp"
        temporary.write_text(json.dumps(completion, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output / "complete.json")
        print(f"[gradient_replay] All ranks verified; combined trajectory results: {output}")
        return results
    finally:
        if workers is not None:
            for worker in workers.workers:
                ray.kill(worker, no_restart=True)
        if pool is not None:
            for placement_group in pool.pgs or []:
                ray.util.remove_placement_group(placement_group)
        if owns_ray:
            ray.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", required=True, help="Completed phase-one replay archive")
    parser.add_argument("--output-dir", required=True, help="New directory for verified combined trajectory JSON")
    args = parser.parse_args()
    run_replay(args.replay_dir, args.output_dir)


if __name__ == "__main__":
    main()
