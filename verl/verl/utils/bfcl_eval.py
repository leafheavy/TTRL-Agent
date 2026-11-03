"""Reuse a trained run's inference settings for its no-update BFCL control."""

import argparse
import hashlib
import json
import math
import os
import shlex
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def get_vllm_eval_settings(config, sampling_params_type):
    """Common engine arguments for the existing rollout server and baseline."""
    actor = config["actor_rollout_ref"]
    rollout = actor["rollout"]
    sampling = sampling_params_type()
    overrides = dict(n=1, logprobs=0, repetition_penalty=1.0, max_new_tokens=rollout["response_length"])
    overrides.update({key: rollout[key] for key in rollout if hasattr(sampling, str(key))})
    return dict(
        enable_sleep_mode=rollout["free_cache_engine"],
        generation_config="auto",
        override_generation_config=overrides,
        tensor_parallel_size=rollout.get("tensor_model_parallel_size", 1),
        dtype=rollout["dtype"],
        enforce_eager=rollout["enforce_eager"],
        gpu_memory_utilization=rollout["gpu_memory_utilization"],
        disable_custom_all_reduce=True,
        skip_tokenizer_init=False,
        max_model_len=int(rollout.get("max_model_len") or rollout["prompt_length"] + rollout["response_length"]),
        load_format="auto",
        disable_log_stats=rollout["disable_log_stats"],
        max_num_batched_tokens=rollout.get("max_num_batched_tokens", 8192),
        enable_chunked_prefill=rollout["enable_chunked_prefill"],
        enable_prefix_caching=True,
        trust_remote_code=actor["model"].get("trust_remote_code", False),
        seed=rollout.get("seed", 0),
    )


def _load_reference_config(reference):
    from omegaconf import OmegaConf

    for path in (reference / "hydra/.hydra/config.yaml", reference / ".hydra/config.yaml"):
        if path.is_file():
            return path, OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    raise FileNotFoundError(f"No saved Hydra config in {reference}; select a concrete training run/category")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get_vllm_runtime_metadata():
    packages = {}
    for name in ("vllm", "torch"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return packages


def _fingerprint_files(package, paths):
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(package).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def record_bfcl_environment(output, compare=None, *, bfcl_model=None, temperature=0.001, num_threads=1):
    """Record/check the actual BFCL handler and scorer implementation."""
    import bfcl_eval

    package = Path(bfcl_eval.__file__).resolve().parent
    package_version = None
    for distribution in ("bfcl", "bfcl-eval"):
        try:
            package_version = version(distribution)
            break
        except PackageNotFoundError:
            pass
    result = dict(
        version=package_version, code_sha256=_fingerprint_files(package, package.rglob("*.py")),
        data_sha256=_fingerprint_files(package, (package / "data").rglob("*.json")),
        client=dict(bfcl_model=bfcl_model, temperature=temperature, num_threads=num_threads),
    )
    if compare:
        reference = Path(compare)
        if reference.is_file():
            previous = json.loads(reference.read_text(encoding="utf-8"))
            if any(result[key] != previous.get(key) for key in result):
                raise ValueError("BFCL version/code/data/client differs from the updated-model evaluation")
            result["reference_check"] = "matched"
        else:
            result["reference_check"] = "unverified_legacy_reference"
            print("Legacy reference has no BFCL fingerprint; historical version/code cannot be verified.", flush=True)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def prepare_baseline(args, sampling_params_type):
    """Freeze the source config, exact case IDs, and client/server settings."""
    reference = Path(args.reference_run).resolve()
    source_config, config = _load_reference_config(reference)
    rollout = config["actor_rollout_ref"]["rollout"]
    serving = config["trainer"].get("post_train_eval", {})
    if rollout["name"] != "vllm" or rollout.get("mode") != "async" or not serving.get("enabled", False):
        raise ValueError("Reference run must use async vLLM post-training evaluation")
    settings = get_vllm_eval_settings(config, sampling_params_type)
    versions = get_vllm_runtime_metadata()
    recorded = reference / "post_train_eval/vllm_eval_config_0.json"
    reference_check = "unverified_legacy_reference"
    if recorded.is_file():
        previous = json.loads(recorded.read_text(encoding="utf-8"))
        if previous.get("engine_settings") != settings or previous.get("versions") != versions:
            raise ValueError("vLLM settings/version differ from the updated-model evaluation")
        if previous.get("model_path") != config["actor_rollout_ref"]["model"]["path"]:
            raise ValueError("Original initialization model differs from the recorded evaluation")
        reference_check = "matched"
    workers = config["trainer"]["n_gpus_per_node"] * config["trainer"]["nnodes"]
    if workers != settings["tensor_parallel_size"]:
        raise ValueError("Select a reference with one inference replica (worker count must equal tensor parallel size)")
    if args.num_threads < 1 or not math.isfinite(args.temperature) or args.temperature < 0:
        raise ValueError("BFCL threads must be positive and temperature must be finite and nonnegative")
    bfcl_reference_check = "unverified_legacy_reference"
    bfcl_record = reference / "eval/bfcl_environment.json"
    if bfcl_record.is_file():
        client = json.loads(bfcl_record.read_text(encoding="utf-8")).get("client")
        if client is not None:
            expected = dict(bfcl_model=args.bfcl_model, temperature=args.temperature, num_threads=args.num_threads)
            if client != expected:
                raise ValueError("BFCL client settings differ from the updated-model evaluation")
            bfcl_reference_check = "matched"

    # Read the recorded registry key as data; do not execute the experiment's shell.
    recorded_model = None
    run_env = reference / "run.env"
    if run_env.is_file():
        for line in run_env.read_text(encoding="utf-8").splitlines():
            if line.startswith("export BFCL_MODEL="):
                recorded_model = shlex.split(line[len("export BFCL_MODEL="):])[0]
    if recorded_model is not None and recorded_model != args.bfcl_model:
        raise ValueError(f"BFCL handler differs from reference: {args.bfcl_model!r} != {recorded_model!r}")

    case_source = (Path(args.case_ids).resolve() if getattr(args, "case_ids", None)
                   else reference / "data/case_ids.json")
    cases = json.loads(case_source.read_text(encoding="utf-8"))
    if not isinstance(cases, dict) or not cases:
        raise ValueError("Reference case_ids.json must contain selected categories")
    all_ids = []
    for category, ids in cases.items():
        if not isinstance(category, str) or not isinstance(ids, list) or not ids:
            raise ValueError("Each reference category must contain a nonempty ID list")
        if not all(isinstance(item, str) and item.startswith(category + "_") for item in ids):
            raise ValueError(f"Invalid case ID for {category}")
        all_ids.extend(ids)
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Reference contains duplicate case IDs")

    model_path = config["actor_rollout_ref"]["model"]["path"]
    model_name = serving.get("served_model_name") or "/".join(model_path.split("/")[-2:])
    settings.update(model=model_path, served_model_name=list(dict.fromkeys([
        model_name, "/".join(model_path.split("/")[-2:]), model_path,
    ])), host=args.host, port=args.port)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Use a new, empty baseline output directory: {output}")
    environment = dict(
        BASELINE_KIND="no_update", BASELINE_REFERENCE_RUN=str(reference),
        BASELINE_CASE_SOURCE=str(case_source), BASELINE_CATEGORIES=",".join(cases),
        MODEL_PATH=model_path, MODEL_NAME=model_name, BFCL_MODEL=args.bfcl_model,
        LOCAL_SERVER_ENDPOINT=args.host, LOCAL_SERVER_PORT=args.port,
        REMOTE_OPENAI_BASE_URL=f"http://{args.host}:{args.port}/v1",
        REMOTE_OPENAI_API_KEY="EMPTY", REMOTE_OPENAI_TOKENIZER_PATH=model_path,
        BFCL_EVAL_TEMPERATURE=args.temperature, BFCL_NUM_THREADS=args.num_threads,
        BFCL_PROJECT_ROOT=str(output),
    )
    manifest = dict(
        baseline_kind="no_update", reference_run=str(reference),
        reference_config=str(source_config), config_sha256=_sha256(source_config),
        case_ids_source=str(case_source), case_ids_sha256=_sha256(case_source),
        engine_args=settings, bfcl_model=args.bfcl_model,
        temperature=args.temperature, num_threads=args.num_threads,
        python=sys.version, packages=versions, reference_check=reference_check,
        bfcl_reference_check=bfcl_reference_check,
    )
    with (output / "baseline.env").open("x", encoding="utf-8") as handle:
        handle.writelines(f"export {key}={shlex.quote(str(value))}\n" for key, value in environment.items())
    with (output / "test_case_ids_to_generate.json").open("xb") as handle:
        handle.write(case_source.read_bytes())
    with (output / "reference_config.yaml").open("xb") as handle:
        handle.write(source_config.read_bytes())
    with (output / "baseline_config.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return settings


def run_server(settings):
    """Use vLLM's own parser and OpenAI server lifecycle."""
    import uvloop
    from vllm.entrypoints.openai.api_server import run_server as serve
    from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
    from vllm.entrypoints.utils import cli_env_setup
    from vllm.utils import FlexibleArgumentParser

    cli_env_setup()
    namespace = make_arg_parser(FlexibleArgumentParser()).parse_args([])
    for key, value in settings.items():
        setattr(namespace, key, value)
    validate_parsed_serve_args(namespace)
    uvloop.run(serve(namespace))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run")
    parser.add_argument("--case-ids", help="Override evaluation IDs, e.g. the reference run's heldout_case_ids.json")
    parser.add_argument("--output-dir")
    parser.add_argument("--bfcl-model")
    parser.add_argument("--record-bfcl-env")
    parser.add_argument("--compare-bfcl-env")
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1053)
    args = parser.parse_args(argv)
    if args.record_bfcl_env:
        if not args.bfcl_model:
            parser.error("--bfcl-model is required to record a BFCL evaluation")
        record_bfcl_environment(
            args.record_bfcl_env, args.compare_bfcl_env, bfcl_model=args.bfcl_model,
            temperature=args.temperature, num_threads=args.num_threads,
        )
        return
    if not all((args.reference_run, args.output_dir, args.bfcl_model)):
        parser.error("--reference-run, --output-dir, and --bfcl-model are required to start a baseline")
    if args.compare_bfcl_env:
        parser.error("--compare-bfcl-env requires --record-bfcl-env")
    # The training rollout server explicitly uses vLLM V1 as well.
    os.environ["VLLM_USE_V1"] = "1"
    os.environ.pop("VLLM_ATTENTION_BACKEND", None)
    from vllm import SamplingParams

    settings = prepare_baseline(args, SamplingParams)
    print(f"Baseline configuration saved in {args.output_dir}", flush=True)
    run_server(settings)


if __name__ == "__main__":
    main()
