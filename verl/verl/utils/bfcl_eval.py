"""Reuse BFCL's evaluator and a trained run's inference settings."""

import argparse
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener


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


def _load_case_ids(path):
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cases, dict) or not cases:
        raise ValueError("case_ids.json must contain selected categories")
    all_ids = []
    for category, ids in cases.items():
        if not isinstance(category, str) or not isinstance(ids, list) or not ids:
            raise ValueError("Each reference category must contain a nonempty ID list")
        if not all(isinstance(item, str) and item.startswith(category + "_") for item in ids):
            raise ValueError(f"Invalid case ID for {category}")
        all_ids.extend(ids)
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Reference contains duplicate case IDs")
    return cases


def _reference_eval_record(reference, cases):
    """Select the fingerprint for the evaluation whose cases are being compared."""
    matched = []
    for side in ("train", "heldout"):
        directory = reference / f"eval_{side}"
        manifest = directory / "test_case_ids_to_generate.json"
        if manifest.is_file():
            selected = _load_case_ids(manifest)
            if selected.keys() == cases.keys() and all(set(selected[key]) == set(cases[key]) for key in cases):
                matched.append(directory / "bfcl_environment.json")
    if len(matched) > 1:
        raise ValueError("Reference evaluations have ambiguous case IDs")
    return matched[0] if matched else reference / "eval/bfcl_environment.json"


def _probe_model(model, output):
    base = os.environ["REMOTE_OPENAI_BASE_URL"].rstrip("/")
    headers = {"Authorization": "Bearer " + os.environ.get("REMOTE_OPENAI_API_KEY", "EMPTY")}
    opener = build_opener(ProxyHandler({}))
    with opener.open(Request(base + "/models", headers=headers), timeout=10) as response:
        names = [item["id"] for item in json.load(response)["data"]]
    if model not in names:
        raise RuntimeError(f"BFCL request model {model!r} is absent from {names!r}")
    payload = {"model": model, "prompt": "Hello", "max_tokens": 8, "temperature": 0, "n": 1}
    request = Request(base + "/completions", data=json.dumps(payload).encode(),
                      headers={**headers, "Content-Type": "application/json"})
    with opener.open(request, timeout=60) as response:
        result = json.load(response)
    if not result.get("choices"):
        raise RuntimeError(f"Model probe returned no choices: {result}")
    (output / "inference_probe.json").write_text(
        json.dumps({"model": model, "served_models": names, "ok": True}, indent=2), encoding="utf-8"
    )


def _run_bfcl(command, log, environment):
    """Stream the unchanged official CLI output to both the terminal and its log."""
    with log.open("w", encoding="utf-8") as handle, subprocess.Popen(
        [sys.executable, "-m", "bfcl_eval", *command], env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
    ) as process:
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            print(line, end="", flush=True)
        status = process.wait()
    if status:
        raise subprocess.CalledProcessError(status, process.args)
    if "generate" == command[0]:
        contents = log.read_text(encoding="utf-8")
        if ("Error code: 404" in contents or "openai.NotFoundError" in contents) and "does not exist" in contents:
            raise RuntimeError("Model-service 404; aborting before scoring")


def _category_file(directory, category, kind):
    paths = list(directory.rglob(f"BFCL_v*_{category}_{kind}.json"))
    if len(paths) != 1:
        raise ValueError(f"Expected one official {kind} file for {category}, got {paths}")
    return paths[0]


def evaluate_cases(args):
    """Evaluate one fixed actor; publish selected raw BFCL files without nesting."""
    source = Path(args.evaluate_case_ids).resolve()
    cases = _load_case_ids(source)
    output = Path(args.output_dir).resolve()
    score_only = args.score_only
    if args.num_threads < 1 or not math.isfinite(args.temperature) or args.temperature < 0:
        raise ValueError("BFCL threads must be positive and temperature must be finite and nonnegative")
    if args.compare_bfcl_env and not Path(args.compare_bfcl_env).is_file():
        raise FileNotFoundError(args.compare_bfcl_env)
    if score_only:
        saved = output / "test_case_ids_to_generate.json"
        if _load_case_ids(saved) != cases:
            raise ValueError("Saved evaluation case IDs differ from the requested cases")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new, empty evaluation output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = output / "bfcl_environment.json"
    if score_only and not fingerprint.is_file():
        raise FileNotFoundError(fingerprint)
    record_bfcl_environment(
        fingerprint, fingerprint if score_only else args.compare_bfcl_env,
        bfcl_model=args.bfcl_model, temperature=args.temperature, num_threads=args.num_threads,
    )
    if score_only and args.compare_bfcl_env:
        record_bfcl_environment(fingerprint, args.compare_bfcl_env, bfcl_model=args.bfcl_model,
                                temperature=args.temperature, num_threads=args.num_threads)
    saved = output / "test_case_ids_to_generate.json"
    if not score_only:
        saved.write_bytes(source.read_bytes())
        (output / "case_ids_source.txt").write_text(str(source) + "\n", encoding="utf-8")
        (output / "case_ids.sha256").write_text(_sha256(saved) + "\n", encoding="utf-8")
        _probe_model(args.local_model_path, output)
    else:
        # A failed rescore must not leave an earlier success marker current.
        (output / "summary.json").unlink(missing_ok=True)

    # BFCL owns its native working layout. Only this newly created temporary
    # directory is removed; the selected raw outputs are published unchanged.
    with tempfile.TemporaryDirectory(prefix=".bfcl_work_", dir=output) as temporary:
        work = Path(temporary).resolve()
        if work.parent != output:
            raise ValueError("BFCL temporary work directory escaped the evaluation output")
        environment = dict(os.environ, BFCL_PROJECT_ROOT=str(work), PYTHONUNBUFFERED="1", PYTHONUTF8="1")
        (work / "test_case_ids_to_generate.json").write_bytes(saved.read_bytes())
        # Delay benchmark imports until its writable project root is available.
        previous_root = os.environ.get("BFCL_PROJECT_ROOT")
        os.environ["BFCL_PROJECT_ROOT"] = str(work)
        try:
            project = str(Path(__file__).resolve().parents[2])
            if project not in sys.path:
                sys.path.insert(0, project)
            from bfcl_eval.utils import get_directory_structure_by_category

            from data.preprocess import check_bfcl_result_ids

            model_directory = work / "result" / args.bfcl_model.replace("/", "_")
            if score_only:
                for category in cases:
                    destination = model_directory / get_directory_structure_by_category(category)
                    destination.mkdir(parents=True, exist_ok=True)
                    original = _category_file(output, category, "result")
                    shutil.copy2(original, destination / original.name)
            else:
                try:
                    _run_bfcl([
                        "generate", "--model", args.bfcl_model, "--local-model-path", args.local_model_path,
                        "--run-ids", "--backend", "vllm", "--skip-server-setup",
                        "--temperature", str(args.temperature), "--num-threads", str(args.num_threads),
                    ], output / "generate.log", environment)
                finally:
                    # Even a failed run leaves available raw trajectories for diagnosis.
                    for category in cases:
                        for result in model_directory.rglob(f"BFCL_v*_{category}_result.json"):
                            shutil.copy2(result, output / result.name)
            for category in cases:
                _category_file(model_directory, category, "result")
            check_bfcl_result_ids(cases, model_directory)
            _run_bfcl([
                "evaluate", "--model", args.bfcl_model, "--test-category", ",".join(cases), "--partial-eval",
            ], output / "evaluate.log", environment)
            rows = []
            for category, ids in cases.items():
                result = _category_file(model_directory, category, "result")
                score = _category_file(work / "score" / args.bfcl_model.replace("/", "_"), category, "score")
                with score.open(encoding="utf-8") as handle:
                    header = json.loads(next(line for line in handle if line.strip()))
                correct, total = header["correct_count"], header["total_count"]
                if (
                    not isinstance(correct, int) or isinstance(correct, bool)
                    or not isinstance(total, int) or isinstance(total, bool)
                    or total != len(ids) or not 0 <= correct <= total
                ):
                    raise ValueError(f"Official score counts disagree with selected cases for {category}: {header}")
                rows.append(dict(category=category, correct_count=correct, total_count=total, accuracy=correct / total,
                                 result_file=result.name, score_file=score.name))
            for row in rows:
                score = _category_file(work / "score" / args.bfcl_model.replace("/", "_"), row["category"], "score")
                shutil.copy2(score, output / score.name)
            summary = dict(bfcl_model=args.bfcl_model, case_ids_sha256=_sha256(saved), categories=rows,
                           average_accuracy=sum(row["accuracy"] for row in rows) / len(rows))
            (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        finally:
            if previous_root is None:
                os.environ.pop("BFCL_PROJECT_ROOT", None)
            else:
                os.environ["BFCL_PROJECT_ROOT"] = previous_root
    print(f"BFCL evaluation: {output / 'summary.json'}", flush=True)
    return summary


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
    cases = _load_case_ids(case_source)
    bfcl_reference_check = "unverified_legacy_reference"
    bfcl_record = _reference_eval_record(reference, cases)
    if bfcl_record.is_file():
        client = json.loads(bfcl_record.read_text(encoding="utf-8")).get("client")
        if client is not None:
            expected = dict(bfcl_model=args.bfcl_model, temperature=args.temperature, num_threads=args.num_threads)
            if client != expected:
                raise ValueError("BFCL client settings differ from the updated-model evaluation")
            bfcl_reference_check = "matched"

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
        BASELINE_CASE_OVERRIDE=bool(getattr(args, "case_ids", None)),
        BASELINE_BFCL_REFERENCE_ENV=str(bfcl_record),
        BASELINE_EVAL_SPLITS=os.environ.get("BASELINE_EVAL_SPLITS", os.environ.get("BFCL_EVAL_SPLIT", "both")),
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
        bfcl_reference_environment=str(bfcl_record),
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
    parser.add_argument(
        "--evaluate-case-ids", help="Generate and score these exact cases using the existing model service"
    )
    parser.add_argument("--local-model-path", help="Tokenizer/config path and request model ID of the existing service")
    parser.add_argument(
        "--score-only", action="store_true", help="Rescore flat raw trajectories without model inference"
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--bfcl-model")
    parser.add_argument("--record-bfcl-env")
    parser.add_argument("--compare-bfcl-env")
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1053)
    args = parser.parse_args(argv)
    if args.evaluate_case_ids:
        if not all((args.output_dir, args.bfcl_model)) or (not args.score_only and not args.local_model_path):
            parser.error(
                "Evaluation requires --output-dir, --bfcl-model, and --local-model-path unless --score-only"
            )
        if args.record_bfcl_env or args.reference_run or args.case_ids:
            parser.error("--evaluate-case-ids cannot be combined with baseline or fingerprint-only modes")
        evaluate_cases(args)
        return
    if args.score_only or args.local_model_path:
        parser.error("--score-only and --local-model-path require --evaluate-case-ids")
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
    # This file shares the installed benchmark package's name. A direct script
    # launch must not resolve `import bfcl_eval` back to this file.
    script_directory = Path(__file__).resolve().parent
    sys.path[:] = [entry for entry in sys.path if Path(entry or os.getcwd()).resolve() != script_directory]
    main()
