#!/usr/bin/env bash
# Sequential, independent BFCL adaptation; reuse the existing trainer and CLI.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
SUITE_ROOT=${RUN_ROOT:?Set RUN_ROOT to the parent experiment directory}
mkdir -p "$SUITE_ROOT"
SUITE_ROOT=$(cd "$SUITE_ROOT" && pwd)
MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH to the original initialization model}
BFCL_PYTHON=${BFCL_PYTHON:?Set BFCL_PYTHON to the Python executable in the BFCL environment}
TRAIN_PYTHON=${TRAIN_PYTHON:-$(command -v python)}
export TRAIN_PYTHON
BFCL_MODEL=${BFCL_MODEL:-Qwen/$(basename "$MODEL_PATH")-FC}
CONSENSUS=${CONSENSUS:-observations}
ENDPOINT=${BENCHMARK_RUNTIME_ENDPOINT:-http://127.0.0.1:1054}
POST_EVAL=False
TTRL_ENABLED=True
ROLLOUT_DATA_DIR=null
VALIDATION_DATA_DIR=null
for override in "$@"; do
  key=${override%%=*}
  key=${key#+}; key=${key#+}
  case "$key" in
    benchmark_runtime.endpoint) ENDPOINT=${override#*=} ;;
    ttrl.consensus.mode) CONSENSUS=${override#*=} ;;
    trainer.post_train_eval.enabled) POST_EVAL=${override#*=} ;;
    ttrl.enable) TTRL_ENABLED=${override#*=} ;;
    trainer.rollout_data_dir) ROLLOUT_DATA_DIR=${override#*=} ;;
    trainer.validation_data_dir) VALIDATION_DATA_DIR=${override#*=} ;;
  esac
done
RUN_LABEL=${RUN_LABEL:-$(basename "$MODEL_PATH")_${CONFIG_ID:-ttrl}_${CONSENSUS}}
if [ "${POST_EVAL,,}" != true ] || [ "${TTRL_ENABLED,,}" != true ]; then
  echo 'multi_turn suite requires TTRL and trainer.post_train_eval.enabled=True; use a concrete category for diagnostics.' >&2
  exit 1
fi
test -x "$BFCL_PYTHON"
test -x "$TRAIN_PYTHON"
# Fail before loading models if the BFCL registry/environment is incompatible.
"$BFCL_PYTHON" -c 'import sys; from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING; assert sys.argv[1] in MODEL_CONFIG_MAPPING, "BFCL model is not registered: " + sys.argv[1]' "$BFCL_MODEL"
read -r RUNTIME_HOST RUNTIME_PORT < <("$TRAIN_PYTHON" -c '
import sys
from urllib.parse import urlsplit
u = urlsplit(sys.argv[1])
if u.scheme != "http" or u.hostname not in ("localhost", "127.0.0.1") or not u.port or u.path not in ("", "/"):
    raise ValueError("The managed BFCL runtime requires a local HTTP endpoint with an explicit port")
print(u.hostname, u.port)
' "$ENDPOINT")
SERVER_SHA=$("$TRAIN_PYTHON" -c 'import hashlib; from pathlib import Path; print(hashlib.sha256(Path("verl/benchmark_runtime/server.py").read_bytes()).hexdigest())')

CATEGORIES=(multi_turn_base multi_turn_miss_func multi_turn_miss_param multi_turn_long_context)
for category in "${CATEGORIES[@]}"; do
  if [ -e "$SUITE_ROOT/$category" ]; then
    echo "Use a fresh RUN_ROOT; category output already exists: $SUITE_ROOT/$category" >&2
    exit 1
  fi
done
mkdir "$SUITE_ROOT/.multi_turn_suite.lock"
runtime_pid=
train_pid=
child_root=
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [ -n "$train_pid" ]; then
    # The process group belongs to this launcher; never stop a shared Ray cluster.
    kill -TERM -- "-$train_pid" 2>/dev/null || true
    wait "$train_pid" 2>/dev/null || true
  fi
  if [ -n "$runtime_pid" ]; then
    kill "$runtime_pid" 2>/dev/null || true
    wait "$runtime_pid" 2>/dev/null || true
  fi
  rmdir "$SUITE_ROOT/.multi_turn_suite.lock"
  if [ "$status" -ne 0 ]; then
    echo "Suite stopped; no subsequent category was started. Logs: ${child_root:-$SUITE_ROOT}" >&2
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for category in "${CATEGORIES[@]}"; do
  child_root="$SUITE_ROOT/$category"
  mkdir -p "$child_root/data" "$child_root/eval"
  export BFCL_PROJECT_ROOT="$child_root/eval"
  echo "[multi_turn] $category: original model=$MODEL_PATH; output=$child_root"
  data_args=(--bfcl-categories "$category" --bfcl-consensus "$CONSENSUS"
    --bfcl-max-samples "${MAX_TRAIN_SAMPLES:-0}" --output-dir "$child_root/data")
  if [ "${FS_ONLY:-False}" = True ]; then data_args+=(--bfcl-fs-only); fi
  "$BFCL_PYTHON" data/preprocess.py "${data_args[@]}" > "$child_root/preprocess.log" 2>&1

  # Restart our own runtime for each category, reusing the same port. PID and
  # loaded-code checks reject an unrelated or stale listener without stopping it.
  "$BFCL_PYTHON" -u verl/benchmark_runtime/server.py \
    --adapter verl/benchmark_runtime/bfcl_adapter.py \
    --host "$RUNTIME_HOST" --port "$RUNTIME_PORT" > "$child_root/bfcl_runtime.log" 2>&1 &
  runtime_pid=$!
  "$TRAIN_PYTHON" verl/benchmark_runtime/client.py --endpoint "$ENDPOINT" --backend bfcl \
    --wait-s 30 --expect-pid "$runtime_pid" --expect-server-sha256 "$SERVER_SHA"
  kill -0 "$runtime_pid"
  printf '%s\n' "$runtime_pid" > "$child_root/bfcl_runtime.pid"

  args=("$@"
    "benchmark_runtime.endpoint=$ENDPOINT"
    "actor_rollout_ref.model.path=$MODEL_PATH"
    "data.train_files=$child_root/data/train.parquet"
    "data.val_files=$child_root/data/test.parquet"
    "hydra.run.dir=$child_root/hydra"
    "trainer.default_local_dir=$child_root/checkpoints"
    "actor_rollout_ref.actor.gradient_record.output_dir=$child_root/gradients"
    "trainer.experiment_name=${RUN_LABEL}_$category"
    trainer.resume_mode=disable
    trainer.post_train_eval.enabled=True
    "trainer.post_train_eval.output_dir=$child_root/post_train_eval")
  if [ "$ROLLOUT_DATA_DIR" != null ]; then
    args+=("trainer.rollout_data_dir=$child_root/rollouts")
  fi
  if [ "$VALIDATION_DATA_DIR" != null ]; then
    args+=("trainer.validation_data_dir=$child_root/validation")
  fi
  # Save executable arguments and independent initialization for inspection.
  {
    if [ -f "$SUITE_ROOT/run.env" ]; then cat "$SUITE_ROOT/run.env"; fi
    printf 'export RUN_ROOT=%q\nexport TRAIN_CATEGORIES=%q\nexport EVAL_CATEGORIES=%q\n' "$child_root" "$category" "$category"
    printf 'export MODEL_PATH=%q\nexport BFCL_MODEL=%q\n' "$MODEL_PATH" "$BFCL_MODEL"
    BFCL_TRAIN_ARGS=("${args[@]}")
    declare -p BFCL_TRAIN_ARGS
  } > "$child_root/run.env"
  # A concrete category prevents recursive suite dispatch in aime.sh.
  RUN_ROOT="$child_root" TRAIN_CATEGORIES="$category" EVAL_CATEGORIES="$category" \
    setsid bash -e examples/ttrl/Qwen2.5/aime.sh "${args[@]}" > "$child_root/train.log" 2>&1 &
  train_pid=$!
  echo "[multi_turn] Training $category; follow $child_root/train.log"
  while [ ! -f "$child_root/post_train_eval/server.env" ]; do
    if ! kill -0 "$train_pid" 2>/dev/null; then
      wait "$train_pid" || exit $?
      echo "Training exited before publishing final weights: $child_root/train.log" >&2
      exit 1
    fi
    sleep 1
  done

  # Only this category's newly published API address serves its final weights.
  source "$child_root/post_train_eval/server.env"
  export NO_PROXY="${NO_PROXY:+$NO_PROXY,}${LOCAL_SERVER_ENDPOINT},127.0.0.1,localhost"
  export no_proxy="$NO_PROXY"
  "$TRAIN_PYTHON" - <<'PY'
import json
import os
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

base = os.environ["REMOTE_OPENAI_BASE_URL"].rstrip("/")
model = os.environ["REMOTE_OPENAI_TOKENIZER_PATH"]
opener = build_opener(ProxyHandler({}))
with opener.open(base + "/models", timeout=10) as response:
    names = [item["id"] for item in json.load(response)["data"]]
if model not in names:
    raise RuntimeError(f"BFCL request model {model!r} is absent from {names!r}")
payload = {"model": model, "prompt": "Hello", "max_tokens": 8, "temperature": 0}
request = Request(base + "/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
with opener.open(request, timeout=60) as response:
    result = json.load(response)
if not result.get("choices"):
    raise RuntimeError(f"Model probe returned no choices: {result}")
(Path(os.environ["BFCL_PROJECT_ROOT"]) / "inference_probe.json").write_text(
    json.dumps({"model": model, "served_models": names, "ok": True}, indent=2), encoding="utf-8")
PY
  cp "$child_root/data/case_ids.json" "$BFCL_PROJECT_ROOT/test_case_ids_to_generate.json"
  "$BFCL_PYTHON" -m bfcl_eval generate --model "$BFCL_MODEL" \
    --local-model-path "$REMOTE_OPENAI_TOKENIZER_PATH" --run-ids \
    --backend vllm --skip-server-setup --num-threads "${BFCL_NUM_THREADS:-1}" \
    2>&1 | tee "$child_root/eval/generate.log"
  if grep -Eq '(Error code: 404|openai[.]NotFoundError).*The model .*does not exist' "$child_root/eval/generate.log"; then
    echo 'Model-service 404; aborting before scoring.' >&2
    exit 1
  fi
  "$BFCL_PYTHON" data/preprocess.py --bfcl-check-results "$child_root/data/case_ids.json" \
    --bfcl-result-dir "$child_root/eval/result"
  "$BFCL_PYTHON" -m bfcl_eval evaluate --model "$BFCL_MODEL" \
    --test-category "$category" --partial-eval 2>&1 | tee "$child_root/eval/evaluate.log"
  touch "$child_root/post_train_eval/stop"
  wait "$train_pid"
  train_pid=
  kill "$runtime_pid"
  wait "$runtime_pid" 2>/dev/null || true
  runtime_pid=
  echo "[multi_turn] Completed $category; model and optimizer discarded."
done
"$BFCL_PYTHON" data/preprocess.py --bfcl-summarize-multi-turn "$SUITE_ROOT" \
  --bfcl-model "$BFCL_MODEL" --bfcl-run-label "$RUN_LABEL"
