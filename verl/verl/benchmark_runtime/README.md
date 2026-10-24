# Benchmark runtime

Model inference, AgentLoop orchestration, consensus voting and optimization stay
in the training environment. A separate process in each benchmark's virtual
environment owns its parsers, simulators, data loaders and official checkers.
Only JSON requests, tool observations, trajectory summaries and explicit
supervised scores cross the process boundary. No benchmark package is imported
by `client.py` or the training-side BFCL reward module.

For BFCL, activate the existing BFCL environment and start the standalone entry
point by file path. This avoids importing `verl.__init__` and does not require
installing verl, Ray, vLLM or the training environment into the BFCL environment.

```bash
source /root/group_data/yz/bfcl/bin/activate
cd /root/group_data/yz/projects/TTRL-Agent/verl
python verl/benchmark_runtime/server.py \
  --adapter verl/benchmark_runtime/bfcl_adapter.py \
  --host 127.0.0.1 --port 1054
```

Set `benchmark_runtime.endpoint=http://127.0.0.1:1054` in the training config.
The endpoint and timeout are carried in the config to Ray workers, rather than
depending on shell-variable inheritance. Startup checks the backend before Ray
or model allocation. No fallback imports BFCL into the training process.
The documented loopback endpoint supports the current single-node recipe.

Health reports the responding process's `pid`, `runtime_id`,
`started_at_unix_s`, and `server_sha256` captured when the server code is loaded.
When launching in the background, verify the returned PID against `$!` with
`client.py --endpoint ... --backend bfcl --expect-pid "$pid" --wait-s 30`.
Optionally pass `--expect-server-sha256` to check the loaded server revision.
Checking backend health and then `kill -0` separately does not prove that the
endpoint belongs to the newly launched process: an existing listener may reply
while the new child is still importing dependencies and has not yet failed to
bind the occupied port. Identity mismatches fail immediately. Readiness is
printed only after binding succeeds.

`active_sessions` counts rollout environments inside that process, not service
processes. A fresh runtime starts with zero sessions; each rollout creates one,
and failed runs may leave sessions until their idle TTL expires. A health check
does not create a session or test session creation under training load.

For BFCL multi-turn runs, keep `rollout.mode=async`,
`data.return_raw_chat=True` and GRPO enabled. Tool schemas and interactions come
from the remote session, so `multi_turn.tool_config_path` and
`multi_turn.interaction_config_path` can remain null. The trainer validates the
explicit benchmark runtime config for this path; ordinary multi-turn tools
still require a native tool or interaction config.

Each rollout owns a separate session. Requests within one session are serialized;
different sessions retain separate simulator namespaces. `close_session` is
idempotent, and abandoned idle sessions expire after `--session-ttl-s` (3600
seconds by default). Tool/scoring request failures fail the run and are never
converted to zero task rewards. Cleanup failures produce a warning and rely on
the session TTL, preserving any earlier generation error. Mutating requests
are not retried.

To add another benchmark, provide a Python file defining `Adapter` and start the
same server with `--adapter /path/to/adapter.py` in that benchmark's environment.
The generic server and client do not depend on BFCL. An adapter implements:

| Interface | Result |
| --- | --- |
| `name` | Backend identifier returned by health checks |
| `health()` | JSON metadata; validates benchmark imports in this process |
| `create_session(context)` | Session with `tool_schemas` and the operations below |
| `decode_responses(context, responses)` | Parsed single-turn calls, or `None` for invalid output |
| `score_outcomes(items, outcomes, answer_dir=None)` | Explicit supervised scores; never used for unlabeled TTRL |

Sessions implement `parse_calls(response)`, `execute(calls)`,
`malformed_response(reason=...)`, `finish_turn(reply)`, `outcome(termination)`
and `close()`. Tools and messages use JSON schemas and chat-message objects.
`outcome` supplies a complete trajectory's consensus key and validity; the
trainer performs the group vote and GRPO update. A benchmark still needs its
own input conversion and any differing agent/reward semantics, but does not
need to share dependencies with the model or other benchmark adapters.

The BFCL adapter keeps its previous simulation, validation and voting-key
semantics. Reference files are accessed only by `score_outcomes`, which is used
by the explicitly enabled supervised baseline. Session execution and the
unlabeled TTRL reward do not read reference answers.

## Operational failure diagnostics

`failure_mode_detection.enabled` is a benchmark-independent training setting,
disabled by default. Set `failure_mode_detection.enabled=True` to enable it in
the current BFCL AgentLoop. The loop copies this setting into each session's
`context["failure_mode_detection"]`, overriding dataset metadata. No server flag
is required. Direct runtime clients can supply the same context explicitly.
Disabling diagnostics skips recording and rule evaluation; the parser, schema
validation, tool behavior and task rewards retain their original behavior.

`failure_modes.py` contains only standard-library code. `FailureDiagnostics`
owns event history and conservative recovery tracking. `validate_calls` accepts
an existing parser's list of calls and a name-to-validator mapping (objects
providing `is_valid` and `iter_errors`), so adapters reuse their validators
without importing a benchmark into the core. It rejects the whole response if
any call is malformed or schema-invalid. With `collect_failures=False`, it only
validates and skips constructing diagnostic evidence. Response framing and
benchmark-specific parsing remain in the adapter.

To add rules for another benchmark, implement `FailureRules`:

| Method | Contract |
| --- | --- |
| `identity(call)` | Stable key for a known operation and its current scope, or `None` when association is uncertain. Include all relevant arguments. |
| `preserves_evidence(call)` | `True` only for a confirmed read-only operation that cannot repair or replace the failed target; defaults to `False`. |
| `observe(call, result)` | Return `(failure, succeeded)`. `failure` is `None` or a dict with `kind`, `rule`, `evidence`; `succeeded` is `True` only for explicit success of the identified operation. Update any scope here before subsequent identity checks. |

Each session creates `FailureDiagnostics(rules, enabled=...)`. Record confirmed
parser/schema failures with `record(kind, rule, evidence, call=..., turn=...)`, feed actual
executions to `observe(call, result, turn=...)`, and put
`snapshot(termination, episode_completed, turn=...)` in its outcome. A failure
from `validate_calls` already includes `call`; use `record(**failure, turn=...)`
for those entries. No task oracle, simulator state query or additional tool
execution is needed. Unknown rules abstain by default.

The core adds user-turn separation to rule-provided operation keys. Explicit
success of the same key can confirm recovery. Other actions conservatively
expire evidence for declaring an old error unrecovered. Only a completed
episode with current, identifiable failure evidence can produce `unrecovered`;
truncation, changed scope or uncertain association stays `unknown`. Recovery
does not delete the original error event. BFCL's public return contracts live
only in `bfcl_diagnostics.py`; its rules are not applied to other benchmarks.

Expose `failure_modes` and `failure_mode_detection_enabled` in AgentLoop
`extra_fields` to use the generic trainer log path. With
`trainer.rollout_data_dir=/path/to/rollouts`, each step's JSONL includes these
fields. An empty list means no recorded failures, and is not a task-success
signal. The enabled flag distinguishes disabled detection from no findings.
BFCL also retains the list inside `bfcl_outcome` for existing consumers. Other
benchmarks must connect their own adapter/AgentLoop to this core; enabling the
switch alone does not supply benchmark-specific rules or a new reward path.

## Diagnosis-guided advantage weighting

Detection and policy weighting have separate switches. Keeping
`failure_mode_detection.enabled=True` and
`failure_mode_detection.advantage_weight.enabled=False` preserves diagnostics
without changing advantages. The weighting switch defaults to false. To compare
only TTRL against TTRL + failure, keep detection enabled in both runs, turn off
`ttrl.q_weight.enabled`, and change only the failure-weighting switch under the
same case IDs, initialization, rollout budget and evaluation settings.

```yaml
failure_mode_detection:
  enabled: true
  advantage_weight:
    enabled: true
    min_weight: 0.5
    max_weight: 2.0
    type_weights:
      format_failure: 2.0
      schema_failure: 2.0
      precondition_violation: 1.5
```

The implementation borrows the positive trajectory-weight structure from
[SEAL §3.4, equations (10)–(12)](https://arxiv.org/html/2605.24426v1#S3.SS4).
Its local labels and utilities are adapted defaults, not a reproduction of the
paper's utility table; the paper's diagnosis and utility taxonomies are not
one-to-one. SEAL's environment-interface evolution is not implemented here.

For each actual user turn with a generation attempt, choose at most one dominant
label with priority `format_failure > schema_failure > precondition_violation`.
A turn with no recorded event has fixed neutral utility 1.0, which does not mean
task success. Repeated events in the same user turn do not add extra weight.
All confirmed historical failures contribute, including events whose recovery
status is `recovered` or `unknown`; recovery status does not select the utility.
`unknown` concerns recovery, not whether the recorded failure occurred.

For rollout $i$ with $L_i$ actual user turns, compute

$$
d_i=\operatorname{clip}\left(\frac1{L_i}\sum_{u=1}^{L_i}\rho(z_{iu}),
w_{\min},w_{\max}\right),\qquad
\widetilde A_i=w(q)d_iA_i^{\mathrm{GRPO}}.
$$

Apply this weight after GRPO normalization and do not normalize again. A positive
weight preserves the advantage sign: it can amplify either reinforcement or a
penalty, but cannot fix an incorrect consensus signal or create a gradient from
zero advantages. Raw rewards, returns, independent KL regularization and the
supervised side of paired gradient diagnostics remain unchanged. No reference
answers or official scores are used to compute the failure weight.

The current weighting path requires BFCL multi-turn asynchronous TTRL, shared
rollouts, GRPO, enabled detection and `algorithm.use_kl_in_reward=False`. Disable
the weighting switch for the independent BFCL supervised baseline. Standalone
failure-weight training is unsupported because a positive weight does not supply
a base advantage; constructing a new failure-derived reward would be a separate
algorithm.

Adapters/AgentLoops additionally expose `failure_mode_num_turns` in `extra_fields`:
the number of user turns actually processed with generation attempts, not the
scheduled total. Keep this field and `failure_modes` aligned with their rollout
rows through sampling and reordering. The trainer stores the per-row tensor
`ttrl_failure_weight`; rollout JSONL adds `failure_advantage_weight` when weighting
is enabled. Metrics include `train/failure_weight_mean`,
`train/failure_weight_min`, `train/failure_weight_max` and
`train/failure_detected_fraction`. Paired diagnostics record the `failure_weight`
configuration and compare the weighted unsupervised direction with the unchanged
supervised reference.

Run the transport and environment-boundary checks without Ray or a GPU:

```bash
python -m unittest discover -s tests/benchmark_runtime -p 'test_*_on_cpu.py' -v
```

These tests create two temporary virtual environments and install a tiny BFCL
fixture only in the benchmark environment. They exercise the real BFCL adapter
over HTTP while the model environment cannot import BFCL. The fixture verifies
integration boundaries; it does not substitute for a run against the installed
official BFCL package on the GPU node. The existing in-process official-library
checks in `tests/trainer/ppo/test_ttrl_bfcl_on_cpu.py` require activating the BFCL
environment and opting in with `BFCL_IN_PROCESS_TESTS=1` (plus that test suite's
training test dependencies).

The config regression tests compile and execute the trainer's actual validation
methods without importing Ray or GPU workers. They cover the smoke and supervised
remote-tool paths, reject invalid runtime/AgentLoop settings and preserve the
native multi-turn guards while blocking BFCL dependency imports.

The dispatch regression tests run the real AgentLoop manager and async worker
methods with metadata-only DataProto inputs and a stub model. They cover small
and uneven batches, task/context ordering, per-prompt sampling and validation,
without Ray or a GPU. Only workers with a nonempty chunk are dispatched; tasks
are never padded or repeated to fill the worker pool.

BFCL runs also exercise every training and validation dataset row on CPU before
initializing reference, actor or vLLM workers. This uses the real dataset path,
including chat templates and tool schemas, and retains all selected case IDs.
Pass `trainer.preflight_only=True` to the TTRL entry point to check configuration
and tokenized data and then exit without loading LLM weights. Prompt, response,
rollout model length and training token budgets should be configured together.
