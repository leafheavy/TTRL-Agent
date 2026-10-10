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

BFCL runs also exercise every training and validation dataset row on CPU before
initializing reference, actor or vLLM workers. This uses the real dataset path,
including chat templates and tool schemas, and retains all selected case IDs.
Pass `trainer.preflight_only=True` to the TTRL entry point to check configuration
and tokenized data and then exit without loading LLM weights. Prompt, response,
rollout model length and training token budgets should be configured together.
