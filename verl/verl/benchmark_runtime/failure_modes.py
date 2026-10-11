"""Benchmark-independent failure evidence and conservative recovery tracking.

Adapters parse their own response format, pass normalized calls to
``validate_calls``, and supply public operation contracts through ``FailureRules``.
This module has no benchmark, model, trainer, or JSON Schema dependency.
"""

import json
from copy import deepcopy


def validate_calls(calls, validators, collect_failures=True):
    """Validate every call; a rejected response never executes a valid prefix.

    Validators are supplied by the adapter and expose ``is_valid`` and
    ``iter_errors``. Disabling diagnostics preserves validation without building
    failure evidence.
    """
    failures = []

    def reject(kind, rule, evidence, call=None):
        if collect_failures:
            failures.append({"kind": kind, "rule": rule, "evidence": evidence, "call": call})

    if not isinstance(calls, list):
        reject("format_failure", "invalid_call_structure", "Tool calls must be a list")
        return None, failures
    normalized = []
    rejected = False
    for call in calls:
        if (
            not isinstance(call, dict)
            or not isinstance(call.get("name"), str)
            or not isinstance(call.get("arguments"), dict)
        ):
            reject("format_failure", "invalid_call_structure", "Each call requires a name and an arguments object")
            rejected = True
            continue
        candidate = {"name": call["name"], "arguments": call["arguments"]}
        try:
            json.dumps(candidate, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError):
            # Invalid values must not leak into JSON-serialized diagnostics.
            reject("format_failure", "non_json_call_value", "Tool calls must contain finite JSON values")
            rejected = True
            continue
        validator = validators.get(candidate["name"])
        if validator is None:
            reject("schema_failure", "unavailable_tool", candidate["name"], candidate)
            rejected = True
            continue
        if collect_failures:
            errors = list(validator.iter_errors(candidate["arguments"]))
            if errors:
                reject(
                    "schema_failure",
                    "invalid_arguments",
                    [
                        {
                            "path": list(error.absolute_path),
                            "constraint": error.validator,
                            "message": error.message[:256],
                        }
                        for error in errors
                    ],
                    candidate,
                )
                rejected = True
        elif not validator.is_valid(candidate["arguments"]):
            rejected = True
        normalized.append(candidate)
    return (None if rejected else normalized), failures


class FailureRules:
    """Adapter contract: unknown identities and outcomes never imply recovery."""

    def identity(self, call):
        """Return a stable operation/target key, or None when linkage is unclear."""
        return None

    def preserves_evidence(self, call):
        """Only known read-only operations may preserve earlier failure evidence."""
        return False

    def observe(self, call, result):
        """Return (failure record arguments or None, explicit success boolean)."""
        return None, False


class FailureDiagnostics:
    """Record confirmed errors and link only explicit corresponding recovery."""

    def __init__(self, rules=None, enabled=False):
        if not isinstance(enabled, bool):
            raise ValueError("failure_mode_detection.enabled must be a boolean")
        self.enabled = enabled
        self.rules = FailureRules() if rules is None else rules
        self.events = []
        self._identities = []
        self._evidence_at = []
        self._observation = 0
        self._turn = 0

    def record(self, kind, rule, evidence, call=None, turn=0):
        if not self.enabled:
            return None
        self._turn = max(self._turn, turn)
        event = {
            "id": len(self.events),
            "type": kind,
            "rule": rule,
            "evidence": deepcopy(evidence),
            "call": deepcopy(call),
            "turn": turn,
            "recovery": "unknown",
        }
        self.events.append(event)
        self._identities.append(deepcopy(self.rules.identity(call)))
        self._evidence_at.append(self._observation)
        return event

    def observe(self, call, result, turn=0):
        if not self.enabled:
            return
        self._turn = max(self._turn, turn)
        if not self.rules.preserves_evidence(call):
            # Later actions may repair or replace a target without a known retry.
            self._observation += 1
        failure, succeeded = self.rules.observe(call, result)
        if failure is not None:
            self.record(**failure, call=call, turn=turn)
        elif succeeded:
            identity = self.rules.identity(call)
            if identity is None:
                return
            for event, candidate in zip(self.events, self._identities):
                if event["turn"] == turn and candidate == identity and event["recovery"] != "recovered":
                    event["recovery"] = "recovered"
                    event["recovered_by"] = {"call": deepcopy(call), "turn": turn}

    def snapshot(self, termination, episode_completed, turn=None):
        if not self.enabled:
            return []
        events = deepcopy(self.events)
        active_turn = self._turn if turn is None else turn
        if termination == "completed" and episode_completed:
            for event, identity, evidence_at in zip(events, self._identities, self._evidence_at):
                if (
                    event["recovery"] != "recovered"
                    and identity is not None
                    and event["turn"] == active_turn
                    and evidence_at == self._observation
                    and self.rules.identity(event["call"]) == identity
                ):
                    event["recovery"] = "unrecovered"
        return events
