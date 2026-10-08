"""Trajectory equivalence and TTRL voting, independent of BFCL and training.

Only deployment-observable records are accepted by callers. This module does
not execute tools, parse model text, infer correctness, or inspect reference
answers. All modes produce one category and one binary reward per trajectory.
"""

import json
from collections import Counter

CONSENSUS_MODES = frozenset({"calls", "observations", "state"})


def validate_consensus_mode(mode):
    if not isinstance(mode, str) or mode not in CONSENSUS_MODES:
        raise ValueError("TTRL consensus mode must be calls, observations or state")
    return mode


def canonical_json(value):
    """Normalize dictionary keys while preserving sequence order and repeats."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def consensus_key(trajectory, mode):
    """Build a category from a complete ordered trace or its final state.

    Empty observations are not evidence. An empty call trace is comparable
    only when the model explicitly emitted the structured no-call result [].
    Natural-language wording never participates in any equivalence relation.
    """
    validate_consensus_mode(mode)
    if mode == "calls":
        value = trajectory["calls"]
        if not value and not trajectory.get("explicit_no_call", False):
            return None
    elif mode == "observations":
        value = trajectory["observations"]
        if not value:
            return None
    else:
        value = trajectory["final_state"]
        if not isinstance(value, dict) or not value:
            raise ValueError("State consensus requires a non-empty observable final-state projection")
    return canonical_json({"mode": mode, "value": value})


def majority_vote_labels(labels):
    """Reuse TTRL's first-observed tie rule; invalid rows remain in N."""
    if not labels:
        raise ValueError("Cannot vote on an empty rollout group")
    counts = Counter(label for label in labels if label is not None)
    if not counts:
        return None, 0.0
    label, count = counts.most_common(1)[0]
    return label, count / len(labels)


def vote_consensus(keys):
    """Select the highest-frequency category and return binary TTRL rewards."""
    majority_key, q = majority_vote_labels(keys)
    counts = Counter(key for key in keys if key is not None)
    scores = [float(key is not None and key == majority_key) for key in keys]
    return {
        "scores": scores,
        "keys": keys,
        "majority_key": majority_key,
        "majority_ratio": q,
        "valid": [key is not None for key in keys],
        "group_valid": majority_key is not None,
        "tied": bool(counts) and sum(count == max(counts.values()) for count in counts.values()) > 1,
        "informative": len(set(scores)) > 1,
    }
