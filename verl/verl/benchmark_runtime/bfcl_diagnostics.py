"""Public BFCL operation contracts for benchmark-independent diagnostics.

Return contracts below come from BFCL's public ``func_source_code`` modules:
``gorilla_file_system.GorillaFileSystem.rmdir`` and
``vehicle_control.VehicleControlAPI.{startEngine,setCruiseControl}`` in
https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard/bfcl_eval/eval_checker/multi_turn_eval
No simulator state, task answers, or additional tool execution is used.
"""

import importlib.util
import json
import math
from pathlib import Path

_rules_spec = importlib.util.spec_from_file_location("_failure_rules", Path(__file__).with_name("failure_modes.py"))
_rules = importlib.util.module_from_spec(_rules_spec)
_rules_spec.loader.exec_module(_rules)
FailureRules = _rules.FailureRules


def _number(value):
    return (isinstance(value, int) and not isinstance(value, bool)) or (
        isinstance(value, float) and math.isfinite(value)
    )


class BFCLFailureRules(FailureRules):
    """Require exact public errors and explicit success for known operations."""

    def __init__(self, classes, name_map=None):
        self.classes = set(classes)
        self.name_map = name_map or {}
        self._path_scope = 0

    def _name(self, call):
        name = call.get("name") if isinstance(call, dict) else None
        return self.name_map.get(name, name) if isinstance(name, str) else None

    def _operation(self, call):
        if not isinstance(call, dict) or not isinstance(call.get("arguments"), dict):
            return None
        name, args = self._name(call), call["arguments"]
        if "GorillaFileSystem" in self.classes and name == "rmdir":
            if set(args) == {"dir_name"} and isinstance(args["dir_name"], str) and args["dir_name"]:
                return name, args, self._path_scope
        if "VehicleControlAPI" in self.classes:
            if name == "startEngine" and args == {"ignitionMode": "START"}:
                return name, args, None
            if (
                name == "setCruiseControl"
                and set(args) == {"speed", "activate", "distanceToNextVehicle"}
                and args["activate"] is True
                and _number(args["speed"])
                and _number(args["distanceToNextVehicle"])
            ):
                return name, args, None
        return None

    def identity(self, call):
        operation = self._operation(call)
        if operation is None:
            return None
        name, args, scope = operation
        return scope, name, json.dumps(args, sort_keys=True, allow_nan=False)

    def preserves_evidence(self, call):
        return "GorillaFileSystem" in self.classes and self._name(call) == "pwd"

    @staticmethod
    def _precondition(name, args, observation):
        error = observation.get("error")
        if name == "rmdir" and error == f"rmdir: cannot remove '{args['dir_name']}': Directory not empty":
            return "directory_not_empty"
        if name == "startEngine":
            if error == "Brake pedal needs to be pressed when starting the engine.":
                return "brake_not_pressed"
            if error == "Must press the brake fully before starting the engine.":
                return "brake_not_fully_pressed"
            prefix = "All doors must be locked before starting the engine. Here are the unlocked doors: "
            if isinstance(error, str) and error.startswith(prefix):
                doors = error[len(prefix) :].split(", ")
                if (
                    doors
                    and len(set(doors)) == len(doors)
                    and set(doors) <= {"driver", "passenger", "rear_left", "rear_right"}
                ):
                    return "doors_not_locked"
        if name == "setCruiseControl" and error == "Start the engine before activating the cruise control.":
            return "engine_not_started"
        return None

    @staticmethod
    def _succeeded(name, args, observation):
        if "error" in observation:
            return False
        if name == "rmdir":
            return observation.get("result") == f"'{args['dir_name']}' removed"
        if name == "startEngine":
            return (
                observation.get("engineState") == "running"
                and _number(observation.get("fuelLevel"))
                and _number(observation.get("batteryVoltage"))
            )
        return (
            observation.get("cruiseStatus") == "active"
            and _number(observation.get("currentSpeed"))
            and observation["currentSpeed"] == args["speed"]
            and _number(observation.get("distanceToNextVehicle"))
            and observation["distanceToNextVehicle"] == args["distanceToNextVehicle"]
        )

    def observe(self, call, result):
        if self._name(call) == "cd":
            # Failed navigation can be ambiguous too; retire relative identities.
            self._path_scope += 1
        operation = self._operation(call)
        if operation is None:
            return None, False
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except ValueError:
                return None, False  # Generic executor exceptions are not policy errors.
        if not isinstance(result, dict):
            return None, False
        name, args, _ = operation
        rule = self._precondition(name, args, result)
        if rule:
            return {"kind": "precondition_violation", "rule": rule, "evidence": result["error"]}, False
        return None, self._succeeded(name, args, result)
