#!/usr/bin/env python3
"""Run exactly five guarded ACT V3 commands on the SO-101 follower.

This is the second intentionally commanding commissioning gate.  It accepts
the completed 60-second live shadow and the reviewed controlled-Home run,
loads the frozen 11K ACT policy, performs one pre-hardware dummy CUDA warmup,
acquires one fresh paired front/wrist observation, performs exactly one live
ACT inference, and transmits exactly the first five guarded delta commands.

The process is deliberately finite:

1. Reproduce the read-only review of the existing controlled-Home trace.
2. Reproduce the zero-write failure of the first five-command attempt and
   require the live pose to remain near the already executed controlled-Home
   trace and inside its narrowly expanded joint envelope.
3. Seed the measured current goal with torque disabled, enable torque, and
   recover monotonically to frozen Home without disabling torque afterward.
4. Capture one fresh paired observation and run ACT exactly once on live data.
5. Reject the whole policy chunk before motion if any of its first five deltas
   needs rate/soft clipping or leaves the commissioning excursion envelope.
6. Send exactly five policy-derived Goal_Position commands at 15 Hz, hold the
   last command for two seconds under a stricter tracking bound, return to
   frozen Home, then disable torque and close the serial bus.

Both successful and failed runs finish with torque disabled.  Stable external
support must prevent a torque-off fall without putting hands in a joint,
gripper, sweep, or pinch zone.  This gate does not authorize an autonomous
episode or general hardware deployment.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import math
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence


MOTOR_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

EXPECTED_BRIDGE_SHA256 = (
    "331ad2a464c22b76eb817bfeaf38f6da43d6457150171ad75d6c579889dc314b"
)
EXPECTED_PREPROCESS_SHA256 = (
    "63215b4ac625afa87abc6262b560b4383d378b61828d4582c9e4e4781e4c65b6"
)
EXPECTED_ASSEMBLER_SHA256 = (
    "866a53d8b4ce1473344779c493a955d03dafa485115fab34a4ffc929ff31d424"
)
EXPECTED_GUARD_SHA256 = (
    "e06e785071ee3ded95e89d670978bcb4a8caf104b2fc69fc87de37cb6fff6d15"
)
EXPECTED_ONE_SHOT_SHA256 = (
    "c2f2be0481e72aa1836bae36e873be59b9c739b776d1baf5b510954264b9196b"
)
EXPECTED_OFFLINE_ADAPTER_SHA256 = (
    "0deb4d70d4fc3b8e0af205bfd047a7167c53736b933c39906d79ffd767d3fbc3"
)
EXPECTED_LIVE_SHADOW_SHA256 = (
    "978928157e4fb441d95c4a385f3192f463065723b61bda3d404f9fe4b7018c02"
)
EXPECTED_CONTROLLED_HOME_SHA256 = (
    "ecb40000ffc60337a2bc56c8b0490086f6fd476acf5e70e21e9c818c7bb8fac6"
)
EXPECTED_FOLLOWER_SHA256 = (
    "26c675c71ade2670fa1bed2d887507b6ef3ad53a49e20182f5ba3d032a0afd7a"
)
EXPECTED_FOLLOWER_CONFIG_SHA256 = (
    "e5702d9b6c4a09d10de83f912a1640169698292d5145091d403f4fdce0211cbe"
)
EXPECTED_FOLLOWER_BY_ID = (
    "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82110810-if00"
)
EXPECTED_HOME_FAILURE_DECISION = (
    "CONTROLLED_HOME_FAILED_MOTION_AND_DEPLOYMENT_REMAIN_BLOCKED"
)
EXPECTED_HOME_REVIEW_DECISION = (
    "CONTROLLED_HOME_REVIEW_ACCEPTED_WITHIN_5_DEG_COMMISSIONING_TOLERANCE"
)
EXPECTED_SHADOW_DECISION = (
    "LIVE_SHADOW_PASS_PIPELINE_START_POSE_NOT_MOTION_READY_"
    "PREPARE_CONTROLLED_HOME_GATE_NEXT"
)
EXPECTED_PREVIOUS_FIVE_SHA256 = (
    "7606293066b081b5d2611eec84344b58be0a354f3864e8861456879d2ef819af"
)
EXPECTED_PREVIOUS_FIVE_FAILURE_DECISION = (
    "FIVE_COMMAND_POLICY_COMMISSIONING_FAILED_DEPLOYMENT_REMAINS_BLOCKED"
)

EXPECTED_POLICY_STEP = 11000
EXPECTED_CHUNK_SIZE = 10
POLICY_COMMAND_COUNT = 5
CONTROL_HZ = 15.0
COUNTDOWN_SECONDS = 5
CURRENT_GOAL_SETTLE_SECONDS = 1.0
HOME_ALIGNMENT_HOLD_SECONDS = 2.0
POLICY_HOLD_SECONDS = 2.0
FINAL_HOME_HOLD_SECONDS = 2.0
CAMERA_STREAM_SECONDS = 40.0
CAMERA_FINISH_GRACE_SECONDS = 15.0
PAIR_WAIT_TIMEOUT_SECONDS = 0.30
MAX_CAMERA_AGE_SECONDS = 0.250
MAX_JOINT_AGE_SECONDS = 0.100
MAX_CAMERA_SKEW_SECONDS = 0.100
MAX_SINGLE_INFERENCE_MS = 500.0

START_HOME_TOLERANCE = 5.0
FINAL_HOME_TOLERANCE = 5.0
COUNTDOWN_DRIFT_LIMIT = 3.0
GOAL_READBACK_TOLERANCE = 1.5
TORQUE_LOCK_TOLERANCE = 3.0
HOME_ALIGNMENT_DEADLINE_SECONDS = 15.0
POLICY_SEQUENCE_DEADLINE_SECONDS = 2.0
RETURN_HOME_DEADLINE_SECONDS = 5.0

# A short five-command gate gets an immediate tighter bound rather than
# waiting two seconds for the general runtime tracking watchdog.
COMMISSIONING_TRACKING_LIMITS = (8.0, 8.0, 8.0, 8.0, 8.0, 10.0)

# Maximum absolute commanded excursion from frozen Home over all five policy
# commands.  These are no larger than five frozen per-command limits and are
# deliberately tighter for wrist/gripper axes during first motion.
POLICY_EXCURSION_LIMITS = (7.5, 6.0, 6.0, 5.0, 5.0, 5.0)

# Recovery is authorized only near the real, previously executed no-trip Home
# trajectory.  One degree is below the existing 1.5-degree goal-readback and
# 3-degree start/lock tolerances, while accommodating sub-degree encoder
# quantization/backlash seen in the zero-write failed gate.
RECOVERY_ENVELOPE_TOLERANCE = 1.0
RECOVERY_TRACE_TOLERANCE = 5.0
FAILED_START_DRIFT_TOLERANCE = 3.0


class FiveCommandError(RuntimeError):
    """A frozen dependency, live input, guard, tracking, or motion gate failed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-bridge", type=Path, required=True)
    parser.add_argument("--preprocess-adapter", type=Path, required=True)
    parser.add_argument("--assembler", type=Path, required=True)
    parser.add_argument("--guard-core", type=Path, required=True)
    parser.add_argument("--one-shot-launcher", type=Path, required=True)
    parser.add_argument("--one-shot-report", type=Path, required=True)
    parser.add_argument("--offline-runtime-adapter", type=Path, required=True)
    parser.add_argument("--live-shadow-runner", type=Path, required=True)
    parser.add_argument("--controlled-home-gate", type=Path, required=True)
    parser.add_argument("--controlled-home-report", type=Path, required=True)
    parser.add_argument("--controlled-home-trace", type=Path, required=True)
    parser.add_argument(
        "--failed-five-command-report",
        type=Path,
        required=True,
        help=(
            "Preserved report from the first zero-policy-write commissioning "
            "failure; anchors the merged recovery pose."
        ),
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--source-checkpoint-root", type=Path, required=True)
    parser.add_argument("--follower-source", type=Path, required=True)
    parser.add_argument("--follower-config-source", type=Path, required=True)
    parser.add_argument("--follower-port", type=Path, required=True)
    parser.add_argument(
        "--windows-python",
        default=(
            "/mnt/c/Users/Administrator/venvs/"
            "act-v3-camera-bridge/Scripts/python.exe"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--authorize-five-command-policy-motion",
        action="store_true",
        help="Authorize exactly five guarded policy-derived Goal_Position writes.",
    )
    parser.add_argument(
        "--confirm-arm-supported",
        action="store_true",
        help=(
            "Confirm stable external support prevents a torque-off fall without "
            "placing hands in the motion or pinch zone."
        ),
    )
    parser.add_argument(
        "--confirm-scene-ready",
        action="store_true",
        help=(
            "Confirm the reviewed red cube is at the training start location and "
            "all other objects are outside the robot sweep."
        ),
    )
    parser.add_argument(
        "--confirm-hands-clear",
        action="store_true",
        help="Confirm every person remains outside all robot and gripper sweep zones.",
    )
    parser.add_argument(
        "--confirm-power-cutoff-ready",
        action="store_true",
        help="Confirm the follower 12 V cutoff is immediately reachable.",
    )
    return parser.parse_args(argv)


def require_file(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def require_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise NotADirectoryError(resolved)
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise FiveCommandError(
            f"{label} SHA256 mismatch: expected={expected}, actual={actual}"
        )
    return actual


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FiveCommandError(f"invalid JSON {path}: {exc}") from exc


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise FiveCommandError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def finite_vector(label: str, values: Sequence[Any]) -> tuple[float, ...]:
    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise FiveCommandError(f"{label} must be numeric") from exc
    if len(vector) != len(MOTOR_ORDER):
        raise FiveCommandError(
            f"{label} must contain {len(MOTOR_ORDER)} values, got {len(vector)}"
        )
    invalid = [index for index, value in enumerate(vector) if not math.isfinite(value)]
    if invalid:
        raise FiveCommandError(f"{label} contains nonfinite values at {invalid}")
    return vector


def maximum_difference(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[str, float]:
    a = finite_vector("left", left)
    b = finite_vector("right", right)
    differences = [abs(x - y) for x, y in zip(a, b, strict=True)]
    index = max(range(len(differences)), key=differences.__getitem__)
    return MOTOR_ORDER[index], differences[index]


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def verify_static_capability_boundary(path: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]

    def attr_name(call: ast.Call) -> str | None:
        return call.func.attr if isinstance(call.func, ast.Attribute) else None

    by_attr: dict[str, list[ast.Call]] = {}
    for call in calls:
        name = attr_name(call)
        if name is not None:
            by_attr.setdefault(name, []).append(call)

    expected_counts = {
        "connect_motor_bus": 1,
        "read_operating_mode": 1,
        "read_present_position": 1,
        "read_goal_position": 1,
        "write_goal_position": 1,
        "enable_position_torque": 1,
        "disable_position_torque": 1,
        "disconnect_motor_bus": 1,
        "inference_chunk": 1,
        "from_pretrained": 1,
    }
    count_mismatches = {
        name: {"expected": expected, "actual": len(by_attr.get(name, []))}
        for name, expected in expected_counts.items()
        if len(by_attr.get(name, [])) != expected
    }
    forbidden = {
        "sync_write",
        "sync_read",
        "enable_torque",
        "disable_torque",
        "send_action",
        "write",
        "write_calibration",
        "configure",
        "configure_motors",
        "calibrate",
        "setup_motors",
    }
    forbidden_found = sorted(forbidden.intersection(by_attr))
    follower_calls = sorted(
        ast.unparse(call)
        for call in calls
        if ast.unparse(call).startswith(
            ("follower.connect(", "follower.disconnect(")
        )
    )
    if count_mismatches or forbidden_found or follower_calls:
        raise FiveCommandError(
            "five-command source capability boundary mismatch: "
            f"counts={count_mismatches}, forbidden={forbidden_found}, "
            f"follower_calls={follower_calls}"
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "indirect_goal_write_wrapper_call_sites": 1,
        "indirect_torque_enable_wrapper_call_sites": 1,
        "indirect_torque_disable_wrapper_call_sites": 1,
        "direct_raw_bus_register_calls": False,
        "full_follower_connect_or_disconnect_calls": False,
        "policy_inference_call_sites": 1,
        "pass": True,
    }


def verify_controlled_home_review(
    *,
    gate_path: Path,
    report_path: Path,
    trace_path: Path,
) -> dict[str, Any]:
    gate_sha = require_sha(
        gate_path,
        EXPECTED_CONTROLLED_HOME_SHA256,
        "controlled Home gate",
    )
    report = load_json(report_path)
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise FiveCommandError("unexpected controlled-Home report schema")
    with trace_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise FiveCommandError("controlled-Home trace is empty")

    motion = report.get("motion_contract") or {}
    run = report.get("run") or {}
    scope = report.get("scope") or {}
    failure = report.get("failure") or {}
    static = report.get("static_motion_boundary") or {}
    if tuple(motion.get("motor_order", ())) != MOTOR_ORDER:
        raise FiveCommandError("controlled-Home motor order mismatch")
    home_command = finite_vector("reviewed Home", motion.get("home_command", ()))
    measured_start = finite_vector(
        "controlled-Home measured start",
        run.get("measured_start", ()),
    )
    final_q = finite_vector(
        "controlled-Home final_q",
        run.get("final_q_before_torque_disable", ()),
    )
    tracking_limits_raw = motion.get("tracking_error_limits") or {}
    if set(tracking_limits_raw) != set(MOTOR_ORDER):
        raise FiveCommandError("controlled-Home tracking limits are missing")
    tracking_limits = tuple(
        float(tracking_limits_raw[motor]) for motor in MOTOR_ORDER
    )

    move_rows = [row for row in rows if row.get("phase") == "move"]
    hold_rows = [row for row in rows if row.get("phase") == "hold"]
    trip_rows = [row for row in rows if int(row["tracking_tripped"]) != 0]
    final_errors = tuple(
        abs(actual - target)
        for actual, target in zip(final_q, home_command, strict=True)
    )
    trace_max = tuple(
        max(float(row[f"tracking_error_{motor}"]) for row in rows)
        for motor in MOTOR_ORDER
    )
    shadow_gate = (
        report.get("frozen_dependencies", {}).get("live_shadow_gate", {})
    )
    checks = {
        "source_identity": static.get("sha256") == gate_sha,
        "expected_precision_only_failure": (
            report.get("decision") == EXPECTED_HOME_FAILURE_DECISION
            and failure.get("type") == "ControlledHomeError"
            and str(failure.get("message", "")).startswith(
                "final Home tolerance failed:"
            )
        ),
        "shadow_gate": (
            shadow_gate.get("decision") == EXPECTED_SHADOW_DECISION
            and shadow_gate.get("runner_sha256") == EXPECTED_LIVE_SHADOW_SHA256
            and shadow_gate.get("no_commands_sent") is True
        ),
        "motion_commands": len(move_rows) == 106,
        "home_hold": len(hold_rows) >= 40,
        "no_tracking_trip": not trip_rows,
        "trace_count": len(rows) == int(run.get("trace_rows", -1)),
        "motion_deadline": float(run.get("trajectory_duration_seconds", math.inf))
        <= float(motion.get("maximum_move_seconds", -1.0)),
        "final_within_5_degrees": max(final_errors) <= START_HOME_TOLERANCE,
        "trace_within_frozen_limits": all(
            value <= limit
            for value, limit in zip(trace_max, tracking_limits, strict=True)
        ),
        "cleanup": (
            run.get("cleanup_errors") == []
            and scope.get("torque_disabled_at_exit") is True
            and scope.get("serial_closed") is True
        ),
    }
    if not all(checks.values()):
        raise FiveCommandError(f"controlled-Home review rejected: {checks}")
    return {
        "review_decision": EXPECTED_HOME_REVIEW_DECISION,
        "gate_path": str(gate_path),
        "gate_sha256": gate_sha,
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "trace_path": str(trace_path),
        "trace_sha256": sha256_file(trace_path),
        "checks": checks,
        "move_rows": len(move_rows),
        "hold_rows": len(hold_rows),
        "measured_start": list(measured_start),
        "home_command": list(home_command),
        "final_q": list(final_q),
        "final_error_by_joint": dict(zip(MOTOR_ORDER, final_errors, strict=True)),
        "trace_max_error_by_joint": dict(zip(MOTOR_ORDER, trace_max, strict=True)),
    }


def load_reviewed_trace_poses(trace_path: Path) -> list[dict[str, Any]]:
    with trace_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    poses: list[dict[str, Any]] = []
    for row in rows:
        poses.append(
            {
                "phase": row.get("phase"),
                "sequence": int(row["sequence"]),
                "actual_q": finite_vector(
                    "controlled-Home trace actual_q",
                    [row[f"actual_{motor}"] for motor in MOTOR_ORDER],
                ),
            }
        )
    if not poses:
        raise FiveCommandError("controlled-Home trace contains no poses")
    return poses


def recovery_pose_status(
    pose: Sequence[float],
    *,
    reviewed_start: Sequence[float],
    home_command: Sequence[float],
    trace_poses: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    vector = finite_vector("recovery pose", pose)
    start = finite_vector("reviewed recovery start", reviewed_start)
    home = finite_vector("recovery Home", home_command)
    envelope_excess = tuple(
        max(min(source, target) - value, value - max(source, target), 0.0)
        for value, source, target in zip(vector, start, home, strict=True)
    )
    nearest = min(
        trace_poses,
        key=lambda item: max(
            abs(value - reference)
            for value, reference in zip(
                vector,
                item["actual_q"],
                strict=True,
            )
        ),
    )
    nearest_errors = tuple(
        abs(value - reference)
        for value, reference in zip(
            vector,
            nearest["actual_q"],
            strict=True,
        )
    )
    return {
        "pose": list(vector),
        "envelope_excess_by_joint": dict(
            zip(MOTOR_ORDER, envelope_excess, strict=True)
        ),
        "max_envelope_excess": max(envelope_excess),
        "nearest_trace_phase": nearest["phase"],
        "nearest_trace_sequence": nearest["sequence"],
        "nearest_trace_error_by_joint": dict(
            zip(MOTOR_ORDER, nearest_errors, strict=True)
        ),
        "nearest_trace_max_error": max(nearest_errors),
        "pass": (
            max(envelope_excess) <= RECOVERY_ENVELOPE_TOLERANCE
            and max(nearest_errors) <= RECOVERY_TRACE_TOLERANCE
        ),
    }


def verify_failed_five_command_gate(
    *,
    report_path: Path,
    home_review: dict[str, Any],
    trace_poses: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    report = load_json(report_path)
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise FiveCommandError("unexpected failed five-command report schema")
    scope = report.get("scope") or {}
    run = report.get("run") or {}
    failure = report.get("failure") or {}
    static = report.get("static_capability_boundary") or {}
    writes = scope.get("goal_position_writes") or {}
    failed_start = finite_vector(
        "failed five-command start_q",
        run.get("start_q", ()),
    )
    pose_status = recovery_pose_status(
        failed_start,
        reviewed_start=home_review["measured_start"],
        home_command=home_review["home_command"],
        trace_poses=trace_poses,
    )
    checks = {
        "source_identity": static.get("sha256") == EXPECTED_PREVIOUS_FIVE_SHA256,
        "expected_decision": (
            report.get("decision") == EXPECTED_PREVIOUS_FIVE_FAILURE_DECISION
        ),
        "expected_prewrite_home_failure": (
            failure.get("type") == "FiveCommandError"
            and str(failure.get("message", "")).startswith(
                "live arm moved away from reviewed Home; no Goal_Position write "
                "was allowed:"
            )
        ),
        "zero_goal_writes": (
            int(writes.get("total_attempted", -1)) == 0
            and int(writes.get("total", -1)) == 0
            and int(scope.get("policy_command_write_attempts", -1)) == 0
            and int(scope.get("policy_command_writes_acknowledged", -1)) == 0
        ),
        "zero_live_inference": int(scope.get("live_policy_inferences", -1)) == 0,
        "cleanup": (
            scope.get("torque_disabled_at_exit") is True
            and scope.get("serial_closed") is True
            and scope.get("torque_may_be_enabled_after_cleanup") is False
            and run.get("cleanup_errors") == []
        ),
        "failed_pose_near_reviewed_trajectory": pose_status["pass"] is True,
    }
    if not all(checks.values()):
        raise FiveCommandError(f"failed five-command gate review rejected: {checks}")
    return {
        "decision": "ZERO_WRITE_FAILURE_REVIEWED_FOR_MERGED_HOME_RECOVERY",
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "checks": checks,
        "failed_start_q": list(failed_start),
        "failed_start_pose_status": pose_status,
    }


# The following wrappers are the only motor-bus capability path in this file.
# Their implementation is pinned by EXPECTED_CONTROLLED_HOME_SHA256.
def connect_bus(home: ModuleType, bus: Any) -> None:
    home.connect_motor_bus(bus)


def read_modes(home: ModuleType, bus: Any) -> dict[str, int]:
    return home.read_operating_mode(bus)


def read_actual(
    home: ModuleType,
    guard: ModuleType,
    limits: Any,
    bus: Any,
) -> tuple[float, ...]:
    values = home.read_present_position(bus)
    return guard.DeltaCommandGuard(
        limits,
        limits.home_command,
    ).validate_measured_positions(values)


def read_goal(home: ModuleType, bus: Any) -> tuple[float, ...]:
    return home.read_goal_position(bus)


def send_goal(
    home: ModuleType,
    bus: Any,
    command: Sequence[float],
    counters: dict[str, int],
    phase: str,
) -> None:
    counters["total_attempted"] += 1
    attempted_key = f"{phase}_attempted"
    counters[attempted_key] = counters.get(attempted_key, 0) + 1
    home.write_goal_position(bus, command)
    counters["total"] += 1
    counters[phase] = counters.get(phase, 0) + 1


def enable_torque(home: ModuleType, bus: Any) -> None:
    home.enable_position_torque(bus)


def disable_torque(home: ModuleType, bus: Any) -> None:
    home.disable_position_torque(bus)


def disconnect_bus(home: ModuleType, bus: Any) -> None:
    home.disconnect_motor_bus(bus)


def make_trace_row(
    *,
    phase: str,
    sequence: int,
    elapsed_seconds: float,
    command: Sequence[float],
    actual: Sequence[float],
    tracking: Any,
    policy_step: Any | None = None,
) -> dict[str, Any]:
    command_vector = finite_vector("trace command", command)
    actual_vector = finite_vector("trace actual", actual)
    row: dict[str, Any] = {
        "phase": phase,
        "sequence": sequence,
        "elapsed_seconds": elapsed_seconds,
        "policy_derived": int(policy_step is not None),
        "tracking_tripped": int(tracking.tripped),
        "tracking_tripped_joints": ";".join(tracking.tripped_joints),
        "any_rate_clip": int(policy_step.any_rate_clip) if policy_step else 0,
        "any_soft_clip": int(policy_step.any_soft_clip) if policy_step else 0,
    }
    for index, motor in enumerate(MOTOR_ORDER):
        row[f"command_{motor}"] = command_vector[index]
        row[f"actual_{motor}"] = actual_vector[index]
        row[f"tracking_error_{motor}"] = float(tracking.absolute_error[index])
        row[f"over_limit_seconds_{motor}"] = float(
            tracking.over_limit_seconds[index]
        )
        row[f"raw_delta_{motor}"] = (
            float(policy_step.raw_delta[index]) if policy_step else 0.0
        )
        row[f"guarded_delta_{motor}"] = (
            float(policy_step.sent_delta[index]) if policy_step else 0.0
        )
    return row


def validate_tracking(
    tracking: Any,
    limits: Any,
    *,
    strict_commissioning: bool,
    phase: str,
) -> None:
    thresholds = (
        COMMISSIONING_TRACKING_LIMITS
        if strict_commissioning
        else tuple(value * 2.0 for value in limits.tracking_error_limits)
    )
    immediate = {
        motor: {
            "error": float(tracking.absolute_error[index]),
            "limit": float(thresholds[index]),
        }
        for index, motor in enumerate(MOTOR_ORDER)
        if tracking.absolute_error[index] > thresholds[index]
    }
    if immediate:
        raise FiveCommandError(
            f"{phase} immediate tracking-error stop: {immediate}"
        )
    if tracking.tripped:
        raise FiveCommandError(
            f"{phase} persistent tracking-error stop: {tracking.tripped_joints}"
        )


def execute_sequence(
    *,
    phase: str,
    commands: Sequence[Sequence[float]],
    policy_steps: Sequence[Any] | None,
    deadline_seconds: float,
    strict_commissioning: bool,
    home: ModuleType,
    guard: ModuleType,
    limits: Any,
    bus: Any,
    counters: dict[str, int],
    trace_rows: list[dict[str, Any]],
    policy_plan_rows: list[dict[str, Any]] | None = None,
) -> tuple[float, ...]:
    if not commands:
        return read_actual(home, guard, limits, bus)
    if policy_steps is not None and len(policy_steps) != len(commands):
        raise FiveCommandError("policy step/command length mismatch")
    if policy_plan_rows is not None and len(policy_plan_rows) != len(commands):
        raise FiveCommandError("policy plan/command length mismatch")
    watchdog = guard.TrackingWatchdog(limits)
    watchdog.reset()
    started = time.monotonic()
    next_tick = started
    actual: tuple[float, ...] | None = None
    for sequence, command_values in enumerate(commands, start=1):
        command = finite_vector(f"{phase} command", command_values)
        if policy_plan_rows is not None:
            policy_plan_rows[sequence - 1]["write_attempted"] = 1
        send_goal(home, bus, command, counters, phase)
        if policy_plan_rows is not None:
            policy_plan_rows[sequence - 1]["write_acknowledged"] = 1
        next_tick += 1.0 / CONTROL_HZ
        delay = next_tick - time.monotonic()
        if delay > 0.0:
            time.sleep(delay)
        else:
            next_tick = time.monotonic()
        actual = read_actual(home, guard, limits, bus)
        now = time.monotonic()
        if now - started > deadline_seconds:
            raise FiveCommandError(
                f"{phase} deadline exceeded: {now - started:.3f}s > "
                f"{deadline_seconds:.3f}s"
            )
        tracking = watchdog.update(actual, command, now)
        validate_tracking(
            tracking,
            limits,
            strict_commissioning=strict_commissioning,
            phase=phase,
        )
        step = policy_steps[sequence - 1] if policy_steps is not None else None
        trace_rows.append(
            make_trace_row(
                phase=phase,
                sequence=sequence,
                elapsed_seconds=now - started,
                command=command,
                actual=actual,
                tracking=tracking,
                policy_step=step,
            )
        )
    if actual is None:
        raise FiveCommandError(f"{phase} produced no readback")
    return actual


def hold_command(
    *,
    phase: str,
    command: Sequence[float],
    seconds: float,
    strict_commissioning: bool,
    home: ModuleType,
    guard: ModuleType,
    limits: Any,
    bus: Any,
    trace_rows: list[dict[str, Any]],
) -> tuple[float, ...]:
    command_vector = finite_vector(f"{phase} command", command)
    watchdog = guard.TrackingWatchdog(limits)
    watchdog.reset()
    started = time.monotonic()
    deadline = started + seconds
    sequence = 0
    actual: tuple[float, ...] | None = None
    while time.monotonic() < deadline:
        time.sleep(1.0 / CONTROL_HZ)
        actual = read_actual(home, guard, limits, bus)
        now = time.monotonic()
        tracking = watchdog.update(actual, command_vector, now)
        validate_tracking(
            tracking,
            limits,
            strict_commissioning=strict_commissioning,
            phase=phase,
        )
        sequence += 1
        trace_rows.append(
            make_trace_row(
                phase=phase,
                sequence=sequence,
                elapsed_seconds=now - started,
                command=command_vector,
                actual=actual,
                tracking=tracking,
            )
        )
    if actual is None:
        raise FiveCommandError(f"{phase} produced no readback")
    return actual


def make_camera_packet(assembler: ModuleType, frame: Any) -> Any:
    return assembler.CameraFramePacket(
        role=frame.role,
        vid=frame.vid,
        pid=frame.pid,
        sequence=frame.sequence,
        capture_monotonic_ns=frame.capture_perf_ns,
        receive_monotonic_s=frame.receive_monotonic_s,
        frame_bgr=frame.frame_bgr,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise FiveCommandError(f"CSV schema changed within {path.name}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    confirmations = {
        "authorize_five_command_policy_motion": (
            args.authorize_five_command_policy_motion
        ),
        "arm_supported": args.confirm_arm_supported,
        "scene_ready": args.confirm_scene_ready,
        "hands_clear": args.confirm_hands_clear,
        "power_cutoff_ready": args.confirm_power_cutoff_ready,
    }
    missing = [name for name, present in confirmations.items() if not present]
    if missing:
        raise FiveCommandError(f"required physical confirmations missing: {missing}")

    script_path = Path(__file__).resolve()
    bridge_path = require_file(args.camera_bridge)
    preprocess_path = require_file(args.preprocess_adapter)
    assembler_path = require_file(args.assembler)
    guard_path = require_file(args.guard_core)
    one_shot_path = require_file(args.one_shot_launcher)
    one_shot_report_path = require_file(args.one_shot_report)
    offline_path = require_file(args.offline_runtime_adapter)
    shadow_path = require_file(args.live_shadow_runner)
    home_path = require_file(args.controlled_home_gate)
    home_report_path = require_file(args.controlled_home_report)
    home_trace_path = require_file(args.controlled_home_trace)
    failed_five_report_path = require_file(args.failed_five_command_report)
    contract_path = require_file(args.contract)
    calibration_path = require_file(args.calibration)
    candidate_root = require_dir(args.candidate_root)
    source_checkpoint_root = require_dir(args.source_checkpoint_root)
    follower_source = require_file(args.follower_source)
    follower_config_source = require_file(args.follower_config_source)
    follower_port = args.follower_port.expanduser()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            "output directory already exists; preserve it and choose a retry path: "
            f"{output_dir}"
        )

    print("===== EXPLICIT FIVE-COMMAND MOTION AUTHORIZATION =====", flush=True)
    print(
        "authorized: one dummy warmup + exactly one live ACT inference + five "
        "policy commands",
        flush=True,
    )
    print("policy execution=15Hz; no second replan; no autonomous loop", flush=True)
    print(
        "reviewed-path Home recovery, policy motion, and post-policy Home "
        "return: bounded",
        flush=True,
    )
    print("external support, reviewed scene, hands clear, 12V cutoff: CONFIRMED", flush=True)
    print("normal and abnormal exit both disable torque", flush=True)
    print("autonomous episode/deployment authorization: false", flush=True)

    print("\n===== VERIFY FROZEN STACK AND CONTROLLED-HOME REVIEW =====", flush=True)
    static_boundary = verify_static_capability_boundary(script_path)
    hashes = {
        "camera_bridge": require_sha(
            bridge_path, EXPECTED_BRIDGE_SHA256, "camera bridge"
        ),
        "preprocess_adapter": require_sha(
            preprocess_path, EXPECTED_PREPROCESS_SHA256, "preprocess adapter"
        ),
        "assembler": require_sha(
            assembler_path, EXPECTED_ASSEMBLER_SHA256, "assembler"
        ),
        "guard_core": require_sha(
            guard_path, EXPECTED_GUARD_SHA256, "guard core"
        ),
        "one_shot_launcher": require_sha(
            one_shot_path, EXPECTED_ONE_SHOT_SHA256, "one-shot launcher"
        ),
        "offline_runtime_adapter": require_sha(
            offline_path,
            EXPECTED_OFFLINE_ADAPTER_SHA256,
            "offline runtime adapter",
        ),
        "live_shadow_runner": require_sha(
            shadow_path, EXPECTED_LIVE_SHADOW_SHA256, "live shadow runner"
        ),
        "controlled_home_gate": require_sha(
            home_path, EXPECTED_CONTROLLED_HOME_SHA256, "controlled Home gate"
        ),
        "follower_source": require_sha(
            follower_source, EXPECTED_FOLLOWER_SHA256, "follower source"
        ),
        "follower_config_source": require_sha(
            follower_config_source,
            EXPECTED_FOLLOWER_CONFIG_SHA256,
            "follower config source",
        ),
    }
    home_review = verify_controlled_home_review(
        gate_path=home_path,
        report_path=home_report_path,
        trace_path=home_trace_path,
    )
    trace_poses = load_reviewed_trace_poses(home_trace_path)
    failed_five_gate = verify_failed_five_command_gate(
        report_path=failed_five_report_path,
        home_review=home_review,
        trace_poses=trace_poses,
    )
    print(
        "controlled Home: original 2-degree precision gate reviewed and "
        "accepted within 5 degrees: PASS",
        flush=True,
    )
    failed_status = failed_five_gate["failed_start_pose_status"]
    print(
        "zero-write five-command failure: reviewed-path recovery candidate "
        f"PASS (envelope={failed_status['max_envelope_excess']:.3f}deg, "
        f"trace={failed_status['nearest_trace_max_error']:.3f}deg)",
        flush=True,
    )

    bridge = load_module(bridge_path, "act_v3_five_command_bridge")
    preprocess = load_module(preprocess_path, "act_v3_five_command_preprocess")
    assembler = load_module(assembler_path, "act_v3_five_command_assembler")
    guard = load_module(guard_path, "act_v3_five_command_guard")
    one_shot = load_module(one_shot_path, "act_v3_five_command_one_shot")
    offline = load_module(offline_path, "act_v3_five_command_offline")
    shadow = load_module(shadow_path, "act_v3_five_command_shadow")
    home = load_module(home_path, "act_v3_five_command_home")

    frozen_inputs = guard.verify_frozen_inputs(contract_path, calibration_path)
    limits = guard.RuntimeLimits.frozen_v3()
    guard_audit = guard.run_algorithm_self_audit(limits)
    if guard_audit["tests_passed"] != guard_audit["tests_total"]:
        raise FiveCommandError("frozen guard self-audit did not fully pass")
    if tuple(home_review["home_command"]) != tuple(limits.home_command):
        raise FiveCommandError("reviewed Home command differs from frozen runtime Home")
    preprocess.verify_static_offline_boundary(preprocess_path)
    one_shot_gate = shadow.verify_one_shot_gate(
        one_shot_path,
        one_shot_report_path,
    )
    print(
        f"source identities and guard={guard_audit['tests_passed']}/"
        f"{guard_audit['tests_total']}: PASS",
        flush=True,
    )

    print("\n===== VERIFY FROZEN 11K RELEASE AND WARM MODEL =====", flush=True)
    release = offline.verify_release(candidate_root, source_checkpoint_root)
    if int(release.get("selected_step", -1)) != EXPECTED_POLICY_STEP:
        raise FiveCommandError("release is not the frozen 11K checkpoint")
    import cv2
    import numpy as np
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy

    if not torch.cuda.is_available():
        raise FiveCommandError("CUDA is unavailable")
    policy = ACTPolicy.from_pretrained(
        candidate_root / "pretrained_model",
        local_files_only=True,
    )
    policy.eval()
    policy_report = offline.verify_policy_contract(policy)
    warmup_ms = shadow.warm_policy(policy, limits, np, torch)
    print(
        "policy state=18D images=2x3x480x480 action=6D chunk=10 queue=5: PASS",
        flush=True,
    )
    print(f"pre-hardware CUDA warmup inference: {warmup_ms:.3f} ms", flush=True)

    print("\n===== VERIFY EXACT FOLLOWER AND CONSTRUCT INERT BUS =====", flush=True)
    device = one_shot.verify_follower_device(follower_port)
    if device.get("stable_path") != EXPECTED_FOLLOWER_BY_ID:
        raise FiveCommandError("unexpected follower by-id identity")
    follower_class, config_class, imported_modules = one_shot.import_frozen_follower(
        follower_source,
        follower_config_source,
    )
    config = config_class(
        port=str(follower_port),
        id="follower_white",
        cameras={},
        disable_torque_on_disconnect=False,
        use_degrees=False,
    )
    follower = follower_class(config)
    follower_summary = one_shot.verify_constructed_follower(
        follower,
        expected_calibration=guard.EXPECTED_FOLLOWER_CALIBRATION,
        expected_port=str(follower_port),
    )
    print(f"follower={device['stable_path']} cameras={{}} bus disconnected: PASS", flush=True)

    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "five_command_commissioning_report.json"
    trace_path = output_dir / "five_command_commissioning_trace.csv"
    plan_path = output_dir / "five_command_policy_plan.csv"
    hub = shadow.LiveFrameHub(bridge=bridge, cv2=cv2, np=np)
    camera_session = shadow.CameraSession(
        bridge=bridge,
        bridge_path=bridge_path,
        windows_python=args.windows_python,
        output_dir=output_dir,
        duration_seconds=CAMERA_STREAM_SECONDS,
        hub=hub,
    )
    policy_guard = guard.DeltaCommandGuard(
        limits,
        limits.home_command,
        (0.0,) * len(MOTOR_ORDER),
    )
    observation_assembler = assembler.LiveObservationAssembler(
        preprocess_module=preprocess,
        delta_guard=policy_guard,
        cv2_module=cv2,
        np_module=np,
        torch_module=torch,
        device=torch.device(policy.config.device),
        max_camera_age_seconds=MAX_CAMERA_AGE_SECONDS,
        max_joint_age_seconds=MAX_JOINT_AGE_SECONDS,
        max_capture_skew_seconds=MAX_CAMERA_SKEW_SECONDS,
        max_receive_skew_seconds=MAX_CAMERA_SKEW_SECONDS,
    )

    bus = follower.bus
    counters: dict[str, int] = {
        "total_attempted": 0,
        "total": 0,
        "current_goal_seed_attempted": 0,
        "current_goal_seed": 0,
        "home_align_attempted": 0,
        "home_align": 0,
        "policy_attempted": 0,
        "policy": 0,
        "return_home_attempted": 0,
        "return_home": 0,
    }
    trace_rows: list[dict[str, Any]] = []
    plan_rows: list[dict[str, Any]] = []
    connect_attempted = False
    connected = False
    operating_mode_read = False
    torque_enable_attempted = False
    torque_may_be_enabled = False
    torque_disabled_at_exit = False
    serial_closed = False
    camera_start_attempted = False
    camera_started = False
    camera_finished = False
    camera_aborted = False
    camera_start_report: dict[str, Any] | None = None
    camera_report: dict[str, Any] | None = None
    camera_failures: list[str] = []
    cleanup_errors: list[str] = []
    failure: dict[str, Any] | None = None
    exit_code = 0
    motion_completed = False
    start_q: tuple[float, ...] | None = None
    live_recovery_status: dict[str, Any] | None = None
    recovery_session_limits: tuple[tuple[float, float], ...] | None = None
    home_aligned_q: tuple[float, ...] | None = None
    pre_inference_q: tuple[float, ...] | None = None
    final_q: tuple[float, ...] | None = None
    policy_steps: list[Any] = []
    inference_ms: float | None = None
    assembly_metrics: Any | None = None

    try:
        print("\n===== START REVIEWED WINDOWS CAMERAS =====", flush=True)
        camera_start_attempted = True
        camera_start_report = camera_session.start()
        camera_started = True
        for role in ("front", "wrist"):
            actual_capture = camera_start_report[role]["actual_capture"]
            print(
                f"{role}: READY {actual_capture['width']}x"
                f"{actual_capture['height']}/"
                f"{actual_capture['fourcc'] or 'UNKNOWN'}",
                flush=True,
            )

        print("\n===== READ-ONLY PRE-WRITE HOME CHECK =====", flush=True)
        connect_attempted = True
        connect_bus(home, bus)
        connected = True
        read_modes(home, bus)
        operating_mode_read = True
        start_q = read_actual(home, guard, limits, bus)
        motor, failed_start_drift = maximum_difference(
            start_q,
            failed_five_gate["failed_start_q"],
        )
        if failed_start_drift > FAILED_START_DRIFT_TOLERANCE:
            raise FiveCommandError(
                "live pose drifted from the preserved zero-write failure; no "
                f"Goal_Position write was allowed: {motor} "
                f"drift={failed_start_drift:.3f} > "
                f"{FAILED_START_DRIFT_TOLERANCE:.3f}"
            )
        live_recovery_status = recovery_pose_status(
            start_q,
            reviewed_start=home_review["measured_start"],
            home_command=home_review["home_command"],
            trace_poses=trace_poses,
        )
        if not live_recovery_status["pass"]:
            raise FiveCommandError(
                "live pose is outside the narrowly expanded reviewed Home path: "
                f"envelope={live_recovery_status['max_envelope_excess']:.3f}, "
                f"trace={live_recovery_status['nearest_trace_max_error']:.3f}"
            )
        print(
            "live pose remains near the prior real Home trajectory: "
            f"failed-start drift={failed_start_drift:.3f}, "
            f"envelope={live_recovery_status['max_envelope_excess']:.3f}, "
            f"trace={live_recovery_status['nearest_trace_max_error']:.3f}: PASS",
            flush=True,
        )

        print("\n===== FIVE-SECOND NO-WRITE SAFETY COUNTDOWN =====", flush=True)
        print(
            "KEEP HANDS CLEAR. Press Ctrl+C or cut follower 12 V on any concern.",
            flush=True,
        )
        countdown_reference = start_q
        for remaining in range(COUNTDOWN_SECONDS, 0, -1):
            print(f"policy commissioning in {remaining}s", flush=True)
            time.sleep(1.0)
            current = read_actual(home, guard, limits, bus)
            drift_motor, drift = maximum_difference(current, countdown_reference)
            if drift > COUNTDOWN_DRIFT_LIMIT:
                raise FiveCommandError(
                    f"arm moved during countdown: {drift_motor} drift={drift:.3f}"
                )

        print(
            "\n===== CURRENT-GOAL LOCK AND REVIEWED-PATH HOME RECOVERY =====",
            flush=True,
        )
        disable_torque(home, bus)
        torque_disabled_at_exit = True
        current = read_actual(home, guard, limits, bus)
        motor, failed_start_drift = maximum_difference(
            current,
            failed_five_gate["failed_start_q"],
        )
        if failed_start_drift > FAILED_START_DRIFT_TOLERANCE:
            raise FiveCommandError(
                "pre-lock pose left the preserved zero-write start: "
                f"{motor} drift={failed_start_drift:.3f}"
            )
        live_recovery_status = recovery_pose_status(
            current,
            reviewed_start=home_review["measured_start"],
            home_command=home_review["home_command"],
            trace_poses=trace_poses,
        )
        if not live_recovery_status["pass"]:
            raise FiveCommandError(
                "pre-lock pose left the reviewed Home recovery path: "
                f"envelope={live_recovery_status['max_envelope_excess']:.3f}, "
                f"trace={live_recovery_status['nearest_trace_max_error']:.3f}"
            )
        send_goal(home, bus, current, counters, "current_goal_seed")
        goal_readback = read_goal(home, bus)
        motor, readback_error = maximum_difference(goal_readback, current)
        if readback_error > GOAL_READBACK_TOLERANCE:
            raise FiveCommandError(
                f"current-goal readback mismatch: {motor}={readback_error:.3f}"
            )
        torque_enable_attempted = True
        torque_may_be_enabled = True
        torque_disabled_at_exit = False
        enable_torque(home, bus)

        settle_deadline = time.monotonic() + CURRENT_GOAL_SETTLE_SECONDS
        while time.monotonic() < settle_deadline:
            actual = read_actual(home, guard, limits, bus)
            motor, drift = maximum_difference(actual, current)
            if drift > TORQUE_LOCK_TOLERANCE:
                raise FiveCommandError(
                    f"current-goal torque lock moved arm: {motor}={drift:.3f}"
                )
            time.sleep(1.0 / CONTROL_HZ)

        recovery_session_limits = home.build_session_limits(
            current,
            limits.soft_limits,
        )
        align_commands = home.build_home_trajectory(current, limits.home_command)
        for command in align_commands:
            home.validate_command_envelope(command, recovery_session_limits)
        print(
            f"reviewed-path Home recovery: {len(align_commands)} commands, "
            f"{len(align_commands) / CONTROL_HZ:.3f}s planned",
            flush=True,
        )
        home_aligned_q = execute_sequence(
            phase="home_align",
            commands=align_commands,
            policy_steps=None,
            deadline_seconds=HOME_ALIGNMENT_DEADLINE_SECONDS,
            strict_commissioning=False,
            home=home,
            guard=guard,
            limits=limits,
            bus=bus,
            counters=counters,
            trace_rows=trace_rows,
        )
        home_aligned_q = hold_command(
            phase="home_align_hold",
            command=limits.home_command,
            seconds=HOME_ALIGNMENT_HOLD_SECONDS,
            strict_commissioning=True,
            home=home,
            guard=guard,
            limits=limits,
            bus=bus,
            trace_rows=trace_rows,
        )
        motor, alignment_error = maximum_difference(
            home_aligned_q,
            limits.home_command,
        )
        if alignment_error > START_HOME_TOLERANCE:
            raise FiveCommandError(
                f"Home alignment failed: {motor} error={alignment_error:.3f}"
            )
        print(
            f"Home command aligned; worst actual error={motor}:{alignment_error:.3f}: PASS",
            flush=True,
        )

        print("\n===== ONE LIVE OBSERVATION AND ONE ACT INFERENCE =====", flush=True)
        front, wrist = hub.wait_pair(
            after_front=-1,
            after_wrist=-1,
            timeout_seconds=PAIR_WAIT_TIMEOUT_SECONDS,
        )
        pre_inference_q = read_actual(home, guard, limits, bus)
        read_finished = time.monotonic()
        motor, pre_inference_error = maximum_difference(
            pre_inference_q,
            limits.home_command,
        )
        if pre_inference_error > START_HOME_TOLERANCE:
            raise FiveCommandError(
                f"pre-inference Home error failed: {motor}={pre_inference_error:.3f}"
            )
        joint_packet = assembler.JointStatePacket(
            actual_q=pre_inference_q,
            read_monotonic_s=read_finished,
        )
        now = time.monotonic()
        observation, assembly_metrics = observation_assembler.assemble(
            front=make_camera_packet(assembler, front),
            wrist=make_camera_packet(assembler, wrist),
            joints=joint_packet,
            now_monotonic_s=now,
        )
        chunk, inference_ms = shadow.inference_chunk(
            policy,
            observation,
            np,
            torch,
        )
        if inference_ms > MAX_SINGLE_INFERENCE_MS:
            raise FiveCommandError(
                f"single live inference exceeded {MAX_SINGLE_INFERENCE_MS:.0f}ms: "
                f"{inference_ms:.3f}ms"
            )
        if tuple(int(value) for value in chunk.shape) != (EXPECTED_CHUNK_SIZE, 6):
            raise FiveCommandError(f"unexpected ACT chunk shape: {chunk.shape}")

        for substep in range(POLICY_COMMAND_COUNT):
            step = policy_guard.apply_delta(chunk[substep])
            policy_steps.append(step)
            if step.any_rate_clip or step.any_soft_clip:
                raise FiveCommandError(
                    "first five policy deltas require guard clipping; aborting before "
                    f"policy motion at substep {substep}"
                )
            excursions = tuple(
                abs(command - target)
                for command, target in zip(
                    step.sent_command,
                    limits.home_command,
                    strict=True,
                )
            )
            exceeded = {
                motor: {
                    "excursion": excursions[index],
                    "limit": POLICY_EXCURSION_LIMITS[index],
                }
                for index, motor in enumerate(MOTOR_ORDER)
                if excursions[index] > POLICY_EXCURSION_LIMITS[index]
            }
            if exceeded:
                raise FiveCommandError(
                    "first-five policy plan left commissioning excursion envelope: "
                    f"{exceeded}"
                )
            plan_row: dict[str, Any] = {
                "substep": substep,
                "write_attempted": 0,
                "write_acknowledged": 0,
                "any_rate_clip": 0,
                "any_soft_clip": 0,
            }
            for index, motor in enumerate(MOTOR_ORDER):
                plan_row[f"raw_delta_{motor}"] = float(step.raw_delta[index])
                plan_row[f"command_{motor}"] = float(step.sent_command[index])
                plan_row[f"excursion_from_home_{motor}"] = excursions[index]
            plan_rows.append(plan_row)
        print(
            f"fresh pair + 18D state + inference={inference_ms:.3f}ms: PASS",
            flush=True,
        )
        print("five-command plan has no clips and stays inside envelope: PASS", flush=True)

        print("\n===== EXECUTE EXACTLY FIVE GUARDED POLICY COMMANDS =====", flush=True)
        policy_commands = [step.sent_command for step in policy_steps]
        last_actual = execute_sequence(
            phase="policy",
            commands=policy_commands,
            policy_steps=policy_steps,
            deadline_seconds=POLICY_SEQUENCE_DEADLINE_SECONDS,
            strict_commissioning=True,
            home=home,
            guard=guard,
            limits=limits,
            bus=bus,
            counters=counters,
            trace_rows=trace_rows,
            policy_plan_rows=plan_rows,
        )
        if counters["policy"] != POLICY_COMMAND_COUNT:
            raise FiveCommandError(
                f"policy write count is {counters['policy']}, expected 5"
            )
        last_command = policy_steps[-1].sent_command
        last_actual = hold_command(
            phase="policy_hold",
            command=last_command,
            seconds=POLICY_HOLD_SECONDS,
            strict_commissioning=True,
            home=home,
            guard=guard,
            limits=limits,
            bus=bus,
            trace_rows=trace_rows,
        )
        _, policy_hold_error = maximum_difference(last_actual, last_command)
        print(
            f"policy commands sent=5/5; hold worst tracking={policy_hold_error:.3f}: PASS",
            flush=True,
        )

        print("\n===== BOUNDED RETURN TO HOME =====", flush=True)
        return_commands = home.build_home_trajectory(
            last_command,
            limits.home_command,
        )
        final_q = execute_sequence(
            phase="return_home",
            commands=return_commands,
            policy_steps=None,
            deadline_seconds=RETURN_HOME_DEADLINE_SECONDS,
            strict_commissioning=False,
            home=home,
            guard=guard,
            limits=limits,
            bus=bus,
            counters=counters,
            trace_rows=trace_rows,
        )
        final_q = hold_command(
            phase="final_home_hold",
            command=limits.home_command,
            seconds=FINAL_HOME_HOLD_SECONDS,
            strict_commissioning=True,
            home=home,
            guard=guard,
            limits=limits,
            bus=bus,
            trace_rows=trace_rows,
        )
        motor, final_error = maximum_difference(final_q, limits.home_command)
        if final_error > FINAL_HOME_TOLERANCE:
            raise FiveCommandError(
                f"final Home return failed: {motor} error={final_error:.3f}"
            )
        motion_completed = True
        print(
            f"returned Home; worst actual error={motor}:{final_error:.3f}: PASS",
            flush=True,
        )
    except KeyboardInterrupt as exc:
        failure = {
            "type": type(exc).__name__,
            "message": "operator cancelled with Ctrl+C",
            "traceback": traceback.format_exc(),
        }
        exit_code = 130
    except Exception as exc:
        failure = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        exit_code = 1
    finally:
        if connect_attempted:
            try:
                disable_torque(home, bus)
                torque_may_be_enabled = False
                torque_disabled_at_exit = True
                print("Follower torque disabled.", flush=True)
            except Exception as exc:
                cleanup_errors.append(
                    f"torque disable failed: {type(exc).__name__}: {exc}"
                )
                print(
                    "CRITICAL: torque disable failed; cut follower 12 V now: "
                    f"{exc}",
                    file=sys.stderr,
                    flush=True,
                )
            try:
                disconnect_bus(home, bus)
                serial_closed = True
                print("Follower serial closed with disconnect(false).", flush=True)
            except Exception as exc:
                cleanup_errors.append(
                    f"serial close failed: {type(exc).__name__}: {exc}"
                )
                print(
                    f"CRITICAL: follower serial close failed: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

        if camera_start_attempted:
            try:
                if camera_started and motion_completed and failure is None:
                    camera_session.finish(CAMERA_FINISH_GRACE_SECONDS)
                    camera_finished = True
                else:
                    camera_session.abort()
                    camera_aborted = True
            except Exception as exc:
                cleanup_errors.append(
                    f"camera cleanup failed: {type(exc).__name__}: {exc}"
                )

        if cleanup_errors and exit_code == 0:
            exit_code = 1
            failure = {
                "type": "CleanupError",
                "message": "; ".join(cleanup_errors),
                "traceback": None,
            }

    if camera_start_attempted:
        try:
            camera_report, camera_failures = camera_session.report()
        except Exception as exc:
            camera_failures = [f"camera_report_failed: {type(exc).__name__}: {exc}"]
        if camera_finished and camera_failures and failure is None:
            failure = {
                "type": "CameraAcceptanceError",
                "message": "; ".join(camera_failures),
                "traceback": None,
            }
            exit_code = 1

    # Preserve camera diagnostics and the final decoded frames even on failure.
    if camera_start_attempted:
        for role in ("front", "wrist"):
            state = hub.states[role]
            write_csv(output_dir / f"{role}_commissioning_frames.csv", state.get("rows", []))
            last_frame = state.get("last_frame")
            if last_frame is not None:
                cv2.imwrite(str(output_dir / f"{role}_commissioning_last.png"), last_frame)
    write_csv(trace_path, trace_rows)
    write_csv(plan_path, plan_rows)

    success = (
        exit_code == 0
        and failure is None
        and motion_completed
        and counters["policy"] == POLICY_COMMAND_COUNT
        and torque_disabled_at_exit
        and serial_closed
        and camera_finished
        and not camera_failures
        and final_q is not None
    )
    decision = (
        "FIVE_COMMAND_POLICY_COMMISSIONING_PASS_PREPARE_ONE_EPISODE_NEXT"
        if success
        else "FIVE_COMMAND_POLICY_COMMISSIONING_FAILED_DEPLOYMENT_REMAINS_BLOCKED"
    )
    report = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "authorization": {
            "kind": "explicit_cli_exactly_five_policy_commands",
            "confirmations": confirmations,
            "authorized_dummy_warmup_inferences": 1,
            "authorized_live_policy_inferences": 1,
            "authorized_policy_commands": POLICY_COMMAND_COUNT,
            "authorized_pre_policy_goal": (
                "monotonic reviewed-path recovery to frozen Home"
            ),
            "autonomous_episode_authorized": False,
            "hardware_deployment_authorized": False,
        },
        "scope": {
            "model_loaded": True,
            "live_camera_start_attempted": camera_start_attempted,
            "live_cameras_started": camera_started,
            "camera_session_finished": camera_finished,
            "camera_session_aborted": camera_aborted,
            "follower_serial_connect_attempted": connect_attempted,
            "follower_serial_opened": connected,
            "operating_mode_read": operating_mode_read,
            "torque_enable_attempted": torque_enable_attempted,
            "torque_may_be_enabled_after_cleanup": torque_may_be_enabled,
            "goal_position_writes": counters,
            "dummy_warmup_inferences": 1,
            "live_policy_inferences": 1 if inference_ms is not None else 0,
            "policy_commands_sent": counters["policy"],
            "policy_command_writes_acknowledged": counters["policy"],
            "policy_command_write_attempts": counters["policy_attempted"],
            "torque_disabled_at_exit": torque_disabled_at_exit,
            "serial_closed": serial_closed,
            "full_autonomous_loop_run": False,
            "autonomous_episode_authorized": False,
            "hardware_deployment_authorized": False,
        },
        "static_capability_boundary": static_boundary,
        "frozen_dependencies": {
            "hashes": hashes,
            "contract_and_calibration": frozen_inputs,
            "guard_self_audit": guard_audit,
            "one_shot_gate": one_shot_gate,
            "controlled_home_review": home_review,
            "failed_five_command_gate": failed_five_gate,
            "release": release,
            "policy": policy_report,
            "policy_warmup_ms": warmup_ms,
            "follower_device": device,
            "constructed_follower": follower_summary,
            "imported_follower_modules": imported_modules,
        },
        "commissioning_contract": {
            "motor_order": list(MOTOR_ORDER),
            "control_hz": CONTROL_HZ,
            "start_home_tolerance": START_HOME_TOLERANCE,
            "final_home_tolerance": FINAL_HOME_TOLERANCE,
            "recovery_envelope_tolerance": RECOVERY_ENVELOPE_TOLERANCE,
            "recovery_trace_tolerance": RECOVERY_TRACE_TOLERANCE,
            "failed_start_drift_tolerance": FAILED_START_DRIFT_TOLERANCE,
            "merged_home_recovery_keeps_torque_enabled_for_policy": True,
            "policy_command_count": POLICY_COMMAND_COUNT,
            "policy_hold_seconds": POLICY_HOLD_SECONDS,
            "commissioning_tracking_limits": dict(
                zip(MOTOR_ORDER, COMMISSIONING_TRACKING_LIMITS, strict=True)
            ),
            "policy_excursion_limits_from_home": dict(
                zip(MOTOR_ORDER, POLICY_EXCURSION_LIMITS, strict=True)
            ),
            "reject_any_rate_or_soft_clip_before_policy_motion": True,
            "return_home_before_torque_disable": True,
            "normal_and_abnormal_exit_disable_torque": True,
        },
        "camera_start": camera_start_report,
        "camera_transport": camera_report,
        "camera_acceptance_failures": camera_failures,
        "run": {
            "start_q": None if start_q is None else list(start_q),
            "live_recovery_pose_status": live_recovery_status,
            "recovery_session_limits": (
                None
                if recovery_session_limits is None
                else {
                    motor: list(bounds)
                    for motor, bounds in zip(
                        MOTOR_ORDER,
                        recovery_session_limits,
                        strict=True,
                    )
                }
            ),
            "home_aligned_q": (
                None if home_aligned_q is None else list(home_aligned_q)
            ),
            "pre_inference_q": (
                None if pre_inference_q is None else list(pre_inference_q)
            ),
            "inference_ms": inference_ms,
            "assembly_metrics": (
                None
                if assembly_metrics is None
                else {
                    field: getattr(assembly_metrics, field)
                    for field in (
                        "front_sequence",
                        "wrist_sequence",
                        "front_age_ms",
                        "wrist_age_ms",
                        "joint_age_ms",
                        "capture_skew_ms",
                        "receive_skew_ms",
                        "state_dimension",
                    )
                }
            ),
            "policy_steps_planned": len(policy_steps),
            "policy_steps_sent": counters["policy"],
            "trace_rows": len(trace_rows),
            "final_q_before_torque_disable": (
                None if final_q is None else list(final_q)
            ),
            "cleanup_errors": cleanup_errors,
        },
        "failure": failure,
        "decision": decision,
        "next_gate_authorized": False,
        "autonomous_episode_authorized": False,
        "hardware_deployment_authorized": False,
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\n===== DECISION =====", flush=True)
    print(f"decision='{decision}'", flush=True)
    if failure is not None:
        print(
            f"failure={failure['type']}: {failure['message']}",
            file=sys.stderr,
            flush=True,
        )
    print(
        f"policy write attempts: {counters['policy_attempted']}/5; "
        f"acknowledged: {counters['policy']}/5",
        flush=True,
    )
    print(f"torque disabled at exit: {torque_disabled_at_exit}", flush=True)
    print(f"serial closed: {serial_closed}", flush=True)
    print("AUTONOMOUS EPISODE AND DEPLOYMENT REMAIN BLOCKED.", flush=True)
    print("\n===== OUTPUT =====", flush=True)
    print(report_path, flush=True)
    if trace_rows:
        print(trace_path, flush=True)
    if plan_rows:
        print(plan_path, flush=True)
    print(
        "ACT V3 FIVE-COMMAND POLICY COMMISSIONING: PASS"
        if success
        else "ACT V3 FIVE-COMMAND POLICY COMMISSIONING: FAIL",
        flush=True,
    )
    return 0 if success else (exit_code or 1)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(
            "ACT V3 FIVE-COMMAND POLICY COMMISSIONING: FAIL-CLOSED: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        print("NO MOTION IS AUTHORIZED BY THIS FAILED PROCESS.", file=sys.stderr)
        raise SystemExit(1)
