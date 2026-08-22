#!/usr/bin/env python3
"""Run one bounded, guarded ACT V3 policy episode on the SO-101 follower.

This is Stage 3C of the frozen ACT V3 commissioning plan.  It consumes the
offline acceptance report for the completed five-command gate, recovers the
gravity-sagged follower to Home, and then runs exactly 180 ACT replans with
five guarded commands per replan.  The policy portion is finite: at most 900
policy-derived Goal_Position writes.

The preferred recovery entry remains the complete reviewed six-joint Home
trace.  A separate, explicitly authorized power-loss branch accepts a pose
that is still inside a fixed 10-degree expansion of the reviewed componentwise
start-to-Home envelope.  That branch does not replay policy from the sagged
pose.  It first validates a finite, per-joint rate-bounded trajectory whose
every step moves monotonically toward frozen Home, locks the current position,
executes the trajectory under the frozen tracking watchdog, and requires a
five-degree Home alignment before cameras or policy execution begin.

The run fails closed before any further policy write if it observes a rate
clip, a new soft-boundary crossing, an unapproved soft clip, stale or skewed
camera/joint data, an invalid policy output, excessive inference latency, a
tracking stop, or an operator cancellation.  The only accepted soft-boundary
saturation is an inherited outward hold at wrist_flex or gripper; this models
the reviewed task behavior of lowering the wrist and closing the gripper at
an already active boundary.  It never authorizes crossing into a new boundary.

On a normal policy completion the process records task-end images, returns to
frozen Home, then reverses the reviewed commissioning path to the preserved
folded Park pose before disabling torque.  It holds and verifies Park for two
seconds, closes the follower serial bus, and completes the native-Windows
camera END/ACK lifecycle.  On an abnormal path it sends no recovery commands:
it immediately disables torque and closes every transport.  Stable external
support is therefore mandatory for the entire run.

A process PASS means the controlled execution and safety evidence passed.  It
does not assert that the cube was picked and placed.  The preserved task-end
images must be reviewed before Stage 3C is accepted or any ten-trial run begins.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gc
import hashlib
import importlib.util
import json
import math
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Sequence


MOTOR_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

EXPECTED_FIVE_RUNNER_SHA256 = (
    "8c9d589feaf8f7409002d60d4156f11934651eeeb832c2e8be16a45d40ce687a"
)
EXPECTED_FIVE_REVIEWER_SHA256 = (
    "c882c6f406a2d4f99c5970d1ce8c60da9c53721197f78def9dafae47d04caed2"
)
EXPECTED_FIVE_REVIEW_DECISION = (
    "FIVE_COMMAND_POLICY_COMMISSIONING_REVIEW_ACCEPTED_"
    "TEARDOWN_ONLY_PREPARE_ONE_EPISODE_NEXT"
)
EXPECTED_FIVE_SOURCE_DECISION = (
    "FIVE_COMMAND_POLICY_COMMISSIONING_FAILED_DEPLOYMENT_REMAINS_BLOCKED"
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
EXPECTED_POLICY_STEP = 11000
EXPECTED_CHUNK_SIZE = 10

REPLAN_COUNT = 180
COMMANDS_PER_REPLAN = 5
POLICY_COMMAND_LIMIT = REPLAN_COUNT * COMMANDS_PER_REPLAN
CONTROL_HZ = 15.0
COUNTDOWN_SECONDS = 5
CURRENT_GOAL_SETTLE_SECONDS = 1.0
HOME_HOLD_SECONDS = 2.0
FINAL_HOME_HOLD_SECONDS = 2.0
FINAL_PARK_HOLD_SECONDS = 2.0
CAMERA_STREAM_SECONDS = 90.0
CAMERA_FINISH_GRACE_SECONDS = 30.0
POLICY_HARD_DEADLINE_SECONDS = 85.0
HOME_RECOVERY_DEADLINE_SECONDS = 15.0
RETURN_HOME_DEADLINE_SECONDS = 15.0
RETURN_PARK_DEADLINE_SECONDS = 15.0
PAIR_WAIT_TIMEOUT_SECONDS = 0.30
MAX_CAMERA_AGE_SECONDS = 0.250
MAX_JOINT_AGE_SECONDS = 0.100
MAX_CAMERA_SKEW_SECONDS = 0.100
MAX_SINGLE_INFERENCE_MS = 500.0
MAX_INFERENCE_P95_MS = 250.0
MAX_QUEUE_SECONDS = 1.0

START_HOME_TOLERANCE = 5.0
FINAL_HOME_TOLERANCE = 5.0
FINAL_PARK_TOLERANCE = 5.0
COUNTDOWN_DRIFT_LIMIT = 3.0
GOAL_READBACK_TOLERANCE = 1.5
TORQUE_LOCK_TOLERANCE = 3.0
RECOVERY_ENVELOPE_TOLERANCE = 1.0
RECOVERY_TRACE_TOLERANCE = 5.0
POWER_LOSS_RECOVERY_ENVELOPE_MARGIN = 10.0
# Retained only as a diagnostic comparison with the prior gravity-sagged
# snapshot.  A torque-off pose is not repeatable across time or power cycles;
# motion authorization is instead gated by proximity to the complete reviewed
# six-joint Home trajectory below.
SOURCE_START_DRIFT_TOLERANCE = 3.0
POLICY_TRACKING_LIMITS = (8.0, 8.0, 8.0, 8.0, 8.0, 10.0)

# Only a no-op projection at an already active task boundary is accepted.
# wrist_flex is needed for the downward grasp pose; gripper saturation is the
# physical close/open endpoint.  A new crossing at either joint is forbidden.
ALLOWED_INHERITED_SOFT_HOLD_INDICES = {3, 5}


class ControlledEpisodeError(RuntimeError):
    """A frozen dependency, live input, safety, tracking, or lifecycle gate failed."""


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
    parser.add_argument("--failed-five-command-report", type=Path, required=True)
    parser.add_argument("--five-command-runner", type=Path, required=True)
    parser.add_argument("--five-command-reviewer", type=Path, required=True)
    parser.add_argument("--five-command-review-report", type=Path, required=True)
    parser.add_argument("--five-command-source-report", type=Path, required=True)
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
        "--authorize-one-controlled-policy-episode",
        action="store_true",
        help="Authorize one finite 180-replan/900-command policy episode.",
    )
    parser.add_argument(
        "--authorize-bounded-power-loss-home-recovery",
        action="store_true",
        help=(
            "Authorize only the guarded current-pose-to-Home recovery that "
            "precedes this one finite policy episode."
        ),
    )
    parser.add_argument(
        "--confirm-stage3b-reviewed",
        action="store_true",
        help="Confirm the preserved Stage 3B offline review was inspected.",
    )
    parser.add_argument(
        "--confirm-arm-supported",
        action="store_true",
        help=(
            "Confirm stable external support prevents a torque-off fall without "
            "placing hands in any sweep or pinch zone."
        ),
    )
    parser.add_argument(
        "--confirm-scene-ready",
        action="store_true",
        help=(
            "Confirm one red cube is at the reviewed training start location and "
            "all other objects are outside the complete robot sweep."
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
        raise ControlledEpisodeError(
            f"{label} SHA256 mismatch: expected={expected}, actual={actual}"
        )
    return actual


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ControlledEpisodeError(f"invalid JSON {path}: {exc}") from exc


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ControlledEpisodeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def finite_vector(label: str, values: Sequence[Any]) -> tuple[float, ...]:
    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ControlledEpisodeError(f"{label} must be numeric") from exc
    if len(vector) != len(MOTOR_ORDER):
        raise ControlledEpisodeError(f"{label} must contain six values")
    if any(not math.isfinite(value) for value in vector):
        raise ControlledEpisodeError(f"{label} contains a nonfinite value")
    return vector


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
    by_attr: dict[str, list[ast.Call]] = {}
    for call in calls:
        if isinstance(call.func, ast.Attribute):
            by_attr.setdefault(call.func.attr, []).append(call)

    expected_once = {
        "connect_bus",
        "read_modes",
        "read_actual",
        "read_goal",
        "send_goal",
        "enable_torque",
        "disable_torque",
        "disconnect_bus",
        "inference_chunk",
        "from_pretrained",
    }
    mismatches = {
        name: len(by_attr.get(name, []))
        for name in sorted(expected_once)
        if len(by_attr.get(name, [])) != 1
    }
    forbidden = {
        "sync_write",
        "sync_read",
        "write_goal_position",
        "enable_position_torque",
        "disable_position_torque",
        "send_action",
        "write",
        "write_calibration",
        "configure",
        "configure_motors",
        "calibrate",
        "setup_motors",
        "VideoCapture",
    }
    forbidden_found = sorted(forbidden.intersection(by_attr))
    follower_calls = sorted(
        ast.unparse(call)
        for call in calls
        if ast.unparse(call).startswith(("follower.connect(", "follower.disconnect("))
    )
    if mismatches or forbidden_found or follower_calls:
        raise ControlledEpisodeError(
            "controlled-episode source capability boundary mismatch: "
            f"counts={mismatches}, forbidden={forbidden_found}, "
            f"follower_calls={follower_calls}"
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "raw_motor_register_calls": False,
        "full_follower_connect_or_disconnect_calls": False,
        "indirect_pinned_motor_capability_only": True,
        "policy_loader_call_sites": 1,
        "policy_inference_wrapper_call_sites": 1,
        "pass": True,
    }


def verify_stage3b_review(
    *,
    runner_path: Path,
    reviewer_path: Path,
    review_path: Path,
    source_report_path: Path,
) -> dict[str, Any]:
    runner_sha = require_sha(
        runner_path, EXPECTED_FIVE_RUNNER_SHA256, "five-command runner"
    )
    reviewer_sha = require_sha(
        reviewer_path, EXPECTED_FIVE_REVIEWER_SHA256, "five-command reviewer"
    )
    review = load_json(review_path)
    source = load_json(source_report_path)
    if not isinstance(review, dict) or review.get("schema_version") != 1:
        raise ControlledEpisodeError("unexpected five-command review schema")
    if not isinstance(source, dict) or source.get("schema_version") != 1:
        raise ControlledEpisodeError("unexpected five-command source report schema")
    motion = review.get("motion_acceptance") or {}
    review_scope = review.get("review_scope") or {}
    preserved = review.get("preserved_run") or {}
    source_scope = source.get("scope") or {}
    source_run = source.get("run") or {}
    checks = {
        "review_decision": review.get("decision") == EXPECTED_FIVE_REVIEW_DECISION,
        "stage": review.get("stage") == "3b_five_command_policy_commissioning",
        "stage_complete": review.get("stage_complete") is True,
        "next_gate": review.get("next_gate") == "ONE_CONTROLLED_POLICY_EPISODE",
        "review_offline": (
            review_scope.get("offline_only") is True
            and review_scope.get("hardware_accessed") is False
            and review_scope.get("source_run_reexecuted") is False
        ),
        "reviewer_identity": (
            (review.get("static_boundary") or {}).get("sha256") == reviewer_sha
        ),
        "runner_identity": (
            (review.get("commissioning_source") or {}).get("sha256") == runner_sha
        ),
        "source_report_identity": (
            preserved.get("report_sha256") == sha256_file(source_report_path)
        ),
        "five_writes": (
            int(motion.get("policy_write_attempts", -1)) == 5
            and int(motion.get("policy_writes_acknowledged", -1)) == 5
            and int(source_scope.get("policy_command_write_attempts", -1)) == 5
            and int(source_scope.get("policy_command_writes_acknowledged", -1)) == 5
        ),
        "one_inference": (
            int(motion.get("live_policy_inferences", -1)) == 1
            and int(source_scope.get("live_policy_inferences", -1)) == 1
        ),
        "safe_motion": (
            int(motion.get("tracking_trip_rows", -1)) == 0
            and int(motion.get("rate_or_soft_clip_policy_rows", -1)) == 0
            and motion.get("returned_home_within_5_degrees") is True
        ),
        "safe_cleanup": (
            motion.get("torque_disabled_at_exit") is True
            and motion.get("serial_closed") is True
            and source_scope.get("torque_disabled_at_exit") is True
            and source_scope.get("serial_closed") is True
        ),
        "source_decision": source.get("decision") == EXPECTED_FIVE_SOURCE_DECISION,
        "teardown_only_failure": (
            (source.get("failure") or {}).get("type") == "CameraAcceptanceError"
        ),
    }
    if not all(checks.values()):
        raise ControlledEpisodeError(f"Stage 3B review rejected: {checks}")
    source_start_q = finite_vector("five-command source start_q", source_run.get("start_q", ()))
    return {
        "decision": EXPECTED_FIVE_REVIEW_DECISION,
        "runner_path": str(runner_path),
        "runner_sha256": runner_sha,
        "reviewer_path": str(reviewer_path),
        "reviewer_sha256": reviewer_sha,
        "review_path": str(review_path),
        "review_sha256": sha256_file(review_path),
        "source_report_path": str(source_report_path),
        "source_report_sha256": sha256_file(source_report_path),
        "source_start_q": list(source_start_q),
        "checks": checks,
    }


def validate_policy_step(step: Any, *, replan: int, substep: int) -> tuple[int, ...]:
    if step.any_rate_clip:
        clipped = [
            MOTOR_ORDER[index]
            for index, flag in enumerate(step.rate_clipped)
            if flag
        ]
        raise ControlledEpisodeError(
            f"replan {replan} substep {substep} requires rate clipping: {clipped}"
        )
    crossing = [
        MOTOR_ORDER[index]
        for index, flag in enumerate(step.new_boundary_crossing)
        if flag
    ]
    if crossing:
        raise ControlledEpisodeError(
            f"replan {replan} substep {substep} creates new boundary crossing: "
            f"{crossing}"
        )
    accepted: list[int] = []
    for index, flag in enumerate(step.soft_clipped):
        if not flag:
            continue
        if (
            index not in ALLOWED_INHERITED_SOFT_HOLD_INDICES
            or not step.inherited_boundary_projection[index]
        ):
            raise ControlledEpisodeError(
                f"replan {replan} substep {substep} has unapproved soft clip at "
                f"{MOTOR_ORDER[index]}"
            )
        accepted.append(index)
    return tuple(accepted)


def run_local_safety_self_audit() -> dict[str, Any]:
    tests: dict[str, bool] = {}

    def fake(
        *,
        rate: int | None = None,
        soft: int | None = None,
        inherited: bool = False,
        crossing: int | None = None,
    ) -> Any:
        rate_flags = [False] * 6
        soft_flags = [False] * 6
        inherited_flags = [False] * 6
        crossing_flags = [False] * 6
        if rate is not None:
            rate_flags[rate] = True
        if soft is not None:
            soft_flags[soft] = True
            inherited_flags[soft] = inherited
        if crossing is not None:
            crossing_flags[crossing] = True
        return SimpleNamespace(
            any_rate_clip=any(rate_flags),
            rate_clipped=tuple(rate_flags),
            soft_clipped=tuple(soft_flags),
            inherited_boundary_projection=tuple(inherited_flags),
            new_boundary_crossing=tuple(crossing_flags),
        )

    def must_fail(label: str, item: Any) -> None:
        try:
            validate_policy_step(item, replan=0, substep=0)
        except ControlledEpisodeError:
            tests[label] = True
        else:
            tests[label] = False

    tests["clean_step"] = validate_policy_step(fake(), replan=0, substep=0) == ()
    tests["inherited_wrist_hold"] = validate_policy_step(
        fake(soft=3, inherited=True), replan=0, substep=0
    ) == (3,)
    tests["inherited_gripper_hold"] = validate_policy_step(
        fake(soft=5, inherited=True), replan=0, substep=0
    ) == (5,)
    must_fail("rate_clip_fails", fake(rate=1))
    must_fail("new_crossing_fails", fake(crossing=3))
    must_fail("noninherited_wrist_fails", fake(soft=3, inherited=False))
    must_fail("other_joint_soft_clip_fails", fake(soft=2, inherited=True))
    tests["finite_write_budget"] = (
        REPLAN_COUNT == 180
        and COMMANDS_PER_REPLAN == 5
        and POLICY_COMMAND_LIMIT == 900
    )

    recovery_start = (2.0, -95.0, 99.0, 69.0, 0.0, 6.0)
    recovery_home = (13.0, -32.0, 48.0, 43.0, 0.4, 28.0)
    recovery_middle = tuple(
        (source + target) / 2.0
        for source, target in zip(recovery_start, recovery_home, strict=True)
    )
    recovery_trace = [
        {"phase": "move", "sequence": 1, "actual_q": recovery_start},
        {"phase": "move", "sequence": 2, "actual_q": recovery_middle},
        {"phase": "hold", "sequence": 1, "actual_q": recovery_home},
    ]
    strict_status = bounded_recovery_pose_status(
        recovery_start,
        reviewed_start=recovery_start,
        home_command=recovery_home,
        trace_poses=recovery_trace,
    )
    tests["reviewed_trace_recovery_entry_passes"] = (
        strict_status["pass"] is True
        and strict_status["recovery_mode"] == "REVIEWED_TRACE"
    )

    power_loss_pose = list(recovery_start)
    power_loss_pose[0] -= 6.0
    power_loss_pose[3] = recovery_middle[3]
    bounded_status = bounded_recovery_pose_status(
        power_loss_pose,
        reviewed_start=recovery_start,
        home_command=recovery_home,
        trace_poses=recovery_trace,
    )
    tests["bounded_power_loss_entry_passes"] = (
        bounded_status["pass"] is True
        and bounded_status["strict_reviewed_path_pass"] is False
        and bounded_status["recovery_mode"] == "BOUNDED_POWER_LOSS"
    )

    rejected_pose = list(recovery_start)
    rejected_pose[0] -= POWER_LOSS_RECOVERY_ENVELOPE_MARGIN + 0.01
    rejected_status = bounded_recovery_pose_status(
        rejected_pose,
        reviewed_start=recovery_start,
        home_command=recovery_home,
        trace_poses=recovery_trace,
    )
    tests["power_loss_margin_is_fail_closed"] = rejected_status["pass"] is False

    session_limits = tuple(
        (min(source, target) - 0.1, max(source, target) + 0.1)
        for source, target in zip(recovery_start, recovery_home, strict=True)
    )
    trajectory_report = validate_monotonic_home_trajectory(
        start=recovery_start,
        commands=(recovery_middle, recovery_home),
        home_command=recovery_home,
        session_limits=session_limits,
        step_limits=(100.0,) * len(MOTOR_ORDER),
    )
    tests["monotonic_home_trajectory_passes"] = trajectory_report["pass"] is True

    park_session_limits = tuple(
        (min(source, target) - 0.1, max(source, target) + 0.1)
        for source, target in zip(recovery_home, recovery_start, strict=True)
    )
    park_report = validate_monotonic_home_trajectory(
        start=recovery_home,
        commands=(recovery_middle, recovery_start),
        home_command=recovery_start,
        session_limits=park_session_limits,
        step_limits=(100.0,) * len(MOTOR_ORDER),
    )
    tests["reviewed_reverse_park_trajectory_passes"] = park_report["pass"] is True

    outward = list(recovery_start)
    outward[0] -= 0.1
    try:
        validate_monotonic_home_trajectory(
            start=recovery_start,
            commands=(outward, recovery_home),
            home_command=recovery_home,
            session_limits=session_limits,
            step_limits=(100.0,) * len(MOTOR_ORDER),
        )
    except ControlledEpisodeError:
        tests["outward_home_recovery_step_fails"] = True
    else:
        tests["outward_home_recovery_step_fails"] = False

    if not all(tests.values()):
        raise ControlledEpisodeError(f"local safety self-audit failed: {tests}")
    return {
        "tests": tests,
        "tests_passed": sum(tests.values()),
        "tests_total": len(tests),
        "pass": True,
    }


# These wrappers are the only motor capability route in this source.  Their
# implementation is pinned by the accepted five-command runner and controlled
# Home module hashes.
def bus_connect(five: ModuleType, home: ModuleType, bus: Any) -> None:
    five.connect_bus(home, bus)


def bus_read_modes(five: ModuleType, home: ModuleType, bus: Any) -> dict[str, int]:
    return five.read_modes(home, bus)


def bus_read_actual(
    five: ModuleType,
    home: ModuleType,
    guard: ModuleType,
    limits: Any,
    bus: Any,
) -> tuple[float, ...]:
    return five.read_actual(home, guard, limits, bus)


def bus_read_goal(five: ModuleType, home: ModuleType, bus: Any) -> tuple[float, ...]:
    return five.read_goal(home, bus)


def bus_send_goal(
    five: ModuleType,
    home: ModuleType,
    bus: Any,
    command: Sequence[float],
    counters: dict[str, int],
    phase: str,
) -> None:
    five.send_goal(home, bus, command, counters, phase)


def bus_enable_torque(five: ModuleType, home: ModuleType, bus: Any) -> None:
    five.enable_torque(home, bus)


def bus_disable_torque(five: ModuleType, home: ModuleType, bus: Any) -> None:
    five.disable_torque(home, bus)


def bus_disconnect(five: ModuleType, home: ModuleType, bus: Any) -> None:
    five.disconnect_bus(home, bus)


def infer_policy(
    shadow: ModuleType,
    policy: Any,
    observation: dict[str, Any],
    np: Any,
    torch: Any,
) -> tuple[Any, float]:
    return shadow.inference_chunk(policy, observation, np, torch)


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


def maximum_difference(
    left: Sequence[float], right: Sequence[float]
) -> tuple[str, float]:
    a = finite_vector("left", left)
    b = finite_vector("right", right)
    differences = [abs(x - y) for x, y in zip(a, b, strict=True)]
    index = max(range(6), key=differences.__getitem__)
    return MOTOR_ORDER[index], differences[index]


def bounded_recovery_pose_status(
    pose: Sequence[float],
    *,
    reviewed_start: Sequence[float],
    home_command: Sequence[float],
    trace_poses: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Classify a live recovery entry without authorizing a policy action.

    The complete reviewed trace remains the preferred entry.  The bounded
    power-loss branch is deliberately componentwise: it only permits a fixed
    expansion around the already reviewed start-to-Home ranges.  A later
    independent gate validates every generated recovery command as monotonic,
    rate bounded, finite, and inside the start-to-soft-limit session envelope.
    """

    vector = finite_vector("recovery pose", pose)
    start = finite_vector("reviewed recovery start", reviewed_start)
    home = finite_vector("recovery Home", home_command)
    if not trace_poses:
        raise ControlledEpisodeError("reviewed recovery trace contains no poses")

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
                finite_vector("reviewed trace pose", item["actual_q"]),
                strict=True,
            )
        ),
    )
    nearest_pose = finite_vector("nearest reviewed trace pose", nearest["actual_q"])
    nearest_errors = tuple(
        abs(value - reference)
        for value, reference in zip(vector, nearest_pose, strict=True)
    )
    max_envelope = max(envelope_excess)
    max_trace = max(nearest_errors)
    strict_reviewed_path_pass = (
        max_envelope <= RECOVERY_ENVELOPE_TOLERANCE
        and max_trace <= RECOVERY_TRACE_TOLERANCE
    )
    bounded_power_loss_entry_pass = (
        max_envelope <= POWER_LOSS_RECOVERY_ENVELOPE_MARGIN
    )
    accepted = strict_reviewed_path_pass or bounded_power_loss_entry_pass
    return {
        "pose": list(vector),
        "envelope_excess_by_joint": dict(
            zip(MOTOR_ORDER, envelope_excess, strict=True)
        ),
        "max_envelope_excess": max_envelope,
        "nearest_trace_phase": nearest.get("phase"),
        "nearest_trace_sequence": nearest.get("sequence"),
        "nearest_trace_error_by_joint": dict(
            zip(MOTOR_ORDER, nearest_errors, strict=True)
        ),
        "nearest_trace_max_error": max_trace,
        "strict_reviewed_path_pass": strict_reviewed_path_pass,
        "bounded_power_loss_entry_pass": bounded_power_loss_entry_pass,
        "recovery_mode": (
            "REVIEWED_TRACE"
            if strict_reviewed_path_pass
            else "BOUNDED_POWER_LOSS"
            if bounded_power_loss_entry_pass
            else "REJECTED"
        ),
        "power_loss_envelope_margin": POWER_LOSS_RECOVERY_ENVELOPE_MARGIN,
        "pass": accepted,
    }


def validate_monotonic_home_trajectory(
    *,
    start: Sequence[float],
    commands: Sequence[Sequence[float]],
    home_command: Sequence[float],
    session_limits: Sequence[Sequence[float]],
    step_limits: Sequence[float],
) -> dict[str, Any]:
    """Validate the entire recovery plan before Goal Position or torque enable."""

    previous = finite_vector("recovery trajectory start", start)
    target = finite_vector("recovery trajectory Home", home_command)
    if len(session_limits) != len(MOTOR_ORDER):
        raise ControlledEpisodeError("recovery session limits must contain six rows")
    if len(step_limits) != len(MOTOR_ORDER):
        raise ControlledEpisodeError("recovery step limits must contain six values")
    if not commands:
        raise ControlledEpisodeError("recovery trajectory contains no commands")
    duration = len(commands) / CONTROL_HZ
    if duration > HOME_RECOVERY_DEADLINE_SECONDS:
        raise ControlledEpisodeError(
            "recovery trajectory exceeds deadline: "
            f"{duration:.3f}s > {HOME_RECOVERY_DEADLINE_SECONDS:.3f}s"
        )

    maximum_step = [0.0] * len(MOTOR_ORDER)
    for sequence, raw_command in enumerate(commands, start=1):
        command = finite_vector(f"recovery command {sequence}", raw_command)
        for index, (prior, value, goal, bounds, raw_step_limit) in enumerate(
            zip(
                previous,
                command,
                target,
                session_limits,
                step_limits,
                strict=True,
            )
        ):
            minimum, maximum = (float(bounds[0]), float(bounds[1]))
            step_limit = float(raw_step_limit)
            if not all(math.isfinite(item) for item in (minimum, maximum, step_limit)):
                raise ControlledEpisodeError("recovery limits contain nonfinite values")
            if minimum > maximum or step_limit <= 0.0:
                raise ControlledEpisodeError("recovery limits are malformed")
            if value < minimum - 1e-9 or value > maximum + 1e-9:
                raise ControlledEpisodeError(
                    f"recovery command {sequence} leaves session envelope at "
                    f"{MOTOR_ORDER[index]}"
                )
            step = abs(value - prior)
            maximum_step[index] = max(maximum_step[index], step)
            if step > step_limit + 1e-9:
                raise ControlledEpisodeError(
                    f"recovery command {sequence} exceeds step limit at "
                    f"{MOTOR_ORDER[index]}"
                )
            if abs(goal - value) > abs(goal - prior) + 1e-9:
                raise ControlledEpisodeError(
                    f"recovery command {sequence} moves outward from Home at "
                    f"{MOTOR_ORDER[index]}"
                )
        previous = command

    final_motor, final_error = maximum_difference(previous, target)
    if final_error > 1e-9:
        raise ControlledEpisodeError(
            f"recovery trajectory does not terminate at Home: "
            f"{final_motor}={final_error:.9f}"
        )
    return {
        "commands": len(commands),
        "planned_duration_seconds": duration,
        "maximum_step_by_joint": dict(
            zip(MOTOR_ORDER, maximum_step, strict=True)
        ),
        "all_steps_monotonic_toward_home": True,
        "all_steps_within_session_envelope": True,
        "all_steps_within_rate_limits": True,
        "terminates_at_home": True,
        "pass": True,
    }


def measured_soft_excess(actual: Sequence[float], limits: Any) -> dict[str, float]:
    vector = finite_vector("measured soft-limit check", actual)
    result: dict[str, float] = {}
    for motor, value, (minimum, maximum) in zip(
        MOTOR_ORDER, vector, limits.soft_limits, strict=True
    ):
        excess = max(minimum - value, value - maximum, 0.0)
        if excess > 0.0:
            result[motor] = excess
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ControlledEpisodeError(f"CSV schema changed within {path.name}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def flatten_command_row(
    *,
    replan: int,
    substep: int,
    global_command: int,
    episode_elapsed_seconds: float,
    step: Any,
    actual: Sequence[float],
    tracking: Any,
    write_elapsed_ms: float,
    read_elapsed_ms: float,
    inherited_indices: Sequence[int],
) -> dict[str, Any]:
    actual_q = finite_vector("command actual", actual)
    row: dict[str, Any] = {
        "replan": replan,
        "substep": substep,
        "global_policy_command": global_command,
        "episode_elapsed_seconds": episode_elapsed_seconds,
        "write_acknowledged": 1,
        "write_elapsed_ms": write_elapsed_ms,
        "read_elapsed_ms": read_elapsed_ms,
        "tracking_tripped": int(tracking.tripped),
        "tracking_tripped_joints": ";".join(tracking.tripped_joints),
        "any_rate_clip": int(step.any_rate_clip),
        "any_soft_clip": int(step.any_soft_clip),
        "accepted_inherited_soft_hold": int(bool(inherited_indices)),
        "accepted_inherited_soft_hold_joints": ";".join(
            MOTOR_ORDER[index] for index in inherited_indices
        ),
    }
    for index, motor in enumerate(MOTOR_ORDER):
        row[f"raw_delta_{motor}"] = float(step.raw_delta[index])
        row[f"rate_limited_delta_{motor}"] = float(step.rate_limited_delta[index])
        row[f"previous_command_{motor}"] = float(step.previous_command[index])
        row[f"candidate_{motor}"] = float(step.candidate_after_rate_limit[index])
        row[f"sent_command_{motor}"] = float(step.sent_command[index])
        row[f"sent_delta_{motor}"] = float(step.sent_delta[index])
        row[f"rate_clipped_{motor}"] = int(step.rate_clipped[index])
        row[f"soft_clipped_{motor}"] = int(step.soft_clipped[index])
        row[f"inherited_boundary_{motor}"] = int(
            step.inherited_boundary_projection[index]
        )
        row[f"new_boundary_crossing_{motor}"] = int(
            step.new_boundary_crossing[index]
        )
        row[f"actual_{motor}"] = actual_q[index]
        row[f"tracking_error_{motor}"] = float(tracking.absolute_error[index])
        row[f"over_limit_seconds_{motor}"] = float(
            tracking.over_limit_seconds[index]
        )
    return row


def flatten_replan_row(
    *,
    index: int,
    episode_elapsed_seconds: float,
    front: Any,
    wrist: Any,
    assembly: Any,
    joint_read_ms: float,
    inference_ms: float,
    pre_actual: Sequence[float],
    pre_command: Sequence[float],
    post_actual: Sequence[float],
    post_command: Sequence[float],
    pre_tracking: Any,
    inherited_holds: int,
) -> dict[str, Any]:
    pre_q = finite_vector("replan pre actual", pre_actual)
    pre_u = finite_vector("replan pre command", pre_command)
    post_q = finite_vector("replan post actual", post_actual)
    post_u = finite_vector("replan post command", post_command)
    row: dict[str, Any] = {
        "replan": index,
        "episode_elapsed_seconds": episode_elapsed_seconds,
        "front_sequence": front.sequence,
        "wrist_sequence": wrist.sequence,
        "front_age_ms": assembly.front_age_ms,
        "wrist_age_ms": assembly.wrist_age_ms,
        "joint_age_ms": assembly.joint_age_ms,
        "capture_skew_ms": assembly.capture_skew_ms,
        "receive_skew_ms": assembly.receive_skew_ms,
        "joint_read_ms": joint_read_ms,
        "inference_ms": inference_ms,
        "state_dimension": assembly.state_dimension,
        "commands_sent": COMMANDS_PER_REPLAN,
        "inherited_soft_holds": inherited_holds,
        "pre_tracking_tripped": int(pre_tracking.tripped),
    }
    for motor_index, motor in enumerate(MOTOR_ORDER):
        row[f"pre_actual_{motor}"] = pre_q[motor_index]
        row[f"pre_command_{motor}"] = pre_u[motor_index]
        row[f"pre_tracking_error_{motor}"] = float(
            pre_tracking.absolute_error[motor_index]
        )
        row[f"post_actual_{motor}"] = post_q[motor_index]
        row[f"post_command_{motor}"] = post_u[motor_index]
    return row


def execute_policy_queue(
    *,
    replan: int,
    steps: Sequence[Any],
    inherited_by_step: Sequence[Sequence[int]],
    episode_started: float,
    five: ModuleType,
    home: ModuleType,
    guard: ModuleType,
    limits: Any,
    bus: Any,
    counters: dict[str, int],
    watchdog: Any,
    trace_rows: list[dict[str, Any]],
    command_rows: list[dict[str, Any]],
) -> tuple[float, ...]:
    if len(steps) != COMMANDS_PER_REPLAN:
        raise ControlledEpisodeError("policy queue length is not five")
    queue_started = time.monotonic()
    next_tick = queue_started
    actual: tuple[float, ...] | None = None
    for substep, step in enumerate(steps):
        write_started = time.monotonic()
        bus_send_goal(
            five,
            home,
            bus,
            step.sent_command,
            counters,
            "policy",
        )
        write_elapsed_ms = (time.monotonic() - write_started) * 1000.0
        next_tick += 1.0 / CONTROL_HZ
        delay = next_tick - time.monotonic()
        if delay > 0.0:
            time.sleep(delay)
        read_started = time.monotonic()
        actual = bus_read_actual(five, home, guard, limits, bus)
        read_finished = time.monotonic()
        read_elapsed_ms = (read_finished - read_started) * 1000.0
        if read_finished - queue_started > MAX_QUEUE_SECONDS:
            raise ControlledEpisodeError(
                f"replan {replan} policy queue exceeded {MAX_QUEUE_SECONDS:.1f}s"
            )
        tracking = watchdog.update(actual, step.sent_command, read_finished)
        five.validate_tracking(
            tracking,
            limits,
            strict_commissioning=True,
            phase="controlled_episode_policy",
        )
        global_command = counters["policy"]
        command_rows.append(
            flatten_command_row(
                replan=replan,
                substep=substep,
                global_command=global_command,
                episode_elapsed_seconds=read_finished - episode_started,
                step=step,
                actual=actual,
                tracking=tracking,
                write_elapsed_ms=write_elapsed_ms,
                read_elapsed_ms=read_elapsed_ms,
                inherited_indices=inherited_by_step[substep],
            )
        )
        trace_rows.append(
            five.make_trace_row(
                phase="policy",
                sequence=global_command,
                elapsed_seconds=read_finished - episode_started,
                command=step.sent_command,
                actual=actual,
                tracking=tracking,
                policy_step=step,
            )
        )
    if actual is None:
        raise ControlledEpisodeError("policy queue produced no readback")
    return actual


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    confirmations = {
        "authorize_one_controlled_policy_episode": (
            args.authorize_one_controlled_policy_episode
        ),
        "authorize_bounded_power_loss_home_recovery": (
            args.authorize_bounded_power_loss_home_recovery
        ),
        "stage3b_reviewed": args.confirm_stage3b_reviewed,
        "arm_supported": args.confirm_arm_supported,
        "scene_ready": args.confirm_scene_ready,
        "hands_clear": args.confirm_hands_clear,
        "power_cutoff_ready": args.confirm_power_cutoff_ready,
    }
    missing = [name for name, present in confirmations.items() if not present]
    if missing:
        raise ControlledEpisodeError(f"required physical confirmations missing: {missing}")

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
    five_path = require_file(args.five_command_runner)
    reviewer_path = require_file(args.five_command_reviewer)
    five_review_path = require_file(args.five_command_review_report)
    five_source_report_path = require_file(args.five_command_source_report)
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
            "output directory already exists; preserve it and choose a new path: "
            f"{output_dir}"
        )

    print("===== EXPLICIT ONE-EPISODE MOTION AUTHORIZATION =====", flush=True)
    print(
        "authorized: bounded power-loss pose -> Home recovery + exactly "
        "180 replans x 5 guarded policy commands + normal Home -> folded Park "
        "return",
        flush=True,
    )
    print(
        "recovery entry: reviewed trace preferred; fixed <=10deg componentwise "
        "power-loss envelope otherwise",
        flush=True,
    )
    print(
        "before first Goal Position write or torque enable: complete Home "
        "trajectory must be finite, rate-bounded, envelope-bounded and "
        "monotonic toward Home",
        flush=True,
    )
    print("maximum policy writes: 900; no automatic second episode", flush=True)
    print(
        "rate clip/new crossing/unapproved soft clip/tracking or sensor failure: "
        "STOP BEFORE FURTHER POLICY WRITES",
        flush=True,
    )
    print(
        "only inherited wrist_flex/gripper boundary no-op holds are accepted",
        flush=True,
    )
    print(
        "normal exit reaches and verifies folded Park before torque disable; "
        "abnormal exit disables torque immediately",
        flush=True,
    )
    print("task success and autonomous deployment authorization: false", flush=True)

    print("\n===== VERIFY STAGE 3B AND FROZEN STACK =====", flush=True)
    static_boundary = verify_static_capability_boundary(script_path)
    local_audit = run_local_safety_self_audit()
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
        "guard_core": require_sha(guard_path, EXPECTED_GUARD_SHA256, "guard core"),
        "one_shot_launcher": require_sha(
            one_shot_path, EXPECTED_ONE_SHOT_SHA256, "one-shot launcher"
        ),
        "offline_runtime_adapter": require_sha(
            offline_path, EXPECTED_OFFLINE_ADAPTER_SHA256, "offline runtime adapter"
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
    stage3b = verify_stage3b_review(
        runner_path=five_path,
        reviewer_path=reviewer_path,
        review_path=five_review_path,
        source_report_path=five_source_report_path,
    )
    print("Stage 3B review, exact source run, and 5/5 motion evidence: PASS", flush=True)
    print(
        f"local full-episode fail-closed tests: {local_audit['tests_passed']}/"
        f"{local_audit['tests_total']} PASS",
        flush=True,
    )

    bridge = load_module(bridge_path, "act_v3_episode_bridge")
    preprocess = load_module(preprocess_path, "act_v3_episode_preprocess")
    assembler = load_module(assembler_path, "act_v3_episode_assembler")
    guard = load_module(guard_path, "act_v3_episode_guard")
    one_shot = load_module(one_shot_path, "act_v3_episode_one_shot")
    offline = load_module(offline_path, "act_v3_episode_offline")
    shadow = load_module(shadow_path, "act_v3_episode_shadow")
    home = load_module(home_path, "act_v3_episode_home")
    five = load_module(five_path, "act_v3_episode_five")

    frozen_inputs = guard.verify_frozen_inputs(contract_path, calibration_path)
    limits = guard.RuntimeLimits.frozen_v3()
    guard_audit = guard.run_algorithm_self_audit(limits)
    if guard_audit["tests_passed"] != guard_audit["tests_total"]:
        raise ControlledEpisodeError("frozen guard self-audit did not fully pass")
    preprocess.verify_static_offline_boundary(preprocess_path)
    one_shot_gate = shadow.verify_one_shot_gate(one_shot_path, one_shot_report_path)
    home_review = five.verify_controlled_home_review(
        gate_path=home_path,
        report_path=home_report_path,
        trace_path=home_trace_path,
    )
    park_command = finite_vector(
        "reviewed folded Park pose", home_review["measured_start"]
    )
    trace_poses = five.load_reviewed_trace_poses(home_trace_path)
    failed_five_gate = five.verify_failed_five_command_gate(
        report_path=failed_five_report_path,
        home_review=home_review,
        trace_poses=trace_poses,
    )
    if tuple(home_review["home_command"]) != tuple(limits.home_command):
        raise ControlledEpisodeError("reviewed Home differs from frozen runtime Home")
    if tuple(five.COMMISSIONING_TRACKING_LIMITS) != POLICY_TRACKING_LIMITS:
        raise ControlledEpisodeError(
            "pinned five-command tracking limits differ from episode limits"
        )
    print(
        f"guard={guard_audit['tests_passed']}/{guard_audit['tests_total']}; "
        "Home and live-read gates: PASS",
        flush=True,
    )

    print("\n===== VERIFY FROZEN 11K RELEASE AND WARM MODEL =====", flush=True)
    release = offline.verify_release(candidate_root, source_checkpoint_root)
    if int(release.get("selected_step", -1)) != EXPECTED_POLICY_STEP:
        raise ControlledEpisodeError("release is not the frozen 11K checkpoint")
    import cv2
    import numpy as np
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy

    if not torch.cuda.is_available():
        raise ControlledEpisodeError("CUDA is unavailable")
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
        raise ControlledEpisodeError("unexpected follower by-id identity")
    follower_class, config_class, imported_modules = one_shot.import_frozen_follower(
        follower_source, follower_config_source
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
    report_path = output_dir / "one_controlled_policy_episode_report.json"
    replan_path = output_dir / "one_controlled_policy_episode_replans.csv"
    command_path = output_dir / "one_controlled_policy_episode_commands.csv"
    trace_path = output_dir / "one_controlled_policy_episode_motion_trace.csv"
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
        limits, limits.home_command, (0.0,) * len(MOTOR_ORDER)
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
        "return_park_attempted": 0,
        "return_park": 0,
    }
    replan_rows: list[dict[str, Any]] = []
    command_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    inference_times_ms: list[float] = []
    joint_read_times_ms: list[float] = []
    assembly_metrics_samples: dict[str, list[float]] = {
        "front_age_ms": [],
        "wrist_age_ms": [],
        "joint_age_ms": [],
        "capture_skew_ms": [],
        "receive_skew_ms": [],
    }
    inherited_soft_holds = Counter({motor: 0 for motor in MOTOR_ORDER})
    measured_soft_excess_max = Counter({motor: 0.0 for motor in MOTOR_ORDER})
    max_guard_invariant_error = 0.0
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
    acceptance_failures: list[str] = []
    failure: dict[str, Any] | None = None
    exit_code = 0
    policy_completed = False
    motion_completed = False
    park_completed = False
    task_outcome_images_saved = False
    start_q: tuple[float, ...] | None = None
    source_start_drift_motor: str | None = None
    source_start_drift: float | None = None
    home_aligned_q: tuple[float, ...] | None = None
    task_end_q: tuple[float, ...] | None = None
    final_home_q: tuple[float, ...] | None = None
    final_park_q: tuple[float, ...] | None = None
    final_q: tuple[float, ...] | None = None
    live_recovery_status: dict[str, Any] | None = None
    recovery_session_limits: Any | None = None
    recovery_trajectory_audit: dict[str, Any] | None = None
    park_trajectory_audit: dict[str, Any] | None = None
    park_session_limits: Any | None = None
    episode_started: float | None = None
    episode_finished: float | None = None
    last_front_sequence = -1
    last_wrist_sequence = -1

    try:
        print("\n===== READ-ONLY PRE-WRITE RECOVERY CHECK =====", flush=True)
        connect_attempted = True
        bus_connect(five, home, bus)
        connected = True
        bus_read_modes(five, home, bus)
        operating_mode_read = True
        start_q = bus_read_actual(five, home, guard, limits, bus)
        source_start_drift_motor, source_start_drift = maximum_difference(
            start_q, stage3b["source_start_q"]
        )
        live_recovery_status = bounded_recovery_pose_status(
            start_q,
            reviewed_start=home_review["measured_start"],
            home_command=home_review["home_command"],
            trace_poses=trace_poses,
        )
        if not live_recovery_status["pass"]:
            raise ControlledEpisodeError(
                "live pose is outside both recovery gates: "
                f"envelope={live_recovery_status['max_envelope_excess']:.3f}, "
                f"allowed={POWER_LOSS_RECOVERY_ENVELOPE_MARGIN:.3f}, "
                f"trace={live_recovery_status['nearest_trace_max_error']:.3f}"
            )
        print(
            f"accepted recovery start mode={live_recovery_status['recovery_mode']}: "
            f"envelope={live_recovery_status['max_envelope_excess']:.3f}, "
            f"trace={live_recovery_status['nearest_trace_max_error']:.3f}: PASS",
            flush=True,
        )
        print(
            "prior torque-off point comparison (diagnostic only): "
            f"{source_start_drift_motor} drift={source_start_drift:.3f}; "
            f"previous reference={SOURCE_START_DRIFT_TOLERANCE:.3f}",
            flush=True,
        )

        print("\n===== FIVE-SECOND NO-WRITE SAFETY COUNTDOWN =====", flush=True)
        print(
            "KEEP HANDS CLEAR. Ctrl+C or follower 12 V cutoff stops this gate.",
            flush=True,
        )
        countdown_reference = start_q
        for remaining in range(COUNTDOWN_SECONDS, 0, -1):
            print(f"one controlled episode in {remaining}s", flush=True)
            time.sleep(1.0)
            current = bus_read_actual(five, home, guard, limits, bus)
            drift_motor, drift = maximum_difference(current, countdown_reference)
            if drift > COUNTDOWN_DRIFT_LIMIT:
                raise ControlledEpisodeError(
                    f"arm moved during countdown: {drift_motor}={drift:.3f}"
                )

        print(
            "\n===== PRE-WRITE FULL TRAJECTORY AUDIT AND HOME RECOVERY =====",
            flush=True,
        )
        current = bus_read_actual(five, home, guard, limits, bus)
        live_recovery_status = bounded_recovery_pose_status(
            current,
            reviewed_start=home_review["measured_start"],
            home_command=home_review["home_command"],
            trace_poses=trace_poses,
        )
        if not live_recovery_status["pass"]:
            raise ControlledEpisodeError(
                "pre-lock pose left the bounded recovery envelope"
            )
        recovery_session_limits = home.build_session_limits(
            current, limits.soft_limits
        )
        home_commands = home.build_home_trajectory(current, limits.home_command)
        recovery_trajectory_audit = validate_monotonic_home_trajectory(
            start=current,
            commands=home_commands,
            home_command=limits.home_command,
            session_limits=recovery_session_limits,
            step_limits=home.HOME_STEP_LIMITS,
        )
        print(
            f"pre-write {live_recovery_status['recovery_mode']} trajectory: "
            f"{len(home_commands)} commands, "
            f"{len(home_commands) / CONTROL_HZ:.3f}s; complete audit PASS",
            flush=True,
        )

        bus_disable_torque(five, home, bus)
        torque_disabled_at_exit = True
        locked_start = bus_read_actual(five, home, guard, limits, bus)
        drift_motor, prelock_drift = maximum_difference(locked_start, current)
        if prelock_drift > COUNTDOWN_DRIFT_LIMIT:
            raise ControlledEpisodeError(
                "arm moved between trajectory audit and current-goal lock: "
                f"{drift_motor}={prelock_drift:.3f}"
            )
        live_recovery_status = bounded_recovery_pose_status(
            locked_start,
            reviewed_start=home_review["measured_start"],
            home_command=home_review["home_command"],
            trace_poses=trace_poses,
        )
        if not live_recovery_status["pass"]:
            raise ControlledEpisodeError(
                "current-goal lock pose left the bounded recovery envelope"
            )
        current = locked_start
        recovery_session_limits = home.build_session_limits(
            current, limits.soft_limits
        )
        home_commands = home.build_home_trajectory(current, limits.home_command)
        recovery_trajectory_audit = validate_monotonic_home_trajectory(
            start=current,
            commands=home_commands,
            home_command=limits.home_command,
            session_limits=recovery_session_limits,
            step_limits=home.HOME_STEP_LIMITS,
        )
        bus_send_goal(
            five, home, bus, current, counters, "current_goal_seed"
        )
        goal_readback = bus_read_goal(five, home, bus)
        motor, readback_error = maximum_difference(goal_readback, current)
        if readback_error > GOAL_READBACK_TOLERANCE:
            raise ControlledEpisodeError(
                f"current-goal readback mismatch: {motor}={readback_error:.3f}"
            )
        torque_enable_attempted = True
        torque_may_be_enabled = True
        torque_disabled_at_exit = False
        bus_enable_torque(five, home, bus)

        settle_deadline = time.monotonic() + CURRENT_GOAL_SETTLE_SECONDS
        while time.monotonic() < settle_deadline:
            actual = bus_read_actual(five, home, guard, limits, bus)
            motor, drift = maximum_difference(actual, current)
            if drift > TORQUE_LOCK_TOLERANCE:
                raise ControlledEpisodeError(
                    f"current-goal torque lock moved arm: {motor}={drift:.3f}"
                )
            time.sleep(1.0 / CONTROL_HZ)

        print(
            f"{live_recovery_status['recovery_mode']} Home recovery: "
            f"{len(home_commands)} commands, "
            f"{len(home_commands) / CONTROL_HZ:.3f}s; trajectory audit PASS",
            flush=True,
        )
        home_aligned_q = five.execute_sequence(
            phase="home_align",
            commands=home_commands,
            policy_steps=None,
            deadline_seconds=HOME_RECOVERY_DEADLINE_SECONDS,
            strict_commissioning=False,
            home=home,
            guard=guard,
            limits=limits,
            bus=bus,
            counters=counters,
            trace_rows=trace_rows,
        )
        home_aligned_q = five.hold_command(
            phase="home_align_hold",
            command=limits.home_command,
            seconds=HOME_HOLD_SECONDS,
            strict_commissioning=True,
            home=home,
            guard=guard,
            limits=limits,
            bus=bus,
            trace_rows=trace_rows,
        )
        motor, alignment_error = maximum_difference(
            home_aligned_q, limits.home_command
        )
        if alignment_error > START_HOME_TOLERANCE:
            raise ControlledEpisodeError(
                f"Home alignment failed: {motor}={alignment_error:.3f}"
            )
        print(
            f"Home aligned; worst actual error={motor}:{alignment_error:.3f}: PASS",
            flush=True,
        )

        print("\n===== START REVIEWED WINDOWS CAMERAS =====", flush=True)
        camera_start_attempted = True
        camera_start_report = camera_session.start()
        camera_started = True
        for role in ("front", "wrist"):
            capture = camera_start_report[role]["actual_capture"]
            print(
                f"{role}: READY {capture['width']}x{capture['height']}/"
                f"{capture['fourcc'] or 'UNKNOWN'}",
                flush=True,
            )

        print("\n===== ONE CONTROLLED POLICY EPISODE =====", flush=True)
        watchdog = guard.TrackingWatchdog(limits)
        watchdog.reset()
        episode_started = time.monotonic()
        for replan in range(REPLAN_COUNT):
            if time.monotonic() - episode_started > POLICY_HARD_DEADLINE_SECONDS:
                raise ControlledEpisodeError(
                    f"policy hard deadline exceeded before replan {replan}"
                )
            front, wrist = hub.wait_pair(
                after_front=last_front_sequence,
                after_wrist=last_wrist_sequence,
                timeout_seconds=PAIR_WAIT_TIMEOUT_SECONDS,
            )
            read_started = time.monotonic()
            pre_actual = bus_read_actual(five, home, guard, limits, bus)
            read_finished = time.monotonic()
            joint_read_ms = (read_finished - read_started) * 1000.0
            pre_command = policy_guard.previous_command
            pre_tracking = watchdog.update(pre_actual, pre_command, read_finished)
            five.validate_tracking(
                pre_tracking,
                limits,
                strict_commissioning=True,
                phase="controlled_episode_pre_replan",
            )
            for motor_name, excess in measured_soft_excess(pre_actual, limits).items():
                measured_soft_excess_max[motor_name] = max(
                    measured_soft_excess_max[motor_name], excess
                )
            joint_packet = assembler.JointStatePacket(
                actual_q=pre_actual,
                read_monotonic_s=read_finished,
            )
            now = time.monotonic()
            observation, assembly = observation_assembler.assemble(
                front=make_camera_packet(assembler, front),
                wrist=make_camera_packet(assembler, wrist),
                joints=joint_packet,
                now_monotonic_s=now,
            )
            chunk, inference_ms = infer_policy(
                shadow, policy, observation, np, torch
            )
            if inference_ms > MAX_SINGLE_INFERENCE_MS:
                raise ControlledEpisodeError(
                    f"replan {replan} inference {inference_ms:.3f}ms exceeds "
                    f"{MAX_SINGLE_INFERENCE_MS:.0f}ms"
                )
            if tuple(int(value) for value in chunk.shape) != (EXPECTED_CHUNK_SIZE, 6):
                raise ControlledEpisodeError(f"unexpected ACT chunk shape: {chunk.shape}")

            steps: list[Any] = []
            inherited_by_step: list[tuple[int, ...]] = []
            replan_inherited = 0
            for substep in range(COMMANDS_PER_REPLAN):
                step = policy_guard.apply_delta(chunk[substep])
                inherited = validate_policy_step(
                    step, replan=replan, substep=substep
                )
                for index in inherited:
                    inherited_soft_holds[MOTOR_ORDER[index]] += 1
                replan_inherited += len(inherited)
                invariant_error = max(
                    abs(
                        step.sent_command[index]
                        - step.previous_command[index]
                        - step.sent_delta[index]
                    )
                    for index in range(6)
                )
                max_guard_invariant_error = max(
                    max_guard_invariant_error, invariant_error
                )
                steps.append(step)
                inherited_by_step.append(inherited)

            post_actual = execute_policy_queue(
                replan=replan,
                steps=steps,
                inherited_by_step=inherited_by_step,
                episode_started=episode_started,
                five=five,
                home=home,
                guard=guard,
                limits=limits,
                bus=bus,
                counters=counters,
                watchdog=watchdog,
                trace_rows=trace_rows,
                command_rows=command_rows,
            )
            for motor_name, excess in measured_soft_excess(post_actual, limits).items():
                measured_soft_excess_max[motor_name] = max(
                    measured_soft_excess_max[motor_name], excess
                )
            replan_rows.append(
                flatten_replan_row(
                    index=replan,
                    episode_elapsed_seconds=time.monotonic() - episode_started,
                    front=front,
                    wrist=wrist,
                    assembly=assembly,
                    joint_read_ms=joint_read_ms,
                    inference_ms=inference_ms,
                    pre_actual=pre_actual,
                    pre_command=pre_command,
                    post_actual=post_actual,
                    post_command=steps[-1].sent_command,
                    pre_tracking=pre_tracking,
                    inherited_holds=replan_inherited,
                )
            )
            inference_times_ms.append(inference_ms)
            joint_read_times_ms.append(joint_read_ms)
            for key in assembly_metrics_samples:
                assembly_metrics_samples[key].append(float(getattr(assembly, key)))
            last_front_sequence = front.sequence
            last_wrist_sequence = wrist.sequence
            if (replan + 1) % 30 == 0:
                print(
                    f"replans={replan + 1}/{REPLAN_COUNT} "
                    f"commands={counters['policy']}/{POLICY_COMMAND_LIMIT} "
                    f"elapsed={time.monotonic() - episode_started:.1f}s "
                    f"inference={inference_ms:.1f}ms",
                    flush=True,
                )

        episode_finished = time.monotonic()
        if counters["policy"] != POLICY_COMMAND_LIMIT:
            raise ControlledEpisodeError(
                f"policy writes={counters['policy']}, expected={POLICY_COMMAND_LIMIT}"
            )
        if len(replan_rows) != REPLAN_COUNT:
            raise ControlledEpisodeError("full replan count was not completed")
        policy_completed = True
        task_end_q = bus_read_actual(five, home, guard, limits, bus)
        for role in ("front", "wrist"):
            frame = hub.states[role].get("last_frame")
            if frame is None or not cv2.imwrite(
                str(output_dir / f"{role}_task_outcome_before_home.png"), frame
            ):
                raise ControlledEpisodeError(
                    f"failed to save {role} task-outcome image"
                )
        task_outcome_images_saved = True
        inference_p95 = percentile(inference_times_ms, 0.95)
        if inference_p95 is None or inference_p95 > MAX_INFERENCE_P95_MS:
            acceptance_failures.append("inference_p95_above_250ms")
        print(
            f"policy episode complete: replans={len(replan_rows)} "
            f"commands={counters['policy']} "
            f"elapsed={episode_finished - episode_started:.3f}s",
            flush=True,
        )
        print(
            "task-outcome images captured before deterministic Home return: PASS",
            flush=True,
        )

        print("\n===== NORMAL-PATH RETURN TO HOME =====", flush=True)
        return_commands = home.build_home_trajectory(
            policy_guard.previous_command, limits.home_command
        )
        final_home_q = five.execute_sequence(
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
        final_home_q = five.hold_command(
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
        motor, final_error = maximum_difference(
            final_home_q, limits.home_command
        )
        if final_error > FINAL_HOME_TOLERANCE:
            raise ControlledEpisodeError(
                f"final Home return failed: {motor}={final_error:.3f}"
            )
        print(
            f"returned Home; worst actual error={motor}:{final_error:.3f}: PASS",
            flush=True,
        )

        print("\n===== REVIEWED HOME -> FOLDED PARK RETURN =====", flush=True)
        park_status = bounded_recovery_pose_status(
            park_command,
            reviewed_start=home_review["measured_start"],
            home_command=home_review["home_command"],
            trace_poses=trace_poses,
        )
        if not park_status["strict_reviewed_path_pass"]:
            raise ControlledEpisodeError(
                "preserved Park command is not the reviewed trajectory start"
            )
        park_session_limits = home.build_session_limits(
            park_command, limits.soft_limits
        )
        park_commands = home.build_home_trajectory(
            limits.home_command,
            park_command,
            step_limits=home.HOME_STEP_LIMITS,
        )
        park_trajectory_audit = validate_monotonic_home_trajectory(
            start=limits.home_command,
            commands=park_commands,
            home_command=park_command,
            session_limits=park_session_limits,
            step_limits=home.HOME_STEP_LIMITS,
        )
        print(
            f"reviewed reverse Park trajectory: {len(park_commands)} commands, "
            f"{len(park_commands) / CONTROL_HZ:.3f}s; complete audit PASS",
            flush=True,
        )
        final_park_q = five.execute_sequence(
            phase="return_park",
            commands=park_commands,
            policy_steps=None,
            deadline_seconds=RETURN_PARK_DEADLINE_SECONDS,
            strict_commissioning=False,
            home=home,
            guard=guard,
            limits=limits,
            bus=bus,
            counters=counters,
            trace_rows=trace_rows,
        )
        final_park_q = five.hold_command(
            phase="final_park_hold",
            command=park_command,
            seconds=FINAL_PARK_HOLD_SECONDS,
            strict_commissioning=True,
            home=home,
            guard=guard,
            limits=limits,
            bus=bus,
            trace_rows=trace_rows,
        )
        motor, park_error = maximum_difference(final_park_q, park_command)
        if park_error > FINAL_PARK_TOLERANCE:
            raise ControlledEpisodeError(
                f"final Park return failed: {motor}={park_error:.3f}"
            )
        final_q = final_park_q
        park_completed = True
        motion_completed = True
        print(
            f"returned folded Park; worst actual error={motor}:{park_error:.3f}: "
            "PASS — torque may now be disabled",
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
                bus_disable_torque(five, home, bus)
                torque_may_be_enabled = False
                torque_disabled_at_exit = True
                print("Follower torque disabled.", flush=True)
            except Exception as exc:
                cleanup_errors.append(
                    f"torque disable failed: {type(exc).__name__}: {exc}"
                )
                print(
                    f"CRITICAL: torque disable failed; cut follower 12 V now: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            try:
                bus_disconnect(five, home, bus)
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

        if cleanup_errors and failure is None:
            failure = {
                "type": "CleanupError",
                "message": "; ".join(cleanup_errors),
                "traceback": None,
            }
            exit_code = 1

    if camera_start_attempted:
        try:
            camera_report, camera_failures = camera_session.report()
        except Exception as exc:
            camera_failures = [f"camera_report_failed: {type(exc).__name__}: {exc}"]
        acceptance_failures.extend(camera_failures)

    if acceptance_failures and failure is None:
        failure = {
            "type": "EpisodeAcceptanceError",
            "message": "; ".join(sorted(set(acceptance_failures))),
            "traceback": None,
        }
        exit_code = 1

    # Preserve every available diagnostic, including failed and partial runs.
    if camera_start_attempted:
        for role in ("front", "wrist"):
            state = hub.states[role]
            write_csv(
                output_dir / f"{role}_one_episode_frames.csv",
                state.get("rows", []),
            )
            last_frame = state.get("last_frame")
            if last_frame is not None:
                cv2.imwrite(str(output_dir / f"{role}_one_episode_last.png"), last_frame)
    write_csv(replan_path, replan_rows)
    write_csv(command_path, command_rows)
    write_csv(trace_path, trace_rows)

    success = (
        exit_code == 0
        and failure is None
        and policy_completed
        and motion_completed
        and park_completed
        and len(replan_rows) == REPLAN_COUNT
        and counters["policy"] == POLICY_COMMAND_LIMIT
        and len(command_rows) == POLICY_COMMAND_LIMIT
        and torque_disabled_at_exit
        and serial_closed
        and camera_finished
        and not camera_failures
        and task_outcome_images_saved
        and final_q is not None
    )
    decision = (
        "ONE_CONTROLLED_POLICY_EPISODE_EXECUTION_PASS_REVIEW_TASK_OUTCOME_NEXT"
        if success
        else "ONE_CONTROLLED_POLICY_EPISODE_FAILED_REVIEW_REQUIRED_MOTION_BLOCKED"
    )
    inference_p95 = percentile(inference_times_ms, 0.95)
    report = {
        "schema_version": 2,
        "created_at_utc": utc_now(),
        "raw_dataset_archive_note": {
            "windows_path": r"F:\episodes_pick_place_pilot_v5",
            "wsl_path": "/mnt/f/episodes_pick_place_pilot_v5",
            "used_by_this_frozen_episode": False,
        },
        "authorization": {
            "kind": "explicit_cli_one_controlled_policy_episode",
            "confirmations": confirmations,
            "authorized_replans": REPLAN_COUNT,
            "authorized_policy_commands": POLICY_COMMAND_LIMIT,
            "authorized_normal_home_return": True,
            "authorized_normal_folded_park_return": True,
            "automatic_retry_authorized": False,
            "ten_trial_evaluation_authorized": False,
            "autonomous_deployment_authorized": False,
        },
        "scope": {
            "model_loaded": True,
            "live_cameras_started": camera_started,
            "follower_serial_opened": connected,
            "operating_mode_read": operating_mode_read,
            "torque_enable_attempted": torque_enable_attempted,
            "torque_may_be_enabled_after_cleanup": torque_may_be_enabled,
            "goal_position_writes": counters,
            "dummy_warmup_inferences": 1,
            "live_policy_inferences": len(replan_rows),
            "policy_command_write_attempts": counters["policy_attempted"],
            "policy_command_writes_acknowledged": counters["policy"],
            "torque_disabled_at_exit": torque_disabled_at_exit,
            "serial_closed": serial_closed,
            "camera_session_finished": camera_finished,
            "camera_session_aborted": camera_aborted,
            "task_outcome_images_saved": task_outcome_images_saved,
            "folded_park_return_completed": park_completed,
            "task_success_claimed": False,
            "autonomous_deployment_authorized": False,
        },
        "static_capability_boundary": static_boundary,
        "local_safety_self_audit": local_audit,
        "frozen_dependencies": {
            "hashes": hashes,
            "stage3b_review": stage3b,
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
        "episode_contract": {
            "motor_order": list(MOTOR_ORDER),
            "replans": REPLAN_COUNT,
            "commands_per_replan": COMMANDS_PER_REPLAN,
            "maximum_policy_writes": POLICY_COMMAND_LIMIT,
            "control_hz_within_each_queue": CONTROL_HZ,
            "policy_hard_deadline_seconds": POLICY_HARD_DEADLINE_SECONDS,
            "camera_stream_seconds": CAMERA_STREAM_SECONDS,
            "camera_finish_grace_seconds": CAMERA_FINISH_GRACE_SECONDS,
            "reject_any_rate_clip": True,
            "reject_any_new_boundary_crossing": True,
            "accepted_inherited_soft_hold_joints": ["wrist_flex", "gripper"],
            "policy_tracking_limits": dict(
                zip(MOTOR_ORDER, POLICY_TRACKING_LIMITS, strict=True)
            ),
            "persistent_tracking_timeout_seconds": limits.tracking_timeout_seconds,
            "recovery_pose_gate": (
                "preferred: nearest complete reviewed six-joint Home trace "
                "<=5deg and reviewed start-to-Home envelope excess <=1deg; "
                "otherwise: fixed componentwise envelope expansion <=10deg "
                "plus a complete monotonic Home trajectory audit"
            ),
            "power_loss_recovery_envelope_margin_degrees": (
                POWER_LOSS_RECOVERY_ENVELOPE_MARGIN
            ),
            "full_recovery_trajectory_validated_before_first_goal_write": True,
            "full_recovery_trajectory_validated_before_torque_enable": True,
            "prior_torque_off_point_is_diagnostic_only": True,
            "normal_path_returns_home_then_folded_park": True,
            "folded_park_reference_source": (
                "accepted controlled Home report measured_start"
            ),
            "folded_park_command": dict(
                zip(MOTOR_ORDER, park_command, strict=True)
            ),
            "folded_park_step_limits": dict(
                zip(MOTOR_ORDER, home.HOME_STEP_LIMITS, strict=True)
            ),
            "folded_park_hold_seconds": FINAL_PARK_HOLD_SECONDS,
            "folded_park_final_tolerance_degrees": FINAL_PARK_TOLERANCE,
            "torque_disable_only_after_verified_park_on_normal_path": True,
            "abnormal_path_sends_no_recovery_commands": True,
        },
        "camera_start": camera_start_report,
        "camera_transport": camera_report,
        "camera_acceptance_failures": camera_failures,
        "run": {
            "start_q": None if start_q is None else list(start_q),
            "prior_stage3b_start_drift_diagnostic": {
                "joint": source_start_drift_motor,
                "drift": source_start_drift,
                "previous_reference_tolerance": SOURCE_START_DRIFT_TOLERANCE,
                "within_previous_reference": (
                    None
                    if source_start_drift is None
                    else source_start_drift <= SOURCE_START_DRIFT_TOLERANCE
                ),
                "used_as_motion_gate": False,
            },
            "live_recovery_pose_status": live_recovery_status,
            "recovery_mode": (
                None
                if live_recovery_status is None
                else live_recovery_status.get("recovery_mode")
            ),
            "recovery_trajectory_audit": recovery_trajectory_audit,
            "recovery_session_limits": (
                None
                if recovery_session_limits is None
                else {
                    motor: list(bounds)
                    for motor, bounds in zip(
                        MOTOR_ORDER, recovery_session_limits, strict=True
                    )
                }
            ),
            "home_aligned_q": (
                None if home_aligned_q is None else list(home_aligned_q)
            ),
            "episode_started_monotonic": episode_started,
            "episode_elapsed_seconds": (
                None
                if episode_started is None or episode_finished is None
                else episode_finished - episode_started
            ),
            "replans_completed": len(replan_rows),
            "policy_commands_completed": counters["policy"],
            "task_end_q": None if task_end_q is None else list(task_end_q),
            "final_home_q": (
                None if final_home_q is None else list(final_home_q)
            ),
            "park_command": list(park_command),
            "park_trajectory_audit": park_trajectory_audit,
            "park_session_limits": (
                None
                if park_session_limits is None
                else {
                    motor: list(bounds)
                    for motor, bounds in zip(
                        MOTOR_ORDER, park_session_limits, strict=True
                    )
                }
            ),
            "final_park_q": (
                None if final_park_q is None else list(final_park_q)
            ),
            "park_completed": park_completed,
            "final_q_before_torque_disable": (
                None if final_q is None else list(final_q)
            ),
            "inference_ms_p50": percentile(inference_times_ms, 0.50),
            "inference_ms_p95": inference_p95,
            "inference_ms_max": (
                max(inference_times_ms) if inference_times_ms else None
            ),
            "joint_read_ms_p95": percentile(joint_read_times_ms, 0.95),
            "joint_read_ms_max": (
                max(joint_read_times_ms) if joint_read_times_ms else None
            ),
            "assembly_metrics_p95": {
                key: percentile(values, 0.95)
                for key, values in assembly_metrics_samples.items()
            },
            "inherited_soft_holds_by_joint": dict(inherited_soft_holds),
            "rate_clips_in_acknowledged_commands": (
                sum(int(row["any_rate_clip"]) for row in command_rows)
            ),
            "new_boundary_crossings_in_acknowledged_commands": sum(
                int(row[f"new_boundary_crossing_{motor}"])
                for row in command_rows
                for motor in MOTOR_ORDER
            ),
            "measured_soft_excess_max_by_joint": dict(measured_soft_excess_max),
            "max_guard_invariant_error": max_guard_invariant_error,
            "trace_rows": len(trace_rows),
            "cleanup_errors": cleanup_errors,
        },
        "task_outcome_review": {
            "pending": success,
            "execution_pass_does_not_claim_pick_place_success": True,
            "front_image": (
                str(output_dir / "front_task_outcome_before_home.png")
                if task_outcome_images_saved
                else None
            ),
            "wrist_image": (
                str(output_dir / "wrist_task_outcome_before_home.png")
                if task_outcome_images_saved
                else None
            ),
            "required_before_next_motion": True,
        },
        "acceptance_failures": sorted(set(acceptance_failures)),
        "failure": failure,
        "decision": decision,
        "stage3c_execution_complete": success,
        "stage3c_task_outcome_accepted": False,
        "next_motion_gate_authorized": False,
        "ten_trial_evaluation_authorized": False,
        "autonomous_deployment_authorized": False,
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\n===== ONE-EPISODE METRICS =====", flush=True)
    print(
        f"replans={len(replan_rows)}/{REPLAN_COUNT} "
        f"policy_commands={counters['policy']}/{POLICY_COMMAND_LIMIT}",
        flush=True,
    )
    if inference_times_ms:
        print(
            f"inference p95={percentile(inference_times_ms, 0.95):.3f}ms "
            f"max={max(inference_times_ms):.3f}ms",
            flush=True,
        )
    print(
        f"accepted inherited boundary holds={sum(inherited_soft_holds.values())} "
        f"{dict(inherited_soft_holds)}",
        flush=True,
    )
    print(
        f"normal folded Park return completed={park_completed} "
        f"commands={counters['return_park']}",
        flush=True,
    )
    print(f"max guard invariant error={max_guard_invariant_error:.10f}", flush=True)

    print("\n===== DECISION =====", flush=True)
    print(f"decision='{decision}'", flush=True)
    if failure is not None:
        print(
            f"failure={failure['type']}: {failure['message']}",
            file=sys.stderr,
            flush=True,
        )
    print(f"torque disabled at exit: {torque_disabled_at_exit}", flush=True)
    print(f"serial closed: {serial_closed}", flush=True)
    if success:
        print(
            "EXECUTION PASSED; FOLDED PARK VERIFIED BEFORE TORQUE DISABLE; "
            "REVIEW TASK-OUTCOME IMAGES BEFORE ANY NEXT MOTION.",
            flush=True,
        )
    else:
        print("MOTION REMAINS BLOCKED; DO NOT AUTOMATICALLY RETRY.", flush=True)
    print("AUTONOMOUS DEPLOYMENT REMAINS BLOCKED.", flush=True)
    print("\n===== OUTPUT =====", flush=True)
    print(report_path, flush=True)
    if replan_rows:
        print(replan_path, flush=True)
    if command_rows:
        print(command_path, flush=True)
    if trace_rows:
        print(trace_path, flush=True)
    if task_outcome_images_saved:
        print(output_dir / "front_task_outcome_before_home.png", flush=True)
        print(output_dir / "wrist_task_outcome_before_home.png", flush=True)
    print(
        "ACT V3 ONE CONTROLLED POLICY EPISODE: PASS — TASK REVIEW REQUIRED"
        if success
        else "ACT V3 ONE CONTROLLED POLICY EPISODE: FAIL",
        flush=True,
    )

    del policy
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0 if success else (exit_code or 1)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(
            "ACT V3 ONE CONTROLLED POLICY EPISODE: FAIL-CLOSED: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        print("NO MOTION IS AUTHORIZED BY THIS FAILED PROCESS.", file=sys.stderr)
        raise SystemExit(1)
