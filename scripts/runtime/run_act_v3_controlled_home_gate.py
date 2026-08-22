#!/usr/bin/env python3
"""Move the frozen SO-101 follower once from its reviewed folded pose to Home.

This is the first intentionally commanding ACT V3 commissioning gate.  It does
not load a model or open cameras.  It accepts exactly one follower serial
identity and performs this bounded sequence:

1. Verify the frozen runtime contract, calibration, vendor sources, completed
   60-second live-shadow report, and last live-shadow joint sample.
2. Open only the follower motor bus and read Operating_Mode/Present_Position.
3. Wait through a five-second physical-safety countdown without motor writes.
4. Disable torque, seed Goal_Position to the measured folded pose, verify the
   goal readback, then enable torque.
5. Interpolate all six joints together from the measured pose to the frozen
   dataset Home command at 15 Hz with conservative per-command steps.
6. Continuously validate calibrated sensor ranges, command envelopes, tracking
   error, and monotonic progress; any exception or Ctrl+C disables torque.
7. Hold and verify Home, then disable torque and close the serial transport.

The arm must have stable external support throughout because both successful
and failed runs finish with torque disabled.  Do not support a moving link by
hand or place hands in a joint/gripper sweep or pinch zone.  Passing this gate
authorizes neither policy actions nor autonomous deployment; it only prepares
the separate five-command commissioning gate.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


MOTOR_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

EXPECTED_GUARD_SHA256 = (
    "e06e785071ee3ded95e89d670978bcb4a8caf104b2fc69fc87de37cb6fff6d15"
)
EXPECTED_ONE_SHOT_LAUNCHER_SHA256 = (
    "c2f2be0481e72aa1836bae36e873be59b9c739b776d1baf5b510954264b9196b"
)
EXPECTED_LIVE_SHADOW_SHA256 = (
    "978928157e4fb441d95c4a385f3192f463065723b61bda3d404f9fe4b7018c02"
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
EXPECTED_SHADOW_DECISION = (
    "LIVE_SHADOW_PASS_PIPELINE_START_POSE_NOT_MOTION_READY_"
    "PREPARE_CONTROLLED_HOME_GATE_NEXT"
)

CONTROL_HZ = 15.0
COUNTDOWN_SECONDS = 5
SETTLE_SECONDS = 1.0
HOME_HOLD_SECONDS = 3.0
MAX_MOVE_SECONDS = 15.0
START_MATCH_TOLERANCE = 3.0
COUNTDOWN_DRIFT_LIMIT = 3.0
GOAL_READBACK_TOLERANCE = 1.5
TORQUE_LOCK_TOLERANCE = 3.0
HOME_TOLERANCE = 2.0
IMMEDIATE_TRACKING_MULTIPLIER = 2.0

# More conservative than the frozen runtime action limits.  A single scalar
# interpolation fraction is used so every joint reaches Home simultaneously.
HOME_STEP_LIMITS = (0.75, 0.60, 0.60, 0.75, 1.00, 1.25)


class ControlledHomeError(RuntimeError):
    """A frozen input, physical precondition, bus operation, or motion gate failed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guard-core", type=Path, required=True)
    parser.add_argument("--one-shot-launcher", type=Path, required=True)
    parser.add_argument("--live-shadow-runner", type=Path, required=True)
    parser.add_argument("--live-shadow-report", type=Path, required=True)
    parser.add_argument("--live-shadow-replans-csv", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--follower-source", type=Path, required=True)
    parser.add_argument("--follower-config-source", type=Path, required=True)
    parser.add_argument("--follower-port", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--authorize-controlled-home-motion",
        action="store_true",
        help="Authorize only this bounded folded-pose-to-Home motor trajectory.",
    )
    parser.add_argument(
        "--confirm-arm-supported",
        action="store_true",
        help=(
            "Confirm stable external support prevents a torque-off fall without "
            "putting hands in the motion or pinch zone."
        ),
    )
    parser.add_argument(
        "--confirm-workspace-clear",
        action="store_true",
        help="Confirm the full unfolding workspace is clear of people and obstacles.",
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ControlledHomeError(
            f"{label} SHA256 mismatch: expected={expected}, actual={actual}"
        )
    return actual


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ControlledHomeError(f"invalid JSON: {path}: {exc}") from exc


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ControlledHomeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def finite_vector(label: str, values: Sequence[Any]) -> tuple[float, ...]:
    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ControlledHomeError(f"{label} must be numeric") from exc
    if len(vector) != len(MOTOR_ORDER):
        raise ControlledHomeError(
            f"{label} must contain {len(MOTOR_ORDER)} values, got {len(vector)}"
        )
    invalid = [index for index, value in enumerate(vector) if not math.isfinite(value)]
    if invalid:
        raise ControlledHomeError(f"{label} contains nonfinite values at {invalid}")
    return vector


def ordered_mapping(label: str, values: Any) -> tuple[float, ...]:
    if not isinstance(values, Mapping):
        raise ControlledHomeError(f"{label} must be a mapping")
    if set(values) != set(MOTOR_ORDER):
        raise ControlledHomeError(
            f"{label} keys mismatch: expected={list(MOTOR_ORDER)}, got={list(values)}"
        )
    return finite_vector(label, [values[motor] for motor in MOTOR_ORDER])


def vector_mapping(values: Sequence[float]) -> dict[str, float]:
    vector = finite_vector("motor vector", values)
    return dict(zip(MOTOR_ORDER, vector, strict=True))


def maximum_difference(left: Sequence[float], right: Sequence[float]) -> tuple[str, float]:
    a = finite_vector("left", left)
    b = finite_vector("right", right)
    differences = [abs(x - y) for x, y in zip(a, b, strict=True)]
    index = max(range(len(differences)), key=differences.__getitem__)
    return MOTOR_ORDER[index], differences[index]


def read_present_position(bus: Any) -> tuple[float, ...]:
    return ordered_mapping(
        "Present_Position",
        bus.sync_read("Present_Position", normalize=True, num_retry=3),
    )


def read_goal_position(bus: Any) -> tuple[float, ...]:
    return ordered_mapping(
        "Goal_Position",
        bus.sync_read("Goal_Position", normalize=True, num_retry=3),
    )


def read_operating_mode(bus: Any) -> dict[str, int]:
    values = bus.sync_read("Operating_Mode", normalize=False, num_retry=3)
    if not isinstance(values, Mapping) or set(values) != set(MOTOR_ORDER):
        raise ControlledHomeError("Operating_Mode mapping keys mismatch")
    try:
        result = {motor: int(values[motor]) for motor in MOTOR_ORDER}
    except (TypeError, ValueError) as exc:
        raise ControlledHomeError("Operating_Mode values must be integers") from exc
    invalid = {motor: value for motor, value in result.items() if value != 0}
    if invalid:
        raise ControlledHomeError(f"non-position Operating_Mode detected: {invalid}")
    return result


def write_goal_position(bus: Any, command: Sequence[float]) -> None:
    bus.sync_write(
        "Goal_Position",
        vector_mapping(command),
        normalize=True,
        num_retry=1,
    )


def enable_position_torque(bus: Any) -> None:
    bus.enable_torque(num_retry=5)


def disable_position_torque(bus: Any) -> None:
    bus.disable_torque(num_retry=5)


def connect_motor_bus(bus: Any) -> None:
    bus.connect()


def disconnect_motor_bus(bus: Any) -> None:
    bus.disconnect(disable_torque=False)


def verify_static_motion_boundary(path: Path) -> dict[str, Any]:
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
        "sync_write": 1,
        "enable_torque": 1,
        "disable_torque": 1,
        "connect": 1,
        "disconnect": 1,
        "sync_read": 3,
    }
    count_mismatches = {
        name: {"expected": expected, "actual": len(by_attr.get(name, []))}
        for name, expected in expected_counts.items()
        if len(by_attr.get(name, [])) != expected
    }
    forbidden = {
        "send_action",
        "configure",
        "configure_motors",
        "calibrate",
        "setup_motors",
        "write",
        "write_calibration",
        "torque_disabled",
    }
    forbidden_found = sorted(forbidden.intersection(by_attr))

    sync_write = by_attr.get("sync_write", [])
    if sync_write:
        call = sync_write[0]
        write_register = (
            call.args[0].value
            if call.args and isinstance(call.args[0], ast.Constant)
            else None
        )
        keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords}
        write_ok = (
            write_register == "Goal_Position"
            and keywords == {"normalize": "True", "num_retry": "1"}
        )
    else:
        write_register = None
        write_ok = False

    read_registers: list[str | None] = []
    read_keywords_ok = True
    for call in by_attr.get("sync_read", []):
        register = (
            call.args[0].value
            if call.args and isinstance(call.args[0], ast.Constant)
            else None
        )
        read_registers.append(register)
        keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords}
        expected = (
            {"normalize": "False", "num_retry": "3"}
            if register == "Operating_Mode"
            else {"normalize": "True", "num_retry": "3"}
        )
        read_keywords_ok = read_keywords_ok and keywords == expected

    disconnect_calls = by_attr.get("disconnect", [])
    disconnect_ok = False
    if disconnect_calls:
        keywords = {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in disconnect_calls[0].keywords
        }
        disconnect_ok = keywords == {"disable_torque": "False"}

    if (
        count_mismatches
        or forbidden_found
        or not write_ok
        or sorted(read_registers) != ["Goal_Position", "Operating_Mode", "Present_Position"]
        or not read_keywords_ok
        or not disconnect_ok
    ):
        raise ControlledHomeError(
            "controlled-home source capability boundary mismatch: "
            f"counts={count_mismatches}, forbidden={forbidden_found}, "
            f"write_register={write_register}, reads={read_registers}, "
            f"read_keywords_ok={read_keywords_ok}, disconnect_ok={disconnect_ok}"
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "allowed_register_reads": sorted(str(value) for value in read_registers),
        "allowed_register_write": "Goal_Position",
        "goal_write_call_sites": 1,
        "torque_enable_call_sites": 1,
        "torque_disable_call_sites": 1,
        "direct_bus_connect_call_sites": 1,
        "torque_preserving_disconnect_call_sites": 1,
        "full_follower_connect_or_disconnect_calls": False,
        "model_or_camera_calls": False,
        "pass": True,
    }


def verify_shadow_gate(
    *,
    runner_path: Path,
    report_path: Path,
    replans_path: Path,
) -> dict[str, Any]:
    runner_sha = require_sha(
        runner_path,
        EXPECTED_LIVE_SHADOW_SHA256,
        "live shadow runner",
    )
    report = load_json(report_path)
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise ControlledHomeError("unexpected live-shadow report schema")
    if report.get("decision") != EXPECTED_SHADOW_DECISION:
        raise ControlledHomeError("live-shadow decision mismatch")
    if report.get("acceptance_failures") != []:
        raise ControlledHomeError("live-shadow report contains acceptance failures")
    if report.get("motion_authorized") is not False:
        raise ControlledHomeError("live-shadow report motion scope mismatch")
    scope = report.get("scope") or {}
    required_scope = {
        "command_sent": False,
        "motor_register_write_api_called": False,
        "goal_position_written": False,
        "torque_api_called": False,
        "action_api_called": False,
        "serial_port_closed": True,
        "motion_authorized": False,
        "hardware_deployment_authorized": False,
    }
    mismatched_scope = {
        key: {"expected": expected, "actual": scope.get(key)}
        for key, expected in required_scope.items()
        if scope.get(key) is not expected
    }
    if mismatched_scope:
        raise ControlledHomeError(
            f"live-shadow safety scope mismatch: {mismatched_scope}"
        )
    metrics = report.get("shadow_metrics") or {}
    if int(metrics.get("completed_replans", -1)) != 180:
        raise ControlledHomeError("live-shadow did not complete 180 replans")
    if int(metrics.get("predictions_sent_to_hardware", -1)) != 0:
        raise ControlledHomeError("live-shadow predictions were not purely diagnostic")
    static = report.get("static_no_command_boundary") or {}
    if static.get("sha256") != runner_sha or static.get("pass") is not True:
        raise ControlledHomeError("live-shadow static source identity mismatch")
    transport = report.get("camera_transport") or {}
    cameras = transport.get("cameras") or {}
    for role in ("front", "wrist"):
        camera = cameras.get(role) or {}
        if camera.get("receiver_status") != "PASS":
            raise ControlledHomeError(f"live-shadow {role} receiver did not pass")
        worker = camera.get("windows_worker") or {}
        if worker.get("status") != "PASS" or worker.get("exit_code") != 0:
            raise ControlledHomeError(f"live-shadow {role} worker did not pass")

    with replans_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 180:
        raise ControlledHomeError(
            f"live-shadow replan CSV row count mismatch: {len(rows)}"
        )
    indices = [int(row["replan_index"]) for row in rows]
    if indices != list(range(180)):
        raise ControlledHomeError("live-shadow replan CSV indices are not 0..179")
    last_q = finite_vector(
        "last live-shadow actual_q",
        [rows[-1][f"actual_q_{motor}"] for motor in MOTOR_ORDER],
    )
    return {
        "runner": str(runner_path),
        "runner_sha256": runner_sha,
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "replans_csv": str(replans_path),
        "replans_csv_sha256": sha256_file(replans_path),
        "decision": report["decision"],
        "completed_replans": 180,
        "last_actual_q": list(last_q),
        "no_commands_sent": True,
        "serial_closed": True,
    }


def build_session_limits(
    start: Sequence[float],
    soft_limits: Sequence[Sequence[float]],
) -> tuple[tuple[float, float], ...]:
    start_vector = finite_vector("start", start)
    if len(soft_limits) != len(MOTOR_ORDER):
        raise ControlledHomeError("soft-limit dimension mismatch")
    result: list[tuple[float, float]] = []
    for value, bounds in zip(start_vector, soft_limits, strict=True):
        minimum, maximum = float(bounds[0]), float(bounds[1])
        if value < minimum:
            result.append((value, maximum))
        elif value > maximum:
            result.append((minimum, value))
        else:
            result.append((minimum, maximum))
    return tuple(result)


def build_home_trajectory(
    start: Sequence[float],
    home: Sequence[float],
    step_limits: Sequence[float] = HOME_STEP_LIMITS,
) -> list[tuple[float, ...]]:
    start_vector = finite_vector("trajectory start", start)
    home_vector = finite_vector("trajectory Home", home)
    limits = finite_vector("trajectory step limits", step_limits)
    if any(limit <= 0.0 for limit in limits):
        raise ControlledHomeError("trajectory step limits must be positive")
    ratios = [
        abs(target - source) / limit
        for source, target, limit in zip(
            start_vector, home_vector, limits, strict=True
        )
    ]
    steps = max(1, int(math.ceil(max(ratios))))
    if steps / CONTROL_HZ > MAX_MOVE_SECONDS:
        raise ControlledHomeError(
            f"Home trajectory exceeds {MAX_MOVE_SECONDS:.1f}s bound: {steps} steps"
        )
    trajectory = [
        tuple(
            source + (target - source) * (step / steps)
            for source, target in zip(start_vector, home_vector, strict=True)
        )
        for step in range(1, steps + 1)
    ]
    trajectory[-1] = home_vector
    previous = start_vector
    for index, command in enumerate(trajectory, start=1):
        for joint, (source, target, prior, value, limit) in enumerate(
            zip(start_vector, home_vector, previous, command, limits, strict=True)
        ):
            if abs(value - prior) > limit + 1e-9:
                raise ControlledHomeError(
                    f"trajectory step {index} exceeds limit for {MOTOR_ORDER[joint]}"
                )
            if abs(target - value) > abs(target - prior) + 1e-9:
                raise ControlledHomeError(
                    f"trajectory step {index} moves outward for {MOTOR_ORDER[joint]}"
                )
            lower, upper = sorted((source, target))
            if not lower - 1e-9 <= value <= upper + 1e-9:
                raise ControlledHomeError(
                    f"trajectory step {index} leaves endpoint envelope for {MOTOR_ORDER[joint]}"
                )
        previous = command
    return trajectory


def validate_command_envelope(
    command: Sequence[float],
    session_limits: Sequence[Sequence[float]],
) -> tuple[float, ...]:
    vector = finite_vector("command", command)
    violations: dict[str, dict[str, Any]] = {}
    for motor, value, bounds in zip(
        MOTOR_ORDER, vector, session_limits, strict=True
    ):
        minimum, maximum = float(bounds[0]), float(bounds[1])
        if not minimum - 1e-9 <= value <= maximum + 1e-9:
            violations[motor] = {
                "value": value,
                "session_limit": [minimum, maximum],
            }
    if violations:
        raise ControlledHomeError(f"command left session envelope: {violations}")
    return vector


def run_algorithm_self_audit(limits: Any) -> dict[str, Any]:
    tests: dict[str, bool] = {}
    folded = (2.37548, -95.00208, 98.73360, 68.66554, -0.05179, 6.40687)
    trajectory = build_home_trajectory(folded, limits.home_command)
    session = build_session_limits(folded, limits.soft_limits)
    previous = folded
    for command in trajectory:
        validate_command_envelope(command, session)
        for value, prior, home in zip(
            command, previous, limits.home_command, strict=True
        ):
            assert abs(home - value) <= abs(home - prior) + 1e-9
        previous = command
    tests["folded_pose_moves_monotonically_inward"] = True
    tests["all_commands_remain_in_start_to_home_envelope"] = True
    tests["all_joints_arrive_at_exact_frozen_home"] = trajectory[-1] == tuple(
        limits.home_command
    )
    tests["trajectory_duration_is_bounded"] = (
        len(trajectory) / CONTROL_HZ <= MAX_MOVE_SECONDS
    )
    tests["home_steps_are_stricter_than_runtime_limits"] = all(
        home_step < runtime_step
        for home_step, runtime_step in zip(
            HOME_STEP_LIMITS,
            limits.max_delta_per_command,
            strict=True,
        )
    )
    tests["current_folded_pose_requires_motion_block"] = any(
        not minimum <= value <= maximum
        for value, (minimum, maximum) in zip(
            folded, limits.soft_limits, strict=True
        )
    )
    if not all(tests.values()):
        raise ControlledHomeError(f"controlled-home self-audit failed: {tests}")
    return {
        "tests": tests,
        "tests_passed": sum(tests.values()),
        "tests_total": len(tests),
        "folded_fixture_steps": len(trajectory),
        "folded_fixture_duration_seconds": len(trajectory) / CONTROL_HZ,
    }


def make_log_row(
    *,
    phase: str,
    sequence: int,
    elapsed_seconds: float,
    command: Sequence[float],
    actual: Sequence[float],
    tracking: Any,
) -> dict[str, Any]:
    command_vector = finite_vector("logged command", command)
    actual_vector = finite_vector("logged actual", actual)
    row: dict[str, Any] = {
        "phase": phase,
        "sequence": sequence,
        "elapsed_seconds": elapsed_seconds,
        "tracking_tripped": int(tracking.tripped),
        "tracking_tripped_joints": ";".join(tracking.tripped_joints),
    }
    for index, motor in enumerate(MOTOR_ORDER):
        row[f"command_{motor}"] = command_vector[index]
        row[f"actual_{motor}"] = actual_vector[index]
        row[f"tracking_error_{motor}"] = float(tracking.absolute_error[index])
        row[f"over_limit_seconds_{motor}"] = float(
            tracking.over_limit_seconds[index]
        )
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ControlledHomeError("controlled-home CSV schema changed")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    confirmations = {
        "authorize_controlled_home_motion": args.authorize_controlled_home_motion,
        "arm_supported": args.confirm_arm_supported,
        "workspace_clear": args.confirm_workspace_clear,
        "power_cutoff_ready": args.confirm_power_cutoff_ready,
    }
    missing = [name for name, present in confirmations.items() if not present]
    if missing:
        raise ControlledHomeError(f"required physical confirmations missing: {missing}")

    script_path = Path(__file__).resolve()
    guard_path = require_file(args.guard_core)
    one_shot_path = require_file(args.one_shot_launcher)
    shadow_runner_path = require_file(args.live_shadow_runner)
    shadow_report_path = require_file(args.live_shadow_report)
    shadow_replans_path = require_file(args.live_shadow_replans_csv)
    contract_path = require_file(args.contract)
    calibration_path = require_file(args.calibration)
    follower_source = require_file(args.follower_source)
    follower_config_source = require_file(args.follower_config_source)
    follower_port = args.follower_port.expanduser()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"output directory already exists; preserve it and choose a retry path: {output_dir}"
        )

    print("===== EXPLICIT CONTROLLED-HOME MOTION AUTHORIZATION =====", flush=True)
    print("authorized: one folded-pose -> frozen Home trajectory only", flush=True)
    print("control=15Hz; joint-space interpolation; <=15s motion", flush=True)
    print("model/cameras/policy actions: FORBIDDEN", flush=True)
    print(
        "external support, clear workspace, hands clear, 12V cutoff: CONFIRMED",
        flush=True,
    )
    print("normal and abnormal exit both disable torque", flush=True)
    print("autonomous deployment authorization: false", flush=True)

    print("\n===== VERIFY FROZEN INPUTS AND PREVIOUS GATE =====", flush=True)
    static_boundary = verify_static_motion_boundary(script_path)
    hashes = {
        "guard_core": require_sha(guard_path, EXPECTED_GUARD_SHA256, "guard core"),
        "one_shot_launcher": require_sha(
            one_shot_path,
            EXPECTED_ONE_SHOT_LAUNCHER_SHA256,
            "one-shot launcher",
        ),
        "live_shadow_runner": require_sha(
            shadow_runner_path,
            EXPECTED_LIVE_SHADOW_SHA256,
            "live shadow runner",
        ),
        "follower_source": require_sha(
            follower_source,
            EXPECTED_FOLLOWER_SHA256,
            "follower source",
        ),
        "follower_config_source": require_sha(
            follower_config_source,
            EXPECTED_FOLLOWER_CONFIG_SHA256,
            "follower config source",
        ),
    }
    shadow_gate = verify_shadow_gate(
        runner_path=shadow_runner_path,
        report_path=shadow_report_path,
        replans_path=shadow_replans_path,
    )
    guard = load_module(guard_path, "act_v3_controlled_home_guard")
    one_shot = load_module(one_shot_path, "act_v3_controlled_home_one_shot")
    frozen_inputs = guard.verify_frozen_inputs(contract_path, calibration_path)
    limits = guard.RuntimeLimits.frozen_v3()
    guard_audit = guard.run_algorithm_self_audit(limits)
    if guard_audit["tests_passed"] != guard_audit["tests_total"]:
        raise ControlledHomeError("frozen guard self-audit did not fully pass")
    home_audit = run_algorithm_self_audit(limits)
    print("live shadow 180/180, no-command scope, source identities: PASS", flush=True)
    print(
        f"guard={guard_audit['tests_passed']}/{guard_audit['tests_total']} "
        f"home={home_audit['tests_passed']}/{home_audit['tests_total']}: PASS",
        flush=True,
    )

    print("\n===== VERIFY EXACT FOLLOWER AND CONSTRUCT INERT BUS =====", flush=True)
    device = one_shot.verify_follower_device(follower_port)
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
    report_path = output_dir / "controlled_home_report.json"
    csv_path = output_dir / "controlled_home_trace.csv"
    expected_start = finite_vector(
        "expected live-shadow start",
        shadow_gate["last_actual_q"],
    )
    bus = follower.bus
    watchdog = guard.TrackingWatchdog(limits)
    trace_rows: list[dict[str, Any]] = []
    connect_attempted = False
    connected = False
    operating_mode_read = False
    torque_enable_attempted = False
    torque_may_be_enabled = False
    torque_disabled_at_exit = False
    serial_closed = False
    goal_write_count = 0
    present_read_count = 0
    move_started: float | None = None
    move_finished: float | None = None
    start_q: tuple[float, ...] | None = None
    final_q: tuple[float, ...] | None = None
    session_limits: tuple[tuple[float, float], ...] | None = None
    trajectory_steps = 0
    failure: dict[str, Any] | None = None
    exit_code = 0
    cleanup_errors: list[str] = []

    try:
        print("\n===== READ-ONLY PRE-MOTION CHECKS =====", flush=True)
        connect_attempted = True
        connect_motor_bus(bus)
        connected = True
        modes = read_operating_mode(bus)
        operating_mode_read = True
        start_q = guard.DeltaCommandGuard(
            limits,
            limits.home_command,
        ).validate_measured_positions(read_present_position(bus))
        present_read_count += 1
        drift_motor, drift = maximum_difference(start_q, expected_start)
        if drift > START_MATCH_TOLERANCE:
            raise ControlledHomeError(
                "live start differs from completed shadow pose: "
                f"{drift_motor} drift={drift:.3f} > {START_MATCH_TOLERANCE:.3f}"
            )
        session_limits = build_session_limits(start_q, limits.soft_limits)
        trajectory = build_home_trajectory(start_q, limits.home_command)
        trajectory_steps = len(trajectory)
        print("Operating_Mode=POSITION for all six joints: PASS", flush=True)
        print(
            "start: "
            + " ".join(
                f"{motor}={value:.3f}"
                for motor, value in zip(MOTOR_ORDER, start_q, strict=True)
            ),
            flush=True,
        )
        print(
            f"planned trajectory: {trajectory_steps} commands, "
            f"{trajectory_steps / CONTROL_HZ:.3f}s",
            flush=True,
        )

        print("\n===== FIVE-SECOND NO-WRITE SAFETY COUNTDOWN =====", flush=True)
        print(
            "VERIFY EXTERNAL SUPPORT; keep hands and obstacles outside the "
            "complete unfolding path.",
            flush=True,
        )
        countdown_reference = start_q
        for remaining in range(COUNTDOWN_SECONDS, 0, -1):
            print(
                f"motion preparation in {remaining}s — Ctrl+C cancels before torque change",
                flush=True,
            )
            time.sleep(1.0)
            current = guard.DeltaCommandGuard(
                limits,
                limits.home_command,
            ).validate_measured_positions(read_present_position(bus))
            present_read_count += 1
            motor, value = maximum_difference(current, countdown_reference)
            if value > COUNTDOWN_DRIFT_LIMIT:
                raise ControlledHomeError(
                    f"arm moved during countdown: {motor} drift={value:.3f}"
                )

        print("\n===== TORQUE-SAFE CURRENT-GOAL LOCK =====", flush=True)
        disable_position_torque(bus)
        torque_disabled_at_exit = True
        current = guard.DeltaCommandGuard(
            limits,
            limits.home_command,
        ).validate_measured_positions(read_present_position(bus))
        present_read_count += 1
        motor, value = maximum_difference(current, countdown_reference)
        if value > COUNTDOWN_DRIFT_LIMIT:
            raise ControlledHomeError(
                f"arm moved while torque was disabled: {motor} drift={value:.3f}"
            )
        start_q = current
        session_limits = build_session_limits(start_q, limits.soft_limits)
        trajectory = build_home_trajectory(start_q, limits.home_command)
        trajectory_steps = len(trajectory)
        write_goal_position(bus, start_q)
        goal_write_count += 1
        goal_readback = read_goal_position(bus)
        motor, value = maximum_difference(goal_readback, start_q)
        if value > GOAL_READBACK_TOLERANCE:
            raise ControlledHomeError(
                f"current-goal readback mismatch: {motor} error={value:.3f}"
            )
        # Treat torque as potentially enabled before making the call.  Some
        # transports can apply a write and still raise while collecting the
        # acknowledgement, so cleanup must not rely on a successful return.
        torque_enable_attempted = True
        torque_may_be_enabled = True
        torque_disabled_at_exit = False
        enable_position_torque(bus)
        print("current Goal seeded and verified; follower torque enabled: PASS", flush=True)

        settle_deadline = time.monotonic() + SETTLE_SECONDS
        while time.monotonic() < settle_deadline:
            actual = guard.DeltaCommandGuard(
                limits,
                limits.home_command,
            ).validate_measured_positions(read_present_position(bus))
            present_read_count += 1
            motor, value = maximum_difference(actual, start_q)
            if value > TORQUE_LOCK_TOLERANCE:
                raise ControlledHomeError(
                    f"arm moved after current-goal torque lock: {motor} drift={value:.3f}"
                )
            time.sleep(1.0 / CONTROL_HZ)

        print("\n===== CONTROLLED FOLDED-POSE -> HOME TRAJECTORY =====", flush=True)
        watchdog.reset()
        move_started = time.monotonic()
        next_tick = move_started
        previous_command = start_q
        for sequence, command in enumerate(trajectory, start=1):
            validate_command_envelope(command, session_limits)
            for index, (prior, value, limit, target) in enumerate(
                zip(
                    previous_command,
                    command,
                    HOME_STEP_LIMITS,
                    limits.home_command,
                    strict=True,
                )
            ):
                if abs(value - prior) > limit + 1e-9:
                    raise ControlledHomeError(
                        f"runtime step exceeded for {MOTOR_ORDER[index]}"
                    )
                if abs(target - value) > abs(target - prior) + 1e-9:
                    raise ControlledHomeError(
                        f"runtime command moved outward for {MOTOR_ORDER[index]}"
                    )
            write_goal_position(bus, command)
            goal_write_count += 1
            next_tick += 1.0 / CONTROL_HZ
            delay = next_tick - time.monotonic()
            if delay > 0.0:
                time.sleep(delay)
            else:
                # Drop accumulated scheduler debt.  Never emit catch-up motor
                # commands back-to-back after a slow serial transaction.
                next_tick = time.monotonic()
            actual = guard.DeltaCommandGuard(
                limits,
                limits.home_command,
            ).validate_measured_positions(read_present_position(bus))
            present_read_count += 1
            now = time.monotonic()
            if now - move_started > MAX_MOVE_SECONDS:
                raise ControlledHomeError(
                    f"motion deadline exceeded: {now - move_started:.3f}s > "
                    f"{MAX_MOVE_SECONDS:.3f}s"
                )
            tracking = watchdog.update(actual, command, now)
            immediate = {
                motor: tracking.absolute_error[index]
                for index, motor in enumerate(MOTOR_ORDER)
                if tracking.absolute_error[index]
                > limits.tracking_error_limits[index]
                * IMMEDIATE_TRACKING_MULTIPLIER
            }
            if immediate:
                raise ControlledHomeError(
                    f"immediate tracking-error stop: {immediate}"
                )
            if tracking.tripped:
                raise ControlledHomeError(
                    f"persistent tracking-error stop: {tracking.tripped_joints}"
                )
            trace_rows.append(
                make_log_row(
                    phase="move",
                    sequence=sequence,
                    elapsed_seconds=now - move_started,
                    command=command,
                    actual=actual,
                    tracking=tracking,
                )
            )
            previous_command = command
            if sequence % 15 == 0 or sequence == trajectory_steps:
                _, worst = maximum_difference(actual, command)
                print(
                    f"commands={sequence}/{trajectory_steps} "
                    f"elapsed={now - move_started:.1f}s "
                    f"worst_tracking={worst:.3f}",
                    flush=True,
                )
        move_finished = time.monotonic()

        print("\n===== HOME HOLD AND FINAL VERIFICATION =====", flush=True)
        hold_deadline = time.monotonic() + HOME_HOLD_SECONDS
        hold_sequence = 0
        while time.monotonic() < hold_deadline:
            time.sleep(1.0 / CONTROL_HZ)
            final_q = guard.DeltaCommandGuard(
                limits,
                limits.home_command,
            ).validate_measured_positions(read_present_position(bus))
            present_read_count += 1
            now = time.monotonic()
            tracking = watchdog.update(final_q, limits.home_command, now)
            immediate = {
                motor: tracking.absolute_error[index]
                for index, motor in enumerate(MOTOR_ORDER)
                if tracking.absolute_error[index]
                > limits.tracking_error_limits[index]
                * IMMEDIATE_TRACKING_MULTIPLIER
            }
            if immediate:
                raise ControlledHomeError(
                    f"Home hold immediate tracking-error stop: {immediate}"
                )
            if tracking.tripped:
                raise ControlledHomeError(
                    f"Home hold tracking-error stop: {tracking.tripped_joints}"
                )
            hold_sequence += 1
            trace_rows.append(
                make_log_row(
                    phase="hold",
                    sequence=hold_sequence,
                    elapsed_seconds=now - move_started,
                    command=limits.home_command,
                    actual=final_q,
                    tracking=tracking,
                )
            )
        if final_q is None:
            raise ControlledHomeError("Home hold produced no final readback")
        motor, home_error = maximum_difference(final_q, limits.home_command)
        if home_error > HOME_TOLERANCE:
            raise ControlledHomeError(
                f"final Home tolerance failed: {motor} error={home_error:.3f}"
            )
        print(
            f"Home reached and held for {HOME_HOLD_SECONDS:.1f}s; "
            f"worst final error={motor}:{home_error:.3f}: PASS",
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
                disable_position_torque(bus)
                torque_may_be_enabled = False
                torque_disabled_at_exit = True
                print("Follower torque disabled.", flush=True)
            except Exception as exc:
                cleanup_errors.append(
                    f"torque disable failed: {type(exc).__name__}: {exc}"
                )
                print(
                    "CRITICAL: software torque disable failed; cut follower 12 V now: "
                    f"{exc}",
                    file=sys.stderr,
                    flush=True,
                )
            try:
                disconnect_motor_bus(bus)
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
        if cleanup_errors and exit_code == 0:
            exit_code = 1
            failure = {
                "type": "CleanupError",
                "message": "; ".join(cleanup_errors),
                "traceback": None,
            }

    write_csv(csv_path, trace_rows)
    success = (
        exit_code == 0
        and failure is None
        and torque_disabled_at_exit
        and serial_closed
        and final_q is not None
    )
    decision = (
        "CONTROLLED_HOME_PASS_TORQUE_DISABLED_"
        "PREPARE_FIVE_COMMAND_COMMISSIONING_NEXT"
        if success
        else "CONTROLLED_HOME_FAILED_MOTION_AND_DEPLOYMENT_REMAIN_BLOCKED"
    )
    report = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "authorization": {
            "kind": "explicit_cli_controlled_home_motion",
            "confirmations": confirmations,
            "authorized_goal": "frozen dataset Home only",
            "policy_action_authorized": False,
            "autonomous_deployment_authorized": False,
        },
        "scope": {
            "model_loaded": False,
            "camera_opened": False,
            "follower_serial_connect_attempted": connect_attempted,
            "follower_serial_opened": connected,
            "operating_mode_read": operating_mode_read,
            "present_position_reads": present_read_count,
            "goal_position_writes": goal_write_count,
            "torque_enable_attempted": torque_enable_attempted,
            "torque_disabled_at_exit": torque_disabled_at_exit,
            "serial_closed": serial_closed,
            "action_api_called": False,
            "policy_inference_run": False,
            "autonomous_deployment_authorized": False,
        },
        "static_motion_boundary": static_boundary,
        "frozen_dependencies": {
            "hashes": hashes,
            "contract_and_calibration": frozen_inputs,
            "guard_self_audit": guard_audit,
            "controlled_home_self_audit": home_audit,
            "live_shadow_gate": shadow_gate,
            "follower_device": device,
            "constructed_follower": follower_summary,
            "imported_follower_modules": imported_modules,
        },
        "motion_contract": {
            "motor_order": list(MOTOR_ORDER),
            "control_hz": CONTROL_HZ,
            "step_limits": dict(zip(MOTOR_ORDER, HOME_STEP_LIMITS, strict=True)),
            "maximum_move_seconds": MAX_MOVE_SECONDS,
            "tracking_error_limits": dict(
                zip(MOTOR_ORDER, limits.tracking_error_limits, strict=True)
            ),
            "tracking_timeout_seconds": limits.tracking_timeout_seconds,
            "immediate_tracking_multiplier": IMMEDIATE_TRACKING_MULTIPLIER,
            "home_command": list(limits.home_command),
            "home_tolerance": HOME_TOLERANCE,
            "normal_and_abnormal_exit_disable_torque": True,
        },
        "run": {
            "expected_start_from_shadow": list(expected_start),
            "measured_start": None if start_q is None else list(start_q),
            "session_limits": (
                None
                if session_limits is None
                else {
                    motor: list(bounds)
                    for motor, bounds in zip(
                        MOTOR_ORDER, session_limits, strict=True
                    )
                }
            ),
            "trajectory_steps": trajectory_steps,
            "trajectory_duration_seconds": (
                None
                if move_started is None or move_finished is None
                else move_finished - move_started
            ),
            "trace_rows": len(trace_rows),
            "final_q_before_torque_disable": (
                None if final_q is None else list(final_q)
            ),
            "cleanup_errors": cleanup_errors,
        },
        "failure": failure,
        "decision": decision,
        "next_gate_authorized": False,
        "motion_authorized_after_process": False,
        "autonomous_deployment_authorized": False,
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
    print(f"torque disabled at exit: {torque_disabled_at_exit}", flush=True)
    print(f"serial closed: {serial_closed}", flush=True)
    print("POLICY ACTIONS AND AUTONOMOUS DEPLOYMENT REMAIN BLOCKED.", flush=True)
    print("\n===== OUTPUT =====", flush=True)
    print(report_path, flush=True)
    if trace_rows:
        print(csv_path, flush=True)
    print(
        "ACT V3 CONTROLLED HOME GATE: PASS"
        if success
        else "ACT V3 CONTROLLED HOME GATE: FAIL",
        flush=True,
    )
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(
            f"ACT V3 CONTROLLED HOME GATE: FAIL-CLOSED: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        print("NO MOTION IS AUTHORIZED BY THIS FAILED PROCESS.", file=sys.stderr)
        raise SystemExit(1)
