#!/usr/bin/env python3
"""Guide a manual SO101 pose reset using read-only joint feedback.

This finite commissioning utility reuses the frozen ACT V3 read-only sensor
adapter.  Its only live transport lifecycle is delegated to that adapter:

* open the reviewed follower motor bus;
* repeatedly read normalized ``Present_Position`` values;
* close the bus with ``disable_torque=False``.

The utility never calls a robot connect/configure path, never changes torque,
never writes a register or Goal Position, never opens a camera, and never loads
or runs a policy.  It compares each complete six-joint read with the preserved
controlled-Home trace.  A PASS requires the frozen recovery gate to remain
true for five consecutive seconds.  Passing this tool does not itself
authorize motion; it only provides a fresh pose-alignment artifact for the
separately authorized one-controlled-episode gate.
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence


EXPECTED_ADAPTER_SHA256 = (
    "516749f47ccd6e6731760aa46c304936ba9e8cbaacc2e7f4baf4b4f101078d67"
)
EXPECTED_ONE_SHOT_SHA256 = (
    "c2f2be0481e72aa1836bae36e873be59b9c739b776d1baf5b510954264b9196b"
)
EXPECTED_GUARD_SHA256 = (
    "e06e785071ee3ded95e89d670978bcb4a8caf104b2fc69fc87de37cb6fff6d15"
)
EXPECTED_ASSEMBLER_SHA256 = (
    "866a53d8b4ce1473344779c493a955d03dafa485115fab34a4ffc929ff31d424"
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
EXPECTED_ADAPTER_DECISION = (
    "READ_ONLY_LIVE_SENSOR_ADAPTER_INERT_PASS_"
    "PREPARE_EXPLICIT_ONE_SHOT_LIVE_READ_NEXT"
)
EXPECTED_HOME_DECISION = (
    "CONTROLLED_HOME_FAILED_MOTION_AND_DEPLOYMENT_REMAIN_BLOCKED"
)
MOTOR_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
ROBOT_ID = "follower_white"
SAMPLE_HZ = 5.0
MAX_DURATION_SECONDS = 120.0
REQUIRED_STABLE_SAMPLES = 25
STABILITY_DELTA_DEGREES = 0.5
RECOVERY_ENVELOPE_TOLERANCE = 1.0
RECOVERY_TRACE_TOLERANCE = 5.0
MAX_READ_DURATION_SECONDS = 0.100


class PoseAlignmentError(RuntimeError):
    """A frozen dependency, read-only lifecycle, or alignment gate failed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--adapter-report", type=Path, required=True)
    parser.add_argument("--one-shot-launcher", type=Path, required=True)
    parser.add_argument("--guard-core", type=Path, required=True)
    parser.add_argument("--assembler", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--controlled-home-report", type=Path, required=True)
    parser.add_argument("--controlled-home-trace", type=Path, required=True)
    parser.add_argument("--follower-source", type=Path, required=True)
    parser.add_argument("--follower-config-source", type=Path, required=True)
    parser.add_argument("--follower-port", type=Path, required=True)
    parser.add_argument("--robot-id", default=ROBOT_ID)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--authorize-continuous-read-only-hardware",
        action="store_true",
        help="Authorize finite repeated Present_Position reads only.",
    )
    parser.add_argument(
        "--confirm-torque-disabled-and-no-command-process",
        action="store_true",
        help=(
            "Confirm follower torque is already disabled and no command-capable "
            "robot process is running."
        ),
    )
    parser.add_argument(
        "--confirm-arm-supported-for-manual-alignment",
        action="store_true",
        help=(
            "Confirm a stable external support is present and the operator will "
            "stop on resistance or unexpected motion."
        ),
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
        raise PoseAlignmentError(
            f"{label} SHA256 mismatch: expected={expected}, actual={actual}"
        )
    return actual


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PoseAlignmentError(f"invalid JSON: {path}: {exc}") from exc


def load_module(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise PoseAlignmentError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def finite_vector(label: str, values: Any) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != len(MOTOR_ORDER):
        raise PoseAlignmentError(f"{label} must be a six-value vector")
    result = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in result):
        raise PoseAlignmentError(f"{label} contains nonfinite values")
    return result


def verify_static_read_only_boundary(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    attributes: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
        elif isinstance(node, ast.Attribute):
            attributes.add(node.attr)
    forbidden_imports = sorted(
        module
        for module in imports
        if module.startswith(
            ("lerobot", "serial", "pyserial", "cv2", "torch", "scservo_sdk")
        )
    )
    forbidden_attributes = sorted(
        attributes.intersection(
            {
                "connect",
                "disconnect",
                "sync_read",
                "sync_write",
                "write",
                "send_action",
                "configure",
                "setup_motors",
                "enable_torque",
                "disable_torque",
                "write_calibration",
                "get_observation",
            }
        )
    )
    if forbidden_imports or forbidden_attributes:
        raise PoseAlignmentError(
            "pose-alignment source exceeded its delegated read-only boundary: "
            f"imports={forbidden_imports}, attributes={forbidden_attributes}"
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "forbidden_imports": forbidden_imports,
        "direct_bus_or_robot_calls": forbidden_attributes,
        "live_calls_delegated_to_frozen_adapter": ["open", "read", "close"],
        "pass": True,
    }


def verify_home_evidence(
    report_path: Path,
    trace_path: Path,
    expected_home: Sequence[float],
) -> dict[str, Any]:
    report = load_json(report_path)
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise PoseAlignmentError("unexpected controlled-Home report schema")
    if report.get("decision") != EXPECTED_HOME_DECISION:
        raise PoseAlignmentError("unexpected controlled-Home decision")
    motion = report.get("motion_contract") or {}
    run = report.get("run") or {}
    scope = report.get("scope") or {}
    failure = report.get("failure") or {}
    static = report.get("static_motion_boundary") or {}
    if tuple(motion.get("motor_order", ())) != MOTOR_ORDER:
        raise PoseAlignmentError("controlled-Home motor order mismatch")
    if static.get("sha256") != EXPECTED_CONTROLLED_HOME_SHA256:
        raise PoseAlignmentError("controlled-Home source identity mismatch")
    if failure.get("type") != "ControlledHomeError" or not str(
        failure.get("message", "")
    ).startswith("final Home tolerance failed:"):
        raise PoseAlignmentError("controlled-Home failure was not precision-only")
    if scope.get("torque_disabled_at_exit") is not True:
        raise PoseAlignmentError("controlled-Home report does not prove torque cleanup")
    if scope.get("serial_closed") is not True:
        raise PoseAlignmentError("controlled-Home report does not prove serial cleanup")
    reviewed_start = finite_vector("reviewed start", run.get("measured_start"))
    home_command = finite_vector("reviewed Home", motion.get("home_command"))
    if tuple(float(value) for value in expected_home) != home_command:
        raise PoseAlignmentError("reviewed Home differs from frozen runtime Home")

    with trace_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len(rows) != int(run.get("trace_rows", -1)):
        raise PoseAlignmentError("controlled-Home trace count mismatch")
    trace_poses: list[dict[str, Any]] = []
    for row in rows:
        if int(row.get("tracking_tripped", -1)) != 0:
            raise PoseAlignmentError("controlled-Home trace contains a tracking trip")
        trace_poses.append(
            {
                "phase": row.get("phase"),
                "sequence": int(row["sequence"]),
                "actual_q": finite_vector(
                    "controlled-Home trace pose",
                    [row[f"actual_{motor}"] for motor in MOTOR_ORDER],
                ),
            }
        )
    return {
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "trace_path": str(trace_path),
        "trace_sha256": sha256_file(trace_path),
        "reviewed_start": list(reviewed_start),
        "home_command": list(home_command),
        "trace_poses": trace_poses,
        "trace_rows": len(trace_poses),
        "torque_disabled_at_exit": True,
        "serial_closed": True,
        "pass": True,
    }


def recovery_pose_status(
    pose: Sequence[float],
    *,
    reviewed_start: Sequence[float],
    home_command: Sequence[float],
    trace_poses: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    vector = finite_vector("live pose", list(pose))
    start = finite_vector("reviewed start", list(reviewed_start))
    home = finite_vector("Home command", list(home_command))
    envelope_excess = tuple(
        max(min(source, target) - value, value - max(source, target), 0.0)
        for value, source, target in zip(vector, start, home, strict=True)
    )
    nearest = min(
        trace_poses,
        key=lambda item: max(
            abs(value - reference)
            for value, reference in zip(vector, item["actual_q"], strict=True)
        ),
    )
    nearest_errors = tuple(
        abs(value - reference)
        for value, reference in zip(vector, nearest["actual_q"], strict=True)
    )
    return {
        "pose": list(vector),
        "target_delta_to_reviewed_start": {
            motor: target - value
            for motor, target, value in zip(MOTOR_ORDER, start, vector, strict=True)
        },
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


@dataclass
class StabilityGate:
    required_samples: int = REQUIRED_STABLE_SAMPLES
    maximum_sample_delta: float = STABILITY_DELTA_DEGREES
    consecutive: int = 0
    previous_pose: tuple[float, ...] | None = None

    def update(self, status: dict[str, Any]) -> bool:
        pose = finite_vector("stability pose", status["pose"])
        stable = (
            self.previous_pose is None
            or max(
                abs(value - previous)
                for value, previous in zip(pose, self.previous_pose, strict=True)
            )
            <= self.maximum_sample_delta
        )
        if status["pass"] and stable:
            self.consecutive += 1
        elif status["pass"]:
            self.consecutive = 1
        else:
            self.consecutive = 0
        self.previous_pose = pose
        return self.consecutive >= self.required_samples


def run_pure_self_audit() -> dict[str, Any]:
    start = (2.0, -95.0, 99.0, 69.0, 0.0, 6.0)
    home = (13.0, -32.0, 48.0, 43.0, 0.4, 28.0)
    middle = tuple((a + b) / 2.0 for a, b in zip(start, home, strict=True))
    trace = [
        {"phase": "move", "sequence": 1, "actual_q": start},
        {"phase": "move", "sequence": 2, "actual_q": middle},
        {"phase": "hold", "sequence": 1, "actual_q": home},
    ]
    exact = recovery_pose_status(
        start, reviewed_start=start, home_command=home, trace_poses=trace
    )
    interior = recovery_pose_status(
        middle, reviewed_start=start, home_command=home, trace_poses=trace
    )
    outside = list(start)
    outside[0] -= 1.01
    outside_status = recovery_pose_status(
        outside, reviewed_start=start, home_command=home, trace_poses=trace
    )
    hybrid = list(middle)
    hybrid[3] = start[3]
    hybrid_status = recovery_pose_status(
        hybrid, reviewed_start=start, home_command=home, trace_poses=trace
    )
    stable_gate = StabilityGate(required_samples=3, maximum_sample_delta=0.5)
    stable_results = [stable_gate.update(exact) for _ in range(3)]
    reset_gate = StabilityGate(required_samples=2, maximum_sample_delta=0.5)
    reset_gate.update(exact)
    reset_gate.update(outside_status)
    checks = {
        "exact_trace_pose_passes": exact["pass"] is True,
        "recorded_interior_pose_passes": interior["pass"] is True,
        "envelope_excess_fails": outside_status["pass"] is False,
        "hybrid_pose_requires_trace_proximity": hybrid_status["pass"] is False,
        "stable_pass_requires_consecutive_samples": stable_results == [False, False, True],
        "failed_sample_resets_stability": reset_gate.consecutive == 0,
    }
    if not all(checks.values()):
        raise PoseAlignmentError(f"pure self-audit failed: {checks}")
    return {
        "tests_total": len(checks),
        "tests_passed": sum(checks.values()),
        "checks": checks,
        "pass": True,
    }


def print_live_status(
    sequence: int,
    status: dict[str, Any],
    stability: StabilityGate,
) -> None:
    delta = status["target_delta_to_reviewed_start"]
    print(
        f"sample={sequence:04d} gate={'PASS' if status['pass'] else 'MOVE'} "
        f"stable={stability.consecutive:02d}/{stability.required_samples} "
        f"env={status['max_envelope_excess']:.2f} "
        f"trace={status['nearest_trace_max_error']:.2f} | "
        f"pan={delta['shoulder_pan']:+.2f} "
        f"lift={delta['shoulder_lift']:+.2f} "
        f"elbow={delta['elbow_flex']:+.2f} "
        f"wrist={delta['wrist_flex']:+.2f} "
        f"roll={delta['wrist_roll']:+.2f} "
        f"grip={delta['gripper']:+.2f}",
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    confirmations = {
        "continuous_read_only_hardware": args.authorize_continuous_read_only_hardware,
        "torque_disabled_and_no_command_process": (
            args.confirm_torque_disabled_and_no_command_process
        ),
        "arm_supported_for_manual_alignment": (
            args.confirm_arm_supported_for_manual_alignment
        ),
    }
    missing = [name for name, present in confirmations.items() if not present]
    if missing:
        raise PoseAlignmentError(f"required confirmations missing: {missing}")
    if args.robot_id != ROBOT_ID:
        raise PoseAlignmentError(f"robot-id must be exactly {ROBOT_ID!r}")

    script_path = Path(__file__).resolve()
    adapter_path = require_file(args.adapter)
    adapter_report_path = require_file(args.adapter_report)
    one_shot_path = require_file(args.one_shot_launcher)
    guard_path = require_file(args.guard_core)
    assembler_path = require_file(args.assembler)
    contract_path = require_file(args.contract)
    calibration_path = require_file(args.calibration)
    home_report_path = require_file(args.controlled_home_report)
    home_trace_path = require_file(args.controlled_home_trace)
    follower_source = require_file(args.follower_source)
    follower_config_source = require_file(args.follower_config_source)
    follower_port = args.follower_port.expanduser()
    output_path = args.output_json.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {output_path}")

    print("===== EXPLICIT READ-ONLY MANUAL ALIGNMENT AUTHORIZATION =====", flush=True)
    print("finite 120s / 5Hz Present_Position reads only", flush=True)
    print("no torque change, Goal Position, action, camera, model or inference", flush=True)
    print("unexpected motion or resistance: stop and cut follower 12 V", flush=True)
    print("passing this process does not authorize motion", flush=True)

    print("\n===== VERIFY FROZEN READ-ONLY STACK =====", flush=True)
    static_boundary = verify_static_read_only_boundary(script_path)
    self_audit = run_pure_self_audit()
    hashes = {
        "adapter": require_sha(adapter_path, EXPECTED_ADAPTER_SHA256, "adapter"),
        "one_shot_launcher": require_sha(
            one_shot_path, EXPECTED_ONE_SHOT_SHA256, "one-shot launcher"
        ),
        "guard": require_sha(guard_path, EXPECTED_GUARD_SHA256, "guard"),
        "assembler": require_sha(
            assembler_path, EXPECTED_ASSEMBLER_SHA256, "assembler"
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
    one_shot = load_module(one_shot_path, "act_v3_alignment_one_shot")
    adapter = load_module(adapter_path, "act_v3_alignment_adapter")
    guard = load_module(guard_path, "act_v3_alignment_guard")
    assembler = load_module(assembler_path, "act_v3_alignment_assembler")
    adapter_gate = one_shot.verify_adapter_gate(adapter_path, adapter_report_path)
    frozen_inputs = guard.verify_frozen_inputs(contract_path, calibration_path)
    limits = guard.RuntimeLimits.frozen_v3()
    home_evidence = verify_home_evidence(
        home_report_path, home_trace_path, limits.home_command
    )
    print(
        f"static boundary PASS; adapter 17/17; alignment audit "
        f"{self_audit['tests_passed']}/{self_audit['tests_total']}: PASS",
        flush=True,
    )
    print("controlled-Home report and complete six-joint trace: PASS", flush=True)

    print("\n===== VERIFY FOLLOWER AND CONSTRUCT INERT BUS =====", flush=True)
    device = one_shot.verify_follower_device(follower_port)
    if device.get("stable_path") != EXPECTED_FOLLOWER_BY_ID:
        raise PoseAlignmentError("unexpected follower by-id identity")
    follower_class, config_class, imported_modules = one_shot.import_frozen_follower(
        follower_source, follower_config_source
    )
    config = config_class(
        port=str(follower_port),
        id=ROBOT_ID,
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

    delta_guard = guard.DeltaCommandGuard(limits, limits.home_command)
    sensor = adapter.ReadOnlyJointSensorAdapter(
        bus=follower.bus,
        validate_positions=delta_guard.validate_measured_positions,
        packet_factory=assembler.JointStatePacket,
        monotonic=time.monotonic,
        max_read_duration_seconds=MAX_READ_DURATION_SECONDS,
    )
    stability = StabilityGate()
    started: float | None = None
    finished: float | None = None
    last_status: dict[str, Any] | None = None
    accepted = False
    failure: dict[str, Any] | None = None
    cleanup_error: dict[str, Any] | None = None
    serial_closed = False
    read_durations: list[float] = []

    print("\n===== LIVE READ-ONLY ALIGNMENT =====", flush=True)
    print(
        "delta signs mean target minus actual; move gently toward zero. "
        "A PASS needs 25 stable samples (5 seconds).",
        flush=True,
    )
    try:
        sensor.open()
        started = time.monotonic()
        next_tick = started
        while time.monotonic() - started < MAX_DURATION_SECONDS:
            result = sensor.read()
            read_durations.append(float(result.metrics.read_duration_ms))
            pose = tuple(float(value) for value in result.packet.actual_q)
            last_status = recovery_pose_status(
                pose,
                reviewed_start=home_evidence["reviewed_start"],
                home_command=home_evidence["home_command"],
                trace_poses=home_evidence["trace_poses"],
            )
            accepted = stability.update(last_status)
            if result.metrics.sequence == 1 or result.metrics.sequence % 5 == 0 or accepted:
                print_live_status(result.metrics.sequence, last_status, stability)
            if accepted:
                break
            next_tick += 1.0 / SAMPLE_HZ
            delay = next_tick - time.monotonic()
            if delay > 0.0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()
        finished = time.monotonic()
        if not accepted:
            failure = {
                "type": "PoseAlignmentTimeout",
                "message": "reviewed trajectory gate was not stable for five seconds",
                "traceback": None,
            }
    except KeyboardInterrupt:
        finished = time.monotonic()
        failure = {
            "type": "KeyboardInterrupt",
            "message": "operator cancelled read-only alignment",
            "traceback": traceback.format_exc(),
        }
    except Exception as exc:
        finished = time.monotonic()
        failure = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        try:
            sensor.close()
            serial_closed = not bool(getattr(follower.bus, "is_connected", True))
        except Exception as exc:
            cleanup_error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            serial_closed = False

    success = accepted and failure is None and cleanup_error is None and serial_closed
    decision = (
        "READ_ONLY_POSE_ALIGNMENT_PASS_RUN_ONE_CONTROLLED_EPISODE_NEXT"
        if success
        else "READ_ONLY_POSE_ALIGNMENT_FAILED_NO_MOTION_AUTHORIZED"
    )
    report = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "authorization": {
            "confirmations": confirmations,
            "maximum_duration_seconds": MAX_DURATION_SECONDS,
            "sample_hz": SAMPLE_HZ,
            "motion_authorized": False,
            "autonomous_deployment_authorized": False,
        },
        "scope": {
            "serial_opened": started is not None,
            "successful_position_reads": sensor.sequence,
            "serial_closed": serial_closed,
            "torque_api_called": False,
            "register_write_called": False,
            "goal_position_written": False,
            "action_api_called": False,
            "camera_opened": False,
            "model_loaded": False,
            "policy_inference_run": False,
            "motion_authorized": False,
        },
        "static_read_only_boundary": static_boundary,
        "pure_self_audit": self_audit,
        "frozen_dependencies": {
            "hashes": hashes,
            "adapter_gate": adapter_gate,
            "contract_and_calibration": frozen_inputs,
            "controlled_home_evidence": {
                key: value
                for key, value in home_evidence.items()
                if key != "trace_poses"
            },
            "device": device,
            "constructed_follower": follower_summary,
            "imported_modules": imported_modules,
        },
        "alignment_contract": {
            "motor_order": list(MOTOR_ORDER),
            "reviewed_start": home_evidence["reviewed_start"],
            "home_command": home_evidence["home_command"],
            "recovery_envelope_tolerance": RECOVERY_ENVELOPE_TOLERANCE,
            "recovery_trace_tolerance": RECOVERY_TRACE_TOLERANCE,
            "required_stable_samples": REQUIRED_STABLE_SAMPLES,
            "stability_seconds": REQUIRED_STABLE_SAMPLES / SAMPLE_HZ,
            "maximum_inter_sample_delta": STABILITY_DELTA_DEGREES,
        },
        "run": {
            "elapsed_seconds": (
                None
                if started is None or finished is None
                else finished - started
            ),
            "successful_reads": sensor.sequence,
            "stable_samples": stability.consecutive,
            "last_pose_status": last_status,
            "read_duration_ms_max": max(read_durations) if read_durations else None,
            "cleanup_error": cleanup_error,
        },
        "failure": failure,
        "decision": decision,
        "alignment_passed": success,
        "motion_authorized": False,
        "autonomous_deployment_authorized": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
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
    if cleanup_error is not None:
        print(
            f"cleanup={cleanup_error['type']}: {cleanup_error['message']}",
            file=sys.stderr,
            flush=True,
        )
    print(f"successful reads: {sensor.sequence}", flush=True)
    print(f"serial closed: {serial_closed}", flush=True)
    print("NO MOTOR WRITE, TORQUE CHANGE, CAMERA OR POLICY CALL OCCURRED.", flush=True)
    print("PASS DOES NOT ITSELF AUTHORIZE MOTION.", flush=True)
    print("\n===== OUTPUT =====", flush=True)
    print(output_path, flush=True)
    print(
        "ACT V3 READ-ONLY POSE ALIGNMENT: PASS"
        if success
        else "ACT V3 READ-ONLY POSE ALIGNMENT: FAIL",
        flush=True,
    )
    return 0 if success else (130 if failure and failure["type"] == "KeyboardInterrupt" else 2)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(
            f"ACT V3 READ-ONLY POSE ALIGNMENT: FAIL-CLOSED: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        print("NO MOTION IS AUTHORIZED BY THIS FAILED PROCESS.", file=sys.stderr)
        raise SystemExit(1)
