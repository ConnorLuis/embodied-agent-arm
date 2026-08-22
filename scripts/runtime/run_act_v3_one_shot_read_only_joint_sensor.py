#!/usr/bin/env python3
"""Perform one explicitly authorized, non-commanding SO101 joint read.

This is the first live-hardware gate after the frozen ACT V3 offline and inert
audits.  The launcher constructs the reviewed SO101 follower object without
calling ``SO101Follower.connect()``.  It delegates to the frozen read-only
adapter, whose only live bus sequence is:

    bus.connect()
    bus.sync_read("Present_Position", normalize=True, num_retry=3)
    bus.disconnect(disable_torque=False)

Exactly one successful six-joint read is allowed.  No camera is configured, no
model is loaded, no policy inference runs, and no calibration, configuration,
torque, register-write, Goal_Position, or action API is called.  The motor bus
does transmit a READ instruction; therefore this script requires the explicit
``--authorize-one-shot-read-only-hardware`` acknowledgement.

Passing this one-shot read does not authorize motion or deployment.  It only
unblocks a subsequent live shadow-mode observation/inference run whose policy
outputs remain suppressed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.util
import json
import math
import os
import stat
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


EXPECTED_ADAPTER_SHA256 = (
    "516749f47ccd6e6731760aa46c304936ba9e8cbaacc2e7f4baf4b4f101078d67"
)
EXPECTED_ADAPTER_DECISION = (
    "READ_ONLY_LIVE_SENSOR_ADAPTER_INERT_PASS_"
    "PREPARE_EXPLICIT_ONE_SHOT_LIVE_READ_NEXT"
)
EXPECTED_GUARD_SHA256 = (
    "e06e785071ee3ded95e89d670978bcb4a8caf104b2fc69fc87de37cb6fff6d15"
)
EXPECTED_ASSEMBLER_SHA256 = (
    "866a53d8b4ce1473344779c493a955d03dafa485115fab34a4ffc929ff31d424"
)
EXPECTED_FOLLOWER_SHA256 = (
    "26c675c71ade2670fa1bed2d887507b6ef3ad53a49e20182f5ba3d032a0afd7a"
)
EXPECTED_FOLLOWER_CONFIG_SHA256 = (
    "e5702d9b6c4a09d10de83f912a1640169698292d5145091d403f4fdce0211cbe"
)
EXPECTED_FOLLOWER_BY_ID_BASENAME = (
    "usb-1a86_USB_Single_Serial_5C82110810-if00"
)
EXPECTED_ROBOT_ID = "follower_white"
MOTOR_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
CALIBRATION_FIELDS = (
    "id",
    "drive_mode",
    "homing_offset",
    "range_min",
    "range_max",
)
DEFAULT_MAX_READ_DURATION_SECONDS = 0.100


class OneShotReadError(RuntimeError):
    """The one-shot authorization, frozen inputs, device, or read failed."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--adapter-report", type=Path, required=True)
    parser.add_argument("--guard-core", type=Path, required=True)
    parser.add_argument("--assembler", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--follower-source", type=Path, required=True)
    parser.add_argument("--follower-config-source", type=Path, required=True)
    parser.add_argument("--follower-port", type=Path, required=True)
    parser.add_argument("--robot-id", default=EXPECTED_ROBOT_ID)
    parser.add_argument(
        "--max-read-duration-seconds",
        type=float,
        default=DEFAULT_MAX_READ_DURATION_SECONDS,
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--authorize-one-shot-read-only-hardware",
        action="store_true",
        help=(
            "Explicitly authorize opening only the reviewed follower serial port, "
            "transmitting one Present_Position READ, and closing without torque change."
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


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OneShotReadError(f"invalid JSON: {path}: {exc}") from exc


def load_module(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise OneShotReadError(f"cannot load module spec: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def finite_positive(label: str, value: Any) -> float:
    if isinstance(value, bool):
        raise OneShotReadError(f"{label} must not be bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise OneShotReadError(f"{label} must be numeric") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise OneShotReadError(f"{label} must be finite and positive")
    return result


def verify_launcher_static_boundary(path: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    attributes: set[str] = set()
    calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            attributes.add(node.attr)
        elif isinstance(node, ast.Call):
            calls.append(ast.unparse(node))
    forbidden_attributes = sorted(
        {
            "send_action",
            "get_observation",
            "sync_write",
            "write",
            "write_calibration",
            "enable_torque",
            "disable_torque",
            "torque_disabled",
            "configure",
            "configure_motors",
            "calibrate",
            "setup_motors",
            "predict_action",
            "predict_action_chunk",
            "from_pretrained",
            "VideoCapture",
        }.intersection(attributes)
    )
    forbidden_full_follower_calls = sorted(
        expression
        for expression in calls
        if expression.startswith(("follower.connect(", "follower.disconnect("))
    )
    if forbidden_attributes or forbidden_full_follower_calls:
        raise OneShotReadError(
            "launcher contains forbidden hardware/model calls: "
            f"attributes={forbidden_attributes}, "
            f"full_follower_calls={forbidden_full_follower_calls}"
        )
    return {
        "source": str(path),
        "sha256": sha256_file(path),
        "full_follower_connect_or_disconnect_calls": False,
        "motor_register_write_calls": False,
        "torque_calls": False,
        "camera_calls": False,
        "model_or_inference_calls": False,
        "pass": True,
    }


def verify_adapter_gate(adapter_path: Path, report_path: Path) -> dict[str, Any]:
    adapter_sha = sha256_file(adapter_path)
    if adapter_sha != EXPECTED_ADAPTER_SHA256:
        raise OneShotReadError(
            f"adapter SHA256 mismatch: expected {EXPECTED_ADAPTER_SHA256}, got {adapter_sha}"
        )
    report = load_json(report_path)
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise OneShotReadError("unexpected inert adapter report schema")
    if report.get("decision") != EXPECTED_ADAPTER_DECISION:
        raise OneShotReadError("inert adapter report decision mismatch")
    self_audit = report.get("self_audit")
    if not isinstance(self_audit, dict):
        raise OneShotReadError("inert adapter self_audit missing")
    passed = self_audit.get("tests_passed")
    total = self_audit.get("tests_total")
    if passed != 17 or total != 17:
        raise OneShotReadError(f"inert adapter tests are not 17/17: {passed}/{total}")
    if report.get("live_read_authorized") is not False:
        raise OneShotReadError("inert adapter report unexpectedly authorized live read")
    if report.get("hardware_deployment_authorized") is not False:
        raise OneShotReadError("inert adapter report unexpectedly authorized deployment")
    contract = report.get("adapter_contract")
    expected_contract = {
        "motor_order": list(MOTOR_ORDER),
        "read_register": "Present_Position",
        "normalize": True,
        "num_retry": 3,
        "disconnect_disable_torque": False,
        "transactional_sequence_commit": True,
    }
    if not isinstance(contract, dict):
        raise OneShotReadError("inert adapter contract missing")
    mismatches = {
        key: {"expected": expected, "actual": contract.get(key)}
        for key, expected in expected_contract.items()
        if contract.get(key) != expected
    }
    if mismatches:
        raise OneShotReadError(f"inert adapter contract mismatch: {mismatches}")
    return {
        "adapter_path": str(adapter_path),
        "adapter_sha256": adapter_sha,
        "adapter_report": str(report_path),
        "adapter_decision": report["decision"],
        "inert_tests": "17/17",
    }


def verify_source_identity(path: Path, expected_sha: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected_sha:
        raise OneShotReadError(
            f"{label} SHA256 mismatch: expected {expected_sha}, got {actual}"
        )
    return actual


def verify_follower_device(path: Path) -> dict[str, Any]:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        raise OneShotReadError("follower port must be an absolute /dev/serial/by-id path")
    if expanded.parent != Path("/dev/serial/by-id"):
        raise OneShotReadError("follower port must be under /dev/serial/by-id")
    if expanded.name != EXPECTED_FOLLOWER_BY_ID_BASENAME:
        raise OneShotReadError(
            "unexpected follower serial identity: "
            f"expected {EXPECTED_FOLLOWER_BY_ID_BASENAME}, got {expanded.name}"
        )
    if not expanded.is_symlink():
        raise OneShotReadError(f"stable follower symlink is missing: {expanded}")
    try:
        resolved = expanded.resolve(strict=True)
    except FileNotFoundError as exc:
        raise OneShotReadError(f"follower serial symlink is broken: {expanded}") from exc
    mode = resolved.stat().st_mode
    if not stat.S_ISCHR(mode):
        raise OneShotReadError(f"follower serial target is not a character device: {resolved}")
    if not os.access(resolved, os.R_OK | os.W_OK):
        raise OneShotReadError(
            f"current user lacks read/write permission for follower serial target: {resolved}"
        )
    return {
        "stable_path": str(expanded),
        "stable_basename": expanded.name,
        "resolved_character_device": str(resolved),
        "read_write_permission": True,
    }


def normalize_calibration_row(row: Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return {field: row.get(field) for field in CALIBRATION_FIELDS}
    return {field: getattr(row, field, None) for field in CALIBRATION_FIELDS}


def verify_constructed_follower(
    follower: Any,
    *,
    expected_calibration: Mapping[str, Any],
    expected_port: str,
) -> dict[str, Any]:
    cameras = getattr(follower, "cameras", None)
    if not isinstance(cameras, Mapping) or cameras:
        raise OneShotReadError("one-shot follower must have cameras={}")
    bus = getattr(follower, "bus", None)
    if bus is None:
        raise OneShotReadError("constructed follower has no motor bus")
    if bool(getattr(bus, "is_connected", False)):
        raise OneShotReadError("follower bus unexpectedly connected during construction")
    if str(getattr(bus, "port", "")) != expected_port:
        raise OneShotReadError(
            f"constructed bus port mismatch: {getattr(bus, 'port', None)!r}"
        )
    motors = getattr(bus, "motors", None)
    if not isinstance(motors, Mapping) or tuple(motors) != MOTOR_ORDER:
        raise OneShotReadError(
            f"constructed bus motor order mismatch: {tuple(motors) if isinstance(motors, Mapping) else motors}"
        )
    actual_calibration = getattr(bus, "calibration", None)
    if not isinstance(actual_calibration, Mapping):
        raise OneShotReadError("constructed bus calibration is missing")
    mismatches: list[dict[str, Any]] = []
    normalized: dict[str, dict[str, Any]] = {}
    for motor in MOTOR_ORDER:
        expected_row = expected_calibration.get(motor)
        actual_row = actual_calibration.get(motor)
        expected_normalized = normalize_calibration_row(expected_row)
        actual_normalized = normalize_calibration_row(actual_row)
        normalized[motor] = actual_normalized
        if actual_normalized != expected_normalized:
            mismatches.append(
                {
                    "motor": motor,
                    "expected": expected_normalized,
                    "actual": actual_normalized,
                }
            )
    if mismatches:
        raise OneShotReadError(
            f"constructed follower calibration differs from frozen values: {mismatches}"
        )
    return {
        "cameras": [],
        "bus_port": expected_port,
        "bus_initially_connected": False,
        "motor_order": list(MOTOR_ORDER),
        "calibration": normalized,
        "calibration_matches_frozen": True,
    }


def import_frozen_follower(
    follower_source: Path,
    config_source: Path,
) -> tuple[type, type, dict[str, str]]:
    follower_module = importlib.import_module(
        "lerobot.robots.so101_follower.so101_follower"
    )
    config_module = importlib.import_module(
        "lerobot.robots.so101_follower.config_so101_follower"
    )
    follower_module_path = Path(follower_module.__file__).resolve()
    config_module_path = Path(config_module.__file__).resolve()
    if follower_module_path != follower_source:
        raise OneShotReadError(
            f"imported follower module path mismatch: {follower_module_path}"
        )
    if config_module_path != config_source:
        raise OneShotReadError(
            f"imported follower config module path mismatch: {config_module_path}"
        )
    follower_class = getattr(follower_module, "SO101Follower", None)
    config_class = getattr(config_module, "SO101FollowerConfig", None)
    if not isinstance(follower_class, type) or not isinstance(config_class, type):
        raise OneShotReadError("frozen SO101 follower classes were not found")
    return follower_class, config_class, {
        "follower_module": str(follower_module_path),
        "config_module": str(config_module_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.authorize_one_shot_read_only_hardware:
        raise OneShotReadError(
            "live read not authorized: add --authorize-one-shot-read-only-hardware "
            "only when the follower is stable and you intend one non-commanding read"
        )
    if args.robot_id != EXPECTED_ROBOT_ID:
        raise OneShotReadError(
            f"robot-id must be exactly {EXPECTED_ROBOT_ID!r}, got {args.robot_id!r}"
        )
    max_read_duration = finite_positive(
        "max_read_duration_seconds", args.max_read_duration_seconds
    )
    if abs(max_read_duration - DEFAULT_MAX_READ_DURATION_SECONDS) > 1e-12:
        raise OneShotReadError(
            "max-read-duration-seconds is frozen at "
            f"{DEFAULT_MAX_READ_DURATION_SECONDS:.3f} for this gate"
        )

    adapter_path = require_file(args.adapter)
    adapter_report_path = require_file(args.adapter_report)
    guard_path = require_file(args.guard_core)
    assembler_path = require_file(args.assembler)
    contract_path = require_file(args.contract)
    calibration_path = require_file(args.calibration)
    follower_source = require_file(args.follower_source)
    config_source = require_file(args.follower_config_source)
    output_path = args.output_json.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {output_path}")

    print("===== EXPLICIT ONE-SHOT READ-ONLY AUTHORIZATION =====")
    print("authorization flag: PRESENT")
    print("allowed: open reviewed follower serial; one Present_Position READ; close(false)")
    print("forbidden: calibration/configuration/torque/write/Goal_Position/action/camera/model")
    print("hardware deployment authorization: false")

    print("\n===== VERIFY FROZEN LIVE-READ GATES =====")
    static_boundary = verify_launcher_static_boundary(Path(__file__).resolve())
    adapter_gate = verify_adapter_gate(adapter_path, adapter_report_path)
    guard_sha = verify_source_identity(guard_path, EXPECTED_GUARD_SHA256, "guard")
    assembler_sha = verify_source_identity(
        assembler_path, EXPECTED_ASSEMBLER_SHA256, "assembler"
    )
    follower_sha = verify_source_identity(
        follower_source, EXPECTED_FOLLOWER_SHA256, "follower source"
    )
    config_sha = verify_source_identity(
        config_source, EXPECTED_FOLLOWER_CONFIG_SHA256, "follower config source"
    )
    guard_module = load_module(guard_path, "act_v3_one_shot_guard")
    assembler_module = load_module(assembler_path, "act_v3_one_shot_assembler")
    adapter_module = load_module(adapter_path, "act_v3_one_shot_adapter")
    frozen_inputs = guard_module.verify_frozen_inputs(contract_path, calibration_path)
    print("launcher boundary, adapter 17/17, guard, assembler and vendor sources: PASS")
    print("runtime contract and follower calibration identities: PASS")

    print("\n===== VERIFY EXACT FOLLOWER SERIAL DEVICE =====")
    device = verify_follower_device(args.follower_port)
    print(f"stable identity: {device['stable_path']}")
    print(f"resolved device: {device['resolved_character_device']}")
    print("character device and current-user read/write permission: PASS")

    print("\n===== CONSTRUCT FOLLOWER WITHOUT CONNECT/CONFIGURE/CAMERAS =====")
    follower_class, config_class, imported_modules = import_frozen_follower(
        follower_source,
        config_source,
    )
    config = config_class(
        port=str(args.follower_port.expanduser()),
        id=args.robot_id,
        cameras={},
        disable_torque_on_disconnect=False,
        use_degrees=False,
    )
    follower = follower_class(config)
    follower_summary = verify_constructed_follower(
        follower,
        expected_calibration=guard_module.EXPECTED_FOLLOWER_CALIBRATION,
        expected_port=str(args.follower_port.expanduser()),
    )
    print("robot object constructed inertly; motor order and frozen calibration: PASS")
    print("follower bus is disconnected; cameras={}: PASS")

    limits = guard_module.RuntimeLimits.frozen_v3()
    delta_guard = guard_module.DeltaCommandGuard(limits, limits.home_command)
    sensor = adapter_module.ReadOnlyJointSensorAdapter(
        bus=follower.bus,
        validate_positions=delta_guard.validate_measured_positions,
        packet_factory=assembler_module.JointStatePacket,
        monotonic=time.monotonic,
        max_read_duration_seconds=max_read_duration,
    )

    print("\n===== ONE-SHOT LIVE PRESENT_POSITION READ =====")
    result = None
    try:
        sensor.open()
        result = sensor.read()
    finally:
        sensor.close()
    if result is None:
        raise OneShotReadError("one-shot read returned no result")
    if sensor.sequence != 1:
        raise OneShotReadError(f"expected exactly one successful read, got {sensor.sequence}")
    if sensor.state is not adapter_module.AdapterState.CLOSED:
        raise OneShotReadError(f"adapter final state is {sensor.state}")
    if bool(getattr(follower.bus, "is_connected", True)):
        raise OneShotReadError("follower bus remained connected after one-shot read")

    actual_q = tuple(float(value) for value in result.packet.actual_q)
    if len(actual_q) != len(MOTOR_ORDER):
        raise OneShotReadError("one-shot JointStatePacket dimension mismatch")
    print("joint             normalized_position")
    for motor, value in zip(MOTOR_ORDER, actual_q, strict=True):
        print(f"{motor:<18} {value:>10.5f}")
    print(f"read duration: {result.metrics.read_duration_ms:.3f} ms")
    print("successful reads: 1/1")
    print("serial closed with disable_torque=False: PASS")

    decision = "ONE_SHOT_READ_ONLY_JOINT_SENSOR_PASS_BUILD_LIVE_SHADOW_MODE_NEXT"
    report = {
        "schema_version": 1,
        "raw_dataset_archive_note": {
            "windows_path": r"F:\episodes_pick_place_pilot_v5",
            "wsl_path": "/mnt/f/episodes_pick_place_pilot_v5",
            "used_by_this_read": False,
        },
        "authorization": {
            "kind": "explicit_cli_one_shot_read_only",
            "flag_present": True,
            "maximum_successful_reads": 1,
            "live_read_authorized_for_this_process": True,
            "motion_authorized": False,
            "hardware_deployment_authorized": False,
        },
        "scope": {
            "vendor_modules_imported": True,
            "robot_object_created": True,
            "serial_port_opened": True,
            "motor_read_instruction_sent": True,
            "successful_motor_reads": 1,
            "motor_register_write_api_called": False,
            "goal_position_written": False,
            "torque_api_called": False,
            "full_follower_connect_called": False,
            "full_follower_disconnect_called": False,
            "camera_configured_or_opened": False,
            "model_loaded": False,
            "policy_inference_run": False,
            "action_sent": False,
            "serial_port_closed": True,
            "hardware_deployment_authorized": False,
        },
        "static_launcher_boundary": static_boundary,
        "adapter_gate": adapter_gate,
        "frozen_dependencies": {
            "guard": {"path": str(guard_path), "sha256": guard_sha},
            "assembler": {"path": str(assembler_path), "sha256": assembler_sha},
            "follower_source": {
                "path": str(follower_source),
                "sha256": follower_sha,
            },
            "follower_config_source": {
                "path": str(config_source),
                "sha256": config_sha,
            },
            "contract_and_calibration": frozen_inputs,
            "imported_modules": imported_modules,
        },
        "device": device,
        "constructed_follower": follower_summary,
        "one_shot_read": {
            "motor_order": list(MOTOR_ORDER),
            "actual_q": list(actual_q),
            "read_started_monotonic_s": result.metrics.read_started_monotonic_s,
            "read_completed_monotonic_s": result.metrics.read_completed_monotonic_s,
            "read_duration_ms": result.metrics.read_duration_ms,
            "successful_reads": 1,
            "adapter_sequence": sensor.sequence,
            "adapter_final_state": sensor.state.value,
        },
        "decision": decision,
        "live_shadow_mode_authorized": False,
        "motion_authorized": False,
        "hardware_deployment_authorized": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\n===== DECISION =====")
    print(f"decision='{decision}'")
    print("ONE-SHOT SENSOR READ COMPLETE; SERIAL PORT CLOSED.")
    print("LIVE SHADOW MODE, MOTION, AND DEPLOYMENT REMAIN BLOCKED.")
    print("NO CAMERA WAS OPENED. NO MODEL WAS LOADED. NO INFERENCE RAN.")
    print("NO REGISTER WRITE, GOAL POSITION, TORQUE, OR ACTION API WAS CALLED.")
    print("\n===== OUTPUT =====")
    print(output_path)
    print("ACT V3 ONE-SHOT READ-ONLY JOINT SENSOR: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"\nACT V3 ONE-SHOT READ-ONLY JOINT SENSOR: FAIL: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print("MOTION AND HARDWARE DEPLOYMENT REMAIN BLOCKED.", file=sys.stderr)
        print(
            "If live access had begun, the reviewed adapter attempted "
            "disconnect(disable_torque=False). Verify the process exited and the port is closed.",
            file=sys.stderr,
        )
        print("NO ACTION OR GOAL POSITION IS AUTHORIZED BY THIS SCRIPT.", file=sys.stderr)
        raise SystemExit(1)
