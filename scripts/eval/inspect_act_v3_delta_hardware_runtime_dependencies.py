#!/usr/bin/env python3
"""Statically inventory the dependencies for a future ACT V3 hardware adapter.

This program is deliberately read-only with respect to robot hardware. It never
imports LeRobot robot/camera modules, never constructs a robot object, never
opens a serial port or camera, and never sends a command. Vendor and project
Python files are parsed as text with :mod:`ast`.

The output is an evidence report, not a deployment authorization. Its purpose
is to freeze the exact local interfaces that a later, separately reviewed live
adapter would have to use: follower configuration, observation readback,
command write, connection/disconnection, camera configuration, calibration,
and the already reviewed safety-controller constants.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence


EXPECTED_GUARD_SHA256 = (
    "e06e785071ee3ded95e89d670978bcb4a8caf104b2fc69fc87de37cb6fff6d15"
)
EXPECTED_CONTRACT_SHA256 = (
    "ff93a28d0698c1367431215185159797b77847db06fd3ea29bdcc1aa916e7083"
)
EXPECTED_CONTRACT_SCHEMA = "so101_act_v3_delta_runtime_limit_contract_v1"
MOTORS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
REQUIRED_FOLLOWER_METHODS = (
    "connect",
    "get_observation",
    "send_action",
    "disconnect",
)
SAFE_CONSTANT_NAMES = (
    "LEADER_PORT",
    "FOLLOWER_PORT",
    "ALL_MOTORS",
    "MAX_STEP",
    "SOFT_LIMITS",
    "FOLLOWER_CALIBRATED_RANGES",
    "FOLLOWER_SENSOR_TOLERANCE",
    "TRACKING_ERROR_LIMIT",
    "TRACKING_ERROR_TIMEOUT",
)
INTERFACE_CALL_TERMS = (
    "connect",
    "disconnect",
    "calibr",
    "torque",
    "read",
    "write",
    "send_action",
    "get_observation",
    "sync_read",
    "sync_write",
    "ensure_safe",
    "camera",
    "capture",
)
SEARCHED_CONFIG_CALLS = {
    "OpenCVCameraConfig",
    "IntelRealSenseCameraConfig",
    "SO101FollowerConfig",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guard-core", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--safe-teleop", type=Path, required=True)
    parser.add_argument("--follower-config-source", type=Path, required=True)
    parser.add_argument("--follower-source", type=Path, required=True)
    parser.add_argument(
        "--scripts-root",
        type=Path,
        required=True,
        help="Project scripts tree searched statically for camera/follower config calls.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
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


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON: {path}: {exc}") from exc


def parse_python(path: Path) -> tuple[str, ast.Module]:
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise RuntimeError(f"cannot parse Python source {path}: {exc}") from exc
    return source, tree


def stable_value(node: ast.AST | None) -> Any:
    if node is None:
        return None
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return {"python_expression": ast.unparse(node)}
    return json_compatible(value)


def json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return repr(value)
        return value
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    return repr(value)


def call_leaf_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ast.unparse(func)


def source_span(node: ast.AST) -> dict[str, int]:
    return {
        "start_line": int(getattr(node, "lineno", 0)),
        "end_line": int(getattr(node, "end_lineno", getattr(node, "lineno", 0))),
    }


def function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{prefix} {node.name}({ast.unparse(node.args)})"


def selected_calls(node: ast.AST) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        expression = ast.unparse(child)
        lowered = expression.casefold()
        if any(term in lowered for term in INTERFACE_CALL_TERMS):
            rows.append(
                {
                    "line": int(child.lineno),
                    "leaf_name": call_leaf_name(child),
                    "expression": expression,
                }
            )
    rows.sort(key=lambda row: (row["line"], row["expression"]))
    return rows


def find_class(tree: ast.Module, class_name: str) -> ast.ClassDef:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {class_name} class, found {len(matches)}"
        )
    return matches[0]


def extract_config_class(path: Path) -> dict[str, Any]:
    source, tree = parse_python(path)
    class_node = find_class(tree, "SO101FollowerConfig")
    fields: list[dict[str, Any]] = []
    for node in class_node.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            fields.append(
                {
                    "name": node.target.id,
                    "annotation": ast.unparse(node.annotation),
                    "default": stable_value(node.value),
                    "line": int(node.lineno),
                }
            )
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    fields.append(
                        {
                            "name": target.id,
                            "annotation": None,
                            "default": stable_value(node.value),
                            "line": int(node.lineno),
                        }
                    )
    if not fields:
        raise RuntimeError("SO101FollowerConfig fields were not found")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "class": "SO101FollowerConfig",
        **source_span(class_node),
        "bases": [ast.unparse(base) for base in class_node.bases],
        "fields": fields,
        "source_excerpt": ast.get_source_segment(source, class_node),
    }


def extract_follower_api(path: Path) -> dict[str, Any]:
    source, tree = parse_python(path)
    class_node = find_class(tree, "SO101Follower")
    methods: dict[str, dict[str, Any]] = {}
    for node in class_node.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in REQUIRED_FOLLOWER_METHODS and node.name not in {
            "calibrate",
            "configure",
        }:
            continue
        methods[node.name] = {
            "signature": function_signature(node),
            **source_span(node),
            "selected_calls": selected_calls(node),
            "source_excerpt": ast.get_source_segment(source, node),
        }
    missing = sorted(set(REQUIRED_FOLLOWER_METHODS) - set(methods))
    if missing:
        raise RuntimeError(f"SO101Follower required methods missing: {missing}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "class": "SO101Follower",
        **source_span(class_node),
        "methods": methods,
    }


def top_level_assignments(tree: ast.Module) -> dict[str, ast.AST | None]:
    result: dict[str, ast.AST | None] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    result[target.id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            result[node.target.id] = node.value
    return result


def extract_safe_teleop(path: Path) -> dict[str, Any]:
    source, tree = parse_python(path)
    assignments = top_level_assignments(tree)
    constants = {
        name: stable_value(assignments[name])
        for name in SAFE_CONSTANT_NAMES
        if name in assignments
    }
    minimum_required = {
        "FOLLOWER_PORT",
        "ALL_MOTORS",
        "MAX_STEP",
        "SOFT_LIMITS",
        "FOLLOWER_SENSOR_TOLERANCE",
        "TRACKING_ERROR_LIMIT",
        "TRACKING_ERROR_TIMEOUT",
    }
    missing = sorted(minimum_required - set(constants))
    if missing:
        raise RuntimeError(f"reviewed safe teleop constants missing: {missing}")

    main_nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "main"
    ]
    main_info: dict[str, Any] | None = None
    if len(main_nodes) == 1:
        main_node = main_nodes[0]
        main_info = {
            "signature": function_signature(main_node),
            **source_span(main_node),
            "selected_calls": selected_calls(main_node),
            "source_excerpt": ast.get_source_segment(source, main_node),
        }
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "constants": constants,
        "main": main_info,
    }


def walk_json_leaves(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from walk_json_leaves(child, child_prefix)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk_json_leaves(child, f"{prefix}[{index}]")
    else:
        yield prefix, value


def relevant_contract_leaves(contract: dict[str, Any]) -> list[dict[str, Any]]:
    terms = (
        "hardware",
        "calibr",
        "motor",
        "normal",
        "home",
        "startup",
        "soft_limit",
        "max_abs_delta",
        "tracking",
        "sensor",
        "timeout",
        "camera",
        "fps",
        "command",
        "robot_object",
        "serial_port",
    )
    rows = []
    for path, value in walk_json_leaves(contract):
        if any(term in path.casefold() for term in terms):
            rows.append({"path": path, "value": json_compatible(value)})
    return rows


def verify_contract(contract: Any) -> dict[str, Any]:
    if not isinstance(contract, dict):
        raise RuntimeError("contract root must be a JSON object")
    if contract.get("schema_version") != EXPECTED_CONTRACT_SCHEMA:
        raise RuntimeError("unexpected runtime contract schema_version")
    scope = contract.get("scope")
    if not isinstance(scope, dict):
        raise RuntimeError("contract.scope must be an object")
    required_false = (
        "hardware_access_allowed",
        "hardware_deployment_authorized",
        "physical_safety_certified",
        "recalibration_authorized",
    )
    invalid = [name for name in required_false if scope.get(name) is not False]
    if invalid:
        raise RuntimeError(f"contract authorization flags are not false: {invalid}")
    robot = contract.get("robot")
    if not isinstance(robot, dict) or tuple(robot.get("motors", ())) != MOTORS:
        raise RuntimeError("contract motor order mismatch")
    policy = contract.get("policy")
    if not isinstance(policy, dict):
        raise RuntimeError("contract.policy must be an object")
    expected_policy = {
        "candidate_step": 11000,
        "observation_state_dim": 18,
        "action_dim": 6,
        "chunk_size": 10,
        "n_action_steps": 5,
        "replan_interval_frames": 5,
        "dataset_fps": 15,
        "use_vae": False,
        "use_amp": False,
    }
    mismatches = {
        key: {"expected": expected, "actual": policy.get(key)}
        for key, expected in expected_policy.items()
        if policy.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"contract policy mismatch: {mismatches}")
    return {
        "schema_version": contract["schema_version"],
        "scope": scope,
        "motor_order": list(MOTORS),
        "policy": {key: policy[key] for key in expected_policy},
    }


def verify_calibration(calibration: Any, contract: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(calibration, dict):
        raise RuntimeError("calibration root must be a JSON object")
    expected = contract["robot"].get("expected_calibration_raw")
    if not isinstance(expected, dict):
        raise RuntimeError("contract expected_calibration_raw is missing")

    if isinstance(calibration.get("motors"), dict):
        actual_motors = calibration["motors"]
        storage = "motors"
    else:
        actual_motors = calibration
        storage = "root"
    if not isinstance(actual_motors, dict):
        raise RuntimeError("calibration motor mapping is missing")

    mismatches: list[dict[str, Any]] = []
    normalized: dict[str, dict[str, Any]] = {}
    for motor in MOTORS:
        actual_row = actual_motors.get(motor)
        expected_row = expected.get(motor)
        if not isinstance(actual_row, dict) or not isinstance(expected_row, dict):
            mismatches.append(
                {"motor": motor, "field": "record", "expected": expected_row, "actual": actual_row}
            )
            continue
        normalized[motor] = {}
        for field in ("id", "drive_mode", "homing_offset", "range_min", "range_max"):
            actual_value = actual_row.get(field)
            expected_value = expected_row.get(field)
            normalized[motor][field] = actual_value
            if actual_value != expected_value:
                mismatches.append(
                    {
                        "motor": motor,
                        "field": field,
                        "expected": expected_value,
                        "actual": actual_value,
                    }
                )
    if mismatches:
        raise RuntimeError(f"calibration differs from frozen contract: {mismatches}")
    return {"storage": storage, "motors": normalized, "matches_contract": True}


def extract_config_calls(path: Path, scripts_root: Path) -> list[dict[str, Any]]:
    try:
        _, tree = parse_python(path)
    except (OSError, UnicodeDecodeError, RuntimeError) as exc:
        return [
            {
                "path": str(path),
                "parse_error": str(exc),
            }
        ]
    rows: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        leaf = call_leaf_name(node)
        if leaf not in SEARCHED_CONFIG_CALLS:
            continue
        try:
            relative = path.relative_to(scripts_root).as_posix()
        except ValueError:
            relative = str(path)
        rows.append(
            {
                "path": relative,
                "line": int(node.lineno),
                "config_type": leaf,
                "expression": ast.unparse(node),
            }
        )
    return rows


def search_project_configs(scripts_root: Path) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []
    files_scanned = 0
    for path in sorted(scripts_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        files_scanned += 1
        extracted = extract_config_calls(path, scripts_root)
        for row in extracted:
            if "parse_error" in row:
                parse_errors.append(row)
            else:
                calls.append(row)
    calls.sort(key=lambda row: (row["path"], row["line"], row["config_type"]))
    camera_calls = [
        row
        for row in calls
        if row["config_type"] in {"OpenCVCameraConfig", "IntelRealSenseCameraConfig"}
    ]
    follower_calls = [row for row in calls if row["config_type"] == "SO101FollowerConfig"]
    return {
        "scripts_root": str(scripts_root),
        "python_files_scanned": files_scanned,
        "camera_config_calls": camera_calls,
        "follower_config_calls": follower_calls,
        "parse_errors": parse_errors,
    }


def static_no_hardware_self_audit(path: Path) -> dict[str, Any]:
    _, tree = parse_python(path)
    imported: set[str] = set()
    called: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Call):
            called.append(
                {
                    "leaf": call_leaf_name(node),
                    "line": int(node.lineno),
                }
            )
    forbidden_import_roots = {"lerobot", "serial", "pyserial", "torch", "cv2", "dynamixel_sdk"}
    bad_imports = sorted(
        module
        for module in imported
        if module.split(".", 1)[0] in forbidden_import_roots
    )
    forbidden_call_leaves = {
        "connect",
        "disconnect",
        "send_action",
        "get_observation",
        "sync_read",
        "sync_write",
        "enable_torque",
        "disable_torque",
        "write_calibration",
        "VideoCapture",
    }
    bad_calls = [row for row in called if row["leaf"] in forbidden_call_leaves]
    if bad_imports or bad_calls:
        raise RuntimeError(
            "inventory script violated static no-hardware boundary: "
            f"imports={bad_imports}, calls={bad_calls}"
        )
    return {
        "source_path": str(path),
        "source_sha256": sha256_file(path),
        "forbidden_imports": bad_imports,
        "forbidden_hardware_calls": bad_calls,
        "pass": True,
    }


def normalize_per_motor(value: Any) -> list[Any] | None:
    if isinstance(value, dict) and set(value) == {"python_expression"}:
        return None
    if isinstance(value, dict):
        if any(motor not in value for motor in MOTORS):
            return None
        return [value[motor] for motor in MOTORS]
    if isinstance(value, (list, tuple)) and len(value) == len(MOTORS):
        return list(value)
    return None


def literal_matches(actual: Any, expected: Any, *, per_motor: bool = False) -> bool:
    if per_motor:
        return normalize_per_motor(actual) == normalize_per_motor(expected)
    if isinstance(actual, dict) and set(actual) == {"python_expression"}:
        return False
    if isinstance(expected, tuple):
        expected = list(expected)
    return actual == expected


def compare_safe_contract(
    safe: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any]:
    constants = safe["constants"]
    guard = contract["guard"]
    expected: dict[str, Any] = {
        "ALL_MOTORS": list(MOTORS),
        "MAX_STEP": guard["max_abs_delta_per_command"],
        "SOFT_LIMITS": guard["normal_soft_limits"],
        "FOLLOWER_SENSOR_TOLERANCE": guard["follower_sensor_tolerance"],
        "TRACKING_ERROR_LIMIT": guard["tracking_error_limit"],
        "TRACKING_ERROR_TIMEOUT": guard["tracking_error_timeout_seconds"],
    }
    per_motor_names = {"MAX_STEP", "SOFT_LIMITS", "TRACKING_ERROR_LIMIT"}
    comparisons = {
        name: {
            "match": literal_matches(
                constants.get(name),
                expected_value,
                per_motor=name in per_motor_names,
            ),
            "safe_teleop": constants.get(name),
            "contract": expected_value,
        }
        for name, expected_value in expected.items()
    }
    mismatches = [name for name, row in comparisons.items() if not row["match"]]
    if mismatches:
        raise RuntimeError(
            "reviewed safe teleop differs from frozen contract: " f"{mismatches}"
        )
    return {"comparisons": comparisons, "all_match": True}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    guard_core = require_file(args.guard_core)
    contract_path = require_file(args.contract)
    calibration_path = require_file(args.calibration)
    safe_teleop_path = require_file(args.safe_teleop)
    follower_config_path = require_file(args.follower_config_source)
    follower_path = require_file(args.follower_source)
    scripts_root = require_dir(args.scripts_root)
    output_path = args.output_json.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing report: {output_path}"
        )

    print("===== STATIC READ-ONLY BOUNDARY =====")
    self_audit = static_no_hardware_self_audit(Path(__file__).resolve())
    print("no LeRobot/serial/torch/cv2 imports or hardware calls: PASS")
    print("NO vendor module will be imported; source is parsed as text.")

    print("\n===== VERIFY FROZEN GUARD, CONTRACT, AND CALIBRATION =====")
    guard_sha = sha256_file(guard_core)
    contract_sha = sha256_file(contract_path)
    if guard_sha != EXPECTED_GUARD_SHA256:
        raise RuntimeError(
            f"guard SHA256 mismatch: expected {EXPECTED_GUARD_SHA256}, got {guard_sha}"
        )
    if contract_sha != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError(
            "contract SHA256 mismatch: "
            f"expected {EXPECTED_CONTRACT_SHA256}, got {contract_sha}"
        )
    contract = load_json(contract_path)
    contract_summary = verify_contract(contract)
    calibration = load_json(calibration_path)
    calibration_summary = verify_calibration(calibration, contract)
    print("guard byte identity: PASS")
    print("contract schema, policy, motor order, and authorization flags: PASS")
    print("calibration byte semantics match frozen contract: PASS")
    print("hardware access allowed: false")

    print("\n===== STATIC FOLLOWER CONFIG AND API =====")
    config_api = extract_config_class(follower_config_path)
    follower_api = extract_follower_api(follower_path)
    field_names = [row["name"] for row in config_api["fields"]]
    print(f"SO101FollowerConfig fields ({len(field_names)}): {field_names}")
    for method in REQUIRED_FOLLOWER_METHODS:
        print(f"{follower_api['methods'][method]['signature']}: FOUND")
    print("follower sources were not imported: PASS")

    print("\n===== REVIEWED SAFE TELEOP CONTRACT =====")
    safe = extract_safe_teleop(safe_teleop_path)
    safe_comparison = compare_safe_contract(safe, contract)
    print("motor order, per-command rate limits, soft limits, and tracking limits: MATCH")
    print(f"follower port expression: {safe['constants'].get('FOLLOWER_PORT')}")

    print("\n===== STATIC PROJECT CAMERA / ROBOT CONFIG SEARCH =====")
    project_configs = search_project_configs(scripts_root)
    print(f"Python files scanned: {project_configs['python_files_scanned']}")
    print(f"camera config calls: {len(project_configs['camera_config_calls'])}")
    print(f"follower config calls: {len(project_configs['follower_config_calls'])}")
    for row in project_configs["camera_config_calls"]:
        print(f"camera {row['path']}:{row['line']} {row['expression']}")
    for row in project_configs["follower_config_calls"]:
        print(f"robot  {row['path']}:{row['line']} {row['expression']}")
    if project_configs["parse_errors"]:
        print(f"parse warnings: {len(project_configs['parse_errors'])}")

    unresolved: list[str] = []
    if not project_configs["camera_config_calls"]:
        unresolved.append(
            "No camera configuration call was found under scripts-root; front/wrist live camera mapping must be supplied and reviewed before building a live adapter."
        )
    if not project_configs["follower_config_calls"]:
        unresolved.append(
            "No SO101FollowerConfig call was found under scripts-root; exact live follower construction must be supplied and reviewed."
        )
    if project_configs["parse_errors"]:
        unresolved.append(
            "One or more project Python files could not be parsed; review parse_errors before relying on the search result."
        )

    decision = (
        "STATIC_DEPENDENCY_INVENTORY_PASS_REVIEW_UNRESOLVED_INPUTS"
        if unresolved
        else "STATIC_DEPENDENCY_INVENTORY_PASS_BUILD_INERT_ADAPTER_NEXT"
    )
    report = {
        "schema_version": "act_v3_delta_hardware_runtime_dependency_inventory_v1",
        "scope": {
            "static_read_only": True,
            "vendor_modules_imported": False,
            "model_loaded": False,
            "dataset_loaded": False,
            "gpu_used": False,
            "robot_object_created": False,
            "serial_port_opened": False,
            "live_camera_opened": False,
            "command_sent": False,
            "hardware_deployment_authorized": False,
        },
        "self_audit": self_audit,
        "frozen_inputs": {
            "guard_core": {"path": str(guard_core), "sha256": guard_sha},
            "contract": {"path": str(contract_path), "sha256": contract_sha},
            "calibration": {
                "path": str(calibration_path),
                "sha256": sha256_file(calibration_path),
            },
            "safe_teleop": {
                "path": str(safe_teleop_path),
                "sha256": sha256_file(safe_teleop_path),
            },
        },
        "contract_summary": contract_summary,
        "contract_runtime_leaves": relevant_contract_leaves(contract),
        "calibration_summary": calibration_summary,
        "follower_config_api": config_api,
        "follower_runtime_api": follower_api,
        "reviewed_safe_teleop": safe,
        "safe_teleop_contract_comparison": safe_comparison,
        "project_config_search": project_configs,
        "task_specific_boundary_semantics": {
            "joint": "wrist_flex",
            "task_endpoint": 60.0,
            "interpretation": (
                "The +60 wrist_flex endpoint is an expected downward pick-task working boundary. "
                "An outward residual while already at that endpoint is BOUNDARY_HOLD, not a fault; "
                "the command remains clamped and motion beyond +60 remains forbidden."
            ),
        },
        "unresolved_requirements": unresolved,
        "decision": decision,
        "hardware_deployment_authorized": False,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("\n===== DECISION =====")
    print(f"decision='{decision}'")
    if unresolved:
        for item in unresolved:
            print(f"UNRESOLVED: {item}")
    else:
        print("Exact local camera and follower construction candidates were found.")
    print("HARDWARE DEPLOYMENT REMAINS BLOCKED.")
    print("NO MODEL OR DATASET WAS LOADED. NO GPU WAS USED.")
    print("NO ROBOT OBJECT WAS CREATED. NO SERIAL PORT OR LIVE CAMERA WAS OPENED.")
    print("NO COMMAND WAS SENT. NO CALIBRATION OR SOURCE FILE WAS MODIFIED.")
    print("\n===== OUTPUT =====")
    print(output_path)
    print("ACT V3 DELTA HARDWARE RUNTIME DEPENDENCY INVENTORY: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nINVENTORY ABORTED: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("HARDWARE DEPLOYMENT REMAINS BLOCKED.", file=sys.stderr)
        print(
            "NO ROBOT OBJECT WAS CREATED. NO SERIAL PORT OR LIVE CAMERA WAS OPENED. "
            "NO COMMAND WAS SENT.",
            file=sys.stderr,
        )
        raise
