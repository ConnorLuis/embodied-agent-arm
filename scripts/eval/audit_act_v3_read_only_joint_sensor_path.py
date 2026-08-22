#!/usr/bin/env python3
"""Statically audit a non-commanding SO101 joint readback path.

This program parses the frozen local LeRobot sources as text.  It deliberately
does not import LeRobot, construct a robot, open a serial port, read a motor,
change torque, write a register, load a policy, or open a camera.

The audit exists because ``SO101Follower.connect(calibrate=False)`` is *not* a
read-only operation in the reviewed vendor version: after opening the bus it
unconditionally calls ``configure()``, which changes torque state and writes
motor registers.  The only candidate advanced by this audit is the narrower
transport path used by a future, separately authorized adapter::

    follower.bus.connect()
    follower.bus.sync_read("Present_Position", normalize=True, num_retry=3)
    follower.bus.disconnect(disable_torque=False)

Passing this audit approves that source-level design for the next inert adapter
stage.  It does not authorize opening the serial port or accessing hardware.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


EXPECTED_ASSEMBLER_SHA256 = (
    "866a53d8b4ce1473344779c493a955d03dafa485115fab34a4ffc929ff31d424"
)
EXPECTED_FOLLOWER_SHA256 = (
    "26c675c71ade2670fa1bed2d887507b6ef3ad53a49e20182f5ba3d032a0afd7a"
)
EXPECTED_FOLLOWER_CONFIG_SHA256 = (
    "e5702d9b6c4a09d10de83f912a1640169698292d5145091d403f4fdce0211cbe"
)
EXPECTED_ASSEMBLER_DECISION = (
    "LIVE_OBSERVATION_ASSEMBLER_OFFLINE_PASS_"
    "BUILD_READ_ONLY_LIVE_SENSOR_ADAPTER_NEXT"
)
EXPECTED_INVENTORY_SCHEMA = (
    "act_v3_delta_hardware_runtime_dependency_inventory_v1"
)
EXPECTED_INVENTORY_DECISIONS = {
    "STATIC_DEPENDENCY_INVENTORY_PASS_REVIEW_UNRESOLVED_INPUTS",
    "STATIC_DEPENDENCY_INVENTORY_PASS_BUILD_INERT_ADAPTER_NEXT",
}
EXPECTED_BUS_CLASS = "FeetechMotorsBus"
EXPECTED_REGISTER = "Present_Position"

MUTATING_CALL_LEAVES = {
    "calibrate",
    "configure",
    "configure_motors",
    "disable_torque",
    "enable_torque",
    "record_ranges_of_motion",
    "send_action",
    "set_half_turn_homings",
    "setup_motor",
    "sync_write",
    "torque_disabled",
    "write",
    "write_calibration",
}
FORBIDDEN_REGISTER_LITERALS = {
    "Goal_Position",
    "Operating_Mode",
    "P_Coefficient",
    "I_Coefficient",
    "D_Coefficient",
    "Max_Torque_Limit",
    "Protection_Current",
    "Overload_Torque",
    "Torque_Enable",
    "Homing_Offset",
    "Min_Position_Limit",
    "Max_Position_Limit",
    "Drive_Mode",
}


class AuditError(RuntimeError):
    """A frozen input or source-side-effect contract was not satisfied."""


@dataclass(frozen=True)
class ClassLocation:
    path: Path
    source: str
    node: ast.ClassDef


@dataclass(frozen=True)
class MethodLocation:
    owner: str
    path: Path
    source: str
    node: ast.FunctionDef | ast.AsyncFunctionDef


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembler", type=Path, required=True)
    parser.add_argument("--assembler-report", type=Path, required=True)
    parser.add_argument("--dependency-inventory", type=Path, required=True)
    parser.add_argument("--follower-config-source", type=Path, required=True)
    parser.add_argument("--follower-source", type=Path, required=True)
    parser.add_argument("--motors-source-root", type=Path, required=True)
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
        raise AuditError(f"invalid JSON: {path}: {exc}") from exc


def parse_python(path: Path) -> tuple[str, ast.Module]:
    source = path.read_text(encoding="utf-8")
    try:
        return source, ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise AuditError(f"cannot parse Python source {path}: {exc}") from exc


def call_leaf(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ast.unparse(func)


def call_expression(call: ast.Call) -> str:
    return ast.unparse(call)


def direct_calls(node: ast.AST) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            rows.append(
                {
                    "line": int(getattr(child, "lineno", 0)),
                    "leaf": call_leaf(child),
                    "expression": call_expression(child),
                }
            )
    rows.sort(key=lambda row: (row["line"], row["expression"]))
    return rows


def string_literals(node: ast.AST) -> list[str]:
    return sorted(
        {
            child.value
            for child in ast.walk(node)
            if isinstance(child, ast.Constant) and isinstance(child.value, str)
        }
    )


def method_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{prefix} {node.name}({ast.unparse(node.args)})"


def method_record(location: MethodLocation) -> dict[str, Any]:
    calls = direct_calls(location.node)
    return {
        "owner": location.owner,
        "path": str(location.path),
        "signature": method_signature(location.node),
        "start_line": int(location.node.lineno),
        "end_line": int(getattr(location.node, "end_lineno", location.node.lineno)),
        "calls": calls,
        "mutating_calls": [row for row in calls if row["leaf"] in MUTATING_CALL_LEAVES],
        "string_literals": string_literals(location.node),
        "source_excerpt": ast.get_source_segment(location.source, location.node),
    }


def find_unique_class(path: Path, class_name: str) -> ClassLocation:
    source, tree = parse_python(path)
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    if len(matches) != 1:
        raise AuditError(
            f"expected exactly one {class_name} in {path}, found {len(matches)}"
        )
    return ClassLocation(path=path, source=source, node=matches[0])


def own_methods(location: ClassLocation) -> dict[str, MethodLocation]:
    result: dict[str, MethodLocation] = {}
    for node in location.node.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result[node.name] = MethodLocation(
                owner=location.node.name,
                path=location.path,
                source=location.source,
                node=node,
            )
    return result


def scan_class_index(root: Path) -> tuple[dict[str, list[ClassLocation]], list[dict[str, str]]]:
    index: dict[str, list[ClassLocation]] = {}
    parse_errors: list[dict[str, str]] = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            source, tree = parse_python(path)
        except (OSError, UnicodeDecodeError, AuditError) as exc:
            parse_errors.append({"path": str(path), "error": str(exc)})
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                index.setdefault(node.name, []).append(
                    ClassLocation(path=path, source=source, node=node)
                )
    return index, parse_errors


def base_leaf(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return base_leaf(node.value)
    return ast.unparse(node)


def unique_index_class(
    index: dict[str, list[ClassLocation]], class_name: str
) -> ClassLocation:
    matches = index.get(class_name, [])
    if len(matches) != 1:
        paths = [str(row.path) for row in matches]
        raise AuditError(
            f"expected exactly one motor class {class_name}, found {len(matches)}: {paths}"
        )
    return matches[0]


def resolve_method(
    index: dict[str, list[ClassLocation]],
    class_name: str,
    method_name: str,
    visiting: tuple[str, ...] = (),
) -> MethodLocation:
    if class_name in visiting:
        raise AuditError(f"class inheritance cycle while resolving {class_name}.{method_name}")
    location = unique_index_class(index, class_name)
    methods = own_methods(location)
    if method_name in methods:
        return methods[method_name]
    for base in location.node.bases:
        base_name = base_leaf(base)
        if base_name not in index:
            continue
        try:
            return resolve_method(
                index,
                base_name,
                method_name,
                visiting=(*visiting, class_name),
            )
        except AuditError as exc:
            if "found 0" not in str(exc):
                raise
    raise AuditError(f"cannot resolve {class_name}.{method_name} under motors source root")


def verify_static_boundary(path: Path) -> dict[str, Any]:
    source, tree = parse_python(path)
    imports: set[str] = set()
    call_expressions: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
        elif isinstance(node, ast.Call):
            call_expressions.append(ast.unparse(node.func))
    bad_imports = sorted(
        module
        for module in imports
        if module.startswith(
            (
                "lerobot",
                "serial",
                "pyserial",
                "cv2",
                "torch",
                "dynamixel_sdk",
                "scservo_sdk",
            )
        )
    )
    forbidden_exact_calls = {
        "SO101Follower",
        "SO101FollowerConfig",
        "VideoCapture",
        "ACTPolicy.from_pretrained",
    }
    bad_calls = sorted(
        expression
        for expression in call_expressions
        if expression in forbidden_exact_calls
        or expression.endswith(
            (
                ".send_action",
                ".sync_write",
                ".enable_torque",
                ".disable_torque",
                ".connect",
                ".disconnect",
                ".sync_read",
            )
        )
    )
    if bad_imports or bad_calls:
        raise AuditError(
            f"audit script violates static boundary: imports={bad_imports}, calls={bad_calls}"
        )
    return {
        "source": str(path),
        "source_sha256": sha256_file(path),
        "vendor_modules_imported": False,
        "robot_object_created": False,
        "serial_port_opened": False,
        "motor_read_instruction_sent": False,
        "motor_register_write_sent": False,
        "torque_state_changed": False,
        "camera_opened": False,
        "model_loaded": False,
        "command_sent": False,
        "pass": True,
        "source_length": len(source),
    }


def verify_previous_gates(
    assembler: Path,
    assembler_report_path: Path,
    inventory_path: Path,
    follower_source: Path,
    config_source: Path,
) -> dict[str, Any]:
    assembler_sha = sha256_file(assembler)
    if assembler_sha != EXPECTED_ASSEMBLER_SHA256:
        raise AuditError(
            f"assembler SHA256 mismatch: expected {EXPECTED_ASSEMBLER_SHA256}, got {assembler_sha}"
        )
    assembler_report = load_json(assembler_report_path)
    if not isinstance(assembler_report, dict):
        raise AuditError("assembler report root must be an object")
    if assembler_report.get("schema_version") != 1:
        raise AuditError("unexpected assembler report schema_version")
    if assembler_report.get("decision") != EXPECTED_ASSEMBLER_DECISION:
        raise AuditError("assembler report decision mismatch")
    if assembler_report.get("fail_closed_tests_passed") != 16:
        raise AuditError("assembler report did not pass all 16 fail-closed tests")
    if assembler_report.get("fail_closed_tests_total") != 16:
        raise AuditError("assembler report fail-closed test total mismatch")
    for flag in (
        "hardware_deployment_authorized",
        "camera_opened",
        "model_loaded",
        "robot_accessed",
        "serial_port_opened",
        "command_sent",
    ):
        if assembler_report.get(flag) is not False:
            raise AuditError(f"assembler report scope flag must be false: {flag}")

    inventory = load_json(inventory_path)
    if not isinstance(inventory, dict):
        raise AuditError("dependency inventory root must be an object")
    if inventory.get("schema_version") != EXPECTED_INVENTORY_SCHEMA:
        raise AuditError("dependency inventory schema mismatch")
    if inventory.get("decision") not in EXPECTED_INVENTORY_DECISIONS:
        raise AuditError("dependency inventory decision is not an accepted static pass")
    scope = inventory.get("scope")
    if not isinstance(scope, dict) or scope.get("static_read_only") is not True:
        raise AuditError("dependency inventory was not static read-only")
    for flag in (
        "hardware_deployment_authorized",
        "live_camera_opened",
        "robot_object_created",
        "serial_port_opened",
        "command_sent",
    ):
        if scope.get(flag) is not False:
            raise AuditError(f"dependency inventory scope flag must be false: {flag}")

    current_follower_sha = sha256_file(follower_source)
    current_config_sha = sha256_file(config_source)
    if current_follower_sha != EXPECTED_FOLLOWER_SHA256:
        raise AuditError(
            f"follower source SHA256 mismatch: expected {EXPECTED_FOLLOWER_SHA256}, got {current_follower_sha}"
        )
    if current_config_sha != EXPECTED_FOLLOWER_CONFIG_SHA256:
        raise AuditError(
            "follower config source SHA256 mismatch: "
            f"expected {EXPECTED_FOLLOWER_CONFIG_SHA256}, got {current_config_sha}"
        )
    inventory_follower = inventory.get("follower_runtime_api", {})
    inventory_config = inventory.get("follower_config_api", {})
    if inventory_follower.get("sha256") != current_follower_sha:
        raise AuditError("current follower source differs from dependency inventory")
    if inventory_config.get("sha256") != current_config_sha:
        raise AuditError("current follower config differs from dependency inventory")
    return {
        "assembler": {
            "path": str(assembler),
            "sha256": assembler_sha,
            "report": str(assembler_report_path),
            "decision": assembler_report["decision"],
            "fail_closed_tests": "16/16",
        },
        "dependency_inventory": {
            "path": str(inventory_path),
            "schema_version": inventory["schema_version"],
            "decision": inventory["decision"],
        },
        "follower_source_sha256": current_follower_sha,
        "follower_config_source_sha256": current_config_sha,
    }


def find_config_default(config_class: ClassLocation, field_name: str) -> Any:
    for node in config_class.node.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == field_name:
                try:
                    return ast.literal_eval(node.value) if node.value is not None else None
                except (ValueError, TypeError, SyntaxError):
                    return {"python_expression": ast.unparse(node.value)}
        elif isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == field_name for target in node.targets):
                try:
                    return ast.literal_eval(node.value)
                except (ValueError, TypeError, SyntaxError):
                    return {"python_expression": ast.unparse(node.value)}
    raise AuditError(f"config field not found: {field_name}")


def leaves(rows: Iterable[dict[str, Any]]) -> set[str]:
    return {str(row["leaf"]) for row in rows}


def has_call_expression(rows: Iterable[dict[str, Any]], suffix: str) -> bool:
    return any(str(row["expression"]).startswith(suffix) for row in rows)


def analyze_follower(config_source: Path, follower_source: Path) -> dict[str, Any]:
    config_class = find_unique_class(config_source, "SO101FollowerConfig")
    follower_class = find_unique_class(follower_source, "SO101Follower")
    methods = own_methods(follower_class)
    required = {
        "__init__",
        "connect",
        "calibrate",
        "configure",
        "get_observation",
        "send_action",
        "disconnect",
    }
    missing = sorted(required - set(methods))
    if missing:
        raise AuditError(f"SO101Follower required methods missing: {missing}")
    records = {name: method_record(methods[name]) for name in sorted(required)}

    init_calls = records["__init__"]["calls"]
    if EXPECTED_BUS_CLASS not in leaves(init_calls):
        raise AuditError(f"SO101Follower no longer constructs {EXPECTED_BUS_CLASS}")
    init_bad = records["__init__"]["mutating_calls"]
    init_transport = [
        row
        for row in init_calls
        if row["leaf"] in {"connect", "open", "openPort", "VideoCapture"}
    ]
    if init_bad or init_transport:
        raise AuditError(
            f"SO101Follower constructor is not inert: mutators={init_bad}, transport={init_transport}"
        )

    connect_calls = records["connect"]["calls"]
    if not has_call_expression(connect_calls, "self.bus.connect("):
        raise AuditError("SO101Follower.connect no longer calls self.bus.connect")
    if not has_call_expression(connect_calls, "self.configure("):
        raise AuditError("SO101Follower.connect no longer exposes the reviewed configure side effect")
    configure_mutators = records["configure"]["mutating_calls"]
    if not configure_mutators:
        raise AuditError("SO101Follower.configure unexpectedly has no detected motor mutators")
    if not {"write", "configure_motors"}.intersection(leaves(configure_mutators)):
        raise AuditError("SO101Follower.configure mutating register writes were not detected")
    calibration_mutators = records["calibrate"]["mutating_calls"]
    if not calibration_mutators:
        raise AuditError("SO101Follower.calibrate unexpectedly has no detected mutators")

    observation_calls = records["get_observation"]["calls"]
    observation_mutators = records["get_observation"]["mutating_calls"]
    present_reads = [
        row
        for row in observation_calls
        if row["leaf"] == "sync_read" and EXPECTED_REGISTER in row["expression"]
    ]
    if len(present_reads) != 1:
        raise AuditError(
            f"expected one get_observation Present_Position sync_read, found {len(present_reads)}"
        )
    if observation_mutators:
        raise AuditError(f"get_observation contains motor mutators: {observation_mutators}")

    disconnect_default = find_config_default(
        config_class, "disable_torque_on_disconnect"
    )
    if disconnect_default is not True:
        raise AuditError(
            "expected reviewed disable_torque_on_disconnect default True; source changed"
        )
    disconnect_calls = records["disconnect"]["calls"]
    if not has_call_expression(disconnect_calls, "self.bus.disconnect("):
        raise AuditError("SO101Follower.disconnect no longer delegates to bus.disconnect")

    return {
        "follower_class": "SO101Follower",
        "bus_class": EXPECTED_BUS_CLASS,
        "constructor_inert": True,
        "full_follower_connect_read_only": False,
        "full_follower_connect_reason": (
            "connect(calibrate=False) still calls configure(), whose reviewed body changes "
            "torque state and writes motor configuration registers"
        ),
        "full_follower_disconnect_read_only": False,
        "full_follower_disconnect_reason": (
            "SO101Follower.disconnect() forwards the config default "
            "disable_torque_on_disconnect=True"
        ),
        "get_observation_motor_read_only": True,
        "disable_torque_on_disconnect_default": disconnect_default,
        "methods": records,
    }


def argument_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
    return [arg.arg for arg in args]


def guarded_disable_torque(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for child in ast.walk(node):
        if not isinstance(child, ast.If):
            continue
        test_names = {
            item.id for item in ast.walk(child.test) if isinstance(item, ast.Name)
        }
        if "disable_torque" not in test_names:
            continue
        if any(
            isinstance(item, ast.Call) and call_leaf(item) == "disable_torque"
            for statement in child.body
            for item in ast.walk(statement)
        ):
            return True
    return False


def analyze_bus(root: Path) -> dict[str, Any]:
    index, parse_errors = scan_class_index(root)
    if parse_errors:
        raise AuditError(f"motor source parse errors: {parse_errors}")
    bus_class = unique_index_class(index, EXPECTED_BUS_CLASS)
    connect = resolve_method(index, EXPECTED_BUS_CLASS, "connect")
    disconnect = resolve_method(index, EXPECTED_BUS_CLASS, "disconnect")
    sync_read = resolve_method(index, EXPECTED_BUS_CLASS, "sync_read")
    internal_sync_read = resolve_method(index, EXPECTED_BUS_CLASS, "_sync_read")
    records = {
        "connect": method_record(connect),
        "disconnect": method_record(disconnect),
        "sync_read": method_record(sync_read),
        "_sync_read": method_record(internal_sync_read),
    }

    connect_mutators = records["connect"]["mutating_calls"]
    if connect_mutators:
        raise AuditError(f"direct bus.connect contains motor mutators: {connect_mutators}")
    connect_literals = set(records["connect"]["string_literals"])
    if connect_literals.intersection(FORBIDDEN_REGISTER_LITERALS):
        raise AuditError("direct bus.connect references forbidden mutable motor registers")

    read_mutators = [
        row
        for key in ("sync_read", "_sync_read")
        for row in records[key]["mutating_calls"]
    ]
    if read_mutators:
        raise AuditError(f"sync_read chain contains motor mutators: {read_mutators}")
    read_literals = {
        literal
        for key in ("sync_read", "_sync_read")
        for literal in records[key]["string_literals"]
    }
    forbidden_read_literals = sorted(read_literals.intersection(FORBIDDEN_REGISTER_LITERALS))
    if forbidden_read_literals:
        raise AuditError(
            f"sync_read chain references mutable register literals: {forbidden_read_literals}"
        )
    if "disable_torque" not in argument_names(disconnect.node):
        raise AuditError("bus.disconnect no longer accepts disable_torque")
    disconnect_mutators = records["disconnect"]["mutating_calls"]
    torque_calls = [row for row in disconnect_mutators if row["leaf"] == "disable_torque"]
    if torque_calls and not guarded_disable_torque(disconnect.node):
        raise AuditError("bus.disconnect torque mutation is not guarded by disable_torque")
    unexpected_disconnect_mutators = [
        row for row in disconnect_mutators if row["leaf"] != "disable_torque"
    ]
    if unexpected_disconnect_mutators:
        raise AuditError(
            "bus.disconnect contains unexpected mutators: "
            f"{unexpected_disconnect_mutators}"
        )

    return {
        "motors_source_root": str(root),
        "classes_scanned": sum(len(rows) for rows in index.values()),
        "python_files_with_classes": len({str(row.path) for rows in index.values() for row in rows}),
        "selected_bus_class": EXPECTED_BUS_CLASS,
        "selected_bus_class_path": str(bus_class.path),
        "direct_bus_connect_has_motor_mutators": False,
        "sync_read_chain_has_motor_mutators": False,
        "disconnect_disable_torque_parameter": True,
        "disconnect_false_preserves_torque_state": True,
        "methods": records,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    assembler = require_file(args.assembler)
    assembler_report = require_file(args.assembler_report)
    dependency_inventory = require_file(args.dependency_inventory)
    follower_config = require_file(args.follower_config_source)
    follower_source = require_file(args.follower_source)
    motors_root = require_dir(args.motors_source_root)
    output_path = args.output_json.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {output_path}")

    print("===== STATIC NO-HARDWARE / NO-VENDOR-IMPORT BOUNDARY =====")
    static_boundary = verify_static_boundary(Path(__file__).resolve())
    print("source parsed as text; no LeRobot/serial/camera/model imports or calls: PASS")

    print("\n===== VERIFY PREVIOUS FROZEN OFFLINE GATES =====")
    previous = verify_previous_gates(
        assembler,
        assembler_report,
        dependency_inventory,
        follower_source,
        follower_config,
    )
    print("observation assembler byte identity and 16/16 fail-closed report: PASS")
    print("dependency inventory and frozen follower source identities: PASS")

    print("\n===== FULL SO101FOLLOWER API SIDE-EFFECT AUDIT =====")
    follower = analyze_follower(follower_config, follower_source)
    print("SO101Follower constructor: inert before connection: PASS")
    print("SO101Follower.connect(calibrate=False): PROHIBITED for read-only use")
    print("reason: connect still calls configure(), which changes torque/register state")
    print("SO101Follower.disconnect(): PROHIBITED for read-only use")
    print("reason: default disable_torque_on_disconnect=True changes torque state")
    print("get_observation Present_Position branch has no motor mutators: PASS")

    print("\n===== DIRECT MOTOR-BUS READ PATH AUDIT =====")
    bus = analyze_bus(motors_root)
    print(f"resolved bus class: {bus['selected_bus_class']}")
    print("bus.connect: serial transport open only; no motor register mutator found: PASS")
    print("sync_read/_sync_read: read transaction only; no mutator found: PASS")
    print("bus.disconnect(disable_torque=False): torque-preserving close path: PASS")

    approved_sequence = [
        {
            "step": 1,
            "operation": "construct SO101Follower with cameras={} and frozen calibration",
            "hardware_effect": "none",
        },
        {
            "step": 2,
            "operation": "follower.bus.connect()",
            "hardware_effect": "open serial transport only",
        },
        {
            "step": 3,
            "operation": (
                'follower.bus.sync_read("Present_Position", normalize=True, num_retry=3)'
            ),
            "hardware_effect": "transmit motor READ instruction; no register mutation",
        },
        {
            "step": 4,
            "operation": "follower.bus.disconnect(disable_torque=False)",
            "hardware_effect": "close serial transport without changing torque",
        },
    ]
    prohibited = [
        "SO101Follower.connect(...) including calibrate=False",
        "SO101Follower.disconnect()",
        "calibrate / configure / setup_motors",
        "write / sync_write / write_calibration",
        "enable_torque / disable_torque / torque_disabled",
        "send_action / Goal_Position",
    ]
    decision = (
        "READ_ONLY_JOINT_SENSOR_PATH_STATIC_PASS_"
        "BUILD_INERT_LIVE_SENSOR_ADAPTER_NEXT"
    )
    report = {
        "schema_version": 1,
        "scope": {
            "static_source_audit": True,
            "vendor_modules_imported": False,
            "robot_object_created": False,
            "serial_port_opened": False,
            "motor_read_instruction_sent": False,
            "motor_register_write_sent": False,
            "torque_state_changed": False,
            "camera_opened": False,
            "model_loaded": False,
            "command_sent": False,
            "hardware_deployment_authorized": False,
            "live_read_authorized": False,
        },
        "static_boundary": static_boundary,
        "previous_gates": previous,
        "follower_side_effect_audit": follower,
        "direct_bus_read_path_audit": bus,
        "candidate_read_only_sequence": approved_sequence,
        "explicitly_prohibited_operations": prohibited,
        "important_semantics": {
            "read_only_does_not_mean_zero_serial_traffic": True,
            "motor_read_instruction_is_transmitted": True,
            "motor_configuration_registers_are_not_written": True,
            "goal_position_is_not_written": True,
            "torque_state_is_not_changed": True,
        },
        "decision": decision,
        "hardware_deployment_authorized": False,
        "live_read_authorized": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\n===== DECISION =====")
    print(f"decision='{decision}'")
    print("The next adapter may encode only the audited direct-bus sequence.")
    print("LIVE HARDWARE READBACK AND HARDWARE DEPLOYMENT REMAIN BLOCKED.")
    print("NO VENDOR MODULE WAS IMPORTED. NO ROBOT OBJECT WAS CREATED.")
    print("NO SERIAL PORT OR CAMERA WAS OPENED. NO MOTOR READ WAS SENT.")
    print("NO REGISTER, GOAL POSITION, OR TORQUE COMMAND WAS SENT.")
    print("NO MODEL WAS LOADED. NO POLICY INFERENCE RAN.")
    print("\n===== OUTPUT =====")
    print(output_path)
    print("ACT V3 READ-ONLY JOINT SENSOR PATH STATIC AUDIT: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"\nACT V3 READ-ONLY JOINT SENSOR PATH STATIC AUDIT: FAIL: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print("LIVE HARDWARE READBACK AND DEPLOYMENT REMAIN BLOCKED.", file=sys.stderr)
        print("NO VENDOR MODULE WAS IMPORTED. NO ROBOT OBJECT WAS CREATED.", file=sys.stderr)
        print("NO SERIAL PORT OR CAMERA WAS OPENED. NO MOTOR READ WAS SENT.", file=sys.stderr)
        print("NO REGISTER, GOAL POSITION, OR TORQUE COMMAND WAS SENT.", file=sys.stderr)
        raise SystemExit(1)
