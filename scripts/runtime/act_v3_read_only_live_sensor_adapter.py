#!/usr/bin/env python3
"""Define and inertly audit the ACT V3 read-only joint sensor adapter.

The reusable adapter deliberately accepts an already-constructed motor bus.  It
uses only the source-audited sequence below:

* ``bus.connect()``
* ``bus.sync_read("Present_Position", normalize=True, num_retry=3)``
* ``bus.disconnect(disable_torque=False)``

It never calls ``SO101Follower.connect()``, ``SO101Follower.disconnect()``,
calibration/configuration APIs, torque APIs, register-write APIs, action APIs,
camera APIs, or policy inference.  A successful read is converted into the
``JointStatePacket`` expected by the reviewed live observation assembler.

The CLI in this file is intentionally inert: it loads only the pure guard and
observation packet definitions and exercises the adapter with fake buses.  It
does not import LeRobot, construct a real robot, open a serial port, read a
motor, open a camera, load a policy, or send a command.  A later, separately
reviewed launcher may inject a real bus only after explicit live-read approval.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Protocol, Sequence


EXPECTED_PATH_AUDIT_SHA256 = (
    "a46439fa977d5cc66040c4b7a570332b7fe1dba9a0adf13674938101443f3850"
)
EXPECTED_PATH_AUDIT_DECISION = (
    "READ_ONLY_JOINT_SENSOR_PATH_STATIC_PASS_"
    "BUILD_INERT_LIVE_SENSOR_ADAPTER_NEXT"
)
EXPECTED_GUARD_SHA256 = (
    "e06e785071ee3ded95e89d670978bcb4a8caf104b2fc69fc87de37cb6fff6d15"
)
EXPECTED_ASSEMBLER_SHA256 = (
    "866a53d8b4ce1473344779c493a955d03dafa485115fab34a4ffc929ff31d424"
)
MOTOR_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
READ_REGISTER = "Present_Position"
READ_NORMALIZE = True
READ_NUM_RETRY = 3
DEFAULT_MAX_READ_DURATION_SECONDS = 0.100


class SensorAdapterError(RuntimeError):
    """The adapter lifecycle, timing, schema, or sensor contract failed."""


class AdapterState(str, Enum):
    NEW = "NEW"
    OPEN = "OPEN"
    FAULTED = "FAULTED"
    CLOSED = "CLOSED"


class ReadOnlyBus(Protocol):
    @property
    def is_connected(self) -> bool: ...

    def connect(self) -> None: ...

    def sync_read(
        self,
        data_name: str,
        motors: str | list[str] | None = None,
        *,
        normalize: bool = True,
        num_retry: int = 1,
    ) -> Mapping[str, Any]: ...

    def disconnect(self, disable_torque: bool = True) -> None: ...


@dataclass(frozen=True)
class JointReadMetrics:
    sequence: int
    read_started_monotonic_s: float
    read_completed_monotonic_s: float
    read_duration_ms: float
    motor_order: tuple[str, ...]


@dataclass(frozen=True)
class JointReadResult:
    packet: Any
    metrics: JointReadMetrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-audit", type=Path, required=True)
    parser.add_argument("--path-audit-report", type=Path, required=True)
    parser.add_argument("--guard-core", type=Path, required=True)
    parser.add_argument("--assembler", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
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
        raise SensorAdapterError(f"invalid JSON: {path}: {exc}") from exc


def load_module(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SensorAdapterError(f"cannot load module spec: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def finite_float(label: str, value: Any) -> float:
    if isinstance(value, bool):
        raise SensorAdapterError(f"{label} must not be bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SensorAdapterError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise SensorAdapterError(f"{label} must be finite")
    return result


def positive_finite(label: str, value: Any) -> float:
    result = finite_float(label, value)
    if result <= 0.0:
        raise SensorAdapterError(f"{label} must be positive")
    return result


class ReadOnlyJointSensorAdapter:
    """Fail-closed lifecycle wrapper around the audited direct-bus read path."""

    def __init__(
        self,
        *,
        bus: ReadOnlyBus,
        validate_positions: Callable[[Sequence[float]], Sequence[float]],
        packet_factory: Callable[..., Any],
        monotonic: Callable[[], float],
        max_read_duration_seconds: float = DEFAULT_MAX_READ_DURATION_SECONDS,
    ) -> None:
        self._bus = bus
        self._validate_positions = validate_positions
        self._packet_factory = packet_factory
        self._monotonic = monotonic
        self._max_read_duration_seconds = positive_finite(
            "max_read_duration_seconds", max_read_duration_seconds
        )
        self._state = AdapterState.NEW
        self._sequence = 0
        self._last_clock: float | None = None
        for method_name in ("connect", "sync_read", "disconnect"):
            if not callable(getattr(bus, method_name, None)):
                raise SensorAdapterError(f"bus is missing callable {method_name}()")
        if bool(getattr(bus, "is_connected", False)):
            raise SensorAdapterError("bus must be disconnected before adapter construction")

    @property
    def state(self) -> AdapterState:
        return self._state

    @property
    def sequence(self) -> int:
        return self._sequence

    def _bus_connected(self) -> bool:
        value = getattr(self._bus, "is_connected", None)
        if not isinstance(value, bool):
            raise SensorAdapterError("bus.is_connected must be bool")
        return value

    def _clock(self, label: str) -> float:
        value = finite_float(label, self._monotonic())
        if value < 0.0:
            raise SensorAdapterError(f"{label} must be nonnegative")
        if self._last_clock is not None and value < self._last_clock:
            raise SensorAdapterError(
                f"monotonic clock reversed: previous={self._last_clock}, current={value}"
            )
        return value

    def _close_bus_preserving_torque(self) -> None:
        self._bus.disconnect(disable_torque=False)

    def _fault(self, original: BaseException) -> SensorAdapterError:
        self._state = AdapterState.FAULTED
        cleanup_error: BaseException | None = None
        try:
            if self._bus_connected():
                self._close_bus_preserving_torque()
        except BaseException as exc:  # Preserve both the primary and cleanup failure.
            cleanup_error = exc
        message = f"read-only joint sensor adapter fault: {type(original).__name__}: {original}"
        if cleanup_error is not None:
            message += (
                "; torque-preserving serial close also failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        return SensorAdapterError(message)

    def open(self) -> None:
        if self._state is not AdapterState.NEW:
            raise SensorAdapterError(f"open requires NEW state, got {self._state.value}")
        try:
            if self._bus_connected():
                raise SensorAdapterError("bus became connected before open")
            self._bus.connect()
            if not self._bus_connected():
                raise SensorAdapterError("bus.connect returned without a connected bus")
        except BaseException as exc:
            raise self._fault(exc) from exc
        self._state = AdapterState.OPEN

    def _ordered_positions(self, values: Any) -> tuple[float, ...]:
        if not isinstance(values, Mapping):
            raise SensorAdapterError("Present_Position sync_read must return a mapping")
        actual_keys = set(values)
        expected_keys = set(MOTOR_ORDER)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise SensorAdapterError(
                f"Present_Position keys mismatch: missing={missing}, extra={extra}"
            )
        vector = tuple(
            finite_float(f"Present_Position[{motor}]", values[motor])
            for motor in MOTOR_ORDER
        )
        validated = tuple(float(value) for value in self._validate_positions(vector))
        if len(validated) != len(MOTOR_ORDER):
            raise SensorAdapterError("position validator returned wrong dimension")
        if any(not math.isfinite(value) for value in validated):
            raise SensorAdapterError("position validator returned nonfinite values")
        return validated

    def read(self) -> JointReadResult:
        if self._state is not AdapterState.OPEN:
            raise SensorAdapterError(f"read requires OPEN state, got {self._state.value}")
        try:
            if not self._bus_connected():
                raise SensorAdapterError("bus disconnected before read")
            started = self._clock("read_started_monotonic_s")
            values = self._bus.sync_read(
                READ_REGISTER,
                normalize=READ_NORMALIZE,
                num_retry=READ_NUM_RETRY,
            )
            completed = self._clock("read_completed_monotonic_s")
            duration = completed - started
            if duration > self._max_read_duration_seconds:
                raise SensorAdapterError(
                    "Present_Position read exceeded deadline: "
                    f"{duration:.6f}s > {self._max_read_duration_seconds:.6f}s"
                )
            actual_q = self._ordered_positions(values)
            next_sequence = self._sequence + 1
            packet = self._packet_factory(
                actual_q=actual_q,
                read_monotonic_s=completed,
            )
            metrics = JointReadMetrics(
                sequence=next_sequence,
                read_started_monotonic_s=started,
                read_completed_monotonic_s=completed,
                read_duration_ms=duration * 1000.0,
                motor_order=MOTOR_ORDER,
            )
        except BaseException as exc:
            raise self._fault(exc) from exc
        self._sequence = next_sequence
        self._last_clock = completed
        return JointReadResult(packet=packet, metrics=metrics)

    def close(self) -> None:
        if self._state is AdapterState.CLOSED:
            return
        if self._state is AdapterState.NEW:
            self._state = AdapterState.CLOSED
            return
        if self._state is AdapterState.FAULTED:
            try:
                if self._bus_connected():
                    self._close_bus_preserving_torque()
            except BaseException as exc:
                raise SensorAdapterError(
                    "torque-preserving close retry failed while adapter was FAULTED: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            return
        try:
            if not self._bus_connected():
                raise SensorAdapterError("bus disconnected before adapter close")
            self._close_bus_preserving_torque()
            if self._bus_connected():
                raise SensorAdapterError("bus remained connected after disconnect(false)")
        except BaseException as exc:
            self._state = AdapterState.FAULTED
            raise SensorAdapterError(
                f"torque-preserving adapter close failed: {type(exc).__name__}: {exc}"
            ) from exc
        self._state = AdapterState.CLOSED

    def __enter__(self) -> "ReadOnlyJointSensorAdapter":
        self.open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.close()
        return False


def verify_static_adapter_boundary(path: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    imports: set[str] = set()
    identifiers: set[str] = set()
    attributes: set[str] = set()
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
        elif isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, ast.Attribute):
            attributes.add(node.attr)
        elif isinstance(node, ast.Call):
            calls.append(node)

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
    bad_identifiers = sorted(
        {
            "SO101Follower",
            "SO101FollowerConfig",
            "VideoCapture",
            "ACTPolicy",
        }.intersection(identifiers)
    )
    bad_attributes = sorted(
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

    connect_calls: list[str] = []
    read_calls: list[str] = []
    disconnect_calls: list[str] = []
    for call in calls:
        leaf = call.func.attr if isinstance(call.func, ast.Attribute) else None
        expression = ast.unparse(call)
        if leaf == "connect":
            connect_calls.append(expression)
        elif leaf == "sync_read":
            read_calls.append(expression)
        elif leaf == "disconnect":
            disconnect_calls.append(expression)
    expected_connect = ["self._bus.connect()"]
    expected_read = [
        "self._bus.sync_read(READ_REGISTER, normalize=READ_NORMALIZE, num_retry=READ_NUM_RETRY)"
    ]
    expected_disconnect = ["self._bus.disconnect(disable_torque=False)"]
    if (
        bad_imports
        or bad_identifiers
        or bad_attributes
        or connect_calls != expected_connect
        or read_calls != expected_read
        or disconnect_calls != expected_disconnect
    ):
        raise SensorAdapterError(
            "inert adapter static boundary failed: "
            f"imports={bad_imports}, identifiers={bad_identifiers}, "
            f"attributes={bad_attributes}, connect={connect_calls}, "
            f"read={read_calls}, disconnect={disconnect_calls}"
        )
    return {
        "source": str(path),
        "source_sha256": sha256_file(path),
        "vendor_or_serial_imports": False,
        "robot_constructor_calls": False,
        "camera_or_model_calls": False,
        "motor_mutator_calls": False,
        "allowed_connect_calls": connect_calls,
        "allowed_read_calls": read_calls,
        "allowed_disconnect_calls": disconnect_calls,
        "pass": True,
    }


def verify_previous_path_audit(path: Path, report_path: Path) -> dict[str, Any]:
    actual_sha = sha256_file(path)
    if actual_sha != EXPECTED_PATH_AUDIT_SHA256:
        raise SensorAdapterError(
            f"path audit SHA256 mismatch: expected {EXPECTED_PATH_AUDIT_SHA256}, got {actual_sha}"
        )
    report = load_json(report_path)
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise SensorAdapterError("unexpected path audit report schema")
    if report.get("decision") != EXPECTED_PATH_AUDIT_DECISION:
        raise SensorAdapterError("path audit report decision mismatch")
    if report.get("hardware_deployment_authorized") is not False:
        raise SensorAdapterError("path audit unexpectedly authorized deployment")
    if report.get("live_read_authorized") is not False:
        raise SensorAdapterError("path audit unexpectedly authorized live read")
    scope = report.get("scope")
    if not isinstance(scope, dict):
        raise SensorAdapterError("path audit scope is missing")
    required_false = (
        "vendor_modules_imported",
        "robot_object_created",
        "serial_port_opened",
        "motor_read_instruction_sent",
        "motor_register_write_sent",
        "torque_state_changed",
        "camera_opened",
        "model_loaded",
        "command_sent",
        "hardware_deployment_authorized",
        "live_read_authorized",
    )
    invalid = [name for name in required_false if scope.get(name) is not False]
    if invalid:
        raise SensorAdapterError(f"path audit scope flags are not false: {invalid}")
    semantics = report.get("important_semantics")
    if not isinstance(semantics, dict):
        raise SensorAdapterError("path audit important_semantics missing")
    expected_true = (
        "read_only_does_not_mean_zero_serial_traffic",
        "motor_read_instruction_is_transmitted",
        "motor_configuration_registers_are_not_written",
        "goal_position_is_not_written",
        "torque_state_is_not_changed",
    )
    invalid_semantics = [name for name in expected_true if semantics.get(name) is not True]
    if invalid_semantics:
        raise SensorAdapterError(
            f"path audit read-only semantics mismatch: {invalid_semantics}"
        )
    return {
        "path": str(path),
        "sha256": actual_sha,
        "report": str(report_path),
        "decision": report["decision"],
        "candidate_sequence": report.get("candidate_read_only_sequence"),
    }


class ScriptedClock:
    def __init__(self, values: Sequence[float]) -> None:
        self._values = list(values)

    def __call__(self) -> float:
        if not self._values:
            raise RuntimeError("scripted clock exhausted")
        return self._values.pop(0)


class FakeReadOnlyBus:
    def __init__(
        self,
        *,
        positions: Mapping[str, Any],
        connect_error: BaseException | None = None,
        connect_error_after_open: bool = False,
        read_error: BaseException | None = None,
        disconnect_error: BaseException | None = None,
    ) -> None:
        self.positions = dict(positions)
        self.connect_error = connect_error
        self.connect_error_after_open = connect_error_after_open
        self.read_error = read_error
        self.disconnect_error = disconnect_error
        self.is_connected = False
        self.events: list[dict[str, Any]] = []
        self.torque_marker = "UNCHANGED"

    def connect(self) -> None:
        self.events.append({"operation": "connect"})
        if self.connect_error is not None:
            if self.connect_error_after_open:
                self.is_connected = True
            raise self.connect_error
        self.is_connected = True

    def sync_read(
        self,
        data_name: str,
        motors: str | list[str] | None = None,
        *,
        normalize: bool = True,
        num_retry: int = 1,
    ) -> Mapping[str, Any]:
        self.events.append(
            {
                "operation": "sync_read",
                "data_name": data_name,
                "motors": motors,
                "normalize": normalize,
                "num_retry": num_retry,
            }
        )
        if not self.is_connected:
            raise RuntimeError("fake bus is disconnected")
        if self.read_error is not None:
            raise self.read_error
        return dict(self.positions)

    def disconnect(self, disable_torque: bool = True) -> None:
        self.events.append(
            {
                "operation": "disconnect",
                "disable_torque": disable_torque,
            }
        )
        if disable_torque:
            self.torque_marker = "CHANGED"
        if self.disconnect_error is not None:
            raise self.disconnect_error
        self.is_connected = False


def expect_sensor_error(label: str, function: Callable[[], Any]) -> str:
    try:
        function()
    except SensorAdapterError as exc:
        return str(exc)
    raise AssertionError(f"expected SensorAdapterError: {label}")


def run_inert_self_audit(
    *,
    guard_module: ModuleType,
    assembler_module: ModuleType,
) -> dict[str, Any]:
    limits = guard_module.RuntimeLimits.frozen_v3()
    if tuple(limits.motor_order) != MOTOR_ORDER:
        raise AssertionError("guard motor order mismatch")
    guard = guard_module.DeltaCommandGuard(limits, limits.home_command)
    home_positions = dict(zip(MOTOR_ORDER, limits.home_command, strict=True))

    def make_adapter(
        bus: FakeReadOnlyBus,
        clock_values: Sequence[float],
        *,
        max_duration: float = DEFAULT_MAX_READ_DURATION_SECONDS,
    ) -> ReadOnlyJointSensorAdapter:
        return ReadOnlyJointSensorAdapter(
            bus=bus,
            validate_positions=guard.validate_measured_positions,
            packet_factory=assembler_module.JointStatePacket,
            monotonic=ScriptedClock(clock_values),
            max_read_duration_seconds=max_duration,
        )

    tests: dict[str, bool] = {}
    evidence: dict[str, Any] = {}

    bus = FakeReadOnlyBus(positions=home_positions)
    adapter = make_adapter(bus, [10.000, 10.020])
    adapter.open()
    result = adapter.read()
    adapter.close()
    expected_events = [
        {"operation": "connect"},
        {
            "operation": "sync_read",
            "data_name": READ_REGISTER,
            "motors": None,
            "normalize": True,
            "num_retry": 3,
        },
        {"operation": "disconnect", "disable_torque": False},
    ]
    if bus.events != expected_events:
        raise AssertionError(f"valid lifecycle event mismatch: {bus.events}")
    if bus.torque_marker != "UNCHANGED" or bus.is_connected:
        raise AssertionError("valid lifecycle changed torque or left bus connected")
    if tuple(result.packet.actual_q) != tuple(limits.home_command):
        raise AssertionError("JointStatePacket actual_q mismatch")
    if result.packet.read_monotonic_s != 10.020:
        raise AssertionError("JointStatePacket timestamp mismatch")
    if result.metrics.sequence != 1 or adapter.sequence != 1:
        raise AssertionError("successful sequence mismatch")
    if adapter.state is not AdapterState.CLOSED:
        raise AssertionError("adapter did not reach CLOSED state")
    tests["exact_connect_read_disconnect_false_lifecycle"] = True
    tests["joint_packet_schema_and_sequence"] = True
    tests["torque_marker_unchanged"] = True
    evidence["valid_events"] = expected_events
    evidence["valid_metrics"] = asdict(result.metrics)

    bus = FakeReadOnlyBus(positions=home_positions)
    adapter = make_adapter(bus, [])
    expect_sensor_error("read_before_open", adapter.read)
    if bus.events or bus.is_connected:
        raise AssertionError("read-before-open touched bus")
    tests["read_before_open_fails_without_bus_access"] = True

    bus = FakeReadOnlyBus(positions=home_positions)
    adapter = make_adapter(bus, [])
    adapter.open()
    expect_sensor_error("duplicate_open", adapter.open)
    adapter.close()
    if bus.events != [
        {"operation": "connect"},
        {"operation": "disconnect", "disable_torque": False},
    ]:
        raise AssertionError("duplicate open caused unexpected bus access")
    tests["duplicate_open_fails_without_second_connect"] = True

    bus = FakeReadOnlyBus(positions=home_positions)
    adapter = make_adapter(bus, [])
    adapter.close()
    adapter.close()
    if bus.events or adapter.state is not AdapterState.CLOSED:
        raise AssertionError("close-before-open was not inert/idempotent")
    tests["close_before_open_and_duplicate_close_are_inert"] = True

    bus = FakeReadOnlyBus(positions=home_positions)
    adapter = make_adapter(bus, [])
    adapter.close()
    expect_sensor_error("read_after_close", adapter.read)
    if bus.events:
        raise AssertionError("read-after-close touched bus")
    tests["read_after_close_fails_without_bus_access"] = True

    malformed_cases: list[tuple[str, dict[str, Any]]] = []
    missing = dict(home_positions)
    missing.pop("gripper")
    malformed_cases.append(("missing_motor", missing))
    extra = dict(home_positions)
    extra["unexpected"] = 0.0
    malformed_cases.append(("extra_motor", extra))
    bool_value = dict(home_positions)
    bool_value["shoulder_pan"] = True
    malformed_cases.append(("boolean_value", bool_value))
    nan_value = dict(home_positions)
    nan_value["shoulder_lift"] = math.nan
    malformed_cases.append(("nonfinite_value", nan_value))
    out_of_range = dict(home_positions)
    out_of_range["elbow_flex"] = 106.0
    malformed_cases.append(("outside_calibrated_sensor_range", out_of_range))
    for label, positions in malformed_cases:
        bus = FakeReadOnlyBus(positions=positions)
        adapter = make_adapter(bus, [20.000, 20.010])
        adapter.open()
        expect_sensor_error(label, adapter.read)
        if adapter.state is not AdapterState.FAULTED:
            raise AssertionError(f"{label} did not fault adapter")
        if bus.is_connected or bus.torque_marker != "UNCHANGED":
            raise AssertionError(f"{label} cleanup was not torque-preserving")
        if adapter.sequence != 0:
            raise AssertionError(f"{label} committed a failed sequence")
        if bus.events[-1] != {"operation": "disconnect", "disable_torque": False}:
            raise AssertionError(f"{label} did not use disconnect(false)")
        tests[f"{label}_fails_closed"] = True

    bus = FakeReadOnlyBus(positions=home_positions)
    adapter = make_adapter(bus, [30.000, 30.101], max_duration=0.100)
    adapter.open()
    expect_sensor_error("read_deadline", adapter.read)
    if bus.is_connected or adapter.sequence != 0:
        raise AssertionError("slow read did not fail transactionally")
    tests["read_deadline_fails_closed"] = True

    bus = FakeReadOnlyBus(positions=home_positions)
    adapter = make_adapter(bus, [40.000, 40.010, 40.009, 40.020])
    adapter.open()
    adapter.read()
    expect_sensor_error("clock_reversal", adapter.read)
    if bus.is_connected or adapter.sequence != 1:
        raise AssertionError("clock reversal committed state or left bus open")
    tests["monotonic_clock_reversal_fails_closed"] = True

    bus = FakeReadOnlyBus(
        positions=home_positions,
        read_error=OSError("synthetic read failure"),
    )
    adapter = make_adapter(bus, [50.000])
    adapter.open()
    expect_sensor_error("bus_read_exception", adapter.read)
    if bus.is_connected or bus.torque_marker != "UNCHANGED":
        raise AssertionError("read exception cleanup was not torque-preserving")
    tests["bus_read_exception_disconnects_false"] = True

    bus = FakeReadOnlyBus(
        positions=home_positions,
        connect_error=OSError("synthetic partial-open failure"),
        connect_error_after_open=True,
    )
    adapter = make_adapter(bus, [])
    expect_sensor_error("partial_open_exception", adapter.open)
    if bus.is_connected or bus.torque_marker != "UNCHANGED":
        raise AssertionError("partial-open cleanup was not torque-preserving")
    if bus.events != [
        {"operation": "connect"},
        {"operation": "disconnect", "disable_torque": False},
    ]:
        raise AssertionError("partial-open cleanup sequence mismatch")
    tests["partial_open_failure_disconnects_false"] = True

    bus = FakeReadOnlyBus(
        positions=home_positions,
        disconnect_error=OSError("synthetic close failure"),
    )
    adapter = make_adapter(bus, [])
    adapter.open()
    expect_sensor_error("disconnect_exception", adapter.close)
    if adapter.state is not AdapterState.FAULTED:
        raise AssertionError("disconnect exception did not fault adapter")
    if bus.torque_marker != "UNCHANGED":
        raise AssertionError("disconnect exception changed torque marker")
    tests["disconnect_failure_faults_without_torque_change"] = True

    return {
        "tests": tests,
        "tests_passed": sum(tests.values()),
        "tests_total": len(tests),
        "evidence": evidence,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    path_audit = require_file(args.path_audit)
    path_audit_report = require_file(args.path_audit_report)
    guard_path = require_file(args.guard_core)
    assembler_path = require_file(args.assembler)
    contract_path = require_file(args.contract)
    calibration_path = require_file(args.calibration)
    output_path = args.output_json.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {output_path}")

    print("===== STATIC ADAPTER CAPABILITY BOUNDARY =====")
    static_boundary = verify_static_adapter_boundary(Path(__file__).resolve())
    print("no LeRobot/serial/camera/model imports or forbidden motor calls: PASS")
    print("allowed bus call sites are exactly connect/read/disconnect(false): PASS")

    print("\n===== VERIFY REVIEWED READ-ONLY PATH =====")
    path_gate = verify_previous_path_audit(path_audit, path_audit_report)
    print("static joint-path script identity and PASS report: PASS")
    print("live read and hardware deployment authorization remain false: PASS")

    print("\n===== VERIFY PURE RUNTIME DEPENDENCIES =====")
    guard_sha = sha256_file(guard_path)
    assembler_sha = sha256_file(assembler_path)
    if guard_sha != EXPECTED_GUARD_SHA256:
        raise SensorAdapterError(
            f"guard SHA256 mismatch: expected {EXPECTED_GUARD_SHA256}, got {guard_sha}"
        )
    if assembler_sha != EXPECTED_ASSEMBLER_SHA256:
        raise SensorAdapterError(
            f"assembler SHA256 mismatch: expected {EXPECTED_ASSEMBLER_SHA256}, got {assembler_sha}"
        )
    guard_module = load_module(guard_path, "act_v3_inert_sensor_guard")
    assembler_module = load_module(assembler_path, "act_v3_inert_sensor_assembler")
    frozen_inputs = guard_module.verify_frozen_inputs(contract_path, calibration_path)
    print("guard, assembler, runtime contract and calibration identities: PASS")

    print("\n===== FAKE-BUS LIFECYCLE AND FAIL-CLOSED SELF-AUDIT =====")
    self_audit = run_inert_self_audit(
        guard_module=guard_module,
        assembler_module=assembler_module,
    )
    print(
        f"inert adapter tests: {self_audit['tests_passed']}/"
        f"{self_audit['tests_total']} PASS"
    )
    print("valid path emitted exactly connect/read/disconnect(false): PASS")
    print("all read/schema/timing failures closed transport without torque mutation: PASS")

    decision = (
        "READ_ONLY_LIVE_SENSOR_ADAPTER_INERT_PASS_"
        "PREPARE_EXPLICIT_ONE_SHOT_LIVE_READ_NEXT"
    )
    report = {
        "schema_version": 1,
        "scope": {
            "inert_fake_bus_only": True,
            "vendor_modules_imported": False,
            "real_bus_constructed": False,
            "robot_object_created": False,
            "serial_port_opened": False,
            "motor_read_instruction_sent": False,
            "motor_register_write_sent": False,
            "torque_state_changed": False,
            "camera_opened": False,
            "model_loaded": False,
            "policy_inference_run": False,
            "command_sent": False,
            "live_read_authorized": False,
            "hardware_deployment_authorized": False,
        },
        "static_adapter_boundary": static_boundary,
        "reviewed_path_gate": path_gate,
        "pure_runtime_dependencies": {
            "guard_core": {"path": str(guard_path), "sha256": guard_sha},
            "assembler": {"path": str(assembler_path), "sha256": assembler_sha},
            "frozen_inputs": frozen_inputs,
        },
        "adapter_contract": {
            "motor_order": list(MOTOR_ORDER),
            "read_register": READ_REGISTER,
            "normalize": READ_NORMALIZE,
            "num_retry": READ_NUM_RETRY,
            "max_read_duration_seconds": DEFAULT_MAX_READ_DURATION_SECONDS,
            "disconnect_disable_torque": False,
            "position_validation": "frozen runtime guard calibrated range plus tolerance",
            "packet_type": "JointStatePacket",
            "transactional_sequence_commit": True,
        },
        "self_audit": self_audit,
        "important_semantics": {
            "this_cli_was_inert": True,
            "future_live_read_will_open_serial": True,
            "future_live_read_will_transmit_motor_read_instruction": True,
            "future_live_read_must_not_write_registers": True,
            "future_live_read_must_not_change_torque": True,
            "future_live_read_requires_separate_explicit_approval": True,
        },
        "decision": decision,
        "live_read_authorized": False,
        "hardware_deployment_authorized": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\n===== DECISION =====")
    print(f"decision='{decision}'")
    print("This is the final non-hardware joint-read gate.")
    print("ONE-SHOT LIVE READ AND HARDWARE DEPLOYMENT REMAIN BLOCKED.")
    print("NO LEROBOT VENDOR MODULE WAS IMPORTED. NO REAL BUS WAS CONSTRUCTED.")
    print("NO ROBOT OBJECT WAS CREATED. NO SERIAL PORT OR CAMERA WAS OPENED.")
    print("NO MOTOR READ, REGISTER WRITE, TORQUE CHANGE, OR ACTION WAS SENT.")
    print("NO MODEL WAS LOADED. NO POLICY INFERENCE RAN.")
    print("\n===== OUTPUT =====")
    print(output_path)
    print("ACT V3 READ-ONLY LIVE SENSOR ADAPTER INERT AUDIT: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"\nACT V3 READ-ONLY LIVE SENSOR ADAPTER INERT AUDIT: FAIL: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print("ONE-SHOT LIVE READ AND DEPLOYMENT REMAIN BLOCKED.", file=sys.stderr)
        print("NO REAL BUS OR ROBOT OBJECT WAS CREATED.", file=sys.stderr)
        print("NO SERIAL PORT OR CAMERA WAS OPENED. NO MOTOR READ WAS SENT.", file=sys.stderr)
        print("NO REGISTER, TORQUE, OR ACTION COMMAND WAS SENT.", file=sys.stderr)
        raise SystemExit(1)
