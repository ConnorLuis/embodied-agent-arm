#!/usr/bin/env python3
"""Pure, hardware-free runtime guard for the frozen ACT V3 delta policy.

This module deliberately imports no LeRobot, Torch, camera, or serial package.
It owns only deterministic numeric safety logic:

1. validate the 6-D measured joint vector;
2. construct the 18-D policy state
   [actual_q, previous_sent_command, previous_sent_delta];
3. rate-limit a predicted command delta;
4. recursively decode it from the last command that was actually sent;
5. project the decoded command into the absolute soft workspace;
6. monitor persistent command/readback tracking error.

Running this file performs an offline self-audit. It never constructs a robot,
opens a serial port, reads a camera, loads a model, or sends a command. A PASS
therefore validates only this pure guard core; it is not hardware authorization.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence


MOTOR_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# Frozen contract copied by the user and already audited offline.
FROZEN_CONTRACT_SHA256 = (
    "ff93a28d0698c1367431215185159797b77847db06fd3ea29bdcc1aa916e7083"
)

# Exact follower calibration values shown in the reviewed local source scan.
# JSON formatting is allowed to differ; the semantic fields must match exactly.
EXPECTED_FOLLOWER_CALIBRATION = {
    "shoulder_pan": {
        "id": 1,
        "drive_mode": 0,
        "homing_offset": 458,
        "range_min": 684,
        "range_max": 3294,
    },
    "shoulder_lift": {
        "id": 2,
        "drive_mode": 0,
        "homing_offset": 1010,
        "range_min": 886,
        "range_max": 3287,
    },
    "elbow_flex": {
        "id": 3,
        "drive_mode": 0,
        "homing_offset": -1080,
        "range_min": 915,
        "range_max": 3126,
    },
    "wrist_flex": {
        "id": 4,
        "drive_mode": 0,
        "homing_offset": -181,
        "range_min": 814,
        "range_max": 3182,
    },
    "wrist_roll": {
        "id": 5,
        "drive_mode": 0,
        "homing_offset": -83,
        "range_min": 93,
        "range_max": 3955,
    },
    "gripper": {
        "id": 6,
        "drive_mode": 0,
        "homing_offset": -35,
        "range_min": 1479,
        "range_max": 2993,
    },
}


class GuardContractError(RuntimeError):
    """Fail-closed input, state, or invariant violation."""


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _finite_vector(label: str, values: Sequence[float], size: int = 6) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise GuardContractError(f"{label} must be a numeric sequence")
    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise GuardContractError(f"{label} is not a numeric sequence") from exc
    if len(vector) != size:
        raise GuardContractError(f"{label} must have {size} values, got {len(vector)}")
    invalid = [index for index, value in enumerate(vector) if not math.isfinite(value)]
    if invalid:
        raise GuardContractError(f"{label} has non-finite values at indices {invalid}")
    return vector


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class RuntimeLimits:
    motor_order: tuple[str, ...]
    max_delta_per_command: tuple[float, ...]
    soft_limits: tuple[tuple[float, float], ...]
    calibrated_ranges: tuple[tuple[float, float], ...]
    sensor_tolerance: float
    tracking_error_limits: tuple[float, ...]
    tracking_timeout_seconds: float
    home_command: tuple[float, ...]

    def __post_init__(self) -> None:
        size = len(self.motor_order)
        fields = {
            "max_delta_per_command": self.max_delta_per_command,
            "soft_limits": self.soft_limits,
            "calibrated_ranges": self.calibrated_ranges,
            "tracking_error_limits": self.tracking_error_limits,
            "home_command": self.home_command,
        }
        for name, value in fields.items():
            if len(value) != size:
                raise GuardContractError(f"{name} length does not match motor_order")
        if tuple(self.motor_order) != MOTOR_ORDER:
            raise GuardContractError("unexpected motor order")
        if self.sensor_tolerance < 0.0 or not math.isfinite(self.sensor_tolerance):
            raise GuardContractError("invalid sensor tolerance")
        if self.tracking_timeout_seconds <= 0.0 or not math.isfinite(
            self.tracking_timeout_seconds
        ):
            raise GuardContractError("invalid tracking timeout")
        for index, motor in enumerate(self.motor_order):
            rate = float(self.max_delta_per_command[index])
            soft_min, soft_max = self.soft_limits[index]
            calibrated_min, calibrated_max = self.calibrated_ranges[index]
            track = float(self.tracking_error_limits[index])
            home = float(self.home_command[index])
            if rate <= 0.0 or not math.isfinite(rate):
                raise GuardContractError(f"invalid rate limit for {motor}")
            if not soft_min < soft_max:
                raise GuardContractError(f"invalid soft limits for {motor}")
            if not calibrated_min < calibrated_max:
                raise GuardContractError(f"invalid calibrated range for {motor}")
            if track <= 0.0 or not math.isfinite(track):
                raise GuardContractError(f"invalid tracking limit for {motor}")
            if not soft_min <= home <= soft_max:
                raise GuardContractError(f"Home is outside soft limits for {motor}")

    @classmethod
    def frozen_v3(cls) -> "RuntimeLimits":
        return cls(
            motor_order=MOTOR_ORDER,
            max_delta_per_command=(1.5, 1.2, 1.2, 1.5, 2.0, 2.5),
            soft_limits=(
                (-60.0, 60.0),
                (-60.0, 65.0),
                (-65.0, 65.0),
                (-60.0, 60.0),
                (-90.0, 90.0),
                (5.0, 95.0),
            ),
            calibrated_ranges=(
                (-100.0, 100.0),
                (-100.0, 100.0),
                (-100.0, 100.0),
                (-100.0, 100.0),
                (-100.0, 100.0),
                (0.0, 100.0),
            ),
            sensor_tolerance=5.0,
            tracking_error_limits=(12.0, 12.0, 12.0, 12.0, 12.0, 15.0),
            tracking_timeout_seconds=2.0,
            home_command=(
                12.994799613952637,
                -31.88789939880371,
                48.33190155029297,
                43.222900390625,
                0.36250001192092896,
                27.65489959716797,
            ),
        )

    def to_jsonable(self) -> dict[str, object]:
        return {
            "motor_order": list(self.motor_order),
            "max_delta_per_command": dict(
                zip(self.motor_order, self.max_delta_per_command, strict=True)
            ),
            "soft_limits": {
                motor: list(bounds)
                for motor, bounds in zip(self.motor_order, self.soft_limits, strict=True)
            },
            "calibrated_ranges": {
                motor: list(bounds)
                for motor, bounds in zip(
                    self.motor_order, self.calibrated_ranges, strict=True
                )
            },
            "sensor_tolerance": self.sensor_tolerance,
            "tracking_error_limits": dict(
                zip(self.motor_order, self.tracking_error_limits, strict=True)
            ),
            "tracking_timeout_seconds": self.tracking_timeout_seconds,
            "home_command": list(self.home_command),
        }


@dataclass(frozen=True)
class GuardStep:
    raw_delta: tuple[float, ...]
    rate_limited_delta: tuple[float, ...]
    previous_command: tuple[float, ...]
    candidate_after_rate_limit: tuple[float, ...]
    sent_command: tuple[float, ...]
    sent_delta: tuple[float, ...]
    rate_clipped: tuple[bool, ...]
    soft_clipped: tuple[bool, ...]
    inherited_boundary_projection: tuple[bool, ...]
    new_boundary_crossing: tuple[bool, ...]

    @property
    def any_rate_clip(self) -> bool:
        return any(self.rate_clipped)

    @property
    def any_soft_clip(self) -> bool:
        return any(self.soft_clipped)


class DeltaCommandGuard:
    """Stateful delta decoder whose state is the last command actually sent."""

    def __init__(
        self,
        limits: RuntimeLimits,
        initial_command: Sequence[float],
        initial_delta: Sequence[float] | None = None,
    ) -> None:
        self.limits = limits
        command = _finite_vector("initial_command", initial_command)
        delta = _finite_vector(
            "initial_delta", initial_delta if initial_delta is not None else (0.0,) * 6
        )
        self._assert_command_within_soft_limits(command, "initial_command")
        self._previous_command = command
        self._previous_delta = delta

    @property
    def previous_command(self) -> tuple[float, ...]:
        return self._previous_command

    @property
    def previous_delta(self) -> tuple[float, ...]:
        return self._previous_delta

    def _assert_command_within_soft_limits(
        self, command: Sequence[float], label: str
    ) -> None:
        violations: dict[str, dict[str, object]] = {}
        for motor, value, (minimum, maximum) in zip(
            self.limits.motor_order, command, self.limits.soft_limits, strict=True
        ):
            if not minimum <= value <= maximum:
                violations[motor] = {
                    "value": value,
                    "allowed": [minimum, maximum],
                }
        if violations:
            raise GuardContractError(f"{label} is outside soft limits: {violations}")

    def validate_measured_positions(
        self, actual_q: Sequence[float]
    ) -> tuple[float, ...]:
        actual = _finite_vector("actual_q", actual_q)
        violations: dict[str, dict[str, object]] = {}
        for motor, value, (minimum, maximum) in zip(
            self.limits.motor_order,
            actual,
            self.limits.calibrated_ranges,
            strict=True,
        ):
            allowed_min = minimum - self.limits.sensor_tolerance
            allowed_max = maximum + self.limits.sensor_tolerance
            if not allowed_min <= value <= allowed_max:
                violations[motor] = {
                    "value": value,
                    "allowed": [allowed_min, allowed_max],
                }
        if violations:
            raise GuardContractError(
                f"actual_q is outside calibrated sensor ranges: {violations}"
            )
        return actual

    def build_policy_state(self, actual_q: Sequence[float]) -> tuple[float, ...]:
        actual = self.validate_measured_positions(actual_q)
        state = actual + self._previous_command + self._previous_delta
        if len(state) != 18:
            raise GuardContractError("internal 18-D state invariant failed")
        return state

    def apply_delta(self, predicted_delta: Sequence[float]) -> GuardStep:
        raw = _finite_vector("predicted_delta", predicted_delta)
        previous = self._previous_command

        rated: list[float] = []
        candidates: list[float] = []
        commands: list[float] = []
        sent_deltas: list[float] = []
        rate_flags: list[bool] = []
        soft_flags: list[bool] = []
        inherited_flags: list[bool] = []
        crossing_flags: list[bool] = []

        epsilon = 1e-12
        for index, value in enumerate(raw):
            rate_limit = self.limits.max_delta_per_command[index]
            soft_min, soft_max = self.limits.soft_limits[index]
            previous_value = previous[index]

            rate_delta = _clamp(value, -rate_limit, rate_limit)
            candidate = previous_value + rate_delta
            command = _clamp(candidate, soft_min, soft_max)
            sent_delta = command - previous_value

            rate_clipped = abs(value - rate_delta) > epsilon
            soft_clipped = abs(candidate - command) > epsilon
            at_min = abs(previous_value - soft_min) <= epsilon
            at_max = abs(previous_value - soft_max) <= epsilon
            inherited = (at_min and rate_delta < 0.0) or (at_max and rate_delta > 0.0)
            new_crossing = (
                (previous_value > soft_min + epsilon and candidate < soft_min)
                or (previous_value < soft_max - epsilon and candidate > soft_max)
            )

            rated.append(rate_delta)
            candidates.append(candidate)
            commands.append(command)
            sent_deltas.append(sent_delta)
            rate_flags.append(rate_clipped)
            soft_flags.append(soft_clipped)
            inherited_flags.append(inherited)
            crossing_flags.append(new_crossing)

        command_tuple = tuple(commands)
        sent_delta_tuple = tuple(sent_deltas)
        self._assert_command_within_soft_limits(command_tuple, "guarded command")
        self._previous_command = command_tuple
        self._previous_delta = sent_delta_tuple

        return GuardStep(
            raw_delta=raw,
            rate_limited_delta=tuple(rated),
            previous_command=previous,
            candidate_after_rate_limit=tuple(candidates),
            sent_command=command_tuple,
            sent_delta=sent_delta_tuple,
            rate_clipped=tuple(rate_flags),
            soft_clipped=tuple(soft_flags),
            inherited_boundary_projection=tuple(inherited_flags),
            new_boundary_crossing=tuple(crossing_flags),
        )

    def action_dict(self) -> dict[str, float]:
        return {
            f"{motor}.pos": value
            for motor, value in zip(
                self.limits.motor_order, self._previous_command, strict=True
            )
        }


@dataclass(frozen=True)
class TrackingStatus:
    tripped: bool
    tripped_joints: tuple[str, ...]
    absolute_error: tuple[float, ...]
    over_limit_seconds: tuple[float, ...]


class TrackingWatchdog:
    """Trips when any command/readback error persists for the frozen timeout."""

    def __init__(self, limits: RuntimeLimits) -> None:
        self.limits = limits
        self._over_since: list[float | None] = [None] * len(limits.motor_order)
        self._last_monotonic_time: float | None = None

    def reset(self) -> None:
        self._over_since = [None] * len(self.limits.motor_order)
        self._last_monotonic_time = None

    def update(
        self,
        actual_q: Sequence[float],
        sent_command: Sequence[float],
        monotonic_time: float,
    ) -> TrackingStatus:
        actual = _finite_vector("actual_q", actual_q)
        command = _finite_vector("sent_command", sent_command)
        now = float(monotonic_time)
        if not math.isfinite(now):
            raise GuardContractError("monotonic_time is not finite")
        if self._last_monotonic_time is not None and now < self._last_monotonic_time:
            raise GuardContractError("monotonic_time moved backwards")
        self._last_monotonic_time = now

        errors = tuple(abs(a - c) for a, c in zip(actual, command, strict=True))
        elapsed: list[float] = []
        tripped: list[str] = []
        for index, (motor, error, limit) in enumerate(
            zip(
                self.limits.motor_order,
                errors,
                self.limits.tracking_error_limits,
                strict=True,
            )
        ):
            if error > limit:
                if self._over_since[index] is None:
                    self._over_since[index] = now
                duration = now - float(self._over_since[index])
                if duration >= self.limits.tracking_timeout_seconds:
                    tripped.append(motor)
            else:
                self._over_since[index] = None
                duration = 0.0
            elapsed.append(duration)

        return TrackingStatus(
            tripped=bool(tripped),
            tripped_joints=tuple(tripped),
            absolute_error=errors,
            over_limit_seconds=tuple(elapsed),
        )


def verify_frozen_inputs(contract_path: Path, calibration_path: Path) -> dict[str, object]:
    if not contract_path.is_file():
        raise GuardContractError(f"contract file not found: {contract_path}")
    if not calibration_path.is_file():
        raise GuardContractError(f"calibration file not found: {calibration_path}")

    contract_sha = _sha256(contract_path)
    if contract_sha != FROZEN_CONTRACT_SHA256:
        raise GuardContractError(
            "runtime contract SHA256 mismatch: "
            f"expected {FROZEN_CONTRACT_SHA256}, got {contract_sha}"
        )
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GuardContractError(f"invalid JSON input: {exc}") from exc
    if not isinstance(contract, dict):
        raise GuardContractError("runtime contract must be a JSON object")
    if calibration != EXPECTED_FOLLOWER_CALIBRATION:
        raise GuardContractError(
            "follower calibration does not match the reviewed follower_white values"
        )

    return {
        "contract_path": str(contract_path),
        "contract_sha256": contract_sha,
        "contract_sha256_matches_frozen": True,
        "calibration_path": str(calibration_path),
        "calibration_sha256": _sha256(calibration_path),
        "calibration_semantic_match": True,
    }


def _expect_guard_error(function, label: str) -> None:
    try:
        function()
    except GuardContractError:
        return
    raise AssertionError(f"expected fail-closed GuardContractError: {label}")


def _assert_close(actual: float, expected: float, label: str, tol: float = 1e-10) -> None:
    if abs(actual - expected) > tol:
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def _audit_import_boundary() -> list[str]:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    forbidden = {"lerobot", "torch", "serial", "cv2"}
    present = sorted(forbidden.intersection(imports))
    if present:
        raise AssertionError(f"hardware/model imports present: {present}")
    return sorted(imports)


def run_algorithm_self_audit(limits: RuntimeLimits) -> dict[str, object]:
    tests: dict[str, bool] = {}

    imports = _audit_import_boundary()
    tests["no_lerobot_torch_serial_or_cv2_import"] = True

    guard = DeltaCommandGuard(limits, limits.home_command)
    state = guard.build_policy_state(limits.home_command)
    assert len(state) == 18
    assert state[6:12] == limits.home_command
    assert state[12:18] == (0.0,) * 6
    tests["state_is_actual_previous_command_previous_sent_delta"] = True

    expected_action_keys = {f"{motor}.pos" for motor in MOTOR_ORDER}
    assert set(guard.action_dict()) == expected_action_keys
    tests["action_dictionary_uses_motor_pos_keys"] = True

    for joint, rate_limit in enumerate(limits.max_delta_per_command):
        center = tuple((lo + hi) / 2.0 for lo, hi in limits.soft_limits)
        for sign in (-1.0, 1.0):
            local = DeltaCommandGuard(limits, center)
            raw = [0.0] * 6
            raw[joint] = sign * rate_limit * 10.0
            step = local.apply_delta(raw)
            _assert_close(
                step.sent_delta[joint], sign * rate_limit, f"rate clamp joint {joint}"
            )
            assert step.rate_clipped[joint]
            assert not step.soft_clipped[joint]
    tests["symmetric_per_joint_rate_clamps"] = True

    for joint, (soft_min, soft_max) in enumerate(limits.soft_limits):
        rate_limit = limits.max_delta_per_command[joint]
        for boundary, sign in ((soft_min, -1.0), (soft_max, 1.0)):
            start = [float((lo + hi) / 2.0) for lo, hi in limits.soft_limits]
            start[joint] = boundary
            local = DeltaCommandGuard(limits, start)
            raw = [0.0] * 6
            raw[joint] = sign * min(0.1, rate_limit / 10.0)
            step = local.apply_delta(raw)
            _assert_close(step.sent_command[joint], boundary, "boundary projection")
            _assert_close(step.sent_delta[joint], 0.0, "boundary no-op delta")
            assert step.soft_clipped[joint]
            assert step.inherited_boundary_projection[joint]
            assert not step.new_boundary_crossing[joint]
    tests["outward_delta_at_active_boundary_becomes_no_op"] = True

    for joint, (soft_min, soft_max) in enumerate(limits.soft_limits):
        rate_limit = limits.max_delta_per_command[joint]
        start = [float((lo + hi) / 2.0) for lo, hi in limits.soft_limits]
        start[joint] = soft_max - rate_limit * 0.1
        local = DeltaCommandGuard(limits, start)
        raw = [0.0] * 6
        raw[joint] = rate_limit * 0.5
        step = local.apply_delta(raw)
        _assert_close(step.sent_command[joint], soft_max, "new upper crossing clamp")
        assert step.soft_clipped[joint]
        assert step.new_boundary_crossing[joint]
        assert not step.inherited_boundary_projection[joint]

        start[joint] = soft_min + rate_limit * 0.1
        local = DeltaCommandGuard(limits, start)
        raw[joint] = -rate_limit * 0.5
        step = local.apply_delta(raw)
        _assert_close(step.sent_command[joint], soft_min, "new lower crossing clamp")
        assert step.soft_clipped[joint]
        assert step.new_boundary_crossing[joint]
    tests["interior_boundary_crossings_are_attributed_and_clamped"] = True

    local = DeltaCommandGuard(limits, limits.home_command)
    step = local.apply_delta((0.1, -0.1, 0.2, 0.0, 0.3, -0.2))
    next_state = local.build_policy_state(limits.home_command)
    assert next_state[6:12] == step.sent_command
    assert next_state[12:18] == step.sent_delta
    tests["recursive_state_uses_guarded_command_and_guarded_delta"] = True

    _expect_guard_error(
        lambda: DeltaCommandGuard(limits, limits.home_command).apply_delta((0.0,) * 5),
        "short predicted delta",
    )
    _expect_guard_error(
        lambda: DeltaCommandGuard(limits, limits.home_command).apply_delta(
            (0.0, 0.0, math.nan, 0.0, 0.0, 0.0)
        ),
        "NaN predicted delta",
    )
    _expect_guard_error(
        lambda: DeltaCommandGuard(limits, limits.home_command).apply_delta(
            (0.0, 0.0, math.inf, 0.0, 0.0, 0.0)
        ),
        "Inf predicted delta",
    )
    tests["malformed_or_nonfinite_policy_output_fails_closed"] = True

    sensor_guard = DeltaCommandGuard(limits, limits.home_command)
    for joint, (minimum, maximum) in enumerate(limits.calibrated_ranges):
        for accepted in (
            minimum - limits.sensor_tolerance,
            maximum + limits.sensor_tolerance,
        ):
            vector = list(limits.home_command)
            vector[joint] = accepted
            sensor_guard.validate_measured_positions(vector)
        vector = list(limits.home_command)
        vector[joint] = maximum + limits.sensor_tolerance + 1e-6
        _expect_guard_error(
            lambda vector=tuple(vector): sensor_guard.validate_measured_positions(vector),
            f"sensor range joint {joint}",
        )
    tests["calibrated_sensor_range_with_tolerance_fails_closed"] = True

    for joint, error_limit in enumerate(limits.tracking_error_limits):
        watchdog = TrackingWatchdog(limits)
        actual = list(limits.home_command)
        actual[joint] += error_limit + 0.5
        status = watchdog.update(actual, limits.home_command, 100.0)
        assert not status.tripped
        status = watchdog.update(
            actual,
            limits.home_command,
            100.0 + limits.tracking_timeout_seconds - 1e-6,
        )
        assert not status.tripped
        status = watchdog.update(
            actual,
            limits.home_command,
            100.0 + limits.tracking_timeout_seconds,
        )
        assert status.tripped
        assert status.tripped_joints == (limits.motor_order[joint],)
    tests["tracking_error_trips_only_after_persistent_timeout"] = True

    watchdog = TrackingWatchdog(limits)
    actual = list(limits.home_command)
    actual[0] += limits.tracking_error_limits[0] + 1.0
    assert not watchdog.update(actual, limits.home_command, 1.0).tripped
    assert not watchdog.update(limits.home_command, limits.home_command, 2.5).tripped
    assert not watchdog.update(actual, limits.home_command, 3.0).tripped
    assert not watchdog.update(actual, limits.home_command, 4.9).tripped
    _expect_guard_error(
        lambda: watchdog.update(actual, limits.home_command, 4.8),
        "monotonic clock reversal",
    )
    tests["tracking_timer_resets_on_recovery_and_rejects_clock_reversal"] = True

    return {
        "tests": tests,
        "tests_passed": sum(tests.values()),
        "tests_total": len(tests),
        "import_roots": imports,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    contract_path = args.contract.expanduser().resolve()
    calibration_path = args.calibration.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve()

    print("===== VERIFY FROZEN OFFLINE INPUTS =====")
    try:
        input_report = verify_frozen_inputs(contract_path, calibration_path)
        print("contract byte identity and follower calibration semantics: PASS")

        print("\n===== PURE GUARD ALGORITHM SELF-AUDIT =====")
        limits = RuntimeLimits.frozen_v3()
        audit = run_algorithm_self_audit(limits)
        for name, passed in audit["tests"].items():
            print(f"{name}: {'PASS' if passed else 'FAIL'}")
        print(f"tests: {audit['tests_passed']}/{audit['tests_total']} PASS")

        report = {
            "schema_version": "act_v3_delta_runtime_guard_self_audit_v1",
            "frozen_inputs": input_report,
            "runtime_limits": limits.to_jsonable(),
            "algorithm_self_audit": audit,
            "hardware_access_authorized": False,
            "hardware_or_model_modules_imported": False,
            "integration_requirements_not_tested": [
                "live camera pairing and freshness",
                "live Present_Position readback freshness",
                "robot connect/calibration behavior",
                "torque enable/disable and emergency-stop lifecycle",
                "physical workspace, collision clearance, payload, and cable routing",
            ],
            "decision": "PURE_GUARD_CORE_PASS_BUILD_OFFLINE_ADAPTER_NEXT",
        }
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"ACT V3 DELTA RUNTIME GUARD SELF-AUDIT: FAIL: {exc}", file=sys.stderr)
        print("HARDWARE DEPLOYMENT REMAINS BLOCKED.", file=sys.stderr)
        print("NO ROBOT OBJECT WAS CREATED. NO SERIAL PORT WAS OPENED.", file=sys.stderr)
        print("NO COMMAND WAS SENT.", file=sys.stderr)
        return 1

    print("\n===== DECISION =====")
    print("decision='PURE_GUARD_CORE_PASS_BUILD_OFFLINE_ADAPTER_NEXT'")
    print("HARDWARE DEPLOYMENT REMAINS BLOCKED.")
    print("NO ROBOT OBJECT WAS CREATED. NO SERIAL PORT WAS OPENED.")
    print("NO COMMAND WAS SENT. NO MODEL OR DATASET WAS LOADED.")
    print("\n===== OUTPUT =====")
    print(output_json)
    print("ACT V3 DELTA RUNTIME GUARD SELF-AUDIT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
