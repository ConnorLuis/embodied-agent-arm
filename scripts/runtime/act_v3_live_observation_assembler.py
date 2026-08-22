#!/usr/bin/env python3
"""Purely assemble and audit one live ACT V3 observation without hardware I/O.

The reusable :class:`LiveObservationAssembler` accepts already-received camera
packets plus one already-read joint-state packet.  It validates camera identity,
sequence monotonicity, freshness, front/wrist skew, joint-read freshness and
the guarded 18-D state contract before delegating image conversion to the
reviewed camera preprocessing adapter.

This module does not own camera capture, networking, robot construction, serial
I/O, policy inference or command transmission.  Its CLI uses completed bridge
snapshots and one frozen validation sample for an offline self-audit only.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence


STATE_KEY = "observation.state"
FRONT_KEY = "observation.images.front"
WRIST_KEY = "observation.images.wrist"

EXPECTED_PREPROCESS_SHA256 = (
    "63215b4ac625afa87abc6262b560b4383d378b61828d4582c9e4e4781e4c65b6"
)
EXPECTED_GUARD_SHA256 = (
    "e06e785071ee3ded95e89d670978bcb4a8caf104b2fc69fc87de37cb6fff6d15"
)
EXPECTED_CAMERA_IDENTITIES = {
    "front": (0x32E6, 0x9221),
    "wrist": (0x32E6, 0x9005),
}

DEFAULT_MAX_CAMERA_AGE_SECONDS = 0.250
DEFAULT_MAX_JOINT_AGE_SECONDS = 0.100
DEFAULT_MAX_CAPTURE_SKEW_SECONDS = 0.100
DEFAULT_MAX_RECEIVE_SKEW_SECONDS = 0.100


class ObservationContractError(RuntimeError):
    """Fail-closed observation metadata, timing, state or tensor violation."""


@dataclass(frozen=True)
class CameraFramePacket:
    role: str
    vid: int
    pid: int
    sequence: int
    capture_monotonic_ns: int
    receive_monotonic_s: float
    frame_bgr: Any


@dataclass(frozen=True)
class JointStatePacket:
    actual_q: Sequence[float]
    read_monotonic_s: float


@dataclass(frozen=True)
class AssemblyMetrics:
    now_monotonic_s: float
    front_sequence: int
    wrist_sequence: int
    front_age_ms: float
    wrist_age_ms: float
    joint_age_ms: float
    capture_skew_ms: float
    receive_skew_ms: float
    state_dimension: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def load_module(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module spec: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def finite_time(label: str, value: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ObservationContractError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ObservationContractError(f"{label} must be finite")
    return result


def positive_finite(label: str, value: float) -> float:
    result = finite_time(label, value)
    if result <= 0.0:
        raise ObservationContractError(f"{label} must be positive")
    return result


def verify_static_offline_boundary(path: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules: set[str] = set()
    identifiers: set[str] = set()
    attributes: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)
        elif isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, ast.Attribute):
            attributes.add(node.attr)

    forbidden_modules = sorted(
        module
        for module in imported_modules
        if module.startswith(("lerobot.robots", "lerobot.teleoperators", "serial", "pyserial"))
    )
    forbidden_identifiers = sorted(
        {"SO101Follower", "SO101FollowerConfig", "VideoCapture", "ACTPolicy"}.intersection(
            identifiers
        )
    )
    forbidden_attributes = sorted(
        {
            "send_action",
            "sync_write",
            "enable_torque",
            "disable_torque",
            "predict_action",
            "predict_action_chunk",
            "from_pretrained",
        }.intersection(attributes)
    )
    if forbidden_modules or forbidden_identifiers or forbidden_attributes:
        raise ObservationContractError(
            "hardware/model-capable code found: "
            f"modules={forbidden_modules}, identifiers={forbidden_identifiers}, "
            f"attributes={forbidden_attributes}"
        )
    return {
        "source": str(path),
        "sha256": sha256_file(path),
        "camera_open_calls": False,
        "robot_or_serial_imports": False,
        "model_load_or_inference_calls": False,
        "command_write_calls": False,
    }


class LiveObservationAssembler:
    """Transactional ACT observation assembly with fail-closed timing checks."""

    def __init__(
        self,
        *,
        preprocess_module: ModuleType,
        delta_guard: Any,
        cv2_module: Any,
        np_module: Any,
        torch_module: Any,
        device: Any | None = None,
        max_camera_age_seconds: float = DEFAULT_MAX_CAMERA_AGE_SECONDS,
        max_joint_age_seconds: float = DEFAULT_MAX_JOINT_AGE_SECONDS,
        max_capture_skew_seconds: float = DEFAULT_MAX_CAPTURE_SKEW_SECONDS,
        max_receive_skew_seconds: float = DEFAULT_MAX_RECEIVE_SKEW_SECONDS,
    ) -> None:
        self.preprocess_module = preprocess_module
        self.delta_guard = delta_guard
        self.cv2 = cv2_module
        self.np = np_module
        self.torch = torch_module
        self.device = device
        self.max_camera_age_seconds = positive_finite(
            "max_camera_age_seconds", max_camera_age_seconds
        )
        self.max_joint_age_seconds = positive_finite(
            "max_joint_age_seconds", max_joint_age_seconds
        )
        self.max_capture_skew_seconds = positive_finite(
            "max_capture_skew_seconds", max_capture_skew_seconds
        )
        self.max_receive_skew_seconds = positive_finite(
            "max_receive_skew_seconds", max_receive_skew_seconds
        )
        self._last_sequence: dict[str, int | None] = {"front": None, "wrist": None}
        self._last_now: float | None = None

    def _validate_camera_packet(
        self,
        packet: CameraFramePacket,
        expected_role: str,
        now: float,
    ) -> tuple[float, float]:
        if not isinstance(packet, CameraFramePacket):
            raise ObservationContractError(f"{expected_role} packet has wrong type")
        if packet.role != expected_role:
            raise ObservationContractError(
                f"camera role mismatch: expected={expected_role}, got={packet.role!r}"
            )
        expected_vid, expected_pid = EXPECTED_CAMERA_IDENTITIES[expected_role]
        if (packet.vid, packet.pid) != (expected_vid, expected_pid):
            raise ObservationContractError(
                f"{expected_role} identity mismatch: "
                f"expected={expected_vid:04X}:{expected_pid:04X}, "
                f"got={packet.vid:04X}:{packet.pid:04X}"
            )
        if isinstance(packet.sequence, bool) or not isinstance(packet.sequence, int):
            raise ObservationContractError(f"{expected_role} sequence must be an integer")
        if packet.sequence < 0:
            raise ObservationContractError(f"{expected_role} sequence must be nonnegative")
        previous = self._last_sequence[expected_role]
        if previous is not None and packet.sequence <= previous:
            raise ObservationContractError(
                f"{expected_role} sequence did not advance: previous={previous}, "
                f"current={packet.sequence}"
            )
        if isinstance(packet.capture_monotonic_ns, bool) or not isinstance(
            packet.capture_monotonic_ns, int
        ):
            raise ObservationContractError(
                f"{expected_role} capture_monotonic_ns must be an integer"
            )
        if packet.capture_monotonic_ns < 0:
            raise ObservationContractError(
                f"{expected_role} capture_monotonic_ns must be nonnegative"
            )
        received = finite_time(
            f"{expected_role}.receive_monotonic_s", packet.receive_monotonic_s
        )
        if received > now:
            raise ObservationContractError(f"{expected_role} receive timestamp is in the future")
        age = now - received
        if age > self.max_camera_age_seconds:
            raise ObservationContractError(
                f"{expected_role} frame is stale: age={age:.6f}s"
            )
        return received, age

    def assemble(
        self,
        *,
        front: CameraFramePacket,
        wrist: CameraFramePacket,
        joints: JointStatePacket,
        now_monotonic_s: float,
    ) -> tuple[dict[str, Any], AssemblyMetrics]:
        """Validate and assemble one observation; commit sequence state only on success."""
        try:
            now = finite_time("now_monotonic_s", now_monotonic_s)
            if self._last_now is not None and now < self._last_now:
                raise ObservationContractError("assembler monotonic clock moved backwards")

            front_received, front_age = self._validate_camera_packet(front, "front", now)
            wrist_received, wrist_age = self._validate_camera_packet(wrist, "wrist", now)
            capture_skew = abs(
                front.capture_monotonic_ns - wrist.capture_monotonic_ns
            ) / 1_000_000_000.0
            if capture_skew > self.max_capture_skew_seconds:
                raise ObservationContractError(
                    f"front/wrist capture skew is too large: {capture_skew:.6f}s"
                )
            receive_skew = abs(front_received - wrist_received)
            if receive_skew > self.max_receive_skew_seconds:
                raise ObservationContractError(
                    f"front/wrist receive skew is too large: {receive_skew:.6f}s"
                )

            if not isinstance(joints, JointStatePacket):
                raise ObservationContractError("joint packet has wrong type")
            joint_time = finite_time("joints.read_monotonic_s", joints.read_monotonic_s)
            if joint_time > now:
                raise ObservationContractError("joint read timestamp is in the future")
            joint_age = now - joint_time
            if joint_age > self.max_joint_age_seconds:
                raise ObservationContractError(
                    f"joint read is stale: age={joint_age:.6f}s"
                )

            state_18 = self.delta_guard.build_policy_state(joints.actual_q)
            if len(state_18) != 18:
                raise ObservationContractError("guard did not produce an 18-D state")
            observation = self.preprocess_module.build_policy_observation(
                front.frame_bgr,
                wrist.frame_bgr,
                state_18,
                cv2_module=self.cv2,
                np_module=self.np,
                torch_module=self.torch,
                device=self.device,
            )
            if set(observation) != {STATE_KEY, FRONT_KEY, WRIST_KEY}:
                raise ObservationContractError("ACT observation keys mismatch")
            expected_shapes = {
                STATE_KEY: (1, 18),
                FRONT_KEY: (1, 3, 480, 480),
                WRIST_KEY: (1, 3, 480, 480),
            }
            for key, expected_shape in expected_shapes.items():
                value = observation[key]
                if not isinstance(value, self.torch.Tensor):
                    raise ObservationContractError(f"{key} is not a torch tensor")
                if tuple(int(x) for x in value.shape) != expected_shape:
                    raise ObservationContractError(
                        f"{key} shape mismatch: {tuple(value.shape)}"
                    )
                if value.dtype != self.torch.float32:
                    raise ObservationContractError(f"{key} dtype is not torch.float32")
                if not bool(self.torch.isfinite(value).all().item()):
                    raise ObservationContractError(f"{key} contains nonfinite values")
            for key in (FRONT_KEY, WRIST_KEY):
                minimum = float(observation[key].min().item())
                maximum = float(observation[key].max().item())
                if minimum < 0.0 or maximum > 1.0:
                    raise ObservationContractError(f"{key} is outside [0,1]")

            metrics = AssemblyMetrics(
                now_monotonic_s=now,
                front_sequence=front.sequence,
                wrist_sequence=wrist.sequence,
                front_age_ms=front_age * 1000.0,
                wrist_age_ms=wrist_age * 1000.0,
                joint_age_ms=joint_age * 1000.0,
                capture_skew_ms=capture_skew * 1000.0,
                receive_skew_ms=receive_skew * 1000.0,
                state_dimension=len(state_18),
            )

            # Transactional commit: no failed assembly consumes a sequence number.
            self._last_sequence = {
                "front": front.sequence,
                "wrist": wrist.sequence,
            }
            self._last_now = now
            return observation, metrics
        except ObservationContractError:
            raise
        except Exception as exc:
            raise ObservationContractError(
                f"observation dependency rejected the input: {type(exc).__name__}: {exc}"
            ) from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocess-adapter", type=Path, required=True)
    parser.add_argument("--guard-core", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--bridge-script", type=Path, required=True)
    parser.add_argument("--bridge-output-dir", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--validation-dataset-root", type=Path, required=True)
    parser.add_argument("--validation-repo-id", required=True)
    parser.add_argument("--video-backend", default="torchcodec")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def expect_closed(label: str, function: Any, results: dict[str, bool]) -> None:
    try:
        function()
    except ObservationContractError:
        results[label] = True
        return
    raise AssertionError(f"expected fail-closed rejection: {label}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_json = args.output_json.expanduser().resolve()
    try:
        if output_json.exists():
            raise RuntimeError(f"output JSON already exists: {output_json}")
        preprocess_path = require_file(args.preprocess_adapter)
        guard_path = require_file(args.guard_core)
        contract_path = require_file(args.contract)
        calibration_path = require_file(args.calibration)
        bridge_script = require_file(args.bridge_script)
        bridge_output_dir = require_dir(args.bridge_output_dir)
        candidate_root = require_dir(args.candidate_root)
        validation_root = require_dir(args.validation_dataset_root)

        print("===== STATIC NO-HARDWARE / NO-MODEL BOUNDARY =====")
        static_report = verify_static_offline_boundary(Path(__file__).resolve())
        print("no camera-open, robot, serial, model-load, inference or command calls: PASS")

        print("\n===== VERIFY FROZEN RUNTIME DEPENDENCIES =====")
        preprocess_sha = sha256_file(preprocess_path)
        guard_sha = sha256_file(guard_path)
        if preprocess_sha != EXPECTED_PREPROCESS_SHA256:
            raise RuntimeError(
                f"preprocess SHA mismatch: expected={EXPECTED_PREPROCESS_SHA256}, "
                f"actual={preprocess_sha}"
            )
        if guard_sha != EXPECTED_GUARD_SHA256:
            raise RuntimeError(
                f"guard SHA mismatch: expected={EXPECTED_GUARD_SHA256}, actual={guard_sha}"
            )
        preprocess = load_module(preprocess_path, "act_v3_camera_preprocess_frozen")
        guard_core = load_module(guard_path, "act_v3_delta_runtime_guard_frozen")
        preprocess_static = preprocess.verify_static_offline_boundary(preprocess_path)
        guard_inputs = guard_core.verify_frozen_inputs(contract_path, calibration_path)
        limits = guard_core.RuntimeLimits.frozen_v3()
        guard_audit = guard_core.run_algorithm_self_audit(limits)
        if guard_audit["tests_passed"] != guard_audit["tests_total"]:
            raise RuntimeError("guard algorithm self-audit did not fully pass")
        _, policy_report = preprocess.load_policy_config(candidate_root)
        _, snapshots, bridge_report = preprocess.verify_bridge_run(
            bridge_script,
            bridge_output_dir,
            "last",
        )
        print("preprocess, guard, contract, calibration, policy config and bridge: PASS")

        import cv2
        import numpy as np
        import torch
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        print("\n===== LOAD ONE FROZEN OFFLINE STATE AND TWO BRIDGE SNAPSHOTS =====")
        dataset = preprocess.load_dataset(
            LeRobotDataset,
            args.validation_repo_id,
            validation_root,
            args.video_backend,
        )
        item = dataset[0]
        recorded = torch.as_tensor(item[STATE_KEY], dtype=torch.float32).reshape(18)
        if not bool(torch.isfinite(recorded).all().item()):
            raise RuntimeError("recorded validation state contains nonfinite values")
        recorded_values = tuple(float(value) for value in recorded.tolist())
        actual_q = recorded_values[:6]
        previous_command = recorded_values[6:12]
        previous_delta = recorded_values[12:18]
        frames: dict[str, Any] = {}
        for role in ("front", "wrist"):
            frame = cv2.imread(str(snapshots[role]), cv2.IMREAD_COLOR)
            if frame is None or frame.size == 0:
                raise RuntimeError(f"failed to decode {role} bridge snapshot")
            frames[role] = frame
        print("recorded 18-D state and completed front/wrist bridge frames: PASS")

        def new_assembler() -> LiveObservationAssembler:
            delta_guard = guard_core.DeltaCommandGuard(
                limits,
                previous_command,
                previous_delta,
            )
            return LiveObservationAssembler(
                preprocess_module=preprocess,
                delta_guard=delta_guard,
                cv2_module=cv2,
                np_module=np,
                torch_module=torch,
            )

        front_sequence = int(
            bridge_report["cameras"]["front"]["received_frames"]
        ) - 1
        wrist_sequence = int(
            bridge_report["cameras"]["wrist"]["received_frames"]
        ) - 1
        front_packet = CameraFramePacket(
            role="front",
            vid=0x32E6,
            pid=0x9221,
            sequence=front_sequence,
            capture_monotonic_ns=1_000_000_000,
            receive_monotonic_s=100.000,
            frame_bgr=frames["front"],
        )
        wrist_packet = CameraFramePacket(
            role="wrist",
            vid=0x32E6,
            pid=0x9005,
            sequence=wrist_sequence,
            capture_monotonic_ns=1_020_000_000,
            receive_monotonic_s=100.010,
            frame_bgr=frames["wrist"],
        )
        joint_packet = JointStatePacket(
            actual_q=actual_q,
            read_monotonic_s=100.015,
        )

        print("\n===== SUCCESSFUL OBSERVATION ASSEMBLY =====")
        assembler = new_assembler()
        observation, metrics = assembler.assemble(
            front=front_packet,
            wrist=wrist_packet,
            joints=joint_packet,
            now_monotonic_s=100.020,
        )
        reconstructed = observation[STATE_KEY][0].detach().cpu()
        state_max_error = float(torch.max(torch.abs(reconstructed - recorded)).item())
        if state_max_error > 1e-6:
            raise RuntimeError(f"18-D runtime state mismatch: {state_max_error}")
        direct_front, _, _ = preprocess.preprocess_bgr_frame(
            frames["front"],
            "front",
            cv2_module=cv2,
            np_module=np,
            torch_module=torch,
        )
        direct_wrist, _, _ = preprocess.preprocess_bgr_frame(
            frames["wrist"],
            "wrist",
            cv2_module=cv2,
            np_module=np,
            torch_module=torch,
        )
        front_image_max_error = float(
            torch.max(torch.abs(observation[FRONT_KEY].cpu() - direct_front.cpu())).item()
        )
        wrist_image_max_error = float(
            torch.max(torch.abs(observation[WRIST_KEY].cpu() - direct_wrist.cpu())).item()
        )
        if front_image_max_error != 0.0 or wrist_image_max_error != 0.0:
            raise RuntimeError("assembler image output differs from frozen preprocessing")
        print(
            "keys/shapes/dtypes/ranges: PASS; "
            f"state_error={state_max_error:.10f} "
            f"front_error={front_image_max_error:.10f} "
            f"wrist_error={wrist_image_max_error:.10f}"
        )
        print(
            f"ages_ms=front:{metrics.front_age_ms:.1f} "
            f"wrist:{metrics.wrist_age_ms:.1f} joints:{metrics.joint_age_ms:.1f}; "
            f"skew_ms=capture:{metrics.capture_skew_ms:.1f} "
            f"receive:{metrics.receive_skew_ms:.1f}"
        )

        print("\n===== FAIL-CLOSED MATRIX =====")
        failures: dict[str, bool] = {}

        expect_closed(
            "wrong_camera_identity",
            lambda: new_assembler().assemble(
                front=replace(front_packet, pid=0x9005),
                wrist=wrist_packet,
                joints=joint_packet,
                now_monotonic_s=100.020,
            ),
            failures,
        )
        expect_closed(
            "swapped_camera_role",
            lambda: new_assembler().assemble(
                front=replace(front_packet, role="wrist"),
                wrist=wrist_packet,
                joints=joint_packet,
                now_monotonic_s=100.020,
            ),
            failures,
        )
        expect_closed(
            "stale_camera_frame",
            lambda: new_assembler().assemble(
                front=replace(front_packet, receive_monotonic_s=99.700),
                wrist=wrist_packet,
                joints=joint_packet,
                now_monotonic_s=100.020,
            ),
            failures,
        )
        expect_closed(
            "future_camera_timestamp",
            lambda: new_assembler().assemble(
                front=replace(front_packet, receive_monotonic_s=100.021),
                wrist=wrist_packet,
                joints=joint_packet,
                now_monotonic_s=100.020,
            ),
            failures,
        )
        expect_closed(
            "capture_skew_exceeded",
            lambda: new_assembler().assemble(
                front=front_packet,
                wrist=replace(wrist_packet, capture_monotonic_ns=1_200_000_001),
                joints=joint_packet,
                now_monotonic_s=100.020,
            ),
            failures,
        )
        expect_closed(
            "receive_skew_exceeded",
            lambda: new_assembler().assemble(
                front=replace(front_packet, receive_monotonic_s=99.900),
                wrist=replace(wrist_packet, receive_monotonic_s=100.010),
                joints=joint_packet,
                now_monotonic_s=100.020,
            ),
            failures,
        )
        expect_closed(
            "stale_joint_read",
            lambda: new_assembler().assemble(
                front=front_packet,
                wrist=wrist_packet,
                joints=replace(joint_packet, read_monotonic_s=99.900),
                now_monotonic_s=100.020,
            ),
            failures,
        )
        expect_closed(
            "future_joint_timestamp",
            lambda: new_assembler().assemble(
                front=front_packet,
                wrist=wrist_packet,
                joints=replace(joint_packet, read_monotonic_s=100.021),
                now_monotonic_s=100.020,
            ),
            failures,
        )

        duplicate_assembler = new_assembler()
        duplicate_assembler.assemble(
            front=front_packet,
            wrist=wrist_packet,
            joints=joint_packet,
            now_monotonic_s=100.020,
        )
        expect_closed(
            "duplicate_sequence",
            lambda: duplicate_assembler.assemble(
                front=front_packet,
                wrist=wrist_packet,
                joints=joint_packet,
                now_monotonic_s=100.030,
            ),
            failures,
        )

        reversed_clock_assembler = new_assembler()
        reversed_clock_assembler.assemble(
            front=front_packet,
            wrist=wrist_packet,
            joints=joint_packet,
            now_monotonic_s=100.020,
        )
        expect_closed(
            "assembler_clock_reversal",
            lambda: reversed_clock_assembler.assemble(
                front=replace(front_packet, sequence=front_sequence + 1),
                wrist=replace(wrist_packet, sequence=wrist_sequence + 1),
                joints=replace(joint_packet, read_monotonic_s=100.005),
                now_monotonic_s=100.010,
            ),
            failures,
        )

        bad_actual = list(actual_q)
        bad_actual[2] = math.nan
        expect_closed(
            "nonfinite_actual_q",
            lambda: new_assembler().assemble(
                front=front_packet,
                wrist=wrist_packet,
                joints=replace(joint_packet, actual_q=bad_actual),
                now_monotonic_s=100.020,
            ),
            failures,
        )
        expect_closed(
            "wrong_frame_shape",
            lambda: new_assembler().assemble(
                front=replace(front_packet, frame_bgr=frames["front"][:, :-1]),
                wrist=wrist_packet,
                joints=joint_packet,
                now_monotonic_s=100.020,
            ),
            failures,
        )
        expect_closed(
            "negative_capture_timestamp",
            lambda: new_assembler().assemble(
                front=replace(front_packet, capture_monotonic_ns=-1),
                wrist=wrist_packet,
                joints=joint_packet,
                now_monotonic_s=100.020,
            ),
            failures,
        )
        expect_closed(
            "boolean_sequence",
            lambda: new_assembler().assemble(
                front=replace(front_packet, sequence=True),
                wrist=wrist_packet,
                joints=joint_packet,
                now_monotonic_s=100.020,
            ),
            failures,
        )
        expect_closed(
            "nonfinite_now",
            lambda: new_assembler().assemble(
                front=front_packet,
                wrist=wrist_packet,
                joints=joint_packet,
                now_monotonic_s=math.inf,
            ),
            failures,
        )

        transactional = new_assembler()
        expect_closed(
            "failed_assembly_does_not_commit",
            lambda: transactional.assemble(
                front=replace(front_packet, pid=0x9005),
                wrist=wrist_packet,
                joints=joint_packet,
                now_monotonic_s=100.020,
            ),
            failures,
        )
        transactional.assemble(
            front=front_packet,
            wrist=wrist_packet,
            joints=joint_packet,
            now_monotonic_s=100.020,
        )
        print(f"fail-closed tests: {sum(failures.values())}/{len(failures)} PASS")

        report = {
            "schema_version": 1,
            "static_offline_boundary": static_report,
            "frozen_dependencies": {
                "preprocess_adapter": {
                    "path": str(preprocess_path),
                    "sha256": preprocess_sha,
                    "static_boundary": preprocess_static,
                },
                "guard_core": {
                    "path": str(guard_path),
                    "sha256": guard_sha,
                    "inputs": guard_inputs,
                    "algorithm_tests_passed": guard_audit["tests_passed"],
                    "algorithm_tests_total": guard_audit["tests_total"],
                },
                "policy": policy_report,
                "bridge": bridge_report,
            },
            "freshness_contract": {
                "max_camera_age_seconds": DEFAULT_MAX_CAMERA_AGE_SECONDS,
                "max_joint_age_seconds": DEFAULT_MAX_JOINT_AGE_SECONDS,
                "max_capture_skew_seconds": DEFAULT_MAX_CAPTURE_SKEW_SECONDS,
                "max_receive_skew_seconds": DEFAULT_MAX_RECEIVE_SKEW_SECONDS,
                "capture_clock_domain": "shared Windows monotonic clock",
                "receive_and_joint_clock_domain": "shared WSL monotonic clock",
            },
            "successful_assembly": {
                "metrics": asdict(metrics),
                "observation_keys": sorted(observation),
                "state_shape": list(observation[STATE_KEY].shape),
                "front_shape": list(observation[FRONT_KEY].shape),
                "wrist_shape": list(observation[WRIST_KEY].shape),
                "dtype": str(observation[STATE_KEY].dtype),
                "state_max_error_vs_recorded": state_max_error,
                "front_max_error_vs_frozen_preprocess": front_image_max_error,
                "wrist_max_error_vs_frozen_preprocess": wrist_image_max_error,
            },
            "fail_closed_tests": failures,
            "fail_closed_tests_passed": sum(failures.values()),
            "fail_closed_tests_total": len(failures),
            "hardware_deployment_authorized": False,
            "camera_opened": False,
            "model_loaded": False,
            "robot_accessed": False,
            "serial_port_opened": False,
            "command_sent": False,
            "decision": (
                "LIVE_OBSERVATION_ASSEMBLER_OFFLINE_PASS_"
                "BUILD_READ_ONLY_LIVE_SENSOR_ADAPTER_NEXT"
            ),
        }
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"ACT V3 LIVE OBSERVATION ASSEMBLER: FAIL: {exc}", file=sys.stderr)
        print("HARDWARE DEPLOYMENT REMAINS BLOCKED.", file=sys.stderr)
        print("NO ROBOT OBJECT WAS CREATED. NO SERIAL PORT WAS OPENED.", file=sys.stderr)
        print("NO CAMERA WAS OPENED. NO MODEL WAS LOADED. NO COMMAND WAS SENT.", file=sys.stderr)
        return 1

    print("\n===== DECISION =====")
    print(
        "decision='LIVE_OBSERVATION_ASSEMBLER_OFFLINE_PASS_"
        "BUILD_READ_ONLY_LIVE_SENSOR_ADAPTER_NEXT'"
    )
    print("HARDWARE DEPLOYMENT REMAINS BLOCKED.")
    print("NO ROBOT OBJECT WAS CREATED. NO SERIAL PORT WAS OPENED.")
    print("NO CAMERA WAS OPENED. NO MODEL WAS LOADED. NO COMMAND WAS SENT.")
    print("\n===== OUTPUT =====")
    print(output_json)
    print("ACT V3 LIVE OBSERVATION ASSEMBLER: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
