#!/usr/bin/env python3
"""Run the frozen ACT V3 stack in a finite, strictly non-commanding live shadow.

The WSL parent loads the reviewed ACT 11K policy, starts the reviewed native-
Windows front/wrist camera workers, opens the reviewed follower motor bus, and
repeatedly performs the following 3 Hz shadow replan sequence for exactly 60 s:

    paired live camera frames + one Present_Position read
      -> reviewed image geometry and 18-D state assembly
      -> ACT predict_action_chunk
      -> local guard projection for diagnostics only
      -> JSON/CSV logs only

No predicted value is passed to a robot, motor bus, or action API.  The only
motor traffic is repeated normalized Present_Position READ instructions through
the already-audited read-only adapter.  The bus is closed with
``disable_torque=False``.  Camera transport uses the already-reviewed Windows
worker implementation and its END/ACK protocol.

This gate proves live sensing, preprocessing, observation assembly, inference,
and timing integration.  It does not authorize motion.  In particular, a
folded arm may be inside calibrated sensor ranges while outside the task soft
workspace; that condition is reported separately and keeps motion blocked.
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
import os
import secrets
import socket
import sys
import threading
import time
import traceback
import zlib
from collections import Counter, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Mapping, Sequence


MOTOR_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
STATE_KEY = "observation.state"
FRONT_KEY = "observation.images.front"
WRIST_KEY = "observation.images.wrist"

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
EXPECTED_SENSOR_ADAPTER_SHA256 = (
    "516749f47ccd6e6731760aa46c304936ba9e8cbaacc2e7f4baf4b4f101078d67"
)
EXPECTED_ONE_SHOT_LAUNCHER_SHA256 = (
    "c2f2be0481e72aa1836bae36e873be59b9c739b776d1baf5b510954264b9196b"
)
EXPECTED_OFFLINE_ADAPTER_SHA256 = (
    "0deb4d70d4fc3b8e0af205bfd047a7167c53736b933c39906d79ffd767d3fbc3"
)
EXPECTED_FOLLOWER_SHA256 = (
    "26c675c71ade2670fa1bed2d887507b6ef3ad53a49e20182f5ba3d032a0afd7a"
)
EXPECTED_FOLLOWER_CONFIG_SHA256 = (
    "e5702d9b6c4a09d10de83f912a1640169698292d5145091d403f4fdce0211cbe"
)
EXPECTED_ONE_SHOT_DECISION = (
    "ONE_SHOT_READ_ONLY_JOINT_SENSOR_PASS_BUILD_LIVE_SHADOW_MODE_NEXT"
)
EXPECTED_FOLLOWER_BY_ID = (
    "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82110810-if00"
)
EXPECTED_POLICY_STEP = 11000
EXPECTED_CHUNK_SIZE = 10
EXPECTED_QUEUE_LENGTH = 5
FROZEN_DURATION_SECONDS = 60.0
FROZEN_REPLAN_HZ = 3.0
CAMERA_STREAM_TAIL_SECONDS = 3.0
PAIR_WAIT_TIMEOUT_SECONDS = 0.30
MAX_CAMERA_AGE_SECONDS = 0.250
MAX_JOINT_AGE_SECONDS = 0.100
MAX_CAMERA_SKEW_SECONDS = 0.100
MIN_CAMERA_FPS = 14.0
MAX_CAMERA_GAP_MS = 250.0
MAX_INFERENCE_P95_MS = 250.0
MIN_REPLAN_COMPLETION_RATIO = 0.90


class ShadowModeError(RuntimeError):
    """A frozen input, live sensor, timing, inference, or no-command gate failed."""


@dataclass(frozen=True)
class LiveFrame:
    role: str
    vid: int
    pid: int
    sequence: int
    capture_perf_ns: int
    receive_monotonic_s: float
    frame_bgr: Any
    jpeg_size: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-bridge", type=Path, required=True)
    parser.add_argument("--preprocess-adapter", type=Path, required=True)
    parser.add_argument("--assembler", type=Path, required=True)
    parser.add_argument("--guard-core", type=Path, required=True)
    parser.add_argument("--sensor-adapter", type=Path, required=True)
    parser.add_argument("--one-shot-launcher", type=Path, required=True)
    parser.add_argument("--one-shot-report", type=Path, required=True)
    parser.add_argument("--offline-runtime-adapter", type=Path, required=True)
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
    parser.add_argument("--duration-seconds", type=float, default=FROZEN_DURATION_SECONDS)
    parser.add_argument("--replan-hz", type=float, default=FROZEN_REPLAN_HZ)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--authorize-live-shadow-read-only-hardware",
        action="store_true",
        help=(
            "Authorize 60 seconds of camera reads and follower Present_Position "
            "reads, local model inference, and logging; no command transmission."
        ),
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


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ShadowModeError(f"invalid JSON: {path}: {exc}") from exc


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ShadowModeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def require_sha(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ShadowModeError(
            f"{label} SHA256 mismatch: expected={expected}, actual={actual}"
        )
    return actual


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


def verify_static_no_command_boundary(path: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    attributes: set[str] = set()
    calls: list[str] = []
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            attributes.add(node.attr)
        elif isinstance(node, ast.Call):
            calls.append(ast.unparse(node))
        elif isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)

    forbidden_attributes = sorted(
        {
            "send_action",
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
        }.intersection(attributes)
    )
    forbidden_follower_calls = sorted(
        expression
        for expression in calls
        if expression.startswith(("follower.connect(", "follower.disconnect("))
    )
    serial_imports = sorted(
        module
        for module in imports
        if module.startswith(("serial", "pyserial"))
    )
    if forbidden_attributes or forbidden_follower_calls or serial_imports:
        raise ShadowModeError(
            "shadow launcher contains a command-capable path: "
            f"attributes={forbidden_attributes}, "
            f"follower_calls={forbidden_follower_calls}, imports={serial_imports}"
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "motor_write_calls": False,
        "torque_calls": False,
        "full_follower_connect_or_disconnect_calls": False,
        "direct_serial_imports": False,
        "pass": True,
    }


def verify_one_shot_gate(
    launcher_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    launcher_sha = require_sha(
        launcher_path,
        EXPECTED_ONE_SHOT_LAUNCHER_SHA256,
        "one-shot launcher",
    )
    report = load_json(report_path)
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise ShadowModeError("unexpected one-shot report schema")
    if report.get("decision") != EXPECTED_ONE_SHOT_DECISION:
        raise ShadowModeError("one-shot report decision mismatch")
    scope = report.get("scope") or {}
    required_true = (
        "serial_port_opened",
        "motor_read_instruction_sent",
        "serial_port_closed",
    )
    if any(scope.get(field) is not True for field in required_true):
        raise ShadowModeError("one-shot report does not prove a completed read/close")
    required_false = (
        "motor_register_write_api_called",
        "goal_position_written",
        "torque_api_called",
        "full_follower_connect_called",
        "full_follower_disconnect_called",
        "camera_configured_or_opened",
        "model_loaded",
        "policy_inference_run",
        "action_sent",
        "hardware_deployment_authorized",
    )
    invalid = [field for field in required_false if scope.get(field) is not False]
    if invalid:
        raise ShadowModeError(f"one-shot no-write scope mismatch: {invalid}")
    read = report.get("one_shot_read") or {}
    if int(read.get("successful_reads", -1)) != 1:
        raise ShadowModeError("one-shot report does not contain exactly one read")
    if read.get("adapter_final_state") != "CLOSED":
        raise ShadowModeError("one-shot adapter did not finish CLOSED")
    actual_q = tuple(float(value) for value in read.get("actual_q", ()))
    if len(actual_q) != 6 or any(not math.isfinite(value) for value in actual_q):
        raise ShadowModeError("one-shot report contains invalid actual_q")
    device = report.get("device") or {}
    if device.get("stable_path") != EXPECTED_FOLLOWER_BY_ID:
        raise ShadowModeError("one-shot report used an unexpected follower identity")
    return {
        "launcher": str(launcher_path),
        "launcher_sha256": launcher_sha,
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "decision": report["decision"],
        "successful_reads": 1,
        "read_duration_ms": float(read["read_duration_ms"]),
        "actual_q": list(actual_q),
        "serial_closed": True,
        "no_write_or_torque_change": True,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ShadowModeError(f"refusing to write empty CSV: {path}")
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ShadowModeError(f"CSV schema changed within {path.name}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class LiveFrameHub:
    """Receive, validate, retain, and pair the latest camera frames."""

    def __init__(self, *, bridge: ModuleType, cv2: Any, np: Any) -> None:
        self.bridge = bridge
        self.cv2 = cv2
        self.np = np
        self.condition = threading.Condition()
        self.frames: dict[str, deque[LiveFrame]] = {
            "front": deque(maxlen=12),
            "wrist": deque(maxlen=12),
        }
        self.states: dict[str, dict[str, Any]] = {
            "front": {"status": "NOT_STARTED", "rows": []},
            "wrist": {"status": "NOT_STARTED", "rows": []},
        }

    def _publish(self, frame: LiveFrame) -> None:
        with self.condition:
            self.frames[frame.role].append(frame)
            self.condition.notify_all()

    def receive(self, role: str, connection: socket.socket, nonce: str) -> None:
        state = self.states[role]
        state.update(
            {
                "status": "RUNNING",
                "decode_failures": 0,
                "crc_failures": 0,
                "protocol_failures": 0,
                "sequence_gap_count": 0,
                "consecutive_exact_duplicate_frames": 0,
                "end_ack_sent": False,
            }
        )
        previous_sequence: int | None = None
        previous_decoded_crc: int | None = None
        try:
            while True:
                header, jpeg = self.bridge.receive_message(connection)
                receive_perf_ns = time.perf_counter_ns()
                receive_monotonic_s = time.monotonic()
                receive_wall_ns = time.time_ns()
                self.bridge.validate_protocol_header(header, nonce)
                if header.get("role") != role:
                    raise ShadowModeError("camera role changed during stream")
                message_type = header.get("type")
                if message_type == "end":
                    if jpeg:
                        state["protocol_failures"] += 1
                        raise ShadowModeError("END contained a payload")
                    frames_sent = int(header.get("frames_sent", -1))
                    if frames_sent != len(state["rows"]):
                        state["protocol_failures"] += 1
                        raise ShadowModeError(
                            f"{role} END count={frames_sent}, received={len(state['rows'])}"
                        )
                    state["end_message"] = dict(header)
                    connection.sendall(self.bridge.STREAM_END_ACK)
                    state["end_ack_sent"] = True
                    state["status"] = "PASS"
                    break
                if message_type != "frame" or not jpeg:
                    state["protocol_failures"] += 1
                    raise ShadowModeError(f"unexpected {role} message: {message_type!r}")

                expected_crc = int(header["jpeg_crc32"])
                actual_crc = zlib.crc32(jpeg) & 0xFFFFFFFF
                if actual_crc != expected_crc:
                    state["crc_failures"] += 1
                    continue
                encoded = self.np.frombuffer(jpeg, dtype=self.np.uint8)
                image = self.cv2.imdecode(encoded, self.cv2.IMREAD_COLOR)
                if image is None or image.size == 0:
                    state["decode_failures"] += 1
                    continue
                if tuple(int(value) for value in image.shape) != (480, 640, 3):
                    state["protocol_failures"] += 1
                    raise ShadowModeError(f"{role} frame shape changed: {image.shape}")

                sequence = int(header["sequence"])
                if sequence < 0:
                    raise ShadowModeError(f"{role} sequence is negative")
                if previous_sequence is not None and sequence != previous_sequence + 1:
                    state["sequence_gap_count"] += max(
                        sequence - previous_sequence - 1,
                        1,
                    )
                previous_sequence = sequence
                decoded_crc = zlib.crc32(
                    memoryview(self.np.ascontiguousarray(image))
                ) & 0xFFFFFFFF
                if previous_decoded_crc is not None and decoded_crc == previous_decoded_crc:
                    state["consecutive_exact_duplicate_frames"] += 1
                previous_decoded_crc = decoded_crc

                capture_perf_ns = int(header["capture_perf_ns"])
                row = {
                    "sequence": sequence,
                    "capture_perf_ns": capture_perf_ns,
                    "capture_wall_ns": int(header["capture_wall_ns"]),
                    "send_perf_ns": int(header["send_perf_ns"]),
                    "send_wall_ns": int(header["send_wall_ns"]),
                    "receive_perf_ns": receive_perf_ns,
                    "receive_wall_ns": receive_wall_ns,
                    "width": int(image.shape[1]),
                    "height": int(image.shape[0]),
                    "channels": int(image.shape[2]),
                    "jpeg_size": len(jpeg),
                    "jpeg_crc32": f"{actual_crc:08x}",
                    "decoded_crc32": f"{decoded_crc:08x}",
                    "capture_to_send_ms": (
                        int(header["send_perf_ns"]) - capture_perf_ns
                    ) / 1_000_000.0,
                    "wall_end_to_end_ms": (
                        receive_wall_ns - int(header["capture_wall_ns"])
                    ) / 1_000_000.0,
                }
                state["rows"].append(row)
                frame = LiveFrame(
                    role=role,
                    vid=0x32E6,
                    pid=0x9221 if role == "front" else 0x9005,
                    sequence=sequence,
                    capture_perf_ns=capture_perf_ns,
                    receive_monotonic_s=receive_monotonic_s,
                    frame_bgr=image,
                    jpeg_size=len(jpeg),
                )
                state["last_frame"] = image.copy()
                self._publish(frame)
        except Exception as exc:
            if state.get("status") == "TIMEOUT":
                state.update(
                    {
                        "cleanup_error_type": type(exc).__name__,
                        "cleanup_error": str(exc),
                        "cleanup_traceback": traceback.format_exc(),
                    }
                )
            else:
                state.update(
                    {
                        "status": "FAILED",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
        finally:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()
            with self.condition:
                self.condition.notify_all()

    def wait_pair(
        self,
        *,
        after_front: int,
        after_wrist: int,
        timeout_seconds: float,
    ) -> tuple[LiveFrame, LiveFrame]:
        deadline = time.monotonic() + timeout_seconds
        with self.condition:
            while True:
                failures = {
                    role: state.get("error")
                    for role, state in self.states.items()
                    if state.get("status") == "FAILED"
                }
                if failures:
                    raise ShadowModeError(f"camera receiver failed: {failures}")
                now = time.monotonic()
                front_candidates = [
                    frame
                    for frame in self.frames["front"]
                    if frame.sequence > after_front
                    and now - frame.receive_monotonic_s <= MAX_CAMERA_AGE_SECONDS
                ]
                wrist_candidates = [
                    frame
                    for frame in self.frames["wrist"]
                    if frame.sequence > after_wrist
                    and now - frame.receive_monotonic_s <= MAX_CAMERA_AGE_SECONDS
                ]
                pairs: list[tuple[float, float, LiveFrame, LiveFrame]] = []
                for front in front_candidates:
                    for wrist in wrist_candidates:
                        capture_skew = abs(
                            front.capture_perf_ns - wrist.capture_perf_ns
                        ) / 1_000_000_000.0
                        receive_skew = abs(
                            front.receive_monotonic_s - wrist.receive_monotonic_s
                        )
                        if (
                            capture_skew <= MAX_CAMERA_SKEW_SECONDS
                            and receive_skew <= MAX_CAMERA_SKEW_SECONDS
                        ):
                            recency = min(
                                front.receive_monotonic_s,
                                wrist.receive_monotonic_s,
                            )
                            pairs.append(
                                (recency, -capture_skew, front, wrist)
                            )
                if pairs:
                    _, _, front, wrist = max(pairs, key=lambda item: (item[0], item[1]))
                    return front, wrist
                if all(
                    state.get("status") == "PASS" for state in self.states.values()
                ):
                    raise ShadowModeError("camera streams ended before a fresh pair arrived")
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise ShadowModeError("timed out waiting for a fresh paired camera frame")
                self.condition.wait(timeout=remaining)


class CameraSession:
    """Lifecycle owner for the reviewed native-Windows camera workers."""

    def __init__(
        self,
        *,
        bridge: ModuleType,
        bridge_path: Path,
        windows_python: str,
        output_dir: Path,
        duration_seconds: float,
        hub: LiveFrameHub,
    ) -> None:
        self.bridge = bridge
        self.bridge_path = bridge_path
        self.output_dir = output_dir
        self.hub = hub
        self.listener: socket.socket | None = None
        self.connections: dict[str, socket.socket] = {}
        self.threads: dict[str, threading.Thread] = {}
        self.processes: dict[str, tuple[Any, Path, float]] = {}
        self.hellos: dict[str, dict[str, Any]] = {}
        self.addresses: dict[str, tuple[str, int]] = {}
        self.process_reports: dict[str, dict[str, Any]] = {}
        self.collected = False
        self.stream_duration_seconds = duration_seconds
        self.nonce = secrets.token_hex(32)
        self.args = SimpleNamespace(
            windows_python=windows_python,
            backend="dshow",
            send_fps=15.0,
            duration_seconds=duration_seconds,
            successful_warmup_frames=5,
            max_warmup_attempts=30,
            jpeg_quality=90,
            connect_timeout_seconds=8.0,
        )
        self.requests = {
            "front": bridge.CameraRequest(
                "front", 0x32E6, 0x9221, 640, 480, 30.0, "MJPG"
            ),
            "wrist": bridge.CameraRequest(
                "wrist", 0x32E6, 0x9005, 640, 480, 15.0, "AUTO"
            ),
        }

    def start(self) -> dict[str, Any]:
        if os.name == "nt":
            raise ShadowModeError("live shadow parent must run in WSL/Linux")
        windows_python = Path(self.args.windows_python)
        if not windows_python.is_file():
            raise FileNotFoundError(f"Windows camera Python missing: {windows_python}")
        target_host = self.bridge.detect_wsl_ipv4()
        script_windows = self.bridge.to_windows_path(self.bridge_path)
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(("0.0.0.0", 0))
        self.listener.listen(2)
        port = int(self.listener.getsockname()[1])

        for role in ("front", "wrist"):
            launched = self.bridge.launch_worker(
                self.requests[role],
                self.args,
                script_windows,
                target_host,
                port,
                self.nonce,
                self.output_dir,
            )
            process = launched[0]
            # The Windows worker can emit OpenCV/driver diagnostics in the
            # active Windows code page.  The reviewed bridge creates a text
            # pipe whose WSL-side default is UTF-8; make teardown diagnostics
            # lossy-but-safe so a non-UTF-8 warning cannot invalidate an
            # otherwise completed shadow run or bypass cleanup/reporting.
            stdout_stream = getattr(process, "stdout", None)
            reconfigure = getattr(stdout_stream, "reconfigure", None)
            if not callable(reconfigure):
                raise ShadowModeError(
                    f"{role} Windows worker stdout cannot be configured safely"
                )
            reconfigure(errors="replace")
            self.processes[role] = launched
            connection, hello, address = self.bridge.accept_expected_role(
                self.listener,
                role,
                self.nonce,
                12.0,
            )
            selected = hello.get("selected_camera") or {}
            if int(selected.get("vid", -1)) != self.requests[role].vid:
                raise ShadowModeError(f"{role} worker VID mismatch")
            if int(selected.get("pid", -1)) != self.requests[role].pid:
                raise ShadowModeError(f"{role} worker PID mismatch")
            actual = hello.get("actual_capture") or {}
            if (int(actual.get("width", -1)), int(actual.get("height", -1))) != (
                640,
                480,
            ):
                raise ShadowModeError(f"{role} camera dimensions mismatch")
            self.connections[role] = connection
            self.hellos[role] = hello
            self.addresses[role] = address

        for role in ("front", "wrist"):
            thread = threading.Thread(
                target=self.hub.receive,
                args=(role, self.connections[role], self.nonce),
                name=f"live-shadow-camera-{role}",
                daemon=True,
            )
            self.threads[role] = thread
            thread.start()
        for role in ("front", "wrist"):
            self.connections[role].sendall(self.bridge.STREAM_START_SIGNAL)
        return {
            "listen_port": port,
            "windows_target_host": target_host,
            "stream_duration_seconds": self.stream_duration_seconds,
            "front": self.hellos["front"],
            "wrist": self.hellos["wrist"],
        }

    def finish(self, grace_seconds: float = 15.0) -> None:
        # finish() is called after the 60 s shadow loop.  The workers have only
        # the short camera tail left, so bound cleanup by the grace period rather
        # than adding the full stream duration a second time.
        deadline = time.monotonic() + grace_seconds
        for role in ("front", "wrist"):
            remaining = max(deadline - time.monotonic(), 0.01)
            self.threads[role].join(timeout=remaining)
            if self.threads[role].is_alive():
                self.hub.states[role].update(
                    {
                        "status": "TIMEOUT",
                        "error": "camera receiver exceeded hard deadline",
                    }
                )
        self._close_listener_and_connections()
        self._collect_processes(12.0)

    def abort(self) -> None:
        self._close_listener_and_connections()
        for process, _, _ in self.processes.values():
            if process.poll() is None:
                process.kill()
        for thread in self.threads.values():
            thread.join(timeout=2.0)
        self._collect_processes(3.0)

    def _close_listener_and_connections(self) -> None:
        if self.listener is not None:
            try:
                self.listener.close()
            except OSError:
                pass
            self.listener = None
        for connection in self.connections.values():
            try:
                connection.close()
            except OSError:
                pass

    def _collect_processes(self, timeout_seconds: float) -> None:
        if self.collected:
            return
        for role, (process, result_path, started) in self.processes.items():
            code, stdout, timed_out = self.bridge.collect_process(
                process,
                timeout_seconds,
            )
            if result_path.is_file():
                result = load_json(result_path)
            else:
                result = {
                    "status": "FAILED",
                    "error": "Windows worker result was not created",
                }
            result["exit_code"] = code
            result["timed_out_during_collection"] = timed_out
            result["stdout"] = stdout
            result["parent_observed_seconds"] = time.monotonic() - started
            self.process_reports[role] = result
        self.collected = True

    def report(self) -> tuple[dict[str, Any], list[str]]:
        cameras: dict[str, Any] = {}
        reasons: list[str] = []
        for role in ("wrist", "front"):
            state = self.hub.states[role]
            rows = state.get("rows", [])
            metrics = (
                self.bridge.stream_metrics(rows, self.stream_duration_seconds)
                if rows
                else None
            )
            worker = self.process_reports.get(role) or {}
            cameras[role] = {
                "receiver_status": state.get("status", "NOT_STARTED"),
                "receiver_error": state.get("error"),
                "end_ack_sent": state.get("end_ack_sent", False),
                "decode_failures": int(state.get("decode_failures", 0)),
                "crc_failures": int(state.get("crc_failures", 0)),
                "protocol_failures": int(state.get("protocol_failures", 0)),
                "sequence_gap_count": int(state.get("sequence_gap_count", 0)),
                "consecutive_exact_duplicate_frames": int(
                    state.get("consecutive_exact_duplicate_frames", 0)
                ),
                "metrics": metrics,
                "hello": self.hellos.get(role),
                "peer_address": (
                    list(self.addresses[role]) if role in self.addresses else None
                ),
                "windows_worker": worker,
            }
            if state.get("status") != "PASS":
                reasons.append(f"{role}_receiver_{state.get('status', 'missing').lower()}")
            if state.get("end_ack_sent") is not True:
                reasons.append(f"{role}_receiver_missing_end_ack")
            if worker.get("status") != "PASS" or worker.get("exit_code") != 0:
                reasons.append(f"{role}_windows_worker_failed")
            if worker.get("end_ack_received") is not True:
                reasons.append(f"{role}_windows_worker_missing_end_ack")
            if worker.get("timed_out_during_collection"):
                reasons.append(f"{role}_windows_worker_collection_timeout")
            for field in (
                "decode_failures",
                "crc_failures",
                "protocol_failures",
                "sequence_gap_count",
            ):
                if int(state.get(field, 0)) != 0:
                    reasons.append(f"{role}_{field}_{state.get(field)}")
            if metrics is None:
                reasons.append(f"{role}_metrics_missing")
            else:
                if float(metrics["achieved_receive_fps"]) < MIN_CAMERA_FPS:
                    reasons.append(f"{role}_fps_below_{MIN_CAMERA_FPS}")
                if float(metrics["receive_span_ratio"]) < 0.90:
                    reasons.append(f"{role}_span_ratio_below_0.90")
                if (
                    metrics.get("gap_ms_max") is None
                    or float(metrics["gap_ms_max"]) > MAX_CAMERA_GAP_MS
                ):
                    reasons.append(f"{role}_frame_gap_above_{MAX_CAMERA_GAP_MS}ms")
                if metrics.get("unique_dimensions") != [[640, 480, 3]]:
                    reasons.append(f"{role}_dimensions_changed")

        front_rows = self.hub.states["front"].get("rows", [])
        wrist_rows = self.hub.states["wrist"].get("rows", [])
        capture_skew = self.bridge.nearest_skew_metrics(
            [int(row["capture_perf_ns"]) for row in wrist_rows],
            [int(row["capture_perf_ns"]) for row in front_rows],
            "Windows producer capture perf_counter_ns",
        )
        receive_skew = self.bridge.nearest_skew_metrics(
            [int(row["receive_perf_ns"]) for row in wrist_rows],
            [int(row["receive_perf_ns"]) for row in front_rows],
            "WSL receiver perf_counter_ns",
        )
        if capture_skew is None:
            reasons.append("capture_skew_unavailable")
        elif float(capture_skew["skew_ms_p95"]) > MAX_CAMERA_SKEW_SECONDS * 1000.0:
            reasons.append("capture_skew_p95_above_100ms")
        return {
            "cameras": cameras,
            "capture_skew": capture_skew,
            "receive_skew": receive_skew,
        }, sorted(set(reasons))


def inference_chunk(policy: Any, observation: dict[str, Any], np: Any, torch: Any):
    started = time.perf_counter()
    policy.reset()
    with torch.inference_mode():
        chunk = policy.predict_action_chunk(observation)
    device = torch.device(policy.config.device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    array = chunk.detach().cpu().float().numpy()
    if array.shape != (1, EXPECTED_CHUNK_SIZE, 6):
        raise ShadowModeError(f"unexpected ACT chunk shape: {array.shape}")
    if not bool(np.isfinite(array).all()):
        raise ShadowModeError("ACT shadow output contains NaN/Inf")
    return array[0], elapsed_ms


def warm_policy(policy: Any, limits: Any, np: Any, torch: Any) -> float:
    device = torch.device(policy.config.device)
    state = tuple(limits.home_command) + tuple(limits.home_command) + (0.0,) * 6
    batch = {
        STATE_KEY: torch.as_tensor(state, dtype=torch.float32, device=device).reshape(1, 18),
        FRONT_KEY: torch.zeros((1, 3, 480, 480), dtype=torch.float32, device=device),
        WRIST_KEY: torch.zeros((1, 3, 480, 480), dtype=torch.float32, device=device),
    }
    _, elapsed_ms = inference_chunk(policy, batch, np, torch)
    return elapsed_ms


def task_workspace_status(actual_q: Sequence[float], limits: Any) -> dict[str, Any]:
    violations: dict[str, dict[str, Any]] = {}
    home_error: dict[str, float] = {}
    tracking_ready: dict[str, bool] = {}
    for index, motor in enumerate(MOTOR_ORDER):
        value = float(actual_q[index])
        minimum, maximum = limits.soft_limits[index]
        if not minimum <= value <= maximum:
            violations[motor] = {
                "value": value,
                "soft_limit": [float(minimum), float(maximum)],
            }
        error = abs(value - float(limits.home_command[index]))
        home_error[motor] = error
        tracking_ready[motor] = error <= float(limits.tracking_error_limits[index])
    return {
        "inside_task_soft_limits": not violations,
        "soft_limit_violations": violations,
        "absolute_error_from_home": home_error,
        "within_home_tracking_limits_by_joint": tracking_ready,
        "motion_ready": not violations and all(tracking_ready.values()),
    }


def flatten_replan_row(
    *,
    index: int,
    loop_elapsed_s: float,
    front: LiveFrame,
    wrist: LiveFrame,
    assembly: Any,
    joint_result: Any,
    inference_ms: float,
    actual_q: Sequence[float],
    workspace: dict[str, Any],
    rate_any: int,
    soft_any: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "replan_index": index,
        "loop_elapsed_s": loop_elapsed_s,
        "front_sequence": front.sequence,
        "wrist_sequence": wrist.sequence,
        "front_age_ms": assembly.front_age_ms,
        "wrist_age_ms": assembly.wrist_age_ms,
        "joint_age_ms": assembly.joint_age_ms,
        "capture_skew_ms": assembly.capture_skew_ms,
        "receive_skew_ms": assembly.receive_skew_ms,
        "joint_read_ms": joint_result.metrics.read_duration_ms,
        "inference_ms": inference_ms,
        "hypothetical_commands_with_rate_clip": rate_any,
        "hypothetical_commands_with_soft_clip": soft_any,
        "actual_inside_task_soft_limits": int(workspace["inside_task_soft_limits"]),
        "motion_ready": int(workspace["motion_ready"]),
    }
    for motor, value in zip(MOTOR_ORDER, actual_q, strict=True):
        row[f"actual_q_{motor}"] = float(value)
        row[f"home_error_{motor}"] = float(
            workspace["absolute_error_from_home"][motor]
        )
    return row


def flatten_command_row(
    replan_index: int,
    substep: int,
    step: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "replan_index": replan_index,
        "substep": substep,
        "diagnostic_only": 1,
        "sent_to_hardware": 0,
        "any_rate_clip": int(step.any_rate_clip),
        "any_soft_clip": int(step.any_soft_clip),
    }
    for index, motor in enumerate(MOTOR_ORDER):
        row[f"raw_delta_{motor}"] = float(step.raw_delta[index])
        row[f"guarded_delta_{motor}"] = float(step.sent_delta[index])
        row[f"hypothetical_command_{motor}"] = float(step.sent_command[index])
        row[f"rate_clipped_{motor}"] = int(step.rate_clipped[index])
        row[f"soft_clipped_{motor}"] = int(step.soft_clipped[index])
    return row


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.authorize_live_shadow_read_only_hardware:
        raise ShadowModeError(
            "live shadow not authorized: explicit read-only hardware flag is required"
        )
    if abs(float(args.duration_seconds) - FROZEN_DURATION_SECONDS) > 1e-12:
        raise ShadowModeError("duration-seconds is frozen at 60.0 for this gate")
    if abs(float(args.replan_hz) - FROZEN_REPLAN_HZ) > 1e-12:
        raise ShadowModeError("replan-hz is frozen at 3.0 for this gate")

    script_path = Path(__file__).resolve()
    bridge_path = require_file(args.camera_bridge)
    preprocess_path = require_file(args.preprocess_adapter)
    assembler_path = require_file(args.assembler)
    guard_path = require_file(args.guard_core)
    sensor_adapter_path = require_file(args.sensor_adapter)
    one_shot_launcher_path = require_file(args.one_shot_launcher)
    one_shot_report_path = require_file(args.one_shot_report)
    offline_path = require_file(args.offline_runtime_adapter)
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
            f"output directory already exists; preserve it and choose a retry path: {output_dir}"
        )

    print("===== LIVE SHADOW AUTHORIZATION BOUNDARY =====", flush=True)
    print("duration=60.0s replan=3.0Hz; cameras + joint READ + ACT inference", flush=True)
    print("policy outputs: diagnostics/CSV only", flush=True)
    print("Goal Position, motor writes, torque calls and action APIs: FORBIDDEN", flush=True)
    print("motion and hardware deployment authorization: false", flush=True)

    print("\n===== VERIFY FROZEN NON-COMMANDING STACK =====", flush=True)
    static_boundary = verify_static_no_command_boundary(script_path)
    hashes = {
        "camera_bridge": require_sha(bridge_path, EXPECTED_BRIDGE_SHA256, "camera bridge"),
        "preprocess_adapter": require_sha(
            preprocess_path, EXPECTED_PREPROCESS_SHA256, "preprocess adapter"
        ),
        "assembler": require_sha(assembler_path, EXPECTED_ASSEMBLER_SHA256, "assembler"),
        "guard_core": require_sha(guard_path, EXPECTED_GUARD_SHA256, "guard core"),
        "sensor_adapter": require_sha(
            sensor_adapter_path, EXPECTED_SENSOR_ADAPTER_SHA256, "sensor adapter"
        ),
        "offline_runtime_adapter": require_sha(
            offline_path, EXPECTED_OFFLINE_ADAPTER_SHA256, "offline runtime adapter"
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
    one_shot_gate = verify_one_shot_gate(
        one_shot_launcher_path,
        one_shot_report_path,
    )
    bridge = load_module(bridge_path, "act_v3_live_shadow_bridge")
    preprocess = load_module(preprocess_path, "act_v3_live_shadow_preprocess")
    assembler_module = load_module(assembler_path, "act_v3_live_shadow_assembler")
    guard = load_module(guard_path, "act_v3_live_shadow_guard")
    sensor_adapter = load_module(sensor_adapter_path, "act_v3_live_shadow_sensor")
    one_shot = load_module(one_shot_launcher_path, "act_v3_live_shadow_one_shot")
    offline = load_module(offline_path, "act_v3_live_shadow_offline")
    frozen_inputs = guard.verify_frozen_inputs(contract_path, calibration_path)
    limits = guard.RuntimeLimits.frozen_v3()
    guard_audit = guard.run_algorithm_self_audit(limits)
    if guard_audit["tests_passed"] != guard_audit["tests_total"]:
        raise ShadowModeError("guard self-audit did not fully pass")
    preprocess.verify_static_offline_boundary(preprocess_path)
    sensor_adapter.verify_static_adapter_boundary(sensor_adapter_path)
    print("source identities, no-command boundary and prior one-shot gate: PASS", flush=True)
    print(
        f"guard algorithm: {guard_audit['tests_passed']}/"
        f"{guard_audit['tests_total']} PASS",
        flush=True,
    )

    print("\n===== VERIFY FROZEN 11K RELEASE AND WARM MODEL =====", flush=True)
    release = offline.verify_release(candidate_root, source_checkpoint_root)
    if int(release.get("selected_step", -1)) != EXPECTED_POLICY_STEP:
        raise ShadowModeError("release is not the frozen 11K checkpoint")
    import cv2
    import numpy as np
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy

    if not torch.cuda.is_available():
        raise ShadowModeError("CUDA is unavailable")
    policy = ACTPolicy.from_pretrained(
        candidate_root / "pretrained_model",
        local_files_only=True,
    )
    policy.eval()
    policy_report = offline.verify_policy_contract(policy)
    warmup_ms = warm_policy(policy, limits, np, torch)
    print(
        "policy state=18D images=2x3x480x480 action=6D chunk=10 queue=5: PASS",
        flush=True,
    )
    print(f"pre-hardware CUDA warmup inference: {warmup_ms:.3f} ms", flush=True)

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
    hub = LiveFrameHub(bridge=bridge, cv2=cv2, np=np)
    camera_session = CameraSession(
        bridge=bridge,
        bridge_path=bridge_path,
        windows_python=args.windows_python,
        output_dir=output_dir,
        duration_seconds=FROZEN_DURATION_SECONDS + CAMERA_STREAM_TAIL_SECONDS,
        hub=hub,
    )
    state_guard = guard.DeltaCommandGuard(
        limits,
        limits.home_command,
        (0.0,) * 6,
    )
    observation_assembler = assembler_module.LiveObservationAssembler(
        preprocess_module=preprocess,
        delta_guard=state_guard,
        cv2_module=cv2,
        np_module=np,
        torch_module=torch,
        device=torch.device(policy.config.device),
        max_camera_age_seconds=MAX_CAMERA_AGE_SECONDS,
        max_joint_age_seconds=MAX_JOINT_AGE_SECONDS,
        max_capture_skew_seconds=MAX_CAMERA_SKEW_SECONDS,
        max_receive_skew_seconds=MAX_CAMERA_SKEW_SECONDS,
    )
    sensor = sensor_adapter.ReadOnlyJointSensorAdapter(
        bus=follower.bus,
        validate_positions=state_guard.validate_measured_positions,
        packet_factory=assembler_module.JointStatePacket,
        monotonic=time.monotonic,
        max_read_duration_seconds=MAX_JOINT_AGE_SECONDS,
    )

    camera_start_report: dict[str, Any] | None = None
    replan_rows: list[dict[str, Any]] = []
    command_rows: list[dict[str, Any]] = []
    inference_times_ms: list[float] = []
    joint_read_times_ms: list[float] = []
    assembly_ages: dict[str, list[float]] = {
        "front": [],
        "wrist": [],
        "joint": [],
        "capture_skew": [],
        "receive_skew": [],
    }
    rate_clips = Counter({motor: 0 for motor in MOTOR_ORDER})
    soft_clips = Counter({motor: 0 for motor in MOTOR_ORDER})
    commands_any_rate = 0
    commands_any_soft = 0
    workspace_first: dict[str, Any] | None = None
    workspace_last: dict[str, Any] | None = None
    shadow_started: float | None = None
    shadow_finished: float | None = None
    last_front_sequence = -1
    last_wrist_sequence = -1
    sensor_opened = False
    sensor_closed = False
    camera_finished = False

    try:
        print("\n===== START WINDOWS CAMERAS AND READ-ONLY FOLLOWER BUS =====", flush=True)
        camera_start_report = camera_session.start()
        for role in ("front", "wrist"):
            actual = camera_start_report[role]["actual_capture"]
            print(
                f"{role}: READY {actual['width']}x{actual['height']}/"
                f"{actual['fourcc'] or 'UNKNOWN'}",
                flush=True,
            )
        sensor.open()
        sensor_opened = True
        print("follower bus: OPEN for Present_Position reads only", flush=True)

        print("\n===== 60-SECOND LIVE SHADOW =====", flush=True)
        interval = 1.0 / FROZEN_REPLAN_HZ
        target_replans = int(round(FROZEN_DURATION_SECONDS * FROZEN_REPLAN_HZ))
        shadow_started = time.monotonic()
        deadline = shadow_started + FROZEN_DURATION_SECONDS
        next_tick = shadow_started
        replan_index = 0
        while replan_index < target_replans:
            now = time.monotonic()
            if now >= deadline:
                break
            if now < next_tick:
                time.sleep(next_tick - now)
            if time.monotonic() >= deadline:
                break

            front, wrist = hub.wait_pair(
                after_front=last_front_sequence,
                after_wrist=last_wrist_sequence,
                timeout_seconds=PAIR_WAIT_TIMEOUT_SECONDS,
            )
            joint_result = sensor.read()
            now = time.monotonic()
            front_packet = assembler_module.CameraFramePacket(
                role="front",
                vid=front.vid,
                pid=front.pid,
                sequence=front.sequence,
                capture_monotonic_ns=front.capture_perf_ns,
                receive_monotonic_s=front.receive_monotonic_s,
                frame_bgr=front.frame_bgr,
            )
            wrist_packet = assembler_module.CameraFramePacket(
                role="wrist",
                vid=wrist.vid,
                pid=wrist.pid,
                sequence=wrist.sequence,
                capture_monotonic_ns=wrist.capture_perf_ns,
                receive_monotonic_s=wrist.receive_monotonic_s,
                frame_bgr=wrist.frame_bgr,
            )
            observation, assembly = observation_assembler.assemble(
                front=front_packet,
                wrist=wrist_packet,
                joints=joint_result.packet,
                now_monotonic_s=now,
            )
            chunk, inference_ms = inference_chunk(policy, observation, np, torch)
            actual_q = tuple(float(value) for value in joint_result.packet.actual_q)
            workspace = task_workspace_status(actual_q, limits)
            workspace_first = workspace_first or workspace
            workspace_last = workspace

            local_guard = guard.DeltaCommandGuard(
                limits,
                limits.home_command,
                (0.0,) * 6,
            )
            local_rate_any = 0
            local_soft_any = 0
            for substep in range(EXPECTED_QUEUE_LENGTH):
                step = local_guard.apply_delta(chunk[substep])
                local_rate_any += int(step.any_rate_clip)
                local_soft_any += int(step.any_soft_clip)
                commands_any_rate += int(step.any_rate_clip)
                commands_any_soft += int(step.any_soft_clip)
                for motor_index, motor in enumerate(MOTOR_ORDER):
                    rate_clips[motor] += int(step.rate_clipped[motor_index])
                    soft_clips[motor] += int(step.soft_clipped[motor_index])
                command_rows.append(flatten_command_row(replan_index, substep, step))

            replan_rows.append(
                flatten_replan_row(
                    index=replan_index,
                    loop_elapsed_s=now - shadow_started,
                    front=front,
                    wrist=wrist,
                    assembly=assembly,
                    joint_result=joint_result,
                    inference_ms=inference_ms,
                    actual_q=actual_q,
                    workspace=workspace,
                    rate_any=local_rate_any,
                    soft_any=local_soft_any,
                )
            )
            inference_times_ms.append(inference_ms)
            joint_read_times_ms.append(joint_result.metrics.read_duration_ms)
            assembly_ages["front"].append(assembly.front_age_ms)
            assembly_ages["wrist"].append(assembly.wrist_age_ms)
            assembly_ages["joint"].append(assembly.joint_age_ms)
            assembly_ages["capture_skew"].append(assembly.capture_skew_ms)
            assembly_ages["receive_skew"].append(assembly.receive_skew_ms)
            last_front_sequence = front.sequence
            last_wrist_sequence = wrist.sequence
            replan_index += 1
            next_tick += interval
            if replan_index % 30 == 0:
                print(
                    f"replans={replan_index}/{target_replans} "
                    f"elapsed={time.monotonic() - shadow_started:.1f}s "
                    f"inference={inference_ms:.1f}ms",
                    flush=True,
                )
        shadow_finished = time.monotonic()
        sensor.close()
        sensor_closed = True
        print("follower bus: CLOSED with disable_torque=False", flush=True)
        camera_session.finish()
        camera_finished = True
    finally:
        if sensor_opened and not sensor_closed:
            try:
                sensor.close()
                sensor_closed = True
            except Exception as cleanup_exc:
                print(
                    f"WARNING: torque-preserving sensor close failed: {cleanup_exc}",
                    file=sys.stderr,
                    flush=True,
                )
        if not camera_finished:
            camera_session.abort()

    if not replan_rows or shadow_started is None or shadow_finished is None:
        raise ShadowModeError("live shadow produced no completed replans")
    if sensor.state is not sensor_adapter.AdapterState.CLOSED:
        raise ShadowModeError(f"sensor adapter final state is {sensor.state}")
    if bool(getattr(follower.bus, "is_connected", True)):
        raise ShadowModeError("follower bus remained connected")

    camera_report, camera_failures = camera_session.report()
    shadow_elapsed = shadow_finished - shadow_started
    replans = len(replan_rows)
    target_replans = int(round(FROZEN_DURATION_SECONDS * FROZEN_REPLAN_HZ))
    minimum_replans = int(math.floor(target_replans * MIN_REPLAN_COMPLETION_RATIO))
    achieved_replan_hz = (
        (replans - 1)
        / (replan_rows[-1]["loop_elapsed_s"] - replan_rows[0]["loop_elapsed_s"])
        if replans >= 2
        and replan_rows[-1]["loop_elapsed_s"] > replan_rows[0]["loop_elapsed_s"]
        else 0.0
    )
    inference_p95 = percentile(inference_times_ms, 0.95)
    failures = list(camera_failures)
    if replans < minimum_replans:
        failures.append(f"replans_{replans}_below_{minimum_replans}")
    if shadow_elapsed < FROZEN_DURATION_SECONDS * 0.98:
        failures.append("shadow_elapsed_below_98_percent")
    if inference_p95 is None or inference_p95 > MAX_INFERENCE_P95_MS:
        failures.append("inference_p95_above_250ms")
    if sensor.sequence != replans:
        failures.append("joint_read_sequence_mismatch")
    if not sensor_closed:
        failures.append("serial_not_closed")
    failures = sorted(set(failures))

    motion_ready = bool(workspace_last and workspace_last["motion_ready"])
    if failures:
        decision = "LIVE_SHADOW_REVIEW_REQUIRED_MOTION_REMAINS_BLOCKED"
    elif motion_ready:
        decision = "LIVE_SHADOW_PASS_PREPARE_GUARDED_HOLD_NEXT"
    else:
        decision = (
            "LIVE_SHADOW_PASS_PIPELINE_START_POSE_NOT_MOTION_READY_"
            "PREPARE_CONTROLLED_HOME_GATE_NEXT"
        )

    for role in ("front", "wrist"):
        state = hub.states[role]
        write_csv(output_dir / f"{role}_shadow_frames.csv", state["rows"])
        last_frame = state.get("last_frame")
        if last_frame is not None:
            if not cv2.imwrite(str(output_dir / f"{role}_shadow_last.png"), last_frame):
                raise ShadowModeError(f"failed to save {role} shadow snapshot")
    write_csv(output_dir / "live_shadow_replans.csv", replan_rows)
    write_csv(output_dir / "live_shadow_hypothetical_commands.csv", command_rows)

    report = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "raw_dataset_archive_note": {
            "windows_path": r"F:\episodes_pick_place_pilot_v5",
            "wsl_path": "/mnt/f/episodes_pick_place_pilot_v5",
            "used_by_live_shadow": False,
        },
        "authorization": {
            "kind": "explicit_cli_live_shadow_read_only",
            "duration_seconds": FROZEN_DURATION_SECONDS,
            "camera_reads_authorized": True,
            "present_position_reads_authorized": True,
            "model_inference_authorized": True,
            "command_transmission_authorized": False,
            "motion_authorized": False,
            "hardware_deployment_authorized": False,
        },
        "scope": {
            "live_front_camera_opened_by_windows_worker": True,
            "live_wrist_camera_opened_by_windows_worker": True,
            "follower_serial_port_opened": True,
            "present_position_reads": sensor.sequence,
            "model_loaded": True,
            "policy_inference_replans": replans,
            "hypothetical_guarded_commands_logged": len(command_rows),
            "motor_register_write_api_called": False,
            "goal_position_written": False,
            "torque_api_called": False,
            "action_api_called": False,
            "command_sent": False,
            "serial_port_closed": sensor_closed,
            "motion_authorized": False,
            "hardware_deployment_authorized": False,
        },
        "static_no_command_boundary": static_boundary,
        "frozen_dependencies": {
            "hashes": hashes,
            "one_shot_gate": one_shot_gate,
            "frozen_inputs": frozen_inputs,
            "guard_algorithm_audit": guard_audit,
            "release": release,
            "policy": policy_report,
            "policy_warmup_ms": warmup_ms,
            "follower_device": device,
            "follower": follower_summary,
            "imported_follower_modules": imported_modules,
        },
        "state_contract": {
            "actual_q": "live normalized Present_Position",
            "previous_command": list(limits.home_command),
            "previous_delta": [0.0] * 6,
            "note": (
                "Home/zero are the frozen training startup baseline. No predicted "
                "command is committed to observation state because no command is sent."
            ),
        },
        "camera_start": camera_start_report,
        "camera_transport": camera_report,
        "shadow_metrics": {
            "requested_duration_seconds": FROZEN_DURATION_SECONDS,
            "elapsed_seconds": shadow_elapsed,
            "target_replans": target_replans,
            "minimum_accepted_replans": minimum_replans,
            "completed_replans": replans,
            "achieved_replan_hz": achieved_replan_hz,
            "joint_reads": sensor.sequence,
            "joint_read_ms_p50": percentile(joint_read_times_ms, 0.50),
            "joint_read_ms_p95": percentile(joint_read_times_ms, 0.95),
            "joint_read_ms_max": max(joint_read_times_ms),
            "inference_ms_p50": percentile(inference_times_ms, 0.50),
            "inference_ms_p95": inference_p95,
            "inference_ms_max": max(inference_times_ms),
            "front_age_ms_p95": percentile(assembly_ages["front"], 0.95),
            "wrist_age_ms_p95": percentile(assembly_ages["wrist"], 0.95),
            "joint_age_ms_p95": percentile(assembly_ages["joint"], 0.95),
            "capture_skew_ms_p95": percentile(
                assembly_ages["capture_skew"], 0.95
            ),
            "receive_skew_ms_p95": percentile(
                assembly_ages["receive_skew"], 0.95
            ),
            "commands_with_any_rate_clip_diagnostic": commands_any_rate,
            "commands_with_any_soft_clip_diagnostic": commands_any_soft,
            "per_joint_rate_clips_diagnostic": dict(rate_clips),
            "per_joint_soft_clips_diagnostic": dict(soft_clips),
            "predictions_sent_to_hardware": 0,
        },
        "startup_workspace": workspace_first,
        "final_workspace": workspace_last,
        "motion_readiness": {
            "ready": motion_ready,
            "pipeline_pass_does_not_imply_motion_ready": True,
            "next_motion_gate": (
                "guarded_hold" if motion_ready else "controlled_home_approach"
            ),
        },
        "acceptance_failures": failures,
        "decision": decision,
        "motion_authorized": False,
        "hardware_deployment_authorized": False,
    }
    report_path = output_dir / "live_shadow_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\n===== LIVE SHADOW METRICS =====", flush=True)
    print(
        f"elapsed={shadow_elapsed:.3f}s replans={replans}/{target_replans} "
        f"rate={achieved_replan_hz:.3f}Hz",
        flush=True,
    )
    print(
        f"joint_read p95={percentile(joint_read_times_ms, 0.95):.3f}ms "
        f"max={max(joint_read_times_ms):.3f}ms",
        flush=True,
    )
    print(
        f"inference p95={inference_p95:.3f}ms "
        f"max={max(inference_times_ms):.3f}ms",
        flush=True,
    )
    print(
        f"hypothetical rate clips={commands_any_rate}/{len(command_rows)} "
        f"soft clips={commands_any_soft}/{len(command_rows)}",
        flush=True,
    )
    print(f"motion-ready start pose: {motion_ready}", flush=True)
    if workspace_last and workspace_last["soft_limit_violations"]:
        print(
            f"task soft-limit violations: {workspace_last['soft_limit_violations']}",
            flush=True,
        )

    print("\n===== DECISION =====", flush=True)
    print(f"decision='{decision}'", flush=True)
    if failures:
        print(f"acceptance_failures={failures}", flush=True)
    print("MOTION AND HARDWARE DEPLOYMENT REMAIN BLOCKED.", flush=True)
    print("NO GOAL POSITION, REGISTER WRITE, TORQUE, OR ACTION API WAS CALLED.", flush=True)
    print("ALL POLICY OUTPUTS WERE LOGGED LOCALLY AND DISCARDED.", flush=True)
    print("FOLLOWER SERIAL AND BOTH CAMERA WORKERS ARE CLOSED.", flush=True)
    print("\n===== OUTPUT =====", flush=True)
    print(report_path, flush=True)
    print(output_dir / "live_shadow_replans.csv", flush=True)
    print(output_dir / "live_shadow_hypothetical_commands.csv", flush=True)
    print("ACT V3 LIVE SHADOW MODE: PASS" if not failures else "ACT V3 LIVE SHADOW MODE: REVIEW", flush=True)

    del policy
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0 if not failures else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(
            f"ACT V3 LIVE SHADOW MODE: FAIL: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        print("MOTION AND HARDWARE DEPLOYMENT REMAIN BLOCKED.", file=sys.stderr, flush=True)
        print(
            "The launcher contains no motor write, Goal Position, torque, or action call.",
            file=sys.stderr,
            flush=True,
        )
        print(
            "If live access began, cleanup attempted to close the follower bus without "
            "changing torque and to terminate both camera workers.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)
