#!/usr/bin/env python3
"""Audit a native-Windows to WSL dual-camera transport bridge.

Parent mode runs under WSL/Linux. It opens a local TCP listener, launches two
native-Windows Python workers, receives JPEG frames, decodes them, and audits
throughput, ordering, frame gaps, dimensions, and cross-camera capture skew.

Worker mode runs under native Windows. Each worker opens exactly one physical
camera by VID:PID with the already reviewed DirectShow mode and sends a
rate-limited frame stream to WSL. The front and wrist cameras remain in
separate Windows processes, matching the topology that passed the 60-second
native-Windows dual-camera audit.

Camera reads only. No LeRobot, robot, teleoperator, serial, torch, policy, or
command API is imported or used.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import ipaddress
import json
import math
import os
import platform
import secrets
import socket
import struct
import subprocess
import sys
import threading
import time
import traceback
import zlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROTOCOL_MAGIC = "ACT_V3_CAMERA_BRIDGE"
PROTOCOL_VERSION = 2
MAX_HEADER_BYTES = 64 * 1024
MAX_JPEG_BYTES = 4 * 1024 * 1024
DEFAULT_VID = 0x32E6
STREAM_START_SIGNAL = b"\x01"
STREAM_END_ACK = b"\x02"


@dataclass(frozen=True)
class CameraRequest:
    role: str
    vid: int
    pid: int
    width: int
    height: int
    device_fps: float
    fourcc: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_hex_id(value: str) -> int:
    text = value.strip().lower()
    if text.startswith("0x"):
        text = text[2:]
    return int(text, 16)


def normalize_usb_id(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text or text == "--":
        return None
    try:
        return parse_hex_id(text)
    except ValueError:
        return None


def decode_fourcc(value: float) -> str:
    number = int(value)
    return "".join(chr((number >> (8 * shift)) & 0xFF) for shift in range(4)).rstrip("\x00")


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def camera_info_payload(info: Any) -> dict[str, Any]:
    vid = normalize_usb_id(info.vid)
    pid = normalize_usb_id(info.pid)
    return {
        "index": int(info.index),
        "name": str(info.name),
        "vid": vid,
        "pid": pid,
        "vid_hex": None if vid is None else f"{vid:04X}",
        "pid_hex": None if pid is None else f"{pid:04X}",
        "path": str(info.path),
        "backend": int(info.backend),
    }


def backend_code(name: str) -> int:
    import cv2

    return {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}[name]


def find_exact_camera(request: CameraRequest, backend: int) -> tuple[Any, list[dict[str, Any]]]:
    from cv2_enumerate_cameras import enumerate_cameras

    discovered = list(enumerate_cameras(backend))
    inventory = [camera_info_payload(item) for item in discovered]
    matches = [
        item
        for item in discovered
        if normalize_usb_id(item.vid) == request.vid and normalize_usb_id(item.pid) == request.pid
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {request.role} camera with VID:PID "
            f"{request.vid:04X}:{request.pid:04X}; found {len(matches)}"
        )
    return matches[0], inventory


def capture_properties(cap: Any, cv2: Any) -> dict[str, Any]:
    return {
        "width": int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
        "height": int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        "fps_reported": float(cap.get(cv2.CAP_PROP_FPS)),
        "fourcc": decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC)),
        "backend_reported": cap.getBackendName(),
    }


def constructor_open_parameters(request: CameraRequest, cv2: Any) -> list[int]:
    if not float(request.device_fps).is_integer():
        raise ValueError("Device FPS must be integer-valued for constructor parameters")
    params: list[int] = []
    if request.fourcc != "AUTO":
        params.extend(
            [int(cv2.CAP_PROP_FOURCC), int(cv2.VideoWriter_fourcc(*request.fourcc))]
        )
    params.extend(
        [
            int(cv2.CAP_PROP_FRAME_WIDTH),
            int(request.width),
            int(cv2.CAP_PROP_FRAME_HEIGHT),
            int(request.height),
            int(cv2.CAP_PROP_FPS),
            int(request.device_fps),
        ]
    )
    return params


def strict_mode_failures(
    actual: dict[str, Any], request: CameraRequest
) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    if actual["width"] != request.width:
        failures.append(f"width_{actual['width']}_expected_{request.width}")
    if actual["height"] != request.height:
        failures.append(f"height_{actual['height']}_expected_{request.height}")
    if request.fourcc != "AUTO" and actual["fourcc"].upper() != request.fourcc.upper():
        failures.append(f"fourcc_{actual['fourcc'] or 'EMPTY'}_expected_{request.fourcc}")
    fps = float(actual["fps_reported"])
    if not math.isfinite(fps) or fps <= 0:
        warnings.append(f"fps_reported_unavailable_{fps}")
    elif abs(fps - request.device_fps) > 0.51:
        failures.append(f"fps_{fps:.3f}_expected_{request.device_fps:.3f}")
    return failures, warnings


def recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise EOFError(f"Connection closed with {remaining} bytes remaining")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_message(connection: socket.socket, header: dict[str, Any], payload: bytes = b"") -> None:
    header_bytes = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if len(header_bytes) > MAX_HEADER_BYTES:
        raise ValueError("Header exceeds protocol maximum")
    if len(payload) > MAX_JPEG_BYTES:
        raise ValueError("Payload exceeds protocol maximum")
    prefix = struct.pack("!II", len(header_bytes), len(payload))
    connection.sendall(prefix + header_bytes + payload)


def receive_message(connection: socket.socket) -> tuple[dict[str, Any], bytes]:
    prefix = recv_exact(connection, 8)
    header_size, payload_size = struct.unpack("!II", prefix)
    if not 1 <= header_size <= MAX_HEADER_BYTES:
        raise ValueError(f"Invalid header size: {header_size}")
    if not 0 <= payload_size <= MAX_JPEG_BYTES:
        raise ValueError(f"Invalid payload size: {payload_size}")
    header = json.loads(recv_exact(connection, header_size).decode("utf-8"))
    payload = recv_exact(connection, payload_size) if payload_size else b""
    return header, payload


def validate_protocol_header(header: dict[str, Any], expected_nonce: str) -> None:
    if header.get("magic") != PROTOCOL_MAGIC:
        raise RuntimeError("Protocol magic mismatch")
    if header.get("version") != PROTOCOL_VERSION:
        raise RuntimeError("Protocol version mismatch")
    if not secrets.compare_digest(str(header.get("nonce", "")), expected_nonce):
        raise RuntimeError("Session nonce mismatch")


def connect_with_retry(host: str, port: int, timeout_seconds: float) -> socket.socket:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        connection.settimeout(5.0)
        try:
            connection.connect((host, port))
            return connection
        except OSError as exc:
            last_error = exc
            connection.close()
            time.sleep(0.1)
    raise RuntimeError(f"Could not connect to WSL receiver at {host}:{port}: {last_error}")


def worker_main(args: argparse.Namespace) -> int:
    if os.name != "nt":
        raise RuntimeError("Worker mode must run under native Windows Python")

    import cv2
    import numpy as np

    request = CameraRequest(
        role=args.role,
        vid=parse_hex_id(args.vid),
        pid=parse_hex_id(args.pid),
        width=args.width,
        height=args.height,
        device_fps=args.device_fps,
        fourcc=args.fourcc,
    )
    result: dict[str, Any] = {
        "status": "FAILED",
        "role": request.role,
        "request": asdict(request),
        "started_at_utc": utc_now(),
        "camera_reads_only": True,
        "robot_accessed": False,
        "serial_port_opened": False,
        "command_sent": False,
    }
    cap = None
    connection = None
    started_ns = time.perf_counter_ns()
    try:
        api = backend_code(args.backend)
        camera, inventory = find_exact_camera(request, api)
        result["inventory"] = inventory
        result["selected_camera"] = camera_info_payload(camera)
        open_params = constructor_open_parameters(request, cv2)
        result["open_parameters_raw"] = open_params

        cap = cv2.VideoCapture()
        open_return = bool(cap.open(int(camera.index), int(camera.backend), open_params))
        result["open_return"] = open_return
        if not cap.isOpened():
            raise RuntimeError("VideoCapture did not open with constructor parameters")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        initial_actual = capture_properties(cap, cv2)
        failures, warnings = strict_mode_failures(initial_actual, request)
        result["actual_immediately_after_open"] = initial_actual
        result["initial_mode_failures"] = failures
        result["initial_mode_warnings"] = warnings
        if failures:
            raise RuntimeError("Strict camera mode failed: " + ", ".join(failures))

        warmup_successes = 0
        warmup_attempts = 0
        last_frame = None
        while warmup_successes < args.successful_warmup_frames:
            warmup_attempts += 1
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                warmup_successes += 1
                last_frame = frame
            if warmup_attempts >= args.max_warmup_attempts:
                raise RuntimeError(
                    f"Warmup failed: successes={warmup_successes}, attempts={warmup_attempts}"
                )

        ready_actual = capture_properties(cap, cv2)
        failures, warnings = strict_mode_failures(ready_actual, request)
        result["actual_after_warmup"] = ready_actual
        result["ready_mode_failures"] = failures
        result["ready_mode_warnings"] = warnings
        if failures:
            raise RuntimeError("Strict camera mode after warmup failed: " + ", ".join(failures))

        connection = connect_with_retry(
            args.target_host, args.target_port, args.connect_timeout_seconds
        )
        hello = {
            "magic": PROTOCOL_MAGIC,
            "version": PROTOCOL_VERSION,
            "nonce": args.nonce,
            "type": "hello",
            "role": request.role,
            "request": asdict(request),
            "selected_camera": camera_info_payload(camera),
            "actual_capture": ready_actual,
            "mode_warnings": warnings,
            "windows_pid": os.getpid(),
            "windows_python": sys.executable,
        }
        send_message(connection, hello)
        start_signal = recv_exact(connection, 1)
        if start_signal != STREAM_START_SIGNAL:
            raise RuntimeError("Invalid stream start signal")

        interval_ns = int(round(1_000_000_000 / args.send_fps))
        stream_started_ns = time.perf_counter_ns()
        deadline_ns = stream_started_ns + int(args.duration_seconds * 1_000_000_000)
        next_send_ns = stream_started_ns
        sequence = 0
        read_failures = 0
        encode_failures = 0
        sent_bytes = 0

        while time.perf_counter_ns() < deadline_ns:
            ok, frame = cap.read()
            capture_perf_ns = time.perf_counter_ns()
            capture_wall_ns = time.time_ns()
            if not ok or frame is None or frame.size == 0:
                read_failures += 1
                continue
            last_frame = frame
            if capture_perf_ns < next_send_ns:
                continue

            encode_ok, encoded = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]
            )
            if not encode_ok:
                encode_failures += 1
                continue
            jpeg = np.ascontiguousarray(encoded).tobytes()
            send_perf_ns = time.perf_counter_ns()
            send_wall_ns = time.time_ns()
            header = {
                "magic": PROTOCOL_MAGIC,
                "version": PROTOCOL_VERSION,
                "nonce": args.nonce,
                "type": "frame",
                "role": request.role,
                "sequence": sequence,
                "capture_perf_ns": capture_perf_ns,
                "capture_wall_ns": capture_wall_ns,
                "send_perf_ns": send_perf_ns,
                "send_wall_ns": send_wall_ns,
                "source_width": int(frame.shape[1]),
                "source_height": int(frame.shape[0]),
                "channels": int(frame.shape[2]),
                "jpeg_size": len(jpeg),
                "jpeg_crc32": zlib.crc32(jpeg) & 0xFFFFFFFF,
            }
            send_message(connection, header, jpeg)
            sent_bytes += len(jpeg)
            sequence += 1
            while next_send_ns <= capture_perf_ns:
                next_send_ns += interval_ns

        send_message(
            connection,
            {
                "magic": PROTOCOL_MAGIC,
                "version": PROTOCOL_VERSION,
                "nonce": args.nonce,
                "type": "end",
                "role": request.role,
                "frames_sent": sequence,
            },
        )
        end_ack = recv_exact(connection, 1)
        if end_ack != STREAM_END_ACK:
            raise RuntimeError("Invalid stream end acknowledgement")
        result.update(
            {
                "status": "PASS",
                "end_ack_received": True,
                "frames_sent": sequence,
                "read_failures": read_failures,
                "encode_failures": encode_failures,
                "jpeg_bytes_sent": sent_bytes,
                "stream_elapsed_seconds": (
                    time.perf_counter_ns() - stream_started_ns
                )
                / 1_000_000_000.0,
                "last_frame_shape": None if last_frame is None else list(last_frame.shape),
            }
        )
        return_code = 0
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        print(f"WORKER_FATAL {request.role}: {type(exc).__name__}: {exc}", flush=True)
        return_code = 2
    finally:
        if connection is not None:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()
        if cap is not None:
            cap.release()
        result["worker_elapsed_seconds"] = (
            time.perf_counter_ns() - started_ns
        ) / 1_000_000_000.0
        result["finished_at_utc"] = utc_now()
        if args.result_json:
            write_json(Path(args.result_json).resolve(), result)
    return return_code


def detect_wsl_ipv4() -> str:
    output = subprocess.check_output(["hostname", "-I"], text=True).strip()
    for token in output.split():
        try:
            address = ipaddress.ip_address(token)
        except ValueError:
            continue
        if isinstance(address, ipaddress.IPv4Address) and not address.is_loopback:
            return str(address)
    raise RuntimeError(f"Could not detect a non-loopback WSL IPv4 address from: {output!r}")


def to_windows_path(path: Path) -> str:
    return subprocess.check_output(
        ["wslpath", "-w", str(path.resolve())], text=True
    ).strip()


def launch_worker(
    request: CameraRequest,
    args: argparse.Namespace,
    script_windows: str,
    target_host: str,
    port: int,
    nonce: str,
    output_dir: Path,
) -> tuple[subprocess.Popen[str], Path, float]:
    result_path = output_dir / f"{request.role}_windows_worker_result.json"
    command = [
        args.windows_python,
        script_windows,
        "--worker",
        "--role",
        request.role,
        "--vid",
        f"{request.vid:04X}",
        "--pid",
        f"{request.pid:04X}",
        "--backend",
        args.backend,
        "--width",
        str(request.width),
        "--height",
        str(request.height),
        "--device-fps",
        str(request.device_fps),
        "--fourcc",
        request.fourcc,
        "--send-fps",
        str(args.send_fps),
        "--duration-seconds",
        str(args.duration_seconds),
        "--successful-warmup-frames",
        str(args.successful_warmup_frames),
        "--max-warmup-attempts",
        str(args.max_warmup_attempts),
        "--jpeg-quality",
        str(args.jpeg_quality),
        "--target-host",
        target_host,
        "--target-port",
        str(port),
        "--connect-timeout-seconds",
        str(args.connect_timeout_seconds),
        "--nonce",
        nonce,
        "--result-json",
        to_windows_path(result_path),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return process, result_path, time.monotonic()


def accept_expected_role(
    listener: socket.socket,
    expected_role: str,
    nonce: str,
    timeout_seconds: float,
) -> tuple[socket.socket, dict[str, Any], tuple[str, int]]:
    listener.settimeout(timeout_seconds)
    connection, address = listener.accept()
    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    connection.settimeout(5.0)
    hello, payload = receive_message(connection)
    validate_protocol_header(hello, nonce)
    if payload:
        raise RuntimeError("Hello message unexpectedly contained a payload")
    if hello.get("type") != "hello":
        raise RuntimeError(f"Expected hello message, got {hello.get('type')!r}")
    if hello.get("role") != expected_role:
        raise RuntimeError(
            f"Expected {expected_role!r} connection, got {hello.get('role')!r}"
        )
    return connection, hello, address


def receive_stream(
    role: str,
    connection: socket.socket,
    nonce: str,
    state: dict[str, Any],
) -> None:
    import cv2
    import numpy as np

    rows: list[dict[str, Any]] = []
    decode_failures = 0
    crc_failures = 0
    protocol_failures = 0
    previous_sequence: int | None = None
    sequence_gap_count = 0
    previous_decoded_crc: int | None = None
    duplicate_count = 0
    first_frame = None
    middle_frame = None
    last_frame = None
    try:
        while True:
            header, jpeg = receive_message(connection)
            receive_perf_ns = time.perf_counter_ns()
            receive_wall_ns = time.time_ns()
            validate_protocol_header(header, nonce)
            if header.get("role") != role:
                raise RuntimeError(f"Role changed within stream: {header.get('role')!r}")
            message_type = header.get("type")
            if message_type == "end":
                if jpeg:
                    protocol_failures += 1
                    raise RuntimeError("End message unexpectedly contained a payload")
                frames_sent = int(header["frames_sent"])
                if frames_sent < 0:
                    protocol_failures += 1
                    raise RuntimeError(f"Invalid end frames_sent value: {frames_sent}")
                if frames_sent != len(rows):
                    protocol_failures += 1
                    raise RuntimeError(
                        f"End frame-count mismatch: worker={frames_sent}, receiver={len(rows)}"
                    )
                state["end_message"] = header
                connection.sendall(STREAM_END_ACK)
                state["end_ack_sent"] = True
                break
            if message_type != "frame":
                protocol_failures += 1
                raise RuntimeError(f"Unexpected message type: {message_type!r}")
            if not jpeg:
                protocol_failures += 1
                raise RuntimeError("Frame message had an empty JPEG payload")

            expected_crc = int(header["jpeg_crc32"])
            actual_crc = zlib.crc32(jpeg) & 0xFFFFFFFF
            if actual_crc != expected_crc:
                crc_failures += 1
                continue
            encoded = np.frombuffer(jpeg, dtype=np.uint8)
            frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if frame is None or frame.size == 0:
                decode_failures += 1
                continue

            sequence = int(header["sequence"])
            if previous_sequence is not None and sequence != previous_sequence + 1:
                sequence_gap_count += max(sequence - previous_sequence - 1, 1)
            previous_sequence = sequence
            decoded_crc = zlib.crc32(memoryview(np.ascontiguousarray(frame))) & 0xFFFFFFFF
            if previous_decoded_crc is not None and decoded_crc == previous_decoded_crc:
                duplicate_count += 1
            previous_decoded_crc = decoded_crc

            row = {
                "sequence": sequence,
                "capture_perf_ns": int(header["capture_perf_ns"]),
                "capture_wall_ns": int(header["capture_wall_ns"]),
                "send_perf_ns": int(header["send_perf_ns"]),
                "send_wall_ns": int(header["send_wall_ns"]),
                "receive_perf_ns": receive_perf_ns,
                "receive_wall_ns": receive_wall_ns,
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
                "channels": int(frame.shape[2]),
                "jpeg_size": len(jpeg),
                "jpeg_crc32": f"{actual_crc:08x}",
                "decoded_crc32": f"{decoded_crc:08x}",
                "capture_to_send_ms": (
                    int(header["send_perf_ns"]) - int(header["capture_perf_ns"])
                )
                / 1_000_000.0,
                "wall_end_to_end_ms": (
                    receive_wall_ns - int(header["capture_wall_ns"])
                )
                / 1_000_000.0,
            }
            rows.append(row)
            if first_frame is None:
                first_frame = frame.copy()
            if middle_frame is None and len(rows) >= 2:
                middle_frame = frame.copy()
            last_frame = frame.copy()
        state.update(
            {
                "status": "PASS",
                "rows": rows,
                "decode_failures": decode_failures,
                "crc_failures": crc_failures,
                "protocol_failures": protocol_failures,
                "sequence_gap_count": sequence_gap_count,
                "consecutive_exact_duplicate_frames": duplicate_count,
                "first_frame": first_frame,
                "middle_frame": middle_frame,
                "last_frame": last_frame,
            }
        )
    except Exception as exc:
        counters = {
            "rows": rows,
            "decode_failures": decode_failures,
            "crc_failures": crc_failures,
            "protocol_failures": protocol_failures,
            "sequence_gap_count": sequence_gap_count,
            "consecutive_exact_duplicate_frames": duplicate_count,
        }
        if state.get("status") == "TIMEOUT":
            # The parent owns the terminal timeout classification. Closing a
            # timed-out socket can wake this thread with EBADF; preserve the
            # causal status instead of replacing it with a cleanup artifact.
            state.update(counters)
            state["cleanup_error_type"] = type(exc).__name__
            state["cleanup_error"] = str(exc)
            state["cleanup_traceback"] = traceback.format_exc()
        else:
            state.update(
                {
                    "status": "FAILED",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    **counters,
                }
            )
    finally:
        try:
            connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        connection.close()


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "sequence",
        "capture_perf_ns",
        "capture_wall_ns",
        "send_perf_ns",
        "send_wall_ns",
        "receive_perf_ns",
        "receive_wall_ns",
        "width",
        "height",
        "channels",
        "jpeg_size",
        "jpeg_crc32",
        "decoded_crc32",
        "capture_to_send_ms",
        "wall_end_to_end_ms",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stream_metrics(rows: list[dict[str, Any]], duration_seconds: float) -> dict[str, Any]:
    receive_times = [int(row["receive_perf_ns"]) for row in rows]
    gaps_ms = [
        (right - left) / 1_000_000.0
        for left, right in zip(receive_times, receive_times[1:])
    ]
    span = (
        (receive_times[-1] - receive_times[0]) / 1_000_000_000.0
        if len(receive_times) >= 2
        else 0.0
    )
    fps = (len(rows) - 1) / span if span > 0 else 0.0
    dimensions = sorted(
        {(int(row["width"]), int(row["height"]), int(row["channels"])) for row in rows}
    )
    jpeg_sizes = [int(row["jpeg_size"]) for row in rows]
    capture_to_send = [float(row["capture_to_send_ms"]) for row in rows]
    wall_end_to_end = [float(row["wall_end_to_end_ms"]) for row in rows]
    return {
        "successful_frames": len(rows),
        "requested_duration_seconds": duration_seconds,
        "receive_span_seconds": span,
        "receive_span_ratio": span / duration_seconds if duration_seconds > 0 else 0.0,
        "achieved_receive_fps": fps,
        "gap_ms_p50": percentile(gaps_ms, 0.50),
        "gap_ms_p95": percentile(gaps_ms, 0.95),
        "gap_ms_p99": percentile(gaps_ms, 0.99),
        "gap_ms_max": max(gaps_ms) if gaps_ms else None,
        "unique_dimensions": [list(item) for item in dimensions],
        "jpeg_size_bytes_p50": percentile([float(v) for v in jpeg_sizes], 0.50),
        "jpeg_size_bytes_p95": percentile([float(v) for v in jpeg_sizes], 0.95),
        "capture_to_send_ms_p95": percentile(capture_to_send, 0.95),
        "wall_end_to_end_ms_p50_diagnostic": percentile(wall_end_to_end, 0.50),
        "wall_end_to_end_ms_p95_diagnostic": percentile(wall_end_to_end, 0.95),
        "receive_timestamps_monotonic": all(
            right > left for left, right in zip(receive_times, receive_times[1:])
        ),
    }


def nearest_skew_metrics(
    reference: list[int], candidates: list[int], label: str
) -> dict[str, Any] | None:
    if not reference or not candidates:
        return None
    overlap_start = max(reference[0], candidates[0])
    overlap_end = min(reference[-1], candidates[-1])
    selected = [value for value in reference if overlap_start <= value <= overlap_end]
    skews_ms: list[float] = []
    for value in selected:
        position = bisect.bisect_left(candidates, value)
        nearby: list[int] = []
        if position < len(candidates):
            nearby.append(candidates[position])
        if position > 0:
            nearby.append(candidates[position - 1])
        if nearby:
            nearest = min(nearby, key=lambda item: abs(item - value))
            skews_ms.append(abs(nearest - value) / 1_000_000.0)
    if not skews_ms:
        return None
    return {
        "basis": label,
        "pairs": len(skews_ms),
        "overlap_seconds": max((overlap_end - overlap_start) / 1_000_000_000.0, 0.0),
        "skew_ms_p50": percentile(skews_ms, 0.50),
        "skew_ms_p95": percentile(skews_ms, 0.95),
        "skew_ms_p99": percentile(skews_ms, 0.99),
        "skew_ms_max": max(skews_ms),
    }


def collect_process(
    process: subprocess.Popen[str], timeout_seconds: float
) -> tuple[int | None, str, bool]:
    try:
        stdout, _ = process.communicate(timeout=timeout_seconds)
        return process.returncode, stdout or "", False
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, _ = process.communicate(timeout=3.0)
        return process.returncode, stdout or "", True


def parent_main(args: argparse.Namespace) -> int:
    if os.name == "nt":
        raise RuntimeError("Parent mode must run under WSL/Linux Python")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory must be absent or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    windows_python = Path(args.windows_python)
    if not windows_python.is_file():
        raise FileNotFoundError(f"Windows Python not found: {windows_python}")

    target_host = args.windows_target_host or detect_wsl_ipv4()
    script_windows = to_windows_path(Path(__file__))
    nonce = secrets.token_hex(32)
    vid = parse_hex_id(args.vid)
    requests = {
        "front": CameraRequest(
            "front", vid, parse_hex_id(args.front_pid), args.width, args.height,
            args.front_device_fps, args.front_fourcc,
        ),
        "wrist": CameraRequest(
            "wrist", vid, parse_hex_id(args.wrist_pid), args.width, args.height,
            args.wrist_device_fps, args.wrist_fourcc,
        ),
    }

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((args.listen_host, args.port))
    listener.listen(2)
    actual_port = int(listener.getsockname()[1])

    print("===== WINDOWS -> WSL CAMERA BRIDGE CONTRACT =====", flush=True)
    print("camera transport only; no model, robot, serial port, or command API", flush=True)
    print(
        f"listener={args.listen_host}:{actual_port} windows_target={target_host}:{actual_port}",
        flush=True,
    )
    print(
        f"duration={args.duration_seconds:.1f}s send_fps={args.send_fps:.2f} "
        f"jpeg_quality={args.jpeg_quality}",
        flush=True,
    )
    print(
        f"front={vid:04X}:{requests['front'].pid:04X}/"
        f"{args.front_fourcc}@{args.front_device_fps:.2f}; "
        f"wrist={vid:04X}:{requests['wrist'].pid:04X}/"
        f"{args.wrist_fourcc}@{args.wrist_device_fps:.2f}",
        flush=True,
    )
    print("startup_order=front-ready-then-wrist-ready", flush=True)

    processes: dict[str, tuple[subprocess.Popen[str], Path, float]] = {}
    connections: dict[str, socket.socket] = {}
    hellos: dict[str, dict[str, Any]] = {}
    addresses: dict[str, tuple[str, int]] = {}
    receiver_states: dict[str, dict[str, Any]] = {"front": {}, "wrist": {}}
    failure: str | None = None

    try:
        for role in ("front", "wrist"):
            processes[role] = launch_worker(
                requests[role], args, script_windows, target_host, actual_port, nonce, output_dir
            )
            try:
                connection, hello, address = accept_expected_role(
                    listener, role, nonce, args.ready_timeout_seconds
                )
            except Exception as exc:
                failure = f"{role} did not establish a valid bridge connection: {exc}"
                break
            connections[role] = connection
            hellos[role] = hello
            addresses[role] = address
            actual = hello["actual_capture"]
            print(
                f"{role}: READY from {address[0]}:{address[1]} "
                f"mode={actual['width']}x{actual['height']}/{actual['fourcc'] or 'UNKNOWN'} "
                f"reported_fps={actual['fps_reported']:.3f}",
                flush=True,
            )

        if failure is None:
            threads: dict[str, threading.Thread] = {}
            for role in ("front", "wrist"):
                thread = threading.Thread(
                    target=receive_stream,
                    args=(role, connections[role], nonce, receiver_states[role]),
                    name=f"receive-{role}",
                    daemon=True,
                )
                threads[role] = thread
                thread.start()
            for role in ("front", "wrist"):
                connections[role].sendall(STREAM_START_SIGNAL)

            deadline = time.monotonic() + args.duration_seconds + args.worker_grace_seconds
            for role in ("front", "wrist"):
                remaining = max(deadline - time.monotonic(), 0.01)
                threads[role].join(timeout=remaining)
                if threads[role].is_alive():
                    receiver_states[role]["status"] = "TIMEOUT"
                    receiver_states[role]["error"] = "Receiver thread exceeded hard deadline"
                    receiver_states[role]["parent_timeout_at_utc"] = utc_now()
                    failure = failure or receiver_states[role]["error"]
    finally:
        listener.close()
        for connection in connections.values():
            try:
                connection.close()
            except OSError:
                pass

    process_reports: dict[str, Any] = {}
    for role, (process, result_path, started) in processes.items():
        code, stdout, timed_out = collect_process(process, args.worker_grace_seconds)
        if result_path.is_file():
            worker_result = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            worker_result = {
                "status": "FAILED",
                "error": "Windows worker result JSON was not created",
            }
        worker_result["exit_code"] = code
        worker_result["timed_out_during_collection"] = timed_out
        worker_result["stdout"] = stdout
        worker_result["parent_observed_seconds"] = time.monotonic() - started
        process_reports[role] = worker_result

    camera_reports: dict[str, Any] = {}
    for role in ("wrist", "front"):
        state = receiver_states[role]
        rows = state.get("rows", [])
        metrics = stream_metrics(rows, args.duration_seconds) if rows else None
        csv_path = output_dir / f"{role}_bridge_frames.csv"
        write_rows_csv(csv_path, rows)
        snapshots: dict[str, str] = {}
        if state.get("status") == "PASS":
            import cv2

            for name in ("first", "middle", "last"):
                frame = state.get(f"{name}_frame")
                if frame is None:
                    continue
                path = output_dir / f"{role}_{name}.png"
                if not cv2.imwrite(str(path), frame):
                    raise RuntimeError(f"Failed to write snapshot: {path}")
                snapshots[name] = str(path)
        camera_reports[role] = {
            "receiver_status": state.get("status", "NOT_STARTED"),
            "receiver_error": state.get("error"),
            "cleanup_error": state.get("cleanup_error"),
            "end_message": state.get("end_message"),
            "end_ack_sent": state.get("end_ack_sent", False),
            "decode_failures": state.get("decode_failures", 0),
            "crc_failures": state.get("crc_failures", 0),
            "protocol_failures": state.get("protocol_failures", 0),
            "sequence_gap_count": state.get("sequence_gap_count", 0),
            "consecutive_exact_duplicate_frames": state.get(
                "consecutive_exact_duplicate_frames", 0
            ),
            "metrics": metrics,
            "frame_csv": str(csv_path),
            "snapshots": snapshots,
            "hello": hellos.get(role),
            "peer_address": list(addresses[role]) if role in addresses else None,
            "windows_worker": process_reports.get(role),
        }

    front_rows = receiver_states["front"].get("rows", [])
    wrist_rows = receiver_states["wrist"].get("rows", [])
    capture_skew = nearest_skew_metrics(
        [int(row["capture_perf_ns"]) for row in wrist_rows],
        [int(row["capture_perf_ns"]) for row in front_rows],
        "Windows producer capture perf_counter_ns",
    )
    receive_skew = nearest_skew_metrics(
        [int(row["receive_perf_ns"]) for row in wrist_rows],
        [int(row["receive_perf_ns"]) for row in front_rows],
        "WSL receiver perf_counter_ns",
    )

    reasons: list[str] = []
    if failure:
        reasons.append(failure)
    for role in ("wrist", "front"):
        report = camera_reports[role]
        worker = report.get("windows_worker") or {}
        if worker.get("status") != "PASS":
            reasons.append(f"{role}_windows_worker_{worker.get('status', 'MISSING').lower()}")
        if worker.get("timed_out_during_collection"):
            reasons.append(f"{role}_windows_worker_collection_timeout")
        if worker.get("status") == "PASS" and worker.get("end_ack_received") is not True:
            reasons.append(f"{role}_windows_worker_missing_end_ack")
        if report["receiver_status"] != "PASS":
            reasons.append(f"{role}_receiver_{report['receiver_status'].lower()}")
            continue
        metrics = report["metrics"]
        if report["end_ack_sent"] is not True:
            reasons.append(f"{role}_receiver_missing_end_ack")
        end_message = report.get("end_message") or {}
        if int(end_message.get("frames_sent", -1)) != metrics["successful_frames"]:
            reasons.append(f"{role}_end_frame_count_mismatch")
        if int(worker.get("frames_sent", -1)) != metrics["successful_frames"]:
            reasons.append(f"{role}_worker_frame_count_mismatch")
        if metrics["achieved_receive_fps"] < args.min_receive_fps:
            reasons.append(f"{role}_receive_fps_below_{args.min_receive_fps}")
        if metrics["receive_span_ratio"] < args.min_receive_span_ratio:
            reasons.append(f"{role}_receive_span_ratio_below_{args.min_receive_span_ratio}")
        if metrics["gap_ms_max"] is None or metrics["gap_ms_max"] > args.max_frame_gap_ms:
            reasons.append(f"{role}_frame_gap_above_{args.max_frame_gap_ms}ms")
        if metrics["unique_dimensions"] != [[args.width, args.height, 3]]:
            reasons.append(f"{role}_unexpected_dimensions")
        if not metrics["receive_timestamps_monotonic"]:
            reasons.append(f"{role}_receive_timestamps_not_monotonic")
        for field in ("decode_failures", "crc_failures", "protocol_failures", "sequence_gap_count"):
            if report[field] != 0:
                reasons.append(f"{role}_{field}_{report[field]}")
    if capture_skew is None:
        reasons.append("capture_skew_unavailable")
    elif capture_skew["skew_ms_p95"] > args.max_capture_skew_p95_ms:
        reasons.append(f"capture_skew_p95_above_{args.max_capture_skew_p95_ms}ms")

    passed = not reasons
    report = {
        "schema_version": 2,
        "created_at_utc": utc_now(),
        "host": {
            "os_name": os.name,
            "platform": platform.platform(),
            "python": sys.version,
            "executable": sys.executable,
        },
        "contract": {
            "camera_transport_only": True,
            "model_loaded": False,
            "robot_accessed": False,
            "serial_port_opened": False,
            "command_sent": False,
            "listen_host": args.listen_host,
            "listen_port": actual_port,
            "windows_target_host": target_host,
            "backend": args.backend,
            "duration_seconds": args.duration_seconds,
            "send_fps": args.send_fps,
            "jpeg_quality": args.jpeg_quality,
            "protocol_version": PROTOCOL_VERSION,
            "termination_handshake": "END_then_ACK_before_close",
            "front_request": asdict(requests["front"]),
            "wrist_request": asdict(requests["wrist"]),
        },
        "acceptance_thresholds": {
            "min_receive_fps": args.min_receive_fps,
            "min_receive_span_ratio": args.min_receive_span_ratio,
            "max_frame_gap_ms": args.max_frame_gap_ms,
            "max_capture_skew_p95_ms": args.max_capture_skew_p95_ms,
            "classification": "transport service criteria, not physical hardware safety limits",
        },
        "camera_reports": camera_reports,
        "capture_skew": capture_skew,
        "receive_skew": receive_skew,
        "wall_latency_note": (
            "Wall-clock end-to-end values are diagnostics only because Windows and WSL "
            "clock synchronization was not independently bounded in this audit."
        ),
        "acceptance_failures": reasons,
        "decision": (
            "CAMERA_BRIDGE_SMOKE_PASS_BUILD_PREPROCESS_ADAPTER_NEXT"
            if passed
            else "CAMERA_BRIDGE_SMOKE_REVIEW_REQUIRED"
        ),
    }
    report_path = output_dir / "windows_wsl_camera_bridge_report.json"
    write_json(report_path, report)

    print("\n===== BRIDGE RECEIVE METRICS =====", flush=True)
    for role in ("wrist", "front"):
        camera = camera_reports[role]
        metrics = camera["metrics"]
        if camera["receiver_status"] != "PASS" or metrics is None:
            print(
                f"{role}: {camera['receiver_status']} {camera.get('receiver_error') or ''}",
                flush=True,
            )
            continue
        print(
            f"{role}: frames={metrics['successful_frames']} "
            f"fps={metrics['achieved_receive_fps']:.3f} "
            f"span={metrics['receive_span_seconds']:.3f}s "
            f"gap_p95={metrics['gap_ms_p95']:.3f}ms "
            f"gap_max={metrics['gap_ms_max']:.3f}ms "
            f"decode={camera['decode_failures']} crc={camera['crc_failures']} "
            f"seq_gap={camera['sequence_gap_count']} end_ack={camera['end_ack_sent']}",
            flush=True,
        )

    print("\n===== CROSS-CAMERA BRIDGE SKEW =====", flush=True)
    if capture_skew is None:
        print("capture skew unavailable", flush=True)
    else:
        print(
            f"capture pairs={capture_skew['pairs']} "
            f"p50={capture_skew['skew_ms_p50']:.3f}ms "
            f"p95={capture_skew['skew_ms_p95']:.3f}ms "
            f"max={capture_skew['skew_ms_max']:.3f}ms",
            flush=True,
        )
    if receive_skew is None:
        print("receive skew unavailable", flush=True)
    else:
        print(
            f"receive pairs={receive_skew['pairs']} "
            f"p50={receive_skew['skew_ms_p50']:.3f}ms "
            f"p95={receive_skew['skew_ms_p95']:.3f}ms "
            f"max={receive_skew['skew_ms_max']:.3f}ms",
            flush=True,
        )

    print("\n===== DECISION =====", flush=True)
    print(f"decision='{report['decision']}'", flush=True)
    if reasons:
        print(f"acceptance_failures={reasons}", flush=True)
    print(f"report={report_path}", flush=True)
    print("NO MODEL WAS LOADED. NO ROBOT OBJECT WAS CREATED.", flush=True)
    print("NO SERIAL PORT WAS OPENED. NO COMMAND WAS SENT.", flush=True)
    return 0 if passed else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit a native-Windows to WSL dual-camera JPEG/TCP bridge."
    )
    parser.add_argument(
        "--windows-python",
        default="/mnt/c/Users/Administrator/venvs/act-v3-camera-bridge/Scripts/python.exe",
    )
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--windows-target-host")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--backend", choices=("dshow", "msmf"), default="dshow")
    parser.add_argument("--vid", default="32E6")
    parser.add_argument("--front-pid", default="9221")
    parser.add_argument("--wrist-pid", default="9005")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--front-device-fps", type=float, default=30.0)
    parser.add_argument("--wrist-device-fps", type=float, default=15.0)
    parser.add_argument("--front-fourcc", default="MJPG")
    parser.add_argument("--wrist-fourcc", default="AUTO")
    parser.add_argument("--send-fps", type=float, default=15.0)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--duration-seconds", type=float, default=15.0)
    parser.add_argument("--successful-warmup-frames", type=int, default=5)
    parser.add_argument("--max-warmup-attempts", type=int, default=30)
    parser.add_argument("--ready-timeout-seconds", type=float, default=12.0)
    parser.add_argument("--connect-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--worker-grace-seconds", type=float, default=12.0)
    parser.add_argument("--min-receive-fps", type=float, default=14.0)
    parser.add_argument("--min-receive-span-ratio", type=float, default=0.90)
    parser.add_argument("--max-frame-gap-ms", type=float, default=250.0)
    parser.add_argument("--max-capture-skew-p95-ms", type=float, default=100.0)
    parser.add_argument("--output-dir")

    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--role", choices=("front", "wrist"), help=argparse.SUPPRESS)
    parser.add_argument("--pid", help=argparse.SUPPRESS)
    parser.add_argument("--device-fps", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--fourcc", default="AUTO", help=argparse.SUPPRESS)
    parser.add_argument("--target-host", help=argparse.SUPPRESS)
    parser.add_argument("--target-port", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--nonce", help=argparse.SUPPRESS)
    parser.add_argument("--result-json", help=argparse.SUPPRESS)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "width": args.width,
        "height": args.height,
        "front_device_fps": args.front_device_fps,
        "wrist_device_fps": args.wrist_device_fps,
        "send_fps": args.send_fps,
        "duration_seconds": args.duration_seconds,
        "successful_warmup_frames": args.successful_warmup_frames,
        "max_warmup_attempts": args.max_warmup_attempts,
        "ready_timeout_seconds": args.ready_timeout_seconds,
        "connect_timeout_seconds": args.connect_timeout_seconds,
        "worker_grace_seconds": args.worker_grace_seconds,
        "min_receive_fps": args.min_receive_fps,
        "max_frame_gap_ms": args.max_frame_gap_ms,
        "max_capture_skew_p95_ms": args.max_capture_skew_p95_ms,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"Arguments must be positive: {invalid}")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be in [1, 100]")
    if not 0 < args.min_receive_span_ratio <= 1:
        raise ValueError("min_receive_span_ratio must be in (0, 1]")
    if not 0 <= args.port <= 65535:
        raise ValueError("port must be in [0, 65535]")
    if args.successful_warmup_frames > args.max_warmup_attempts:
        raise ValueError("successful warmup frames cannot exceed max warmup attempts")
    for name, value in (("front_fourcc", args.front_fourcc), ("wrist_fourcc", args.wrist_fourcc)):
        if value != "AUTO" and len(value) != 4:
            raise ValueError(f"{name} must be AUTO or exactly four characters")
    if args.worker:
        required = (args.role, args.pid, args.device_fps, args.target_host, args.target_port, args.nonce)
        if not all(required):
            raise ValueError("Worker invocation is missing internal arguments")
        if args.fourcc != "AUTO" and len(args.fourcc) != 4:
            raise ValueError("worker fourcc must be AUTO or exactly four characters")
    elif not args.output_dir:
        raise ValueError("Parent invocation requires --output-dir")


def main() -> int:
    args = build_parser().parse_args()
    try:
        validate_args(args)
        return worker_main(args) if args.worker else parent_main(args)
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
