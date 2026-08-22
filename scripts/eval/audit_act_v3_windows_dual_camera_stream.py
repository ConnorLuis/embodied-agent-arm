#!/usr/bin/env python3
"""Audit simultaneous native-Windows front/wrist camera streaming.

Each physical camera runs in its own child process.  The parent imposes one
global hard deadline and kills any blocked worker.  The audit records achieved
FPS, frame gaps, failed reads, exact consecutive duplicates, stable dimensions,
and nearest-frame cross-camera timestamp skew.

Camera reads only: no LeRobot, robot, teleoperator, serial, or command API.

Schema v3 adds an explicit capture-open strategy. ``constructor`` passes the
requested mode to OpenCV's three-argument ``VideoCapture.open`` call so the
backend can negotiate the mode during open. ``post-open`` retains the earlier
diagnostic behavior for comparison. Strict negotiation can require that the
reported width, height, FPS, and requested FOURCC match before the next camera
is allowed to start.

Schema v4 treats a non-positive or non-finite backend-reported FPS as
unavailable rather than mismatched. DirectShow can return ``-1`` before frames
are read. Requested throughput is still checked from monotonic timestamps over
the full stream; dimensions and an explicitly requested FOURCC remain strict.

Schema v5 separates sensor-mode requests from application throughput needs.
For this ACT runtime, acceptance defaults to an absolute 15 FPS floor for both
camera roles; the request-relative FPS ratio remains available as a diagnostic
or an explicitly selected legacy acceptance mode.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
import zlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_VID = 0x32E6
DEFAULT_CAMERAS = {
    "wrist": {"pid": 0x9005, "fps": 15.0},
    "front": {"pid": 0x9221, "fps": 30.0},
}


@dataclass(frozen=True)
class CameraRequest:
    role: str
    vid: int
    pid: int
    width: int
    height: int
    fps: float


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_hex_id(value: str) -> int:
    cleaned = value.strip().lower()
    if cleaned.startswith("0x"):
        cleaned = cleaned[2:]
    return int(cleaned, 16)


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


def backend_code(name: str) -> int:
    import cv2

    return {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}[name]


def decode_fourcc(value: float) -> str:
    number = int(value)
    return "".join(chr((number >> (8 * shift)) & 0xFF) for shift in range(4)).rstrip("\x00")


def capture_properties(cap: Any, cv2: Any) -> dict[str, Any]:
    return {
        "width": int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
        "height": int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        "fps_reported": float(cap.get(cv2.CAP_PROP_FPS)),
        "fourcc": decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC)),
        "backend_reported": cap.getBackendName(),
    }


def constructor_open_parameters(
    request: CameraRequest, fourcc: str, cv2: Any
) -> tuple[list[int], list[dict[str, Any]]]:
    if not float(request.fps).is_integer():
        raise ValueError(
            "constructor open strategy requires an integer FPS because OpenCV's "
            f"open-parameter vector is integer-valued; got {request.fps}"
        )

    raw: list[int] = []
    readable: list[dict[str, Any]] = []

    def add(name: str, property_id: int, value: int, display: Any | None = None) -> None:
        raw.extend((int(property_id), int(value)))
        readable.append(
            {
                "name": name,
                "property_id": int(property_id),
                "value": int(value),
                "display": value if display is None else display,
            }
        )

    if fourcc != "AUTO":
        encoded = int(cv2.VideoWriter_fourcc(*fourcc))
        add("CAP_PROP_FOURCC", cv2.CAP_PROP_FOURCC, encoded, fourcc)
    add("CAP_PROP_FRAME_WIDTH", cv2.CAP_PROP_FRAME_WIDTH, request.width)
    add("CAP_PROP_FRAME_HEIGHT", cv2.CAP_PROP_FRAME_HEIGHT, request.height)
    add("CAP_PROP_FPS", cv2.CAP_PROP_FPS, int(request.fps))
    return raw, readable


def negotiation_assessment(
    actual: dict[str, Any], request: CameraRequest, requested_fourcc: str
) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    if actual["width"] != request.width:
        failures.append(f"width_{actual['width']}_expected_{request.width}")
    if actual["height"] != request.height:
        failures.append(f"height_{actual['height']}_expected_{request.height}")
    reported_fps = float(actual["fps_reported"])
    if not math.isfinite(reported_fps) or reported_fps <= 0:
        warnings.append(f"fps_reported_unavailable_{reported_fps}")
    elif abs(reported_fps - request.fps) > 0.51:
        failures.append(
            f"fps_{reported_fps:.3f}_expected_{request.fps:.3f}"
        )
    if requested_fourcc != "AUTO" and actual["fourcc"].upper() != requested_fourcc.upper():
        failures.append(f"fourcc_{actual['fourcc'] or 'EMPTY'}_expected_{requested_fourcc}")
    return failures, warnings


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


def write_frame_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "sequence",
        "capture_monotonic_ns",
        "elapsed_seconds",
        "width",
        "height",
        "channels",
        "pixel_mean",
        "pixel_std",
        "crc32",
        "same_as_previous",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def worker_main(args: argparse.Namespace) -> int:
    import cv2
    import numpy as np

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = Path(args.result_json).resolve()
    request = CameraRequest(
        role=args.role,
        vid=parse_hex_id(args.vid),
        pid=parse_hex_id(args.pid),
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    result: dict[str, Any] = {
        "status": "FAILED",
        "role": request.role,
        "request": asdict(request),
        "backend_name": args.backend,
        "started_at_utc": utc_now(),
    }
    cap = None
    worker_started_ns = time.perf_counter_ns()

    try:
        api = backend_code(args.backend)
        camera, inventory = find_exact_camera(request, api)
        result["inventory"] = inventory
        result["selected_camera"] = camera_info_payload(camera)

        if args.open_strategy == "constructor":
            open_params, readable_open_params = constructor_open_parameters(
                request, args.fourcc, cv2
            )
            result["requested_open_parameters"] = readable_open_params
            result["requested_open_parameters_raw"] = open_params
            cap = cv2.VideoCapture()
            open_return = bool(
                cap.open(int(camera.index), int(camera.backend), open_params)
            )
        else:
            result["requested_open_parameters"] = []
            result["requested_open_parameters_raw"] = []
            cap = cv2.VideoCapture(int(camera.index), int(camera.backend))
            open_return = bool(cap.isOpened())

        result["open_strategy"] = args.open_strategy
        result["open_call_return"] = open_return
        result["opened"] = bool(cap.isOpened())
        if not cap.isOpened():
            raise RuntimeError(
                f"VideoCapture did not open with strategy={args.open_strategy}"
            )

        property_set_results: dict[str, Any] = {
            "buffer_size_1": bool(cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)),
        }
        if args.open_strategy == "post-open":
            if args.fourcc != "AUTO":
                property_set_results["fourcc"] = bool(
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
                )
            else:
                property_set_results["fourcc"] = "not_requested"
            property_set_results.update(
                {
                    "width": bool(cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(request.width))),
                    "height": bool(cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(request.height))),
                    "fps": bool(cap.set(cv2.CAP_PROP_FPS, float(request.fps))),
                }
            )
        else:
            property_set_results.update(
                {
                    "fourcc": "supplied_during_open" if args.fourcc != "AUTO" else "AUTO",
                    "width": "supplied_during_open",
                    "height": "supplied_during_open",
                    "fps": "supplied_during_open",
                }
            )
        result["property_set_results"] = property_set_results
        result["actual_capture_immediately_after_open"] = capture_properties(cap, cv2)

        initial_negotiation_failures, initial_negotiation_warnings = negotiation_assessment(
            result["actual_capture_immediately_after_open"], request, args.fourcc
        )
        result["initial_negotiation_failures"] = initial_negotiation_failures
        result["initial_negotiation_warnings"] = initial_negotiation_warnings
        if args.strict_negotiation and initial_negotiation_failures:
            raise RuntimeError(
                "Strict capture negotiation failed immediately after open: "
                + ", ".join(initial_negotiation_failures)
            )

        warmup_successes = 0
        warmup_attempts = 0
        while warmup_successes < args.successful_warmup_frames:
            warmup_attempts += 1
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                warmup_successes += 1
            if warmup_attempts >= args.max_warmup_attempts:
                raise RuntimeError(
                    f"Warmup failed: successes={warmup_successes}, attempts={warmup_attempts}"
                )

        ready_actual_capture = capture_properties(cap, cv2)
        ready_negotiation_failures, ready_negotiation_warnings = negotiation_assessment(
            ready_actual_capture, request, args.fourcc
        )
        result["ready_negotiation_failures"] = ready_negotiation_failures
        result["ready_negotiation_warnings"] = ready_negotiation_warnings
        if args.strict_negotiation and ready_negotiation_failures:
            raise RuntimeError(
                "Strict capture negotiation failed after warmup: "
                + ", ".join(ready_negotiation_failures)
            )

        ready_payload = {
            "status": "READY",
            "role": request.role,
            "ready_at_utc": utc_now(),
            "ready_monotonic_ns": time.perf_counter_ns(),
            "selected_camera": camera_info_payload(camera),
            "open_strategy": args.open_strategy,
            "strict_negotiation": bool(args.strict_negotiation),
            "actual_capture": ready_actual_capture,
            "negotiation_failures": ready_negotiation_failures,
            "negotiation_warnings": ready_negotiation_warnings,
        }
        write_json(Path(args.ready_json).resolve(), ready_payload)

        capture_started_ns = time.perf_counter_ns()
        capture_deadline_ns = capture_started_ns + int(args.duration_seconds * 1_000_000_000)
        rows: list[dict[str, Any]] = []
        failed_reads = 0
        previous_crc: int | None = None
        first_frame = None
        middle_frame = None
        last_frame = None

        while time.perf_counter_ns() < capture_deadline_ns:
            ok, frame = cap.read()
            timestamp_ns = time.perf_counter_ns()
            if not ok or frame is None or frame.size == 0:
                failed_reads += 1
                continue

            if frame.ndim != 3 or frame.shape[2] != 3:
                raise RuntimeError(f"Unexpected frame shape: {frame.shape}")
            crc = zlib.crc32(memoryview(np.ascontiguousarray(frame))) & 0xFFFFFFFF
            elapsed = (timestamp_ns - capture_started_ns) / 1_000_000_000.0
            row = {
                "sequence": len(rows),
                "capture_monotonic_ns": timestamp_ns,
                "elapsed_seconds": elapsed,
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
                "channels": int(frame.shape[2]),
                "pixel_mean": float(np.mean(frame)),
                "pixel_std": float(np.std(frame)),
                "crc32": f"{crc:08x}",
                "same_as_previous": int(previous_crc == crc) if previous_crc is not None else 0,
            }
            rows.append(row)
            previous_crc = crc
            if first_frame is None:
                first_frame = frame.copy()
            if middle_frame is None and elapsed >= args.duration_seconds / 2.0:
                middle_frame = frame.copy()
            last_frame = frame.copy()

        if len(rows) < 2 or first_frame is None or last_frame is None:
            raise RuntimeError(f"Too few successful frames: {len(rows)}")
        if middle_frame is None:
            middle_frame = last_frame.copy()

        timestamps_ns = [int(row["capture_monotonic_ns"]) for row in rows]
        gaps_ms = [
            (right - left) / 1_000_000.0
            for left, right in zip(timestamps_ns, timestamps_ns[1:])
        ]
        span_seconds = (timestamps_ns[-1] - timestamps_ns[0]) / 1_000_000_000.0
        achieved_fps = (len(rows) - 1) / span_seconds if span_seconds > 0 else 0.0
        dimensions = sorted({(int(row["width"]), int(row["height"]), int(row["channels"])) for row in rows})
        duplicate_count = sum(int(row["same_as_previous"]) for row in rows)
        total_reads = len(rows) + failed_reads

        frame_csv = output_dir / f"{request.role}_stream_frames.csv"
        write_frame_csv(frame_csv, rows)
        snapshot_paths: dict[str, str] = {}
        for name, snapshot in (("first", first_frame), ("middle", middle_frame), ("last", last_frame)):
            path = output_dir / f"{request.role}_{name}.png"
            if not cv2.imwrite(str(path), snapshot):
                raise RuntimeError(f"Failed to write snapshot: {path}")
            snapshot_paths[name] = str(path)

        result.update(
            {
                "status": "PASS",
                "actual_capture": capture_properties(cap, cv2),
                "stream_metrics": {
                    "duration_requested_seconds": args.duration_seconds,
                    "successful_frames": len(rows),
                    "failed_reads": failed_reads,
                    "total_reads": total_reads,
                    "read_failure_rate": failed_reads / total_reads if total_reads else 1.0,
                    "span_seconds": span_seconds,
                    "achieved_fps": achieved_fps,
                    "fps_ratio": achieved_fps / request.fps if request.fps > 0 else 0.0,
                    "gap_ms_p50": percentile(gaps_ms, 0.50),
                    "gap_ms_p95": percentile(gaps_ms, 0.95),
                    "gap_ms_p99": percentile(gaps_ms, 0.99),
                    "gap_ms_max": max(gaps_ms) if gaps_ms else None,
                    "consecutive_exact_duplicate_frames": duplicate_count,
                    "consecutive_exact_duplicate_rate": duplicate_count / max(len(rows) - 1, 1),
                    "unique_dimensions": [list(item) for item in dimensions],
                    "timestamp_monotonic": all(right > left for left, right in zip(timestamps_ns, timestamps_ns[1:])),
                    "first_timestamp_ns": timestamps_ns[0],
                    "last_timestamp_ns": timestamps_ns[-1],
                },
                "frame_csv": str(frame_csv),
                "snapshots": snapshot_paths,
            }
        )
        return_code = 0
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        return_code = 1
    finally:
        if cap is not None:
            cap.release()
        result["worker_elapsed_seconds"] = (time.perf_counter_ns() - worker_started_ns) / 1_000_000_000.0
        result["finished_at_utc"] = utc_now()
        write_json(result_path, result)

    return return_code


def worker_command(
    request: CameraRequest, args: argparse.Namespace, output_dir: Path
) -> tuple[list[str], Path, Path]:
    result_path = output_dir / f"{request.role}_stream_worker_result.json"
    ready_path = output_dir / f"{request.role}_stream_worker_ready.json"
    fourcc = args.wrist_fourcc if request.role == "wrist" else args.front_fourcc
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
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
        "--fps",
        str(request.fps),
        "--fourcc",
        fourcc,
        "--open-strategy",
        args.open_strategy,
        "--duration-seconds",
        str(args.duration_seconds),
        "--successful-warmup-frames",
        str(args.successful_warmup_frames),
        "--max-warmup-attempts",
        str(args.max_warmup_attempts),
        "--output-dir",
        str(output_dir),
        "--result-json",
        str(result_path),
        "--ready-json",
        str(ready_path),
    ]
    if args.strict_negotiation:
        command.append("--strict-negotiation")
    return command, result_path, ready_path


def kill_and_collect(process: subprocess.Popen[str]) -> str:
    process.kill()
    try:
        stdout, _ = process.communicate(timeout=3.0)
    except subprocess.TimeoutExpired:
        stdout = "Worker did not terminate within the post-kill grace period."
    return stdout or ""


def wait_for_worker_ready(
    process: subprocess.Popen[str], ready_path: Path, timeout_seconds: float
) -> str:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if ready_path.is_file():
            return "READY"
        if process.poll() is not None:
            return "EXITED"
        time.sleep(0.05)
    return "TIMEOUT"


def read_finished_worker_result(
    request: CameraRequest,
    process: subprocess.Popen[str],
    result_path: Path,
    started: float,
    stdout: str,
) -> dict[str, Any]:
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        result = {
            "status": "FAILED",
            "role": request.role,
            "request": asdict(request),
            "error": "Worker result JSON was not created",
        }
    result["worker_exit_code"] = process.returncode
    result["worker_stdout"] = stdout or ""
    result["parent_observed_elapsed_seconds"] = time.monotonic() - started
    return result


def launch_workers(
    requests: list[CameraRequest], args: argparse.Namespace, output_dir: Path
) -> dict[str, dict[str, Any]]:
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    running: dict[str, tuple[subprocess.Popen[str], Path, Path, float]] = {}

    request_by_role = {request.role: request for request in requests}
    if args.startup_order == "wrist-first":
        launch_order = [request_by_role["wrist"], request_by_role["front"]]
    elif args.startup_order == "front-first":
        launch_order = [request_by_role["front"], request_by_role["wrist"]]
    else:
        launch_order = list(requests)

    def start(request: CameraRequest) -> None:
        command, result_path, ready_path = worker_command(request, args, output_dir)
        started = time.monotonic()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=creation_flags,
        )
        running[request.role] = (process, result_path, ready_path, started)

    results: dict[str, dict[str, Any]] = {}
    if args.startup_order == "concurrent":
        for request in launch_order:
            start(request)
    else:
        first, second = launch_order
        start(first)
        first_process, first_result_path, first_ready_path, first_started = running[first.role]
        ready_state = wait_for_worker_ready(
            first_process, first_ready_path, args.startup_ready_timeout_seconds
        )
        if ready_state != "READY":
            if first_process.poll() is None:
                stdout = kill_and_collect(first_process)
                results[first.role] = {
                    "status": "TIMEOUT",
                    "role": first.role,
                    "request": asdict(first),
                    "error": "First camera did not reach READY before startup timeout",
                    "worker_stdout": stdout,
                    "parent_observed_elapsed_seconds": time.monotonic() - first_started,
                }
            else:
                stdout, _ = first_process.communicate()
                results[first.role] = read_finished_worker_result(
                    first, first_process, first_result_path, first_started, stdout
                )
            results[second.role] = {
                "status": "SKIPPED",
                "role": second.role,
                "request": asdict(second),
                "error": f"Not started because first camera readiness state was {ready_state}",
            }
            return results
        start(second)

    global_timeout = args.duration_seconds + args.worker_timeout_grace_seconds
    global_deadline = time.monotonic() + global_timeout
    for request in requests:
        if request.role in results:
            continue
        process, result_path, _ready_path, started = running[request.role]
        remaining = max(global_deadline - time.monotonic(), 0.01)
        try:
            stdout, _ = process.communicate(timeout=remaining)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            stdout = kill_and_collect(process)

        if timed_out:
            results[request.role] = {
                "status": "TIMEOUT",
                "role": request.role,
                "request": asdict(request),
                "global_timeout_seconds": global_timeout,
                "parent_observed_elapsed_seconds": time.monotonic() - started,
                "worker_stdout": stdout,
            }
            continue

        results[request.role] = read_finished_worker_result(
            request, process, result_path, started, stdout
        )
    return results


def cross_camera_skew_metrics(results: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if not all(results.get(role, {}).get("status") == "PASS" for role in ("wrist", "front")):
        return None
    wrist_csv = Path(results["wrist"]["frame_csv"])
    front_csv = Path(results["front"]["frame_csv"])

    def read_timestamps(path: Path) -> list[int]:
        with path.open("r", newline="", encoding="utf-8") as handle:
            return [int(row["capture_monotonic_ns"]) for row in csv.DictReader(handle)]

    wrist_times = read_timestamps(wrist_csv)
    front_times = read_timestamps(front_csv)
    if not wrist_times or not front_times:
        return None

    overlap_start = max(wrist_times[0], front_times[0])
    overlap_end = min(wrist_times[-1], front_times[-1])
    reference_times = [timestamp for timestamp in wrist_times if overlap_start <= timestamp <= overlap_end]
    skews_ms: list[float] = []
    for timestamp in reference_times:
        position = bisect.bisect_left(front_times, timestamp)
        candidates = []
        if position < len(front_times):
            candidates.append(front_times[position])
        if position > 0:
            candidates.append(front_times[position - 1])
        if candidates:
            nearest = min(candidates, key=lambda value: abs(value - timestamp))
            skews_ms.append(abs(nearest - timestamp) / 1_000_000.0)

    return {
        "reference": "each wrist frame paired with nearest front frame in common monotonic interval",
        "paired_frames": len(skews_ms),
        "overlap_seconds": max((overlap_end - overlap_start) / 1_000_000_000.0, 0.0),
        "skew_ms_p50": percentile(skews_ms, 0.50),
        "skew_ms_p95": percentile(skews_ms, 0.95),
        "skew_ms_p99": percentile(skews_ms, 0.99),
        "skew_ms_max": max(skews_ms) if skews_ms else None,
    }


def acceptance_reasons(
    results: dict[str, dict[str, Any]], cross_skew: dict[str, Any] | None, args: argparse.Namespace
) -> list[str]:
    reasons: list[str] = []
    for role in ("wrist", "front"):
        result = results.get(role, {})
        if result.get("status") != "PASS":
            reasons.append(f"{role}_worker_{result.get('status', 'MISSING').lower()}")
            continue
        metrics = result["stream_metrics"]
        if args.fps_acceptance == "ratio":
            if metrics["fps_ratio"] < args.min_fps_ratio:
                reasons.append(f"{role}_fps_ratio_below_{args.min_fps_ratio}")
        else:
            minimum_fps = (
                args.min_wrist_achieved_fps
                if role == "wrist"
                else args.min_front_achieved_fps
            )
            if metrics["achieved_fps"] < minimum_fps:
                reasons.append(f"{role}_achieved_fps_below_{minimum_fps}")
        if metrics["read_failure_rate"] > args.max_read_failure_rate:
            reasons.append(f"{role}_read_failure_rate_above_{args.max_read_failure_rate}")
        if metrics["gap_ms_max"] is None or metrics["gap_ms_max"] > args.max_frame_gap_ms:
            reasons.append(f"{role}_frame_gap_above_{args.max_frame_gap_ms}ms")
        if len(metrics["unique_dimensions"]) != 1:
            reasons.append(f"{role}_dimensions_changed")
        if not metrics["timestamp_monotonic"]:
            reasons.append(f"{role}_timestamp_not_monotonic")
    if cross_skew is None or cross_skew.get("skew_ms_p95") is None:
        reasons.append("cross_camera_skew_unavailable")
    elif cross_skew["skew_ms_p95"] > args.max_cross_camera_skew_p95_ms:
        reasons.append(f"cross_camera_skew_p95_above_{args.max_cross_camera_skew_p95_ms}ms")
    return reasons


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simultaneous native-Windows dual-camera stream audit with hard worker deadlines."
    )
    parser.add_argument("--backend", choices=("dshow", "msmf"), default="dshow")
    parser.add_argument("--vid", default="32E6")
    parser.add_argument("--front-pid", default="9221")
    parser.add_argument("--wrist-pid", default="9005")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--front-fps", type=float, default=30.0)
    parser.add_argument("--wrist-fps", type=float, default=15.0)
    parser.add_argument("--front-fourcc", default="AUTO")
    parser.add_argument("--wrist-fourcc", default="AUTO")
    parser.add_argument(
        "--open-strategy",
        choices=("constructor", "post-open"),
        default="constructor",
        help="Apply capture mode in VideoCapture.open params or with set() after open.",
    )
    parser.add_argument(
        "--strict-negotiation",
        action="store_true",
        help="Require reported width, height, FPS and requested FOURCC before READY.",
    )
    parser.add_argument(
        "--startup-order",
        choices=("wrist-first", "front-first", "concurrent"),
        default="wrist-first",
    )
    parser.add_argument("--startup-ready-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--duration-seconds", type=float, default=15.0)
    parser.add_argument("--successful-warmup-frames", type=int, default=5)
    parser.add_argument("--max-warmup-attempts", type=int, default=30)
    parser.add_argument("--worker-timeout-grace-seconds", type=float, default=10.0)
    parser.add_argument(
        "--fps-acceptance",
        choices=("absolute", "ratio"),
        default="absolute",
        help="Use role-specific achieved-FPS floors or the request-relative ratio.",
    )
    parser.add_argument("--min-front-achieved-fps", type=float, default=15.0)
    parser.add_argument("--min-wrist-achieved-fps", type=float, default=15.0)
    parser.add_argument("--min-fps-ratio", type=float, default=0.80)
    parser.add_argument("--max-read-failure-rate", type=float, default=0.01)
    parser.add_argument("--max-frame-gap-ms", type=float, default=250.0)
    parser.add_argument("--max-cross-camera-skew-p95-ms", type=float, default=100.0)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--role", choices=("front", "wrist"), help=argparse.SUPPRESS)
    parser.add_argument("--pid", help=argparse.SUPPRESS)
    parser.add_argument("--fps", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--fourcc", default="AUTO", help=argparse.SUPPRESS)
    parser.add_argument("--result-json", help=argparse.SUPPRESS)
    parser.add_argument("--ready-json", help=argparse.SUPPRESS)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if os.name != "nt":
        raise RuntimeError("This audit must run under native Windows Python, not WSL/Linux Python")
    positive = {
        "width": args.width,
        "height": args.height,
        "duration_seconds": args.duration_seconds,
        "successful_warmup_frames": args.successful_warmup_frames,
        "max_warmup_attempts": args.max_warmup_attempts,
        "worker_timeout_grace_seconds": args.worker_timeout_grace_seconds,
        "startup_ready_timeout_seconds": args.startup_ready_timeout_seconds,
        "max_frame_gap_ms": args.max_frame_gap_ms,
        "max_cross_camera_skew_p95_ms": args.max_cross_camera_skew_p95_ms,
        "min_front_achieved_fps": args.min_front_achieved_fps,
        "min_wrist_achieved_fps": args.min_wrist_achieved_fps,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"Arguments must be positive: {invalid}")
    if not 0 < args.min_fps_ratio <= 1:
        raise ValueError("min_fps_ratio must be in (0, 1]")
    if not 0 <= args.max_read_failure_rate <= 1:
        raise ValueError("max_read_failure_rate must be in [0, 1]")
    if args.successful_warmup_frames > args.max_warmup_attempts:
        raise ValueError("successful warmup frames cannot exceed max warmup attempts")
    for name, value in (("front_fourcc", args.front_fourcc), ("wrist_fourcc", args.wrist_fourcc)):
        if value != "AUTO" and len(value) != 4:
            raise ValueError(f"{name} must be AUTO or exactly four characters")
    if args.worker and args.fourcc != "AUTO" and len(args.fourcc) != 4:
        raise ValueError("worker fourcc must be AUTO or exactly four characters")
    if args.open_strategy == "constructor":
        non_integer_fps = [
            value
            for value in (args.front_fps, args.wrist_fps)
            if not float(value).is_integer()
        ]
        if non_integer_fps:
            raise ValueError(
                "constructor open strategy requires integer-valued front/wrist FPS; "
                f"got {non_integer_fps}"
            )
    if args.worker and not all((args.role, args.pid, args.fps, args.result_json, args.ready_json)):
        raise ValueError("worker invocation is missing internal arguments")


def parent_main(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory must be absent or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    vid = parse_hex_id(args.vid)
    requests = [
        CameraRequest("wrist", vid, parse_hex_id(args.wrist_pid), args.width, args.height, args.wrist_fps),
        CameraRequest("front", vid, parse_hex_id(args.front_pid), args.width, args.height, args.front_fps),
    ]

    print("===== WINDOWS DUAL-CAMERA STREAM AUDIT CONTRACT =====", flush=True)
    print("simultaneous camera reads only; no robot, serial port, or command API", flush=True)
    print(
        f"backend={args.backend} duration={args.duration_seconds:.1f}s "
        f"global_hard_timeout={args.duration_seconds + args.worker_timeout_grace_seconds:.1f}s",
        flush=True,
    )
    print(
        f"startup_order={args.startup_order} "
        f"ready_timeout={args.startup_ready_timeout_seconds:.1f}s "
        f"open_strategy={args.open_strategy} strict_negotiation={args.strict_negotiation}",
        flush=True,
    )
    print(
        f"wrist={vid:04X}:{parse_hex_id(args.wrist_pid):04X}@{args.wrist_fps:.2f}fps/"
        f"{args.wrist_fourcc}; "
        f"front={vid:04X}:{parse_hex_id(args.front_pid):04X}@{args.front_fps:.2f}fps/"
        f"{args.front_fourcc}",
        flush=True,
    )
    if args.fps_acceptance == "absolute":
        print(
            "fps_acceptance=absolute "
            f"wrist>={args.min_wrist_achieved_fps:.2f} "
            f"front>={args.min_front_achieved_fps:.2f}",
            flush=True,
        )
    else:
        print(f"fps_acceptance=ratio minimum={args.min_fps_ratio:.3f}", flush=True)
    print("Starting camera workers under the requested startup contract...", flush=True)

    results = launch_workers(requests, args, output_dir)
    cross_skew = cross_camera_skew_metrics(results)
    reasons = acceptance_reasons(results, cross_skew, args)
    passed = not reasons
    run_kind = "full" if args.duration_seconds >= 60.0 else "smoke"

    print("\n===== PER-CAMERA STREAM METRICS =====", flush=True)
    for role in ("wrist", "front"):
        result = results.get(role, {})
        if result.get("status") != "PASS":
            print(f"{role}: {result.get('status')} {result.get('error', '')}", flush=True)
            continue
        metrics = result["stream_metrics"]
        actual = result["actual_capture"]
        print(
            f"{role}: frames={metrics['successful_frames']} failed={metrics['failed_reads']} "
            f"fps={metrics['achieved_fps']:.3f} ratio={metrics['fps_ratio']:.3f} "
            f"gap_p95={metrics['gap_ms_p95']:.3f}ms gap_max={metrics['gap_ms_max']:.3f}ms "
            f"duplicates={metrics['consecutive_exact_duplicate_frames']} "
            f"mode={actual['width']}x{actual['height']}/{actual['fourcc'] or 'UNKNOWN'} "
            f"reported_fps={actual['fps_reported']:.3f}",
            flush=True,
        )
        negotiation_warnings = result.get("ready_negotiation_warnings", [])
        if negotiation_warnings:
            print(f"{role}: negotiation_warnings={negotiation_warnings}", flush=True)

    print("\n===== CROSS-CAMERA TIMESTAMP SKEW =====", flush=True)
    if cross_skew is None:
        print("unavailable", flush=True)
    else:
        print(
            f"pairs={cross_skew['paired_frames']} overlap={cross_skew['overlap_seconds']:.3f}s "
            f"p50={cross_skew['skew_ms_p50']:.3f}ms "
            f"p95={cross_skew['skew_ms_p95']:.3f}ms "
            f"max={cross_skew['skew_ms_max']:.3f}ms",
            flush=True,
        )

    thresholds = {
        "fps_acceptance": args.fps_acceptance,
        "min_front_achieved_fps": args.min_front_achieved_fps,
        "min_wrist_achieved_fps": args.min_wrist_achieved_fps,
        "min_fps_ratio": args.min_fps_ratio,
        "max_read_failure_rate": args.max_read_failure_rate,
        "max_frame_gap_ms": args.max_frame_gap_ms,
        "max_cross_camera_skew_p95_ms": args.max_cross_camera_skew_p95_ms,
        "classification": "engineering service criteria, not physical hardware safety limits",
    }
    report = {
        "schema_version": 5,
        "created_at_utc": utc_now(),
        "host": {
            "os_name": os.name,
            "platform": platform.platform(),
            "python": sys.version,
            "executable": sys.executable,
        },
        "contract": {
            "camera_reads_only": True,
            "robot_accessed": False,
            "serial_port_opened": False,
            "command_sent": False,
            "backend": args.backend,
            "duration_seconds": args.duration_seconds,
            "startup_order": args.startup_order,
            "startup_ready_timeout_seconds": args.startup_ready_timeout_seconds,
            "open_strategy": args.open_strategy,
            "strict_negotiation": bool(args.strict_negotiation),
            "fps_acceptance": args.fps_acceptance,
            "front_fourcc": args.front_fourcc,
            "wrist_fourcc": args.wrist_fourcc,
        },
        "run_kind": run_kind,
        "acceptance_thresholds": thresholds,
        "camera_results": results,
        "cross_camera_skew": cross_skew,
        "acceptance_failures": reasons,
        "decision": (
            "DUAL_CAMERA_STREAM_FULL_PASS_BUILD_BRIDGE_NEXT"
            if passed and run_kind == "full"
            else "DUAL_CAMERA_STREAM_SMOKE_PASS_RUN_FULL_NEXT"
            if passed
            else "DUAL_CAMERA_STREAM_REVIEW_REQUIRED"
        ),
    }
    report_path = output_dir / "windows_dual_camera_stream_audit_report.json"
    write_json(report_path, report)

    print("\n===== DECISION =====", flush=True)
    print(f"decision='{report['decision']}'", flush=True)
    if reasons:
        print(f"acceptance_failures={reasons}", flush=True)
    print(f"report={report_path}", flush=True)
    print("NO ROBOT OBJECT WAS CREATED. NO SERIAL PORT WAS OPENED. NO COMMAND WAS SENT.", flush=True)
    return 0 if passed else 2


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
        return worker_main(args) if args.worker else parent_main(args)
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
