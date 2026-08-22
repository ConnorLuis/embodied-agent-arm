#!/usr/bin/env python3
"""Safely probe the two ACT V3 cameras with native Windows OpenCV.

The parent process starts one worker per physical camera.  Every worker has a
hard wall-clock timeout, so a blocked VideoCapture.open/read cannot stall the
whole audit indefinitely.

This script only enumerates cameras and reads frames.  It contains no LeRobot,
robot, teleoperator, serial-port, or motor-command code.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_VID = 0x32E6
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

    values = {
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
    }
    return values[name]


def decode_fourcc(value: float) -> str:
    number = int(value)
    chars = [chr((number >> (8 * shift)) & 0xFF) for shift in range(4)]
    return "".join(chars).rstrip("\x00")


def camera_info_payload(info: Any) -> dict[str, Any]:
    return {
        "index": int(info.index),
        "name": str(info.name),
        "vid": normalize_usb_id(info.vid),
        "pid": normalize_usb_id(info.pid),
        "vid_hex": None if normalize_usb_id(info.vid) is None else f"{normalize_usb_id(info.vid):04X}",
        "pid_hex": None if normalize_usb_id(info.pid) is None else f"{normalize_usb_id(info.pid):04X}",
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


def worker_main(args: argparse.Namespace) -> int:
    result_path = Path(args.result_json).resolve()
    request = CameraRequest(
        role=args.role,
        vid=parse_hex_id(args.vid),
        pid=parse_hex_id(args.pid),
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    started = time.monotonic()
    cap = None
    result: dict[str, Any] = {
        "status": "FAILED",
        "role": request.role,
        "request": asdict(request),
        "backend_name": args.backend,
        "started_at_utc": utc_now(),
    }

    try:
        import cv2
        import numpy as np

        api = backend_code(args.backend)
        camera, inventory = find_exact_camera(request, api)
        result["inventory"] = inventory
        result["selected_camera"] = camera_info_payload(camera)

        cap = cv2.VideoCapture(int(camera.index), int(camera.backend))
        result["opened"] = bool(cap.isOpened())
        if not cap.isOpened():
            raise RuntimeError("VideoCapture did not open")

        property_sets = {
            "buffer_size_1": bool(cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)),
            "width": bool(cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(request.width))),
            "height": bool(cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(request.height))),
            "fps": bool(cap.set(cv2.CAP_PROP_FPS, float(request.fps))),
        }
        result["property_set_results"] = property_sets

        successful_reads = 0
        failed_reads = 0
        frame = None
        for _ in range(args.max_read_attempts):
            ok, candidate = cap.read()
            if ok and candidate is not None and candidate.size > 0:
                frame = candidate
                successful_reads += 1
                if successful_reads >= args.successful_warmup_frames:
                    break
            else:
                failed_reads += 1

        if frame is None or successful_reads < args.successful_warmup_frames:
            raise RuntimeError(
                f"Insufficient valid frames: successes={successful_reads}, failures={failed_reads}"
            )

        actual = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "fourcc": decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC)),
            "backend_reported": cap.getBackendName(),
            "shape": list(frame.shape),
            "dtype": str(frame.dtype),
            "pixel_min": int(np.min(frame)),
            "pixel_max": int(np.max(frame)),
            "pixel_mean": float(np.mean(frame)),
            "pixel_std": float(np.std(frame)),
            "successful_reads": successful_reads,
            "failed_reads": failed_reads,
        }
        result["actual_capture"] = actual

        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_path = output_dir / f"{request.role}_raw.png"
        annotated_path = output_dir / f"{request.role}_annotated.jpg"

        if not cv2.imwrite(str(raw_path), frame):
            raise RuntimeError(f"Failed to write {raw_path}")

        annotated = frame.copy()
        lines = [
            f"role={request.role} VID:PID={request.vid:04X}:{request.pid:04X}",
            f"backend={args.backend} index={camera.index} actual={actual['width']}x{actual['height']}@{actual['fps']:.2f}",
            f"fourcc={actual['fourcc'] or 'UNKNOWN'} utc={utc_now()}",
        ]
        y = 30
        for line in lines:
            cv2.putText(annotated, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(annotated, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            y += 26
        if not cv2.imwrite(str(annotated_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"Failed to write {annotated_path}")

        result.update(
            {
                "status": "PASS",
                "raw_image": str(raw_path),
                "annotated_image": str(annotated_path),
            }
        )
        return_code = 0
    except Exception as exc:  # worker must preserve diagnostics for the parent
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        return_code = 1
    finally:
        if cap is not None:
            cap.release()
        result["elapsed_seconds"] = time.monotonic() - started
        result["finished_at_utc"] = utc_now()
        write_json(result_path, result)

    return return_code


def terminate_worker(process: subprocess.Popen[str]) -> str:
    process.kill()
    try:
        stdout, _ = process.communicate(timeout=3.0)
    except subprocess.TimeoutExpired:
        stdout = "Worker did not terminate within the post-kill grace period."
    return stdout or ""


def run_one_worker(
    request: CameraRequest,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    result_path = output_dir / f"{request.role}_worker_result.json"
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
        "--successful-warmup-frames",
        str(args.successful_warmup_frames),
        "--max-read-attempts",
        str(args.max_read_attempts),
        "--output-dir",
        str(output_dir),
        "--result-json",
        str(result_path),
    ]

    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=creation_flags,
    )
    try:
        stdout, _ = process.communicate(timeout=args.worker_timeout_seconds)
        timed_out = False
    except subprocess.TimeoutExpired:
        timed_out = True
        stdout = terminate_worker(process)

    elapsed = time.monotonic() - started
    if timed_out:
        return {
            "status": "TIMEOUT",
            "role": request.role,
            "request": asdict(request),
            "timeout_seconds": args.worker_timeout_seconds,
            "elapsed_seconds": elapsed,
            "worker_stdout": stdout,
        }

    if result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        payload = {
            "status": "FAILED",
            "role": request.role,
            "request": asdict(request),
            "error": "Worker result JSON was not created",
        }
    payload["worker_exit_code"] = process.returncode
    payload["worker_stdout"] = stdout
    payload["parent_observed_elapsed_seconds"] = elapsed
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Native-Windows front/wrist camera probe with per-camera hard timeouts."
    )
    parser.add_argument("--backend", choices=("dshow", "msmf"), default="dshow")
    parser.add_argument("--vid", default="32E6")
    parser.add_argument("--front-pid", default="9221")
    parser.add_argument("--wrist-pid", default="9005")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--front-fps", type=float, default=30.0)
    parser.add_argument("--wrist-fps", type=float, default=15.0)
    parser.add_argument("--successful-warmup-frames", type=int, default=5)
    parser.add_argument("--max-read-attempts", type=int, default=30)
    parser.add_argument("--worker-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--role", choices=("front", "wrist"), help=argparse.SUPPRESS)
    parser.add_argument("--pid", help=argparse.SUPPRESS)
    parser.add_argument("--fps", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--result-json", help=argparse.SUPPRESS)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if os.name != "nt":
        raise RuntimeError("This probe must run under native Windows Python, not WSL/Linux Python")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("width and height must be positive")
    if args.successful_warmup_frames <= 0 or args.max_read_attempts <= 0:
        raise ValueError("frame counts must be positive")
    if args.successful_warmup_frames > args.max_read_attempts:
        raise ValueError("successful warmup frames cannot exceed max read attempts")
    if args.worker_timeout_seconds <= 0:
        raise ValueError("worker timeout must be positive")
    if args.worker and not all((args.role, args.pid, args.fps, args.result_json)):
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

    print("===== WINDOWS CAMERA PROBE CONTRACT =====", flush=True)
    print("camera reads only; no robot, serial port, or command API", flush=True)
    print(f"backend={args.backend} hard_timeout={args.worker_timeout_seconds:.1f}s per camera", flush=True)
    print("wrist=32E6:9005; front=32E6:9221", flush=True)

    results: dict[str, dict[str, Any]] = {}
    for request in requests:
        print(
            f"\nProbing {request.role}: VID:PID={request.vid:04X}:{request.pid:04X} "
            f"requested={request.width}x{request.height}@{request.fps:.2f}",
            flush=True,
        )
        result = run_one_worker(request, args, output_dir)
        results[request.role] = result
        if result["status"] == "PASS":
            actual = result["actual_capture"]
            print(
                f"{request.role}: PASS index={result['selected_camera']['index']} "
                f"actual={actual['width']}x{actual['height']}@{actual['fps']:.2f} "
                f"fourcc={actual['fourcc'] or 'UNKNOWN'} std={actual['pixel_std']:.3f}",
                flush=True,
            )
        else:
            print(
                f"{request.role}: {result['status']} "
                f"error={result.get('error', 'hard worker timeout')}",
                flush=True,
            )

    both_pass = all(result.get("status") == "PASS" for result in results.values())
    report = {
        "schema_version": 1,
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
            "worker_timeout_seconds": args.worker_timeout_seconds,
        },
        "results": results,
        "decision": "BOTH_CAMERAS_CAPTURED_REVIEW_IMAGES" if both_pass else "CAMERA_PROBE_INCOMPLETE_REVIEW_BACKEND",
    }
    report_path = output_dir / "windows_camera_probe_report.json"
    write_json(report_path, report)

    print("\n===== DECISION =====", flush=True)
    print(f"decision='{report['decision']}'", flush=True)
    print(f"report={report_path}", flush=True)
    print("NO ROBOT OBJECT WAS CREATED. NO SERIAL PORT WAS OPENED. NO COMMAND WAS SENT.", flush=True)
    return 0 if both_pass else 2


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
