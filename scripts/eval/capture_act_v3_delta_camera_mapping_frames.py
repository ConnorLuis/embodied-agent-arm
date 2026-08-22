#!/usr/bin/env python3
"""Capture one labeled frame from each ACT V3 camera candidate.

This is a camera-only identification tool. It never imports LeRobot, never
creates a robot object, never opens a serial port, and never sends a motor
command. The two cameras are opened sequentially through explicit V4L2 paths,
then immediately released. The saved JPEGs are for a human to label as
``front`` and ``wrist``; this program deliberately does not guess from pixels.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


DEFAULT_CAMERA_A = Path(
    "/dev/v4l/by-id/usb-icSpring_icspring_camera_202404160005-video-index0"
)
DEFAULT_CAMERA_B = Path(
    "/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0"
)
FORBIDDEN_IMPORT_ROOTS = {
    "lerobot",
    "serial",
    "pyserial",
    "torch",
    "dynamixel_sdk",
}
FORBIDDEN_CALL_LEAVES = {
    "send_action",
    "sync_write",
    "write_calibration",
    "enable_torque",
    "disable_torque",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-a", type=Path, default=DEFAULT_CAMERA_A)
    parser.add_argument("--camera-b", type=Path, default=DEFAULT_CAMERA_B)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument(
        "--successful-warmup-frames",
        type=int,
        default=8,
        help="Successful frames discarded before saving the labeled frame.",
    )
    parser.add_argument(
        "--max-read-attempts",
        type=int,
        default=40,
        help="Maximum read attempts per camera, including warm-up.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    if not 64 <= args.width <= 4096:
        parser.error("--width must be 64..4096")
    if not 64 <= args.height <= 4096:
        parser.error("--height must be 64..4096")
    if not 1.0 <= args.fps <= 120.0:
        parser.error("--fps must be 1..120")
    if not 1 <= args.successful_warmup_frames <= 100:
        parser.error("--successful-warmup-frames must be 1..100")
    if args.max_read_attempts <= args.successful_warmup_frames:
        parser.error(
            "--max-read-attempts must exceed --successful-warmup-frames"
        )
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def call_leaf_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ast.unparse(call.func)


def static_no_robot_audit(path: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    imported: set[str] = set()
    calls: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Call):
            leaf = call_leaf_name(node)
            if leaf in FORBIDDEN_CALL_LEAVES:
                calls.append({"line": int(node.lineno), "call": leaf})

    bad_imports = sorted(
        module
        for module in imported
        if module.split(".", 1)[0] in FORBIDDEN_IMPORT_ROOTS
    )
    if bad_imports or calls:
        raise RuntimeError(
            "camera probe violates the no-robot boundary: "
            f"imports={bad_imports}, calls={calls}"
        )
    return {
        "script": str(path),
        "sha256": sha256_file(path),
        "forbidden_imports": bad_imports,
        "forbidden_robot_write_calls": calls,
        "pass": True,
    }


def resolve_video_device(path: Path) -> dict[str, Any]:
    requested = path.expanduser()
    if not requested.exists():
        raise FileNotFoundError(requested)
    resolved = requested.resolve(strict=True)
    if not resolved.name.startswith("video"):
        raise RuntimeError(f"not a /dev/video* device: {requested} -> {resolved}")
    if resolved.parent != Path("/dev"):
        raise RuntimeError(f"video device resolves outside /dev: {resolved}")
    mode = resolved.stat().st_mode
    if not stat.S_ISCHR(mode):
        raise RuntimeError(f"video device is not a character device: {resolved}")
    if not os.access(resolved, os.R_OK | os.W_OK):
        raise PermissionError(
            f"current user cannot read/write {resolved}; check membership in the video group"
        )

    sysfs = Path("/sys/class/video4linux") / resolved.name
    if not sysfs.exists():
        raise RuntimeError(f"missing sysfs record for {resolved}")
    device_root = (sysfs / "device").resolve(strict=True)
    usb_root: Path | None = None
    for candidate in (device_root, *device_root.parents):
        if (candidate / "idVendor").is_file() and (candidate / "idProduct").is_file():
            usb_root = candidate
            break
    if usb_root is None:
        raise RuntimeError(f"cannot locate USB identity for {resolved}")

    def read_optional(name: str) -> str | None:
        candidate = usb_root / name
        if not candidate.is_file():
            return None
        value = candidate.read_text(encoding="utf-8", errors="replace").strip()
        return value or None

    vendor = read_optional("idVendor")
    product_id = read_optional("idProduct")
    if not vendor or not product_id:
        raise RuntimeError(f"incomplete USB identity for {resolved}")
    return {
        "requested_path": str(requested),
        "resolved_path": str(resolved),
        "symlink": requested.is_symlink(),
        "video_node": resolved.name,
        "sysfs_path": str(sysfs.resolve()),
        "usb_sysfs_path": str(usb_root),
        "usb_vendor_id": vendor.casefold(),
        "usb_product_id": product_id.casefold(),
        "usb_vid_pid": f"{vendor.casefold()}:{product_id.casefold()}",
        "manufacturer": read_optional("manufacturer"),
        "product": read_optional("product"),
        "serial": read_optional("serial"),
        "busnum": read_optional("busnum"),
        "devnum": read_optional("devnum"),
    }


def decode_fourcc(value: float) -> str:
    integer = int(value)
    return "".join(chr((integer >> (8 * index)) & 0xFF) for index in range(4))


def label_frame(cv2: Any, frame: Any, lines: Sequence[str]) -> Any:
    annotated = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    margin = 10
    line_height = 24
    overlay_height = margin * 2 + line_height * len(lines)
    overlay = annotated.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (annotated.shape[1], min(overlay_height, annotated.shape[0])),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.65, annotated, 0.35, 0, annotated)
    for index, line in enumerate(lines):
        y = margin + 17 + index * line_height
        cv2.putText(
            annotated,
            line,
            (margin, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return annotated


def capture_camera(
    label: str,
    device: dict[str, Any],
    output_dir: Path,
    width: int,
    height: int,
    fps: float,
    warmup_successes: int,
    max_attempts: int,
) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is unavailable in the active environment; install/import cv2 before probing"
        ) from exc

    path = device["resolved_path"]
    capture = cv2.VideoCapture(path, cv2.CAP_V4L2)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV could not open {path}")

        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        capture.set(cv2.CAP_PROP_FPS, float(fps))

        successful = 0
        failures = 0
        frame = None
        attempts = 0
        for attempts in range(1, max_attempts + 1):
            ok, candidate = capture.read()
            if not ok or candidate is None or getattr(candidate, "size", 0) == 0:
                failures += 1
                continue
            successful += 1
            frame = candidate
            if successful > warmup_successes:
                break
        if frame is None or successful <= warmup_successes:
            raise RuntimeError(
                f"failed to obtain a post-warmup frame from {path}: "
                f"attempts={attempts}, successes={successful}, failures={failures}"
            )

        actual_height, actual_width = int(frame.shape[0]), int(frame.shape[1])
        actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
        fourcc = decode_fourcc(capture.get(cv2.CAP_PROP_FOURCC))
        timestamp = datetime.now(timezone.utc).isoformat()
        filename = (
            f"camera_{label}_{device['usb_vendor_id']}_{device['usb_product_id']}_"
            f"{device['video_node']}.jpg"
        )
        image_path = output_dir / filename
        annotated = label_frame(
            cv2,
            frame,
            (
                f"CAMERA {label} - human label required",
                f"USB {device['usb_vid_pid']}  {device['video_node']}",
                f"{actual_width}x{actual_height}  reported FPS {actual_fps:.2f}",
            ),
        )
        if not cv2.imwrite(str(image_path), annotated):
            raise RuntimeError(f"cv2.imwrite failed: {image_path}")
        if not image_path.is_file() or image_path.stat().st_size <= 0:
            raise RuntimeError(f"captured image is missing or empty: {image_path}")

        return {
            "label": label,
            "device": device,
            "requested_capture": {"width": width, "height": height, "fps": fps},
            "actual_capture": {
                "width": actual_width,
                "height": actual_height,
                "reported_fps": actual_fps,
                "fourcc": fourcc,
            },
            "read_attempts": attempts,
            "successful_frames": successful,
            "failed_reads": failures,
            "captured_at_utc": timestamp,
            "image_path": str(image_path),
            "image_sha256": sha256_file(image_path),
            "image_size_bytes": image_path.stat().st_size,
        }
    finally:
        capture.release()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")

    print("===== STATIC NO-ROBOT BOUNDARY =====")
    self_audit = static_no_robot_audit(Path(__file__).resolve())
    print("no LeRobot/serial/torch imports or robot-write calls: PASS")
    print("camera-only access: two explicit V4L2 devices, opened sequentially")

    print("\n===== RESOLVE CAMERA IDENTITIES =====")
    camera_a = resolve_video_device(args.camera_a)
    camera_b = resolve_video_device(args.camera_b)
    if camera_a["resolved_path"] == camera_b["resolved_path"]:
        raise RuntimeError("camera A and B resolve to the same video device")
    if camera_a["usb_vid_pid"] == camera_b["usb_vid_pid"]:
        raise RuntimeError(
            "camera A and B have the same VID:PID; explicit identity review is required"
        )
    print(
        f"A: {camera_a['requested_path']} -> {camera_a['resolved_path']} "
        f"USB={camera_a['usb_vid_pid']} serial={camera_a['serial']}"
    )
    print(
        f"B: {camera_b['requested_path']} -> {camera_b['resolved_path']} "
        f"USB={camera_b['usb_vid_pid']} serial={camera_b['serial']}"
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    print("\n===== SEQUENTIAL SINGLE-FRAME CAPTURE =====")
    results: list[dict[str, Any]] = []
    for label, device in (("A", camera_a), ("B", camera_b)):
        print(f"opening camera {label}: {device['resolved_path']}")
        result = capture_camera(
            label=label,
            device=device,
            output_dir=output_dir,
            width=args.width,
            height=args.height,
            fps=args.fps,
            warmup_successes=args.successful_warmup_frames,
            max_attempts=args.max_read_attempts,
        )
        results.append(result)
        print(
            f"camera {label}: USB={device['usb_vid_pid']} "
            f"frame={result['actual_capture']['width']}x"
            f"{result['actual_capture']['height']} "
            f"saved={result['image_path']} PASS"
        )

    report = {
        "schema_version": "act_v3_delta_camera_mapping_probe_v1",
        "scope": {
            "camera_only": True,
            "camera_streams_opened_sequentially": True,
            "robot_module_imported": False,
            "robot_object_created": False,
            "serial_port_opened": False,
            "motor_command_sent": False,
            "model_loaded": False,
            "dataset_loaded": False,
            "hardware_deployment_authorized": False,
        },
        "self_audit": self_audit,
        "captures": results,
        "human_label_required": {
            "instruction": (
                "Open both JPEGs and assign exactly one image to front and the other to wrist."
            ),
            "camera_A_role": None,
            "camera_B_role": None,
        },
        "task_specific_note": (
            "The wrist camera is needed to observe the gripper and downward cube approach; "
            "this probe does not infer roles automatically."
        ),
        "decision": "CAMERA_FRAMES_CAPTURED_AWAIT_HUMAN_FRONT_WRIST_LABEL",
        "hardware_deployment_authorized": False,
    }
    report_path = output_dir / "camera_mapping_probe_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("\n===== DECISION =====")
    print("decision='CAMERA_FRAMES_CAPTURED_AWAIT_HUMAN_FRONT_WRIST_LABEL'")
    print("Open the two JPEGs and report: CAMERA A=front/wrist, CAMERA B=front/wrist.")
    print("HARDWARE DEPLOYMENT REMAINS BLOCKED.")
    print("NO ROBOT OBJECT WAS CREATED. NO SERIAL PORT WAS OPENED.")
    print("NO MOTOR COMMAND WAS SENT. NO MODEL OR DATASET WAS LOADED.")
    print("\n===== OUTPUT =====")
    for result in results:
        print(result["image_path"])
    print(report_path)
    print("ACT V3 DELTA CAMERA MAPPING FRAME CAPTURE: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nCAMERA PROBE ABORTED: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("HARDWARE DEPLOYMENT REMAINS BLOCKED.", file=sys.stderr)
        print(
            "NO ROBOT OBJECT WAS CREATED. NO SERIAL PORT WAS OPENED. "
            "NO MOTOR COMMAND WAS SENT.",
            file=sys.stderr,
        )
        raise
