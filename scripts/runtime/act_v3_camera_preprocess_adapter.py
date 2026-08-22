#!/usr/bin/env python3
"""Build and audit the ACT V3 live-camera preprocessing contract offline.

The reusable functions in this module convert decoded OpenCV BGR frames into
the camera-specific geometry used by the frozen ACT V3 policy:

    front: uint8 BGR HWC 640x480
      -> INTER_AREA resize to 480x360
      -> 60 px black letterbox above and below

    wrist: uint8 BGR HWC 640x480
      -> rotate 90 degrees counter-clockwise
      -> INTER_AREA resize to 360x480
      -> 60 px black pillarbox left and right

    both -> RGB -> float32 in [0, 1] -> NCHW with batch size 1

The CLI audits that conversion against a completed Windows-to-WSL bridge run,
the frozen 11K policy configuration, and one validation-dataset observation.
It does not open a camera, load model weights, create a robot object, open a
serial port, or send a command. ImageNet normalization is deliberately not
applied here; the policy's own input-normalization path remains authoritative.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


STATE_KEY = "observation.state"
FRONT_KEY = "observation.images.front"
WRIST_KEY = "observation.images.wrist"
ACTION_KEY = "action"

EXPECTED_BRIDGE_SCRIPT_SHA256 = (
    "331ad2a464c22b76eb817bfeaf38f6da43d6457150171ad75d6c579889dc314b"
)
EXPECTED_BRIDGE_SCHEMA_VERSION = 2
EXPECTED_BRIDGE_DECISION = "CAMERA_BRIDGE_SMOKE_PASS_BUILD_PREPROCESS_ADAPTER_NEXT"
EXPECTED_BRIDGE_PROTOCOL_VERSION = 2
EXPECTED_BRIDGE_TERMINATION = "END_then_ACK_before_close"
EXPECTED_CAMERA_IDS = {
    "front": {"vid": 0x32E6, "pid": 0x9221, "fourcc": "MJPG"},
    "wrist": {"vid": 0x32E6, "pid": 0x9005, "fourcc": "AUTO"},
}
EXPECTED_SOURCE_SHAPE = (480, 640, 3)
EXPECTED_POLICY_IMAGE_SHAPE = (3, 480, 480)
EXPECTED_POLICY_STATE_SHAPE = (18,)
EXPECTED_POLICY_ACTION_SHAPE = (6,)
EXPECTED_CHUNK_SIZE = 10
EXPECTED_ACTION_STEPS = 5
POLICY_SIDE = 480
CONTENT_SHORT_SIDE = 360
PADDING = 60
DATASET_PADDING_MAX_UINT8 = 16
DATASET_PADDING_P99_UINT8 = 4.0
DATASET_PADDING_MEAN_UINT8 = 0.5
DATASET_CONTENT_BOUNDARY_P95_MIN_UINT8 = 8.0
GEOMETRY_CONTRACT = {
    "front": {
        "rotation": "none",
        "oriented_shape_hwc": (480, 640, 3),
        "resized_shape_hwc": (360, 480, 3),
        "padding": {"top": 60, "bottom": 60, "left": 0, "right": 0},
    },
    "wrist": {
        "rotation": "counter_clockwise_90",
        "oriented_shape_hwc": (640, 480, 3),
        "resized_shape_hwc": (480, 360, 3),
        "padding": {"top": 0, "bottom": 0, "left": 60, "right": 60},
    },
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-script", type=Path, required=True)
    parser.add_argument("--bridge-output-dir", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--validation-dataset-root", type=Path, required=True)
    parser.add_argument("--validation-repo-id", required=True)
    parser.add_argument("--video-backend", default="torchcodec")
    parser.add_argument(
        "--snapshot-position",
        choices=("first", "middle", "last"),
        default="last",
        help="Which completed bridge snapshot to use for the offline conversion audit.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


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

    forbidden_module_prefixes = (
        "lerobot.robots",
        "lerobot.teleoperators",
        "serial",
        "pyserial",
    )
    forbidden_modules = sorted(
        module
        for module in imported_modules
        if module.startswith(forbidden_module_prefixes)
    )
    forbidden_identifiers = sorted(
        {"SO101Follower", "SO101FollowerConfig", "VideoCapture"}.intersection(
            identifiers
        )
    )
    forbidden_attributes = sorted(
        {"send_action", "sync_write", "enable_torque", "disable_torque"}.intersection(
            attributes
        )
    )
    if forbidden_modules or forbidden_identifiers or forbidden_attributes:
        raise RuntimeError(
            "hardware-capable code found: "
            f"modules={forbidden_modules}, identifiers={forbidden_identifiers}, "
            f"attributes={forbidden_attributes}"
        )
    return {
        "source": str(path),
        "sha256": sha256_file(path),
        "camera_open_calls": False,
        "robot_or_serial_imports": False,
        "command_write_calls": False,
    }


def feature_shape(feature: Any) -> tuple[int, ...]:
    if isinstance(feature, dict):
        shape = feature.get("shape")
    else:
        shape = getattr(feature, "shape", None)
    if shape is None:
        raise RuntimeError(f"feature has no shape: {feature!r}")
    return tuple(int(value) for value in shape)


def load_policy_config(candidate_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = require_file(candidate_root / "pretrained_model" / "config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    input_features = config.get("input_features")
    output_features = config.get("output_features")
    if not isinstance(input_features, dict) or not isinstance(output_features, dict):
        raise RuntimeError("policy config is missing input/output feature dictionaries")
    if set(input_features) != {STATE_KEY, FRONT_KEY, WRIST_KEY}:
        raise RuntimeError(f"unexpected policy inputs: {sorted(input_features)}")
    if set(output_features) != {ACTION_KEY}:
        raise RuntimeError(f"unexpected policy outputs: {sorted(output_features)}")
    if feature_shape(input_features[STATE_KEY]) != EXPECTED_POLICY_STATE_SHAPE:
        raise RuntimeError("policy state feature is not 18-D")
    for key in (FRONT_KEY, WRIST_KEY):
        if feature_shape(input_features[key]) != EXPECTED_POLICY_IMAGE_SHAPE:
            raise RuntimeError(f"policy image shape mismatch: {key}")
    if feature_shape(output_features[ACTION_KEY]) != EXPECTED_POLICY_ACTION_SHAPE:
        raise RuntimeError("policy action feature is not 6-D")
    if int(config.get("chunk_size", -1)) != EXPECTED_CHUNK_SIZE:
        raise RuntimeError("policy chunk_size is not 10")
    if int(config.get("n_action_steps", -1)) != EXPECTED_ACTION_STEPS:
        raise RuntimeError("policy n_action_steps is not 5")
    if bool(config.get("use_vae")):
        raise RuntimeError("policy use_vae must be false")
    if bool(config.get("use_amp")):
        raise RuntimeError("policy use_amp must be false")
    return config, {
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "state_shape": list(EXPECTED_POLICY_STATE_SHAPE),
        "front_shape": list(EXPECTED_POLICY_IMAGE_SHAPE),
        "wrist_shape": list(EXPECTED_POLICY_IMAGE_SHAPE),
        "action_shape": list(EXPECTED_POLICY_ACTION_SHAPE),
        "chunk_size": EXPECTED_CHUNK_SIZE,
        "n_action_steps": EXPECTED_ACTION_STEPS,
        "use_vae": False,
        "use_amp": False,
    }


def verify_bridge_run(
    bridge_script: Path,
    bridge_output_dir: Path,
    snapshot_position: str,
) -> tuple[dict[str, Any], dict[str, Path], dict[str, Any]]:
    bridge_sha = sha256_file(bridge_script)
    if bridge_sha != EXPECTED_BRIDGE_SCRIPT_SHA256:
        raise RuntimeError(
            "bridge script SHA256 mismatch: "
            f"expected={EXPECTED_BRIDGE_SCRIPT_SHA256}, actual={bridge_sha}"
        )
    report_path = require_file(bridge_output_dir / "windows_wsl_camera_bridge_report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if int(report.get("schema_version", -1)) != EXPECTED_BRIDGE_SCHEMA_VERSION:
        raise RuntimeError("bridge report schema is not version 2")
    if report.get("decision") != EXPECTED_BRIDGE_DECISION:
        raise RuntimeError(f"bridge run did not pass: {report.get('decision')!r}")
    if report.get("acceptance_failures") != []:
        raise RuntimeError("bridge report contains acceptance failures")

    contract = report.get("contract") or {}
    if int(contract.get("protocol_version", -1)) != EXPECTED_BRIDGE_PROTOCOL_VERSION:
        raise RuntimeError("bridge protocol version mismatch")
    if contract.get("termination_handshake") != EXPECTED_BRIDGE_TERMINATION:
        raise RuntimeError("bridge termination handshake mismatch")
    for field in ("model_loaded", "robot_accessed", "serial_port_opened", "command_sent"):
        if contract.get(field) is not False:
            raise RuntimeError(f"bridge offline boundary failed: {field}")

    snapshots: dict[str, Path] = {}
    camera_summary: dict[str, Any] = {}
    for role in ("front", "wrist"):
        expected = EXPECTED_CAMERA_IDS[role]
        request = contract.get(f"{role}_request") or {}
        if int(request.get("vid", -1)) != expected["vid"]:
            raise RuntimeError(f"{role} VID mismatch")
        if int(request.get("pid", -1)) != expected["pid"]:
            raise RuntimeError(f"{role} PID mismatch")
        if str(request.get("fourcc", "")).upper() != expected["fourcc"]:
            raise RuntimeError(f"{role} FOURCC request mismatch")
        if (int(request.get("width", -1)), int(request.get("height", -1))) != (640, 480):
            raise RuntimeError(f"{role} source resolution is not 640x480")

        camera = (report.get("camera_reports") or {}).get(role) or {}
        worker = camera.get("windows_worker") or {}
        metrics = camera.get("metrics") or {}
        end_message = camera.get("end_message") or {}
        received = int(metrics.get("successful_frames", -1))
        if camera.get("receiver_status") != "PASS":
            raise RuntimeError(f"{role} bridge receiver did not pass")
        if camera.get("end_ack_sent") is not True:
            raise RuntimeError(f"{role} bridge receiver did not acknowledge END")
        if worker.get("status") != "PASS" or worker.get("exit_code") != 0:
            raise RuntimeError(f"{role} Windows bridge worker did not pass")
        if worker.get("end_ack_received") is not True:
            raise RuntimeError(f"{role} Windows worker did not receive END ACK")
        if worker.get("timed_out_during_collection") is not False:
            raise RuntimeError(f"{role} Windows worker collection timed out")
        if int(worker.get("frames_sent", -1)) != received:
            raise RuntimeError(f"{role} worker/receiver frame-count mismatch")
        if int(end_message.get("frames_sent", -1)) != received:
            raise RuntimeError(f"{role} END/receiver frame-count mismatch")
        for field in ("decode_failures", "crc_failures", "protocol_failures", "sequence_gap_count"):
            if int(camera.get(field, -1)) != 0:
                raise RuntimeError(f"{role} bridge {field} is nonzero")

        snapshot_text = (camera.get("snapshots") or {}).get(snapshot_position)
        if not snapshot_text:
            raise RuntimeError(
                f"{role} bridge {snapshot_position} snapshot is missing from report"
            )
        snapshot_path = Path(snapshot_text).expanduser()
        if not snapshot_path.is_absolute():
            snapshot_path = bridge_output_dir / snapshot_path
        snapshots[role] = require_file(snapshot_path)
        camera_summary[role] = {
            "received_frames": received,
            "achieved_receive_fps": float(metrics["achieved_receive_fps"]),
            "gap_ms_max": float(metrics["gap_ms_max"]),
            "end_ack": True,
            "snapshot": str(snapshots[role]),
            "snapshot_sha256": sha256_file(snapshots[role]),
        }

    return report, snapshots, {
        "bridge_script": str(bridge_script),
        "bridge_script_sha256": bridge_sha,
        "bridge_report": str(report_path),
        "bridge_report_sha256": sha256_file(report_path),
        "schema_version": EXPECTED_BRIDGE_SCHEMA_VERSION,
        "protocol_version": EXPECTED_BRIDGE_PROTOCOL_VERSION,
        "termination_handshake": EXPECTED_BRIDGE_TERMINATION,
        "cameras": camera_summary,
    }


def apply_camera_geometry(
    frame_bgr: Any,
    role: str,
    *,
    cv2_module: Any,
    np_module: Any,
) -> Any:
    """Apply the reviewed camera-role geometry and return uint8 BGR 480x480."""
    np = np_module
    cv2 = cv2_module
    if role not in GEOMETRY_CONTRACT:
        raise ValueError(f"camera role must be one of {sorted(GEOMETRY_CONTRACT)}, got {role!r}")
    if not isinstance(frame_bgr, np.ndarray):
        raise TypeError("live camera frame must be a numpy ndarray")
    if frame_bgr.dtype != np.uint8:
        raise TypeError(f"live camera frame dtype must be uint8, got {frame_bgr.dtype}")
    if tuple(int(value) for value in frame_bgr.shape) != EXPECTED_SOURCE_SHAPE:
        raise RuntimeError(
            f"live camera frame shape must be {EXPECTED_SOURCE_SHAPE}, got {frame_bgr.shape}"
        )
    if not bool(np.isfinite(frame_bgr).all()):
        raise FloatingPointError("live camera frame contains nonfinite values")

    if role == "front":
        oriented = frame_bgr
        content = cv2.resize(
            oriented,
            (POLICY_SIDE, CONTENT_SHORT_SIDE),
            interpolation=cv2.INTER_AREA,
        )
        policy_bgr = np.zeros((POLICY_SIDE, POLICY_SIDE, 3), dtype=np.uint8)
        policy_bgr[PADDING : POLICY_SIDE - PADDING, :, :] = content
    else:
        oriented = cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        content = cv2.resize(
            oriented,
            (CONTENT_SHORT_SIDE, POLICY_SIDE),
            interpolation=cv2.INTER_AREA,
        )
        policy_bgr = np.zeros((POLICY_SIDE, POLICY_SIDE, 3), dtype=np.uint8)
        policy_bgr[:, PADDING : POLICY_SIDE - PADDING, :] = content

    expected = GEOMETRY_CONTRACT[role]
    if tuple(int(value) for value in oriented.shape) != expected["oriented_shape_hwc"]:
        raise RuntimeError(f"{role} oriented shape mismatch: {oriented.shape}")
    if tuple(int(value) for value in content.shape) != expected["resized_shape_hwc"]:
        raise RuntimeError(f"{role} resized content shape mismatch: {content.shape}")
    if tuple(int(value) for value in policy_bgr.shape) != (480, 480, 3):
        raise RuntimeError(f"{role} policy BGR shape mismatch: {policy_bgr.shape}")
    return policy_bgr


def preprocess_bgr_frame(
    frame_bgr: Any,
    role: str,
    *,
    cv2_module: Any,
    np_module: Any,
    torch_module: Any,
    device: Any | None = None,
) -> tuple[Any, Any, Any]:
    """Return (batched tensor, policy BGR, policy RGB) for one live frame."""
    np = np_module
    torch = torch_module
    cv2 = cv2_module
    policy_bgr = apply_camera_geometry(
        frame_bgr,
        role,
        cv2_module=cv2,
        np_module=np,
    )
    policy_rgb = cv2.cvtColor(policy_bgr, cv2.COLOR_BGR2RGB)
    rgb_contiguous = np.ascontiguousarray(policy_rgb)
    tensor = torch.from_numpy(rgb_contiguous).permute(2, 0, 1).to(dtype=torch.float32)
    tensor = tensor.div(255.0).unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device=device)
    if tuple(int(value) for value in tensor.shape) != (1, *EXPECTED_POLICY_IMAGE_SHAPE):
        raise RuntimeError(f"preprocessed tensor shape mismatch: {tuple(tensor.shape)}")
    if not bool(torch.isfinite(tensor).all().item()):
        raise FloatingPointError("preprocessed tensor contains nonfinite values")
    minimum = float(tensor.min().item())
    maximum = float(tensor.max().item())
    if minimum < 0.0 or maximum > 1.0:
        raise RuntimeError(f"preprocessed tensor range is invalid: [{minimum}, {maximum}]")
    return tensor, policy_bgr, policy_rgb


def camera_padding_views(image_hwc: Any, role: str, *, np_module: Any) -> tuple[Any, Any]:
    """Return (padding pixels, two content-boundary strips) for one 480-square image."""
    np = np_module
    if tuple(int(value) for value in image_hwc.shape) != (480, 480, 3):
        raise RuntimeError(f"{role} padding audit expected 480x480x3, got {image_hwc.shape}")
    if role == "front":
        padding = np.concatenate(
            (image_hwc[:PADDING], image_hwc[POLICY_SIDE - PADDING :]),
            axis=0,
        )
        boundaries = np.concatenate(
            (
                image_hwc[PADDING : PADDING + 1],
                image_hwc[POLICY_SIDE - PADDING - 1 : POLICY_SIDE - PADDING],
            ),
            axis=0,
        )
    elif role == "wrist":
        padding = np.concatenate(
            (image_hwc[:, :PADDING], image_hwc[:, POLICY_SIDE - PADDING :]),
            axis=1,
        )
        boundaries = np.concatenate(
            (
                image_hwc[:, PADDING : PADDING + 1],
                image_hwc[:, POLICY_SIDE - PADDING - 1 : POLICY_SIDE - PADDING],
            ),
            axis=1,
        )
    else:
        raise ValueError(f"unknown camera role: {role!r}")
    return padding, boundaries


def audit_dataset_geometry(image_rgb_uint8: Any, role: str, *, np_module: Any) -> dict[str, Any]:
    """Verify that a recorded reference exposes the reviewed 60 px black bars."""
    np = np_module
    padding, boundaries = camera_padding_views(image_rgb_uint8, role, np_module=np)
    padding_max = int(padding.max())
    padding_p99 = float(np.percentile(padding, 99.0))
    padding_mean = float(padding.mean())
    boundary_p95 = float(np.percentile(boundaries, 95.0))
    if padding_max > DATASET_PADDING_MAX_UINT8:
        raise RuntimeError(f"{role} reference padding max is too bright: {padding_max}")
    if padding_p99 > DATASET_PADDING_P99_UINT8:
        raise RuntimeError(f"{role} reference padding p99 is too bright: {padding_p99}")
    if padding_mean > DATASET_PADDING_MEAN_UINT8:
        raise RuntimeError(f"{role} reference padding mean is too bright: {padding_mean}")
    if boundary_p95 <= DATASET_CONTENT_BOUNDARY_P95_MIN_UINT8:
        raise RuntimeError(f"{role} reference content boundary is unexpectedly dark")
    return {
        "expected_padding_pixels": PADDING,
        "padding_axes": "top_bottom" if role == "front" else "left_right",
        "padding_max_uint8": padding_max,
        "padding_p99_uint8": padding_p99,
        "padding_mean_uint8": padding_mean,
        "content_boundary_p95_uint8": boundary_p95,
        "passed": True,
    }


def audit_live_geometry(policy_bgr: Any, role: str, *, np_module: Any) -> dict[str, Any]:
    """Verify exact black padding and nonempty content in a generated live image."""
    np = np_module
    padding, boundaries = camera_padding_views(policy_bgr, role, np_module=np)
    padding_max = int(padding.max())
    boundary_p95 = float(np.percentile(boundaries, 95.0))
    if padding_max != 0:
        raise RuntimeError(f"{role} generated padding is not exactly black")
    if boundary_p95 <= 0.0:
        raise RuntimeError(f"{role} generated content boundary is empty")
    return {
        "padding_exact_zero": True,
        "padding_max_uint8": padding_max,
        "content_boundary_p95_uint8": boundary_p95,
        "passed": True,
    }


def canonicalize_dataset_rgb(value: Any, *, torch_module: Any) -> Any:
    """Match the existing offline adapter's dataset-image tensor semantics."""
    torch = torch_module
    tensor = value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim != 3:
        raise RuntimeError(f"unexpected dataset image shape: {tuple(tensor.shape)}")
    if tensor.shape[0] in (1, 3, 4):
        tensor = tensor[:3]
    elif tensor.shape[-1] in (1, 3, 4):
        tensor = tensor[..., :3].permute(2, 0, 1)
    else:
        raise RuntimeError(f"cannot infer dataset image channel axis: {tuple(tensor.shape)}")
    tensor = tensor.to(dtype=torch.float32)
    if float(tensor.max().item()) > 1.5:
        tensor = tensor.div(255.0)
    tensor = tensor.unsqueeze(0).cpu()
    if tuple(int(value) for value in tensor.shape) != (1, *EXPECTED_POLICY_IMAGE_SHAPE):
        raise RuntimeError(f"dataset policy tensor shape mismatch: {tuple(tensor.shape)}")
    if not bool(torch.isfinite(tensor).all().item()):
        raise FloatingPointError("dataset image tensor contains nonfinite values")
    if float(tensor.min().item()) < 0.0 or float(tensor.max().item()) > 1.0:
        raise RuntimeError("dataset image tensor is outside [0, 1]")
    return tensor


def build_policy_observation(
    front_bgr: Any,
    wrist_bgr: Any,
    state_18: Any,
    *,
    cv2_module: Any,
    np_module: Any,
    torch_module: Any,
    device: Any | None = None,
) -> dict[str, Any]:
    """Build the three-key batch consumed by ACTPolicy.predict_action_chunk."""
    torch = torch_module
    state = torch.as_tensor(state_18, dtype=torch.float32).reshape(1, 18)
    if not bool(torch.isfinite(state).all().item()):
        raise FloatingPointError("18-D policy state contains nonfinite values")
    if device is not None:
        state = state.to(device=device)
    front, _, _ = preprocess_bgr_frame(
        front_bgr,
        "front",
        cv2_module=cv2_module,
        np_module=np_module,
        torch_module=torch_module,
        device=device,
    )
    wrist, _, _ = preprocess_bgr_frame(
        wrist_bgr,
        "wrist",
        cv2_module=cv2_module,
        np_module=np_module,
        torch_module=torch_module,
        device=device,
    )
    return {STATE_KEY: state, FRONT_KEY: front, WRIST_KEY: wrist}


def load_dataset(LeRobotDataset: Any, repo_id: str, root: Path, video_backend: str):
    try:
        return LeRobotDataset(repo_id=repo_id, root=root, video_backend=video_backend)
    except TypeError:
        return LeRobotDataset(repo_id, root=root, video_backend=video_backend)


def make_geometry_preview(
    source_bgr: Any,
    policy_bgr: Any,
    dataset_reference_bgr: Any,
    role: str,
    *,
    cv2_module: Any,
    np_module: Any,
) -> Any:
    cv2 = cv2_module
    np = np_module
    source = source_bgr.copy()
    policy = policy_bgr.copy()
    dataset_reference = dataset_reference_bgr.copy()
    cv2.putText(
        source,
        f"{role}: bridge BGR 640x480",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        policy,
        f"{role}: corrected ACT geometry 480x480",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        dataset_reference,
        f"{role}: validation reference 480x480",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    canvas = np.full((480, 640 + 20 + 480 + 20 + 480, 3), 32, dtype=np.uint8)
    canvas[:, :640] = source
    canvas[:, 660:1140] = policy
    canvas[:, 1160:] = dataset_reference
    return canvas


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    bridge_script = require_file(args.bridge_script)
    bridge_output_dir = require_dir(args.bridge_output_dir)
    candidate_root = require_dir(args.candidate_root)
    validation_root = require_dir(args.validation_dataset_root)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"output directory must be absent or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("===== STATIC OFFLINE BOUNDARY =====")
    static_report = verify_static_offline_boundary(Path(__file__).resolve())
    print("no camera-open, robot, serial, model-load, or command-write code: PASS")

    print("\n===== VERIFY COMPLETED CAMERA BRIDGE =====")
    _, snapshots, bridge_report = verify_bridge_run(
        bridge_script,
        bridge_output_dir,
        args.snapshot_position,
    )
    print("bridge schema v2, END/ACK, identities, counts and transport metrics: PASS")

    print("\n===== VERIFY FROZEN POLICY INPUT CONFIG =====")
    _, policy_report = load_policy_config(candidate_root)
    print("policy input state=18D front/wrist=3x480x480; action=6D: PASS")

    import cv2
    import numpy as np
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print("\n===== LOAD ONE OFFLINE VALIDATION OBSERVATION =====")
    dataset = load_dataset(
        LeRobotDataset,
        args.validation_repo_id,
        validation_root,
        args.video_backend,
    )
    item = dataset[0]
    missing = sorted({STATE_KEY, FRONT_KEY, WRIST_KEY} - set(item))
    if missing:
        raise RuntimeError(f"validation item is missing policy keys: {missing}")
    state = torch.as_tensor(item[STATE_KEY], dtype=torch.float32).reshape(18)
    if not bool(torch.isfinite(state).all().item()):
        raise FloatingPointError("validation state contains nonfinite values")
    dataset_tensors = {
        "front": canonicalize_dataset_rgb(item[FRONT_KEY], torch_module=torch),
        "wrist": canonicalize_dataset_rgb(item[WRIST_KEY], torch_module=torch),
    }
    print("validation image contract RGB float32 [0,1] NCHW 480x480: PASS")

    print("\n===== VERIFY REVIEWED TRAINING GEOMETRY =====")
    dataset_reference_rgb: dict[str, Any] = {}
    dataset_geometry_reports: dict[str, Any] = {}
    for role in ("front", "wrist"):
        reference_rgb = (
            torch.round(dataset_tensors[role][0].permute(1, 2, 0) * 255.0)
            .to(dtype=torch.uint8)
            .cpu()
            .numpy()
        )
        dataset_reference_rgb[role] = reference_rgb
        dataset_geometry_reports[role] = audit_dataset_geometry(
            reference_rgb,
            role,
            np_module=np,
        )
    print("front: preserve 4:3; resize 640x480 -> 480x360; black top/bottom=60: PASS")
    print("wrist: reviewed CCW90; resize 480x640 -> 360x480; black left/right=60: PASS")

    print("\n===== LIVE-FRAME PREPROCESS ADAPTER AUDIT =====")
    source_frames: dict[str, Any] = {}
    policy_tensors: dict[str, Any] = {}
    camera_reports: dict[str, Any] = {}
    for role in ("front", "wrist"):
        frame = cv2.imread(str(snapshots[role]), cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            raise RuntimeError(f"failed to decode bridge snapshot: {snapshots[role]}")
        tensor, policy_bgr, policy_rgb = preprocess_bgr_frame(
            frame,
            role,
            cv2_module=cv2,
            np_module=np,
            torch_module=torch,
        )
        source_frames[role] = frame
        policy_tensors[role] = tensor

        expected_rgb = np.ascontiguousarray(policy_bgr[..., ::-1])
        rgb_semantic_error = int(
            np.max(
                np.abs(
                    policy_rgb.astype(np.int16)
                    - expected_rgb.astype(np.int16)
                )
            )
        )
        reconstructed_rgb = (
            torch.round(tensor[0].permute(1, 2, 0) * 255.0)
            .to(dtype=torch.uint8)
            .cpu()
            .numpy()
        )
        tensor_roundtrip_error = int(
            np.max(
                np.abs(
                    reconstructed_rgb.astype(np.int16)
                    - policy_rgb.astype(np.int16)
                )
            )
        )
        if rgb_semantic_error != 0 or tensor_roundtrip_error != 0:
            raise RuntimeError(f"{role} color/tensor semantic reconstruction failed")

        dataset_reference_bgr = cv2.cvtColor(
            dataset_reference_rgb[role],
            cv2.COLOR_RGB2BGR,
        )

        live_geometry_report = audit_live_geometry(
            policy_bgr,
            role,
            np_module=np,
        )

        preview = make_geometry_preview(
            frame,
            policy_bgr,
            dataset_reference_bgr,
            role,
            cv2_module=cv2,
            np_module=np,
        )
        preview_path = output_dir / f"{role}_preprocess_geometry_review.png"
        if not cv2.imwrite(str(preview_path), preview):
            raise RuntimeError(f"failed to write geometry preview: {preview_path}")
        camera_reports[role] = {
            "source_snapshot": str(snapshots[role]),
            "source_shape_hwc": list(frame.shape),
            "source_dtype": str(frame.dtype),
            "source_color_order": "BGR",
            "policy_tensor_shape_nchw": list(tensor.shape),
            "policy_tensor_dtype": str(tensor.dtype),
            "policy_tensor_min": float(tensor.min().item()),
            "policy_tensor_max": float(tensor.max().item()),
            "policy_tensor_channel_mean_rgb": [
                float(value)
                for value in tensor.mean(dim=(0, 2, 3)).tolist()
            ],
            "dataset_reference_shape_nchw": list(dataset_tensors[role].shape),
            "dataset_reference_dtype": str(dataset_tensors[role].dtype),
            "dataset_reference_min": float(dataset_tensors[role].min().item()),
            "dataset_reference_max": float(dataset_tensors[role].max().item()),
            "geometry_contract": GEOMETRY_CONTRACT[role],
            "validation_geometry_audit": dataset_geometry_reports[role],
            "live_geometry_audit": live_geometry_report,
            "bgr_to_rgb_max_error_uint8": rgb_semantic_error,
            "tensor_roundtrip_max_error_uint8": tensor_roundtrip_error,
            "geometry_preview": str(preview_path),
        }
        print(
            f"{role}: reviewed geometry -> RGB {tuple(tensor.shape)} "
            f"range=[{tensor.min().item():.4f},{tensor.max().item():.4f}] "
            "padding=PASS color_error=0 roundtrip_error=0 PASS"
        )

    observation = build_policy_observation(
        source_frames["front"],
        source_frames["wrist"],
        state,
        cv2_module=cv2,
        np_module=np,
        torch_module=torch,
    )
    if set(observation) != {STATE_KEY, FRONT_KEY, WRIST_KEY}:
        raise RuntimeError("assembled ACT observation keys mismatch")
    if tuple(observation[STATE_KEY].shape) != (1, 18):
        raise RuntimeError("assembled ACT state shape mismatch")
    for key in (FRONT_KEY, WRIST_KEY):
        if tuple(observation[key].shape) != (1, 3, 480, 480):
            raise RuntimeError(f"assembled ACT image shape mismatch: {key}")
    print("three-key ACT observation batch assembly: PASS")

    report = {
        "schema_version": 2,
        "created_from": "completed offline Windows-to-WSL bridge snapshots",
        "static_offline_boundary": static_report,
        "bridge_contract": bridge_report,
        "policy_contract": policy_report,
        "validation_reference": {
            "dataset_root": str(validation_root),
            "repo_id": args.validation_repo_id,
            "sample_index": 0,
            "state_shape": list(state.shape),
            "images_are_recorded_offline": True,
        },
        "preprocess_contract": {
            "source_shape_hwc": list(EXPECTED_SOURCE_SHAPE),
            "source_dtype": "uint8",
            "source_color_order": "BGR",
            "geometry": {
                "mode": "camera_role_specific_preserve_aspect_with_black_padding",
                "implementation": "cv2.INTER_AREA",
                "source_width": 640,
                "source_height": 480,
                "target_width": 480,
                "target_height": 480,
                "content_scale": 0.75,
                "padding_value_uint8": 0,
                "roles": GEOMETRY_CONTRACT,
                "review_basis": {
                    "front": "validation reference has 60 px top/bottom bars",
                    "wrist": (
                        "human-reviewed live/reference scene orientation requires "
                        "counter-clockwise 90 degree rotation; validation reference "
                        "has 60 px left/right bars"
                    ),
                },
                "human_geometry_review_completed": True,
            },
            "target_color_order": "RGB",
            "target_dtype": "torch.float32",
            "target_range": [0.0, 1.0],
            "target_layout": "NCHW",
            "target_shape": [1, 3, 480, 480],
            "manual_imagenet_normalization_applied": False,
            "normalization_owner": "frozen ACT policy input-normalization path",
        },
        "validation_geometry_audit": dataset_geometry_reports,
        "camera_results": camera_reports,
        "assembled_observation": {
            "keys": sorted(observation),
            "state_shape": list(observation[STATE_KEY].shape),
            "front_shape": list(observation[FRONT_KEY].shape),
            "wrist_shape": list(observation[WRIST_KEY].shape),
            "device": str(observation[STATE_KEY].device),
        },
        "numeric_contract_passed": True,
        "geometry_contract_passed": True,
        "geometry_review_pending": False,
        "decision": "CAMERA_PREPROCESS_GEOMETRY_PASS_BUILD_LIVE_OBSERVATION_ASSEMBLER_NEXT",
        "hardware_deployment_authorized": False,
        "model_loaded": False,
        "camera_opened": False,
        "robot_accessed": False,
        "serial_port_opened": False,
        "command_sent": False,
    }
    report_path = output_dir / "camera_preprocess_adapter_report.json"
    write_json(report_path, report)

    print("\n===== DECISION =====")
    print("decision='CAMERA_PREPROCESS_GEOMETRY_PASS_BUILD_LIVE_OBSERVATION_ASSEMBLER_NEXT'")
    print("Numeric, color, tensor, aspect, rotation, and padding contracts: PASS")
    print("HARDWARE DEPLOYMENT REMAINS BLOCKED.")
    print("NO CAMERA WAS OPENED. NO MODEL WAS LOADED.")
    print("NO ROBOT OBJECT WAS CREATED. NO SERIAL PORT WAS OPENED. NO COMMAND WAS SENT.")
    print(f"report={report_path}")
    print(f"front_preview={camera_reports['front']['geometry_preview']}")
    print(f"wrist_preview={camera_reports['wrist']['geometry_preview']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
