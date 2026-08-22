#!/usr/bin/env python3
"""Smoke-test the frozen ACT V3 delta runtime sequence without hardware.

This is an integration adapter, not a robot deployment script. It connects:

    recorded validation observation
      -> runtime 18-D state construction
      -> frozen ACT 11K chunk inference
      -> explicit 5-command queue
      -> shared DeltaCommandGuard
      -> CSV-only command sink

The adapter never imports a LeRobot robot or teleoperator module, never creates
a robot object, never opens a serial port or live camera, and never sends a
command. Recorded images and actual_q are used only as an offline observation
source. The command state is recursively maintained from the commands that the
guard would have sent.

For wrist_flex, an outward residual while already at the +60 task endpoint is
classified as BOUNDARY_HOLD. It is expected saturation for this pick task, not
a failure. A persistent live tracking error cannot be validated by this hybrid
offline replay; the watchdog is exercised only as an integration diagnostic.
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
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


MOTORS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
STATE_KEY = "observation.state"
ACTION_KEY = "action"
FRONT_KEY = "observation.images.front"
WRIST_KEY = "observation.images.wrist"

EXPECTED_GUARD_CORE_SHA256 = (
    "e06e785071ee3ded95e89d670978bcb4a8caf104b2fc69fc87de37cb6fff6d15"
)
EXPECTED_STEP = 11000
EXPECTED_CHUNK_SIZE = 10
EXPECTED_QUEUE_LENGTH = 5
EXPECTED_VALIDATION_FRAMES = 10704
EXPECTED_VALIDATION_EPISODES = 12
EXPECTED_FPS = 15.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guard-core", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--source-checkpoint-root", type=Path, required=True)
    parser.add_argument("--validation-dataset-root", type=Path, required=True)
    parser.add_argument("--validation-repo-id", required=True)
    parser.add_argument("--video-backend", default="torchcodec")
    parser.add_argument(
        "--episode-indices",
        type=int,
        nargs="+",
        required=True,
        help="Validation-dataset episode indices; use 0 11 for the smoke run.",
    )
    parser.add_argument(
        "--max-replans-per-episode",
        type=int,
        default=5,
        help="Smoke cap. Each replan writes exactly five guarded commands.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    if not args.episode_indices:
        parser.error("--episode-indices cannot be empty")
    if len(set(args.episode_indices)) != len(args.episode_indices):
        parser.error("--episode-indices contains duplicates")
    if not 1 <= args.max_replans_per_episode <= 20:
        parser.error("--max-replans-per-episode must be 1..20 for this smoke adapter")
    return args


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


def verify_no_hardware_code(path: Path) -> dict[str, object]:
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
    forbidden_names = {
        "SO101Follower",
        "SO101FollowerConfig",
        "SO101Leader",
        "SO101LeaderConfig",
    }
    forbidden_attributes = {
        "send_action",
        "sync_write",
        "write",
        "enable_torque",
        "disable_torque",
        "connect",
    }
    present_names = sorted(forbidden_names.intersection(identifiers))
    present_attributes = sorted(forbidden_attributes.intersection(attributes))
    if forbidden_modules or present_names or present_attributes:
        raise RuntimeError(
            "hardware-capable code found in offline adapter: "
            f"modules={forbidden_modules}, names={present_names}, "
            f"attributes={present_attributes}"
        )

    return {
        "source_path": str(path),
        "source_sha256": sha256_file(path),
        "imported_modules": sorted(imported_modules),
        "hardware_capable_imports_or_calls": False,
    }


def verify_release(candidate_root: Path, source_checkpoint_root: Path) -> dict[str, object]:
    manifest = require_file(candidate_root / "SHA256SUMS")
    manifest_rows: list[dict[str, str]] = []
    for line_number, raw_line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise RuntimeError(f"invalid SHA256SUMS line {line_number}")
        expected, relative_text = parts
        relative_text = relative_text.lstrip("* ")
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe SHA256SUMS path: {relative_text}")
        target = require_file(candidate_root / relative)
        actual = sha256_file(target)
        if actual != expected:
            raise RuntimeError(f"release manifest mismatch: {relative_text}")
        manifest_rows.append(
            {"path": relative.as_posix(), "sha256": actual}
        )

    required_release_files = {
        "pretrained_model/config.json",
        "pretrained_model/model.safetensors",
        "pretrained_model/train_config.json",
        "best_checkpoint.json",
        "checkpoint_metrics.csv",
        "validation_selection_report.json",
    }
    present = {row["path"] for row in manifest_rows}
    missing = sorted(required_release_files - present)
    if missing:
        raise RuntimeError(f"release manifest is missing files: {missing}")

    selection_path = require_file(candidate_root / "best_checkpoint.json")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if int(selection.get("best_step", -1)) != EXPECTED_STEP:
        raise RuntimeError("release best_step is not the frozen 11000 checkpoint")
    if selection.get("hardware_deployment_authorized") is not False:
        raise RuntimeError("release must keep hardware deployment authorization false")

    identity: dict[str, str] = {}
    for filename in ("config.json", "model.safetensors", "train_config.json"):
        candidate_file = require_file(candidate_root / "pretrained_model" / filename)
        source_file = require_file(source_checkpoint_root / filename)
        candidate_sha = sha256_file(candidate_file)
        source_sha = sha256_file(source_file)
        if candidate_sha != source_sha:
            raise RuntimeError(f"candidate/source checkpoint mismatch: {filename}")
        identity[filename] = candidate_sha

    return {
        "candidate_root": str(candidate_root),
        "source_checkpoint_root": str(source_checkpoint_root),
        "selected_step": EXPECTED_STEP,
        "manifest_entries": manifest_rows,
        "source_checkpoint_identity": identity,
        "hardware_deployment_authorized": False,
    }


def load_guard_core(path: Path):
    actual_sha = sha256_file(path)
    if actual_sha != EXPECTED_GUARD_CORE_SHA256:
        raise RuntimeError(
            "guard core SHA256 mismatch: "
            f"expected {EXPECTED_GUARD_CORE_SHA256}, got {actual_sha}"
        )
    spec = importlib.util.spec_from_file_location("act_v3_delta_runtime_guard", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load guard core: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_runtime_dependencies():
    import numpy as np
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy

    return np, torch, LeRobotDataset, ACTPolicy


def load_dataset(LeRobotDataset, repo_id: str, root: Path, video_backend: str):
    try:
        return LeRobotDataset(
            repo_id=repo_id,
            root=root,
            video_backend=video_backend,
        )
    except TypeError:
        return LeRobotDataset(repo_id, root=root, video_backend=video_backend)


def feature_shape(feature: Any) -> tuple[int, ...]:
    if isinstance(feature, dict):
        value = feature.get("shape")
    else:
        value = getattr(feature, "shape", None)
    if value is None:
        raise RuntimeError(f"policy feature has no shape: {feature!r}")
    return tuple(int(item) for item in value)


def verify_policy_contract(policy: Any) -> dict[str, object]:
    config = policy.config
    if int(config.chunk_size) != EXPECTED_CHUNK_SIZE:
        raise RuntimeError(f"chunk_size={config.chunk_size}, expected 10")
    if int(config.n_action_steps) != EXPECTED_QUEUE_LENGTH:
        raise RuntimeError(f"n_action_steps={config.n_action_steps}, expected 5")
    if bool(config.use_vae):
        raise RuntimeError("frozen V3 policy must use_vae=false")
    if bool(config.use_amp):
        raise RuntimeError("frozen V3 policy must use_amp=false")

    expected_inputs = {STATE_KEY, FRONT_KEY, WRIST_KEY}
    input_features = config.input_features
    output_features = config.output_features
    if set(input_features) != expected_inputs:
        raise RuntimeError(f"unexpected policy inputs: {sorted(input_features)}")
    if set(output_features) != {ACTION_KEY}:
        raise RuntimeError(f"unexpected policy outputs: {sorted(output_features)}")
    if feature_shape(input_features[STATE_KEY]) != (18,):
        raise RuntimeError("policy state feature is not 18-D")
    if feature_shape(input_features[FRONT_KEY]) != (3, 480, 480):
        raise RuntimeError("front image feature shape mismatch")
    if feature_shape(input_features[WRIST_KEY]) != (3, 480, 480):
        raise RuntimeError("wrist image feature shape mismatch")
    if feature_shape(output_features[ACTION_KEY]) != (6,):
        raise RuntimeError("policy action feature is not 6-D")

    return {
        "device": str(config.device),
        "chunk_size": int(config.chunk_size),
        "n_action_steps": int(config.n_action_steps),
        "use_vae": bool(config.use_vae),
        "use_amp": bool(config.use_amp),
        "input_shapes": {
            key: list(feature_shape(value)) for key, value in input_features.items()
        },
        "output_shapes": {
            key: list(feature_shape(value)) for key, value in output_features.items()
        },
    }


def image_tensor(value: Any, torch: Any, device: Any):
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    tensor = value.detach()
    if tensor.ndim != 3:
        raise RuntimeError(f"unexpected image shape: {tuple(tensor.shape)}")
    if tensor.shape[0] in (1, 3, 4):
        tensor = tensor[:3]
    elif tensor.shape[-1] in (1, 3, 4):
        tensor = tensor[..., :3].permute(2, 0, 1)
    else:
        raise RuntimeError(f"cannot infer image channel axis: {tuple(tensor.shape)}")
    tensor = tensor.float()
    if float(tensor.max()) > 1.5:
        tensor = tensor / 255.0
    return tensor.unsqueeze(0).to(device=device)


def infer_chunk(policy: Any, item: dict[str, Any], state: Sequence[float], np: Any, torch: Any):
    device = torch.device(policy.config.device)
    state_tensor = torch.as_tensor(state, dtype=torch.float32).reshape(1, 18).to(device)
    batch = {
        STATE_KEY: state_tensor,
        FRONT_KEY: image_tensor(item[FRONT_KEY], torch, device),
        WRIST_KEY: image_tensor(item[WRIST_KEY], torch, device),
    }
    policy.reset()
    with torch.no_grad():
        chunk = policy.predict_action_chunk(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    array = chunk.detach().cpu().float().numpy()
    if array.shape != (1, EXPECTED_CHUNK_SIZE, 6):
        raise RuntimeError(f"unexpected ACT chunk shape: {array.shape}")
    if not np.isfinite(array).all():
        raise FloatingPointError("ACT output contains NaN/Inf")
    return array[0]


def dataset_fps(dataset: Any) -> float:
    candidates = [
        getattr(dataset, "fps", None),
        getattr(getattr(dataset, "meta", None), "fps", None),
        getattr(getattr(dataset, "meta", None), "info", {}).get("fps")
        if isinstance(getattr(getattr(dataset, "meta", None), "info", None), dict)
        else None,
    ]
    for candidate in candidates:
        if candidate is not None:
            return float(candidate)
    raise RuntimeError("cannot determine validation dataset fps")


def build_frame_lookup(episodes: Any, frames: Any) -> dict[tuple[int, int], int]:
    lookup: dict[tuple[int, int], int] = {}
    for global_index, (episode, frame) in enumerate(zip(episodes, frames, strict=True)):
        key = (int(episode), int(frame))
        if key in lookup:
            raise RuntimeError(f"duplicate dataset episode/frame key: {key}")
        lookup[key] = global_index
    return lookup


def wrist_runtime_mode(step: Any, limits: Any) -> str:
    index = MOTORS.index("wrist_flex")
    _, upper = limits.soft_limits[index]
    epsilon = 1e-9
    if step.inherited_boundary_projection[index]:
        return "BOUNDARY_HOLD"
    if step.new_boundary_crossing[index] and step.sent_command[index] >= upper - epsilon:
        return "APPROACH_TASK_ENDPOINT"
    if step.previous_command[index] >= upper - epsilon and step.sent_delta[index] < -epsilon:
        return "LEAVE_TASK_ENDPOINT"
    if step.sent_command[index] >= upper - epsilon:
        return "AT_TASK_ENDPOINT"
    return "INTERIOR"


def flatten_replan_row(
    base: dict[str, object],
    actual: Sequence[float],
    runtime_previous: Sequence[float],
    recorded_previous: Sequence[float],
    runtime_delta: Sequence[float],
    recorded_delta: Sequence[float],
    tracking: Any,
) -> dict[str, object]:
    row = dict(base)
    for index, motor in enumerate(MOTORS):
        row[f"actual_q_{motor}"] = float(actual[index])
        row[f"runtime_previous_command_{motor}"] = float(runtime_previous[index])
        row[f"recorded_previous_command_{motor}"] = float(recorded_previous[index])
        row[f"runtime_previous_delta_{motor}"] = float(runtime_delta[index])
        row[f"recorded_previous_delta_{motor}"] = float(recorded_delta[index])
        row[f"tracking_error_{motor}"] = float(tracking.absolute_error[index])
        row[f"tracking_over_seconds_{motor}"] = float(
            tracking.over_limit_seconds[index]
        )
    return row


def flatten_command_row(
    base: dict[str, object],
    step: Any,
    recorded_delta: Sequence[float] | None,
) -> dict[str, object]:
    row = dict(base)
    for index, motor in enumerate(MOTORS):
        row[f"raw_delta_{motor}"] = float(step.raw_delta[index])
        row[f"rate_limited_delta_{motor}"] = float(
            step.rate_limited_delta[index]
        )
        row[f"previous_command_{motor}"] = float(step.previous_command[index])
        row[f"candidate_command_{motor}"] = float(
            step.candidate_after_rate_limit[index]
        )
        row[f"sent_command_{motor}"] = float(step.sent_command[index])
        row[f"sent_delta_{motor}"] = float(step.sent_delta[index])
        row[f"rate_clipped_{motor}"] = int(step.rate_clipped[index])
        row[f"soft_clipped_{motor}"] = int(step.soft_clipped[index])
        row[f"boundary_hold_{motor}"] = int(
            step.inherited_boundary_projection[index]
        )
        row[f"new_boundary_crossing_{motor}"] = int(
            step.new_boundary_crossing[index]
        )
        row[f"recorded_delta_{motor}"] = (
            float(recorded_delta[index]) if recorded_delta is not None else math.nan
        )
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fieldnames = list(rows[0])
    for row in rows:
        if list(row) != fieldnames:
            raise RuntimeError("CSV row schemas are inconsistent")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    script_path = Path(__file__).resolve()
    guard_path = require_file(args.guard_core)
    contract_path = require_file(args.contract)
    calibration_path = require_file(args.calibration)
    candidate_root = require_dir(args.candidate_root)
    source_checkpoint_root = require_dir(args.source_checkpoint_root)
    dataset_root = require_dir(args.validation_dataset_root)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"output directory already exists; preserve it and choose a new path: {output_dir}"
        )

    print("===== STATIC NO-HARDWARE BOUNDARY =====")
    static_report = verify_no_hardware_code(script_path)
    print("no robot/teleoperator/serial imports or command-write calls: PASS")

    print("\n===== VERIFY SHARED GUARD AND FROZEN INPUTS =====")
    guard_module = load_guard_core(guard_path)
    frozen_inputs = guard_module.verify_frozen_inputs(
        contract_path, calibration_path
    )
    limits = guard_module.RuntimeLimits.frozen_v3()
    guard_audit = guard_module.run_algorithm_self_audit(limits)
    if guard_audit["tests_passed"] != guard_audit["tests_total"]:
        raise RuntimeError("shared guard core self-audit did not pass")
    print(
        f"guard byte identity, contract, calibration and algorithms: "
        f"{guard_audit['tests_passed']}/{guard_audit['tests_total']} PASS"
    )

    print("\n===== VERIFY FROZEN 11K RELEASE =====")
    release_report = verify_release(candidate_root, source_checkpoint_root)
    print("release manifest, selection and source-checkpoint identity: PASS")

    print("\n===== LOAD OFFLINE VALIDATION SOURCE =====")
    np, torch, LeRobotDataset, ACTPolicy = load_runtime_dependencies()
    dataset = load_dataset(
        LeRobotDataset,
        args.validation_repo_id,
        dataset_root,
        args.video_backend,
    )
    hf = dataset.hf_dataset
    states = np.asarray(hf[STATE_KEY], dtype=np.float32)
    actions = np.asarray(hf[ACTION_KEY], dtype=np.float32)
    episodes = np.asarray(hf["episode_index"], dtype=np.int64)
    frames = np.asarray(hf["frame_index"], dtype=np.int64)
    unique_episodes = sorted(int(value) for value in np.unique(episodes))
    fps = dataset_fps(dataset)

    if states.shape != (EXPECTED_VALIDATION_FRAMES, 18):
        raise RuntimeError(f"unexpected validation state shape: {states.shape}")
    if actions.shape != (EXPECTED_VALIDATION_FRAMES, 6):
        raise RuntimeError(f"unexpected validation action shape: {actions.shape}")
    if len(states) != EXPECTED_VALIDATION_FRAMES:
        raise RuntimeError("unexpected validation frame count")
    if len(unique_episodes) != EXPECTED_VALIDATION_EPISODES:
        raise RuntimeError("unexpected validation episode count")
    if abs(fps - EXPECTED_FPS) > 1e-9:
        raise RuntimeError(f"validation fps={fps}, expected 15")
    invalid_episodes = sorted(set(args.episode_indices) - set(unique_episodes))
    if invalid_episodes:
        raise RuntimeError(f"unknown validation episodes: {invalid_episodes}")
    lookup = build_frame_lookup(episodes, frames)
    print(
        f"validation: frames={len(states)} episodes={len(unique_episodes)} "
        f"fps={fps:g} PASS"
    )
    print(f"selected episodes: {args.episode_indices}")
    print(f"smoke replans per episode: {args.max_replans_per_episode}")

    print("\n===== LOAD FROZEN POLICY (NO ROBOT OBJECT) =====")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; offline smoke inference stopped")
    policy = ACTPolicy.from_pretrained(
        candidate_root / "pretrained_model",
        local_files_only=True,
    )
    policy.eval()
    policy.reset()
    policy_report = verify_policy_contract(policy)
    print("policy state=18D action=6D chunk=10 queue=5 vae=false amp=false: PASS")

    command_rows: list[dict[str, object]] = []
    replan_rows: list[dict[str, object]] = []
    event_counts: Counter[str] = Counter()
    rate_counts = Counter({motor: 0 for motor in MOTORS})
    soft_counts = Counter({motor: 0 for motor in MOTORS})
    hold_counts = Counter({motor: 0 for motor in MOTORS})
    crossing_counts = Counter({motor: 0 for motor in MOTORS})
    commands_any_rate = 0
    commands_any_soft = 0
    tracking_trip_replans = 0
    max_guard_invariant_error = 0.0
    max_runtime_recorded_command_gap = 0.0
    max_runtime_recorded_delta_gap = 0.0

    print("\n===== OFFLINE RUNTIME-SEQUENCE SMOKE =====")
    for episode in args.episode_indices:
        first_key = (episode, 0)
        if first_key not in lookup:
            raise RuntimeError(f"episode {episode}: frame 0 missing")
        first_state = states[lookup[first_key]]
        if float(np.max(np.abs(first_state[6:12] - np.asarray(limits.home_command)))) > 1e-5:
            raise RuntimeError(f"episode {episode}: initial previous_command is not Home")
        if float(np.max(np.abs(first_state[12:18]))) > 1e-6:
            raise RuntimeError(f"episode {episode}: initial previous_delta is not zero")

        runtime_guard = guard_module.DeltaCommandGuard(
            limits,
            first_state[6:12],
            first_state[12:18],
        )
        tracking_watchdog = guard_module.TrackingWatchdog(limits)
        episode_commands = 0

        for replan_index in range(args.max_replans_per_episode):
            anchor_frame = replan_index * EXPECTED_QUEUE_LENGTH
            key = (episode, anchor_frame)
            if key not in lookup:
                raise RuntimeError(
                    f"episode {episode}: anchor frame {anchor_frame} missing"
                )
            global_index = lookup[key]
            recorded_state = states[global_index]
            actual_q = recorded_state[:6]
            runtime_previous = runtime_guard.previous_command
            runtime_previous_delta = runtime_guard.previous_delta

            policy_state = runtime_guard.build_policy_state(actual_q)
            if tuple(policy_state[6:12]) != tuple(runtime_previous):
                raise RuntimeError("runtime previous-command state invariant failed")
            if tuple(policy_state[12:18]) != tuple(runtime_previous_delta):
                raise RuntimeError("runtime previous-delta state invariant failed")

            command_gap = float(
                np.max(
                    np.abs(
                        np.asarray(runtime_previous, dtype=np.float64)
                        - recorded_state[6:12].astype(np.float64)
                    )
                )
            )
            delta_gap = float(
                np.max(
                    np.abs(
                        np.asarray(runtime_previous_delta, dtype=np.float64)
                        - recorded_state[12:18].astype(np.float64)
                    )
                )
            )
            max_runtime_recorded_command_gap = max(
                max_runtime_recorded_command_gap, command_gap
            )
            max_runtime_recorded_delta_gap = max(
                max_runtime_recorded_delta_gap, delta_gap
            )

            tracking = tracking_watchdog.update(
                actual_q,
                runtime_previous,
                anchor_frame / fps,
            )
            if tracking.tripped:
                tracking_trip_replans += 1

            replan_rows.append(
                flatten_replan_row(
                    {
                        "episode_index": episode,
                        "replan_index": replan_index,
                        "anchor_frame": anchor_frame,
                        "global_index": global_index,
                        "sim_time_seconds": anchor_frame / fps,
                        "runtime_recorded_command_gap_max": command_gap,
                        "runtime_recorded_delta_gap_max": delta_gap,
                        "tracking_tripped": int(tracking.tripped),
                        "tracking_tripped_joints": ";".join(
                            tracking.tripped_joints
                        ),
                    },
                    actual_q,
                    runtime_previous,
                    recorded_state[6:12],
                    runtime_previous_delta,
                    recorded_state[12:18],
                    tracking,
                )
            )

            item = dataset[int(global_index)]
            chunk = infer_chunk(policy, item, policy_state, np, torch)

            for substep in range(EXPECTED_QUEUE_LENGTH):
                raw_delta = chunk[substep]
                step = runtime_guard.apply_delta(raw_delta)
                reconstructed = np.asarray(step.previous_command) + np.asarray(
                    step.sent_delta
                )
                invariant_error = float(
                    np.max(np.abs(reconstructed - np.asarray(step.sent_command)))
                )
                max_guard_invariant_error = max(
                    max_guard_invariant_error, invariant_error
                )
                if invariant_error > 1e-9:
                    raise RuntimeError("guard recursive command invariant failed")

                demo_key = (episode, anchor_frame + substep)
                recorded_delta = (
                    actions[lookup[demo_key]] if demo_key in lookup else None
                )
                wrist_mode = wrist_runtime_mode(step, limits)
                event_counts[wrist_mode] += 1
                commands_any_rate += int(step.any_rate_clip)
                commands_any_soft += int(step.any_soft_clip)
                for index, motor in enumerate(MOTORS):
                    rate_counts[motor] += int(step.rate_clipped[index])
                    soft_counts[motor] += int(step.soft_clipped[index])
                    hold_counts[motor] += int(
                        step.inherited_boundary_projection[index]
                    )
                    crossing_counts[motor] += int(step.new_boundary_crossing[index])

                command_rows.append(
                    flatten_command_row(
                        {
                            "episode_index": episode,
                            "replan_index": replan_index,
                            "anchor_frame": anchor_frame,
                            "substep_index": substep,
                            "sim_command_frame": anchor_frame + substep,
                            "wrist_flex_runtime_mode": wrist_mode,
                            "any_rate_clip": int(step.any_rate_clip),
                            "any_soft_clip": int(step.any_soft_clip),
                            "guard_invariant_error": invariant_error,
                        },
                        step,
                        recorded_delta,
                    )
                )
                episode_commands += 1

        print(
            f"episode {episode:02d}: replans={args.max_replans_per_episode} "
            f"commands={episode_commands} PASS"
        )

    total_replans = len(replan_rows)
    total_commands = len(command_rows)
    expected_replans = len(args.episode_indices) * args.max_replans_per_episode
    expected_commands = expected_replans * EXPECTED_QUEUE_LENGTH
    if total_replans != expected_replans or total_commands != expected_commands:
        raise RuntimeError("offline adapter replan/command count invariant failed")

    decision = "OFFLINE_ADAPTER_SMOKE_PASS_STATIC_HARDWARE_ADAPTER_NEXT"
    if tracking_trip_replans:
        decision = "OFFLINE_ADAPTER_SMOKE_REVIEW_TRACKING_DIAGNOSTIC"

    report = {
        "schema_version": "act_v3_delta_offline_runtime_adapter_smoke_v1",
        "scope": {
            "kind": "offline_smoke",
            "recorded_observation_source": True,
            "recursive_guarded_command_state": True,
            "hardware_access_authorized": False,
            "robot_object_created": False,
            "serial_port_opened": False,
            "live_camera_opened": False,
            "command_sent": False,
        },
        "static_no_hardware_boundary": static_report,
        "guard_core": {
            "path": str(guard_path),
            "sha256": sha256_file(guard_path),
            "frozen_inputs": frozen_inputs,
            "self_audit": guard_audit,
        },
        "release": release_report,
        "validation_dataset": {
            "root": str(dataset_root),
            "repo_id": args.validation_repo_id,
            "frames": len(states),
            "episodes": len(unique_episodes),
            "fps": fps,
            "selected_episodes": args.episode_indices,
        },
        "policy": policy_report,
        "queue_contract": {
            "chunk_size": EXPECTED_CHUNK_SIZE,
            "commands_per_replan": EXPECTED_QUEUE_LENGTH,
            "replans": total_replans,
            "commands": total_commands,
        },
        "diagnostics": {
            "commands_with_any_rate_clip": commands_any_rate,
            "commands_with_any_soft_clip": commands_any_soft,
            "per_joint_rate_clips": dict(rate_counts),
            "per_joint_soft_clips": dict(soft_counts),
            "per_joint_boundary_holds": dict(hold_counts),
            "per_joint_new_boundary_crossings": dict(crossing_counts),
            "wrist_flex_runtime_modes": dict(event_counts),
            "tracking_trip_replans": tracking_trip_replans,
            "max_guard_invariant_error": max_guard_invariant_error,
            "max_runtime_recorded_command_gap": max_runtime_recorded_command_gap,
            "max_runtime_recorded_delta_gap": max_runtime_recorded_delta_gap,
            "tracking_is_hybrid_replay_diagnostic_not_live_prediction": True,
            "boundary_hold_is_expected_task_saturation_not_failure": True,
        },
        "remaining_untested_live_integrations": [
            "live front/wrist camera pairing and freshness",
            "live Present_Position readback and timestamp freshness",
            "connect without calibration mutation",
            "torque lifecycle and emergency stop",
            "physical collision clearance, payload, and cable routing",
        ],
        "decision": decision,
        "hardware_deployment_authorized": False,
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "offline_runtime_adapter_report.json"
    commands_path = output_dir / "offline_runtime_adapter_commands.csv"
    replans_path = output_dir / "offline_runtime_adapter_replans.csv"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(commands_path, command_rows)
    write_csv(replans_path, replan_rows)

    print("\n===== SMOKE METRICS =====")
    print(f"replans: {total_replans}")
    print(f"commands: {total_commands}")
    print(f"commands with any rate clip: {commands_any_rate}/{total_commands}")
    print(f"commands with any soft clip: {commands_any_soft}/{total_commands}")
    print(f"wrist_flex runtime modes: {dict(event_counts)}")
    print(f"tracking-trip replans: {tracking_trip_replans}/{total_replans}")
    print(f"max guard invariant error: {max_guard_invariant_error:.10f}")
    print(
        "Boundary hold is expected task saturation and is not a failure condition."
    )

    print("\n===== DECISION =====")
    print(f"decision='{decision}'")
    print("HARDWARE DEPLOYMENT REMAINS BLOCKED.")
    print("NO ROBOT OBJECT WAS CREATED. NO SERIAL PORT WAS OPENED.")
    print("NO LIVE CAMERA WAS OPENED. NO COMMAND WAS SENT.")

    print("\n===== OUTPUT =====")
    print(report_path)
    print(commands_path)
    print(replans_path)
    print("ACT V3 DELTA OFFLINE RUNTIME ADAPTER SMOKE: PASS")

    del policy
    del dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0 if tracking_trip_replans == 0 else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print("ACT V3 DELTA OFFLINE RUNTIME ADAPTER SMOKE: FAIL", file=sys.stderr)
        print("HARDWARE DEPLOYMENT REMAINS BLOCKED.", file=sys.stderr)
        print("NO ROBOT OBJECT WAS CREATED. NO SERIAL PORT WAS OPENED.", file=sys.stderr)
        print("NO LIVE CAMERA WAS OPENED. NO COMMAND WAS SENT.", file=sys.stderr)
        raise
