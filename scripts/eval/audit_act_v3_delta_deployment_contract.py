#!/usr/bin/env python
"""Audit the frozen ACT v3 delta candidate before any robot-side work.

This program is deliberately offline-only. It loads local LeRobot datasets,
the frozen policy release, and recorded camera frames. It never imports a
robot, motor, serial-port, calibration, teleoperation, or live-camera module.

The audit answers one narrow question: is the frozen policy package and its
delta-action interface internally consistent enough to design a separate,
guarded runtime adapter? A PASS from this script is *not* permission to move
hardware.

Runtime contract checked here
-----------------------------
observation.state = [
    actual_q_t[6],
    last_sent_command_t[6],
    last_sent_delta_t[6],
]

policy output = command_delta[10, 6]

For the first five queued actions before replanning:
    next_command = last_sent_command + command_delta
    last_sent_command = next_command
    last_sent_delta = command_delta

The five commands therefore equal:
    base_command + cumsum(predicted_delta[:5], axis=0)

Dataset-derived ranges in the report are diagnostics only. They are not
manufacturer limits and are not a physical safety certification.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy


STATE_KEY = "observation.state"
ACTION_KEY = "action"
FRONT_KEY = "observation.images.front"
WRIST_KEY = "observation.images.wrist"

MOTORS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

EXPECTED_TRAIN_FRAMES = 42678
EXPECTED_TRAIN_EPISODES = 48
EXPECTED_VALIDATION_FRAMES = 10704
EXPECTED_VALIDATION_EPISODES = 12
EXPECTED_STATE_DIM = 18
EXPECTED_ACTION_DIM = 6
EXPECTED_STEP = 11000
EXPECTED_FPS = 15
CHUNK_SIZE = 10
EXECUTION_HORIZON = 5

REQUIRED_RELEASE_FILES = (
    "pretrained_model/config.json",
    "pretrained_model/model.safetensors",
    "pretrained_model/train_config.json",
    "best_checkpoint.json",
    "checkpoint_metrics.csv",
    "validation_selection_report.json",
)


@dataclass(frozen=True)
class Anchor:
    episode_index: int
    frame_index: int
    global_index: int
    phases: tuple[str, ...]
    activity_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=Path("outputs/release/act_red_cube_v3_delta_011000"),
    )
    parser.add_argument(
        "--source-checkpoint-root",
        type=Path,
        default=Path(
            "outputs/train/act_red_cube_v3_delta_stage1/"
            "checkpoints/011000/pretrained_model"
        ),
    )
    parser.add_argument(
        "--train-dataset-root",
        type=Path,
        default=Path(
            "data/lerobot/so101_red_cube_pick_place_v3_delta_train"
        ),
    )
    parser.add_argument(
        "--train-repo-id",
        default="connorluis/so101_red_cube_pick_place_v3_delta_train",
    )
    parser.add_argument(
        "--validation-dataset-root",
        type=Path,
        default=Path(
            "data/lerobot/so101_red_cube_pick_place_v3_delta_validation"
        ),
    )
    parser.add_argument(
        "--validation-repo-id",
        default="connorluis/so101_red_cube_pick_place_v3_delta_validation",
    )
    parser.add_argument(
        "--v3-plan",
        type=Path,
        default=Path(
            "outputs/dataset/so101_red_cube_pick_place_v3_delta_plan/"
            "dataset_v3_delta_plan.json"
        ),
    )
    parser.add_argument("--video-backend", default="torchcodec")
    parser.add_argument("--uniform-samples-per-episode", type=int, default=12)
    parser.add_argument("--moving-samples-per-episode", type=int, default=4)
    parser.add_argument("--stationary-samples-per-episode", type=int, default=4)
    parser.add_argument("--stress-min-separation", type=int, default=20)
    parser.add_argument("--queue-check-anchors", type=int, default=12)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/eval/act_red_cube_v3_delta_deployment_contract"
        ),
    )
    args = parser.parse_args()

    for name in (
        "uniform_samples_per_episode",
        "moving_samples_per_episode",
        "stationary_samples_per_episode",
        "stress_min_separation",
        "queue_check_anchors",
    ):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256_manifest(candidate_root: Path) -> dict[str, str]:
    manifest_path = candidate_root / "SHA256SUMS"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    entries: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise RuntimeError(
                f"Malformed SHA256SUMS line {line_number}: {raw_line!r}"
            )
        expected, relative_text = parts
        relative_text = relative_text.lstrip("*")
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"Unsafe SHA256SUMS path: {relative_text}")
        if relative_text in entries:
            raise RuntimeError(f"Duplicate SHA256SUMS entry: {relative_text}")
        path = candidate_root / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"SHA-256 mismatch for {relative_text}: "
                f"expected={expected}, actual={actual}"
            )
        entries[relative_text] = actual

    missing = sorted(set(REQUIRED_RELEASE_FILES) - set(entries))
    if missing:
        raise RuntimeError(
            "SHA256SUMS does not cover required release files: "
            + ", ".join(missing)
        )
    return entries


def verify_release(
    candidate_root: Path,
    source_checkpoint_root: Path,
) -> dict[str, Any]:
    candidate_root = candidate_root.resolve()
    source_checkpoint_root = source_checkpoint_root.resolve()
    if not candidate_root.is_dir():
        raise FileNotFoundError(candidate_root)
    if not source_checkpoint_root.is_dir():
        raise FileNotFoundError(source_checkpoint_root)

    manifest = verify_sha256_manifest(candidate_root)
    selection = load_json(candidate_root / "best_checkpoint.json")
    validation_report = load_json(
        candidate_root / "validation_selection_report.json"
    )

    if int(selection.get("best_step", -1)) != EXPECTED_STEP:
        raise RuntimeError("Frozen best_checkpoint.json does not select step 11000")
    if selection.get("decision") != "SELECT_INTERIOR_STAGE1_BEST":
        raise RuntimeError("Frozen selection is not an interior-stage best")
    if bool(selection.get("boundary_censored", True)):
        raise RuntimeError("Frozen selection is unexpectedly boundary-censored")
    if bool(selection.get("hardware_deployment_authorized", True)):
        raise RuntimeError(
            "best_checkpoint.json must keep hardware authorization false"
        )

    report_selection = validation_report.get("selection")
    if not isinstance(report_selection, dict):
        raise RuntimeError("Validation report is missing selection metadata")
    if int(report_selection.get("best_step", -1)) != EXPECTED_STEP:
        raise RuntimeError("Validation report does not select step 11000")
    if bool(report_selection.get("hardware_deployment_authorized", True)):
        raise RuntimeError("Validation report incorrectly authorizes hardware")

    source_match: dict[str, dict[str, Any]] = {}
    for filename in ("config.json", "model.safetensors", "train_config.json"):
        candidate = candidate_root / "pretrained_model" / filename
        source = source_checkpoint_root / filename
        if not candidate.is_file() or not source.is_file():
            raise FileNotFoundError(f"Missing candidate/source file: {filename}")
        candidate_hash = sha256_file(candidate)
        source_hash = sha256_file(source)
        if candidate_hash != source_hash:
            raise RuntimeError(
                f"Frozen candidate differs from source checkpoint: {filename}"
            )
        source_match[filename] = {
            "bytes": int(candidate.stat().st_size),
            "sha256": candidate_hash,
            "matches_source_checkpoint": True,
        }

    return {
        "candidate_root": str(candidate_root),
        "source_checkpoint_root": str(source_checkpoint_root),
        "manifest_entries": manifest,
        "selection": selection,
        "source_match": source_match,
    }


def load_dataset(
    repo_id: str,
    root: Path,
    video_backend: str,
) -> LeRobotDataset:
    resolved = root.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    return LeRobotDataset(
        repo_id=repo_id,
        root=resolved,
        video_backend=video_backend,
    )


def dataset_arrays(
    dataset: LeRobotDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hf = dataset.hf_dataset
    states = np.asarray(hf[STATE_KEY], dtype=np.float32)
    actions = np.asarray(hf[ACTION_KEY], dtype=np.float32)
    episodes = np.asarray(hf["episode_index"], dtype=np.int64)
    frames = np.asarray(hf["frame_index"], dtype=np.int64)
    lengths = [len(states), len(actions), len(episodes), len(frames)]
    if len(set(lengths)) != 1:
        raise RuntimeError(f"Dataset column lengths differ: {lengths}")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise FloatingPointError("Dataset state/action contains NaN or Inf")
    return states, actions, episodes, frames


def validate_dataset_metadata(
    *,
    label: str,
    dataset: LeRobotDataset,
    states: np.ndarray,
    actions: np.ndarray,
    episodes: np.ndarray,
    frames: np.ndarray,
    expected_frames: int,
    expected_episodes: int,
) -> None:
    if states.shape != (expected_frames, EXPECTED_STATE_DIM):
        raise RuntimeError(f"{label}: unexpected state shape {states.shape}")
    if actions.shape != (expected_frames, EXPECTED_ACTION_DIM):
        raise RuntimeError(f"{label}: unexpected action shape {actions.shape}")
    unique_episodes = sorted(int(value) for value in np.unique(episodes))
    if unique_episodes != list(range(expected_episodes)):
        raise RuntimeError(f"{label}: episode indices are not contiguous")
    if int(dataset.fps) != EXPECTED_FPS:
        raise RuntimeError(
            f"{label}: fps={dataset.fps}, expected={EXPECTED_FPS}"
        )
    if set(dataset.meta.camera_keys) != {FRONT_KEY, WRIST_KEY}:
        raise RuntimeError(f"{label}: camera keys differ from the contract")
    for episode in unique_episodes:
        indices = np.flatnonzero(episodes == episode)
        expected = np.arange(len(indices), dtype=np.int64)
        if not np.array_equal(frames[indices], expected):
            raise RuntimeError(f"{label}: episode {episode} frames not contiguous")


def validate_sequential_delta_semantics(
    *,
    label: str,
    states: np.ndarray,
    actions: np.ndarray,
    episodes: np.ndarray,
) -> dict[str, float | int]:
    max_next_previous_command_error = 0.0
    max_next_previous_delta_error = 0.0
    max_startup_action = 0.0
    max_startup_previous_delta = 0.0
    checked_transitions = 0

    for episode in sorted(int(value) for value in np.unique(episodes)):
        indices = np.flatnonzero(episodes == episode)
        episode_state = states[indices].astype(np.float64)
        episode_action = actions[indices].astype(np.float64)
        max_startup_action = max(
            max_startup_action,
            float(np.max(np.abs(episode_action[0]))),
        )
        max_startup_previous_delta = max(
            max_startup_previous_delta,
            float(np.max(np.abs(episode_state[0, 12:18]))),
        )
        if len(indices) <= 1:
            continue

        decoded_current_command = (
            episode_state[:-1, 6:12] + episode_action[:-1]
        )
        next_previous_command = episode_state[1:, 6:12]
        next_previous_delta = episode_state[1:, 12:18]
        max_next_previous_command_error = max(
            max_next_previous_command_error,
            float(
                np.max(
                    np.abs(decoded_current_command - next_previous_command)
                )
            ),
        )
        max_next_previous_delta_error = max(
            max_next_previous_delta_error,
            float(np.max(np.abs(episode_action[:-1] - next_previous_delta))),
        )
        checked_transitions += len(indices) - 1

    tolerance = 1e-5
    values = (
        max_next_previous_command_error,
        max_next_previous_delta_error,
        max_startup_action,
        max_startup_previous_delta,
    )
    if max(values) > tolerance:
        raise RuntimeError(
            f"{label}: sequential delta semantics failed; max={max(values)}"
        )
    return {
        "checked_frames": int(len(states)),
        "checked_transitions": int(checked_transitions),
        "max_next_previous_command_error": max_next_previous_command_error,
        "max_next_previous_delta_error": max_next_previous_delta_error,
        "max_startup_action": max_startup_action,
        "max_startup_previous_delta": max_startup_previous_delta,
        "tolerance": tolerance,
    }


def frame_map(
    episodes: np.ndarray,
    frames: np.ndarray,
) -> dict[tuple[int, int], int]:
    mapping = {
        (int(episode), int(frame)): int(index)
        for index, (episode, frame) in enumerate(
            zip(episodes, frames, strict=True)
        )
    }
    if len(mapping) != len(episodes):
        raise RuntimeError("Duplicate (episode_index, frame_index) rows")
    return mapping


def evenly_spaced_frames(last_start: int, count: int) -> list[int]:
    if last_start < 0:
        raise RuntimeError("Episode is shorter than the action chunk")
    count = min(count, last_start + 1)
    values = np.rint(np.linspace(0, last_start, count)).astype(np.int64)
    return sorted(set(int(value) for value in values))


def separated_extremes(
    scores: np.ndarray,
    count: int,
    min_separation: int,
    descending: bool,
) -> list[int]:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    frame_count = len(values)
    weights = values if descending else -values
    negative_infinity = -np.inf
    dp = np.full((count + 1, frame_count + 1), negative_infinity)
    take = np.zeros((count + 1, frame_count + 1), dtype=bool)
    dp[0, :] = 0.0

    for selected in range(1, count + 1):
        for prefix_size in range(1, frame_count + 1):
            skip_value = dp[selected, prefix_size - 1]
            previous_prefix = max(0, prefix_size - min_separation)
            previous_value = dp[selected - 1, previous_prefix]
            take_value = (
                previous_value + weights[prefix_size - 1]
                if np.isfinite(previous_value)
                else negative_infinity
            )
            if take_value > skip_value:
                dp[selected, prefix_size] = take_value
                take[selected, prefix_size] = True
            else:
                dp[selected, prefix_size] = skip_value

    if not np.isfinite(dp[count, frame_count]):
        raise RuntimeError(
            f"Cannot select {count} anchors from {frame_count} starts "
            f"with min_separation={min_separation}"
        )

    chosen: list[int] = []
    selected = count
    prefix_size = frame_count
    while selected > 0:
        if prefix_size <= 0:
            raise RuntimeError("Anchor DP reconstruction failed")
        if take[selected, prefix_size]:
            chosen.append(prefix_size - 1)
            prefix_size = max(0, prefix_size - min_separation)
            selected -= 1
        else:
            prefix_size -= 1
    return sorted(chosen)


def build_anchors(
    *,
    validation_actions: np.ndarray,
    validation_episodes: np.ndarray,
    validation_frames: np.ndarray,
    action_scale: np.ndarray,
    uniform_per_episode: int,
    moving_per_episode: int,
    stationary_per_episode: int,
    min_separation: int,
) -> tuple[list[Anchor], dict[str, int]]:
    mapping = frame_map(validation_episodes, validation_frames)
    anchors: list[Anchor] = []
    memberships = {"startup": 0, "uniform": 0, "moving": 0, "stationary": 0}

    for episode in range(EXPECTED_VALIDATION_EPISODES):
        episode_indices = np.flatnonzero(validation_episodes == episode)
        episode_actions = validation_actions[episode_indices].astype(np.float64)
        last_start = len(episode_actions) - CHUNK_SIZE
        if last_start < 0:
            raise RuntimeError(f"Validation episode {episode} is too short")

        prefix = np.concatenate(
            [
                np.zeros((1, EXPECTED_ACTION_DIM), dtype=np.float64),
                np.cumsum(episode_actions, axis=0),
            ],
            axis=0,
        )
        starts = np.arange(last_start + 1, dtype=np.int64)
        endpoints = prefix[starts + EXECUTION_HORIZON] - prefix[starts]
        scores = np.mean(
            np.abs(endpoints) / action_scale.reshape(1, -1), axis=1
        )

        phase_by_frame: dict[int, set[str]] = {}

        def add_phase(selected_frames: list[int], phase: str) -> None:
            for frame in selected_frames:
                phase_by_frame.setdefault(int(frame), set()).add(phase)
                memberships[phase] += 1

        add_phase([0], "startup")
        add_phase(evenly_spaced_frames(last_start, uniform_per_episode), "uniform")
        add_phase(
            separated_extremes(
                scores, moving_per_episode, min_separation, descending=True
            ),
            "moving",
        )
        add_phase(
            separated_extremes(
                scores, stationary_per_episode, min_separation, descending=False
            ),
            "stationary",
        )

        for frame in sorted(phase_by_frame):
            anchors.append(
                Anchor(
                    episode_index=episode,
                    frame_index=frame,
                    global_index=mapping[(episode, frame)],
                    phases=tuple(sorted(phase_by_frame[frame])),
                    activity_score=float(scores[frame]),
                )
            )

    return anchors, memberships


def image_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    tensor = value.detach()
    if tensor.ndim != 3:
        raise RuntimeError(f"Unexpected image shape {tuple(tensor.shape)}")
    if tensor.shape[0] in (1, 3, 4):
        tensor = tensor[:3]
    elif tensor.shape[-1] in (1, 3, 4):
        tensor = tensor[..., :3].permute(2, 0, 1)
    else:
        raise RuntimeError(f"Cannot infer image channels {tuple(tensor.shape)}")
    tensor = tensor.float()
    if float(tensor.max()) > 1.5:
        tensor = tensor / 255.0
    if tuple(tensor.shape) != (3, 480, 480):
        raise RuntimeError(f"Unexpected image tensor shape {tuple(tensor.shape)}")
    return tensor.unsqueeze(0).to(device)


def item_batch(
    item: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    state = item[STATE_KEY]
    if not isinstance(state, torch.Tensor):
        state = torch.as_tensor(state)
    state = state.float().reshape(-1)
    if state.numel() != EXPECTED_STATE_DIM:
        raise RuntimeError(f"Item state has {state.numel()} values")
    return {
        STATE_KEY: state.reshape(1, EXPECTED_STATE_DIM).to(device),
        FRONT_KEY: image_tensor(item[FRONT_KEY], device),
        WRIST_KEY: image_tensor(item[WRIST_KEY], device),
    }


def load_policy(model_path: Path) -> tuple[ACTPolicy, torch.device]:
    required = (
        model_path / "config.json",
        model_path / "model.safetensors",
        model_path / "train_config.json",
    )
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; stopping offline policy audit")

    policy = ACTPolicy.from_pretrained(model_path, local_files_only=True)
    policy.eval()
    policy.reset()

    state_feature = policy.config.input_features.get(STATE_KEY)
    action_feature = policy.config.output_features.get(ACTION_KEY)
    if state_feature is None or tuple(state_feature.shape) != (EXPECTED_STATE_DIM,):
        raise RuntimeError("Policy state feature is not 18D")
    if action_feature is None or tuple(action_feature.shape) != (EXPECTED_ACTION_DIM,):
        raise RuntimeError("Policy action feature is not 6D")
    for camera_key in (FRONT_KEY, WRIST_KEY):
        feature = policy.config.input_features.get(camera_key)
        if feature is None or tuple(feature.shape) != (3, 480, 480):
            raise RuntimeError(f"Policy camera feature mismatch: {camera_key}")
    if int(policy.config.chunk_size) != CHUNK_SIZE:
        raise RuntimeError(f"Policy chunk_size is not {CHUNK_SIZE}")
    if int(policy.config.n_action_steps) != EXECUTION_HORIZON:
        raise RuntimeError(f"Policy n_action_steps is not {EXECUTION_HORIZON}")
    if bool(policy.config.use_vae):
        raise RuntimeError("Policy use_vae must remain false")
    if bool(policy.config.use_amp):
        raise RuntimeError("Policy use_amp must remain false")

    device = torch.device(policy.config.device)
    if device.type != "cuda":
        raise RuntimeError(f"Frozen policy device is {device}, expected cuda")
    return policy, device


@torch.inference_mode()
def infer_chunk(
    policy: ACTPolicy,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    policy.reset()
    prediction = policy.predict_action_chunk(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    result = prediction.detach().cpu().float().numpy()
    expected_shape = (1, CHUNK_SIZE, EXPECTED_ACTION_DIM)
    if result.shape != expected_shape:
        raise RuntimeError(
            f"Prediction shape={result.shape}, expected={expected_shape}"
        )
    if not np.isfinite(result).all():
        raise FloatingPointError("ACT prediction contains NaN or Inf")
    return result[0]


@torch.inference_mode()
def queue_equivalence_error(
    policy: ACTPolicy,
    batch: dict[str, torch.Tensor],
    expected_first_five: np.ndarray,
    device: torch.device,
) -> float:
    policy.reset()
    queued: list[np.ndarray] = []
    for _ in range(EXECUTION_HORIZON):
        action = policy.select_action(batch)
        array = action.detach().cpu().float().numpy().reshape(-1)
        if array.shape != (EXPECTED_ACTION_DIM,):
            raise RuntimeError(f"Queued action has shape {array.shape}")
        queued.append(array)
    if device.type == "cuda":
        torch.cuda.synchronize()
    policy.reset()
    return float(
        np.max(
            np.abs(np.stack(queued).astype(np.float64) - expected_first_five)
        )
    )


def recursive_decode(base: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    base_array = np.asarray(base, dtype=np.float64).reshape(EXPECTED_ACTION_DIM)
    delta_array = np.asarray(deltas, dtype=np.float64)
    if delta_array.shape != (EXECUTION_HORIZON, EXPECTED_ACTION_DIM):
        raise RuntimeError(f"Unexpected execution deltas: {delta_array.shape}")
    output = []
    running = base_array.copy()
    for delta in delta_array:
        running = running + delta
        output.append(running.copy())
    return np.stack(output)


def per_joint_distribution(values: np.ndarray) -> list[dict[str, float | str]]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape[-1] != EXPECTED_ACTION_DIM or not np.isfinite(array).all():
        raise RuntimeError("Invalid array for per-joint distribution")
    flat = np.abs(array.reshape(-1, EXPECTED_ACTION_DIM))
    result: list[dict[str, float | str]] = []
    for joint, motor in enumerate(MOTORS):
        column = flat[:, joint]
        result.append(
            {
                "joint": motor,
                "abs_mean": float(np.mean(column)),
                "abs_p50": float(np.percentile(column, 50.0)),
                "abs_p95": float(np.percentile(column, 95.0)),
                "abs_p99": float(np.percentile(column, 99.0)),
                "abs_max": float(np.max(column)),
            }
        )
    return result


def main() -> int:
    args = parse_args()

    print("===== FROZEN RELEASE INTEGRITY =====")
    release = verify_release(
        args.candidate_root,
        args.source_checkpoint_root,
    )
    model_path = args.candidate_root.resolve() / "pretrained_model"
    print(f"candidate: {args.candidate_root.resolve()}")
    print(f"selected step: {EXPECTED_STEP}")
    print("manifest and source-checkpoint byte identity: PASS")

    print("\n===== LOAD REVIEWED V3 PLAN =====")
    plan = load_json(args.v3_plan.resolve())
    if plan.get("status") != "PASS":
        raise RuntimeError("V3 plan status is not PASS")
    schema = plan.get("proposed_v3_schema")
    if not isinstance(schema, dict):
        raise RuntimeError("V3 plan is missing proposed_v3_schema")
    if int(schema.get("observation_state_dim", -1)) != EXPECTED_STATE_DIM:
        raise RuntimeError("V3 plan state dimension is not 18")
    if int(schema.get("action_dim", -1)) != EXPECTED_ACTION_DIM:
        raise RuntimeError("V3 plan action dimension is not 6")
    expected_state_order = [
        "actual_q_t[6]",
        "previous_command_t[6]",
        "previous_command_delta_t[6]",
    ]
    if schema.get("observation_state_order") != expected_state_order:
        raise RuntimeError("V3 plan observation-state order has changed")
    if schema.get("action_formula") != "command[t] - previous_command[t]":
        raise RuntimeError("V3 plan action formula has changed")
    runtime_decode = str(schema.get("runtime_decode", ""))
    if "sequential cumulative sum" not in runtime_decode:
        raise RuntimeError("V3 plan runtime decoder is not cumulative")
    guardrail = np.asarray(
        plan["command_delta"]["empirical_guardrail_diagnostic"][
            "candidate_max_abs_delta_by_joint"
        ],
        dtype=np.float64,
    )
    if guardrail.shape != (EXPECTED_ACTION_DIM,) or np.any(guardrail <= 0):
        raise RuntimeError("Invalid empirical guardrail diagnostic in V3 plan")
    print("state/action order and recursive runtime decode contract: PASS")
    print("empirical values loaded as diagnostics, not hardware limits")

    print("\n===== FULL DATASET SEQUENTIAL SEMANTICS =====")
    train_dataset = load_dataset(
        args.train_repo_id, args.train_dataset_root, args.video_backend
    )
    validation_dataset = load_dataset(
        args.validation_repo_id,
        args.validation_dataset_root,
        args.video_backend,
    )
    train_state, train_action, train_episode, train_frame = dataset_arrays(
        train_dataset
    )
    val_state, val_action, val_episode, val_frame = dataset_arrays(
        validation_dataset
    )
    validate_dataset_metadata(
        label="train",
        dataset=train_dataset,
        states=train_state,
        actions=train_action,
        episodes=train_episode,
        frames=train_frame,
        expected_frames=EXPECTED_TRAIN_FRAMES,
        expected_episodes=EXPECTED_TRAIN_EPISODES,
    )
    validate_dataset_metadata(
        label="validation",
        dataset=validation_dataset,
        states=val_state,
        actions=val_action,
        episodes=val_episode,
        frames=val_frame,
        expected_frames=EXPECTED_VALIDATION_FRAMES,
        expected_episodes=EXPECTED_VALIDATION_EPISODES,
    )
    train_semantics = validate_sequential_delta_semantics(
        label="train",
        states=train_state,
        actions=train_action,
        episodes=train_episode,
    )
    validation_semantics = validate_sequential_delta_semantics(
        label="validation",
        states=val_state,
        actions=val_action,
        episodes=val_episode,
    )
    print(
        "train: "
        f"frames={len(train_state)} transitions="
        f"{train_semantics['checked_transitions']} PASS"
    )
    print(
        "validation: "
        f"frames={len(val_state)} transitions="
        f"{validation_semantics['checked_transitions']} PASS"
    )
    print(
        "max train next-command error: "
        f"{train_semantics['max_next_previous_command_error']:.10f}"
    )
    print(
        "max validation next-command error: "
        f"{validation_semantics['max_next_previous_command_error']:.10f}"
    )

    action_scale = np.std(train_action.astype(np.float64), axis=0)
    action_scale = np.maximum(action_scale, 1e-3)
    train_action_max = np.max(np.abs(train_action.astype(np.float64)), axis=0)
    train_commands = (
        train_state[:, 6:12].astype(np.float64)
        + train_action.astype(np.float64)
    )
    train_command_min = np.min(train_commands, axis=0)
    train_command_max = np.max(train_commands, axis=0)

    print("\n===== POLICY AND QUEUE CONTRACT =====")
    policy, device = load_policy(model_path)
    print(
        "policy: state=18D action=6D chunk=10 queue=5 "
        "vae=false amp=false PASS"
    )

    anchors, memberships = build_anchors(
        validation_actions=val_action,
        validation_episodes=val_episode,
        validation_frames=val_frame,
        action_scale=action_scale,
        uniform_per_episode=args.uniform_samples_per_episode,
        moving_per_episode=args.moving_samples_per_episode,
        stationary_per_episode=args.stationary_samples_per_episode,
        min_separation=args.stress_min_separation,
    )
    print(f"deterministic unique anchors: {len(anchors)}")
    print(f"phase memberships: {memberships}")

    predictions: list[np.ndarray] = []
    base_commands: list[np.ndarray] = []
    target_deltas: list[np.ndarray] = []
    queue_errors: list[float] = []
    mapping = frame_map(val_episode, val_frame)
    target_transition_error = 0.0

    print("\n===== OFFLINE INFERENCE AND DELTA DECODER =====")
    for number, anchor in enumerate(anchors, start=1):
        item = validation_dataset[int(anchor.global_index)]
        batch = item_batch(item, device)
        chunk = infer_chunk(policy, batch, device)
        predictions.append(chunk)
        base = val_state[anchor.global_index, 6:12].astype(np.float64)
        base_commands.append(base)

        target_indices = [
            mapping[(anchor.episode_index, anchor.frame_index + offset)]
            for offset in range(CHUNK_SIZE)
        ]
        target_chunk = val_action[np.asarray(target_indices)].astype(np.float64)
        target_deltas.append(target_chunk)

        decoded_target = recursive_decode(base, target_chunk[:EXECUTION_HORIZON])
        expected_target = np.stack(
            [
                val_state[
                    mapping[(anchor.episode_index, anchor.frame_index + offset + 1)],
                    6:12,
                ]
                for offset in range(EXECUTION_HORIZON)
            ]
        ).astype(np.float64)
        target_transition_error = max(
            target_transition_error,
            float(np.max(np.abs(decoded_target - expected_target))),
        )

        if number <= args.queue_check_anchors:
            queue_errors.append(
                queue_equivalence_error(
                    policy,
                    batch,
                    chunk[:EXECUTION_HORIZON].astype(np.float64),
                    device,
                )
            )
        if number % 40 == 0 or number == len(anchors):
            print(f"processed {number}/{len(anchors)}")

    prediction_array = np.stack(predictions).astype(np.float64)
    target_array = np.stack(target_deltas).astype(np.float64)
    base_array = np.stack(base_commands).astype(np.float64)
    max_queue_error = max(queue_errors, default=0.0)
    if max_queue_error > 1e-6:
        raise RuntimeError(
            f"select_action queue differs from first five chunk rows: "
            f"{max_queue_error}"
        )
    if target_transition_error > 1e-5:
        raise RuntimeError(
            f"Recorded target recursive decode mismatch: {target_transition_error}"
        )

    predicted_first_five = prediction_array[:, :EXECUTION_HORIZON]
    vectorized_commands = (
        base_array[:, None, :] + np.cumsum(predicted_first_five, axis=1)
    )
    loop_commands = np.stack(
        [
            recursive_decode(base, deltas)
            for base, deltas in zip(
                base_array, predicted_first_five, strict=True
            )
        ]
    )
    decoder_equivalence_error = float(
        np.max(np.abs(vectorized_commands - loop_commands))
    )
    if decoder_equivalence_error > 1e-10:
        raise RuntimeError(
            f"Vectorized/recursive decoder mismatch: {decoder_equivalence_error}"
        )

    # This is a scale/schema sanity ceiling, not a robot limit. A model that
    # emits values ten times larger than every training delta is likely being
    # interpreted with the wrong action semantics.
    scale_denominator = np.maximum(train_action_max, 1e-3)
    catastrophic_scale_ratio_by_joint = (
        np.max(np.abs(prediction_array), axis=(0, 1)) / scale_denominator
    )
    catastrophic_scale_ratio = float(
        np.max(catastrophic_scale_ratio_by_joint)
    )
    if catastrophic_scale_ratio > 10.0:
        raise RuntimeError(
            "Predicted action scale exceeds the offline schema sanity ceiling: "
            f"ratio={catastrophic_scale_ratio}"
        )

    empirical_exceed = np.abs(predicted_first_five) > guardrail.reshape(1, 1, -1)
    empirical_exceed_rate_by_joint = np.mean(empirical_exceed, axis=(0, 1))
    empirical_exceed_rate_any_step = float(np.mean(np.any(empirical_exceed, axis=2)))
    outside_train_envelope = (
        (vectorized_commands < train_command_min.reshape(1, 1, -1))
        | (vectorized_commands > train_command_max.reshape(1, 1, -1))
    )
    outside_train_envelope_rate_by_joint = np.mean(
        outside_train_envelope, axis=(0, 1)
    )
    wrong_absolute_interpretation_mae = float(
        np.mean(np.abs(predicted_first_five - vectorized_commands))
    )

    predicted_stats = per_joint_distribution(predicted_first_five)
    target_stats = per_joint_distribution(target_array[:, :EXECUTION_HORIZON])
    rows: list[dict[str, Any]] = []
    for joint, motor in enumerate(MOTORS):
        predicted_row = predicted_stats[joint]
        target_row = target_stats[joint]
        rows.append(
            {
                "joint": motor,
                "train_abs_delta_max": float(train_action_max[joint]),
                "empirical_diagnostic_threshold": float(guardrail[joint]),
                "predicted_abs_p50": predicted_row["abs_p50"],
                "predicted_abs_p95": predicted_row["abs_p95"],
                "predicted_abs_p99": predicted_row["abs_p99"],
                "predicted_abs_max": predicted_row["abs_max"],
                "target_abs_p99": target_row["abs_p99"],
                "empirical_threshold_exceed_rate": float(
                    empirical_exceed_rate_by_joint[joint]
                ),
                "decoded_command_outside_train_envelope_rate": float(
                    outside_train_envelope_rate_by_joint[joint]
                ),
                "catastrophic_scale_ratio": float(
                    catastrophic_scale_ratio_by_joint[joint]
                ),
            }
        )

    print(f"queue equivalence max error: {max_queue_error:.10f}")
    print(
        "recorded target recursive decode max error: "
        f"{target_transition_error:.10f}"
    )
    print(
        "predicted recursive/vectorized decode max error: "
        f"{decoder_equivalence_error:.10f}"
    )
    print(
        "wrong delta-as-absolute interpretation MAE: "
        f"{wrong_absolute_interpretation_mae:.6f}"
    )

    print("\n===== EMPIRICAL DIAGNOSTICS (NOT HARDWARE LIMITS) =====")
    print(
        "joint            pred_p99  pred_max  diagnostic  exceed_rate  "
        "outside_train_cmd"
    )
    for row in rows:
        print(
            f"{row['joint']:<16} "
            f"{row['predicted_abs_p99']:9.5f} "
            f"{row['predicted_abs_max']:9.5f} "
            f"{row['empirical_diagnostic_threshold']:11.5f} "
            f"{row['empirical_threshold_exceed_rate']:11.3%} "
            f"{row['decoded_command_outside_train_envelope_rate']:17.3%}"
        )

    try:
        lerobot_version = importlib.metadata.version("lerobot")
    except importlib.metadata.PackageNotFoundError:
        lerobot_version = "unknown-editable-install"

    report: dict[str, Any] = {
        "schema_version": "act_v3_delta_deployment_contract_v1",
        "status": "PASS",
        "decision": "OFFLINE_CONTRACT_PASS_HARDWARE_REMAINS_BLOCKED",
        "scope": {
            "offline_only": True,
            "dataset_modified": False,
            "hardware_accessed": False,
            "hardware_deployment_authorized": False,
            "physical_safety_certified": False,
        },
        "environment": {
            "lerobot_version": lerobot_version,
            "torch_version": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0),
        },
        "release": release,
        "contract": {
            "selected_step": EXPECTED_STEP,
            "state_order": [
                "actual_q_t[6]",
                "last_sent_command_t[6]",
                "last_sent_delta_t[6]",
            ],
            "policy_output_semantics": "command_delta[10,6]",
            "execution_horizon": EXECUTION_HORIZON,
            "replan_formula": (
                "command[i] = last_sent_command + predicted_delta[i]; "
                "update last_sent_command and last_sent_delta after each send"
            ),
            "vectorized_formula": (
                "base_command + cumsum(predicted_delta[:5], axis=0)"
            ),
            "forbidden_interpretation": (
                "Never send predicted_delta directly as an absolute joint command"
            ),
            "reset_rule": (
                "Initialize last_sent_command from the verified startup/Home "
                "command and last_sent_delta=zeros[6]"
            ),
        },
        "dataset_semantics": {
            "train": train_semantics,
            "validation": validation_semantics,
        },
        "inference_protocol": {
            "unique_anchor_count": len(anchors),
            "phase_memberships": memberships,
            "queue_checked_anchor_count": len(queue_errors),
            "human_motion_assumption": "No uniform-speed assumption",
        },
        "decoder_checks": {
            "queue_equivalence_max_error": max_queue_error,
            "target_recursive_decode_max_error": target_transition_error,
            "predicted_recursive_vectorized_max_error": (
                decoder_equivalence_error
            ),
            "wrong_delta_as_absolute_interpretation_mae": (
                wrong_absolute_interpretation_mae
            ),
        },
        "offline_scale_sanity": {
            "hard_failure_ratio": 10.0,
            "max_ratio_observed": catastrophic_scale_ratio,
            "ratio_by_joint": catastrophic_scale_ratio_by_joint,
            "meaning": (
                "Prediction max divided by train action max; schema sanity "
                "check only, not a hardware limit"
            ),
        },
        "empirical_diagnostics": {
            "warning": (
                "Dataset-derived diagnostics only; not manufacturer limits, "
                "not control limits, and not a physical safety certification."
            ),
            "candidate_threshold_by_joint": guardrail,
            "first5_any_joint_exceed_rate": empirical_exceed_rate_any_step,
            "joint_rows": rows,
            "train_command_min": train_command_min,
            "train_command_max": train_command_max,
        },
        "required_next_stage": {
            "name": "guarded dry-run runtime adapter",
            "hardware_access_allowed": False,
            "requirements": [
                "No robot or serial object",
                "Construct 18D state in the frozen order",
                "Recursively integrate each delta from last_sent_command",
                "Apply separately reviewed hardware joint and rate limits",
                "Log proposed commands without sending them",
                "Fail closed on stale frames, NaN/Inf, or shape mismatch",
            ],
        },
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "deployment_contract_report.json"
    report_path.write_text(
        json.dumps(to_jsonable(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    csv_path = output_dir / "deployment_contract_joint_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    del policy
    del predictions
    del prediction_array
    gc.collect()
    torch.cuda.empty_cache()

    print("\n===== OUTPUT =====")
    print(report_path)
    print(csv_path)
    print("ACT V3 DELTA DEPLOYMENT CONTRACT AUDIT: PASS")
    print("HARDWARE DEPLOYMENT REMAINS BLOCKED.")
    print("NO DATASET WAS MODIFIED. NO HARDWARE WAS ACCESSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
