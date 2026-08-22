#!/usr/bin/env python
"""Offline held-out checkpoint selection for SO-101 ACT v3 delta actions.

This evaluator is intentionally independent from the training loop. It loads
only the physically separate v3 validation dataset and saved ACT checkpoints.
No robot, serial-port, motor, or live-camera module is imported.

Pre-registered selection rule
-----------------------------
1. Build the same deterministic validation anchors for every checkpoint.
2. Use evenly spaced anchors as the primary representation of the recorded
   human validation distribution. Human motion is not assumed to be uniform.
3. Decode predicted delta actions by cumulative sum for the five actions that
   would be executed before replanning.
4. Select the checkpoint with the lowest normalized cumulative-command MAE on
   the uniform anchors. Moving-anchor error is the first tie-breaker, followed
   by the earlier step.
5. If the best checkpoint is the last checkpoint evaluated, report that the
   optimum is boundary-censored and training should be extended before a final
   checkpoint is selected.

The moving and stationary anchor sets are diagnostic stress sets. They do not
replace or reweight the primary held-out distribution after results are seen.
"""

from __future__ import annotations

import argparse
import csv
import gc
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
CHUNK_SIZE = 10
EXECUTION_HORIZON = 5
DEFAULT_CHECKPOINT_STEPS = [1000, 2000, 3000, 4000, 5000, 6000]


@dataclass(frozen=True)
class Anchor:
    episode_index: int
    frame_index: int
    global_index: int
    phases: tuple[str, ...]
    activity_score: float
    target_chunk: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
        "--train-output-root",
        type=Path,
        default=Path("outputs/train/act_red_cube_v3_delta_stage1"),
    )
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        nargs="+",
        default=DEFAULT_CHECKPOINT_STEPS,
    )
    parser.add_argument("--uniform-samples-per-episode", type=int, default=24)
    parser.add_argument("--moving-samples-per-episode", type=int, default=4)
    parser.add_argument("--stationary-samples-per-episode", type=int, default=4)
    parser.add_argument("--stress-min-separation", type=int, default=20)
    parser.add_argument("--video-backend", default="torchcodec")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/eval/act_red_cube_v3_delta_stage1_validation"),
    )
    args = parser.parse_args()

    steps = sorted(set(int(step) for step in args.checkpoint_steps))
    if not steps or steps[0] <= 0:
        parser.error("--checkpoint-steps must contain positive integers")
    args.checkpoint_steps = steps
    for name in (
        "uniform_samples_per_episode",
        "moving_samples_per_episode",
        "stationary_samples_per_episode",
        "stress_min_separation",
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


def validate_dataset_contract(
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
    unique_episodes = sorted(int(x) for x in np.unique(episodes))
    if unique_episodes != list(range(expected_episodes)):
        raise RuntimeError(f"{label}: episode indices are not contiguous")
    if int(dataset.fps) != 15:
        raise RuntimeError(f"{label}: FPS is {dataset.fps}, expected 15")
    if set(dataset.meta.camera_keys) != {FRONT_KEY, WRIST_KEY}:
        raise RuntimeError(f"{label}: camera keys differ from the contract")
    for episode in unique_episodes:
        indices = np.flatnonzero(episodes == episode)
        if not np.array_equal(
            frames[indices], np.arange(len(indices), dtype=np.int64)
        ):
            raise RuntimeError(f"{label}: episode {episode} frames are not contiguous")


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
    if frame_count == 0:
        raise RuntimeError("Cannot select stress anchors from an empty episode")

    # Exact dynamic programming, rather than greedy ranking. Greedy selection
    # can choose one central peak that blocks two valid separated peaks, which
    # is especially undesirable for irregular human demonstrations.
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
            f"Cannot select {count} stress anchors from {frame_count} starts "
            f"with min_separation={min_separation}"
        )

    chosen: list[int] = []
    selected = count
    prefix_size = frame_count
    while selected > 0:
        if prefix_size <= 0:
            raise RuntimeError("Stress-anchor DP reconstruction failed")
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
    phase_counts = {"startup": 0, "uniform": 0, "moving": 0, "stationary": 0}

    for episode in range(EXPECTED_VALIDATION_EPISODES):
        episode_indices = np.flatnonzero(validation_episodes == episode)
        episode_actions = validation_actions[episode_indices]
        last_start = len(episode_actions) - CHUNK_SIZE
        if last_start < 0:
            raise RuntimeError(f"Validation episode {episode} is too short")

        # Activity is the normalized command displacement after the five
        # deltas that would be executed before replanning.
        prefix = np.concatenate(
            [
                np.zeros((1, EXPECTED_ACTION_DIM), dtype=np.float64),
                np.cumsum(episode_actions.astype(np.float64), axis=0),
            ],
            axis=0,
        )
        starts = np.arange(last_start + 1, dtype=np.int64)
        endpoints = prefix[starts + EXECUTION_HORIZON] - prefix[starts]
        scores = np.mean(
            np.abs(endpoints) / action_scale.reshape(1, -1),
            axis=1,
        )

        phase_by_frame: dict[int, set[str]] = {}

        def add_phase(frames_to_add: list[int], phase: str) -> None:
            for frame in frames_to_add:
                phase_by_frame.setdefault(int(frame), set()).add(phase)
                phase_counts[phase] += 1

        add_phase([0], "startup")
        add_phase(
            evenly_spaced_frames(last_start, uniform_per_episode),
            "uniform",
        )
        add_phase(
            separated_extremes(
                scores,
                moving_per_episode,
                min_separation,
                descending=True,
            ),
            "moving",
        )
        add_phase(
            separated_extremes(
                scores,
                stationary_per_episode,
                min_separation,
                descending=False,
            ),
            "stationary",
        )

        for frame in sorted(phase_by_frame):
            indices = [
                mapping[(episode, frame + offset)]
                for offset in range(CHUNK_SIZE)
            ]
            target = validation_actions[
                np.asarray(indices, dtype=np.int64)
            ].copy()
            anchors.append(
                Anchor(
                    episode_index=episode,
                    frame_index=frame,
                    global_index=mapping[(episode, frame)],
                    phases=tuple(sorted(phase_by_frame[frame])),
                    activity_score=float(scores[frame]),
                    target_chunk=target,
                )
            )

    for phase, expected_per_episode in (
        ("startup", 1),
        ("uniform", uniform_per_episode),
        ("moving", moving_per_episode),
        ("stationary", stationary_per_episode),
    ):
        expected = expected_per_episode * EXPECTED_VALIDATION_EPISODES
        if phase_counts[phase] != expected:
            raise RuntimeError(
                f"{phase} anchor count={phase_counts[phase]}, expected={expected}"
            )
    return anchors, phase_counts


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


def checkpoint_model_path(train_root: Path, step: int) -> Path:
    return (
        train_root.resolve()
        / "checkpoints"
        / f"{step:06d}"
        / "pretrained_model"
    )


def validate_checkpoint_files(path: Path) -> None:
    required = [
        path / "config.json",
        path / "model.safetensors",
        path / "train_config.json",
    ]
    missing = [item for item in required if not item.is_file() or item.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(
            "Checkpoint files are missing or empty:\n"
            + "\n".join(f"  - {item}" for item in missing)
        )


def load_policy(path: Path, step: int) -> ACTPolicy:
    validate_checkpoint_files(path)
    print(f"Loading step {step}: {path}")
    policy = ACTPolicy.from_pretrained(path, local_files_only=True)
    policy.eval()
    policy.reset()

    state_feature = policy.config.input_features.get(STATE_KEY)
    action_feature = policy.config.output_features.get(ACTION_KEY)
    if state_feature is None or tuple(state_feature.shape) != (EXPECTED_STATE_DIM,):
        raise RuntimeError(f"step {step}: policy state feature is not 18D")
    if action_feature is None or tuple(action_feature.shape) != (EXPECTED_ACTION_DIM,):
        raise RuntimeError(f"step {step}: policy action feature is not 6D")
    if int(policy.config.chunk_size) != CHUNK_SIZE:
        raise RuntimeError(f"step {step}: chunk_size is not {CHUNK_SIZE}")
    if int(policy.config.n_action_steps) != EXECUTION_HORIZON:
        raise RuntimeError(
            f"step {step}: n_action_steps is not {EXECUTION_HORIZON}"
        )
    if bool(policy.config.use_vae):
        raise RuntimeError(f"step {step}: use_vae must remain false")
    if bool(policy.config.use_amp):
        raise RuntimeError(f"step {step}: use_amp must remain false")
    return policy


@torch.inference_mode()
def infer_anchor(
    policy: ACTPolicy,
    item: dict[str, Any],
) -> np.ndarray:
    device = torch.device(policy.config.device)
    state = item[STATE_KEY]
    if not isinstance(state, torch.Tensor):
        state = torch.as_tensor(state)
    state = state.float().reshape(-1)
    if state.numel() != EXPECTED_STATE_DIM:
        raise RuntimeError(f"Validation item state has {state.numel()} values")
    batch = {
        STATE_KEY: state.reshape(1, EXPECTED_STATE_DIM).to(device),
        FRONT_KEY: image_tensor(item[FRONT_KEY], device),
        WRIST_KEY: image_tensor(item[WRIST_KEY], device),
    }
    policy.reset()
    prediction = policy.predict_action_chunk(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    array = prediction.detach().cpu().float().numpy()
    expected_shape = (1, CHUNK_SIZE, EXPECTED_ACTION_DIM)
    if array.shape != expected_shape:
        raise RuntimeError(
            f"Prediction shape={array.shape}, expected={expected_shape}"
        )
    if not np.isfinite(array).all():
        raise FloatingPointError("ACT prediction contains NaN or Inf")
    return array[0]


def scalar_summary(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise RuntimeError("Cannot summarize empty or non-finite values")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90.0)),
        "max": float(np.max(array)),
    }


def phase_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    action_scale: np.ndarray,
) -> dict[str, Any]:
    if predictions.shape != targets.shape:
        raise RuntimeError(
            f"Prediction/target shapes differ: {predictions.shape} vs {targets.shape}"
        )
    if predictions.shape[1:] != (CHUNK_SIZE, EXPECTED_ACTION_DIM):
        raise RuntimeError(f"Unexpected phase array shape {predictions.shape}")

    error = predictions.astype(np.float64) - targets.astype(np.float64)
    action0_mae = np.mean(np.abs(error[:, 0]), axis=1)
    delta_first5_mae = np.mean(
        np.abs(error[:, :EXECUTION_HORIZON]), axis=(1, 2)
    )
    delta_full10_mae = np.mean(np.abs(error), axis=(1, 2))

    cumulative_error = np.cumsum(error[:, :EXECUTION_HORIZON], axis=1)
    cumulative_first5_mae = np.mean(
        np.abs(cumulative_error), axis=(1, 2)
    )
    normalized_cumulative = (
        np.abs(cumulative_error)
        / action_scale.reshape(1, 1, EXPECTED_ACTION_DIM)
    )
    cumulative_first5_normalized_mae = np.mean(
        normalized_cumulative, axis=(1, 2)
    )
    endpoint_error = cumulative_error[:, -1]
    endpoint5_mae = np.mean(np.abs(endpoint_error), axis=1)
    endpoint5_normalized_mae = np.mean(
        np.abs(endpoint_error) / action_scale.reshape(1, -1), axis=1
    )

    endpoint_by_joint = {
        motor: {
            "mae": float(np.mean(np.abs(endpoint_error[:, joint]))),
            "mean_signed_error": float(np.mean(endpoint_error[:, joint])),
            "p90_abs_error": float(
                np.percentile(np.abs(endpoint_error[:, joint]), 90.0)
            ),
            "max_abs_error": float(np.max(np.abs(endpoint_error[:, joint]))),
        }
        for joint, motor in enumerate(MOTORS)
    }

    return {
        "sample_count": int(predictions.shape[0]),
        "delta_action0_mae": scalar_summary(action0_mae),
        "delta_first5_mae": scalar_summary(delta_first5_mae),
        "delta_full10_mae": scalar_summary(delta_full10_mae),
        "cumulative_first5_mae": scalar_summary(cumulative_first5_mae),
        "cumulative_first5_normalized_mae": scalar_summary(
            cumulative_first5_normalized_mae
        ),
        "endpoint5_mae": scalar_summary(endpoint5_mae),
        "endpoint5_normalized_mae": scalar_summary(
            endpoint5_normalized_mae
        ),
        "endpoint5_error_by_joint": endpoint_by_joint,
    }


def metrics_for_all_phases(
    *,
    anchors: list[Anchor],
    predictions: np.ndarray,
    action_scale: np.ndarray,
) -> dict[str, Any]:
    phases = ("startup", "uniform", "moving", "stationary")
    result: dict[str, Any] = {}
    for phase in phases:
        indices = [
            index for index, anchor in enumerate(anchors) if phase in anchor.phases
        ]
        if not indices:
            raise RuntimeError(f"No anchors for phase {phase}")
        index_array = np.asarray(indices, dtype=np.int64)
        targets = np.stack([anchors[index].target_chunk for index in indices])
        result[phase] = phase_metrics(
            predictions[index_array], targets, action_scale
        )
    return result


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; stopping checkpoint validation")

    print("===== LOAD PHYSICALLY SEPARATE V3 DATASETS =====")
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
    validate_dataset_contract(
        label="train",
        dataset=train_dataset,
        states=train_state,
        actions=train_action,
        episodes=train_episode,
        frames=train_frame,
        expected_frames=EXPECTED_TRAIN_FRAMES,
        expected_episodes=EXPECTED_TRAIN_EPISODES,
    )
    validate_dataset_contract(
        label="validation",
        dataset=validation_dataset,
        states=val_state,
        actions=val_action,
        episodes=val_episode,
        frames=val_frame,
        expected_frames=EXPECTED_VALIDATION_FRAMES,
        expected_episodes=EXPECTED_VALIDATION_EPISODES,
    )
    print(
        f"train: episodes={EXPECTED_TRAIN_EPISODES}, "
        f"frames={EXPECTED_TRAIN_FRAMES}"
    )
    print(
        f"validation: episodes={EXPECTED_VALIDATION_EPISODES}, "
        f"frames={EXPECTED_VALIDATION_FRAMES}"
    )

    action_scale = np.std(train_action.astype(np.float64), axis=0)
    action_scale = np.maximum(action_scale, 1e-3)
    print(f"train delta std by joint: {action_scale.tolist()}")

    print("\n===== FREEZE VALIDATION ANCHORS =====")
    anchors, phase_counts = build_anchors(
        validation_actions=val_action,
        validation_episodes=val_episode,
        validation_frames=val_frame,
        action_scale=action_scale,
        uniform_per_episode=args.uniform_samples_per_episode,
        moving_per_episode=args.moving_samples_per_episode,
        stationary_per_episode=args.stationary_samples_per_episode,
        min_separation=args.stress_min_separation,
    )
    print(f"unique decoded anchors per checkpoint: {len(anchors)}")
    print(f"phase memberships: {phase_counts}")
    print(
        "selection metric: uniform cumulative first-5 normalized MAE "
        "(lower is better)"
    )

    try:
        lerobot_version = importlib.metadata.version("lerobot")
    except importlib.metadata.PackageNotFoundError:
        lerobot_version = "unknown-editable-install"

    report: dict[str, Any] = {
        "schema_version": "act_v3_delta_validation_selection_v1",
        "status": "PASS",
        "scope": {
            "offline_only": True,
            "dataset_modified": False,
            "hardware_accessed": False,
            "physical_safety_certified": False,
        },
        "environment": {
            "lerobot_version": lerobot_version,
            "torch_version": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0),
        },
        "contract": {
            "state_dim": EXPECTED_STATE_DIM,
            "action_dim": EXPECTED_ACTION_DIM,
            "action_semantics": "command delta",
            "chunk_size": CHUNK_SIZE,
            "n_action_steps": EXECUTION_HORIZON,
            "use_vae": False,
            "use_amp": False,
            "train_frames": EXPECTED_TRAIN_FRAMES,
            "validation_frames": EXPECTED_VALIDATION_FRAMES,
        },
        "protocol": {
            "primary_distribution": (
                "24 deterministic evenly spaced valid chunk starts per "
                "validation episode"
            ),
            "primary_metric": "uniform.cumulative_first5_normalized_mae.mean",
            "primary_metric_semantics": (
                "Mean absolute error after cumulative decoding of the five "
                "delta actions executed before replanning, normalized per "
                "joint by train-only delta standard deviation"
            ),
            "first_tiebreaker": "moving.cumulative_first5_normalized_mae.mean",
            "second_tiebreaker": "earlier checkpoint step",
            "boundary_rule": (
                "If the best checkpoint is the final evaluated step, extend "
                "training; do not declare a boundary checkpoint final."
            ),
            "human_motion_assumption": "No uniform-speed assumption",
            "action_scale_train_only": action_scale.tolist(),
            "phase_memberships": phase_counts,
            "unique_anchors": len(anchors),
        },
        "paths": {
            "train_dataset": str(args.train_dataset_root.resolve()),
            "validation_dataset": str(args.validation_dataset_root.resolve()),
            "train_output_root": str(args.train_output_root.resolve()),
        },
        "anchors": [
            {
                "episode_index": anchor.episode_index,
                "frame_index": anchor.frame_index,
                "global_index": anchor.global_index,
                "phases": list(anchor.phases),
                "activity_score": anchor.activity_score,
            }
            for anchor in anchors
        ],
        "checkpoints": {},
    }

    print("\n===== CHECKPOINT INFERENCE =====")
    for step in args.checkpoint_steps:
        model_path = checkpoint_model_path(args.train_output_root, step)
        policy = load_policy(model_path, step)
        predictions: list[np.ndarray] = []
        for number, anchor in enumerate(anchors, start=1):
            item = validation_dataset[int(anchor.global_index)]
            predictions.append(infer_anchor(policy, item))
            if number % 50 == 0 or number == len(anchors):
                print(f"step {step}: processed {number}/{len(anchors)}")
        prediction_array = np.stack(predictions)
        phase_result = metrics_for_all_phases(
            anchors=anchors,
            predictions=prediction_array,
            action_scale=action_scale,
        )
        report["checkpoints"][str(step)] = {
            "step": step,
            "checkpoint": str(model_path),
            "model_bytes": int((model_path / "model.safetensors").stat().st_size),
            "metrics": phase_result,
        }
        print(
            f"step {step}: "
            f"uniform_cum5_norm="
            f"{phase_result['uniform']['cumulative_first5_normalized_mae']['mean']:.6f} "
            f"moving_cum5_norm="
            f"{phase_result['moving']['cumulative_first5_normalized_mae']['mean']:.6f}"
        )
        del policy
        del predictions
        del prediction_array
        gc.collect()
        torch.cuda.empty_cache()

    def rank_key(step: int) -> tuple[float, float, int]:
        metrics = report["checkpoints"][str(step)]["metrics"]
        return (
            float(
                metrics["uniform"]["cumulative_first5_normalized_mae"]["mean"]
            ),
            float(
                metrics["moving"]["cumulative_first5_normalized_mae"]["mean"]
            ),
            step,
        )

    ranked_steps = sorted(args.checkpoint_steps, key=rank_key)
    best_step = ranked_steps[0]
    final_step = max(args.checkpoint_steps)
    boundary_censored = best_step == final_step
    decision = (
        "EXTEND_TRAINING_BEST_IS_FINAL_BOUNDARY"
        if boundary_censored
        else "SELECT_INTERIOR_STAGE1_BEST"
    )
    best_path = checkpoint_model_path(args.train_output_root, best_step)
    report["selection"] = {
        "decision": decision,
        "best_step": best_step,
        "best_checkpoint": str(best_path),
        "boundary_censored": boundary_censored,
        "ranked_steps": ranked_steps,
        "rank_keys": {
            str(step): list(rank_key(step)) for step in ranked_steps
        },
        "hardware_deployment_authorized": False,
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "validation_selection_report.json"
    report_path.write_text(
        json.dumps(to_jsonable(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    csv_path = output_dir / "checkpoint_metrics.csv"
    columns = [
        "step",
        "uniform_delta_first5_mae",
        "uniform_cumulative_first5_mae",
        "uniform_cumulative_first5_normalized_mae",
        "uniform_endpoint5_mae",
        "moving_cumulative_first5_normalized_mae",
        "stationary_endpoint5_normalized_mae",
        "startup_endpoint5_normalized_mae",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for step in args.checkpoint_steps:
            metrics = report["checkpoints"][str(step)]["metrics"]
            writer.writerow(
                {
                    "step": step,
                    "uniform_delta_first5_mae": metrics["uniform"][
                        "delta_first5_mae"
                    ]["mean"],
                    "uniform_cumulative_first5_mae": metrics["uniform"][
                        "cumulative_first5_mae"
                    ]["mean"],
                    "uniform_cumulative_first5_normalized_mae": metrics[
                        "uniform"
                    ]["cumulative_first5_normalized_mae"]["mean"],
                    "uniform_endpoint5_mae": metrics["uniform"][
                        "endpoint5_mae"
                    ]["mean"],
                    "moving_cumulative_first5_normalized_mae": metrics[
                        "moving"
                    ]["cumulative_first5_normalized_mae"]["mean"],
                    "stationary_endpoint5_normalized_mae": metrics[
                        "stationary"
                    ]["endpoint5_normalized_mae"]["mean"],
                    "startup_endpoint5_normalized_mae": metrics["startup"][
                        "endpoint5_normalized_mae"
                    ]["mean"],
                }
            )

    best_path_json = output_dir / "best_checkpoint.json"
    best_path_json.write_text(
        json.dumps(
            to_jsonable(report["selection"]),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\n===== HELD-OUT VALIDATION METRICS (LOWER IS BETTER) =====")
    print(
        "step   uniform_cum5_norm  moving_cum5_norm  "
        "stationary_end5_norm  uniform_delta5"
    )
    for step in args.checkpoint_steps:
        metrics = report["checkpoints"][str(step)]["metrics"]
        print(
            f"{step:6d} "
            f"{metrics['uniform']['cumulative_first5_normalized_mae']['mean']:18.6f} "
            f"{metrics['moving']['cumulative_first5_normalized_mae']['mean']:17.6f} "
            f"{metrics['stationary']['endpoint5_normalized_mae']['mean']:20.6f} "
            f"{metrics['uniform']['delta_first5_mae']['mean']:14.6f}"
        )

    print("\n===== PRE-REGISTERED SELECTION =====")
    print(f"ranked steps: {ranked_steps}")
    print(f"best step: {best_step}")
    print(f"decision: {decision}")
    if boundary_censored:
        print(
            "The best metric is at the final evaluated boundary. "
            "Do not declare it final; extend training first."
        )
    else:
        print(
            "The best metric is an interior checkpoint. It is the stage-1 "
            "offline candidate, not yet a hardware-safety authorization."
        )

    print("\n===== OUTPUT =====")
    print(report_path)
    print(csv_path)
    print(best_path_json)
    print("ACT V3 DELTA HELD-OUT VALIDATION: PASS")
    print("NO DATASET WAS MODIFIED. NO HARDWARE WAS ACCESSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
