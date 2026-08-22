#!/usr/bin/env python
"""Paired offline audit for ACT v1 versus ACT v2.

This script is deliberately hardware-free. It never imports robot, camera, or
motor-control modules. It compares the frozen v1 and v2 datasets at exactly
paired observations:

    v1 source frame = trim_start_source_frame + v2 frame

The v1 15k checkpoint is retained as the established baseline. The four v2
checkpoints are evaluated on the same 60 Home anchors and on deterministic
moving-phase anchors selected only from demonstration actions.

Reported metrics (physical action space):

* Home action[0] MAE and six-joint signed bias.
* First-10 action MAE (indices 0..9; the runtime execution horizon).
* Full 50-action chunk MAE.
* Delta 0->10 MAE (index 10 minus index 0, matching the v1 audit).
* Relative first-10/full-chunk MAE after subtracting action[0]. These are the
  moving-phase trajectory-consistency metrics independent of absolute anchor.

An audit PASS means dataset pairing, checkpoint loading, finite inference, and
report generation passed. It does not mean a checkpoint is physically safe or
that v2 improved; improvement is decided from the emitted metrics.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy


MOTORS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

STATE_KEY = "observation.state"
ACTION_KEY = "action"
FRONT_KEY = "observation.images.front"
WRIST_KEY = "observation.images.wrist"
DEFAULT_V2_STEPS = [5000, 10000, 15000, 20000]


@dataclass(frozen=True)
class PairedSample:
    phase: str
    episode_index: int
    v2_frame: int
    source_frame: int
    v1_global_index: int
    v2_global_index: int
    activity_score: float
    target_chunk: np.ndarray


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--v1-dataset-root",
        type=Path,
        default=Path("data/lerobot/so101_red_cube_pick_place_v1"),
    )
    p.add_argument(
        "--v1-repo-id",
        default="connorluis/so101_red_cube_pick_place_v1",
    )
    p.add_argument(
        "--v2-dataset-root",
        type=Path,
        default=Path("data/lerobot/so101_red_cube_pick_place_v2"),
    )
    p.add_argument(
        "--v2-repo-id",
        default="connorluis/so101_red_cube_pick_place_v2",
    )
    p.add_argument(
        "--plan-json",
        type=Path,
        default=Path(
            "outputs/dataset/so101_red_cube_pick_place_v2_plan/"
            "dataset_v2_plan.json"
        ),
    )
    p.add_argument(
        "--v1-checkpoint",
        type=Path,
        default=Path(
            "outputs/train/act_red_cube_v1/checkpoints/015000/"
            "pretrained_model"
        ),
    )
    p.add_argument(
        "--v2-train-root",
        type=Path,
        default=Path("outputs/train/act_red_cube_v2"),
    )
    p.add_argument(
        "--v2-checkpoint-steps",
        type=int,
        nargs="+",
        default=DEFAULT_V2_STEPS,
    )
    p.add_argument("--video-backend", default="torchcodec")
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--first-action-count", type=int, default=10)
    p.add_argument("--future-offset", type=int, default=10)
    p.add_argument("--moving-samples-per-episode", type=int, default=3)
    p.add_argument("--moving-min-separation", type=int, default=50)
    p.add_argument("--motion-threshold", type=float, default=0.20)
    p.add_argument(
        "--episode-indices",
        type=int,
        nargs="*",
        default=None,
        help="Optional deterministic subset for a smoke run.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/eval/act_paired_offline_v2"),
    )
    args = p.parse_args()

    if args.chunk_size != 50:
        p.error("--chunk-size must remain 50 for this controlled audit")
    if args.first_action_count != 10:
        p.error("--first-action-count must remain 10")
    if not 1 <= args.future_offset < args.chunk_size:
        p.error("--future-offset must be within the action chunk")
    if args.moving_samples_per_episode < 1:
        p.error("--moving-samples-per-episode must be >= 1")
    if args.moving_min_separation < 1:
        p.error("--moving-min-separation must be >= 1")
    if args.motion_threshold <= 0:
        p.error("--motion-threshold must be positive")
    return args


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def load_dataset(
    repo_id: str,
    root: Path,
    video_backend: str,
) -> LeRobotDataset:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    return LeRobotDataset(
        repo_id=repo_id,
        root=root,
        video_backend=video_backend,
    )


def arrays_from_dataset(
    dataset: LeRobotDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hf = dataset.hf_dataset
    states = np.asarray(hf[STATE_KEY], dtype=np.float32)
    actions = np.asarray(hf[ACTION_KEY], dtype=np.float32)
    episodes = np.asarray(hf["episode_index"], dtype=np.int64)
    frames = np.asarray(hf["frame_index"], dtype=np.int64)
    return states, actions, episodes, frames


def build_frame_map(
    episodes: np.ndarray,
    frames: np.ndarray,
) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    for gi, (ep, frame) in enumerate(zip(episodes, frames, strict=True)):
        key = (int(ep), int(frame))
        if key in out:
            raise RuntimeError(f"Duplicate dataset frame key: {key}")
        out[key] = gi
    return out


def image_tensor(x: Any, device: torch.device) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    t = x.detach()
    if t.ndim != 3:
        raise RuntimeError(f"Unexpected image shape: {tuple(t.shape)}")
    if t.shape[0] in (1, 3, 4):
        t = t[:3]
    elif t.shape[-1] in (1, 3, 4):
        t = t[..., :3].permute(2, 0, 1)
    else:
        raise RuntimeError(f"Cannot infer image channels: {tuple(t.shape)}")
    t = t.float()
    if float(t.max()) > 1.5:
        t = t / 255.0
    return t.unsqueeze(0).to(device)


def validate_checkpoint_files(path: Path) -> None:
    required = [
        path / "config.json",
        path / "model.safetensors",
        path / "train_config.json",
    ]
    missing = [p for p in required if not p.is_file() or p.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(
            "Checkpoint files missing/empty:\n"
            + "\n".join(f"  - {p}" for p in missing)
        )


def load_policy(
    path: Path,
    label: str,
    expected_state_dim: int,
    chunk_size: int,
) -> ACTPolicy:
    path = path.resolve()
    validate_checkpoint_files(path)
    print(f"Loading {label}: {path}")
    policy = ACTPolicy.from_pretrained(path, local_files_only=True)
    policy.eval()
    policy.reset()

    feature = policy.config.input_features.get(STATE_KEY)
    if feature is None:
        raise RuntimeError(f"{label}: missing {STATE_KEY} input feature")
    state_shape = tuple(int(x) for x in feature.shape)
    if state_shape != (expected_state_dim,):
        raise RuntimeError(
            f"{label}: state shape={state_shape}, expected={(expected_state_dim,)}"
        )
    if policy.config.chunk_size != chunk_size:
        raise RuntimeError(
            f"{label}: chunk_size={policy.config.chunk_size}, expected={chunk_size}"
        )
    if policy.config.n_action_steps != 10:
        raise RuntimeError(
            f"{label}: n_action_steps={policy.config.n_action_steps}, expected=10"
        )
    return policy


@torch.no_grad()
def infer_one(
    policy: ACTPolicy,
    item: dict[str, Any],
    expected_state_dim: int,
    chunk_size: int,
) -> np.ndarray:
    device = torch.device(policy.config.device)
    state = item[STATE_KEY]
    if not isinstance(state, torch.Tensor):
        state = torch.as_tensor(state)
    state = state.float().reshape(-1)
    if state.numel() != expected_state_dim:
        raise RuntimeError(
            f"State has {state.numel()} values, expected {expected_state_dim}"
        )

    batch = {
        STATE_KEY: state.reshape(1, expected_state_dim).to(device),
        FRONT_KEY: image_tensor(item[FRONT_KEY], device),
        WRIST_KEY: image_tensor(item[WRIST_KEY], device),
    }
    policy.reset()
    chunk = policy.predict_action_chunk(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    arr = chunk.detach().cpu().float().numpy()
    expected_shape = (1, chunk_size, 6)
    if arr.shape != expected_shape:
        raise RuntimeError(f"Chunk shape={arr.shape}, expected={expected_shape}")
    if not np.isfinite(arr).all():
        raise FloatingPointError("NaN/Inf in ACT prediction")
    return arr[0]


def chunk_from_map(
    actions: np.ndarray,
    frame_map: dict[tuple[int, int], int],
    episode: int,
    start_frame: int,
    chunk_size: int,
) -> tuple[np.ndarray, list[int]]:
    indices = []
    for offset in range(chunk_size):
        key = (episode, start_frame + offset)
        if key not in frame_map:
            raise RuntimeError(f"Missing target frame: episode={episode}, frame={key[1]}")
        indices.append(frame_map[key])
    return actions[np.asarray(indices, dtype=np.int64)], indices


def select_moving_frames(
    ep_actions: np.ndarray,
    action_scale: np.ndarray,
    chunk_size: int,
    future_offset: int,
    count: int,
    min_separation: int,
    motion_threshold: float,
) -> list[tuple[int, float]]:
    candidates: list[tuple[float, int]] = []
    last_start = len(ep_actions) - chunk_size
    for frame in range(last_start + 1):
        delta = ep_actions[frame + future_offset] - ep_actions[frame]
        if float(np.max(np.abs(delta))) <= motion_threshold:
            continue
        score = float(np.mean(np.abs(delta) / action_scale))
        candidates.append((score, frame))

    candidates.sort(key=lambda x: (-x[0], x[1]))
    chosen: list[tuple[int, float]] = []
    for score, frame in candidates:
        if all(abs(frame - old_frame) >= min_separation for old_frame, _ in chosen):
            chosen.append((frame, score))
            if len(chosen) == count:
                break
    if len(chosen) != count:
        raise RuntimeError(
            f"Could select only {len(chosen)}/{count} moving frames "
            f"with min_separation={min_separation}"
        )
    return sorted(chosen)


def scalar_summary(values: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size == 0 or not np.isfinite(x).all():
        raise RuntimeError("Cannot summarize empty/non-finite metric array")
    return {
        "count": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p90": float(np.percentile(x, 90)),
        "max": float(np.max(x)),
    }


def per_joint_action0_summary(errors: np.ndarray) -> dict[str, Any]:
    if errors.ndim != 2 or errors.shape[1] != 6:
        raise RuntimeError(f"Unexpected action0 error shape: {errors.shape}")
    out: dict[str, Any] = {}
    for joint, name in enumerate(MOTORS):
        e = errors[:, joint]
        ae = np.abs(e)
        out[name] = {
            "mean_signed_error": float(np.mean(e)),
            "median_signed_error": float(np.median(e)),
            "mae": float(np.mean(ae)),
            "median_abs_error": float(np.median(ae)),
            "p90_abs_error": float(np.percentile(ae, 90)),
            "max_abs_error": float(np.max(ae)),
            "positive_fraction_gt_0p1": float(np.mean(e > 0.10)),
            "negative_fraction_lt_minus_0p1": float(np.mean(e < -0.10)),
        }
    return out


def phase_summary(
    predictions: np.ndarray,
    targets: np.ndarray,
    first_action_count: int,
    future_offset: int,
) -> dict[str, Any]:
    if predictions.shape != targets.shape or predictions.ndim != 3:
        raise RuntimeError(
            f"Prediction/target shapes differ: {predictions.shape} vs {targets.shape}"
        )
    if predictions.shape[1:] != (50, 6):
        raise RuntimeError(f"Unexpected prediction shape: {predictions.shape}")

    error = predictions - targets
    action0_error = error[:, 0]
    action0_mae = np.mean(np.abs(action0_error), axis=1)
    first10_mae = np.mean(
        np.abs(error[:, :first_action_count]), axis=(1, 2)
    )
    full_chunk_mae = np.mean(np.abs(error), axis=(1, 2))

    pred_delta = predictions[:, future_offset] - predictions[:, 0]
    target_delta = targets[:, future_offset] - targets[:, 0]
    delta_mae = np.mean(np.abs(pred_delta - target_delta), axis=1)

    pred_relative = predictions - predictions[:, :1]
    target_relative = targets - targets[:, :1]
    relative_error = pred_relative - target_relative
    relative_first10_mae = np.mean(
        np.abs(relative_error[:, :first_action_count]), axis=(1, 2)
    )
    relative_full_chunk_mae = np.mean(
        np.abs(relative_error), axis=(1, 2)
    )

    return {
        "sample_count": int(predictions.shape[0]),
        "action0_mae": scalar_summary(action0_mae),
        "action0_error_by_joint": per_joint_action0_summary(action0_error),
        "first10_mae": scalar_summary(first10_mae),
        "full_chunk_mae": scalar_summary(full_chunk_mae),
        "delta_0_to_10_mae": scalar_summary(delta_mae),
        "relative_first10_mae": scalar_summary(relative_first10_mae),
        "relative_full_chunk_mae": scalar_summary(relative_full_chunk_mae),
    }


def relative_improvement(baseline: float, candidate: float) -> float | None:
    if abs(baseline) < 1e-12:
        return None
    return float((baseline - candidate) / baseline)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; stopping offline checkpoint audit")

    plan_path = args.plan_json.resolve()
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != "so101_red_cube_pick_place_v2_plan_v1":
        raise RuntimeError(f"Unexpected plan schema: {plan.get('schema_version')}")
    if plan["target"]["planned_episodes"] != 60:
        raise RuntimeError("Plan does not contain 60 episodes")
    if plan["target"]["planned_frames"] != 53382:
        raise RuntimeError("Plan does not contain 53382 target frames")

    plan_by_ep = {
        int(row["episode_index"]): row for row in plan["episodes"]
    }
    if set(plan_by_ep) != set(range(60)):
        raise RuntimeError("Plan episode indices are not exactly 0..59")

    if args.episode_indices is None or len(args.episode_indices) == 0:
        selected_episodes = list(range(60))
    else:
        selected_episodes = sorted(set(args.episode_indices))
        invalid = [ep for ep in selected_episodes if ep not in plan_by_ep]
        if invalid:
            raise RuntimeError(f"Invalid episode indices: {invalid}")

    print("===== LOAD FROZEN DATASETS =====")
    v1 = load_dataset(args.v1_repo_id, args.v1_dataset_root, args.video_backend)
    v2 = load_dataset(args.v2_repo_id, args.v2_dataset_root, args.video_backend)
    v1_states, v1_actions, v1_eps, v1_frames = arrays_from_dataset(v1)
    v2_states, v2_actions, v2_eps, v2_frames = arrays_from_dataset(v2)

    if v1_states.shape != (54000, 6) or v1_actions.shape != (54000, 6):
        raise RuntimeError(
            f"Unexpected v1 shapes: state={v1_states.shape}, action={v1_actions.shape}"
        )
    if v2_states.shape != (53382, 12) or v2_actions.shape != (53382, 6):
        raise RuntimeError(
            f"Unexpected v2 shapes: state={v2_states.shape}, action={v2_actions.shape}"
        )
    if int(v1.meta.total_episodes) != 60 or int(v2.meta.total_episodes) != 60:
        raise RuntimeError("Both datasets must contain exactly 60 episodes")
    if int(v1.fps) != 15 or int(v2.fps) != 15:
        raise RuntimeError("Both datasets must use 15 FPS")

    v1_map = build_frame_map(v1_eps, v1_frames)
    v2_map = build_frame_map(v2_eps, v2_frames)
    action_scale = np.maximum(v2_actions.std(axis=0), 1.0)

    print(f"v1: frames={len(v1_actions)}, state={v1_states.shape[1]}D")
    print(f"v2: frames={len(v2_actions)}, state={v2_states.shape[1]}D")
    print(f"episodes selected: {selected_episodes}")

    home_samples: list[PairedSample] = []
    moving_samples: list[PairedSample] = []
    max_actual_error = 0.0
    max_previous_command_error = 0.0
    max_target_chunk_error = 0.0

    print()
    print("===== BUILD STRICTLY PAIRED SAMPLES =====")
    for ep in selected_episodes:
        row = plan_by_ep[ep]
        trim_start = int(row["trim_start_source_frame"])
        retained = int(row["retained_frames"])

        ep_v2_gi = [v2_map[(ep, frame)] for frame in range(retained)]
        ep_v2_actions = v2_actions[np.asarray(ep_v2_gi, dtype=np.int64)]

        moving = select_moving_frames(
            ep_actions=ep_v2_actions,
            action_scale=action_scale,
            chunk_size=args.chunk_size,
            future_offset=args.future_offset,
            count=args.moving_samples_per_episode,
            min_separation=args.moving_min_separation,
            motion_threshold=args.motion_threshold,
        )
        phase_frames = [("home", 0, 0.0)] + [
            ("moving", frame, score) for frame, score in moving
        ]

        for phase, v2_frame, activity_score in phase_frames:
            source_frame = trim_start + v2_frame
            v1_gi = v1_map[(ep, source_frame)]
            v2_gi = v2_map[(ep, v2_frame)]

            target_v1, _ = chunk_from_map(
                v1_actions, v1_map, ep, source_frame, args.chunk_size
            )
            target_v2, _ = chunk_from_map(
                v2_actions, v2_map, ep, v2_frame, args.chunk_size
            )

            actual_error = float(
                np.max(np.abs(v1_states[v1_gi] - v2_states[v2_gi, :6]))
            )
            target_error = float(np.max(np.abs(target_v1 - target_v2)))

            if source_frame > 0:
                prev_gi = v1_map[(ep, source_frame - 1)]
                expected_previous = v1_actions[prev_gi]
            else:
                expected_previous = np.asarray(
                    row["first_previous_command"], dtype=np.float32
                )
            previous_error = float(
                np.max(
                    np.abs(expected_previous - v2_states[v2_gi, 6:])
                )
            )

            max_actual_error = max(max_actual_error, actual_error)
            max_previous_command_error = max(
                max_previous_command_error, previous_error
            )
            max_target_chunk_error = max(max_target_chunk_error, target_error)

            if actual_error > 1e-6:
                raise RuntimeError(
                    f"Actual-state mismatch ep={ep}, v2_frame={v2_frame}: "
                    f"{actual_error}"
                )
            if previous_error > 1e-6:
                raise RuntimeError(
                    f"Previous-command mismatch ep={ep}, v2_frame={v2_frame}: "
                    f"{previous_error}"
                )
            if target_error > 1e-6:
                raise RuntimeError(
                    f"Action-chunk mismatch ep={ep}, v2_frame={v2_frame}: "
                    f"{target_error}"
                )

            sample = PairedSample(
                phase=phase,
                episode_index=ep,
                v2_frame=v2_frame,
                source_frame=source_frame,
                v1_global_index=v1_gi,
                v2_global_index=v2_gi,
                activity_score=activity_score,
                target_chunk=target_v2.copy(),
            )
            if phase == "home":
                home_samples.append(sample)
            else:
                moving_samples.append(sample)

        print(
            f"ep={ep:02d} trim={trim_start:02d} home=v2:000/v1:{trim_start:03d} "
            f"moving={[frame for frame, _ in moving]}"
        )

    home_targets = np.stack([s.target_chunk for s in home_samples])
    home_commands = home_targets[:, 0]
    locked_home_command = np.median(home_commands, axis=0)
    home_command_spread = float(
        np.max(np.abs(home_commands - locked_home_command[None, :]))
    )

    print()
    print("===== PAIRING INTEGRITY =====")
    print(f"home samples: {len(home_samples)}")
    print(f"moving samples: {len(moving_samples)}")
    print(f"max actual_q error: {max_actual_error:.8f}")
    print(f"max previous_command error: {max_previous_command_error:.8f}")
    print(f"max target chunk error: {max_target_chunk_error:.8f}")
    print(f"startup action[0] spread around median: {home_command_spread:.8f}")

    model_specs: list[tuple[str, str, Path, int, LeRobotDataset]] = [
        ("v1_015000", "v1", args.v1_checkpoint, 6, v1)
    ]
    for step in args.v2_checkpoint_steps:
        ckpt = (
            args.v2_train_root
            / "checkpoints"
            / f"{step:06d}"
            / "pretrained_model"
        )
        model_specs.append((f"v2_{step:06d}", "v2", ckpt, 12, v2))

    report: dict[str, Any] = {
        "schema_version": "act_paired_offline_v2_v1",
        "protocol": {
            "pairing": "v1_source_frame = trim_start_source_frame + v2_frame",
            "home_anchor": (
                "v2 frame 0 (the frozen pre-roll start) for every selected episode; "
                "the per-episode demonstration action is the target"
            ),
            "moving_selection": (
                "Top normalized demonstration delta(0->10), selected with "
                "deterministic minimum frame separation"
            ),
            "first10_definition": "chunk indices 0..9",
            "delta_definition": "action[10] - action[0]",
            "trajectory_consistency_definition": (
                "MAE after subtracting action[0] from predicted and target chunks"
            ),
        },
        "paths": {
            "v1_dataset_root": str(args.v1_dataset_root.resolve()),
            "v2_dataset_root": str(args.v2_dataset_root.resolve()),
            "plan_json": str(plan_path),
        },
        "selected_episodes": selected_episodes,
        "parameters": {
            "chunk_size": args.chunk_size,
            "first_action_count": args.first_action_count,
            "future_offset": args.future_offset,
            "moving_samples_per_episode": args.moving_samples_per_episode,
            "moving_min_separation": args.moving_min_separation,
            "motion_threshold": args.motion_threshold,
        },
        "integrity": {
            "max_actual_q_error": max_actual_error,
            "max_previous_command_error": max_previous_command_error,
            "max_target_chunk_error": max_target_chunk_error,
            "startup_action0_spread_around_median": home_command_spread,
            "median_startup_action0": locked_home_command.tolist(),
        },
        "samples": {
            "home": [
                {
                    "episode_index": s.episode_index,
                    "v2_frame": s.v2_frame,
                    "source_frame": s.source_frame,
                    "v1_global_index": s.v1_global_index,
                    "v2_global_index": s.v2_global_index,
                }
                for s in home_samples
            ],
            "moving": [
                {
                    "episode_index": s.episode_index,
                    "v2_frame": s.v2_frame,
                    "source_frame": s.source_frame,
                    "v1_global_index": s.v1_global_index,
                    "v2_global_index": s.v2_global_index,
                    "activity_score": s.activity_score,
                }
                for s in moving_samples
            ],
        },
        "models": {},
        "comparisons_to_v1_015000": {},
    }

    all_samples = home_samples + moving_samples
    n_home = len(home_samples)
    target_all = np.stack([s.target_chunk for s in all_samples])

    print()
    print("===== MODEL INFERENCE =====")
    for label, dataset_version, ckpt, state_dim, dataset in model_specs:
        policy = load_policy(
            ckpt,
            label,
            expected_state_dim=state_dim,
            chunk_size=args.chunk_size,
        )
        predictions = []
        record_rows = []
        for n, sample in enumerate(all_samples, start=1):
            gi = (
                sample.v1_global_index
                if dataset_version == "v1"
                else sample.v2_global_index
            )
            pred = infer_one(
                policy,
                dataset[int(gi)],
                expected_state_dim=state_dim,
                chunk_size=args.chunk_size,
            )
            predictions.append(pred)

            target = sample.target_chunk
            record_rows.append(
                {
                    "phase": sample.phase,
                    "episode_index": sample.episode_index,
                    "v2_frame": sample.v2_frame,
                    "source_frame": sample.source_frame,
                    "action0_mae": float(
                        np.mean(np.abs(pred[0] - target[0]))
                    ),
                    "first10_mae": float(
                        np.mean(
                            np.abs(
                                pred[: args.first_action_count]
                                - target[: args.first_action_count]
                            )
                        )
                    ),
                    "full_chunk_mae": float(np.mean(np.abs(pred - target))),
                    "delta_0_to_10_mae": float(
                        np.mean(
                            np.abs(
                                (pred[args.future_offset] - pred[0])
                                - (
                                    target[args.future_offset]
                                    - target[0]
                                )
                            )
                        )
                    ),
                    "pred_action0": pred[0].tolist(),
                    "target_action0": target[0].tolist(),
                    "action0_signed_error": (pred[0] - target[0]).tolist(),
                }
            )
            if n % 20 == 0 or n == len(all_samples):
                print(f"{label}: processed {n}/{len(all_samples)}")

        pred_all = np.stack(predictions)
        home_pred = pred_all[:n_home]
        moving_pred = pred_all[n_home:]
        home_target = target_all[:n_home]
        moving_target = target_all[n_home:]

        report["models"][label] = {
            "dataset_version": dataset_version,
            "checkpoint": str(ckpt.resolve()),
            "state_dim": state_dim,
            "home": phase_summary(
                home_pred,
                home_target,
                args.first_action_count,
                args.future_offset,
            ),
            "moving": phase_summary(
                moving_pred,
                moving_target,
                args.first_action_count,
                args.future_offset,
            ),
            "records": record_rows,
        }

        del policy
        del predictions
        del pred_all
        gc.collect()
        torch.cuda.empty_cache()

    baseline = report["models"]["v1_015000"]
    metric_names = [
        "action0_mae",
        "first10_mae",
        "full_chunk_mae",
        "delta_0_to_10_mae",
        "relative_first10_mae",
        "relative_full_chunk_mae",
    ]
    for label, model in report["models"].items():
        if not label.startswith("v2_"):
            continue
        comp: dict[str, Any] = {}
        for phase in ["home", "moving"]:
            comp[phase] = {}
            for metric in metric_names:
                base_value = baseline[phase][metric]["mean"]
                value = model[phase][metric]["mean"]
                comp[phase][metric] = {
                    "v1_015000": base_value,
                    "candidate": value,
                    "absolute_change_candidate_minus_v1": value - base_value,
                    "relative_improvement": relative_improvement(
                        base_value, value
                    ),
                }
        report["comparisons_to_v1_015000"][label] = comp

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "paired_offline_summary.json"
    json_path.write_text(
        json.dumps(to_jsonable(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_path = output_dir / "checkpoint_metrics.csv"
    columns = [
        "model",
        "home_action0_mae",
        "home_first10_mae",
        "home_full_chunk_mae",
        "home_delta_0_to_10_mae",
        "moving_action0_mae",
        "moving_first10_mae",
        "moving_full_chunk_mae",
        "moving_delta_0_to_10_mae",
        "moving_relative_first10_mae",
        "moving_relative_full_chunk_mae",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for label, model in report["models"].items():
            writer.writerow(
                {
                    "model": label,
                    "home_action0_mae": model["home"]["action0_mae"]["mean"],
                    "home_first10_mae": model["home"]["first10_mae"]["mean"],
                    "home_full_chunk_mae": model["home"]["full_chunk_mae"]["mean"],
                    "home_delta_0_to_10_mae": model["home"]["delta_0_to_10_mae"]["mean"],
                    "moving_action0_mae": model["moving"]["action0_mae"]["mean"],
                    "moving_first10_mae": model["moving"]["first10_mae"]["mean"],
                    "moving_full_chunk_mae": model["moving"]["full_chunk_mae"]["mean"],
                    "moving_delta_0_to_10_mae": model["moving"]["delta_0_to_10_mae"]["mean"],
                    "moving_relative_first10_mae": model["moving"]["relative_first10_mae"]["mean"],
                    "moving_relative_full_chunk_mae": model["moving"]["relative_full_chunk_mae"]["mean"],
                }
            )

    print()
    print("===== CHECKPOINT METRICS (LOWER IS BETTER) =====")
    header = (
        "model          home_a0  home_f10  home_full  home_d10  "
        "move_f10  move_full  move_rel_full"
    )
    print(header)
    for label, model in report["models"].items():
        h = model["home"]
        m = model["moving"]
        print(
            f"{label:13s} "
            f"{h['action0_mae']['mean']:8.4f} "
            f"{h['first10_mae']['mean']:9.4f} "
            f"{h['full_chunk_mae']['mean']:10.4f} "
            f"{h['delta_0_to_10_mae']['mean']:9.4f} "
            f"{m['first10_mae']['mean']:9.4f} "
            f"{m['full_chunk_mae']['mean']:10.4f} "
            f"{m['relative_full_chunk_mae']['mean']:13.4f}"
        )

    print()
    print("===== OUTPUT =====")
    print(json_path)
    print(csv_path)
    print("PAIRED OFFLINE ACT V1/V2 AUDIT: PASS")
    print("NO HARDWARE WAS ACCESSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
