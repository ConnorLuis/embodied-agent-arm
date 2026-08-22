#!/usr/bin/env python
"""
Full offline validation for the SO-101 v3 delta-action datasets.

The validator checks the physically separate train and validation datasets
against frozen v1/v2 data, the reviewed v3 plan, and the v3 build report.
It never opens cameras, serial ports, or robot devices and never modifies any
dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


MOTORS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
STATE_NAMES_V3 = [
    *(f"actual.{motor}" for motor in MOTORS),
    *(f"prev_command.{motor}" for motor in MOTORS),
    *(f"prev_command_delta.{motor}" for motor in MOTORS),
]

STATE_KEY = "observation.state"
ACTION_KEY = "action"
FRONT_KEY = "observation.images.front"
WRIST_KEY = "observation.images.wrist"

LOCKED_SPLITS = {
    "train": {"episodes": 48, "frames": 42678},
    "validation": {"episodes": 12, "frames": 10704},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-v1-root",
        type=Path,
        default=Path("data/lerobot/so101_red_cube_pick_place_v1"),
    )
    parser.add_argument(
        "--source-v1-repo-id",
        default="connorluis/so101_red_cube_pick_place_v1",
    )
    parser.add_argument(
        "--reference-v2-root",
        type=Path,
        default=Path("data/lerobot/so101_red_cube_pick_place_v2"),
    )
    parser.add_argument(
        "--reference-v2-repo-id",
        default="connorluis/so101_red_cube_pick_place_v2",
    )
    parser.add_argument(
        "--train-root",
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
        "--validation-root",
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
        "--v2-plan",
        type=Path,
        default=Path(
            "outputs/dataset/so101_red_cube_pick_place_v2_plan/"
            "dataset_v2_plan.json"
        ),
    )
    parser.add_argument(
        "--v3-plan",
        type=Path,
        default=Path(
            "outputs/dataset/so101_red_cube_pick_place_v3_delta_plan/"
            "dataset_v3_delta_plan.json"
        ),
    )
    parser.add_argument(
        "--build-report",
        type=Path,
        default=Path(
            "outputs/dataset/so101_red_cube_pick_place_v3_delta_build/"
            "dataset_v3_delta_build_report.json"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "outputs/dataset/so101_red_cube_pick_place_v3_delta_validation/"
            "dataset_v3_delta_validation_report.json"
        ),
    )
    parser.add_argument("--video-backend", default="torchcodec")
    parser.add_argument("--integrity-atol", type=float, default=1e-5)
    parser.add_argument("--stats-atol", type=float, default=1e-4)
    parser.add_argument("--camera-mae-max", type=float, default=0.10)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {resolved}")
    return value


def load_dataset(
    repo_id: str,
    root: Path,
    video_backend: str | None,
) -> LeRobotDataset:
    resolved = root.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    kwargs: dict[str, Any] = {"repo_id": repo_id, "root": resolved}
    if video_backend:
        kwargs["video_backend"] = video_backend
    try:
        return LeRobotDataset(**kwargs)
    except TypeError:
        kwargs.pop("video_backend", None)
        return LeRobotDataset(**kwargs)


def parquet_arrays(
    dataset: LeRobotDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hf = dataset.hf_dataset
    state = np.asarray(hf[STATE_KEY], dtype=np.float32)
    action = np.asarray(hf[ACTION_KEY], dtype=np.float32)
    episode = np.asarray(hf["episode_index"], dtype=np.int64)
    frame = np.asarray(hf["frame_index"], dtype=np.int64)
    lengths = [len(state), len(action), len(episode), len(frame)]
    if len(set(lengths)) != 1:
        raise RuntimeError(f"Parquet column lengths differ: {lengths}")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise FloatingPointError("State/action contains NaN or Inf.")
    return state, action, episode, frame


def index_map(
    episode: np.ndarray,
    frame: np.ndarray,
) -> dict[tuple[int, int], int]:
    mapping = {
        (int(ep), int(fr)): int(index)
        for index, (ep, fr) in enumerate(zip(episode, frame, strict=True))
    }
    if len(mapping) != len(episode):
        raise RuntimeError("Duplicate (episode_index, frame_index) rows.")
    return mapping


def to_array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return to_jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def image_chw_float(value: Any) -> np.ndarray:
    array = to_array(value)
    if array.ndim != 3:
        raise RuntimeError(f"Unexpected image shape: {array.shape}")
    if array.shape[0] in (1, 3, 4):
        array = array[:3]
    elif array.shape[-1] in (1, 3, 4):
        array = np.transpose(array[..., :3], (2, 0, 1))
    else:
        raise RuntimeError(f"Cannot infer image channels: {array.shape}")
    array = np.asarray(array, dtype=np.float32)
    if float(np.nanmax(array)) > 1.5:
        array = array / 255.0
    if array.shape != (3, 480, 480):
        raise RuntimeError(f"Expected CHW 3x480x480, got {array.shape}")
    if not np.isfinite(array).all():
        raise FloatingPointError("Image contains NaN or Inf.")
    return array


def summarize(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "p50": float(np.percentile(array, 50.0)),
        "p95": float(np.percentile(array, 95.0)),
        "p99": float(np.percentile(array, 99.0)),
        "p99_9": float(np.percentile(array, 99.9)),
        "max": float(np.max(array)),
    }


def basic_dataset_check(
    *,
    split: str,
    dataset: LeRobotDataset,
    root: Path,
    state: np.ndarray,
    action: np.ndarray,
    episode: np.ndarray,
    frame: np.ndarray,
    expected_episodes: int,
    expected_frames: int,
) -> dict[str, Any]:
    unique_episodes = sorted(int(x) for x in np.unique(episode))
    if unique_episodes != list(range(expected_episodes)):
        raise RuntimeError(
            f"{split}: target episode indices are not 0..{expected_episodes - 1}"
        )
    if len(state) != expected_frames:
        raise RuntimeError(
            f"{split}: frame count {len(state)} != {expected_frames}"
        )
    if state.shape != (expected_frames, 18):
        raise RuntimeError(f"{split}: state shape {state.shape}")
    if action.shape != (expected_frames, 6):
        raise RuntimeError(f"{split}: action shape {action.shape}")
    if float(dataset.fps) != 15.0:
        raise RuntimeError(f"{split}: FPS {dataset.fps} != 15")
    if set(dataset.meta.camera_keys) != {FRONT_KEY, WRIST_KEY}:
        raise RuntimeError(
            f"{split}: unexpected camera keys {dataset.meta.camera_keys}"
        )

    for target_ep in unique_episodes:
        indices = np.flatnonzero(episode == target_ep)
        if not np.array_equal(
            frame[indices], np.arange(len(indices), dtype=np.int64)
        ):
            raise RuntimeError(
                f"{split}: episode {target_ep} frame_index is not contiguous."
            )

    video_files = sorted((root.resolve() / "videos").rglob("*.mp4"))
    expected_video_files = expected_episodes * 2
    if len(video_files) != expected_video_files:
        raise RuntimeError(
            f"{split}: video file count {len(video_files)} "
            f"!= {expected_video_files}"
        )
    empty_videos = [str(path) for path in video_files if path.stat().st_size == 0]
    if empty_videos:
        raise RuntimeError(f"{split}: empty video files: {empty_videos[:5]}")

    return {
        "root": str(root.resolve()),
        "episodes": expected_episodes,
        "frames": expected_frames,
        "state_shape": list(state.shape),
        "action_shape": list(action.shape),
        "fps": float(dataset.fps),
        "camera_keys": list(dataset.meta.camera_keys),
        "video_files": len(video_files),
        "video_bytes": int(sum(path.stat().st_size for path in video_files)),
    }


def metadata_feature_check(
    split: str,
    dataset: LeRobotDataset,
) -> dict[str, Any]:
    features = dataset.meta.features
    state_feature = features[STATE_KEY]
    action_feature = features[ACTION_KEY]
    if tuple(state_feature["shape"]) != (18,):
        raise RuntimeError(f"{split}: metadata state is not 18D.")
    if list(state_feature["names"]) != STATE_NAMES_V3:
        raise RuntimeError(f"{split}: state feature names differ from v3 contract.")
    if tuple(action_feature["shape"]) != (6,):
        raise RuntimeError(f"{split}: metadata action is not 6D.")
    if list(action_feature["names"]) != MOTORS:
        raise RuntimeError(f"{split}: action feature names differ from contract.")
    return {
        "state": state_feature,
        "action": action_feature,
        "front": features[FRONT_KEY],
        "wrist": features[WRIST_KEY],
    }


def strict_metadata_stats_check(
    *,
    split: str,
    dataset: LeRobotDataset,
    action: np.ndarray,
    atol: float,
) -> dict[str, Any]:
    metadata_stats = dataset.meta.stats[ACTION_KEY]
    computed = {
        "min": np.min(action.astype(np.float64), axis=0),
        "max": np.max(action.astype(np.float64), axis=0),
        "mean": np.mean(action.astype(np.float64), axis=0),
    }
    checked: dict[str, Any] = {}
    for key, expected in computed.items():
        if key not in metadata_stats:
            continue
        actual = np.asarray(to_array(metadata_stats[key]), dtype=np.float64).reshape(-1)
        if actual.shape != (6,):
            raise RuntimeError(
                f"{split}: metadata action {key} shape is {actual.shape}"
            )
        error = float(np.max(np.abs(actual - expected)))
        if error > atol:
            raise RuntimeError(
                f"{split}: metadata action {key} error {error} > {atol}"
            )
        checked[key] = {
            "metadata": actual.tolist(),
            "computed": expected.tolist(),
            "max_error": error,
        }
    if "min" not in checked or "max" not in checked:
        raise RuntimeError(
            f"{split}: metadata action stats do not expose min/max."
        )
    return checked


def semantic_check(
    *,
    split: str,
    target_state: np.ndarray,
    target_action: np.ndarray,
    target_episode: np.ndarray,
    target_frame: np.ndarray,
    v2_state: np.ndarray,
    v2_action: np.ndarray,
    v2_map: dict[tuple[int, int], int],
    mappings: list[dict[str, Any]],
    atol: float,
) -> dict[str, Any]:
    target_map = index_map(target_episode, target_frame)
    max_actual_error = 0.0
    max_previous_error = 0.0
    max_previous_delta_error = 0.0
    max_action_delta_error = 0.0
    max_direct_decode_error = 0.0
    max_recursive_decode_error = 0.0
    max_startup_action_delta = 0.0
    max_startup_previous_delta = 0.0
    checked = 0

    expected_target_eps = sorted(int(row["target_episode_index"]) for row in mappings)
    if expected_target_eps != list(range(len(mappings))):
        raise RuntimeError(f"{split}: build mapping target episodes are invalid.")

    for mapping in sorted(mappings, key=lambda row: row["target_episode_index"]):
        source_ep = int(mapping["source_episode_index"])
        target_ep = int(mapping["target_episode_index"])
        expected_frames = int(mapping["frames"])
        target_indices = np.flatnonzero(target_episode == target_ep)
        if len(target_indices) != expected_frames:
            raise RuntimeError(
                f"{split} target ep={target_ep}: frames {len(target_indices)} "
                f"!= {expected_frames}"
            )

        running_command: np.ndarray | None = None
        for local_frame in range(expected_frames):
            target_global = target_map[(target_ep, local_frame)]
            v2_global = v2_map[(source_ep, local_frame)]
            actual_q = v2_state[v2_global, :6]
            previous_command = v2_state[v2_global, 6:12]
            absolute_command = v2_action[v2_global]
            if local_frame == 0:
                previous_delta = np.zeros(6, dtype=np.float32)
            else:
                previous_v2_global = v2_map[(source_ep, local_frame - 1)]
                previous_delta = (
                    previous_command - v2_state[previous_v2_global, 6:12]
                ).astype(np.float32)
            action_delta = (absolute_command - previous_command).astype(np.float32)

            actual_error = float(
                np.max(np.abs(target_state[target_global, :6] - actual_q))
            )
            previous_error = float(
                np.max(
                    np.abs(target_state[target_global, 6:12] - previous_command)
                )
            )
            previous_delta_error = float(
                np.max(
                    np.abs(target_state[target_global, 12:18] - previous_delta)
                )
            )
            action_delta_error = float(
                np.max(np.abs(target_action[target_global] - action_delta))
            )
            direct_decode_error = float(
                np.max(
                    np.abs(
                        target_state[target_global, 6:12].astype(np.float64)
                        + target_action[target_global].astype(np.float64)
                        - absolute_command.astype(np.float64)
                    )
                )
            )
            if running_command is None:
                running_command = previous_command.astype(np.float64)
            running_command = (
                running_command + target_action[target_global].astype(np.float64)
            )
            recursive_decode_error = float(
                np.max(
                    np.abs(
                        running_command - absolute_command.astype(np.float64)
                    )
                )
            )

            errors = [
                actual_error,
                previous_error,
                previous_delta_error,
                action_delta_error,
                direct_decode_error,
                recursive_decode_error,
            ]
            if max(errors) > atol:
                raise RuntimeError(
                    f"{split} target_ep={target_ep} frame={local_frame}: "
                    f"semantic error {max(errors)} > {atol}"
                )

            max_actual_error = max(max_actual_error, actual_error)
            max_previous_error = max(max_previous_error, previous_error)
            max_previous_delta_error = max(
                max_previous_delta_error, previous_delta_error
            )
            max_action_delta_error = max(max_action_delta_error, action_delta_error)
            max_direct_decode_error = max(
                max_direct_decode_error, direct_decode_error
            )
            max_recursive_decode_error = max(
                max_recursive_decode_error, recursive_decode_error
            )
            if local_frame == 0:
                max_startup_action_delta = max(
                    max_startup_action_delta,
                    float(np.max(np.abs(target_action[target_global]))),
                )
                max_startup_previous_delta = max(
                    max_startup_previous_delta,
                    float(np.max(np.abs(target_state[target_global, 12:18]))),
                )
            checked += 1

    return {
        "checked_frames": checked,
        "max_actual_q_error": max_actual_error,
        "max_previous_command_error": max_previous_error,
        "max_previous_command_delta_error": max_previous_delta_error,
        "max_action_delta_error": max_action_delta_error,
        "max_direct_delta_decode_error": max_direct_decode_error,
        "max_recursive_delta_decode_error": max_recursive_decode_error,
        "max_startup_action_delta": max_startup_action_delta,
        "max_startup_previous_command_delta": max_startup_previous_delta,
    }


def locked_contract_check(
    *,
    v2_plan: dict[str, Any],
    v3_plan: dict[str, Any],
    build_report: dict[str, Any],
) -> dict[str, Any]:
    if int(v2_plan["summary"]["retained_total_frames"]) != 53382:
        raise RuntimeError("Frozen v2 plan retained frame count is not 53382.")

    plan_rows = v2_plan["episodes"]
    if len(plan_rows) != 60:
        raise RuntimeError(f"Frozen v2 plan has {len(plan_rows)} episodes, not 60.")
    plan_by_source_ep = {
        int(row["episode_index"]): row for row in plan_rows
    }
    if set(plan_by_source_ep) != set(range(60)):
        raise RuntimeError("Frozen v2 plan does not cover source episodes 0..59.")

    mappings = build_report["episode_mapping"]
    if len(mappings) != 60:
        raise RuntimeError(f"Build report has {len(mappings)} mappings, not 60.")
    mapping_by_source_ep = {
        int(row["source_episode_index"]): row for row in mappings
    }
    if set(mapping_by_source_ep) != set(range(60)):
        raise RuntimeError(
            "Build report mappings do not uniquely cover source episodes 0..59."
        )

    train_source = set(int(x) for x in v3_plan["split"]["train_episodes"])
    validation_source = set(
        int(x) for x in v3_plan["split"]["validation_episodes"]
    )
    if train_source & validation_source or train_source | validation_source != set(
        range(60)
    ):
        raise RuntimeError("Reviewed v3 split is not disjoint and complete.")

    next_target_ep = {"train": 0, "validation": 0}
    totals = {
        "train": {"episodes": 0, "frames": 0},
        "validation": {"episodes": 0, "frames": 0},
    }
    for source_ep in range(60):
        plan_row = plan_by_source_ep[source_ep]
        mapping = mapping_by_source_ep[source_ep]
        expected_split = "train" if source_ep in train_source else "validation"
        if mapping["split"] != expected_split:
            raise RuntimeError(
                f"source ep={source_ep}: build split {mapping['split']} "
                f"!= reviewed split {expected_split}"
            )
        if int(mapping["target_episode_index"]) != next_target_ep[expected_split]:
            raise RuntimeError(
                f"source ep={source_ep}: target episode order is invalid."
            )
        if int(mapping["trim_start_source_frame"]) != int(
            plan_row["trim_start_source_frame"]
        ):
            raise RuntimeError(
                f"source ep={source_ep}: trim start differs from frozen v2 plan."
            )
        if int(mapping["frames"]) != int(plan_row["retained_frames"]):
            raise RuntimeError(
                f"source ep={source_ep}: retained frames differ from frozen v2 plan."
            )
        next_target_ep[expected_split] += 1
        totals[expected_split]["episodes"] += 1
        totals[expected_split]["frames"] += int(mapping["frames"])

    for split, locked in LOCKED_SPLITS.items():
        reviewed = {
            "episodes": int(v3_plan["split"][f"{split}_episode_count"]),
            "frames": int(v3_plan["split"][f"{split}_frame_count"]),
        }
        reported = {
            "episodes": int(build_report["split_outputs"][split]["episodes"]),
            "frames": int(build_report["split_outputs"][split]["frames"]),
        }
        if reviewed != locked:
            raise RuntimeError(
                f"Reviewed v3 {split} counts {reviewed} != locked {locked}."
            )
        if reported != locked:
            raise RuntimeError(
                f"Build report {split} counts {reported} != locked {locked}."
            )
        if totals[split] != locked:
            raise RuntimeError(
                f"Mapping-derived {split} counts {totals[split]} != locked {locked}."
            )

    return {
        "v2_plan_episodes": len(plan_rows),
        "v2_plan_frames": 53382,
        "mapping_rows": len(mappings),
        "derived_split_counts": totals,
        "all_mapping_rows_match_frozen_v2_plan": True,
    }


def camera_sanity(
    *,
    split: str,
    target_dataset: LeRobotDataset,
    target_episode: np.ndarray,
    target_frame: np.ndarray,
    source_v1: LeRobotDataset,
    source_v1_map: dict[tuple[int, int], int],
    mappings: list[dict[str, Any]],
    same_camera_mae_max: float,
) -> list[dict[str, Any]]:
    target_map = index_map(target_episode, target_frame)
    mapping_positions = sorted({0, len(mappings) // 2, len(mappings) - 1})
    records: list[dict[str, Any]] = []

    for mapping_position in mapping_positions:
        mapping = sorted(
            mappings, key=lambda row: row["target_episode_index"]
        )[mapping_position]
        source_ep = int(mapping["source_episode_index"])
        target_ep = int(mapping["target_episode_index"])
        trim_start = int(mapping["trim_start_source_frame"])
        frame_count = int(mapping["frames"])
        local_frames = sorted({0, frame_count // 2, frame_count - 1})

        for local_frame in local_frames:
            target_global = target_map[(target_ep, local_frame)]
            source_global = source_v1_map[(source_ep, trim_start + local_frame)]
            target_item = target_dataset[target_global]
            source_item = source_v1[source_global]

            target_front = image_chw_float(target_item[FRONT_KEY])
            target_wrist = image_chw_float(target_item[WRIST_KEY])
            source_front = image_chw_float(source_item[FRONT_KEY])
            source_wrist = image_chw_float(source_item[WRIST_KEY])

            front_same_mae = float(np.mean(np.abs(target_front - source_front)))
            front_cross_mae = float(np.mean(np.abs(target_front - source_wrist)))
            wrist_same_mae = float(np.mean(np.abs(target_wrist - source_wrist)))
            wrist_cross_mae = float(np.mean(np.abs(target_wrist - source_front)))
            if not front_same_mae < front_cross_mae:
                raise RuntimeError(
                    f"{split} target_ep={target_ep} frame={local_frame}: "
                    "front camera identity check failed."
                )
            if not wrist_same_mae < wrist_cross_mae:
                raise RuntimeError(
                    f"{split} target_ep={target_ep} frame={local_frame}: "
                    "wrist camera identity check failed."
                )
            if max(front_same_mae, wrist_same_mae) > same_camera_mae_max:
                raise RuntimeError(
                    f"{split} target_ep={target_ep} frame={local_frame}: "
                    "same-camera source/target MAE is unexpectedly high: "
                    f"front={front_same_mae}, wrist={wrist_same_mae}, "
                    f"limit={same_camera_mae_max}."
                )

            records.append(
                {
                    "target_episode_index": target_ep,
                    "target_frame_index": local_frame,
                    "source_episode_index": source_ep,
                    "source_frame_index": trim_start + local_frame,
                    "front_same_camera_mae": front_same_mae,
                    "front_cross_camera_mae": front_cross_mae,
                    "wrist_same_camera_mae": wrist_same_mae,
                    "wrist_cross_camera_mae": wrist_cross_mae,
                }
            )
            print(
                f"{split} ep={target_ep:02d} frame={local_frame:03d}: "
                f"front_same={front_same_mae:.5f} "
                f"front_cross={front_cross_mae:.5f} "
                f"wrist_same={wrist_same_mae:.5f} "
                f"wrist_cross={wrist_cross_mae:.5f} PASS"
            )
    return records


def compare_to_reviewed_plan(
    *,
    train_action: np.ndarray,
    validation_action: np.ndarray,
    v3_plan: dict[str, Any],
    atol: float,
) -> dict[str, Any]:
    combined = np.concatenate([train_action, validation_action], axis=0)
    per_frame = np.max(np.abs(combined.astype(np.float64)), axis=1)
    computed = summarize(per_frame)
    expected = v3_plan["command_delta"]["all_max_joint_abs_per_frame"]
    for key in ("mean", "p50", "p95", "p99_9", "max"):
        error = abs(float(computed[key]) - float(expected[key]))
        if error > atol:
            raise RuntimeError(
                f"Combined delta statistic {key} differs from v3 plan: "
                f"error={error} > {atol}"
            )
    return {"computed": computed, "reviewed_plan": expected}


def main() -> int:
    args = parse_args()
    if (
        args.integrity_atol <= 0.0
        or args.stats_atol <= 0.0
        or args.camera_mae_max <= 0.0
    ):
        raise ValueError("Tolerances must be positive.")

    v2_plan = load_json(args.v2_plan)
    v3_plan = load_json(args.v3_plan)
    build_report = load_json(args.build_report)
    if v3_plan.get("status") != "PASS":
        raise RuntimeError("Reviewed v3 plan is not PASS.")
    if build_report.get("status") != "BUILD_COMPLETE_REQUIRES_VALIDATION":
        raise RuntimeError("Build report is not awaiting validation.")
    if build_report.get("hardware_accessed") is not False:
        raise RuntimeError("Build report hardware_accessed contract is invalid.")

    locked_contract = locked_contract_check(
        v2_plan=v2_plan,
        v3_plan=v3_plan,
        build_report=build_report,
    )

    print("===== LOAD FROZEN DATASETS =====")
    source_v1 = load_dataset(
        args.source_v1_repo_id, args.source_v1_root, args.video_backend
    )
    reference_v2 = load_dataset(
        args.reference_v2_repo_id, args.reference_v2_root, args.video_backend
    )
    train = load_dataset(args.train_repo_id, args.train_root, args.video_backend)
    validation = load_dataset(
        args.validation_repo_id, args.validation_root, args.video_backend
    )

    source_v1_state, source_v1_action, source_v1_ep, source_v1_fr = (
        parquet_arrays(source_v1)
    )
    v2_state, v2_action, v2_ep, v2_fr = parquet_arrays(reference_v2)
    train_state, train_action, train_ep, train_fr = parquet_arrays(train)
    val_state, val_action, val_ep, val_fr = parquet_arrays(validation)

    if (
        source_v1_state.shape != (54000, 6)
        or source_v1_action.shape != (54000, 6)
        or set(int(x) for x in np.unique(source_v1_ep)) != set(range(60))
    ):
        raise RuntimeError("Frozen v1 shapes/episodes differ from the contract.")
    del source_v1_state, source_v1_action
    if (
        v2_state.shape != (53382, 12)
        or v2_action.shape != (53382, 6)
        or set(int(x) for x in np.unique(v2_ep)) != set(range(60))
    ):
        raise RuntimeError("Frozen v2 shapes differ from the reviewed contract.")

    basic_train = basic_dataset_check(
        split="train",
        dataset=train,
        root=args.train_root,
        state=train_state,
        action=train_action,
        episode=train_ep,
        frame=train_fr,
        expected_episodes=LOCKED_SPLITS["train"]["episodes"],
        expected_frames=LOCKED_SPLITS["train"]["frames"],
    )
    basic_validation = basic_dataset_check(
        split="validation",
        dataset=validation,
        root=args.validation_root,
        state=val_state,
        action=val_action,
        episode=val_ep,
        frame=val_fr,
        expected_episodes=LOCKED_SPLITS["validation"]["episodes"],
        expected_frames=LOCKED_SPLITS["validation"]["frames"],
    )
    print(
        f"train: episodes={basic_train['episodes']} "
        f"frames={basic_train['frames']} videos={basic_train['video_files']}"
    )
    print(
        f"validation: episodes={basic_validation['episodes']} "
        f"frames={basic_validation['frames']} "
        f"videos={basic_validation['video_files']}"
    )

    print("\n===== FEATURE METADATA =====")
    feature_train = metadata_feature_check("train", train)
    feature_validation = metadata_feature_check("validation", validation)
    print("train metadata: PASS")
    print("validation metadata: PASS")

    mappings = build_report["episode_mapping"]
    train_mappings = [row for row in mappings if row["split"] == "train"]
    validation_mappings = [
        row for row in mappings if row["split"] == "validation"
    ]
    source_train_eps = {int(row["source_episode_index"]) for row in train_mappings}
    source_validation_eps = {
        int(row["source_episode_index"]) for row in validation_mappings
    }
    if source_train_eps & source_validation_eps:
        raise RuntimeError("Source train/validation episode sets overlap.")
    if source_train_eps | source_validation_eps != set(range(60)):
        raise RuntimeError("Source split does not cover episodes 0..59.")
    if source_train_eps != set(v3_plan["split"]["train_episodes"]):
        raise RuntimeError("Train split differs from reviewed v3 plan.")
    if source_validation_eps != set(v3_plan["split"]["validation_episodes"]):
        raise RuntimeError("Validation split differs from reviewed v3 plan.")

    print("\n===== FULL PARQUET SEMANTIC CHECK =====")
    v2_map = index_map(v2_ep, v2_fr)
    semantic_train = semantic_check(
        split="train",
        target_state=train_state,
        target_action=train_action,
        target_episode=train_ep,
        target_frame=train_fr,
        v2_state=v2_state,
        v2_action=v2_action,
        v2_map=v2_map,
        mappings=train_mappings,
        atol=args.integrity_atol,
    )
    semantic_validation = semantic_check(
        split="validation",
        target_state=val_state,
        target_action=val_action,
        target_episode=val_ep,
        target_frame=val_fr,
        v2_state=v2_state,
        v2_action=v2_action,
        v2_map=v2_map,
        mappings=validation_mappings,
        atol=args.integrity_atol,
    )
    for split, result in (
        ("train", semantic_train),
        ("validation", semantic_validation),
    ):
        print(
            f"{split}: checked={result['checked_frames']} "
            f"actual={result['max_actual_q_error']:.10f} "
            f"prev={result['max_previous_command_error']:.10f} "
            f"prev_delta={result['max_previous_command_delta_error']:.10f} "
            f"action_delta={result['max_action_delta_error']:.10f} "
            f"recursive={result['max_recursive_delta_decode_error']:.10f}"
        )
    print("FULL PARQUET SEMANTICS: PASS")

    print("\n===== TRAIN-ONLY METADATA STATS =====")
    stats_train = strict_metadata_stats_check(
        split="train",
        dataset=train,
        action=train_action,
        atol=args.stats_atol,
    )
    stats_validation = strict_metadata_stats_check(
        split="validation",
        dataset=validation,
        action=val_action,
        atol=args.stats_atol,
    )
    print("train action metadata stats match train parquet: PASS")
    print("validation action metadata stats match validation parquet: PASS")
    print("train and validation normalization populations are physically separate.")

    plan_comparison = compare_to_reviewed_plan(
        train_action=train_action,
        validation_action=val_action,
        v3_plan=v3_plan,
        atol=args.stats_atol,
    )
    print("combined delta distribution matches reviewed v3 plan: PASS")

    print("\n===== CAMERA / RANDOM ACCESS SANITY =====")
    source_v1_map = index_map(source_v1_ep, source_v1_fr)
    camera_train = camera_sanity(
        split="train",
        target_dataset=train,
        target_episode=train_ep,
        target_frame=train_fr,
        source_v1=source_v1,
        source_v1_map=source_v1_map,
        mappings=train_mappings,
        same_camera_mae_max=args.camera_mae_max,
    )
    camera_validation = camera_sanity(
        split="validation",
        target_dataset=validation,
        target_episode=val_ep,
        target_frame=val_fr,
        source_v1=source_v1,
        source_v1_map=source_v1_map,
        mappings=validation_mappings,
        same_camera_mae_max=args.camera_mae_max,
    )
    print("CAMERA IDENTITY / DECODE: PASS")

    report = {
        "schema_version": "so101_red_cube_pick_place_v3_delta_validation_v1",
        "status": "PASS",
        "scope": {
            "full_parquet_semantic_check": True,
            "camera_random_access_check": True,
            "dataset_modified": False,
            "hardware_accessed": False,
        },
        "locked_contract": locked_contract,
        "basic": {"train": basic_train, "validation": basic_validation},
        "features": {
            "train": feature_train,
            "validation": feature_validation,
        },
        "split": {
            "train_source_episodes": sorted(source_train_eps),
            "validation_source_episodes": sorted(source_validation_eps),
            "disjoint": True,
            "complete": True,
        },
        "semantics": {
            "train": semantic_train,
            "validation": semantic_validation,
        },
        "metadata_action_stats": {
            "train": stats_train,
            "validation": stats_validation,
        },
        "reviewed_plan_delta_distribution": plan_comparison,
        "camera_samples": {
            "train": camera_train,
            "validation": camera_validation,
        },
        "training_contract_after_validation": {
            "dataset_root": str(args.train_root.resolve()),
            "repo_id": args.train_repo_id,
            "validation_root": str(args.validation_root.resolve()),
            "validation_repo_id": args.validation_repo_id,
            "observation_state_dim": 18,
            "action_dim": 6,
            "action_semantics": "per-frame command delta",
            "chunk_size": 10,
            "n_action_steps": 5,
        },
    }
    output_path = args.output_json.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(to_jsonable(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n===== OUTPUT =====")
    print(output_path)
    print("DATASET V3 DELTA VALIDATION: PASS")
    print("NO DATASET WAS MODIFIED. NO HARDWARE WAS ACCESSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
