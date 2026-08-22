#!/usr/bin/env python
"""
Read-only planning audit for the SO-101 red-cube v3 delta-action dataset.

This script:
  * reads the frozen v2 LeRobot dataset from local disk;
  * never opens cameras, serial ports, or robot devices;
  * never modifies v1/v2 dataset files;
  * proposes the v3 causal schema

        observation.state[t] = [
            actual_q[t],                 # 6D
            previous_command[t],         # 6D = command[t-1]
            previous_command_delta[t],   # 6D
        ]

        action[t] = command_delta[t]
                  = command[t] - previous_command[t]

  * audits v2 causal integrity;
  * measures per-frame command-delta distributions and stationary fractions;
  * compares a fixed episode-level train/validation split;
  * summarizes 5/10/20/50-frame human motion horizons;
  * writes planning reports only (no v3 dataset is built).

The empirical guardrail values written by this script are dataset diagnostics,
not certified robot or servo safety limits.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

DEFAULT_VALIDATION_EPISODES = [4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59]
DEFAULT_HORIZONS = [5, 10, 20, 50]
DEFAULT_STATIONARY_THRESHOLDS = [0.01, 0.05, 0.10, 0.20]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only v3 delta-action dataset planning audit."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/lerobot/so101_red_cube_pick_place_v2"),
    )
    parser.add_argument(
        "--dataset-repo-id",
        default="connorluis/so101_red_cube_pick_place_v2",
    )
    parser.add_argument("--video-backend", default="torchcodec")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/dataset/so101_red_cube_pick_place_v3_delta_plan"
        ),
    )
    parser.add_argument(
        "--validation-episodes",
        type=int,
        nargs="+",
        default=DEFAULT_VALIDATION_EPISODES,
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=DEFAULT_HORIZONS,
    )
    parser.add_argument(
        "--stationary-thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_STATIONARY_THRESHOLDS,
    )
    parser.add_argument("--expected-frames", type=int, default=53382)
    parser.add_argument("--expected-episodes", type=int, default=60)
    parser.add_argument("--expected-fps", type=float, default=15.0)
    parser.add_argument("--integrity-atol", type=float, default=1e-6)
    parser.add_argument(
        "--guardrail-quantile",
        type=float,
        default=99.9,
        help="Percentile used only for an empirical diagnostic guardrail.",
    )
    parser.add_argument(
        "--guardrail-multiplier",
        type=float,
        default=1.25,
        help="Multiplier used only for an empirical diagnostic guardrail.",
    )
    return parser.parse_args()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def stack_vector_column(
    column: Iterable[Any],
    *,
    name: str,
    expected_dim: int,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for row_index, value in enumerate(column):
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
        else:
            array = np.asarray(value)
        array = np.asarray(array, dtype=np.float64).reshape(-1)
        if array.size != expected_dim:
            raise AssertionError(
                f"{name}[{row_index}] has {array.size} values; "
                f"expected {expected_dim}."
            )
        rows.append(array)
    if not rows:
        raise AssertionError(f"{name} is empty.")
    result = np.stack(rows, axis=0)
    if not np.isfinite(result).all():
        bad = np.argwhere(~np.isfinite(result))[0]
        raise AssertionError(
            f"{name} contains NaN/Inf at row={int(bad[0])}, "
            f"column={int(bad[1])}."
        )
    return result


def scalar_int_column(column: Iterable[Any], *, name: str) -> np.ndarray:
    values: list[int] = []
    for row_index, value in enumerate(column):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().reshape(-1).item()
        array = np.asarray(value).reshape(-1)
        if array.size != 1:
            raise AssertionError(f"{name}[{row_index}] is not scalar.")
        values.append(int(array[0]))
    return np.asarray(values, dtype=np.int64)


def summarize(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty array.")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "p50": float(np.percentile(array, 50.0)),
        "p90": float(np.percentile(array, 90.0)),
        "p95": float(np.percentile(array, 95.0)),
        "p99": float(np.percentile(array, 99.0)),
        "p99_5": float(np.percentile(array, 99.5)),
        "p99_9": float(np.percentile(array, 99.9)),
        "max": float(np.max(array)),
    }


def format_vector(values: np.ndarray, digits: int = 6) -> str:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return "[" + ", ".join(f"{x:.{digits}f}" for x in array) + "]"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows supplied for {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def subset_mask(episode_indices: np.ndarray, episodes: list[int]) -> np.ndarray:
    return np.isin(episode_indices, np.asarray(episodes, dtype=np.int64))


def split_joint_rows(
    command_delta: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    guardrail_quantile: float,
    guardrail_multiplier: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stats_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    for joint_index, joint_name in enumerate(JOINT_NAMES):
        for split_name, mask in (
            ("all", np.ones(len(command_delta), dtype=bool)),
            ("train", train_mask),
            ("validation", validation_mask),
        ):
            signed = command_delta[mask, joint_index]
            absolute = np.abs(signed)
            signed_summary = summarize(signed)
            absolute_summary = summarize(absolute)
            row: dict[str, Any] = {
                "split": split_name,
                "joint_index": joint_index,
                "joint": joint_name,
                "signed_mean": signed_summary["mean"],
                "signed_std": signed_summary["std"],
            }
            row.update({f"abs_{key}": value for key, value in absolute_summary.items()})
            if split_name == "train":
                row["empirical_guardrail_candidate"] = float(
                    np.percentile(absolute, guardrail_quantile)
                    * guardrail_multiplier
                )
            else:
                row["empirical_guardrail_candidate"] = ""
            stats_rows.append(row)

        train_abs = np.abs(command_delta[train_mask, joint_index])
        validation_abs = np.abs(command_delta[validation_mask, joint_index])
        train_p95 = float(np.percentile(train_abs, 95.0))
        validation_p95 = float(np.percentile(validation_abs, 95.0))
        train_p99 = float(np.percentile(train_abs, 99.0))
        validation_p99 = float(np.percentile(validation_abs, 99.0))
        eps = 1e-12
        comparison_rows.append(
            {
                "joint_index": joint_index,
                "joint": joint_name,
                "train_abs_mean": float(np.mean(train_abs)),
                "validation_abs_mean": float(np.mean(validation_abs)),
                "validation_to_train_abs_mean_ratio": float(
                    np.mean(validation_abs) / max(np.mean(train_abs), eps)
                ),
                "train_abs_p95": train_p95,
                "validation_abs_p95": validation_p95,
                "validation_to_train_p95_ratio": float(
                    validation_p95 / max(train_p95, eps)
                ),
                "train_abs_p99": train_p99,
                "validation_abs_p99": validation_p99,
                "validation_to_train_p99_ratio": float(
                    validation_p99 / max(train_p99, eps)
                ),
            }
        )

    return stats_rows, comparison_rows


def horizon_summaries(
    actions: np.ndarray,
    previous_commands: np.ndarray,
    command_delta: np.ndarray,
    episode_rows: dict[int, np.ndarray],
    horizons: list[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []

    for horizon in horizons:
        max_joint_displacements: list[float] = []
        l2_displacements: list[float] = []
        path_lengths: list[float] = []
        home_displacements: list[np.ndarray] = []

        for indices in episode_rows.values():
            episode_length = len(indices)
            if episode_length < horizon:
                continue

            first_global = int(indices[0])
            home_displacements.append(
                actions[int(indices[horizon - 1])]
                - previous_commands[first_global]
            )

            for local_start in range(0, episode_length - horizon + 1):
                window = indices[local_start : local_start + horizon]
                displacement = (
                    actions[int(window[-1])]
                    - previous_commands[int(window[0])]
                )
                max_joint_displacements.append(float(np.max(np.abs(displacement))))
                l2_displacements.append(float(np.linalg.norm(displacement, ord=2)))
                path_lengths.append(
                    float(np.sum(np.linalg.norm(command_delta[window], axis=1)))
                )

        if not max_joint_displacements:
            raise AssertionError(f"No valid windows for horizon={horizon}.")

        home_array = np.stack(home_displacements, axis=0)
        horizon_report = {
            "seconds_at_dataset_fps": None,
            "valid_window_count": len(max_joint_displacements),
            "max_joint_abs_displacement": summarize(
                np.asarray(max_joint_displacements)
            ),
            "l2_displacement": summarize(np.asarray(l2_displacements)),
            "joint_space_path_length": summarize(np.asarray(path_lengths)),
            "home_anchor": {
                "episode_count": int(home_array.shape[0]),
                "mean_displacement_by_joint": np.mean(home_array, axis=0),
                "std_displacement_by_joint": np.std(home_array, axis=0),
                "max_abs_displacement_by_joint": np.max(
                    np.abs(home_array), axis=0
                ),
                "l2_displacement": summarize(
                    np.linalg.norm(home_array, axis=1)
                ),
            },
        }
        report[str(horizon)] = horizon_report

        for metric_name in (
            "max_joint_abs_displacement",
            "l2_displacement",
            "joint_space_path_length",
        ):
            summary = horizon_report[metric_name]
            csv_rows.append(
                {
                    "horizon_frames": horizon,
                    "metric": metric_name,
                    **summary,
                }
            )

    return report, csv_rows


def main() -> int:
    args = parse_args()

    if not 0.0 < args.guardrail_quantile < 100.0:
        raise ValueError("--guardrail-quantile must be in (0, 100).")
    if args.guardrail_multiplier <= 0.0:
        raise ValueError("--guardrail-multiplier must be positive.")
    if any(horizon <= 0 for horizon in args.horizons):
        raise ValueError("All --horizons values must be positive.")
    if any(threshold < 0.0 for threshold in args.stationary_thresholds):
        raise ValueError("All --stationary-thresholds must be non-negative.")

    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    if not dataset_root.is_dir():
        raise NotADirectoryError(dataset_root)

    print("===== LOAD FROZEN V2 DATASET (READ ONLY) =====")
    dataset = LeRobotDataset(
        repo_id=args.dataset_repo_id,
        root=dataset_root,
        video_backend=args.video_backend,
    )
    hf_dataset = dataset.hf_dataset

    required_columns = {
        "observation.state",
        "action",
        "episode_index",
        "frame_index",
    }
    missing_columns = sorted(required_columns.difference(hf_dataset.column_names))
    if missing_columns:
        raise KeyError(f"Dataset is missing columns: {missing_columns}")

    print("Loading state/action parquet columns; videos are not decoded.")
    states = stack_vector_column(
        hf_dataset["observation.state"],
        name="observation.state",
        expected_dim=12,
    )
    actions = stack_vector_column(
        hf_dataset["action"],
        name="action",
        expected_dim=6,
    )
    episode_indices = scalar_int_column(
        hf_dataset["episode_index"], name="episode_index"
    )
    frame_indices = scalar_int_column(
        hf_dataset["frame_index"], name="frame_index"
    )

    lengths = {
        "observation.state": len(states),
        "action": len(actions),
        "episode_index": len(episode_indices),
        "frame_index": len(frame_indices),
    }
    if len(set(lengths.values())) != 1:
        raise AssertionError(f"Dataset column lengths differ: {lengths}")

    frame_count = len(states)
    unique_episodes = sorted(int(x) for x in np.unique(episode_indices))
    episode_count = len(unique_episodes)
    fps = float(dataset.fps)

    print(f"dataset root: {dataset_root}")
    print(f"frames: {frame_count}")
    print(f"episodes: {episode_count}")
    print(f"fps: {fps:g}")
    print(f"camera keys: {list(dataset.meta.camera_keys)}")

    if frame_count != args.expected_frames:
        raise AssertionError(
            f"Expected {args.expected_frames} frames, found {frame_count}."
        )
    if episode_count != args.expected_episodes:
        raise AssertionError(
            f"Expected {args.expected_episodes} episodes, found {episode_count}."
        )
    if unique_episodes != list(range(args.expected_episodes)):
        raise AssertionError(
            "Episode indices are not contiguous 0..expected_episodes-1: "
            f"{unique_episodes}"
        )
    if not np.isclose(fps, args.expected_fps, rtol=0.0, atol=1e-9):
        raise AssertionError(f"Expected fps={args.expected_fps}, found {fps}.")

    validation_episodes = sorted(set(args.validation_episodes))
    invalid_validation = sorted(set(validation_episodes).difference(unique_episodes))
    if invalid_validation:
        raise ValueError(
            f"Validation episodes are outside the dataset: {invalid_validation}"
        )
    train_episodes = sorted(set(unique_episodes).difference(validation_episodes))
    if not train_episodes or not validation_episodes:
        raise ValueError("Both train and validation episode splits must be non-empty.")

    train_mask = subset_mask(episode_indices, train_episodes)
    validation_mask = subset_mask(episode_indices, validation_episodes)
    if np.any(train_mask & validation_mask):
        raise AssertionError("Train/validation masks overlap.")
    if not np.all(train_mask | validation_mask):
        raise AssertionError("Train/validation masks do not cover every frame.")

    actual_q = states[:, :6]
    previous_commands = states[:, 6:12]
    command_delta = actions - previous_commands
    previous_command_delta = np.zeros_like(command_delta)

    episode_rows: dict[int, np.ndarray] = {}
    max_previous_command_error = 0.0
    max_reconstruction_error = 0.0
    max_recursive_reconstruction_error = 0.0
    episode_motion_rows: list[dict[str, Any]] = []

    print("\n===== CAUSAL INTEGRITY =====")
    for episode in unique_episodes:
        indices = np.flatnonzero(episode_indices == episode)
        if indices.size == 0:
            raise AssertionError(f"Episode {episode} has no frames.")
        episode_rows[episode] = indices

        expected_local_frames = np.arange(indices.size, dtype=np.int64)
        if not np.array_equal(frame_indices[indices], expected_local_frames):
            raise AssertionError(
                f"Episode {episode} frame_index is not contiguous from zero."
            )
        if indices.size > 1:
            previous_error = np.max(
                np.abs(previous_commands[indices[1:]] - actions[indices[:-1]])
            )
            max_previous_command_error = max(
                max_previous_command_error, float(previous_error)
            )
            previous_command_delta[indices[1:]] = (
                previous_commands[indices[1:]]
                - previous_commands[indices[:-1]]
            )

        reconstructed_direct = previous_commands[indices] + command_delta[indices]
        direct_error = np.max(np.abs(reconstructed_direct - actions[indices]))
        max_reconstruction_error = max(
            max_reconstruction_error, float(direct_error)
        )

        running_command = previous_commands[int(indices[0])].copy()
        recursive_errors: list[float] = []
        for global_index in indices:
            running_command = running_command + command_delta[int(global_index)]
            recursive_errors.append(
                float(np.max(np.abs(running_command - actions[int(global_index)])))
            )
        max_recursive_reconstruction_error = max(
            max_recursive_reconstruction_error,
            max(recursive_errors),
        )

        episode_abs_step = np.max(np.abs(command_delta[indices]), axis=1)
        episode_motion_rows.append(
            {
                "episode_index": episode,
                "split": "validation" if episode in validation_episodes else "train",
                "frames": int(indices.size),
                "max_joint_abs_delta_mean": float(np.mean(episode_abs_step)),
                "max_joint_abs_delta_p50": float(
                    np.percentile(episode_abs_step, 50.0)
                ),
                "max_joint_abs_delta_p95": float(
                    np.percentile(episode_abs_step, 95.0)
                ),
                "max_joint_abs_delta_p99": float(
                    np.percentile(episode_abs_step, 99.0)
                ),
                "max_joint_abs_delta_max": float(np.max(episode_abs_step)),
                **{
                    f"fraction_max_abs_delta_le_{threshold:g}": float(
                        np.mean(episode_abs_step <= threshold)
                    )
                    for threshold in args.stationary_thresholds
                },
            }
        )

    print(f"max previous_command error: {max_previous_command_error:.10f}")
    print(f"max direct reconstruction error: {max_reconstruction_error:.10f}")
    print(
        "max recursive reconstruction error: "
        f"{max_recursive_reconstruction_error:.10f}"
    )
    if max_previous_command_error > args.integrity_atol:
        raise AssertionError(
            "v2 previous_command semantics failed: "
            f"{max_previous_command_error} > {args.integrity_atol}"
        )
    if max_reconstruction_error > args.integrity_atol:
        raise AssertionError(
            "v3 direct delta reconstruction failed: "
            f"{max_reconstruction_error} > {args.integrity_atol}"
        )
    if max_recursive_reconstruction_error > args.integrity_atol:
        raise AssertionError(
            "v3 recursive delta reconstruction failed: "
            f"{max_recursive_reconstruction_error} > {args.integrity_atol}"
        )

    first_indices = np.asarray(
        [int(episode_rows[episode][0]) for episode in unique_episodes],
        dtype=np.int64,
    )
    startup_delta = command_delta[first_indices]
    startup_previous_delta = previous_command_delta[first_indices]

    overall_max_abs_step = np.max(np.abs(command_delta), axis=1)
    train_max_abs_step = overall_max_abs_step[train_mask]
    validation_max_abs_step = overall_max_abs_step[validation_mask]

    stationary_report: dict[str, Any] = {}
    print("\n===== STATIONARY / MOVING FRACTIONS =====")
    for threshold in args.stationary_thresholds:
        key = f"max_abs_delta_le_{threshold:g}"
        values = {
            "all": float(np.mean(overall_max_abs_step <= threshold)),
            "train": float(np.mean(train_max_abs_step <= threshold)),
            "validation": float(
                np.mean(validation_max_abs_step <= threshold)
            ),
        }
        stationary_report[key] = values
        print(
            f"threshold={threshold:g}: "
            f"all={values['all']:.3%} "
            f"train={values['train']:.3%} "
            f"validation={values['validation']:.3%}"
        )

    delta_stats_rows, split_comparison_rows = split_joint_rows(
        command_delta=command_delta,
        train_mask=train_mask,
        validation_mask=validation_mask,
        guardrail_quantile=args.guardrail_quantile,
        guardrail_multiplier=args.guardrail_multiplier,
    )

    print("\n===== TRAIN DELTA DISTRIBUTION BY JOINT =====")
    train_rows = [row for row in delta_stats_rows if row["split"] == "train"]
    for row in train_rows:
        print(
            f"{row['joint']:<16} "
            f"abs_p50={row['abs_p50']:.6f} "
            f"abs_p95={row['abs_p95']:.6f} "
            f"abs_p99={row['abs_p99']:.6f} "
            f"abs_p99.9={row['abs_p99_9']:.6f} "
            f"max={row['abs_max']:.6f}"
        )

    horizons = sorted(set(args.horizons))
    horizon_report, horizon_csv_rows = horizon_summaries(
        actions=actions,
        previous_commands=previous_commands,
        command_delta=command_delta,
        episode_rows=episode_rows,
        horizons=horizons,
    )
    for horizon in horizons:
        horizon_report[str(horizon)]["seconds_at_dataset_fps"] = horizon / fps

    print("\n===== HUMAN MOTION HORIZONS =====")
    for horizon in horizons:
        values = horizon_report[str(horizon)]
        displacement = values["max_joint_abs_displacement"]
        home = values["home_anchor"]["l2_displacement"]
        print(
            f"H={horizon:>2} ({horizon / fps:.3f}s): "
            f"all-window max-joint disp p50={displacement['p50']:.4f} "
            f"p95={displacement['p95']:.4f}; "
            f"home L2 disp p50={home['p50']:.4f} p95={home['p95']:.4f}"
        )

    train_guardrail = np.asarray(
        [float(row["empirical_guardrail_candidate"]) for row in train_rows],
        dtype=np.float64,
    )

    split_report = {
        "train_episodes": train_episodes,
        "validation_episodes": validation_episodes,
        "train_episode_count": len(train_episodes),
        "validation_episode_count": len(validation_episodes),
        "train_frame_count": int(np.sum(train_mask)),
        "validation_frame_count": int(np.sum(validation_mask)),
        "disjoint": bool(not np.any(train_mask & validation_mask)),
        "complete": bool(np.all(train_mask | validation_mask)),
    }

    report = {
        "schema_version": "so101_red_cube_pick_place_v3_delta_plan_v1",
        "status": "PASS",
        "scope": {
            "read_only_source_audit": True,
            "v3_dataset_built": False,
            "model_trained": False,
            "hardware_accessed": False,
            "videos_decoded": False,
        },
        "source": {
            "dataset_repo_id": args.dataset_repo_id,
            "dataset_root": dataset_root,
            "frames": frame_count,
            "episodes": episode_count,
            "fps": fps,
            "camera_keys": list(dataset.meta.camera_keys),
            "state_dim": int(states.shape[1]),
            "action_dim": int(actions.shape[1]),
        },
        "proposed_v3_schema": {
            "observation_state_dim": 18,
            "observation_state_order": [
                "actual_q_t[6]",
                "previous_command_t[6]",
                "previous_command_delta_t[6]",
            ],
            "previous_command_delta_formula": (
                "episode frame 0: zeros[6]; otherwise "
                "previous_command[t] - previous_command[t-1]"
            ),
            "action_dim": 6,
            "action_formula": "command[t] - previous_command[t]",
            "runtime_decode": (
                "sequential cumulative sum from safety-controller "
                "last_sent_command; replan after n_action_steps"
            ),
            "provisional_chunk_size": 20,
            "provisional_n_action_steps": 5,
        },
        "split": split_report,
        "integrity": {
            "atol": args.integrity_atol,
            "max_previous_command_error": max_previous_command_error,
            "max_direct_delta_reconstruction_error": max_reconstruction_error,
            "max_recursive_delta_reconstruction_error": (
                max_recursive_reconstruction_error
            ),
            "startup_command_delta": {
                "mean_by_joint": np.mean(startup_delta, axis=0),
                "std_by_joint": np.std(startup_delta, axis=0),
                "max_abs_by_joint": np.max(np.abs(startup_delta), axis=0),
            },
            "startup_previous_command_delta": {
                "mean_by_joint": np.mean(startup_previous_delta, axis=0),
                "std_by_joint": np.std(startup_previous_delta, axis=0),
                "max_abs_by_joint": np.max(
                    np.abs(startup_previous_delta), axis=0
                ),
            },
        },
        "command_delta": {
            "all_max_joint_abs_per_frame": summarize(overall_max_abs_step),
            "train_max_joint_abs_per_frame": summarize(train_max_abs_step),
            "validation_max_joint_abs_per_frame": summarize(
                validation_max_abs_step
            ),
            "stationary_fractions": stationary_report,
            "empirical_guardrail_diagnostic": {
                "warning": (
                    "Dataset-derived diagnostic only; not a hardware safety limit."
                ),
                "quantile_percent": args.guardrail_quantile,
                "multiplier": args.guardrail_multiplier,
                "joint_names": JOINT_NAMES,
                "candidate_max_abs_delta_by_joint": train_guardrail,
            },
        },
        "tracking_gap_previous_command_minus_actual_q": {
            joint_name: summarize(
                np.abs(previous_commands[:, joint_index] - actual_q[:, joint_index])
            )
            for joint_index, joint_name in enumerate(JOINT_NAMES)
        },
        "horizons": horizon_report,
        "decision_rule_for_next_stage": {
            "automatic_build_authorized": False,
            "required_review": [
                "startup command deltas",
                "train/validation delta distribution",
                "stationary-frame proportion",
                "20-frame horizon dispersion",
                "candidate empirical per-joint guardrails",
            ],
            "note": (
                "This planner intentionally does not decide robot safety and "
                "does not build or train v3."
            ),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "dataset_v3_delta_plan.json"
    delta_stats_path = output_dir / "delta_stats_by_joint.csv"
    episode_stats_path = output_dir / "episode_motion_stats.csv"
    split_comparison_path = output_dir / "split_comparison_by_joint.csv"
    horizon_stats_path = output_dir / "horizon_motion_stats.csv"

    report_path.write_text(
        json.dumps(to_jsonable(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(delta_stats_path, delta_stats_rows)
    write_csv(episode_stats_path, episode_motion_rows)
    write_csv(split_comparison_path, split_comparison_rows)
    write_csv(horizon_stats_path, horizon_csv_rows)

    print("\n===== PROVISIONAL V3 SCHEMA =====")
    print("state: 18D [actual_q, previous_command, previous_command_delta]")
    print("action: 6D command_delta = command - previous_command")
    print("train episodes:", train_episodes)
    print("validation episodes:", validation_episodes)
    print(
        "empirical guardrail candidates (NOT hardware limits):",
        format_vector(train_guardrail),
    )

    print("\n===== OUTPUT =====")
    for path in (
        report_path,
        delta_stats_path,
        episode_stats_path,
        split_comparison_path,
        horizon_stats_path,
    ):
        print(path)
    print("V3 DELTA DATASET PLANNING AUDIT: PASS")
    print("NO DATASET WAS MODIFIED. NO HARDWARE WAS ACCESSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
