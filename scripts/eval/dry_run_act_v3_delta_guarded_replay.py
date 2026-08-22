#!/usr/bin/env python
"""Guarded, counterfactual ACT v3 delta replay with zero hardware access.

The script runs the frozen 11K policy over recorded validation images. At each
five-frame replan boundary it constructs the exact 18D runtime state from:

    recorded actual_q[6]
    simulated guarded last_sent_command[6]
    simulated guarded last_sent_delta[6]

The predicted deltas are passed through the frozen JSON guard contract. Every
would-be command is written to CSV, but no robot class, serial port, motor bus,
teleoperator, calibration procedure, or live camera is imported or opened.

This is a numerical runtime-stability audit. Recorded images and actual_q do
not react to simulated commands, so the result is not a closed-loop task-success
evaluation and never authorizes hardware deployment.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from audit_act_v3_delta_deployment_contract import (
    ACTION_KEY,
    CHUNK_SIZE,
    EXECUTION_HORIZON,
    EXPECTED_ACTION_DIM,
    EXPECTED_STATE_DIM,
    EXPECTED_TRAIN_EPISODES,
    EXPECTED_TRAIN_FRAMES,
    EXPECTED_VALIDATION_EPISODES,
    EXPECTED_VALIDATION_FRAMES,
    MOTORS,
    STATE_KEY,
    dataset_arrays,
    frame_map,
    infer_chunk,
    item_batch,
    load_dataset,
    load_policy,
    to_jsonable,
    validate_dataset_metadata,
    verify_release,
)


EXPECTED_CONTRACT_SCHEMA = "so101_act_v3_delta_runtime_limit_contract_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/safety/act_v3_delta_runtime_limit_contract.json"),
    )
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
        "--calibration",
        type=Path,
        default=Path(
            "~/.cache/huggingface/lerobot/calibration/robots/"
            "so101_follower/follower_white.json"
        ),
    )
    parser.add_argument("--video-backend", default="torchcodec")
    parser.add_argument(
        "--episode-indices",
        type=int,
        nargs="*",
        default=None,
        help="Validation episode indices. Omit to replay all 12 episodes.",
    )
    parser.add_argument(
        "--max-replans-per-episode",
        type=int,
        default=0,
        help="0 means every valid replan; a positive value creates a smoke run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/eval/act_red_cube_v3_delta_guarded_dry_run"),
    )
    args = parser.parse_args()

    if args.max_replans_per_episode < 0:
        parser.error("--max-replans-per-episode must be >= 0")
    if args.episode_indices is not None:
        values = sorted(set(int(value) for value in args.episode_indices))
        if not values:
            parser.error("--episode-indices cannot be empty when supplied")
        invalid = [
            value
            for value in values
            if value < 0 or value >= EXPECTED_VALIDATION_EPISODES
        ]
        if invalid:
            parser.error(f"Invalid validation episode indices: {invalid}")
        args.episode_indices = values
    return args


def load_json(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {resolved}")
    return value


def ordered_vector(mapping: dict[str, Any], label: str) -> np.ndarray:
    if set(mapping) != set(MOTORS):
        raise RuntimeError(f"{label} motors differ from the frozen contract")
    result = np.asarray([mapping[motor] for motor in MOTORS], dtype=np.float64)
    if result.shape != (EXPECTED_ACTION_DIM,) or not np.isfinite(result).all():
        raise RuntimeError(f"Invalid {label}: {result}")
    return result


def ordered_limits(
    mapping: dict[str, Any],
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    if set(mapping) != set(MOTORS):
        raise RuntimeError(f"{label} motors differ from the frozen contract")
    pairs = np.asarray([mapping[motor] for motor in MOTORS], dtype=np.float64)
    if pairs.shape != (EXPECTED_ACTION_DIM, 2):
        raise RuntimeError(f"Invalid {label} shape: {pairs.shape}")
    lower = pairs[:, 0]
    upper = pairs[:, 1]
    if (
        not np.isfinite(pairs).all()
        or np.any(lower >= upper)
    ):
        raise RuntimeError(f"Invalid {label}: {pairs}")
    return lower, upper


def validate_contract(contract: dict[str, Any]) -> dict[str, np.ndarray]:
    if contract.get("schema_version") != EXPECTED_CONTRACT_SCHEMA:
        raise RuntimeError("Runtime-limit contract schema differs")
    if contract.get("status") != "FROZEN_FOR_OFFLINE_DRY_RUN":
        raise RuntimeError("Runtime-limit contract is not frozen for dry-run")
    scope = contract.get("scope")
    if not isinstance(scope, dict):
        raise RuntimeError("Runtime-limit contract is missing scope")
    if not bool(scope.get("offline_dry_run_only")):
        raise RuntimeError("Contract does not require offline-only execution")
    if bool(scope.get("hardware_access_allowed")):
        raise RuntimeError("Contract unexpectedly allows hardware access")
    if bool(scope.get("hardware_deployment_authorized")):
        raise RuntimeError("Contract unexpectedly authorizes deployment")

    policy = contract.get("policy")
    if not isinstance(policy, dict):
        raise RuntimeError("Contract is missing policy metadata")
    expected_policy = {
        "candidate_step": 11000,
        "observation_state_dim": EXPECTED_STATE_DIM,
        "action_dim": EXPECTED_ACTION_DIM,
        "action_semantics": "command_delta",
        "chunk_size": CHUNK_SIZE,
        "n_action_steps": EXECUTION_HORIZON,
        "replan_interval_frames": EXECUTION_HORIZON,
        "dataset_fps": 15,
        "use_vae": False,
        "use_amp": False,
    }
    for key, expected in expected_policy.items():
        if policy.get(key) != expected:
            raise RuntimeError(
                f"Contract policy {key}={policy.get(key)!r}, expected={expected!r}"
            )

    robot = contract.get("robot")
    if not isinstance(robot, dict):
        raise RuntimeError("Contract is missing robot metadata")
    if robot.get("type") != "so101_follower" or robot.get("id") != "follower_white":
        raise RuntimeError("Contract robot identity differs")
    if bool(robot.get("use_degrees")):
        raise RuntimeError("Contract must use range normalization, not degrees")
    normalization = robot.get("normalization")
    if not isinstance(normalization, dict) or set(normalization) != set(MOTORS):
        raise RuntimeError("Contract normalization map differs")
    for motor in MOTORS[:5]:
        value = normalization[motor]
        if value.get("mode") != "RANGE_M100_100" or value.get(
            "normalized_range"
        ) != [-100.0, 100.0]:
            raise RuntimeError(f"Body normalization differs for {motor}")
    gripper_norm = normalization["gripper"]
    if gripper_norm.get("mode") != "RANGE_0_100" or gripper_norm.get(
        "normalized_range"
    ) != [0.0, 100.0]:
        raise RuntimeError("Gripper normalization differs")

    startup = contract.get("startup")
    guard = contract.get("guard")
    if not isinstance(startup, dict) or not isinstance(guard, dict):
        raise RuntimeError("Contract is missing startup/guard metadata")
    home = np.asarray(startup["required_previous_command"], dtype=np.float64)
    previous_delta = np.asarray(
        startup["required_previous_delta"], dtype=np.float64
    )
    if home.shape != (EXPECTED_ACTION_DIM,) or previous_delta.shape != (
        EXPECTED_ACTION_DIM,
    ):
        raise RuntimeError("Contract startup vectors are not 6D")
    if not np.isfinite(home).all() or not np.isfinite(previous_delta).all():
        raise RuntimeError("Contract startup vectors contain NaN/Inf")
    if not np.array_equal(previous_delta, np.zeros(EXPECTED_ACTION_DIM)):
        raise RuntimeError("Contract startup previous delta is not zero")
    tolerance = float(startup["dataset_match_tolerance"])
    if tolerance <= 0.0:
        raise RuntimeError("Contract startup tolerance must be positive")

    max_delta = ordered_vector(
        guard["max_abs_delta_per_command"],
        "max_abs_delta_per_command",
    )
    if np.any(max_delta <= 0.0):
        raise RuntimeError("Per-command delta limits must be positive")
    soft_min, soft_max = ordered_limits(
        guard["normal_soft_limits"], "normal_soft_limits"
    )
    tracking = ordered_vector(
        guard["tracking_error_limit"], "tracking_error_limit"
    )
    if np.any(tracking <= 0.0):
        raise RuntimeError("Tracking limits must be positive")
    if np.any(home < soft_min) or np.any(home > soft_max):
        raise RuntimeError("Frozen Home command is outside normal soft limits")
    numerical_tolerance = float(guard["numerical_tolerance"])
    if numerical_tolerance <= 0.0:
        raise RuntimeError("Guard numerical tolerance must be positive")

    return {
        "home": home,
        "previous_delta": previous_delta,
        "startup_tolerance": np.asarray([tolerance], dtype=np.float64),
        "max_delta": max_delta,
        "soft_min": soft_min,
        "soft_max": soft_max,
        "tracking": tracking,
        "numerical_tolerance": np.asarray(
            [numerical_tolerance], dtype=np.float64
        ),
    }


def validate_calibration(
    contract: dict[str, Any], calibration: dict[str, Any]
) -> None:
    expected = contract["robot"]["expected_calibration_raw"]
    if set(calibration) != set(MOTORS):
        raise RuntimeError("Follower calibration motor set differs")
    if calibration != expected:
        differences: dict[str, Any] = {}
        for motor in MOTORS:
            if calibration.get(motor) != expected.get(motor):
                differences[motor] = {
                    "expected": expected.get(motor),
                    "actual": calibration.get(motor),
                }
        raise RuntimeError(
            "Follower calibration changed; stopping dry-run: "
            + json.dumps(differences, ensure_ascii=False)
        )


def validate_dataset_startup(
    *,
    label: str,
    states: np.ndarray,
    actions: np.ndarray,
    episodes: np.ndarray,
    home: np.ndarray,
    tolerance: float,
) -> dict[str, float | int]:
    first_indices = np.asarray(
        [
            int(np.flatnonzero(episodes == episode)[0])
            for episode in sorted(int(value) for value in np.unique(episodes))
        ],
        dtype=np.int64,
    )
    previous_command_error = float(
        np.max(np.abs(states[first_indices, 6:12].astype(np.float64) - home))
    )
    previous_delta_error = float(
        np.max(np.abs(states[first_indices, 12:18].astype(np.float64)))
    )
    startup_action_error = float(
        np.max(np.abs(actions[first_indices].astype(np.float64)))
    )
    if max(previous_command_error, previous_delta_error, startup_action_error) > tolerance:
        raise RuntimeError(
            f"{label} startup contract failed: "
            f"previous_command={previous_command_error}, "
            f"previous_delta={previous_delta_error}, action={startup_action_error}"
        )
    return {
        "episode_count": int(len(first_indices)),
        "max_previous_command_error": previous_command_error,
        "max_previous_delta_error": previous_delta_error,
        "max_startup_action": startup_action_error,
        "tolerance": tolerance,
    }


def scalar_summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise RuntimeError("Cannot summarize empty/non-finite values")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "p99": float(np.percentile(array, 99.0)),
        "max": float(np.max(array)),
    }


def main() -> int:
    args = parse_args()

    print("===== LOAD FROZEN OFFLINE CONTRACT =====")
    contract_path = args.contract.expanduser().resolve()
    contract = load_json(contract_path)
    limits = validate_contract(contract)
    calibration_path = args.calibration.expanduser().resolve()
    calibration = load_json(calibration_path)
    validate_calibration(contract, calibration)
    print(f"contract: {contract_path}")
    print(f"calibration: {calibration_path}")
    print("normalization, calibration, Home, delta and soft limits: PASS")
    print("hardware access allowed by contract: false")

    print("\n===== VERIFY FROZEN 11K RELEASE =====")
    release = verify_release(args.candidate_root, args.source_checkpoint_root)
    print("manifest, selection, and source-checkpoint identity: PASS")

    print("\n===== LOAD FROZEN V3 DATASETS =====")
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
    home = limits["home"]
    tolerance = float(limits["startup_tolerance"][0])
    train_startup = validate_dataset_startup(
        label="train",
        states=train_state,
        actions=train_action,
        episodes=train_episode,
        home=home,
        tolerance=tolerance,
    )
    validation_startup = validate_dataset_startup(
        label="validation",
        states=val_state,
        actions=val_action,
        episodes=val_episode,
        home=home,
        tolerance=tolerance,
    )
    print(
        f"train: frames={len(train_state)}, startup episodes="
        f"{train_startup['episode_count']} PASS"
    )
    print(
        f"validation: frames={len(val_state)}, startup episodes="
        f"{validation_startup['episode_count']} PASS"
    )

    soft_min = limits["soft_min"]
    soft_max = limits["soft_max"]
    validation_target_command = (
        val_state[:, 6:12].astype(np.float64)
        + val_action.astype(np.float64)
    )
    target_soft_violation = (
        (
            validation_target_command
            < soft_min.reshape(1, -1) - limits["numerical_tolerance"][0]
        )
        | (
            validation_target_command
            > soft_max.reshape(1, -1) + limits["numerical_tolerance"][0]
        )
    )
    target_soft_violation_by_joint = np.sum(target_soft_violation, axis=0)
    print(
        "recorded validation target soft-limit violations by joint: "
        f"{target_soft_violation_by_joint.astype(int).tolist()}"
    )

    selected_episodes = (
        list(range(EXPECTED_VALIDATION_EPISODES))
        if args.episode_indices is None
        else list(args.episode_indices)
    )
    run_is_full = (
        selected_episodes == list(range(EXPECTED_VALIDATION_EPISODES))
        and args.max_replans_per_episode == 0
    )
    run_kind = "full" if run_is_full else "smoke"
    print(f"selected validation episodes: {selected_episodes}")
    print(f"run kind: {run_kind}")

    print("\n===== LOAD POLICY (NO ROBOT OBJECT) =====")
    policy, device = load_policy(
        args.candidate_root.resolve() / "pretrained_model"
    )
    print("frozen ACT 11K policy: PASS")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "guarded_dry_run_commands.csv"
    report_path = output_dir / "guarded_dry_run_report.json"

    metadata_columns = [
        "episode_index",
        "replan_index",
        "replan_frame",
        "substep",
        "dataset_frame",
        "any_rate_clipped",
        "any_soft_clipped",
    ]
    joint_fields = [
        "recorded_actual_q",
        "recorded_target_command",
        "raw_predicted_delta",
        "rate_limited_delta",
        "unbounded_command",
        "guarded_command",
        "actual_guarded_delta",
        "rate_clipped",
        "soft_clipped",
        "counterfactual_tracking_error",
    ]
    csv_columns = metadata_columns + [
        f"{field}.{motor}" for motor in MOTORS for field in joint_fields
    ]

    stats: dict[str, dict[str, list[float]]] = {
        motor: {
            "raw_abs_delta": [],
            "actual_abs_delta": [],
            "guarded_command": [],
            "target_abs_error": [],
            "counterfactual_tracking_error": [],
            "rate_clipped": [],
            "soft_clipped": [],
            "at_soft_min": [],
            "at_soft_max": [],
        }
        for motor in MOTORS
    }
    episode_summaries: list[dict[str, Any]] = []
    mapping = frame_map(val_episode, val_frame)
    total_replans = 0
    total_commands = 0
    any_rate_clipped_commands = 0
    any_soft_clipped_commands = 0
    max_guard_invariant_error = 0.0

    print("\n===== GUARDED COUNTERFACTUAL REPLAY =====")
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
        writer.writeheader()

        for episode in selected_episodes:
            episode_indices = np.flatnonzero(val_episode == episode)
            episode_length = len(episode_indices)
            last_valid_start = episode_length - CHUNK_SIZE
            replan_frames = list(
                range(0, last_valid_start + 1, EXECUTION_HORIZON)
            )
            if args.max_replans_per_episode > 0:
                replan_frames = replan_frames[: args.max_replans_per_episode]

            last_command = home.copy()
            last_delta = np.zeros(EXPECTED_ACTION_DIM, dtype=np.float64)
            episode_rate_clips = 0
            episode_soft_clips = 0

            for replan_index, replan_frame in enumerate(replan_frames):
                global_index = mapping[(episode, replan_frame)]
                actual_q = val_state[global_index, :6].astype(np.float64)
                runtime_state = np.concatenate(
                    [actual_q, last_command, last_delta]
                ).astype(np.float32)
                if runtime_state.shape != (EXPECTED_STATE_DIM,):
                    raise RuntimeError("Constructed runtime state is not 18D")
                if not np.isfinite(runtime_state).all():
                    raise FloatingPointError("Runtime state contains NaN/Inf")

                recorded_item = validation_dataset[int(global_index)]
                runtime_item = dict(recorded_item)
                runtime_item[STATE_KEY] = torch.from_numpy(runtime_state)
                batch = item_batch(runtime_item, device)
                chunk = infer_chunk(policy, batch, device)

                for substep in range(EXECUTION_HORIZON):
                    dataset_frame = replan_frame + substep
                    step_global = mapping[(episode, dataset_frame)]
                    recorded_actual = val_state[step_global, :6].astype(
                        np.float64
                    )
                    recorded_target = (
                        val_state[step_global, 6:12].astype(np.float64)
                        + val_action[step_global].astype(np.float64)
                    )
                    raw_delta = chunk[substep].astype(np.float64)
                    if raw_delta.shape != (EXPECTED_ACTION_DIM,):
                        raise RuntimeError("Predicted delta is not 6D")
                    if not np.isfinite(raw_delta).all():
                        raise FloatingPointError("Predicted delta contains NaN/Inf")

                    rate_limited_delta = np.clip(
                        raw_delta, -limits["max_delta"], limits["max_delta"]
                    )
                    unbounded_command = last_command + rate_limited_delta
                    guarded_command = np.clip(
                        unbounded_command, soft_min, soft_max
                    )
                    actual_guarded_delta = guarded_command - last_command
                    rate_clipped = (
                        np.abs(raw_delta - rate_limited_delta)
                        > limits["numerical_tolerance"][0]
                    )
                    soft_clipped = (
                        np.abs(unbounded_command - guarded_command)
                        > limits["numerical_tolerance"][0]
                    )
                    tracking_error = np.abs(guarded_command - recorded_actual)

                    invariant_error = max(
                        float(
                            np.max(
                                np.maximum(
                                    np.abs(actual_guarded_delta)
                                    - limits["max_delta"],
                                    0.0,
                                )
                            )
                        ),
                        float(
                            np.max(
                                np.maximum(soft_min - guarded_command, 0.0)
                            )
                        ),
                        float(
                            np.max(
                                np.maximum(guarded_command - soft_max, 0.0)
                            )
                        ),
                    )
                    max_guard_invariant_error = max(
                        max_guard_invariant_error, invariant_error
                    )
                    if invariant_error > limits["numerical_tolerance"][0]:
                        raise RuntimeError(
                            f"Guard invariant failed: {invariant_error}"
                        )

                    any_rate = bool(np.any(rate_clipped))
                    any_soft = bool(np.any(soft_clipped))
                    any_rate_clipped_commands += int(any_rate)
                    any_soft_clipped_commands += int(any_soft)
                    episode_rate_clips += int(any_rate)
                    episode_soft_clips += int(any_soft)
                    total_commands += 1

                    row: dict[str, Any] = {
                        "episode_index": episode,
                        "replan_index": replan_index,
                        "replan_frame": replan_frame,
                        "substep": substep,
                        "dataset_frame": dataset_frame,
                        "any_rate_clipped": int(any_rate),
                        "any_soft_clipped": int(any_soft),
                    }
                    for joint, motor in enumerate(MOTORS):
                        values = {
                            "recorded_actual_q": recorded_actual[joint],
                            "recorded_target_command": recorded_target[joint],
                            "raw_predicted_delta": raw_delta[joint],
                            "rate_limited_delta": rate_limited_delta[joint],
                            "unbounded_command": unbounded_command[joint],
                            "guarded_command": guarded_command[joint],
                            "actual_guarded_delta": actual_guarded_delta[joint],
                            "rate_clipped": int(rate_clipped[joint]),
                            "soft_clipped": int(soft_clipped[joint]),
                            "counterfactual_tracking_error": tracking_error[joint],
                        }
                        for field, value in values.items():
                            row[f"{field}.{motor}"] = value

                        joint_stats = stats[motor]
                        joint_stats["raw_abs_delta"].append(
                            float(abs(raw_delta[joint]))
                        )
                        joint_stats["actual_abs_delta"].append(
                            float(abs(actual_guarded_delta[joint]))
                        )
                        joint_stats["guarded_command"].append(
                            float(guarded_command[joint])
                        )
                        joint_stats["target_abs_error"].append(
                            float(abs(guarded_command[joint] - recorded_target[joint]))
                        )
                        joint_stats["counterfactual_tracking_error"].append(
                            float(tracking_error[joint])
                        )
                        joint_stats["rate_clipped"].append(
                            float(rate_clipped[joint])
                        )
                        joint_stats["soft_clipped"].append(
                            float(soft_clipped[joint])
                        )
                        joint_stats["at_soft_min"].append(
                            float(
                                abs(guarded_command[joint] - soft_min[joint])
                                <= 1e-9
                            )
                        )
                        joint_stats["at_soft_max"].append(
                            float(
                                abs(guarded_command[joint] - soft_max[joint])
                                <= 1e-9
                            )
                        )
                    writer.writerow(row)
                    last_command = guarded_command.copy()
                    last_delta = actual_guarded_delta.copy()

                total_replans += 1
                if total_replans % 50 == 0:
                    print(
                        f"processed replans={total_replans} "
                        f"commands={total_commands}"
                    )

            episode_summaries.append(
                {
                    "episode_index": episode,
                    "episode_frames": episode_length,
                    "replans_processed": len(replan_frames),
                    "commands_simulated": len(replan_frames)
                    * EXECUTION_HORIZON,
                    "commands_with_any_rate_clip": episode_rate_clips,
                    "commands_with_any_soft_clip": episode_soft_clips,
                    "final_guarded_command": last_command.tolist(),
                    "final_guarded_delta": last_delta.tolist(),
                }
            )
            print(
                f"episode {episode:02d}: replans={len(replan_frames)} "
                f"commands={len(replan_frames) * EXECUTION_HORIZON} "
                f"rate_clips={episode_rate_clips} "
                f"soft_clips={episode_soft_clips}"
            )

    if total_commands == 0:
        raise RuntimeError("Dry-run produced no simulated commands")

    joint_report: dict[str, Any] = {}
    print("\n===== GUARDED DRY-RUN METRICS =====")
    print(
        "joint             raw_p99  raw_max  rate_clip  soft_clip  "
        "guarded_min  guarded_max  tracking>limit"
    )
    for joint, motor in enumerate(MOTORS):
        joint_stats = stats[motor]
        raw_summary = scalar_summary(joint_stats["raw_abs_delta"])
        actual_summary = scalar_summary(joint_stats["actual_abs_delta"])
        guarded_array = np.asarray(
            joint_stats["guarded_command"], dtype=np.float64
        )
        tracking_array = np.asarray(
            joint_stats["counterfactual_tracking_error"], dtype=np.float64
        )
        rate_clip_rate = float(np.mean(joint_stats["rate_clipped"]))
        soft_clip_rate = float(np.mean(joint_stats["soft_clipped"]))
        tracking_violation_rate = float(
            np.mean(tracking_array > limits["tracking"][joint])
        )
        joint_report[motor] = {
            "max_abs_delta_contract": float(limits["max_delta"][joint]),
            "soft_limits": [float(soft_min[joint]), float(soft_max[joint])],
            "tracking_error_limit": float(limits["tracking"][joint]),
            "raw_predicted_abs_delta": raw_summary,
            "actual_guarded_abs_delta": actual_summary,
            "rate_clip_count": int(sum(joint_stats["rate_clipped"])),
            "rate_clip_rate": rate_clip_rate,
            "soft_clip_count": int(sum(joint_stats["soft_clipped"])),
            "soft_clip_rate": soft_clip_rate,
            "soft_min_occupancy_rate": float(
                np.mean(joint_stats["at_soft_min"])
            ),
            "soft_max_occupancy_rate": float(
                np.mean(joint_stats["at_soft_max"])
            ),
            "guarded_command_min": float(np.min(guarded_array)),
            "guarded_command_max": float(np.max(guarded_array)),
            "recorded_target_command_mae": float(
                np.mean(joint_stats["target_abs_error"])
            ),
            "counterfactual_tracking_error": scalar_summary(
                joint_stats["counterfactual_tracking_error"]
            ),
            "counterfactual_tracking_violation_count": int(
                np.sum(tracking_array > limits["tracking"][joint])
            ),
            "counterfactual_tracking_violation_rate": tracking_violation_rate,
            "recorded_validation_target_soft_violation_count": int(
                target_soft_violation_by_joint[joint]
            ),
        }
        print(
            f"{motor:<16} "
            f"{raw_summary['p99']:8.5f} "
            f"{raw_summary['max']:8.5f} "
            f"{rate_clip_rate:9.3%} "
            f"{soft_clip_rate:9.3%} "
            f"{np.min(guarded_array):11.4f} "
            f"{np.max(guarded_array):11.4f} "
            f"{tracking_violation_rate:14.3%}"
        )

    if np.any(target_soft_violation_by_joint > 0):
        decision = "REVIEW_RECORDED_TARGETS_OUTSIDE_FROZEN_SOFT_LIMITS"
    elif run_is_full:
        decision = "FULL_DRY_RUN_COMPLETE_REVIEW_GUARD_INTERVENTIONS"
    else:
        decision = "SMOKE_PASS_RUN_FULL_DRY_RUN_NEXT"

    try:
        lerobot_version = importlib.metadata.version("lerobot")
    except importlib.metadata.PackageNotFoundError:
        lerobot_version = "unknown-editable-install"

    report: dict[str, Any] = {
        "schema_version": "so101_act_v3_delta_guarded_dry_run_v1",
        "status": "PASS",
        "decision": decision,
        "scope": {
            "offline_only": True,
            "recorded_images_only": True,
            "recorded_actual_q_only": True,
            "dataset_modified": False,
            "hardware_accessed": False,
            "robot_object_created": False,
            "serial_port_opened": False,
            "command_sent": False,
            "hardware_deployment_authorized": False,
            "physical_safety_certified": False,
        },
        "counterfactual_warning": (
            "Recorded images and actual_q do not react to simulated model "
            "commands. Tracking errors are diagnostics, not live-controller "
            "trip predictions."
        ),
        "environment": {
            "lerobot_version": lerobot_version,
            "torch_version": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0),
        },
        "paths": {
            "contract": str(contract_path),
            "calibration": str(calibration_path),
            "candidate": str(args.candidate_root.resolve()),
            "validation_dataset": str(args.validation_dataset_root.resolve()),
        },
        "release": release,
        "contract": contract,
        "startup_checks": {
            "train": train_startup,
            "validation": validation_startup,
        },
        "run": {
            "kind": run_kind,
            "episode_indices": selected_episodes,
            "max_replans_per_episode": args.max_replans_per_episode,
            "total_replans": total_replans,
            "total_commands": total_commands,
            "commands_with_any_rate_clip": any_rate_clipped_commands,
            "commands_with_any_rate_clip_rate": (
                any_rate_clipped_commands / total_commands
            ),
            "commands_with_any_soft_clip": any_soft_clipped_commands,
            "commands_with_any_soft_clip_rate": (
                any_soft_clipped_commands / total_commands
            ),
            "max_guard_invariant_error": max_guard_invariant_error,
        },
        "recorded_validation_target_soft_violation_count_by_joint": {
            motor: int(target_soft_violation_by_joint[index])
            for index, motor in enumerate(MOTORS)
        },
        "joint_metrics": joint_report,
        "episodes": episode_summaries,
        "required_next_decision": (
            "Review rate-clamp, soft-clamp, boundary occupancy, and "
            "counterfactual tracking metrics. Hardware remains blocked."
        ),
    }
    report_path.write_text(
        json.dumps(to_jsonable(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    del policy
    gc.collect()
    torch.cuda.empty_cache()

    print("\n===== DECISION =====")
    print(f"{decision=}")
    print(f"commands with any rate clip: {any_rate_clipped_commands}/{total_commands}")
    print(f"commands with any soft clip: {any_soft_clipped_commands}/{total_commands}")
    print(f"max guard invariant error: {max_guard_invariant_error:.10f}")
    print("\n===== OUTPUT =====")
    print(report_path)
    print(csv_path)
    print("ACT V3 DELTA GUARDED DRY-RUN: PASS")
    print("HARDWARE DEPLOYMENT REMAINS BLOCKED.")
    print("NO ROBOT OBJECT WAS CREATED. NO SERIAL PORT WAS OPENED.")
    print("NO COMMAND WAS SENT. NO DATASET WAS MODIFIED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
