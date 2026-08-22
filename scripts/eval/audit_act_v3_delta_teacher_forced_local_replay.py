#!/usr/bin/env python
"""Teacher-forced local five-command audit for the frozen ACT v3 policy.

At every replan anchor, this audit uses the recorded in-distribution 18D state:

    [recorded actual_q, recorded previous_command, recorded previous_delta]

It predicts one ten-action ACT chunk, applies the frozen guard recursively only
to the first five commands, and then discards the simulated local command state.
The next replan starts again from that frame's recorded 18D state. This isolates
local queue behavior from the long counterfactual replay's hybrid-state drift.

This remains an offline diagnostic. Recorded images and state do not react to
predicted commands, so it is not a closed-loop task-success evaluation and does
not authorize hardware deployment. No robot, motor bus, serial port, live
camera, or teleoperator module is imported or opened.
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
    validate_sequential_delta_semantics,
    verify_release,
)
from dry_run_act_v3_delta_guarded_replay import (
    load_json,
    scalar_summary,
    validate_calibration,
    validate_contract,
    validate_dataset_startup,
)


EXPECTED_ATTRIBUTION_SCHEMA = (
    "so101_act_v3_delta_guard_intervention_attribution_v1"
)
EXPECTED_ATTRIBUTION_DECISION = (
    "RUN_TEACHER_FORCED_LOCAL_GUARD_REPLAY_NEXT"
)
OUTPUT_SCHEMA = "so101_act_v3_delta_teacher_forced_local_replay_v1"


def parse_anchor(text: str) -> tuple[int, int]:
    parts = text.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Anchor must be EPISODE:FRAME, got {text!r}"
        )
    try:
        episode, frame = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Anchor must contain integers, got {text!r}"
        ) from exc
    return episode, frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/safety/act_v3_delta_runtime_limit_contract.json"),
    )
    parser.add_argument(
        "--attribution-report",
        type=Path,
        default=Path(
            "outputs/eval/act_red_cube_v3_delta_guard_intervention_attribution/"
            "guard_intervention_attribution_report.json"
        ),
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
        "--anchors",
        nargs="*",
        type=parse_anchor,
        default=None,
        metavar="EPISODE:FRAME",
        help=(
            "Exact replan anchors for a targeted run. Frames must be divisible "
            "by five. Omit for episode-based or full replay."
        ),
    )
    parser.add_argument(
        "--episode-indices",
        type=int,
        nargs="*",
        default=None,
        help="Validation episode indices; omit for all 12 episodes.",
    )
    parser.add_argument(
        "--max-replans-per-episode",
        type=int,
        default=0,
        help="0 means every valid replan; positive values create a smoke run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/eval/act_red_cube_v3_delta_teacher_forced_local_replay"
        ),
    )
    args = parser.parse_args()

    if args.max_replans_per_episode < 0:
        parser.error("--max-replans-per-episode must be >= 0")
    if args.anchors is not None:
        if not args.anchors:
            parser.error("--anchors cannot be supplied without values")
        if args.episode_indices is not None or args.max_replans_per_episode != 0:
            parser.error(
                "--anchors cannot be combined with --episode-indices or "
                "--max-replans-per-episode"
            )
        args.anchors = sorted(set(args.anchors))
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


def validate_attribution(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("schema_version") != EXPECTED_ATTRIBUTION_SCHEMA:
        raise RuntimeError("Unexpected guard-attribution report schema")
    if report.get("status") != "PASS":
        raise RuntimeError("Guard-attribution status is not PASS")
    if report.get("decision") != EXPECTED_ATTRIBUTION_DECISION:
        raise RuntimeError("Guard attribution did not request local replay")
    scope = report.get("scope")
    if not isinstance(scope, dict):
        raise RuntimeError("Guard attribution is missing scope")
    if not bool(scope.get("offline_postprocessing_only")):
        raise RuntimeError("Guard attribution was not offline-only")
    for key in (
        "hardware_accessed",
        "robot_object_created",
        "serial_port_opened",
        "command_sent",
        "hardware_deployment_authorized",
        "physical_safety_certified",
    ):
        if bool(scope.get(key, True)):
            raise RuntimeError(f"Guard-attribution scope violation: {key}")

    integrity = report.get("integrity")
    events = report.get("event_summary")
    episodes = report.get("episodes")
    if not isinstance(integrity, dict) or not isinstance(events, dict):
        raise RuntimeError("Guard attribution is missing integrity/event summary")
    if not isinstance(episodes, dict):
        raise RuntimeError("Guard attribution is missing episode summary")
    if int(integrity.get("commands_checked", 0)) <= 0:
        raise RuntimeError("Guard attribution checked no commands")
    if int(events.get("soft_clip_joint_events", 0)) <= 0:
        raise RuntimeError("Guard attribution did not find the expected drift")
    if int(events.get("soft_clip_events_after_command_state_drift_gt_joint_tracking_limit", -1)) != int(
        events.get("soft_clip_joint_events", -2)
    ):
        raise RuntimeError(
            "Not all counterfactual soft clips followed large command-state drift"
        )
    return {
        "commands_checked": int(integrity["commands_checked"]),
        "commands_with_any_rate_clip": int(
            sum(
                int(value["commands_with_any_rate_clip"])
                for value in episodes.values()
            )
        ),
        "commands_with_any_soft_clip": int(
            sum(
                int(value["commands_with_any_soft_clip"])
                for value in episodes.values()
            )
        ),
        "rate_clip_joint_events": int(events["rate_clip_joint_events"]),
        "soft_clip_joint_events": int(events["soft_clip_joint_events"]),
    }


def make_anchor_plan(
    *,
    args: argparse.Namespace,
    episodes: np.ndarray,
) -> tuple[str, list[tuple[int, int]]]:
    episode_lengths = {
        episode: int(np.sum(episodes == episode))
        for episode in range(EXPECTED_VALIDATION_EPISODES)
    }
    if args.anchors is not None:
        anchors = list(args.anchors)
        run_kind = "targeted"
    else:
        selected_episodes = (
            list(range(EXPECTED_VALIDATION_EPISODES))
            if args.episode_indices is None
            else list(args.episode_indices)
        )
        anchors = []
        for episode in selected_episodes:
            last_valid_start = episode_lengths[episode] - CHUNK_SIZE
            frames = list(range(0, last_valid_start + 1, EXECUTION_HORIZON))
            if args.max_replans_per_episode > 0:
                frames = frames[: args.max_replans_per_episode]
            anchors.extend((episode, frame) for frame in frames)
        run_kind = (
            "full"
            if selected_episodes == list(range(EXPECTED_VALIDATION_EPISODES))
            and args.max_replans_per_episode == 0
            else "smoke"
        )

    if not anchors:
        raise RuntimeError("Anchor plan is empty")
    for episode, frame in anchors:
        if episode < 0 or episode >= EXPECTED_VALIDATION_EPISODES:
            raise RuntimeError(f"Invalid anchor episode: {(episode, frame)}")
        if frame < 0 or frame % EXECUTION_HORIZON != 0:
            raise RuntimeError(
                f"Anchor frame must be nonnegative and divisible by five: "
                f"{(episode, frame)}"
            )
        if frame > episode_lengths[episode] - CHUNK_SIZE:
            raise RuntimeError(
                f"Anchor lacks a full ten-frame target chunk: {(episode, frame)}"
            )
    return run_kind, anchors


def main() -> int:
    args = parse_args()

    print("===== LOAD FROZEN OFFLINE CONTRACT AND PRIOR ATTRIBUTION =====")
    contract_path = args.contract.expanduser().resolve()
    contract = load_json(contract_path)
    limits = validate_contract(contract)
    calibration_path = args.calibration.expanduser().resolve()
    calibration = load_json(calibration_path)
    validate_calibration(contract, calibration)
    attribution_path = args.attribution_report.expanduser().resolve()
    attribution = validate_attribution(load_json(attribution_path))
    print("contract, calibration, and counterfactual attribution: PASS")
    print(
        "counterfactual baseline: "
        f"commands={attribution['commands_checked']} "
        f"rate_any={attribution['commands_with_any_rate_clip']} "
        f"soft_any={attribution['commands_with_any_soft_clip']} "
        f"soft_joint_events={attribution['soft_clip_joint_events']}"
    )
    print("hardware access allowed by contract: false")

    print("\n===== VERIFY FROZEN 11K RELEASE =====")
    release = verify_release(args.candidate_root, args.source_checkpoint_root)
    print("manifest, selection, and source-checkpoint identity: PASS")

    print("\n===== LOAD FROZEN V3 VALIDATION DATASET =====")
    validation_dataset = load_dataset(
        args.validation_repo_id,
        args.validation_dataset_root,
        args.video_backend,
    )
    val_state, val_action, val_episode, val_frame = dataset_arrays(
        validation_dataset
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
    sequential = validate_sequential_delta_semantics(
        label="validation",
        states=val_state,
        actions=val_action,
        episodes=val_episode,
    )
    startup = validate_dataset_startup(
        label="validation",
        states=val_state,
        actions=val_action,
        episodes=val_episode,
        home=limits["home"],
        tolerance=float(limits["startup_tolerance"][0]),
    )
    target_commands = (
        val_state[:, 6:12].astype(np.float64)
        + val_action.astype(np.float64)
    )
    target_soft_violation = (
        target_commands < limits["soft_min"].reshape(1, -1)
        - limits["numerical_tolerance"][0]
    ) | (
        target_commands > limits["soft_max"].reshape(1, -1)
        + limits["numerical_tolerance"][0]
    )
    target_soft_violation_by_joint = np.sum(target_soft_violation, axis=0)
    if np.any(target_soft_violation_by_joint):
        raise RuntimeError(
            "Recorded validation targets violate soft limits: "
            f"{target_soft_violation_by_joint.astype(int).tolist()}"
        )
    run_kind, anchors = make_anchor_plan(args=args, episodes=val_episode)
    print(
        f"validation: frames={len(val_state)}, episodes={startup['episode_count']} PASS"
    )
    print(
        "sequential command/delta semantics: PASS; "
        f"max={max(sequential['max_next_previous_command_error'], sequential['max_next_previous_delta_error']):.10f}"
    )
    print("recorded validation target soft-limit violations: [0, 0, 0, 0, 0, 0]")
    print(f"run kind: {run_kind}")
    print(f"replan anchors: {anchors}")

    print("\n===== LOAD POLICY (NO ROBOT OBJECT) =====")
    policy, device = load_policy(
        args.candidate_root.resolve() / "pretrained_model"
    )
    print("frozen ACT 11K policy: PASS")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "teacher_forced_local_commands.csv"
    report_path = output_dir / "teacher_forced_local_replay_report.json"
    if csv_path.exists() or report_path.exists():
        raise FileExistsError(
            "Refusing to overwrite existing teacher-forced local outputs"
        )

    metadata_columns = [
        "run_kind",
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
        "recorded_previous_command",
        "local_previous_command",
        "local_state_gap_before",
        "recorded_target_delta",
        "raw_predicted_delta",
        "raw_delta_abs_error",
        "rate_limited_delta",
        "unbounded_command",
        "guarded_command",
        "actual_guarded_delta",
        "recorded_target_command",
        "guarded_target_abs_error",
        "rate_clipped",
        "soft_clipped",
    ]
    csv_columns = metadata_columns + [
        f"{field}.{motor}" for motor in MOTORS for field in joint_fields
    ]

    stats: dict[str, dict[str, list[float]]] = {
        motor: {
            "raw_abs_delta": [],
            "raw_delta_abs_error": [],
            "local_state_gap_before": [],
            "guarded_target_abs_error": [],
            "rate_clipped": [],
            "soft_clipped": [],
            "at_soft_min": [],
            "at_soft_max": [],
        }
        for motor in MOTORS
    }
    episode_stats: dict[int, dict[str, Any]] = {}
    mapping = frame_map(val_episode, val_frame)
    total_commands = 0
    any_rate_commands = 0
    any_soft_commands = 0
    max_guard_invariant_error = 0.0
    max_recorded_item_state_error = 0.0

    print("\n===== TEACHER-FORCED LOCAL FIVE-COMMAND REPLAY =====")
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
        writer.writeheader()

        for replan_index, (episode, replan_frame) in enumerate(anchors):
            global_index = mapping[(episode, replan_frame)]
            recorded_item = validation_dataset[int(global_index)]
            item_state = recorded_item[STATE_KEY]
            if not isinstance(item_state, torch.Tensor):
                item_state = torch.as_tensor(item_state)
            item_state_array = (
                item_state.detach().cpu().float().numpy().reshape(-1)
            )
            if item_state_array.shape != (EXPECTED_STATE_DIM,):
                raise RuntimeError("Recorded dataset item state is not 18D")
            item_error = float(
                np.max(
                    np.abs(
                        item_state_array.astype(np.float64)
                        - val_state[global_index].astype(np.float64)
                    )
                )
            )
            max_recorded_item_state_error = max(
                max_recorded_item_state_error, item_error
            )
            if item_error > limits["numerical_tolerance"][0]:
                raise RuntimeError(
                    f"Dataset item/parquet state mismatch: {item_error}"
                )

            batch = item_batch(recorded_item, device)
            chunk = infer_chunk(policy, batch, device)
            local_last_command = val_state[
                global_index, 6:12
            ].astype(np.float64).copy()

            current_episode_stats = episode_stats.setdefault(
                episode,
                {
                    "replan_count": 0,
                    "command_count": 0,
                    "commands_with_any_rate_clip": 0,
                    "commands_with_any_soft_clip": 0,
                    "max_local_state_gap_before": 0.0,
                    "max_guarded_target_abs_error": 0.0,
                },
            )
            current_episode_stats["replan_count"] += 1

            for substep in range(EXECUTION_HORIZON):
                dataset_frame = replan_frame + substep
                step_global = mapping[(episode, dataset_frame)]
                recorded_actual = val_state[
                    step_global, :6
                ].astype(np.float64)
                recorded_previous = val_state[
                    step_global, 6:12
                ].astype(np.float64)
                recorded_delta = val_action[step_global].astype(np.float64)
                recorded_target = recorded_previous + recorded_delta
                raw_delta = chunk[substep].astype(np.float64)
                if raw_delta.shape != (EXPECTED_ACTION_DIM,):
                    raise RuntimeError("Predicted delta is not 6D")
                if not np.isfinite(raw_delta).all():
                    raise FloatingPointError("Predicted delta contains NaN/Inf")

                local_state_gap = np.abs(
                    local_last_command - recorded_previous
                )
                rate_limited_delta = np.clip(
                    raw_delta, -limits["max_delta"], limits["max_delta"]
                )
                unbounded_command = local_last_command + rate_limited_delta
                guarded_command = np.clip(
                    unbounded_command, limits["soft_min"], limits["soft_max"]
                )
                actual_guarded_delta = guarded_command - local_last_command
                raw_delta_abs_error = np.abs(raw_delta - recorded_delta)
                guarded_target_abs_error = np.abs(
                    guarded_command - recorded_target
                )
                rate_clipped = (
                    np.abs(raw_delta - rate_limited_delta)
                    > limits["numerical_tolerance"][0]
                )
                soft_clipped = (
                    np.abs(unbounded_command - guarded_command)
                    > limits["numerical_tolerance"][0]
                )
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
                            np.maximum(
                                limits["soft_min"] - guarded_command, 0.0
                            )
                        )
                    ),
                    float(
                        np.max(
                            np.maximum(
                                guarded_command - limits["soft_max"], 0.0
                            )
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
                total_commands += 1
                any_rate_commands += int(any_rate)
                any_soft_commands += int(any_soft)
                current_episode_stats["command_count"] += 1
                current_episode_stats["commands_with_any_rate_clip"] += int(
                    any_rate
                )
                current_episode_stats["commands_with_any_soft_clip"] += int(
                    any_soft
                )
                current_episode_stats["max_local_state_gap_before"] = max(
                    current_episode_stats["max_local_state_gap_before"],
                    float(np.max(local_state_gap)),
                )
                current_episode_stats["max_guarded_target_abs_error"] = max(
                    current_episode_stats["max_guarded_target_abs_error"],
                    float(np.max(guarded_target_abs_error)),
                )

                row: dict[str, Any] = {
                    "run_kind": run_kind,
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
                        "recorded_previous_command": recorded_previous[joint],
                        "local_previous_command": local_last_command[joint],
                        "local_state_gap_before": local_state_gap[joint],
                        "recorded_target_delta": recorded_delta[joint],
                        "raw_predicted_delta": raw_delta[joint],
                        "raw_delta_abs_error": raw_delta_abs_error[joint],
                        "rate_limited_delta": rate_limited_delta[joint],
                        "unbounded_command": unbounded_command[joint],
                        "guarded_command": guarded_command[joint],
                        "actual_guarded_delta": actual_guarded_delta[joint],
                        "recorded_target_command": recorded_target[joint],
                        "guarded_target_abs_error": guarded_target_abs_error[joint],
                        "rate_clipped": int(rate_clipped[joint]),
                        "soft_clipped": int(soft_clipped[joint]),
                    }
                    for field, value in values.items():
                        row[f"{field}.{motor}"] = value

                    motor_stats = stats[motor]
                    motor_stats["raw_abs_delta"].append(
                        float(abs(raw_delta[joint]))
                    )
                    motor_stats["raw_delta_abs_error"].append(
                        float(raw_delta_abs_error[joint])
                    )
                    motor_stats["local_state_gap_before"].append(
                        float(local_state_gap[joint])
                    )
                    motor_stats["guarded_target_abs_error"].append(
                        float(guarded_target_abs_error[joint])
                    )
                    motor_stats["rate_clipped"].append(
                        float(rate_clipped[joint])
                    )
                    motor_stats["soft_clipped"].append(
                        float(soft_clipped[joint])
                    )
                    motor_stats["at_soft_min"].append(
                        float(
                            abs(guarded_command[joint] - limits["soft_min"][joint])
                            <= limits["numerical_tolerance"][0]
                        )
                    )
                    motor_stats["at_soft_max"].append(
                        float(
                            abs(guarded_command[joint] - limits["soft_max"][joint])
                            <= limits["numerical_tolerance"][0]
                        )
                    )
                writer.writerow(row)
                local_last_command = guarded_command

            if (replan_index + 1) % 50 == 0 or replan_index + 1 == len(anchors):
                print(
                    f"processed replans={replan_index + 1}/{len(anchors)} "
                    f"commands={total_commands}"
                )

    print("\n===== TEACHER-FORCED LOCAL METRICS =====")
    print(
        "joint             raw_max  delta_mae  local_gap_p95  local_gap_max  "
        "command_mae  command_p95  rate_clip  soft_clip"
    )
    joint_report: dict[str, Any] = {}
    for motor in MOTORS:
        motor_stats = stats[motor]
        raw_summary = scalar_summary(motor_stats["raw_abs_delta"])
        delta_error_summary = scalar_summary(
            motor_stats["raw_delta_abs_error"]
        )
        gap_summary = scalar_summary(motor_stats["local_state_gap_before"])
        command_error_summary = scalar_summary(
            motor_stats["guarded_target_abs_error"]
        )
        rate_count = int(sum(motor_stats["rate_clipped"]))
        soft_count = int(sum(motor_stats["soft_clipped"]))
        boundary_min_count = int(sum(motor_stats["at_soft_min"]))
        boundary_max_count = int(sum(motor_stats["at_soft_max"]))
        joint_report[motor] = {
            "raw_abs_delta": raw_summary,
            "raw_delta_abs_error": delta_error_summary,
            "local_state_gap_before": gap_summary,
            "guarded_target_abs_error": command_error_summary,
            "rate_clip_count": rate_count,
            "rate_clip_rate": rate_count / total_commands,
            "soft_clip_count": soft_count,
            "soft_clip_rate": soft_count / total_commands,
            "at_soft_min_count": boundary_min_count,
            "at_soft_max_count": boundary_max_count,
        }
        print(
            f"{motor:16s} "
            f"{raw_summary['max']:8.5f} "
            f"{delta_error_summary['mean']:10.5f} "
            f"{gap_summary['p95']:13.5f} "
            f"{gap_summary['max']:13.5f} "
            f"{command_error_summary['mean']:11.5f} "
            f"{command_error_summary['p95']:11.5f} "
            f"{rate_count / total_commands:9.3%} "
            f"{soft_count / total_commands:9.3%}"
        )

    if run_kind == "targeted":
        decision = (
            "TARGETED_LOCAL_REPLAY_FOUND_SOFT_CLIPS_REVIEW_BEFORE_FULL"
            if any_soft_commands
            else "TARGETED_LOCAL_REPLAY_NO_SOFT_CLIPS_RUN_FULL_NEXT"
        )
    elif run_kind == "smoke":
        decision = (
            "SMOKE_LOCAL_REPLAY_FOUND_SOFT_CLIPS_REVIEW_BEFORE_FULL"
            if any_soft_commands
            else "SMOKE_LOCAL_REPLAY_PASS_RUN_FULL_NEXT"
        )
    elif any_soft_commands:
        decision = "FULL_LOCAL_REPLAY_REVIEW_SOFT_CLIPS"
    elif any_rate_commands:
        decision = "FULL_LOCAL_REPLAY_NO_SOFT_CLIPS_REVIEW_RATE_CLIPS"
    else:
        decision = "FULL_LOCAL_REPLAY_NO_GUARD_CLIPS_REVIEW_RUNTIME_DESIGN_NEXT"

    try:
        lerobot_version = importlib.metadata.version("lerobot")
    except importlib.metadata.PackageNotFoundError:
        lerobot_version = "unknown-editable-install"

    report = {
        "schema_version": OUTPUT_SCHEMA,
        "status": "PASS",
        "decision": decision,
        "scope": {
            "offline_only": True,
            "recorded_validation_images_only": True,
            "recorded_validation_state_at_every_replan": True,
            "simulated_command_state_resets_every_replan": True,
            "local_recursive_commands_per_replan": EXECUTION_HORIZON,
            "closed_loop_task_success_evaluated": False,
            "dataset_modified": False,
            "hardware_accessed": False,
            "robot_object_created": False,
            "serial_port_opened": False,
            "command_sent": False,
            "hardware_deployment_authorized": False,
            "physical_safety_certified": False,
        },
        "interpretation": (
            "This in-distribution local replay isolates five-command queue "
            "behavior by resetting command state at each replan. It does not "
            "simulate visual or physical closed-loop response."
        ),
        "environment": {
            "lerobot_version": lerobot_version,
            "torch_version": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0),
        },
        "paths": {
            "contract": str(contract_path),
            "calibration": str(calibration_path),
            "attribution_report": str(attribution_path),
            "candidate": str(args.candidate_root.resolve()),
            "validation_dataset": str(args.validation_dataset_root.resolve()),
        },
        "release": release,
        "prior_counterfactual_baseline": attribution,
        "validation_integrity": {
            "metadata": "PASS",
            "sequential_semantics": sequential,
            "startup": startup,
            "recorded_target_soft_violation_count_by_joint": {
                motor: int(target_soft_violation_by_joint[index])
                for index, motor in enumerate(MOTORS)
            },
            "max_recorded_item_state_error": max_recorded_item_state_error,
        },
        "run": {
            "kind": run_kind,
            "anchors": [
                {"episode_index": episode, "replan_frame": frame}
                for episode, frame in anchors
            ],
            "total_replans": len(anchors),
            "total_commands": total_commands,
            "commands_with_any_rate_clip": any_rate_commands,
            "commands_with_any_rate_clip_rate": any_rate_commands
            / total_commands,
            "commands_with_any_soft_clip": any_soft_commands,
            "commands_with_any_soft_clip_rate": any_soft_commands
            / total_commands,
            "max_guard_invariant_error": max_guard_invariant_error,
        },
        "joint_metrics": joint_report,
        "episodes": {
            str(episode): values
            for episode, values in sorted(episode_stats.items())
        },
        "required_next_step": (
            "Review local rate/soft clips and command errors. Hardware remains "
            "blocked. Run the full local replay only if the targeted result "
            "does not request pre-full review."
        ),
    }
    report_path.write_text(
        json.dumps(to_jsonable(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n===== DECISION =====")
    print(f"decision='{decision}'")
    print(f"commands with any rate clip: {any_rate_commands}/{total_commands}")
    print(f"commands with any soft clip: {any_soft_commands}/{total_commands}")
    print(f"max guard invariant error: {max_guard_invariant_error:.10f}")
    print("HARDWARE DEPLOYMENT REMAINS BLOCKED.")
    print("NO ROBOT OBJECT WAS CREATED. NO SERIAL PORT WAS OPENED.")
    print("NO COMMAND WAS SENT. NO DATASET WAS MODIFIED.")
    print("\n===== OUTPUT =====")
    print(report_path)
    print(csv_path)
    print("ACT V3 DELTA TEACHER-FORCED LOCAL REPLAY: PASS")

    del policy, validation_dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
