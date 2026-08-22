#!/usr/bin/env python
"""Attribute ACT v3 decoded-command envelope excursions offline.

This diagnostic explains the aggregate ``outside_train_cmd`` percentage from
the deployment-contract audit. It separates:

1. validation commands already outside the train-only min/max envelope;
2. anchor base commands already outside that envelope;
3. recorded human target commands outside the envelope;
4. model-decoded commands outside the envelope;
5. model steps that newly cross from inside to outside; and
6. the magnitude of every excursion.

All thresholds in this file are numerical attribution bins in dataset units.
They are not robot limits or physical safety limits. The script never imports
or constructs robot, serial, motor, calibration, teleoperation, or live-camera
objects. It never sends a command.
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
    EXPECTED_TRAIN_EPISODES,
    EXPECTED_TRAIN_FRAMES,
    EXPECTED_VALIDATION_EPISODES,
    EXPECTED_VALIDATION_FRAMES,
    MOTORS,
    build_anchors,
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
    parser.add_argument("--video-backend", default="torchcodec")
    parser.add_argument("--uniform-samples-per-episode", type=int, default=12)
    parser.add_argument("--moving-samples-per-episode", type=int, default=4)
    parser.add_argument("--stationary-samples-per-episode", type=int, default=4)
    parser.add_argument("--stress-min-separation", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/eval/act_red_cube_v3_delta_command_envelope"
        ),
    )
    args = parser.parse_args()
    for name in (
        "uniform_samples_per_episode",
        "moving_samples_per_episode",
        "stationary_samples_per_episode",
        "stress_min_separation",
    ):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def outside_mask(
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    tolerance: float = 0.0,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return (array < lower - tolerance) | (array > upper + tolerance)


def excursion_distance(
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.maximum(lower - array, 0.0) + np.maximum(array - upper, 0.0)


def finite_summary(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    if not np.isfinite(array).all():
        raise FloatingPointError("Cannot summarize NaN/Inf")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "p99": float(np.percentile(array, 99.0)),
        "max": float(np.max(array)),
    }


def rate(mask: np.ndarray) -> float:
    array = np.asarray(mask, dtype=bool)
    if array.size == 0:
        raise RuntimeError("Cannot calculate a rate from an empty array")
    return float(np.mean(array))


def main() -> int:
    args = parse_args()

    print("===== VERIFY FROZEN 11K RELEASE =====")
    release = verify_release(args.candidate_root, args.source_checkpoint_root)
    print("manifest, selection, and source-checkpoint identity: PASS")

    print("\n===== LOAD FROZEN DATASETS =====")
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
    print(f"train: frames={len(train_state)} episodes={EXPECTED_TRAIN_EPISODES}")
    print(
        f"validation: frames={len(val_state)} "
        f"episodes={EXPECTED_VALIDATION_EPISODES}"
    )

    train_commands = (
        train_state[:, 6:12].astype(np.float64)
        + train_action.astype(np.float64)
    )
    validation_commands = (
        val_state[:, 6:12].astype(np.float64)
        + val_action.astype(np.float64)
    )
    train_min = np.min(train_commands, axis=0)
    train_max = np.max(train_commands, axis=0)
    validation_min = np.min(validation_commands, axis=0)
    validation_max = np.max(validation_commands, axis=0)
    action_scale = np.std(train_action.astype(np.float64), axis=0)
    action_scale = np.maximum(action_scale, 1e-3)

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

    print("\n===== 11K OFFLINE INFERENCE =====")
    policy, device = load_policy(
        args.candidate_root.resolve() / "pretrained_model"
    )
    mapping = frame_map(val_episode, val_frame)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    bases: list[np.ndarray] = []

    for number, anchor in enumerate(anchors, start=1):
        item = validation_dataset[int(anchor.global_index)]
        batch = item_batch(item, device)
        predictions.append(infer_chunk(policy, batch, device))
        bases.append(val_state[anchor.global_index, 6:12].astype(np.float64))
        target_indices = [
            mapping[(anchor.episode_index, anchor.frame_index + offset)]
            for offset in range(CHUNK_SIZE)
        ]
        targets.append(
            val_action[np.asarray(target_indices, dtype=np.int64)].astype(
                np.float64
            )
        )
        if number % 40 == 0 or number == len(anchors):
            print(f"processed {number}/{len(anchors)}")

    prediction = np.stack(predictions).astype(np.float64)
    target = np.stack(targets).astype(np.float64)
    base = np.stack(bases).astype(np.float64)
    predicted_delta = prediction[:, :EXECUTION_HORIZON]
    target_delta = target[:, :EXECUTION_HORIZON]
    predicted_after = base[:, None, :] + np.cumsum(predicted_delta, axis=1)
    target_after = base[:, None, :] + np.cumsum(target_delta, axis=1)
    predicted_before = np.concatenate(
        [base[:, None, :], predicted_after[:, :-1, :]], axis=1
    )
    target_before = np.concatenate(
        [base[:, None, :], target_after[:, :-1, :]], axis=1
    )

    full_validation_distance = excursion_distance(
        validation_commands, train_min, train_max
    )
    base_distance = excursion_distance(base, train_min, train_max)
    target_distance = excursion_distance(target_after, train_min, train_max)
    predicted_distance = excursion_distance(
        predicted_after, train_min, train_max
    )
    predicted_before_outside = outside_mask(
        predicted_before, train_min, train_max
    )
    predicted_after_outside = outside_mask(
        predicted_after, train_min, train_max
    )
    target_before_outside = outside_mask(target_before, train_min, train_max)
    target_after_outside = outside_mask(target_after, train_min, train_max)

    rows: list[dict[str, Any]] = []
    joint_details: dict[str, Any] = {}
    for joint, motor in enumerate(MOTORS):
        pred_before_out = predicted_before_outside[:, :, joint]
        pred_after_out = predicted_after_outside[:, :, joint]
        target_before_out = target_before_outside[:, :, joint]
        target_after_out = target_after_outside[:, :, joint]
        pred_distance = predicted_distance[:, :, joint]
        tgt_distance = target_distance[:, :, joint]
        val_distance = full_validation_distance[:, joint]

        predicted_new_crossing = (~pred_before_out) & pred_after_out
        predicted_inherited_outside = pred_before_out & pred_after_out
        predicted_returned_inside = pred_before_out & (~pred_after_out)
        target_new_crossing = (~target_before_out) & target_after_out

        predicted_positive_excess = np.maximum(
            predicted_after[:, :, joint] - train_max[joint], 0.0
        )
        predicted_negative_excess = np.maximum(
            train_min[joint] - predicted_after[:, :, joint], 0.0
        )

        detail = {
            "train_command_min": float(train_min[joint]),
            "train_command_max": float(train_max[joint]),
            "validation_command_min": float(validation_min[joint]),
            "validation_command_max": float(validation_max[joint]),
            "anchor_base_min": float(np.min(base[:, joint])),
            "anchor_base_max": float(np.max(base[:, joint])),
            "target_decoded_min": float(np.min(target_after[:, :, joint])),
            "target_decoded_max": float(np.max(target_after[:, :, joint])),
            "predicted_decoded_min": float(
                np.min(predicted_after[:, :, joint])
            ),
            "predicted_decoded_max": float(
                np.max(predicted_after[:, :, joint])
            ),
            "full_validation_outside_strict_rate": rate(val_distance > 0.0),
            "full_validation_outside_gt_0p01_rate": rate(val_distance > 0.01),
            "anchor_base_outside_strict_rate": rate(
                base_distance[:, joint] > 0.0
            ),
            "target_outside_strict_rate": rate(tgt_distance > 0.0),
            "target_outside_gt_0p01_rate": rate(tgt_distance > 0.01),
            "target_new_crossing_rate": rate(target_new_crossing),
            "predicted_outside_strict_rate": rate(pred_distance > 0.0),
            "predicted_outside_gt_0p01_rate": rate(pred_distance > 0.01),
            "predicted_outside_gt_0p10_rate": rate(pred_distance > 0.10),
            "predicted_new_crossing_rate": rate(predicted_new_crossing),
            "predicted_inherited_outside_rate": rate(
                predicted_inherited_outside
            ),
            "predicted_returned_inside_rate": rate(predicted_returned_inside),
            "predicted_excursion_all": finite_summary(pred_distance),
            "predicted_excursion_outside_only": finite_summary(
                pred_distance[pred_distance > 0.0]
            ),
            "target_excursion_outside_only": finite_summary(
                tgt_distance[tgt_distance > 0.0]
            ),
            "predicted_positive_excess_max": float(
                np.max(predicted_positive_excess)
            ),
            "predicted_negative_excess_max": float(
                np.max(predicted_negative_excess)
            ),
            "predicted_delta_mean": float(
                np.mean(predicted_delta[:, :, joint])
            ),
            "target_delta_mean": float(np.mean(target_delta[:, :, joint])),
            "predicted_vs_target_command_mae": float(
                np.mean(
                    np.abs(
                        predicted_after[:, :, joint]
                        - target_after[:, :, joint]
                    )
                )
            ),
        }
        joint_details[motor] = detail
        outside_summary = detail["predicted_excursion_outside_only"]
        rows.append(
            {
                "joint": motor,
                "train_min": detail["train_command_min"],
                "train_max": detail["train_command_max"],
                "validation_outside_rate": detail[
                    "full_validation_outside_strict_rate"
                ],
                "base_outside_rate": detail["anchor_base_outside_strict_rate"],
                "target_outside_rate": detail["target_outside_strict_rate"],
                "predicted_outside_rate": detail[
                    "predicted_outside_strict_rate"
                ],
                "predicted_outside_gt_0p01_rate": detail[
                    "predicted_outside_gt_0p01_rate"
                ],
                "predicted_outside_gt_0p10_rate": detail[
                    "predicted_outside_gt_0p10_rate"
                ],
                "predicted_new_crossing_rate": detail[
                    "predicted_new_crossing_rate"
                ],
                "predicted_inherited_outside_rate": detail[
                    "predicted_inherited_outside_rate"
                ],
                "outside_excursion_p95": outside_summary["p95"],
                "outside_excursion_max": outside_summary["max"],
                "predicted_vs_target_command_mae": detail[
                    "predicted_vs_target_command_mae"
                ],
            }
        )

    print("\n===== COMMAND ENVELOPE ATTRIBUTION =====")
    print(
        "joint             val_out  base_out  target_out  pred_out  "
        "pred>0.01  pred>0.10  new_cross  excursion_max"
    )
    for row in rows:
        print(
            f"{row['joint']:<16} "
            f"{row['validation_outside_rate']:8.3%} "
            f"{row['base_outside_rate']:9.3%} "
            f"{row['target_outside_rate']:11.3%} "
            f"{row['predicted_outside_rate']:8.3%} "
            f"{row['predicted_outside_gt_0p01_rate']:10.3%} "
            f"{row['predicted_outside_gt_0p10_rate']:10.3%} "
            f"{row['predicted_new_crossing_rate']:9.3%} "
            f"{row['outside_excursion_max']:14.6f}"
        )

    wrist = joint_details["wrist_flex"]
    print("\n===== WRIST_FLEX FOCUS =====")
    print(
        "train envelope: "
        f"[{wrist['train_command_min']:.6f}, "
        f"{wrist['train_command_max']:.6f}]"
    )
    print(
        "validation full range: "
        f"[{wrist['validation_command_min']:.6f}, "
        f"{wrist['validation_command_max']:.6f}]"
    )
    print(
        "predicted decoded range: "
        f"[{wrist['predicted_decoded_min']:.6f}, "
        f"{wrist['predicted_decoded_max']:.6f}]"
    )
    print(
        "predicted outside rate: "
        f"strict={wrist['predicted_outside_strict_rate']:.3%}, "
        f">0.01={wrist['predicted_outside_gt_0p01_rate']:.3%}, "
        f">0.10={wrist['predicted_outside_gt_0p10_rate']:.3%}"
    )
    print(
        "new crossing vs inherited outside: "
        f"{wrist['predicted_new_crossing_rate']:.3%} vs "
        f"{wrist['predicted_inherited_outside_rate']:.3%}"
    )
    print(
        "outside-only excursion: "
        f"p95={wrist['predicted_excursion_outside_only']['p95']:.6f}, "
        f"max={wrist['predicted_excursion_outside_only']['max']:.6f}"
    )

    try:
        lerobot_version = importlib.metadata.version("lerobot")
    except importlib.metadata.PackageNotFoundError:
        lerobot_version = "unknown-editable-install"

    report: dict[str, Any] = {
        "schema_version": "act_v3_delta_command_envelope_attribution_v1",
        "status": "PASS",
        "decision": "ATTRIBUTION_COMPLETE_HARDWARE_REMAINS_BLOCKED",
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
        "protocol": {
            "unique_anchors": len(anchors),
            "phase_memberships": memberships,
            "execution_horizon": EXECUTION_HORIZON,
            "attribution_thresholds_in_dataset_units": [0.0, 0.01, 0.10],
            "threshold_warning": (
                "Numerical attribution bins only; not hardware limits."
            ),
        },
        "joint_details": joint_details,
        "interpretation_rule": {
            "validation_outside": (
                "Recorded validation absolute command lies outside train-only "
                "command min/max"
            ),
            "base_outside": (
                "Recorded previous command at an inference anchor already lies "
                "outside the train-only envelope"
            ),
            "target_outside": (
                "Recorded human target, cumulatively decoded for five steps, "
                "lies outside the train-only envelope"
            ),
            "predicted_new_crossing": (
                "Model-decoded command moves from inside to outside on that step"
            ),
            "predicted_inherited_outside": (
                "Model-decoded command was already outside before that step and "
                "remains outside"
            ),
        },
        "required_next_decision": (
            "Review wrist_flex new-crossing rate and excursion magnitude before "
            "designing the guarded dry-run adapter."
        ),
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "command_envelope_attribution_report.json"
    report_path.write_text(
        json.dumps(to_jsonable(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    csv_path = output_dir / "command_envelope_attribution.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    del policy
    del prediction
    del predictions
    gc.collect()
    torch.cuda.empty_cache()

    print("\n===== OUTPUT =====")
    print(report_path)
    print(csv_path)
    print("ACT V3 DELTA COMMAND ENVELOPE ATTRIBUTION: PASS")
    print("HARDWARE DEPLOYMENT REMAINS BLOCKED.")
    print("NO DATASET WAS MODIFIED. NO HARDWARE WAS ACCESSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
