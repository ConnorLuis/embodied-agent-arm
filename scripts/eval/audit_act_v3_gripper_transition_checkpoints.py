#!/usr/bin/env python
"""Offline full-chunk gripper-transition audit for ACT v3 checkpoints.

This script evaluates only frozen validation data and local model checkpoints.
It never imports a robot, motor bus, serial API, live camera, or command API.

The audit was added after the first Stage 3C episode completed 900 guarded
commands without ever producing a meaningful gripper-close transition.  It
answers two bounded questions in one run:

1. Did any existing checkpoint learn close and release transitions?
2. Are those transitions present in executable chunk steps 0..4, or only in
   discarded steps 5..9?

It also compares each checkpoint with the zero-delta baseline on the sparse
transition steps.  A checkpoint is not accepted merely because its global
average action error is low.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np


OUTPUT_SCHEMA = "so101_act_v3_gripper_transition_checkpoint_audit_v1"
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
CHUNK_SIZE = 10
EXECUTION_HORIZON = 5
EXPECTED_STATE_DIM = 18
EXPECTED_ACTION_DIM = 6
EXPECTED_VALIDATION_FRAMES = 10704
EXPECTED_VALIDATION_EPISODES = 12
EXPECTED_FPS = 15.0
GRIPPER_INDEX = 5
DIRECTIONS = ("close", "open")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument(
        "--train-output-root",
        type=Path,
        default=Path("outputs/train/act_red_cube_v3_delta_stage1"),
    )
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        nargs="+",
        default=[8000, 9000, 10000, 11000, 12000],
    )
    parser.add_argument(
        "--stage3c-run-dir",
        type=Path,
        default=Path(
            "outputs/eval/"
            "act_red_cube_v3_one_controlled_policy_episode_powerloss_"
            "20260821T101025Z"
        ),
    )
    parser.add_argument(
        "--target-step-threshold",
        type=float,
        default=0.25,
        help="Minimum absolute recorded gripper delta for an active step.",
    )
    parser.add_argument(
        "--target-window-amplitude",
        type=float,
        default=2.5,
        help="Minimum cumulative gripper displacement defining an event.",
    )
    parser.add_argument(
        "--prediction-fraction",
        type=float,
        default=0.25,
        help="Predicted/target amplitude fraction used for event recall.",
    )
    parser.add_argument(
        "--minimum-direction-recall",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--minimum-zero-baseline-improvement",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--minimum-median-amplitude-ratio",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--maximum-tail-only-rate",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/eval/act_red_cube_v3_gripper_transition_checkpoints"
        ),
    )
    args = parser.parse_args()

    args.checkpoint_steps = sorted(set(args.checkpoint_steps))
    if not args.checkpoint_steps or min(args.checkpoint_steps) <= 0:
        parser.error("--checkpoint-steps must contain positive values")
    for name in (
        "target_step_threshold",
        "target_window_amplitude",
        "prediction_fraction",
        "minimum_direction_recall",
        "minimum_zero_baseline_improvement",
        "minimum_median_amplitude_ratio",
        "maximum_tail_only_rate",
    ):
        value = float(getattr(args, name))
        if value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "prediction_fraction",
        "minimum_direction_recall",
        "minimum_zero_baseline_improvement",
        "minimum_median_amplitude_ratio",
        "maximum_tail_only_rate",
    ):
        value = float(getattr(args, name))
        if value > 1.0:
            parser.error(f"--{name.replace('_', '-')} must be <= 1")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise NotADirectoryError(resolved)
    return resolved


def load_dataset(repo_id: str, root: Path, video_backend: str) -> Any:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset_root = require_dir(root)
    try:
        return LeRobotDataset(
            repo_id=repo_id,
            root=dataset_root,
            video_backend=video_backend,
        )
    except TypeError:
        return LeRobotDataset(
            repo_id,
            root=dataset_root,
            video_backend=video_backend,
        )


def dataset_arrays(
    dataset: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hf = dataset.hf_dataset
    state = np.asarray(hf[STATE_KEY], dtype=np.float32)
    action = np.asarray(hf[ACTION_KEY], dtype=np.float32)
    episode = np.asarray(hf["episode_index"], dtype=np.int64)
    frame = np.asarray(hf["frame_index"], dtype=np.int64)
    return state, action, episode, frame


def dataset_fps(dataset: Any) -> float:
    meta = getattr(dataset, "meta", None)
    info = getattr(meta, "info", None)
    candidates = [
        getattr(dataset, "fps", None),
        getattr(meta, "fps", None),
        info.get("fps") if isinstance(info, dict) else None,
    ]
    for candidate in candidates:
        if candidate is not None:
            return float(candidate)
    raise RuntimeError("cannot determine validation dataset fps")


def frame_map(
    episodes: np.ndarray,
    frames: np.ndarray,
) -> dict[tuple[int, int], int]:
    result: dict[tuple[int, int], int] = {}
    for index, (episode, frame) in enumerate(
        zip(episodes, frames, strict=True)
    ):
        key = (int(episode), int(frame))
        if key in result:
            raise RuntimeError(f"duplicate validation episode/frame key: {key}")
        result[key] = index
    return result


def validate_dataset_metadata(
    *,
    dataset: Any,
    states: np.ndarray,
    actions: np.ndarray,
    episodes: np.ndarray,
    frames: np.ndarray,
) -> dict[str, Any]:
    if states.shape != (EXPECTED_VALIDATION_FRAMES, EXPECTED_STATE_DIM):
        raise RuntimeError(f"unexpected validation state shape: {states.shape}")
    if actions.shape != (EXPECTED_VALIDATION_FRAMES, EXPECTED_ACTION_DIM):
        raise RuntimeError(f"unexpected validation action shape: {actions.shape}")
    if episodes.shape != (EXPECTED_VALIDATION_FRAMES,):
        raise RuntimeError(f"unexpected episode-index shape: {episodes.shape}")
    if frames.shape != (EXPECTED_VALIDATION_FRAMES,):
        raise RuntimeError(f"unexpected frame-index shape: {frames.shape}")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise FloatingPointError("validation state/action contains NaN or Inf")
    unique_episodes = sorted(int(value) for value in np.unique(episodes))
    if unique_episodes != list(range(EXPECTED_VALIDATION_EPISODES)):
        raise RuntimeError(f"unexpected validation episodes: {unique_episodes}")
    for episode in unique_episodes:
        episode_frames = frames[episodes == episode]
        expected = np.arange(len(episode_frames), dtype=np.int64)
        if not np.array_equal(episode_frames, expected):
            raise RuntimeError(
                f"episode {episode} frames are not contiguous from zero"
            )
    fps = dataset_fps(dataset)
    if abs(fps - EXPECTED_FPS) > 1e-9:
        raise RuntimeError(f"validation fps={fps}, expected {EXPECTED_FPS}")
    return {
        "frames": int(len(states)),
        "episodes": len(unique_episodes),
        "fps": fps,
    }


def validate_sequential_delta_semantics(
    *,
    states: np.ndarray,
    actions: np.ndarray,
    episodes: np.ndarray,
) -> dict[str, float]:
    max_command_error = 0.0
    max_delta_error = 0.0
    max_start_delta = 0.0
    for episode in range(EXPECTED_VALIDATION_EPISODES):
        indices = np.flatnonzero(episodes == episode)
        if indices.size < 2:
            raise RuntimeError(f"episode {episode} is too short")
        initial_previous_delta = states[indices[0], 12:18]
        max_start_delta = max(
            max_start_delta,
            float(np.max(np.abs(initial_previous_delta))),
        )
        current = indices[:-1]
        following = indices[1:]
        expected_command = states[current, 6:12] + actions[current]
        command_error = np.max(
            np.abs(states[following, 6:12] - expected_command)
        )
        delta_error = np.max(
            np.abs(states[following, 12:18] - actions[current])
        )
        max_command_error = max(max_command_error, float(command_error))
        max_delta_error = max(max_delta_error, float(delta_error))
    tolerance = 1e-5
    if max(max_command_error, max_delta_error, max_start_delta) > tolerance:
        raise RuntimeError(
            "validation sequential delta semantics failed: "
            f"command={max_command_error}, delta={max_delta_error}, "
            f"startup={max_start_delta}"
        )
    return {
        "max_previous_command_recurrence_error": max_command_error,
        "max_previous_delta_shift_error": max_delta_error,
        "max_startup_previous_delta": max_start_delta,
    }


def feature_shape(feature: Any) -> tuple[int, ...]:
    value = feature.get("shape") if isinstance(feature, dict) else getattr(
        feature, "shape", None
    )
    if value is None:
        raise RuntimeError(f"policy feature has no shape: {feature!r}")
    return tuple(int(item) for item in value)


def verify_policy_contract(policy: Any) -> dict[str, Any]:
    config = policy.config
    if int(config.chunk_size) != CHUNK_SIZE:
        raise RuntimeError(f"chunk_size={config.chunk_size}, expected 10")
    if int(config.n_action_steps) != EXECUTION_HORIZON:
        raise RuntimeError(
            f"n_action_steps={config.n_action_steps}, expected 5"
        )
    expected_inputs = {STATE_KEY, FRONT_KEY, WRIST_KEY}
    if set(config.input_features) != expected_inputs:
        raise RuntimeError(
            f"unexpected policy inputs: {sorted(config.input_features)}"
        )
    if set(config.output_features) != {ACTION_KEY}:
        raise RuntimeError(
            f"unexpected policy outputs: {sorted(config.output_features)}"
        )
    if feature_shape(config.input_features[STATE_KEY]) != (EXPECTED_STATE_DIM,):
        raise RuntimeError("policy state feature is not 18-D")
    if feature_shape(config.input_features[FRONT_KEY]) != (3, 480, 480):
        raise RuntimeError("front image feature shape mismatch")
    if feature_shape(config.input_features[WRIST_KEY]) != (3, 480, 480):
        raise RuntimeError("wrist image feature shape mismatch")
    if feature_shape(config.output_features[ACTION_KEY]) != (EXPECTED_ACTION_DIM,):
        raise RuntimeError("policy action feature is not 6-D")
    return {
        "device": str(config.device),
        "chunk_size": int(config.chunk_size),
        "n_action_steps": int(config.n_action_steps),
    }


def load_policy(model_path: Path) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy

    model_root = require_dir(model_path)
    model_file = model_root / "model.safetensors"
    if not model_file.is_file():
        raise FileNotFoundError(model_file)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; checkpoint audit stopped")
    policy = ACTPolicy.from_pretrained(model_root, local_files_only=True)
    policy.eval()
    policy.reset()
    contract = verify_policy_contract(policy)
    return policy, torch.device(policy.config.device), contract


def image_tensor(value: Any, device: Any) -> Any:
    import torch

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
        raise RuntimeError(
            f"cannot infer image channel axis: {tuple(tensor.shape)}"
        )
    tensor = tensor.float()
    if float(tensor.max()) > 1.5:
        tensor = tensor / 255.0
    return tensor.unsqueeze(0).to(device=device)


def item_batch(item: dict[str, Any], device: Any) -> dict[str, Any]:
    import torch

    state = torch.as_tensor(
        item[STATE_KEY], dtype=torch.float32, device=device
    ).reshape(1, EXPECTED_STATE_DIM)
    return {
        STATE_KEY: state,
        FRONT_KEY: image_tensor(item[FRONT_KEY], device),
        WRIST_KEY: image_tensor(item[WRIST_KEY], device),
    }


def infer_chunk(
    policy: Any,
    batch: dict[str, Any],
    device: Any,
) -> np.ndarray:
    import torch

    policy.reset()
    with torch.inference_mode():
        chunk = policy.predict_action_chunk(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    result = chunk.detach().cpu().float().numpy()
    if result.shape != (1, CHUNK_SIZE, EXPECTED_ACTION_DIM):
        raise RuntimeError(f"unexpected ACT chunk shape: {result.shape}")
    if not np.isfinite(result).all():
        raise FloatingPointError("ACT output contains NaN or Inf")
    return result[0]


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def finite_float(value: Any, label: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise RuntimeError(f"{label} is not finite")
    return result


def cumulative_amplitude(values: Iterable[float], direction: str) -> float:
    array = np.asarray(list(values), dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise RuntimeError("transition delta vector is empty or non-finite")
    cumulative = np.concatenate(
        [np.zeros(1, dtype=np.float64), np.cumsum(array)]
    )
    if direction == "close":
        return float(max(0.0, -float(np.min(cumulative))))
    if direction == "open":
        return float(max(0.0, float(np.max(cumulative))))
    raise ValueError(direction)


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def event_threshold(target_amplitude: float, args: argparse.Namespace) -> float:
    return max(
        float(args.target_step_threshold),
        float(args.prediction_fraction) * float(target_amplitude),
    )


def build_event_anchors(
    *,
    actions: np.ndarray,
    episodes: np.ndarray,
    mapping: dict[tuple[int, int], int],
    args: argparse.Namespace,
) -> dict[tuple[int, int], dict[str, Any]]:
    anchors: dict[tuple[int, int], dict[str, Any]] = {}
    for episode in range(EXPECTED_VALIDATION_EPISODES):
        episode_indices = np.flatnonzero(episodes == episode)
        last_start = len(episode_indices) - CHUNK_SIZE
        if last_start < 0:
            raise RuntimeError(f"episode {episode} is shorter than chunk")
        for frame in range(0, last_start + 1, EXECUTION_HORIZON):
            indices = [mapping[(episode, frame + offset)] for offset in range(CHUNK_SIZE)]
            target = actions[indices, GRIPPER_INDEX].astype(np.float64)
            record: dict[str, Any] = {
                "episode": episode,
                "frame": frame,
                "target": target,
                "directions": {},
            }
            for direction in DIRECTIONS:
                full = cumulative_amplitude(target, direction)
                first = cumulative_amplitude(target[:EXECUTION_HORIZON], direction)
                tail = cumulative_amplitude(target[EXECUTION_HORIZON:], direction)
                active = (
                    target <= -float(args.target_step_threshold)
                    if direction == "close"
                    else target >= float(args.target_step_threshold)
                )
                if full >= float(args.target_window_amplitude):
                    record["directions"][direction] = {
                        "target_full_amplitude": full,
                        "target_first5_amplitude": first,
                        "target_tail5_amplitude": tail,
                        "active_mask": active,
                    }
            if record["directions"]:
                anchors[(episode, frame)] = record
    if not anchors:
        raise RuntimeError("no gripper transition anchors were found")
    for direction in DIRECTIONS:
        count = sum(direction in value["directions"] for value in anchors.values())
        first_count = sum(
            direction in value["directions"]
            and value["directions"][direction]["target_first5_amplitude"]
            >= float(args.target_window_amplitude)
            for value in anchors.values()
        )
        if count == 0 or first_count == 0:
            raise RuntimeError(
                f"validation data has no usable {direction} transition windows"
            )
    return anchors


def validate_stage3c_evidence(run_dir: Path) -> dict[str, Any]:
    resolved = run_dir.resolve()
    report_path = resolved / "one_controlled_policy_episode_report.json"
    command_path = resolved / "one_controlled_policy_episode_commands.csv"
    for path in (report_path, command_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("decision") != (
        "ONE_CONTROLLED_POLICY_EPISODE_EXECUTION_PASS_REVIEW_TASK_OUTCOME_NEXT"
    ):
        raise RuntimeError("Stage 3C source run did not pass execution")
    with command_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 900:
        raise RuntimeError(f"Stage 3C command count is {len(rows)}, expected 900")
    if sum(int(row["write_acknowledged"]) for row in rows) != 900:
        raise RuntimeError("Stage 3C did not acknowledge all commands")
    if any(
        int(row[field])
        for row in rows
        for field in ("any_rate_clip", "any_soft_clip", "tracking_tripped")
    ):
        raise RuntimeError("Stage 3C source run contains a clip or tracking trip")
    commands = [finite_float(row["sent_command_gripper"], "gripper command") for row in rows]
    raw = [finite_float(row["raw_delta_gripper"], "gripper delta") for row in rows]
    actual = [finite_float(row["actual_gripper"], "gripper actual") for row in rows]
    return {
        "report": str(report_path),
        "commands_csv": str(command_path),
        "report_sha256": sha256(report_path),
        "commands_sha256": sha256(command_path),
        "commands": len(rows),
        "gripper_command_min": min(commands),
        "gripper_command_max": max(commands),
        "gripper_command_net_change": sum(raw),
        "gripper_raw_abs_max": max(abs(value) for value in raw),
        "gripper_actual_min": min(actual),
        "gripper_actual_max": max(actual),
        "meaningful_close_observed": False,
    }


def summarize_direction(
    *,
    direction: str,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    relevant = [row for row in records if row["direction"] == direction]
    first_events = [
        row
        for row in relevant
        if row["target_first5_amplitude"]
        >= float(args.target_window_amplitude)
    ]
    if not relevant or not first_events:
        raise RuntimeError(f"checkpoint has no {direction} event records")

    detected = 0
    tail_only = 0
    amplitude_ratios: list[float] = []
    model_errors: list[float] = []
    zero_errors: list[float] = []
    for row in first_events:
        threshold = event_threshold(row["target_first5_amplitude"], args)
        is_detected = row["predicted_first5_amplitude"] >= threshold
        detected += int(is_detected)
        tail_threshold = event_threshold(row["target_full_amplitude"], args)
        is_tail_only = (
            not is_detected
            and row["predicted_tail5_amplitude"] >= tail_threshold
        )
        tail_only += int(is_tail_only)
        amplitude_ratios.append(
            row["predicted_first5_amplitude"]
            / row["target_first5_amplitude"]
        )
        model_errors.extend(row["active_model_abs_errors"])
        zero_errors.extend(row["active_zero_abs_errors"])

    model_mae = float(np.mean(model_errors))
    zero_mae = float(np.mean(zero_errors))
    if zero_mae <= 0.0:
        raise RuntimeError(f"{direction} zero baseline MAE is not positive")
    improvement = 1.0 - model_mae / zero_mae
    recall = detected / len(first_events)
    tail_rate = tail_only / len(first_events)
    ratio_median = float(median(amplitude_ratios))
    passed = (
        recall >= float(args.minimum_direction_recall)
        and improvement >= float(args.minimum_zero_baseline_improvement)
        and ratio_median >= float(args.minimum_median_amplitude_ratio)
        and tail_rate <= float(args.maximum_tail_only_rate)
    )
    return {
        "direction": direction,
        "all_event_windows": len(relevant),
        "first5_target_event_windows": len(first_events),
        "first5_detected_windows": detected,
        "first5_direction_recall": recall,
        "tail_only_windows": tail_only,
        "tail_only_rate": tail_rate,
        "active_step_model_mae": model_mae,
        "active_step_zero_baseline_mae": zero_mae,
        "active_step_improvement_vs_zero": improvement,
        "first5_amplitude_ratio_median": ratio_median,
        "first5_amplitude_ratio_p10": percentile(amplitude_ratios, 0.10),
        "first5_amplitude_ratio_p90": percentile(amplitude_ratios, 0.90),
        "passed": passed,
    }


def evaluate_checkpoint(
    *,
    step: int,
    model_path: Path,
    dataset: Any,
    mapping: dict[tuple[int, int], int],
    anchors: dict[tuple[int, int], dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    policy, device, policy_contract = load_policy(model_path)
    records: list[dict[str, Any]] = []
    raw_first: list[float] = []
    raw_tail: list[float] = []
    try:
        for index, ((episode, frame), anchor) in enumerate(sorted(anchors.items())):
            item = dataset[mapping[(episode, frame)]]
            batch = item_batch(item, device)
            prediction = infer_chunk(policy, batch, device).astype(np.float64)
            if prediction.shape != (CHUNK_SIZE, EXPECTED_ACTION_DIM):
                raise RuntimeError("unexpected prediction shape")
            predicted = prediction[:, GRIPPER_INDEX]
            raw_first.extend(abs(value) for value in predicted[:EXECUTION_HORIZON])
            raw_tail.extend(abs(value) for value in predicted[EXECUTION_HORIZON:])
            target = anchor["target"]
            for direction, event in anchor["directions"].items():
                active_mask = event["active_mask"]
                sign = -1.0 if direction == "close" else 1.0
                active_indices = [
                    offset
                    for offset in range(EXECUTION_HORIZON)
                    if bool(active_mask[offset])
                ]
                row = {
                    "checkpoint_step": step,
                    "episode": episode,
                    "frame": frame,
                    "direction": direction,
                    "target_full_amplitude": event["target_full_amplitude"],
                    "target_first5_amplitude": event["target_first5_amplitude"],
                    "target_tail5_amplitude": event["target_tail5_amplitude"],
                    "predicted_full_amplitude": cumulative_amplitude(predicted, direction),
                    "predicted_first5_amplitude": cumulative_amplitude(
                        predicted[:EXECUTION_HORIZON], direction
                    ),
                    "predicted_tail5_amplitude": cumulative_amplitude(
                        predicted[EXECUTION_HORIZON:], direction
                    ),
                    "active_target_step_count": len(active_indices),
                    "active_model_abs_errors": [
                        abs(float(predicted[offset] - target[offset]))
                        for offset in active_indices
                    ],
                    "active_zero_abs_errors": [
                        abs(float(target[offset])) for offset in active_indices
                    ],
                    "target_net": float(np.sum(target)),
                    "predicted_net": float(np.sum(predicted)),
                    "predicted_directional_net": float(sign * np.sum(predicted)),
                }
                records.append(row)
            if (index + 1) % 25 == 0 or index + 1 == len(anchors):
                print(
                    f"step {step}: processed transition anchors "
                    f"{index + 1}/{len(anchors)}",
                    flush=True,
                )
    finally:
        del policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    directions = {
        direction: summarize_direction(
            direction=direction,
            records=records,
            args=args,
        )
        for direction in DIRECTIONS
    }
    passed = all(value["passed"] for value in directions.values())
    rank_score = sum(
        directions[direction]["first5_direction_recall"]
        + directions[direction]["active_step_improvement_vs_zero"]
        + min(directions[direction]["first5_amplitude_ratio_median"], 1.0)
        - directions[direction]["tail_only_rate"]
        for direction in DIRECTIONS
    )
    return (
        {
            "step": step,
            "model_path": str(model_path.resolve()),
            "model_sha256": sha256(model_path / "model.safetensors"),
            "policy_contract": policy_contract,
            "event_anchor_count": len(anchors),
            "predicted_gripper_abs_max_first5": max(raw_first),
            "predicted_gripper_abs_max_tail5": max(raw_tail),
            "directions": directions,
            "rank_score": rank_score,
            "passed": passed,
        },
        records,
    )


def csv_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key not in {"active_model_abs_errors", "active_zero_abs_errors"}
    } | {
        "active_model_mae": (
            float(np.mean(record["active_model_abs_errors"]))
            if record["active_model_abs_errors"]
            else ""
        ),
        "active_zero_mae": (
            float(np.mean(record["active_zero_abs_errors"]))
            if record["active_zero_abs_errors"]
            else ""
        ),
    }


def main() -> int:
    args = parse_args()
    if CHUNK_SIZE != 10 or EXECUTION_HORIZON != 5:
        raise RuntimeError("audit requires the frozen chunk=10/horizon=5 contract")
    print("===== STATIC OFFLINE CAPABILITY BOUNDARY =====")
    print("validation images + local checkpoints only")
    print("NO robot, motor bus, serial port, live camera, or command API")

    print("\n===== VERIFY PRESERVED STAGE 3C EVIDENCE =====")
    stage3c = validate_stage3c_evidence(args.stage3c_run_dir)
    print(
        "900/900 acknowledged, clips=0, trips=0; "
        f"gripper command={stage3c['gripper_command_min']:.6f}.."
        f"{stage3c['gripper_command_max']:.6f}: PASS"
    )

    print("\n===== LOAD FROZEN VALIDATION DATASET =====")
    dataset = load_dataset(
        args.validation_repo_id,
        args.validation_dataset_root,
        args.video_backend,
    )
    state, action, episode, frame = dataset_arrays(dataset)
    validate_dataset_metadata(
        dataset=dataset,
        states=state,
        actions=action,
        episodes=episode,
        frames=frame,
    )
    sequential = validate_sequential_delta_semantics(
        states=state,
        actions=action,
        episodes=episode,
    )
    mapping = frame_map(episode, frame)
    anchors = build_event_anchors(
        actions=action,
        episodes=episode,
        mapping=mapping,
        args=args,
    )
    direction_counts = {
        direction: sum(
            direction in value["directions"] for value in anchors.values()
        )
        for direction in DIRECTIONS
    }
    print(
        f"validation frames={len(state)} event anchors={len(anchors)} "
        f"directions={direction_counts}: PASS"
    )

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"output directory already exists; preserve it: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    report_path = output_dir / "gripper_transition_checkpoint_report.json"
    csv_path = output_dir / "gripper_transition_anchor_metrics.csv"

    print("\n===== FULL TEN-STEP CHECKPOINT AUDIT =====")
    checkpoint_reports: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    for step in args.checkpoint_steps:
        model_path = (
            args.train_output_root.resolve()
            / "checkpoints"
            / f"{step:06d}"
            / "pretrained_model"
        )
        print(f"Loading step {step}: {model_path}", flush=True)
        checkpoint_report, records = evaluate_checkpoint(
            step=step,
            model_path=model_path,
            dataset=dataset,
            mapping=mapping,
            anchors=anchors,
            args=args,
        )
        checkpoint_reports.append(checkpoint_report)
        all_records.extend(records)
        close = checkpoint_report["directions"]["close"]
        opened = checkpoint_report["directions"]["open"]
        print(
            f"step {step}: close recall={close['first5_direction_recall']:.3f} "
            f"zero_improvement={close['active_step_improvement_vs_zero']:.3f}; "
            f"open recall={opened['first5_direction_recall']:.3f} "
            f"zero_improvement={opened['active_step_improvement_vs_zero']:.3f}; "
            f"PASS={checkpoint_report['passed']}",
            flush=True,
        )

    passing = [item for item in checkpoint_reports if item["passed"]]
    selected = max(passing, key=lambda item: item["rank_score"]) if passing else None
    decision = (
        "EXISTING_CHECKPOINT_HAS_EXECUTABLE_GRIPPER_TRANSITIONS_"
        "RUN_FULL_OFFLINE_VALIDATION_NEXT"
        if selected is not None
        else "NO_EXISTING_CHECKPOINT_PASSES_GRIPPER_TRANSITION_GATE_"
        "RESUME_TRAINING_WITH_TRANSITION_AWARE_SELECTION"
    )
    resume_command = (
        "HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 "
        "python -m lerobot.scripts.train "
        "--config_path=outputs/train/act_red_cube_v3_delta_stage1/"
        "checkpoints/012000/pretrained_model/train_config.json "
        "--resume=true --steps=40000 --save_freq=4000"
    )

    rows = [csv_row(record) for record in all_records]
    if not rows:
        raise RuntimeError("no checkpoint rows were produced")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "schema_version": OUTPUT_SCHEMA,
        "decision": decision,
        "scope": {
            "offline_only": True,
            "hardware_accessed": False,
            "robot_object_created": False,
            "serial_port_opened": False,
            "live_camera_opened": False,
            "command_sent": False,
            "hardware_retry_authorized": False,
        },
        "contract": {
            "chunk_size": CHUNK_SIZE,
            "execution_horizon": EXECUTION_HORIZON,
            "closing_direction": "negative gripper delta",
            "target_step_threshold": args.target_step_threshold,
            "target_window_amplitude": args.target_window_amplitude,
            "prediction_fraction": args.prediction_fraction,
            "minimum_direction_recall": args.minimum_direction_recall,
            "minimum_zero_baseline_improvement": (
                args.minimum_zero_baseline_improvement
            ),
            "minimum_median_amplitude_ratio": (
                args.minimum_median_amplitude_ratio
            ),
            "maximum_tail_only_rate": args.maximum_tail_only_rate,
        },
        "stage3c_evidence": stage3c,
        "validation": {
            "root": str(args.validation_dataset_root.resolve()),
            "repo_id": args.validation_repo_id,
            "frames": len(state),
            "episodes": EXPECTED_VALIDATION_EPISODES,
            "sequential_semantics": sequential,
            "unique_event_anchors": len(anchors),
            "event_windows_by_direction": direction_counts,
        },
        "checkpoints": checkpoint_reports,
        "selected_existing_checkpoint": (
            None if selected is None else int(selected["step"])
        ),
        "suggested_bounded_resume_command_if_none_pass": (
            None if selected is not None else resume_command
        ),
        "required_next_step": (
            "Run the existing full offline safety/teacher-forced suite for the "
            "selected checkpoint; do not run hardware yet."
            if selected is not None
            else "Resume once to 40K, rerun this audit on 16K..40K, and select "
            "only a passing checkpoint. If none passes, stop V3 and build a "
            "transition-balanced V4 dataset; do not run more V3 hardware."
        ),
    }
    report_path.write_text(
        json.dumps(to_jsonable(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n===== DECISION =====")
    print(f"decision='{decision}'")
    if selected is not None:
        print(f"selected existing checkpoint: {selected['step']}")
        print("HARDWARE REMAINS BLOCKED; run full offline validation next.")
    else:
        print("No 8K-12K checkpoint passed the transition gate.")
        print(f"bounded resume command: {resume_command}")
        print("HARDWARE REMAINS BLOCKED.")
    print("NO ROBOT OBJECT WAS CREATED. NO SERIAL PORT OR CAMERA WAS OPENED.")
    print("NO COMMAND WAS SENT.")
    print("\n===== OUTPUT =====")
    print(report_path)
    print(csv_path)
    print("ACT V3 GRIPPER TRANSITION CHECKPOINT AUDIT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
