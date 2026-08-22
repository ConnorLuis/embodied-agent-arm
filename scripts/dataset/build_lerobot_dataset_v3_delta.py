#!/usr/bin/env python
"""
Build SO-ARM101 LeRobot v3 delta-action train/validation datasets.

The script uses:
  * frozen v1 for original state/action values and one-generation image decode;
  * frozen v2 plus its frozen trim plan as a semantic reference;
  * the reviewed v3 planning report for the episode-level split.

V3 schema
---------
observation.state[t] = concat(
    actual_q[t],                  # 6D
    previous_command[t],          # 6D
    previous_command_delta[t],    # 6D
)                                # total 18D

action[t] = command_delta[t]
          = command[t] - previous_command[t]  # 6D

At the first retained frame of every episode,
previous_command_delta is reset to zeros. This matches runtime reset state.

Safety / reproducibility
------------------------
* v1 and v2 are always read-only.
* output roots must not already exist for a real build.
* --preflight-only performs full parquet/plan checks but writes no dataset.
* train and validation are physically separate datasets, so train statistics
  cannot include validation episodes.
* no cameras, serial ports, or robot devices are accessed.
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

DEFAULT_TASK = "抓取无压纹红色方块并放入固定目标区"


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
        "--build-report",
        type=Path,
        default=Path(
            "outputs/dataset/so101_red_cube_pick_place_v3_delta_build/"
            "dataset_v3_delta_build_report.json"
        ),
    )
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--video-backend", default="torchcodec")
    parser.add_argument("--integrity-atol", type=float, default=1e-5)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=150,
        help="Print progress every N frames inside each episode.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate all contracts without creating either v3 dataset.",
    )
    return parser.parse_args()


def load_dataset(
    repo_id: str,
    root: Path,
    video_backend: str | None,
) -> LeRobotDataset:
    resolved_root = root.resolve()
    if not resolved_root.is_dir():
        raise FileNotFoundError(resolved_root)

    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "root": resolved_root,
    }
    if video_backend:
        kwargs["video_backend"] = video_backend
    try:
        return LeRobotDataset(**kwargs)
    except TypeError:
        kwargs.pop("video_backend", None)
        return LeRobotDataset(**kwargs)


def image_to_hwc_uint8(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.ndim != 3:
            raise RuntimeError(
                f"Unexpected image tensor shape: {tuple(tensor.shape)}"
            )
        if tensor.shape[0] in (1, 3, 4):
            tensor = tensor[:3].permute(1, 2, 0)
        elif tensor.shape[-1] in (1, 3, 4):
            tensor = tensor[..., :3]
        else:
            raise RuntimeError(
                f"Cannot infer image channels: {tuple(tensor.shape)}"
            )
        array = tensor.numpy()
    else:
        array = np.asarray(value)
        if array.ndim != 3:
            raise RuntimeError(f"Unexpected image array shape: {array.shape}")
        if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = np.transpose(array[:3], (1, 2, 0))
        else:
            array = array[..., :3]

    if np.issubdtype(array.dtype, np.floating):
        if float(np.nanmax(array)) <= 1.5:
            array = array * 255.0

    array = np.clip(array, 0, 255).astype(np.uint8)
    if array.shape != (480, 480, 3):
        raise RuntimeError(f"Expected image 480x480x3, got {array.shape}")
    return np.ascontiguousarray(array)


def create_target_dataset(
    *,
    repo_id: str,
    root: Path,
    fps: int,
) -> LeRobotDataset:
    features = {
        STATE_KEY: {
            "dtype": "float32",
            "shape": (18,),
            "names": STATE_NAMES_V3,
        },
        ACTION_KEY: {
            "dtype": "float32",
            "shape": (6,),
            "names": MOTORS,
        },
        FRONT_KEY: {
            "dtype": "video",
            "shape": (480, 480, 3),
            "names": ["height", "width", "channels"],
        },
        WRIST_KEY: {
            "dtype": "video",
            "shape": (480, 480, 3),
            "names": ["height", "width", "channels"],
        },
    }
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=fps,
        features=features,
        robot_type="so101_follower",
        use_videos=True,
    )


def load_json(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {resolved}")
    return value


def parquet_arrays(
    dataset: LeRobotDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hf_dataset = dataset.hf_dataset
    state = np.asarray(hf_dataset[STATE_KEY], dtype=np.float32)
    action = np.asarray(hf_dataset[ACTION_KEY], dtype=np.float32)
    episode = np.asarray(hf_dataset["episode_index"], dtype=np.int64)
    frame = np.asarray(hf_dataset["frame_index"], dtype=np.int64)
    if not (
        len(state) == len(action) == len(episode) == len(frame)
    ):
        raise RuntimeError(
            "Dataset parquet column lengths differ: "
            f"state={len(state)} action={len(action)} "
            f"episode={len(episode)} frame={len(frame)}"
        )
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise FloatingPointError("Dataset state/action contains NaN or Inf.")
    return state, action, episode, frame


def global_index_map(
    episode: np.ndarray,
    frame: np.ndarray,
) -> dict[tuple[int, int], int]:
    mapping = {
        (int(ep), int(fr)): int(index)
        for index, (ep, fr) in enumerate(zip(episode, frame, strict=True))
    }
    if len(mapping) != len(episode):
        raise RuntimeError("Duplicate (episode_index, frame_index) rows found.")
    return mapping


def home_command_from_v2_plan(v2_plan: dict[str, Any]) -> np.ndarray:
    candidates = [
        np.asarray(row["first_previous_command"], dtype=np.float32)
        for row in v2_plan["episodes"]
        if int(row["trim_start_source_frame"]) == 0
    ]
    if not candidates:
        raise RuntimeError("No trim_start=0 episode in v2 plan.")
    home = candidates[0]
    if home.shape != (6,):
        raise RuntimeError(f"Unexpected Home command shape: {home.shape}")
    for candidate in candidates[1:]:
        if not np.allclose(candidate, home, rtol=0.0, atol=1e-6):
            raise RuntimeError("Inconsistent Home commands in frozen v2 plan.")
    return home


def command_features(
    *,
    source_frame: int,
    new_frame: int,
    source_episode_indices: np.ndarray,
    source_actions: np.ndarray,
    home_command: np.ndarray,
    previous_previous_command: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_global = int(source_episode_indices[source_frame])
    command = source_actions[source_global]
    if source_frame > 0:
        previous_global = int(source_episode_indices[source_frame - 1])
        previous_command = source_actions[previous_global]
    else:
        previous_command = home_command

    if new_frame == 0:
        previous_command_delta = np.zeros(6, dtype=np.float32)
    else:
        if previous_previous_command is None:
            raise RuntimeError("Missing previous command history.")
        previous_command_delta = (
            previous_command - previous_previous_command
        ).astype(np.float32)

    command_delta = (command - previous_command).astype(np.float32)
    return previous_command, previous_command_delta, command_delta


def preflight(
    *,
    source_state: np.ndarray,
    source_action: np.ndarray,
    source_episode: np.ndarray,
    source_frame: np.ndarray,
    v2_state: np.ndarray,
    v2_action: np.ndarray,
    v2_episode: np.ndarray,
    v2_frame: np.ndarray,
    v2_plan: dict[str, Any],
    v3_plan: dict[str, Any],
    home_command: np.ndarray,
    atol: float,
) -> dict[str, Any]:
    source_map = global_index_map(source_episode, source_frame)
    v2_map = global_index_map(v2_episode, v2_frame)
    train_episodes = [int(x) for x in v3_plan["split"]["train_episodes"]]
    validation_episodes = [
        int(x) for x in v3_plan["split"]["validation_episodes"]
    ]
    train_set = set(train_episodes)
    validation_set = set(validation_episodes)

    if train_set & validation_set:
        raise RuntimeError("Train and validation episodes overlap.")
    if train_set | validation_set != set(range(60)):
        raise RuntimeError("Train/validation split does not cover episodes 0..59.")
    if len(train_episodes) != 48 or len(validation_episodes) != 12:
        raise RuntimeError(
            "Expected 48 train and 12 validation source episodes."
        )

    checked = 0
    train_frames = 0
    validation_frames = 0
    startup_deltas: list[np.ndarray] = []
    max_actual_error = 0.0
    max_previous_error = 0.0
    max_absolute_action_error = 0.0
    max_direct_decode_error = 0.0
    max_recursive_decode_error = 0.0
    all_command_deltas: list[np.ndarray] = []

    print("===== FULL PREFLIGHT SEMANTIC CHECK =====")
    for row in v2_plan["episodes"]:
        episode = int(row["episode_index"])
        trim_start = int(row["trim_start_source_frame"])
        source_length = int(row["source_length"])
        retained = int(row["retained_frames"])

        source_indices = np.asarray(
            [source_map[(episode, frame)] for frame in range(source_length)],
            dtype=np.int64,
        )
        previous_previous_command: np.ndarray | None = None
        running_command: np.ndarray | None = None

        for new_frame, source_local_frame in enumerate(
            range(trim_start, source_length)
        ):
            source_global = int(source_indices[source_local_frame])
            v2_global = v2_map[(episode, new_frame)]

            previous_command, previous_delta, command_delta = command_features(
                source_frame=source_local_frame,
                new_frame=new_frame,
                source_episode_indices=source_indices,
                source_actions=source_action,
                home_command=home_command,
                previous_previous_command=previous_previous_command,
            )
            actual_q = source_state[source_global]
            absolute_command = source_action[source_global]

            actual_error = float(
                np.max(np.abs(v2_state[v2_global, :6] - actual_q))
            )
            previous_error = float(
                np.max(np.abs(v2_state[v2_global, 6:] - previous_command))
            )
            absolute_action_error = float(
                np.max(np.abs(v2_action[v2_global] - absolute_command))
            )
            direct_decode_error = float(
                np.max(
                    np.abs(
                        previous_command.astype(np.float64)
                        + command_delta.astype(np.float64)
                        - absolute_command.astype(np.float64)
                    )
                )
            )

            if running_command is None:
                running_command = previous_command.astype(np.float64)
            running_command = running_command + command_delta.astype(np.float64)
            recursive_decode_error = float(
                np.max(
                    np.abs(
                        running_command - absolute_command.astype(np.float64)
                    )
                )
            )

            max_actual_error = max(max_actual_error, actual_error)
            max_previous_error = max(max_previous_error, previous_error)
            max_absolute_action_error = max(
                max_absolute_action_error, absolute_action_error
            )
            max_direct_decode_error = max(
                max_direct_decode_error, direct_decode_error
            )
            max_recursive_decode_error = max(
                max_recursive_decode_error, recursive_decode_error
            )

            if max(
                actual_error,
                previous_error,
                absolute_action_error,
                direct_decode_error,
                recursive_decode_error,
            ) > atol:
                raise RuntimeError(
                    f"ep={episode} new_frame={new_frame}: semantic error "
                    f"actual={actual_error:.9g} prev={previous_error:.9g} "
                    f"absolute_action={absolute_action_error:.9g} "
                    f"direct_decode={direct_decode_error:.9g} "
                    f"recursive_decode={recursive_decode_error:.9g}"
                )

            if new_frame == 0:
                startup_deltas.append(command_delta.copy())
                if np.max(np.abs(previous_delta)) > atol:
                    raise RuntimeError(
                        f"ep={episode}: first previous_command_delta is not zero."
                    )

            all_command_deltas.append(command_delta.copy())
            previous_previous_command = previous_command.copy()
            checked += 1
            if episode in train_set:
                train_frames += 1
            else:
                validation_frames += 1

        if new_frame + 1 != retained:
            raise RuntimeError(
                f"Episode {episode}: retained frame mismatch "
                f"{new_frame + 1} != {retained}"
            )

        split_name = "train" if episode in train_set else "validation"
        print(
            f"ep={episode:02d} split={split_name:<10} "
            f"trim={trim_start:02d} retained={retained:03d}: PASS"
        )

    expected_total = int(v2_plan["summary"]["retained_total_frames"])
    if checked != expected_total or checked != len(v2_state):
        raise RuntimeError(
            f"Checked count mismatch: checked={checked}, "
            f"plan={expected_total}, v2={len(v2_state)}"
        )

    expected_train_frames = int(v3_plan["split"]["train_frame_count"])
    expected_validation_frames = int(
        v3_plan["split"]["validation_frame_count"]
    )
    if train_frames != expected_train_frames:
        raise RuntimeError(
            f"Train frame mismatch: {train_frames} != {expected_train_frames}"
        )
    if validation_frames != expected_validation_frames:
        raise RuntimeError(
            "Validation frame mismatch: "
            f"{validation_frames} != {expected_validation_frames}"
        )

    startup_array = np.stack(startup_deltas, axis=0)
    all_delta_array = np.stack(all_command_deltas, axis=0)
    max_startup_delta = float(np.max(np.abs(startup_array)))
    if max_startup_delta > atol:
        raise RuntimeError(
            f"Startup command delta is not zero: {max_startup_delta}"
        )

    max_joint_abs_per_frame = np.max(np.abs(all_delta_array), axis=1)
    planned_summary = v3_plan["command_delta"][
        "all_max_joint_abs_per_frame"
    ]
    computed_quantiles = {
        "mean": float(np.mean(max_joint_abs_per_frame)),
        "p50": float(np.percentile(max_joint_abs_per_frame, 50.0)),
        "p95": float(np.percentile(max_joint_abs_per_frame, 95.0)),
        "p99_9": float(np.percentile(max_joint_abs_per_frame, 99.9)),
        "max": float(np.max(max_joint_abs_per_frame)),
    }
    for key, computed in computed_quantiles.items():
        planned = float(planned_summary[key])
        if not np.isclose(computed, planned, rtol=0.0, atol=atol):
            raise RuntimeError(
                f"Delta statistic {key} differs from reviewed v3 plan: "
                f"{computed} != {planned}"
            )

    result = {
        "checked_frames": checked,
        "train_frames": train_frames,
        "validation_frames": validation_frames,
        "train_episodes": train_episodes,
        "validation_episodes": validation_episodes,
        "max_actual_q_error_vs_v2": max_actual_error,
        "max_previous_command_error_vs_v2": max_previous_error,
        "max_absolute_action_error_vs_v2": max_absolute_action_error,
        "max_direct_delta_decode_error": max_direct_decode_error,
        "max_recursive_delta_decode_error": max_recursive_decode_error,
        "max_startup_command_delta": max_startup_delta,
        "delta_statistics": computed_quantiles,
    }

    print()
    print("===== PREFLIGHT SUMMARY =====")
    print(f"checked frames: {checked}")
    print(f"train: {len(train_episodes)} episodes, {train_frames} frames")
    print(
        "validation: "
        f"{len(validation_episodes)} episodes, {validation_frames} frames"
    )
    print(f"max actual_q error vs v2: {max_actual_error:.10f}")
    print(f"max previous_command error vs v2: {max_previous_error:.10f}")
    print(
        "max absolute action error vs v2: "
        f"{max_absolute_action_error:.10f}"
    )
    print(f"max direct delta decode error: {max_direct_decode_error:.10f}")
    print(
        "max recursive delta decode error: "
        f"{max_recursive_decode_error:.10f}"
    )
    print(f"max startup command delta: {max_startup_delta:.10f}")
    print("FULL PREFLIGHT: PASS")
    return result


def v3_frame_values(
    *,
    source_frame: int,
    new_frame: int,
    source_episode_indices: np.ndarray,
    source_state: np.ndarray,
    source_action: np.ndarray,
    home_command: np.ndarray,
    previous_previous_command: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_global = int(source_episode_indices[source_frame])
    previous_command, previous_delta, command_delta = command_features(
        source_frame=source_frame,
        new_frame=new_frame,
        source_episode_indices=source_episode_indices,
        source_actions=source_action,
        home_command=home_command,
        previous_previous_command=previous_previous_command,
    )
    state_v3 = np.concatenate(
        [
            source_state[source_global],
            previous_command,
            previous_delta,
        ]
    ).astype(np.float32)
    if state_v3.shape != (18,) or command_delta.shape != (6,):
        raise RuntimeError(
            f"Unexpected v3 shapes: state={state_v3.shape}, "
            f"action={command_delta.shape}"
        )
    if not np.isfinite(state_v3).all() or not np.isfinite(command_delta).all():
        raise FloatingPointError("Non-finite v3 state/action.")
    return state_v3, command_delta, previous_command


def main() -> int:
    args = parse_args()
    if args.fps != 15:
        raise ValueError("This frozen experiment expects --fps 15.")
    if args.integrity_atol <= 0.0:
        raise ValueError("--integrity-atol must be positive.")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive.")

    source_v1_root = args.source_v1_root.resolve()
    reference_v2_root = args.reference_v2_root.resolve()
    train_root = args.train_root.resolve()
    validation_root = args.validation_root.resolve()
    build_report_path = args.build_report.resolve()

    if train_root == validation_root:
        raise ValueError("Train and validation roots must differ.")
    if not args.preflight_only:
        existing = [root for root in (train_root, validation_root) if root.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing v3 output root(s):\n"
                + "\n".join(str(root) for root in existing)
                + "\nInspect and move/remove them manually before a fresh build."
            )

    v2_plan = load_json(args.v2_plan)
    v3_plan = load_json(args.v3_plan)
    if v2_plan.get("schema_version") != "so101_red_cube_pick_place_v2_plan_v1":
        raise RuntimeError("Unexpected frozen v2 plan schema.")
    if v3_plan.get("schema_version") != (
        "so101_red_cube_pick_place_v3_delta_plan_v1"
    ):
        raise RuntimeError("Unexpected reviewed v3 plan schema.")
    if v3_plan.get("status") != "PASS":
        raise RuntimeError("Reviewed v3 plan status is not PASS.")

    print("===== V3 DELTA BUILD CONTRACT =====")
    print(f"v1 source: {source_v1_root} (READ ONLY)")
    print(f"v2 reference: {reference_v2_root} (READ ONLY)")
    print(f"v2 trim plan: {args.v2_plan.resolve()}")
    print(f"v3 reviewed plan: {args.v3_plan.resolve()}")
    print(f"train target: {train_root}")
    print(f"validation target: {validation_root}")
    print("state: 18D [actual_q, previous_command, previous_command_delta]")
    print("action: 6D command_delta")
    print("images: decoded once from original v1 videos")
    print("hardware access: NONE")

    print()
    print("===== LOAD FROZEN SOURCES =====")
    source_v1 = load_dataset(
        args.source_v1_repo_id,
        source_v1_root,
        args.video_backend,
    )
    reference_v2 = load_dataset(
        args.reference_v2_repo_id,
        reference_v2_root,
        args.video_backend,
    )
    source_state, source_action, source_episode, source_frame = parquet_arrays(
        source_v1
    )
    v2_state, v2_action, v2_episode, v2_frame = parquet_arrays(reference_v2)

    if source_state.shape != (54000, 6):
        raise RuntimeError(f"Unexpected v1 state shape: {source_state.shape}")
    if source_action.shape != (54000, 6):
        raise RuntimeError(f"Unexpected v1 action shape: {source_action.shape}")
    if v2_state.shape != (53382, 12):
        raise RuntimeError(f"Unexpected v2 state shape: {v2_state.shape}")
    if v2_action.shape != (53382, 6):
        raise RuntimeError(f"Unexpected v2 action shape: {v2_action.shape}")
    if float(source_v1.fps) != 15.0 or float(reference_v2.fps) != 15.0:
        raise RuntimeError("Source/reference dataset FPS is not 15.")

    home_command = home_command_from_v2_plan(v2_plan)
    print(f"v1: frames={len(source_state)}, episodes={len(np.unique(source_episode))}")
    print(f"v2: frames={len(v2_state)}, episodes={len(np.unique(v2_episode))}")
    print(f"locked Home command: {home_command.tolist()}")

    preflight_result = preflight(
        source_state=source_state,
        source_action=source_action,
        source_episode=source_episode,
        source_frame=source_frame,
        v2_state=v2_state,
        v2_action=v2_action,
        v2_episode=v2_episode,
        v2_frame=v2_frame,
        v2_plan=v2_plan,
        v3_plan=v3_plan,
        home_command=home_command,
        atol=args.integrity_atol,
    )

    if args.preflight_only:
        print("PREFLIGHT ONLY: no v3 dataset or build report was created.")
        print("NO HARDWARE WAS ACCESSED.")
        return 0

    print()
    print("===== CREATE PHYSICALLY SEPARATE V3 DATASETS =====")
    train_target = create_target_dataset(
        repo_id=args.train_repo_id,
        root=train_root,
        fps=args.fps,
    )
    validation_target = create_target_dataset(
        repo_id=args.validation_repo_id,
        root=validation_root,
        fps=args.fps,
    )

    source_map = global_index_map(source_episode, source_frame)
    train_set = set(preflight_result["train_episodes"])
    written = {"train": 0, "validation": 0}
    target_episode_counters = {"train": 0, "validation": 0}
    episode_mapping: list[dict[str, Any]] = []

    try:
        for row in v2_plan["episodes"]:
            source_ep = int(row["episode_index"])
            trim_start = int(row["trim_start_source_frame"])
            source_length = int(row["source_length"])
            retained = int(row["retained_frames"])
            split = "train" if source_ep in train_set else "validation"
            target = train_target if split == "train" else validation_target
            target_ep = target_episode_counters[split]

            source_indices = np.asarray(
                [
                    source_map[(source_ep, frame)]
                    for frame in range(source_length)
                ],
                dtype=np.int64,
            )
            previous_previous_command: np.ndarray | None = None

            print()
            print(
                f"===== {split.upper()} TARGET EP {target_ep:03d} "
                f"<- SOURCE EP {source_ep:03d} FRAME "
                f"{trim_start}..{source_length - 1} ====="
            )

            for new_frame, source_local_frame in enumerate(
                range(trim_start, source_length)
            ):
                source_global = int(source_indices[source_local_frame])
                state_v3, action_delta, previous_command = v3_frame_values(
                    source_frame=source_local_frame,
                    new_frame=new_frame,
                    source_episode_indices=source_indices,
                    source_state=source_state,
                    source_action=source_action,
                    home_command=home_command,
                    previous_previous_command=previous_previous_command,
                )

                source_item = source_v1[source_global]
                front = image_to_hwc_uint8(source_item[FRONT_KEY])
                wrist = image_to_hwc_uint8(source_item[WRIST_KEY])

                target.add_frame(
                    {
                        STATE_KEY: state_v3,
                        ACTION_KEY: action_delta,
                        FRONT_KEY: front,
                        WRIST_KEY: wrist,
                        "task": args.task,
                    }
                )

                previous_previous_command = previous_command.copy()
                written[split] += 1
                if (
                    (new_frame + 1) % args.progress_every == 0
                    or new_frame + 1 == retained
                ):
                    print(
                        f"  frames: {new_frame + 1}/{retained}; "
                        f"split written={written[split]}"
                    )

            target.save_episode()
            episode_mapping.append(
                {
                    "split": split,
                    "source_episode_index": source_ep,
                    "target_episode_index": target_ep,
                    "trim_start_source_frame": trim_start,
                    "frames": retained,
                }
            )
            target_episode_counters[split] += 1

        if hasattr(train_target, "finalize"):
            train_target.finalize()
        if hasattr(validation_target, "finalize"):
            validation_target.finalize()

    except Exception:
        print()
        print("BUILD ABORTED. Partial output roots were kept for diagnosis:")
        print(train_root)
        print(validation_root)
        raise

    expected_train_frames = int(preflight_result["train_frames"])
    expected_validation_frames = int(preflight_result["validation_frames"])
    if written["train"] != expected_train_frames:
        raise RuntimeError(
            f"Written train frames {written['train']} != {expected_train_frames}"
        )
    if written["validation"] != expected_validation_frames:
        raise RuntimeError(
            "Written validation frames "
            f"{written['validation']} != {expected_validation_frames}"
        )
    if target_episode_counters != {"train": 48, "validation": 12}:
        raise RuntimeError(
            f"Written episode counts are wrong: {target_episode_counters}"
        )

    report = {
        "schema_version": "so101_red_cube_pick_place_v3_delta_build_v1",
        "status": "BUILD_COMPLETE_REQUIRES_VALIDATION",
        "source": {
            "v1_root": str(source_v1_root),
            "v1_repo_id": args.source_v1_repo_id,
            "v2_reference_root": str(reference_v2_root),
            "v2_reference_repo_id": args.reference_v2_repo_id,
            "v2_plan": str(args.v2_plan.resolve()),
            "v3_plan": str(args.v3_plan.resolve()),
        },
        "schema": {
            "state_shape": [18],
            "state_names": STATE_NAMES_V3,
            "state_semantics": (
                "[actual_q, previous_command, previous_command_delta]"
            ),
            "action_shape": [6],
            "action_names": MOTORS,
            "action_semantics": "command[t] - previous_command[t]",
            "episode_first_previous_command_delta": "zeros[6]",
        },
        "split_outputs": {
            "train": {
                "root": str(train_root),
                "repo_id": args.train_repo_id,
                "episodes": 48,
                "frames": written["train"],
                "stats_scope": "train episodes only",
            },
            "validation": {
                "root": str(validation_root),
                "repo_id": args.validation_repo_id,
                "episodes": 12,
                "frames": written["validation"],
                "stats_scope": "validation episodes only; not used for training",
            },
        },
        "preflight": preflight_result,
        "episode_mapping": episode_mapping,
        "images": {
            "source": "original frozen v1 videos",
            "reason": "avoid decoding and re-encoding v2 videos again",
            "shape_hwc": [480, 480, 3],
            "camera_keys": [FRONT_KEY, WRIST_KEY],
        },
        "hardware_accessed": False,
        "task": args.task,
        "fps": args.fps,
    }

    build_report_path.parent.mkdir(parents=True, exist_ok=True)
    build_report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    for root, split in (
        (train_root, "train"),
        (validation_root, "validation"),
    ):
        provenance_path = root / "meta" / "v3_delta_build_provenance.json"
        provenance_path.write_text(
            json.dumps(
                {
                    "schema_version": report["schema_version"],
                    "split": split,
                    "schema": report["schema"],
                    "output": report["split_outputs"][split],
                    "source": report["source"],
                    "episode_mapping": [
                        row for row in episode_mapping if row["split"] == split
                    ],
                    "hardware_accessed": False,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    print()
    print("===== BUILD COMPLETE; VALIDATION STILL REQUIRED =====")
    print(
        f"train: {train_root} "
        f"episodes=48 frames={written['train']}"
    )
    print(
        f"validation: {validation_root} "
        f"episodes=12 frames={written['validation']}"
    )
    print(f"build report: {build_report_path}")
    print("V1 AND V2 WERE NOT MODIFIED.")
    print("NO HARDWARE WAS ACCESSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
