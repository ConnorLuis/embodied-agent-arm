#!/usr/bin/env python
"""
Validate SO-ARM101 LeRobot dataset v2 against source v1 + frozen trim plan.

Checks
------
- 60 episodes
- exact total frame count from dataset_v2_plan.json
- observation.state is 12D
- action remains 6D absolute command
- front/wrist remain 480x480
- every v2 parquet state/action frame matches source semantics:
    state[:6]  == source observation.state[t]
    state[6:]  == source action[t-1] (or Home fallback at source frame 0)
    action     == source action[t]
- random video access sanity
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
STATE_NAMES_V2 = [
    *(f"actual.{m}" for m in MOTORS),
    *(f"prev_command.{m}" for m in MOTORS),
]

STATE_KEY = "observation.state"
ACTION_KEY = "action"
FRONT_KEY = "observation.images.front"
WRIST_KEY = "observation.images.wrist"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/lerobot/so101_red_cube_pick_place_v1"),
    )
    p.add_argument(
        "--source-repo-id",
        default="connorluis/so101_red_cube_pick_place_v1",
    )
    p.add_argument(
        "--root",
        type=Path,
        default=Path("data/lerobot/so101_red_cube_pick_place_v2"),
    )
    p.add_argument(
        "--repo-id",
        default="connorluis/so101_red_cube_pick_place_v2",
    )
    p.add_argument(
        "--plan",
        type=Path,
        default=Path(
            "outputs/dataset/so101_red_cube_pick_place_v2_plan/"
            "dataset_v2_plan.json"
        ),
    )
    p.add_argument("--video-backend", default="torchcodec")
    return p.parse_args()


def load_dataset(
    repo_id: str,
    root: Path,
    video_backend: str | None,
) -> LeRobotDataset:
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "root": root.resolve(),
    }
    if video_backend:
        kwargs["video_backend"] = video_backend
    try:
        return LeRobotDataset(**kwargs)
    except TypeError:
        kwargs.pop("video_backend", None)
        return LeRobotDataset(**kwargs)


def main() -> int:
    args = parse_args()

    plan = json.loads(args.plan.resolve().read_text(encoding="utf-8"))

    src = load_dataset(
        args.source_repo_id,
        args.source_root,
        args.video_backend,
    )
    dst = load_dataset(
        args.repo_id,
        args.root,
        args.video_backend,
    )

    src_hf = src.hf_dataset
    dst_hf = dst.hf_dataset

    src_state = np.asarray(src_hf[STATE_KEY], dtype=np.float32)
    src_action = np.asarray(src_hf[ACTION_KEY], dtype=np.float32)
    src_ep = np.asarray(src_hf["episode_index"], dtype=np.int64)
    src_fr = np.asarray(src_hf["frame_index"], dtype=np.int64)

    dst_state = np.asarray(dst_hf[STATE_KEY], dtype=np.float32)
    dst_action = np.asarray(dst_hf[ACTION_KEY], dtype=np.float32)
    dst_ep = np.asarray(dst_hf["episode_index"], dtype=np.int64)
    dst_fr = np.asarray(dst_hf["frame_index"], dtype=np.int64)

    expected_total = int(plan["summary"]["retained_total_frames"])

    print("===== BASIC =====")
    print(f"root: {args.root.resolve()}")
    print(f"repo_id: {args.repo_id}")
    print(f"episodes: {len(np.unique(dst_ep))}")
    print(f"frames: {len(dst_state)}")
    print(f"state shape: {dst_state.shape}")
    print(f"action shape: {dst_action.shape}")

    if len(np.unique(dst_ep)) != 60:
        raise RuntimeError("Expected 60 v2 episodes.")
    if len(dst_state) != expected_total:
        raise RuntimeError(
            f"Frame count mismatch: {len(dst_state)} != {expected_total}"
        )
    if dst_state.shape != (expected_total, 12):
        raise RuntimeError(f"Unexpected v2 state shape: {dst_state.shape}")
    if dst_action.shape != (expected_total, 6):
        raise RuntimeError(f"Unexpected v2 action shape: {dst_action.shape}")

    # Recover Home fallback from frozen plan.
    home_rows = [
        row for row in plan["episodes"]
        if int(row["trim_start_source_frame"]) == 0
    ]
    if not home_rows:
        raise RuntimeError("No trim_start=0 row in frozen plan.")
    home_command = np.asarray(
        home_rows[0]["first_previous_command"],
        dtype=np.float32,
    )

    src_map = {
        (int(ep), int(fr)): int(i)
        for i, (ep, fr) in enumerate(zip(src_ep, src_fr))
    }
    dst_map = {
        (int(ep), int(fr)): int(i)
        for i, (ep, fr) in enumerate(zip(dst_ep, dst_fr))
    }

    print()
    print("===== FULL PARQUET SEMANTIC CHECK =====")

    checked = 0
    max_actual_err = 0.0
    max_prev_err = 0.0
    max_action_err = 0.0

    for row in plan["episodes"]:
        ep = int(row["episode_index"])
        trim_start = int(row["trim_start_source_frame"])
        retained = int(row["retained_frames"])

        for new_fr in range(retained):
            src_fr_idx = trim_start + new_fr
            si = src_map[(ep, src_fr_idx)]
            di = dst_map[(ep, new_fr)]

            expected_actual = src_state[si]
            expected_action = src_action[si]

            if src_fr_idx > 0:
                spi = src_map[(ep, src_fr_idx - 1)]
                expected_prev = src_action[spi]
            else:
                expected_prev = home_command

            actual_err = float(
                np.max(np.abs(dst_state[di, :6] - expected_actual))
            )
            prev_err = float(
                np.max(np.abs(dst_state[di, 6:] - expected_prev))
            )
            action_err = float(
                np.max(np.abs(dst_action[di] - expected_action))
            )

            max_actual_err = max(max_actual_err, actual_err)
            max_prev_err = max(max_prev_err, prev_err)
            max_action_err = max(max_action_err, action_err)

            if actual_err > 1e-6:
                raise RuntimeError(
                    f"ep={ep} new_fr={new_fr}: actual_q mismatch {actual_err}"
                )
            if prev_err > 1e-6:
                raise RuntimeError(
                    f"ep={ep} new_fr={new_fr}: previous_command mismatch "
                    f"{prev_err}"
                )
            if action_err > 1e-6:
                raise RuntimeError(
                    f"ep={ep} new_fr={new_fr}: action mismatch {action_err}"
                )

            checked += 1

    print(f"checked frames: {checked}/{expected_total}")
    print(f"max actual_q error:       {max_actual_err:.8f}")
    print(f"max previous_command err: {max_prev_err:.8f}")
    print(f"max action error:         {max_action_err:.8f}")
    print("PARQUET SEMANTICS: PASS")

    print()
    print("===== CAMERA / RANDOM ACCESS SANITY =====")

    sample_indices = sorted(
        {
            0,
            min(100, expected_total - 1),
            expected_total // 2,
            max(0, expected_total - 2),
            expected_total - 1,
        }
    )

    for idx in sample_indices:
        item = dst[idx]
        state = item[STATE_KEY]
        action = item[ACTION_KEY]
        front = item[FRONT_KEY]
        wrist = item[WRIST_KEY]

        if tuple(state.shape) != (12,):
            raise RuntimeError(f"idx={idx}: state shape {tuple(state.shape)}")
        if tuple(action.shape) != (6,):
            raise RuntimeError(f"idx={idx}: action shape {tuple(action.shape)}")
        if tuple(front.shape) != (3, 480, 480):
            raise RuntimeError(f"idx={idx}: front shape {tuple(front.shape)}")
        if tuple(wrist.shape) != (3, 480, 480):
            raise RuntimeError(f"idx={idx}: wrist shape {tuple(wrist.shape)}")

        print(
            f"idx={idx}: state={tuple(state.shape)} "
            f"action={tuple(action.shape)} "
            f"front={tuple(front.shape)} "
            f"wrist={tuple(wrist.shape)} PASS"
        )

    print()
    print("===== FEATURE METADATA =====")
    try:
        features = dst.meta.features
    except AttributeError:
        features = None

    if features is not None:
        print(f"{STATE_KEY}: {features.get(STATE_KEY)}")
        print(f"{ACTION_KEY}: {features.get(ACTION_KEY)}")
        print(f"{FRONT_KEY}: {features.get(FRONT_KEY)}")
        print(f"{WRIST_KEY}: {features.get(WRIST_KEY)}")

        state_feature = features[STATE_KEY]
        if tuple(state_feature["shape"]) != (12,):
            raise RuntimeError("Metadata state shape is not 12D.")
        if list(state_feature["names"]) != STATE_NAMES_V2:
            raise RuntimeError("Metadata state names do not match v2 contract.")

    print()
    print("DATASET V2 VALIDATION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
