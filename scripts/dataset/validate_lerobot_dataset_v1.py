#!/usr/bin/env python
"""
验证生成后的 LeRobot v2.1 数据集。
训练前必须执行。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


REQUIRED_FEATURES = {
    "observation.state",
    "action",
    "observation.images.front",
    "observation.images.wrist",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--repo-id",
        default="connorluis/so101_red_cube_pick_place_v1",
    )
    parser.add_argument(
        "--expected-episodes",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--expected-frames-per-episode",
        type=int,
        default=900,
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=480,
    )
    parser.add_argument(
        "--video-backend",
        default="pyav",
    )
    return parser.parse_args()


def tensor_finite(x) -> bool:
    if isinstance(x, torch.Tensor):
        return bool(torch.isfinite(x).all())

    arr = np.asarray(x)
    return bool(np.isfinite(arr).all())


def main() -> int:
    args = parse_args()
    root = args.root.resolve()

    if not root.is_dir():
        raise NotADirectoryError(root)

    print("===== LOAD DATASET =====")
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=root,
        video_backend=args.video_backend,
    )

    expected_total_frames = (
        args.expected_episodes
        * args.expected_frames_per_episode
    )

    print(f"root: {root}")
    print(f"repo_id: {dataset.repo_id}")
    print(f"fps: {dataset.fps}")
    print(
        f"episodes: "
        f"{dataset.meta.total_episodes}"
    )
    print(
        f"frames: "
        f"{dataset.meta.total_frames}"
    )
    print(f"len(dataset): {len(dataset)}")
    print(
        f"camera keys: "
        f"{dataset.meta.camera_keys}"
    )
    print()

    if dataset.fps != args.fps:
        raise AssertionError(
            f"fps={dataset.fps}, expected={args.fps}"
        )

    if (
        dataset.meta.total_episodes
        != args.expected_episodes
    ):
        raise AssertionError(
            f"episodes="
            f"{dataset.meta.total_episodes}, "
            f"expected={args.expected_episodes}"
        )

    if (
        dataset.meta.total_frames
        != expected_total_frames
    ):
        raise AssertionError(
            f"frames={dataset.meta.total_frames}, "
            f"expected={expected_total_frames}"
        )

    if len(dataset) != expected_total_frames:
        raise AssertionError(
            f"len(dataset)={len(dataset)}, "
            f"expected={expected_total_frames}"
        )

    missing = REQUIRED_FEATURES.difference(
        dataset.meta.features
    )
    if missing:
        raise AssertionError(
            f"缺少必要 feature：{sorted(missing)}"
        )

    print("===== FEATURES =====")
    for key, feature in dataset.meta.features.items():
        print(f"{key}: {feature}")
    print()

    target_hwc = (
        args.image_size,
        args.image_size,
        3,
    )

    for camera_key in (
        "observation.images.front",
        "observation.images.wrist",
    ):
        actual_shape = tuple(
            dataset.meta.features[camera_key]["shape"]
        )
        if actual_shape != target_hwc:
            raise AssertionError(
                f"{camera_key} metadata shape="
                f"{actual_shape}, expected={target_hwc}"
            )

    source_map_path = (
        root
        / "meta"
        / "source_episode_map.json"
    )

    if not source_map_path.is_file():
        raise AssertionError(
            f"缺少 provenance 文件："
            f"{source_map_path}"
        )

    source_map = json.loads(
        source_map_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        source_map["selected_episode_count"]
        != args.expected_episodes
    ):
        raise AssertionError(
            "source_episode_map "
            "selected_episode_count 不一致"
        )

    preprocessing = source_map.get(
        "image_preprocessing",
        {},
    )

    if (
        preprocessing.get("method")
        != "letterbox"
    ):
        raise AssertionError(
            "provenance 缺少 letterbox 记录"
        )

    if preprocessing.get(
        "extra_rotation"
    ) is not False:
        raise AssertionError(
            "provenance extra_rotation 应为 false"
        )

    print("===== RANDOM ACCESS SANITY =====")

    check_indices = sorted(
        set(
            [
                0,
                args.expected_frames_per_episode - 1,
                len(dataset) // 2,
                len(dataset) - 1,
            ]
        )
    )

    expected_chw = (
        3,
        args.image_size,
        args.image_size,
    )

    for idx in check_indices:
        item = dataset[idx]

        state = item["observation.state"]
        action = item["action"]
        front = item[
            "observation.images.front"
        ]
        wrist = item[
            "observation.images.wrist"
        ]

        if tuple(state.shape) != (6,):
            raise AssertionError(
                f"idx={idx}: "
                f"state shape="
                f"{tuple(state.shape)}"
            )

        if tuple(action.shape) != (6,):
            raise AssertionError(
                f"idx={idx}: "
                f"action shape="
                f"{tuple(action.shape)}"
            )

        if tuple(front.shape) != expected_chw:
            raise AssertionError(
                f"idx={idx}: front shape="
                f"{tuple(front.shape)}, "
                f"expected={expected_chw}"
            )

        if tuple(wrist.shape) != expected_chw:
            raise AssertionError(
                f"idx={idx}: wrist shape="
                f"{tuple(wrist.shape)}, "
                f"expected={expected_chw}"
            )

        if not tensor_finite(state):
            raise AssertionError(
                f"idx={idx}: "
                "state 有非有限值"
            )

        if not tensor_finite(action):
            raise AssertionError(
                f"idx={idx}: "
                "action 有非有限值"
            )

        print(
            f"idx={idx}: "
            f"state={tuple(state.shape)} "
            f"action={tuple(action.shape)} "
            f"front={tuple(front.shape)} "
            f"wrist={tuple(wrist.shape)} "
            "PASS"
        )

    print()
    print("===== STATS SANITY =====")
    for key in (
        "observation.state",
        "action",
    ):
        stats = dataset.meta.stats[key]
        print(
            f"{key}: "
            f"min={stats['min']} "
            f"max={stats['max']}"
        )

    print()
    print("DATASET VALIDATION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
