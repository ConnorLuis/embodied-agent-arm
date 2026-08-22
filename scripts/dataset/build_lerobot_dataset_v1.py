#!/usr/bin/env python
"""
将自定义 Episode Recorder v5 数据转换为 LeRobot v2.1 数据集。

重要：
- 只读取 training_manifest.csv 中 valid_for_training=TRUE 的 Episode。
- 永不修改/删除原始 Episode。
- observation.state = Follower 实际关节位置（observation_*）。
- action = Recorder 实际发送的最终绝对目标（action_*）。
- Front 原始保存尺寸为 640x480（W×H）。
- Wrist 已在 Recorder 内执行 ccw90，原始保存尺寸为 480x640（W×H）。
- ACT 多相机输入需要相同图像 shape，因此 Dataset v1 将两路图像
  等比例 letterbox 到 480x480：不旋转、不裁剪、不拉伸。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset


MOTORS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

DEFAULT_TASK = "抓取无压纹红色方块并放入固定目标区"

FRONT_SOURCE_WH = (640, 480)
WRIST_SOURCE_WH = (480, 640)
DEFAULT_TARGET_SIZE = 480


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="人工清洗后的 training_manifest_v1.csv",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="包含所有原始 Episode 的 outputs 根目录；脚本递归查找 samples.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="生成的 LeRobot 数据集根目录；默认拒绝覆盖已有目录",
    )
    parser.add_argument(
        "--repo-id",
        default="connorluis/so101_red_cube_pick_place_v1",
        help="LeRobot 数据集逻辑 repo_id；本地训练不要求上传 Hub",
    )
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--expected-frames", type=int, default=900)
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_TARGET_SIZE,
        help="两路图像统一 letterbox 到 image-size × image-size",
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="仅删除并重建 output-root；绝不触碰 raw-root",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], str]:
    errors = []
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            with path.open("r", encoding=encoding, newline="") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                raise ValueError("CSV 为空")
            return rows, encoding
        except (UnicodeDecodeError, ValueError) as exc:
            errors.append(f"{encoding}: {exc}")
    raise RuntimeError(
        f"无法读取 CSV 编码：{path}\n" + "\n".join(errors)
    )


def is_true(value: str | None) -> bool:
    return (value or "").strip().lower() in {
        "true", "1", "yes", "y", "是"
    }


def load_manifest(path: Path) -> tuple[list[str], str]:
    rows, encoding = read_csv_rows(path)

    required = {
        "episode_id",
        "technical_valid",
        "task_success",
        "demo_quality_valid",
        "valid_for_training",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"manifest 缺少字段：{sorted(missing)}")

    selected = []
    all_ids = set()

    for line_no, row in enumerate(rows, start=2):
        episode_id = (row.get("episode_id") or "").strip()
        if not episode_id:
            raise ValueError(f"manifest 第 {line_no} 行 episode_id 为空")
        if episode_id in all_ids:
            raise ValueError(f"manifest 存在重复 episode_id：{episode_id}")
        all_ids.add(episode_id)

        if is_true(row.get("valid_for_training")):
            prerequisites = {
                "technical_valid": is_true(row.get("technical_valid")),
                "task_success": is_true(row.get("task_success")),
                "demo_quality_valid": is_true(
                    row.get("demo_quality_valid")
                ),
            }
            bad = [k for k, ok in prerequisites.items() if not ok]
            if bad:
                raise ValueError(
                    f"{episode_id}: valid_for_training=TRUE，"
                    f"但前置字段为 FALSE：{bad}"
                )
            selected.append(episode_id)

    if not selected:
        raise ValueError("manifest 中没有 valid_for_training=TRUE 的 Episode")

    return selected, encoding


def index_episode_dirs(
    raw_root: Path,
    wanted_ids: set[str],
) -> dict[str, Path]:
    found: dict[str, Path] = {}

    for samples_csv in raw_root.rglob("samples.csv"):
        episode_dir = samples_csv.parent
        episode_id = episode_dir.name

        if episode_id not in wanted_ids:
            continue

        if episode_id in found:
            raise RuntimeError(
                f"同一个 episode_id 找到多个目录：\n"
                f"  {found[episode_id]}\n"
                f"  {episode_dir}"
            )
        found[episode_id] = episode_dir

    missing = sorted(wanted_ids.difference(found))
    if missing:
        preview = "\n".join(f"  - {x}" for x in missing[:20])
        raise FileNotFoundError(
            f"有 {len(missing)} 个白名单 Episode 未在 raw-root 找到：\n"
            f"{preview}"
        )

    return found


def load_episode_rows(samples_csv: Path) -> list[dict[str, str]]:
    rows, _ = read_csv_rows(samples_csv)
    return rows


def require_columns(rows: list[dict[str, str]], episode_id: str) -> None:
    required = {
        "frame_index",
        "front_image",
        "wrist_image",
    }
    required.update(f"observation_{m}" for m in MOTORS)
    required.update(f"action_{m}" for m in MOTORS)

    missing = required.difference(rows[0])
    if missing:
        raise ValueError(
            f"{episode_id}: samples.csv 缺少列：{sorted(missing)}"
        )


def parse_vector(
    row: dict[str, str],
    prefix: str,
    episode_id: str,
    frame_index: int,
) -> np.ndarray:
    values = []
    for motor in MOTORS:
        key = f"{prefix}_{motor}"
        try:
            value = float(row[key])
        except Exception as exc:
            raise ValueError(
                f"{episode_id} frame={frame_index}: "
                f"无法解析 {key}={row.get(key)!r}"
            ) from exc

        if not math.isfinite(value):
            raise ValueError(
                f"{episode_id} frame={frame_index}: "
                f"{key} 非有限值 {value}"
            )
        values.append(value)

    return np.asarray(values, dtype=np.float32)


def resolve_image_path(
    episode_dir: Path,
    raw_value: str,
    episode_id: str,
    frame_index: int,
    key: str,
) -> Path:
    raw_value = raw_value.strip()
    if not raw_value:
        raise ValueError(
            f"{episode_id} frame={frame_index}: {key} 路径为空"
        )

    p = Path(raw_value)
    if not p.is_absolute():
        p = episode_dir / p

    if not p.is_file():
        raise FileNotFoundError(
            f"{episode_id} frame={frame_index}: "
            f"{key} 不存在：{p}"
        )
    return p


def letterbox_rgb(
    path: Path,
    expected_source_wh: tuple[int, int],
    target_size: int,
    episode_id: str,
    frame_index: int,
    key: str,
) -> np.ndarray:
    """
    等比例缩放并居中填充到正方形。
    不旋转、不裁剪、不非等比拉伸。
    """
    with Image.open(path) as image:
        image = image.convert("RGB")

        if image.size != expected_source_wh:
            raise ValueError(
                f"{episode_id} frame={frame_index}: "
                f"{key} 尺寸={image.size}，"
                f"期望源尺寸={expected_source_wh}"
            )

        source_w, source_h = image.size
        scale = min(
            target_size / source_w,
            target_size / source_h,
        )

        resized_w = int(round(source_w * scale))
        resized_h = int(round(source_h * scale))

        resized = image.resize(
            (resized_w, resized_h),
            resample=Image.Resampling.LANCZOS,
        )

        canvas = Image.new(
            "RGB",
            (target_size, target_size),
            color=(0, 0, 0),
        )
        offset_x = (target_size - resized_w) // 2
        offset_y = (target_size - resized_h) // 2
        canvas.paste(
            resized,
            (offset_x, offset_y),
        )

        return np.asarray(
            canvas,
            dtype=np.uint8,
        ).copy()


def check_source_image_size(
    path: Path,
    expected_wh: tuple[int, int],
    episode_id: str,
    frame_index: int,
    key: str,
) -> None:
    with Image.open(path) as image:
        if image.size != expected_wh:
            raise ValueError(
                f"{episode_id} frame={frame_index}: "
                f"{key} 尺寸={image.size}，期望={expected_wh}"
            )


def preflight_episode(
    episode_id: str,
    episode_dir: Path,
    expected_frames: int,
) -> None:
    samples_csv = episode_dir / "samples.csv"
    rows = load_episode_rows(samples_csv)
    require_columns(rows, episode_id)

    if len(rows) != expected_frames:
        raise ValueError(
            f"{episode_id}: samples={len(rows)}，"
            f"期望={expected_frames}"
        )

    expected_indices = list(range(expected_frames))
    actual_indices = []

    for row in rows:
        try:
            actual_indices.append(int(row["frame_index"]))
        except Exception as exc:
            raise ValueError(
                f"{episode_id}: frame_index 无法解析"
            ) from exc

    if actual_indices != expected_indices:
        raise ValueError(
            f"{episode_id}: frame_index 不是严格 "
            f"0..{expected_frames - 1}"
        )

    # 全量检查数值字段与图片存在性。
    for i, row in enumerate(rows):
        parse_vector(row, "observation", episode_id, i)
        parse_vector(row, "action", episode_id, i)

        resolve_image_path(
            episode_dir,
            row["front_image"],
            episode_id,
            i,
            "front_image",
        )
        resolve_image_path(
            episode_dir,
            row["wrist_image"],
            episode_id,
            i,
            "wrist_image",
        )

    # 首/中/末检查真实保存尺寸。
    for i in (0, expected_frames // 2, expected_frames - 1):
        row = rows[i]

        front_path = resolve_image_path(
            episode_dir,
            row["front_image"],
            episode_id,
            i,
            "front_image",
        )
        wrist_path = resolve_image_path(
            episode_dir,
            row["wrist_image"],
            episode_id,
            i,
            "wrist_image",
        )

        check_source_image_size(
            front_path,
            FRONT_SOURCE_WH,
            episode_id,
            i,
            "front_image",
        )
        check_source_image_size(
            wrist_path,
            WRIST_SOURCE_WH,
            episode_id,
            i,
            "wrist_image",
        )


def build_features(target_size: int) -> dict:
    shape = (target_size, target_size, 3)

    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(MOTORS),),
            "names": MOTORS,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(MOTORS),),
            "names": MOTORS,
        },
        "observation.images.front": {
            "dtype": "video",
            "shape": shape,
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": shape,
            "names": ["height", "width", "channels"],
        },
    }


def main() -> int:
    args = parse_args()

    args.manifest = args.manifest.resolve()
    args.raw_root = args.raw_root.resolve()
    args.output_root = args.output_root.resolve()

    if args.image_size <= 0:
        raise ValueError("--image-size 必须 > 0")

    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    if not args.raw_root.is_dir():
        raise NotADirectoryError(args.raw_root)

    selected_ids, manifest_encoding = load_manifest(
        args.manifest
    )
    wanted = set(selected_ids)

    print("===== MANIFEST =====")
    print(f"path: {args.manifest}")
    print(f"encoding: {manifest_encoding}")
    print(f"valid_for_training: {len(selected_ids)}")
    print()

    print("===== IMAGE PREPROCESSING =====")
    print(f"front source W×H: {FRONT_SOURCE_WH}")
    print(f"wrist source W×H: {WRIST_SOURCE_WH}")
    print(
        f"target: {args.image_size}×{args.image_size} "
        "(letterbox, no crop, no rotation, no stretch)"
    )
    print()

    print("===== INDEX RAW EPISODES =====")
    episode_dirs = index_episode_dirs(
        args.raw_root,
        wanted,
    )
    print(
        f"找到白名单 Episode: "
        f"{len(episode_dirs)}/{len(selected_ids)}"
    )
    print()

    print("===== PREFLIGHT =====")
    for n, episode_id in enumerate(
        selected_ids,
        start=1,
    ):
        preflight_episode(
            episode_id,
            episode_dirs[episode_id],
            args.expected_frames,
        )
        print(
            f"[{n:02d}/{len(selected_ids):02d}] "
            f"{episode_id}: PASS"
        )

    print("所有白名单 Episode 预检 PASS。")
    print()

    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"输出目录已存在：{args.output_root}\n"
                "为保护已有数据，默认拒绝覆盖。"
                "确认它只是生成物后，可显式添加 --overwrite。"
            )

        print(
            f"删除旧生成目录：{args.output_root}"
        )
        shutil.rmtree(args.output_root)

    print("===== CREATE LEROBOT DATASET =====")
    features = build_features(args.image_size)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=args.output_root,
        robot_type="so101_follower",
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=8,
        batch_encoding_size=1,
    )

    source_map = []

    try:
        for lerobot_episode_index, episode_id in enumerate(
            selected_ids
        ):
            episode_dir = episode_dirs[episode_id]
            rows = load_episode_rows(
                episode_dir / "samples.csv"
            )

            print(
                f"\n===== EPISODE "
                f"{lerobot_episode_index:03d} <- "
                f"{episode_id} ====="
            )

            for i, row in enumerate(rows):
                state = parse_vector(
                    row,
                    "observation",
                    episode_id,
                    i,
                )
                action = parse_vector(
                    row,
                    "action",
                    episode_id,
                    i,
                )

                front_path = resolve_image_path(
                    episode_dir,
                    row["front_image"],
                    episode_id,
                    i,
                    "front_image",
                )
                wrist_path = resolve_image_path(
                    episode_dir,
                    row["wrist_image"],
                    episode_id,
                    i,
                    "wrist_image",
                )

                frame = {
                    "observation.state": state,
                    "action": action,
                    "observation.images.front": (
                        letterbox_rgb(
                            front_path,
                            FRONT_SOURCE_WH,
                            args.image_size,
                            episode_id,
                            i,
                            "front_image",
                        )
                    ),
                    "observation.images.wrist": (
                        letterbox_rgb(
                            wrist_path,
                            WRIST_SOURCE_WH,
                            args.image_size,
                            episode_id,
                            i,
                            "wrist_image",
                        )
                    ),
                }

                # 不传 timestamp：LeRobot 自动使用 frame_index/fps。
                dataset.add_frame(
                    frame=frame,
                    task=args.task,
                )

                if (
                    (i + 1) % 150 == 0
                    or i + 1 == len(rows)
                ):
                    print(
                        f"  frames: "
                        f"{i + 1}/{len(rows)}"
                    )

            dataset.save_episode()

            source_map.append(
                {
                    "lerobot_episode_index": (
                        lerobot_episode_index
                    ),
                    "source_episode_id": episode_id,
                    "source_episode_dir": str(
                        episode_dir
                    ),
                    "num_frames": len(rows),
                }
            )

    finally:
        dataset.stop_image_writer()

    meta_dir = args.output_root / "meta"
    meta_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    provenance = {
        "repo_id": args.repo_id,
        "fps": args.fps,
        "task": args.task,
        "source_manifest": str(args.manifest),
        "source_raw_root": str(args.raw_root),
        "selected_episode_count": len(selected_ids),
        "image_preprocessing": {
            "front_source_wh": list(FRONT_SOURCE_WH),
            "wrist_source_wh": list(WRIST_SOURCE_WH),
            "target_hwc": [
                args.image_size,
                args.image_size,
                3,
            ],
            "method": "letterbox",
            "padding_rgb": [0, 0, 0],
            "crop": False,
            "extra_rotation": False,
            "non_uniform_stretch": False,
        },
        "episode_map": source_map,
    }

    (
        meta_dir / "source_episode_map.json"
    ).write_text(
        json.dumps(
            provenance,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    shutil.copy2(
        args.manifest,
        meta_dir / "training_manifest_source.csv",
    )

    print()
    print("===== BUILD COMPLETE =====")
    print(f"dataset root: {args.output_root}")
    print(f"episodes: {len(selected_ids)}")
    print(
        f"frames: "
        f"{len(selected_ids) * args.expected_frames}"
    )
    print(
        "image shape: "
        f"({args.image_size}, "
        f"{args.image_size}, 3)"
    )
    print(
        f"source map: "
        f"{meta_dir / 'source_episode_map.json'}"
    )
    print(
        f"manifest copy: "
        f"{meta_dir / 'training_manifest_source.csv'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
