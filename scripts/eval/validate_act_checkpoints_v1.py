#!/usr/bin/env python
"""
ACT v1 checkpoint 离线验证。

目的：
- 完全不连接机械臂、不打开相机、不发送 Goal_Position。
- 加载 5k/10k/15k/20k 四个 ACT checkpoint。
- 从已验证通过的 LeRobot Dataset 读取真实 observation。
- 调用 ACTPolicy.predict_action_chunk()，检查输出 shape、有限性、动作范围、
  第一步动作与当前 state / 当前示范 action 的差值，以及 chunk 连续性。
- 所有 checkpoint 使用相同样本，便于横向比较。

注意：
- 这是离线 sanity check，不是 checkpoint 最终优劣评估。
- 不会访问串口、USB、Leader、Follower 或摄像头。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy


DEFAULT_CHECKPOINT_STEPS = [5000, 10000, 15000, 20000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-root",
        type=Path,
        default=Path("outputs/train/act_red_cube_v1"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/lerobot/so101_red_cube_pick_place_v1"),
    )
    parser.add_argument(
        "--dataset-repo-id",
        default="connorluis/so101_red_cube_pick_place_v1",
    )
    parser.add_argument(
        "--video-backend",
        default="torchcodec",
    )
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        nargs="+",
        default=DEFAULT_CHECKPOINT_STEPS,
    )
    parser.add_argument(
        "--sample-indices",
        type=int,
        nargs="*",
        default=None,
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/eval/offline_act_checkpoints_v1.json"),
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


def tensor1d(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().reshape(-1)
    return torch.as_tensor(x, dtype=torch.float32).reshape(-1)


def auto_sample_indices(dataset_len: int) -> list[int]:
    episode_indices = [0, 15, 30, 45, 59]
    frame_indices = [60, 240, 420, 600, 780]

    result = []
    for ep, frame in zip(episode_indices, frame_indices, strict=True):
        idx = ep * 900 + frame
        if 0 <= idx < dataset_len:
            result.append(idx)

    if not result:
        raise RuntimeError("无法生成离线验证 sample indices")
    return result


def prepare_policy_batch(
    item: dict[str, Any],
    policy: ACTPolicy,
) -> dict[str, torch.Tensor]:
    device = torch.device(policy.config.device)
    batch: dict[str, torch.Tensor] = {}

    for key in policy.config.input_features:
        if key not in item:
            raise KeyError(f"Dataset item 缺少 policy 所需输入：{key}")

        value = item[key]
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)

        batch[key] = value.unsqueeze(0).to(device=device)

    return batch


def stats_min_max(dataset: LeRobotDataset) -> tuple[torch.Tensor, torch.Tensor]:
    stats = dataset.meta.stats["action"]
    action_min = tensor1d(stats["min"])
    action_max = tensor1d(stats["max"])

    if action_min.numel() != 6 or action_max.numel() != 6:
        raise AssertionError(
            f"action stats 维度异常：min={action_min.shape}, max={action_max.shape}"
        )

    return action_min, action_max


def checkpoint_path(train_root: Path, step: int) -> Path:
    return train_root / "checkpoints" / f"{step:06d}" / "pretrained_model"


def validate_checkpoint_files(path: Path) -> None:
    required = [
        path / "config.json",
        path / "model.safetensors",
        path / "train_config.json",
    ]
    missing = [p for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "checkpoint 文件不完整：\n"
            + "\n".join(f"  - {p}" for p in missing)
        )


def format_vec(values: list[float] | np.ndarray, digits: int = 3) -> str:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return "[" + ", ".join(f"{x:.{digits}f}" for x in arr) + "]"


def main() -> int:
    args = parse_args()

    train_root = args.train_root.resolve()
    dataset_root = args.dataset_root.resolve()
    output_json = args.output_json.resolve()

    if not train_root.is_dir():
        raise NotADirectoryError(train_root)
    if not dataset_root.is_dir():
        raise NotADirectoryError(dataset_root)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，停止 checkpoint 离线验证。")

    print("===== LOAD DATASET =====")
    dataset = LeRobotDataset(
        repo_id=args.dataset_repo_id,
        root=dataset_root,
        video_backend=args.video_backend,
    )

    print(f"dataset root: {dataset_root}")
    print(f"frames: {len(dataset)}")
    print(f"episodes: {dataset.meta.total_episodes}")
    print(f"fps: {dataset.fps}")
    print(f"camera keys: {dataset.meta.camera_keys}")

    if dataset.meta.total_episodes != 60:
        raise AssertionError(
            f"期望 60 episodes，实际 {dataset.meta.total_episodes}"
        )
    if len(dataset) != 54000:
        raise AssertionError(
            f"期望 54000 frames，实际 {len(dataset)}"
        )

    action_min, action_max = stats_min_max(dataset)
    print(f"action min: {format_vec(action_min.numpy())}")
    print(f"action max: {format_vec(action_max.numpy())}")
    print()

    sample_indices = (
        args.sample_indices if args.sample_indices else auto_sample_indices(len(dataset))
    )

    for idx in sample_indices:
        if not 0 <= idx < len(dataset):
            raise IndexError(
                f"sample index 越界：{idx}，dataset len={len(dataset)}"
            )

    print("===== FIXED SAMPLE INDICES =====")
    for idx in sample_indices:
        ep = idx // 900
        frame = idx % 900
        print(f"global={idx:05d}  episode={ep:02d}  frame={frame:03d}")
    print()

    report: dict[str, Any] = {
        "train_root": str(train_root),
        "dataset_root": str(dataset_root),
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_frames": len(dataset),
        "dataset_episodes": dataset.meta.total_episodes,
        "fps": dataset.fps,
        "sample_indices": sample_indices,
        "action_min": action_min.tolist(),
        "action_max": action_max.tolist(),
        "checkpoints": {},
    }

    for step in args.checkpoint_steps:
        ckpt = checkpoint_path(train_root, step)
        validate_checkpoint_files(ckpt)

        print("=" * 80)
        print(f"CHECKPOINT {step:06d}")
        print(f"path: {ckpt}")

        policy = ACTPolicy.from_pretrained(
            ckpt,
            local_files_only=True,
        )
        policy.eval()
        policy.reset()

        print(f"device: {policy.config.device}")
        print(f"chunk_size: {policy.config.chunk_size}")
        print(f"n_action_steps: {policy.config.n_action_steps}")
        print(f"input_features: {list(policy.config.input_features)}")
        print(f"output_features: {list(policy.config.output_features)}")

        if policy.config.chunk_size != 50:
            raise AssertionError(
                f"{step}: chunk_size={policy.config.chunk_size}, expected=50"
            )
        if policy.config.n_action_steps != 10:
            raise AssertionError(
                f"{step}: n_action_steps={policy.config.n_action_steps}, expected=10"
            )

        expected_inputs = {
            "observation.state",
            "observation.images.front",
            "observation.images.wrist",
        }
        actual_inputs = set(policy.config.input_features)
        if actual_inputs != expected_inputs:
            raise AssertionError(
                f"{step}: input_features 异常：{sorted(actual_inputs)}"
            )

        sample_reports = []
        checkpoint_pass = True

        first_action_maes = []
        max_chunk_step_deltas = []
        max_first_state_deltas = []
        total_outside_values = 0

        for sample_idx in sample_indices:
            item = dataset[sample_idx]
            state = tensor1d(item["observation.state"])
            demo_action = tensor1d(item["action"])

            batch = prepare_policy_batch(item, policy)

            with torch.inference_mode():
                chunk = policy.predict_action_chunk(batch)

            chunk_cpu = chunk.detach().cpu().float()

            expected_shape = (1, 50, 6)
            shape_ok = tuple(chunk_cpu.shape) == expected_shape
            finite_ok = bool(torch.isfinite(chunk_cpu).all())

            if not shape_ok or not finite_ok:
                checkpoint_pass = False

            if not shape_ok:
                raise AssertionError(
                    f"{step} sample={sample_idx}: "
                    f"chunk shape={tuple(chunk_cpu.shape)}, expected={expected_shape}"
                )
            if not finite_ok:
                raise FloatingPointError(
                    f"{step} sample={sample_idx}: chunk 出现 NaN/Inf"
                )

            actions = chunk_cpu[0]
            first_action = actions[0]

            first_state_abs_delta = (first_action - state).abs()
            first_demo_abs_error = (first_action - demo_action).abs()
            chunk_step_abs_delta = (actions[1:] - actions[:-1]).abs()

            outside = (
                (actions < action_min.unsqueeze(0))
                | (actions > action_max.unsqueeze(0))
            )
            outside_count = int(outside.sum().item())

            first_action_mae = float(first_demo_abs_error.mean().item())
            max_chunk_step_delta = float(chunk_step_abs_delta.max().item())
            max_first_state_delta = float(first_state_abs_delta.max().item())

            first_action_maes.append(first_action_mae)
            max_chunk_step_deltas.append(max_chunk_step_delta)
            max_first_state_deltas.append(max_first_state_delta)
            total_outside_values += outside_count

            ep = sample_idx // 900
            frame = sample_idx % 900

            print(
                f"sample global={sample_idx:05d} ep={ep:02d} frame={frame:03d}"
            )
            print(f"  state:        {format_vec(state.numpy())}")
            print(f"  demo action:  {format_vec(demo_action.numpy())}")
            print(f"  pred first:   {format_vec(first_action.numpy())}")
            print(f"  first-vs-demo MAE: {first_action_mae:.4f}")
            print(f"  max |pred_first-state|: {max_first_state_delta:.4f}")
            print(
                f"  max chunk consecutive delta: {max_chunk_step_delta:.4f}"
            )
            print(
                f"  values outside dataset action min/max: {outside_count}/300"
            )

            sample_reports.append(
                {
                    "sample_index": sample_idx,
                    "episode_index": ep,
                    "frame_index": frame,
                    "state": state.tolist(),
                    "demo_action": demo_action.tolist(),
                    "pred_first_action": first_action.tolist(),
                    "first_vs_demo_mae": first_action_mae,
                    "first_state_abs_delta": first_state_abs_delta.tolist(),
                    "max_first_state_abs_delta": max_first_state_delta,
                    "max_chunk_consecutive_abs_delta": max_chunk_step_delta,
                    "outside_dataset_action_range_values": outside_count,
                    "chunk_shape": list(chunk_cpu.shape),
                    "finite": finite_ok,
                }
            )

        mean_first_action_mae = float(np.mean(first_action_maes))
        max_chunk_step_delta = float(np.max(max_chunk_step_deltas))
        max_first_state_delta = float(np.max(max_first_state_deltas))

        print()
        print(f"CHECKPOINT {step:06d} SUMMARY")
        print(f"  shape/finite: {'PASS' if checkpoint_pass else 'FAIL'}")
        print(
            f"  mean first-action MAE vs demonstration: {mean_first_action_mae:.4f}"
        )
        print(
            f"  max |pred_first-state| across samples: {max_first_state_delta:.4f}"
        )
        print(
            f"  max consecutive action delta in chunks: {max_chunk_step_delta:.4f}"
        )
        print(
            f"  total values outside dataset action min/max: {total_outside_values}"
        )
        print()

        report["checkpoints"][f"{step:06d}"] = {
            "path": str(ckpt),
            "shape_and_finite_pass": checkpoint_pass,
            "mean_first_action_mae_vs_demo": mean_first_action_mae,
            "max_first_state_abs_delta": max_first_state_delta,
            "max_chunk_consecutive_abs_delta": max_chunk_step_delta,
            "total_outside_dataset_action_range_values": total_outside_values,
            "samples": sample_reports,
        }

        del policy
        torch.cuda.empty_cache()

    print("=" * 80)
    print("===== OFFLINE CHECKPOINT SCOREBOARD =====")
    rows = []

    for step_key, result in report["checkpoints"].items():
        rows.append(
            (
                int(step_key),
                result["mean_first_action_mae_vs_demo"],
                result["max_first_state_abs_delta"],
                result["max_chunk_consecutive_abs_delta"],
                result["total_outside_dataset_action_range_values"],
                result["shape_and_finite_pass"],
            )
        )

    for step, mae, first_jump, chunk_jump, outside, ok in rows:
        print(
            f"{step:06d}  "
            f"shape/finite={'PASS' if ok else 'FAIL'}  "
            f"mean_first_demo_MAE={mae:.4f}  "
            f"max_first_state_delta={first_jump:.4f}  "
            f"max_chunk_step_delta={chunk_jump:.4f}  "
            f"outside_range_values={outside}"
        )

    valid_rows = [r for r in rows if r[-1]]
    if valid_rows:
        offline_rank = sorted(valid_rows, key=lambda x: x[1])
        report["offline_reference_rank_by_first_action_mae"] = [
            step for step, *_ in offline_rank
        ]
        print()
        print(
            "offline reference rank by first-action MAE "
            "(NOT real-arm ranking):"
        )
        print(
            "  " + " < ".join(f"{step:06d}" for step, *_ in offline_rank)
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(to_jsonable(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"report: {output_json}")
    print("OFFLINE ACT CHECKPOINT VALIDATION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
