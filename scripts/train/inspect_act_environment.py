#!/usr/bin/env python
"""
ACT 训练环境预检：版本、Git commit、CUDA、显存、ACT 默认配置。
不连接机械臂，不访问相机。
"""

from __future__ import annotations

import argparse
import importlib.metadata
import subprocess
from pathlib import Path

import torch

from lerobot.policies.act.configuration_act import ACTConfig


EXPECTED_LEROBOT_VERSION = "0.3.4"
EXPECTED_COMMIT = "882c80d446a63a44868c67ae535467af32ce0e80"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vendor-root",
        type=Path,
        default=Path("vendor/hiwonder/lerobot"),
    )
    return parser.parse_args()


def git_output(root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        text=True,
        capture_output=True,
    )
    return proc.stdout.strip()


def main() -> int:
    args = parse_args()
    vendor_root = args.vendor_root.resolve()

    print("===== PYTHON / LEROBOT =====")
    version = importlib.metadata.version("lerobot")
    print(f"lerobot version: {version}")
    if version != EXPECTED_LEROBOT_VERSION:
        raise RuntimeError(
            f"LeRobot 版本不一致：{version} != "
            f"{EXPECTED_LEROBOT_VERSION}"
        )

    if not vendor_root.is_dir():
        raise NotADirectoryError(vendor_root)

    commit = git_output(vendor_root, "rev-parse", "HEAD")
    print(f"vendor commit: {commit}")
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(
            f"vendor commit 不一致：{commit} != {EXPECTED_COMMIT}"
        )

    status = git_output(vendor_root, "status", "--short")
    print("vendor working tree:")
    if status:
        print(status)
        print(
            "说明：存在项目已知的本地修改时可以继续；"
            "不要执行 git reset/restore/checkout/clean。"
        )
    else:
        print("(clean)")

    print()
    print("===== CUDA =====")
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，停止 ACT 训练。")

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    print(f"gpu: {props.name}")
    print(
        f"total VRAM: "
        f"{props.total_memory / 1024**3:.2f} GiB"
    )

    x = torch.zeros((1024, 1024), device=device)
    del x
    torch.cuda.synchronize()
    print("CUDA allocation smoke: PASS")

    print()
    print("===== ACT CONFIG =====")
    cfg = ACTConfig(
        device="cuda",
        push_to_hub=False,
    )
    print(f"default chunk_size: {cfg.chunk_size}")
    print(f"default n_action_steps: {cfg.n_action_steps}")
    print(f"vision_backbone: {cfg.vision_backbone}")
    print(
        f"pretrained_backbone_weights: "
        f"{cfg.pretrained_backbone_weights}"
    )
    print(f"optimizer_lr: {cfg.optimizer_lr}")

    print()
    print("ACT ENVIRONMENT CHECK: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
