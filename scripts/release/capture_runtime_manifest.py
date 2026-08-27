#!/usr/bin/env python3
"""Capture software versions without accessing robot, serial, or cameras."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PACKAGE_NAMES = (
    "lerobot",
    "torch",
    "torchvision",
    "numpy",
    "opencv-python",
    "transformers",
    "datasets",
    "safetensors",
    "torchcodec",
    "feetech-servo-sdk",
    "cv2-enumerate-cameras",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture a read-only local runtime version manifest."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--vendor-root",
        type=Path,
        default=Path("vendor/hiwonder/lerobot"),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_value(root: Path, *args: str) -> str | None:
    if not (root / ".git").exists():
        return None
    proc = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def torch_manifest() -> dict[str, Any]:
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError:
        return {"installed": False}

    result: dict[str, Any] = {
        "installed": True,
        "version": getattr(torch, "__version__", None),
        "compiled_cuda": getattr(getattr(torch, "version", None), "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if result["cuda_available"]:
        result["device_name"] = torch.cuda.get_device_name(0)
        result["device_capability"] = list(torch.cuda.get_device_capability(0))
        result["cudnn_version"] = torch.backends.cudnn.version()
    return result


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and not args.force:
        raise FileExistsError(f"output exists; use --force to replace: {output}")

    vendor_root = args.vendor_root.resolve()
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "capability_boundary": {
            "robot_accessed": False,
            "serial_port_opened": False,
            "camera_opened": False,
            "model_loaded": False,
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": {name: package_version(name) for name in PACKAGE_NAMES},
        "torch": torch_manifest(),
        "vendor_lerobot": {
            "commit": git_value(vendor_root, "rev-parse", "HEAD"),
            "working_tree_status": git_value(vendor_root, "status", "--short"),
        },
        "privacy_review_required_before_public_commit": True,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("RUNTIME MANIFEST CAPTURE: PASS")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
