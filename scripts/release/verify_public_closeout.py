#!/usr/bin/env python3
"""Stdlib-only verification for the public Stage 1 closeout package."""

from __future__ import annotations

import csv
import hashlib
import json
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

REQUIRED_FILES = (
    "README.md",
    "environment.yml",
    "requirements-camera-bridge.txt",
    "RELEASE_NOTES_v0.1.0-stage1.md",
    "docs/PROJECT_CLOSEOUT.md",
    "docs/DEPENDENCIES.md",
    "evidence/release/README.md",
    "evidence/release/stage1_summary.json",
    "evidence/release/v3_checkpoint_metrics.csv",
    "evidence/release/v4_checkpoint_metrics.csv",
    "evidence/release/front_task_outcome.png",
    "evidence/release/wrist_task_outcome.png",
    "evidence/release/fixtures/follower_white_calibration_semantics.json",
    "scripts/release/capture_runtime_manifest.py",
    ".github/workflows/public-closeout-ci.yml",
)

EXPECTED_IMAGE_HASHES = {
    "evidence/release/front_task_outcome.png": (
        "95313646aff5590855072c8048170554ba5eb5256658d94f8741981e9b179f11"
    ),
    "evidence/release/wrist_task_outcome.png": (
        "6ef68da1e4b7682073805fed5043b4f30ae5626bbb8f74bb759af6885c41eafc"
    ),
}


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def read_text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def load_csv(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def png_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    require(data[:8] == b"\x89PNG\r\n\x1a\n", f"not a PNG: {path}")
    require(data[12:16] == b"IHDR", f"PNG is missing IHDR: {path}")
    return struct.unpack(">II", data[16:24])


def verify_wording() -> None:
    readme = read_text("README.md")
    closeout = read_text("docs/PROJECT_CLOSEOUT.md")
    public_evidence = read_text("evidence/release/README.md")

    require("Stage 1: CLOSED / FROZEN" in readme, "README status is missing")
    require("工程链路完成，策略任务未达标" in readme, "README conclusion missing")
    require("task_success=false" in public_evidence, "execution/task distinction missing")

    forbidden = {
        "README.md": ("Work in progress", "LLM Agent", "MCP"),
        "docs/PROJECT_CLOSEOUT.md": ("双目", "预注册"),
    }
    for relative, phrases in forbidden.items():
        text = read_text(relative)
        for phrase in phrases:
            require(phrase not in text, f"stale wording {phrase!r} in {relative}")

    public_text = "\n".join((readme, closeout, public_evidence))
    for private_fragment in ("/home/", "/mnt/", "C:\\Users", "\\\\wsl.localhost"):
        require(
            private_fragment not in public_text,
            f"private/local path leaked into public narrative: {private_fragment}",
        )


def verify_summary() -> None:
    summary = json.loads(read_text("evidence/release/stage1_summary.json"))
    require(summary["schema_version"] == 1, "unexpected summary schema")
    require(summary["status"] == "CLOSED_FROZEN", "stage is not frozen")
    require(summary["dataset"]["total_frames"] == 53382, "frame total mismatch")

    episode = summary["controlled_episode"]
    require(episode["execution_pass"] is True, "execution must be recorded as pass")
    require(episode["replans_completed"] == episode["replans_expected"] == 180, "replan mismatch")
    require(
        episode["policy_commands_acknowledged"]
        == episode["policy_commands_expected"]
        == 900,
        "policy command mismatch",
    )
    require(episode["rate_clips"] == 0, "rate clips must be zero")
    require(episode["new_boundary_crossings"] == 0, "boundary crossings must be zero")
    require(episode["tracking_trips"] == 0, "tracking trips must be zero")
    require(episode["torque_disabled_at_exit"] is True, "torque cleanup missing")
    require(episode["serial_closed"] is True, "serial cleanup missing")
    require(episode["camera_workers_closed"] is True, "camera cleanup missing")

    outcome = summary["task_outcome"]
    require(outcome["manual_review_complete"] is True, "task review incomplete")
    require(outcome["task_success"] is False, "must not claim task success")
    require(outcome["cube_grasped"] is False, "must not claim cube grasp")
    require(
        outcome["gripper_command_max"] - outcome["gripper_command_min"] < 0.2,
        "gripper collapse range changed",
    )

    diagnosis = summary["checkpoint_diagnosis"]
    require(
        diagnosis["v4_best_close_recall"] == {"step": 16000, "value": 0.5},
        "best close metric mismatch",
    )
    require(
        diagnosis["v4_best_open_recall"] == {"step": 20000, "value": 0.324},
        "best open metric mismatch",
    )
    require(
        diagnosis["same_checkpoint_achieved_both_best_values"] is False,
        "cross-checkpoint metric caveat missing",
    )
    require(
        diagnosis["any_checkpoint_passed_joint_direction_and_amplitude_gate"] is False,
        "a passing checkpoint would invalidate the frozen conclusion",
    )
    require(summary["freeze"]["autonomous_deployment_authorized"] is False, "deployment must stay blocked")
    require(summary["freeze"]["home_to_park_live_validated"] is False, "Park must remain unclaimed")


def verify_checkpoint_csvs() -> None:
    v3 = load_csv("evidence/release/v3_checkpoint_metrics.csv")
    v4 = load_csv("evidence/release/v4_checkpoint_metrics.csv")
    require([int(row["step"]) for row in v3] == [16000, 20000, 24000, 28000, 32000, 36000, 40000], "V3 steps mismatch")
    require(len(v4) == 5, "V4 row count mismatch")
    require(all(float(row["close_recall"]) == 0.0 for row in v3), "V3 close recall changed")
    require(all(float(row["open_recall"]) == 0.0 for row in v3), "V3 open recall changed")
    require(all(row["passes_joint_gate"] == "false" for row in v3 + v4), "a checkpoint unexpectedly passes")

    best_close = max(v4, key=lambda row: float(row["close_recall"]))
    best_open = max(v4, key=lambda row: float(row["open_recall"]))
    require((int(best_close["step"]), float(best_close["close_recall"])) == (16000, 0.5), "V4 best close mismatch")
    require((int(best_open["step"]), float(best_open["open_recall"])) == (20000, 0.324), "V4 best open mismatch")


def verify_images() -> None:
    for relative, expected_hash in EXPECTED_IMAGE_HASHES.items():
        path = ROOT / relative
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual_hash == expected_hash, f"image SHA-256 mismatch: {relative}")
        require(png_dimensions(path) == (640, 480), f"image dimensions mismatch: {relative}")


def verify_dependency_records() -> None:
    environment = read_text("environment.yml")
    camera = read_text("requirements-camera-bridge.txt")
    dependencies = read_text("docs/DEPENDENCIES.md")
    for fragment in (
        "python=3.10.18",
        "ffmpeg=7.1.1",
        "numpy==2.2.6",
        "opencv-python==5.0.0.93",
        "torch>=2.2.1,<2.8.0",
        "torchvision>=0.21.0,<0.23.0",
        "torchcodec>=0.2.1,<0.6.0",
        "datasets>=2.19.0,<=3.6.0",
        "transformers>=4.50.3,<4.52.0",
        "safetensors>=0.4.3",
        "feetech-servo-sdk>=1.0.0",
    ):
        require(fragment in environment, f"WSL dependency missing: {fragment}")
    for fragment in ("numpy==2.5.2", "opencv-python==5.0.0.93", "cv2-enumerate-cameras==1.3.3"):
        require(fragment in camera, f"camera dependency missing: {fragment}")
    require("882c80d446a63a44868c67ae535467af32ce0e80" in dependencies, "vendor commit missing")
    require("没有猜测的版本" in dependencies, "unknown-version disclosure missing")


def main() -> int:
    for relative in REQUIRED_FILES:
        require((ROOT / relative).is_file(), f"required file missing: {relative}")

    verify_wording()
    verify_summary()
    verify_checkpoint_csvs()
    verify_images()
    verify_dependency_records()

    print("PUBLIC CLOSEOUT VERIFY: PASS")
    print(f"required files: {len(REQUIRED_FILES)}/{len(REQUIRED_FILES)}")
    print("V3 checkpoints: 7; V4 checkpoints: 5; passing: 0")
    print("controlled episode: 180/180 replans, 900/900 commands; task success: false")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as error:
        print(f"PUBLIC CLOSEOUT VERIFY: FAIL: {error}")
        raise SystemExit(2) from error
