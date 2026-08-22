#!/usr/bin/env python
"""
Build SO-ARM101 LeRobot dataset v2 from the validated v1 dataset + v2 plan.

SOURCE v1 (READ ONLY)
---------------------
observation.state[t] : 6D actual follower joint state
action[t]            : 6D absolute command target
front / wrist        : 480x480 video observations

TARGET v2
---------
observation.state[t] : 12D [actual_q_t, previous_command_t]
previous_command_t   : source action[t-1]
                       if source frame t == 0 -> locked Home command
action[t]            : unchanged 6D absolute command target

Episode trimming is NOT recomputed here. This builder consumes the frozen
dataset_v2_plan.json produced by plan_lerobot_dataset_v2.py.

Safety / reproducibility
------------------------
- source v1 is never modified
- output root must not already exist
- plan/source counts and semantics are verified before writing
- task text and camera geometry stay unchanged
"""

from __future__ import annotations

import argparse
import inspect
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

DEFAULT_TASK = "抓取无压纹红色方块并放入固定目标区"


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
        "--plan",
        type=Path,
        default=Path(
            "outputs/dataset/so101_red_cube_pick_place_v2_plan/"
            "dataset_v2_plan.json"
        ),
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/lerobot/so101_red_cube_pick_place_v2"),
    )
    p.add_argument(
        "--repo-id",
        default="connorluis/so101_red_cube_pick_place_v2",
    )
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--video-backend", default="torchcodec")
    p.add_argument(
        "--progress-every",
        type=int,
        default=150,
        help="Print progress every N retained frames.",
    )
    p.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate source + frozen plan but do not create v2.",
    )
    p.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help=(
            "Optional smoke-build limit. Use with a separate --output-root. "
            "Example: --max-episodes 1 --output-root "
            "data/lerobot/so101_red_cube_pick_place_v2_smoke"
        ),
    )
    args = p.parse_args()
    if args.max_episodes is not None and not 1 <= args.max_episodes <= 60:
        p.error("--max-episodes must be 1..60")
    return args


def load_dataset(
    repo_id: str,
    root: Path,
    video_backend: str | None,
) -> LeRobotDataset:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "root": root,
    }
    if video_backend:
        kwargs["video_backend"] = video_backend

    try:
        return LeRobotDataset(**kwargs)
    except TypeError:
        # Compatibility fallback for pinned/custom LeRobot variants.
        kwargs.pop("video_backend", None)
        return LeRobotDataset(**kwargs)


def image_to_hwc_uint8(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu()
        if t.ndim != 3:
            raise RuntimeError(f"Unexpected image tensor shape: {tuple(t.shape)}")
        if t.shape[0] in (1, 3, 4):
            t = t[:3].permute(1, 2, 0)
        elif t.shape[-1] in (1, 3, 4):
            t = t[..., :3]
        else:
            raise RuntimeError(f"Cannot infer image channels: {tuple(t.shape)}")
        arr = t.numpy()
    else:
        arr = np.asarray(x)
        if arr.ndim != 3:
            raise RuntimeError(f"Unexpected image array shape: {arr.shape}")
        if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr[:3], (1, 2, 0))
        else:
            arr = arr[..., :3]

    if np.issubdtype(arr.dtype, np.floating):
        if float(np.nanmax(arr)) <= 1.5:
            arr = arr * 255.0

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape != (480, 480, 3):
        raise RuntimeError(f"Expected 480x480x3 image, got {arr.shape}")
    return np.ascontiguousarray(arr)


def create_target_dataset(
    *,
    repo_id: str,
    root: Path,
    fps: int,
) -> LeRobotDataset:
    features = {
        STATE_KEY: {
            "dtype": "float32",
            "shape": (12,),
            "names": STATE_NAMES_V2,
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

    kwargs = dict(
        repo_id=repo_id,
        root=root,
        fps=fps,
        features=features,
        robot_type="so101_follower",
        use_videos=True,
    )

    return LeRobotDataset.create(**kwargs)



def describe_add_frame_api(dataset: LeRobotDataset) -> tuple[str, str]:
    """
    Return (signature_string, mode).

    Pinned LeRobot 0.3.x uses add_frame(frame, task, timestamp=None).
    Newer LeRobot versions use add_frame(frame) with 'task' inside frame.
    """
    sig = inspect.signature(dataset.add_frame)
    params = sig.parameters
    mode = "separate_task_argument" if "task" in params else "task_inside_frame"
    return str(sig), mode


def add_frame_compat(
    dataset: LeRobotDataset,
    frame: dict[str, Any],
    task: str,
) -> None:
    """
    Write one frame against both LeRobot dataset APIs without mutating vendor code.
    """
    params = inspect.signature(dataset.add_frame).parameters

    if "task" in params:
        # LeRobot 0.3.x / v2.x-style API:
        #   add_frame(frame, task, timestamp=None)
        dataset.add_frame(frame, task=task)
        return

    # Newer API:
    #   add_frame(frame)
    # where 'task' is a required special key in frame.
    frame_with_task = dict(frame)
    frame_with_task["task"] = task
    dataset.add_frame(frame_with_task)


def finalize_dataset_compat(dataset: LeRobotDataset) -> str:
    """
    Finalize/consolidate against different LeRobot dataset APIs.
    Returns a short mode string for provenance/logging.
    """
    if hasattr(dataset, "finalize") and callable(dataset.finalize):
        sig = inspect.signature(dataset.finalize)
        dataset.finalize()
        return f"finalize{sig}"

    if hasattr(dataset, "consolidate") and callable(dataset.consolidate):
        sig = inspect.signature(dataset.consolidate)
        params = sig.parameters
        if "run_compute_stats" in params:
            dataset.consolidate(run_compute_stats=True)
        else:
            dataset.consolidate()
        return f"consolidate{sig}"

    return "none_available"


def main() -> int:
    args = parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    plan_path = args.plan.resolve()

    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)

    if output_root.exists() and not args.preflight_only:
        raise FileExistsError(
            f"Refusing to overwrite existing output root:\n{output_root}\n"
            "Rename/remove it manually only after inspecting its contents."
        )

    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    print("===== BUILD CONTRACT =====")
    print(f"source: {source_root}")
    print(f"source repo_id: {args.source_repo_id}")
    print(f"plan: {plan_path}")
    print(f"target: {output_root}")
    print(f"target repo_id: {args.repo_id}")
    print(f"fps: {args.fps}")
    print("state v2: 12D [actual_q_t, previous_command]")
    print("action: unchanged 6D absolute command")
    print("images: reuse decoded v1 480x480 front/wrist frames")
    print("source dataset: READ ONLY")

    source = load_dataset(
        args.source_repo_id,
        source_root,
        args.video_backend,
    )
    hf = source.hf_dataset

    states = np.asarray(hf[STATE_KEY], dtype=np.float32)
    actions = np.asarray(hf[ACTION_KEY], dtype=np.float32)
    episodes = np.asarray(hf["episode_index"], dtype=np.int64)
    frames = np.asarray(hf["frame_index"], dtype=np.int64)

    source_info = plan["source"]
    target_info = plan["target"]
    plan_eps_all = plan["episodes"]
    plan_eps = (
        plan_eps_all
        if args.max_episodes is None
        else plan_eps_all[: args.max_episodes]
    )

    if args.max_episodes is not None:
        print()
        print("===== SMOKE BUILD LIMIT =====")
        print(f"max episodes: {args.max_episodes}")
        print(
            "IMPORTANT: use a separate --output-root for smoke builds; "
            "do not use the final v2 root."
        )

    if int(source_info["episodes"]) != 60:
        raise RuntimeError("Frozen plan does not describe 60 source episodes.")
    if int(source_info["frames"]) != 54000:
        raise RuntimeError("Frozen plan does not describe 54000 source frames.")
    if len(states) != 54000 or len(actions) != 54000:
        raise RuntimeError(
            f"Source frame count mismatch: state={len(states)} action={len(actions)}"
        )
    if states.shape != (54000, 6):
        raise RuntimeError(f"Unexpected source state shape: {states.shape}")
    if actions.shape != (54000, 6):
        raise RuntimeError(f"Unexpected source action shape: {actions.shape}")
    if target_info["observation_state_shape"] != [12]:
        raise RuntimeError("Frozen plan target state shape is not [12].")
    if target_info["action_shape"] != [6]:
        raise RuntimeError("Frozen plan target action shape is not [6].")

    home_command = np.asarray(
        plan_eps[0]["first_previous_command"]
        if plan_eps[0]["trim_start_source_frame"] == 0
        else None,
        dtype=np.float32,
    ) if False else None

    # Read exact Home fallback from target plan semantics. For source frame 0
    # episodes, first_previous_command is already frozen in plan.
    home_candidates = [
        np.asarray(row["first_previous_command"], dtype=np.float32)
        for row in plan_eps_all
        if int(row["trim_start_source_frame"]) == 0
    ]
    if not home_candidates:
        raise RuntimeError("No source-frame-0 episode found to recover Home command.")
    home_command = home_candidates[0]
    for x in home_candidates[1:]:
        if not np.allclose(x, home_command, atol=1e-6):
            raise RuntimeError("Frozen plan has inconsistent Home fallback commands.")

    print()
    print("===== PREFLIGHT PLAN / SOURCE ALIGNMENT =====")

    global_by_ep_frame = {
        (int(ep), int(fr)): int(i)
        for i, (ep, fr) in enumerate(zip(episodes, frames))
    }

    planned_total = 0
    for row in plan_eps:
        ep = int(row["episode_index"])
        trim_start = int(row["trim_start_source_frame"])
        source_len = int(row["source_length"])
        retained = int(row["retained_frames"])

        ep_idx = np.flatnonzero(episodes == ep)
        if len(ep_idx) != source_len:
            raise RuntimeError(
                f"Episode {ep}: source length mismatch "
                f"{len(ep_idx)} != {source_len}"
            )
        if retained != source_len - trim_start:
            raise RuntimeError(
                f"Episode {ep}: retained count inconsistent with trim_start."
            )
        if (ep, trim_start) not in global_by_ep_frame:
            raise RuntimeError(
                f"Episode {ep}: source trim frame {trim_start} missing."
            )

        gi = global_by_ep_frame[(ep, trim_start)]
        actual_first = states[gi]
        action_first = actions[gi]
        if trim_start > 0:
            prev_gi = global_by_ep_frame[(ep, trim_start - 1)]
            prev_first = actions[prev_gi]
        else:
            prev_first = home_command

        frozen_state = np.asarray(row["first_v2_state_12d"], dtype=np.float32)
        expected_state = np.concatenate([actual_first, prev_first]).astype(np.float32)

        if not np.allclose(frozen_state, expected_state, atol=1e-6):
            raise RuntimeError(f"Episode {ep}: frozen first 12D state mismatch.")
        if not np.allclose(
            np.asarray(row["first_action"], dtype=np.float32),
            action_first,
            atol=1e-6,
        ):
            raise RuntimeError(f"Episode {ep}: frozen first action mismatch.")

        planned_total += retained
        print(
            f"ep={ep:02d} trim_start={trim_start:3d} "
            f"retained={retained:3d}: PASS"
        )

    expected_total = sum(
        int(row["retained_frames"]) for row in plan_eps
    )
    if planned_total != expected_total:
        raise RuntimeError(
            f"Planned total mismatch: {planned_total} != {expected_total}"
        )

    print(f"planned total frames: {planned_total}")
    print("PREFLIGHT: PASS")

    if args.preflight_only:
        print("PRECHECK ONLY: no target dataset was created.")
        return 0

    print()
    print("===== CREATE V2 LEROBOT DATASET =====")
    target = create_target_dataset(
        repo_id=args.repo_id,
        root=output_root,
        fps=args.fps,
    )

    add_frame_signature, add_frame_mode = describe_add_frame_api(target)
    print(f"LeRobot add_frame signature: {add_frame_signature}")
    print(f"add_frame compatibility mode: {add_frame_mode}")

    written_total = 0

    try:
        for row in plan_eps:
            ep = int(row["episode_index"])
            trim_start = int(row["trim_start_source_frame"])
            source_len = int(row["source_length"])
            retained = int(row["retained_frames"])

            print()
            print(
                f"===== V2 EPISODE {ep:03d} "
                f"<- SOURCE FRAME {trim_start}..{source_len-1} ====="
            )

            for new_frame, source_frame in enumerate(
                range(trim_start, source_len)
            ):
                gi = global_by_ep_frame[(ep, source_frame)]

                actual_q = states[gi]
                action = actions[gi]

                if source_frame > 0:
                    prev_gi = global_by_ep_frame[(ep, source_frame - 1)]
                    previous_command = actions[prev_gi]
                else:
                    previous_command = home_command

                state_v2 = np.concatenate(
                    [actual_q, previous_command]
                ).astype(np.float32)

                item = source[gi]
                front = image_to_hwc_uint8(item[FRONT_KEY])
                wrist = image_to_hwc_uint8(item[WRIST_KEY])

                add_frame_compat(
                    target,
                    {
                        STATE_KEY: state_v2,
                        ACTION_KEY: action.astype(np.float32, copy=False),
                        FRONT_KEY: front,
                        WRIST_KEY: wrist,
                    },
                    args.task,
                )

                written_total += 1

                if (
                    (new_frame + 1) % args.progress_every == 0
                    or new_frame + 1 == retained
                ):
                    print(
                        f"  frames: {new_frame + 1}/{retained} "
                        f"(global written={written_total}/{expected_total})"
                    )

            target.save_episode()

        finalization_mode = finalize_dataset_compat(target)
        print(f"dataset finalization mode: {finalization_mode}")

    except Exception:
        # Do not attempt destructive cleanup. Keep partial target for diagnosis.
        print()
        print("BUILD ABORTED.")
        print(f"Partial target kept at: {output_root}")
        raise

    if written_total != expected_total:
        raise RuntimeError(
            f"Final written frame count mismatch: "
            f"{written_total} != {expected_total}"
        )

    provenance = {
        "schema_version": "so101_red_cube_pick_place_v2_build_v1",
        "source_root": str(source_root),
        "source_repo_id": args.source_repo_id,
        "source_frames": int(len(states)),
        "source_episodes": 60,
        "plan_path": str(plan_path),
        "target_root": str(output_root),
        "target_repo_id": args.repo_id,
        "target_frames": written_total,
        "target_episodes": len(plan_eps),
        "fps": args.fps,
        "task": args.task,
        "lerobot_add_frame_signature": add_frame_signature,
        "lerobot_add_frame_mode": add_frame_mode,
        "lerobot_finalization_mode": finalization_mode,
        "state_semantics": {
            "shape": [12],
            "names": STATE_NAMES_V2,
            "first_6": "source observation.state[t] actual follower state",
            "last_6": (
                "source action[t-1] previous absolute command; "
                "source frame 0 uses locked Home command"
            ),
        },
        "action_semantics": {
            "shape": [6],
            "names": MOTORS,
            "meaning": "unchanged source absolute command target action[t]",
        },
        "startup_trim": plan["target"]["startup_trim"],
    }

    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = meta_dir / "v2_build_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("===== BUILD COMPLETE =====")
    print(f"dataset root: {output_root}")
    print(f"episodes: {len(plan_eps)}")
    print(f"frames: {written_total}")
    print("observation.state: 12D")
    print("action: 6D absolute command")
    print(f"provenance: {provenance_path}")
    print("SOURCE V1 WAS NOT MODIFIED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
