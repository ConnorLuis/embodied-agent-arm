#!/usr/bin/env python
"""
Plan SO-ARM101 LeRobot dataset v2 without mutating the source dataset.

v2 design
---------
Source v1:
    observation.state[t] = 6D actual follower joint state
    action[t]            = 6D absolute command target

Target v2:
    observation.state[t] = concat(
        actual_q[t],          # 6D
        previous_command[t],  # 6D = action[t-1] in source trajectory
    )                        # total 12D
    action[t] = unchanged 6D absolute command target

Startup trimming:
    action_onset = first frame where max(|action - locked_home_command|)
                   > threshold for N consecutive frames
    trim_start   = max(0, action_onset - pre_roll)

The source dataset is READ-ONLY. This script only writes a plan JSON/CSV.

For the first retained frame:
- if source frame > 0: previous_command = source action[frame-1]
- if source frame == 0: previous_command = locked Home command

This preserves the true command-history state while removing variable idle prefix.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


MOTORS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

STATE_KEY = "observation.state"
ACTION_KEY = "action"


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
        "--run-dir",
        type=Path,
        required=True,
        help="Validated run containing report.json and locked Home command.",
    )
    p.add_argument("--action-threshold", type=float, default=0.20)
    p.add_argument("--consecutive", type=int, default=2)
    p.add_argument("--pre-roll", type=int, default=2)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dataset/so101_red_cube_pick_place_v2_plan"),
    )
    args = p.parse_args()

    if not 0.01 <= args.action_threshold <= 2.0:
        p.error("--action-threshold must be 0.01..2.0")
    if not 1 <= args.consecutive <= 5:
        p.error("--consecutive must be 1..5")
    if not 0 <= args.pre_roll <= 20:
        p.error("--pre-roll must be 0..20")
    return args


def first_consecutive_true(mask: np.ndarray, consecutive: int) -> int:
    mask = np.asarray(mask, dtype=bool)
    for i in range(0, len(mask) - consecutive + 1):
        if bool(np.all(mask[i : i + consecutive])):
            return i
    return len(mask)


def load_dataset(repo_id: str, root: Path) -> LeRobotDataset:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    try:
        return LeRobotDataset(repo_id=repo_id, root=root)
    except TypeError:
        return LeRobotDataset(repo_id, root=root)


def summarize(values: list[int]) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {
        "min": int(np.min(a)),
        "q25": float(np.percentile(a, 25)),
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "q75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
        "max": int(np.max(a)),
        "total": int(np.sum(a)),
    }


def main() -> int:
    args = parse_args()

    report_path = args.run_dir.resolve() / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    home_cmd_dict = report["home_result"]["command_target_positions"]
    home_command = np.asarray(
        [home_cmd_dict[m] for m in MOTORS],
        dtype=np.float32,
    )

    print("===== V2 DESIGN CONTRACT =====")
    print("source observation.state: 6D actual follower state")
    print("source action:            6D absolute command")
    print("target observation.state: 12D [actual_q_t, previous_command]")
    print("target action:            unchanged 6D absolute command")
    print(
        f"startup trim: threshold={args.action_threshold:.2f}, "
        f"consecutive={args.consecutive}, pre_roll={args.pre_roll}"
    )
    print(f"locked Home command: {home_command.tolist()}")

    print()
    print("===== LOAD SOURCE V1 =====")
    ds = load_dataset(args.source_repo_id, args.source_root)
    hf = ds.hf_dataset

    states = np.asarray(hf[STATE_KEY], dtype=np.float32)
    actions = np.asarray(hf[ACTION_KEY], dtype=np.float32)
    episodes = np.asarray(hf["episode_index"], dtype=np.int64)
    frames = np.asarray(hf["frame_index"], dtype=np.int64)

    unique_eps = sorted(int(x) for x in np.unique(episodes))
    if len(unique_eps) != 60:
        raise RuntimeError(f"Expected 60 episodes, got {len(unique_eps)}")
    if states.shape[1:] != (6,):
        raise RuntimeError(f"Expected 6D state, got {states.shape}")
    if actions.shape[1:] != (6,):
        raise RuntimeError(f"Expected 6D action, got {actions.shape}")

    print(f"source frames:   {len(states)}")
    print(f"source episodes: {len(unique_eps)}")
    print(f"source state:    {states.shape}")
    print(f"source action:   {actions.shape}")

    plan_rows = []
    removed_counts = []
    retained_counts = []
    onsets = []

    print()
    print("===== EPISODE TRIM PLAN =====")
    print(
        "episode  source_len  action_onset  trim_start  removed  retained  "
        "first_prev_source"
    )

    for ep in unique_eps:
        idx = np.flatnonzero(episodes == ep)
        idx = idx[np.argsort(frames[idx])]

        ep_frames = frames[idx]
        ep_actions = actions[idx]

        expected = np.arange(len(idx), dtype=np.int64)
        if not np.array_equal(ep_frames, expected):
            raise RuntimeError(
                f"Episode {ep}: frame_index not contiguous 0..N-1"
            )

        action_dev = np.max(
            np.abs(ep_actions - home_command[None, :]),
            axis=1,
        )
        onset = first_consecutive_true(
            action_dev > args.action_threshold,
            args.consecutive,
        )
        if onset >= len(idx):
            raise RuntimeError(
                f"Episode {ep}: no action onset found with current rule."
            )

        trim_start = max(0, onset - args.pre_roll)
        removed = trim_start
        retained = len(idx) - trim_start
        first_prev_source = trim_start - 1 if trim_start > 0 else None

        # Construct first retained v2 state in-memory as a semantic sanity check.
        actual_first = ep_actions[0]  # placeholder overwritten below
        actual_first = states[idx[trim_start]]
        if trim_start > 0:
            prev_cmd_first = actions[idx[trim_start - 1]]
        else:
            prev_cmd_first = home_command
        state_v2_first = np.concatenate(
            [actual_first, prev_cmd_first]
        ).astype(np.float32)

        if state_v2_first.shape != (12,):
            raise RuntimeError(
                f"Episode {ep}: v2 first state shape {state_v2_first.shape}"
            )
        if not np.isfinite(state_v2_first).all():
            raise FloatingPointError(
                f"Episode {ep}: non-finite v2 first state."
            )

        plan_rows.append(
            {
                "episode_index": ep,
                "source_length": int(len(idx)),
                "action_onset_frame": int(onset),
                "trim_start_source_frame": int(trim_start),
                "removed_frames": int(removed),
                "retained_frames": int(retained),
                "first_previous_command_source_frame": (
                    None if first_prev_source is None else int(first_prev_source)
                ),
                "first_actual_q": actual_first.tolist(),
                "first_previous_command": prev_cmd_first.tolist(),
                "first_action": actions[idx[trim_start]].tolist(),
                "first_v2_state_12d": state_v2_first.tolist(),
            }
        )

        removed_counts.append(removed)
        retained_counts.append(retained)
        onsets.append(onset)

        prev_label = (
            "HOME"
            if first_prev_source is None
            else str(first_prev_source)
        )
        print(
            f"ep={ep:02d}      {len(idx):4d}        {onset:4d}       "
            f"{trim_start:4d}      {removed:4d}     {retained:4d}      "
            f"{prev_label}"
        )

    total_removed = int(sum(removed_counts))
    total_retained = int(sum(retained_counts))

    print()
    print("===== PLAN SUMMARY =====")
    print(
        f"action onset frames: median={np.median(onsets):.1f}, "
        f"mean={np.mean(onsets):.1f}, "
        f"min={min(onsets)}, max={max(onsets)}"
    )
    print(
        f"removed per episode: median={np.median(removed_counts):.1f}, "
        f"mean={np.mean(removed_counts):.1f}, "
        f"min={min(removed_counts)}, max={max(removed_counts)}"
    )
    print(f"source total frames:  {len(states)}")
    print(f"removed total frames: {total_removed}")
    print(f"v2 planned frames:    {total_retained}")
    print(
        f"retained ratio:        "
        f"{100.0 * total_retained / len(states):.2f}%"
    )

    # Check previous-command semantics on every retained source frame without
    # allocating/decoding any images.
    semantic_checks = 0
    for row in plan_rows:
        ep = row["episode_index"]
        trim_start = row["trim_start_source_frame"]
        idx = np.flatnonzero(episodes == ep)
        idx = idx[np.argsort(frames[idx])]

        for local_src_frame in range(trim_start, len(idx)):
            if local_src_frame == 0:
                prev_cmd = home_command
            else:
                prev_cmd = actions[idx[local_src_frame - 1]]

            v2_state = np.concatenate(
                [states[idx[local_src_frame]], prev_cmd]
            )
            if v2_state.shape != (12,) or not np.isfinite(v2_state).all():
                raise RuntimeError(
                    f"Episode {ep}, source frame {local_src_frame}: "
                    "invalid 12D v2 state"
                )
            semantic_checks += 1

    if semantic_checks != total_retained:
        raise RuntimeError(
            f"Semantic-check count mismatch: {semantic_checks} vs {total_retained}"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "dataset_v2_plan.json"
    csv_path = output_dir / "dataset_v2_trim_plan.csv"

    result = {
        "schema_version": "so101_red_cube_pick_place_v2_plan_v1",
        "source": {
            "root": str(args.source_root.resolve()),
            "repo_id": args.source_repo_id,
            "episodes": len(unique_eps),
            "frames": int(len(states)),
        },
        "target": {
            "suggested_root": "data/lerobot/so101_red_cube_pick_place_v2",
            "suggested_repo_id": "connorluis/so101_red_cube_pick_place_v2",
            "observation_state_shape": [12],
            "observation_state_names": [
                *(f"actual.{m}" for m in MOTORS),
                *(f"prev_command.{m}" for m in MOTORS),
            ],
            "action_shape": [6],
            "action_names": MOTORS,
            "previous_command_semantics": (
                "source action[t-1]; if source frame t=0, locked Home command"
            ),
            "startup_trim": {
                "action_threshold": args.action_threshold,
                "consecutive_frames": args.consecutive,
                "pre_roll_frames": args.pre_roll,
                "formula": "trim_start=max(0, action_onset-pre_roll)",
            },
            "planned_episodes": len(unique_eps),
            "planned_frames": total_retained,
        },
        "summary": {
            "action_onset_frames": summarize(onsets),
            "removed_frames_per_episode": summarize(removed_counts),
            "retained_frames_per_episode": summarize(retained_counts),
            "removed_total_frames": total_removed,
            "retained_total_frames": total_retained,
            "retained_ratio": total_retained / len(states),
            "semantic_checks": semantic_checks,
        },
        "episodes": plan_rows,
    }

    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode_index",
                "source_length",
                "action_onset_frame",
                "trim_start_source_frame",
                "removed_frames",
                "retained_frames",
                "first_previous_command_source_frame",
            ],
        )
        writer.writeheader()
        for row in plan_rows:
            writer.writerow(
                {k: row[k] for k in writer.fieldnames}
            )

    print()
    print("===== OUTPUT =====")
    print(json_path)
    print(csv_path)
    print("V2 DATASET PLAN: PASS")
    print("SOURCE DATASET WAS NOT MODIFIED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
