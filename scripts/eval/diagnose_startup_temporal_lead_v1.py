#!/usr/bin/env python
"""
Diagnose startup idle-prefix ambiguity and ACT effective temporal lead.

Inputs
------
- home_anchor_bias_summary.json produced by audit_act_home_anchor_bias_v1.py
- local 60-episode LeRobot dataset

What this checks
----------------
1. How long each episode remains at the locked Home command from frame 0.
2. For the already-computed 15k/20k predicted action[0], which FUTURE
   demonstration action (offset 0..49) it most closely resembles.
3. Whether that best-match future offset tracks the number of frames until
   the demonstration actually leaves Home.

This separates two hypotheses:
A) temporal anticipation caused by variable idle prefixes / phase ambiguity;
B) a more general absolute-action calibration error.

OFFLINE / READ-ONLY:
- no model inference
- no serial ports
- no cameras
- no motor writes
"""

from __future__ import annotations

import argparse
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

ACTION_KEY = "action"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--bias-json",
        type=Path,
        default=Path(
            "outputs/eval/act_home_anchor_bias_v1/"
            "home_anchor_bias_summary.json"
        ),
    )
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/lerobot/so101_red_cube_pick_place_v1"),
    )
    p.add_argument(
        "--repo-id",
        default="connorluis/so101_red_cube_pick_place_v1",
    )
    p.add_argument(
        "--max-future-offset",
        type=int,
        default=49,
    )
    p.add_argument(
        "--motion-threshold",
        type=float,
        default=0.20,
        help=(
            "Home departure threshold: max absolute joint-action deviation "
            "from locked Home command."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/eval/startup_temporal_lead_v1"),
    )
    args = p.parse_args()
    if not 10 <= args.max_future_offset <= 100:
        p.error("--max-future-offset must be 10..100")
    if not 0.01 <= args.motion_threshold <= 2.0:
        p.error("--motion-threshold must be 0.01..2.0")
    return args


def load_dataset(repo_id: str, root: Path) -> LeRobotDataset:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    try:
        return LeRobotDataset(repo_id=repo_id, root=root)
    except TypeError:
        return LeRobotDataset(repo_id, root=root)


def hold_length(
    ep_actions: np.ndarray,
    home_command: np.ndarray,
    threshold: float,
) -> int:
    dev = np.max(
        np.abs(ep_actions - home_command[None, :]),
        axis=1,
    )
    moved = np.flatnonzero(dev > threshold)
    if len(moved) == 0:
        return len(ep_actions)
    return int(moved[0])


def summarize_int(values: list[int]) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {
        "min": int(np.min(a)),
        "q25": float(np.percentile(a, 25)),
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "q75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
        "max": int(np.max(a)),
    }


def main() -> int:
    args = parse_args()

    bias_path = args.bias_json.resolve()
    if not bias_path.is_file():
        raise FileNotFoundError(bias_path)

    bias = json.loads(bias_path.read_text(encoding="utf-8"))
    home_command = np.asarray(
        bias["home_command"],
        dtype=np.float32,
    )

    print("===== LOAD DATASET =====")
    ds = load_dataset(args.repo_id, args.dataset_root)
    hf = ds.hf_dataset

    actions = np.asarray(hf[ACTION_KEY], dtype=np.float32)
    episodes = np.asarray(hf["episode_index"], dtype=np.int64)
    frames = np.asarray(hf["frame_index"], dtype=np.int64)

    unique_eps = sorted(int(x) for x in np.unique(episodes))
    if len(unique_eps) != 60:
        raise RuntimeError(f"Expected 60 episodes, got {len(unique_eps)}")

    action_std = np.maximum(actions.std(axis=0), 1.0)

    print(f"frames: {len(actions)}")
    print(f"episodes: {len(unique_eps)}")
    print(f"locked Home command: {home_command.tolist()}")

    # --------------------------------------------------------------
    # Part 1: startup idle prefix audit
    # --------------------------------------------------------------
    thresholds = [0.05, 0.10, 0.20, 0.50]
    hold_by_threshold: dict[str, list[int]] = {
        f"{t:.2f}": [] for t in thresholds
    }

    motion_onset_by_ep: dict[int, int] = {}

    print()
    print("===== STARTUP HOME-HOLD LENGTH =====")
    print("episode  " + "  ".join(f"thr={t:.2f}" for t in thresholds))

    for ep in unique_eps:
        idx = np.flatnonzero(episodes == ep)
        order = np.argsort(frames[idx])
        idx = idx[order]
        ep_actions = actions[idx]

        row = []
        for t in thresholds:
            n = hold_length(ep_actions, home_command, t)
            hold_by_threshold[f"{t:.2f}"].append(n)
            row.append(n)

        onset = hold_length(
            ep_actions,
            home_command,
            args.motion_threshold,
        )
        motion_onset_by_ep[ep] = onset

        print(
            f"ep={ep:02d}    "
            + "  ".join(f"{x:3d}" for x in row)
        )

    print()
    print("===== HOLD-LENGTH SUMMARY =====")
    hold_summary = {}
    for t in thresholds:
        key = f"{t:.2f}"
        s = summarize_int(hold_by_threshold[key])
        hold_summary[key] = s
        print(
            f"threshold={key}: "
            f"min={s['min']} "
            f"median={s['median']:.1f} "
            f"mean={s['mean']:.1f} "
            f"p90={s['p90']:.1f} "
            f"max={s['max']} "
            f"(median={s['median']/15.0:.2f}s)"
        )

    # --------------------------------------------------------------
    # Part 2: for stored action[0] predictions, find which future
    # demo action they most closely resemble.
    # --------------------------------------------------------------
    output = {
        "schema_version": "startup_temporal_lead_v1",
        "bias_json": str(bias_path),
        "repo_id": args.repo_id,
        "dataset_root": str(args.dataset_root.resolve()),
        "home_command": home_command.tolist(),
        "motion_threshold": args.motion_threshold,
        "hold_length_summary": hold_summary,
        "checkpoints": {},
    }

    print()
    print("===== EFFECTIVE TEMPORAL LEAD =====")

    for ckpt in ["15k", "20k"]:
        records = bias["checkpoints"][ckpt]["records"]

        best_offsets = []
        frames_until_motion = []
        current_raw_mae = []
        best_raw_mae = []
        selected_records = []

        print()
        print(f"--- {ckpt} ---")

        for rec in records:
            ep = int(rec["episode_index"])
            frame = int(rec["frame_index"])
            pred0 = np.asarray(
                rec["pred_action_0"],
                dtype=np.float32,
            )

            ep_idx = np.flatnonzero(episodes == ep)
            frame_to_global = {
                int(frames[i]): int(i) for i in ep_idx
            }

            candidates = []
            max_off = args.max_future_offset
            for off in range(max_off + 1):
                gi = frame_to_global.get(frame + off)
                if gi is None:
                    break
                target = actions[gi]
                raw_mae = float(np.mean(np.abs(pred0 - target)))
                norm_rmse = float(
                    np.sqrt(
                        np.mean(
                            ((pred0 - target) / action_std) ** 2
                        )
                    )
                )
                candidates.append(
                    (norm_rmse, raw_mae, off)
                )

            if not candidates:
                raise RuntimeError(
                    f"No future candidates for ep={ep} frame={frame}"
                )

            candidates.sort()
            best_norm, best_mae, best_off = candidates[0]

            gi0 = frame_to_global[frame]
            mae0 = float(
                np.mean(np.abs(pred0 - actions[gi0]))
            )

            onset = motion_onset_by_ep[ep]
            until_motion = max(onset - frame, 0)

            best_offsets.append(best_off)
            frames_until_motion.append(until_motion)
            current_raw_mae.append(mae0)
            best_raw_mae.append(best_mae)

            selected_records.append(
                {
                    "episode_index": ep,
                    "selected_frame": frame,
                    "motion_onset_frame": onset,
                    "frames_until_motion": until_motion,
                    "best_future_offset": best_off,
                    "current_action_mae": mae0,
                    "best_future_action_mae": best_mae,
                    "best_future_normalized_rmse": best_norm,
                    "pred_action_0": pred0.tolist(),
                }
            )

        bo = np.asarray(best_offsets, dtype=np.float64)
        fm = np.asarray(frames_until_motion, dtype=np.float64)
        mae0 = np.asarray(current_raw_mae)
        maeb = np.asarray(best_raw_mae)

        positive_lead_fraction = float(np.mean(bo > 0))
        lead_ge5_fraction = float(np.mean(bo >= 5))
        lead_ge10_fraction = float(np.mean(bo >= 10))

        if np.std(bo) > 1e-9 and np.std(fm) > 1e-9:
            corr = float(np.corrcoef(bo, fm)[0, 1])
        else:
            corr = None

        summary = {
            "best_offset": summarize_int(
                [int(x) for x in best_offsets]
            ),
            "frames_until_motion": summarize_int(
                [int(x) for x in frames_until_motion]
            ),
            "positive_lead_fraction": positive_lead_fraction,
            "lead_ge5_fraction": lead_ge5_fraction,
            "lead_ge10_fraction": lead_ge10_fraction,
            "mean_current_action_mae": float(np.mean(mae0)),
            "mean_best_future_action_mae": float(np.mean(maeb)),
            "mean_mae_improvement": float(
                np.mean(mae0 - maeb)
            ),
            "corr_best_offset_vs_frames_until_motion": corr,
        }

        output["checkpoints"][ckpt] = {
            "summary": summary,
            "records": selected_records,
        }

        b = summary["best_offset"]
        f = summary["frames_until_motion"]

        print(
            f"best future offset: "
            f"median={b['median']:.1f} "
            f"mean={b['mean']:.1f} "
            f"p90={b['p90']:.1f} "
            f"max={b['max']}"
        )
        print(
            f"frames until demo leaves Home: "
            f"median={f['median']:.1f} "
            f"mean={f['mean']:.1f} "
            f"p90={f['p90']:.1f}"
        )
        print(
            f"best offset >0:   "
            f"{positive_lead_fraction*100:.1f}%"
        )
        print(
            f"best offset >=5:  "
            f"{lead_ge5_fraction*100:.1f}%"
        )
        print(
            f"best offset >=10: "
            f"{lead_ge10_fraction*100:.1f}%"
        )
        print(
            f"mean MAE current action: "
            f"{summary['mean_current_action_mae']:.4f}"
        )
        print(
            f"mean MAE best future action: "
            f"{summary['mean_best_future_action_mae']:.4f}"
        )
        print(
            f"mean improvement by allowing future offset: "
            f"{summary['mean_mae_improvement']:.4f}"
        )
        print(
            "corr(best offset, frames until motion): "
            f"{corr if corr is not None else 'n/a'}"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "startup_temporal_lead_summary.json"
    json_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("===== OUTPUT =====")
    print(json_path)
    print("STARTUP TEMPORAL-LEAD DIAGNOSIS: PASS")
    print("NO MODEL INFERENCE OR HARDWARE WAS ACCESSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
