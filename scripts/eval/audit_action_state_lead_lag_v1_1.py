#!/usr/bin/env python
"""
Offline audit v1.1: commanded action lead vs observed follower state.

Dataset semantics under audit
-----------------------------
observation.state[t] = actual follower joint positions
action[t]            = absolute command target sent at t

The question is whether command targets start changing several frames before the
Follower state visibly responds. If yes, very similar current observations can
coexist with different current commands, making BC partially history-dependent.

v1.1 deliberately does NOT compare observed state against one global Home actual
reference for motion onset, because the Follower's static Home actual position
has repeatable load/hold variation. Instead, each episode gets its own observed
state baseline from the frames before command motion begins.

For robustness, state-response onset is reported at four margins:
0.25, 0.50, 0.75, 1.00 normalized joint units, requiring two consecutive frames.

It also estimates per-joint command lead by maximizing correlation between:
    action_delta[t]
and
    observed_state_delta[t + lag]
for lag 0..20 frames.

OFFLINE / READ-ONLY:
- no model inference
- no serial
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

STATE_KEY = "observation.state"
ACTION_KEY = "action"

STATE_MARGINS = (0.25, 0.50, 0.75, 1.00)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run containing report.json with the locked Home command.",
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
        "--action-threshold",
        type=float,
        default=0.20,
        help="Command departure threshold from locked Home command.",
    )
    p.add_argument(
        "--consecutive",
        type=int,
        default=2,
        help="Required consecutive frames for onset detection.",
    )
    p.add_argument(
        "--max-lag",
        type=int,
        default=20,
    )
    p.add_argument(
        "--alignment-first-frames",
        type=int,
        default=180,
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/eval/action_state_lead_lag_v1_1"),
    )
    args = p.parse_args()

    if not 0.01 <= args.action_threshold <= 2.0:
        p.error("--action-threshold must be 0.01..2.0")
    if not 1 <= args.consecutive <= 5:
        p.error("--consecutive must be 1..5")
    if not 0 <= args.max_lag <= 60:
        p.error("--max-lag must be 0..60")
    return args


def load_dataset(repo_id: str, root: Path) -> LeRobotDataset:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    try:
        return LeRobotDataset(repo_id=repo_id, root=root)
    except TypeError:
        return LeRobotDataset(repo_id, root=root)


def first_consecutive_true(mask: np.ndarray, consecutive: int, start: int = 0) -> int:
    mask = np.asarray(mask, dtype=bool)
    n = len(mask)
    for i in range(max(0, start), n - consecutive + 1):
        if bool(np.all(mask[i : i + consecutive])):
            return i
    return n


def summarize(values) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(a)),
        "q25": float(np.percentile(a, 25)),
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "q75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
        "max": float(np.max(a)),
    }


def best_corr_lag(
    action_delta: np.ndarray,
    state_delta: np.ndarray,
    max_lag: int,
) -> tuple[int, float]:
    """
    Positive lag means action[t] best matches observed state[t+lag].
    Uses Pearson correlation so actuator gain / command-actual scale mismatch
    does not dominate the timing estimate.
    """
    best_lag = 0
    best_corr = -np.inf

    n = len(action_delta)
    for lag in range(max_lag + 1):
        usable = n - lag
        if usable < 12:
            continue

        a = action_delta[:usable]
        s = state_delta[lag : lag + usable]

        if float(np.std(a)) < 1e-6 or float(np.std(s)) < 1e-6:
            continue

        corr = float(np.corrcoef(a, s)[0, 1])
        if np.isfinite(corr) and corr > best_corr:
            best_corr = corr
            best_lag = lag

    if best_corr == -np.inf:
        return 0, float("nan")
    return best_lag, best_corr


def main() -> int:
    args = parse_args()

    report = json.loads(
        (args.run_dir.resolve() / "report.json").read_text(encoding="utf-8")
    )
    home_cmd_dict = report["home_result"]["command_target_positions"]
    home_cmd = np.asarray(
        [home_cmd_dict[m] for m in MOTORS],
        dtype=np.float32,
    )

    print("===== LOCKED COMMAND REFERENCE =====")
    print(f"Home command: {home_cmd.tolist()}")

    print()
    print("===== LOAD DATASET =====")
    ds = load_dataset(args.repo_id, args.dataset_root)
    hf = ds.hf_dataset

    states = np.asarray(hf[STATE_KEY], dtype=np.float32)
    actions = np.asarray(hf[ACTION_KEY], dtype=np.float32)
    episodes = np.asarray(hf["episode_index"], dtype=np.int64)
    frames = np.asarray(hf["frame_index"], dtype=np.int64)

    unique_eps = sorted(int(x) for x in np.unique(episodes))
    print(f"frames: {len(states)}")
    print(f"episodes: {len(unique_eps)}")

    if len(unique_eps) != 60:
        raise RuntimeError(f"Expected 60 episodes, got {len(unique_eps)}")

    per_margin_lags = {f"{m:.2f}": [] for m in STATE_MARGINS}
    per_margin_positive = {f"{m:.2f}": 0 for m in STATE_MARGINS}
    episode_records = []

    per_joint_lags = {m: [] for m in MOTORS}
    per_joint_corrs = {m: [] for m in MOTORS}

    print()
    print("===== PER-EPISODE COMMAND ONSET -> OBSERVED RESPONSE =====")
    print(
        "episode  action_onset  "
        + "  ".join(f"state@{m:.2f}" for m in STATE_MARGINS)
    )

    for ep in unique_eps:
        idx = np.flatnonzero(episodes == ep)
        order = np.argsort(frames[idx])
        idx = idx[order]

        ep_state = states[idx]
        ep_action = actions[idx]

        action_dev = np.max(
            np.abs(ep_action - home_cmd[None, :]),
            axis=1,
        )
        action_onset = first_consecutive_true(
            action_dev > args.action_threshold,
            args.consecutive,
            start=0,
        )

        # Per-episode observed-state baseline from command-stationary frames.
        if action_onset >= 2:
            baseline_slice = ep_state[:action_onset]
        else:
            baseline_slice = ep_state[: max(1, min(5, len(ep_state)))]

        state_baseline = np.median(baseline_slice, axis=0)

        onset_by_margin = {}
        lag_by_margin = {}

        for margin in STATE_MARGINS:
            state_dev = np.max(
                np.abs(ep_state - state_baseline[None, :]),
                axis=1,
            )
            state_onset = first_consecutive_true(
                state_dev > margin,
                args.consecutive,
                start=min(action_onset, len(ep_state) - 1),
            )
            lag = state_onset - action_onset

            key = f"{margin:.2f}"
            onset_by_margin[key] = state_onset
            lag_by_margin[key] = lag
            per_margin_lags[key].append(lag)
            if lag > 0:
                per_margin_positive[key] += 1

        episode_records.append(
            {
                "episode_index": ep,
                "action_onset_frame": action_onset,
                "state_baseline": state_baseline.tolist(),
                "state_onset_by_margin": onset_by_margin,
                "state_minus_action_lag_by_margin": lag_by_margin,
            }
        )

        print(
            f"ep={ep:02d}      {action_onset:4d}       "
            + "  ".join(
                f"{onset_by_margin[f'{m:.2f}']:4d}"
                for m in STATE_MARGINS
            )
        )

        # Correlation-based joint lead estimate over early task motion.
        n = min(args.alignment_first_frames, len(ep_state))
        a_delta = ep_action[:n] - home_cmd[None, :]
        s_delta = ep_state[:n] - state_baseline[None, :]

        for j, motor in enumerate(MOTORS):
            lag, corr = best_corr_lag(
                a_delta[:, j],
                s_delta[:, j],
                args.max_lag,
            )
            per_joint_lags[motor].append(lag)
            per_joint_corrs[motor].append(corr)

    print()
    print("===== ROBUST ONSET-LAG SUMMARY =====")
    onset_summary = {}

    for margin in STATE_MARGINS:
        key = f"{margin:.2f}"
        vals = per_margin_lags[key]
        s = summarize(vals)
        positive = per_margin_positive[key]
        onset_summary[key] = {
            "state_minus_action_lag_frames": s,
            "positive_lag_fraction": positive / len(unique_eps),
        }
        print(
            f"state margin={key}: "
            f"lag median={s['median']:.1f} "
            f"mean={s['mean']:.1f} "
            f"p90={s['p90']:.1f} "
            f"min={s['min']:.0f} "
            f"max={s['max']:.0f} | "
            f"command leads in {positive}/60 "
            f"({100.0*positive/60:.1f}%)"
        )

    print()
    print("===== PER-JOINT CORRELATION LEAD =====")
    joint_summary = {}

    for motor in MOTORS:
        lags = np.asarray(per_joint_lags[motor], dtype=np.float64)
        corrs = np.asarray(per_joint_corrs[motor], dtype=np.float64)

        valid = np.isfinite(corrs)
        if not np.any(valid):
            joint_summary[motor] = {
                "valid_episodes": 0,
            }
            print(f"{motor:20s} no valid correlation estimates")
            continue

        lag_s = summarize(lags[valid])
        corr_s = summarize(corrs[valid])
        joint_summary[motor] = {
            "valid_episodes": int(np.sum(valid)),
            "best_lead_frames": lag_s,
            "best_correlation": corr_s,
        }

        print(
            f"{motor:20s} "
            f"lead median={lag_s['median']:.1f} "
            f"mean={lag_s['mean']:.1f} "
            f"p90={lag_s['p90']:.1f} | "
            f"corr median={corr_s['median']:.3f}"
        )

    result = {
        "schema_version": "action_state_lead_lag_v1_1",
        "repo_id": args.repo_id,
        "dataset_root": str(args.dataset_root.resolve()),
        "home_command": home_cmd.tolist(),
        "action_threshold": args.action_threshold,
        "required_consecutive_frames": args.consecutive,
        "state_margins": list(STATE_MARGINS),
        "max_lag": args.max_lag,
        "alignment_first_frames": args.alignment_first_frames,
        "onset_summary": onset_summary,
        "per_joint_correlation_lead": joint_summary,
        "episodes": episode_records,
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "action_state_lead_lag_summary.json"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("===== OUTPUT =====")
    print(json_path)
    print("OFFLINE ACTION/STATE LEAD-LAG AUDIT v1.1: PASS")
    print("NO MODEL OR HARDWARE WAS ACCESSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
