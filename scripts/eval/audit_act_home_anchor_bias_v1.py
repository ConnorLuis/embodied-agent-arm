#!/usr/bin/env python
"""
Offline audit of ACT absolute action[0] bias across all 60 episodes.

Purpose
-------
The exact live Home observation showed that ACT's first absolute action can be
offset from the demonstration Home command even when the local chunk trajectory
shape is good. This script determines whether that offset is systematic across
episodes or peculiar to one live query.

For each episode:
1. Search only the early episode window for the frame closest to the locked
   Home state + Home command.
2. Load that frame's Front/Wrist images.
3. Run 15k and 20k ACT checkpoints on the exact training observation.
4. Compare predicted action[0] with the exact demonstration action[t].
5. Compare predicted chunk delta 0->10 with demonstration action[t+10]-action[t].

Outputs per-joint signed error, MAE, p90 absolute error, sign consistency,
and per-episode records.

OFFLINE / READ-ONLY:
- no serial ports
- no camera devices
- no motor writes
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy


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
FRONT_KEY = "observation.images.front"
WRIST_KEY = "observation.images.wrist"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
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
        "--checkpoint-15k",
        type=Path,
        default=Path(
            "outputs/train/act_red_cube_v1/checkpoints/015000/pretrained_model"
        ),
    )
    p.add_argument(
        "--checkpoint-20k",
        type=Path,
        default=Path(
            "outputs/train/act_red_cube_v1/checkpoints/020000/pretrained_model"
        ),
    )
    p.add_argument(
        "--search-first-frames",
        type=int,
        default=120,
        help="Only search this many early frames per episode for Home.",
    )
    p.add_argument(
        "--future-offset",
        type=int,
        default=10,
        help="Compare chunk delta against demonstration continuation.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/eval/act_home_anchor_bias_v1"),
    )
    return p.parse_args()


def fmt(x: np.ndarray, digits: int = 3) -> str:
    x = np.asarray(x).reshape(-1)
    return "[" + ", ".join(f"{float(v):.{digits}f}" for v in x) + "]"


def image_tensor(x, device: torch.device) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)

    t = x.detach()
    if t.ndim != 3:
        raise RuntimeError(f"Unexpected image shape: {tuple(t.shape)}")

    if t.shape[0] in (1, 3, 4):
        t = t[:3]
    elif t.shape[-1] in (1, 3, 4):
        t = t[..., :3].permute(2, 0, 1)
    else:
        raise RuntimeError(f"Cannot infer image channels: {tuple(t.shape)}")

    t = t.float()
    if float(t.max()) > 1.5:
        t = t / 255.0
    return t.unsqueeze(0).to(device)


def load_dataset(repo_id: str, root: Path) -> LeRobotDataset:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    try:
        return LeRobotDataset(repo_id=repo_id, root=root)
    except TypeError:
        return LeRobotDataset(repo_id, root=root)


def load_policy(path: Path, name: str) -> ACTPolicy:
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    print(f"Loading {name}: {path}")
    policy = ACTPolicy.from_pretrained(path, local_files_only=True)
    policy.eval()
    policy.reset()
    return policy


@torch.no_grad()
def infer_one(
    policy: ACTPolicy,
    item: dict,
) -> np.ndarray:
    device = torch.device(policy.config.device)

    state = item[STATE_KEY]
    if not isinstance(state, torch.Tensor):
        state = torch.as_tensor(state)
    state = state.float().reshape(1, 6).to(device)

    batch = {
        STATE_KEY: state,
        FRONT_KEY: image_tensor(item[FRONT_KEY], device),
        WRIST_KEY: image_tensor(item[WRIST_KEY], device),
    }

    policy.reset()
    chunk = policy.predict_action_chunk(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()

    arr = chunk.detach().cpu().float().numpy()
    if arr.shape != (1, 50, 6):
        raise RuntimeError(f"Unexpected chunk shape: {arr.shape}")
    if not np.isfinite(arr).all():
        raise FloatingPointError("NaN/Inf in ACT output.")
    return arr[0]


def summarize_errors(errors: np.ndarray) -> dict:
    # errors: [episodes, motors]
    abs_e = np.abs(errors)
    out = {}
    for j, motor in enumerate(MOTORS):
        col = errors[:, j]
        abs_col = abs_e[:, j]

        pos_frac = float(np.mean(col > 0.10))
        neg_frac = float(np.mean(col < -0.10))
        near_frac = 1.0 - pos_frac - neg_frac

        out[motor] = {
            "mean_signed_error": float(np.mean(col)),
            "median_signed_error": float(np.median(col)),
            "mae": float(np.mean(abs_col)),
            "median_abs_error": float(np.median(abs_col)),
            "p90_abs_error": float(np.percentile(abs_col, 90)),
            "max_abs_error": float(np.max(abs_col)),
            "positive_fraction_gt_0p1": pos_frac,
            "negative_fraction_lt_minus_0p1": neg_frac,
            "near_zero_fraction": near_frac,
        }
    out["_overall"] = {
        "mean_mae_all_values": float(np.mean(abs_e)),
        "median_abs_all_values": float(np.median(abs_e)),
        "p90_abs_all_values": float(np.percentile(abs_e, 90)),
    }
    return out


def main() -> int:
    args = parse_args()

    run_dir = args.run_dir.resolve()
    report = json.loads(
        (run_dir / "report.json").read_text(encoding="utf-8")
    )

    home_state_dict = report["home_result"]["target_positions"]
    home_cmd_dict = report["home_result"]["command_target_positions"]

    home_state = np.asarray(
        [home_state_dict[m] for m in MOTORS],
        dtype=np.float32,
    )
    home_command = np.asarray(
        [home_cmd_dict[m] for m in MOTORS],
        dtype=np.float32,
    )

    print("===== LOCKED HOME REFERENCES =====")
    print(f"Home actual reference: {fmt(home_state)}")
    print(f"Home command:          {fmt(home_command)}")

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

    state_scale = np.maximum(states.std(axis=0), 1.0)
    action_scale = np.maximum(actions.std(axis=0), 1.0)

    selected = []
    future_indices = []

    print()
    print("===== SELECT ONE HOME-LIKE FRAME PER EPISODE =====")
    for ep in unique_eps:
        ep_idx = np.flatnonzero(
            (episodes == ep)
            & (frames < args.search_first_frames)
        )
        if len(ep_idx) == 0:
            raise RuntimeError(f"Episode {ep}: no early frames.")

        s_term = np.mean(
            ((states[ep_idx] - home_state) / state_scale) ** 2,
            axis=1,
        )
        a_term = np.mean(
            ((actions[ep_idx] - home_command) / action_scale) ** 2,
            axis=1,
        )
        score = 0.5 * s_term + 0.5 * a_term

        gi = int(ep_idx[int(np.argmin(score))])
        future_frame = int(frames[gi]) + args.future_offset

        hits = np.flatnonzero(
            (episodes == ep)
            & (frames == future_frame)
        )
        if len(hits) != 1:
            raise RuntimeError(
                f"Episode {ep}: missing future frame {future_frame}"
            )

        selected.append(gi)
        future_indices.append(int(hits[0]))

        print(
            f"ep={ep:02d} frame={int(frames[gi]):03d} "
            f"state_err={float(np.max(np.abs(states[gi]-home_state))):.3f} "
            f"action_err={float(np.max(np.abs(actions[gi]-home_command))):.3f}"
        )

    selected = np.asarray(selected, dtype=np.int64)
    future_indices = np.asarray(future_indices, dtype=np.int64)

    demo_t = actions[selected]
    demo_future = actions[future_indices]
    demo_delta = demo_future - demo_t

    print()
    print("Selected frame-index stats:")
    chosen_frames = frames[selected]
    print(
        f"min={int(chosen_frames.min())} "
        f"median={float(np.median(chosen_frames)):.1f} "
        f"max={int(chosen_frames.max())}"
    )
    print(f"median demo action[t]: {fmt(np.median(demo_t, axis=0))}")
    print(
        f"median demo delta t+{args.future_offset}: "
        f"{fmt(np.median(demo_delta, axis=0))}"
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")

    policies = {
        "15k": load_policy(args.checkpoint_15k, "15k"),
        "20k": load_policy(args.checkpoint_20k, "20k"),
    }

    records = {
        "15k": [],
        "20k": [],
    }
    pred0 = {
        "15k": [],
        "20k": [],
    }
    pred_delta = {
        "15k": [],
        "20k": [],
    }

    print()
    print("===== INFERENCE ACROSS 60 HOME-LIKE TRAINING OBSERVATIONS =====")

    for n, (gi, fi) in enumerate(
        zip(selected, future_indices),
        start=1,
    ):
        item = ds[int(gi)]
        ep = int(episodes[gi])
        frame = int(frames[gi])

        for name, policy in policies.items():
            chunk = infer_one(policy, item)
            first = chunk[0]
            delta = chunk[args.future_offset] - chunk[0]

            pred0[name].append(first)
            pred_delta[name].append(delta)

            records[name].append(
                {
                    "episode_index": ep,
                    "frame_index": frame,
                    "global_index": int(gi),
                    "state": states[gi].tolist(),
                    "demo_action_t": actions[gi].tolist(),
                    "pred_action_0": first.tolist(),
                    "action0_error": (
                        first - actions[gi]
                    ).tolist(),
                    "demo_future_action": actions[fi].tolist(),
                    "demo_delta": demo_delta[n - 1].tolist(),
                    "pred_delta": delta.tolist(),
                    "delta_error": (
                        delta - demo_delta[n - 1]
                    ).tolist(),
                }
            )

        if n % 10 == 0 or n == len(selected):
            print(f"processed {n}/60 episode anchors")

    for name in list(policies):
        del policies[name]
    gc.collect()
    torch.cuda.empty_cache()

    result = {
        "schema_version": "audit_act_home_anchor_bias_v1",
        "run_dir": str(run_dir),
        "repo_id": args.repo_id,
        "dataset_root": str(args.dataset_root.resolve()),
        "home_actual_reference": home_state.tolist(),
        "home_command": home_command.tolist(),
        "search_first_frames": args.search_first_frames,
        "future_offset": args.future_offset,
        "selected_frame_indices": chosen_frames.tolist(),
        "checkpoints": {},
    }

    print()
    print("===== SUMMARY =====")

    ranking = []

    for name in ["15k", "20k"]:
        p0 = np.stack(pred0[name])
        pd = np.stack(pred_delta[name])

        action0_errors = p0 - demo_t
        delta_errors = pd - demo_delta

        action0_summary = summarize_errors(action0_errors)
        delta_summary = summarize_errors(delta_errors)

        result["checkpoints"][name] = {
            "action0_error_summary": action0_summary,
            "delta_error_summary": delta_summary,
            "records": records[name],
        }

        score = (
            action0_summary["_overall"]["mean_mae_all_values"]
            + delta_summary["_overall"]["mean_mae_all_values"]
        )
        ranking.append((score, name))

        print()
        print(f"--- {name} absolute action[0] error ---")
        print(
            "motor                 mean_signed   MAE     p90_abs   "
            "pos_frac  neg_frac"
        )
        for motor in MOTORS:
            s = action0_summary[motor]
            print(
                f"{motor:20s} "
                f"{s['mean_signed_error']:10.3f} "
                f"{s['mae']:7.3f} "
                f"{s['p90_abs_error']:9.3f} "
                f"{s['positive_fraction_gt_0p1']:8.2f} "
                f"{s['negative_fraction_lt_minus_0p1']:8.2f}"
            )

        print(
            f"{name} overall action[0] MAE: "
            f"{action0_summary['_overall']['mean_mae_all_values']:.4f}"
        )
        print(
            f"{name} overall delta(0->{args.future_offset}) MAE: "
            f"{delta_summary['_overall']['mean_mae_all_values']:.4f}"
        )

    ranking.sort()
    result["ranking"] = [
        {
            "rank": i + 1,
            "checkpoint": name,
            "score": float(score),
        }
        for i, (score, name) in enumerate(ranking)
    ]

    print()
    print("===== RANKING =====")
    for r in result["ranking"]:
        print(
            f"#{r['rank']} {r['checkpoint']} "
            f"score={r['score']:.4f}"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "home_anchor_bias_summary.json"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("===== OUTPUT =====")
    print(json_path)
    print("OFFLINE HOME-ANCHOR BIAS AUDIT: PASS")
    print("NO HARDWARE WAS ACCESSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
