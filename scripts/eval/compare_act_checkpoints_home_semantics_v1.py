#!/usr/bin/env python
"""
Offline ACT checkpoint semantic comparison at the exact Home observation.

Goal
----
Determine whether the absolute action baseline mismatch seen in the 15k policy
(elbow/wrist especially) is checkpoint-specific or shared by 20k.

This script:
1. Reads the exact Home state + saved Front/Wrist images from a completed
   act_replan_after_10transition_v1 run.
2. Searches the local 60-episode LeRobot dataset for visually/state-similar
   demonstration frames.
3. Builds a median demonstration continuation at t,t+1,t+5,t+10,t+20.
4. Runs ACT checkpoints 15k and 20k on the SAME Home observation.
5. Compares:
   - absolute action[0] vs demonstration action[t]
   - absolute action[0] vs current actual state
   - absolute action[0] vs validated Home command
   - chunk trajectory deltas vs demonstration continuation deltas
6. Writes JSON + CSV.

OFFLINE / READ-ONLY:
- no serial ports
- no camera devices
- no motor writes
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
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

OFFSETS = [0, 1, 5, 10, 20]


@dataclass
class Candidate:
    global_index: int
    episode_index: int
    frame_index: int
    state_distance: float
    visual_distance: float
    final_score: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help=(
            "Completed act_replan_after_10transition_v1 run containing "
            "report.json, front_home.png and wrist_home.png"
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
    p.add_argument("--candidate-pool", type=int, default=240)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--max-per-episode", type=int, default=1)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/eval/checkpoint_home_semantics_v1"),
    )
    args = p.parse_args()

    if not 80 <= args.candidate_pool <= 1000:
        p.error("--candidate-pool must be 80..1000")
    if not 5 <= args.top_k <= 20:
        p.error("--top-k must be 5..20")
    return args


def fmt(x: np.ndarray, digits: int = 3) -> str:
    x = np.asarray(x).reshape(-1)
    return "[" + ", ".join(f"{float(v):.{digits}f}" for v in x) + "]"


def tensor_image_to_rgb_uint8(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu()
        if t.ndim != 3:
            raise RuntimeError(f"Unexpected image tensor shape: {tuple(t.shape)}")
        if t.shape[0] in (1, 3, 4):
            t = t[:3].permute(1, 2, 0)
        elif t.shape[-1] in (1, 3, 4):
            t = t[..., :3]
        else:
            raise RuntimeError(f"Cannot infer channels: {tuple(t.shape)}")
        arr = t.numpy()
    else:
        arr = np.asarray(x)
        if arr.ndim != 3:
            raise RuntimeError(f"Unexpected image array shape: {arr.shape}")
        if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr[:3], (1, 2, 0))

    if np.issubdtype(arr.dtype, np.floating):
        if float(np.nanmax(arr)) <= 1.5:
            arr = arr * 255.0
    return np.clip(arr[..., :3], 0, 255).astype(np.uint8)


def load_rgb(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    arr = np.asarray(Image.open(path).convert("RGB"))
    if arr.shape != (480, 480, 3):
        raise RuntimeError(f"{path}: expected 480x480 RGB, got {arr.shape}")
    return arr


def appearance_feature(rgb: np.ndarray, size: int = 64) -> np.ndarray:
    small = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0]
    L = (L - L.mean()) / max(float(L.std()), 8.0)
    A = (lab[..., 1] - 128.0) / 64.0
    B = (lab[..., 2] - 128.0) / 64.0
    return np.stack([L, A, B], axis=-1)


def red_mask(rgb: np.ndarray, size: int = 96) -> np.ndarray:
    small = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    m1 = cv2.inRange(
        hsv,
        np.array([0, 80, 55], np.uint8),
        np.array([12, 255, 255], np.uint8),
    )
    m2 = cv2.inRange(
        hsv,
        np.array([168, 80, 55], np.uint8),
        np.array([179, 255, 255], np.uint8),
    )
    mask = ((m1 > 0) | (m2 > 0)).astype(np.float32)
    return cv2.GaussianBlur(mask, (5, 5), 0)


def robust_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    med = float(np.median(values))
    q25, q75 = np.percentile(values, [25, 75])
    iqr = max(float(q75 - q25), 1e-8)
    return (values - med) / iqr


def select_diverse(
    candidates: list[Candidate],
    top_k: int,
    max_per_episode: int,
) -> list[Candidate]:
    selected = []
    counts: dict[int, int] = {}
    for c in sorted(candidates, key=lambda x: x.final_score):
        if counts.get(c.episode_index, 0) >= max_per_episode:
            continue
        selected.append(c)
        counts[c.episode_index] = counts.get(c.episode_index, 0) + 1
        if len(selected) >= top_k:
            break
    return selected


def load_dataset(repo_id: str, root: Path) -> LeRobotDataset:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    try:
        return LeRobotDataset(repo_id=repo_id, root=root)
    except TypeError:
        return LeRobotDataset(repo_id, root=root)


def safe_future_index(
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    ep: int,
    frame: int,
) -> int | None:
    hits = np.flatnonzero(
        (episode_indices == ep) & (frame_indices == frame)
    )
    if len(hits) != 1:
        return None
    return int(hits[0])


def image_to_policy_tensor(
    rgb: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    t = (
        torch.from_numpy(rgb.copy())
        .permute(2, 0, 1)
        .float()
        .div_(255.0)
        .unsqueeze(0)
    )
    return t.to(device)


def run_checkpoint(
    name: str,
    checkpoint: Path,
    state: np.ndarray,
    front: np.ndarray,
    wrist: np.ndarray,
) -> np.ndarray:
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)

    print()
    print(f"===== LOAD {name} =====")
    policy = ACTPolicy.from_pretrained(
        checkpoint,
        local_files_only=True,
    )
    policy.eval()
    policy.reset()

    device = torch.device(policy.config.device)
    batch = {
        "observation.state": (
            torch.from_numpy(state.astype(np.float32))
            .unsqueeze(0)
            .to(device)
        ),
        "observation.images.front": image_to_policy_tensor(front, device),
        "observation.images.wrist": image_to_policy_tensor(wrist, device),
    }

    # Warm using the real batch, discard, reset, then retain next output.
    with torch.inference_mode():
        _ = policy.predict_action_chunk(batch)
    torch.cuda.synchronize()
    policy.reset()

    with torch.inference_mode():
        chunk = policy.predict_action_chunk(batch)
    torch.cuda.synchronize()

    arr = chunk.detach().cpu().float().numpy()
    if arr.shape != (1, 50, 6):
        raise RuntimeError(f"{name}: unexpected chunk shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise FloatingPointError(f"{name}: NaN/Inf")

    result = arr[0]
    del policy, batch, chunk
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    args = parse_args()

    run_dir = args.run_dir.resolve()
    report_path = run_dir / "report.json"
    front_path = run_dir / "front_home.png"
    wrist_path = run_dir / "wrist_home.png"

    if not report_path.is_file():
        raise FileNotFoundError(report_path)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    query_state = np.asarray(report["initial_actual"], dtype=np.float32)

    home_cmd_dict = report["home_result"]["command_target_positions"]
    home_command = np.asarray(
        [home_cmd_dict[m] for m in MOTORS],
        dtype=np.float32,
    )

    query_front = load_rgb(front_path)
    query_wrist = load_rgb(wrist_path)

    print("===== EXACT HOME QUERY =====")
    print(f"run dir:      {run_dir}")
    print(f"actual state: {fmt(query_state)}")
    print(f"Home command: {fmt(home_command)}")

    print()
    print("===== LOAD DATASET =====")
    ds = load_dataset(args.repo_id, args.dataset_root)
    hf = ds.hf_dataset

    states = np.asarray(hf[STATE_KEY], dtype=np.float32)
    actions = np.asarray(hf[ACTION_KEY], dtype=np.float32)
    episodes = np.asarray(hf["episode_index"], dtype=np.int64)
    frames = np.asarray(hf["frame_index"], dtype=np.int64)

    print(f"frames: {len(states)}")
    print(f"episodes: {len(np.unique(episodes))}")

    # Stage 1: state prefilter.
    state_std = np.maximum(states.std(axis=0), 1.0)
    z = (states - query_state[None, :]) / state_std[None, :]
    state_dist = np.sqrt(np.mean(z * z, axis=1))

    pool_n = min(args.candidate_pool, len(states))
    pool = np.argpartition(state_dist, pool_n - 1)[:pool_n]
    pool = pool[np.argsort(state_dist[pool])]

    # Stage 2: visual rerank.
    qff = appearance_feature(query_front)
    qwf = appearance_feature(query_wrist)
    qfr = red_mask(query_front)
    qwr = red_mask(query_wrist)

    raw = []
    print()
    print("===== VISUAL RE-RANK =====")
    for n, gi in enumerate(pool, start=1):
        item = ds[int(gi)]
        f = tensor_image_to_rgb_uint8(item[FRONT_KEY])
        w = tensor_image_to_rgb_uint8(item[WRIST_KEY])

        fd = float(np.mean(np.abs(appearance_feature(f) - qff)))
        wd = float(np.mean(np.abs(appearance_feature(w) - qwf)))
        fr = float(np.mean(np.abs(red_mask(f) - qfr)))
        wr = float(np.mean(np.abs(red_mask(w) - qwr)))
        vd = 0.35 * fd + 0.55 * wd + 0.10 * (0.30 * fr + 0.70 * wr)

        raw.append(
            (
                int(gi),
                int(episodes[gi]),
                int(frames[gi]),
                float(state_dist[gi]),
                vd,
            )
        )

        if n % 40 == 0 or n == len(pool):
            print(f"decoded {n}/{len(pool)} candidate pairs")

    state_scaled = robust_scale(np.asarray([r[3] for r in raw]))
    visual_scaled = robust_scale(np.asarray([r[4] for r in raw]))

    candidates = []
    for i, r in enumerate(raw):
        score = 0.35 * state_scaled[i] + 0.65 * visual_scaled[i]
        candidates.append(
            Candidate(
                global_index=r[0],
                episode_index=r[1],
                frame_index=r[2],
                state_distance=r[3],
                visual_distance=r[4],
                final_score=float(score),
            )
        )

    selected = select_diverse(
        candidates,
        top_k=args.top_k,
        max_per_episode=args.max_per_episode,
    )

    print()
    print("===== NEAREST DEMONSTRATION FRAMES =====")
    demo_sequences = []
    match_records = []

    for rank, c in enumerate(selected, start=1):
        seq = []
        valid = True
        for off in OFFSETS:
            gi = safe_future_index(
                episodes,
                frames,
                c.episode_index,
                c.frame_index + off,
            )
            if gi is None:
                valid = False
                break
            seq.append(actions[gi])

        if not valid:
            continue

        seq = np.stack(seq)
        demo_sequences.append(seq)
        match_records.append(
            {
                "rank": rank,
                "episode_index": c.episode_index,
                "frame_index": c.frame_index,
                "time_seconds": c.frame_index / 15.0,
                "state_distance": c.state_distance,
                "visual_distance": c.visual_distance,
                "final_score": c.final_score,
                "action_t": seq[0].tolist(),
            }
        )

        print(
            f"#{rank:02d} ep={c.episode_index:02d} "
            f"frame={c.frame_index:03d} "
            f"state={c.state_distance:.4f} "
            f"visual={c.visual_distance:.4f}"
        )

    if len(demo_sequences) < 5:
        raise RuntimeError(
            f"Only {len(demo_sequences)} valid demo sequences found."
        )

    demo_sequences = np.stack(demo_sequences, axis=0)
    demo_median = np.median(demo_sequences, axis=0)
    demo_delta = demo_sequences - demo_sequences[:, 0:1, :]
    demo_median_delta = np.median(demo_delta, axis=0)

    print()
    print("demo median action[t]:")
    print(fmt(demo_median[0]))
    print("demo median delta t->t+10:")
    print(fmt(demo_median_delta[OFFSETS.index(10)]))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")

    chunks = {
        "15k": run_checkpoint(
            "15k",
            args.checkpoint_15k,
            query_state,
            query_front,
            query_wrist,
        ),
        "20k": run_checkpoint(
            "20k",
            args.checkpoint_20k,
            query_state,
            query_front,
            query_wrist,
        ),
    }

    result = {
        "schema_version": "compare_act_checkpoints_home_semantics_v1",
        "run_dir": str(run_dir),
        "query_state": query_state.tolist(),
        "home_command": home_command.tolist(),
        "offsets": OFFSETS,
        "matches": match_records,
        "demo_median_action": {
            str(off): demo_median[i].tolist()
            for i, off in enumerate(OFFSETS)
        },
        "demo_median_delta": {
            str(off): demo_median_delta[i].tolist()
            for i, off in enumerate(OFFSETS)
        },
        "checkpoints": {},
    }

    print()
    print("===== CHECKPOINT COMPARISON =====")

    csv_rows = []

    for name, chunk in chunks.items():
        act = np.stack([chunk[off] for off in OFFSETS], axis=0)
        act_delta = act - act[0:1, :]

        abs0_error = act[0] - demo_median[0]
        abs0_mae = float(np.mean(np.abs(abs0_error)))
        vs_actual = act[0] - query_state
        vs_home_command = act[0] - home_command

        trajectory_mae = np.mean(
            np.abs(act_delta - demo_median_delta),
            axis=1,
        )

        print()
        print(f"--- {name} ---")
        print(f"action[0]:                 {fmt(act[0])}")
        print(f"demo median action[t]:     {fmt(demo_median[0])}")
        print(f"action[0]-demo:            {fmt(abs0_error)}")
        print(f"absolute action[0] MAE:    {abs0_mae:.4f}")
        print(f"action[0]-actual state:    {fmt(vs_actual)}")
        print(f"action[0]-Home command:    {fmt(vs_home_command)}")
        for oi, off in enumerate(OFFSETS[1:], start=1):
            print(
                f"delta-MAE t+{off:02d}: "
                f"{trajectory_mae[oi]:.4f}"
            )

        result["checkpoints"][name] = {
            "action_0": act[0].tolist(),
            "action_0_error_vs_demo_median": abs0_error.tolist(),
            "action_0_mae_vs_demo_median": abs0_mae,
            "action_0_delta_vs_actual_state": vs_actual.tolist(),
            "action_0_delta_vs_home_command": vs_home_command.tolist(),
            "actions_at_offsets": {
                str(off): act[i].tolist()
                for i, off in enumerate(OFFSETS)
            },
            "delta_from_action_0": {
                str(off): act_delta[i].tolist()
                for i, off in enumerate(OFFSETS)
            },
            "trajectory_delta_mae_vs_demo_median": {
                str(off): float(trajectory_mae[i])
                for i, off in enumerate(OFFSETS)
            },
        }

        for oi, off in enumerate(OFFSETS):
            for j, motor in enumerate(MOTORS):
                csv_rows.append(
                    [
                        name,
                        off,
                        motor,
                        float(act[oi, j]),
                        float(demo_median[oi, j]),
                        float(act_delta[oi, j]),
                        float(demo_median_delta[oi, j]),
                    ]
                )

    # Simple deterministic ranking:
    # primary = absolute action[0] MAE to nearest demos
    # secondary = t+10 trajectory delta MAE
    scored = []
    for name, rec in result["checkpoints"].items():
        score = (
            float(rec["action_0_mae_vs_demo_median"])
            + float(rec["trajectory_delta_mae_vs_demo_median"]["10"])
        )
        scored.append((score, name))
    scored.sort()
    result["ranking"] = [
        {"rank": i + 1, "checkpoint": name, "score": float(score)}
        for i, (score, name) in enumerate(scored)
    ]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "checkpoint_home_semantics_summary.json"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_path = output_dir / "checkpoint_home_semantics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "checkpoint",
                "offset_frames",
                "motor",
                "act_abs",
                "demo_median_abs",
                "act_delta_from_action0",
                "demo_median_delta",
            ]
        )
        writer.writerows(csv_rows)

    print()
    print("===== RANKING =====")
    for r in result["ranking"]:
        print(
            f"#{r['rank']} {r['checkpoint']} "
            f"score={r['score']:.4f}"
        )

    print()
    print("===== OUTPUT =====")
    print(json_path)
    print(csv_path)
    print("OFFLINE CHECKPOINT HOME-SEMANTICS COMPARISON: PASS")
    print("NO HARDWARE WAS ACCESSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
