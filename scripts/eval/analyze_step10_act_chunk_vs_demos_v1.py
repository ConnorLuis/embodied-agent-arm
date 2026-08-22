#!/usr/bin/env python
"""
Offline Step-10 ACT chunk vs nearest-demonstration continuation analysis.

Inputs
------
1) Live short-rollout directory containing:
     report.json
     front_step_10.jpg
     wrist_step_10.jpg
2) nearest_neighbors_summary.json from compare_live_step10_to_training_v1.py
3) local ACT checkpoint

Outputs
-------
- Re-run ACT predict_action_chunk() on the saved Step-10 observation.
- Compare ACT chunk actions at t,t+1,t+5,t+10,t+20 against the
  nearest demonstration continuations.
- Compare trajectory DELTAS relative to t, which is more informative than
  absolute command offsets when the live state has already drifted from the
  demonstration manifold.
- Save JSON + CSV summaries.

OFFLINE / READ-ONLY:
- no serial port
- no cameras
- no motor writes
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lerobot.policies.act.modeling_act import ACTPolicy


MOTORS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

OFFSETS = [0, 1, 5, 10, 20]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--query-run-dir",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--neighbors-json",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "outputs/train/act_red_cube_v1/checkpoints/015000/pretrained_model"
        ),
    )
    p.add_argument("--query-step", type=int, default=10)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/eval/step10_chunk_vs_demos_v1"),
    )

    args = p.parse_args()
    if not 3 <= args.top_k <= 8:
        p.error("--top-k must be 3..8")
    return args


def load_rgb_tensor(path: Path, device: torch.device) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(path)

    rgb = np.asarray(Image.open(path).convert("RGB"))
    if rgb.shape != (480, 480, 3):
        raise RuntimeError(f"{path}: expected 480x480 RGB, got {rgb.shape}")

    x = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0
    return x.unsqueeze(0).to(device)


def fmt(v: np.ndarray, digits: int = 3) -> str:
    return "[" + ", ".join(f"{float(x):.{digits}f}" for x in v) + "]"


def percentile_rows(a: np.ndarray, q: float) -> np.ndarray:
    return np.percentile(a, q, axis=0)


def get_future_action(match: dict, offset: int) -> np.ndarray:
    rec = next(
        (
            f
            for f in match["future"]
            if int(f["offset_frames"]) == offset
        ),
        None,
    )
    if rec is None:
        raise RuntimeError(
            f"Missing future offset={offset} for "
            f"rank={match.get('rank')}"
        )
    return np.asarray(rec["action"], dtype=np.float32)


def main() -> int:
    args = parse_args()

    query_dir = args.query_run_dir.resolve()
    neighbors_path = args.neighbors_json.resolve()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    live_report_path = query_dir / "report.json"
    front_path = query_dir / f"front_step_{args.query_step:02d}.jpg"
    wrist_path = query_dir / f"wrist_step_{args.query_step:02d}.jpg"

    live_report = json.loads(
        live_report_path.read_text(encoding="utf-8")
    )
    nn = json.loads(neighbors_path.read_text(encoding="utf-8"))

    cycle = next(
        (
            c
            for c in live_report["cycles"]
            if int(c["step"]) == args.query_step
        ),
        None,
    )
    if cycle is None:
        raise RuntimeError(f"Step {args.query_step} missing from report.")

    query_state = np.asarray(
        cycle["actual_before"],
        dtype=np.float32,
    )
    live_recorded_first = np.asarray(
        cycle["raw_first_action"],
        dtype=np.float32,
    )

    matches = nn["matches"][: args.top_k]
    if len(matches) < args.top_k:
        raise RuntimeError(
            f"Only {len(matches)} nearest matches available."
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")

    print("===== LOAD ACT =====")
    policy = ACTPolicy.from_pretrained(
        checkpoint,
        local_files_only=True,
    )
    policy.eval()
    policy.reset()
    device = torch.device(policy.config.device)

    print(f"checkpoint: {checkpoint}")
    print(f"device: {device}")
    print(f"chunk_size: {policy.config.chunk_size}")

    state_t = torch.from_numpy(query_state).unsqueeze(0).to(device)
    batch = {
        "observation.state": state_t,
        "observation.images.front": load_rgb_tensor(front_path, device),
        "observation.images.wrist": load_rgb_tensor(wrist_path, device),
    }

    # Warm once with the actual batch, discard output, reset, then run
    # the measured/reported prediction.
    with torch.inference_mode():
        _ = policy.predict_action_chunk(batch)
    torch.cuda.synchronize()
    policy.reset()

    with torch.inference_mode():
        chunk = policy.predict_action_chunk(batch)
    torch.cuda.synchronize()

    chunk_np = chunk.detach().cpu().float().numpy()
    if chunk_np.shape[0] != 1 or chunk_np.shape[2] != 6:
        raise RuntimeError(f"Unexpected ACT chunk shape: {chunk_np.shape}")

    actions = chunk_np[0]

    print()
    print("===== REPRODUCTION CHECK =====")
    print(f"query state:          {fmt(query_state)}")
    print(f"recorded live first:  {fmt(live_recorded_first)}")
    print(f"offline saved-JPEG:   {fmt(actions[0])}")

    reproduce_abs = np.abs(actions[0] - live_recorded_first)
    reproduce_mae = float(reproduce_abs.mean())
    reproduce_max = float(reproduce_abs.max())

    print(f"first-action MAE: {reproduce_mae:.4f}")
    print(f"first-action MAX: {reproduce_max:.4f}")
    if reproduce_mae > 1.0:
        print(
            "WARNING: saved-JPEG offline prediction differs materially "
            "from the live in-memory prediction. Interpret trajectory "
            "comparison cautiously."
        )

    # Demo action tensors:
    # demo_abs[match_i, offset_i, motor]
    demo_abs = np.stack(
        [
            np.stack(
                [get_future_action(m, off) for off in OFFSETS],
                axis=0,
            )
            for m in matches
        ],
        axis=0,
    )

    demo_delta = demo_abs - demo_abs[:, 0:1, :]

    act_abs = np.stack(
        [actions[off] for off in OFFSETS],
        axis=0,
    )
    act_delta = act_abs - act_abs[0:1, :]

    demo_median_abs = np.median(demo_abs, axis=0)
    demo_median_delta = np.median(demo_delta, axis=0)
    demo_q25_delta = percentile_rows(
        demo_delta.transpose(1, 0, 2).reshape(len(OFFSETS), -1, 6),
        25,
    )
    demo_q75_delta = percentile_rows(
        demo_delta.transpose(1, 0, 2).reshape(len(OFFSETS), -1, 6),
        75,
    )

    print()
    print("===== ACT CHUNK vs TOP DEMONSTRATIONS =====")
    for oi, off in enumerate(OFFSETS):
        print()
        print(f"--- offset +{off} frames ({off/15.0:.3f}s demo time) ---")
        print(f"ACT abs:          {fmt(act_abs[oi])}")
        print(f"demo median abs:  {fmt(demo_median_abs[oi])}")
        print(f"ACT delta from 0: {fmt(act_delta[oi])}")
        print(f"demo med delta:   {fmt(demo_median_delta[oi])}")

    # Direction consensus on demonstration trajectory deltas.
    consensus = {}
    for oi, off in enumerate(OFFSETS[1:], start=1):
        rec = {}
        for j, motor in enumerate(MOTORS):
            vals = demo_delta[:, oi, j]
            pos = int(np.sum(vals > 0.05))
            neg = int(np.sum(vals < -0.05))
            zero = int(len(vals) - pos - neg)

            act_v = float(act_delta[oi, j])
            if act_v > 0.05:
                act_sign = "+"
            elif act_v < -0.05:
                act_sign = "-"
            else:
                act_sign = "0"

            majority = (
                "+"
                if pos > max(neg, zero)
                else "-"
                if neg > max(pos, zero)
                else "0"
                if zero > max(pos, neg)
                else "mixed"
            )

            rec[motor] = {
                "demo_positive": pos,
                "demo_negative": neg,
                "demo_near_zero": zero,
                "demo_majority": majority,
                "act_delta": act_v,
                "act_sign": act_sign,
                "act_matches_demo_majority": (
                    majority == act_sign if majority != "mixed" else None
                ),
            }
        consensus[str(off)] = rec

    print()
    print("===== DIRECTION CONSENSUS =====")
    for off in OFFSETS[1:]:
        print(f"offset +{off}:")
        pieces = []
        for motor in MOTORS:
            r = consensus[str(off)][motor]
            pieces.append(
                f"{motor}:demo={r['demo_majority']} "
                f"ACT={r['act_sign']} "
                f"match={r['act_matches_demo_majority']}"
            )
        print("  " + " | ".join(pieces))

    # Per-offset trajectory-shape error to demonstration median.
    trajectory_mae = np.mean(
        np.abs(act_delta - demo_median_delta),
        axis=1,
    )

    result = {
        "schema_version": "step10_act_chunk_vs_demos_v1",
        "query_run_dir": str(query_dir),
        "query_step": args.query_step,
        "checkpoint": str(checkpoint),
        "top_k": args.top_k,
        "offsets": OFFSETS,
        "query_state": query_state.tolist(),
        "live_recorded_first_action": live_recorded_first.tolist(),
        "offline_reproduced_first_action": actions[0].tolist(),
        "offline_reproduction_first_action_mae": reproduce_mae,
        "offline_reproduction_first_action_max_abs": reproduce_max,
        "act_chunk_actions": {
            str(off): act_abs[i].tolist()
            for i, off in enumerate(OFFSETS)
        },
        "act_chunk_delta_from_t": {
            str(off): act_delta[i].tolist()
            for i, off in enumerate(OFFSETS)
        },
        "demo_median_actions": {
            str(off): demo_median_abs[i].tolist()
            for i, off in enumerate(OFFSETS)
        },
        "demo_median_delta_from_t": {
            str(off): demo_median_delta[i].tolist()
            for i, off in enumerate(OFFSETS)
        },
        "demo_delta_q25": {
            str(off): demo_q25_delta[i].tolist()
            for i, off in enumerate(OFFSETS)
        },
        "demo_delta_q75": {
            str(off): demo_q75_delta[i].tolist()
            for i, off in enumerate(OFFSETS)
        },
        "trajectory_delta_mae_vs_demo_median": {
            str(off): float(trajectory_mae[i])
            for i, off in enumerate(OFFSETS)
        },
        "direction_consensus": consensus,
    }

    json_path = output_dir / "step10_chunk_vs_demos_summary.json"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_path = output_dir / "step10_chunk_vs_demos.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "offset_frames",
                "motor",
                "act_abs",
                "demo_median_abs",
                "act_delta_from_t",
                "demo_median_delta_from_t",
                "demo_q25_delta",
                "demo_q75_delta",
            ]
        )
        for oi, off in enumerate(OFFSETS):
            for j, motor in enumerate(MOTORS):
                writer.writerow(
                    [
                        off,
                        motor,
                        float(act_abs[oi, j]),
                        float(demo_median_abs[oi, j]),
                        float(act_delta[oi, j]),
                        float(demo_median_delta[oi, j]),
                        float(demo_q25_delta[oi, j]),
                        float(demo_q75_delta[oi, j]),
                    ]
                )

    print()
    print("===== OUTPUT =====")
    print(json_path)
    print(csv_path)
    print("OFFLINE ACT-CHUNK / DEMO CONTINUATION ANALYSIS: PASS")
    print("NO HARDWARE WAS ACCESSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
