#!/usr/bin/env python
"""
Compare a live ACT rollout step with visually/state-similar training frames.

Designed for:
  dataset:
    data/lerobot/so101_red_cube_pick_place_v1
  live rollout:
    outputs/eval/act_short_task_rollout_v1/<timestamp>

Method
------
1. Load the local LeRobotDataset.
2. Use the live rollout's `actual_before` at --query-step as the query state.
3. Search all dataset states cheaply and keep a state-nearest candidate pool.
4. Decode Front/Wrist images only for that candidate pool.
5. Re-rank candidates with:
      - standardized robot-state distance
      - Front appearance distance
      - Wrist appearance distance
      - red-object mask distance
6. Enforce episode/frame diversity so the top list is not just adjacent frames.
7. For each top match, inspect demonstration actions at:
      t, t+1, t+5, t+10, t+20
   and save visual evidence.
8. Write a JSON report and contact sheets.

This script is OFFLINE / READ-ONLY:
- no serial ports
- no camera devices
- no motor writes
- no calibration writes
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

FRONT_KEY = "observation.images.front"
WRIST_KEY = "observation.images.wrist"
STATE_KEY = "observation.state"
ACTION_KEY = "action"


@dataclass
class Candidate:
    global_index: int
    episode_index: int
    frame_index: int
    state_distance: float
    front_distance: float
    wrist_distance: float
    red_distance: float
    visual_distance: float
    final_score: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

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
        "--query-run-dir",
        type=Path,
        required=True,
        help=(
            "Live rollout directory containing report.json, "
            "front_step_XX.jpg and wrist_step_XX.jpg"
        ),
    )
    p.add_argument("--query-step", type=int, default=10)
    p.add_argument("--candidate-pool", type=int, default=240)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--max-per-episode", type=int, default=2)
    p.add_argument("--min-frame-spacing", type=int, default=15)
    p.add_argument(
        "--future-offsets",
        default="0,1,5,10,20",
        help="Comma-separated frame offsets.",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/eval/training_nn_step10_v1"),
    )

    args = p.parse_args()

    if args.query_step < 1:
        p.error("--query-step must be >= 1")
    if not 40 <= args.candidate_pool <= 1000:
        p.error("--candidate-pool must be 40..1000")
    if not 3 <= args.top_k <= 20:
        p.error("--top-k must be 3..20")
    if not 1 <= args.max_per_episode <= 4:
        p.error("--max-per-episode must be 1..4")
    if not 0 <= args.min_frame_spacing <= 120:
        p.error("--min-frame-spacing must be 0..120")

    try:
        offsets = sorted(
            set(int(x.strip()) for x in args.future_offsets.split(","))
        )
    except ValueError as exc:
        p.error(f"Invalid --future-offsets: {exc}")

    if not offsets or offsets[0] < 0 or offsets[-1] > 120:
        p.error("future offsets must be within 0..120 frames")
    if 0 not in offsets:
        offsets.insert(0, 0)

    args.future_offsets_parsed = offsets
    return args


def load_lerobot_dataset(repo_id: str, root: Path) -> LeRobotDataset:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    # LeRobot 0.3.x accepts repo_id + root. Keep the call minimal so local
    # version differences do not introduce optional-argument failures.
    try:
        ds = LeRobotDataset(
            repo_id=repo_id,
            root=root,
        )
    except TypeError:
        ds = LeRobotDataset(
            repo_id,
            root=root,
        )
    return ds


def tensor_image_to_rgb_uint8(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu()
        if t.ndim != 3:
            raise RuntimeError(f"Unexpected image tensor shape: {tuple(t.shape)}")

        # LeRobot returns camera images as C,H,W float tensors.
        if t.shape[0] in (1, 3, 4):
            t = t[:3].permute(1, 2, 0)
        elif t.shape[-1] in (1, 3, 4):
            t = t[..., :3]
        else:
            raise RuntimeError(
                f"Cannot infer channel dimension for {tuple(t.shape)}"
            )

        arr = t.numpy()
        if np.issubdtype(arr.dtype, np.floating):
            # Expected [0,1], but make conversion robust.
            vmax = float(np.nanmax(arr))
            if vmax <= 1.5:
                arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    arr = np.asarray(x)
    if arr.ndim != 3:
        raise RuntimeError(f"Unexpected image array shape: {arr.shape}")
    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr[:3], (1, 2, 0))
    if np.issubdtype(arr.dtype, np.floating):
        vmax = float(np.nanmax(arr))
        if vmax <= 1.5:
            arr = arr * 255.0
    return np.clip(arr[..., :3], 0, 255).astype(np.uint8)


def load_query_rgb(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)
    if arr.shape[:2] != (480, 480):
        raise RuntimeError(
            f"Query image must be 480x480, got {arr.shape} at {path}"
        )
    return arr


def appearance_feature(rgb: np.ndarray, size: int = 64) -> np.ndarray:
    # Fixed cameras/background make low-resolution appearance matching
    # surprisingly strong. LAB reduces sensitivity to small RGB exposure shifts.
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    small = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Normalize luminance per-image to reduce exposure drift while preserving
    # chroma and geometry.
    L = lab[..., 0]
    L = (L - L.mean()) / max(float(L.std()), 8.0)
    A = (lab[..., 1] - 128.0) / 64.0
    B = (lab[..., 2] - 128.0) / 64.0

    feat = np.stack([L, A, B], axis=-1)
    return feat


def red_mask(rgb: np.ndarray, size: int = 96) -> np.ndarray:
    small = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)

    # Red wraps around hue=0 in HSV.
    lower1 = np.array([0, 80, 55], dtype=np.uint8)
    upper1 = np.array([12, 255, 255], dtype=np.uint8)
    lower2 = np.array([168, 80, 55], dtype=np.uint8)
    upper2 = np.array([179, 255, 255], dtype=np.uint8)

    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    mask = ((m1 > 0) | (m2 > 0)).astype(np.float32)

    # Slight blur makes a 1–2 pixel shift less punitive.
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return mask


def mean_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def robust_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    med = float(np.median(values))
    q75 = float(np.percentile(values, 75))
    q25 = float(np.percentile(values, 25))
    iqr = max(q75 - q25, 1e-8)
    return (values - med) / iqr


def get_column(hf_dataset, key: str, dtype=None) -> np.ndarray:
    values = hf_dataset[key]
    arr = np.asarray(values, dtype=dtype)
    return arr


def select_diverse(
    ranked: list[Candidate],
    top_k: int,
    max_per_episode: int,
    min_frame_spacing: int,
) -> list[Candidate]:
    selected: list[Candidate] = []
    episode_counts: dict[int, int] = {}

    for cand in ranked:
        if episode_counts.get(cand.episode_index, 0) >= max_per_episode:
            continue

        too_close = any(
            s.episode_index == cand.episode_index
            and abs(s.frame_index - cand.frame_index) < min_frame_spacing
            for s in selected
        )
        if too_close:
            continue

        selected.append(cand)
        episode_counts[cand.episode_index] = (
            episode_counts.get(cand.episode_index, 0) + 1
        )

        if len(selected) >= top_k:
            break

    return selected


def safe_frame_global_index(
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


def annotate(rgb: np.ndarray, lines: list[str]) -> Image.Image:
    img = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    pad = 6
    line_h = 13
    box_h = pad * 2 + line_h * len(lines)
    overlay = Image.new("RGBA", (img.width, box_h), (0, 0, 0, 190))
    img_rgba = img.convert("RGBA")
    img_rgba.alpha_composite(overlay, (0, 0))
    draw = ImageDraw.Draw(img_rgba)

    y = pad
    for line in lines:
        draw.text((pad, y), line, fill=(255, 255, 255, 255), font=font)
        y += line_h

    return img_rgba.convert("RGB")


def make_grid(
    images: list[Image.Image],
    columns: int,
    output: Path,
) -> None:
    if not images:
        return
    w = max(img.width for img in images)
    h = max(img.height for img in images)
    rows = math.ceil(len(images) / columns)
    canvas = Image.new("RGB", (columns * w, rows * h), "black")
    for i, img in enumerate(images):
        x = (i % columns) * w
        y = (i // columns) * h
        canvas.paste(img, (x, y))
    canvas.save(output, quality=92)


def main() -> int:
    args = parse_args()

    dataset_root = args.dataset_root.resolve()
    query_run_dir = args.query_run_dir.resolve()

    report_path = query_run_dir / "report.json"
    front_query_path = (
        query_run_dir / f"front_step_{args.query_step:02d}.jpg"
    )
    wrist_query_path = (
        query_run_dir / f"wrist_step_{args.query_step:02d}.jpg"
    )

    if not report_path.is_file():
        raise FileNotFoundError(report_path)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    cycles = report.get("cycles", [])
    query_cycle = next(
        (c for c in cycles if int(c.get("step", -1)) == args.query_step),
        None,
    )
    if query_cycle is None:
        raise RuntimeError(
            f"Step {args.query_step} not found in {report_path}"
        )

    query_state = np.asarray(
        query_cycle["actual_before"],
        dtype=np.float32,
    )
    live_raw_action = np.asarray(
        query_cycle["raw_first_action"],
        dtype=np.float32,
    )
    if query_state.shape != (6,) or live_raw_action.shape != (6,):
        raise RuntimeError("Live report state/action shape mismatch.")

    query_front = load_query_rgb(front_query_path)
    query_wrist = load_query_rgb(wrist_query_path)

    print("===== QUERY =====")
    print(f"run dir: {query_run_dir}")
    print(f"step: {args.query_step}")
    print(
        "state: "
        + np.array2string(query_state, precision=3, separator=", ")
    )
    print(
        "live ACT raw first: "
        + np.array2string(live_raw_action, precision=3, separator=", ")
    )

    print()
    print("===== LOAD LOCAL LEROBOT DATASET =====")
    ds = load_lerobot_dataset(args.repo_id, dataset_root)
    hf = ds.hf_dataset

    states = get_column(hf, STATE_KEY, dtype=np.float32)
    actions = get_column(hf, ACTION_KEY, dtype=np.float32)
    episode_indices = get_column(hf, "episode_index", dtype=np.int64)
    frame_indices = get_column(hf, "frame_index", dtype=np.int64)

    if states.ndim != 2 or states.shape[1] != 6:
        raise RuntimeError(f"State matrix invalid: {states.shape}")
    if actions.shape != states.shape:
        raise RuntimeError(
            f"Action matrix mismatch: states={states.shape}, actions={actions.shape}"
        )

    print(f"frames: {len(states)}")
    print(f"episodes: {len(np.unique(episode_indices))}")
    print(f"state shape: {states.shape}")

    # ------------------------------------------------------------------
    # Stage 1: cheap state-nearest prefilter across all 54k frames.
    # ------------------------------------------------------------------
    state_std = states.std(axis=0)
    state_std = np.maximum(state_std, 1.0)

    state_z = (states - query_state[None, :]) / state_std[None, :]
    state_dist = np.sqrt(np.mean(state_z * state_z, axis=1))

    pool_n = min(args.candidate_pool, len(states))
    pool_indices = np.argpartition(state_dist, pool_n - 1)[:pool_n]
    pool_indices = pool_indices[
        np.argsort(state_dist[pool_indices])
    ]

    print()
    print("===== STATE PREFILTER =====")
    print(f"candidate pool: {len(pool_indices)}")
    print(
        "state std: "
        + np.array2string(state_std, precision=3, separator=", ")
    )
    print(
        f"best state distance: {state_dist[pool_indices[0]]:.5f}"
    )

    # ------------------------------------------------------------------
    # Stage 2: decode only candidate images and compute visual distances.
    # ------------------------------------------------------------------
    q_front_feat = appearance_feature(query_front)
    q_wrist_feat = appearance_feature(query_wrist)
    q_front_red = red_mask(query_front)
    q_wrist_red = red_mask(query_wrist)

    raw_candidates = []
    decoded = 0

    print()
    print("===== VISUAL RE-RANK =====")

    for n, gi in enumerate(pool_indices, start=1):
        item = ds[int(gi)]

        cand_front = tensor_image_to_rgb_uint8(item[FRONT_KEY])
        cand_wrist = tensor_image_to_rgb_uint8(item[WRIST_KEY])

        f_dist = mean_abs(
            appearance_feature(cand_front),
            q_front_feat,
        )
        w_dist = mean_abs(
            appearance_feature(cand_wrist),
            q_wrist_feat,
        )

        # Red mask is especially useful for object-relative geometry.
        f_red = mean_abs(red_mask(cand_front), q_front_red)
        w_red = mean_abs(red_mask(cand_wrist), q_wrist_red)
        r_dist = 0.30 * f_red + 0.70 * w_red

        # Wrist is weighted slightly more because the current question is
        # whether the end effector is in the same local grasp phase.
        visual = (
            0.35 * f_dist
            + 0.55 * w_dist
            + 0.10 * r_dist
        )

        raw_candidates.append(
            {
                "global_index": int(gi),
                "episode_index": int(episode_indices[gi]),
                "frame_index": int(frame_indices[gi]),
                "state_distance": float(state_dist[gi]),
                "front_distance": float(f_dist),
                "wrist_distance": float(w_dist),
                "red_distance": float(r_dist),
                "visual_distance": float(visual),
            }
        )
        decoded += 1

        if n % 40 == 0 or n == len(pool_indices):
            print(
                f"decoded {n}/{len(pool_indices)} candidate frame pairs"
            )

    # Normalize within prefiltered pool before combining state+vision.
    state_vals = np.asarray(
        [x["state_distance"] for x in raw_candidates],
        dtype=np.float64,
    )
    visual_vals = np.asarray(
        [x["visual_distance"] for x in raw_candidates],
        dtype=np.float64,
    )

    state_scaled = robust_scale(state_vals)
    visual_scaled = robust_scale(visual_vals)

    candidates: list[Candidate] = []
    for i, x in enumerate(raw_candidates):
        # Visual match dominates final order; state was already used as a
        # hard prefilter across the entire dataset.
        score = 0.35 * state_scaled[i] + 0.65 * visual_scaled[i]
        candidates.append(
            Candidate(
                global_index=x["global_index"],
                episode_index=x["episode_index"],
                frame_index=x["frame_index"],
                state_distance=x["state_distance"],
                front_distance=x["front_distance"],
                wrist_distance=x["wrist_distance"],
                red_distance=x["red_distance"],
                visual_distance=x["visual_distance"],
                final_score=float(score),
            )
        )

    ranked = sorted(candidates, key=lambda c: c.final_score)
    selected = select_diverse(
        ranked,
        top_k=args.top_k,
        max_per_episode=args.max_per_episode,
        min_frame_spacing=args.min_frame_spacing,
    )

    if len(selected) < args.top_k:
        print(
            f"WARNING: diversity filter returned only "
            f"{len(selected)}/{args.top_k} matches."
        )

    output_dir = (
        args.output_root.resolve()
        / f"step_{args.query_step:02d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("===== TOP MATCHES =====")

    result = {
        "schema_version": "training_nn_step10_v1",
        "dataset_root": str(dataset_root),
        "repo_id": args.repo_id,
        "query_run_dir": str(query_run_dir),
        "query_step": args.query_step,
        "query_state": query_state.tolist(),
        "live_act_raw_first": live_raw_action.tolist(),
        "candidate_pool": len(pool_indices),
        "ranking": {
            "state_prefilter": "standardized Euclidean over 6D observation.state",
            "visual": (
                "0.35*front appearance + 0.55*wrist appearance "
                "+ 0.10*red-mask distance"
            ),
            "final": "0.35*robust_state + 0.65*robust_visual",
        },
        "future_offsets": args.future_offsets_parsed,
        "matches": [],
    }

    query_pair = [
        annotate(
            query_front,
            [
                f"QUERY front step={args.query_step}",
                "live rollout",
            ],
        ),
        annotate(
            query_wrist,
            [
                f"QUERY wrist step={args.query_step}",
                "live rollout",
            ],
        ),
    ]

    contact_images = query_pair.copy()
    trajectory_images: list[Image.Image] = []

    for rank, cand in enumerate(selected, start=1):
        gi = cand.global_index
        ep = cand.episode_index
        frame = cand.frame_index

        current_demo_action = actions[gi]
        action_mae_vs_live = float(
            np.mean(np.abs(current_demo_action - live_raw_action))
        )

        match_record = {
            "rank": rank,
            "global_index": gi,
            "episode_index": ep,
            "frame_index": frame,
            "time_seconds": float(frame / 15.0),
            "state_distance": cand.state_distance,
            "front_distance": cand.front_distance,
            "wrist_distance": cand.wrist_distance,
            "red_distance": cand.red_distance,
            "visual_distance": cand.visual_distance,
            "final_score": cand.final_score,
            "state": states[gi].tolist(),
            "demo_action_t": current_demo_action.tolist(),
            "demo_action_t_mae_vs_live_act_raw": action_mae_vs_live,
            "future": [],
        }

        print(
            f"#{rank:02d} ep={ep:02d} frame={frame:03d} "
            f"t={frame/15.0:6.2f}s "
            f"score={cand.final_score:8.4f} "
            f"state={cand.state_distance:6.3f} "
            f"visual={cand.visual_distance:7.4f} "
            f"demo-vs-live-MAE={action_mae_vs_live:6.3f}"
        )

        item = ds[gi]
        front = tensor_image_to_rgb_uint8(item[FRONT_KEY])
        wrist = tensor_image_to_rgb_uint8(item[WRIST_KEY])

        contact_images.extend(
            [
                annotate(
                    front,
                    [
                        f"rank {rank} FRONT",
                        f"ep={ep} frame={frame}",
                        f"score={cand.final_score:.3f}",
                    ],
                ),
                annotate(
                    wrist,
                    [
                        f"rank {rank} WRIST",
                        f"ep={ep} frame={frame}",
                        f"score={cand.final_score:.3f}",
                    ],
                ),
            ]
        )

        # Save future action/image evidence for top matches.
        match_dir = (
            output_dir
            / f"rank_{rank:02d}_ep_{ep:02d}_frame_{frame:03d}"
        )
        match_dir.mkdir(parents=True, exist_ok=True)

        for off in args.future_offsets_parsed:
            future_frame = frame + off
            fgi = safe_frame_global_index(
                episode_indices,
                frame_indices,
                ep,
                future_frame,
            )
            if fgi is None:
                continue

            demo_action = actions[fgi]
            demo_state = states[fgi]

            future_rec = {
                "offset_frames": off,
                "offset_seconds": float(off / 15.0),
                "global_index": int(fgi),
                "frame_index": int(future_frame),
                "state": demo_state.tolist(),
                "action": demo_action.tolist(),
                "action_delta_from_match_t": (
                    demo_action - current_demo_action
                ).tolist(),
            }
            match_record["future"].append(future_rec)

            # Visualize future trajectory only for the first 5 nearest matches.
            if rank <= min(5, args.top_k):
                future_item = ds[int(fgi)]
                ffront = tensor_image_to_rgb_uint8(
                    future_item[FRONT_KEY]
                )
                fwrist = tensor_image_to_rgb_uint8(
                    future_item[WRIST_KEY]
                )

                Image.fromarray(ffront).save(
                    match_dir / f"front_t_plus_{off:03d}.jpg",
                    quality=92,
                )
                Image.fromarray(fwrist).save(
                    match_dir / f"wrist_t_plus_{off:03d}.jpg",
                    quality=92,
                )

                trajectory_images.extend(
                    [
                        annotate(
                            ffront,
                            [
                                f"rank {rank} front",
                                f"ep={ep} f={future_frame}",
                                f"t+{off} ({off/15.0:.2f}s)",
                            ],
                        ),
                        annotate(
                            fwrist,
                            [
                                f"rank {rank} wrist",
                                f"ep={ep} f={future_frame}",
                                f"t+{off} ({off/15.0:.2f}s)",
                            ],
                        ),
                    ]
                )

        result["matches"].append(match_record)

    result_path = output_dir / "nearest_neighbors_summary.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Contact sheet: query pair + each selected match pair.
    make_grid(
        contact_images,
        columns=2,
        output=output_dir / "nearest_neighbors_contact_sheet.jpg",
    )

    # Trajectory sheet: for top 5 matches, each offset contributes front+wrist.
    make_grid(
        trajectory_images,
        columns=2,
        output=output_dir / "top5_future_trajectories.jpg",
    )

    print()
    print("===== OUTPUT =====")
    print(f"dir: {output_dir}")
    print(f"summary: {result_path}")
    print(
        "contact sheet: "
        f"{output_dir / 'nearest_neighbors_contact_sheet.jpg'}"
    )
    print(
        "future trajectories: "
        f"{output_dir / 'top5_future_trajectories.jpg'}"
    )
    print()
    print("OFFLINE TRAINING-ALIGNMENT SEARCH: PASS")
    print("NO HARDWARE WAS ACCESSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
