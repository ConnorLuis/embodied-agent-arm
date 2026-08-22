#!/usr/bin/env python
"""
Offline ACT CVAE prior-vs-posterior audit (15k).

Question
--------
The Home/startup audit found that normal ACT inference has a systematic
absolute action[0] bias. Is that mainly because:

A) the trained CVAE can reconstruct the correct demonstration when its VAE
   encoder is given the ground-truth action chunk, but normal inference uses
   the zero latent prior; or

B) even the training/posterior path cannot reconstruct the target well, so
   the issue is broader model fit / representation / target semantics?

Method
------
For the same 60 Home-like training observations used by the previous audit:

1. Build the exact 50-step ground-truth action chunk.
2. Call ACTPolicy.forward(...) in TRAINING mode so the VAE encoder sees the
   ground-truth action sequence.
3. Force the reparameterization noise epsilon to zero, so latent_sample = mu
   (deterministic posterior-mean diagnostic).
4. Disable dropout in all child modules while keeping the ACT root module's
   `training=True`, so only the VAE training branch remains active.
5. Capture the exact batch that reaches policy.model. This automatically
   preserves whatever normalization/preprocessing the installed LeRobot
   version uses.
6. On that exact model-space batch:
     - posterior_mu prediction: training VAE path
     - prior_zero prediction: normal inference path (model.eval -> z=0)
7. Compare both against the exact model-space target.

All reported reconstruction errors are MODEL-SPACE errors. This is deliberate:
it avoids making assumptions about whether this pinned LeRobot version keeps
normalization inside ACTPolicy or in external processors.

OFFLINE / READ-ONLY:
- no serial ports
- no cameras
- no motor writes
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy


STATE_KEY = "observation.state"
ACTION_KEY = "action"
FRONT_KEY = "observation.images.front"
WRIST_KEY = "observation.images.wrist"


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
        "--checkpoint",
        type=Path,
        default=Path(
            "outputs/train/act_red_cube_v1/checkpoints/015000/"
            "pretrained_model"
        ),
    )
    p.add_argument("--max-episodes", type=int, default=60)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/eval/act_cvae_prior_posterior_v1"),
    )
    args = p.parse_args()
    if not 5 <= args.max_episodes <= 60:
        p.error("--max-episodes must be 5..60")
    return args


def load_dataset(repo_id: str, root: Path) -> LeRobotDataset:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    try:
        return LeRobotDataset(repo_id=repo_id, root=root)
    except TypeError:
        return LeRobotDataset(repo_id, root=root)


def to_batched_tensor(x: Any, device: torch.device) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    return x.unsqueeze(0).to(device)


@contextlib.contextmanager
def zero_randn_like():
    """Temporarily force VAE epsilon=0 so latent_sample == posterior mu."""
    original = torch.randn_like

    def _zeros_like(x, *args, **kwargs):
        return torch.zeros_like(x)

    torch.randn_like = _zeros_like
    try:
        yield
    finally:
        torch.randn_like = original


def clone_model_batch(batch: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().clone()
        elif isinstance(v, list):
            out[k] = [
                x.detach().clone() if isinstance(x, torch.Tensor) else x
                for x in v
            ]
        else:
            out[k] = v
    return out


def summarize(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "q25": float(np.percentile(x, 25)),
        "q75": float(np.percentile(x, 75)),
        "p90": float(np.percentile(x, 90)),
        "max": float(np.max(x)),
    }


def main() -> int:
    args = parse_args()

    bias_path = args.bias_json.resolve()
    if not bias_path.is_file():
        raise FileNotFoundError(bias_path)

    bias = json.loads(bias_path.read_text(encoding="utf-8"))
    anchor_records = bias["checkpoints"]["15k"]["records"][: args.max_episodes]

    print("===== LOAD DATASET =====")
    ds = load_dataset(args.repo_id, args.dataset_root)
    hf = ds.hf_dataset

    actions = np.asarray(hf[ACTION_KEY], dtype=np.float32)
    episodes = np.asarray(hf["episode_index"], dtype=np.int64)
    frames = np.asarray(hf["frame_index"], dtype=np.int64)

    print(f"frames: {len(actions)}")
    print(f"episodes used: {len(anchor_records)}")

    print()
    print("===== LOAD 15k ACT =====")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)

    policy = ACTPolicy.from_pretrained(
        checkpoint,
        local_files_only=True,
    )

    if not policy.config.use_vae:
        raise RuntimeError("Checkpoint use_vae=False; CVAE audit is not applicable.")
    if policy.config.chunk_size != 50:
        raise RuntimeError(
            f"Expected chunk_size=50, got {policy.config.chunk_size}"
        )

    device = torch.device(policy.config.device)
    print(f"device: {device}")
    print(f"chunk_size: {policy.config.chunk_size}")
    print(f"latent_dim: {policy.config.latent_dim}")
    print(f"kl_weight: {policy.config.kl_weight}")

    # Map (episode, frame) -> global dataset index.
    index_map = {
        (int(ep), int(fr)): int(i)
        for i, (ep, fr) in enumerate(zip(episodes, frames))
    }

    records = []
    prior_action0_mae = []
    posterior_action0_mae = []
    prior_chunk_mae = []
    posterior_chunk_mae = []
    prior_first10_mae = []
    posterior_first10_mae = []
    mu_norms = []
    kld_values = []

    print()
    print("===== PRIOR-ZERO vs POSTERIOR-MU =====")

    for n, rec in enumerate(anchor_records, start=1):
        ep = int(rec["episode_index"])
        frame = int(rec["frame_index"])

        gi = index_map.get((ep, frame))
        if gi is None:
            raise RuntimeError(f"Missing ep={ep} frame={frame}")

        # Exact 50-step action chunk, same episode only.
        future_indices = []
        for off in range(policy.config.chunk_size):
            idx = index_map.get((ep, frame + off))
            if idx is None:
                raise RuntimeError(
                    f"ep={ep} frame={frame}: missing future offset {off}"
                )
            future_indices.append(idx)

        item = ds[gi]
        action_chunk = torch.from_numpy(
            actions[np.asarray(future_indices)]
        ).float()

        batch = {
            STATE_KEY: to_batched_tensor(item[STATE_KEY], device),
            FRONT_KEY: to_batched_tensor(item[FRONT_KEY], device),
            WRIST_KEY: to_batched_tensor(item[WRIST_KEY], device),
            ACTION_KEY: action_chunk.unsqueeze(0).to(device),
            "action_is_pad": torch.zeros(
                (1, policy.config.chunk_size),
                dtype=torch.bool,
                device=device,
            ),
        }

        captured: dict[str, Any] = {}

        def pre_hook(module, args_tuple):
            if len(args_tuple) != 1 or not isinstance(args_tuple[0], dict):
                raise RuntimeError("Unexpected ACT model input signature.")
            captured["model_batch"] = clone_model_batch(args_tuple[0])

        def out_hook(module, args_tuple, output):
            actions_hat, latent = output
            captured["posterior_actions"] = actions_hat.detach().clone()
            mu, log_sigma_x2 = latent
            captured["mu"] = None if mu is None else mu.detach().clone()
            captured["logvar"] = (
                None if log_sigma_x2 is None else log_sigma_x2.detach().clone()
            )

        h1 = policy.model.register_forward_pre_hook(pre_hook)
        h2 = policy.model.register_forward_hook(out_hook)

        # Activate only the root ACT training branch. All children are eval
        # to disable attention/FFN dropout.
        policy.train()
        for child in policy.model.children():
            child.eval()
        policy.model.training = True

        try:
            with torch.no_grad(), zero_randn_like():
                _loss, _loss_dict = policy.forward(batch)
        finally:
            h1.remove()
            h2.remove()

        model_batch = captured["model_batch"]
        posterior = captured["posterior_actions"]
        mu = captured["mu"]
        logvar = captured["logvar"]

        if mu is None or logvar is None:
            raise RuntimeError("VAE posterior parameters were not captured.")

        if ACTION_KEY not in model_batch:
            raise RuntimeError("Captured model batch has no ACTION target.")

        target = model_batch[ACTION_KEY]
        if target.shape != posterior.shape:
            raise RuntimeError(
                f"target/posterior shape mismatch: "
                f"{tuple(target.shape)} vs {tuple(posterior.shape)}"
            )

        # Normal inference branch on the IDENTICAL model-space observation.
        policy.model.eval()
        prior_batch = clone_model_batch(model_batch)
        with torch.no_grad():
            prior, _ = policy.model(prior_batch)

        prior_err = torch.abs(prior - target)
        post_err = torch.abs(posterior - target)

        p0 = float(prior_err[:, 0].mean().item())
        q0 = float(post_err[:, 0].mean().item())
        pc = float(prior_err.mean().item())
        qc = float(post_err.mean().item())
        p10 = float(prior_err[:, :10].mean().item())
        q10 = float(post_err[:, :10].mean().item())

        # KL(q(z|a,o) || N(0,I)) per sample.
        kld = float(
            (
                -0.5
                * (1 + logvar - mu.pow(2) - logvar.exp())
            )
            .sum(-1)
            .mean()
            .item()
        )
        mu_norm = float(torch.linalg.vector_norm(mu, dim=-1).mean().item())

        prior_action0_mae.append(p0)
        posterior_action0_mae.append(q0)
        prior_chunk_mae.append(pc)
        posterior_chunk_mae.append(qc)
        prior_first10_mae.append(p10)
        posterior_first10_mae.append(q10)
        mu_norms.append(mu_norm)
        kld_values.append(kld)

        records.append(
            {
                "episode_index": ep,
                "frame_index": frame,
                "prior_zero_action0_mae_model_space": p0,
                "posterior_mu_action0_mae_model_space": q0,
                "prior_zero_first10_mae_model_space": p10,
                "posterior_mu_first10_mae_model_space": q10,
                "prior_zero_full_chunk_mae_model_space": pc,
                "posterior_mu_full_chunk_mae_model_space": qc,
                "posterior_mu_norm": mu_norm,
                "posterior_kld_to_standard_normal": kld,
            }
        )

        if n <= 5 or n % 10 == 0 or n == len(anchor_records):
            print(
                f"ep={ep:02d} frame={frame:03d} "
                f"a0 prior={p0:.4f} posterior_mu={q0:.4f} | "
                f"first10 prior={p10:.4f} posterior_mu={q10:.4f} | "
                f"mu_norm={mu_norm:.3f} kld={kld:.3f}"
            )

    def arr(x):
        return np.asarray(x, dtype=np.float64)

    p0 = arr(prior_action0_mae)
    q0 = arr(posterior_action0_mae)
    p10 = arr(prior_first10_mae)
    q10 = arr(posterior_first10_mae)
    pc = arr(prior_chunk_mae)
    qc = arr(posterior_chunk_mae)

    action0_reduction = (p0 - q0) / np.maximum(p0, 1e-12)
    first10_reduction = (p10 - q10) / np.maximum(p10, 1e-12)
    chunk_reduction = (pc - qc) / np.maximum(pc, 1e-12)

    print()
    print("===== SUMMARY =====")
    print(
        f"action[0] model-space MAE: "
        f"prior_zero={p0.mean():.4f} "
        f"posterior_mu={q0.mean():.4f} "
        f"reduction={action0_reduction.mean()*100:.1f}%"
    )
    print(
        f"first10 model-space MAE:   "
        f"prior_zero={p10.mean():.4f} "
        f"posterior_mu={q10.mean():.4f} "
        f"reduction={first10_reduction.mean()*100:.1f}%"
    )
    print(
        f"full50 model-space MAE:    "
        f"prior_zero={pc.mean():.4f} "
        f"posterior_mu={qc.mean():.4f} "
        f"reduction={chunk_reduction.mean()*100:.1f}%"
    )
    print(
        f"episodes posterior_mu improves action[0]: "
        f"{np.mean(q0 < p0)*100:.1f}%"
    )
    print(
        f"episodes posterior_mu improves first10: "
        f"{np.mean(q10 < p10)*100:.1f}%"
    )
    print(
        f"posterior mu norm: mean={np.mean(mu_norms):.3f} "
        f"median={np.median(mu_norms):.3f}"
    )
    print(
        f"posterior KL to N(0,I): mean={np.mean(kld_values):.3f} "
        f"median={np.median(kld_values):.3f}"
    )

    result = {
        "schema_version": "act_cvae_prior_posterior_v1",
        "checkpoint": str(checkpoint),
        "episodes": len(records),
        "metric_space": "exact model input/output space captured from ACTPolicy.forward",
        "summary": {
            "prior_zero_action0_mae": summarize(p0),
            "posterior_mu_action0_mae": summarize(q0),
            "action0_relative_error_reduction": summarize(action0_reduction),
            "prior_zero_first10_mae": summarize(p10),
            "posterior_mu_first10_mae": summarize(q10),
            "first10_relative_error_reduction": summarize(first10_reduction),
            "prior_zero_full_chunk_mae": summarize(pc),
            "posterior_mu_full_chunk_mae": summarize(qc),
            "full_chunk_relative_error_reduction": summarize(chunk_reduction),
            "posterior_mu_norm": summarize(np.asarray(mu_norms)),
            "posterior_kld_to_standard_normal": summarize(
                np.asarray(kld_values)
            ),
            "fraction_posterior_improves_action0": float(np.mean(q0 < p0)),
            "fraction_posterior_improves_first10": float(np.mean(q10 < p10)),
            "fraction_posterior_improves_full_chunk": float(np.mean(qc < pc)),
        },
        "records": records,
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "cvae_prior_posterior_summary.json"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("===== OUTPUT =====")
    print(json_path)
    print("OFFLINE ACT CVAE PRIOR/POSTERIOR AUDIT: PASS")
    print("NO HARDWARE WAS ACCESSED.")

    del policy
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
