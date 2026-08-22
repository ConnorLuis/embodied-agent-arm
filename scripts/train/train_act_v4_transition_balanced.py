#!/usr/bin/env python3
"""Train ACT with a deterministic transition-balanced virtual dataset.

The underlying LeRobot dataset and videos remain read-only.  A deterministic
WeightedRandomSampler raises the probability of samples whose next ten
gripper deltas contain a meaningful closing or opening transition.  The
validation dataset is not touched and this wrapper has no hardware capability.

Wrapper arguments must appear before ``--``.  Arguments after ``--`` are
passed unchanged to ``lerobot.scripts.train``.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SCHEMA_VERSION = "so101_act_v4_transition_balanced_sampler_v1"
ACTION_KEY = "action"
EXPECTED_ACTION_DIM = 6
GRIPPER_INDEX = 5
EXPECTED_TRAIN_FRAMES = 42678
EXPECTED_TRAIN_EPISODES = 48
EXPECTED_FPS = 15.0
DEFAULT_CHUNK_SIZE = 10


@dataclass(frozen=True)
class SamplingPlan:
    categories: np.ndarray
    weights: np.ndarray
    report: dict[str, Any]


def parse_wrapper_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=False)
    parser.add_argument("--dataset-repo-id", required=False)
    parser.add_argument("--video-backend", default="torchcodec")
    parser.add_argument("--sampler-report", type=Path, required=False)
    parser.add_argument("--target-close-fraction", type=float, default=0.25)
    parser.add_argument("--target-open-fraction", type=float, default=0.25)
    parser.add_argument("--target-step-threshold", type=float, default=0.25)
    parser.add_argument("--target-window-amplitude", type=float, default=2.5)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--sampler-seed", type=int, default=1000)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")

    separator = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(list(argv[:separator]))
    train_args = list(argv[separator + 1 :]) if separator < len(argv) else []

    if args.self_test:
        return args, train_args
    for name in ("dataset_root", "dataset_repo_id", "sampler_report"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")
    if args.chunk_size != DEFAULT_CHUNK_SIZE:
        parser.error("the frozen ACT V4 contract requires --chunk-size=10")
    if args.target_step_threshold <= 0 or args.target_window_amplitude <= 0:
        parser.error("transition thresholds must be positive")
    for name in ("target_close_fraction", "target_open_fraction"):
        value = float(getattr(args, name))
        if not 0.0 < value < 1.0:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1")
    if args.target_close_fraction + args.target_open_fraction >= 1.0:
        parser.error("close + open target fractions must be below 1")
    if not args.plan_only and not train_args:
        parser.error("training arguments are required after --")
    return args, train_args


def cumulative_amplitude(values: np.ndarray, sign: int) -> float:
    cumulative = np.concatenate(
        [np.zeros(1, dtype=np.float64), np.cumsum(values, dtype=np.float64)]
    )
    if sign < 0:
        return float(max(0.0, -float(np.min(cumulative))))
    return float(max(0.0, float(np.max(cumulative))))


def validate_arrays(
    actions: np.ndarray,
    episodes: np.ndarray,
    frames: np.ndarray,
    *,
    expected_frames: int,
    expected_episodes: int,
) -> None:
    if actions.shape != (expected_frames, EXPECTED_ACTION_DIM):
        raise RuntimeError(f"unexpected action shape: {actions.shape}")
    if episodes.shape != (expected_frames,) or frames.shape != (expected_frames,):
        raise RuntimeError("episode/frame index shape mismatch")
    if not np.isfinite(actions).all():
        raise FloatingPointError("action data contains NaN or Inf")
    unique = sorted(int(value) for value in np.unique(episodes))
    if unique != list(range(expected_episodes)):
        raise RuntimeError(f"unexpected episode indices: {unique}")
    for episode in unique:
        values = frames[episodes == episode]
        if not np.array_equal(values, np.arange(len(values), dtype=np.int64)):
            raise RuntimeError(f"episode {episode} frames are not contiguous")


def classify_samples(
    actions: np.ndarray,
    episodes: np.ndarray,
    *,
    chunk_size: int,
    step_threshold: float,
    amplitude_threshold: float,
) -> tuple[np.ndarray, dict[str, int]]:
    # 0=background, 1=close, 2=open.  Mixed windows take the larger amplitude.
    categories = np.zeros(len(actions), dtype=np.int8)
    mixed = 0
    for episode in sorted(int(value) for value in np.unique(episodes)):
        indices = np.flatnonzero(episodes == episode)
        for local_index, global_index in enumerate(indices):
            end = min(local_index + chunk_size, len(indices))
            window = actions[indices[local_index:end], GRIPPER_INDEX].astype(
                np.float64
            )
            close_amplitude = cumulative_amplitude(window, -1)
            open_amplitude = cumulative_amplitude(window, 1)
            close_active = bool(np.any(window <= -step_threshold))
            open_active = bool(np.any(window >= step_threshold))
            close_event = close_active and close_amplitude >= amplitude_threshold
            open_event = open_active and open_amplitude >= amplitude_threshold
            if close_event and open_event:
                mixed += 1
            if close_event or open_event:
                categories[global_index] = (
                    1 if close_amplitude >= open_amplitude else 2
                )
    counts = {
        "background": int(np.sum(categories == 0)),
        "close": int(np.sum(categories == 1)),
        "open": int(np.sum(categories == 2)),
        "mixed_assigned_by_larger_amplitude": mixed,
    }
    if counts["close"] == 0 or counts["open"] == 0:
        raise RuntimeError(f"transition classification is empty: {counts}")
    return categories, counts


def create_sampling_plan(
    *,
    actions: np.ndarray,
    episodes: np.ndarray,
    frames: np.ndarray,
    expected_frames: int,
    expected_episodes: int,
    chunk_size: int,
    step_threshold: float,
    amplitude_threshold: float,
    close_fraction: float,
    open_fraction: float,
    seed: int,
) -> SamplingPlan:
    validate_arrays(
        actions,
        episodes,
        frames,
        expected_frames=expected_frames,
        expected_episodes=expected_episodes,
    )
    categories, counts = classify_samples(
        actions,
        episodes,
        chunk_size=chunk_size,
        step_threshold=step_threshold,
        amplitude_threshold=amplitude_threshold,
    )
    desired = {
        "background": 1.0 - close_fraction - open_fraction,
        "close": close_fraction,
        "open": open_fraction,
    }
    labels = {"background": 0, "close": 1, "open": 2}
    raw_weights = np.zeros(len(actions), dtype=np.float64)
    factors: dict[str, float] = {}
    observed: dict[str, float] = {}
    for name, label in labels.items():
        fraction = counts[name] / len(actions)
        if fraction <= 0.0:
            raise RuntimeError(f"category {name} is empty")
        observed[name] = fraction
        factors[name] = desired[name] / fraction
        raw_weights[categories == label] = factors[name]
    weights = raw_weights / float(np.mean(raw_weights))
    expected = {
        name: float(np.sum(weights[categories == label]) / np.sum(weights))
        for name, label in labels.items()
    }
    action_digest = hashlib.sha256(
        np.ascontiguousarray(actions.astype(np.float32)).tobytes()
    ).hexdigest()
    category_digest = hashlib.sha256(categories.tobytes()).hexdigest()
    report = {
        "schema_version": SCHEMA_VERSION,
        "scope": {
            "virtual_dataset": True,
            "source_dataset_modified": False,
            "validation_dataset_modified": False,
            "hardware_accessed": False,
            "robot_or_camera_imported": False,
        },
        "source": {
            "frames": len(actions),
            "episodes": expected_episodes,
            "action_sha256": action_digest,
            "category_sha256": category_digest,
        },
        "transition_contract": {
            "chunk_size": chunk_size,
            "gripper_index": GRIPPER_INDEX,
            "closing_direction": "negative delta",
            "target_step_threshold": step_threshold,
            "target_window_amplitude": amplitude_threshold,
        },
        "observed": {"counts": counts, "fractions": observed},
        "target_fractions": desired,
        "sampling": {
            "replacement": True,
            "samples_per_virtual_epoch": len(actions),
            "seed": seed,
            "category_weight_factors_before_mean_normalization": factors,
            "normalized_weight_min": float(np.min(weights)),
            "normalized_weight_max": float(np.max(weights)),
            "expected_sampled_fractions": expected,
        },
        "dataloader_patch_applied": False,
        "training_completed": False,
        "decision": "V4_TRANSITION_BALANCED_PLAN_PASS_TRAINING_NOT_STARTED",
    }
    for name in desired:
        if abs(expected[name] - desired[name]) > 1e-12:
            raise RuntimeError(f"weighted expectation mismatch for {name}")
    return SamplingPlan(categories=categories, weights=weights, report=report)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_source_dataset(args: argparse.Namespace) -> tuple[Any, SamplingPlan]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = args.dataset_root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    try:
        dataset = LeRobotDataset(
            repo_id=args.dataset_repo_id,
            root=root,
            video_backend=args.video_backend,
        )
    except TypeError:
        dataset = LeRobotDataset(
            args.dataset_repo_id,
            root=root,
            video_backend=args.video_backend,
        )
    hf = dataset.hf_dataset
    actions = np.asarray(hf[ACTION_KEY], dtype=np.float32)
    episodes = np.asarray(hf["episode_index"], dtype=np.int64)
    frames = np.asarray(hf["frame_index"], dtype=np.int64)
    fps_candidates = [
        getattr(dataset, "fps", None),
        getattr(getattr(dataset, "meta", None), "fps", None),
    ]
    fps = next((float(value) for value in fps_candidates if value is not None), None)
    if fps is not None and abs(fps - EXPECTED_FPS) > 1e-9:
        raise RuntimeError(f"dataset fps={fps}, expected {EXPECTED_FPS}")
    plan = create_sampling_plan(
        actions=actions,
        episodes=episodes,
        frames=frames,
        expected_frames=EXPECTED_TRAIN_FRAMES,
        expected_episodes=EXPECTED_TRAIN_EPISODES,
        chunk_size=args.chunk_size,
        step_threshold=args.target_step_threshold,
        amplitude_threshold=args.target_window_amplitude,
        close_fraction=args.target_close_fraction,
        open_fraction=args.target_open_fraction,
        seed=args.sampler_seed,
    )
    plan.report["source"].update(
        {
            "root": str(root),
            "repo_id": args.dataset_repo_id,
            "fps": fps,
        }
    )
    return dataset, plan


def parse_forwarded_cli(train_args: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in train_args:
        if item.startswith("--") and "=" in item:
            key, value = item[2:].split("=", 1)
            values[key] = value
    return values


def nested_config_value(config: dict[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def canonical_cli_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def validate_forwarded_training_args(
    args: argparse.Namespace, train_args: list[str]
) -> dict[str, Any]:
    values = parse_forwarded_cli(train_args)
    config_path_text = values.get("config_path")
    if not config_path_text:
        raise RuntimeError(
            "--config_path=<frozen V3 train_config.json> is required so V4 "
            "changes only the sampling distribution"
        )
    config_path = Path(config_path_text).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    base_config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(base_config, dict):
        raise RuntimeError("base train config must contain a JSON object")

    def effective(key: str) -> str | None:
        if key in values:
            return values[key]
        return canonical_cli_value(nested_config_value(base_config, key))

    required = {
        "dataset.repo_id": args.dataset_repo_id,
        "dataset.video_backend": args.video_backend,
        "dataset.image_transforms.enable": "false",
        "policy.type": "act",
        "policy.device": "cuda",
        "policy.chunk_size": "10",
        "policy.n_action_steps": "5",
        "policy.use_vae": "false",
        "policy.use_amp": "false",
        "policy.push_to_hub": "false",
        "policy.dim_model": "512",
        "policy.dim_feedforward": "3200",
        "policy.n_heads": "8",
        "policy.n_encoder_layers": "4",
        "policy.n_decoder_layers": "1",
        "policy.vision_backbone": "resnet18",
        "batch_size": "4",
        "num_workers": "4",
        "seed": str(args.sampler_seed),
        "save_checkpoint": "true",
        "eval_freq": "0",
        "wandb.enable": "false",
    }
    for key, expected in required.items():
        actual = effective(key)
        if actual != expected:
            raise RuntimeError(
                f"effective training value {key}={expected} required; "
                f"got {actual!r}"
            )
    forwarded_root = Path(effective("dataset.root") or "").expanduser().resolve()
    if forwarded_root != args.dataset_root.expanduser().resolve():
        raise RuntimeError("forwarded dataset.root does not match wrapper dataset")
    if effective("resume") != "false":
        raise RuntimeError("V4 must explicitly set --resume=false")
    steps = int(effective("steps") or "-1")
    save_frequency = int(effective("save_freq") or "-1")
    if steps != 20000 or save_frequency != 4000:
        raise RuntimeError("frozen V4 budget requires steps=20000/save_freq=4000")
    output_text = effective("output_dir")
    output = Path(output_text or "").expanduser().resolve()
    if not output_text or output.exists():
        raise RuntimeError(f"V4 output_dir must be new: {output}")
    return {
        "base_config": str(config_path),
        "base_config_sha256": hashlib.sha256(
            config_path.read_bytes()
        ).hexdigest(),
        "overrides": values,
        "effective_contract": {key: effective(key) for key in required},
        "effective_steps": steps,
        "effective_save_freq": save_frequency,
        "effective_output_dir": str(output),
        "effective_resume": effective("resume"),
    }


def verify_train_loader_patchability() -> dict[str, str]:
    """Fail before training if the installed entry point bypasses our patch."""
    module_name = "lerobot.scripts.train"
    if module_name in sys.modules:
        raise RuntimeError(
            "lerobot.scripts.train was imported before the DataLoader patch"
        )
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError(module_name)
    source_path = Path(spec.origin).resolve()
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))

    loader_aliases: set[str] = set()
    torch_imported = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "torch.utils.data":
            for name in node.names:
                if name.name == "DataLoader":
                    loader_aliases.add(name.asname or name.name)
        elif isinstance(node, ast.Import):
            for name in node.names:
                if name.name == "torch":
                    torch_imported = True

    direct_call = False
    attribute_call = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in loader_aliases:
            direct_call = True
        if isinstance(node.func, ast.Attribute) and node.func.attr == "DataLoader":
            attribute_call = True

    if not direct_call and not (torch_imported and attribute_call):
        raise RuntimeError(
            "installed lerobot.scripts.train does not construct a patchable "
            "torch DataLoader; aborting before training"
        )
    return {
        "module": module_name,
        "source": str(source_path),
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "call_style": "direct_import" if direct_call else "torch_attribute",
    }


def dataset_fingerprint(dataset: Any) -> str | None:
    current = dataset
    for _ in range(4):
        hf = getattr(current, "hf_dataset", None)
        if hf is not None:
            actions = np.asarray(hf[ACTION_KEY], dtype=np.float32)
            return hashlib.sha256(
                np.ascontiguousarray(actions).tobytes()
            ).hexdigest()
        current = getattr(current, "dataset", None)
        if current is None:
            return None
    return None


def run_training_with_sampler(
    *,
    args: argparse.Namespace,
    train_args: list[str],
    plan: SamplingPlan,
) -> int:
    import torch

    report_path = args.sampler_report.expanduser().resolve()
    expected_digest = str(plan.report["source"]["action_sha256"])
    weights = torch.as_tensor(plan.weights, dtype=torch.double)
    state = {"applied": False, "calls": 0}
    original_public = torch.utils.data.DataLoader
    dataloader_module = torch.utils.data.dataloader
    original_module = dataloader_module.DataLoader

    class TransitionBalancedDataLoader(original_public):
        def __init__(self, dataset: Any, *positional: Any, **keyword: Any) -> None:
            fingerprint = dataset_fingerprint(dataset)
            if fingerprint == expected_digest:
                if state["applied"]:
                    raise RuntimeError("transition sampler applied more than once")
                if len(dataset) != len(weights):
                    raise RuntimeError("training dataset length changed")
                if keyword.get("batch_sampler") is not None:
                    raise RuntimeError("custom batch_sampler is incompatible")
                positional_values = list(positional)
                if len(positional_values) >= 3 and positional_values[2] is not None:
                    raise RuntimeError("pre-existing positional sampler is forbidden")
                if len(positional_values) < 3 and keyword.get("sampler") is not None:
                    raise RuntimeError("pre-existing sampler is forbidden")
                preview_generator = torch.Generator()
                preview_generator.manual_seed(args.sampler_seed)
                preview_sampler = torch.utils.data.WeightedRandomSampler(
                    weights=weights,
                    num_samples=len(weights),
                    replacement=True,
                    generator=preview_generator,
                )
                preview_indices = np.fromiter(
                    preview_sampler,
                    dtype=np.int64,
                    count=len(weights),
                )
                preview_categories = plan.categories[preview_indices]
                realized = {
                    "background": float(np.mean(preview_categories == 0)),
                    "close": float(np.mean(preview_categories == 1)),
                    "open": float(np.mean(preview_categories == 2)),
                }
                if any(
                    abs(realized[name] - target) > 0.015
                    for name, target in {
                        "background": 0.50,
                        "close": 0.25,
                        "open": 0.25,
                    }.items()
                ):
                    raise RuntimeError(
                        f"seeded virtual epoch is unexpectedly imbalanced: {realized}"
                    )
                generator = torch.Generator()
                generator.manual_seed(args.sampler_seed)
                sampler = torch.utils.data.WeightedRandomSampler(
                    weights=weights,
                    num_samples=len(weights),
                    replacement=True,
                    generator=generator,
                )
                if len(positional_values) >= 2:
                    positional_values[1] = False
                    keyword.pop("shuffle", None)
                else:
                    keyword["shuffle"] = False
                if len(positional_values) >= 3:
                    positional_values[2] = sampler
                    keyword.pop("sampler", None)
                else:
                    keyword["sampler"] = sampler
                positional = tuple(positional_values)
                state["applied"] = True
                state["calls"] += 1
                plan.report["dataloader_patch_applied"] = True
                plan.report["sampling"][
                    "first_virtual_epoch_realized_fractions"
                ] = realized
                plan.report["decision"] = (
                    "V4_TRANSITION_BALANCED_SAMPLER_APPLIED_TRAINING_RUNNING"
                )
                atomic_json(report_path, plan.report)
                print("\n===== V4 TRANSITION-BALANCED DATALOADER =====")
                print("WeightedRandomSampler applied before the first batch: PASS")
                print(
                    "expected fractions: background=0.500 close=0.250 open=0.250"
                )
                print(f"seeded first virtual epoch: {realized}")
            super().__init__(dataset, *positional, **keyword)

    torch.utils.data.DataLoader = TransitionBalancedDataLoader
    dataloader_module.DataLoader = TransitionBalancedDataLoader
    old_argv = sys.argv
    exit_code = 0
    try:
        sys.argv = ["lerobot.scripts.train", *train_args]
        runpy.run_module("lerobot.scripts.train", run_name="__main__")
    except SystemExit as error:
        exit_code = int(error.code or 0)
        if exit_code != 0:
            raise
    finally:
        sys.argv = old_argv
        torch.utils.data.DataLoader = original_public
        dataloader_module.DataLoader = original_module
    if not state["applied"] or state["calls"] != 1:
        raise RuntimeError(f"transition sampler application failed: {state}")
    plan.report["training_completed"] = True
    plan.report["decision"] = (
        "V4_TRANSITION_BALANCED_TRAINING_COMPLETE_RUN_OFFLINE_AUDITS_NEXT"
    )
    atomic_json(report_path, plan.report)
    return exit_code


def run_self_test() -> None:
    episodes = np.repeat(np.arange(4, dtype=np.int64), 30)
    frames = np.tile(np.arange(30, dtype=np.int64), 4)
    actions = np.zeros((120, EXPECTED_ACTION_DIM), dtype=np.float32)
    for episode in range(4):
        start = episode * 30
        actions[start + 5 : start + 10, GRIPPER_INDEX] = -0.6
        actions[start + 18 : start + 23, GRIPPER_INDEX] = 0.6
    plan = create_sampling_plan(
        actions=actions,
        episodes=episodes,
        frames=frames,
        expected_frames=120,
        expected_episodes=4,
        chunk_size=10,
        step_threshold=0.25,
        amplitude_threshold=2.5,
        close_fraction=0.25,
        open_fraction=0.25,
        seed=1000,
    )
    expected = plan.report["sampling"]["expected_sampled_fractions"]
    target = {"background": 0.5, "close": 0.25, "open": 0.25}
    if any(abs(float(expected[key]) - value) > 1e-12 for key, value in target.items()):
        raise RuntimeError(expected)
    print("ACT V4 TRANSITION-BALANCED SAMPLER SELF-TEST: PASS")


def main(argv: Sequence[str] | None = None) -> int:
    args, train_args = parse_wrapper_args(list(sys.argv[1:] if argv is None else argv))
    if args.self_test:
        run_self_test()
        return 0

    report_path = args.sampler_report.expanduser().resolve()
    if report_path.exists():
        raise FileExistsError(f"sampler report already exists: {report_path}")
    print("===== ACT V4 STATIC CAPABILITY BOUNDARY =====")
    print("offline dataset sampling + ACT training only")
    print("NO robot, motor bus, serial port, live camera, or command API")

    print("\n===== BUILD READ-ONLY VIRTUAL DATASET PLAN =====")
    _dataset, plan = load_source_dataset(args)
    atomic_json(report_path, plan.report)
    observed = plan.report["observed"]
    print(f"source counts: {observed['counts']}")
    print(f"source fractions: {observed['fractions']}")
    print(
        "target fractions: "
        f"{plan.report['sampling']['expected_sampled_fractions']}: PASS"
    )
    print(f"report={report_path}")
    if args.plan_only:
        print("ACT V4 TRANSITION-BALANCED PLAN: PASS")
        return 0

    forwarded = validate_forwarded_training_args(args, train_args)
    plan.report["training_cli_contract"] = forwarded
    plan.report["train_entry_point"] = verify_train_loader_patchability()
    atomic_json(report_path, plan.report)
    print("installed LeRobot DataLoader interception: PASS")
    print("\n===== START FRESH BOUNDED V4 TRAINING =====")
    exit_code = run_training_with_sampler(
        args=args,
        train_args=train_args,
        plan=plan,
    )
    print("\n===== DECISION =====")
    print("decision='V4_TRANSITION_BALANCED_TRAINING_COMPLETE_RUN_OFFLINE_AUDITS_NEXT'")
    print("HARDWARE REMAINS BLOCKED.")
    print("ACT V4 TRANSITION-BALANCED TRAINING: PASS")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
