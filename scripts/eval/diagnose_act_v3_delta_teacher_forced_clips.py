#!/usr/bin/env python
"""Attribute rate and soft clips from the full teacher-forced local replay.

This program reads only the frozen JSON contract, the completed local-replay
JSON report, and its command CSV. It does not import Torch or LeRobot, load a
model or dataset, create a robot object, open a serial port, or send commands.

The report separates soft-limit events that inherit an already-active boundary
from new crossings out of the interior. It also records overshoot magnitude,
the demonstrated target direction, episode, and queue substep. Diagnostic
magnitude bins are descriptive only; they are not physical safety thresholds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


EXPECTED_CONTRACT_SCHEMA = "so101_act_v3_delta_runtime_limit_contract_v1"
EXPECTED_LOCAL_SCHEMA = "so101_act_v3_delta_teacher_forced_local_replay_v1"
EXPECTED_LOCAL_DECISION = "FULL_LOCAL_REPLAY_REVIEW_SOFT_CLIPS"
OUTPUT_SCHEMA = "so101_act_v3_delta_teacher_forced_clip_attribution_v1"
DIAGNOSTIC_OVERSHOOT_LEVELS = (0.001, 0.01, 0.05, 0.10, 0.25, 0.50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--local-report", type=Path, required=True)
    parser.add_argument("--commands-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Missing JSON file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def finite_float(row: dict[str, str], column: str, row_number: int) -> float:
    require(column in row, f"CSV is missing column: {column}")
    try:
        value = float(row[column])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"CSV row {row_number}: invalid float in {column}: {row[column]!r}"
        ) from exc
    require(math.isfinite(value), f"CSV row {row_number}: NaN/Inf in {column}")
    return value


def integer(row: dict[str, str], column: str, row_number: int) -> int:
    require(column in row, f"CSV is missing column: {column}")
    try:
        return int(row[column])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"CSV row {row_number}: invalid integer in {column}: {row[column]!r}"
        ) from exc


def percentile(values: Iterable[float], probability: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def ordered_vector(
    mapping: dict[str, Any], motors: list[str], label: str
) -> list[float]:
    require(isinstance(mapping, dict), f"{label} must be an object")
    require(set(mapping) == set(motors), f"{label} motor keys differ")
    values = [float(mapping[motor]) for motor in motors]
    require(all(math.isfinite(value) for value in values), f"{label} is non-finite")
    return values


def classify_motion(delta: float, direction: str, tolerance: float) -> str:
    if abs(delta) <= tolerance:
        return "hold"
    outward = delta > 0.0 if direction == "max" else delta < 0.0
    return "outward" if outward else "inward"


def event_sort_key(event: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(event["episode_index"]),
        int(event["dataset_frame"]),
        int(event["joint_index"]),
        0 if event["event_type"] == "rate_clip" else 1,
    )


def main() -> int:
    args = parse_args()
    contract_path = args.contract.expanduser().resolve()
    local_report_path = args.local_report.expanduser().resolve()
    commands_path = args.commands_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    contract = load_json(contract_path)
    local_report = load_json(local_report_path)
    require(commands_path.is_file(), f"Missing commands CSV: {commands_path}")

    print("===== VERIFY FROZEN OFFLINE INPUTS =====")
    require(
        contract.get("schema_version") == EXPECTED_CONTRACT_SCHEMA,
        "Unexpected runtime contract schema",
    )
    contract_scope = contract.get("scope", {})
    require(
        contract_scope.get("offline_dry_run_only") is True,
        "Contract is not offline-only",
    )
    require(
        contract_scope.get("hardware_access_allowed") is False,
        "Contract unexpectedly allows hardware access",
    )
    require(
        contract_scope.get("hardware_deployment_authorized") is False,
        "Contract unexpectedly authorizes hardware deployment",
    )

    require(
        local_report.get("schema_version") == EXPECTED_LOCAL_SCHEMA,
        "Unexpected local-replay report schema",
    )
    require(local_report.get("status") == "PASS", "Local replay status is not PASS")
    require(
        local_report.get("decision") == EXPECTED_LOCAL_DECISION,
        "Local replay did not request soft-clip review",
    )
    local_scope = local_report.get("scope", {})
    require(local_scope.get("offline_only") is True, "Local replay is not offline-only")
    for key in (
        "hardware_accessed",
        "robot_object_created",
        "serial_port_opened",
        "command_sent",
        "hardware_deployment_authorized",
        "physical_safety_certified",
    ):
        require(local_scope.get(key) is False, f"Local-replay scope violation: {key}")

    motors = contract.get("robot", {}).get("motors")
    require(isinstance(motors, list) and len(motors) == 6, "Expected six motors")
    require(all(isinstance(motor, str) for motor in motors), "Invalid motor names")
    motors = list(motors)

    guard = contract.get("guard", {})
    tolerance = float(guard.get("numerical_tolerance"))
    require(math.isfinite(tolerance) and tolerance > 0.0, "Invalid tolerance")
    equation_tolerance = max(5.0 * tolerance, 1e-8)
    max_delta = ordered_vector(
        guard.get("max_abs_delta_per_command", {}), motors, "max delta"
    )
    soft_mapping = guard.get("normal_soft_limits", {})
    require(set(soft_mapping) == set(motors), "Soft-limit motor keys differ")
    soft_min: list[float] = []
    soft_max: list[float] = []
    for motor in motors:
        pair = soft_mapping[motor]
        require(isinstance(pair, list) and len(pair) == 2, f"Invalid limits for {motor}")
        low, high = float(pair[0]), float(pair[1])
        require(math.isfinite(low) and math.isfinite(high) and low < high, f"Invalid limits for {motor}")
        soft_min.append(low)
        soft_max.append(high)

    run = local_report.get("run", {})
    require(run.get("kind") == "full", "Local replay is not a full run")
    expected_commands = int(run.get("total_commands", 0))
    expected_rate_any = int(run.get("commands_with_any_rate_clip", -1))
    expected_soft_any = int(run.get("commands_with_any_soft_clip", -1))
    require(expected_commands > 0, "Local report has no commands")
    integrity = local_report.get("validation_integrity", {})
    target_violations = integrity.get(
        "recorded_target_soft_violation_count_by_joint", {}
    )
    require(set(target_violations) == set(motors), "Target-violation keys differ")
    require(
        all(int(target_violations[motor]) == 0 for motor in motors),
        "Recorded validation targets violate soft limits",
    )
    print("contract/report scope, schemas, and recorded target limits: PASS")

    base_columns = {
        "run_kind",
        "episode_index",
        "replan_index",
        "replan_frame",
        "substep",
        "dataset_frame",
        "any_rate_clipped",
        "any_soft_clipped",
    }
    value_fields = (
        "recorded_previous_command",
        "local_previous_command",
        "recorded_target_delta",
        "raw_predicted_delta",
        "rate_limited_delta",
        "unbounded_command",
        "guarded_command",
        "actual_guarded_delta",
        "recorded_target_command",
        "rate_clipped",
        "soft_clipped",
    )

    total_commands = 0
    any_rate_commands = 0
    any_soft_commands = 0
    max_equation_error = 0.0
    events: list[dict[str, Any]] = []
    episode_metrics: dict[int, dict[str, int]] = {}
    joint_metrics: dict[str, dict[str, Any]] = {
        motor: {
            "rate_events": [],
            "soft_events": [],
            "rate_clip_count": 0,
            "soft_clip_count": 0,
            "local_at_min_count": 0,
            "local_at_max_count": 0,
            "recorded_previous_at_min_count": 0,
            "recorded_previous_at_max_count": 0,
        }
        for motor in motors
    }

    print("\n===== CSV EQUATION AND EVENT AUDIT =====")
    with commands_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        require(base_columns <= fieldnames, "CSV base columns are incomplete")
        for motor in motors:
            for field in value_fields:
                require(f"{field}.{motor}" in fieldnames, f"Missing {field}.{motor}")

        for row_number, row in enumerate(reader, start=2):
            require(row["run_kind"] == "full", f"CSV row {row_number}: run kind differs")
            episode = integer(row, "episode_index", row_number)
            replan_index = integer(row, "replan_index", row_number)
            replan_frame = integer(row, "replan_frame", row_number)
            substep = integer(row, "substep", row_number)
            dataset_frame = integer(row, "dataset_frame", row_number)
            any_rate_csv = integer(row, "any_rate_clipped", row_number)
            any_soft_csv = integer(row, "any_soft_clipped", row_number)
            require(any_rate_csv in (0, 1), f"CSV row {row_number}: invalid rate flag")
            require(any_soft_csv in (0, 1), f"CSV row {row_number}: invalid soft flag")
            require(0 <= substep < 5, f"CSV row {row_number}: invalid substep")
            require(dataset_frame == replan_frame + substep, f"CSV row {row_number}: frame mismatch")

            episode_entry = episode_metrics.setdefault(
                episode,
                {
                    "commands": 0,
                    "commands_with_any_rate_clip": 0,
                    "commands_with_any_soft_clip": 0,
                },
            )
            row_has_rate = False
            row_has_soft = False

            for joint, motor in enumerate(motors):
                recorded_previous = finite_float(
                    row, f"recorded_previous_command.{motor}", row_number
                )
                local_previous = finite_float(
                    row, f"local_previous_command.{motor}", row_number
                )
                target_delta = finite_float(
                    row, f"recorded_target_delta.{motor}", row_number
                )
                raw_delta = finite_float(
                    row, f"raw_predicted_delta.{motor}", row_number
                )
                rate_limited = finite_float(
                    row, f"rate_limited_delta.{motor}", row_number
                )
                unbounded = finite_float(
                    row, f"unbounded_command.{motor}", row_number
                )
                guarded = finite_float(
                    row, f"guarded_command.{motor}", row_number
                )
                actual_delta = finite_float(
                    row, f"actual_guarded_delta.{motor}", row_number
                )
                target_command = finite_float(
                    row, f"recorded_target_command.{motor}", row_number
                )
                rate_flag = integer(row, f"rate_clipped.{motor}", row_number)
                soft_flag = integer(row, f"soft_clipped.{motor}", row_number)
                require(rate_flag in (0, 1), f"CSV row {row_number}: invalid joint rate flag")
                require(soft_flag in (0, 1), f"CSV row {row_number}: invalid joint soft flag")

                expected_rate = max(-max_delta[joint], min(max_delta[joint], raw_delta))
                expected_unbounded = local_previous + expected_rate
                expected_guarded = max(
                    soft_min[joint], min(soft_max[joint], expected_unbounded)
                )
                expected_actual_delta = expected_guarded - local_previous
                expected_target = recorded_previous + target_delta
                errors = (
                    abs(rate_limited - expected_rate),
                    abs(unbounded - expected_unbounded),
                    abs(guarded - expected_guarded),
                    abs(actual_delta - expected_actual_delta),
                    abs(target_command - expected_target),
                )
                max_equation_error = max(max_equation_error, *errors)
                require(
                    max(errors) <= equation_tolerance,
                    f"CSV row {row_number}, {motor}: equation mismatch",
                )
                expected_rate_flag = int(abs(raw_delta - expected_rate) > tolerance)
                expected_soft_flag = int(abs(expected_unbounded - expected_guarded) > tolerance)
                require(rate_flag == expected_rate_flag, f"CSV row {row_number}, {motor}: rate flag mismatch")
                require(soft_flag == expected_soft_flag, f"CSV row {row_number}, {motor}: soft flag mismatch")

                local_at_min = abs(local_previous - soft_min[joint]) <= tolerance
                local_at_max = abs(local_previous - soft_max[joint]) <= tolerance
                recorded_at_min = abs(recorded_previous - soft_min[joint]) <= tolerance
                recorded_at_max = abs(recorded_previous - soft_max[joint]) <= tolerance
                metric = joint_metrics[motor]
                metric["local_at_min_count"] += int(local_at_min)
                metric["local_at_max_count"] += int(local_at_max)
                metric["recorded_previous_at_min_count"] += int(recorded_at_min)
                metric["recorded_previous_at_max_count"] += int(recorded_at_max)

                if rate_flag:
                    row_has_rate = True
                    direction = "positive" if raw_delta > 0.0 else "negative"
                    event = {
                        "event_type": "rate_clip",
                        "episode_index": episode,
                        "replan_index": replan_index,
                        "replan_frame": replan_frame,
                        "substep": substep,
                        "dataset_frame": dataset_frame,
                        "joint_index": joint,
                        "joint": motor,
                        "direction": direction,
                        "clip_limit": max_delta[joint],
                        "clip_overshoot": abs(raw_delta) - max_delta[joint],
                        "recorded_previous_command": recorded_previous,
                        "local_previous_command": local_previous,
                        "recorded_target_delta": target_delta,
                        "raw_predicted_delta": raw_delta,
                        "rate_limited_delta": rate_limited,
                        "unbounded_command": unbounded,
                        "guarded_command": guarded,
                        "recorded_target_command": target_command,
                        "boundary_source": "not_applicable",
                        "local_distance_to_boundary_before": None,
                        "recorded_previous_distance_to_boundary": None,
                        "recorded_target_distance_to_boundary": None,
                        "target_motion_relative_to_boundary": "not_applicable",
                        "prediction_motion_relative_to_boundary": "not_applicable",
                    }
                    events.append(event)
                    metric["rate_events"].append(event)
                    metric["rate_clip_count"] += 1

                if soft_flag:
                    row_has_soft = True
                    direction = "max" if expected_unbounded > soft_max[joint] else "min"
                    boundary = soft_max[joint] if direction == "max" else soft_min[joint]
                    local_distance = (
                        boundary - local_previous
                        if direction == "max"
                        else local_previous - boundary
                    )
                    recorded_previous_distance = (
                        boundary - recorded_previous
                        if direction == "max"
                        else recorded_previous - boundary
                    )
                    target_distance = (
                        boundary - target_command
                        if direction == "max"
                        else target_command - boundary
                    )
                    boundary_source = (
                        "inherited_at_boundary"
                        if local_distance <= tolerance
                        else "new_crossing_from_interior"
                    )
                    event = {
                        "event_type": "soft_clip",
                        "episode_index": episode,
                        "replan_index": replan_index,
                        "replan_frame": replan_frame,
                        "substep": substep,
                        "dataset_frame": dataset_frame,
                        "joint_index": joint,
                        "joint": motor,
                        "direction": direction,
                        "clip_limit": boundary,
                        "clip_overshoot": abs(expected_unbounded - boundary),
                        "recorded_previous_command": recorded_previous,
                        "local_previous_command": local_previous,
                        "recorded_target_delta": target_delta,
                        "raw_predicted_delta": raw_delta,
                        "rate_limited_delta": rate_limited,
                        "unbounded_command": unbounded,
                        "guarded_command": guarded,
                        "recorded_target_command": target_command,
                        "boundary_source": boundary_source,
                        "local_distance_to_boundary_before": local_distance,
                        "recorded_previous_distance_to_boundary": recorded_previous_distance,
                        "recorded_target_distance_to_boundary": target_distance,
                        "target_motion_relative_to_boundary": classify_motion(
                            target_delta, direction, tolerance
                        ),
                        "prediction_motion_relative_to_boundary": classify_motion(
                            raw_delta, direction, tolerance
                        ),
                    }
                    events.append(event)
                    metric["soft_events"].append(event)
                    metric["soft_clip_count"] += 1

            require(any_rate_csv == int(row_has_rate), f"CSV row {row_number}: any-rate mismatch")
            require(any_soft_csv == int(row_has_soft), f"CSV row {row_number}: any-soft mismatch")
            total_commands += 1
            any_rate_commands += int(row_has_rate)
            any_soft_commands += int(row_has_soft)
            episode_entry["commands"] += 1
            episode_entry["commands_with_any_rate_clip"] += int(row_has_rate)
            episode_entry["commands_with_any_soft_clip"] += int(row_has_soft)

    require(total_commands == expected_commands, "CSV/report command count mismatch")
    require(any_rate_commands == expected_rate_any, "CSV/report rate count mismatch")
    require(any_soft_commands == expected_soft_any, "CSV/report soft count mismatch")
    report_joint_metrics = local_report.get("joint_metrics", {})
    require(set(report_joint_metrics) == set(motors), "Report joint keys differ")
    for motor in motors:
        require(
            joint_metrics[motor]["rate_clip_count"]
            == int(report_joint_metrics[motor]["rate_clip_count"]),
            f"Rate count differs for {motor}",
        )
        require(
            joint_metrics[motor]["soft_clip_count"]
            == int(report_joint_metrics[motor]["soft_clip_count"]),
            f"Soft count differs for {motor}",
        )
    print(f"commands checked: {total_commands}")
    print(f"guard/target equation max error: {max_equation_error:.10f}")
    print("CSV equations, flags, and report counts: PASS")

    print("\n===== CLIP ATTRIBUTION BY EPISODE =====")
    print("episode  commands  rate_any  soft_any  soft_rate")
    for episode, values in sorted(episode_metrics.items()):
        soft_rate = values["commands_with_any_soft_clip"] / values["commands"]
        print(
            f"{episode:7d}  {values['commands']:8d}  "
            f"{values['commands_with_any_rate_clip']:8d}  "
            f"{values['commands_with_any_soft_clip']:8d}  {soft_rate:9.3%}"
        )

    print("\n===== RATE CLIP ATTRIBUTION =====")
    print("joint             count  overshoot_p95  overshoot_max")
    joint_report: dict[str, Any] = {}
    for motor in motors:
        metric = joint_metrics[motor]
        rate_overshoots = [
            float(event["clip_overshoot"]) for event in metric["rate_events"]
        ]
        rate_summary = summary(rate_overshoots)
        p95 = rate_summary["p95"]
        maximum = rate_summary["max"]
        print(
            f"{motor:16s} {metric['rate_clip_count']:5d}  "
            f"{('-' if p95 is None else f'{p95:.6f}'):>13s}  "
            f"{('-' if maximum is None else f'{maximum:.6f}'):>13s}"
        )

    print("\n===== SOFT CLIP ATTRIBUTION =====")
    print(
        "joint             count   min   max  inherited  new_cross  "
        "over_p95  over_max  target_hold  target_in  target_out"
    )
    soft_joint_names: list[str] = []
    all_soft_directions: set[str] = set()
    for motor in motors:
        metric = joint_metrics[motor]
        soft_events = metric["soft_events"]
        if soft_events:
            soft_joint_names.append(motor)
        directions = Counter(str(event["direction"]) for event in soft_events)
        sources = Counter(str(event["boundary_source"]) for event in soft_events)
        target_motion = Counter(
            str(event["target_motion_relative_to_boundary"])
            for event in soft_events
        )
        substeps = Counter(int(event["substep"]) for event in soft_events)
        all_soft_directions.update(directions)
        overshoots = [float(event["clip_overshoot"]) for event in soft_events]
        local_distances = [
            float(event["local_distance_to_boundary_before"])
            for event in soft_events
        ]
        recorded_previous_distances = [
            float(event["recorded_previous_distance_to_boundary"])
            for event in soft_events
        ]
        target_distances = [
            float(event["recorded_target_distance_to_boundary"])
            for event in soft_events
        ]
        overshoot_summary = summary(overshoots)
        local_distance_summary = summary(local_distances)
        recorded_previous_distance_summary = summary(recorded_previous_distances)
        target_distance_summary = summary(target_distances)
        threshold_counts = {
            f"gt_{level:g}": sum(value > level for value in overshoots)
            for level in DIAGNOSTIC_OVERSHOOT_LEVELS
        }
        joint_report[motor] = {
            "rate_clip_count": metric["rate_clip_count"],
            "rate_clip_overshoot": summary(
                [float(event["clip_overshoot"]) for event in metric["rate_events"]]
            ),
            "soft_clip_count": metric["soft_clip_count"],
            "soft_direction_counts": dict(sorted(directions.items())),
            "soft_boundary_source_counts": dict(sorted(sources.items())),
            "soft_target_motion_counts": dict(sorted(target_motion.items())),
            "soft_substep_counts": {
                str(key): value for key, value in sorted(substeps.items())
            },
            "soft_overshoot": overshoot_summary,
            "soft_overshoot_diagnostic_threshold_counts": threshold_counts,
            "local_distance_to_boundary_before_soft_clip": local_distance_summary,
            "recorded_previous_distance_to_boundary": recorded_previous_distance_summary,
            "recorded_target_distance_to_boundary": target_distance_summary,
            "local_at_min_count_all_commands": metric["local_at_min_count"],
            "local_at_max_count_all_commands": metric["local_at_max_count"],
            "recorded_previous_at_min_count_all_commands": metric[
                "recorded_previous_at_min_count"
            ],
            "recorded_previous_at_max_count_all_commands": metric[
                "recorded_previous_at_max_count"
            ],
        }
        p95 = overshoot_summary["p95"]
        maximum = overshoot_summary["max"]
        print(
            f"{motor:16s} {metric['soft_clip_count']:5d} "
            f"{directions['min']:5d} {directions['max']:5d} "
            f"{sources['inherited_at_boundary']:10d} "
            f"{sources['new_crossing_from_interior']:10d} "
            f"{('-' if p95 is None else f'{p95:.6f}'):>9s} "
            f"{('-' if maximum is None else f'{maximum:.6f}'):>9s} "
            f"{target_motion['hold']:11d} "
            f"{target_motion['inward']:9d} "
            f"{target_motion['outward']:10d}"
        )

    print("\n===== SOFT OVERSHOOT DIAGNOSTIC BINS =====")
    print("These are descriptive bins, not physical safety thresholds.")
    print("joint             " + "  ".join(f">{level:g}" for level in DIAGNOSTIC_OVERSHOOT_LEVELS))
    for motor in motors:
        events_for_motor = joint_metrics[motor]["soft_events"]
        overshoots = [float(event["clip_overshoot"]) for event in events_for_motor]
        counts = [sum(value > level for value in overshoots) for level in DIAGNOSTIC_OVERSHOOT_LEVELS]
        print(f"{motor:16s} " + "  ".join(f"{value:6d}" for value in counts))

    print("\n===== ACTIVE SOFT-BOUNDARY EXPOSURE =====")
    for motor in soft_joint_names:
        metric = joint_metrics[motor]
        soft_events = metric["soft_events"]
        substeps = Counter(int(event["substep"]) for event in soft_events)
        print(
            f"{motor}: local_at_min={metric['local_at_min_count']} "
            f"local_at_max={metric['local_at_max_count']} "
            f"recorded_previous_at_min={metric['recorded_previous_at_min_count']} "
            f"recorded_previous_at_max={metric['recorded_previous_at_max_count']}"
        )
        print(
            f"{motor}: soft clips by substep="
            f"{[substeps[index] for index in range(5)]}"
        )

    if soft_joint_names == ["wrist_flex"] and all_soft_directions == {"max"}:
        decision = "WRIST_FLEX_MAX_BOUNDARY_PROJECTION_ATTRIBUTED_REVIEW_MAGNITUDES"
    else:
        decision = "NON_WRIST_OR_MIXED_SOFT_CLIPS_REQUIRE_REVIEW"

    output_dir.mkdir(parents=True, exist_ok=True)
    output_report_path = output_dir / "teacher_forced_clip_attribution_report.json"
    events_path = output_dir / "teacher_forced_clip_events.csv"
    require(
        not output_report_path.exists() and not events_path.exists(),
        "Refusing to overwrite existing clip-attribution outputs",
    )

    output_report = {
        "schema_version": OUTPUT_SCHEMA,
        "status": "PASS",
        "decision": decision,
        "scope": {
            "offline_postprocessing_only": True,
            "model_loaded": False,
            "dataset_loaded": False,
            "gpu_used": False,
            "hardware_accessed": False,
            "robot_object_created": False,
            "serial_port_opened": False,
            "command_sent": False,
            "hardware_deployment_authorized": False,
            "physical_safety_certified": False,
        },
        "diagnostic_bin_warning": (
            "Overshoot levels are descriptive bins only, not physical safety "
            "limits or deployment thresholds."
        ),
        "inputs": {
            "contract": str(contract_path),
            "local_report": str(local_report_path),
            "commands_csv": str(commands_path),
        },
        "integrity": {
            "commands_checked": total_commands,
            "commands_with_any_rate_clip": any_rate_commands,
            "commands_with_any_soft_clip": any_soft_commands,
            "max_guard_and_target_equation_error": max_equation_error,
            "report_counts_match": True,
            "recorded_target_soft_violation_count_by_joint": target_violations,
        },
        "soft_clip_joints": soft_joint_names,
        "soft_clip_directions": sorted(all_soft_directions),
        "joint_metrics": joint_report,
        "episodes": {
            str(episode): values for episode, values in sorted(episode_metrics.items())
        },
        "required_next_step": (
            "Review wrist boundary source, overshoot magnitudes, target motion, "
            "and rate-clip overshoots. Hardware remains blocked."
        ),
    }
    with output_report_path.open("w", encoding="utf-8") as handle:
        json.dump(output_report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    event_columns = [
        "event_type",
        "episode_index",
        "replan_index",
        "replan_frame",
        "substep",
        "dataset_frame",
        "joint_index",
        "joint",
        "direction",
        "clip_limit",
        "clip_overshoot",
        "recorded_previous_command",
        "local_previous_command",
        "recorded_target_delta",
        "raw_predicted_delta",
        "rate_limited_delta",
        "unbounded_command",
        "guarded_command",
        "recorded_target_command",
        "boundary_source",
        "local_distance_to_boundary_before",
        "recorded_previous_distance_to_boundary",
        "recorded_target_distance_to_boundary",
        "target_motion_relative_to_boundary",
        "prediction_motion_relative_to_boundary",
    ]
    with events_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=event_columns)
        writer.writeheader()
        for event in sorted(events, key=event_sort_key):
            writer.writerow({column: event.get(column) for column in event_columns})

    print("\n===== DECISION =====")
    print(f"decision='{decision}'")
    print("HARDWARE DEPLOYMENT REMAINS BLOCKED.")
    print("NO MODEL OR DATASET WAS LOADED. NO GPU WAS USED.")
    print("NO ROBOT OBJECT WAS CREATED. NO SERIAL PORT WAS OPENED.")
    print("NO COMMAND WAS SENT.")
    print("\n===== OUTPUT =====")
    print(output_report_path)
    print(events_path)
    print("ACT V3 DELTA TEACHER-FORCED CLIP ATTRIBUTION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
