#!/usr/bin/env python
"""Attribute guard interventions in an existing ACT v3 delta dry-run.

This program is deliberately post-processing only. It reads the frozen guard
contract plus the JSON/CSV emitted by ``dry_run_act_v3_delta_guarded_replay.py``.
It does not import LeRobot, Torch, camera, robot, motor-bus, teleoperator, or
serial modules; it cannot create a robot object or send a command.

The long counterfactual replay mixes recorded images/actual_q with recursively
simulated command state. Therefore this audit attributes numerical drift and
guard interventions, but it does not estimate real closed-loop task success or
authorize hardware deployment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


EXPECTED_CONTRACT_SCHEMA = "so101_act_v3_delta_runtime_limit_contract_v1"
EXPECTED_DRY_RUN_SCHEMA = "so101_act_v3_delta_guarded_dry_run_v1"
EXPECTED_FULL_DECISION = "FULL_DRY_RUN_COMPLETE_REVIEW_GUARD_INTERVENTIONS"
OUTPUT_SCHEMA = "so101_act_v3_delta_guard_intervention_attribution_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--dry-run-report", type=Path, required=True)
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
    require(
        math.isfinite(value),
        f"CSV row {row_number}: NaN/Inf in {column}",
    )
    return value


def integer(row: dict[str, str], column: str, row_number: int) -> int:
    require(column in row, f"CSV is missing column: {column}")
    try:
        return int(row[column])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"CSV row {row_number}: invalid integer in {column}: {row[column]!r}"
        ) from exc


def ordered_values(
    mapping: dict[str, Any], motors: list[str], name: str
) -> list[float]:
    require(isinstance(mapping, dict), f"{name} must be an object")
    require(set(mapping) == set(motors), f"{name} motor keys do not match")
    result = [float(mapping[motor]) for motor in motors]
    require(all(math.isfinite(value) for value in result), f"{name} is non-finite")
    return result


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


def close(actual: float, expected: float, tolerance: float) -> bool:
    return abs(actual - expected) <= tolerance


def event_sort_key(event: dict[str, Any]) -> tuple[int, int, int, int]:
    event_type_rank = 0 if event["event_type"] == "rate_clip" else 1
    return (
        int(event["episode_index"]),
        int(event["dataset_frame"]),
        int(event["joint_index"]),
        event_type_rank,
    )


def main() -> int:
    args = parse_args()
    contract_path = args.contract.expanduser().resolve()
    report_path = args.dry_run_report.expanduser().resolve()
    commands_path = args.commands_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    contract = load_json(contract_path)
    dry_run = load_json(report_path)
    require(commands_path.is_file(), f"Missing commands CSV: {commands_path}")

    print("===== VERIFY OFFLINE INPUT CONTRACTS =====")
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
        dry_run.get("schema_version") == EXPECTED_DRY_RUN_SCHEMA,
        "Unexpected dry-run report schema",
    )
    require(dry_run.get("status") == "PASS", "Dry-run report status is not PASS")
    require(
        dry_run.get("decision") == EXPECTED_FULL_DECISION,
        "Input is not the completed full dry-run report",
    )
    dry_scope = dry_run.get("scope", {})
    for key in (
        "hardware_accessed",
        "robot_object_created",
        "serial_port_opened",
        "command_sent",
        "hardware_deployment_authorized",
        "physical_safety_certified",
    ):
        require(dry_scope.get(key) is False, f"Dry-run scope violation: {key}")
    require(dry_scope.get("offline_only") is True, "Dry-run is not offline-only")

    motors = contract.get("robot", {}).get("motors")
    require(isinstance(motors, list) and len(motors) == 6, "Expected six motors")
    require(all(isinstance(motor, str) for motor in motors), "Invalid motor name")
    motors = list(motors)

    guard = contract.get("guard", {})
    tolerance = float(guard.get("numerical_tolerance"))
    require(math.isfinite(tolerance) and tolerance > 0.0, "Invalid tolerance")
    equation_tolerance = max(5.0 * tolerance, 1e-8)
    max_delta = ordered_values(
        guard.get("max_abs_delta_per_command", {}), motors, "max delta"
    )
    tracking_limit = ordered_values(
        guard.get("tracking_error_limit", {}), motors, "tracking limit"
    )
    soft_mapping = guard.get("normal_soft_limits", {})
    require(set(soft_mapping) == set(motors), "Soft-limit motor keys do not match")
    soft_min: list[float] = []
    soft_max: list[float] = []
    for motor in motors:
        limits = soft_mapping[motor]
        require(
            isinstance(limits, list) and len(limits) == 2,
            f"Invalid soft limits for {motor}",
        )
        low, high = float(limits[0]), float(limits[1])
        require(math.isfinite(low) and math.isfinite(high) and low < high, f"Invalid soft limits for {motor}")
        soft_min.append(low)
        soft_max.append(high)

    home = [float(value) for value in contract.get("startup", {}).get("required_previous_command", [])]
    require(len(home) == len(motors), "Invalid Home command")
    require(all(math.isfinite(value) for value in home), "Home contains NaN/Inf")
    fps = float(contract.get("policy", {}).get("dataset_fps"))
    execution_horizon = int(contract.get("policy", {}).get("n_action_steps"))
    replan_interval = int(contract.get("policy", {}).get("replan_interval_frames"))
    require(fps > 0.0, "Invalid dataset FPS")
    require(execution_horizon > 0, "Invalid execution horizon")
    require(replan_interval == execution_horizon, "Unexpected replan interval")

    reported_run = dry_run.get("run", {})
    require(reported_run.get("kind") == "full", "Dry-run report is not full")
    expected_commands = int(reported_run.get("total_commands"))
    expected_rate_any = int(reported_run.get("commands_with_any_rate_clip"))
    expected_soft_any = int(reported_run.get("commands_with_any_soft_clip"))
    require(expected_commands > 0, "Dry-run report has no commands")
    target_violations = dry_run.get(
        "recorded_validation_target_soft_violation_count_by_joint", {}
    )
    require(
        set(target_violations) == set(motors),
        "Recorded-target violation keys do not match motors",
    )
    require(
        all(int(target_violations[motor]) == 0 for motor in motors),
        "Recorded validation targets violate the frozen soft limits",
    )
    print("contract/report scope, schemas, and recorded target limits: PASS")

    required_base_columns = {
        "episode_index",
        "replan_index",
        "replan_frame",
        "substep",
        "dataset_frame",
        "any_rate_clipped",
        "any_soft_clipped",
    }
    value_fields = (
        "recorded_actual_q",
        "recorded_target_command",
        "raw_predicted_delta",
        "rate_limited_delta",
        "unbounded_command",
        "guarded_command",
        "actual_guarded_delta",
        "rate_clipped",
        "soft_clipped",
        "counterfactual_tracking_error",
    )

    total_commands = 0
    any_rate_commands = 0
    any_soft_commands = 0
    maximum_equation_error = 0.0
    maximum_tracking_recompute_error = 0.0
    events: list[dict[str, Any]] = []
    episode_stats: dict[int, dict[str, Any]] = {}
    joint_stats: dict[str, dict[str, Any]] = {
        motor: {
            "rate_clip_count": 0,
            "soft_clip_count": 0,
            "soft_min_clip_count": 0,
            "soft_max_clip_count": 0,
            "boundary_occupancy_count": 0,
            "soft_overshoots": [],
            "drift_before_soft_clip": [],
            "target_distance_to_boundary_at_soft_clip": [],
            "first_rate_clip": None,
            "first_soft_clip": None,
        }
        for motor in motors
    }

    current_episode: int | None = None
    previous_frame: int | None = None
    previous_guarded = list(home)
    previous_recorded_target = list(home)

    print("\n===== REPLAY CSV STRUCTURAL AND EQUATION AUDIT =====")
    with commands_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        require(required_base_columns <= fieldnames, "CSV base columns are incomplete")
        for motor in motors:
            for field in value_fields:
                require(
                    f"{field}.{motor}" in fieldnames,
                    f"CSV is missing {field}.{motor}",
                )

        for row_number, row in enumerate(reader, start=2):
            episode = integer(row, "episode_index", row_number)
            replan_index = integer(row, "replan_index", row_number)
            replan_frame = integer(row, "replan_frame", row_number)
            substep = integer(row, "substep", row_number)
            dataset_frame = integer(row, "dataset_frame", row_number)
            any_rate_csv = integer(row, "any_rate_clipped", row_number)
            any_soft_csv = integer(row, "any_soft_clipped", row_number)
            require(any_rate_csv in (0, 1), f"CSV row {row_number}: invalid any-rate flag")
            require(any_soft_csv in (0, 1), f"CSV row {row_number}: invalid any-soft flag")
            require(0 <= substep < execution_horizon, f"CSV row {row_number}: invalid substep")
            require(
                dataset_frame == replan_frame + substep,
                f"CSV row {row_number}: frame/replan/substep mismatch",
            )
            require(
                replan_frame == replan_index * replan_interval,
                f"CSV row {row_number}: replan index mismatch",
            )

            if episode != current_episode:
                require(
                    current_episode is None or episode > current_episode,
                    "Episodes are not strictly ordered",
                )
                require(dataset_frame == 0, f"Episode {episode} does not start at frame 0")
                current_episode = episode
                previous_frame = None
                previous_guarded = list(home)
                previous_recorded_target = list(home)
                episode_stats[episode] = {
                    "command_count": 0,
                    "commands_with_any_rate_clip": 0,
                    "commands_with_any_soft_clip": 0,
                    "first_any_rate_clip_frame": None,
                    "first_any_soft_clip_frame": None,
                    "first_tracking_limit_crossing_frame": None,
                    "max_command_state_drift_before": [0.0] * len(motors),
                    "joint_rate_clip_count": {motor: 0 for motor in motors},
                    "joint_soft_clip_count": {motor: 0 for motor in motors},
                }
            require(
                previous_frame is None or dataset_frame == previous_frame + 1,
                f"Episode {episode}: dataset frames are not contiguous",
            )

            per_joint: list[dict[str, Any]] = []
            row_has_rate = False
            row_has_soft = False
            row_tracking_cross = False

            for joint, motor in enumerate(motors):
                actual = finite_float(row, f"recorded_actual_q.{motor}", row_number)
                target = finite_float(row, f"recorded_target_command.{motor}", row_number)
                raw = finite_float(row, f"raw_predicted_delta.{motor}", row_number)
                rate_limited = finite_float(row, f"rate_limited_delta.{motor}", row_number)
                unbounded = finite_float(row, f"unbounded_command.{motor}", row_number)
                guarded = finite_float(row, f"guarded_command.{motor}", row_number)
                actual_delta = finite_float(row, f"actual_guarded_delta.{motor}", row_number)
                tracking = finite_float(row, f"counterfactual_tracking_error.{motor}", row_number)
                rate_flag = integer(row, f"rate_clipped.{motor}", row_number)
                soft_flag = integer(row, f"soft_clipped.{motor}", row_number)
                require(rate_flag in (0, 1), f"CSV row {row_number}: invalid rate flag")
                require(soft_flag in (0, 1), f"CSV row {row_number}: invalid soft flag")

                expected_rate = max(-max_delta[joint], min(max_delta[joint], raw))
                expected_unbounded = previous_guarded[joint] + expected_rate
                expected_guarded = max(
                    soft_min[joint], min(soft_max[joint], expected_unbounded)
                )
                expected_actual_delta = expected_guarded - previous_guarded[joint]
                expected_tracking = abs(expected_guarded - actual)
                expected_rate_flag = int(abs(raw - expected_rate) > tolerance)
                expected_soft_flag = int(
                    abs(expected_unbounded - expected_guarded) > tolerance
                )
                equation_errors = (
                    abs(rate_limited - expected_rate),
                    abs(unbounded - expected_unbounded),
                    abs(guarded - expected_guarded),
                    abs(actual_delta - expected_actual_delta),
                )
                maximum_equation_error = max(maximum_equation_error, *equation_errors)
                maximum_tracking_recompute_error = max(
                    maximum_tracking_recompute_error,
                    abs(tracking - expected_tracking),
                )
                require(
                    max(equation_errors) <= equation_tolerance,
                    f"CSV row {row_number}, {motor}: guard equation mismatch",
                )
                require(
                    abs(tracking - expected_tracking) <= equation_tolerance,
                    f"CSV row {row_number}, {motor}: tracking recompute mismatch",
                )
                require(rate_flag == expected_rate_flag, f"CSV row {row_number}, {motor}: rate flag mismatch")
                require(soft_flag == expected_soft_flag, f"CSV row {row_number}, {motor}: soft flag mismatch")

                command_state_drift = abs(
                    previous_guarded[joint] - previous_recorded_target[joint]
                )
                episode_stats[episode]["max_command_state_drift_before"][joint] = max(
                    episode_stats[episode]["max_command_state_drift_before"][joint],
                    command_state_drift,
                )
                if tracking > tracking_limit[joint]:
                    row_tracking_cross = True

                at_min = close(guarded, soft_min[joint], tolerance)
                at_max = close(guarded, soft_max[joint], tolerance)
                joint_stats[motor]["boundary_occupancy_count"] += int(at_min or at_max)

                details = {
                    "joint_index": joint,
                    "joint": motor,
                    "recorded_actual_q": actual,
                    "recorded_target_command": target,
                    "raw_predicted_delta": raw,
                    "rate_limited_delta": rate_limited,
                    "unbounded_command": unbounded,
                    "guarded_command": guarded,
                    "actual_guarded_delta": actual_delta,
                    "counterfactual_tracking_error": tracking,
                    "command_state_drift_before": command_state_drift,
                    "rate_clipped": rate_flag,
                    "soft_clipped": soft_flag,
                }
                per_joint.append(details)

                if rate_flag:
                    row_has_rate = True
                    joint_stats[motor]["rate_clip_count"] += 1
                    episode_stats[episode]["joint_rate_clip_count"][motor] += 1
                    event = {
                        "event_type": "rate_clip",
                        "episode_index": episode,
                        "dataset_frame": dataset_frame,
                        "elapsed_seconds": dataset_frame / fps,
                        "replan_index": replan_index,
                        "substep": substep,
                        **details,
                        "direction": "positive" if raw > 0.0 else "negative",
                        "limit_value": max_delta[joint],
                        "overshoot": abs(raw) - max_delta[joint],
                        "target_distance_to_hit_boundary": None,
                    }
                    events.append(event)
                    if joint_stats[motor]["first_rate_clip"] is None:
                        joint_stats[motor]["first_rate_clip"] = event

                if soft_flag:
                    row_has_soft = True
                    direction = "max" if unbounded > soft_max[joint] else "min"
                    boundary = soft_max[joint] if direction == "max" else soft_min[joint]
                    overshoot = abs(unbounded - boundary)
                    target_distance = (
                        boundary - target if direction == "max" else target - boundary
                    )
                    joint_stats[motor]["soft_clip_count"] += 1
                    joint_stats[motor][f"soft_{direction}_clip_count"] += 1
                    joint_stats[motor]["soft_overshoots"].append(overshoot)
                    joint_stats[motor]["drift_before_soft_clip"].append(command_state_drift)
                    joint_stats[motor]["target_distance_to_boundary_at_soft_clip"].append(target_distance)
                    episode_stats[episode]["joint_soft_clip_count"][motor] += 1
                    event = {
                        "event_type": "soft_clip",
                        "episode_index": episode,
                        "dataset_frame": dataset_frame,
                        "elapsed_seconds": dataset_frame / fps,
                        "replan_index": replan_index,
                        "substep": substep,
                        **details,
                        "direction": direction,
                        "limit_value": boundary,
                        "overshoot": overshoot,
                        "target_distance_to_hit_boundary": target_distance,
                    }
                    events.append(event)
                    if joint_stats[motor]["first_soft_clip"] is None:
                        joint_stats[motor]["first_soft_clip"] = event

            require(
                any_rate_csv == int(row_has_rate),
                f"CSV row {row_number}: any-rate flag mismatch",
            )
            require(
                any_soft_csv == int(row_has_soft),
                f"CSV row {row_number}: any-soft flag mismatch",
            )
            any_rate_commands += int(row_has_rate)
            any_soft_commands += int(row_has_soft)
            total_commands += 1
            episode_stats[episode]["command_count"] += 1
            episode_stats[episode]["commands_with_any_rate_clip"] += int(row_has_rate)
            episode_stats[episode]["commands_with_any_soft_clip"] += int(row_has_soft)
            if row_has_rate and episode_stats[episode]["first_any_rate_clip_frame"] is None:
                episode_stats[episode]["first_any_rate_clip_frame"] = dataset_frame
            if row_has_soft and episode_stats[episode]["first_any_soft_clip_frame"] is None:
                episode_stats[episode]["first_any_soft_clip_frame"] = dataset_frame
            if row_tracking_cross and episode_stats[episode]["first_tracking_limit_crossing_frame"] is None:
                episode_stats[episode]["first_tracking_limit_crossing_frame"] = dataset_frame

            previous_guarded = [details["guarded_command"] for details in per_joint]
            previous_recorded_target = [
                details["recorded_target_command"] for details in per_joint
            ]
            previous_frame = dataset_frame

    require(total_commands == expected_commands, "CSV/report total command mismatch")
    require(any_rate_commands == expected_rate_any, "CSV/report rate-clip mismatch")
    require(any_soft_commands == expected_soft_any, "CSV/report soft-clip mismatch")
    print(f"commands checked: {total_commands}")
    print(f"guard equation max error: {maximum_equation_error:.10f}")
    print(f"tracking recompute max error: {maximum_tracking_recompute_error:.10f}")
    print("CSV ordering, guard equations, flags, and report counts: PASS")

    print("\n===== INTERVENTION ATTRIBUTION BY EPISODE =====")
    print("episode  commands  rate_any  soft_any  first_track  first_rate  first_soft")
    for episode, values in sorted(episode_stats.items()):
        print(
            f"{episode:7d}  {values['command_count']:8d}  "
            f"{values['commands_with_any_rate_clip']:8d}  "
            f"{values['commands_with_any_soft_clip']:8d}  "
            f"{str(values['first_tracking_limit_crossing_frame']):>11}  "
            f"{str(values['first_any_rate_clip_frame']):>10}  "
            f"{str(values['first_any_soft_clip_frame']):>10}"
        )

    print("\n===== INTERVENTION ATTRIBUTION BY JOINT =====")
    print(
        "joint             rate  soft   min   max  boundary  "
        "first_soft  drift_p50  target_dist_p50"
    )
    joint_report: dict[str, Any] = {}
    soft_events = [event for event in events if event["event_type"] == "soft_clip"]
    rate_events = [event for event in events if event["event_type"] == "rate_clip"]
    for motor in motors:
        values = joint_stats[motor]
        first_soft = values["first_soft_clip"]
        first_soft_label = (
            "None"
            if first_soft is None
            else f"e{first_soft['episode_index']}:f{first_soft['dataset_frame']}"
        )
        drift_summary = summary(values["drift_before_soft_clip"])
        target_distance_summary = summary(
            values["target_distance_to_boundary_at_soft_clip"]
        )
        joint_report[motor] = {
            "rate_clip_count": values["rate_clip_count"],
            "soft_clip_count": values["soft_clip_count"],
            "soft_min_clip_count": values["soft_min_clip_count"],
            "soft_max_clip_count": values["soft_max_clip_count"],
            "boundary_occupancy_count": values["boundary_occupancy_count"],
            "soft_overshoot": summary(values["soft_overshoots"]),
            "command_state_drift_before_soft_clip": drift_summary,
            "recorded_target_distance_to_hit_boundary_at_soft_clip": target_distance_summary,
            "first_rate_clip": values["first_rate_clip"],
            "first_soft_clip": values["first_soft_clip"],
        }
        drift_p50 = drift_summary["p50"]
        target_p50 = target_distance_summary["p50"]
        print(
            f"{motor:16s} {values['rate_clip_count']:5d} "
            f"{values['soft_clip_count']:5d} "
            f"{values['soft_min_clip_count']:5d} "
            f"{values['soft_max_clip_count']:5d} "
            f"{values['boundary_occupancy_count']:9d}  "
            f"{first_soft_label:>10s}  "
            f"{('-' if drift_p50 is None else f'{drift_p50:.4f}'):>9s}  "
            f"{('-' if target_p50 is None else f'{target_p50:.4f}'):>15s}"
        )

    soft_after_drift_1 = sum(
        float(event["command_state_drift_before"]) > 1.0 for event in soft_events
    )
    soft_after_tracking_limit = sum(
        float(event["command_state_drift_before"])
        > tracking_limit[int(event["joint_index"])]
        for event in soft_events
    )
    soft_without_rate_clip = sum(
        int(event["rate_clipped"]) == 0 for event in soft_events
    )
    event_summary = {
        "rate_clip_joint_events": len(rate_events),
        "soft_clip_joint_events": len(soft_events),
        "soft_clip_events_without_same_joint_rate_clip": soft_without_rate_clip,
        "soft_clip_events_after_command_state_drift_gt_1": soft_after_drift_1,
        "soft_clip_events_after_command_state_drift_gt_joint_tracking_limit": soft_after_tracking_limit,
    }

    print("\n===== COUNTERFACTUAL INTERPRETATION =====")
    print(
        "Recorded images and actual_q follow the demonstration, while the command "
        "state above is recursively simulated."
    )
    print(
        f"soft-clip joint events: {len(soft_events)}; "
        f"without same-joint rate clip: {soft_without_rate_clip}; "
        f"after command-state drift >1: {soft_after_drift_1}; "
        f"after drift >tracking limit: {soft_after_tracking_limit}"
    )
    print(
        "The tracking columns are mismatch diagnostics for this hybrid replay, "
        "not predictions of a live tracking trip."
    )

    if soft_events or rate_events:
        decision = "RUN_TEACHER_FORCED_LOCAL_GUARD_REPLAY_NEXT"
    else:
        decision = "NO_GUARD_INTERVENTION_FOUND_REVIEW_BEFORE_NEXT_STAGE"

    output_dir.mkdir(parents=True, exist_ok=True)
    output_report_path = output_dir / "guard_intervention_attribution_report.json"
    events_path = output_dir / "guard_intervention_events.csv"
    require(
        not output_report_path.exists() and not events_path.exists(),
        "Refusing to overwrite existing attribution outputs",
    )

    episode_report = {
        str(episode): values for episode, values in sorted(episode_stats.items())
    }
    output_report = {
        "schema_version": OUTPUT_SCHEMA,
        "status": "PASS",
        "decision": decision,
        "scope": {
            "offline_postprocessing_only": True,
            "model_loaded": False,
            "dataset_loaded": False,
            "hardware_accessed": False,
            "robot_object_created": False,
            "serial_port_opened": False,
            "command_sent": False,
            "hardware_deployment_authorized": False,
            "physical_safety_certified": False,
        },
        "counterfactual_warning": (
            "Recorded images and actual_q did not react to recursively simulated "
            "commands. Attribution does not estimate live closed-loop behavior."
        ),
        "inputs": {
            "contract": str(contract_path),
            "dry_run_report": str(report_path),
            "commands_csv": str(commands_path),
        },
        "integrity": {
            "commands_checked": total_commands,
            "max_guard_equation_error": maximum_equation_error,
            "max_tracking_recompute_error": maximum_tracking_recompute_error,
            "reported_count_match": True,
            "recorded_validation_target_soft_violations": target_violations,
        },
        "event_summary": event_summary,
        "joint_metrics": joint_report,
        "episodes": episode_report,
        "required_next_step": (
            "Run a separate teacher-forced local five-command replay that resets "
            "the command state to recorded in-distribution state at every replan. "
            "Do not access hardware."
        ),
    }
    with output_report_path.open("w", encoding="utf-8") as handle:
        json.dump(output_report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    event_columns = [
        "event_type",
        "episode_index",
        "dataset_frame",
        "elapsed_seconds",
        "replan_index",
        "substep",
        "joint_index",
        "joint",
        "direction",
        "limit_value",
        "overshoot",
        "recorded_actual_q",
        "recorded_target_command",
        "raw_predicted_delta",
        "rate_limited_delta",
        "unbounded_command",
        "guarded_command",
        "actual_guarded_delta",
        "counterfactual_tracking_error",
        "command_state_drift_before",
        "target_distance_to_hit_boundary",
        "rate_clipped",
        "soft_clipped",
    ]
    with events_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=event_columns)
        writer.writeheader()
        for event in sorted(events, key=event_sort_key):
            writer.writerow({column: event.get(column) for column in event_columns})

    print("\n===== DECISION =====")
    print(f"decision='{decision}'")
    print("HARDWARE DEPLOYMENT REMAINS BLOCKED.")
    print("NO MODEL OR DATASET WAS LOADED.")
    print("NO ROBOT OBJECT WAS CREATED. NO SERIAL PORT WAS OPENED.")
    print("NO COMMAND WAS SENT.")
    print("\n===== OUTPUT =====")
    print(output_report_path)
    print(events_path)
    print("ACT V3 DELTA GUARD INTERVENTION ATTRIBUTION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
