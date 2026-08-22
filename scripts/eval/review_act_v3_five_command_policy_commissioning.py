#!/usr/bin/env python3
"""Offline acceptance review for the completed five-command commissioning run.

This reviewer performs no hardware, camera, model, CUDA, or vendor imports.  It
accepts only the preserved run whose five policy writes, tracking, Home return,
and torque/serial cleanup completed, while the final camera END/ACK collection
failed solely because a 40-second producer duration was paired with a
15-second finish grace after an approximately 20-second motion session.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


MOTOR_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

EXPECTED_COMMISSIONING_SHA256 = (
    "8c9d589feaf8f7409002d60d4156f11934651eeeb832c2e8be16a45d40ce687a"
)
EXPECTED_PREVIOUS_FIVE_SHA256 = (
    "7606293066b081b5d2611eec84344b58be0a354f3864e8861456879d2ef819af"
)
EXPECTED_RUN_DECISION = (
    "FIVE_COMMAND_POLICY_COMMISSIONING_FAILED_DEPLOYMENT_REMAINS_BLOCKED"
)
REVIEW_DECISION = (
    "FIVE_COMMAND_POLICY_COMMISSIONING_REVIEW_ACCEPTED_"
    "TEARDOWN_ONLY_PREPARE_ONE_EPISODE_NEXT"
)
EXPECTED_CAMERA_FAILURES = {
    "front_receiver_missing_end_ack",
    "front_receiver_timeout",
    "front_span_ratio_below_0.90",
    "front_windows_worker_failed",
    "front_windows_worker_missing_end_ack",
    "wrist_receiver_missing_end_ack",
    "wrist_receiver_timeout",
    "wrist_span_ratio_below_0.90",
    "wrist_windows_worker_failed",
    "wrist_windows_worker_missing_end_ack",
}
POLICY_COMMAND_COUNT = 5
MAX_LIVE_INFERENCE_MS = 500.0
MAX_CAMERA_AGE_MS = 250.0
MAX_JOINT_AGE_MS = 100.0
MAX_CAMERA_SKEW_MS = 100.0
MIN_CAMERA_FPS = 14.0
MAX_CAMERA_GAP_MS = 250.0
NUMERIC_TOLERANCE = 1e-6


class ReviewError(RuntimeError):
    """The preserved run does not satisfy the offline acceptance contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commissioning-script", type=Path, required=True)
    parser.add_argument("--run-report", type=Path, required=True)
    parser.add_argument("--trace-csv", type=Path, required=True)
    parser.add_argument("--plan-csv", type=Path, required=True)
    parser.add_argument("--front-frames-csv", type=Path, required=True)
    parser.add_argument("--wrist-frames-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def require_file(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReviewError(f"invalid JSON {path}: {exc}") from exc


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ReviewError(f"CSV is empty: {path}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ReviewError(f"CSV schema changes within {path}")
    return rows


def finite_float(label: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ReviewError(f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise ReviewError(f"{label} is nonfinite")
    return result


def finite_vector(label: str, values: Sequence[Any]) -> tuple[float, ...]:
    vector = tuple(finite_float(f"{label}[{index}]", value) for index, value in enumerate(values))
    if len(vector) != len(MOTOR_ORDER):
        raise ReviewError(f"{label} must contain six values")
    return vector


def verify_static_offline_boundary(path: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    allowed = {
        "__future__",
        "argparse",
        "ast",
        "csv",
        "hashlib",
        "json",
        "math",
        "sys",
        "datetime",
        "pathlib",
        "typing",
    }
    unexpected = sorted(imports - allowed)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    forbidden_attributes = {
        "connect",
        "disconnect",
        "send_action",
        "sync_read",
        "sync_write",
        "write_goal_position",
        "enable_torque",
        "disable_torque",
        "from_pretrained",
        "predict_action_chunk",
        "VideoCapture",
    }
    found = sorted(
        {
            node.func.attr
            for node in calls
            if isinstance(node.func, ast.Attribute)
            and node.func.attr in forbidden_attributes
        }
    )
    if unexpected or found:
        raise ReviewError(
            f"offline boundary mismatch: imports={unexpected}, calls={found}"
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "standard_library_only": True,
        "hardware_or_model_calls": False,
        "pass": True,
    }


def verify_source_contract(path: Path) -> dict[str, Any]:
    actual_sha = sha256_file(path)
    if actual_sha != EXPECTED_COMMISSIONING_SHA256:
        raise ReviewError(
            "commissioning source SHA256 mismatch: "
            f"expected={EXPECTED_COMMISSIONING_SHA256}, actual={actual_sha}"
        )
    source = path.read_text(encoding="utf-8")
    required_fragments = {
        "camera_duration_40": "CAMERA_STREAM_SECONDS = 40.0",
        "finish_grace_15": "CAMERA_FINISH_GRACE_SECONDS = 15.0",
        "finish_uses_short_grace": (
            "camera_session.finish(CAMERA_FINISH_GRACE_SECONDS)"
        ),
        "exactly_five_policy_commands": "POLICY_COMMAND_COUNT = 5",
        "previous_zero_write_source_pinned": EXPECTED_PREVIOUS_FIVE_SHA256,
        "normal_cleanup_disables_torque": "disable_torque(home, bus)",
        "normal_cleanup_closes_serial": "disconnect_bus(home, bus)",
    }
    checks = {name: fragment in source for name, fragment in required_fragments.items()}
    if not all(checks.values()):
        raise ReviewError(f"commissioning source contract mismatch: {checks}")
    return {"path": str(path), "sha256": actual_sha, "checks": checks}


def verify_run_report(report_path: Path, source_sha: str) -> dict[str, Any]:
    report = load_json(report_path)
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise ReviewError("unexpected commissioning report schema")
    authorization = report.get("authorization") or {}
    scope = report.get("scope") or {}
    frozen = report.get("frozen_dependencies") or {}
    run = report.get("run") or {}
    failure = report.get("failure") or {}
    contract = report.get("commissioning_contract") or {}
    writes = scope.get("goal_position_writes") or {}
    static = report.get("static_capability_boundary") or {}
    camera_failures = set(report.get("camera_acceptance_failures") or [])
    failed_gate = frozen.get("failed_five_command_gate") or {}
    controlled_home = frozen.get("controlled_home_review") or {}
    guard_audit = frozen.get("guard_self_audit") or {}
    release = frozen.get("release") or {}

    failure_message = str(failure.get("message", ""))
    checks = {
        "source_identity": static.get("sha256") == source_sha,
        "expected_failed_decision": report.get("decision") == EXPECTED_RUN_DECISION,
        "camera_failure_only": (
            failure.get("type") == "CameraAcceptanceError"
            and camera_failures == EXPECTED_CAMERA_FAILURES
            and set(failure_message.split("; ")) == EXPECTED_CAMERA_FAILURES
        ),
        "physical_confirmations": all(
            (authorization.get("confirmations") or {}).values()
        ),
        "authorization_scope": (
            int(authorization.get("authorized_live_policy_inferences", -1)) == 1
            and int(authorization.get("authorized_policy_commands", -1)) == 5
            and authorization.get("autonomous_episode_authorized") is False
            and authorization.get("hardware_deployment_authorized") is False
        ),
        "exact_policy_writes": (
            int(scope.get("policy_command_write_attempts", -1)) == 5
            and int(scope.get("policy_command_writes_acknowledged", -1)) == 5
            and int(scope.get("policy_commands_sent", -1)) == 5
            and int(writes.get("policy_attempted", -1)) == 5
            and int(writes.get("policy", -1)) == 5
        ),
        "single_live_inference": (
            int(scope.get("dummy_warmup_inferences", -1)) == 1
            and int(scope.get("live_policy_inferences", -1)) == 1
            and finite_float("inference_ms", run.get("inference_ms"))
            <= MAX_LIVE_INFERENCE_MS
        ),
        "motion_scope": (
            scope.get("follower_serial_opened") is True
            and scope.get("operating_mode_read") is True
            and scope.get("torque_enable_attempted") is True
            and scope.get("full_autonomous_loop_run") is False
        ),
        "cleanup": (
            scope.get("torque_disabled_at_exit") is True
            and scope.get("torque_may_be_enabled_after_cleanup") is False
            and scope.get("serial_closed") is True
            and run.get("cleanup_errors") == []
        ),
        "five_step_plan_completed": (
            int(run.get("policy_steps_planned", -1)) == 5
            and int(run.get("policy_steps_sent", -1)) == 5
        ),
        "final_pose_present": run.get("final_q_before_torque_disable") is not None,
        "prior_zero_write_gate": (
            failed_gate.get("decision")
            == "ZERO_WRITE_FAILURE_REVIEWED_FOR_MERGED_HOME_RECOVERY"
            and failed_gate.get("checks")
            and all(failed_gate["checks"].values())
            and (
                frozen.get("failed_five_command_gate", {})
                .get("checks", {})
                .get("source_identity")
                is True
            )
        ),
        "controlled_home_review": (
            controlled_home.get("checks")
            and all(controlled_home["checks"].values())
        ),
        "guard_review": (
            int(guard_audit.get("tests_passed", -1))
            == int(guard_audit.get("tests_total", -2))
            == 11
        ),
        "frozen_release_11k": int(release.get("selected_step", -1)) == 11000,
        "next_stage_not_previously_authorized": (
            report.get("next_gate_authorized") is False
            and report.get("autonomous_episode_authorized") is False
            and report.get("hardware_deployment_authorized") is False
        ),
        "camera_duration_contract": (
            float((report.get("camera_start") or {}).get("stream_duration_seconds", -1))
            == 40.0
        ),
        "report_policy_count": int(contract.get("policy_command_count", -1)) == 5,
    }
    if not all(checks.values()):
        raise ReviewError(f"commissioning report review rejected: {checks}")
    return {"report": report, "checks": checks, "sha256": sha256_file(report_path)}


def verify_trace(
    rows: list[dict[str, str]],
    report: dict[str, Any],
) -> dict[str, Any]:
    run = report["run"]
    contract = report["commissioning_contract"]
    home = finite_vector(
        "Home command",
        report["frozen_dependencies"]["controlled_home_review"]["home_command"],
    )
    if len(rows) != int(run.get("trace_rows", -1)):
        raise ReviewError("trace row count differs from report")
    phases: dict[str, list[dict[str, str]]] = {}
    max_tracking = [0.0] * len(MOTOR_ORDER)
    for row in rows:
        phase = row["phase"]
        phases.setdefault(phase, []).append(row)
        if int(row["tracking_tripped"]) != 0:
            raise ReviewError(f"tracking trip found in phase={phase}")
        command = finite_vector(
            "trace command",
            [row[f"command_{motor}"] for motor in MOTOR_ORDER],
        )
        actual = finite_vector(
            "trace actual",
            [row[f"actual_{motor}"] for motor in MOTOR_ORDER],
        )
        for index, motor in enumerate(MOTOR_ORDER):
            logged = finite_float(
                f"tracking_error_{motor}", row[f"tracking_error_{motor}"]
            )
            expected = abs(command[index] - actual[index])
            if abs(logged - expected) > NUMERIC_TOLERANCE:
                raise ReviewError(f"tracking equation mismatch for {motor}")
            max_tracking[index] = max(max_tracking[index], logged)
    required_phases = {
        "home_align",
        "home_align_hold",
        "policy",
        "policy_hold",
        "return_home",
        "final_home_hold",
    }
    if set(phases) != required_phases:
        raise ReviewError(f"unexpected trace phases: {sorted(phases)}")
    for phase, phase_rows in phases.items():
        sequences = [int(row["sequence"]) for row in phase_rows]
        if sequences != list(range(1, len(phase_rows) + 1)):
            raise ReviewError(f"noncontiguous sequence in phase={phase}")
    policy_rows = phases["policy"]
    if len(policy_rows) != POLICY_COMMAND_COUNT:
        raise ReviewError("trace does not contain exactly five policy rows")
    strict_limits_raw = contract.get("commissioning_tracking_limits") or {}
    strict_limits = finite_vector(
        "commissioning tracking limits",
        [strict_limits_raw[motor] for motor in MOTOR_ORDER],
    )
    for row in policy_rows + phases["policy_hold"] + phases["final_home_hold"]:
        for index, motor in enumerate(MOTOR_ORDER):
            if finite_float(
                f"strict tracking {motor}", row[f"tracking_error_{motor}"]
            ) > strict_limits[index] + NUMERIC_TOLERANCE:
                raise ReviewError(f"strict tracking limit exceeded for {motor}")
    if any(
        int(row["policy_derived"]) != 1
        or int(row["any_rate_clip"]) != 0
        or int(row["any_soft_clip"]) != 0
        for row in policy_rows
    ):
        raise ReviewError("policy trace flags are not clean")
    final_row = rows[-1]
    if final_row["phase"] != "final_home_hold":
        raise ReviewError("trace does not end in final Home hold")
    final_actual = finite_vector(
        "trace final actual",
        [final_row[f"actual_{motor}"] for motor in MOTOR_ORDER],
    )
    report_final = finite_vector(
        "report final actual", run["final_q_before_torque_disable"]
    )
    if max(abs(a - b) for a, b in zip(final_actual, report_final, strict=True)) > NUMERIC_TOLERANCE:
        raise ReviewError("trace final pose differs from report")
    final_errors = tuple(abs(a - b) for a, b in zip(final_actual, home, strict=True))
    if max(final_errors) > float(contract["final_home_tolerance"]):
        raise ReviewError("final Home error exceeds commissioning tolerance")
    return {
        "rows": len(rows),
        "phase_counts": {phase: len(items) for phase, items in phases.items()},
        "policy_rows": len(policy_rows),
        "tracking_trip_rows": 0,
        "rate_or_soft_clip_policy_rows": 0,
        "max_tracking_error_by_joint": dict(zip(MOTOR_ORDER, max_tracking, strict=True)),
        "final_error_by_joint": dict(zip(MOTOR_ORDER, final_errors, strict=True)),
        "max_final_home_error": max(final_errors),
        "policy_rows_raw": policy_rows,
    }


def verify_plan(
    rows: list[dict[str, str]],
    trace_review: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    if len(rows) != POLICY_COMMAND_COUNT:
        raise ReviewError("policy plan must contain five rows")
    if [int(row["substep"]) for row in rows] != list(range(POLICY_COMMAND_COUNT)):
        raise ReviewError("policy plan substeps are not 0..4")
    home = finite_vector(
        "Home command",
        report["frozen_dependencies"]["controlled_home_review"]["home_command"],
    )
    excursion_raw = report["commissioning_contract"][
        "policy_excursion_limits_from_home"
    ]
    excursion_limits = finite_vector(
        "policy excursion limits",
        [excursion_raw[motor] for motor in MOTOR_ORDER],
    )
    prior = home
    trace_policy = trace_review["policy_rows_raw"]
    max_excursion = [0.0] * len(MOTOR_ORDER)
    for index, (plan, trace) in enumerate(zip(rows, trace_policy, strict=True)):
        if (
            int(plan["write_attempted"]) != 1
            or int(plan["write_acknowledged"]) != 1
            or int(plan["any_rate_clip"]) != 0
            or int(plan["any_soft_clip"]) != 0
        ):
            raise ReviewError(f"policy plan flags failed at substep={index}")
        raw_delta = finite_vector(
            "plan raw delta",
            [plan[f"raw_delta_{motor}"] for motor in MOTOR_ORDER],
        )
        command = finite_vector(
            "plan command",
            [plan[f"command_{motor}"] for motor in MOTOR_ORDER],
        )
        trace_command = finite_vector(
            "trace policy command",
            [trace[f"command_{motor}"] for motor in MOTOR_ORDER],
        )
        trace_delta = finite_vector(
            "trace policy delta",
            [trace[f"guarded_delta_{motor}"] for motor in MOTOR_ORDER],
        )
        if max(abs(a - b) for a, b in zip(command, trace_command, strict=True)) > NUMERIC_TOLERANCE:
            raise ReviewError(f"plan/trace command mismatch at substep={index}")
        if max(abs(a - b) for a, b in zip(raw_delta, trace_delta, strict=True)) > NUMERIC_TOLERANCE:
            raise ReviewError(f"plan/trace delta mismatch at substep={index}")
        reconstructed = tuple(a + b for a, b in zip(prior, raw_delta, strict=True))
        if max(abs(a - b) for a, b in zip(reconstructed, command, strict=True)) > 1e-5:
            raise ReviewError(f"recursive delta equation mismatch at substep={index}")
        for joint, motor in enumerate(MOTOR_ORDER):
            excursion = abs(command[joint] - home[joint])
            logged = finite_float(
                f"logged excursion {motor}", plan[f"excursion_from_home_{motor}"]
            )
            if abs(excursion - logged) > NUMERIC_TOLERANCE:
                raise ReviewError(f"excursion equation mismatch for {motor}")
            if excursion > excursion_limits[joint] + NUMERIC_TOLERANCE:
                raise ReviewError(f"policy excursion exceeded for {motor}")
            max_excursion[joint] = max(max_excursion[joint], excursion)
        prior = command
    return {
        "rows": len(rows),
        "write_attempts": 5,
        "write_acknowledged": 5,
        "rate_or_soft_clips": 0,
        "recursive_delta_equation": "PASS",
        "max_excursion_from_home_by_joint": dict(
            zip(MOTOR_ORDER, max_excursion, strict=True)
        ),
    }


def verify_camera_frames(
    *,
    role: str,
    rows: list[dict[str, str]],
    report: dict[str, Any],
) -> dict[str, Any]:
    sequences = [int(row["sequence"]) for row in rows]
    if sequences != list(range(sequences[0], sequences[0] + len(sequences))):
        raise ReviewError(f"{role} frame sequence has gaps")
    if any(
        (int(row["width"]), int(row["height"]), int(row["channels"]))
        != (640, 480, 3)
        for row in rows
    ):
        raise ReviewError(f"{role} frame dimensions changed")
    camera = report["camera_transport"]["cameras"][role]
    metrics = camera.get("metrics") or {}
    integrity = {
        "decode_failures": int(camera.get("decode_failures", -1)) == 0,
        "crc_failures": int(camera.get("crc_failures", -1)) == 0,
        "protocol_failures": int(camera.get("protocol_failures", -1)) == 0,
        "sequence_gap_count": int(camera.get("sequence_gap_count", -1)) == 0,
        "fps": finite_float(f"{role} fps", metrics.get("achieved_receive_fps"))
        >= MIN_CAMERA_FPS,
        "gap": finite_float(f"{role} max gap", metrics.get("gap_ms_max"))
        <= MAX_CAMERA_GAP_MS,
        "dimensions": metrics.get("unique_dimensions") == [[640, 480, 3]],
        "frame_count": int(metrics.get("successful_frames", -1)) == len(rows),
        "partial_span_expected": finite_float(
            f"{role} span ratio", metrics.get("receive_span_ratio")
        )
        < 0.90,
    }
    if not all(integrity.values()):
        raise ReviewError(f"{role} camera evidence rejected: {integrity}")
    return {
        "frames": len(rows),
        "first_sequence": sequences[0],
        "last_sequence": sequences[-1],
        "achieved_receive_fps": float(metrics["achieved_receive_fps"]),
        "receive_span_seconds": float(metrics["receive_span_seconds"]),
        "receive_span_ratio": float(metrics["receive_span_ratio"]),
        "gap_ms_max": float(metrics["gap_ms_max"]),
        "integrity_checks": integrity,
    }


def verify_live_observation(
    report: dict[str, Any],
    front_rows: list[dict[str, str]],
    wrist_rows: list[dict[str, str]],
) -> dict[str, Any]:
    assembly = report["run"].get("assembly_metrics") or {}
    front_sequences = {int(row["sequence"]) for row in front_rows}
    wrist_sequences = {int(row["sequence"]) for row in wrist_rows}
    checks = {
        "front_frame_present": int(assembly.get("front_sequence", -1))
        in front_sequences,
        "wrist_frame_present": int(assembly.get("wrist_sequence", -1))
        in wrist_sequences,
        "front_fresh": finite_float("front age", assembly.get("front_age_ms"))
        <= MAX_CAMERA_AGE_MS,
        "wrist_fresh": finite_float("wrist age", assembly.get("wrist_age_ms"))
        <= MAX_CAMERA_AGE_MS,
        "joint_fresh": finite_float("joint age", assembly.get("joint_age_ms"))
        <= MAX_JOINT_AGE_MS,
        "capture_skew": finite_float(
            "capture skew", assembly.get("capture_skew_ms")
        )
        <= MAX_CAMERA_SKEW_MS,
        "receive_skew": finite_float(
            "receive skew", assembly.get("receive_skew_ms")
        )
        <= MAX_CAMERA_SKEW_MS,
        "state_dimension": int(assembly.get("state_dimension", -1)) == 18,
    }
    transport_skew = report["camera_transport"].get("capture_skew") or {}
    checks["stream_capture_skew"] = finite_float(
        "stream capture skew p95", transport_skew.get("skew_ms_p95")
    ) <= MAX_CAMERA_SKEW_MS
    if not all(checks.values()):
        raise ReviewError(f"live observation evidence rejected: {checks}")
    return {"metrics": assembly, "checks": checks}


def write_report(path: Path, report: dict[str, Any]) -> None:
    resolved = path.expanduser().resolve()
    if resolved.exists():
        raise FileExistsError(
            f"output already exists; preserve it and choose another path: {resolved}"
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    script_path = Path(__file__).resolve()
    commissioning_path = require_file(args.commissioning_script)
    run_report_path = require_file(args.run_report)
    trace_path = require_file(args.trace_csv)
    plan_path = require_file(args.plan_csv)
    front_path = require_file(args.front_frames_csv)
    wrist_path = require_file(args.wrist_frames_csv)

    print("===== STATIC OFFLINE REVIEW BOUNDARY =====")
    static = verify_static_offline_boundary(script_path)
    source = verify_source_contract(commissioning_path)
    print("standard-library only; no hardware, camera, model or CUDA calls: PASS")

    print("\n===== VERIFY PRESERVED COMMISSIONING REPORT =====")
    reviewed = verify_run_report(run_report_path, source["sha256"])
    report = reviewed["report"]
    print("exact 5/5 policy writes, single inference and safe cleanup: PASS")
    print("only final camera END/ACK duration/grace mismatch remains: PASS")

    print("\n===== TRACE AND POLICY EQUATION AUDIT =====")
    trace = verify_trace(read_csv(trace_path), report)
    plan = verify_plan(read_csv(plan_path), trace, report)
    print(
        f"trace rows={trace['rows']} policy=5/5 trips=0 clips=0; "
        f"final_home_error={trace['max_final_home_error']:.3f}: PASS"
    )

    print("\n===== LIVE CAMERA EVIDENCE BEFORE TEARDOWN =====")
    front_rows = read_csv(front_path)
    wrist_rows = read_csv(wrist_path)
    front = verify_camera_frames(role="front", rows=front_rows, report=report)
    wrist = verify_camera_frames(role="wrist", rows=wrist_rows, report=report)
    observation = verify_live_observation(report, front_rows, wrist_rows)
    print(
        f"front={front['frames']} frames @{front['achieved_receive_fps']:.3f}fps; "
        f"wrist={wrist['frames']} frames @{wrist['achieved_receive_fps']:.3f}fps"
    )
    print("fresh paired observation, zero decode/CRC/protocol/sequence failures: PASS")

    output = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "decision": REVIEW_DECISION,
        "stage": "3b_five_command_policy_commissioning",
        "stage_complete": True,
        "review_scope": {
            "offline_only": True,
            "hardware_accessed": False,
            "camera_opened": False,
            "model_loaded": False,
            "cuda_used": False,
            "motor_command_sent": False,
            "source_run_reexecuted": False,
        },
        "static_boundary": static,
        "commissioning_source": source,
        "preserved_run": {
            "report_path": str(run_report_path),
            "report_sha256": reviewed["sha256"],
            "checks": reviewed["checks"],
        },
        "motion_acceptance": {
            "policy_write_attempts": 5,
            "policy_writes_acknowledged": 5,
            "live_policy_inferences": 1,
            "tracking_trip_rows": 0,
            "rate_or_soft_clip_policy_rows": 0,
            "returned_home_within_5_degrees": True,
            "torque_disabled_at_exit": True,
            "serial_closed": True,
            "trace": {key: value for key, value in trace.items() if key != "policy_rows_raw"},
            "plan": plan,
        },
        "camera_evidence": {
            "front": front,
            "wrist": wrist,
            "live_observation": observation,
            "teardown_failure_set": sorted(EXPECTED_CAMERA_FAILURES),
            "attribution": (
                "40-second producer duration exceeded the remaining 15-second "
                "finish grace after motion completed; the parent closed transport "
                "before normal END/ACK. Frame integrity and the inference pair passed."
            ),
            "affects_motion_acceptance": False,
        },
        "next_gate": "ONE_CONTROLLED_POLICY_EPISODE",
        "next_gate_motion_authorized": False,
        "autonomous_deployment_authorized": False,
    }
    write_report(args.output_json, output)

    print("\n===== REVIEW DECISION =====")
    print(REVIEW_DECISION)
    print("STAGE 3B: COMPLETE; DO NOT REPEAT FIVE-COMMAND HARDWARE MOTION.")
    print("NEXT: BUILD ONE CONTROLLED POLICY EPISODE.")
    print("NO HARDWARE WAS ACCESSED BY THIS REVIEW.")
    print(f"report={args.output_json.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"ACT V3 FIVE-COMMAND OFFLINE REVIEW: FAIL: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print("STAGE 3B IS NOT ACCEPTED BY THIS FAILED REVIEW.", file=sys.stderr)
        print("NO HARDWARE WAS ACCESSED.", file=sys.stderr)
        raise SystemExit(1)
