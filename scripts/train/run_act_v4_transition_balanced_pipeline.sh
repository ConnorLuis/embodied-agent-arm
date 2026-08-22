#!/usr/bin/env bash
set -Eeuo pipefail

# One bounded offline pipeline: fresh V4 training, transition gate, then general
# held-out validation only for checkpoints that passed the transition gate.
# This script has no hardware/camera/serial command.

project_root="${1:-$PWD}"
cd "$project_root"

trainer=scripts/train/train_act_v4_transition_balanced.py
trainer_sha256=aab738ec3c8205c83a8691ea4137913b30b43f2b3c3cf1c08423fa72a74e9501

train_dataset=data/lerobot/so101_red_cube_pick_place_v3_delta_train
train_repo=connorluis/so101_red_cube_pick_place_v3_delta_train
validation_dataset=data/lerobot/so101_red_cube_pick_place_v3_delta_validation
validation_repo=connorluis/so101_red_cube_pick_place_v3_delta_validation
raw_archive=/mnt/f/episodes_pick_place_pilot_v5

base_config=outputs/train/act_red_cube_v3_delta_stage1/checkpoints/040000/pretrained_model/train_config.json
v3_terminal_report=outputs/eval/act_red_cube_v3_gripper_transition_checkpoints_16k_40k/gripper_transition_checkpoint_report.json
stage3c_run=outputs/eval/act_red_cube_v3_one_controlled_policy_episode_powerloss_20260821T101025Z

train_root=outputs/train/act_red_cube_v4_transition_balanced
sampler_output=outputs/eval/act_red_cube_v4_transition_balanced_sampler
sampler_report="$sampler_output/transition_balanced_sampler_report.json"
gripper_output=outputs/eval/act_red_cube_v4_gripper_transition_checkpoints_4k_20k
gripper_report="$gripper_output/gripper_transition_checkpoint_report.json"
general_output=outputs/eval/act_red_cube_v4_general_validation_passing
summary_output=outputs/eval/act_red_cube_v4_transition_balanced_pipeline
summary_report="$summary_output/pipeline_decision.json"

train_log=outputs/logs/act_red_cube_v4_transition_balanced.log
gripper_log=outputs/logs/act_red_cube_v4_gripper_transition_checkpoints_4k_20k.log
general_log=outputs/logs/act_red_cube_v4_general_validation_passing.log

echo "===== V4 PIPELINE CAPABILITY BOUNDARY ====="
echo "offline datasets + local checkpoints + CUDA training/inference only"
echo "NO robot, serial port, live camera, torque, Goal_Position, or action API"

echo
echo "===== PRE-FLIGHT ====="
for required_file in \
  "$trainer" \
  "$base_config" \
  "$v3_terminal_report" \
  "$stage3c_run/one_controlled_policy_episode_report.json" \
  "$stage3c_run/one_controlled_policy_episode_commands.csv" \
  scripts/eval/audit_act_v3_gripper_transition_checkpoints.py \
  scripts/eval/audit_act_v3_delta_validation.py
do
  test -f "$required_file" || {
    echo "required file missing: $required_file" >&2
    exit 1
  }
done

for required_directory in \
  "$train_dataset" \
  "$validation_dataset"
do
  test -d "$required_directory" || {
    echo "required directory missing: $required_directory" >&2
    exit 1
  }
done

for new_path in \
  "$train_root" \
  "$sampler_output" \
  "$gripper_output" \
  "$general_output" \
  "$summary_output" \
  "$train_log" \
  "$gripper_log" \
  "$general_log"
do
  test ! -e "$new_path" || {
    echo "preserve existing path and choose a new V4 run name: $new_path" >&2
    exit 1
  }
done

actual_trainer_sha256="$(sha256sum "$trainer" | awk '{print $1}')"
test "$actual_trainer_sha256" = "$trainer_sha256" || {
  echo "trainer SHA-256 mismatch" >&2
  echo "expected: $trainer_sha256" >&2
  echo "actual:   $actual_trainer_sha256" >&2
  exit 1
}

python -m py_compile \
  "$trainer" \
  scripts/eval/audit_act_v3_gripper_transition_checkpoints.py \
  scripts/eval/audit_act_v3_delta_validation.py
python "$trainer" --self-test

python - "$v3_terminal_report" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
checkpoints = report["checkpoints"]
steps = [int(item["step"]) for item in checkpoints]
expected = [16000, 20000, 24000, 28000, 32000, 36000, 40000]
if steps != expected:
    raise SystemExit(f"unexpected terminal V3 steps: {steps}")
if any(bool(item["passed"]) for item in checkpoints):
    raise SystemExit("V3 terminal report unexpectedly contains a passing checkpoint")
if report.get("selected_existing_checkpoint") is not None:
    raise SystemExit("V3 terminal report unexpectedly selected a checkpoint")
print("V3 terminal failure evidence: VERIFIED")
print("V3 training/hardware: FROZEN")
PY

available_kib="$(df -Pk . | awk 'NR==2 {print $4}')"
minimum_kib=$((20 * 1024 * 1024))
test "$available_kib" -ge "$minimum_kib" || {
  echo "less than 20 GiB free; stop before V4 training" >&2
  exit 1
}
mkdir -p outputs/logs
if test -d "$raw_archive"; then
  echo "raw archive (identity/reference only): $raw_archive — PASS"
else
  echo "raw archive is not mounted; continuing because V4 reads only $train_dataset"
fi
echo "free disk preflight: PASS"

echo
echo "===== FRESH TRANSITION-BALANCED V4 TRAINING: 0 -> 20K ====="
HF_HUB_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
python "$trainer" \
  --dataset-root "$train_dataset" \
  --dataset-repo-id "$train_repo" \
  --video-backend torchcodec \
  --sampler-report "$sampler_report" \
  --target-close-fraction 0.25 \
  --target-open-fraction 0.25 \
  --target-step-threshold 0.25 \
  --target-window-amplitude 2.5 \
  --chunk-size 10 \
  --sampler-seed 1000 \
  -- \
  --config_path="$base_config" \
  --dataset.repo_id="$train_repo" \
  --dataset.root="$train_dataset" \
  --dataset.video_backend=torchcodec \
  --output_dir="$train_root" \
  --job_name=act_red_cube_v4_transition_balanced \
  --resume=false \
  --steps=20000 \
  --save_freq=4000 \
  --eval_freq=0 \
  --seed=1000 \
  --wandb.enable=false \
  2>&1 | tee "$train_log"

python - "$sampler_report" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not report["dataloader_patch_applied"]:
    raise SystemExit("weighted sampler was not applied")
if not report["training_completed"]:
    raise SystemExit("V4 training did not complete")
expected = report["sampling"]["expected_sampled_fractions"]
target = {"background": 0.5, "close": 0.25, "open": 0.25}
for key, value in target.items():
    if abs(float(expected[key]) - value) > 1e-12:
        raise SystemExit(f"sampling fraction mismatch: {key}={expected[key]}")
print("sampler application and completed training: PASS")
PY

echo
echo "===== VERIFY FIVE V4 CHECKPOINTS ====="
for checkpoint_step in 004000 008000 012000 016000 020000
do
  model_file="$train_root/checkpoints/$checkpoint_step/pretrained_model/model.safetensors"
  test -f "$model_file"
  printf '%s: OK\n' "$checkpoint_step"
done

echo
echo "===== V4 GRIPPER TRANSITION GATE: 4K -> 20K ====="
HF_HUB_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
python scripts/eval/audit_act_v3_gripper_transition_checkpoints.py \
  --validation-dataset-root "$validation_dataset" \
  --validation-repo-id "$validation_repo" \
  --video-backend torchcodec \
  --train-output-root "$train_root" \
  --checkpoint-steps 4000 8000 12000 16000 20000 \
  --stage3c-run-dir "$stage3c_run" \
  --output-dir "$gripper_output" \
  2>&1 | tee "$gripper_log"

mapfile -t passing_steps < <(
  python - "$gripper_report" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for checkpoint in report["checkpoints"]:
    if checkpoint["passed"]:
        print(int(checkpoint["step"]))
PY
)

echo
echo "===== COMPACT V4 TRANSITION RESULT ====="
python - "$gripper_report" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for checkpoint in report["checkpoints"]:
    close = checkpoint["directions"]["close"]
    opened = checkpoint["directions"]["open"]
    print(
        f"step={int(checkpoint['step']):5d} "
        f"close_recall={close['first5_direction_recall']:.3f} "
        f"close_amp={close['first5_amplitude_ratio_median']:.3f} "
        f"open_recall={opened['first5_direction_recall']:.3f} "
        f"open_amp={opened['first5_amplitude_ratio_median']:.3f} "
        f"pass={bool(checkpoint['passed'])}"
    )
PY

if (( ${#passing_steps[@]} == 0 )); then
  mkdir -p "$summary_output"
  python - "$sampler_report" "$gripper_report" "$summary_report" <<'PY'
import json
import sys
from pathlib import Path

sampler_path, gripper_path, output_path = map(Path, sys.argv[1:])
sampler = json.loads(sampler_path.read_text(encoding="utf-8"))
gripper = json.loads(gripper_path.read_text(encoding="utf-8"))
summary = {
    "schema_version": "act_v4_transition_balanced_pipeline_v1",
    "decision": "V4_20K_NO_TRANSITION_CHECKPOINT_PASSED_FINAL_PROJECT_STOP",
    "training_completed": True,
    "sampler_report": str(sampler_path.resolve()),
    "gripper_report": str(gripper_path.resolve()),
    "evaluated_steps": [int(x["step"]) for x in gripper["checkpoints"]],
    "passing_steps": [],
    "general_validation_run": False,
    "hardware_authorized": False,
    "next": "Document V3/V4 negative result; do not train or move hardware again.",
}
output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
PY
  echo
  echo "===== FINAL V4 DECISION ====="
  echo "V4_20K_NO_TRANSITION_CHECKPOINT_PASSED_FINAL_PROJECT_STOP"
  echo "No more V3/V4 training and no hardware retest. Preserve this as the final negative result."
  echo "report=$summary_report"
  exit 2
fi

echo "passing transition checkpoints: ${passing_steps[*]}"
echo
echo "===== GENERAL HELD-OUT VALIDATION OF PASSING V4 CHECKPOINTS ====="
HF_HUB_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
python scripts/eval/audit_act_v3_delta_validation.py \
  --train-dataset-root "$train_dataset" \
  --train-repo-id "$train_repo" \
  --validation-dataset-root "$validation_dataset" \
  --validation-repo-id "$validation_repo" \
  --train-output-root "$train_root" \
  --checkpoint-steps "${passing_steps[@]}" \
  --output-dir "$general_output" \
  2>&1 | tee "$general_log"

test -f "$general_output/best_checkpoint.json"
mkdir -p "$summary_output"
python - \
  "$sampler_report" \
  "$gripper_report" \
  "$general_output/best_checkpoint.json" \
  "$summary_report" \
  "${passing_steps[@]}" <<'PY'
import json
import sys
from pathlib import Path

sampler_path = Path(sys.argv[1])
gripper_path = Path(sys.argv[2])
best_path = Path(sys.argv[3])
output_path = Path(sys.argv[4])
passing = [int(value) for value in sys.argv[5:]]
best = json.loads(best_path.read_text(encoding="utf-8"))
summary = {
    "schema_version": "act_v4_transition_balanced_pipeline_v1",
    "decision": "V4_OFFLINE_CANDIDATE_SELECTED_BUILD_FROZEN_RELEASE_AND_SHADOW_GATE_NEXT",
    "training_completed": True,
    "sampler_report": str(sampler_path.resolve()),
    "gripper_report": str(gripper_path.resolve()),
    "general_best_checkpoint": best,
    "passing_transition_steps": passing,
    "general_validation_run": True,
    "hardware_authorized": False,
    "next": "Freeze the selected V4 checkpoint, then run one 60-second no-command shadow gate.",
}
output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
PY

echo
echo "===== FINAL V4 DECISION ====="
python -m json.tool "$general_output/best_checkpoint.json"
echo "V4_OFFLINE_CANDIDATE_SELECTED_BUILD_FROZEN_RELEASE_AND_SHADOW_GATE_NEXT"
echo "HARDWARE REMAINS BLOCKED. Do not run a policy episode from this pipeline."
echo "report=$summary_report"
echo "ACT V4 TRANSITION-BALANCED OFFLINE PIPELINE: PASS"
