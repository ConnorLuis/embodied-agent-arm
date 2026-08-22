#!/usr/bin/env bash
set -euo pipefail

# 先在当前 shell 中执行：
#   conda activate embodiedarm
# 再运行本脚本。

DATASET_ROOT="data/lerobot/so101_red_cube_pick_place_v1"
DATASET_REPO_ID="connorluis/so101_red_cube_pick_place_v1"
RUN_NAME="${1:-act_red_cube_v1}"
STEPS="${2:-20000}"
BATCH_SIZE="${3:-4}"

OUTPUT_DIR="outputs/train/${RUN_NAME}"
LOG_DIR="logs/train"
LOG_FILE="${LOG_DIR}/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "ERROR: dataset root 不存在: ${DATASET_ROOT}" >&2
  exit 1
fi

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "ERROR: output_dir 已存在: ${OUTPUT_DIR}" >&2
  echo "LeRobot 0.3.4 默认拒绝覆盖；请换 RUN_NAME，或明确处理旧的生成目录。" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

echo "===== ACT TRAIN ====="
echo "dataset: ${DATASET_ROOT}"
echo "run: ${RUN_NAME}"
echo "steps: ${STEPS}"
echo "batch_size: ${BATCH_SIZE}"
echo "output: ${OUTPUT_DIR}"
echo "log: ${LOG_FILE}"

# 本项目 v1 选择：
# chunk_size=50  -> 15 Hz 下预测约 3.33 s action horizon
# n_action_steps=10 -> 推理时每约 0.67 s 重新规划一次
#
# 这是针对当前慢速双视角 Pick-and-Place 的项目基线，不是 LeRobot 官方默认值。
lerobot-train \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.chunk_size=50 \
  --policy.n_action_steps=10 \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${RUN_NAME}" \
  --batch_size="${BATCH_SIZE}" \
  --num_workers=4 \
  --steps="${STEPS}" \
  --log_freq=100 \
  --save_checkpoint=true \
  --save_freq=5000 \
  --wandb.enable=false \
  2>&1 | tee "${LOG_FILE}"
