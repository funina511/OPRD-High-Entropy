#!/usr/bin/env bash
set -euo pipefail

SFT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SFT_DIR}/.." && pwd)"

: "${MODEL_PATH:?Set MODEL_PATH to the student/base/SFT checkpoint to evaluate}"

EVAL_PYTHON="${EVAL_PYTHON:-${SAMPLE_PYTHON:-python}}"
TASK_NAME="${TASK_NAME:-AMC23}"
EVAL_PARQUET="${EVAL_PARQUET:-${REPO_ROOT}/datasets/test_data/${TASK_NAME}/test.parquet}"
GPU_IDS="${EVAL_GPU_IDS:-${GPU_IDS:-0,1,2,3}}"
MODEL_NAME="$(basename "${MODEL_PATH%/}")"
RUN_ID="${RUN_ID:-eval-${MODEL_NAME}-${TASK_NAME}-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${SFT_DIR}/runs/${RUN_ID}/eval_${TASK_NAME}}"
GEN_DIR="${OUTPUT_DIR}/generation"
SCORE_DIR="${OUTPUT_DIR}/scores"

WANDB_PROJECT="${WANDB_PROJECT:-OPRD-High-Entropy}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_RUN_GROUP:-${RUN_ID}}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_DIR="${WANDB_DIR:-${SFT_DIR}/runs/${RUN_ID}}"
mkdir -p "${WANDB_DIR}"
export WANDB_DIR

cd "${REPO_ROOT}"
"${EVAL_PYTHON}" -m sft.sample_teacher \
  --input-parquet "${EVAL_PARQUET}" \
  --model-path "${MODEL_PATH}" \
  --output-dir "${GEN_DIR}" \
  --gpu-ids "${GPU_IDS}" \
  --num-rollouts "${EVAL_NUM_ROLLOUTS:-4}" \
  --max-prompts "${EVAL_MAX_PROMPTS:-0}" \
  --temperature "${EVAL_TEMPERATURE:-0.7}" \
  --top-p "${EVAL_TOP_P:-0.95}" \
  --top-k "${EVAL_TOP_K:--1}" \
  --repetition-penalty "${EVAL_REPETITION_PENALTY:-1.0}" \
  --max-new-tokens "${EVAL_MAX_NEW_TOKENS:-8192}" \
  --max-model-len "${EVAL_MAX_MODEL_LEN:-10480}" \
  --seed "${EVAL_SEED:-42}" \
  --no-enable-thinking \
  --no-basic-rejection \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-entity "${WANDB_ENTITY}" \
  --wandb-group "${WANDB_GROUP}" \
  --wandb-run-name "${RUN_ID}-generate" \
  --wandb-job-type eval_generation \
  --wandb-mode "${WANDB_MODE}"

"${EVAL_PYTHON}" -m sft.score_math \
  --raw-jsonl "${GEN_DIR}/raw_samples.jsonl" \
  --output-dir "${SCORE_DIR}" \
  --task-name "${TASK_NAME}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-entity "${WANDB_ENTITY}" \
  --wandb-group "${WANDB_GROUP}" \
  --wandb-run-name "${RUN_ID}-score" \
  --wandb-mode "${WANDB_MODE}"

echo "Evaluation metrics: ${SCORE_DIR}/metrics.json"
