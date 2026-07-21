#!/usr/bin/env bash
set -euo pipefail

SFT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SFT_DIR}/.." && pwd)"

: "${TEACHER_MODEL_PATH:?Set TEACHER_MODEL_PATH to the local teacher model directory}"

SAMPLE_PYTHON="${SAMPLE_PYTHON:-python}"
DATA_PARQUET="${DATA_PARQUET:-${REPO_ROOT}/datasets/dapo-math-5k-seed42.parquet}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
TEACHER_NAME="$(basename "${TEACHER_MODEL_PATH%/}")"
RUN_ID="${RUN_ID:-teacher-${TEACHER_NAME}-dapo5k-s42-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${SAMPLE_OUTPUT_DIR:-${SFT_DIR}/runs/${RUN_ID}/sampling}"

NUM_ROLLOUTS="${NUM_ROLLOUTS:-1}"
MAX_PROMPTS="${MAX_PROMPTS:-0}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:--1}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-10480}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
REQUEST_BATCH_SIZE="${REQUEST_BATCH_SIZE:-256}"
SEED="${SEED:-42}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"
BASIC_REJECTION="${BASIC_REJECTION:-false}"
MAX_ATTEMPTS_PER_ROLLOUT="${MAX_ATTEMPTS_PER_ROLLOUT:-3}"

WANDB_PROJECT="${WANDB_PROJECT:-OPRD-High-Entropy}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_RUN_GROUP:-${RUN_ID}}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_ID}-sample}"
WANDB_DIR="${WANDB_DIR:-${SFT_DIR}/runs/${RUN_ID}}"
mkdir -p "${WANDB_DIR}"
export WANDB_DIR

args=(
  --input-parquet "${DATA_PARQUET}"
  --model-path "${TEACHER_MODEL_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --gpu-ids "${GPU_IDS}"
  --num-rollouts "${NUM_ROLLOUTS}"
  --max-prompts "${MAX_PROMPTS}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --top-k "${TOP_K}"
  --repetition-penalty "${REPETITION_PENALTY}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --request-batch-size "${REQUEST_BATCH_SIZE}"
  --seed "${SEED}"
  --max-attempts-per-rollout "${MAX_ATTEMPTS_PER_ROLLOUT}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-entity "${WANDB_ENTITY}"
  --wandb-group "${WANDB_GROUP}"
  --wandb-run-name "${WANDB_RUN_NAME}"
  --wandb-mode "${WANDB_MODE}"
)

if [[ "${ENABLE_THINKING,,}" == "true" ]]; then
  args+=(--enable-thinking)
else
  args+=(--no-enable-thinking)
fi
if [[ "${BASIC_REJECTION,,}" == "true" ]]; then
  args+=(--basic-rejection)
else
  args+=(--no-basic-rejection)
fi
if [[ "${ALLOW_INCOMPLETE:-false}" == "true" ]]; then
  args+=(--allow-incomplete)
fi
if [[ "${ALLOW_CONFIG_MISMATCH:-false}" == "true" ]]; then
  args+=(--allow-config-mismatch)
fi

echo "RUN_ID=${RUN_ID}"
echo "Sampling output: ${OUTPUT_DIR}"
cd "${REPO_ROOT}"
exec "${SAMPLE_PYTHON}" -m sft.sample_teacher "${args[@]}"
