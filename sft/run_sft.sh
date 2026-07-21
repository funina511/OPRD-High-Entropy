#!/usr/bin/env bash
set -euo pipefail

SFT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SFT_DIR}/.." && pwd)"
LLAMAFACTORY_DIR="${LLAMAFACTORY_DIR:-${REPO_ROOT}/LlamaFactory}"

: "${RUN_ID:?Set RUN_ID to the teacher-sampling run id}"
: "${STUDENT_MODEL_PATH:?Set STUDENT_MODEL_PATH to the initial student model directory}"

SELECTION="${SELECTION:-all}"
SAFE_RUN_ID="${RUN_ID//-/_}"
SFT_DATASET_NAME="${SFT_DATASET_NAME:-teacher_sft_${SELECTION}_${SAFE_RUN_ID}}"
SFT_DATA_DIR="${SFT_DATA_DIR:-${SFT_DIR}/runs/${RUN_ID}/data_${SELECTION}}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-${SFT_DIR}/runs/${RUN_ID}/checkpoints_${SELECTION}}"
SFT_CONFIG="${SFT_CONFIG:-${SFT_DIR}/configs/teacher_sft_qwen3_06b_full.yaml}"
LLAMAFACTORY_CLI="${LLAMAFACTORY_CLI:-llamafactory-cli}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${LLAMAFACTORY_DIR}/examples/deepspeed/ds_z2_config.json}"

GPU_IDS="${GPU_IDS:-0,1,2,3}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-16}"
IFS=',' read -r -a gpu_array <<< "${GPU_IDS}"
NUM_GPUS="${#gpu_array[@]}"
denominator=$((NUM_GPUS * PER_DEVICE_BATCH_SIZE))
if [[ -z "${GRADIENT_ACCUMULATION_STEPS:-}" ]]; then
  if (( TARGET_GLOBAL_BATCH % denominator != 0 )); then
    echo "TARGET_GLOBAL_BATCH=${TARGET_GLOBAL_BATCH} is not divisible by GPUs*per_device=${denominator}." >&2
    exit 1
  fi
  GRADIENT_ACCUMULATION_STEPS=$((TARGET_GLOBAL_BATCH / denominator))
fi

MAX_STEPS="${MAX_STEPS:--1}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1.0}"
LEARNING_RATE="${LEARNING_RATE:-1.0e-5}"
CUTOFF_LEN="${CUTOFF_LEN:-10240}"
FLASH_ATTN="${FLASH_ATTN:-auto}"
SEED="${SEED:-42}"
RUN_NAME="${SFT_RUN_NAME:-${RUN_ID}-sft-${SELECTION}}"

WANDB_PROJECT="${WANDB_PROJECT:-OPRD-High-Entropy}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-${RUN_ID}}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_TAGS="${WANDB_TAGS:-baseline,teacher-sft,dapo5k,${SELECTION}}"
WANDB_LOG_MODEL="${WANDB_LOG_MODEL:-false}"
WANDB_DIR="${WANDB_DIR:-${SFT_DIR}/runs/${RUN_ID}}"
mkdir -p "${WANDB_DIR}" "${SFT_OUTPUT_DIR}"
export WANDB_PROJECT WANDB_RUN_GROUP WANDB_MODE WANDB_TAGS WANDB_LOG_MODEL WANDB_DIR
if [[ -n "${WANDB_ENTITY}" ]]; then
  export WANDB_ENTITY
else
  unset WANDB_ENTITY
fi

if [[ ! -f "${SFT_DATA_DIR}/dataset_info.json" || ! -f "${SFT_DATA_DIR}/teacher_sft.jsonl" ]]; then
  echo "Prepared dataset not found under ${SFT_DATA_DIR}; run sft/run_prepare.sh first." >&2
  exit 1
fi
if [[ ! -d "${STUDENT_MODEL_PATH}" ]]; then
  echo "Student model directory not found: ${STUDENT_MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${SFT_CONFIG}" || ! -f "${DEEPSPEED_CONFIG}" ]]; then
  echo "Missing SFT or DeepSpeed config: ${SFT_CONFIG} / ${DEEPSPEED_CONFIG}" >&2
  exit 1
fi
if ! command -v "${LLAMAFACTORY_CLI}" >/dev/null 2>&1; then
  echo "Cannot find ${LLAMAFACTORY_CLI}; install the vendored LlamaFactory in the SFT environment." >&2
  exit 1
fi

REPORT_TO=wandb
if [[ "${WANDB_MODE}" == "disabled" ]]; then
  REPORT_TO=none
fi

args=(
  model_name_or_path="${STUDENT_MODEL_PATH}"
  dataset="${SFT_DATASET_NAME}"
  dataset_dir="${SFT_DATA_DIR}"
  output_dir="${SFT_OUTPUT_DIR}"
  deepspeed="${DEEPSPEED_CONFIG}"
  per_device_train_batch_size="${PER_DEVICE_BATCH_SIZE}"
  gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}"
  learning_rate="${LEARNING_RATE}"
  cutoff_len="${CUTOFF_LEN}"
  flash_attn="${FLASH_ATTN}"
  max_steps="${MAX_STEPS}"
  num_train_epochs="${NUM_TRAIN_EPOCHS}"
  seed="${SEED}"
  data_seed="${SEED}"
  report_to="${REPORT_TO}"
  run_name="${RUN_NAME}"
)
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  args+=(resume_from_checkpoint="${RESUME_FROM_CHECKPOINT}")
fi

echo "Effective global batch: ${NUM_GPUS} GPUs * ${PER_DEVICE_BATCH_SIZE} * ${GRADIENT_ACCUMULATION_STEPS} = $((denominator * GRADIENT_ACCUMULATION_STEPS))"
echo "SFT output: ${SFT_OUTPUT_DIR}"
cd "${LLAMAFACTORY_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export FORCE_TORCHRUN=1
exec "${LLAMAFACTORY_CLI}" train "${SFT_CONFIG}" "${args[@]}"
