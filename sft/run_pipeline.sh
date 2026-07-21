#!/usr/bin/env bash
set -euo pipefail

SFT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Comma-separated subset of: sample,prepare,sft,eval.
PIPELINE_STAGES="${PIPELINE_STAGES:-sample,prepare,sft}"

if [[ -z "${RUN_ID:-}" ]]; then
  : "${TEACHER_MODEL_PATH:?Set TEACHER_MODEL_PATH or provide an explicit RUN_ID}"
  : "${STUDENT_MODEL_PATH:?Set STUDENT_MODEL_PATH or provide an explicit RUN_ID}"
  TEACHER_NAME="$(basename "${TEACHER_MODEL_PATH%/}")"
  STUDENT_NAME="$(basename "${STUDENT_MODEL_PATH%/}")"
  RUN_ID="teacher-sft-${TEACHER_NAME}-to-${STUDENT_NAME}-dapo5k-s42-$(date +%Y%m%d-%H%M%S)"
fi
export RUN_ID
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-${RUN_ID}}"
export SELECTION="${SELECTION:-all}"
export WANDB_DIR="${WANDB_DIR:-${SFT_DIR}/runs/${RUN_ID}}"
mkdir -p "${WANDB_DIR}"

has_stage() {
  [[ ",${PIPELINE_STAGES}," == *",$1,"* ]]
}

echo "RUN_ID=${RUN_ID}"
echo "PIPELINE_STAGES=${PIPELINE_STAGES}"
echo "W&B group=${WANDB_RUN_GROUP}"

if has_stage sample; then
  bash "${SFT_DIR}/run_sampling.sh"
fi

if has_stage prepare; then
  bash "${SFT_DIR}/run_prepare.sh"
fi

if has_stage sft; then
  bash "${SFT_DIR}/run_sft.sh"
fi

if has_stage eval; then
  export MODEL_PATH="${MODEL_PATH:-${SFT_DIR}/runs/${RUN_ID}/checkpoints_${SELECTION}}"
  bash "${SFT_DIR}/run_eval.sh"
fi

echo "Pipeline completed: ${SFT_DIR}/runs/${RUN_ID}"
