#!/usr/bin/env bash
set -euo pipefail

SFT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SFT_DIR}/.." && pwd)"

: "${RUN_ID:?Set RUN_ID to the same id used by run_sampling.sh}"

PREP_PYTHON="${PREP_PYTHON:-${SAMPLE_PYTHON:-python}}"
SELECTION="${SELECTION:-all}"
MAX_PER_PROMPT="${MAX_PER_PROMPT:-1}"
RAW_JSONL="${RAW_JSONL:-${SFT_DIR}/runs/${RUN_ID}/sampling/raw_samples.jsonl}"
OUTPUT_DIR="${SFT_DATA_DIR:-${SFT_DIR}/runs/${RUN_ID}/data_${SELECTION}}"
SAFE_RUN_ID="${RUN_ID//-/_}"
DATASET_NAME="${SFT_DATASET_NAME:-teacher_sft_${SELECTION}_${SAFE_RUN_ID}}"

WANDB_PROJECT="${WANDB_PROJECT:-OPRD-High-Entropy}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_RUN_GROUP:-${RUN_ID}}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_RUN_NAME="${WANDB_PREP_RUN_NAME:-${RUN_ID}-prepare-${SELECTION}}"
WANDB_DIR="${WANDB_DIR:-${SFT_DIR}/runs/${RUN_ID}}"
mkdir -p "${WANDB_DIR}"
export WANDB_DIR

args=(
  --raw-jsonl "${RAW_JSONL}"
  --output-dir "${OUTPUT_DIR}"
  --dataset-name "${DATASET_NAME}"
  --selection "${SELECTION}"
  --max-per-prompt "${MAX_PER_PROMPT}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-entity "${WANDB_ENTITY}"
  --wandb-group "${WANDB_GROUP}"
  --wandb-run-name "${WANDB_RUN_NAME}"
  --wandb-mode "${WANDB_MODE}"
)

if [[ "${GRADE_CORRECTNESS:-true}" == "true" ]]; then
  args+=(--grade-correctness)
else
  args+=(--no-grade-correctness)
fi
if [[ "${KEEP_EMPTY:-false}" == "true" ]]; then
  args+=(--keep-empty)
fi

echo "Prepared dataset: ${OUTPUT_DIR}"
echo "LlamaFactory dataset name: ${DATASET_NAME}"
cd "${REPO_ROOT}"
exec "${PREP_PYTHON}" -m sft.prepare_dataset "${args[@]}"
