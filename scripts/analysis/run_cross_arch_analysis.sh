#!/bin/bash
# OPRD-Bridge STAGE 0 — collect on-policy (student-response) pairs for bridge construction.
#
# Student generates on-policy responses; the prompt+response text is saved to
#   ${OUTPUT_DIR}/on_policy_pairs.jsonl   (fields: raw_prompt, response)
# which Stage 1 (run_cross_arch_preexp2.sh) consumes to build the frozen bridge.
#
# NOTE (why this file was rewritten): the previous version passed flags the current
# cross_arch_repr_analysis.py argparse does NOT accept (--generate-backend,
# --max-batch-tokens, --vllm-max-model-len), so it crashed immediately. This version
# passes only supported flags and sets VLLM_WORKER_MULTIPROC_METHOD=spawn (vLLM v1's
# forked engine core dies with "Cannot re-initialize CUDA in forked subprocess" otherwise).
#
# The trailing representation-alignment DIAGNOSTIC (results.json) can OOM on a 24GB card
# with many prompts/long seqs; that happens AFTER on_policy_pairs.jsonl is written, so a
# non-zero exit there is harmless — check that the jsonl has NUM_PROMPTS lines.

set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --- env (verl py3.12; keep localhost off the clash proxy; models are local) ---
source /mnt/lxy/miniconda3/etc/profile.d/conda.sh
conda activate verl
export PATH=/mnt/lxy/miniconda3/envs/verl/bin:$PATH
export PYTHONPATH="${REPO_ROOT}/verl:${PYTHONPATH:-}"
export NO_PROXY=localhost,127.0.0.1,0.0.0.0 no_proxy=localhost,127.0.0.1,0.0.0.0
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export VLLM_WORKER_MULTIPROC_METHOD=spawn          # REQUIRED: vLLM v1 engine core forks
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"   # single GPU

STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:-/mnt/lxy/hf_models/Qwen3-0.6B}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-/mnt/lxy/hf_models/Qwen3-4B}"
DATA_PARQUET="${DATA_PARQUET:-${REPO_ROOT}/datasets/dapo-math-5k-seed42.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/cross_arch_preexp1}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
LAST_K="${LAST_K:-1024}"
GENERATE_BATCH_SIZE="${GENERATE_BATCH_SIZE:-16}"
FORWARD_BATCH_SIZE="${FORWARD_BATCH_SIZE:-4}"
VLLM_TP="${VLLM_TP:-1}"
VLLM_GPU_MEM="${VLLM_GPU_MEM:-0.40}"

python3 "${SCRIPT_DIR}/cross_arch_repr_analysis.py" \
  --student-model-path "${STUDENT_MODEL_PATH}" \
  --teacher-model-path "${TEACHER_MODEL_PATH}" \
  --data-parquet "${DATA_PARQUET}" \
  --num-prompts "${NUM_PROMPTS}" \
  --output-dir "${OUTPUT_DIR}" \
  --last-k "${LAST_K}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --generate-batch-size "${GENERATE_BATCH_SIZE}" \
  --batch-size "${FORWARD_BATCH_SIZE}" \
  --vllm-tensor-parallel-size "${VLLM_TP}" \
  --vllm-gpu-memory-utilization "${VLLM_GPU_MEM}" \
  --generate-responses \
  ${RESPONSES_JSONL:+--responses-jsonl "${RESPONSES_JSONL}"} || \
  echo "NOTE: non-zero exit (likely the trailing alignment diagnostic OOM). Check pairs below."

echo "Pairs file: ${OUTPUT_DIR}/on_policy_pairs.jsonl"
wc -l "${OUTPUT_DIR}/on_policy_pairs.jsonl" 2>/dev/null || echo "  (missing — Stage 0 failed before saving pairs)"
