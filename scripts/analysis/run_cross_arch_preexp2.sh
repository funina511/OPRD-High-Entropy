#!/bin/bash
# OPRD-Bridge STAGE 1 — build + freeze the low-rank cross-arch bridge (P_T via PCA, P_S trained).
#
# Reads Stage 0's on_policy_pairs.jsonl, recomputes teacher/student hidden states on frozen
# models, fits teacher PCA bases P_T (top-r directions per layer) and trains the student
# projectors P_S to align into that r-dim subspace. Writes:
#   ${OUTPUT_DIR}/rank_${RANKS}/ps_bank.pt      <- the frozen bridge used by Stage 2
#   ${OUTPUT_DIR}/rank_${RANKS}/results.json    <- fitted subspace cosine/mse curves
#   ${OUTPUT_DIR}/summary.json
#
# NOTE (why this file was rewritten): the previous version passed flags the current
# cross_arch_preexp2_train_ps.py argparse does NOT accept (--subspace-mode, --position-mode,
# --first-k, --max-pca-rows, --projector, --mlp-hidden-mult, --compute-probe-cosine) and used
# --layer-mode even (only all|last|mid are valid) -> it crashed immediately. Fixed here.
#
# IMPORTANT: --ranks and --layer-mode MUST match the Stage 2 distillation knobs
#   REP_LOW_RANK and REP_DISTILLATION_LAYERS (both default to 8 / all in run_oprd_3090.sh).

set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --- env (verl py3.12; no vLLM here, so expandable_segments is safe and avoids OOM) ---
source /mnt/lxy/miniconda3/etc/profile.d/conda.sh
conda activate verl
export PATH=/mnt/lxy/miniconda3/envs/verl/bin:$PATH
export PYTHONPATH="${REPO_ROOT}/verl:${PYTHONPATH:-}"
export NO_PROXY=localhost,127.0.0.1,0.0.0.0 no_proxy=localhost,127.0.0.1,0.0.0.0
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"   # single GPU

STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:-/mnt/lxy/hf_models/Qwen3-0.6B}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-/mnt/lxy/hf_models/Qwen3-4B}"
RESPONSES_JSONL="${RESPONSES_JSONL:-${REPO_ROOT}/outputs/cross_arch_preexp1/on_policy_pairs.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/bridge_construction}"
RANKS="${RANKS:-8}"                 # cross-arch default r=8 (must match REP_LOW_RANK in Stage 2)
LAYER_MODE="${LAYER_MODE:-all}"     # all | last | mid  (must match REP_DISTILLATION_LAYERS)
LAST_K="${LAST_K:-1024}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_BATCH_TOKENS="${MAX_BATCH_TOKENS:-4096}"
EVAL_EVERY="${EVAL_EVERY:-1}"

python3 "${SCRIPT_DIR}/cross_arch_preexp2_train_ps.py" \
  --responses-jsonl "${RESPONSES_JSONL}" \
  --student-model-path "${STUDENT_MODEL_PATH}" \
  --teacher-model-path "${TEACHER_MODEL_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --ranks ${RANKS} \
  --layer-mode "${LAYER_MODE}" \
  --last-k "${LAST_K}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --batch-size "${BATCH_SIZE}" \
  --max-batch-tokens "${MAX_BATCH_TOKENS}" \
  --eval-every "${EVAL_EVERY}"

echo "Bridge: ${OUTPUT_DIR}/rank_${RANKS}/ps_bank.pt"
echo "Inspect: python ${SCRIPT_DIR}/inspect_ps_bank.py ${OUTPUT_DIR}/rank_${RANKS}/ps_bank.pt"
