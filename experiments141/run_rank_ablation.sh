#!/bin/bash
# RANK ABLATION (base student) — isolate whether the AMC23 collapse is caused by the
# rank-8 bottleneck being too thin, or by PCA picking the wrong (variance != task) directions.
#
# Baseline (already run): rank-8, Qwen3-0.6B-Base, bridge_construction_base/rank_8, rep-only.
#   -> AMC23 acc@4: 0.006(step0) ->0.088(peak@25) -> ... -> 0.019(step150)  [spike then decay]
#
# This driver reproduces that EXACT base recipe and changes ONLY the bridge rank.
#   Stage 1: rebuild + freeze the base bridge at rank R (single GPU).
#   Stage 2: rep-only cross-arch distillation with that frozen rank-R bridge (4 GPUs).
# Every other knob (model, data, coef=10, layers=all, freeze_ps=True, steps=150, ...) is held fixed.
#
# Usage (run the two groups in parallel from two shells / with &):
#   RANK=64  GPUS=0,1,2,3 bash experiments/run_rank_ablation.sh
#   RANK=128 GPUS=4,5,6,7 bash experiments/run_rank_ablation.sh
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AN="${REPO_ROOT}/scripts/analysis"

# ---- required knobs (the ONLY things that differ from the rank-8 base run) ----
RANK="${RANK:?set RANK=64 or RANK=128}"
GPUS="${GPUS:?set GPUS=0,1,2,3 (rank64) or 4,5,6,7 (rank128)}"
BRIDGE_GPU="${BRIDGE_GPU:-${GPUS%%,*}}"   # first GPU of the group builds the bridge

# ---- fixed base recipe (must match the rank-8 base run exactly) ----
STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:-/root/siton-tmp/home/liuxinyu/hf_models/Qwen3-0.6B-Base}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-/root/siton-tmp/home/liuxinyu/hf_models/Qwen3-4B}"
PAIRS="${PAIRS:-${REPO_ROOT}/outputs/cross_arch_preexp1_base/on_policy_pairs.jsonl}"
BRIDGE_ROOT="${BRIDGE_ROOT:-${REPO_ROOT}/outputs/bridge_construction_base}"
BANK="${BRIDGE_ROOT}/rank_${RANK}/ps_bank.pt"

echo "=================================================================="
echo " RANK ABLATION | rank=${RANK} | bridge_gpu=${BRIDGE_GPU} | stage2_gpus=${GPUS}"
echo " student=${STUDENT_MODEL_PATH}"
echo " pairs=${PAIRS}"
echo " bank =${BANK}"
echo "=================================================================="
[ -s "$PAIRS" ] || { echo "FATAL: base on-policy pairs missing: $PAIRS"; exit 1; }

# ================== STAGE 1: build + freeze rank-R base bridge ==================
if [ "${FORCE:-0}" != "1" ] && [ -s "$BANK" ]; then
  echo "SKIP stage1: $BANK already exists (set FORCE=1 to rebuild)."
else
  echo ">>> Building rank-${RANK} bridge on GPU ${BRIDGE_GPU} ..."
  CUDA_VISIBLE_DEVICES="${BRIDGE_GPU}" \
  STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH}" \
  TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH}" \
  RESPONSES_JSONL="${PAIRS}" \
  OUTPUT_DIR="${BRIDGE_ROOT}" \
  RANKS="${RANK}" \
  LAYER_MODE=all \
  LAST_K=1024 \
  EPOCHS=20 \
  LR=1e-4 \
  BATCH_SIZE=2 \
  MAX_BATCH_TOKENS=4096 \
  EVAL_EVERY=1 \
    bash "${AN}/run_cross_arch_preexp2.sh"
fi
[ -s "$BANK" ] || { echo "FATAL: stage1 produced no $BANK"; exit 1; }
echo ">>> Bridge ready: $BANK"
python "${AN}/inspect_ps_bank.py" "$BANK" 2>/dev/null | grep -iE "rank|num layer_pairs" || true

# ================== STAGE 2: rep-only distillation with frozen rank-R bridge ==================
# Reuses exp_oprd_bridge.sh; overrides ONLY rank + bridge path + student model + GPUs.
# All rep/RL knobs (coef=10, layers=all, freeze_ps=True, positions=last_k, steps=150,
# test_freq=25, save_freq=50, n=2, mbs=8) inherit the base-run defaults unchanged.
echo ">>> Stage 2: rep-only distillation (rank=${RANK}) on GPUs ${GPUS} ..."
LOG="${REPO_ROOT}/logs/rank_ablation_${RANK}.log"
mkdir -p "${REPO_ROOT}/logs"

CUDA_VISIBLE_DEVICES="${GPUS}" \
RAY_PORT="${RAY_PORT:-$((6391 + RANK))}" \
SKIP_RAY_STOP="${SKIP_RAY_STOP:-1}" \
ACTOR_MODEL_PATH="${STUDENT_MODEL_PATH}" \
REWARD_MODEL_PATH="${TEACHER_MODEL_PATH}" \
REP_PROJECTOR_MODE=low_rank \
REP_LOW_RANK="${RANK}" \
REP_LOW_RANK_INIT_CHECKPOINT="${BANK}" \
REP_FREEZE_PS=True \
REP_DISTILLATION_COEF=10.0 \
REP_DISTILLATION_LAYERS=all \
REP_DISTILLATION_POSITIONS=last_k \
REP_DISTILLATION_LAST_K=1024 \
EXPERIMENT_NAME="rank${RANK}_ablation_$(date +%Y-%m-%d_%H-%M-%S)" \
  bash "${REPO_ROOT}/experiments/exp_oprd_bridge.sh" 2>&1 | tee "${LOG}"

echo "=================================================================="
echo " DONE rank=${RANK}. Stage2 log: ${LOG}"
echo "=================================================================="
