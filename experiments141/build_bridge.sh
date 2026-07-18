#!/bin/bash
# OPRD-Bridge prerequisites: build the FROZEN cross-arch bridge (Stage 0 + Stage 1) that
# experiments/exp_oprd_bridge.sh (Stage 2) consumes. Single GPU. Idempotent: skips a stage
# whose output already exists (set FORCE=1 to rebuild).
#
#   bash experiments/build_bridge.sh            # 4B -> 0.6B, rank 8, all layers
#   RANKS=4 LAYER_MODE=mid bash experiments/build_bridge.sh
#
# Output: outputs/bridge_construction/rank_${RANKS}/ps_bank.pt
# Then run Stage 2:  REP_LOW_RANK=${RANKS} bash experiments/exp_oprd_bridge.sh
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AN="${REPO_ROOT}/scripts/analysis"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export RANKS="${RANKS:-8}" LAYER_MODE="${LAYER_MODE:-all}"
PAIRS="${REPO_ROOT}/outputs/cross_arch_preexp1/on_policy_pairs.jsonl"
BANK="${REPO_ROOT}/outputs/bridge_construction/rank_${RANKS}/ps_bank.pt"

echo "==================== STAGE 0: collect on-policy pairs ===================="
if [ "${FORCE:-0}" != "1" ] && [ -s "$PAIRS" ]; then
  echo "SKIP: $PAIRS already exists ($(wc -l < "$PAIRS") pairs). Set FORCE=1 to rebuild."
else
  bash "${AN}/run_cross_arch_analysis.sh"
fi
[ -s "$PAIRS" ] || { echo "FATAL: Stage 0 produced no pairs"; exit 1; }

echo "==================== STAGE 1: build + freeze rank-${RANKS} bridge ===================="
if [ "${FORCE:-0}" != "1" ] && [ -s "$BANK" ]; then
  echo "SKIP: $BANK already exists. Set FORCE=1 to rebuild."
else
  bash "${AN}/run_cross_arch_preexp2.sh"
fi
[ -s "$BANK" ] || { echo "FATAL: Stage 1 produced no ps_bank.pt"; exit 1; }

echo "==================== BRIDGE READY ===================="
echo "  $BANK"
python "${AN}/inspect_ps_bank.py" "$BANK" 2>/dev/null | grep -iE "rank|num layer_pairs|Key format" || true
echo "Next:  REP_LOW_RANK=${RANKS} REP_DISTILLATION_LAYERS=${LAYER_MODE} bash experiments/exp_oprd_bridge.sh"
