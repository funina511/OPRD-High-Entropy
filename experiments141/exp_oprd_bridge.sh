#!/bin/bash
# OPRD-Bridge (cross-arch, low-rank PCA subspace). The paper's main method.
# Aligns student->teacher inside a rank-R subspace: frozen teacher PCA bases P_T +
# trained/loaded student projector P_S. Needs a prebuilt bridge (build_bridge.sh).
#
#   bash experiments/exp_oprd_bridge.sh
#   REP_LOW_RANK=64 REP_LOW_RANK_INIT_CHECKPOINT=.../rank_64/ps_bank.pt bash experiments/exp_oprd_bridge.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup/common.sh"

export REP_PROJECTOR_MODE=low_rank
export REP_LOW_RANK=${REP_LOW_RANK:-8}                                  # MUST match the bridge's rank
export REP_LOW_RANK_INIT_CHECKPOINT=${REP_LOW_RANK_INIT_CHECKPOINT:-/root/siton-tmp/home/liuxinyu/OPRD-High-Entropy/outputs/bridge_construction_base/rank_8/ps_bank.pt}
export REP_FREEZE_PS=${REP_FREEZE_PS:-True}                             # freeze the offline bridge
export REP_DISTILLATION_COEF=${REP_DISTILLATION_COEF:-10.0}
export REP_DISTILLATION_LAYERS=${REP_DISTILLATION_LAYERS:-all}          # MUST match the bridge's --layer-mode
export REP_DISTILLATION_POSITIONS=${REP_DISTILLATION_POSITIONS:-last_k}
export REP_DISTILLATION_LAST_K=${REP_DISTILLATION_LAST_K:-1024}

if [ ! -f "$REP_LOW_RANK_INIT_CHECKPOINT" ]; then
  echo "ERROR: frozen bridge not found: $REP_LOW_RANK_INIT_CHECKPOINT"
  echo "Build it first:  RANKS=$REP_LOW_RANK bash experiments/build_bridge.sh"
  exit 1
fi

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-oprd_bridge_r${REP_LOW_RANK}_$(date +%Y-%m-%d_%H-%M-%S)}
run_experiment oprd
