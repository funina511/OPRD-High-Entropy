#!/bin/bash
# Arm C: chunk-level RKD-D relational alignment, NO projector / NO bridge at all.
# Student (1024) and teacher (2560) each keep their own space; we match only the
# mean-normalized pairwise DISTANCE matrices over B*K chunk vectors (off-diagonal),
# logging within-response vs cross-response contributions separately. Tests whether
# cross-arch distillation can drop the bridge entirely.
#
#   bash experiments/exp_rel_rkd.sh
#   REP_CHUNKS=8 REP_DISTILLATION_COEF=3.0 bash experiments/exp_rel_rkd.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"

export REP_PROJECTOR_MODE=full                                         # ignored by RKD (dispatch skips projector)
export REP_ALIGN_LOSS=rkd
export REP_CHUNKS=${REP_CHUNKS:-8}
export REP_DISTILLATION_COEF=${REP_DISTILLATION_COEF:-3.0}             # sweep {1,3,10}
export REP_DISTILLATION_LAYERS=${REP_DISTILLATION_LAYERS:-all}
export REP_DISTILLATION_POSITIONS=${REP_DISTILLATION_POSITIONS:-last_k}
export REP_DISTILLATION_LAST_K=${REP_DISTILLATION_LAST_K:-1024}

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-rel_rkd_c${REP_DISTILLATION_COEF}_$(date +%Y-%m-%d_%H-%M-%S)}
run_experiment oprd
