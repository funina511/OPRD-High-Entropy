#!/bin/bash
# P0: RKD-D, last layer only (vs baseline layers=all). Same c=3 / chunks=8 / last_k=1024.
#
#   bash experiments141/exp_rel_rkd_last.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup/common.sh"

export REP_PROJECTOR_MODE=full
export REP_ALIGN_LOSS=rkd
export REP_CHUNKS=${REP_CHUNKS:-8}
export REP_DISTILLATION_COEF=${REP_DISTILLATION_COEF:-3.0}
export REP_DISTILLATION_LAYERS=last
export REP_DISTILLATION_POSITIONS=${REP_DISTILLATION_POSITIONS:-last_k}
export REP_DISTILLATION_LAST_K=${REP_DISTILLATION_LAST_K:-1024}
export REP_RKD_ANGLE_COEF=${REP_RKD_ANGLE_COEF:-0.0}

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-rel_rkd_last_c${REP_DISTILLATION_COEF}_$(date +%Y-%m-%d_%H-%M-%S)}
run_experiment oprd
