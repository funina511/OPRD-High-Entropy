#!/bin/bash
# P1: RKD-D on ALL response tokens (positions=all), layers=all, c=3.
#
#   bash experiments141/exp_rel_rkd_alltok.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup/common.sh"

export REP_PROJECTOR_MODE=full
export REP_ALIGN_LOSS=rkd
export REP_CHUNKS=${REP_CHUNKS:-8}
export REP_DISTILLATION_COEF=${REP_DISTILLATION_COEF:-3.0}
export REP_DISTILLATION_LAYERS=${REP_DISTILLATION_LAYERS:-all}
export REP_DISTILLATION_POSITIONS=all
export REP_RKD_ANGLE_COEF=${REP_RKD_ANGLE_COEF:-0.0}

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-rel_rkd_alltok_c${REP_DISTILLATION_COEF}_$(date +%Y-%m-%d_%H-%M-%S)}
run_experiment oprd
