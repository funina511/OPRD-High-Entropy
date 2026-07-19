#!/bin/bash
# P2 (retargeted): RKD-D + within-response RKD-A on ALL layers.
# Fair vs baseline rel_rkd_c3.0: same positions=last_k=1024, chunks=8, coef=3; only adds A.
# (P0 showed last-only is weaker — do not put A on last alone.)
#
#   bash experiments141/exp_rel_rkd_da_all.sh
#   REP_RKD_ANGLE_COEF=1.0 bash experiments141/exp_rel_rkd_da_all.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup/common.sh"

export REP_PROJECTOR_MODE=full
export REP_ALIGN_LOSS=rkd
export REP_CHUNKS=${REP_CHUNKS:-8}
export REP_DISTILLATION_COEF=${REP_DISTILLATION_COEF:-3.0}
export REP_DISTILLATION_LAYERS=all
export REP_DISTILLATION_POSITIONS=${REP_DISTILLATION_POSITIONS:-last_k}
export REP_DISTILLATION_LAST_K=${REP_DISTILLATION_LAST_K:-1024}
export REP_RKD_ANGLE_COEF=${REP_RKD_ANGLE_COEF:-2.0}

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-rel_rkd_da_all_a${REP_RKD_ANGLE_COEF}_c${REP_DISTILLATION_COEF}_$(date +%Y-%m-%d_%H-%M-%S)}
run_experiment oprd
