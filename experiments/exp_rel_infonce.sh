#!/bin/bash
# Arm D: chunk-level InfoNCE relational alignment (no bridge, full trainable projector).
# Identical to exp_full_linear EXCEPT the loss: MSE -> InfoNCE. This is the cleanest
# controlled test of "is it the LOSS, not the frozen bottleneck, that resists collapse?"
# Student chunks are projected to teacher dim by the same full nn.Linear(1024->2560),
# then each chunk must pick its own teacher chunk out of the batch (in-batch negatives).
#
#   bash experiments/exp_rel_infonce.sh
#   REP_INFONCE_TAU=0.1 REP_DISTILLATION_COEF=1.0 bash experiments/exp_rel_infonce.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup/common.sh"

export REP_PROJECTOR_MODE=full                                         # trainable Linear(1024->2560)
export REP_ALIGN_LOSS=infonce
export REP_CHUNKS=${REP_CHUNKS:-8}
export REP_INFONCE_TAU=${REP_INFONCE_TAU:-0.07}
export REP_DISTILLATION_COEF=${REP_DISTILLATION_COEF:-1.0}             # sweep {0.3,1,3}
export REP_DISTILLATION_LAYERS=${REP_DISTILLATION_LAYERS:-all}
export REP_DISTILLATION_POSITIONS=${REP_DISTILLATION_POSITIONS:-last_k}
export REP_DISTILLATION_LAST_K=${REP_DISTILLATION_LAST_K:-1024}

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-rel_infonce_c${REP_DISTILLATION_COEF}_$(date +%Y-%m-%d_%H-%M-%S)}
run_experiment oprd
