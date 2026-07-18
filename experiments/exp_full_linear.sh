#!/bin/bash
# Full-linear variant (no PCA, no rank bottleneck). Ablation vs OPRD-Bridge.
# One trainable nn.Linear(d_S=1024 -> d_T=2560, bias=False) maps the student hidden
# straight to the teacher hidden dim, aligned by L2-normalized MSE (REP_PROJECTOR_MODE=full),
# trained on-policy from scratch. All other knobs match exp_oprd_bridge for a clean comparison.
#
#   bash experiments/exp_full_linear.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"

export REP_PROJECTOR_MODE=full                                         # no checkpoint, trains from scratch
export REP_DISTILLATION_COEF=${REP_DISTILLATION_COEF:-10.0}
export REP_DISTILLATION_LAYERS=${REP_DISTILLATION_LAYERS:-all}
export REP_DISTILLATION_POSITIONS=${REP_DISTILLATION_POSITIONS:-last_k}
export REP_DISTILLATION_LAST_K=${REP_DISTILLATION_LAST_K:-1024}

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-full_linear_$(date +%Y-%m-%d_%H-%M-%S)}
run_experiment oprd
