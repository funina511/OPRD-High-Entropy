#!/bin/bash
# OPRD+OPD combined: token-level OPD (log_prob_top_k=16) is the primary signal that
# anchors the output distribution; the representation loss is a small auxiliary term
# (coef=0.1). Use this when rep-only collapses and you want a well-behaved baseline.
#
#   bash experiments/exp_oprd_opd.sh
#   REP_DISTILLATION_COEF=0.3 MINI_BATCH_SIZE=16 bash experiments/exp_oprd_opd.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"

export REP_DISTILLATION_COEF=${REP_DISTILLATION_COEF:-0.1}             # rep is AUXILIARY; OPD dominates
export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-16}                            # token-level OPD signal
export REP_DISTILLATION_LAYERS=${REP_DISTILLATION_LAYERS:-even}
export REP_DISTILLATION_LAST_K=${REP_DISTILLATION_LAST_K:-1024}

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-oprd_opd_$(date +%Y-%m-%d_%H-%M-%S)}
run_experiment oprd_opd
