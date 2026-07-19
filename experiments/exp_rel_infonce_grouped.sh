#!/bin/bash
# Arm D2: GRPO-style "grouped" InfoNCE. Same full-projector InfoNCE as exp_rel_infonce,
# but with the POISON-NEGATIVE FIX: same-response chunks are masked out of each anchor's
# negatives (REP_INFONCE_MASK_WITHIN=True), so only CROSS-response chunks are negatives.
# Statistical grouping: rollout.n=4 (up from 2) raises same-prompt co-occurrence AND pool
# size; K=4 (down from 8) makes chunk pairing cleaner and cuts within-response poison.
#
# Single-variable contrast vs exp_rel_infonce (D arm): flip REP_INFONCE_MASK_WITHIN.
#   REP_INFONCE_MASK_WITHIN=False  -> reproduces D arm (nce_acc caps ~0.11)
#   REP_INFONCE_MASK_WITHIN=True   -> this script (does acc break the ceiling?)
#
#   bash experiments/exp_rel_infonce_grouped.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup/common.sh"

export REP_PROJECTOR_MODE=full
export REP_ALIGN_LOSS=infonce
export REP_INFONCE_MASK_WITHIN=${REP_INFONCE_MASK_WITHIN:-True}        # the fix
export REP_CHUNKS=${REP_CHUNKS:-4}                                     # K: 8 -> 4
export REP_INFONCE_TAU=${REP_INFONCE_TAU:-0.07}
export N_RESPONSES=${N_RESPONSES:-2}                                   # back to 2 (n=4 caused OOM; test masking fix first)
export REP_DISTILLATION_COEF=${REP_DISTILLATION_COEF:-1.0}
export REP_DISTILLATION_LAYERS=${REP_DISTILLATION_LAYERS:-all}
export REP_DISTILLATION_POSITIONS=${REP_DISTILLATION_POSITIONS:-last_k}
export REP_DISTILLATION_LAST_K=${REP_DISTILLATION_LAST_K:-1024}

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-rel_infonce_grouped_k${REP_CHUNKS}_n${N_RESPONSES}_$(date +%Y-%m-%d_%H-%M-%S)}
run_experiment oprd
