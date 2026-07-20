#!/bin/bash
# ARM C — RKD (process channel) + surface (②, text-manifold distillation).
#
# This is OPRD's composite objective with the vocab-locked output term (logit
# reverse-KL / OPD) REPLACED by a vocab-free surface term:
#   - RKD-D + RKD-A: differentiable relational loss on hidden states (bypasses lm_head)
#   - surface (②):   PG reward = length-normalized teacher log-lik of decoded text
# NO outcome reward, NO format reward — pure distillation on both channels.
# Balance knob between the two channels is mu = REP_DISTILLATION_COEF (vs PG loss).
#
# Pairs with:
#   exp_rl_surf.sh                (ARM A: surface-only)
#   exp_rel_rkd_da_rl.sh          (ARM B: RKD-only, rep_distillation_only)
#
#   bash experiments/exp_rel_rkd_da_rl_surf.sh
#
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup/common.sh"

# --- RKD process channel (match P2 da_all) ---
export REP_PROJECTOR_MODE=full
export REP_ALIGN_LOSS=rkd
export REP_CHUNKS=${REP_CHUNKS:-8}
export REP_DISTILLATION_COEF=${REP_DISTILLATION_COEF:-3.0}
export REP_DISTILLATION_LAYERS=all
export REP_DISTILLATION_POSITIONS=${REP_DISTILLATION_POSITIONS:-last_k}
export REP_DISTILLATION_LAST_K=${REP_DISTILLATION_LAST_K:-1024}
export REP_RKD_ANGLE_COEF=${REP_RKD_ANGLE_COEF:-2.0}
export USE_REP_DISTILLATION=True
export REP_DISTILLATION_ONLY=False
export LOG_PROB_TOP_K=0

# --- surface channel (②) as the policy-gradient signal; no outcome/format ---
export USE_SURFACE_REWARD=True
export USE_TOKEN_KL_REWARD=False
export ENABLE_FORMAT_REWARD=False
export ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}
# n=2: unified across arms A/B/C; n=4 OOMs in teacher hidden-repr extraction on 4x3090.
export N_RESPONSES=2

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-armC_rkd_da_surf_a${REP_RKD_ANGLE_COEF}_c${REP_DISTILLATION_COEF}_$(date +%Y-%m-%d_%H-%M-%S)}
run_experiment oprd
