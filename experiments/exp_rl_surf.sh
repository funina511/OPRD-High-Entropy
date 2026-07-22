#!/bin/bash
# ARM A — surface-only (②): on-policy text-manifold distillation.
#
# Reward = length-normalized teacher log-likelihood of the student's decoded text
# (teacher reads the SAME tokens in this same-vocab prototype). token_level_rewards
# is OVERWRITTEN by this signal, so the policy gradient is PURE distillation:
# NO outcome reward, NO format reward, NO RKD (rep coef = 0). The teacher forward is
# kept only to read its per-token log-prob.
#
# Pairs with:
#   exp_rel_rkd_da_rl.sh          (ARM B: RKD-only, rep_distillation_only)
#   exp_rel_rkd_da_rl_surf.sh     (ARM C: RKD + surface)
#
#   bash experiments/exp_rl_surf.sh
#
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup/common.sh"

# --- keep OPRD rig alive but zero the RKD gradient (teacher forward still runs) ---
# USE_REP_DISTILLATION=False: arm A is surface-only. teacher_logp (surface reward
# source) is computed unconditionally in compute_rm_score; the all-layer hidden
# extraction (return_last_hidden_repr=use_rep_distillation) is what costs ~40s/step
# and it's pure waste here (coef=0). Turning it off is exactly equivalent (×0) but
# skips the ~9.5GB/step D2H copy -> compute_rm_score drops from ~45s to a few s.
export USE_REP_DISTILLATION=False
export REP_DISTILLATION_ONLY=False
export REP_DISTILLATION_COEF=0.0
export REP_ALIGN_LOSS=rkd
export REP_PROJECTOR_MODE=full
export LOG_PROB_TOP_K=0

# --- surface channel is the ONLY training signal ---
export USE_SURFACE_REWARD=True
export USE_TOKEN_KL_REWARD=False
export ENABLE_FORMAT_REWARD=False
export ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}
# n: GRPO's within-group baseline needs a large enough group or the std-normalized
# advantage collapses to a sign bit (at n=2, adv == +-1/sqrt(2) regardless of the
# teacher-LL gap). Default lifted to 8; override via N_RESPONSES env. The old n=2 was
# a 4x3090 memory limit, not a modeling choice.
export N_RESPONSES=${N_RESPONSES:-8}

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-armA_surf_$(date +%Y-%m-%d_%H-%M-%S)}
run_experiment oprd
