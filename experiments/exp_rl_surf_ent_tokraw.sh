#!/bin/bash
# ARM A-ent-RAW — SAME-VOCAB clean anchor: r_t = logp_T(y_t) - lam*logp_S(y_t).
#
# Fully raw per-token reward (NO length norm, NO baseline), spread over the whole
# response, under token_reward_direct so the PG sum telescopes:
#
#   Sum_t r_t = logp_T(y) - lam*logp_S(y)
#   lam=1  ==  same-vocab OPD EXACTLY (per-token reverse-KL, both terms same nats
#              scale -> auto-balanced, no per-channel tuning needed).
#
# Purpose: DECOUPLE "does the per-token entropy MECHANISM work?" from "is the
# cross-vocab dual-channel SCALE tuned right?". This arm at lam=1 should match the
# existing token-OPD run (exp_rl_opd_rkl.sh). If it does, the mechanism is sound and
# the only open problem for cross-vocab is the scale balance (-> token_dual).
#
# SAME-VOCAB ONLY (needs per-token teacher_full_logp). Requires student==teacher vocab.
#   SURFACE_STUDENT_ENTROPY_COEF=1.0 bash experiments/exp_rl_surf_ent_tokraw.sh
#
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup/common.sh"

# --- keep OPRD rig alive but zero the RKD gradient (teacher forward still runs) ---
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

# --- token_raw: raw per-token reward + token_reward_direct (telescopes to OPD) ---
export SURFACE_ENTROPY_MODE=token_raw
export ADV_ESTIMATOR=${ADV_ESTIMATOR:-token_reward_direct}
export N_RESPONSES=${N_RESPONSES:-8}
export SURFACE_STUDENT_ENTROPY_COEF=${SURFACE_STUDENT_ENTROPY_COEF:-1.0}

_LAM_TAG=$(printf "%s" "$SURFACE_STUDENT_ENTROPY_COEF" | tr '.' 'p')
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-armA_surf_raw_lam${_LAM_TAG}_$(date +%Y-%m-%d_%H-%M-%S)}
run_experiment oprd
