#!/bin/bash
# ARM A-ent — surface + student-entropy term (route A: sequence-level OPD).
#
# Counterfactual for "surface collapses because it drops the student entropy term".
# Identical to exp_rl_surf.sh (same teacher Qwen3-4B / student Qwen3-0.6B-Base /
# data / n=8 / GRPO) EXCEPT the reward gains a -lam * logp_S(y) term:
#
#   surface (lam=0):  r(y) = mean_t logp_T(y_t)                 = E[logp_T]
#   this arm (lam>0): r(y) = mean_t [ logp_T(y_t) - lam*logp_S(y_t) ]
#                          = E[logp_T] + lam*H(pi)  (length-normalized, telescoped)
#
# logp_S is the DETACHED old_log_probs, so the reward stays a constant (no PG break).
# lam=1.0 is the full sequence-level OPD objective; sweep lam in {0.5, 1.0} and
# reuse the existing cmp_surf_n8 run as the lam=0 baseline.
#
#   SURFACE_STUDENT_ENTROPY_COEF=0.5 bash experiments/exp_rl_surf_ent.sh
#   SURFACE_STUDENT_ENTROPY_COEF=1.0 bash experiments/exp_rl_surf_ent.sh
#
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup/common.sh"

# --- keep OPRD rig alive but zero the RKD gradient (teacher forward still runs) ---
export USE_REP_DISTILLATION=False
export REP_DISTILLATION_ONLY=False
export REP_DISTILLATION_COEF=0.0
export REP_ALIGN_LOSS=rkd
export REP_PROJECTOR_MODE=full
export LOG_PROB_TOP_K=0

# --- surface channel is the ONLY training signal (same as ARM A) ---
export USE_SURFACE_REWARD=True
export USE_TOKEN_KL_REWARD=False
export ENABLE_FORMAT_REWARD=False
export ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}
export N_RESPONSES=${N_RESPONSES:-8}

# --- route A: add back the student entropy term (this is the whole point) ---
export SURFACE_STUDENT_ENTROPY_COEF=${SURFACE_STUDENT_ENTROPY_COEF:-1.0}

# lam in the run name so 0.5 / 1.0 sweeps land in distinct dirs.
_LAM_TAG=$(printf "%s" "$SURFACE_STUDENT_ENTROPY_COEF" | tr '.' 'p')
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-armA_surf_ent_lam${_LAM_TAG}_$(date +%Y-%m-%d_%H-%M-%S)}
run_experiment oprd
