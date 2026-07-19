#!/bin/bash
# RKD-D + within-response RKD-A (same as P2 da_all) PLUS student outcome/format RL.
#
# Unlike plain `rep_only` RKD, this turns on policy gradients so lm_head gets updates
# from student rule rewards (AMC boxed accuracy + soft format bonus). Teacher is still
# used for hidden RKD only — NO token-level reverse-KL / OPD (cross-vocab safe).
#
#   bash experiments141/exp_rel_rkd_da_rl.sh
#   N_RESPONSES=4 bash experiments141/exp_rel_rkd_da_rl.sh   # stronger GRPO groups
#
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup/common.sh"

# --- RKD (match P2 da_all) ---
export REP_PROJECTOR_MODE=full
export REP_ALIGN_LOSS=rkd
export REP_CHUNKS=${REP_CHUNKS:-8}
export REP_DISTILLATION_COEF=${REP_DISTILLATION_COEF:-3.0}
export REP_DISTILLATION_LAYERS=all
export REP_DISTILLATION_POSITIONS=${REP_DISTILLATION_POSITIONS:-last_k}
export REP_DISTILLATION_LAST_K=${REP_DISTILLATION_LAST_K:-1024}
export REP_RKD_ANGLE_COEF=${REP_RKD_ANGLE_COEF:-2.0}

# --- Student RL (outcome + format); keep OPRD method but disable rep-only ---
export USE_REP_DISTILLATION=True
export REP_DISTILLATION_ONLY=False
export LOG_PROB_TOP_K=0
export USE_TOKEN_KL_REWARD=False
export ENABLE_FORMAT_REWARD=True
export FORMAT_REWARD_COEF=${FORMAT_REWARD_COEF:-0.1}
export ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}
export GRPO_OUTCOME_WEIGHT=${GRPO_OUTCOME_WEIGHT:-1.0}
# n=2 matches P2 compute; bump to 4 if GRPO variance is too high
export N_RESPONSES=${N_RESPONSES:-2}

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-rel_rkd_da_rl_a${REP_RKD_ANGLE_COEF}_c${REP_DISTILLATION_COEF}_f${FORMAT_REWARD_COEF}_$(date +%Y-%m-%d_%H-%M-%S)}
run_experiment oprd
