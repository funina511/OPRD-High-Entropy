#!/bin/bash
# ARM A-opd — reverse-KL token-level OPD (on-policy distillation).
#
# Matched sibling of exp_rl_surf.sh (ARM A, surface-only). SAME teacher (Qwen3-4B),
# SAME student (Qwen3-0.6B-Base), SAME rollout budget (n=2), SAME data — the ONLY
# difference is the training signal:
#   surface (ARM A):  reward = length-normalized teacher log-lik of student TEXT,
#                     scalar-on-last-token, credit via GRPO group baseline.
#   opd    (this arm): reward = per-token reverse-KL  rm_scores = teacher_logp -
#                     student_logp  on the sampled tokens (LOG_PROB_TOP_K=0, no
#                     top-k distribution matching), credit via token_reward_direct
#                     (each token's reverse-KL IS its advantage, no group baseline).
#
# Run this on GPUs 0-3 alongside a Qwen3-4B surface arm on 4-7 to form the pair.
#
#   CUDA_VISIBLE_DEVICES=0,1,2,3 RAY_PORT=6480 bash experiments/exp_rl_opd_rkl.sh
#
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup/common.sh"

# --- reverse-KL token OPD is the ONLY training signal ---
export USE_TOKEN_KL_REWARD=True          # rm_scores = teacher_logp - student_logp
export LOG_PROB_TOP_K=0                   # pure reverse-KL on sampled tokens (no top-k match)
export ADV_ESTIMATOR=${ADV_ESTIMATOR:-token_reward_direct}  # per-token reward = advantage
export USE_SURFACE_REWARD=False          # not the surface arm
export ENABLE_FORMAT_REWARD=False        # pure distillation (match surface arm)

# --- rep / hidden-repr OFF: this is token-distribution OPD, not representation KD.
# Skips the ~9.5GB/step all-layer hidden extraction (pure waste here). ---
export USE_REP_DISTILLATION=False
export REP_DISTILLATION_ONLY=False
export REP_DISTILLATION_COEF=0.0

# n=2 matches the surface arm's rollout budget and fits 4x3090.
export N_RESPONSES=${N_RESPONSES:-2}

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-armA_opd_rkl_qwen3-4b_$(date +%Y-%m-%d_%H-%M-%S)}
run_experiment opd
