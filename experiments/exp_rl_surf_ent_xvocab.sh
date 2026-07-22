#!/bin/bash
# ARM A'-ent — CROSS-VOCAB surface + student-entropy, SEQUENCE-LEVEL (route A).
#
# The cross-vocab counterpart of exp_rl_surf_ent.sh (which converged stably at
# lam=0.5, same-vocab). Teacher tokenizer != student, so there is NO per-token
# teacher logp -> the entropy term is folded into the SAME length-normalized
# sequence scalar as the teacher term, then GRPO (group-mean + std) normalizes
# the combined scalar:
#
#   r(y) = teacher_surface_ll(y)            # teacher-token-mean logp (cross-vocab)
#          - lam * student_mean_logp(y)     # = + lam * H(pi), student-token-mean
#   A_i  = ( r_i - mean_group r ) / std_group     # GRPO, one scalar per response
#
# WHY THIS DOESN'T BLOW UP (unlike token_dual): entropy and teacher live in ONE
# jointly-normalized scalar, so the teacher can veto a high-entropy sample; the
# baseline is GROUP-relative (rewards "more random THAN its 8 siblings", not an
# absolute entropy target -> self-limiting as the group drifts up together); and
# std re-normalizes the combined advantage to O(1) every step. student logp is the
# DETACHED old_log_probs so the reward stays constant (no PG break).
#
# Cross-vocab handled entirely by the code path (ray_trainer.py teacher_surface_ll
# branch, ent_mode="seq"): no per-token teacher needed. NOT the per-token dual
# channel (that is same-vocab-only via token_raw / exp_rl_surf_ent_tokraw.sh).
#
# Scale MATCHED to cmp_surf_n8_r8192 (n=8, resp=8192, mbs=8, steps=150) so it is a
# budget-fair comparison against that surface-only baseline.
#
#   SURFACE_STUDENT_ENTROPY_COEF=0.5 bash experiments/exp_rl_surf_ent_xvocab.sh
#
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup/common.sh"

# --- cross-vocab teacher + student tokenizer for decode/re-tokenize bridge ---
# common.sh exports REWARD_MODEL_PATH=Qwen3-4B, so a ${VAR:-...} fallback would NOT
# override it. Force the teacher unconditionally; allow override via XVOCAB_TEACHER.
export REWARD_MODEL_PATH=${XVOCAB_TEACHER:-${MODEL_DIR}/Phi-4-mini-instruct}
export REWARD_INPUT_TOKENIZER=${REWARD_INPUT_TOKENIZER:-$ACTOR_MODEL_PATH}

# --- surface channel is the ONLY training signal (cross-vocab scalar path) ---
export USE_SURFACE_REWARD=True
export SURFACE_REWARD_CROSS_VOCAB=True
export USE_TOKEN_KL_REWARD=False
export ENABLE_FORMAT_REWARD=False
export ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}

# --- RKD / hidden-repr OFF (cross-arch dims + vocab differ): keep surface-only ---
export USE_REP_DISTILLATION=False
export REP_DISTILLATION_ONLY=False
export REP_DISTILLATION_COEF=0.0
export LOG_PROB_TOP_K=0

# --- route A: sequence-level entropy folded into the surface scalar; GRPO keeps
#     its group-mean baseline AND std (the setting that converged same-vocab) ---
export SURFACE_ENTROPY_MODE=${SURFACE_ENTROPY_MODE:-seq}
export SURFACE_STUDENT_ENTROPY_COEF=${SURFACE_STUDENT_ENTROPY_COEF:-0.5}
export NORM_ADV_BY_STD=${NORM_ADV_BY_STD:-True}

# --- scale matched to cmp_surf_n8_r8192 (budget-fair baseline comparison) ---
export N_RESPONSES=${N_RESPONSES:-8}
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-8192}
export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-8192}
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-8}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-150}

_LAM_TAG=$(printf "%s" "$SURFACE_STUDENT_ENTROPY_COEF" | tr '.' 'p')
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-armAx_surf_ent_seq_lam${_LAM_TAG}_$(date +%Y-%m-%d_%H-%M-%S)}
run_experiment oprd
