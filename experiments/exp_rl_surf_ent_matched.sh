#!/bin/bash
# ARM A-ent (MATCHED) — SAME-VOCAB surface + student-entropy, SEQUENCE-LEVEL (route A),
# scale-locked for a direct comparison against cmp_surf_n8_r8192.
#
# Identical rig to that surface-only baseline (same teacher Qwen3-4B, same-vocab,
# n=8, resp=8192, mbs=8, steps=150) EXCEPT the reward gains the student entropy term:
#
#   r(y) = mean_t [ logp_T(y_t) - lam * logp_S(y_t) ]  = E[logp_T] + lam*H(pi)
#   A_i  = ( r_i - mean_group r ) / std_group          # GRPO, one scalar per response
#
# So this is the lam>0 point that pairs with cmp_surf_n8_r8192 as the lam=0 point:
# same-vocab, IDENTICAL scale, only the entropy coefficient differs. logp_S is the
# DETACHED old_log_probs (reward stays constant, no PG break). This is the setting
# that converged stably in armA_surf_ent_lam0p5; here we just pin scale to the
# baseline so the two curves are directly overlayable.
#
# Pairs with the cross-vocab twin exp_rl_surf_ent_xvocab.sh (same route A, Phi-4-mini
# teacher) so same-vocab vs cross-vocab is a clean apples-to-apples at fixed scale.
#
#   SURFACE_STUDENT_ENTROPY_COEF=0.5 bash experiments/exp_rl_surf_ent_matched.sh
#
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup/common.sh"

# --- keep OPRD rig alive but zero the RKD gradient (teacher forward still runs) ---
export USE_REP_DISTILLATION=False
export REP_DISTILLATION_ONLY=False
export REP_DISTILLATION_COEF=0.0
export REP_ALIGN_LOSS=rkd
export REP_PROJECTOR_MODE=full
export LOG_PROB_TOP_K=0

# --- same-vocab surface (Qwen3-4B teacher, inherited from common.sh) is the ONLY
#     training signal; cross-vocab flag stays OFF (matches cmp_surf_n8_r8192) ---
export USE_SURFACE_REWARD=True
export USE_TOKEN_KL_REWARD=False
export ENABLE_FORMAT_REWARD=False
export ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}

# --- route A: sequence-level entropy folded into the surface scalar; GRPO keeps
#     its group-mean baseline AND std (the setting that converged) ---
export SURFACE_ENTROPY_MODE=${SURFACE_ENTROPY_MODE:-seq}
export SURFACE_STUDENT_ENTROPY_COEF=${SURFACE_STUDENT_ENTROPY_COEF:-0.5}
export NORM_ADV_BY_STD=${NORM_ADV_BY_STD:-True}

# --- scale HARD-PINNED to cmp_surf_n8_r8192 (n=8, resp=8192, mbs=8, steps=150).
#     Written with plain '=' (NOT ':-') so any stale env values from the shell
#     session CANNOT override them — this run must be scale-identical to the
#     baseline or the comparison is meaningless. Use a different script to sweep. ---
export N_RESPONSES=8
export MAX_RESP_LENGTH=8192
export MAX_VAL_RESP_LENGTH=8192
export MINI_BATCH_SIZE=8
export TOTAL_TRAINING_STEPS=150

_LAM_TAG=$(printf "%s" "$SURFACE_STUDENT_ENTROPY_COEF" | tr '.' 'p')
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-armA_surf_ent_seq_matched_lam${_LAM_TAG}_$(date +%Y-%m-%d_%H-%M-%S)}
run_experiment oprd
