#!/bin/bash
# ARM A' — CROSS-VOCAB surface-only (①): on-policy text-manifold distillation
# where the teacher's vocabulary DIFFERS from the student's.
#
# Student: Qwen3-0.6B-Base (vocab 151936).  Teacher: Phi-4-mini-instruct (vocab
# 200064, Phi3 arch). The student's decoded TEXT — not its token ids — crosses the
# vocab boundary: the RM worker decodes the student response, re-tokenizes it with
# the TEACHER tokenizer/chat template, and returns a length-normalized scalar
# seq_ll = sum(teacher_logp over teacher response span) / teacher_resp_token_count.
# token_level_rewards is OVERWRITTEN by this signal (pure distillation: no outcome,
# no format, no RKD). Compare against arm A (same-vocab Qwen3-4B teacher) to test
# whether the surface channel survives a genuinely different tokenizer + model.
#
#   bash experiments/exp_rl_surf_xvocab.sh
#
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup/common.sh"

# --- cross-vocab teacher + student tokenizer for decode/re-tokenize bridge ---
# NOTE: common.sh (sourced above) already exports REWARD_MODEL_PATH=Qwen3-4B, so a
# `${REWARD_MODEL_PATH:-...}` fallback would NOT override it. Force the teacher here
# unconditionally; allow a deliberate override via XVOCAB_TEACHER.
export REWARD_MODEL_PATH=${XVOCAB_TEACHER:-${MODEL_DIR}/Phi-4-mini-instruct}
export REWARD_INPUT_TOKENIZER=${REWARD_INPUT_TOKENIZER:-$ACTOR_MODEL_PATH}

# --- surface channel is the ONLY training signal (cross-vocab scalar path) ---
export USE_SURFACE_REWARD=True
export SURFACE_REWARD_CROSS_VOCAB=True
export USE_TOKEN_KL_REWARD=False
export ENABLE_FORMAT_REWARD=False
export ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}

# --- RKD / hidden-repr OFF: teacher hidden dim (3072) != student (1024) AND vocab
# differs, so rep distillation would need a cross-arch projector. Keep this arm
# surface-only to isolate the cross-vocab surface question. The lean cross-vocab
# path also skips the all-layer hidden extraction entirely. ---
export USE_REP_DISTILLATION=False
export REP_DISTILLATION_ONLY=False
export REP_DISTILLATION_COEF=0.0
export LOG_PROB_TOP_K=0

# n=2 matches arms A/B/C rollout budget.
export N_RESPONSES=${N_RESPONSES:-2}

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-armAx_surf_xvocab_phi4mini_$(date +%Y-%m-%d_%H-%M-%S)}
run_experiment oprd
