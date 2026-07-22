#!/bin/bash
# ARM A-ent-DUAL — surface + PER-TOKEN entropy, per-seq de-meaned (token_dual, plan A).
#
# Keeps cross-vocab support. Two channels combined by a raw-nats lam:
#
#   A_t = grpo_adv(teacher_seq_ll; group-mean baseline, NO std)         # teacher channel
#         + lam * ( -logp_S(y_t) - mean_t(-logp_S) )                    # entropy channel (per-token, raw nats)
#
# WHY NO std on the entropy channel (this is the fix): an earlier version divided the
# entropy term by its batch std to force unit RMS. But -logp_S's variance is exactly
# what the entropy term optimizes -- as the policy went uniform the std SHRANK
# (2.24 -> 0.64), so dividing AMPLIFIED the push -> positive feedback -> runaway to
# uniform (entropy ~11.7 = log|V|, acc 0.03). Per-seq de-mean WITHOUT std self-
# corrects: near uniform, -logp_S -> log|V| for all t, per-seq mean -> log|V| too, so
# (ent - mean) -> 0 and the entropy advantage vanishes -- same negative feedback that
# keeps token_raw stable.
#
# Teacher keeps its group-mean baseline but NO std norm (NORM_ADV_BY_STD=False) so it
# retains absolute nats scale and can fight the entropy push (before, std-flattening
# left teacher_ll at -7.7 from step 1, unable to hold).
#
# NOTE: NOT OPD (teacher term is sequence-level; cross-vocab has no per-token teacher).
# For the exact same-vocab OPD anchor use exp_rl_surf_ent_tokraw.sh.
#
# The entropy channel only touches the student's OWN token logp -> works cross-vocab.
# Sweep lam small (teacher must stay in charge; entropy raw ~O(2-3) vs teacher ~O(1)):
#   SURFACE_STUDENT_ENTROPY_COEF=0.2 bash experiments/exp_rl_surf_ent_tok.sh
#   SURFACE_STUDENT_ENTROPY_COEF=0.1 bash experiments/exp_rl_surf_ent_tok.sh
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

# --- token_dual (plan A): GRPO teacher (group-mean, no std) + per-seq-demeaned entropy ---
export SURFACE_ENTROPY_MODE=token_dual
export ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}
export N_RESPONSES=${N_RESPONSES:-8}
export SURFACE_STUDENT_ENTROPY_COEF=${SURFACE_STUDENT_ENTROPY_COEF:-0.2}
# Teacher keeps its group-mean baseline but drops GRPO std normalization: std-
# flattening left teacher too weak to hold against the entropy push. Group-mean alone
# still cancels the absolute PPL scale while preserving relative nats magnitude.
export NORM_ADV_BY_STD=${NORM_ADV_BY_STD:-False}

_LAM_TAG=$(printf "%s" "$SURFACE_STUDENT_ENTROPY_COEF" | tr '.' 'p')
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-armA_surf_dual_lam${_LAM_TAG}_$(date +%Y-%m-%d_%H-%M-%S)}
run_experiment oprd
