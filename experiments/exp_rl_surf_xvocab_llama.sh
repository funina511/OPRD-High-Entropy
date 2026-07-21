#!/bin/bash
# ARM A'-llama — CROSS-VOCAB surface-only (①), teacher = Llama-3.2-3B-Instruct.
#
# Identical to exp_rl_surf_xvocab.sh (surface-only, pure text-manifold distillation,
# token_level_rewards OVERWRITTEN by the length-normalized teacher seq_ll) but swaps
# the teacher from Phi-4-mini-instruct to Llama-3.2-3B-Instruct (vocab 128256, Llama
# arch) to test whether the cross-vocab surface channel generalizes across a second,
# architecturally different teacher tokenizer. Student stays Qwen3-0.6B-Base.
#
#   bash experiments/exp_rl_surf_xvocab_llama.sh
#
# Override knobs (all optional):
#   XVOCAB_TEACHER, EXPERIMENT_NAME, N_RESPONSES, RAY_PORT

export XVOCAB_TEACHER=${XVOCAB_TEACHER:-/mnt/lxy/hf_models/Llama-3.2-3B-Instruct}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-armAx_surf_xvocab_llama32-3b_$(date +%Y-%m-%d_%H-%M-%S)}

# hand off to the shared cross-vocab arm (reads XVOCAB_TEACHER / EXPERIMENT_NAME)
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/exp_rl_surf_xvocab.sh"
