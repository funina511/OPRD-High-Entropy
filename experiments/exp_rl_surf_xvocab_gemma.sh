#!/bin/bash
# ARM A'-Gemma — CROSS-VOCAB surface-only (①), teacher = Gemma 3 4B IT
# text-only language tower.
#
# Identical to exp_rl_surf_xvocab.sh (surface-only, pure text-manifold distillation,
# token_level_rewards OVERWRITTEN by the length-normalized teacher seq_ll) but swaps
# the teacher from Phi-4-mini-instruct to the Gemma 3 4B IT language tower
# (Gemma3ForCausalLM, vocab size 262208). The text-only checkpoint is extracted
# from gemma-3-4b-it and omits its SigLIP vision tower, avoiding multimodal FSDP
# wrapping while preserving the 4B text model. Student stays Qwen3-0.6B-Base.
#
#   bash experiments/exp_rl_surf_xvocab_gemma.sh
#
# Override knobs (all optional):
#   XVOCAB_TEACHER, EXPERIMENT_NAME, N_RESPONSES, RAY_PORT

export XVOCAB_TEACHER=${XVOCAB_TEACHER:-/mnt/lxy/hf_models/gemma-3-4b-it-text-only}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-armAx_surf_xvocab_gemma3-4b_$(date +%Y-%m-%d_%H-%M-%S)}

# hand off to the shared cross-vocab arm (reads XVOCAB_TEACHER / EXPERIMENT_NAME)
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/exp_rl_surf_xvocab.sh"
