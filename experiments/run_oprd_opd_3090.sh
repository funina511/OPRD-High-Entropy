#!/bin/bash
# OPRD combined (token-level OPD + representation) on 4x RTX 3090.
# Student: Qwen3-0.6B   Teacher: Qwen3-4B   Data: DAPO-Math-5k   Eval: AMC23
#
# Why this instead of rep-only: pure rep-only OPRD collapses here (the random
# full projector never aligns, cosine~0, output degenerates to repetition and
# AMC23 -> 0 by ~step 50). Token-level OPD (log_prob_top_k=16) is the primary,
# well-behaved distillation signal that anchors the output distribution; the
# representation loss is kept as a small auxiliary term (coef=0.1).
#
#   bash experiments/run_oprd_opd_3090.sh
#   REP_DISTILLATION_COEF=0.3 MINI_BATCH_SIZE=16 bash experiments/run_oprd_opd_3090.sh
set -eo pipefail

# --- env: use the verl py3.12 interpreter, and keep localhost off the clash proxy ---
source /mnt/lxy/miniconda3/etc/profile.d/conda.sh
conda activate verl
export PATH=/mnt/lxy/miniconda3/envs/verl/bin:$PATH        # shell profile shadows conda PATH
export NO_PROXY=localhost,127.0.0.1,0.0.0.0,.wandb.ai,wandb.ai,api.wandb.ai
export no_proxy=localhost,127.0.0.1,0.0.0.0,.wandb.ai,wandb.ai,api.wandb.ai
export WANDB_MODE=${WANDB_MODE:-online}                    # creds in ~/.netrc; api.wandb.ai reachable
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}                 # models are local dirs; don't call the hub
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

# --- hardware: GPU0 is used by another user; take the 4 free cards ---
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export CUDA_LAUNCH_BLOCKING=0                              # never 1 for real runs
# NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True — vLLM's memory pool asserts against it.

# --- paths (datasets live inside the repo, not repo/../datasets) ---
export DATA_DIR=${DATA_DIR:-/mnt/lxy/OPRD-High-Entropy/datasets}
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-/mnt/lxy/hf_models/Qwen3-0.6B}
export REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-/mnt/lxy/hf_models/Qwen3-4B}

# --- combined-distillation knobs (the point of this script) ---
export REP_DISTILLATION_COEF=${REP_DISTILLATION_COEF:-0.1} # rep is AUXILIARY; keep small so OPD dominates
export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-16}                # token-level OPD signal (set by oprd_opd preset too)
export REP_DISTILLATION_LAYERS=${REP_DISTILLATION_LAYERS:-even}
export REP_DISTILLATION_LAST_K=${REP_DISTILLATION_LAST_K:-1024}

# --- batch / signal: stable scale first (verified not to collapse) ---
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-8}              # train_batch_size = mbs * PARALLEL_SIZE
export N_RESPONSES=${N_RESPONSES:-2}
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-2048}           # bump to 4096/8192 once curve looks healthy
export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-2048}
export REWARD_MICRO_BSZ=${REWARD_MICRO_BSZ:-8}            # teacher fwd; lower if OOM

# --- logging: skip the is_plot viz block (it calls swanlab.log without swanlab
#     initialized -> harmless but noisy errors every 10 steps; scalars still go to wandb) ---
export IS_PLOT=${IS_PLOT:-False}

# --- schedule: eval + save WITHIN the run ---
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-150}
export TEST_FREQ=${TEST_FREQ:-25}
export SAVE_FREQ=${SAVE_FREQ:-50}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}         # baseline AMC23 score at step 0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_distillation.sh" oprd_opd
