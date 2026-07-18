#!/bin/bash
# Shared runtime base for all experiments. SOURCE this (do not exec it):
#   source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
# then set the few knobs your experiment differs on, and finally:
#   run_experiment <oprd|opd|oprd_opd>
#
# Everything here is a shared default via ${VAR:-...}, so an experiment can
# override any value either BEFORE sourcing (env on the command line) or AFTER
# sourcing (plain assignment) — both win over these defaults.
set -eo pipefail

# --- interpreter + network: verl py3.12, keep localhost/wandb off the clash proxy ---
source /mnt/lxy/miniconda3/etc/profile.d/conda.sh
conda activate verl
export PATH=/mnt/lxy/miniconda3/envs/verl/bin:$PATH        # shell profile shadows conda PATH
export NO_PROXY=localhost,127.0.0.1,0.0.0.0,.wandb.ai,wandb.ai,api.wandb.ai
export no_proxy="$NO_PROXY"
export WANDB_MODE=${WANDB_MODE:-online}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

# --- hardware (override CUDA_VISIBLE_DEVICES / RAY_PORT to run several at once) ---
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export CUDA_LAUNCH_BLOCKING=0
export RAY_PORT=${RAY_PORT:-6379}
export SKIP_RAY_STOP=${SKIP_RAY_STOP:-1}                   # never kill another run's ray cluster

# --- paths ---
export DATA_DIR=${DATA_DIR:-/mnt/lxy/OPRD-High-Entropy/datasets}
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-/mnt/lxy/hf_models/Qwen3-0.6B-Base}
export REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-/mnt/lxy/hf_models/Qwen3-4B}

# --- batch / signal: stable scale verified not to OOM on 3090 ---
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-8}
export N_RESPONSES=${N_RESPONSES:-2}
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-2048}
export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-2048}
export REWARD_MICRO_BSZ=${REWARD_MICRO_BSZ:-8}

# --- schedule + logging (eval/save WITHIN the run; skip swanlab viz block) ---
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-150}
export TEST_FREQ=${TEST_FREQ:-25}
export SAVE_FREQ=${SAVE_FREQ:-50}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
export IS_PLOT=${IS_PLOT:-False}

# run_experiment <method>: hand off to the core engine. `exec` so signals/exit map 1:1.
run_experiment() {
  local method="${1:?run_experiment needs a method: oprd | opd | oprd_opd}"
  exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/run_distillation.sh" "$method"
}
