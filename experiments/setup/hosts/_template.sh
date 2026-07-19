#!/bin/bash
# Host profile TEMPLATE. Adding a new machine:
#   1. cp _template.sh <name>.sh          (<name> = hostname, or the OPRD_HOST you'll export)
#   2. fill in every REQUIRED value below  (common.sh fails fast if any is empty)
#   3. if the box's hostname is stable, it auto-selects; if it's an ephemeral
#      container id, add `export OPRD_HOST=<name>` to that box's shell rc.
# Only MACHINE-SPECIFIC values belong here — shared logic stays in common.sh /
# run_distillation.sh, so `git pull` never clobbers a box.
#
# The leading files (good3090.sh, siton141.sh) are worked examples — copy the one
# closest to your machine (proxy vs no-proxy) rather than this bare template.

# --- REQUIRED: interpreter (conda) -----------------------------------------
export OPRD_CONDA_SH=${OPRD_CONDA_SH:-/PATH/TO/miniconda3/etc/profile.d/conda.sh}
export OPRD_CONDA_ENV=${OPRD_CONDA_ENV:-verl}
export OPRD_CONDA_BIN=${OPRD_CONDA_BIN:-/PATH/TO/miniconda3/envs/verl/bin}

# --- REQUIRED: hardware ----------------------------------------------------
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}     # device list for this box
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-1}              # count that trainer/ray use

# --- REQUIRED: paths (model files must sit under $MODEL_DIR/<HF name>) ------
export OPRD_REPO_ROOT=${OPRD_REPO_ROOT:-/PATH/TO/OPRD-High-Entropy}
export MODEL_DIR=${MODEL_DIR:-/PATH/TO/hf_models}
export DATA_DIR=${DATA_DIR:-${OPRD_REPO_ROOT}/datasets}

# --- network: pick ONE of the two blocks -----------------------------------
# (a) direct egress (no proxy); keep wandb reachable:
export NO_PROXY=localhost,127.0.0.1,0.0.0.0,.wandb.ai,wandb.ai,api.wandb.ai
export no_proxy="$NO_PROXY"
# (b) behind an HTTP proxy (wandb online THROUGH it) — uncomment and drop block (a):
# export NO_PROXY=localhost,127.0.0.1,0.0.0.0
# export no_proxy="$NO_PROXY"
# export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:7890}"; export http_proxy="$HTTP_PROXY"
# export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:7890}"; export https_proxy="$HTTPS_PROXY"
# unset ALL_PROXY all_proxy         # socks ALL_PROXY makes wandb-core hang

# --- ray tuning: leave 0/unset unless the host actually needs it -----------
export RAY_ZOMBIE_SWEEP=${RAY_ZOMBIE_SWEEP:-0}            # 1 = kill zombie gcs/raylet + stale temp-dir
# export RAY_OBJECT_STORE_MEMORY=${RAY_OBJECT_STORE_MEMORY:-10000000000}  # cap if ENOMEM at ray.init
# export RAY_NUM_CPUS=${RAY_NUM_CPUS:-8}                                  # cap prestart workers
