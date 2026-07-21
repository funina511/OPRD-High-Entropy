#!/bin/bash
# Host profile: good3090 (bare-metal, user lxy, GPUs 4-7 free, no proxy).
# Sourced by setup/common.sh. Only MACHINE-SPECIFIC values belong here; anything
# shared across machines lives in common.sh / run_distillation.sh.

# --- interpreter ---
export OPRD_CONDA_SH=${OPRD_CONDA_SH:-/mnt/lxy/miniconda3/etc/profile.d/conda.sh}
export OPRD_CONDA_ENV=${OPRD_CONDA_ENV:-verl}
export OPRD_CONDA_BIN=${OPRD_CONDA_BIN:-/mnt/lxy/miniconda3/envs/verl/bin}

# --- network: no proxy on this box ---
export NO_PROXY=localhost,127.0.0.1,0.0.0.0
export no_proxy="$NO_PROXY"
# no HTTP(S)_PROXY exports -> direct egress

# --- hardware ---
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}

# --- paths ---
export OPRD_REPO_ROOT=${OPRD_REPO_ROOT:-/mnt/lxy/OPRD-High-Entropy}
export MODEL_DIR=${MODEL_DIR:-/mnt/lxy/hf_models}
export DATA_DIR=${DATA_DIR:-${OPRD_REPO_ROOT}/datasets}

# --- ray tuning: defaults are fine here (no ENOMEM / zombie issues observed) ---
export RAY_ZOMBIE_SWEEP=${RAY_ZOMBIE_SWEEP:-0}
# This box runs a system Redis on the standard port 6379, so Ray's GCS cannot bind
# there. Pin a non-Redis port (other concurrent runs sit on 6392/6401).
export RAY_PORT=${RAY_PORT:-6399}
