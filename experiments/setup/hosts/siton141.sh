#!/bin/bash
# Host profile: siton141 (Docker container, user root, single GPU, clash HTTP proxy).
# Container hostname is an ephemeral ID, so select this profile via `export OPRD_HOST=siton141`
# in the container's shell rc (see setup/common.sh selection order).

# --- interpreter ---
export OPRD_CONDA_SH=${OPRD_CONDA_SH:-/root/siton-tmp/home/liuxinyu/miniconda3/etc/profile.d/conda.sh}
export OPRD_CONDA_ENV=${OPRD_CONDA_ENV:-verl}
export OPRD_CONDA_BIN=${OPRD_CONDA_BIN:-/root/siton-tmp/home/liuxinyu/miniconda3/envs/verl/bin}

# --- network: wandb online THROUGH the clash proxy ---
export NO_PROXY=localhost,127.0.0.1,0.0.0.0
export no_proxy="$NO_PROXY"
export HTTP_PROXY="${HTTP_PROXY:-${http_proxy:-http://127.0.0.1:7890}}"
export HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy:-http://127.0.0.1:7890}}"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
# socks ALL_PROXY makes wandb-core init hang; prefer HTTP(S)_PROXY.
unset ALL_PROXY all_proxy

# --- hardware ---
# This box is a SINGLE NVIDIA H200 NVL (143 GB HBM), not the 4x3090 in the README.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-1}

# --- memory (single 143 GB card, PARALLEL_SIZE=1) ---
# The student (Qwen3-0.6B) is tiny, so vLLM's KV-cache reservation dwarfs everything
# else. Cap it at 0.6 (~86 GB) so the <=4B teacher/RM forward + actor FSDP have ~57 GB
# of headroom and never collide with the reservation. Surface-only skips hidden
# extraction entirely, so 8192-token rollouts fit comfortably here.
export GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.6}
# Park the teacher/RM weights on CPU when not scoring — cheap insurance against the
# worst-case "vLLM reservation + 4B teacher resident at once" overlap.
export REWARD_PARAM_OFFLOAD=${REWARD_PARAM_OFFLOAD:-True}

# --- paths ---
export OPRD_REPO_ROOT=${OPRD_REPO_ROOT:-/root/siton-tmp/home/liuxinyu/OPRD-High-Entropy}
export MODEL_DIR=${MODEL_DIR:-/root/siton-tmp/home/liuxinyu/hf_models}
export DATA_DIR=${DATA_DIR:-${OPRD_REPO_ROOT}/datasets}

# --- ray tuning: this host needs zombie sweep + capped object-store/CPUs (ENOMEM guard) ---
export RAY_ZOMBIE_SWEEP=${RAY_ZOMBIE_SWEEP:-1}
export RAY_OBJECT_STORE_MEMORY=${RAY_OBJECT_STORE_MEMORY:-10000000000}
export RAY_NUM_CPUS=${RAY_NUM_CPUS:-8}
