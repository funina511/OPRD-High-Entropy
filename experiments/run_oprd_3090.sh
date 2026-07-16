#!/bin/bash
# CROSS-ARCH OPRD-Bridge on 4x RTX 3090 (the paper's cross-architecture setting).
# Student: Qwen3-0.6B (28L, d=1024)   Teacher: Qwen3-4B (36L, d=2560)   Data: DAPO-Math-5k   Eval: AMC23
#
# Why the low-rank bridge (not the vanilla `full` projector):
#   OPRD-Vanilla requires SAME architecture (identical depth/width). For a cross-arch pair the
#   naive full Linear(d_S->d_T) never aligns (cosine ~0) and rep-only collapses (repetition, acc->0).
#   The paper's fix is OPRD-Bridge: align in a shared low-rank subspace built from the teacher's
#   PCA directions (P_T) + a student projector (P_S). Rank r=8 reaches ~95% cosine cross-arch.
#
# Bridge source: no prebuilt ps_bank.pt here, so P_T is estimated by PCA on the first teacher
#   batch and P_S is trained jointly (REP_FREEZE_PS=False). To use an offline-frozen bridge instead,
#   set REP_LOW_RANK_INIT_CHECKPOINT=/path/to/ps_bank.pt and REP_FREEZE_PS=True.
#
#   bash experiments/run_oprd_3090.sh
set -eo pipefail

# --- env: use the verl py3.12 interpreter, and keep localhost off the clash proxy ---
source /mnt/lxy/miniconda3/etc/profile.d/conda.sh
conda activate verl
export PATH=/mnt/lxy/miniconda3/envs/verl/bin:$PATH        # shell profile shadows conda PATH
export NO_PROXY=localhost,127.0.0.1,0.0.0.0,.wandb.ai,wandb.ai,api.wandb.ai
export no_proxy=localhost,127.0.0.1,0.0.0.0,.wandb.ai,wandb.ai,api.wandb.ai
export WANDB_MODE=${WANDB_MODE:-online}                    # creds in ~/.netrc; api.wandb.ai reachable
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

# --- hardware: GPU0 is used by another user; take the 4 free cards ---
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export CUDA_LAUNCH_BLOCKING=0

# --- paths (datasets live inside the repo, not repo/../datasets) ---
export DATA_DIR=${DATA_DIR:-/mnt/lxy/OPRD-High-Entropy/datasets}
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-/mnt/lxy/hf_models/Qwen3-0.6B}
export REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-/mnt/lxy/hf_models/Qwen3-4B}

# --- OPRD-Bridge (low-rank cross-arch projector) — the point of this script ---
# Uses the OFFLINE-BUILT, FROZEN bridge from Stage 0/1 (scripts/analysis + experiments/README.md).
# Frozen P_S removes the degenerate "rubber-ruler" co-adaptation: to lower rep loss the backbone
# must genuinely move toward the teacher's low-rank subspace. Build the bridge first:
#   bash experiments/run_stage01_bridge.sh        # -> outputs/bridge_construction/rank_8/ps_bank.pt
export REP_PROJECTOR_MODE=${REP_PROJECTOR_MODE:-low_rank}   # PCA teacher bases P_T + student P_S
export REP_LOW_RANK=${REP_LOW_RANK:-8}                      # MUST match the bridge's rank (Stage 1 --ranks)
export REP_LOW_RANK_INIT_CHECKPOINT=${REP_LOW_RANK_INIT_CHECKPOINT:-/mnt/lxy/OPRD-High-Entropy/outputs/bridge_construction/rank_8/ps_bank.pt}
export REP_FREEZE_PS=${REP_FREEZE_PS:-True}                 # freeze the offline bridge (P_S + P_T)
export REP_DISTILLATION_COEF=${REP_DISTILLATION_COEF:-10.0} # bridge default; rep is the only loss here
export REP_DISTILLATION_LAYERS=${REP_DISTILLATION_LAYERS:-all}  # MUST match the bridge's --layer-mode (all)
export REP_DISTILLATION_POSITIONS=${REP_DISTILLATION_POSITIONS:-last_k}
export REP_DISTILLATION_LAST_K=${REP_DISTILLATION_LAST_K:-1024}

if [ ! -f "$REP_LOW_RANK_INIT_CHECKPOINT" ]; then
  echo "ERROR: frozen bridge not found: $REP_LOW_RANK_INIT_CHECKPOINT"
  echo "Build it first with: bash experiments/run_stage01_bridge.sh   (see experiments/README.md)"
  exit 1
fi

# --- logging: skip the is_plot viz block (calls swanlab.log w/o init) ---
export IS_PLOT=${IS_PLOT:-False}

# --- batch / signal: stable scale (verified not to OOM on 3090) ---
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-8}
export N_RESPONSES=${N_RESPONSES:-2}
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-2048}
export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-2048}
export REWARD_MICRO_BSZ=${REWARD_MICRO_BSZ:-8}

# --- schedule: eval + save WITHIN the run ---
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-150}
export TEST_FREQ=${TEST_FREQ:-25}
export SAVE_FREQ=${SAVE_FREQ:-50}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}

# oprd = rep-only; the low_rank overrides above turn it into the cross-arch Bridge run
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_distillation.sh" oprd
