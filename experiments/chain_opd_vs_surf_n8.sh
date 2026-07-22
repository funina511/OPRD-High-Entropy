#!/bin/bash
# Sequential same-vocab comparison on the single H200: OPD (token reverse-KL) vs
# surface-only, MATCHED at n=8, resp=8192, teacher=Qwen3-4B, student=Qwen3-0.6B-Base.
# Single card -> run serially; each arm's vLLM reservation would collide if concurrent.
set -uo pipefail
export OPRD_HOST=${OPRD_HOST:-siton141}
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- matched knobs for BOTH arms (exported so the exp_*.sh inherit them) ---
export N_RESPONSES=8
export MAX_RESP_LENGTH=8192
export MAX_VAL_RESP_LENGTH=8192
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-8}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-150}

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p logs
CHAIN_LOG="logs/chain_opd_surf_n8_${TS}.log"

cleanup_between() {   # free ray + GPU before the next arm starts
  echo "[chain] cleanup: stopping ray + killing main_ppo/vllm" | tee -a "$CHAIN_LOG"
  ray stop --force 2>/dev/null
  pkill -9 -f main_ppo 2>/dev/null
  pkill -9 -f "vllm"    2>/dev/null
  sleep 15
  nvidia-smi --query-gpu=memory.used --format=csv,noheader | tee -a "$CHAIN_LOG"
}

run_arm() {   # $1=exp script  $2=RAY_PORT  $3=EXPERIMENT_NAME
  local script="$1" port="$2" name="$3"
  echo "[chain] === START $name ($script, RAY_PORT=$port) $(date) ===" | tee -a "$CHAIN_LOG"
  # subshell so a set -e exit inside the exp script can't kill the chain
  ( RAY_PORT="$port" EXPERIMENT_NAME="$name" bash "experiments/$script" ) \
      >> "$CHAIN_LOG" 2>&1
  echo "[chain] === END $name rc=$? $(date) ===" | tee -a "$CHAIN_LOG"
}

echo "[chain] host=$OPRD_HOST n=$N_RESPONSES resp=$MAX_RESP_LENGTH steps=$TOTAL_TRAINING_STEPS" | tee -a "$CHAIN_LOG"

# Arm 1: OPD (token reverse-KL). Signal quality is n-independent, but run at n=8 too
# so the rollout budget matches surface exactly (fair, same #sequences/step).
run_arm exp_rl_opd_rkl.sh 6401 "cmp_opd_rkl_n8_r8192_${TS}"
cleanup_between

# Arm 2: surface-only. This is the arm n=8 actually rescues.
run_arm exp_rl_surf.sh    6402 "cmp_surf_n8_r8192_${TS}"

echo "[chain] ALL DONE $(date)" | tee -a "$CHAIN_LOG"
