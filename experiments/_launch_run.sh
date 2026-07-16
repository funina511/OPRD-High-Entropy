#!/bin/bash
# Robust launcher: kill any prior run, start OPRD with memory-safe overrides for 4x 3090.
# The 4B teacher forward is the OOM bottleneck -> offload its params to CPU + micro-batch 1.
set +e
cd /mnt/lxy/OPRD-High-Entropy
pkill -9 -f "main_ppo" 2>/dev/null
sleep 5
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/oprd_run_${TS}.log"

export REWARD_PARAM_OFFLOAD="${REWARD_PARAM_OFFLOAD:-True}"   # offload 4B teacher params to CPU (~8GB saved)
export REWARD_MICRO_BSZ="${REWARD_MICRO_BSZ:-1}"             # teacher forward peak scales with this
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.45}"                  # leave room for FSDP teacher+actor
export MAX_RESP_LENGTH="${MAX_RESP_LENGTH:-2048}"
export MAX_VAL_RESP_LENGTH="${MAX_VAL_RESP_LENGTH:-2048}"
export N_RESPONSES="${N_RESPONSES:-2}"                       # fewer sequences through the teacher
export MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-8}"

setsid bash experiments/run_oprd_3090.sh oprd > "$LOG" 2>&1 &
PGID=$!
echo "$PGID" > logs/oprd_run.pgid
echo "$LOG" > logs/oprd_run.logpath
echo "launched PGID=$PGID LOG=$LOG"
echo "  REWARD_PARAM_OFFLOAD=$REWARD_PARAM_OFFLOAD REWARD_MICRO_BSZ=$REWARD_MICRO_BSZ GPU_MEM_UTIL=$GPU_MEM_UTIL"
echo "  MAX_RESP_LENGTH=$MAX_RESP_LENGTH N_RESPONSES=$N_RESPONSES MINI_BATCH_SIZE=$MINI_BATCH_SIZE"
