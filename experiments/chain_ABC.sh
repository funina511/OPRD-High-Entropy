#!/bin/bash
# Auto-chain arms B and C after arm A finishes (all n=2, same 4 GPUs 0,5,6,7).
# A is already running as $A_PID; this waits for it, then runs B, then C.
# Each arm gets its own RAY_PORT so stale sessions never collide.
set -u
REPO=/mnt/lxy/OPRD-High-Entropy
A_PID="${A_PID:?set A_PID to arm A launcher pid}"
export OPRD_HOST=good3090 CUDA_VISIBLE_DEVICES=0,5,6,7

wait_pid() { while kill -0 "$1" 2>/dev/null; do sleep 60; done; }
free_ray() { RAY_ADDRESS="127.0.0.1:$1" ray stop >/dev/null 2>&1 || true; sleep 5; }

echo "[chain] waiting for arm A (pid $A_PID) ..."
wait_pid "$A_PID"
echo "[chain] arm A exited at $(date). launching arm B."

free_ray 6396
RAY_PORT=6396 nohup bash "$REPO/experiments/exp_rel_rkd_da_all.sh" \
  > "$REPO/nohup_armB_rkd.log" 2>&1 &
B_PID=$!
echo "[chain] arm B pid $B_PID (RKD-only, da_all). waiting ..."
wait_pid "$B_PID"
echo "[chain] arm B exited at $(date). launching arm C."

free_ray 6397
RAY_PORT=6397 nohup bash "$REPO/experiments/exp_rel_rkd_da_rl_surf.sh" \
  > "$REPO/nohup_armC_rkd_surf.log" 2>&1 &
C_PID=$!
echo "[chain] arm C pid $C_PID (RKD + surface). waiting ..."
wait_pid "$C_PID"
echo "[chain] arm C exited at $(date). ABC done."
