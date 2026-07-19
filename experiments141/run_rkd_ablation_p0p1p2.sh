#!/bin/bash
# Sequential RKD ablations (1 GPU): P0 last-layer → P1 all-tokens → P2 RKD-DA all-layers.
# Baseline reference: rel_rkd_c3.0 (layers=all, last_k=1024, angle=0).
# P2 retargeted to layers=all after P0 showed last-only underperforms.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /root/siton-tmp/home/liuxinyu/miniconda3/etc/profile.d/conda.sh
conda activate verl
export PATH=/root/siton-tmp/home/liuxinyu/miniconda3/envs/verl/bin:$PATH

echo "=========================================="
echo "RKD P0/P1/P2 ablation start: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader)"
echo "=========================================="

run_one() {
  local script="$1"
  local tag="$2"
  echo ""
  echo ">>> START ${tag} (${script}) at $(date)"
  bash "${script}"
  local ec=$?
  echo ">>> END ${tag} exit=${ec} at $(date)"
  return $ec
}

run_one exp_rel_rkd_last.sh P0_last
ec0=$?
run_one exp_rel_rkd_alltok.sh P1_alltok
ec1=$?
# Filename kept for in-flight queue; script forwards to da_all (layers=all + A).
run_one exp_rel_rkd_da_last.sh P2_da_all
ec2=$?

echo "=========================================="
echo "RKD ablation done: $(date)"
echo "P0 exit=${ec0}  P1 exit=${ec1}  P2 exit=${ec2}"
echo "=========================================="
exit $(( ec0 != 0 || ec1 != 0 || ec2 != 0 ? 1 : 0 ))
