#!/bin/bash
# One-shot env build for OPRD (verl py3.12).
# pip -> tuna mirror DIRECT (no proxy); only the flash-attn github wheel uses the clash proxy.
set -eo pipefail
source /root/siton-tmp/home/liuxinyu/miniconda3/etc/profile.d/conda.sh
conda activate verl
export PATH=/root/siton-tmp/home/liuxinyu/miniconda3/envs/verl/bin:$PATH   # shell profile shadows conda PATH

# pip must NOT go through the flaky clash proxy; tuna is a domestic mirror, reachable direct.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
PIP="python -m pip"
PIPOPT="--no-cache-dir --retries 5 --timeout 120 -i https://pypi.tuna.tsinghua.edu.cn/simple"
export MAX_JOBS=32

cd /root/siton-tmp/home/liuxinyu/OPRD-High-Entropy/verl
echo "===== python: $(python --version) @ $(which python) ====="
[ "$(which python)" = "/root/siton-tmp/home/liuxinyu/miniconda3/envs/verl/bin/python" ] || { echo "WRONG PYTHON"; exit 1; }

echo "===== [1] vllm 0.11.0 (pulls torch 2.8) ====="
$PIP install $PIPOPT "vllm==0.11.0"

echo "===== [2] basic packages ====="
$PIP install $PIPOPT "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy tensorboard \
    "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"

echo "===== [3] flashinfer ====="
$PIP install $PIPOPT flashinfer-python==0.3.1

echo "===== [4] flash-attn wheel (github release via clash proxy) ====="
WHL=flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
if [ ! -s "/tmp/$WHL" ]; then
  https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 \
    curl -fL --retry 5 --retry-delay 3 -m 600 -o "/tmp/$WHL" \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/$WHL"
fi
$PIP install --no-cache-dir "/tmp/$WHL"

echo "===== [5] editable verl + math-verify + swanlab ====="
$PIP install $PIPOPT -e . --no-deps
$PIP install $PIPOPT math-verify swanlab

echo "===== [6] verify imports ====="
python - <<'PY'
import torch, vllm, verl, flash_attn
from verl.utils import rep_distillation
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
print("vllm", vllm.__version__)
print("verl", verl.__file__)
print("flash_attn", flash_attn.__version__)
print("rep_distillation OK:", rep_distillation.__file__)
PY
echo "===== ENV BUILD DONE ====="
