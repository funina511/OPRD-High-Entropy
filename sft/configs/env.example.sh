#!/usr/bin/env bash
# Copy the relevant exports into your shell or a private host-specific file.
# Do not commit API keys or other credentials.

export TEACHER_MODEL_PATH=/mnt/models/Qwen3-4B
export STUDENT_MODEL_PATH=/mnt/models/Qwen3-0.6B-Base

# Sampling/preparation environment (vLLM + pandas/pyarrow + verl math dependencies).
export SAMPLE_PYTHON=/mnt/miniconda3/envs/verl/bin/python
export PREP_PYTHON=/mnt/miniconda3/envs/verl/bin/python

# Training environment with this repository's vendored LlamaFactory installed.
export LLAMAFACTORY_CLI=/mnt/miniconda3/envs/llamafactory/bin/llamafactory-cli

export GPU_IDS=0,1,2,3
export WANDB_PROJECT=OPRD-High-Entropy
export WANDB_ENTITY=
export WANDB_MODE=online

# Set this explicitly when you want multiple commands to share one output/W&B group.
export RUN_ID=teacher-sft-qwen3-4b-to-qwen3-06b-dapo5k-seed42
