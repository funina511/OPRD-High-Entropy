#!/bin/bash
# Shared runtime base for all experiments. SOURCE this (do not exec it):
#   source "$(dirname "${BASH_SOURCE[0]}")/setup/common.sh"
# then set the few knobs your experiment differs on, and finally:
#   run_experiment <oprd|opd|oprd_opd>
#
# Everything here is a shared default via ${VAR:-...}, so an experiment can
# override any value either BEFORE sourcing (env on the command line) or AFTER
# sourcing (plain assignment) — both win over these defaults.
#
# MACHINE-SPECIFIC values (conda path, model/data dirs, CUDA devices, proxy, ray
# tuning) live in setup/hosts/<host>.sh — NOT here. This file is identical on every
# machine; only the host profile differs, so `git pull` never clobbers a box.
set -eo pipefail

_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- host profile selection: OPRD_HOST env > hostname match > error ---------
# good3090 resolves by hostname with zero config. The siton141 container has an
# ephemeral hostname (a docker id), so set `export OPRD_HOST=siton141` in its rc.
_host="${OPRD_HOST:-$(hostname -s 2>/dev/null || hostname)}"
_host_file="${_SETUP_DIR}/hosts/${_host}.sh"
if [ ! -f "$_host_file" ]; then
  echo "ERROR: no host profile for '${_host}' (looked for ${_host_file})." >&2
  echo "  Available: $(cd "${_SETUP_DIR}/hosts" && ls -1 ./*.sh 2>/dev/null | sed 's|.*/||; s/\.sh$//' | grep -v '^_' | paste -sd', ')" >&2
  echo "  Fix: create that file, or 'export OPRD_HOST=<name>' to pick an existing one." >&2
  return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1090
source "$_host_file"
echo "[common.sh] host profile: ${_host}"

# --- fail-fast if the host profile forgot a required var (else PATH/paths silently
#     degrade to empty and python/ray point at the wrong place with no error). ----
_missing=()
for _v in OPRD_CONDA_SH OPRD_CONDA_ENV OPRD_CONDA_BIN CUDA_VISIBLE_DEVICES \
          N_GPUS_PER_NODE OPRD_REPO_ROOT MODEL_DIR DATA_DIR; do
  [ -z "${!_v:-}" ] && _missing+=("$_v")
done
if [ ${#_missing[@]} -gt 0 ]; then
  echo "ERROR: host profile '${_host_file}' is missing required var(s): ${_missing[*]}" >&2
  echo "  Every hosts/<name>.sh must define them — see hosts/_template.sh." >&2
  return 1 2>/dev/null || exit 1
fi

# --- interpreter + offline flags (host file provides the conda locations) ----
# shellcheck disable=SC1090
source "$OPRD_CONDA_SH"
conda activate "$OPRD_CONDA_ENV"
export PATH="$OPRD_CONDA_BIN:$PATH"           # shell profile shadows conda PATH
export WANDB_MODE=${WANDB_MODE:-online}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

# --- hardware knobs shared across hosts (device list / gpu count set by host) -
export CUDA_LAUNCH_BLOCKING=0
export RAY_PORT=${RAY_PORT:-6379}
export SKIP_RAY_STOP=${SKIP_RAY_STOP:-1}      # never kill another run's ray cluster

# --- model/data paths: derived from host MODEL_DIR/DATA_DIR (names are shared) -
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-${MODEL_DIR}/Qwen3-0.6B-Base}
export REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-${MODEL_DIR}/Qwen3-4B}

# --- batch / signal: stable scale verified not to OOM on 3090 ---
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-8}
export N_RESPONSES=${N_RESPONSES:-2}
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-2048}
export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-2048}
export REWARD_MICRO_BSZ=${REWARD_MICRO_BSZ:-8}

# --- schedule + logging (eval/save WITHIN the run; skip swanlab viz block) ---
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-150}
export TEST_FREQ=${TEST_FREQ:-25}
export SAVE_FREQ=${SAVE_FREQ:-50}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
export IS_PLOT=${IS_PLOT:-False}

# run_experiment <method>: hand off to the core engine. `exec` so signals/exit map 1:1.
run_experiment() {
  local method="${1:?run_experiment needs a method: oprd | opd | oprd_opd}"
  exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/run_distillation.sh" "$method"
}
