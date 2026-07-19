#!/bin/bash
# Anti-drift guard: every tunable knob a user can set MUST actually reach the trainer.
#
# Failure mode this prevents (the "Bug 1" class): a var is exported + baked into the
# experiment name, but no hydra override consumes it, so it silently has no effect.
#
# Rule: for each `export VAR=` in run_distillation.sh, VAR passes if ANY of:
#   1. $VAR / ${VAR appears in the hydra python command block, OR
#   2. its lowercase name appears there as a hydra key (key=), OR
#   3. it is injected via an assembled *_ARGS="..." string used in the block, OR
#   4. it is in the INFRA/NAME whitelist below (intentionally not a hydra key).
# Anything else => FAIL (exit 1), naming the offending vars.
#
# Usage:  bash setup/check_plumbing.sh [path/to/run_distillation.sh]
# Skip:   SKIP_PLUMBING_CHECK=1 (run_distillation.sh honors this)
set -uo pipefail

RUN_FILE="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/run_distillation.sh}"
[ -f "$RUN_FILE" ] || { echo "check_plumbing: no such file: $RUN_FILE" >&2; exit 2; }

# Vars that intentionally do NOT map to a hydra key: pure env/infra + name-building.
# Collapsed to single-space-delimited below so `case` matching is newline-safe.
WHITELIST="ACTOR_MODEL_NAME REWARD_MODEL_NAME TRAIN_DATASET_NAME TEST_DATA_DIR
 ACTOR_UPDATE_MEM_PROFILE CUDA_LAUNCH_BLOCKING CUDA_VISIBLE_DEVICES HYDRA_FULL_ERROR
 NCCL_DEBUG NCCL_TIMEOUT OUTLINES_CACHE_DIR PYTHONUNBUFFERED RAY_ADDRESS RAY_PORT
 RAY_TEMP_DIR RAY_ZOMBIE_SWEEP RAY_OBJECT_STORE_MEMORY RAY_NUM_CPUS SWANLAB_LOG_DIR
 TOKENIZERS_PARALLELISM TORCH_DISTRIBUTED_DEBUG TORCH_NCCL_BLOCKING_WAIT LOG_DIR
 SKIP_RAY_STOP DATA_ROOT RAY_memory_usage_threshold RAY_ADDRESS"
WHITELIST=" $(echo $WHITELIST) "     # normalize all whitespace to single spaces

# Hydra command block = from the python entrypoint to end of file.
BLOCK="$(awk '/python3 -m verl.trainer.main_ppo/{f=1} f' "$RUN_FILE")"
# Assembled arg strings (e.g. KL_ARGS="...") that ARE referenced in the block.
ARGS_BLOB=""
while IFS= read -r name; do
  if grep -qE "[$]\{?${name}\b" <<<"$BLOCK"; then
    ARGS_BLOB+="$(sed -n "/^[[:space:]]*${name}=/,/[^\\\\]$/p" "$RUN_FILE")"$'\n'
  fi
done < <(grep -oE "^[[:space:]]*[A-Z_][A-Z0-9_]*_ARGS=" "$RUN_FILE" | sed -E 's/[^A-Z0-9_]//g; s/^_+//')

fails=()
while IFS= read -r var; do
  case " $WHITELIST " in *" $var "*) continue;; esac
  lc="$(echo "$var" | tr 'A-Z' 'a-z')"
  # 1+2: direct value ref or lowercase key in the command block
  grep -qE "[$]\{?${var}([:}]|[^A-Z0-9_]|$)" <<<"$BLOCK" && continue
  grep -qE "(^|[^a-z0-9_])${lc}=" <<<"$BLOCK" && continue
  # 3: injected via an assembled *_ARGS string
  grep -qE "[$]\{?${var}([:}]|[^A-Z0-9_]|$)" <<<"$ARGS_BLOB" && continue
  fails+=("$var")
done < <(grep -oE "^export [A-Za-z_][A-Za-z0-9_]*" "$RUN_FILE" | awk '{print $2}' | sort -u)

if [ ${#fails[@]} -gt 0 ]; then
  echo "❌ check_plumbing FAILED: exported knob(s) never reach the trainer (silent no-op):" >&2
  for v in "${fails[@]}"; do echo "     - $v" >&2; done
  echo "   Fix: add a hydra override in $RUN_FILE, or whitelist it in check_plumbing.sh if intentional." >&2
  exit 1
fi
echo "✅ check_plumbing: all exported knobs reach the trainer."
