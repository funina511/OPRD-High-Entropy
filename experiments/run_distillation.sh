#!/bin/bash
# Unified distillation entry: OPRD-Vanilla / OPD / OPD+rep.
#
# Usage (from anywhere):
#   bash experiments/run_distillation.sh              # OPRD rep-only (default)
#   bash experiments/run_distillation.sh oprd         # same
#   bash experiments/run_distillation.sh opd          # token-level OPD only
#   bash experiments/run_distillation.sh oprd_opd     # OPD + representation
#
# Override any knob via env, e.g.:
#   REP_DISTILLATION_LAYERS=even MINI_BATCH_SIZE=8 bash experiments/run_distillation.sh
#   MODEL_DIR=/path/to/models bash experiments/run_distillation.sh

#SBATCH --job-name=oprd
#SBATCH --output=logs/slurm_%j.log
#SBATCH --error=logs/slurm_%j.err
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --nodes=1

set -euo pipefail
set -x

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Method preset: oprd | opd | oprd_opd
# ---------------------------------------------------------------------------
METHOD="${1:-${METHOD:-oprd}}"
case "$METHOD" in
  oprd)
    export USE_REP_DISTILLATION=${USE_REP_DISTILLATION:-True}
    export REP_DISTILLATION_ONLY=${REP_DISTILLATION_ONLY:-True}
    export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-0}
    ;;
  opd)
    export USE_REP_DISTILLATION=${USE_REP_DISTILLATION:-False}
    export REP_DISTILLATION_ONLY=${REP_DISTILLATION_ONLY:-False}
    export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-16}
    ;;
  oprd_opd|opd_oprd)
    export USE_REP_DISTILLATION=${USE_REP_DISTILLATION:-True}
    export REP_DISTILLATION_ONLY=${REP_DISTILLATION_ONLY:-False}
    export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-16}
    ;;
  *)
    echo "Unknown METHOD='$METHOD'. Use: oprd | opd | oprd_opd"
    exit 1
    ;;
esac
export PROJECT_NAME=${PROJECT_NAME:-OPD_RepDistillation}

# ---------------------------------------------------------------------------
# Rep / attention knobs (safe defaults for OPRD-Vanilla)
# ---------------------------------------------------------------------------
export REP_DISTILLATION_COEF=${REP_DISTILLATION_COEF:-1.0}
export REP_DISTILLATION_POSITIONS=${REP_DISTILLATION_POSITIONS:-last_k}
export REP_DISTILLATION_LAST_K=${REP_DISTILLATION_LAST_K:-2000}
export REP_DISTILLATION_FIRST_K=${REP_DISTILLATION_FIRST_K:-2000}
export REP_DISTILLATION_LAYERS=${REP_DISTILLATION_LAYERS:-all}
export REP_PROJECTOR_MODE=${REP_PROJECTOR_MODE:-full}
export REP_LOW_RANK=${REP_LOW_RANK:-256}

export USE_ATT_DISTILLATION=${USE_ATT_DISTILLATION:-False}
export ATT_DISTILLATION_COEF=${ATT_DISTILLATION_COEF:-1.0}
export ATT_DISTILLATION_LAYERS=${ATT_DISTILLATION_LAYERS:-last}
export ATT_DISTILLATION_POSITIONS=${ATT_DISTILLATION_POSITIONS:-first_k}
export ATT_DISTILLATION_LAST_K=${ATT_DISTILLATION_LAST_K:-32}
export ATT_DISTILLATION_FIRST_K=${ATT_DISTILLATION_FIRST_K:-50}
export ATT_DISTILLATION_MAX_KEY_LEN=${ATT_DISTILLATION_MAX_KEY_LEN:-4096}
export ATT_DISTILLATION_LOSS=${ATT_DISTILLATION_LOSS:-kl}
export ATT_DISTILLATION_TEMPERATURE=${ATT_DISTILLATION_TEMPERATURE:-1.0}

# ---------------------------------------------------------------------------
# Runtime / hardware
# ---------------------------------------------------------------------------
if [ -z "${SLURM_JOB_ID:-}" ]; then
    LOG_DIR=${LOG_DIR:-"$REPO_ROOT/logs"}
    mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "=========================================="
    echo "Log file: $LOG_FILE"
    echo "METHOD=$METHOD"
    echo "Start time: $(date)"
    echo "=========================================="
fi

ray stop --force || true
export RAY_memory_usage_threshold=0.99
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTHONUNBUFFERED=1
export ACTOR_UPDATE_MEM_PROFILE=${ACTOR_UPDATE_MEM_PROFILE:-0}
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=7200
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR=1

# ---------------------------------------------------------------------------
# Algorithm / sampling
# ---------------------------------------------------------------------------
export ADV_ESTIMATOR=${ADV_ESTIMATOR:-token_reward_direct}
export GRPO_OUTCOME_WEIGHT=${GRPO_OUTCOME_WEIGHT:-1.0}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-8192}
export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-8192}
export MAX_MODEL_LEN=$(( MAX_RESP_LENGTH + MAX_PROMPT_LENGTH > MAX_VAL_RESP_LENGTH + MAX_PROMPT_LENGTH ? MAX_RESP_LENGTH + MAX_PROMPT_LENGTH : MAX_VAL_RESP_LENGTH + MAX_PROMPT_LENGTH ))
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-4}
export TEMPERATURE=${TEMPERATURE:-1.0}
export TEACHER_TEMPERATURE=${TEACHER_TEMPERATURE:-1.0}
export REPETITION_PENALTY=${REPETITION_PENALTY:-1.0}
export N_RESPONSES=${N_RESPONSES:-2}
export TOP_K_STRATEGY=${TOP_K_STRATEGY:-only_stu}
export REWARD_WEIGHT_MODE=${REWARD_WEIGHT_MODE:-student_p}
export USE_KL=${USE_KL:-False}
export ENABLE_FORMAT_REWARD=${ENABLE_FORMAT_REWARD:-False}
export MODEL_DTYPE=${MODEL_DTYPE:-bfloat16}
export IS_PLOT=${IS_PLOT:-True}
export LOSS_AGG_MODE=${LOSS_AGG_MODE:-token-mean}
export ENABLE_ACTIVATION_OFFLOAD=${ENABLE_ACTIVATION_OFFLOAD:-False}

# ---------------------------------------------------------------------------
# Data / models (override via MODEL_DIR, DATA_DIR, or individual paths)
# ---------------------------------------------------------------------------
DATA_ROOT="${DATA_DIR:-$REPO_ROOT/../datasets}"
export TRAIN_DATASET=${TRAIN_DATASET:-"$DATA_ROOT/dapo-math-5k-seed42.parquet"}
export TRAIN_DATASET_NAME=${TRAIN_DATASET_NAME:-DAPO-Math-5k}
export TEST_DATA_DIR=${TEST_DATA_DIR:-"$DATA_ROOT/test_data"}
TEST_DATASET=${TEST_FILE:-["$TEST_DATA_DIR/AMC23/test.parquet"]}

export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-${MODEL_DIR}/Qwen3-0.6B-Base}
export REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-${MODEL_DIR}/Qwen3-1.7B}
export ACTOR_MODEL_NAME=$(basename "$ACTOR_MODEL_PATH")
export REWARD_MODEL_NAME=$(basename "$REWARD_MODEL_PATH")

export PROJECT_PATH=${PROJECT_PATH:-"$REPO_ROOT/outputs"}
export PARALLEL_SIZE=${PARALLEL_SIZE:-1}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
export SAVE_FREQ=${SAVE_FREQ:-200}
export TEST_FREQ=${TEST_FREQ:-500}

export CKPT_PATH=${CKPT_PATH:-${PROJECT_PATH}/${ADV_ESTIMATOR}_${TRAIN_DATASET_NAME}_${ACTOR_MODEL_NAME}_${REWARD_MODEL_NAME}_${MAX_RESP_LENGTH}-T_${TEMPERATURE}-Tch_${TEACHER_TEMPERATURE}-n_${N_RESPONSES}-mbs_${MINI_BATCH_SIZE}-topk_${LOG_PROB_TOP_K}-topk_strategy_${TOP_K_STRATEGY}-rw_${REWARD_WEIGHT_MODE}-$(date +%Y-%m-%d_%H-%M-%S)-${METHOD}}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-${ADV_ESTIMATOR}_${TRAIN_DATASET_NAME}_${ACTOR_MODEL_NAME}_${REWARD_MODEL_NAME}_${MAX_RESP_LENGTH}-T_${TEMPERATURE}-Tch_${TEACHER_TEMPERATURE}-n_${N_RESPONSES}-mbs_${MINI_BATCH_SIZE}-topk_${LOG_PROB_TOP_K}-topk_strategy_${TOP_K_STRATEGY}-rw_${REWARD_WEIGHT_MODE}-$(date +%Y-%m-%d_%H-%M-%S)-${METHOD}}
export OUTLINES_CACHE_DIR=${OUTLINES_CACHE_DIR:-~/.cache/outlines/$(uuidgen)}
export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-${PROJECT_PATH}/swanlab_log}

mkdir -p "$PROJECT_PATH/logs/terminal" "$PROJECT_PATH/logs/validation_log"

echo "METHOD=$METHOD USE_REP=$USE_REP_DISTILLATION REP_ONLY=$REP_DISTILLATION_ONLY TOP_K=$LOG_PROB_TOP_K"
echo "student=$ACTOR_MODEL_PATH teacher=$REWARD_MODEL_PATH"
echo "train=$TRAIN_DATASET"

KL_ARGS=""
if [ "$USE_KL" = "True" ]; then
    KL_ARGS="actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.005 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl"
else
    KL_ARGS="actor_rollout_ref.actor.use_kl_loss=False"
fi

LR_ARGS=""
if [ "${LR_SCHEDULER:-}" = "cosine" ]; then
    LR_ARGS="actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03"
fi

PPO_MAX_TOKEN_LEN_PER_GPU=$(( ((1024 + MAX_RESP_LENGTH) > 32768) ? (1024 + MAX_RESP_LENGTH) : 32768))
echo "PPO_MAX_TOKEN_LEN_PER_GPU: $PPO_MAX_TOKEN_LEN_PER_GPU"

export RAY_PORT=${RAY_PORT:-6391}
ray start --head --port="$RAY_PORT"
sleep 5

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$ADV_ESTIMATOR \
    algorithm.grpo_outcome_weight=$GRPO_OUTCOME_WEIGHT \
    data.shuffle=False \
    data.train_files="$TRAIN_DATASET" \
    data.val_files="$TEST_DATASET" \
    data.train_batch_size=$((MINI_BATCH_SIZE * PARALLEL_SIZE)) \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESP_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_activation_offload=$ENABLE_ACTIVATION_OFFLOAD \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    $LR_ARGS \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$PARALLEL_SIZE \
    $KL_ARGS \
    actor_rollout_ref.actor.loss_agg_mode=$LOSS_AGG_MODE \
    +actor_rollout_ref.actor.use_rep_distillation=$USE_REP_DISTILLATION \
    +actor_rollout_ref.actor.rep_distillation_coef=$REP_DISTILLATION_COEF \
    +actor_rollout_ref.actor.rep_distillation_only=$REP_DISTILLATION_ONLY \
    +actor_rollout_ref.actor.rep_distillation_positions=$REP_DISTILLATION_POSITIONS \
    +actor_rollout_ref.actor.rep_distillation_last_k=$REP_DISTILLATION_LAST_K \
    +actor_rollout_ref.actor.rep_distillation_first_k=$REP_DISTILLATION_FIRST_K \
    +actor_rollout_ref.actor.rep_distillation_layers=$REP_DISTILLATION_LAYERS \
    +actor_rollout_ref.actor.rep_projector_mode=${REP_PROJECTOR_MODE} \
    +actor_rollout_ref.actor.rep_low_rank=${REP_LOW_RANK} \
    ${REP_LOW_RANK_INIT_CHECKPOINT:++actor_rollout_ref.actor.rep_low_rank_init_checkpoint="$REP_LOW_RANK_INIT_CHECKPOINT"} \
    +actor_rollout_ref.actor.rep_ps_projector=${REP_PS_PROJECTOR:-auto} \
    +actor_rollout_ref.actor.rep_mlp_hidden_mult=${REP_MLP_HIDDEN_MULT:-4} \
    +actor_rollout_ref.actor.rep_freeze_ps=${REP_FREEZE_PS:-False} \
    +actor_rollout_ref.actor.rep_head_rank=${REP_HEAD_RANK:-16} \
    ${REP_HEAD_INIT_CHECKPOINT:++actor_rollout_ref.actor.rep_head_init_checkpoint="$REP_HEAD_INIT_CHECKPOINT"} \
    +actor_rollout_ref.actor.use_att_distillation=$USE_ATT_DISTILLATION \
    +actor_rollout_ref.actor.att_distillation_coef=$ATT_DISTILLATION_COEF \
    +actor_rollout_ref.actor.att_distillation_layers=$ATT_DISTILLATION_LAYERS \
    +actor_rollout_ref.actor.att_distillation_positions=$ATT_DISTILLATION_POSITIONS \
    +actor_rollout_ref.actor.att_distillation_last_k=$ATT_DISTILLATION_LAST_K \
    +actor_rollout_ref.actor.att_distillation_first_k=$ATT_DISTILLATION_FIRST_K \
    +actor_rollout_ref.actor.att_distillation_max_key_len=$ATT_DISTILLATION_MAX_KEY_LEN \
    +actor_rollout_ref.actor.att_distillation_loss=$ATT_DISTILLATION_LOSS \
    +actor_rollout_ref.actor.att_distillation_temperature=$ATT_DISTILLATION_TEMPERATURE \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=$MODEL_DTYPE \
    actor_rollout_ref.rollout.max_num_batched_tokens=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=$MODEL_DTYPE \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    +actor_rollout_ref.rollout.log_prob_top_k=$LOG_PROB_TOP_K \
    +actor_rollout_ref.rollout.top_k_strategy=$TOP_K_STRATEGY \
    +actor_rollout_ref.rollout.reward_weight_mode=$REWARD_WEIGHT_MODE \
    +actor_rollout_ref.rollout.teacher_temperature=$TEACHER_TEMPERATURE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$PARALLEL_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL:-0.8} \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.n=$N_RESPONSES \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    +actor_rollout_ref.rollout.val_kwargs.max_tokens=$MAX_VAL_RESP_LENGTH \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.repetition_penalty=$REPETITION_PENALTY \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    reward_model.enable=True \
    +reward_model.reward_kwargs.enable_format_reward=$ENABLE_FORMAT_REWARD \
    reward_model.model.path=$REWARD_MODEL_PATH \
    reward_model.model.input_tokenizer=null \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=${REWARD_PARAM_OFFLOAD:-False} \
    +reward_model.model.dtype=$MODEL_DTYPE \
    reward_model.micro_batch_size_per_gpu=${REWARD_MICRO_BSZ:-8} \
    custom_reward_function.path="${CUSTOM_REWARD_PATH:-$REPO_ROOT/verl/verl/utils/reward_score/ttrl_math/__init__.py}" \
    custom_reward_function.name=reward_func \
    trainer.val_before_train=${VAL_BEFORE_TRAIN:-True} \
    trainer.log_val_generations=2 \
    trainer.logger=['console','wandb'] \
    trainer.output_log_path=${PROJECT_PATH}/logs/terminal/${EXPERIMENT_NAME}.log \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.validation_data_dir=${PROJECT_PATH}/logs/validation_log/$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=1 \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
    trainer.default_local_dir="$CKPT_PATH" \
    trainer.is_plot=$IS_PLOT

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "=========================================="
    echo "End time: $(date)"
    echo "=========================================="
fi
