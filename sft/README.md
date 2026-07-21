# Teacher sampling → LlamaFactory SFT baseline

这个目录提供一条与现有 OPRD/verl 训练代码相互独立的 baseline：用本地 teacher 模型通过 vLLM 在 `dapo-math-5k-seed42.parquet` 上离线采样，再把采样结果转换为 LlamaFactory 数据集，对本地 student 模型做全参数 SFT。采样、数据处理、训练和评测均可写入同一个 W&B project，并通过同一 `RUN_ID` 分组。

默认配置面向当前实验设定：Qwen3-4B teacher、Qwen3-0.6B-Base student、4×3090、thinking off。模型路径没有硬编码，换服务器后只需修改环境变量。

```text
DAPO-Math-5k parquet
        │
        ├─ vLLM teacher sampling ── raw_samples.jsonl
        │                              │
        │                              ├─ all / valid / correct selection
        │                              ▼
        │                         teacher_sft.jsonl
        │                              │
        │                              └─ LlamaFactory full SFT ── student checkpoint
        │
        └─ W&B: sampling / data / training / evaluation runs, grouped by RUN_ID
```

## 1. 环境准备

建议保留两个环境，避免 vLLM 与 LlamaFactory 的 PyTorch/Transformers 依赖互相覆盖：

- sampling/preparation 环境：已经装好的 vLLM，加上 pandas、pyarrow、W&B 和 verl 数学判分依赖。
- SFT 环境：仓库内 `LlamaFactory/` 的依赖、DeepSpeed 和 W&B。

如果现有 verl/vLLM 环境缺少数据处理依赖，可在仓库根目录执行：

```bash
/path/to/vllm-env/bin/python -m pip install -r sft/requirements-tools.txt
```

不要为了执行上面的命令重装一个已经与服务器 CUDA/PyTorch 匹配好的 vLLM wheel。先检查现有环境：

```bash
/path/to/vllm-env/bin/python -c "import vllm, pandas, pyarrow, wandb; print(vllm.__version__)"
```

LlamaFactory 环境可按仓库内版本安装：

```bash
cd /home/elysia/code/OPRD-High-Entropy/LlamaFactory
/path/to/sft-env/bin/python -m pip install -e .
/path/to/sft-env/bin/python -m pip install -r requirements/deepspeed.txt wandb
```

登录 W&B，或使用集群已有的 `WANDB_API_KEY`：

```bash
/path/to/vllm-env/bin/wandb login
/path/to/sft-env/bin/wandb login
```

路径模板见 `configs/env.example.sh`。不要把 W&B key 写进脚本或提交到 Git。

## 2. 先做 8 条 smoke test

正式跑 5k 之前，建议先验证模型、数据、CUDA 和 W&B 均能工作：

```bash
cd /home/elysia/code/OPRD-High-Entropy

export TEACHER_MODEL_PATH=/mnt/models/Qwen3-4B
export STUDENT_MODEL_PATH=/mnt/models/Qwen3-0.6B-Base
export SAMPLE_PYTHON=/path/to/vllm-env/bin/python
export PREP_PYTHON=/path/to/vllm-env/bin/python
export LLAMAFACTORY_CLI=/path/to/sft-env/bin/llamafactory-cli

export RUN_ID=smoke-teacher-sft
export GPU_IDS=0,1,2,3
export MAX_PROMPTS=8
export MAX_NEW_TOKENS=512
export MAX_MODEL_LEN=4096
export CUTOFF_LEN=4096
export MAX_STEPS=2
export WANDB_MODE=offline

bash sft/run_pipeline.sh
```

smoke test 成功后更换一个新的 `RUN_ID`，并取消 `MAX_PROMPTS`、`MAX_STEPS` 和缩短长度的设置；不要在 smoke test 输出目录中直接启动正式实验。

## 3. 推荐的正式 baseline

标准 `Teacher-SFT-1`：每道题采样一次，保留所有非空 teacher 输出，每个 prompt 恰好最多一个训练样本。

```bash
cd /home/elysia/code/OPRD-High-Entropy

export TEACHER_MODEL_PATH=/mnt/models/Qwen3-4B
export STUDENT_MODEL_PATH=/mnt/models/Qwen3-0.6B-Base
export SAMPLE_PYTHON=/path/to/vllm-env/bin/python
export PREP_PYTHON=/path/to/vllm-env/bin/python
export LLAMAFACTORY_CLI=/path/to/sft-env/bin/llamafactory-cli

export RUN_ID=teacher-sft-qwen3-4b-to-06b-dapo5k-s42-v1
export GPU_IDS=0,1,2,3
export NUM_ROLLOUTS=1
export SELECTION=all
export MAX_PER_PROMPT=1

export WANDB_PROJECT=OPRD-High-Entropy
export WANDB_ENTITY=your_wandb_entity
export WANDB_MODE=online

bash sft/run_pipeline.sh
```

`run_pipeline.sh` 默认依次执行 `sample,prepare,sft`。三个 W&B run 使用同一个 group（`RUN_ID`）：

- `${RUN_ID}-sample`：采样速度、长度、boxed/有效率、原始样本 artifact。
- `${RUN_ID}-prepare-all`：teacher 正确率、数据覆盖率、SFT 数据 artifact。
- `${RUN_ID}-sft-all`：LlamaFactory/Transformers 的 loss、learning rate、epoch、吞吐和 checkpoint 指标。

默认不把模型 checkpoint 上传到 W&B，避免意外占用大量 artifact 空间。如明确需要，可设置 `WANDB_LOG_MODEL=end`。

## 4. 分阶段执行与断点续跑

需要切换 shell/conda 环境时，可分三步执行；三步必须使用相同的 `RUN_ID`：

```bash
RUN_ID=my-baseline TEACHER_MODEL_PATH=/mnt/models/Qwen3-4B \
SAMPLE_PYTHON=/path/to/vllm-env/bin/python \
bash sft/run_sampling.sh

RUN_ID=my-baseline PREP_PYTHON=/path/to/vllm-env/bin/python \
SELECTION=all MAX_PER_PROMPT=1 \
bash sft/run_prepare.sh

RUN_ID=my-baseline STUDENT_MODEL_PATH=/mnt/models/Qwen3-0.6B-Base \
LLAMAFACTORY_CLI=/path/to/sft-env/bin/llamafactory-cli \
bash sft/run_sft.sh
```

teacher sampling 每个 GPU 使用一个独立的 vLLM 实例，按 prompt/rollout 槽写入 `temp_rollout/worker_*.jsonl`。作业中断后原命令重跑会跳过已完成槽。脚本会校验输入文件 hash、模型 config hash 和采样参数；参数发生变化时应使用新的 `RUN_ID`，而不是混写旧目录。只有确认要承担混合数据风险时才设置 `ALLOW_CONFIG_MISMATCH=true`。

LlamaFactory 从已有 checkpoint 继续训练时，传入明确路径：

```bash
export RESUME_FROM_CHECKPOINT=/absolute/path/to/checkpoint-150
bash sft/run_sft.sh
```

也可以只重跑部分流水线：

```bash
PIPELINE_STAGES=prepare,sft bash sft/run_pipeline.sh
```

## 5. 可选数据策略

采样与筛选是分开的，因此同一份 `raw_samples.jsonl` 可以生成多个 baseline 数据集。

| 实验 | `NUM_ROLLOUTS` | `SELECTION` | `MAX_PER_PROMPT` | 含义 |
|---|---:|---|---:|---|
| Teacher-SFT-1（推荐主 baseline） | 1 | `all` | 1 | 保留一次采样，数据预算最清楚 |
| Teacher-SFT-valid@4 | 4 | `valid` | 1 | 取第一个有 boxed 且无明显重复的输出 |
| Teacher-SFT-correct@4 | 4 | `correct` | 1 | 取第一个通过 verl 数学判分的输出；无正确输出的 prompt 被丢弃 |
| Teacher-SFT-all@4 | 4 | `all` | 4 | 四条全部训练，约 4 倍样本/token 预算，不能与 SFT-1 直接视为等算力比较 |

例如从四次采样中筛选正确答案：

```bash
export RUN_ID=teacher-correct-at4
export NUM_ROLLOUTS=4
bash sft/run_sampling.sh

export SELECTION=correct
export MAX_PER_PROMPT=1
bash sft/run_prepare.sh
bash sft/run_sft.sh
```

`run_prepare.sh` 默认对所有候选做正确性诊断，判分实现复用 `verl.utils.reward_score.ttrl_math`。如果只做 `SELECTION=all` 且当前环境缺少 `math_verify` 等依赖，可以设置 `GRADE_CORRECTNESS=false`；此时 W&B 不会有 teacher accuracy。

不同筛选策略会改变样本数。比较算法时应同时报告 `selected_samples`、`prompt_coverage` 和训练 token 数；若想固定优化步数，设置相同的正数 `MAX_STEPS`。默认 `MAX_STEPS=-1, NUM_TRAIN_EPOCHS=1.0`，表示对各自数据完整训练一轮。

## 6. 核心参数

默认值刻意与当前长数学回答设置保持一致：

| 阶段 | 参数 | 默认值 | 环境变量 |
|---|---|---:|---|
| sampling | GPU 数据并行 | 4 个独立实例 | `GPU_IDS=0,1,2,3` |
| sampling | rollouts / prompt | 1 | `NUM_ROLLOUTS` |
| sampling | temperature / top-p / top-k | 1.0 / 0.95 / -1 | `TEMPERATURE`, `TOP_P`, `TOP_K` |
| sampling | repetition penalty | 1.0 | `REPETITION_PENALTY` |
| sampling | thinking | off | `ENABLE_THINKING=false` |
| sampling | response / context 上限 | 8192 / 10480 | `MAX_NEW_TOKENS`, `MAX_MODEL_LEN` |
| sampling | 基础 rejection | off | `BASIC_REJECTION=false` |
| sampling | seed | 42 | `SEED` |
| preparation | 策略 / 每题上限 | all / 1 | `SELECTION`, `MAX_PER_PROMPT` |
| SFT | 方法 / 精度 | full / bf16 | YAML 配置 |
| SFT | prompt loss | 不训练 prompt | `train_on_prompt: false` |
| SFT | cutoff | 10240 | `CUTOFF_LEN` |
| SFT | effective global batch | 4×1×4 = 16 | `PER_DEVICE_BATCH_SIZE`, `TARGET_GLOBAL_BATCH` |
| SFT | learning rate / scheduler | 1e-5 / cosine | `LEARNING_RATE` |
| SFT | epoch / max steps | 1.0 / -1 | `NUM_TRAIN_EPOCHS`, `MAX_STEPS` |
| SFT | distributed | DeepSpeed ZeRO-2 | `DEEPSPEED_CONFIG` |

`MAX_MODEL_LEN` 必须覆盖 prompt 加生成长度；`CUTOFF_LEN` 决定进入 SFT 的总序列长度。降低二者能省显存，但会改变 baseline 的有效训练 token，必须记录在实验名或 W&B config 中。采样输出中的 `truncated_rate` 和 SFT 的 `num_input_tokens_seen` 可辅助检查截断。

## 7. 评测

默认在 `datasets/test_data/AMC23/test.parquet` 上生成 4 个 rollout，并输出 sample accuracy、first-rollout accuracy 和 empirical best-of-n：

```bash
cd /home/elysia/code/OPRD-High-Entropy

MODEL_PATH=sft/runs/my-baseline/checkpoints_all \
EVAL_PYTHON=/path/to/vllm-env/bin/python \
RUN_ID=my-baseline \
TASK_NAME=AMC23 \
EVAL_NUM_ROLLOUTS=4 \
WANDB_MODE=online \
bash sft/run_eval.sh
```

切换数据集只需设置 `TASK_NAME=AIME24`、`AIME25` 等；默认路径为 `datasets/test_data/${TASK_NAME}/test.parquet`，也可以用 `EVAL_PARQUET` 指定任意 verl 格式 parquet。评测参数使用独立的 `EVAL_NUM_ROLLOUTS`、`EVAL_TEMPERATURE`、`EVAL_TOP_P`、`EVAL_MAX_NEW_TOKENS` 等变量，不会意外继承 teacher sampling 的同名参数。为保证对比公平，base student、SFT student 和 OPRD checkpoint 应使用完全相同的评测采样参数。

## 8. 输出目录

```text
sft/runs/<RUN_ID>/
├── sampling/
│   ├── raw_samples.jsonl
│   ├── sampling_config.json
│   ├── sampling_stats.json
│   ├── manifest.json
│   └── temp_rollout/                 # 断点状态
├── data_<selection>/
│   ├── teacher_sft.jsonl
│   ├── dataset_info.json
│   ├── stats.json
│   └── manifest.json
├── checkpoints_<selection>/          # LlamaFactory 最终模型与 checkpoints
├── wandb/
└── eval_<TASK>/
    ├── generation/
    └── scores/metrics.json
```

manifest 保存输入 hash、模型 `config.json` hash、Git commit、依赖版本和核心参数。模型权重很大，因此只记录路径和 config hash，不逐个 hash 权重文件；正式实验还应在实验记录中固定模型目录的版本或外部 checksum。

## 9. 数据模板说明

生成的每条训练数据采用直观的 OpenAI-style `messages` 字段：

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

但自动生成的 `dataset_info.json` 有意声明为 `formatting: sharegpt`，并显式把 `role/content/user/assistant` tags 映射回上述字段。原因是本仓库修改过 `LlamaFactory/src/llamafactory/data/converter.py`：其 OpenAI converter 会额外插入 `detailed thinking off` system prompt。使用 ShareGPT converter 可以保持 teacher sampling 与 student SFT 的原始 prompt 完全一致。

Qwen3 的采样和训练默认都关闭 thinking：vLLM tokenizer 使用 `enable_thinking=False`；LlamaFactory 使用 `template: qwen3, enable_thinking: false`。不要只修改其中一侧。

## 10. 常见问题

- vLLM OOM：先降低 `GPU_MEMORY_UTILIZATION` 或 `REQUEST_BATCH_SIZE`；仍然 OOM 再缩短 `MAX_NEW_TOKENS/MAX_MODEL_LEN`，并把变化计入实验配置。
- SFT OOM：先保持数据 cutoff 不变，降低 `PER_DEVICE_BATCH_SIZE=1`（默认已是 1）并确认 gradient checkpointing 生效；必要时使用更激进的 DeepSpeed 配置。缩短 `CUTOFF_LEN` 会改变实验。
- 缺少 FlashAttention：默认 `FLASH_ATTN=auto`，无需安装 FA2；若服务器已有兼容版本可设置 `FLASH_ATTN=fa2`。
- 数学判分导入失败：在 `PREP_PYTHON` 环境安装 `sft/requirements-tools.txt`，并确保从仓库根目录启动脚本。
- W&B 无网络：设置 `WANDB_MODE=offline`，完成后用对应环境的 `wandb sync sft/runs/<RUN_ID>/wandb/...` 上传。
- 修改采样参数后提示 config mismatch：使用新的 `RUN_ID`。这是为了防止两套采样分布被合并到同一 JSONL。
