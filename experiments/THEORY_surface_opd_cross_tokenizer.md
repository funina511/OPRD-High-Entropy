# Surface 与 OPD 的数学联系 —— Cross-Tokenizer On-Policy Distillation 的理论支撑

> 本文把 token-level on-policy distillation (OPD / reverse-KL) 与 sequence-level
> surface reward 放在同一个框架下,证明二者**只差一个 student 熵项**,并说明为什么
> "surface + 熵项" 是 OPD 在**跨 tokenizer**(teacher 词表 ≠ student 词表)场景下
> 唯一自然、且良定义的推广。这构成我们 cross-tokenizer 蒸馏方法的理论依据。
>
> 记号里 `π = π_θ` 表示 student 当前策略,`p_T` 表示 teacher 分布。所有 log 以 e 为底。

---

## 0. 动机:为什么需要 sequence-level 的形式

标准 on-policy distillation(Thinking Machines, *On-Policy Distillation*, 2025)把
teacher 当作 per-token reward 模型,在 student 自己采样的轨迹上做 RL。它的每 token
reward 是**逐 token 的 reverse-KL**,这**要求 teacher 与 student 共享词表**:只有
共享词表,`log p_T(y_t\mid y_{<t})` 和 `log p_S(y_t\mid y_{<t})` 才定义在同一个
token 序列上,才能逐位相减。

但在真实蒸馏里,teacher 往往是**另一族模型**(不同 tokenizer)。此时 student 采样出
的是 *student token 序列*,teacher 无法逐 token 对齐地读它。我们的做法是让 teacher 读
student **解码出的文本**(teacher 用自己的 tokenizer 重新切分),得到一个 *sequence-level*
的 teacher 对数似然 —— 这就是 **surface reward**。

本文要回答两个问题:

1. surface reward 与 token-level OPD 在数学上是什么关系?(答:**只差一个 student 熵项**)
2. 为什么这个联系恰好让 surface 成为 OPD 的跨 tokenizer 推广?(答:token-level
   reverse-KL 需要词表对齐才有定义;而 sequence-level 的 teacher 似然 + student 熵
   **不需要词表对齐**)

---

## 1. 记号与设定

- Prompt `x`,student 生成 response `y = (y_1, \dots, y_{|y|})`,`y \sim \pi_\theta(\cdot\mid x)`。
- 链式法则(概率恒等式,恒成立):
$$
\log p(y\mid x) = \sum_{t=1}^{|y|} \log p(y_t \mid y_{<t}, x).
$$
- Teacher 序列对数似然 `\log p_T(y\mid x)`,student 序列对数似然 `\log p_S(y\mid x) = \log \pi_\theta(y\mid x)`。

---

## 2. Token-level OPD 与 telescoping 恒等式

OPD 的每 token reward(discount = 0,即每个 token 只看当前一步):
$$
r_t \;=\; \log p_T(y_t\mid y_{<t}) \;-\; \log p_S(y_t\mid y_{<t}).
$$
沿序列求和,由链式法则**逐项 telescoping**:
$$
\sum_{t=1}^{|y|} r_t
= \sum_t \log p_T(y_t\mid y_{<t}) - \sum_t \log p_S(y_t\mid y_{<t})
= \log p_T(y) - \log p_S(y).
\tag{2.1}
$$
这一步是**严格恒等式**,不含任何近似。它说明:token-level OPD 的轨迹总 reward,
等于 **teacher 与 student 的 sequence 对数似然之差**。

> 代码对应(same-vocab, `LOG_PROB_TOP_K=0`):
> `verl/verl/workers/fsdp_workers.py:3185`
> ```python
> reverse_kl = student_logp - teacher_logp   # = logp_S - logp_T, 逐 token
> rm_scores  = -reverse_kl                    # = logp_T - logp_S = r_t
> ```
> 随后 `token_reward_direct` 把每 token reward 直接当 advantage(无 baseline、无
> 折扣、无 telescoping 求和 —— PG 会隐式地对整条序列求和)。

---

## 3. Surface reward = teacher 的 sequence 似然

Surface reward 定义为 teacher 对 student response 的(长度归一)对数似然:
$$
R_{\text{surf}}(y) \;=\; \frac{1}{|y|}\log p_T(y)
\;=\; \frac{1}{|y|}\sum_t \log p_T(y_t\mid y_{<t}).
\tag{3.1}
$$
(去掉 `1/|y|` 就是 raw sum `\log p_T(y)`;长度归一的作用见 §7。)

对照 (2.1):**surface 就是 OPD telescoping 结果里的 teacher 项,丢掉了 student 项。**
$$
\underbrace{\sum_t r_t}_{\text{OPD 轨迹 reward}}
= \underbrace{\log p_T(y)}_{\text{surface (raw)}} \;-\; \log p_S(y).
\tag{3.2}
$$

> 代码对应(same-vocab surface):`verl/verl/trainer/ppo/ray_trainer.py`,
> `teacher_full_logp` 分支 —— `seq_ll = (teacher_full_logp * mask).sum(-1) / valid`,
> 即 (3.1)。

---

## 4. 核心结论:两个目标只差一个 student 熵项

把 reward 取期望(`y \sim \pi_\theta`)得到被最大化的目标 `J`。

**OPD 目标.** 由 (3.2):
$$
J_{\text{OPD}} = \mathbb{E}_{y\sim\pi}\!\big[\log p_T(y) - \log p_S(y)\big].
$$
若 reward 里的 `\log p_S = \log \pi_\theta`(当前策略),则 `\mathbb{E}_\pi[-\log\pi] = H(\pi)`,于是
$$
\boxed{\,J_{\text{OPD}} = \mathbb{E}_\pi[\log p_T(y)] + H(\pi) = -\,\mathrm{KL}\!\big(\pi \,\|\, p_T\big)\,}
\tag{4.1}
$$
这是 **reverse-KL** 的标准展开:最大化 `J_OPD` ⇔ 最小化 `KL(π‖p_T)`。

**Surface 目标.**
$$
\boxed{\,J_{\text{surf}} = \mathbb{E}_\pi[\log p_T(y)] = -\,H(\pi, p_T)\,}
\tag{4.2}
$$
即 π 相对 teacher 的**交叉熵**的负值。

**两者之差恰好是 student 熵:**
$$
\boxed{\,J_{\text{OPD}} \;=\; J_{\text{surf}} \;+\; H(\pi)\,}
\tag{4.3}
$$

我们据此定义**带熵项的 surface**(实现里的 route A,`λ` 为熵项权重):
$$
J_\lambda \;=\; \mathbb{E}_\pi\!\big[\log p_T(y)\big] + \lambda\,H(\pi)
\;=\; \mathbb{E}_\pi\!\big[\log p_T(y) - \lambda\log p_S(y)\big].
\tag{4.4}
$$
- `λ = 0`:纯 surface (4.2)。
- `λ = 1`:完整 sequence-level OPD (4.1),即 `-KL(π‖p_T)`。
- `λ ∈ (0,1)`:在交叉熵与 reverse-KL 之间插值。

> 代码对应(route A):`ray_trainer.py` 的 `teacher_full_logp` 分支
> `per_tok = tfl - lam * slp`,其中 `slp = old_log_probs`(detached student logp);
> `lam=0` 与原 surface **逐位 bit-identical**(已数值验证)。

---

## 5. 不动点分析:为什么 surface 会 mode-collapse,而 OPD 不会

熵项不是可有可无的正则,它**改变了最优解的位置**:

- **OPD (4.1)** 最小化 `KL(π‖p_T)`,唯一最优解是 `π^\* = p_T` —— 一个**分布**,
  其熵等于 teacher 的熵(有限、非零)。⇒ **存在熵地板**。
- **Surface (4.2)** 最大化 `\mathbb{E}_\pi[\log p_T(y)]`,在无约束下最优解是
  把全部质量压到 `\arg\max_y \log p_T(y)` 上的 **δ 分布**,熵 → 0。⇒ **必然 mode collapse**。

一句话:`-\log p_S` 这一项把不动点从 "**teacher 的众数(一个点)**" 搬回到 "**整个
teacher 分布**"。

**等价的、更直觉的视角(自适应 reward):** reward per token 是 `\log p_T - \lambda\log p_S`。
当 student 在某个模式上过量堆概率时,它自己的 `\log p_S` 在那里升高,`\log p_T - \log p_S \to 0`
—— 这个模式的 reward **自动断供**。纯 surface 付的是绝对的 `\log p_T`,与 student 当前
已经在做什么无关,所以一旦发现高 `\log p_T` 的模式就会 all-in,越陷越深。

> 实测(`cmp_opd_rkl_n8` vs `cmp_surf_n8`,同 teacher/student/data/n=8):
> surface 的 `actor/entropy` 单调塌到 ~0.027,val_acc 在熵跌破 ~0.04 的**同一步**
> 从 0.20 跳崖到 0.09;OPD 的熵触底 ~0.09–0.13 后**企稳/回升**,val_acc 稳在
> 0.16–0.25。崩塌形态是"流畅但不终止、从不写 `\boxed{}`"的 low-entropy stylistic
> collapse(95% 样本抽不出答案),不是经典 n-gram 复读。

---

## 6. 严谨性:reward 里的 `\log p_S` 到底是谁?

实现中 reward 用的 student logp 是 **detached 的 `old_log_probs`**(采样时的 π_old),
不是当前可导的 π。因此严格写应为
$$
J = \mathbb{E}_{y\sim\pi}\!\big[\log p_T(y) - \lambda\log p_{S,\text{old}}(y)\big].
$$
- 单步内,`\mathbb{E}_\pi[-\log p_{S,\text{old}}]` 是 π 对 π_old 的**交叉熵** `H(\pi,\pi_{\text{old}})`,
  不是 `H(\pi)`;严格的熵梯度还差一个 score-function 项。这与 detach reward 后走
  policy-gradient / importance-sampling 的标准做法一致(reward 视为常数)。
- 跨步看,on-policy 训练中 `π_old` 紧跟 `π`(每步重采样),`H(\pi,\pi_{\text{old}}) \approx H(\pi)`。
  **当策略每步变化不大时,(4.1) 的熵项解释成立**,这也是 OPD 熵能"触底企稳"而非塌陷
  的原因:哪一步在某模式尖峰,下一步 rollout 的 `\log p_{S,\text{old}}` 就升上去把那里
  的 reward 清零,形成负反馈。

> 这一点对 route A 同样成立:`slp = old_log_probs` 已 detach,reward 保持常量,不破坏 PG。

---

## 7. 长度归一与 GRPO baseline 的作用(两条正交的工程轴)

**长度归一 `1/|y|`.** (2.1) 的 telescoping 用的是 raw sum `\log p_T(y) - \log p_S(y)`;
surface 默认用 token-mean(除以 `|y|`)。二者对应不同目标:

- raw sum:精确对应序列级 `-KL`,但把长度耦合进 reward(长序列 log-lik 绝对值更大),
  容易复现 OPD 那种"向长 CoT teacher 拉长度"的问题。
- token-mean:对应**每 token 平均对数似然**(≈ 负 perplexity)。它把"是否像 teacher"
  与"有多长"**解耦**,是我们在 surface 上采用的形式。代价:它不再是精确的序列 KL,
  而是长度归一化后的似然比;结论 §4–§5 的定性(熵项决定不动点)不受影响,但定量上
  `J_λ` 与 `-KL` 相差一个 `1/|y|` 的重加权。

**cross-vocab 下长度归一是必需的,不只是可选.** 跨 tokenizer 时 student 序列长度 `|y|`
与 teacher 序列长度 `|y^T|` **不相等**。若用 raw sum,GRPO 组内各样本的 teacher-token 数
不同,reward 被"teacher 侧切了多少 token"直接污染,组内比较失去意义。长度归一(teacher
项除以 `|y^T|`、student 熵项除以 `|y|`,见 §8 量纲对齐)把两项都变成 tokenizer 长度无关
的每 token 平均对数似然,组内才可比。故 same-vocab 下长度归一是"建议",cross-vocab 下是
"必需"。

**GRPO group baseline.** surface 走 `ADV_ESTIMATOR=grpo`:同 prompt 组内减均值(可选
除以 std)。它**消掉 teacher 似然的绝对 PPL 尺度**,只留下"组内哪条更像 teacher"的相对
排序。OPD 走 `token_reward_direct`(无 baseline),每 token reverse-KL 直接做 advantage。
因此 surface+熵项 与 token-OPD 除了熵项,还差 **credit 粒度(sequence 标量+组基线 vs
逐 token)**。这给了一个干净的消融:

> 若 `λ=1` 的 sequence-level 形式就能防住 collapse,则说明**关键是熵项本身,而非 OPD
> 的逐-token credit**。

---

## 8. Cross-tokenizer 推广:本方法的理论落点

这一节是全文目的。设 teacher 与 student **tokenizer 不同**。student 采样出 student
token 序列 `y`;teacher 只能读它**解码出的文本** `\mathrm{text}(y)`,并用自己的
tokenizer 切成 teacher token 序列 `y^T = \mathrm{tok}_T(\mathrm{text}(y))`。

**(a) token-level OPD 在跨 tokenizer 下没有定义.**
逐 token reward `r_t = \log p_T(y_t) - \log p_S(y_t)` 要求两个 log-prob 定义在**同一
token 序列**上。跨 tokenizer 时 `y`(student 词表)与 `y^T`(teacher 词表)长度不同、
边界不对齐,`\log p_T(y_t)` 中的 "第 t 个 student token" 在 teacher 词表里根本不存在。
telescoping 恒等式 (2.1) 的**逐项对应关系断裂**。⇒ 原始 OPD 无法直接迁移。

**(b) 但 sequence-level 的两项都仍然良定义.** 回到 (4.4):
$$
J_\lambda = \mathbb{E}_\pi\big[\log p_T(\mathrm{text}(y))\big] + \lambda\,H_{\text{student}}(\pi).
$$
- 第一项 `\log p_T(\mathrm{text}(y))`:teacher 在**自己的词表**上对文本打分,是 sequence /
  text 级的量,**与 student tokenizer 无关**。这正是 cross-vocab surface reward 返回的
  长度归一标量 `seq_ll`。
- 第二项 `H_{\text{student}}(\pi) = \mathbb{E}_\pi[-\log p_S(y)]`:熵是 **student 侧**的量,
  完全在 **student 词表**里、用 student 自己的 `\log p_S` 算,**同样与 teacher tokenizer 无关**。

**量纲对齐(cross-vocab 实现要点).** 两项的长度归一分母**不同**:teacher 项除以
**teacher 侧** token 数 `|y^T|`(得 teacher-token-mean logp),student 项除以 **student 侧**
token 数 `|y|`(得 student-token-mean logp)。二者都是"**各自词表下的每 token 平均对数
似然**"(即各自的负 log-perplexity),是 **tokenizer 长度无关的强度量**,因此可直接线性
组合 `r = \text{teacher\_mean} - \lambda\cdot\text{student\_mean}`。若改用 raw sum,则
`|y^T| \neq |y|` 会把 tokenizer 的切分粒度耦合进 reward,组内不可比——这也是 cross-vocab
下**必须**长度归一的原因(§7)。

> 代码对应(cross-vocab route A):`ray_trainer.py` 的 `teacher_surface_ll` 分支,
> `seq_ll = teacher_surface_ll - lam * student_ll`,其中 `student_ll` 由 detached
> `old_log_probs` 按 response_mask 归一。`teacher_surface_ll` 由 RM worker 返回时已是
> teacher-token-mean(`fsdp_workers.py:2828`)。

**结论(方法论核心):**

> token-level reverse-KL 是"**token 流形**上的 OPD",要求词表对齐;
> **surface + student 熵项** 是"**文本流形**上的 OPD",只需要
> (i) teacher 能给文本打对数似然,(ii) student 能报自己的熵 ——
> **两者都不需要跨 tokenizer 的 token 对齐**。
>
> 因此 `J_λ = E[log p_T(text)] + λ·H_student(π)` 是 OPD 在跨 tokenizer 场景下
> **唯一自然且良定义**的推广;`λ=1` 时它是文本流形上的 reverse-KL 的直接对应物。

**为什么熵项在这里尤其不能省.** 跨 tokenizer 时我们**只有** sequence-level 的
teacher 文本似然(拿不到逐 token reverse-KL 的自归一化结构),纯 surface 就是 (4.2) 的
交叉熵,其不动点是 δ(mode collapse,见 §5)。熵项 `λ·H_student(π)` 是把不动点拉回
teacher 分布、阻止 collapse 的**唯一**结构性来源。这就是本方法必须带熵项的理论理由。

---

## 9. 已知偏差与 caveats(诚实边界)

1. **top-k 配分函数偏差 (S2) —— 已提供精确开关.** cross-vocab 下 teacher LM head 是
   全词表(可达 ~200k),一次性的 fp32 `[span, V]` `log_softmax` 会 OOM,故**默认**用
   **top-k logsumexp 近似分母**(`_compute_surface_reward_cross_vocab`,`fsdp_workers.py`)。
   该近似**低估配分函数 ⇒ 高估 `\log p_T`**,偏差**随位置熵增大**(尾部质量更多落在 top-k
   外),是有方向的 bias、GRPO baseline **消不掉**。可用 `surface_reward_log_tail_gap=True`
   测量其量级。**surface-only 场景**(hidden-repr 提取关闭,显存宽裕)可置
   `surface_reward_exact_denom=True`:分母改用**全词表 logsumexp**,且对 span 行做
   **chunk 分块**(每次仅 `[ROWS=128, V].float()`,峰值 ~0.1GB),从而在不 OOM 的前提下
   **彻底消除该 bias**。这是本文默认推荐的 surface-only 配置。
2. **文本似然的边缘化 —— 标准近似,误差可忽略.** 严格的 `p_T(\mathrm{text})` 是对所有能
   解码出该文本的 teacher 分词求和 `\sum_{y^T:\,\mathrm{decode}(y^T)=\mathrm{text}} p_T(y^T)`。
   实现直接在 teacher tokenizer 的 **canonical 分词**(确定性 BPE/merge 规则产出的那一种
   切法,`tgt_tokenizer(full_text)`,`fsdp_workers.py:2721`)上求 `\log p_T(y^T)`,即用
   canonical 项代替整个求和。对确定性 tokenizer(Qwen/Phi/Llama 等),canonical 切法占
   该求和的绝对主导,分词歧义带来的误差通常可忽略。换言之——"直接用 teacher 分词求和"
   正是当前实现,且是对文本似然的良好近似;主要误差来源是上面第 1 条的分母,而非此处。
3. **长度归一 ≠ 精确 KL.** 见 §7:token-mean 形式是长度归一似然比,不是序列 KL;定性
   结论不变,定量差一个 `1/|y|` 重加权。
4. **detach 近似.** 见 §6:熵项解释依赖"策略每步变化不大"的 on-policy 近似。
5. **熵项管不了长度.** 熵项防的是 mode collapse(低熵尖峰),**不**防"向长 CoT teacher
   拉长度"。length 增长与 collapse 是两个正交问题,判断 collapse 请用 entropy /
   empty_pred / val_acc,不要用 length。

---

## 10. 实验预测与验证协议

route A 的 `λ` 扫描({0, 0.5, 1.0};`λ=0` 复用现有 `cmp_surf_n8` 曲线)应验证:

| 预测 | 观测量 |
|---|---|
| `λ>0` 出现熵地板,不再单调塌向 0.03 | `actor/entropy` |
| `λ>0` val_acc 不跳崖;`λ=1` 最接近 OPD 的熵企稳形态 | val jsonl 的 `acc` / `empty_pred` |
| sequence-level reverse-KL 随训练下降(student 越来越像 teacher) | `surface/seq_reverse_kl_mean` |
| student 似然快速冲高 = collapse 前兆 | `surface/student_ll_mean` |
| 若 `λ=1`(sequence 标量+GRPO 基线)即可防 collapse ⇒ 关键是熵项,非逐-token credit | 上述指标 + 与 OPD 对照 |

运行:
```bash
SURFACE_STUDENT_ENTROPY_COEF=0.5 bash experiments/exp_rl_surf_ent.sh
SURFACE_STUDENT_ENTROPY_COEF=1.0 bash experiments/exp_rl_surf_ent.sh
```

---

## 11. 代码索引

| 概念 | 位置 |
|---|---|
| token-level reverse-KL reward (same-vocab) | `verl/verl/workers/fsdp_workers.py:3185` |
| `token_reward_direct` advantage(逐 token,无 baseline) | `verl/verl/trainer/ppo/core_algos.py:854` |
| same-vocab surface reward 构造 + route A 熵项 | `verl/verl/trainer/ppo/ray_trainer.py`(`teacher_full_logp` 分支) |
| cross-vocab surface + route A 熵项(teacher/student 各自 token-mean,量纲对齐) | `verl/verl/trainer/ppo/ray_trainer.py`(`teacher_surface_ll` 分支) |
| cross-vocab surface 打分(文本流形,无 token 对齐;top-k 或精确全词表分母) | `verl/verl/workers/fsdp_workers.py:2766` |
| `λ` 熵项系数配置字段 | `verl/verl/workers/config/rollout.py: surface_student_entropy_coef` |
| cross-vocab 精确分母开关 | `verl/verl/workers/config/rollout.py: surface_reward_exact_denom` |
| CLI 透传(`SURFACE_STUDENT_ENTROPY_COEF`, `SURFACE_REWARD_EXACT_DENOM`) | `experiments/run_distillation.sh` |
| 实验脚本(same-vocab λ 扫描) | `experiments/exp_rl_surf_ent.sh` |

---

## 附:一页纸推导链

1. 链式法则:`log p(y) = Σ_t log p(y_t | y_<t)`(恒等式)。
2. OPD token reward:`r_t = log p_T(y_t) - log p_S(y_t)`。
3. Telescoping:`Σ_t r_t = log p_T(y) - log p_S(y)`(恒等式,§2)。
4. Surface = teacher 项:`R_surf = log p_T(y)`(§3)。
5. 取期望 + `E_π[-log π] = H(π)`:`J_OPD = E[log p_T] + H(π) = -KL(π‖p_T)`(§4)。
6. `J_surf = E[log p_T] = -H(π, p_T)`;故 `J_OPD = J_surf + H(π)`(§4)。
7. 不动点:OPD → `π=p_T`(有熵地板);surface → δ(mode collapse)(§5)。
8. 严谨:reward 里 `log p_S = log p_{S,old}`(detached),`H(π,π_old) ≈ H(π)`,策略慢变时成立(§6)。
9. 跨 tokenizer:token-level reverse-KL 需词表对齐 → 断裂;而 `E[log p_T(text)]`(teacher 侧,文本级)
   与 `H_student(π)`(student 侧,student 词表)均不需对齐 → **surface + 熵项 = 文本流形上的 OPD**(§8)。
