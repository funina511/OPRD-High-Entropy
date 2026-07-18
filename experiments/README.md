# OPRD on 4× RTX 3090 — experiment scripts

Preliminary reproduction of **OPRD** (arXiv 2606.06021) on this machine.
Student **Qwen3-0.6B** (28 layers, d=1024), teacher **Qwen3-4B** (36 layers, d=2560),
data **DAPO-Math-5k**, eval **AMC23**. Env is the `verl` conda env (Python 3.12).

## TL;DR — which script for what

Every experiment is one thin `exp_*.sh` file: it `source`s `lib/common.sh` (all shared
env — conda, proxy, paths, batch, schedule), sets only the few knobs it differs on, then
calls `run_experiment <method>`. Experiments never call each other. To add one, copy an
`exp_*.sh`, change its knobs, done.

| Script | Method | Notes |
| --- | --- | --- |
| `run_distillation.sh {oprd\|opd\|oprd_opd}` | **core engine** | the only caller of `main_ppo`; all knobs env-overridable. Don't run directly — use an `exp_*.sh`. |
| `lib/common.sh` | shared base | sourced by every `exp_*.sh`; defines env + the `run_experiment` helper |
| `exp_oprd_opd.sh` | **OPD + rep** (combined) | token-OPD anchors output, rep is a small aux (coef 0.1). Stable; good first result. |
| `exp_oprd_bridge.sh` | **cross-arch OPRD-Bridge** | rep-only in a frozen low-rank subspace. **Requires the bridge built first** (below). |
| `exp_full_linear.sh` | **full-linear ablation** | no PCA; one trainable Linear(1024→2560) + normalized MSE. Compare vs the bridge. |
| `build_bridge.sh` | bridge prerequisites | builds + freezes the bridge (Stage 0 + Stage 1) that `exp_oprd_bridge.sh` needs |

> Same-architecture **OPRD-Vanilla** (naive `full` projector) does **not** apply to a 4B→0.6B pair:
> the projector never aligns (cosine ≈ 0) and rep-only collapses (repetition, AMC23 → 0). Cross-arch
> **must** use the low-rank Bridge below. See the paper §3 ("the method breaks down … cosine … zero").

## Why cross-arch needs the two-stage bridge

`exp_oprd_bridge.sh` (Stage 2) aligns student↔teacher hidden states inside a shared **rank-8**
subspace defined by a **frozen** bridge `(P_T, P_S)`:

- `P_T` — teacher PCA bases (top-8 principal directions per layer). Fixed.
- `P_S` — student projector (1024→8 per layer). **Trained offline, then frozen.**

Freezing matters: with a *trainable* P_S the bridge co-adapts with the backbone (a "rubber ruler")
— cosine hits ~0.97 but AMC23 still collapses, because high cosine is reachable without the backbone
becoming teacher-like. Freezing removes that shortcut. Build the frozen bridge first:

```bash
# One command (Stage 0 -> Stage 1), single GPU, idempotent:
bash experiments/build_bridge.sh
# -> outputs/bridge_construction/rank_8/ps_bank.pt
```

Then run the distillation:

```bash
bash experiments/exp_oprd_bridge.sh          # reads the rank_8 bridge, REP_FREEZE_PS=True
```

### The two stages in detail

**Stage 0 — collect on-policy pairs** (`scripts/analysis/run_cross_arch_analysis.sh`)
Student generates on-policy responses; `{raw_prompt, response}` saved to
`outputs/cross_arch_preexp1/on_policy_pairs.jsonl`. Single GPU, uses vLLM.

**Stage 1 — build + freeze the bridge** (`scripts/analysis/run_cross_arch_preexp2.sh`)
Recomputes teacher/student hidden states on frozen models, fits `P_T` (PCA) + trains `P_S`,
saves `outputs/bridge_construction/rank_${RANKS}/ps_bank.pt`. Inspect with
`python scripts/analysis/inspect_ps_bank.py <ps_bank.pt>`.

### Consistency rules (or the bridge silently won't load)

- `RANKS` (Stage 1) **==** `REP_LOW_RANK` (Stage 2). Default **8**.
- `LAYER_MODE` (Stage 1) **==** `REP_DISTILLATION_LAYERS` (Stage 2). Default **all**
  (Stage 1 only accepts `all|last|mid`; do not use `even/odd` here).
- Same student/teacher paths and tokenizer in both stages.
- At Stage 2 startup, confirm `rep/projector_loaded_from_checkpoint: 1.0` in the log — if it's
  `0.0`, the keys didn't match and P_S fell back to random (→ collapse).

## Environment gotchas baked into these scripts

These are handled automatically; listed so they're not re-discovered the hard way.

1. **PATH shadowing** — the shell profile puts another conda env ahead of `verl` on PATH, so
   `conda activate verl` alone leaves `python`/`ray`/`pip` pointing elsewhere. Every script forces
   `export PATH=/mnt/lxy/miniconda3/envs/verl/bin:$PATH`.
2. **Clash proxy** — `http_proxy=127.0.0.1:7890` hijacks localhost and breaks Ray/vLLM. Scripts set
   `NO_PROXY` (incl. `wandb.ai`) and models load offline (`HF_HUB_OFFLINE=1`).
3. **vLLM fork crash (Stage 0)** — `VLLM_WORKER_MULTIPROC_METHOD=spawn`, else the v1 engine core dies
   with "Cannot re-initialize CUDA in forked subprocess".
4. **`expandable_segments` (Stage 1 only)** — set to reduce fragmentation OOM. Do **not** set it for
   anything running vLLM (Stage 0 / Stage 2); vLLM's memory pool asserts against it.
5. **GPU 0 is shared** with another user → scripts default to `CUDA_VISIBLE_DEVICES=4,5,6,7`
   (Stage 0/1 use a single card, default 4).
6. **`IS_PLOT=False`** in Stage 2 — the `is_plot` viz block calls `swanlab.log` without initializing
   swanlab (only when `log_prob_top_k>0`), which throws harmlessly every 10 steps. Turned off.

> The `scripts/analysis/*.sh` launchers were rewritten: the originals passed CLI flags the current
> Python argparsers reject (e.g. `--subspace-mode`, `--generate-backend`, `--max-batch-tokens`) and
> used `--layer-mode even`, so they crashed on launch. The versions here pass only supported flags.

## What we observed (4B → 0.6B, DAPO-Math-5k, AMC23 acc@4)

| Setup | Result |
| --- | --- |
| rep-only, `full` projector (vanilla) | cosine ~0; collapse to repetition, acc 0.28 → 0 by step 50 |
| `oprd_opd` (OPD + rep 0.1) | stable, acc 0.28 → 0.30 → 0.33 (step 0/50/75) |
| Bridge, **trainable** P_S (on-the-fly) | cosine 0.17 → 0.97 **but** acc → 0.037 (rubber-ruler) |
| Bridge, **frozen** P_S (this pipeline) | see current `exp_oprd_bridge.sh` run |
