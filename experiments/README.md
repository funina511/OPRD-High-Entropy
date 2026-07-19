# OPRD on 4× RTX 3090 — experiment scripts

Preliminary reproduction of **OPRD** (arXiv 2606.06021) on this machine.
Student **Qwen3-0.6B** (28 layers, d=1024), teacher **Qwen3-4B** (36 layers, d=2560),
data **DAPO-Math-5k**, eval **AMC23**. Env is the `verl` conda env (Python 3.12).

## TL;DR — which script for what

Every experiment is one thin `exp_*.sh` file: it `source`s `setup/common.sh` (all shared
env — conda, batch, schedule), sets only the few knobs it differs on, then calls
`run_experiment <method>`. Experiments never call each other. To add one, copy an
`exp_*.sh`, change its knobs, done.

### Single source of truth + per-host profiles (read this before editing)

There is **one** `experiments/` dir, shared across machines via git. The ML logic
(`run_distillation.sh` + the `exp_*.sh` knobs) is identical everywhere — edit it once,
`git pull` on the other box, done. **Machine-specific** values (conda path, model/data
dirs, CUDA devices, proxy, ray tuning) live only in `setup/hosts/<host>.sh`:

- `common.sh` picks the profile by `$OPRD_HOST`, else `hostname`. `good3090` resolves by
  hostname automatically. The `siton141` container has an ephemeral hostname (a docker id),
  so it sets `export OPRD_HOST=siton141` in its shell rc.
- **Adding a machine:** `cp setup/hosts/_template.sh setup/hosts/<name>.sh` and fill it in
  (the template lists every required var; copy `good3090.sh` / `siton141.sh` instead if one
  is closer to your box). If the hostname is stable it auto-selects; if it's a container id,
  `export OPRD_HOST=<name>` in that box's rc. Never put a host path in `run_distillation.sh`
  or an `exp_*.sh`.
- **Fail-fast on incomplete profiles:** after sourcing the host file, `common.sh` asserts the
  required vars (`OPRD_CONDA_SH/ENV/BIN`, `CUDA_VISIBLE_DEVICES`, `N_GPUS_PER_NODE`,
  `OPRD_REPO_ROOT`, `MODEL_DIR`, `DATA_DIR`) are non-empty and errors naming any that are
  missing — so a half-filled profile can't silently degrade `PATH`/paths.

**Anti-drift guard:** `run_distillation.sh` runs `setup/check_plumbing.sh` at startup — it
fails fast if any exported knob has no matching hydra override (the silent no-op bug where a
var is set + baked into the run name but never reaches the trainer). Skip with
`SKIP_PLUMBING_CHECK=1` while iterating on the script itself.

| Script | Method | Notes |
| --- | --- | --- |
| `run_distillation.sh {oprd\|opd\|oprd_opd}` | **core engine** | the only caller of `main_ppo`; all knobs env-overridable. Don't run directly — use an `exp_*.sh`. |
| `setup/common.sh` | shared base | sourced by every `exp_*.sh`; selects host profile, defines env + the `run_experiment` helper |
| `setup/hosts/<host>.sh` | per-machine | the ONLY place host paths/CUDA/proxy/ray-tuning live |
| `setup/check_plumbing.sh` | anti-drift guard | asserts every exported knob reaches the trainer; auto-run at startup |
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
   `conda activate verl` alone leaves `python`/`ray`/`pip` pointing elsewhere. `common.sh` forces
   `export PATH=$OPRD_CONDA_BIN:$PATH` (the bin dir comes from the host profile).
2. **Proxy** — host-dependent, set in the host profile. `good3090` runs direct with `wandb.ai` in
   `NO_PROXY`; `siton141` must route wandb *through* its clash `HTTP(S)_PROXY` (direct GraphQL times
   out there) and unsets socks `ALL_PROXY` (makes wandb-core hang). Models load offline everywhere
   (`HF_HUB_OFFLINE=1`).
3. **vLLM fork crash (Stage 0)** — `VLLM_WORKER_MULTIPROC_METHOD=spawn`, else the v1 engine core dies
   with "Cannot re-initialize CUDA in forked subprocess".
4. **`expandable_segments` (Stage 1 only)** — set to reduce fragmentation OOM. Do **not** set it for
   anything running vLLM (Stage 0 / Stage 2); vLLM's memory pool asserts against it.
5. **GPU visibility** is per-host (`CUDA_VISIBLE_DEVICES` in the host profile): `good3090` defaults
   to `4,5,6,7` (GPU 0 shared with another user), `siton141` to `0`.
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
