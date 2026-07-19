# experiments141 — experiment entrypoints

Thin `exp_*.sh` scripts: each `source`s `setup/common.sh`, sets a few knobs, then
calls `run_experiment <oprd|opd|oprd_opd>` → `run_distillation.sh` → `main_ppo`.

**Do not add queue / “run A then B” wrappers here.** Chain jobs outside this folder
(or by hand). One experiment = one `exp_*.sh`.

## Layout

| Path | Role |
| --- | --- |
| `run_distillation.sh` | Core engine (only place that launches `main_ppo`) |
| `setup/common.sh` | Shared env + `run_experiment` helper |
| `exp_*.sh` | One config per experiment |
| `build_bridge.sh` | Prerequisite for bridge / low-rank OPRD |

## RKD / OPRD (current line)

```bash
# Baseline RKD-D (coef=3, all layers, last_k=1024)
bash experiments141/exp_rel_rkd.sh

# RKD-D + RKD-A (angle_coef=2)
bash experiments141/exp_rel_rkd_da_all.sh

# RKD-DA + student outcome/format RL (no token-KL / OPD)
bash experiments141/exp_rel_rkd_da_rl.sh
```

Optional ablations (same engine, different knobs):

- `exp_rel_rkd_last.sh` — layers=last  
- `exp_rel_rkd_alltok.sh` — positions=all  
- `exp_rel_infonce.sh` — InfoNCE instead of RKD  

## Other methods

- `exp_oprd_opd.sh` — OPD + rep  
- `exp_oprd_bridge.sh` / `exp_full_linear.sh` — bridge / full-linear (need `build_bridge.sh` first)  

## Override knobs

Any env var accepted by `run_distillation.sh` can be set before the `exp_*.sh`, e.g.:

```bash
N_RESPONSES=4 FORMAT_REWARD_COEF=0.1 bash experiments141/exp_rel_rkd_da_rl.sh
```
