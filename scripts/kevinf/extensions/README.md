# Extension Pipelines

Reusable 3-stage pipeline for extending MoE models with new experts.

## Quick start

```bash
bash scripts/kevinf/extensions/pipeline/run.sh math-ta-01 status
bash scripts/kevinf/extensions/pipeline/run.sh math-ta-01 next
```

## Layout

```
extensions/
├── pipeline/          # generic scripts (don't edit per-experiment)
│   ├── common.sh
│   ├── parse_config.py
│   ├── run.sh
│   ├── stage1_compute_activations.sh
│   ├── stage2_add_experts.sh
│   └── stage3_train_experts.sh
└── experiments/       # one YAML per experiment
    └── math-ta-01.yaml
```

## Creating a new experiment

1. Copy an existing YAML. The YAML filename is the experiment name:

```bash
cp scripts/kevinf/extensions/experiments/math-ta-01.yaml scripts/kevinf/extensions/experiments/code-ta-01.yaml
# edit base_model, mix, hyperparameters, etc.
```

2. Run it:

```bash
bash scripts/kevinf/extensions/pipeline/run.sh code-ta-01 status
bash scripts/kevinf/extensions/pipeline/run.sh code-ta-01 next
```

## Naming

The YAML filename is the experiment name (e.g. `math-ta-01`).
The `base_model.path` inside the YAML is the source of truth for artifact identity.

All artifacts for one experiment live under a single root:
`/weka/.../extension-experiments/<experiment>/<step>/`.
The step name comes from the checkpoint path, which keeps names short while still
separating runs from different base checkpoints.

| Artifact | Path |
|---|---|
| Experiment root | `/weka/.../extension-experiments/math-ta-01/step30995/` |
| Activations | `/weka/.../extension-experiments/math-ta-01/step30995/activations/...` |
| Extended checkpoint | `/weka/.../extension-experiments/math-ta-01/step30995/extended-checkpoint/` |
| Training save folder | `/weka/.../extension-experiments/math-ta-01/step30995/runs/math-ta-01_lr4e-4_10B_20260407-231540/` |
| W&B run | `allennlp/flex2-extensions-kevinf/math-ta-01_lr4e-4_10B_20260407-231540` |
| Beaker stage jobs | `math-ta-01-stage1`, `math-ta-01-stage2` |

Stage 3 auto-generated run names include the learning rate and token budget so
you can do simple sweeps under the same experiment YAML without relying only on
timestamps. You can still pass a custom run name explicitly if you want.

## Overriding config values

Config fields exported by `parse_config.py` can be overridden via environment variable:

```bash
STAGE3_LR=1e-4 bash scripts/kevinf/extensions/pipeline/run.sh math-ta-01 stage3
STAGE3_NUM_BILLION_TOKENS=5 bash scripts/kevinf/extensions/pipeline/run.sh math-ta-01 stage3
```

## Commands

| Command | What it does |
|---|---|
| `status` | Print resolved paths and check artifact existence |
| `stage1` | Compute training-data router activations |
| `stage2` | Add new experts to create extended checkpoint |
| `stage3 [name]` | Train the new experts (auto-generates timestamped name) |
| `next [name]` | Launch the next missing stage |

## Safety checks

- Stage 2 verifies Stage 1 activation exists
- Stage 2 refuses to overwrite an existing checkpoint (set `ALLOW_STAGE2_REUSE=1` to skip)
- Stage 3 verifies Stage 2 checkpoint exists
- Stage 3 refuses to reuse a save folder (set `ALLOW_STAGE3_RESUME=1` to skip)

## Expert layout

Base 128-expert model: regular `0..126`, shared `127`.
After adding 4 new experts: old `0..126`, new `127..130`, shared `131`.
Stage 3 trains: `127,128,129,130`.
