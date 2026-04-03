# Running `olmo-eval-internal` on FlexMoE checkpoints

This guide documents how to evaluate FlexMoE checkpoints with a separate checkout
of `~/repos/olmo-eval-internal`.

Use this workflow when you want to compare FlexMoE checkpoints with the newer
`olmo-eval-internal` stack without changing FlexMoE's existing evaluation code.
It is intentionally a docs-first, external-usage path rather than an integration
or migration of `launch_eval.py` / `offline_evals`.

## What this guide does and does not do

- FlexMoE continues to own training, checkpoint conversion, and checkpoint storage.
- `olmo-eval-internal` is used as an external benchmark runner.
- This guide does **not** add `olmo-eval-internal` to FlexMoE's environment.
- This guide does **not** modify FlexMoE's `pyproject.toml`.
- This guide does **not** replace FlexMoE's existing `offline_evals` codepaths.

## Use a separate `olmo-eval-internal` environment

Run `olmo-eval-internal` from its own checkout and repo-local `uv` virtual
environment:

```bash
cd ~/repos/olmo-eval-internal
uv sync --dev --extra beaker --extra storage
source .venv/bin/activate

olmo-eval tasks
olmo-eval suites
```

If you prefer not to activate `.venv`, run the same commands with
`uv run olmo-eval ...` from the `~/repos/olmo-eval-internal` checkout.

Keep this environment separate from FlexMoE for three reasons:

- `olmo-eval-internal` has a different dependency surface than FlexMoE's current eval stack.
- `olmo-eval-internal` currently targets a newer Python baseline than FlexMoE.
- Keeping the environments separate avoids breaking existing FlexMoE training and eval workflows
  while we decide whether a deeper migration is worth doing.

## Task mapping for current parity targets

| FlexMoE / historical usage | `olmo-eval-internal` target |
| --- | --- |
| `mt_mbpp_*_gold_bpb_3shot` | `mt_mbpp_v2fix:3shot:bpb` |
| `code_fresh_rolling:bpb` | `code_fresh:bpb` |

The live `olmo-eval-internal` CLI currently exposes the exact suite names used in
the `main`-branch commands below:

- `mt_mbpp_v2fix:3shot:bpb`
- `code_fresh:bpb`
- `minerva_math_olmo3`

Additional OLMo 3 base-eval suites currently live on David's
`davidh/olmobasesuite` branch (and are expected to land via PR #117). Those are
covered in a separate section below because they are not yet part of the
current `main` checkout.

## Common launch variables

All launch commands below should be run from `~/repos/olmo-eval-internal`.
Start by setting the checkpoint path and common Beaker settings you want to use:

```bash
MODEL=/weka/oe-training-default/<user>/checkpoints/<run>/step2385-hf
WORKSPACE=ai2/flex2
BUDGET=ai2/oceo
DATE=$(date +%Y%m%d)
```

Use `--dry-run` first when you want to inspect the generated Beaker spec before
submitting a real job.

## Smoke tests

Start with small smoke tests before launching full suites.

### MT-MBPP smoke test

```bash
olmo-eval beaker launch \
  -n "<user>-mtmbpp-v2fix-python-smoke" \
  -g "<user>-mtmbpp-v2fix-smoke-${DATE}" \
  -m "$MODEL" \
  -H default \
  -t mt_mbpp_v2fix_python:3shot:bpb -o limit=5 \
  -c h100 \
  -w "$WORKSPACE" \
  -B "$BUDGET" \
  -p urgent \
  --store \
  --no-follow \
  -y
```

### CodeFresh smoke test

```bash
olmo-eval beaker launch \
  -n "<user>-codefresh-python-smoke" \
  -g "<user>-codefresh-smoke-${DATE}" \
  -m "$MODEL" \
  -H default \
  -t code_fresh_python:bpb -o limit=5 \
  -c h100 \
  -w "$WORKSPACE" \
  -B "$BUDGET" \
  -p urgent \
  --store \
  --no-follow \
  -y
```

## Full suite launches

### Full `mt_mbpp_v2fix:3shot:bpb`

```bash
olmo-eval beaker launch \
  -n "<user>-mtmbpp-v2fix-full" \
  -g "<user>-mtmbpp-v2fix-${DATE}" \
  -m "$MODEL" \
  -H default \
  -t mt_mbpp_v2fix:3shot:bpb \
  -c h100 \
  -w "$WORKSPACE" \
  -B "$BUDGET" \
  -p urgent \
  --store \
  --no-follow \
  -y
```

### Full `code_fresh:bpb`

```bash
olmo-eval beaker launch \
  -n "<user>-codefresh-full" \
  -g "<user>-codefresh-${DATE}" \
  -m "$MODEL" \
  -H default \
  -t code_fresh:bpb \
  -c h100 \
  -w "$WORKSPACE" \
  -B "$BUDGET" \
  -p urgent \
  --store \
  --no-follow \
  -y
```

### General suite example: `minerva_math_olmo3`

```bash
olmo-eval beaker launch \
  -n "<user>-minerva-math-olmo3" \
  -g "<user>-minerva-math-${DATE}" \
  -m "$MODEL" \
  -H default \
  -t minerva_math_olmo3 \
  -c h100 \
  -w "$WORKSPACE" \
  -B "$BUDGET" \
  -p urgent \
  --store \
  --no-follow \
  -y
```

## Reusable Beaker template

FlexMoE includes a reusable Beaker template at
`src/scripts/eval/examples/olmo_eval_internal_beaker.yaml`.

Launch it from the `olmo-eval-internal` checkout:

```bash
olmo-eval beaker launch \
  -f ~/repos/FlexMoE/src/scripts/eval/examples/olmo_eval_internal_beaker.yaml \
  -m "$MODEL" \
  -H default \
  -g "<user>-flexmoe-olmo-eval-${DATE}" \
  --store \
  --dry-run
```

The checked-in template keeps shared task and Beaker settings in one place while
letting you override the model path on the command line.

`olmo-eval-internal`'s README currently shows structured model entries such as
`name_or_path` plus `provider: vllm`. In this environment, the live Beaker config
loader dry-runs more reliably with plain string model specs in YAML. For
FlexMoE-style checkpoint paths, the current launcher also auto-detects a vLLM
provider path during dry-run, so the runnable examples in this guide keep the
launch shape as close as possible to the maintainer-recommended commands.

## OLMo 3 base evals on `davidh/olmobasesuite`

If you want the OLMo 3 base-eval suites from Table 3, use David's
`davidh/olmobasesuite` branch (or the equivalent commit after PR #117 merges).

The branch defines these top-level suites:

- `olmobase:mcqa_stem`
- `olmobase:mcqa_non_stem`
- `olmobase:gen`
- `olmobase:math`
- `olmobase:easy:qa:rc`
- `olmobase:easy:qa:bpb`
- `olmobase:easy:math:bpb`
- `olmobase:easy:code:bpb`

The maintainer-provided launch shape for "all base evals in the OLMo 3 paper"
looks like this:

```bash
olmo-eval beaker launch \
  -n "<user>-olmobase-debug" \
  -m "$MODEL" \
  -H default \
  -c h100 \
  -B ai2/oe-base \
  -p high \
  --inspect \
  --store \
  -y \
  -g olmo-3-parity-baseline \
  -w ai2/olmo-3-evals \
  --gpus 8 \
  -t olmobase:mcqa_stem \
  -t olmobase:mcqa_non_stem \
  -t olmobase:gen \
  -t olmobase:math \
  -t olmobase:easy:qa:rc \
  -t olmobase:easy:qa:bpb \
  -t olmobase:easy:math:bpb \
  -t olmobase:easy:code:bpb
```

For a published baseline model you can use a Hugging Face identifier such as
`allenai/Olmo-3-1025-7B`. For FlexMoE checkpoint parity work, replace the model
with your Weka checkpoint path.

## Querying stored results

When `--store` is enabled, Beaker runs write artifacts and store queryable metadata.
After the run finishes, query results by group:

```bash
export OLMO_EVAL_DB_HOST="<database-host>"
export OLMO_EVAL_DB_SECRET_ARN="arn:aws:secretsmanager:us-west-2:..."

olmo-eval results query -G "<user>-mtmbpp-v2fix-${DATE}"
olmo-eval results query -G "<user>-mtmbpp-v2fix-${DATE}" --format csv
olmo-eval results query -G "<user>-mtmbpp-v2fix-${DATE}" --format json
```

For parity analysis, CSV output is often the easiest format to diff and inspect locally.

This step requires a shell environment that can actually reach the shared
results database. In practice that means:

- `OLMO_EVAL_DB_HOST` and `OLMO_EVAL_DB_SECRET_ARN` must be set correctly
- AWS credentials must be available for secret lookup
- your current network path / exit node must be able to reach the database host

If those are not set up, `olmo-eval results query` may fall back to `localhost`
and fail with a PostgreSQL connection error. Even with the correct env vars, a
misconfigured exit node or VPN path can still fail with a connection timeout.

If you do not have the storage secrets yet, drop `--store` from the launch command.
The run will still write Beaker `/results`, but it will not be queryable through
`olmo-eval results query`.

## Troubleshooting

- Launch from `~/repos/olmo-eval-internal`, not from the FlexMoE checkout. Gantry clones the
  current repo into `/gantry-runtime`, so launching from the wrong checkout can install the
  wrong package set.
- For FlexMoE checkpoint paths, the current launcher auto-detects a vLLM-style provider path on
  dry-run. Start without a provider override unless you have a specific reason to force one.
- If you do need to force a provider override, attach it to the harness after `-H default`, for
  example `-H default -o provider.kind=vllm`.
- If Beaker install logs mention `ai2-olmo-core @ file:///gantry-runtime` or complain about
  missing `vllm` extras on the wrong repo, rerun the launch from `~/repos/olmo-eval-internal`.
- Local macOS is useful for CLI inspection, `--dry-run`, and `results query`, but the actual
  `vllm` inference path should be expected to run inside Beaker/Linux.
- `--store` requires the Beaker secrets `olmo_eval_PGHOST` and `olmo_eval_DB_SECRET_ARN`.
  Local querying uses the corresponding `OLMO_EVAL_DB_HOST` and `OLMO_EVAL_DB_SECRET_ARN`
  environment variables.
- Start with `limit=5` smoke tests before scaling to full suites. This catches task-resolution,
  model-path, and storage issues much earlier.
