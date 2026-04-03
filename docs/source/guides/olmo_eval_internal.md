# Running `olmo-eval-internal` on FlexMoE checkpoints

This guide shows how to run FlexMoE evaluations from a separate
`olmo-eval-internal` checkout.

Use a separate environment for `olmo-eval-internal`.

## Setup

From the `olmo-eval-internal` repo:

```bash
cd olmo-eval-internal
uv sync --dev --extra beaker --extra storage
source .venv/bin/activate

olmo-eval tasks
olmo-eval suites
```

Can also use `uv run olmo-eval ...`.

## Task mapping

Some common mappings are:

| FlexMoE / historical usage | `olmo-eval-internal` target |
| --- | --- |
| `mt_mbpp_*_gold_bpb_3shot` | `mt_mbpp_v2fix:3shot:bpb` |
| `code_fresh_rolling:bpb` | `code_fresh:bpb` |

This is not a complete list. Other evals and suites also exist in
`olmo-eval-internal`, and more of the OLMo 3 evals are expected to show up on
`main` as the pending review work lands.

Useful suites:

- `mt_mbpp_v2fix:3shot:bpb`
- `code_fresh:bpb`
- `minerva_math_olmo3`

## Common variables

Run the commands below from `olmo-eval-internal`.

```bash
USER_TAG=your-name
MODEL=/weka/oe-training-default/kevinf/checkpoints/train-olmo3-1b-dolma50-stackedu-python50-10B-lr5e-5-ctd/step2385-hf
WORKSPACE=ai2/flex2
BUDGET=ai2/oceo
DATE=$(date +%Y%m%d)
```

Replace `MODEL` with your checkpoint path.

Use `--dry-run` first if you want to inspect the Beaker spec before launching.

## Flags used below

- `-n`: experiment name
- `-g`: Beaker group name
- `-m`: model or checkpoint path
- `-H`: harness preset
- `-t`: task or suite
- `-o`: override for the preceding task or harness
- `-c`: cluster or GPU type
- `-w`: Beaker workspace
- `-B`: Beaker budget
- `-p`: Beaker priority
- `--gpus`: number of GPUs
- `--store`: store results in S3 and Postgres
- `--no-follow`: launch without streaming logs
- `-y`: skip confirmation

## Smoke tests

### MT-MBPP smoke test

```bash
olmo-eval beaker launch \
  -n "${USER_TAG}-mtmbpp-v2fix-python-smoke" \
  -g "${USER_TAG}-mtmbpp-v2fix-smoke-${DATE}" \
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

## Full suite examples

### Full `mt_mbpp_v2fix:3shot:bpb`

```bash
olmo-eval beaker launch \
  -n "${USER_TAG}-mtmbpp-v2fix-full" \
  -g "${USER_TAG}-mtmbpp-v2fix-${DATE}" \
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

### General suite example: `minerva_math_olmo3`

```bash
olmo-eval beaker launch \
  -n "${USER_TAG}-minerva-math-olmo3" \
  -g "${USER_TAG}-minerva-math-${DATE}" \
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

Launch it from `olmo-eval-internal`:

```bash
olmo-eval beaker launch \
  -f ../FlexMoE/src/scripts/eval/examples/olmo_eval_internal_beaker.yaml \
  -m "$MODEL" \
  -H default \
  -g "${USER_TAG}-flexmoe-olmo-eval-${DATE}" \
  --store \
  --dry-run
```

The template keeps shared settings in one place while letting you override the
model path on the command line. If we don't like this can remove and not use

## OLMo 3 base evals

If you want the OLMo 3 base-eval suites from Table 45 & 46 in the paper, make sure the
`olmo-eval-internal` checkout you launch from already includes those task
definitions. Once they merge to `main`, the normal `main` checkout is enough.

Right now many of these evals are not on `main` yet. They are expected to
be available on `main` after TylerM reviews David's PR.

The top-level suites look like:

- `olmobase:mcqa_stem`
- `olmobase:mcqa_non_stem`
- `olmobase:gen`
- `olmobase:math`
- `olmobase:easy:qa:rc`
- `olmobase:easy:qa:bpb`
- `olmobase:easy:math:bpb`
- `olmobase:easy:code:bpb`

The launch shape for all base evals in the OLMo 3 paper looks like this:

```bash
olmo-eval beaker launch \
  -n "${USER_TAG}-olmobase-debug" \
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
export OLMO_EVAL_DB_HOST="your-db-host"
export OLMO_EVAL_DB_SECRET_ARN="your-secret-arn"

olmo-eval results query -G "${USER_TAG}-mtmbpp-v2fix-${DATE}"
olmo-eval results query -G "${USER_TAG}-mtmbpp-v2fix-${DATE}" --format csv
olmo-eval results query -G "${USER_TAG}-mtmbpp-v2fix-${DATE}" --format json
```

Local querying needs both `OLMO_EVAL_DB_HOST` and `OLMO_EVAL_DB_SECRET_ARN`.
If `results query` times out even with those set, switch to the exit node or
VPN path that can reach the shared Postgres host.

## Troubleshooting

- Launch from the `olmo-eval-internal` checkout, not from the FlexMoE checkout.
- For FlexMoE checkpoint paths, start with `-H default`.
- `results query` needs `OLMO_EVAL_DB_HOST` and `OLMO_EVAL_DB_SECRET_ARN`.
- If `results query` times out even with the correct env vars, switch to the exit node or VPN
  path that can reach the shared Postgres host.
- If you do not have the storage secrets yet, drop `--store`. The run will still write Beaker
  `/results`, but it will not be queryable through `olmo-eval results query`.
- Start with `limit=5` smoke tests before launching full suites.

## Next step

If this workflow looks good and the evals are working as expected, the next step
is to move toward using this evaluation path in FlexMoE directly.

For the most up-to-date view of which evals are implemented and their current
parity status, see:

```text
https://docs.google.com/spreadsheets/d/1fBrMmk0G0VGoKrjlyJ08YVELlZZBB0RPGseK6vz7-SI/edit?gid=1326019620#gid=1326019620
```
