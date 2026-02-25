---
name: read-eval-results
description: Read, compare, and display evaluation results across models. Use when the user asks to check eval results, compare models, see scores, or generate eval tables.
allowed-tools: Bash(source:*), Bash(python:*), Bash(cd:*), Bash(ls:*)
---

# Read Eval Results

Display and compare evaluation results across models: $ARGUMENTS

Parse the user's request to determine which models and tasks to show, then run `print_evals.py`.

## Quick Reference

Results are stored at: `/data/input/kevinf/flexmoe/eval/results/`
Each model has a subdirectory containing `task-*metrics.json` files.

### Core command

```bash
source .venv/bin/activate && cd /Users/kevinfarhat/repos/FlexMoE

python src/scripts/eval/print_evals.py \
    -r \
    -t \
    -b /data/input/kevinf/flexmoe/eval/results/ \
    --show-models "model1,model2" \
    --show-tasks "task1,task2" \
    --avg-all
```

**Always use `-r` (reset cache).** The cache (`cached_results.pkl`) can be stale; always reset it.

### Flags

| Flag | Description |
|------|-------------|
| `-r` | **Always use.** Reset cache and re-scan directories |
| `-t` | Transpose table (models as rows, tasks as columns) — usually what you want |
| `-b <dir>` | Base directory for results (always `/data/input/kevinf/flexmoe/eval/results/`) |
| `--show-models "a,b"` | Only show models whose names contain any of these substrings |
| `--hide-models "a,b"` | Hide models whose names contain any of these substrings |
| `--show-tasks "a,b"` | Only show tasks whose names contain any of these substrings |
| `--hide-tasks "a,b"` | Hide tasks whose names contain any of these substrings |
| `--nicknames "pattern:Nick,..."` | Shorten model names in output (e.g., `"dolma:Dolma,olmoe:OLMoE"`) |
| `--export-csv results.csv` | Export results to CSV |

### Averaging flags

| Flag | What it averages |
|------|-----------------|
| `--avg-all` | **Shortcut: enables ALL averaging flags below** |
| `--avg-core` | Core 9 tasks (arc_easy, arc_challenge, boolq, csqa, hellaswag, openbookqa, piqa, socialiqa, winogrande) |
| `--avg-mmlu` | All MMLU subtasks → single `mmlu:mc` / `mmlu:rc` |
| `--avg-mmlu-pro` | MMLU Pro subtasks |
| `--avg-bbh` | BIG-Bench Hard subtasks |
| `--avg-gen` | Generation tasks (coqa, squad, naturalqs_open, triviaqa, drop) |
| `--avg-agi-eval` | AGI Eval subtasks |
| `--avg-mm` | Minerva Math subtasks |
| `--avg-ruler` | RULER subtasks |
| `--avg-sciriff` | SciRIFF subtasks |
| `--avg-chembench` | ChemBench (separately: mc, rc, gen) |
| `--avg-legalbench` | LegalBench RC subtasks |
| `--avg-frenchbench` | FrenchBench RC subtasks |
| `--avg-code` | Coding tasks (humaneval, mbpp — excludes @10 variants) |

### Task filter flags

| Flag | Description |
|------|-------------|
| `--stem-only` | Only STEM-related tasks |
| `--code-only` | Only coding tasks |
| `--hide-code` | Hide coding tasks |
| `--core-and-gen-only` | Only core 9 + gen 5 tasks |

## How to handle user requests

### "Show me eval results for X"
1. Identify model name substring(s) from the user's request
2. Run the command with `--show-models` set to those substrings
3. Use `--avg-all` unless the user wants individual subtask scores
4. Use `-t` for transposed view (models as rows) — easier to compare

### "Compare model A vs model B on task X"
1. Use `--show-models "modelA,modelB"`
2. Use `--show-tasks "taskX"` if a specific task is requested
3. Use `--avg-all` to collapse subtasks into aggregates

### "What models have been evaluated?"
```bash
ls /data/input/kevinf/flexmoe/eval/results/
```

### "What tasks were run for model X?"
```bash
ls /data/input/kevinf/flexmoe/eval/results/<model-dir>/
```
The files are named `task-<task_name>-metrics.json`.

### "Show me the raw results for a specific task"
```bash
python -c "
import json
with open('/data/input/kevinf/flexmoe/eval/results/<model-dir>/task-<task_name>-metrics.json') as f:
    print(json.dumps(json.load(f), indent=2))
"
```

## Example commands

```bash
# Compare code eval results across code-trained models
python src/scripts/eval/print_evals.py -r -t \
    -b /data/input/kevinf/flexmoe/eval/results/ \
    --show-models "code_fim_cpp,code_fim_java,code_fim_python" \
    --show-tasks "human,mbpp" \
    --avg-all

# See all results for the base model
python src/scripts/eval/print_evals.py -r -t \
    -b /data/input/kevinf/flexmoe/eval/results/ \
    --show-models "dolma3-0625-150Bsample" \
    --avg-all

# Compare domain-adapted models against base on legalbench
python src/scripts/eval/print_evals.py -r -t \
    -b /data/input/kevinf/flexmoe/eval/results/ \
    --show-models "dolma3-0625,the-pile-of-law" \
    --show-tasks "legalbench" \
    --avg-all

# Export comparison to CSV
python src/scripts/eval/print_evals.py -r -t \
    -b /data/input/kevinf/flexmoe/eval/results/ \
    --show-models "model1,model2" \
    --avg-all \
    --export-csv comparison.csv
```

## Output format

- Best score per task is highlighted in red
- Scores are displayed as floats (0.000–1.000)
- `N/A` means the task wasn't evaluated for that model
- A comparability check warns if models were evaluated on different numbers of instances for the same task
