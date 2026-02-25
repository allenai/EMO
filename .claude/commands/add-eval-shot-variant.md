Add N-shot evaluation variants for an existing benchmark: $ARGUMENTS

The argument should be in the format: `<benchmark> <shot_counts>` (e.g., `frenchbench 0,5` or `chembench 0,3,5` or `legalbench 5`).

## Steps

### 1. Find the base task classes and TASK_REGISTRY entries

Look in `src/offline_evals/tasks/<benchmark>.py` to find the base task classes (e.g., `FrenchBenchBoolQRC`).
Look in `src/offline_evals/__init__.py` to find the registered task names (e.g., `frenchbench_boolq:rc`).

### 2. Add stub subclasses in the benchmark's task file

In `src/offline_evals/tasks/<benchmark>.py`, add an empty stub subclass for each base task × shot count combination. These exist purely so each variant gets a unique `task_name` in TASK_REGISTRY (required for unique output filenames — `run_eval.py` uses `task_name` for file naming, so without unique names different shot variants overwrite each other).

```python
class FrenchBenchBoolQRC_0shot(FrenchBenchBoolQRC):
    pass

class FrenchBenchBoolQRC_5shot(FrenchBenchBoolQRC):
    pass
```

Also add a `BASE_TASKS` constant listing just the base task names (without shot suffixes), so suites can reference them cleanly:

```python
FRENCHBENCH_BASE_TASKS = [
    "frenchbench_boolq",
    "frenchbench_arc_challenge",
    ...
]
```

Register all variants in the `create_*_tasks()` function:

```python
"frenchbench_boolq:rc": FrenchBenchBoolQRC,
"frenchbench_boolq:rc:0shot": FrenchBenchBoolQRC_0shot,
"frenchbench_boolq:rc:5shot": FrenchBenchBoolQRC_5shot,
```

### 3. Add TASK_CONFIGS entries in `src/scripts/eval/tasks.py`

For each base task and each shot count, add a config entry:

```python
for fb_task in ["frenchbench_boolq", "frenchbench_arc_challenge", ...]:
    TASK_CONFIGS[f"{fb_task}:rc:{N}shot::olmes"] = {
        "task_name": f"{fb_task}:rc:{N}shot",  # matches the new TASK_REGISTRY key
        "num_shots": N,
        "primary_metric": "acc_per_char",  # match the benchmark's existing metric
        "metadata": {"regimes": []},
    }
```

Key rules:
- `task_name` must match the TASK_REGISTRY key (including the shot count suffix)
- `num_shots` is what actually controls the shot count at runtime
- `::olmes` suffix is a namespace convention on the config key, no runtime effect
- Check the benchmark's existing TASK_CONFIG_DEFAULTS for the right `primary_metric` and any other fields needed (e.g., `split`, `fewshot_source`)

### 4. Add task suites in `src/scripts/eval/task_suites.py` (only if benchmark has multiple subtasks)

Use the `BASE_TASKS` constant to build suites:

```python
"<benchmark>:rc:{N}shot": {
    "tasks": [f"{t}:rc:{N}shot::olmes" for t in <benchmark>.BENCHMARK_BASE_TASKS],
    "primary_metric": "macro",
},
```

### 5. Verify the setup

Run `/test-eval <benchmark>:rc:{N}shot` to test locally with inspect mode.

### 6. Add to launch script

Add the new suite names to `scripts/kevinf/eval/launch.sh` TASKS array:
```bash
<benchmark>:rc:0shot
<benchmark>:rc:5shot
```

## How it works

The eval system has 3 layers:
1. **TASK_REGISTRY** (`src/offline_evals/__init__.py`) - maps task names to Python classes
2. **TASK_CONFIGS** (`src/scripts/eval/tasks.py`) - maps config aliases to parameter dicts with runtime overrides
3. **TASK_SUITE_CONFIGS** (`src/scripts/eval/task_suites.py`) - groups configs under shorthand names

Each shot variant needs a unique stub subclass + TASK_REGISTRY entry because `run_eval.py` uses `task.task_name` for output filenames. Without unique names, the second variant sees the first's output file and skips it ("already processed").

The stub classes are empty (`pass`) — they inherit all behavior from the parent. The `num_shots` field in TASK_CONFIGS is what actually controls the shot count at runtime.

## Notes

- For 5-shot, tasks need a source of fewshot examples. Check if `has_training_docs()` returns True. If not, fewshot examples fall back to the eval set.
- Result files are named from `task_name` (colons → underscores), so `frenchbench_boolq:rc:5shot` → `task-frenchbench_boolq_rc_5shot-metrics.json`.
- See CLAUDE.md "Adding Shot Variants to an Existing Benchmark" for the reference documentation.
