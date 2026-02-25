Run a local test evaluation using inspect mode (tiny model, 5 instances): $ARGUMENTS

The argument should be a task or task suite name (e.g., `chembench:mc`, `chembench:gen`, `legalbench:rc`, `frenchbench:rc`, or individual tasks like `chembench_organic_chemistry:mc`).

Steps:
1. Activate the environment: `source /data/input/kf/FlexMoE/.venv/bin/activate`
2. Run the eval from the eval scripts directory:
```bash
cd /data/input/kf/FlexMoE/src/scripts/eval && PYTHONPATH=/data/input/kf/FlexMoE/src python -u launch_eval.py \
    --task <TASK> \
    --inspect \
    --output-dir /tmp/eval_test_<TASK_SAFE_NAME> \
    --gpus 1 \
    --batch-size 1
```
3. Check the output for errors and report the summary of primary scores at the end.

Notes:
- `--inspect` uses pythia-160m (tiny model) with limit=5 instances for fast local testing
- Task suites are defined in `src/scripts/eval/task_suites.py`
- Individual task configs are in `src/scripts/eval/tasks.py` (overrides) and `src/offline_evals/tasks/` (implementations)
- The entry point is `src/scripts/eval/launch_eval.py` which calls `offline_evals.run_eval`
- To list available tasks: `PYTHONPATH=src python -u src/scripts/eval/launch_eval.py --list-tasks`
- To list available task suites: `PYTHONPATH=src python -u src/scripts/eval/launch_eval.py --list-task-suites`

Running on Beaker (production evals with real models):
- Edit `scripts/kevinf/eval/launch.sh` to set MODELS and TASKS arrays, then run: `bash scripts/kevinf/eval/launch.sh`
- Uses Gantry to launch jobs on ai2/saturn cluster with Weka mounts and Beaker secrets for HF/AWS
- Results go to BASE_OUTPUT_DIR; view with: `bash scripts/kevinf/eval/print_evals.sh`
- See CLAUDE.md "Running on Beaker" section for full details
