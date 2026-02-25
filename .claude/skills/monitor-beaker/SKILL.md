---
name: monitor-beaker
description: Monitor, diagnose, and restart Beaker experiments. Use when the user asks to monitor, check, track, restart, or debug Beaker experiments/jobs.
allowed-tools: Bash(beaker:*), Bash(source:*), Bash(python:*), Bash(export:*)
---

# Monitor & Manage Beaker Experiments

## Monitoring

When monitoring a Beaker experiment:

1. Get the experiment status using `beaker experiment get <experiment-id>`
2. Check if the experiment has completed by looking at `status.exited`
3. If still running, wait 30 seconds and check again
4. When complete:
   - If exitCode is 0: Report success
   - If exitCode is non-zero: Fetch and display logs with `beaker experiment logs <experiment-id>`
5. Continue monitoring until the experiment finishes or the user asks you to stop

## Listing & Filtering Experiments

Use `beaker-py` to list and filter experiments programmatically:

```python
from beaker import Beaker
b = Beaker.from_env()

# List recent experiments in workspace
experiments = list(b.workspace.experiments("ai2/flex2", limit=150))

# Filter by status
for exp in experiments:
    for job in exp.jobs:
        # job.status.exit_code: None (running), 0 (success), 1+ (failed)
        # job.status.exited: datetime if finished, None if running
        print(f"{exp.id} | exit={job.status.exit_code} | {exp.name}")
```

## Diagnosing Failures

Check logs for failed experiments:

```python
logs = b.experiment.logs(exp_id, quiet=True)
log_text = ""
for chunk in logs:
    log_text += chunk.decode("utf-8", errors="replace")

# Search for specific errors
if "whoami-v2" in log_text:
    print("HF rate limit error")
```

## Restarting Failed Experiments

Use `b.experiment.resume()` to restart failed/canceled experiments in-place (new job, same experiment ID):

```python
b.experiment.resume(experiment_id)
```

- Works for failed and canceled experiments (despite docstring saying "preempted")
- Creates a new job within the same experiment, preserving history
- If many jobs failed from the same transient error (e.g., rate limits), stagger restarts with `time.sleep(3)` to avoid re-triggering

### Bulk restart pattern

```python
import time
from beaker import Beaker

b = Beaker.from_env()
experiments = list(b.workspace.experiments("ai2/flex2", limit=150))

# Find failed experiments matching criteria
failed = []
for exp in experiments:
    if not exp.name.startswith("eval-"):
        continue
    for job in exp.jobs:
        if job.status.exit_code is not None and job.status.exit_code != 0:
            failed.append(exp)
            break

# Check logs to filter by specific error
whoami_failures = []
for exp in failed:
    logs = b.experiment.logs(exp.id, quiet=True)
    log_text = "".join(chunk.decode("utf-8", errors="replace") for chunk in logs)
    if "whoami-v2" in log_text:
        whoami_failures.append(exp)

# Resume with stagger
for exp in whoami_failures:
    b.experiment.resume(exp.id)
    time.sleep(3)
```

### Re-creating experiments (when resume doesn't work)

Get the spec from a failed experiment and create a new one:

```python
spec = b.experiment.spec(experiment_id)
new_exp = b.experiment.create("new-name", spec, workspace="ai2/flex2")
```

## Other Useful Operations

```python
# Get experiment spec (full job definition)
spec = b.experiment.spec(experiment_id)

# Get experiment URL
url = b.experiment.url(experiment_id)

# Stop a running experiment
b.experiment.stop(experiment_id)

# Delete an experiment
b.experiment.delete(experiment_id)

# Wait for experiments to complete
b.experiment.wait_for(exp1_id, exp2_id, timeout=3600)
```

## CLI Examples

Check experiment status:
```bash
beaker experiment get 01KCW39T5JBZTYV69BXHWJJ83P
```

Get experiment logs on failure:
```bash
beaker experiment logs 01KCW39T5JBZTYV69BXHWJJ83P
```

Stream logs in real-time for running experiments:
```bash
beaker experiment logs --follow 01KCW39T5JBZTYV69BXHWJJ83P
```
