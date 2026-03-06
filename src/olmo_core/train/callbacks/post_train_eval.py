"""
Callback to automatically launch evaluations when training completes successfully.

This callback expects an HF-format checkpoint to already exist (created by HFConverterCallback).
It looks for a checkpoint with the "-hf" suffix and launches evaluation jobs for it.
"""

import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, List, Optional

from olmo_core.distributed.utils import get_rank
from olmo_core.io import file_exists, join_path

from .callback import Callback

log = logging.getLogger(__name__)


# Default eval tasks (from src/scripts/kevinf/eval/launch.sh)
DEFAULT_EVAL_TASKS = [
    # MC9 tasks
    "arc_easy:mc::olmes",
    "arc_challenge:mc::olmes",
    "boolq:mc::olmes",
    "csqa:mc::olmes",
    "hellaswag:mc::olmes",
    "openbookqa:mc::olmes",
    "piqa:mc::olmes",
    "socialiqa:mc::olmes",
    "winogrande:mc::olmes",
    # Gen5 tasks
    "coqa::olmes",
    "squad::olmes",
    "naturalqs::olmes",
    "triviaqa::olmes",
    "drop::olmes",
    # MMLU
    "mmlu:mc::olmes",
    # AGI eval
    "agi_eval_english:1shot::olmes",
    # BBH
    "bbh:cot-v1::olmes",
    # Math2 tasks
    "gsm8k::olmes",
    "minerva_math_algebra::olmes",
    "minerva_math_counting_and_probability::olmes",
    "minerva_math_geometry::olmes",
    "minerva_math_intermediate_algebra::olmes",
    "minerva_math_number_theory::olmes",
    "minerva_math_prealgebra::olmes",
    "minerva_math_precalculus::olmes",
    # Code4 tasks
    "codex_humaneval:temp0.8",
    "codex_humanevalplus:temp0.8",
    "mbpp::none",
    "mbppplus::none",
    # ChemBench
    # "chembench:mc",
    "chembench:gen",
    "chembench:rc",
    "legalbench:rc",
    "frenchbench:rc",
]


@dataclass
class PostTrainEvalCallback(Callback):
    """
    Callback that launches evaluation jobs for an HF-format checkpoint when training completes.

    This callback expects an HF-format checkpoint to already exist (created by HFConverterCallback).
    It looks for a checkpoint with the "-hf" suffix and launches evaluation jobs via Beaker/gantry.

    This callback only runs on successful training completion (post_train), not on
    preemption or errors.

    .. note::
        Use this callback together with :class:`HFConverterCallback`. The HFConverterCallback
        should convert the checkpoint to HF format first, then this callback will launch evals.
    """

    priority: ClassVar[int] = -2  # Run after HFConverterCallback (priority=-1)

    eval_output_base_dir: str = "/data/input/kevinf/eval_results/flexmoe"
    """Base directory for evaluation results."""

    tasks: List[str] = field(default_factory=lambda: DEFAULT_EVAL_TASKS.copy())
    """List of evaluation tasks to run."""

    cluster: str = "ai2/saturn"
    """Beaker cluster to run evaluations on."""

    workspace: str = "ai2/flex2"
    """Beaker workspace."""

    budget: str = "ai2/oceo"
    """Beaker budget."""

    job_priority: str = "urgent"
    """Job priority for eval jobs."""

    github_repo: str = "allenai/FlexMoE"
    """GitHub repo for gantry (needed when remote URL has embedded token)."""

    batch_size: int = 4
    """Default batch size for evaluations."""

    limit: int = 1000
    """Limit for number of evaluation examples."""

    enabled: bool = True
    """Set to False to disable this callback."""

    def post_train(self):
        """
        Called when training completes successfully.
        Looks for an HF-format checkpoint and launches evaluation jobs.
        """
        if not self.enabled:
            log.info("PostTrainEvalCallback is disabled, skipping")
            return

        # Only run on rank 0
        if get_rank() != 0:
            return

        if self.trainer.is_canceled:
            log.info("Training was canceled, skipping post-train eval pipeline")
            return

        log.info("Training completed successfully! Starting eval pipeline...")

        # Find the HF checkpoint (expects HFConverterCallback to have run first)
        hf_checkpoint = self._find_hf_checkpoint()
        if hf_checkpoint is None:
            log.warning(
                "No HF checkpoint found. Make sure HFConverterCallback is enabled and ran successfully, or that the converted checkpoint exists.\n"
                "Looking for checkpoint with '-hf' suffix."
            )
            return

        log.info(f"Found HF checkpoint: {hf_checkpoint}")

        # Launch evaluations
        self._launch_evals(hf_checkpoint)

    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint in the save folder."""
        save_folder = self.trainer.save_folder

        try:
            checkpoints = list(self.trainer.checkpointer.find_checkpoints(save_folder))
            if not checkpoints:
                return None

            # find_checkpoints returns (step_num, path) tuples
            # Sort by step number and get the latest
            checkpoints.sort(key=lambda x: x[0], reverse=True)
            latest_step, latest_path = checkpoints[0]
            log.info(f"Found {len(checkpoints)} checkpoints, latest is step {latest_step}")
            return latest_path
        except FileNotFoundError:
            log.warning(f"Save folder not found: {save_folder}")
            return None

    def _find_hf_checkpoint(self) -> Optional[str]:
        """Find the HF-format checkpoint (expects HFConverterCallback ran first).

        Looks for a checkpoint with the '-hf' suffix that contains a config.json file.
        """
        latest_checkpoint = self._find_latest_checkpoint()
        if latest_checkpoint is None:
            return None

        # HF checkpoint should be at {checkpoint_path}-hf
        hf_checkpoint = f"{latest_checkpoint.rstrip('/')}-hf"

        # Check if HF checkpoint exists (config.json is a good indicator)
        config_path = join_path(hf_checkpoint, "config.json")
        if file_exists(config_path):
            return hf_checkpoint

        log.warning(f"HF checkpoint not found at expected path: {hf_checkpoint}")
        return None

    def _fix_git_remote(self):
        """Fix git remote URL to remove embedded token (which confuses gantry)."""
        try:
            # Set the remote to a clean URL without the token
            clean_url = f"https://github.com/{self.github_repo}.git"
            subprocess.run(
                ["git", "remote", "set-url", "origin", clean_url],
                check=True,
                capture_output=True,
            )
            log.info(f"Fixed git remote to: {clean_url}")
        except subprocess.CalledProcessError as e:
            log.warning(f"Could not fix git remote: {e.stderr}")

    def _transform_path_for_eval(self, path: str) -> str:
        """Transform weka path to the mount point used in eval jobs.

        Training jobs use: /weka/oe-training-default/...
        Eval jobs mount:   --weka oe-training-default:/data/input
        So we need:        /weka/oe-training-default/... -> /data/input/...
        """
        if path.startswith("/weka/oe-training-default/"):
            return path.replace("/weka/oe-training-default/", "/data/input/")
        return path

    def _launch_evals(self, hf_checkpoint_path: str):
        """Launch beaker evaluation jobs for the HF checkpoint."""
        # Derive the eval run name from the checkpoint path
        # e.g., /checkpoints/my-run/step1000-hf -> my-run_step1000-hf
        path = Path(hf_checkpoint_path)
        step_dir = path.name  # e.g., step1000-hf
        run_name = path.parent.name  # e.g., my-run
        eval_run_name = f"{run_name}_{step_dir}"

        # Transform path for eval container's mount point
        eval_checkpoint_path = self._transform_path_for_eval(hf_checkpoint_path)

        output_dir = f"{self.eval_output_base_dir}/{eval_run_name}"

        log.info(f"Launching evaluations for {eval_run_name}")
        log.info(f"  HF checkpoint (local): {hf_checkpoint_path}")
        log.info(f"  HF checkpoint (eval):  {eval_checkpoint_path}")
        log.info(f"  Output dir: {output_dir}")
        log.info(f"  Tasks: {len(self.tasks)} tasks")

        # Fix git remote URL (remove embedded token which confuses gantry)
        self._fix_git_remote()

        launched_count = 0
        failed_count = 0

        for task in self.tasks:
            # Determine batch size based on task type
            if any(
                x in task
                for x in ["cot", "minerva_math_", "mbpp", "bigcodebench", "ruler", "sciriff"]
            ):
                batch_size = 1
            else:
                batch_size = self.batch_size

            # Create safe job name
            safe_run_name = "".join(c for c in eval_run_name if c.isalnum() or c in "_-")
            safe_task_name = "".join(c for c in task if c.isalnum() or c in "_-")
            job_name = f"eval-{safe_run_name}-{safe_task_name}"

            # Build gantry command (try gantry directly, fall back to python -m)
            gantry_args = [
                "run",
                "--name",
                job_name,
                "--gh-token-secret",
                "KEVINF_GITHUB_TOKEN",  # For private repo clone
                "--weka",
                "oe-training-default:/data/input",
                "--install",
                'pip install uv && UV_CACHE_DIR=/tmp/uv-cache uv pip install -e ".[eval]"',
                "--budget",
                self.budget,
                "--workspace",
                self.workspace,
                "--cluster",
                self.cluster,
                "--priority",
                self.job_priority,
                "--gpus",
                "1",
                "--env-secret",
                "GITHUB_TOKEN=KEVINF_GITHUB_TOKEN",  # For private repo access
                "--env-secret",
                "HF_TOKEN=KEVINF_HF_TOKEN",
                "--env-secret",
                "AWS_ACCESS_KEY_ID=KEVINF_AWS_ACCESS_KEY_ID",
                "--env-secret",
                "AWS_SECRET_ACCESS_KEY=KEVINF_AWS_SECRET_ACCESS_KEY",
                "--allow-dirty",
                "--",
                "bash",
                "-c",
                f"PYTHONPATH=. python -u src/scripts/eval/launch_eval.py "
                f"--model {eval_checkpoint_path} "
                f"--model-type hf "
                f"--task {task} "
                f"--limit {self.limit} "
                f"--output-dir {output_dir} "
                f"--batch-size {batch_size} "
                f"--gpus 1",
            ]

            # Try different ways to invoke gantry
            cmd = ["gantry"] + gantry_args

            try:
                log.info(f"Launching eval job: {job_name}")
                # Try gantry directly first
                try:
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                except FileNotFoundError:
                    # gantry not in PATH, try python -m gantry
                    cmd = ["python", "-m", "gantry"] + gantry_args
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                launched_count += 1
            except FileNotFoundError:
                log.error("gantry command not found even after install attempt")
                failed_count += 1
                break  # No point trying more tasks if gantry isn't available
            except subprocess.CalledProcessError as e:
                log.error(f"Failed to launch eval job {job_name}: {e.stderr}")
                failed_count += 1

        log.info(f"Eval job launch complete: {launched_count} launched, {failed_count} failed")
        if failed_count > 0 and launched_count == 0:
            log.info("To manually launch evals, run locally:")
            log.info(f'  MODELS=("{hf_checkpoint_path}") bash src/scripts/kevinf/eval/launch.sh')
