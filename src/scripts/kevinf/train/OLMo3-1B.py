# isort: skip_file
"""
Train OLMo3-1B on OLMoE-mix-0824.

Launch this with torchrun:

    torchrun --nproc-per-node=8 src/scripts/kevinf/train/OLMo3-1B.py run_name [OVERRIDES...]

Or use the launch script for Beaker:

    bash src/scripts/kevinf/train/launch_olmo3_1b.sh
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from typing import cast, List, Optional

import rich

from olmo_core.config import Config, DType
from olmo_core.data import (
    DataMix,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.data.numpy_dataset import NumpyDatasetConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank
from olmo_core.eval.task_groups import TASK_GROUPS
from olmo_core.float8 import Float8Config
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    GPUMemoryMonitorCallback,
    HFConverterCallback,
    PostTrainEvalCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all, setup_logging

log = logging.getLogger(__name__)

DATA_ROOT = "/weka/oe-training-default/ai2-llm"

SEQUENCE_LENGTH = 8 * 1024
GLOBAL_BATCH_SIZE = 4 * 1024 * 1024


@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    """Model config."""
    dataset: NumpyDatasetConfig
    """Dataset config."""
    data_loader: NumpyDataLoaderConfig
    """Data loader config."""
    trainer: TrainerConfig
    """Trainer config."""
    train_module: TransformerTrainModuleConfig
    """Train module config. Contains settings for optimizer."""
    init_seed: int = 12536
    """Random seed to initialize model weights."""
    load_path: Optional[str] = None
    """Path to load checkpoint from if no checkpoint is found in the save folder.
    Mainly used when you want to fine-tune from a pretrained model."""
    load_trainer_state: bool = False
    """Whether to load the trainer state (including data loader state) when loading from `load_path`.
    This only makes sense when trainer state is available in the checkpoint and you're resuming
    on the same dataset."""


def train(config: ExperimentConfig):
    if get_rank() == 0:
        rich.print(config)

    # Set RNG states on all devices.
    seed_all(config.init_seed)

    # Build components.
    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = config.trainer.build(train_module, data_loader)

    # Save config to W&B and each checkpoint dir.
    config_dict = config.as_config_dict()
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    # Try to load checkpoint: save_folder takes priority over load_path.
    if not trainer.no_checkpoints:
        checkpoint_loaded_from_save_folder = trainer.maybe_load_checkpoint()

        if checkpoint_loaded_from_save_folder:
            if config.load_path:
                log.warning(
                    f"Ignoring load_path ('{config.load_path}') since checkpoint was found in save folder"
                )
        elif config.load_path:
            log.info(
                f"Loading checkpoint from {config.load_path} since no checkpoints were found in the save folder..."
            )
            trainer.load_checkpoint(config.load_path, load_trainer_state=config.load_trainer_state)
        else:
            log.info("No checkpoint found, training from scratch...")

    # Train.
    trainer.fit()


def build_config(opts, overrides: List[str]) -> ExperimentConfig:
    save_folder = opts.save_folder
    if not save_folder:
        save_folder = f"/weka/oe-training-default/kevinf/checkpoints/{opts.run_name}/"

    work_dir = opts.work_dir
    if not work_dir:
        work_dir = "/tmp/dataset-cache"

    tokenizer_config = TokenizerConfig.dolma2()

    # Model config: OLMo3-1B
    model_config = TransformerConfig.olmo3_1B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        attn_backend=AttentionBackendName.flash_2,
    )

    log.info(f"Using data root: {DATA_ROOT}")

    # Dataset config
    dataset_config = NumpyFSLDatasetConfig.from_data_mix(
        DataMix.OLMoE_mix_0824,
        tokenizer=tokenizer_config,
        mix_base_dir=DATA_ROOT,
        sequence_length=SEQUENCE_LENGTH,
        max_target_sequence_length=max(8192, SEQUENCE_LENGTH),
        work_dir=work_dir,
    )

    # Data loader config
    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=34521,
        num_workers=4,
    )

    # Train module config
    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=SEQUENCE_LENGTH,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=SkipStepAdamWConfig(
            lr=4e-4,
            weight_decay=0.033,  # Not ablated; from OLMo3 defaults
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_fraction=0.1),
    )

    # Trainer config
    trainer_config = (
        TrainerConfig(
            save_folder=save_folder,
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.tokens(int(130e9)),
            hard_stop=Duration.tokens(int(130e9)),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=250,
                save_async=False,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=opts.run_name,
                group=opts.run_name,
                entity="allennlp",
                project="dolma3-ablation-kevinf",
                enabled=True,
                cancel_check_interval=10,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=TASK_GROUPS["fast"],
                tokenizer=tokenizer_config,
                eval_interval=250,
            ),
        )
        .with_callback(
            "hf_converter",
            HFConverterCallback(
                enabled=True,
                validate=False,  # Skip validation for speed
            ),
        )
        .with_callback(
            "post_train_eval",
            PostTrainEvalCallback(
                eval_output_base_dir="/data/input/kevinf/flexmoe/eval/results",
                cluster="ai2/saturn",
                enabled=True,
            ),
        )
    )

    config = ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
    )

    # Apply overrides.
    config = config.merge(overrides)

    return config


def parse_args():
    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        usage=f"python {sys.argv[0]} RUN_NAME [OPTIONS...] [CONFIG_OVERRIDES...]",
        description="Train OLMo3-1B",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run_name", type=str, help="""The name of the run.""")
    parser.add_argument(
        "--save-folder",
        type=str,
        help="""A local or remote directory to save checkpoints to.
        Defaults to /weka/oe-training-default/kevinf/checkpoints/{run_name}/""",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        help="""A local working directory for dataset preprocessing.
        Defaults to a temporary directory if not provided.""",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="""Print the config and exit.""",
    )
    opts, overrides = parser.parse_known_args()
    return opts, overrides


def main():
    opts, overrides = parse_args()
    setup_logging()
    config = build_config(opts, overrides)

    if opts.dry_run:
        rich.print(config)
        return

    prepare_training_environment()
    train(config)
    teardown_training_environment()


if __name__ == "__main__":
    main()
