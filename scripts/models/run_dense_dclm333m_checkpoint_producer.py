#!/usr/bin/env python3
"""Run one retry-safe DCLM-333M producer with isolated POST branches."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_dense_1b_dr_wt_embedwd_grid as dense1b
import run_small_dense_dr_wt_embedwd_chain as small
import run_small_dense_saturation_chain as common

POLICY = "dense_dclm333m_integrated_producer_eval_v1"
TRAINING_SCRIPT = "src/scripts/train/olmo2-1B.py"
DATA_BASE_DIR = Path("/weka/oe-training-default/ai2-llm")
OUTPUT_ROOT = Path("/weka/oe-training-default/sewonm/icsl/models")
POOL_TOKENS = 333_000_000
SEQUENCE_LENGTH = 4096
DECAY_FRACTION = 0.1
NPROC_PER_NODE = 8
REFERENCE_WARMUP_SEQUENCE_STEPS = 24_576
EXPECTED_DATASET_MANIFEST = Path(
    "src/olmo_core/data/subsets/0802/dclm_0802_repeated_train_333m.json"
)
EXPECTED_MATERIALIZED_PATH = (
    "../sewonm/icsl/data/dclm_0802_nested_333m_from_1b/dclm_0802_repeated_train_333m_uint32.npy"
)

MODEL_POLICIES: dict[str, dict[str, Any]] = {
    "474m": {
        "batches": (64, 128),
        "lr": "2e-3",
        "wds": ("0.3", "1.0"),
        "retained_checkpoint_epochs": (8, 16, 24, 32),
        "evaluation_epochs": (16, 32),
        "max_epoch": 32,
        "base_experiment": "01KZ7307CK7ZZQ1XCJ2QQ08KD4",
    },
    "153m": {
        "batches": (64, 128),
        "lr": "2e-3",
        "wds": ("0.1", "0.3"),
        "retained_checkpoint_epochs": (16, 32, 48, 64, 80, 96, 112, 128),
        "evaluation_epochs": (32, 64, 96, 128),
        "max_epoch": 128,
        "base_experiment": "01KZ6Q4DJ8J994A6SQ39MEGTZ2",
    },
}

BS64_474M_CONTINUATION_TARGETS = (48, 64)
BS64_153M_WD03_CONTINUATION_TARGETS = (160,)
ALL_CONTINUATION_TARGETS = tuple(
    sorted(set(BS64_474M_CONTINUATION_TARGETS + BS64_153M_WD03_CONTINUATION_TARGETS))
)


def authorized_continuation_targets(item: dict[str, Any]) -> tuple[int, ...]:
    model = str(item["model"])
    batch = int(item["batchSequences"])
    wd = str(item["weightDecay"])
    if model == "474m" and batch == 64:
        return BS64_474M_CONTINUATION_TARGETS
    if model == "153m" and batch == 64 and wd == "0.3":
        return BS64_153M_WD03_CONTINUATION_TARGETS
    return ()


def continuation_source_epoch(item: dict[str, Any], target_epoch: int) -> int:
    candidates = [
        int(epoch) for epoch in item["evaluationEpochs"] if int(epoch) < target_epoch
    ]
    if not candidates:
        raise ValueError(f"E{target_epoch} continuation has no preceding POST frontier")
    return max(candidates)


def total_step(epoch: int, batch_sequences: int) -> int:
    return math.ceil(epoch * POOL_TOKENS / (batch_sequences * SEQUENCE_LENGTH))


def stable_step(epoch: int, batch_sequences: int) -> int:
    endpoint = total_step(epoch, batch_sequences)
    return endpoint - round(DECAY_FRACTION * endpoint) - 1


def warmup_steps(batch_sequences: int) -> int:
    if REFERENCE_WARMUP_SEQUENCE_STEPS % batch_sequences:
        raise ValueError(f"BS{batch_sequences} does not preserve token-matched warmup")
    return REFERENCE_WARMUP_SEQUENCE_STEPS // batch_sequences


def expected_output(item: dict[str, Any]) -> Path:
    return (
        OUTPUT_ROOT
        / f"dense_{item['model']}_dclm333m"
        / (
            f"bs{item['batchSequences']}_dr_wt_embwd_"
            f"lr{item['learningRate']}_wd{item['weightDecay']}"
        )
    )


def validate_dataset_manifest(path: Path, *, check_artifacts: bool = False) -> dict[str, Any]:
    value = json.loads(path.read_text())
    selection = value.get("selection", {})
    audit = value.get("nestedness_audit", {})
    if value.get("format") != "olmo-token-subset-v1":
        raise ValueError("DCLM-333M manifest has the wrong format")
    if int(selection.get("requested_tokens", 0)) != POOL_TOKENS:
        raise ValueError("DCLM-333M manifest has the wrong requested-token count")
    if selection.get("method") != "global-sha256-document-order-prefix":
        raise ValueError("DCLM-333M must retain the Pool-1B global document order prefix")
    if (
        selection.get("domain") != "dclm-train-repeated-sample-v1"
        or int(selection.get("seed", -1)) != 1
    ):
        raise ValueError("DCLM-333M selection domain/seed does not match Pool-1B")
    if value.get("materialized", {}).get("path") != EXPECTED_MATERIALIZED_PATH:
        raise ValueError("DCLM-333M materialized path has the wrong directory semantics")
    if value.get("source", {}).get("nested_base_manifest") != str(
        Path("src/olmo_core/data/subsets/0802/dclm_0802_repeated_train_1b.json")
    ):
        raise ValueError("DCLM-333M manifest has the wrong Pool-1B provenance pointer")
    base_manifest = Path(str(value["source"]["nested_base_manifest"]))
    base_digest = hashlib.sha256(base_manifest.read_bytes()).hexdigest()
    if base_digest != value["source"].get("nested_base_manifest_sha256"):
        raise ValueError("DCLM-333M Pool-1B provenance hash does not match")
    entries_digest = hashlib.sha256(
        json.dumps(value.get("entries"), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if entries_digest != value.get("entries_sha256"):
        raise ValueError("DCLM-333M manifest entry checksum mismatch")
    if audit.get("passed") is not True:
        raise ValueError("DCLM-333M nestedness audit did not pass")
    required_audit = (
        "selection_is_exact_leading_ledger_prefix",
        "all_selected_documents_are_in_base",
        "document_boundary_preserving",
        "boundary_is_strictly_before_base_boundary",
    )
    if not all(audit.get(key) is True for key in required_audit):
        raise ValueError("DCLM-333M nestedness audit is incomplete")
    if int(audit.get("base_requested_tokens", 0)) != 1_000_000_000:
        raise ValueError("DCLM-333M base is not the sealed Pool-1B")
    if int(audit.get("selected_document_intersection_with_base", -1)) != int(
        selection.get("selected_documents", -2)
    ):
        raise ValueError("DCLM-333M selected-document intersection is incomplete")
    if check_artifacts:
        materialized = value["materialized"]
        artifacts = {
            "tokens": DATA_BASE_DIR / str(materialized["path"]),
            "metadata": DATA_BASE_DIR / str(materialized["document_metadata_path"]),
            "ledger": DATA_BASE_DIR / str(value["source_document_ledger"]["path"]),
        }
        missing = [f"{name}:{path}" for name, path in artifacts.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing DCLM-333M artifacts: " + ", ".join(missing))
        expected_bytes = int(selection["selected_tokens"]) * 4
        if artifacts["tokens"].stat().st_size != expected_bytes:
            raise RuntimeError("DCLM-333M token artifact size does not match its manifest")
    return value


def validate_coordinate(item: dict[str, Any]) -> None:
    model = str(item.get("model"))
    if model not in MODEL_POLICIES:
        raise ValueError(f"unsupported model {model}")
    policy = MODEL_POLICIES[model]
    batch = int(item["batchSequences"])
    lr = str(item["learningRate"])
    wd = str(item["weightDecay"])
    if batch not in policy["batches"]:
        raise ValueError(f"{model} does not authorize BS{batch}")
    if Decimal(lr) != Decimal(str(policy["lr"])):
        raise ValueError(f"{model} has the wrong LR")
    if wd not in policy["wds"]:
        raise ValueError(f"{model}/BS{batch} has unauthorized WD{wd}")
    retained = [int(epoch) for epoch in item["retainedCheckpointEpochs"]]
    evaluations = [int(epoch) for epoch in item["evaluationEpochs"]]
    max_epoch = int(item["maxEpoch"])
    continuation_targets = authorized_continuation_targets(item)
    if model == "474m" and max_epoch in continuation_targets:
        expected_retained = list(range(8, max_epoch + 1, 8))
        expected_evaluations = [epoch for epoch in (16, 32, 48, 64) if epoch <= max_epoch]
    elif model == "153m" and max_epoch in continuation_targets:
        expected_retained = list(range(16, max_epoch + 1, 16))
        expected_evaluations = [
            epoch for epoch in (32, 64, 96, 128, 160) if epoch <= max_epoch
        ]
    else:
        expected_retained = list(policy["retained_checkpoint_epochs"])
        expected_evaluations = list(policy["evaluation_epochs"])
    if retained != expected_retained:
        raise ValueError(f"{model} has the wrong retained-checkpoint ladder")
    if evaluations != expected_evaluations:
        raise ValueError(f"{model} has the wrong evaluation ladder")
    if not set(evaluations).issubset(retained):
        raise ValueError("every evaluation epoch must have an exact retained PD source")
    if max_epoch != retained[-1]:
        raise ValueError(f"{model} max epoch must match the retained-checkpoint frontier")
    if max_epoch != int(policy["max_epoch"]) and max_epoch not in continuation_targets:
        raise ValueError(f"{model} has the wrong producer ceiling")
    if int(item["rankMicrobatchSequences"]) * NPROC_PER_NODE != batch:
        raise ValueError("rank microbatch must produce the global batch without accumulation")
    if str(item["baseExperiment"]) != policy["base_experiment"]:
        raise ValueError(f"{model} has the wrong trusted base experiment")
    if Path(str(item["output"])) != expected_output(item):
        raise ValueError(f"output must be exactly {expected_output(item)}")
    if int(item.get("estimatedDeviceTokensPerSecond", 0)) <= 0:
        raise ValueError("runtime estimate requires positive throughput")


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("policy") != POLICY:
        raise ValueError(f"manifest policy must be {POLICY}")
    if value.get("pool") != "dclm333m" or int(value.get("poolTokens", 0)) != POOL_TOKENS:
        raise ValueError("producer manifest must remain scoped to DCLM-333M")
    if Path(str(value.get("datasetManifest"))) != EXPECTED_DATASET_MANIFEST:
        raise ValueError("producer manifest references the wrong dataset manifest")
    if int(value.get("sequenceLength", 0)) != SEQUENCE_LENGTH:
        raise ValueError("producer manifest has the wrong sequence length")
    if Decimal(str(value.get("decayFraction"))) != Decimal(str(DECAY_FRACTION)):
        raise ValueError("pre-decay step convention must remain the uncapped 10% WSD boundary")
    coordinates = value.get("producerCoordinates", [])
    if len(coordinates) != 8:
        raise ValueError("producer manifest must contain exactly eight small-model coordinates")
    ids = [str(item["id"]) for item in coordinates]
    outputs = [str(item["output"]) for item in coordinates]
    if len(ids) != len(set(ids)) or len(outputs) != len(set(outputs)):
        raise ValueError("producer IDs and output directories must be unique")
    for item in coordinates:
        validate_coordinate(item)
    for model, policy in MODEL_POLICIES.items():
        expected = len(policy["batches"]) * len(policy["wds"])
        if sum(item["model"] == model for item in coordinates) != expected:
            raise ValueError(f"manifest does not contain exactly {expected} {model} coordinates")
    validate_dataset_manifest(EXPECTED_DATASET_MANIFEST)
    return value


def coordinate(manifest: dict[str, Any], coordinate_id: str) -> dict[str, Any]:
    matches = [item for item in manifest["producerCoordinates"] if item["id"] == coordinate_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one coordinate {coordinate_id}")
    return matches[0]


def coordinate_for_target(
    manifest: dict[str, Any], coordinate_id: str, target_epoch: int | None
) -> dict[str, Any]:
    item = copy.deepcopy(coordinate(manifest, coordinate_id))
    if target_epoch is None:
        return item
    allowed_targets = authorized_continuation_targets(item)
    if target_epoch not in allowed_targets:
        raise ValueError(
            f"{item['id']} does not authorize continuation target E{target_epoch}"
        )
    if item["model"] == "474m":
        item["retainedCheckpointEpochs"] = list(range(8, target_epoch + 1, 8))
        item["evaluationEpochs"] = [
            epoch for epoch in (16, 32, 48, 64) if epoch <= target_epoch
        ]
    else:
        item["retainedCheckpointEpochs"] = list(range(16, target_epoch + 1, 16))
        item["evaluationEpochs"] = [
            epoch for epoch in (32, 64, 96, 128, 160) if epoch <= target_epoch
        ]
    item["maxEpoch"] = target_epoch
    item["continuationTargetEpoch"] = target_epoch
    validate_coordinate(item)
    return item


def base_arguments(item: dict[str, Any], *, heldout_enabled: bool = False) -> list[str]:
    model = str(item["model"])
    batch = int(item["batchSequences"])
    common_arguments = list(small.COMMON_ARGUMENTS)
    common_arguments = common.upsert(
        common_arguments,
        "--dataset.subset_manifest=",
        f"--dataset.subset_manifest={EXPECTED_DATASET_MANIFEST}",
    )
    common_arguments = common.upsert(
        common_arguments,
        "--trainer.callbacks.downstream_evaluator.tasks=",
        "--trainer.callbacks.downstream_evaluator.tasks=[]",
    )
    common_arguments = common.upsert(
        common_arguments,
        "--trainer.callbacks.downstream_evaluator.eval_on_finish=",
        "--trainer.callbacks.downstream_evaluator.eval_on_finish=false",
    )
    heldout = small.HELDOUT_EVALUATOR
    if not heldout_enabled:
        heldout = heldout.replace("eval_on_finish: true", "eval_on_finish: false")
    common_arguments = common.upsert(
        common_arguments,
        "--trainer.callbacks.heldout_evaluator=",
        f"--trainer.callbacks.heldout_evaluator={heldout}",
    )
    model_arguments = list(small.MODEL_ARGUMENTS[model])
    return [
        *common_arguments,
        *model_arguments,
        f"--data_loader.global_batch_size={batch * SEQUENCE_LENGTH}",
        f"--train_module.rank_microbatch_size={int(item['rankMicrobatchSequences']) * SEQUENCE_LENGTH}",
        f"--train_module.optim.weight_decay={item['weightDecay']}",
        f"--lr={item['learningRate']}",
        "--model.tie_embeddings=true",
        "--decay-embeddings",
    ]


def checkpoint_complete(path: Path) -> bool:
    model_and_optim = path / "model_and_optim"
    train_state = path / "train"
    return (
        path.is_dir()
        and (path / "config.json").is_file()
        and model_and_optim.is_dir()
        and (model_and_optim / ".metadata").is_file()
        and any(model_and_optim.glob("*.distcp"))
        and train_state.is_dir()
        and all((train_state / f"rank{rank}.pt").is_file() for rank in range(NPROC_PER_NODE))
    )


def state_dir(item: dict[str, Any]) -> Path:
    return Path(str(item["output"])) / ".dclm333m_integrated_producer_eval_v1"


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def initialize_state(item: dict[str, Any], steps: list[int]) -> dict[str, Any]:
    output = Path(str(item["output"]))
    marker = state_dir(item) / "producer.json"
    if marker.is_file():
        state = json.loads(marker.read_text())
        if state.get("policy") != POLICY or state.get("id") != item["id"]:
            raise RuntimeError(f"output ownership marker mismatch: {marker}")
        return state
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing nonempty unowned output directory {output}")
    output.mkdir(parents=True, exist_ok=True)
    state = {
        "policy": POLICY,
        "id": item["id"],
        "status": "starting",
        "output": str(output),
        "retainedCheckpointEpochs": item["retainedCheckpointEpochs"],
        "evaluationEpochs": item["evaluationEpochs"],
        "targetSteps": steps,
        "scheduler": "constant_with_warmup",
        "dynamicRepacking": True,
        "weightTying": True,
        "embeddingWeightDecay": True,
        "evaluationEnabled": True,
        "decayEnabled": True,
        "postBranchesIsolatedFromConstantFrontier": True,
    }
    atomic_json(marker, state)
    return state


def run_torch(name: str, arguments: list[str], log_path: Path) -> None:
    subprocess.run(["python", TRAINING_SCRIPT, name, "--dry-run", *arguments], check=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log_file:
        common.stream_command(
            [
                "torchrun",
                f"--nproc-per-node={NPROC_PER_NODE}",
                TRAINING_SCRIPT,
                name,
                *arguments,
            ],
            log_file,
        )


def constant_arguments(
    item: dict[str, Any],
    epoch: int,
    checkpoint_step: int,
    name: str,
    source: Path | None = None,
) -> list[str]:
    batch = int(item["batchSequences"])
    output = Path(str(item["output"]))
    arguments = [
        *base_arguments(item),
        f"--save-folder={output}",
        f"--trainer.max_duration={{value: {checkpoint_step}, unit: steps}}",
        f"--trainer.callbacks.wandb.name={name}",
        (
            "--trainer.callbacks.wandb.tags=[pretraining,step1,0802,dclm333m,"
            "integrated-producer,constant-lr,pre-decay,dr,wt,embwd,"
            f"dense-{item['model']},bs{batch},e{epoch},lr{item['learningRate']},"
            f"wd{item['weightDecay']}]"
        ),
        f"--trainer.callbacks.checkpointer.fixed_steps=[{checkpoint_step}]",
        "--trainer.callbacks.checkpointer.save_interval=1000000000",
        "--trainer.callbacks.checkpointer.ephemeral_save_interval=1000",
        "--data_loader.restore_data_order_from_state=true",
        "--data_loader.ignore_fingerprint_mismatch=false",
        (
            "--train_module.scheduler={_CLASS_: "
            "olmo_core.optim.scheduler.ConstantWithWarmup, "
            f"warmup: {warmup_steps(batch)}}}"
        ),
        "--dynamic-repacking",
    ]
    if source is not None:
        arguments.extend(
            [
                "--force_exact_trainer_load_path=true",
                f"--trainer.load_path={source}",
                "--trainer.load_trainer_state=true",
                "--trainer.load_optim_state=true",
                "--trainer.reset_data_loader_state_on_load_path=false",
                "--train_module.validate_optimizer_hyperparameters_on_load=true",
            ]
        )
    return arguments


def validate_predecay_source(item: dict[str, Any], epoch: int) -> Path:
    batch = int(item["batchSequences"])
    source = Path(str(item["output"])) / f"step{stable_step(epoch, batch)}"
    if not checkpoint_complete(source):
        raise FileNotFoundError(f"incomplete exact pre-decay checkpoint {source}")
    value = json.loads((source / "config.json").read_text())
    checks = {
        "global_batch_size": (
            int(value["data_loader"]["global_batch_size"]),
            batch * SEQUENCE_LENGTH,
        ),
        "tie_embeddings": (bool(value["model"]["tie_embeddings"]), True),
        "dynamic_repacking": (bool(value["dataset"]["dynamic_repacking"]), True),
        "subset_manifest": (
            str(value["dataset"]["subset_manifest"]),
            str(EXPECTED_DATASET_MANIFEST),
        ),
    }
    mismatches = [name for name, pair in checks.items() if pair[0] != pair[1]]
    optim = value["train_module"]["optim"]
    if Decimal(str(optim["lr"])) != Decimal(str(item["learningRate"])):
        mismatches.append("learning_rate")
    if Decimal(str(optim["weight_decay"])) != Decimal(str(item["weightDecay"])):
        mismatches.append("weight_decay")
    if optim.get("group_overrides") != []:
        mismatches.append("embedding_weight_decay")
    if mismatches:
        raise RuntimeError(f"pre-decay checkpoint recipe mismatch {mismatches}: {source}")
    return source


def postdecay_arguments(
    item: dict[str, Any], epoch: int, source: Path, output: Path, name: str
) -> list[str]:
    batch = int(item["batchSequences"])
    endpoint = total_step(epoch, batch)
    arguments = common.upsert(
        base_arguments(item, heldout_enabled=True),
        "--train_module.scheduler=",
        (
            "--train_module.scheduler={_CLASS_: olmo_core.optim.scheduler.WSD, "
            f"units: steps, warmup: {warmup_steps(batch)}, decay_fraction: {DECAY_FRACTION}}}"
        ),
    )
    return [
        *arguments,
        "--dynamic-repacking",
        f"--save-folder={output}",
        f"--trainer.max_duration={{value: {endpoint}, unit: steps}}",
        f"--trainer.callbacks.wandb.name={name}",
        (
            "--trainer.callbacks.wandb.tags=[pretraining,step1,0802,dclm333m,"
            "integrated-producer,post-decay,heldout-eval,dr,wt,embwd,"
            f"dense-{item['model']},bs{batch},e{epoch},lr{item['learningRate']},"
            f"wd{item['weightDecay']}]"
        ),
        f"--trainer.callbacks.checkpointer.fixed_steps=[{endpoint}]",
        "--trainer.callbacks.checkpointer.save_interval=1000000000",
        "--trainer.callbacks.checkpointer.ephemeral_save_interval=999999999",
        "--data_loader.restore_data_order_from_state=false",
        "--data_loader.ignore_fingerprint_mismatch=true",
        "--force_exact_trainer_load_path=true",
        f"--trainer.load_path={source}",
        "--trainer.load_trainer_state=true",
        "--trainer.load_optim_state=true",
        "--trainer.reset_data_loader_state_on_load_path=false",
        "--train_module.validate_optimizer_hyperparameters_on_load=true",
    ]


def recovered_evaluation_arguments(
    item: dict[str, Any], checkpoint: Path, output: Path, name: str
) -> list[str]:
    heldout = small.HELDOUT_EVALUATOR.replace(
        "eval_on_finish: true", "eval_on_finish: false, eval_on_startup: true"
    )
    arguments = common.upsert(
        base_arguments(item),
        "--trainer.callbacks.heldout_evaluator=",
        f"--trainer.callbacks.heldout_evaluator={heldout}",
    )
    return [
        *arguments,
        "--dynamic-repacking",
        f"--save-folder={output}",
        f"--trainer.callbacks.wandb.name={name}",
        "--trainer.callbacks.wandb.tags=[pretraining,step1,0802,dclm333m,recovered-heldout-eval]",
        "--trainer.max_duration={value: 1000000000000, unit: steps}",
        "--trainer.callbacks.checkpointer.enabled=false",
        f"--load_path={checkpoint}",
        "--load_trainer_state=false",
    ]


def evaluate(item: dict[str, Any], epoch: int) -> dict[str, Any]:
    result_path = state_dir(item) / "results" / f"e{epoch}.json"
    if result_path.is_file():
        return json.loads(result_path.read_text())
    source = validate_predecay_source(item, epoch)
    output = state_dir(item) / "post_decay_runs" / f"e{epoch}"
    endpoint = output / f"step{total_step(epoch, int(item['batchSequences']))}"
    name = f"{item['id']}-post-e{epoch}-v1"
    if checkpoint_complete(endpoint):
        log_path = state_dir(item) / "logs" / f"e{epoch}_recovered_eval.log"
        recovered_name = f"{name}-recovered-eval"
        run_torch(
            recovered_name,
            recovered_evaluation_arguments(
                item, endpoint, output / "recovered_eval", recovered_name
            ),
            log_path,
        )
    else:
        log_path = state_dir(item) / "logs" / f"e{epoch}.log"
        print(
            f"DENSE_DCLM333M_POST_START id={item['id']} epoch={epoch} "
            f"source={source} output={output}",
            flush=True,
        )
        run_torch(name, postdecay_arguments(item, epoch, source, output, name), log_path)
    if not checkpoint_complete(endpoint):
        raise RuntimeError(f"E{epoch} decay exited without complete endpoint {endpoint}")
    result = dense1b.parse_validation(log_path, epoch, "post_decay", endpoint)
    result.update(
        {
            "policy": POLICY,
            "model": str(item["model"]),
            "batchSequences": int(item["batchSequences"]),
            "lr": str(item["learningRate"]),
            "wd": str(item["weightDecay"]),
            "variant": "DR+WT+EmbedWD",
            "sourcePreDecayCheckpoint": str(source),
            "source": "integrated_isolated_wsd_decay_and_heldout_eval",
        }
    )
    atomic_json(result_path, result)
    print(
        f"DENSE_DCLM333M_POST_RESULT id={item['id']} epoch={epoch} "
        f"json={json.dumps(result, separators=(',', ':'), sort_keys=True)}",
        flush=True,
    )
    return result


def run(item: dict[str, Any]) -> None:
    validate_coordinate(item)
    validate_dataset_manifest(EXPECTED_DATASET_MANIFEST, check_artifacts=True)
    batch = int(item["batchSequences"])
    epochs = [int(epoch) for epoch in item["retainedCheckpointEpochs"]]
    evaluation_epochs = {int(epoch) for epoch in item["evaluationEpochs"]}
    steps = [stable_step(epoch, batch) for epoch in epochs]
    output = Path(str(item["output"]))
    state = initialize_state(item, steps)
    state.update(
        {
            "retainedCheckpointEpochs": epochs,
            "evaluationEpochs": sorted(evaluation_epochs),
            "targetSteps": steps,
            "continuationTargetEpoch": item.get("continuationTargetEpoch"),
        }
    )
    atomic_json(state_dir(item) / "producer.json", state)

    for index, (epoch, checkpoint_step) in enumerate(zip(epochs, steps)):
        checkpoint = output / f"step{checkpoint_step}"
        if not checkpoint_complete(checkpoint):
            prior_sources = [
                (prior_epoch, output / f"step{prior_step}")
                for prior_epoch, prior_step in zip(epochs[:index], steps[:index])
                if checkpoint_complete(output / f"step{prior_step}")
            ]
            source_epoch: int | None = None
            source: Path | None = None
            if prior_sources:
                source_epoch, source = prior_sources[-1]
                source = validate_predecay_source(item, source_epoch)
            name = f"{item['id']}-pd-e{epoch}-v1"
            state.update({"status": "producer_running", "currentEpoch": epoch})
            atomic_json(state_dir(item) / "producer.json", state)
            print(
                f"DENSE_DCLM333M_PD_START id={item['id']} epoch={epoch} "
                f"step={checkpoint_step} sourceEpoch={source_epoch} source={source} "
                f"output={output}",
                flush=True,
            )
            run_torch(
                name,
                constant_arguments(item, epoch, checkpoint_step, name, source),
                state_dir(item) / "logs" / f"pd_e{epoch}.log",
            )
        if not checkpoint_complete(checkpoint):
            raise RuntimeError(f"producer exited without complete E{epoch} checkpoint")
        print(
            f"DENSE_DCLM333M_PD_RETAINED id={item['id']} epoch={epoch} checkpoint={checkpoint}",
            flush=True,
        )
        if epoch in evaluation_epochs:
            state.update({"status": "post_running", "currentEpoch": epoch})
            atomic_json(state_dir(item) / "producer.json", state)
            evaluate(item, epoch)
            print(
                f"DENSE_DCLM333M_POST_COMPLETE id={item['id']} epoch={epoch}",
                flush=True,
            )

    missing = [
        epoch
        for epoch, step in zip(epochs, steps)
        if not checkpoint_complete(output / f"step{step}")
    ]
    missing_results = [
        epoch
        for epoch in sorted(evaluation_epochs)
        if not (state_dir(item) / "results" / f"e{epoch}.json").is_file()
    ]
    if missing or missing_results:
        raise RuntimeError(
            f"integrated job incomplete: missing PD={missing}, POST={missing_results}"
        )
    state.update({"status": "complete", "currentEpoch": epochs[-1]})
    atomic_json(state_dir(item) / "producer.json", state)
    print(
        f"DENSE_DCLM333M_JOB_COMPLETE id={item['id']} retained={epochs} "
        f"evaluated={sorted(evaluation_epochs)}",
        flush=True,
    )


def runtime_estimate(item: dict[str, Any], runtime_policy: dict[str, Any]) -> dict[str, Any]:
    batch = int(item["batchSequences"])
    final_step = stable_step(int(item["maxEpoch"]), batch)
    training_tokens = final_step * batch * SEQUENCE_LENGTH
    aggregate_tokens_per_second = int(item["estimatedDeviceTokensPerSecond"]) * NPROC_PER_NODE
    training_seconds = training_tokens / aggregate_tokens_per_second
    evaluation_epochs = [int(epoch) for epoch in item["evaluationEpochs"]]
    decay_tokens = sum(
        (total_step(epoch, batch) - stable_step(epoch, batch)) * batch * SEQUENCE_LENGTH
        for epoch in evaluation_epochs
    )
    heldout_tokens = int(runtime_policy["heldoutTokensPerEvaluation"]) * len(evaluation_epochs)
    evaluation_throughput = int(
        runtime_policy["evaluationAggregateTokensPerSecond"][str(item["model"])]
    )
    evaluation_seconds = (decay_tokens + heldout_tokens) / evaluation_throughput
    overhead_seconds = int(runtime_policy["checkpointOverheadSeconds"]) * (
        len(item["retainedCheckpointEpochs"]) + len(evaluation_epochs)
    ) + int(runtime_policy["evaluationStartupOverheadSeconds"]) * len(evaluation_epochs)
    raw_seconds = training_seconds + evaluation_seconds + overhead_seconds
    buffered_seconds = raw_seconds * (1 + float(runtime_policy["bufferFraction"]))
    round_seconds = int(runtime_policy["roundMinRuntimeSeconds"])
    min_runtime_seconds = math.ceil(buffered_seconds / round_seconds) * round_seconds
    return {
        "finalStep": final_step,
        "trainingTokens": training_tokens,
        "decayTokens": decay_tokens,
        "heldoutTokens": heldout_tokens,
        "aggregateTokensPerSecond": aggregate_tokens_per_second,
        "trainingSeconds": training_seconds,
        "evaluationSeconds": evaluation_seconds,
        "overheadSeconds": overhead_seconds,
        "rawSeconds": raw_seconds,
        "bufferedSeconds": buffered_seconds,
        "minRuntimeSeconds": min_runtime_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--coordinate", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--check-data-artifacts", action="store_true")
    parser.add_argument("--dry-run-stages", action="store_true")
    parser.add_argument("--target-epoch", type=int, choices=ALL_CONTINUATION_TARGETS)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    item = coordinate_for_target(manifest, args.coordinate, args.target_epoch)
    validate_coordinate(item)
    if args.validate_only:
        if args.check_data_artifacts:
            validate_dataset_manifest(EXPECTED_DATASET_MANIFEST, check_artifacts=True)
        print(f"validated DCLM-333M producer {args.coordinate}")
        return
    if args.dry_run_stages:
        batch = int(item["batchSequences"])
        retained_epoch = int(item["retainedCheckpointEpochs"][0])
        evaluation_epoch = int(item["evaluationEpochs"][0])
        constant_name = f"{item['id']}-pd-e{retained_epoch}-dry-run"
        subprocess.run(
            [
                sys.executable,
                TRAINING_SCRIPT,
                constant_name,
                "--dry-run",
                *constant_arguments(
                    item,
                    retained_epoch,
                    stable_step(retained_epoch, batch),
                    constant_name,
                ),
            ],
            check=True,
        )
        post_name = f"{item['id']}-post-e{evaluation_epoch}-dry-run"
        subprocess.run(
            [
                sys.executable,
                TRAINING_SCRIPT,
                post_name,
                "--dry-run",
                *postdecay_arguments(
                    item,
                    evaluation_epoch,
                    Path(str(item["output"])) / f"step{stable_step(evaluation_epoch, batch)}",
                    state_dir(item) / "post_decay_runs" / f"e{evaluation_epoch}",
                    post_name,
                ),
            ],
            check=True,
        )
        print(f"dry-run validated producer and POST stages for {args.coordinate}")
        return
    run(item)


if __name__ == "__main__":
    main()
