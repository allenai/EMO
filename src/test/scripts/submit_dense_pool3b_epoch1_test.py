from argparse import Namespace

import pytest

from scripts.models.submit_dense_pool3b_epoch1 import SEQ, audit, stable_step, topology


SOURCE_EXPERIMENT = "01SOURCE"
SOURCE_CHECKPOINT = f"/weka/models/dense_153m_source/step{stable_step(1_000_000_000, 64)}"


def source_spec(weight_decay: str = "0.1"):
    return {
        "tasks": [
            {
                "arguments": [
                    "python",
                    "src/scripts/train/olmo2-1B.py",
                    "source",
                    "--model-size=153M",
                    "--save-folder=/weka/models/dense_153m_source",
                    f"--data_loader.global_batch_size={64 * SEQ}",
                    "--lr=2e-3",
                    f"--train_module.optim.weight_decay={weight_decay}",
                ]
            }
        ]
    }


def report():
    return {
        "poolPlan": {
            "fixedLearningRateByBatch": {"64": "2e-3"},
            "initialWeightDecayByBatch": {"64": ["0.1", "0.033"]},
        },
        "baseline1b": {
            "batchSweeps": [
                {
                    "batchSequences": 64,
                    "lr": "2e-3",
                    "wd": "0.1",
                    "beaker": SOURCE_EXPERIMENT,
                    "results": {
                        "1": {
                            "status": "complete",
                            "resumeCheckpoint": SOURCE_CHECKPOINT,
                        }
                    },
                }
            ]
        },
        "batchSweeps": [],
    }


def args(weight_decay: str):
    return Namespace(
        model="153m",
        global_sequences=64,
        target_epoch=1,
        weight_decay=weight_decay,
        source_checkpoint=SOURCE_CHECKPOINT,
        base_experiment=SOURCE_EXPERIMENT,
        recover_failed_experiment=None,
    )


def test_epoch_one_accepts_exact_weight_decay_matched_source():
    predecessor = audit(args("0.1"), report(), source_spec("0.1"))

    assert predecessor["wd"] == "0.1"


def test_epoch_one_rejects_cross_weight_decay_source():
    with pytest.raises(SystemExit, match="report predecessor mismatch"):
        audit(args("0.033"), report(), source_spec("0.1"))


@pytest.mark.parametrize(
    ("model", "batch", "expected"),
    [
        ("153m", 64, (4, 1, 4, 1)),
        ("153m", 128, (8, 1, 8, 1)),
        ("153m", 256, (8, 1, 8, 2)),
        ("153m", 512, (8, 1, 8, 4)),
        ("1b", 64, (8, 1, 8, 1)),
        ("1b", 128, (8, 1, 8, 2)),
        ("1b", 256, (8, 1, 8, 4)),
        ("1b", 512, (8, 1, 8, 8)),
    ],
)
def test_topology_caps_each_experiment_at_eight_gpus(model, batch, expected):
    assert topology(model, batch) == expected
