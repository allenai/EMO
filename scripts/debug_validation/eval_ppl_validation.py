"""Offline v3-small perplexity validation of OLMoE3-ladder checkpoints (pinned submodule stack).

Reproduces OLMo-core's in-loop LM eval (``LMEvaluatorCallbackConfig`` on
``DataMix.v3_small_ppl_validation``) for checkpoints that were trained without it: the same
padded fixed-sequence-length dataset (one instance per validation document, truncated to the
sequence length, padded with the tokenizer's pad id), the same ``LMEvaluator`` per-set metric
(mean per-token CE over the label-masked positions, PPL = exp(CE)), the same ``get_labels``
shift. Validation sets: c4_en, dolma_{books,common-crawl,pes2o,reddit,stack,wiki}, ice,
m2d2_s2orc, pile, wikitext_103 (``eval-data/perplexity/v3_small_dolma2-tokenizer/<set>/val``
under the data root).

Model loading follows scripts/sparse_experts/olmoe3_routing/extract_routing.py (bf16 weights,
flat OLMoDDP ``.main`` optimizer tensors via the pinned HF-converter loader, bf16 autocast).

    PYTHONPATH=external/OLMo-core/src python scripts/debug_validation/eval_ppl_validation.py \
        --checkpoints /weka/.../olmoe3_275m_10b/step5000 [...] --out-dir claude_outputs/debug_validation/ppl_validation

Writes ``<out-dir>/<run>/step<N>.json`` per checkpoint (skips existing). Two CE variants are
stored per set: ``CE loss`` = OLMo-core's in-loop number (label_mask positions, which include
each document's last real token whose shifted label is ignored -> contributes 0, exactly as
in-loop), and ``CE loss (labeled tokens)`` = mean over positions with a real label only.
``--dummy-model`` exercises the data/metric path on CPU without a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sparse_experts" / "olmoe3_routing"))
# The parallel instance-index prep forks a process pool that shares the parent's S3 client; with
# eleven files it occasionally returns another file's size (seen once: out-of-bounds error). The
# files are tiny, so prepare them serially unless the caller says otherwise.
os.environ.setdefault("OLMO_DATA_PREP_WORKERS", "1")

DEFAULT_DATA_ROOT = "s3://ai2-llm"
DEFAULT_WORK_DIR = "/weka/oe-training-default/ryanwang/dataset-cache"
SEQUENCE_LENGTH = 8192  # the OLMoE3 275M runs' training sequence length


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_evaluator(
    data_root: str, work_dir: str, sequence_length: int, batch_instances: int, device
):
    from olmo_core.data import DataCollator, DataMix, NumpyPaddedFSLDatasetConfig, TokenizerConfig
    from olmo_core.eval import LMEvaluator

    tokenizer = TokenizerConfig.dolma2()
    dataset = NumpyPaddedFSLDatasetConfig.from_data_mix(
        DataMix.v3_small_ppl_validation,
        mix_base_dir=data_root,
        sequence_length=sequence_length,
        tokenizer=tokenizer,
        work_dir=work_dir,
    ).build()
    evaluator = LMEvaluator.from_numpy_dataset(
        dataset,
        name="lm",
        global_batch_size=batch_instances * sequence_length,
        collator=DataCollator(pad_token_id=tokenizer.pad_token_id),
        device=device,
    )
    return evaluator, dataset


class DummyModel(torch.nn.Module):
    """CE = 1 at every position with a real label (so per-set 'labeled' CE must be exactly 1)."""

    def forward(self, input_ids, labels=None, loss_reduction="none", return_logits=False):
        from olmo_core.nn.lm_head import LMOutputWithLoss

        ce = (labels != -100).float()
        return LMOutputWithLoss(logits=None, loss=ce.mean(), ce_loss=ce, z_loss=None)


def evaluate(model, evaluator, device, log_every: int = 20) -> Dict[str, object]:
    from olmo_core.data.utils import get_labels
    from olmo_core.utils import move_to_device

    evaluator.reset_metrics()
    labeled_sum: Dict[str, float] = {}
    labeled_n: Dict[str, int] = {}
    masked_n: Dict[str, int] = {}
    n_batches = 0
    t0 = time.time()
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast("cpu", enabled=False)
    )
    with torch.no_grad(), autocast:
        for batch in evaluator:
            n_batches += 1
            batch = move_to_device(batch, device)
            labels = get_labels(batch)
            out = model(
                batch["input_ids"], labels=labels, loss_reduction="none", return_logits=False
            )
            ce = out.ce_loss.float()
            assert ce.shape == batch["input_ids"].shape, ce.shape
            evaluator.update_metrics(batch, ce, None)
            for i, md in enumerate(batch["metadata"]):
                lab = md["label"]
                valid = labels[i] != -100
                labeled_sum[lab] = labeled_sum.get(lab, 0.0) + float(ce[i][valid].sum())
                labeled_n[lab] = labeled_n.get(lab, 0) + int(valid.sum())
                masked_n[lab] = masked_n.get(lab, 0) + int(batch["label_mask"][i].sum())
            if n_batches % log_every == 0 or n_batches == evaluator.total_batches:
                log(f"  batch {n_batches}/{evaluator.total_batches}, {time.time() - t0:.0f}s")
    metrics = {k: float(v) for k, v in evaluator.compute_metrics().items()}
    sets = sorted(labeled_n)
    per_set = {}
    for lab in sets:
        per_set[lab] = {
            "CE loss": metrics[f"{lab}/CE loss"],
            "PPL": metrics[f"{lab}/PPL"],
            "CE loss (labeled tokens)": labeled_sum[lab] / max(labeled_n[lab], 1),
            "labeled tokens": labeled_n[lab],
            "label_mask tokens": masked_n[lab],
        }
    return {"per_set": per_set, "n_batches": n_batches, "elapsed_s": time.time() - t0}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--checkpoints",
        nargs="*",
        type=Path,
        default=[],
        help="step dirs (config.json + model_and_optim)",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    ap.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    ap.add_argument("--batch-size", type=int, default=8, help="instances per forward")
    ap.add_argument(
        "--attn-backend", default=None, help="override the checkpoint's attention backend"
    )
    ap.add_argument("--use-cute-kda", action="store_true")
    ap.add_argument(
        "--max-first-ce", type=float, default=6.0, help="guard against mis-loaded weights"
    )
    ap.add_argument(
        "--dummy-model", action="store_true", help="CPU self-test of the data/metric path"
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu" if args.dummy_model else "cuda")
    evaluator, dataset = build_evaluator(
        args.data_root, args.work_dir, args.sequence_length, args.batch_size, device
    )
    log(
        f"validation mix: {len(dataset):,} instances x {args.sequence_length} tokens from {len(dataset.paths)} sets; "
        f"{evaluator.total_batches} batches of {args.batch_size}"
    )

    if args.dummy_model:
        res = evaluate(DummyModel(), evaluator, device)
        for lab, m in res["per_set"].items():
            assert abs(m["CE loss (labeled tokens)"] - 1.0) < 1e-6, (lab, m)
            # in-loop metric averages over label_mask positions (one more per document than labels)
            assert m["label_mask tokens"] >= m["labeled tokens"], (lab, m)
            log(
                f"  {lab}: instances={m['label_mask tokens'] - m['labeled tokens']} labeled={m['labeled tokens']:,} in-loop CE={m['CE loss']:.4f}"
            )
        log("dummy-model self-test OK")
        return

    from extract_routing import build_model

    for ckpt in args.checkpoints:
        ckpt = ckpt.resolve()
        run, step = ckpt.parent.name, ckpt.name
        out = args.out_dir / run / f"{step}.json"
        if out.exists() and not args.overwrite:
            log(f"skip {run}/{step} (exists)")
            continue
        log(f"loading {ckpt}")
        model, _, mcfg = build_model(ckpt, device, args.attn_backend, args.use_cute_kda)
        res = evaluate(model, evaluator, device)
        ces = {lab: m["CE loss"] for lab, m in res["per_set"].items()}
        worst = max(ces.values())
        assert (
            worst < args.max_first_ce
        ), f"{run}/{step}: max per-set CE {worst:.3f} > {args.max_first_ce}: weights probably mis-loaded"
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(
            {
                "checkpoint": str(ckpt),
                "run": run,
                "step": int(step.replace("step", "")),
                "num_experts": mcfg["block"]["routed_experts"]["num_experts"],
                "emo": mcfg["block"]["routed_experts_router"].get("emo") is not None,
                "sequence_length": args.sequence_length,
                "data_root": args.data_root,
                "mix": "v3-small-ppl-validation",
                **res,
            },
            open(out, "w"),
            indent=1,
        )
        log(
            f"{run}/{step}: "
            + ", ".join(
                f"{lab.replace('-validation', '')}={ce:.3f}" for lab, ce in sorted(ces.items())
            )
        )
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
