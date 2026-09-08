#!/usr/bin/env python3
"""Extract the exact training-stream instances for a step range of an olmoe3_275m run.

Rebuilds the run's dataset + data loader from the checkpoint's config.json / train state with the
PINNED OLMo-core (PYTHONPATH=external/OLMo-core/src), exactly like
external/OLMo-core/src/scripts/dump_training_batch.py, then pulls `data_loader[step]` for every
step in [--step-start, --step-end) and writes the raw 8192-token instances.

Correctness guards (all fatal):
  * dataset fingerprint must equal the one stored in the checkpoint's train state;
  * the global data-order index file that the RUN wrote must already exist in the work dir and
    must not be rewritten (we only ever read it);
  * every batch must have exactly global_batch_size / seq_len instances of seq_len tokens.

Output (--out-dir), one shard per worker process, plus a manifest:
  shard_<k>.tokens.npy   int32 memmap (n_inst, 8192) token ids
  shard_<k>.meta.npz     step, slot, dataset_index, label_id, instance_mask (bool; False = the run's
                         repetition filter REJECTED it) per instance
  labels.json            label_id -> mix label (source)
  manifest.json          steps covered, counts, fingerprint

Usage (from repo root, emo env):
  PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_routing/extract_stream.py \
      --checkpoint sparse_experts/olmoe3_275m_10b/step19074 --step-start 19074 --step-end 38148 \
      --out-dir sparse_experts/olmoe3_routing/stream_10b_20b --workers 32
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger("extract_stream")

WEKA_PREFIX = "/weka/oe-training-default/ryanwang"


def _local(path: str) -> str:
    """Weka paths from the run config resolve to ~ in a GPU-attached session."""
    if path.startswith(WEKA_PREFIX) and not os.path.exists(path):
        return os.path.expanduser("~" + path[len(WEKA_PREFIX):])
    return path


def build_loader(checkpoint: Path, work_dir: str, num_threads: int):
    from olmo_core.data import DataCollator, NumpyFSLDataLoader, NumpyFSLDatasetConfig
    from olmo_core.data.numpy_dataset import NumpyFSLDatasetBase

    config = json.load(open(checkpoint / "config.json"))
    state = torch.load(checkpoint / "train" / "rank0.pt", map_location="cpu", weights_only=False)
    dl_state = state["data_loader"]
    assert dl_state["dataset_type"] == "fsl", dl_state["dataset_type"]

    fields = dict(config["dataset"])
    fields.pop("_CLASS_", None)
    fields.pop("name", None)
    fields["work_dir"] = work_dir
    fields["tokenizer"] = {k: v for k, v in fields["tokenizer"].items() if k != "_CLASS_"}
    ds_cfg = NumpyFSLDatasetConfig.from_dict(fields)
    dataset = ds_cfg.build()
    assert isinstance(dataset, NumpyFSLDatasetBase)
    dataset.prepare()
    if dataset.fingerprint != dl_state["dataset_fingerprint"]:
        raise RuntimeError(
            f"dataset fingerprint mismatch: checkpoint={dl_state['dataset_fingerprint']} "
            f"rebuilt={dataset.fingerprint}"
        )
    gbs = config["data_loader"]["global_batch_size"]
    loader = NumpyFSLDataLoader(
        dataset,
        collator=DataCollator(pad_token_id=dataset.pad_token_id),
        global_batch_size=gbs,
        work_dir=work_dir,
        seed=dl_state["seed"],
        shuffle=True,
        dp_world_size=1,
        dp_rank=0,
        num_threads=num_threads,
    )
    loader._epoch = dl_state["epoch"]  # needed to resolve the index filename before reshuffle()
    gi_file = loader._global_indices_file
    if not gi_file.exists():
        raise RuntimeError(f"global indices file written by the run is missing: {gi_file}")
    mtime = gi_file.stat().st_mtime
    loader.reshuffle(epoch=dl_state["epoch"], in_memory=False)
    if gi_file.stat().st_mtime != mtime:
        raise RuntimeError(f"global indices file was rewritten: {gi_file}")
    seq_len = dl_state["sequence_length"]
    return loader, dataset, gbs // seq_len, seq_len, dl_state


def worker(args_tuple):
    (k, step_start, step_end, checkpoint, work_dir, out_dir, num_threads, log_every) = args_tuple
    logging.basicConfig(level=logging.INFO, format=f"[shard {k}] %(asctime)s %(message)s")
    loader, dataset, inst_per_step, seq_len, _ = build_loader(Path(checkpoint), work_dir, num_threads)
    labels = {}
    n = (step_end - step_start) * inst_per_step
    tok = np.lib.format.open_memmap(
        Path(out_dir) / f"shard_{k}.tokens.npy", mode="w+", dtype=np.int32, shape=(n, seq_len)
    )
    step_a = np.empty(n, np.int32); slot_a = np.empty(n, np.int16)
    idx_a = np.empty(n, np.int64); lab_a = np.empty(n, np.int16); mask_a = np.ones(n, bool)
    p = 0; t0 = time.time()
    for i, step in enumerate(range(step_start, step_end)):
        batch = loader[step]
        ids = batch["input_ids"]
        if ids.shape != (inst_per_step, seq_len):
            raise RuntimeError(f"step {step}: unexpected batch shape {tuple(ids.shape)}")
        tok[p : p + inst_per_step] = ids.numpy().astype(np.int32)
        step_a[p : p + inst_per_step] = step
        slot_a[p : p + inst_per_step] = np.arange(inst_per_step)
        idx_a[p : p + inst_per_step] = batch["index"].numpy()
        if "instance_mask" in batch:
            mask_a[p : p + inst_per_step] = batch["instance_mask"].numpy()
        metas = batch.get("metadata") or [{}] * inst_per_step
        for j, m in enumerate(metas):
            lab = m.get("label", "unknown")
            lab_a[p + j] = labels.setdefault(lab, len(labels))
        p += inst_per_step
        if (i + 1) % log_every == 0:
            rate = (i + 1) / (time.time() - t0)
            log.info(f"{i+1}/{step_end-step_start} steps, {rate*60:.1f} steps/min, eta {((step_end-step_start)-(i+1))/rate/60:.1f} min")
    tok.flush(); del tok
    np.savez(Path(out_dir) / f"shard_{k}.meta.npz", step=step_a, slot=slot_a, dataset_index=idx_a,
             label_id=lab_a, instance_mask=mask_a)
    json.dump(labels, open(Path(out_dir) / f"shard_{k}.labels.json", "w"))
    return k, n, labels


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--step-start", type=int, required=True)
    ap.add_argument("--step-end", type=int, required=True, help="exclusive")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--num-threads", type=int, default=8, help="S3 reader threads per worker")
    ap.add_argument("--work-dir", default=None, help="dataset cache (default: the run's, weka->~ mapped)")
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    config = json.load(open(args.checkpoint / "config.json"))
    work_dir = args.work_dir or _local(config["dataset"]["work_dir"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    total = args.step_end - args.step_start
    per = -(-total // args.workers)
    ranges = [(args.step_start + k * per, min(args.step_end, args.step_start + (k + 1) * per)) for k in range(args.workers)]
    ranges = [(a, b) for a, b in ranges if b > a]
    log.info(f"{total} steps over {len(ranges)} workers ({per} steps each); work_dir={work_dir}")
    jobs = [(k, a, b, str(args.checkpoint), work_dir, str(args.out_dir), args.num_threads, args.log_every) for k, (a, b) in enumerate(ranges)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(len(jobs)) as pool:
        results = pool.map(worker, jobs)
    # unify labels
    all_labels = {}
    for k, n, labels in results:
        for lab in labels: all_labels.setdefault(lab, len(all_labels))
    for k, n, labels in results:
        remap = np.array([all_labels[l] for l, _ in sorted(labels.items(), key=lambda kv: kv[1])], np.int16)
        m = dict(np.load(args.out_dir / f"shard_{k}.meta.npz"))
        m["label_id"] = remap[m["label_id"]]
        np.savez(args.out_dir / f"shard_{k}.meta.npz", **m)
        (args.out_dir / f"shard_{k}.labels.json").unlink()
    json.dump(all_labels, open(args.out_dir / "labels.json", "w"), indent=1)
    json.dump({"checkpoint": str(args.checkpoint), "step_start": args.step_start, "step_end": args.step_end,
               "shards": [{"k": k, "steps": [a, b], "n_instances": n} for (k, n, _), (a, b) in zip(results, ranges)],
               "n_instances": sum(n for _, n, _ in results), "seq_len": config["dataset"]["sequence_length"],
               "global_batch_size": config["data_loader"]["global_batch_size"]},
              open(args.out_dir / "manifest.json", "w"), indent=1)
    log.info(f"done: {sum(n for _, n, _ in results):,} instances in {len(results)} shards -> {args.out_dir}")


if __name__ == "__main__":
    main()
