#!/usr/bin/env python3
"""
Extract the exact set of WHOLE documents a training run consumes in a token window.

Given a checkpoint of a run (used only for its dataset config + data-loader state) and a
token window [start_tokens, end_tokens), this reconstructs the run's deterministic data
order, finds every instance the run consumes in that window, and reconstructs every
document those instances touch — whole (any-overlap superset semantics: instances that
straddle a document boundary pull in the complete document, stitched from neighboring
dataset positions outside the window).

Correctness notes (verified against olmo_core.data):
  - Step/position -> instance mapping is `global_indices[:total_size][pos]`, a pure
    chunked shuffle: `get_rng(seed + epoch)` permutation of chunk indices, expanded.
  - CRITICAL: when the dataset has `max_target_sequence_length` (8192 for the models_v2
    runs), the trainer's loader uses chunk_size = max_target_sequence_length //
    sequence_length (= 2), NOT 1 (see NumpyDataLoaderBase.wrap_numpy_dataset). We set it
    the same way, and with the training run's own work-dir we reuse the exact
    global-indices file the run itself wrote (e.g. global_indices_chunk2_..._seed0_v1.npy).
  - Per-file instance counts truncate at a max_target_sequence_length multiple
    (NumpyFSLDataset._get_file_size_and_length), so instance -> (file, offset) mapping
    goes through dataset.offsets, never recomputed by hand.
  - instance_filter_config only ANNOTATES instances (instance_mask zeroes their loss);
    it never drops or reorders them. We optionally recompute the mask as provenance.
  - Documents never straddle files; EOS (dolma2: 100257) separates documents within a
    file. File start/end also act as document boundaries.

Sharding is BY FILE (shard i takes files i::num_shards), so dedup by
(source_path, doc_start_offset) is complete within a shard.

Usage (validate on one step's worth of instances):
    PYTHONPATH=src python scripts/modular_extension/extract_training_doc_window.py \
        --checkpoint models_v2/emo_64exp_50b_wsd_lr2e-3/step11921 \
        --start-tokens 100e9 --end-tokens 110e9 --limit 1024 \
        --output-dir modular_extension/data/validation

Full 10B window, 16 shards (run one process per shard):
    ... --shard $i --num-shards 16 --output-dir modular_extension/data/<name>
"""

import argparse
import gzip
import json
import logging
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from olmo_core.data import DataCollator, NumpyFSLDataLoader, NumpyFSLDatasetConfig
from olmo_core.data.numpy_dataset import NumpyFSLDataset
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.data.utils import find_periodic_sequences, load_array_slice
from olmo_core.io import normalize_path

from scripts.dump_training_batch import (  # noqa: E402  (src/scripts via PYTHONPATH)
    load_checkpoint_config,
    load_data_paths,
    load_trainer_state,
    verify_paths_match_mix,
)

log = logging.getLogger("extract_training_doc_window")

DEFAULT_CHECKPOINT = str(REPO_ROOT / "models_v2/emo_64exp_50b_wsd_lr2e-3/step11921")
DEFAULT_WORK_DIR = os.path.expanduser("~/dataset-cache")


def build_dataset_and_loader(checkpoint_dir: str, work_dir: str):
    """Reconstruct the dataset + loader exactly as the training run built them."""
    checkpoint_dir = normalize_path(checkpoint_dir)
    config = load_checkpoint_config(checkpoint_dir)
    data_paths = load_data_paths(checkpoint_dir)
    trainer_state = load_trainer_state(checkpoint_dir)
    data_loader_state = trainer_state["data_loader"]

    if data_loader_state["dataset_type"] != "fsl":
        raise RuntimeError(f"Only FSL datasets supported, got {data_loader_state['dataset_type']}")

    dataset_config_dict = config["dataset"]
    data_loader_config_dict = config["data_loader"]

    tokenizer_config = TokenizerConfig(
        **{k: v for k, v in dataset_config_dict["tokenizer"].items() if k != "_CLASS_"}
    )
    if not verify_paths_match_mix(
        data_paths,
        dataset_config_dict["mix"],
        tokenizer_config,
        dataset_config_dict.get("mix_base_dir", "gs://ai2-llm"),
    ):
        raise RuntimeError("data_paths.txt does not match the mix from config.json")

    config_fields = dict(dataset_config_dict)
    config_fields["work_dir"] = work_dir
    if isinstance(config_fields.get("tokenizer"), dict):
        config_fields["tokenizer"] = {
            k: v for k, v in config_fields["tokenizer"].items() if k != "_CLASS_"
        }
    config_fields.pop("name", None)
    config_fields.pop("_CLASS_", None)

    dataset_config = NumpyFSLDatasetConfig.from_dict(config_fields)
    dataset = dataset_config.build()
    # NumpyFSLDatasetMixture has different offset semantics; only the plain class is supported.
    assert type(dataset) is NumpyFSLDataset, f"Expected NumpyFSLDataset, got {type(dataset)}"
    dataset.prepare()

    if dataset.fingerprint != data_loader_state["dataset_fingerprint"]:
        raise RuntimeError(
            f"Dataset fingerprint mismatch! checkpoint={data_loader_state['dataset_fingerprint']} "
            f"computed={dataset.fingerprint}"
        )
    log.info(f"Dataset fingerprint verified: {dataset.fingerprint}")

    data_loader = NumpyFSLDataLoader(
        dataset,
        collator=DataCollator(pad_token_id=dataset.pad_token_id),
        global_batch_size=data_loader_config_dict["global_batch_size"],
        work_dir=work_dir,
        seed=data_loader_state["seed"],
        shuffle=True,
        dp_world_size=1,
        dp_rank=0,
        num_threads=10,
    )
    # CRITICAL: match NumpyDataLoaderBase.wrap_numpy_dataset (chunk_size=2 for these runs).
    if dataset.max_target_sequence_length is not None:
        data_loader.chunk_size = dataset.max_target_sequence_length // dataset.sequence_length
    log.info(f"Loader chunk_size={data_loader.chunk_size}")

    epoch = data_loader_state["epoch"]
    # in_memory=False -> reuse (or write) the on-disk global-indices file. With the training
    # run's own work-dir this memmaps the EXACT file the run consumed.
    data_loader._epoch = epoch
    idx_file = data_loader._global_indices_file
    if idx_file.is_file():
        log.info(f"Reusing existing global-indices file (written by the training run): {idx_file}")
    else:
        log.warning(f"Global-indices file not found, building fresh: {idx_file}")
    data_loader.reshuffle(epoch=epoch, in_memory=False)

    meta = {
        "checkpoint": checkpoint_dir,
        "run_name": config.get("run_name"),
        "dataset_fingerprint": dataset.fingerprint,
        "seed": data_loader_state["seed"],
        "epoch": epoch,
        "chunk_size": data_loader.chunk_size,
        "sequence_length": dataset.sequence_length,
        "global_batch_size": data_loader_config_dict["global_batch_size"],
        "eos_token_id": dataset.eos_token_id,
        "instance_filter_config": dataset_config_dict.get("instance_filter_config"),
        "global_indices_file": str(idx_file),
    }
    return dataset, data_loader, meta


def coalesce_runs(locals_sorted: np.ndarray, merge_gap: int) -> List[Tuple[int, int]]:
    """Coalesce sorted local instance indices into inclusive [lo, hi] runs, merging runs
    separated by <= merge_gap untouched instances (one bigger read beats two requests)."""
    runs: List[Tuple[int, int]] = []
    lo = hi = int(locals_sorted[0])
    for v in locals_sorted[1:]:
        v = int(v)
        if v <= hi + 1 + merge_gap:
            hi = v
        else:
            runs.append((lo, hi))
            lo = hi = v
    runs.append((lo, hi))
    return runs


class RunProcessor:
    """Processes one coalesced run of touched instances in one source file."""

    def __init__(
        self,
        path: str,
        file_token_len: int,
        seq_len: int,
        eos_id: int,
        dtype,
        pad_tokens: int,
        max_scan_tokens: int,
        stats: Dict[str, int],
        stats_lock: threading.Lock,
    ):
        self.path = path
        self.file_token_len = file_token_len
        self.S = seq_len
        self.eos = eos_id
        self.dtype = dtype
        self.pad = pad_tokens
        self.max_scan = max_scan_tokens
        self.stats = stats
        self.lock = stats_lock

    def _read(self, start: int, end: int) -> np.ndarray:
        arr = load_array_slice(self.path, start, end, self.dtype)
        with self.lock:
            self.stats["requests"] += 1
            self.stats["bytes_read"] += arr.nbytes
        return arr

    def __call__(self, run: Tuple[int, int, List[Tuple[int, int]]]) -> Dict[str, Any]:
        """run = (lo, hi, touched) where touched = [(local_index, consumption_position), ...]
        sorted by local_index; lo/hi are inclusive local instance indices."""
        lo, hi, touched = run
        S = self.S
        T0 = lo * S
        T1 = min((hi + 1) * S, self.file_token_len)
        base = max(0, T0 - self.pad)
        end = min(self.file_token_len, T1 + self.pad)
        buf = self._read(base, end)

        truncated_left = truncated_right = False

        # Left document boundary: latest EOS at absolute pos < T0 (doc starts one past it).
        left_region = buf[: T0 - base]
        eos_left = np.flatnonzero(left_region == self.eos)
        ext = self.pad
        while eos_left.size == 0 and base > 0:
            if T0 - base >= self.max_scan:
                truncated_left = True
                break
            new_base = max(0, base - ext)
            chunk = self._read(new_base, base)
            with self.lock:
                self.stats["ext_requests"] += 1
            buf = np.concatenate([chunk, buf])
            base = new_base
            ext *= 2
            eos_left = np.flatnonzero(buf[: T0 - base] == self.eos)
        if eos_left.size > 0:
            span_start = base + int(eos_left[-1]) + 1  # first token of leftmost overlapping doc
        elif base == 0 and not truncated_left:
            span_start = 0
        else:  # truncated
            span_start = base

        # Right document boundary: first EOS at absolute pos >= T1 - 1 (closes the doc
        # containing/ending at the last touched token).
        rel_from = (T1 - 1) - base
        eos_right = np.flatnonzero(buf[rel_from:] == self.eos)
        cur_end = base + len(buf)
        ext = self.pad
        while eos_right.size == 0 and cur_end < self.file_token_len:
            if cur_end - T1 >= self.max_scan:
                truncated_right = True
                break
            new_end = min(self.file_token_len, cur_end + ext)
            chunk = self._read(cur_end, new_end)
            with self.lock:
                self.stats["ext_requests"] += 1
            buf = np.concatenate([buf, chunk])
            cur_end = new_end
            ext *= 2
            eos_right = np.flatnonzero(buf[rel_from:] == self.eos)
        if eos_right.size > 0:
            span_end = base + rel_from + int(eos_right[0])  # abs index of closing EOS
            span_has_final_eos = True
        else:  # file end (unterminated final doc) or truncated
            span_end = min(cur_end, self.file_token_len) - 1
            span_has_final_eos = False

        # Split [span_start, span_end] on EOS into whole documents.
        span = buf[span_start - base : span_end + 1 - base]
        eos_positions = np.flatnonzero(span == self.eos)  # relative to span_start
        docs = []
        prev = -1  # relative index of previous EOS
        boundaries = [int(e) for e in eos_positions]
        if not span_has_final_eos:
            boundaries.append(len(span))  # virtual terminator at file end / scan cap
        touched_locals = [t[0] for t in touched]
        touched_positions = [t[1] for t in touched]
        for e in boundaries:
            s_rel, e_rel = prev + 1, e  # content = span[s_rel:e_rel], separator at e_rel
            prev = e
            if e_rel <= s_rel:
                continue  # empty doc (consecutive EOS)
            s_abs = span_start + s_rel
            # doc-with-separator covers [s_abs, sep_end_abs)
            sep_end_abs = span_start + e_rel + (1 if e < len(span) else 0)
            # Which touched instances overlap [s_abs, sep_end_abs)?
            i_min = s_abs // self.S
            i_max_excl = math.ceil(sep_end_abs / self.S) if sep_end_abs > s_abs else i_min
            a = bisect_left(touched_locals, i_min)
            b = bisect_right(touched_locals, i_max_excl - 1)
            if b <= a:
                with self.lock:
                    self.stats["empty_gap_docs_skipped"] += 1
                continue  # doc lies entirely in a merged gap -> not touched by the window
            doc = {
                "doc_start_offset": int(s_abs),
                "token_ids": span[s_rel:e_rel].tolist(),
                "touch_positions": touched_positions[a:b],
                "touch_locals": touched_locals[a:b],
            }
            if truncated_left and s_abs == span_start and span_start != 0 and eos_left.size == 0:
                doc["truncated_left"] = True
            if truncated_right and e == len(span) and not span_has_final_eos:
                # only truly truncated if we gave up before file end
                if span_start + len(span) < self.file_token_len:
                    doc["truncated_right"] = True
            docs.append(doc)

        # Instance slices (for --dump-instances / instance-mask computation).
        instance_tokens = {}
        for li in touched_locals:
            a_tok, b_tok = li * S, (li + 1) * S
            if a_tok >= base and b_tok <= base + len(buf):
                instance_tokens[li] = buf[a_tok - base : b_tok - base]
        return {"docs": docs, "instance_tokens": instance_tokens, "touched": touched}


def compute_instance_mask(tokens: np.ndarray, filt: Dict[str, int]) -> bool:
    """Mirror NumpyDatasetBase._validate_instance: True = kept, False = loss-masked."""
    for m in find_periodic_sequences(
        tokens.astype(np.int64),
        max_period=filt["repetition_max_period"],
        min_period=filt["repetition_min_period"],
    ):
        if m.times >= filt["repetition_max_count"]:
            return False
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--start-tokens", type=float, required=True, help="window start (e.g. 100e9)")
    p.add_argument("--end-tokens", type=float, required=True, help="window end, exclusive (e.g. 110e9)")
    p.add_argument("--work-dir", default=DEFAULT_WORK_DIR, help="dataset cache (training run's own)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--pad-tokens", type=int, default=8192, help="padding per side of each run read")
    p.add_argument("--merge-gap", type=int, default=4, help="merge runs <= this many instances apart")
    p.add_argument("--max-scan-tokens", type=int, default=524288, help="EOS scan cap per side")
    p.add_argument("--limit", type=int, default=None, help="only the first N window instances (validation)")
    p.add_argument("--instance-mask", action=argparse.BooleanOptionalAction, default=True,
                   help="recompute the training-time instance_mask as provenance")
    p.add_argument("--dump-instances", default=None,
                   help="also dump raw touched instances to this .npz (positions + 4096-token rows)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [shard {}] %(message)s".format(args.shard),
    )
    t_start = time.monotonic()

    dataset, loader, meta = build_dataset_and_loader(args.checkpoint, args.work_dir)
    S = dataset.sequence_length
    eos_id = dataset.eos_token_id
    dtype = dataset.dtype
    item_size = dtype(0).itemsize

    pos_start = int(args.start_tokens) // S
    pos_end = math.ceil(int(args.end_tokens) / S)
    total_size = loader.total_size  # instances per epoch actually consumed
    if pos_end > total_size:
        raise RuntimeError(
            f"Window end position {pos_end} exceeds epoch size {total_size} instances; "
            f"multi-epoch windows are not supported (would need per-epoch reshuffle handling)."
        )
    gi = loader.get_global_indices()
    window_ids = np.asarray(gi[pos_start:pos_end], dtype=np.int64)
    positions = np.arange(pos_start, pos_end, dtype=np.int64)
    if args.limit is not None:
        window_ids = window_ids[: args.limit]
        positions = positions[: args.limit]
    log.info(
        f"Window: tokens [{int(args.start_tokens):,}, {int(args.end_tokens):,}) -> "
        f"positions [{pos_start:,}, {pos_end:,}) -> {len(window_ids):,} instances"
        + (f" (limited to {args.limit})" if args.limit else "")
    )

    # Map instance ids -> (file index, local index) via dataset.offsets.
    offsets = dataset.offsets
    starts = np.array([s for s, _ in offsets], dtype=np.int64)
    ends = np.array([e for _, e in offsets], dtype=np.int64)
    file_idx = np.searchsorted(starts, window_ids, side="right") - 1
    assert (window_ids < ends[file_idx]).all() and (window_ids >= starts[file_idx]).all()
    local_idx = window_ids - starts[file_idx]

    # This shard's files.
    shard_mask = (file_idx % args.num_shards) == args.shard
    file_idx, local_idx, positions = file_idx[shard_mask], local_idx[shard_mask], positions[shard_mask]
    shard_files = sorted(set(file_idx.tolist()))
    log.info(f"Shard {args.shard}/{args.num_shards}: {len(file_idx):,} instances across {len(shard_files)} files")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_path = out_dir / f"docs-{args.shard:03d}.jsonl.gz"

    stats: Dict[str, int] = {
        "requests": 0, "ext_requests": 0, "bytes_read": 0,
        "instances": int(len(file_idx)), "runs": 0, "docs": 0, "doc_tokens": 0,
        "masked_instances": 0, "truncated_docs": 0, "empty_gap_docs_skipped": 0,
    }
    stats_lock = threading.Lock()
    filt = meta.get("instance_filter_config")
    dump_positions: List[int] = []
    dump_rows: List[np.ndarray] = []

    n_files_done = 0
    with gzip.open(docs_path, "wt", compresslevel=4) as out_f, \
            ThreadPoolExecutor(max_workers=args.threads) as pool:
        for fi in shard_files:
            m = file_idx == fi
            locs = local_idx[m]
            poss = positions[m]
            order = np.argsort(locs)
            locs, poss = locs[order], poss[order]
            path = str(dataset.paths[fi])
            file_token_len = dataset.file_sizes[fi] // item_size

            runs = coalesce_runs(locs, args.merge_gap)
            # attach touched (local, position) pairs to each run
            run_args = []
            li = 0
            for lo, hi in runs:
                touched = []
                while li < len(locs) and locs[li] <= hi:
                    touched.append((int(locs[li]), int(poss[li])))
                    li += 1
                run_args.append((lo, hi, touched))
            stats["runs"] += len(run_args)

            proc = RunProcessor(
                path, file_token_len, S, eos_id, dtype,
                args.pad_tokens, args.max_scan_tokens, stats, stats_lock,
            )
            by_start: Dict[int, Dict[str, Any]] = {}
            mask_by_local: Dict[int, bool] = {}
            for result in pool.map(proc, run_args):
                for doc in result["docs"]:
                    key = doc["doc_start_offset"]
                    if key in by_start:  # doc spanning two runs: merge provenance
                        prev_doc = by_start[key]
                        prev_doc["touch_positions"] = sorted(
                            set(prev_doc["touch_positions"]) | set(doc["touch_positions"]))
                        prev_doc["touch_locals"] = sorted(
                            set(prev_doc["touch_locals"]) | set(doc["touch_locals"]))
                    else:
                        by_start[key] = doc
                if args.instance_mask and filt:
                    for li_, toks in result["instance_tokens"].items():
                        if li_ not in mask_by_local:
                            mask_by_local[li_] = compute_instance_mask(toks, filt)
                if args.dump_instances:
                    for (li_, pos_) in result["touched"]:
                        toks = result["instance_tokens"].get(li_)
                        if toks is not None:
                            dump_positions.append(pos_)
                            dump_rows.append(np.asarray(toks, dtype=np.int64))

            local_to_pos = {int(l): int(pp) for l, pp in zip(locs, poss)}
            for key in sorted(by_start):
                doc = by_start[key]
                rec = {
                    "source_path": path,
                    "doc_start_offset": doc["doc_start_offset"],
                    "doc_len": len(doc["token_ids"]),
                    "n_touching_instances": len(doc["touch_locals"]),
                    "first_touch_position": min(
                        local_to_pos[l] for l in doc["touch_locals"]),
                    "touch_positions": sorted(doc["touch_positions"]),
                }
                if args.instance_mask and filt:
                    filtered = [local_to_pos[l] for l in doc["touch_locals"]
                                if not mask_by_local.get(l, True)]
                    if filtered:
                        rec["filtered_touch_positions"] = sorted(filtered)
                for flag in ("truncated_left", "truncated_right"):
                    if doc.get(flag):
                        rec[flag] = True
                        stats["truncated_docs"] += 1
                rec["token_ids"] = doc["token_ids"]
                out_f.write(json.dumps(rec) + "\n")
                stats["docs"] += 1
                stats["doc_tokens"] += rec["doc_len"]
            if args.instance_mask and filt:
                stats["masked_instances"] += sum(1 for v in mask_by_local.values() if not v)

            n_files_done += 1
            elapsed = time.monotonic() - t_start
            done_frac = stats["runs"] and n_files_done / len(shard_files)
            log.info(
                f"file {n_files_done}/{len(shard_files)} ({Path(path).name}): "
                f"{stats['docs']:,} docs, {stats['doc_tokens']:,} doc-tokens, "
                f"{stats['requests']:,} reqs ({stats['requests']/elapsed:.0f}/s), "
                f"{stats['bytes_read']/1e9:.1f}GB, ETA {elapsed*(1/done_frac - 1)/60:.0f} min"
            )

    if args.dump_instances and dump_rows:
        order = np.argsort(dump_positions)
        np.savez(
            args.dump_instances,
            positions=np.asarray(dump_positions, dtype=np.int64)[order],
            input_ids=np.stack(dump_rows)[order],
        )
        log.info(f"Dumped {len(dump_rows)} raw instances to {args.dump_instances}")

    elapsed = time.monotonic() - t_start
    manifest = {
        **meta,
        "window_start_tokens": int(args.start_tokens),
        "window_end_tokens": int(args.end_tokens),
        "pos_start": pos_start,
        "pos_end": pos_end,
        "limit": args.limit,
        "shard": args.shard,
        "num_shards": args.num_shards,
        "pad_tokens": args.pad_tokens,
        "merge_gap": args.merge_gap,
        "stats": stats,
        "elapsed_seconds": round(elapsed, 1),
        "docs_file": str(docs_path),
    }
    manifest_path = out_dir / f"manifest-{args.shard:03d}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info(
        f"DONE in {elapsed/60:.1f} min: {stats['docs']:,} docs, {stats['doc_tokens']:,} doc-tokens "
        f"({stats['doc_tokens']/max(1, stats['instances']*S):.3f}x window tokens), "
        f"{stats['requests']:,} requests, {stats['bytes_read']/1e9:.1f}GB read. "
        f"Manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()
