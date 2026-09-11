#!/usr/bin/env python3
"""Build a resumable checkpoint dir for post-merge finetuning (olmoe3_squares).

Combines a model(+optimizer) DCP dir (a merged model, or any step dir) with the TRAINER state of a
finished baseline checkpoint, rewritten so the run resumes at --start-step: global_step, tokens seen
and the data-loader position are set to that step, so training continues on the training stream
from that point (past the window the held-out sample was drawn from) at constant LR with no
warmup. Output: <out>/step<S>/{model_and_optim -> symlink, train/rank*.pt, .metadata.json, config.json}.

Usage: python make_finetune_start.py --model <dir with model_and_optim> --train-from <baseline step dir> --start-step 38548 --out <dir>
"""
import argparse, json, os, shutil
from pathlib import Path
import torch

SEQ_TOKENS = 64 * 8192


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True); ap.add_argument("--train-from", type=Path, required=True)
    ap.add_argument("--start-step", type=int, required=True); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    S = a.start_step; d = a.out / f"step{S}"
    if d.exists(): shutil.rmtree(d)
    (d / "train").mkdir(parents=True)
    src = (a.model / "model_and_optim").resolve(); assert (src / ".metadata").exists(), src
    os.symlink(src, d / "model_and_optim")
    for f in sorted((a.train_from / "train").glob("rank*.pt")):
        st = torch.load(f, map_location="cpu", weights_only=False)
        st["global_step"] = S; st["global_train_tokens_seen"] = S * SEQ_TOKENS
        st["data_loader"]["batches_processed"] = S; st["data_loader"]["tokens_processed"] = S * SEQ_TOKENS
        torch.save(st, d / "train" / f.name)
    shutil.copy(a.train_from / ".metadata.json", d / ".metadata.json")
    cfg = a.model / "config.json" if (a.model / "config.json").exists() else a.train_from / "config.json"
    shutil.copy(cfg, d / "config.json")
    print(json.dumps(dict(out=str(d), model=str(src), trainer_state_from=str(a.train_from), start_step=S, tokens_seen=S * SEQ_TOKENS, ranks=len(list((d / "train").glob("rank*.pt"))))))


if __name__ == "__main__":
    main()
