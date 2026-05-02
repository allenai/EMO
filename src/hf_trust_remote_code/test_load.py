#!/usr/bin/env python3
"""
Sanity-check that an uploaded FlexMoE checkpoint loads via trust_remote_code
and can run a forward pass + short generation.

Usage:
    python test_load.py --repo-id allenai/Dense_1b_130B
    python test_load.py --repo-id allenai/Dense_1b_130B --device cuda --dtype bfloat16
    python test_load.py --repo-id allenai/Dense_1b_130B --prompt "The capital of France is"
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--repo-id", required=True, help="HF repo id, or path to a local checkpoint dir."
    )
    p.add_argument("--prompt", default="Hello, my name is")
    p.add_argument("--max-new-tokens", type=int, default=20)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument(
        "--token",
        default=None,
        help="HF token for private repos. Defaults to cached login / HF_TOKEN.",
    )
    args = p.parse_args()

    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
        args.dtype
    ]

    print(f"loading tokenizer: {args.repo_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.repo_id, trust_remote_code=True, token=args.token
    )

    print(f"loading model:     {args.repo_id} (dtype={args.dtype}, device={args.device})")
    model = AutoModelForCausalLM.from_pretrained(
        args.repo_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        token=args.token,
    ).to(args.device)
    model.eval()

    print(f"architecture:      {type(model).__name__}")
    print(f"parameters:        {sum(p.numel() for p in model.parameters()):,}")

    inputs = tokenizer(args.prompt, return_tensors="pt").to(args.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)

    print("---")
    print(f"prompt:     {args.prompt!r}")
    print(f"generation: {text!r}")
    print("OK")


if __name__ == "__main__":
    main()
