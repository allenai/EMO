#!/usr/bin/env python3
"""Parse experiment YAML config and emit shell variable assignments.

Environment variables already set take precedence over YAML values,
so you can override any config value with:

    STAGE3_LR=1e-4 bash run.sh math-ta-01 stage3
"""
import os
import sys

import yaml


def main():
    config_path = sys.argv[1]

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    experiment_name = os.path.splitext(os.path.basename(config_path))[0]

    bm = cfg.get("base_model", {})
    exp = cfg.get("experts", {})
    data = cfg.get("data", {})
    s1 = cfg.get("stage1", {})
    s2 = cfg.get("stage2", {})
    s3 = cfg.get("stage3", {})
    wb = cfg.get("wandb", {})

    base_path = bm["path"]

    shell_vars = {
        "EXPERIMENT_NAME": experiment_name,
        "BASE_MODEL_PATH": base_path,
        "BASE_MODEL_HF_PATH": bm.get("hf_path", f"{base_path}-hf"),
        "NUM_NEW_EXPERTS": exp.get("num_new", 4),
        "NUM_SHARED_EXPERTS": exp.get("num_shared", 1),
        "INIT_METHOD": exp.get("init_method", "similar"),
        "INIT_K": exp.get("init_k", 2),
        "EXCLUDE_EXPERTS": ",".join(str(x) for x in exp.get("exclude_experts", [127])),
        "MIX": data.get("mix", "mj_finemath4plus"),
        "MIX_BASE_DIR": data.get("mix_base_dir", "/weka/oe-training-default/ai2-llm"),
        "STAGE1_BATCH_SIZE": s1.get("batch_size", 16),
        "STAGE1_SEQ_LENGTH": s1.get("seq_length", 4096),
        "STAGE1_MAX_TOKENS": s1.get("max_tokens", 25000000),
        "STAGE1_CLUSTER": s1.get("cluster", "ai2/jupiter-cirrascale-2"),
        "STAGE1_GPUS": s1.get("gpus", 4),
        "STAGE2_CLUSTER": s2.get("cluster", "ai2/jupiter-cirrascale-2"),
        "STAGE2_GPUS": s2.get("gpus", 1),
        "STAGE3_LR": s3.get("lr", "4e-4"),
        "STAGE3_NUM_BILLION_TOKENS": s3.get("num_billion_tokens", 10),
        "STAGE3_WARMUP_FRACTION": s3.get("warmup_fraction", 0.1),
        "STAGE3_LB_LOSS_WEIGHT": s3.get("lb_loss_weight", "1e-2"),
        "STAGE3_CLUSTER": s3.get("cluster", "ai2/jupiter"),
        "STAGE3_GPUS": s3.get("gpus", 8),
        "STAGE3_NODES": s3.get("nodes", 4),
        "WANDB_ENTITY": wb.get("entity", "allennlp"),
        "WANDB_PROJECT": wb.get("project", "flex2-extensions-kevinf"),
    }

    for key, value in shell_vars.items():
        if key not in os.environ:
            safe_val = str(value).replace("'", "'\\''")
            print(f"{key}='{safe_val}'")


if __name__ == "__main__":
    main()
