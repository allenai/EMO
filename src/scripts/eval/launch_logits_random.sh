
# sample router probability format

sample_file = "/root/ryanwang/phdbrainstorm/FlexMoE/prune/moe_1b7b_128experts_olmoe-mix_130B_1103_step30995-hf/task-hellaswag_rc_validation_0shot-router.jsonl"

# read the file

import json
with open(sample_file, "r") as f:
    lines = f.readlines()
    breakpoint()
