
# sample router probability format

sample_file = "/root/ryanwang/phdbrainstorm/FlexMoE/prune/moe_1b7b_128experts_olmoe-mix_130B_1103_step30995-hf/task-hellaswag_rc_validation_0shot-router.jsonl"

# load the sample file and print the first line
import json
import torch

with open(sample_file, "r") as f:
    logits = f.readline()
    logits_json = json.loads(logits)
    logits_values = logits_json["avg_router_probabilities"]

    logits_tensor = torch.tensor(logits_values)

    # create a random tensor with the same shape as logits_tensor
    random_tensor = torch.rand_like(logits_tensor)

    breakpoint()
