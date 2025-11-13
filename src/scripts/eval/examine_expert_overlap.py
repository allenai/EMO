from collections import defaultdict

model_path = "/root/ryanwang/phdbrainstorm/FlexMoE/prune/moe_1b7b_128experts_olmoe-mix_130B_1103_step30995-hf"

router_paths = [
    "task-arc_challenge_rc_validation_0shot-router.jsonl",
    "task-arc_easy_rc_validation_0shot-router.jsonl",
    "task-boolq_rc_validation_0shot-router.jsonl",
    "task-csqa_rc_validation_0shot-router.jsonl",
    "task-hellaswag_rc_validation_0shot-router.jsonl",
    "task-openbookqa_rc_validation_0shot-router.jsonl",
    "task-piqa_rc_validation_0shot-router.jsonl",
    "task-socialiqa_rc_validation_0shot-router.jsonl",
    "task-winogrande_rc_validation_0shot-router.jsonl",
]

# load the sample file and print the first line
import json
import torch


k_range = [4, 8, 16, 32, 64]

for k in k_range:

    router_dict = defaultdict(set)
    for router_path in router_paths:
        # load router probabilities
        with open(f"{model_path}/{router_path}", "r") as f:
            logits = f.readline()
            logits_json = json.loads(logits)
            logits_values = logits_json["avg_router_probabilities"]

            logits_tensor = torch.tensor(logits_values)
            topk = torch.topk(
                logits_tensor,
                k
            ).indices.tolist()

            breakpoint()





