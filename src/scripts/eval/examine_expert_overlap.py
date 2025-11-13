from collections import defaultdict

model_path = "/root/ryanwang/phdbrainstorm/FlexMoE/prune/twolevel-32_1b7b_128experts_olmoe-mix_130B_1110_step30995-hf"
out_dir = "/root/ryanwang/phdbrainstorm/FlexMoE/eval_plots"

router_paths = [
    ("arc_challenge", "task-arc_challenge_rc_validation_0shot-router.jsonl"),
    ("arc_easy", "task-arc_easy_rc_validation_0shot-router.jsonl"),
    ("boolq", "task-boolq_rc_validation_0shot-router.jsonl"),
    ("csqa", "task-csqa_rc_validation_0shot-router.jsonl"),
    ("hellaswag", "task-hellaswag_rc_validation_0shot-router.jsonl"),
    ("openbooqa", "task-openbookqa_rc_validation_0shot-router.jsonl"),
    ("piqa", "task-piqa_rc_validation_0shot-router.jsonl"),
    ("socialiqa", "task-socialiqa_rc_validation_0shot-router.jsonl"),
    ("winogrande", "task-winogrande_rc_validation_0shot-router.jsonl")
]

# load the sample file and print the first line
import json
import torch


k_range = [4, 8, 16, 32, 64]

for k in k_range:

    router_dict = defaultdict(set)
    for name, router_path in router_paths:
        # load router probabilities
        with open(f"{model_path}/{router_path}", "r") as f:
            logits = f.readline()
            logits_json = json.loads(logits)
            logits_values = logits_json["avg_router_probabilities"]

            logits_tensor = torch.tensor(logits_values) # layers x experts

            topk = torch.topk(
                logits_tensor,
                k
            ).indices

            # create a unique id for each expert per layer
            for layer_idx in range(topk.shape[0]):
                topk[layer_idx] += layer_idx * 1000  # assuming less than 1000 experts per layer

            # squash to 1D
            topk_1d = topk.flatten().tolist()
            router_dict[name] = set(topk_1d)

    # create a overlap matrix and plot it
    import numpy as np
    import matplotlib.pyplot as plt
    overlap_matrix = np.zeros((len(router_paths), len(router_paths)))
    for i, (name_i, _) in enumerate(router_paths):
        for j, (name_j, _) in enumerate(router_paths):
            if i <= j:
                overlap = len(router_dict[name_i].intersection(router_dict[name_j]))
                overlap_matrix[i, j] = overlap
                overlap_matrix[j, i] = overlap
    plt.figure(figsize=(10, 8))
    # save the overlap matrix as a heatmap
    plt.imshow(overlap_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.xticks(range(len(router_paths)), [name for name, _ in router_paths], rotation=45)
    plt.yticks(range(len(router_paths)), [name for name, _ in router_paths])
    plt.title(f'Twolevel Expert Overlap Matrix (Top-{k})')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/Twolevel_expert_overlap_top{k}.png")
    plt.close()
    print(f"Saved expert overlap matrix for top-{k} to expert_overlap_top{k}.png")

    # we also get the average overlap
    total_overlap = 0
    count = 0
    for i in range(len(router_paths)):
        for j in range(i + 1, len(router_paths)):
            total_overlap += overlap_matrix[i, j]
            count += 1
    avg_overlap = total_overlap / count
    print(f"Average Twolevel expert overlap for top-{k}: {avg_overlap}")







