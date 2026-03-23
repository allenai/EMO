
# Summary of extension experiments

The goal is to experiment with expert modularity in MoEs, to enable cheaper continual learning.


Extension: Add new fine-grained experts to existing MoE specializing in new domains (eg. math, code, medical/biology, french, etc.)
Selective training: Train selected experts in existing MoEs, and train them further.


- Merge newly added experts in multiple domains, lightly train the router.