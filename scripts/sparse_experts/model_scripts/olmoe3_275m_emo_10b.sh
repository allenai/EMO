# DESCRIPTION:
#     EMO arm of olmoe3_275m_10b.sh: the identical OLMoE3-ladder 275M model, data, recipe and
#     4-node jupiter topology, with EMO document-pool routing switched on in the router
#     (OLMOE3_EMO=1 in scripts/sparse_experts/olmoe3_275m.py). Settings follow the OLMo-core
#     team's own EMO branch of the ladder (allenai/scaling-ladders akshitab/emo-integration):
#     each document draws its expert pool uniformly from [top_k=16, 512] experts at train time,
#     eval uses all 512, load-balancing loss at local-batch granularity with global (DP-wide)
#     balancing. Segment ids come from EOS positions inside the model; no data changes.
#     Parameter counts are unchanged (276.7M active / 2.608B total).
#
#   git add ... && git commit && git push origin <branch>   # gantry clones from origin!
#   bash scripts/sparse_experts/model_scripts/olmoe3_275m_emo_10b.sh [dry_run]
##############################################################
export OLMOE3_EMO=1
export OLMOE3_RUNNAME=olmoe3_275m_emo_10b
export OLMOE3_WANDB_TAGS=10b
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_10b.sh" "$@"
