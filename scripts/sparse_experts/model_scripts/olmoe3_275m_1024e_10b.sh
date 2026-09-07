# DESCRIPTION:
#     1024-expert arm of olmoe3_275m_10b.sh: the identical OLMoE3-ladder 275M model, data, WSD-trunk
#     recipe (LR 8e-4, no decay) and 4-node jupiter topology, with the routed expert count doubled
#     from 512 to 1024 and nothing else changed (expert width 544, latent 320, top-16, one shared
#     expert, same LB/z losses). This mirrors the ladder's own variable-sparsity -> uniform-512 move.
#     Size: 5.017B total / 279.6M active (vs 2.608B / 276.7M); only the router grows in the active
#     path. Expect ~10-15% lower TPS than the 512-expert run (the ladder measured -14% for 275M
#     going 64 -> 512 experts on B300s).
#
#   git add ... && git commit && git push origin <branch>   # gantry clones from origin!
#   bash scripts/sparse_experts/model_scripts/olmoe3_275m_1024e_10b.sh [dry_run]
##############################################################
export OLMOE3_NUM_EXPERTS=1024
export OLMOE3_RUNNAME=olmoe3_275m_1024e_10b
export OLMOE3_WANDB_TAGS=10b
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_10b.sh" "$@"
