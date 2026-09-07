# DESCRIPTION:
#     1000-expert arm of olmoe3_275m_10b.sh: the identical OLMoE3-ladder 275M model, data, WSD-trunk
#     recipe (LR 8e-4, no decay) and 4-node jupiter topology, with the routed expert count raised
#     from 512 to 1000 and nothing else changed. Why 1000 not 1024: the pinned grouped-GEMM
#     kernel caps groups at 1023 per rank at EP1; 1000 is the OLMo-core team's own EP1 stand-in
#     for the 1024-expert point (their LatentMoE ladder), so we follow it (expert width 544, latent 320, top-16, one shared
#     expert, same LB/z losses). This mirrors the ladder's own variable-sparsity -> uniform-512 move.
#     Size: 4.904B total / 279.5M active (vs 2.608B / 276.7M); only the router grows in the active
#     path. Expect ~10-15% lower TPS than the 512-expert run (the ladder measured -14% for 275M
#     going 64 -> 512 experts on B300s).
#
#   git add ... && git commit && git push origin <branch>   # gantry clones from origin!
#   bash scripts/sparse_experts/model_scripts/olmoe3_275m_1000e_10b.sh [dry_run]
##############################################################
export OLMOE3_NUM_EXPERTS=1000
export OLMOE3_RUNNAME=olmoe3_275m_1000e_10b
export OLMOE3_WANDB_TAGS=10b
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_10b.sh" "$@"
