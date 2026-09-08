# DESCRIPTION:
#     128-expert arm of olmoe3_275m_10b.sh: the identical OLMoE3-ladder 275M model, data, WSD-trunk
#     recipe (LR 8e-4, no decay) and 4-node jupiter topology, with the routed expert count lowered
#     from 512 to 128 and nothing else changed (expert width 544, latent 320, top-16, one shared
#     expert, same LB/z losses). Size: 0.801B total / 274.5M active (vs 2.608B / 276.7M at 512);
#     only the router shrinks in the active path. Launched ALLOCATED (OLMOE3_PREEMPTIBLE=0).
#
#   git add ... && git commit && git push origin <branch>   # gantry clones from origin!
#   bash scripts/sparse_experts/model_scripts/olmoe3_275m_128e_10b.sh [dry_run]
##############################################################
export OLMOE3_NUM_EXPERTS=128
export OLMOE3_RUNNAME=olmoe3_275m_128e_10b
export OLMOE3_WANDB_TAGS=10b
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_10b.sh" "$@"
