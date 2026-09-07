# DESCRIPTION:
#     EMO + 1024-expert arm: olmoe3_275m_1024e_10b.sh (512 -> 1024 routed experts, everything else
#     fixed; 5.017B total / 279.6M active) with EMO document-pool routing on (OLMOE3_EMO=1, the
#     ladder's EMO settings: per-document pool drawn uniformly from [top_k=16, 1024] experts, eval
#     pool 1024, local-batch LB loss with global balancing). Same data, WSD-trunk recipe (LR 8e-4,
#     no decay) and 4-node jupiter topology as the other olmoe3_275m arms.
#
#   git add ... && git commit && git push origin <branch>   # gantry clones from origin!
#   bash scripts/sparse_experts/model_scripts/olmoe3_275m_1024e_emo_10b.sh [dry_run]
##############################################################
export OLMOE3_NUM_EXPERTS=1024
export OLMOE3_EMO=1
export OLMOE3_RUNNAME=olmoe3_275m_1024e_emo_10b
export OLMOE3_WANDB_TAGS=10b
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_10b.sh" "$@"
