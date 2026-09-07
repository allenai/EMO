# DESCRIPTION:
#     EMO + 1000-expert arm: olmoe3_275m_1024e_10b.sh (512 -> 1000 routed experts, everything else
#     fixed; 4.904B total / 279.5M active) with EMO document-pool routing on (OLMOE3_EMO=1, the
#     ladder's EMO settings: per-document pool drawn uniformly from [top_k=16, 1000] experts, eval
#     pool 1000, local-batch LB loss with global balancing). Same data, WSD-trunk recipe (LR 8e-4,
#     no decay) and 4-node jupiter topology as the other olmoe3_275m arms.
#
#   git add ... && git commit && git push origin <branch>   # gantry clones from origin!
#   bash scripts/sparse_experts/model_scripts/olmoe3_275m_1000e_emo_10b.sh [dry_run]
##############################################################
export OLMOE3_NUM_EXPERTS=1000
export OLMOE3_EMO=1
export OLMOE3_RUNNAME=olmoe3_275m_1000e_emo_10b
export OLMOE3_WANDB_TAGS=10b
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_10b.sh" "$@"
