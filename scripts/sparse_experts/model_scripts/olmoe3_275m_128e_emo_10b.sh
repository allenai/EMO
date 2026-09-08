# DESCRIPTION:
#     EMO + 128-expert arm: olmoe3_275m_128e_10b.sh (512 -> 128 routed experts, everything else
#     fixed; 0.801B total / 274.5M active) with EMO document-pool routing on (OLMOE3_EMO=1: per-document
#     pool drawn uniformly from [top_k=16, 128] experts, eval pool 128, local-batch LB loss with global
#     balancing). Same data, WSD-trunk recipe (LR 8e-4, no decay) and 4-node jupiter topology as the
#     other olmoe3_275m arms. Launched ALLOCATED (OLMOE3_PREEMPTIBLE=0).
#
#   git add ... && git commit && git push origin <branch>   # gantry clones from origin!
#   bash scripts/sparse_experts/model_scripts/olmoe3_275m_128e_emo_10b.sh [dry_run]
##############################################################
export OLMOE3_NUM_EXPERTS=128
export OLMOE3_EMO=1
export OLMOE3_RUNNAME=olmoe3_275m_128e_emo_10b
export OLMOE3_WANDB_TAGS=10b
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_10b.sh" "$@"
