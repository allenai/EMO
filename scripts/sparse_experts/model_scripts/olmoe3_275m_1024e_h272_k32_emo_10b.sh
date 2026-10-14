##############################################################
# PARENT: "scripts/sparse_experts/model_scripts/olmoe3_275m_emo_10b.sh"
# DESCRIPTION:
#     EMO with half-size experts (user request 2026-10-14): the 512e EMO recipe with 1024 routed experts of hidden size 272
#     (half of 544; latent dim 320, shared expert and everything else unchanged), top-32 of 1024, per-document pools uniform in
#     [32, 1024]. Routed-expert parameters are exactly the 512e model's (2.406B); the total is 2.611B vs 2.608B (the router has
#     twice as many rows). Active parameters 279.6M = the 512e model's active FFN width (32 x 272 = 16 x 544).
#     1024 experts need EP=2 (the grouped-GEMM kernel caps local experts at 1023): sync_1d transport, 512 local experts per rank,
#     like the 2000e arm. Same data, WSD trunk (LR 8e-4, no decay), 10B tokens, 4 nodes jupiter, allocated.
#
#   git add ... && git commit && git push origin <branch>   # gantry clones from origin!
#   bash scripts/sparse_experts/model_scripts/olmoe3_275m_1024e_h272_k32_emo_10b.sh [dry_run]
##############################################################
export OLMOE3_NUM_EXPERTS=1024
export OLMOE3_EXPERT_HIDDEN=272
export OLMOE3_TOP_K=32
export OLMOE3_EP_SIZE=2
export OLMOE3_EP_PATH=sync_1d
export OLMOE3_EMO=1
export OLMOE3_RUNNAME=olmoe3_275m_1024e_h272_k32_emo_10b
export OLMOE3_WANDB_TAGS=10b,halfexperts
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_10b.sh" "$@"
