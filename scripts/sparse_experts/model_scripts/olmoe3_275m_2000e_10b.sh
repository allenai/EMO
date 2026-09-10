##############################################################
# PARENT: "scripts/sparse_experts/model_scripts/olmoe3_275m_10b.sh"
# DESCRIPTION:
#     2000-expert arm of the olmoe3_275m series: 512 -> 2000 routed experts, everything else fixed
#     (expert hidden 544, top-16, LatentMoE, shared expert); ~9.61B total / 285.2M active.
#     1000+ experts per rank exceed the grouped-GEMM cap (< 1024 groups), so this arm runs
#     expert-parallel degree 2 (1000 local experts per GPU, EP pairs = NVLink neighbours) over the
#     sync_1d transport: plain torch all-to-all, dropless, no NVSHMEM (the rowwise_nvshmem path's
#     extension does not build on jupiter). DP stays 32-way; expert grads reduce over the 16
#     EP-shard replicas. Same data, WSD-trunk recipe (LR 8e-4, no decay), 4-node jupiter topology
#     and 5000-step checkpoints as the other arms. Standard (non-EMO) routing.
#
#   git add ... && git commit && git push origin <branch>   # gantry clones from origin!
#   bash scripts/sparse_experts/model_scripts/olmoe3_275m_2000e_10b.sh [dry_run]
##############################################################
export OLMOE3_NUM_EXPERTS=2000
export OLMOE3_EP_SIZE=2
export OLMOE3_EP_PATH=sync_1d
export OLMOE3_RUNNAME="${OLMOE3_RUNNAME:-olmoe3_275m_2000e_10b}"
export OLMOE3_WANDB_TAGS="${OLMOE3_WANDB_TAGS:-10b}"
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_10b.sh" "$@"
