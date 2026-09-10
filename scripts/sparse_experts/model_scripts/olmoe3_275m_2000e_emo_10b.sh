##############################################################
# PARENT: "scripts/sparse_experts/model_scripts/olmoe3_275m_2000e_10b.sh"
# DESCRIPTION:
#     EMO + 2000-expert arm: olmoe3_275m_2000e_10b.sh (2000 routed experts, EP=2 over sync_1d)
#     with EMO document-pool routing on (OLMOE3_EMO=1; per-document pool drawn uniformly from
#     [top_k=16, 2000] experts, eval pool 2000, local-batch LB loss with global balancing), the
#     same "pool range = [16, num_experts]" convention as the 128e / 512e / 1000e EMO arms. The
#     EMO router is DP-replicated and routes over all 2000 experts before dispatch, so EP changes
#     nothing about its semantics.
#
#   bash scripts/sparse_experts/model_scripts/olmoe3_275m_2000e_emo_10b.sh [dry_run]
##############################################################
export OLMOE3_EMO=1
export OLMOE3_RUNNAME="${OLMOE3_RUNNAME:-olmoe3_275m_2000e_emo_10b}"
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_2000e_10b.sh" "$@"
