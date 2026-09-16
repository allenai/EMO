##############################################################
# PARENT: "scripts/sparse_experts/model_scripts/olmoe3_275m_ft.sh"
# DESCRIPTION:
#     olmoe3_squares router-recovery test: the same post-merge finetuning as olmoe3_275m_ft.sh (0.5B tokens at constant
#     LR from a hybrid start checkpoint), but ONLY the routed-expert routers train: base LR 0 for every other parameter
#     (OLMOE3_ROUTER_ONLY_LR=8e-4 gives the routers the usual LR). Everything else identical.
#   FT_START=<hybrid dir> FT_START_STEP=38548 OLMOE3_RUNNAME=<name> bash scripts/sparse_experts/model_scripts/olmoe3_275m_ft_router.sh [dry_run]
##############################################################
export OLMOE3_ROUTER_ONLY_LR="${OLMOE3_ROUTER_ONLY_LR:-8e-4}"
export OLMOE3_WANDB_TAGS="${OLMOE3_WANDB_TAGS:-olmoe3_squares,finetune,router_only}"
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_ft.sh" "$@"
