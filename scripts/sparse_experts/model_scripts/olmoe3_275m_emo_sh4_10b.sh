##############################################################
# PARENT: "scripts/sparse_experts/model_scripts/olmoe3_275m_emo_10b.sh"
# DESCRIPTION:
#     EMO 512e with FOUR shared experts per MoE block instead of one (user request 2026-11-19), from scratch to 10B tokens,
#     otherwise the EMO 512e recipe (LR 8e-4, 4 nodes). The shared experts are full-width (d_model 640, hidden 544), all always
#     active, mixed by a 4-way sigmoid router (OLMOE3_NUM_SHARED in olmoe3_275m.py); the routed side is unchanged (16 of 512), so
#     the active size grows from 276.7M to 304.9M parameters. Start model of the 4- / 8-sub-model controls (squares_shared4.sh).
#
#   git add ... && git commit && git push origin <branch>   # gantry clones from origin!
#   bash scripts/sparse_experts/model_scripts/olmoe3_275m_emo_sh4_10b.sh [dry_run]
##############################################################
export OLMOE3_EMO=1
export OLMOE3_NUM_SHARED=4
export OLMOE3_RUNNAME=olmoe3_275m_emo_sh4_10b
export OLMOE3_WANDB_TAGS=10b,shared4
source "$(dirname "${BASH_SOURCE[0]}")/olmoe3_275m_10b.sh" "$@"
