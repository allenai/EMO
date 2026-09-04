# DESCRIPTION:
#     1-GPU Beaker job running scripts/sparse_experts/check_grouped_gemm.py: numerical
#     verification (forward + backward vs a per-expert matmul loop) of grouped_gemm with
#     HOST batch_sizes at 1024 groups, host-vs-device agreement at 512 groups, and
#     DroplessMoEMLP end to end at 1024 experts. Job status `failed` == a check failed.
#
#   git add . && git commit && git push origin <branch>   # gantry clones from origin!
#   MODE=beaker BEAKER_NO_FOLLOW=1 bash scripts/sparse_experts/model_scripts/check_grouped_gemm_beaker.sh
##############################################################
source "$(dirname "${BASH_SOURCE[0]}")/../../launch_common.sh"

BEAKER_NODES=1
BEAKER_GPUS=1
NPROC=1
PREEMPTIBLE=0   # allocated: even 1-GPU smoke/check jobs schedule much faster than preemptible

launch scripts/sparse_experts/check_grouped_gemm.py check_grouped_gemm_1024 "$@"
