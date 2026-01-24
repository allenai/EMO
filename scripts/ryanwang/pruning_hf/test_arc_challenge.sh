scripts/hf_finetune_with_pruning.sh \
   --model /root/ryanwang/phdbrainstorm/FlexMoE/models/moe_1b4b_32experts_1224/step30995-hf \
   --task arc_challenge \
   --prune-keep-k 16 \
   --output-dir /root/ryanwang/phdbrainstorm/evals/testbed_arc_challenge \
   --num-gpus 1 \
   --skip-activation
#   --skip-prune