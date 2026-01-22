scripts/hf_finetune_with_pruning.sh \
   --model /root/ryanwang/phdbrainstorm/FlexMoE/models/dense_1b_olmoe-mix_prenorm_noqknorm_1123/step30995-hf \
   --task arc_challenge \
   --prune-keep-k 32 \
   --output-dir /root/ryanwang/phdbrainstorm/evals/testbed_arc_challenge \
   --num-gpus 1