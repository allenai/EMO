
python src/examples/huggingface/convert_checkpoint_to_hf.py \
  --checkpoint-input-path /root/ryanwang/phdbrainstorm/FlexMoE/models/moe_1b7b_olmoe-mix_300B_1030/step71526 \
  --max-sequence-length 4096 \
  --huggingface-output-dir /root/ryanwang/phdbrainstorm/FlexMoE/models/moe_1b7b_olmoe-mix_300B_1030/step71526-hf \
  --dtype float32