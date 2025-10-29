
python src/examples/huggingface/convert_checkpoint_to_hf.py \
  --checkpoint-input-path /root/ryanwang/phdbrainstorm/models/dense_1b_olmoe-mix_1028/step30995 \
  --max-sequence-length 4096 \
  --huggingface-output-dir /root/ryanwang/phdbrainstorm/models/dense_1b_olmoe-mix_1028/step30995 \
  --dtype float32