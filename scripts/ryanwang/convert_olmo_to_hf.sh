
python src/examples/huggingface/convert_checkpoint_to_hf.py \
  --checkpoint-input-path /root/ryanwang/phdbrainstorm/models/olmoe-pretrain-mose-natural-1022/step30995/olmoe-finetune-arc_easy-mc/step1686 \
  --max-sequence-length 4096 \
  --huggingface-output-dir /root/ryanwang/phdbrainstorm/models/olmoe-pretrain-mose-natural-1022/step30995/olmoe-finetune-arc_easy-mc/step1686-hf \
  --dtype float32