
base_dir="/root/ryanwang/phdbrainstorm/data/finetune/arc_easy\:mc_train"

jsonl_file="${base_dir}/processed/out.jsonl"
destination="${base_dir}/tokenized"
tokenizer_name="allenai/OLMo-2-1124-7B"

# gzip the data
gzip ${jsonl_file}

# tokenize the files
dolma tokens \
  --documents ${jsonl_file}.gz \
  --tokenizer.name_or_path ${tokenizer_name} \
  --tokenizer.eos_token_id 100257 \
  --tokenizer.pad_token_id 100277 \
  --destination ${destination} \
  --dtype uint32 \
  --processes 1