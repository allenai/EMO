from huggingface_hub import snapshot_download

BASE_PATH = "/data/input/ai2-llm/pretraining-data/sources/the-pile-of-law"

# Download all .jsonl.xz files at once
snapshot_download(
    repo_id="pile-of-law/pile-of-law",
    repo_type="dataset",
    local_dir=BASE_PATH,
    allow_patterns=["*.jsonl.xz"],  # Only download data files
    local_dir_use_symlinks=False
)