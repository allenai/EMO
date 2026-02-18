# Dataset Tokenization Pipeline Guide

## Purpose
Step-by-step guide for tokenizing any new dataset for OLMo pretraining using AI2 infrastructure. Covers EC2 provisioning, data preparation, Dolma format conversion, tokenization, and S3 upload.

---

## Fast Path: YAML Config (Preferred)

For standard datasets (CSV, Arrow, or JSONL from S3 or local), use the generic tokenization pipeline. Write a YAML config and run one command.

### Usage
```bash
# Remote (provisions EC2 via poormanray, runs full pipeline):
./scripts/kevinf/data/ec2_scripts/tokenize_remote.sh scripts/kevinf/data/configs/my-dataset.yaml

# Local (runs pipeline directly on this machine):
./scripts/kevinf/data/ec2_scripts/tokenize_remote.sh scripts/kevinf/data/configs/my-local-dataset.yaml
```

### YAML Config Template
Create a new file in `scripts/kevinf/data/configs/<dataset>.yaml`:
```yaml
dataset:
  name: my-dataset              # Used as cluster name and directory prefix

source:
  format: csv                   # csv | arrow | jsonl
  s3_path: s3://ai2-llm/pretraining-data/sources/my-dataset/raw/
  # local_path: /data/my-dataset/   # For local data (instead of s3_path)
  # documents_pattern: "*.jsonl.gz" # For jsonl format: glob pattern for source docs

  # CSV-specific:
  text_field: text
  id_field: doc_id
  metadata_fields: [field1, field2]
  files: [data.csv.gz]

  # Arrow-specific:
  # data_dirs: [config1, config2]

destination:
  # local_dolma_docs: /custom/path/dolma_docs/  # Optional, defaults to /mnt/raid0/<name>/dolma_docs
  # local_tokenized: /custom/path/tokenized/    # Optional, defaults to /mnt/raid0/<name>/tokenized
  s3_dolma_docs: s3://ai2-llm/pretraining-data/sources/my-dataset/dolma_docs/
  s3_tokenized: s3://ai2-llm/preprocessed/my-dataset/dolma2-tokenizer/

compute:
  mode: remote                  # remote | local
  instance_type: c6a.48xlarge
  storage_size: 200
  num_processes: 192
  convert_workers: 64
  docs_per_shard: 50000

tokenizer:
  name: allenai/dolma2-tokenizer
  eos_token_id: 100257
  pad_token_id: 100277
  dtype: uint32
  use_uv: true                  # uv run dolma vs dolma
  extra_args:                   # Optional additional dolma flags
    - --no-tokenizer.segment_before_tokenization
    - --tokenizer.encode_special_tokens
    - --max_size 1_000_000_000
    - --sample_ring_prop
```

### What the Pipeline Does
1. **Download** from S3 (skipped if `s3_path` is empty)
2. **Convert** based on `format`: csv → `convert_csv_to_jsonl.py`, arrow → `convert_arrow_to_jsonl.py`, jsonl → skip
3. **Tokenize** with dolma using config settings
4. **Count tokens** and report
5. **Upload** to S3 (skipped if `s3_tokenized` is empty)

### Monitoring (Remote Mode)
```bash
poormanray run -n <dataset-name> -c 'tail -50 ~/worker.log'
poormanray ssh -n <dataset-name>    # then: tail -f ~/worker.log
poormanray terminate -n <dataset-name>   # when done
```

### Existing Configs
| Config | Format | Mode | Dataset |
|--------|--------|------|---------|
| `configs/mimic-iv-note.yaml` | csv | remote | MIMIC-IV clinical notes |
| `configs/the-pile-of-law.yaml` | jsonl | local | Pile of Law legal docs |

### HuggingFace Datasets (Download + Convert + Tokenize)
For HF datasets, use the download pipeline first, then tokenize:
```bash
# Step 1: Download and convert Arrow → JSONL
./scripts/kevinf/data/download_and_convert.sh <hf-repo> <output-base> [config1 config2 ...]

# Step 2: Tokenize the JSONL output
./scripts/kevinf/data/tokenize_datasets.sh <output-base>/jsonl <output-base>/tokenized <processes> [config1 ...]

# Step 3: Count tokens
./scripts/kevinf/data/count_tokens.sh <output-base>/tokenized
```

### Key Files
| File | Purpose |
|------|---------|
| `scripts/kevinf/data/configs/*.yaml` | Dataset configs |
| `scripts/kevinf/data/ec2_scripts/tokenize_remote.sh` | Orchestrator (local or remote) |
| `scripts/kevinf/data/ec2_scripts/tokenize_worker.sh` | Pipeline worker |
| `src/scripts/kevinf/data/parse_config.py` | YAML → shell env vars |
| `src/scripts/kevinf/data/convert_csv_to_jsonl.py` | CSV → Dolma JSONL |
| `src/scripts/kevinf/data/convert_arrow_to_jsonl.py` | Arrow → Dolma JSONL |
| `scripts/kevinf/data/download_and_convert.sh` | HF download + Arrow→JSONL |
| `scripts/kevinf/data/tokenize_datasets.sh` | Local multi-config tokenization |
| `scripts/kevinf/data/count_tokens.sh` | Count tokens in .npy files |

---

## Manual Pipeline (Custom Datasets)

Use the manual steps below when the YAML config approach doesn't fit — e.g., custom text cleaning, unusual formats (XML, raw text files), or multi-node distributed processing.

## Quick Reference

### Key Tools
| Tool | Purpose | Install |
|------|---------|---------|
| poormanray | EC2 cluster management | `uv tool install poormanray` |
| dolma | Tokenization | `uv pip install dolma` |
| s5cmd | Fast parallel S3 transfers | Pre-installed on instances via setup commands |
| datamap-rs | Resharding, filtering, dedup (Rust) | Pre-installed via `setup-d2tk` |

### Key Tokenizer Settings (Dolma2)
| Setting | Value |
|---------|-------|
| Tokenizer | `allenai/dolma2-tokenizer` |
| EOS token ID | 100257 |
| PAD token ID | 100277 |
| dtype | uint32 |
| segment_before_tokenization | false |
| encode_special_tokens | true |

### S3 Conventions
```
s3://ai2-llm/
├── pretraining-data/sources/{dataset_name}/
│   ├── raw/              # Original source files
│   └── dolma_docs/       # Sharded .jsonl.gz in Dolma format
└── preprocessed/{dataset_name}/
    └── dolma2-tokenizer/  # Tokenized .npy files (training-ready)
```

---

## Phase 1: Planning

### Assess Your Dataset
Before spinning up any infrastructure, answer these questions:

1. **Where is the source data?**
   - S3 bucket (same region = free transfer, fastest)
   - Public HTTP/FTP (use wget/aria2c)
   - Hugging Face (use `huggingface-cli download`)
   - Already on WEKA/Ceph

2. **How big is it?**
   - Check with `s5cmd du` or `aws s3 ls --summarize --recursive`
   - Rough token estimate: total_bytes / 4 ≈ token count (for English text)

3. **How many files?**
   - Few large files → may need splitting before tokenization
   - Many small files (>100K) → must combine into shards first
   - Sweet spot for processing: 256MB per file (datamap-rs recommendation)

4. **What format?**
   - Already JSONL with `id` and `text` fields → skip to sharding
   - Raw text files → need conversion to Dolma JSONL format
   - Hugging Face dataset → export to JSONL first (see HF section below)
   - XML/HTML → extract text first

5. **Any filtering needed?**
   - Retracted/withdrawn content
   - License restrictions (commercial vs non-commercial)
   - Language filtering
   - Quality filtering
   - Boilerplate removal (references, headers, footers)

### Choose Instance Type

| Instance | vCPUs | RAM | Storage | Best For |
|----------|-------|-----|---------|----------|
| `c6a.48xlarge` | 192 | 384GB | EBS only | CPU-heavy processing, large memory needs |
| `i4i.32xlarge` | 128 | 1024GB | 8x3.75TB NVMe | IO-heavy work (many small files, tokenization) |
| `i4i.2xlarge` | 8 | 64GB | 1x468GB NVMe | Small datasets, light processing |

**Key considerations:**
- **EBS-only instances (c6a, c5, m5)**: Storage persists across stop/start. Need `--storage-size` flag. Can hit IO bottleneck with many parallel processes (high iowait).
- **NVMe instances (i4i, i7i)**: Much faster IO. `setup-d2tk` creates RAID0 array at `/mnt/raid0/`. Data is **lost on terminate**. Recommended by datamap-rs docs.
- For tokenization specifically, **IO is often the bottleneck**, so prefer NVMe instances.
- Provision storage at 3-4x the raw data size (raw + dolma shards + tokenized output + headroom).

---

## Phase 2: Infrastructure Setup

### Create Cluster
```bash
export cluster_name="your-dataset-name"

# For IO-heavy work (recommended for tokenization)
poormanray create -n $cluster_name -t i4i.32xlarge --number 1

# For CPU-heavy work with large storage needs
poormanray create -n $cluster_name -t c6a.48xlarge --number 1 --storage-size 2000

# For distributed processing across multiple nodes
poormanray create -n $cluster_name -t i4i.2xlarge --number 10
```

### Install Tools on Instances
```bash
poormanray setup -n $cluster_name              # AWS credentials
poormanray setup-d2tk -n $cluster_name -d       # Rust tools, RAID setup (for NVMe instances)
poormanray setup-dolma-python -n $cluster_name -d  # Python 3.12, uv, dolma
```

Wait a few minutes, then verify:
```bash
poormanray run -n $cluster_name -c 'ls ~/datamap-rs'      # d2tk installed
poormanray run -n $cluster_name -c 'uv run dolma --help'   # dolma installed
```

### SSH into Instance
```bash
poormanray list -n $cluster_name   # get IP
ssh ec2-user@<ip>
```

**Always work inside tmux** — SSH drops are common, especially under load:
```bash
tmux new -s work
# Ctrl+B, D to detach
# tmux attach -s work to reattach
```

---

## Phase 3: Download Source Data

### From S3 (same region — fastest)
```bash
# Public bucket (no auth needed)
s5cmd --no-sign-request --numworkers 256 cp "s3://source-bucket/path/*" /mnt/raid0/data/

# Private bucket (uses instance AWS creds)
s5cmd --numworkers 256 cp "s3://ai2-llm/path/*" /mnt/raid0/data/
```

### s5cmd Tips
- `cp` starts immediately, no resume on interrupt
- `sync` compares files first (slow on millions of files), but can resume
- **Strategy**: Use `cp` first for speed. If interrupted, switch to `sync` to grab what's missing.
- `--numworkers 256` for high-bandwidth instances (50+ Gbps)
- Use `--include "*.txt"` to filter file types
- Do NOT use `-sp` (source path) unless you want to preserve the S3 directory structure locally

### From Hugging Face
```bash
uv run huggingface-cli download <org>/<dataset> --local-dir /mnt/raid0/data/
```

### Checking Data Size
```bash
# S3
s5cmd --no-sign-request du -H "s3://bucket/path/*"

# Local
du -sh /mnt/raid0/data/
find /mnt/raid0/data/ -name "*.txt" | wc -l   # file count
```

---

## Phase 4: Convert to Dolma Format

### Dolma Document Format
Every document must be a JSON object with at minimum:
```json
{"id": "unique-document-id", "text": "the full document text"}
```

Optional but recommended fields:
```json
{
    "id": "dataset-subset-docname",
    "text": "...",
    "source": "dataset_name",
    "license": "CC BY",
    "metadata_field": "any extra info for later filtering"
}
```

### Input Scenarios

#### Many small text files → Sharded JSONL
This is the most common case (e.g., PMC articles, web crawl pages). You MUST combine them into larger shards. Dolma and datamap-rs choke on millions of tiny files due to per-file overhead.

**Target shard size: ~256MB uncompressed** (sweet spot for parallel processing).

Estimate files per shard: `256MB / avg_file_size`. For 60KB avg files → ~4000 files per shard.

Key decisions:
- Use gzip compression (`.jsonl.gz`) — dolma handles it natively, saves 3-4x disk space
- Use `ProcessPoolExecutor` for parallel conversion
- 64 workers is a good default for IO-bound file reading

```python
#!/usr/bin/env python3
"""Generic template: convert text files to sharded Dolma JSONL."""

import gzip
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

INPUT_DIR = "/mnt/raid0/data/raw"
OUTPUT_DIR = "/mnt/raid0/data/dolma_docs"
DATASET_NAME = "my_dataset"
MAX_SHARD_BYTES = 256 * 1024 * 1024  # 256MB
WORKERS = 64

def process_text(text: str) -> str:
    """Apply any text cleaning here. Modify as needed."""
    # Example: strip references section
    match = re.search(
        r'\n\s*(References|REFERENCES|====+\s*Refs)\s*\n',
        text, re.IGNORECASE
    )
    if match:
        text = text[:match.start()]
    return text.strip()

def process_batch(args):
    batch_files, shard_id, output_dir, dataset_name = args
    output_path = os.path.join(output_dir, f"{dataset_name}_{shard_id:05d}.jsonl.gz")
    count = 0
    with gzip.open(output_path, 'wt', encoding='utf-8') as out:
        for filepath in batch_files:
            try:
                with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                    text = f.read().strip()
                if not text:
                    continue
                text = process_text(text)
                if not text:
                    continue
                doc = {
                    "id": f"{dataset_name}-{Path(filepath).stem}",
                    "text": text,
                    "source": dataset_name,
                }
                out.write(json.dumps(doc) + "\n")
                count += 1
            except Exception as e:
                pass
    return shard_id, count

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect files
    files = sorted([
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR) if f.endswith('.txt')
    ])
    print(f"Found {len(files):,} files")

    # Estimate batch size from sample
    sample_sizes = [os.path.getsize(f) for f in files[:1000]]
    avg_size = sum(sample_sizes) / len(sample_sizes) if sample_sizes else 50000
    batch_size = max(100, int(MAX_SHARD_BYTES / (avg_size * 0.85)))
    print(f"~{batch_size} files per shard")

    # Create batches
    batches = []
    for i in range(0, len(files), batch_size):
        batches.append((files[i:i+batch_size], len(batches), OUTPUT_DIR, DATASET_NAME))

    # Process in parallel
    total = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for future in as_completed({ex.submit(process_batch, b): b for b in batches}):
            _, count = future.result()
            total += count
    print(f"Done: {total:,} docs in {len(batches)} shards")

if __name__ == "__main__":
    main()
```

#### One large JSONL file → Split into shards
SFT/HF datasets often come as a single huge file. Tokenizing one file is extremely slow. Split first:

```python
def partition_jsonl(input_file, max_size_mb=256, output_dir=None):
    """Split JSONL into ~256MB chunks."""
    max_bytes = max_size_mb * 1024 * 1024
    input_path = Path(input_file)
    output_dir = Path(output_dir or input_path.parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    part_num = 0
    current_size = 0
    current_file = None

    with open(input_file, 'r') as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            line_size = len(line.encode('utf-8')) + 1
            if current_file is None or current_size + line_size > max_bytes:
                if current_file:
                    current_file.close()
                output_path = output_dir / f"{input_path.stem}_{part_num:04d}.jsonl"
                current_file = open(output_path, 'w')
                current_size = 0
                part_num += 1
            current_file.write(line + '\n')
            current_size += line_size

    if current_file:
        current_file.close()
    print(f"Created {part_num} shards")
```

#### Hugging Face dataset → Dolma JSONL
```python
from datasets import load_dataset
from jinja2 import Template

# 1. Export to JSONL
ds = load_dataset("org/dataset")
with open("/mnt/raid0/data/raw.jsonl", "w") as f:
    for item in ds['train']:
        f.write(json.dumps(item) + "\n")

# 2. Convert messages to flat text (for SFT data)
EOS_TOKEN = "<|endoftext|>"
TEMPLATE = Template(
    "{% for message in messages %}"
    "{{ '\\n' if not loop.first else '' }}"
    "{{ message['content'] }}"
    "{% if loop.last %}{{ eos_token }}{% endif %}"
    "{% endfor %}"
)

# 3. Reformat into Dolma docs
with open("raw.jsonl") as fin, open("dolma.jsonl", "w") as fout:
    for line in fin:
        item = json.loads(line)
        text = TEMPLATE.render(messages=item["messages"], eos_token=EOS_TOKEN)
        doc = {"id": item["id"], "text": text}
        fout.write(json.dumps(doc) + "\n")

# 4. Split into shards (see partition_jsonl above)
```

For the EOS token, check the base model's tokenizer:
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer")
print(repr(tokenizer.eos_token))
```

### Using datamap-rs for Resharding
If you already have JSONL files of uneven sizes, datamap-rs can normalize them:
```bash
~/datamap-rs/target/release/datamap reshard \
    --input /mnt/raid0/data/uneven_shards/ \
    --output /mnt/raid0/data/resharded/ \
    --max-size-mb 256
```

---

## Phase 5: Tokenize

### Single Subset
```bash
uv run dolma tokens \
    --documents "/mnt/raid0/data/dolma_docs/*.jsonl.gz" \
    --destination "/mnt/raid0/data/tokenized" \
    --tokenizer.name_or_path allenai/dolma2-tokenizer \
    --tokenizer.eos_token_id 100257 \
    --tokenizer.pad_token_id 100277 \
    --no-tokenizer.segment_before_tokenization \
    --tokenizer.encode_special_tokens \
    --processes $(python3 -c "import multiprocessing; print(multiprocessing.cpu_count())") \
    --max_size 1_000_000_000 \
    --sample_ring_prop \
    --dtype uint32
```

### Multiple Subsets in Parallel
When tokenizing multiple subsets on the same machine, split the CPUs:
```bash
# In tmux pane 1
uv run dolma tokens \
    --documents "/mnt/raid0/data/subset_a/*.jsonl.gz" \
    --destination "/mnt/raid0/data/subset_a_tokenized" \
    ... \
    --processes 96

# In tmux pane 2
uv run dolma tokens \
    --documents "/mnt/raid0/data/subset_b/*.jsonl.gz" \
    --destination "/mnt/raid0/data/subset_b_tokenized" \
    ... \
    --processes 96
```

### Tokenizer Parameters Explained
| Parameter | Value | Why |
|-----------|-------|-----|
| `name_or_path` | `allenai/dolma2-tokenizer` | Use HF name directly; local paths can cause validation errors |
| `eos_token_id` | 100257 | End-of-sequence marker inserted after each document |
| `pad_token_id` | 100277 | Padding token for alignment |
| `segment_before_tokenization` | false | Don't split into sentences — feed whole documents |
| `encode_special_tokens` | true | Encode special tokens like `<\|endoftext\|>` into IDs |
| `dtype` | uint32 | Vocabulary > 65535 so uint16 won't fit |
| `max_size` | 1_000_000_000 | 1GB per output .npy file. Smaller = easier to shuffle/mix |
| `sample_ring_prop` | true | Distribute docs proportionally across ring for even file sizes |

### Monitoring Tokenization
```bash
# Check progress (file count growing)
watch -n 10 'ls /mnt/raid0/data/tokenized/*.npy 2>/dev/null | wc -l'

# Check system pressure
top -bn1 | head -5   # look at %wa (iowait) — if >30%, you're IO-bound
free -h               # memory usage
df -h /mnt/raid0/     # disk space remaining
```

---

## Phase 6: Upload to S3 and Clean Up

### Upload
```bash
# Tokenized output (most important)
s5cmd cp "/mnt/raid0/data/tokenized/*" \
    "s3://ai2-llm/preprocessed/{dataset_name}/dolma2-tokenizer/"

# Raw source data (for reproducibility)
s5cmd cp "/mnt/raid0/data/raw/*" \
    "s3://ai2-llm/pretraining-data/sources/{dataset_name}/raw/"

# Dolma docs (intermediate, useful for re-tokenization)
s5cmd cp "/mnt/raid0/data/dolma_docs/*" \
    "s3://ai2-llm/pretraining-data/sources/{dataset_name}/dolma_docs/"

# Processing scripts (for reproducibility)
s5cmd cp /home/ec2-user/convert.py \
    "s3://ai2-llm/pretraining-data/sources/{dataset_name}/scripts/convert.py"
```

### Verify Before Terminating
```bash
# Check every destination has files
aws s3 ls s3://ai2-llm/preprocessed/{dataset_name}/dolma2-tokenizer/ | head -3
aws s3 ls s3://ai2-llm/preprocessed/{dataset_name}/dolma2-tokenizer/ | wc -l

# Spot check a .npy file size (should be close to max_size)
aws s3 ls s3://ai2-llm/preprocessed/{dataset_name}/dolma2-tokenizer/ | head -5
```

### Terminate
```bash
poormanray terminate -n $cluster_name
```

**WARNING**: For NVMe instances (i4i), all local data is permanently lost on terminate. For EBS instances, data persists if you only stop (not terminate), but you still pay for storage.

---

## Distributed Processing (Multiple Nodes)

For very large datasets, use multiple instances with `poormanray map`:

### 1. Create cluster
```bash
poormanray create -n $cluster_name -t i4i.2xlarge --number 10
poormanray setup-d2tk -n $cluster_name -d
poormanray setup-dolma-python -n $cluster_name -d
```

### 2. Create per-node scripts
Create a directory with one bash script per job unit. `poormanray map` distributes them round-robin across instances.

```bash
mkdir -p /tmp/tokenize_jobs/
# Generate one script per shard or per subset
for i in $(seq 0 9); do
cat > /tmp/tokenize_jobs/job_${i}.sh << EOF
#!/bin/bash
s5cmd cp "s3://bucket/shards/part_${i}*" /mnt/raid0/input/
uv run dolma tokens --documents "/mnt/raid0/input/*.jsonl.gz" ...
s5cmd cp "/mnt/raid0/output/*" "s3://bucket/tokenized/part_${i}/"
EOF
chmod +x /tmp/tokenize_jobs/job_${i}.sh
done
```

### 3. Distribute and run
```bash
poormanray map -n $cluster_name --script /tmp/tokenize_jobs/
```

### 4. Monitor
```bash
poormanray run -n $cluster_name -c "tail -f ~/*/run_all.log"
```

---

## Performance Tuning

### IO-Bound (high iowait, low CPU usage)
- Switch to NVMe instance (i4i, i7i)
- Reduce `--processes` to lower IO contention
- Use larger shards (fewer file open/close operations)
- Compress input with gzip (less bytes to read from disk)

### CPU-Bound (high CPU, low iowait)
- Use more vCPUs (larger instance or more nodes)
- Increase `--processes`
- Check if text processing (regex, cleaning) is the bottleneck

### Memory Issues
- Reduce `--processes` (each dolma worker loads data into memory)
- Each worker uses ~1-1.5GB RAM
- 192 processes × 1.5GB = 288GB — close to limit on 384GB instances

### Shard Size Guidelines
| Stage | Recommended Size | Reason |
|-------|-----------------|--------|
| Dolma JSONL shards | ~256MB uncompressed | Sweet spot for parallel processing (datamap-rs) |
| Tokenized .npy files | 1-4GB | Smaller = more flexible mixing; larger = less overhead |
| max_size 1GB | Good default | Easy to shuffle and combine for training mixes |
| max_size 4GB | Convention for large runs | Fewer files to manage |

---

## Troubleshooting

### SSH drops / can't connect
```bash
# Check instance status from local
poormanray list -n $cluster_name
# If running but unresponsive, instance may be overloaded
poormanray run -n $cluster_name -c "uptime"  # uses paramiko, may work when SSH doesn't
poormanray run -n $cluster_name -c "free -h"
```

### "No space left on device"
```bash
df -h /mnt/raid0/
du -sh /mnt/raid0/*/   # find what's using space
# For EBS: resize volume via AWS console/CLI, then growpart + resize2fs
# For NVMe: can't resize, need larger instance or clean up data
```

### dolma tokenizer path errors
- Use HF name `allenai/dolma2-tokenizer` not a local path
- Local path `/mnt/raid0/tokenizer` can fail with HuggingFace validation errors
- If you must use local: point to the directory, not the `.json` file

### s5cmd "flag not defined"
- `s5cmd du` does NOT support `--include`, only `--exclude`
- Workaround: use multiple `--exclude` flags to exclude everything you don't want
- `s5cmd cp` supports both `--include` and `--exclude`

### Conversion script using too much memory
- If passing a large metadata dict to ProcessPoolExecutor workers, each worker gets a copy
- Solution: load metadata once, pass only what each batch needs, or skip metadata