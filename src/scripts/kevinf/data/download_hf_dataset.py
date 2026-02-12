"""
Download HuggingFace datasets using snapshot_download.

This is ~123x faster than streaming from HF because it downloads files locally
first, then you can process them with convert_arrow_to_jsonl.py.

Usage:
    # Download specific subdirectories in parallel:
    python download_hf_dataset.py \
        --repo croissantllm/croissant_dataset \
        --output-dir /data/croissant/raw \
        --dirs french_303b_1 french_303b_2 code_140b \
        --parallel

    # Download entire dataset:
    python download_hf_dataset.py \
        --repo allenai/dolma \
        --output-dir /data/dolma

    # Enable fast transfer (recommended):
    HF_HUB_ENABLE_HF_TRANSFER=1 python download_hf_dataset.py ...

Arguments:
    --repo          HuggingFace repo ID (e.g., croissantllm/croissant_dataset)
    --output-dir    Local directory to download to
    --dirs          Specific subdirectories to download (optional)
    --parallel      Download each subdir in parallel (one process per dir)
    --max-workers   Max parallel downloads (default: number of dirs)

Output:
    Downloads Arrow files to: <output-dir>/<subdir>/train/*.arrow

Next step:
    python convert_arrow_to_jsonl.py --input-dir <output-dir> --output-dir <jsonl-dir> --data-dirs <dirs>
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

from huggingface_hub import snapshot_download


def download_one_dir(repo: str, output_dir: str, subdir: str) -> str:
    """Download a single subdirectory."""
    allow_patterns = [f"{subdir}/**/*"]
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=output_dir,
        allow_patterns=allow_patterns,
    )
    return subdir


def main():
    parser = argparse.ArgumentParser(
        description="Download HuggingFace datasets using snapshot_download"
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="HF repo (e.g., croissantllm/croissant_dataset)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Local output directory",
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        help="Specific subdirs to download (optional)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Download directories in parallel (one process per dir)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Max parallel downloads (default: number of dirs)",
    )
    args = parser.parse_args()

    print(f"Downloading {args.repo} to {args.output_dir}")
    if args.dirs:
        print(f"Directories: {args.dirs}")

    if args.parallel and args.dirs and len(args.dirs) > 1:
        # Download each directory in parallel
        max_workers = args.max_workers or len(args.dirs)
        print(f"Parallel mode: {max_workers} workers for {len(args.dirs)} directories")

        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(download_one_dir, args.repo, args.output_dir, d): d for d in args.dirs
            }
            for future in as_completed(futures):
                subdir = futures[future]
                try:
                    future.result()
                    print(f"[DONE] {subdir}")
                except Exception as e:
                    print(f"[FAILED] {subdir}: {e}")
    else:
        # Sequential download (single call)
        allow_patterns = None
        if args.dirs:
            allow_patterns = [f"{d}/**/*" for d in args.dirs]

        snapshot_download(
            repo_id=args.repo,
            repo_type="dataset",
            local_dir=args.output_dir,
            allow_patterns=allow_patterns,
        )

    print(f"Download complete: {args.output_dir}")


if __name__ == "__main__":
    main()
