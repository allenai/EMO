#!/usr/bin/env python3
"""
Upload a Emo-trained HF checkpoint to the Hugging Face Hub with the
configuration_*.py / modeling_*.py scripts needed for
``trust_remote_code=True`` loading.

Detects the architecture from ``<model_path>/config.json``, copies the matching
remote-code files into the checkpoint directory, patches ``config.json`` with
the appropriate ``auto_map`` entry, and pushes the folder to the given HF repo.

Usage:
    python upload_to_hf.py \\
        --model-path /path/to/dense_1b_lr-4e-3_0213/step30995-hf \\
        --repo-id    allenai/my-checkpoint
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

SCRIPT_DIR = Path(__file__).resolve().parent

ARCH_TO_FOLDER = {
    "EmoNoQKNormPrenormForCausalLM": "emo_noqknorm_prenorm",
    "Olmo2NoQKNormPrenormForCausalLM": "olmo2_noqknorm_prenorm",
}

ARCH_TO_AUTO_MAP = {
    "EmoNoQKNormPrenormForCausalLM": {
        "AutoConfig": "configuration_emo_noqknorm_prenorm.EmoNoQKNormPrenormConfig",
        "AutoModelForCausalLM": "modeling_emo_noqknorm_prenorm.EmoNoQKNormPrenormForCausalLM",
    },
    "Olmo2NoQKNormPrenormForCausalLM": {
        "AutoConfig": "configuration_olmo2_noqknorm_prenorm.Olmo2NoQKNormPrenormConfig",
        "AutoModelForCausalLM": "modeling_olmo2_noqknorm_prenorm.Olmo2NoQKNormPrenormForCausalLM",
    },
}


def detect_architecture(config_path: Path) -> str:
    cfg = json.loads(config_path.read_text())
    archs = cfg.get("architectures") or []
    if len(archs) != 1:
        raise ValueError(f"Expected exactly one architecture in {config_path}, got: {archs}")
    arch = archs[0]
    if arch not in ARCH_TO_FOLDER:
        raise ValueError(
            f"Unsupported architecture {arch!r}. " f"Supported: {sorted(ARCH_TO_FOLDER)}"
        )
    return arch


def stage_remote_code(model_path: Path, arch: str) -> list[Path]:
    """Copy modeling/config .py files into model_path and patch config.json's auto_map.

    Returns the list of files written/modified inside ``model_path``.
    """
    src_dir = SCRIPT_DIR / ARCH_TO_FOLDER[arch]
    if not src_dir.is_dir():
        raise FileNotFoundError(f"missing trust_remote_code source dir: {src_dir}")

    written: list[Path] = []
    for py_file in sorted(src_dir.glob("*.py")):
        dest = model_path / py_file.name
        shutil.copy2(py_file, dest)
        written.append(dest)
        print(f"  copied {py_file.name}")

    config_path = model_path / "config.json"
    cfg = json.loads(config_path.read_text())
    cfg["auto_map"] = ARCH_TO_AUTO_MAP[arch]
    config_path.write_text(json.dumps(cfg, indent=2) + "\n")
    written.append(config_path)
    print(f"  patched config.json: auto_map -> {arch}")

    return written


def upload(
    model_path: Path,
    repo_id: str,
    private: bool,
    commit_message: str,
    token: str | None,
) -> None:
    api = HfApi(token=token)
    create_repo(
        repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
        token=token,
    )
    print(f"uploading {model_path} -> https://huggingface.co/{repo_id}")
    api.upload_folder(
        folder_path=str(model_path),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
    print("done.")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model-path",
        required=True,
        type=Path,
        help="Local directory containing the HF checkpoint (must hold config.json).",
    )
    p.add_argument(
        "--repo-id",
        required=True,
        help="Target HF repo id, e.g. 'allenai/my-model'.",
    )
    p.add_argument(
        "--public",
        dest="private",
        action="store_false",
        help="Create the repo as public. Default is private.",
    )
    p.set_defaults(private=True)
    p.add_argument(
        "--commit-message",
        default="Add model and trust_remote_code files.",
    )
    p.add_argument(
        "--token",
        default=None,
        help="HF token. Defaults to the HF_TOKEN env var / cached login.",
    )
    p.add_argument(
        "--no-upload",
        action="store_true",
        help="Stage the trust_remote_code files locally only; skip the HF push.",
    )
    args = p.parse_args()

    model_path: Path = args.model_path.resolve()
    if not model_path.is_dir():
        sys.exit(f"error: {model_path} is not a directory")
    if not (model_path / "config.json").exists():
        sys.exit(f"error: {model_path}/config.json not found")

    arch = detect_architecture(model_path / "config.json")
    print(f"architecture: {arch}")

    stage_remote_code(model_path, arch)

    if args.no_upload:
        print("--no-upload set; skipping push.")
        return

    upload(
        model_path,
        args.repo_id,
        args.private,
        args.commit_message,
        args.token,
    )


if __name__ == "__main__":
    main()
