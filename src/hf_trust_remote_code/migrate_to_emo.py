#!/usr/bin/env python3
"""
Migrate an already-uploaded HF checkpoint to the Emo names without re-uploading
the model weights.

For one repo:
  1. Rename the repo if --old-repo-id != --new-repo-id (e.g. ModMoE_* -> Emo_*).
  2. Patch ``config.json``: rewrite ``architectures``, ``model_type``, and
     ``auto_map`` from the old FlexOlmo names to the Emo names.
  3. Upload the Emo-named ``configuration_*.py`` / ``modeling_*.py`` from this
     directory and delete the obsolete FlexOlmo-named ones.

Safetensors / tokenizer files are untouched.

Usage:
    python migrate_to_emo.py \\
        --old-repo-id allenai/ModMoE_1b14b_1T \\
        --new-repo-id allenai/Emo_1b14b_1T
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import (
    EntryNotFoundError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)

SCRIPT_DIR = Path(__file__).resolve().parent

ARCH_TO_FOLDER = {
    "EmoForCausalLM": "emo",
    "Olmo2NoQKNormPrenormForCausalLM": "olmo2_noqknorm_prenorm",
}

ARCH_TO_AUTO_MAP = {
    "EmoForCausalLM": {
        "AutoConfig": "configuration_emo.EmoConfig",
        "AutoModelForCausalLM": "modeling_emo.EmoForCausalLM",
    },
    "Olmo2NoQKNormPrenormForCausalLM": {
        "AutoConfig": "configuration_olmo2_noqknorm_prenorm.Olmo2NoQKNormPrenormConfig",
        "AutoModelForCausalLM": "modeling_olmo2_noqknorm_prenorm.Olmo2NoQKNormPrenormForCausalLM",
    },
}

# Old → new mappings for symbols rewritten by the Emo rename.
OLD_TO_NEW_ARCH = {
    "FlexOlmoNoQKNormPrenormForCausalLM": "EmoForCausalLM",
}
OLD_TO_NEW_MODEL_TYPE = {
    "flex_olmo_noqknorm_prenorm": "emo",
}
# Files renamed in the trust_remote_code dir; the old names should be deleted
# from the repo after the new ones are uploaded.
OBSOLETE_REMOTE_FILES = [
    "configuration_flex_olmo_noqknorm_prenorm.py",
    "modeling_flex_olmo_noqknorm_prenorm.py",
]


def rename_repo_if_needed(api: HfApi, old_id: str, new_id: str, token: str | None) -> str:
    """Move the repo on the Hub if old != new. Returns the id to use afterwards."""
    if old_id == new_id:
        print(f"  repo name unchanged: {new_id}")
        return new_id

    if api.repo_exists(repo_id=new_id, repo_type="model", token=token):
        if api.repo_exists(repo_id=old_id, repo_type="model", token=token):
            sys.exit(
                f"  error: both {old_id} and {new_id} exist on the Hub; "
                f"resolve manually before running this script"
            )
        print(f"  already renamed: {new_id}")
        return new_id

    print(f"  renaming {old_id} -> {new_id}")
    api.move_repo(from_id=old_id, to_id=new_id, repo_type="model", token=token)
    return new_id


def patch_config_json(api: HfApi, repo_id: str, token: str | None) -> str | None:
    """Fetch config.json, rewrite FlexOlmo→Emo fields, push it back. Returns the
    final architecture name (post-patch) or None if config.json is missing.
    """
    with tempfile.TemporaryDirectory() as td:
        try:
            local_cfg = api.hf_hub_download(
                repo_id=repo_id,
                filename="config.json",
                local_dir=td,
                token=token,
            )
        except EntryNotFoundError:
            print("  config.json not found in repo; skipping patch")
            return None
        cfg = json.loads(Path(local_cfg).read_text())

    changed = False

    archs = cfg.get("architectures") or []
    new_archs = [OLD_TO_NEW_ARCH.get(a, a) for a in archs]
    if new_archs != archs:
        cfg["architectures"] = new_archs
        changed = True

    mt = cfg.get("model_type")
    if mt in OLD_TO_NEW_MODEL_TYPE:
        cfg["model_type"] = OLD_TO_NEW_MODEL_TYPE[mt]
        changed = True

    arch = (cfg.get("architectures") or [None])[0]
    if arch in ARCH_TO_AUTO_MAP:
        wanted = ARCH_TO_AUTO_MAP[arch]
        if cfg.get("auto_map") != wanted:
            cfg["auto_map"] = wanted
            changed = True

    if not changed:
        print(f"  config.json already up to date (arch={arch})")
        return arch

    payload = (json.dumps(cfg, indent=2) + "\n").encode()
    api.upload_file(
        path_or_fileobj=payload,
        path_in_repo="config.json",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Emo rename: patch config.json",
        token=token,
    )
    print(f"  patched config.json (arch={arch})")
    return arch


def replace_remote_code(api: HfApi, repo_id: str, arch: str, token: str | None) -> None:
    """Upload the Emo-named .py files; delete the obsolete FlexOlmo-named ones."""
    if arch != "EmoForCausalLM":
        print(f"  no remote-code changes needed for arch={arch}")
        return

    src_dir = SCRIPT_DIR / ARCH_TO_FOLDER[arch]
    if not src_dir.is_dir():
        sys.exit(f"  error: missing trust_remote_code source dir: {src_dir}")

    for py_file in sorted(src_dir.glob("*.py")):
        api.upload_file(
            path_or_fileobj=str(py_file),
            path_in_repo=py_file.name,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Emo rename: upload {py_file.name}",
            token=token,
        )
        print(f"  uploaded {py_file.name}")

    for old_name in OBSOLETE_REMOTE_FILES:
        try:
            api.delete_file(
                path_in_repo=old_name,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Emo rename: delete obsolete {old_name}",
                token=token,
            )
            print(f"  deleted {old_name}")
        except (EntryNotFoundError, HfHubHTTPError) as e:
            # Tolerate already-absent files (idempotent re-runs).
            msg = str(e).lower()
            if "404" in msg or "not found" in msg or "does not exist" in msg:
                print(f"  skip {old_name} (not present)")
            else:
                raise


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--old-repo-id",
        required=True,
        help="Current HF repo id, e.g. 'allenai/ModMoE_1b14b_1T'.",
    )
    p.add_argument(
        "--new-repo-id",
        required=True,
        help="Target HF repo id, e.g. 'allenai/Emo_1b14b_1T'. May equal --old-repo-id.",
    )
    p.add_argument(
        "--token",
        default=None,
        help="HF token. Defaults to the HF_TOKEN env var / cached login.",
    )
    args = p.parse_args()

    api = HfApi(token=args.token)
    print(f"==> {args.old_repo_id} -> {args.new_repo_id}")

    try:
        repo_id = rename_repo_if_needed(api, args.old_repo_id, args.new_repo_id, args.token)
    except RepositoryNotFoundError:
        sys.exit(f"  error: source repo {args.old_repo_id} not found on the Hub")

    arch = patch_config_json(api, repo_id, args.token)
    if arch is None:
        return
    replace_remote_code(api, repo_id, arch, args.token)


if __name__ == "__main__":
    main()
