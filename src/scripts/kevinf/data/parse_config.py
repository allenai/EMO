"""
Parse a tokenization YAML config and emit shell export statements.

Reads a YAML config file and prints `export VAR=value` lines that can be
eval'd by bash scripts to load all configuration as environment variables.

Usage:
    eval $(python3 parse_config.py configs/mimic-iv-note.yaml)

    # Then use in bash:
    echo $DATASET_NAME      # mimic-iv-note
    echo $SOURCE_FORMAT      # csv
    echo $NUM_PROCESSES      # 192

Dependencies: PyYAML (pip install pyyaml) — available on EC2 instances via
dolma's dependencies. Falls back to a simple built-in parser for the subset
of YAML used by tokenization configs.
"""

import re
import sys


def _parse_yaml_simple(text: str) -> dict:
    """Minimal YAML parser for our config subset (nested dicts, scalars, lists).

    Handles:
      - Nested keys via indentation
      - Scalar values (strings, ints, bools)
      - Inline lists: [a, b, c]
      - Block list items: - value
      - Comments (#)
      - Quoted strings
    """
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]

    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1

        # Strip comments (but not inside quotes)
        stripped = line.split("#")[0] if '"' not in line and "'" not in line else line
        if not stripped.strip():
            continue

        indent = len(stripped) - len(stripped.lstrip())
        stripped = stripped.strip()

        # Skip pure comment lines
        if stripped.startswith("#"):
            continue

        # Pop stack to find parent
        while stack and stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1] if stack else root

        if ":" in stripped:
            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip()

            if not val:
                # Nested dict — check if next non-empty line is a list item
                new_dict: dict = {}
                parent[key] = new_dict
                stack.append((indent, new_dict))
            elif val.startswith("[") and val.endswith("]"):
                # Inline list
                items = val[1:-1]
                if items.strip():
                    parent[key] = [_parse_scalar(x.strip()) for x in items.split(",")]
                else:
                    parent[key] = []
            else:
                parent[key] = _parse_scalar(val)
        elif stripped.startswith("- "):
            # Block list item — convert parent's last key from dict to list
            val = _parse_scalar(stripped[2:].strip())
            # Find the key in grandparent that points to parent
            grandparent = stack[-2][1] if len(stack) >= 2 else root
            for k, v in grandparent.items():
                if v is parent:
                    if not isinstance(grandparent[k], list):
                        grandparent[k] = []
                    grandparent[k].append(val)
                    # Update stack reference
                    stack[-1] = (stack[-1][0], grandparent[k])
                    parent = grandparent[k]
                    break

    return root


def _parse_scalar(val: str):
    """Parse a YAML scalar value."""
    # Strip quotes
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        return val[1:-1]
    # Strip inline comments
    for sep in ["  #", "\t#"]:
        if sep in val:
            val = val[: val.index(sep)].strip()
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    if val == "":
        return ""
    try:
        return int(val.replace("_", ""))
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


def load_yaml(path: str) -> dict:
    """Load YAML config, using PyYAML if available, else simple parser."""
    with open(path) as f:
        text = f.read()
    try:
        import yaml

        return yaml.safe_load(text)
    except ImportError:
        return _parse_yaml_simple(text)


def quote(val: str) -> str:
    """Shell-quote a string value."""
    return f'"{val}"'


def parse_config(path: str) -> dict[str, str]:
    """Parse YAML config into flat key=value env vars."""
    cfg = load_yaml(path)

    env = {}

    # dataset
    env["DATASET_NAME"] = cfg["dataset"]["name"]

    # source
    src = cfg.get("source", {})
    env["SOURCE_FORMAT"] = src.get("format", "jsonl")
    env["S3_SOURCE"] = src.get("s3_path", "")
    env["LOCAL_SOURCE"] = src.get("local_path", "")
    env["DOCUMENTS_PATTERN"] = src.get("documents_pattern", "")
    env["TEXT_FIELD"] = src.get("text_field", "text")
    env["ID_FIELD"] = src.get("id_field", "id")
    env["METADATA_FIELDS"] = " ".join(src.get("metadata_fields", []))
    env["FILES"] = " ".join(src.get("files", []))
    env["DATA_DIRS"] = " ".join(src.get("data_dirs", []))

    # destination
    dest = cfg.get("destination", {})
    env["LOCAL_DOLMA_DOCS"] = dest.get("local_dolma_docs", "")
    env["LOCAL_TOKENIZED"] = dest.get("local_tokenized", "")
    env["S3_DOLMA_DEST"] = dest.get("s3_dolma_docs", "")
    env["S3_TOKENIZED_DEST"] = dest.get("s3_tokenized", "")

    # compute
    comp = cfg.get("compute", {})
    env["COMPUTE_MODE"] = comp.get("mode", "local")
    env["INSTANCE_TYPE"] = comp.get("instance_type", "c6a.48xlarge")
    env["STORAGE_SIZE"] = str(comp.get("storage_size", 200))
    env["NUM_PROCESSES"] = str(comp.get("num_processes", 200))
    env["CONVERT_WORKERS"] = str(comp.get("convert_workers", 64))
    env["DOCS_PER_SHARD"] = str(comp.get("docs_per_shard", 50000))

    # tokenizer
    tok = cfg.get("tokenizer", {})
    env["TOKENIZER_NAME"] = tok.get("name", "allenai/dolma2-tokenizer")
    env["EOS_TOKEN_ID"] = str(tok.get("eos_token_id", 100257))
    env["PAD_TOKEN_ID"] = str(tok.get("pad_token_id", 100277))
    env["DTYPE"] = tok.get("dtype", "uint32")
    env["USE_UV"] = str(tok.get("use_uv", False)).lower()
    env["DOLMA_EXTRA_ARGS"] = " ".join(tok.get("extra_args", []))

    return env


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <config.yaml>", file=sys.stderr)
        sys.exit(1)

    env = parse_config(sys.argv[1])
    for key, val in env.items():
        print(f"export {key}={quote(val)}")


if __name__ == "__main__":
    main()
