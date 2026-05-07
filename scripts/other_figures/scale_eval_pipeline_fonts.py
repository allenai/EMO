#!/usr/bin/env python3
"""Bump text/cell sizes inside eval_pipeline.html.

The figure source (a JSX asset) is bundled into eval_pipeline.html as a
gzip+base64-encoded entry under the ``__bundler/manifest`` script tag.
This script decodes that entry, applies a fixed set of substitutions
that scale up font sizes and the layout chrome that holds them, then
re-encodes and writes the file back in place.

After running, regenerate the cropped PDF via:

    python scripts/other_figures/crop_html_to_pdf.py \
        --x1 230 --y1 20 --x2 2200 --y2 660

(adjust crop coords if the layout grows beyond these bounds — use
``--preview`` to read off new bounds.)
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HTML = REPO_ROOT / "claude_outputs" / "other_figures" / "eval_pipeline.html"
FIGURE_UUID = "1aa301c7-f791-46ea-9374-8f1ad251b944"

# Each entry must match exactly once in the JSX (we assert that). Order
# is irrelevant for correctness but kept readable.
JSX_REPLACEMENTS: List[Tuple[str, str]] = [
    # ── Cell + chrome sizing (modest 1.18x) ──────────────────────────
    ("const POOL_CELL_W = 78;", "const POOL_CELL_W = 92;"),
    ("const POOL_CELL_H = 86;", "const POOL_CELL_H = 102;"),
    ("const MOD_CELL_W = 70;",  "const MOD_CELL_W = 82;"),
    ("const MOD_CELL_H = 78;",  "const MOD_CELL_H = 92;"),
    ("const ROUTER_W = 150;",   "const ROUTER_W = 220;"),
    ("const ROUTER_H = 30;",    "const ROUTER_H = 44;"),
    ("const MOD_ROUTER_W = 90;", "const MOD_ROUTER_W = 130;"),
    ("const MOD_ROUTER_H = 20;", "const MOD_ROUTER_H = 30;"),
    ("const PAD_BOT = 64;",     "const PAD_BOT = 96;"),

    # ── ECell baseline + label/subscript sizes (1.5x text) ───────────
    ("labelSize = 28, subSize = 18", "labelSize = 42, subSize = 27"),
    ("y={cy + 8}", "y={cy + 12}"),

    # ── Router-pill text baselines (proportional to taller pills) ────
    ("y={ry + ROUTER_H / 2 + 6}", "y={ry + ROUTER_H / 2 + 9}"),
    ("y={ry + MOD_ROUTER_H / 2 + 5}", "y={ry + MOD_ROUTER_H / 2 + 7}"),

    # ── Inline SVG/JSX font sizes ────────────────────────────────────
    # Module Router (15 -> 22)
    ("fontSize: 15, fontStyle: 'italic',", "fontSize: 22, fontStyle: 'italic',"),
    # Pool Router (20 -> 30) — 29 spaces of indent in the JSX.
    ("                             fontSize: 20, fontStyle: 'italic',",
     "                             fontSize: 30, fontStyle: 'italic',"),
    # Caption subtitle (20 -> 30) — 23 spaces of indent.
    ("                       fontSize: 20, fontStyle: 'italic',",
     "                       fontSize: 30, fontStyle: 'italic',"),
    # Domain label (30 -> 45, weight 600)
    ("fontSize: 30, fontWeight: 600,", "fontSize: 45, fontWeight: 600,"),
    # Caption title (30 -> 45, weight 500)
    ("fontSize: 30, fontWeight: 500,", "fontSize: 45, fontWeight: 500,"),
    # "X experts" mono caption (17 -> 25)
    ("mono(17, MUTED)", "mono(25, MUTED)"),

    # ── foreignObject (domain label container) ───────────────────────
    ("y={my + MOD_BOX_H / 2 - 32}", "y={my + MOD_BOX_H / 2 - 48}"),
    ('height="64">', 'height="96">'),

    # ── Caption y offsets below pool box (proportional to bigger caption)
    ("y={poolY + poolBoxH + 32}", "y={poolY + poolBoxH + 48}"),
    ("y={poolY + poolBoxH + 56}", "y={poolY + poolBoxH + 84}"),
]

# In the bundler template the host scaler uses DESIGN_W=1900 to decide
# when to shrink-to-fit. Bump it so our wider figure renders at scale 1.
TEMPLATE_REPLACEMENTS: List[Tuple[str, str]] = [
    ("const DESIGN_W = 1900;", "const DESIGN_W = 2200;"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--dry-run", action="store_true",
                        help="Verify substitutions match without writing.")
    return parser.parse_args()


def _apply(text: str, replacements: List[Tuple[str, str]], label: str) -> str:
    for old, new in replacements:
        count = text.count(old)
        if count == 0:
            raise SystemExit(f"[{label}] substitution did not match: {old!r}")
        if count > 1:
            raise SystemExit(f"[{label}] substitution matched {count} times (expected 1): {old!r}")
        text = text.replace(old, new)
    return text


def main() -> None:
    args = parse_args()
    html_text = args.html.read_text()
    lines = html_text.splitlines(keepends=True)

    # ── Locate the manifest script tag and the JSON payload line. ────
    # Structure (from the file): line indices are 0-based.
    #   ...
    #   <script type="__bundler/manifest">  (line ~177)
    #   <giant JSON>                         (line ~178)
    #   </script>                            (line ~179)
    #   ...
    manifest_idx = None
    for i, line in enumerate(lines):
        if '<script type="__bundler/manifest">' in line:
            # Manifest JSON is on the very next line.
            manifest_idx = i + 1
            break
    if manifest_idx is None:
        raise SystemExit("Could not find __bundler/manifest script tag")

    manifest_line = lines[manifest_idx].rstrip("\n").rstrip("\r")
    manifest = json.loads(manifest_line)
    if FIGURE_UUID not in manifest:
        raise SystemExit(f"Manifest missing figure UUID {FIGURE_UUID}")

    entry = manifest[FIGURE_UUID]
    raw = base64.b64decode(entry["data"])
    if entry.get("compressed"):
        raw = gzip.decompress(raw)
    src = raw.decode("utf-8")

    # ── Apply substitutions to the JSX. ──────────────────────────────
    new_src = _apply(src, JSX_REPLACEMENTS, "JSX")

    # Re-encode (gzip + base64) and stash back into the manifest entry.
    encoded = new_src.encode("utf-8")
    if entry.get("compressed"):
        encoded = gzip.compress(encoded)
    entry["data"] = base64.b64encode(encoded).decode("ascii")
    new_manifest_line = json.dumps(manifest, separators=(",", ":"))
    lines[manifest_idx] = new_manifest_line + "\n"

    # ── Apply substitutions to the bundler template (line 187 region).
    # The template lives inside <script type="__bundler/template">. It's
    # a single very long JSON string. We rewrite the line in place.
    template_idx = None
    for i, line in enumerate(lines):
        if '<script type="__bundler/template">' in line:
            template_idx = i + 1
            break
    if template_idx is None:
        raise SystemExit("Could not find __bundler/template script tag")

    template_line = lines[template_idx]
    # Template is a JSON-encoded HTML string (starts with `"` ends with `"\n`).
    inner = json.loads(template_line)
    inner = _apply(inner, TEMPLATE_REPLACEMENTS, "template")
    # IMPORTANT: this JSON literal is embedded inside a <script> tag, so any
    # raw "</" sequence (e.g. "</script>") would prematurely end the tag.
    # The original tool encoded "</" as "<\u002F" — preserve that.
    encoded_template = json.dumps(inner).replace("</", "<\\u002F")
    lines[template_idx] = encoded_template + "\n"

    if args.dry_run:
        print("Dry run: all substitutions matched. No files written.")
        return

    args.html.write_text("".join(lines))
    print(f"Updated {args.html}")
    print(
        "Now regenerate the cropped PDF with:\n"
        "  python scripts/other_figures/crop_html_to_pdf.py \\\n"
        "      --x1 230 --y1 20 --x2 2200 --y2 660"
    )


if __name__ == "__main__":
    main()
