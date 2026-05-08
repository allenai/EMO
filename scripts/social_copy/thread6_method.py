#!/usr/bin/env python3
"""Thread 6 figure: method_v2 copied to the social-post folder, no caption.

The figure already labels Standard MoE / EMO and shows the document-pool
constraint visually, so it stands on its own next to the thread text.

Output: claude_outputs/social_post/thread6_method.png
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "claude_outputs" / "other_figures" / "method_v2.png"
OUTPUT_DIR = REPO_ROOT / "claude_outputs" / "social_post"
OUTPUT = OUTPUT_DIR / "thread6_method.png"


def main() -> None:
    src = Image.open(SOURCE).convert("RGB")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    src.save(OUTPUT, format="PNG", optimize=True)
    print(f"Wrote {OUTPUT}  ({src.size[0]}x{src.size[1]})")


if __name__ == "__main__":
    main()
