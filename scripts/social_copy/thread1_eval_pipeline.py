#!/usr/bin/env python3
"""Thread 1 figure: combined eval-pipeline + Pareto plot with social caption.

Merges the original thread 1 (EMO architecture diagram) and thread 7 (130B
MMLU memory-accuracy Pareto) into a single figure by rendering the existing
``eval_pipeline_V2_with_pareto.pdf`` to PNG and adding a header caption that
covers both stories.

Output: claude_outputs/social_post/thread1_eval_pipeline.png
"""

from __future__ import annotations

from pathlib import Path

from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PDF = (
    REPO_ROOT / "claude_outputs" / "other_figures"
    / "eval_pipeline_V2_with_pareto.pdf"
)
OUTPUT_DIR = REPO_ROOT / "claude_outputs" / "social_post"
OUTPUT = OUTPUT_DIR / "thread1_eval_pipeline.png"

DPI = 220

HEADER_TITLE = (
    "EMO: one MoE → many domain-specific subsets, "
    "with a stronger memory-accuracy trade-off"
)
# Stacked subtitle. Line 1 sells the modularity story (thread 1); lines 2-3
# explain the Pareto plot's reference points (thread 7).
HEADER_SUBTITLE_LINES = (
    "Pick the experts relevant to your task — each subset functions as a standalone model.",
    "EMO / Reg. MoE: a single model evaluated across different expert subset sizes.",
    "Dense @8 / Reg. MoE @32: separate models trained from scratch at a fixed expert budget.",
)

BG_COLOR = (255, 255, 255)
TITLE_COLOR = (26, 32, 44)
SUBTITLE_COLOR = (85, 96, 112)


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _wrap_text(
    text: str, font: ImageFont.FreeTypeFont, max_width: int, draw: ImageDraw.ImageDraw
) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        w = draw.textbbox((0, 0), candidate, font=font)[2]
        if w <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def main() -> None:
    pages = convert_from_path(str(SOURCE_PDF), dpi=DPI)
    src = pages[0].convert("RGB")
    W, H = src.size

    # Title sized for the wider canvas; subtitle smaller.
    title_font = _load_font(int(W * 0.024), bold=True)
    subtitle_font = _load_font(int(W * 0.014), bold=False)

    pad_top = int(H * 0.06)
    pad_between = int(H * 0.025)
    pad_line = int(H * 0.008)
    pad_bottom = int(H * 0.04)
    side_pad = int(W * 0.04)
    text_max_w = W - 2 * side_pad

    dummy = Image.new("RGB", (10, 10))
    d = ImageDraw.Draw(dummy)
    title_lines = _wrap_text(HEADER_TITLE, title_font, text_max_w, d)
    subtitle_lh = d.textbbox((0, 0), "Hg", font=subtitle_font)[3]
    title_lh = d.textbbox((0, 0), "Hg", font=title_font)[3]

    title_block_h = (
        len(title_lines) * title_lh + (len(title_lines) - 1) * pad_line
    )
    subtitle_block_h = (
        len(HEADER_SUBTITLE_LINES) * subtitle_lh
        + (len(HEADER_SUBTITLE_LINES) - 1) * pad_line
    )

    header_h = (
        pad_top + title_block_h + pad_between + subtitle_block_h + pad_bottom
    )
    out_h = header_h + H

    canvas = Image.new("RGB", (W, out_h), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    y = pad_top
    for line in title_lines:
        lw = draw.textbbox((0, 0), line, font=title_font)[2]
        draw.text(((W - lw) // 2, y), line, fill=TITLE_COLOR, font=title_font)
        y += title_lh + pad_line

    y = pad_top + title_block_h + pad_between
    for line in HEADER_SUBTITLE_LINES:
        lw = draw.textbbox((0, 0), line, font=subtitle_font)[2]
        draw.text(
            ((W - lw) // 2, y),
            line,
            fill=SUBTITLE_COLOR,
            font=subtitle_font,
        )
        y += subtitle_lh + pad_line

    canvas.paste(src, (0, header_h))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT, format="PNG", optimize=True)
    print(f"Wrote {OUTPUT}  ({canvas.size[0]}x{canvas.size[1]})")


if __name__ == "__main__":
    main()
