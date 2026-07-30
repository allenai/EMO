#!/usr/bin/env python3
"""Crop an HTML render to a user-specified pixel rectangle and save as PDF.

Workflow:

  1) Run with ``--preview`` to get a full screenshot with axis tick marks. Open
     the screenshot in any image viewer and read off the pixel coordinates of
     the top-left and bottom-right corners of the region you want.
  2) Run again with ``--x1 ... --y2 ...`` to crop the HTML render to that
     rectangle and save it as a PDF whose page size equals the crop.
  3) Iterate.

Examples
--------

  # 1. Generate a preview to read coordinates off (saved next to the PDF):
  python scripts/other_figures/crop_html_to_pdf.py \\
    --html  claude_outputs/other_figures/eval_pipeline.html \\
    --output claude_outputs/other_figures/eval_pipeline.pdf \\
    --preview

  # 2. Crop with explicit coordinates (top-left=274,24, bottom-right=2005,524):
  python scripts/other_figures/crop_html_to_pdf.py \\
    --html  claude_outputs/other_figures/eval_pipeline.html \\
    --output claude_outputs/other_figures/eval_pipeline.pdf \\
    --x1 274 --y1 24 --x2 2005 --y2 524

Defaults are tuned for the eval_pipeline figure that lives at
``claude_outputs/other_figures/eval_pipeline.html`` — running with no flags is
equivalent to the last working coordinates.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HTML = REPO_ROOT / "claude_outputs" / "other_figures" / "eval_pipeline.html"
DEFAULT_PDF = REPO_ROOT / "claude_outputs" / "other_figures" / "eval_pipeline.pdf"
CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


# --- Minimal Chrome-DevTools-Protocol websocket client ---------------------
class _CDP:
    def __init__(self, ws_url: str):
        from urllib.parse import urlparse

        u = urlparse(ws_url)
        self.s = socket.create_connection((u.hostname, u.port))
        key = base64.b64encode(os.urandom(16)).decode()
        handshake = (
            f"GET {u.path} HTTP/1.1\r\n"
            f"Host: {u.hostname}:{u.port}\r\n"
            "Upgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        self.s.sendall(handshake.encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            buf += self.s.recv(4096)
        self._id = 0
        self._buf = b""

    def call(self, method: str, params: Optional[dict] = None) -> dict:
        self._id += 1
        msg = json.dumps({"id": self._id, "method": method, "params": params or {}}).encode()
        mask = os.urandom(4)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(msg))
        if len(masked) < 126:
            hdr = bytes([0x81, 0x80 | len(masked)]) + mask
        elif len(masked) < 65536:
            hdr = bytes([0x81, 0xFE]) + struct.pack("!H", len(masked)) + mask
        else:
            hdr = bytes([0x81, 0xFF]) + struct.pack("!Q", len(masked)) + mask
        self.s.sendall(hdr + masked)
        while True:
            while len(self._buf) < 2:
                self._buf += self.s.recv(8192)
            ln = self._buf[1] & 0x7F
            off = 2
            if ln == 126:
                while len(self._buf) < 4:
                    self._buf += self.s.recv(8192)
                ln = struct.unpack("!H", self._buf[2:4])[0]
                off = 4
            elif ln == 127:
                while len(self._buf) < 10:
                    self._buf += self.s.recv(8192)
                ln = struct.unpack("!Q", self._buf[2:10])[0]
                off = 10
            while len(self._buf) < off + ln:
                self._buf += self.s.recv(8192)
            payload = self._buf[off : off + ln]
            self._buf = self._buf[off + ln :]
            d = json.loads(payload)
            if d.get("id") == self._id:
                return d


def _launch_chrome() -> Tuple[subprocess.Popen, _CDP]:
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    profile_dir = f"/tmp/chrome_cdp_{port}"
    proc = subprocess.Popen(
        [
            CHROME_BIN,
            "--headless=new",
            "--disable-gpu",
            f"--remote-debugging-port={port}",
            f"--user-data-dir={profile_dir}",
            "--no-first-run",
            "--no-default-browser-check",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(50):
        try:
            urllib.request.urlopen(f"http://localhost:{port}/json/version", timeout=0.5).read()
            break
        except Exception:
            time.sleep(0.2)
    else:
        proc.terminate()
        raise RuntimeError("Chrome did not come up on the debugging port")
    tabs = json.loads(urllib.request.urlopen(f"http://localhost:{port}/json").read())
    target = next((t for t in tabs if t["type"] == "page"), None)
    if target is None:
        proc.terminate()
        raise RuntimeError("No 'page' target in Chrome")
    return proc, _CDP(target["webSocketDebuggerUrl"])


def _navigate_and_wait(
    cdp: _CDP, html: Path, viewport_w: int, viewport_h: int, scale: int, wait_seconds: float
) -> None:
    cdp.call("Page.enable")
    cdp.call("Runtime.enable")
    cdp.call(
        "Emulation.setDeviceMetricsOverride",
        {
            "width": viewport_w,
            "height": viewport_h,
            "deviceScaleFactor": scale,
            "mobile": False,
        },
    )
    cdp.call("Page.navigate", {"url": f"file://{html.resolve()}"})
    time.sleep(wait_seconds)


def _save_preview(cdp: _CDP, viewport_w: int, viewport_h: int, scale: int, out_path: Path) -> None:
    """Full-viewport screenshot with axis ticks every 100 px."""
    res = cdp.call("Page.captureScreenshot", {"format": "png"})
    png_b = base64.b64decode(res["result"]["data"])
    from PIL import Image, ImageDraw, ImageFont

    im = Image.open_io = Image.open  # quiet a linter
    from io import BytesIO

    im = Image.open(BytesIO(png_b)).convert("RGB")
    draw = ImageDraw.Draw(im)
    # Try to load a default font; fall back if unavailable.
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18 * max(1, scale))
    except Exception:
        font = ImageFont.load_default()
    color = (220, 50, 50)
    grid = (220, 220, 220)
    step = 100 * scale
    # Light grid every 100 css px:
    for x in range(0, im.size[0], step):
        draw.line([(x, 0), (x, im.size[1])], fill=grid, width=1)
    for y in range(0, im.size[1], step):
        draw.line([(0, y), (im.size[0], y)], fill=grid, width=1)
    # Tick labels (CSS px units, ie. divide by `scale`):
    for x in range(0, im.size[0], step):
        draw.text((x + 4, 4), f"x={x // scale}", fill=color, font=font)
    for y in range(0, im.size[1], step):
        draw.text((4, y + 4), f"y={y // scale}", fill=color, font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)


def _crop_to_pdf(
    cdp: _CDP, x1: int, y1: int, x2: int, y2: int, scale: int, out_pdf: Path, dpi: int
) -> Tuple[float, float]:
    """Raster path: capture clip (x1,y1)→(x2,y2) as PNG and save as a tightly-sized PDF."""
    w_px = x2 - x1
    h_px = y2 - y1
    if w_px <= 0 or h_px <= 0:
        raise SystemExit(f"Bad crop rect: {x1},{y1} → {x2},{y2}")
    clip = {"x": x1, "y": y1, "width": w_px, "height": h_px, "scale": scale}
    res = cdp.call(
        "Page.captureScreenshot",
        {"format": "png", "clip": clip, "captureBeyondViewport": True},
    )
    png_b = base64.b64decode(res["result"]["data"])
    from io import BytesIO

    from PIL import Image

    im = Image.open(BytesIO(png_b)).convert("RGB")
    page_w_in = w_px / 96.0
    page_h_in = h_px / 96.0
    target_w_px = round(page_w_in * dpi)
    target_h_px = round(page_h_in * dpi)
    im = im.resize((target_w_px, target_h_px), Image.LANCZOS)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_pdf, "PDF", resolution=float(dpi))
    return page_w_in, page_h_in


def _crop_to_pdf_vector(
    cdp: _CDP, x1: int, y1: int, x2: int, y2: int, viewport_w: int, viewport_h: int, out_pdf: Path
) -> Tuple[float, float]:
    """Vector path: full-page Page.printToPDF, then crop the MediaBox/CropBox.

    Chrome's printToPDF preserves SVG/text as vector data, so the output is a
    real vector PDF. We print at paper size = viewport size so positions match
    what we measured at preview time, then use pypdf to set MediaBox+CropBox to
    just the requested CSS-coordinate rectangle (translated to PDF pt with the
    y-axis flip).
    """
    w_px = x2 - x1
    h_px = y2 - y1
    if w_px <= 0 or h_px <= 0:
        raise SystemExit(f"Bad crop rect: {x1},{y1} → {x2},{y2}")

    paper_w_in = viewport_w / 96.0
    paper_h_in = viewport_h / 96.0
    res = cdp.call(
        "Page.printToPDF",
        {
            "paperWidth": paper_w_in,
            "paperHeight": paper_h_in,
            "marginTop": 0,
            "marginBottom": 0,
            "marginLeft": 0,
            "marginRight": 0,
            "printBackground": True,
            "scale": 1.0,
            "preferCSSPageSize": False,
        },
    )
    pdf_full = base64.b64decode(res["result"]["data"])
    full_path = out_pdf.with_suffix(".full.pdf")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_bytes(pdf_full)

    # Now crop the page with pypdf. PDF coordinates: 1px(css) ≈ 0.75pt; origin
    # is bottom-left, while CSS origin is top-left → flip y.
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import RectangleObject

    reader = PdfReader(str(full_path))
    page = reader.pages[0]
    page_h_pt = float(page.mediabox.height)
    # Convert CSS px → PDF pt (1 css px = 1/96 in = 72/96 pt = 0.75 pt).
    PT_PER_PX = 72.0 / 96.0
    pdf_x1 = x1 * PT_PER_PX
    pdf_x2 = x2 * PT_PER_PX
    pdf_y_top = y1 * PT_PER_PX  # top edge in CSS (smaller y)
    pdf_y_bot = y2 * PT_PER_PX  # bottom edge in CSS (larger y)
    new_box = RectangleObject(
        [
            pdf_x1,
            page_h_pt - pdf_y_bot,
            pdf_x2,
            page_h_pt - pdf_y_top,
        ]
    )
    page.mediabox = new_box
    page.cropbox = new_box
    writer = PdfWriter()
    writer.add_page(page)
    with out_pdf.open("wb") as f:
        writer.write(f)
    full_path.unlink()  # remove the uncropped intermediate
    return w_px / 96.0, h_px / 96.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--html", type=Path, default=DEFAULT_HTML, help="Source HTML file.")
    p.add_argument("--output", type=Path, default=DEFAULT_PDF, help="Output PDF path.")
    p.add_argument("--viewport-width", type=int, default=2400)
    p.add_argument("--viewport-height", type=int, default=1200)
    p.add_argument(
        "--scale",
        type=int,
        default=2,
        help="Device-scale-factor for the screenshot (capture DPI multiplier). 2 ≈ retina.",
    )
    p.add_argument(
        "--dpi", type=int, default=300, help="DPI of the final PDF (raster resampling target)."
    )
    p.add_argument(
        "--wait-seconds",
        type=float,
        default=15.0,
        help="How long to wait after navigation for JS-driven content to render.",
    )
    # Crop rectangle in CSS pixel coordinates (NOT screenshot pixels).
    # Defaults are the canonical crop for eval_pipeline.html.
    p.add_argument("--x1", type=int, default=270, help="Top-left X (CSS px).")
    p.add_argument("--y1", type=int, default=40, help="Top-left Y (CSS px).")
    p.add_argument("--x2", type=int, default=1840, help="Bottom-right X (CSS px).")
    p.add_argument("--y2", type=int, default=480, help="Bottom-right Y (CSS px).")
    p.add_argument(
        "--preview",
        action="store_true",
        help="Save a full-viewport screenshot annotated with axis ticks "
        "next to the output PDF and exit. Use this to read off coordinates.",
    )
    p.add_argument(
        "--vector",
        action="store_true",
        help="Produce a vector PDF (Chrome printToPDF + MediaBox crop) "
        "instead of the default raster PNG-in-PDF. Output is much "
        "sharper at any zoom and usually smaller for SVG-heavy pages.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.html.is_file():
        sys.exit(f"HTML not found: {args.html}")

    proc, cdp = _launch_chrome()
    try:
        _navigate_and_wait(
            cdp,
            args.html,
            args.viewport_width,
            args.viewport_height,
            args.scale,
            args.wait_seconds,
        )
        if args.preview:
            preview_path = args.output.with_name(args.output.stem + "_preview.png")
            _save_preview(cdp, args.viewport_width, args.viewport_height, args.scale, preview_path)
            print(f"[preview] wrote {preview_path}")
            print(
                "Open that image, read off pixel coordinates of the top-left and "
                "bottom-right corners of your desired crop region, then re-run with:"
            )
            print("  --x1 ... --y1 ... --x2 ... --y2 ...")
            return
        if args.vector:
            w_in, h_in = _crop_to_pdf_vector(
                cdp,
                args.x1,
                args.y1,
                args.x2,
                args.y2,
                args.viewport_width,
                args.viewport_height,
                args.output,
            )
            mode = "vector"
        else:
            w_in, h_in = _crop_to_pdf(
                cdp,
                args.x1,
                args.y1,
                args.x2,
                args.y2,
                args.scale,
                args.output,
                args.dpi,
            )
            mode = "raster"
        size_kb = args.output.stat().st_size / 1024.0
        print(f"[pdf] {args.output}  ({mode})")
        print(
            f"      crop : ({args.x1}, {args.y1}) → ({args.x2}, {args.y2})  "
            f"(CSS px: {args.x2 - args.x1} × {args.y2 - args.y1})"
        )
        print(f"      page : {w_in:.4f}in × {h_in:.4f}in  ({size_kb:.1f} KB)")
    finally:
        try:
            proc.terminate()
            proc.wait(5)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    main()
