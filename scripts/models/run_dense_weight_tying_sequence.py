#!/usr/bin/env python3
"""Run pre-rendered sequential training stages without a giant argv payload."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    for stage in manifest["stages"]:
        subprocess.run(["bash", "-lc", stage["shell"]], check=True)
        decision = stage.get("stopDecision")
        if decision and Path(decision).is_file() and Path(decision).read_text().strip() == "stop":
            print(f"SEQUENTIAL_RUNTIME_STOP epoch={stage['epoch']}", flush=True)
            return


if __name__ == "__main__":
    main()
