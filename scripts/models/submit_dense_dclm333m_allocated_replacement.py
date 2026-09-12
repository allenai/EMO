#!/usr/bin/env python3
"""Replace the exact pending 474M Pool-333M LR1e-3 WD1.0 run with allocation."""

from __future__ import annotations

import argparse
import json
import re

import submit_dense_dclm333m_checkpoint_producers as submit


COORDINATE = "dense-474m-dclm333m-bs64-lr1e-3-wd1.0"
NAME = f"{COORDINATE}-integrated-producer-eval-allocated-v2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--print-spec", action="store_true")
    parser.add_argument("--submit-if-ready", action="store_true")
    args = parser.parse_args()
    submit.validate_revision(args.revision)
    config = submit.load_manifest()
    matches = [item for item in config["producerCoordinates"] if item["id"] == COORDINATE]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one manifest coordinate {COORDINATE}")
    item = matches[0]
    spec = submit.spec_for(
        item,
        args.revision,
        args.priority,
        omit_min_runtime=False,
    )
    if "minRuntime" not in spec["tasks"][0]["context"]:
        raise RuntimeError("allocated replacement must include minRuntime")
    if args.print_spec:
        print(json.dumps(spec, indent=2))
        return
    if not args.submit_if_ready:
        print(f"{NAME}: ready")
        return
    existing = submit.existing_named_experiment(NAME)
    if existing:
        print(existing)
        return
    output = submit.command(
        ["beaker", "experiment", "create", "-", "--name", NAME, "--workspace", submit.WORKSPACE],
        input_text=json.dumps(spec),
    )
    identifiers = re.findall(r"\b[0-9A-HJKMNP-TV-Z]{26}\b", output)
    if not identifiers:
        raise RuntimeError("submission returned no experiment ID")
    print(identifiers[0])


if __name__ == "__main__":
    main()
