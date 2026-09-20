#!/usr/bin/env python3
"""Run the reviewer-requested 1.5B Pool-1B REPACK+untied+EmbedWD chains."""

from __future__ import annotations

from pathlib import Path

import run_dense_1b_dr_wt_embedwd_grid as base

POLICY = "dense_1b_repack_untied_embedwd_reviewer_v1"
OUTPUT_ROOT = (
    "/weka/oe-training-default/sewonm/icsl/models/"
    "dense_1b_dclm1b_repack_untied_embedwd_reviewer_v1"
)


def configure_base() -> None:
    base.POLICY = POLICY
    base.POST_DECAY_SOURCE_COUNT = 2
    base.CHECKPOINT_ONLY_EPOCHS = []
    base.OUTPUT_ROOT = OUTPUT_ROOT
    base.ALLOWED_COORDINATES = {
        64: {("1e-3", "0.3")},
        512: {("1e-3", "1.0")},
    }


configure_base()

validate_config = base.validate_config
coordinate_config = base.coordinate_config
base_arguments = base.base_arguments
predecay_training_arguments = base.predecay_training_arguments
phase_metadata = base.phase_metadata
stable_step = base.stable_step
total_step = base.total_step
run = base.run


if __name__ == "__main__":
    base.main()
