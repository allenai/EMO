#!/bin/bash
# Dedicated WSD decay branch: anneal the lr4e-4 stable trunk (stdmoe_64exp_50b_wsd_lr4e-4) from its
# 45B checkpoint (step10729) for 5B tokens -> ends at step11921 = 50B total. The anneal entry point
# auto-extracts the trunk's flat peak LR (4e-4) from the checkpoint and decays it linearly to 0.
#
# Thin wrapper over the reusable scripts/models_v2/launch_wsd_decay.sh; hardcodes the branch spec so
# the exact config is versioned (per the CLAUDE.md "launch from a checked-in script" convention).
# Saves hierarchically to .../stdmoe_64exp_50b_wsd_lr4e-4/anneals/s10729_5b/ .
#
#   git add . && git commit && git push origin <branch>
#   MODE=beaker bash scripts/models_v2/anneal_lr4e-4_s10729_5b.sh
export TRUNK_RUN="stdmoe_64exp_50b_wsd_lr4e-4"
export CKPT_STEP=10729   # 45B (a fixed_step permanent checkpoint in the lr4e-4 trunk)
export DECAY_B=5         # decay 5B tokens -> 45B + 5B = 50B (end step 11921)
exec bash "$(dirname "${BASH_SOURCE[0]}")/launch_wsd_decay.sh"
