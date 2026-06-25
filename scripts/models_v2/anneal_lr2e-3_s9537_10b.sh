#!/bin/bash
# Dedicated WSD decay branch: anneal the lr2e-3 stable trunk (stdmoe_64exp_50b_wsd_lr2e-3) from its
# 40B checkpoint (step9537) for 10B tokens -> ends at step11921 = 50B total. The anneal entry point
# auto-extracts the trunk's flat peak LR (2e-3) from the checkpoint and decays it linearly to 0.
#
# Thin wrapper over the reusable scripts/models_v2/launch_wsd_decay.sh; hardcodes the branch spec so
# the exact config is versioned (per the CLAUDE.md "launch from a checked-in script" convention).
# Saves hierarchically to .../stdmoe_64exp_50b_wsd_lr2e-3/anneals/s9537_10b/ .
#
#   git add . && git commit && git push origin <branch>
#   MODE=beaker bash scripts/models_v2/anneal_lr2e-3_s9537_10b.sh
export TRUNK_RUN="stdmoe_64exp_50b_wsd_lr2e-3"
export CKPT_STEP=9537    # 40B (a fixed_step permanent checkpoint in the lr2e-3 trunk)
export DECAY_B=10        # decay 10B tokens -> 40B + 10B = 50B (end step 11921)
exec bash "$(dirname "${BASH_SOURCE[0]}")/launch_wsd_decay.sh"
