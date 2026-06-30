#!/bin/bash
# Expert-upcycling ablation leaf, 16e -> 128e source: jitter_keptcarry_newcopy.
#   init=upcycle_jitter  kept-optim=carry  new-optim=copy
# Source trunk is the LOWER-capacity stdmoe_16exp_50b_wsd_lr2e-3 at 25B (step5960). The 16-expert
# checkpoint is 15 standard + 1 shared; surgery expands it to 127 standard + 1 shared (each of the 15
# sources copied once into the first 15 new slots, the remaining 97 new slots filled by seeded-random
# sources -- see scripts/models_v2/expand_moe_experts.py). Then continue WSD training the remaining
# 25B (step5960 -> step11921 = 50B total) in ONE pass (MAX_B=50). Because the source 16e trunk is much
# further below the 128e upper bound than the 64e source, this measures how much upcycling closes a
# WIDE lower->upper gap. Compare against from-scratch stdmoe_128exp_50b_wsd_lr2e-3 and the 64e->128e
# upcycle leaves.
# See scripts/models_v2/launch_upcycle.sh for the full mechanism.
#
#   git add . && git commit && git push origin <branch>
#   MODE=beaker bash scripts/models_v2/upcycle_16to128_jitter_keptcarry_newcopy.sh
export INIT_MODE="upcycle_jitter"
export KEPT_OPTIM="carry"
export NEW_OPTIM="copy"
export LEAF="jitter_keptcarry_newcopy"
export FROM_RUN="stdmoe_16exp_50b_wsd_lr2e-3"
export FROM_STEP=5960
export FROM_EXPERTS=16
export MAX_B=50
exec bash "$(dirname "${BASH_SOURCE[0]}")/launch_upcycle.sh"
