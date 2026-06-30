#!/bin/bash
# Expert-upcycling ablation leaf, 16e -> 128e source: jitter_keptcarry_newzero.
#   init=upcycle_jitter  kept-optim=carry  new-optim=zero
# Identical to upcycle_16to128_jitter_keptcarry_newcopy.sh EXCEPT new-expert optimizer moments start
# at zero instead of inheriting their source expert's moments. Pairs with the newcopy leaf to test
# whether inheriting new-expert momentum matters when upcycling from the wide-gap 16e source (it
# washed out for the 64e source). Source = stdmoe_16exp_50b_wsd_lr2e-3 @ 25B (step5960); continue the
# remaining 25B to 50B (step11921) in one pass.
# See scripts/models_v2/launch_upcycle.sh for the full mechanism.
#
#   git add . && git commit && git push origin <branch>
#   MODE=beaker bash scripts/models_v2/upcycle_16to128_jitter_keptcarry_newzero.sh
export INIT_MODE="upcycle_jitter"
export KEPT_OPTIM="carry"
export NEW_OPTIM="zero"
export LEAF="jitter_keptcarry_newzero"
export FROM_RUN="stdmoe_16exp_50b_wsd_lr2e-3"
export FROM_STEP=5960
export FROM_EXPERTS=16
export MAX_B=50
exec bash "$(dirname "${BASH_SOURCE[0]}")/launch_upcycle.sh"
