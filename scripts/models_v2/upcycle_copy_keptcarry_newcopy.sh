#!/bin/bash
# Expert-upcycling ablation leaf: copy_keptcarry_newcopy
#   init=upcycle  kept-optim=carry  new-optim=copy
# Seeds an expanded 128-expert step5960 from stdmoe_64exp_50b_wsd_lr2e-3 (25B) and continues WSD
# training. Default MAX_B=30 (5B convergence check); rerun with MAX_B=50 to extend to 50B.
# See scripts/models_v2/launch_upcycle.sh for the full mechanism.
export INIT_MODE="upcycle"
export KEPT_OPTIM="carry"
export NEW_OPTIM="copy"
export LEAF="copy_keptcarry_newcopy"
exec bash "$(dirname "${BASH_SOURCE[0]}")/launch_upcycle.sh"
