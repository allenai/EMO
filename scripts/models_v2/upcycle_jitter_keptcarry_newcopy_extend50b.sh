#!/bin/bash
# Expert-upcycling ablation leaf: jitter_keptcarry_newcopy -- FULL 50B EXTEND PASS.
#   init=upcycle_jitter  kept-optim=carry  new-optim=copy
# The 5B convergence check (MAX_B=30) already trained this leaf 25B->30B (step5960->7153) and wrote a
# permanent step7153 checkpoint. This pass sets MAX_B=50 so the trainer AUTO-RESUMES from that
# step7153 (model + optim + global_step + data cursor) and continues the remaining 20B to step11921
# (50B total) -- the full 25B 128-expert continuation, comparable to the from-scratch 128e WSD-2e-3
# trunk. Seeding is skipped (seed checkpoint already present); load_path is ignored once a
# trainer-written checkpoint exists in the save folder.
# See scripts/models_v2/launch_upcycle.sh for the full mechanism.
#
#   git add . && git commit && git push origin <branch>
#   MODE=beaker bash scripts/models_v2/upcycle_jitter_keptcarry_newcopy_extend50b.sh
export INIT_MODE="upcycle_jitter"
export KEPT_OPTIM="carry"
export NEW_OPTIM="copy"
export LEAF="jitter_keptcarry_newcopy"
export MAX_B=50
exec bash "$(dirname "${BASH_SOURCE[0]}")/launch_upcycle.sh"
