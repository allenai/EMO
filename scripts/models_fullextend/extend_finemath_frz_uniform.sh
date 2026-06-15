#!/usr/bin/env bash
# New-expert extension for the UNIFORM-ghost model: take the uniform-ghost checkpoint, add one
# instantiated expert (128->129), and continually pretrain on FineMath with everything but the
# new expert's MLP frozen. Full recipe in extend_finemath_frz_common.sh.
#
#   MODE=beaker bash scripts/models_fullextend/extend_finemath_frz_uniform.sh
VARIANT=uniform source "$(dirname "${BASH_SOURCE[0]}")/extend_finemath_frz_common.sh"
