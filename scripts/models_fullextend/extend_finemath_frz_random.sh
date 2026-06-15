#!/usr/bin/env bash
# New-expert extension for the RANDOM-ghost model: take the random-ghost checkpoint, add one
# instantiated expert (128->129), and continually pretrain on FineMath with everything but the
# new expert's MLP frozen. Full recipe in extend_finemath_frz_common.sh.
#
#   MODE=beaker bash scripts/models_fullextend/extend_finemath_frz_random.sh
VARIANT=random source "$(dirname "${BASH_SOURCE[0]}")/extend_finemath_frz_common.sh"
