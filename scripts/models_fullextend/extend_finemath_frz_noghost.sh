#!/usr/bin/env bash
# New-expert extension for the NO-GHOST EMO baseline: take the unmodified EMO 1B/14B 50B
# checkpoint (step11921; 130B LR schedule), add one instantiated expert (128->129, uniform-average init), and
# continually pretrain on FineMath with everything but the new expert's MLP frozen. This is
# the apples-to-apples control for the ghost extension runs -- identical grow+freeze recipe,
# but the base was pretrained WITHOUT the ghost mechanism. Full recipe in
# extend_finemath_frz_common.sh.
#
#   MODE=beaker bash scripts/models_fullextend/extend_finemath_frz_noghost.sh
VARIANT=noghost source "$(dirname "${BASH_SOURCE[0]}")/extend_finemath_frz_common.sh"
