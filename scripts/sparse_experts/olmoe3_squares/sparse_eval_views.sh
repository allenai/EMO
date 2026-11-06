#!/usr/bin/env bash
# Sparser squares (2026-11-06): evaluate full models the way the sparser squares were trained. The merged models of the two arms, the existing
# 16-of-64 arm's 30B merge, the joint 30B baseline and the 10B start model are run on the reporting held-out sample with 4 active experts in
# layers 2-9 (layer 1 at 16), under the stock weights (e4) or the top-16 weights with the 4 largest kept (e4n16); mean CE only
# (extract_routing.py --ce-only). Views are config-only (make_eval_view.py). Two batched allocated 1-GPU jobs; idempotent.
#   bash scripts/sparse_experts/olmoe3_squares/sparse_eval_views.sh      (commit + push first)
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; SAMPLE=sample_8k_300b.npz; A=olmoe3_squares_emorand8_lr4e-4; V=$S/${A}_top4/eval_views; mkdir -p $V
say() { echo "$(date -u +%m-%d\ %H:%M) [views] $*"; }
CMDS=""; NP=0
# pass <view name> <source checkpoint rel. to sparse_experts> <rule e4|e4n16> <output dir rel. to olmoe3_routing>
pass() { local name=$1 src=$2 rule=$3 out=$4
  [ -f $S/olmoe3_routing/$out/none/meta.json ] && return 0
  [ -f $S/$src/config.json ] || { say "skip $name: no checkpoint $src"; return 0; }
  python3 scripts/sparse_experts/olmoe3_squares/make_eval_view.py --src $S/$src --out $V/$name --top-k 4 $([ $rule = e4n16 ] && echo --norm-top-k 16) | cut -c1-150
  CMDS="$CMDS python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint $W/${A}_top4/eval_views/$name --instances $W/olmoe3_routing/$SAMPLE --out-dir $W/olmoe3_routing/$out/none --restrict none --batch-size 8 --log-every 100 --ce-only;"; NP=$((NP+1)); }
flush() { local name=$1 mark=$V/$1.launched; [ -n "$CMDS" ] || return 0; [ -f $mark ] && { CMDS=""; NP=0; return 0; }
  local u; u=$(python3 scripts/sparse_experts/olmoe3_beaker_cmd_cli.py --name "$name" --gpus 1 -- bash -c "$CMDS" 2>&1 | tail -n 1); say "$name ($NP passes): $u"; echo "$u" | grep -q beaker.org && echo "$u" > $mark; CMDS=""; NP=0; }
for s in 30000 38148 57221; do pass top4_merged${s}_e4 ${A}_top4/merged/match$s e4 runs_heldout300b_emorand8_lr4e-4_top4/merged_match${s}_e4; done
pass ref_merged57221_e4 $A/merged/match57221 e4 runs_heldout300b_emorand8_lr4e-4/merged_match57221_e4
pass joint57221_e4 olmoe3_275m_emo_30b_1node/step57221 e4 runs_heldout300b_emo/baseline_step57221_e4
pass start_e4 olmoe3_275m_emo_10b/step19074 e4 runs_heldout300b_emo/emo_step19074_e4
flush sparse-views-e4
for s in 30000 38148 57221; do pass top4n16_merged${s}_e4n16 ${A}_top4n16/merged/match$s e4n16 runs_heldout300b_emorand8_lr4e-4_top4n16/merged_match${s}_e4n16; done
pass ref_merged57221_e4n16 $A/merged/match57221 e4n16 runs_heldout300b_emorand8_lr4e-4/merged_match57221_e4n16
pass joint57221_e4n16 olmoe3_275m_emo_30b_1node/step57221 e4n16 runs_heldout300b_emo/baseline_step57221_e4n16
pass start_e4n16 olmoe3_275m_emo_10b/step19074 e4n16 runs_heldout300b_emo/emo_step19074_e4n16
flush sparse-views-e4n16
say done
