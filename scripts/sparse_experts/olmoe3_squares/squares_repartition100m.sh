#!/usr/bin/env bash
# Re-partition every ~100M tokens (user request 2026-11-15): the mixed-pool EMO 512e (20% random pools) 8-square control at sub-model LR
# 4e-4, 10B -> 30B, with ONE change -- every ~100M tokens of the stream (about 24 steps of each square; 200 cycles) the eight squares are
# merged (Adam state carried, weights = steps trained) and the merged model is cut into eight NEW random squares (fresh random expert
# groups of 64 in layers 2-9, layer 1 whole; seed 1000 + cycle). Documents: exactly the control's random packs -- square position g always
# continues on pack g of olmoe3_squares_mix20rand8 (pack for 10B -> 20B, pack2 for 20B -> 30B) from where it stopped, so the comparison arm
# olmoe3_squares_mix20rand8 saw the same documents in the same order. Cycle ends fall exactly on the control's matched points: 57 cycles
# to 15.7B (baseline step 30000), 43 more to 20B (38148), 100 more to 30B (57221). Kept: the merged model (with Adam state) and that
# cycle's eight squares (weights only) at those three points; every other cycle's checkpoints are deleted two cycles later
# (repartition_worker.sh). A CE-only reading of the merge after cycle 3 is a correctness probe (cf. the first-steps probe of the same model).
# Evaluations: merged model and every square alone on the 300B held-out sample and the v3-small ppl sets, one 8-GPU job per point.
#   bash scripts/sparse_experts/olmoe3_squares/squares_repartition100m.sh      (idempotent; detach; commit + push first)   NWORK=<jobs> (default 2)
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; FULL=olmoe3_275m_emo_mix20_10b; SAMPLE=sample_8k_300b.npz; LR=4e-4; K=8; NWORK="${NWORK:-2}"
CTRL=olmoe3_squares_mix20rand8; CRP=olmoe3_275m_mix20rand8_square
SQN=olmoe3_squares_mix20rand8_rp100m; SQ=$S/$SQN; RP=olmoe3_275m_mix20rand8_rp100m; HR=runs_heldout300b_mix20rand8_rp100m; H=$S/olmoe3_routing/$HR; LOG=$SQ/logs; mkdir -p $LOG $H $SQ/templates
say() { echo "$(date -u +%m-%d\ %H:%M) [rp100m] $*"; }
GS=$(seq 0 $((K-1)))
for d in pack pack2; do [ -e $SQ/$d ] || ln -s ../$CTRL/$d $SQ/$d; done
# plan: per cycle and square the window, start and end step on its pack (control's own matched mid point and finals); trainer-state templates
[ -f $SQ/plan.tsv ] || python - <<PY
import json, shutil, os
t1 = json.load(open("$SQ/pack/stats.json"))["tokens_per_group"]; t2 = json.load(open("$SQ/pack2/stats.json"))["tokens_per_group"]; rows = []
for g in range($K):
    fin = t1[g] // 524288; mid = max(1, round(fin * 10926 / 19074)); fin2 = t2[g] // 524288
    b = [round(mid * c / 57) for c in range(58)] + [mid + round((fin - mid) * c / 43) for c in range(1, 44)]
    rows += [(c + 1, g, 1, b[c], b[c + 1]) for c in range(100)]; b2 = [round(fin2 * c / 100) for c in range(101)]; rows += [(c + 101, g, 2, b2[c], b2[c + 1]) for c in range(100)]
    for w, src in ((1, f"$S/$CRP{g}/step{mid}"), (2, f"$S/$CRP{g}_w2/step{fin2}")):   # trainer state (data order of this pack) + config of a 64-expert square
        d = f"$SQ/templates/w{w}/group{g}"; shutil.rmtree(d, ignore_errors=True); os.makedirs(d); shutil.copytree(f"{src}/train", f"{d}/train")
        for f in (".metadata.json", "config.json"): shutil.copy(f"{src}/{f}", f"{d}/{f}")
assert all(r[4] > r[3] for r in rows)
open("$SQ/keep.tsv", "w").write("3 probe_c3\n57 match30000\n100 match38148\n200 match57221\n")
open("$SQ/plan.tsv.tmp", "w").write("".join("%d %d %d %d %d\n" % r for r in sorted(rows))); os.replace("$SQ/plan.tsv.tmp", "$SQ/plan.tsv")
print("plan: 200 cycles, steps per square and cycle", min(r[4] - r[3] for r in rows), "to", max(r[4] - r[3] for r in rows))
PY
[ -f $SQ/plan.tsv ] || { say "ABORT: plan failed"; exit 1; }
fin_of() { awk -v c="$1" -v g="$2" -v p="$S/${RP}_c$1_g$2/step" '$1 == c && $2 == g { print p $5 }' $SQ/plan.tsv; }
# ---- workers: NWORK identical 8-GPU jobs; one that exits before the run is complete is replaced (at most 8 launches per slot) ----
workers() { local i e n u; [ -f $SQ/DONE ] && return 0
  [ "$(df --output=avail -BG /root | tail -n 1 | tr -dc 0-9)" -gt 500 ] || { [ "${fullsaid:-}" = 1 ] || say "less than 500 GB free on the shared filesystem: no worker launched until space returns"; fullsaid=1; return 0; }; fullsaid=0
  for i in $(seq 1 $NWORK); do e=$(tail -n 1 $LOG/worker$i.launched 2>/dev/null | grep -o '01[A-Z0-9]\{20,\}' | head -1)
    if [ -n "$e" ]; then beaker experiment get $e --format json 2>/dev/null | grep -q '"exited"\|"canceled"' || continue; fi
    n=$(cat $LOG/worker$i.launched 2>/dev/null | wc -l); [ $n -ge 12 ] && continue
    u=$(python3 scripts/sparse_experts/olmoe3_beaker_cmd_cli.py --name "rp100m-worker$i-$(date +%m%d%H%M%S)" --gpus 8 -- bash -c "SQUARES_NAME=$SQN RUN_PREFIX=$RP FULL=$FULL OLMOE3_LR=$LR OLMOE3_EMO_POOL_SELECT=mix:0.2 bash scripts/sparse_experts/olmoe3_squares/repartition_worker.sh" 2>&1 | tail -n 1)
    say "worker $i (launch $((n+1))): $u"; echo "$u" | grep -q beaker.org && echo "$u" >> $LOG/worker$i.launched; done; }
# ---- evaluations of a kept point ----
have() { [ -s "$H/$1/none/meta.json" ] || [ -f "$H/$1/none/rank0/DONE" ]; }
hline() { echo "python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint $2 --instances $W/olmoe3_routing/$SAMPLE --out-dir $W/olmoe3_routing/$HR/$1/none/rank0 --restrict none --batch-size 8 --log-every 100"; }
pline() { echo "python scripts/debug_validation/eval_ppl_validation.py --checkpoints $1 --out-dir $W/$SQN/ppl_validation --batch-size 8"; }
build() { local c=$1 s=$2 g d; : > $CF
  have merged_match$s || hline merged_match$s $W/$SQN/merged/match$s >> $CF; [ -f $SQ/ppl_validation/merged/match$s.json ] || pline $W/$SQN/merged/match$s >> $CF
  for g in $GS; do d=$(fin_of $c $g); have sub${g}_match$s || hline sub${g}_match$s ${d/$S/$W} >> $CF; [ -f $SQ/ppl_validation/$(basename $(dirname $d))/$(basename $d).json ] || pline ${d/$S/$W} >> $CF; done; }
point() { local c=$1 s=$2 e u tries; CF=$LOG/evals_${s}_cur.txt;   # rebuilt every pass; a job reads its own snapshot (a list rewritten under a running job garbled its commands, 2026-11-16)
  [ -f $LOG/point_$s.done ] && return 0; [ -f $SQ/merged/match$s/merge_info.json ] || return 1
  build $c $s
  if [ -s $CF ]; then e=$(tail -n 1 $LOG/evals_$s.launched 2>/dev/null | grep -o '01[A-Z0-9]\{20,\}' | head -1)
    if [ -z "$e" ] || beaker experiment get $e --format json 2>/dev/null | grep -q '"exited"\|"canceled"'; then
      [ -n "$e" ] && { sleep 60; build $c $s; }; tries=$(cat $LOG/evals_$s.launched 2>/dev/null | wc -l)
      if [ -s $CF ] && [ $tries -lt 4 ]; then cp $CF $LOG/evals_${s}_try$tries.txt; u=$(python3 scripts/sparse_experts/olmoe3_beaker_cmd_cli.py --name "rp100m-evals-$s-$(date +%m%d%H%M%S)" --gpus 8 -- bash scripts/sparse_experts/olmoe3_squares/eval_batch_8gpu.sh $W/$SQN/logs/evals_${s}_try$tries.txt 2>&1 | tail -n 1)
        say "evals at $s ($(wc -l < $CF) commands): $u"; echo "$u" | grep -q beaker.org && echo "$u" >> $LOG/evals_$s.launched; fi; fi
    [ -s $CF ] && return 1; fi
  # all passes in: the point's squares are cut down to their weights (the merged model keeps its Adam state)
  PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/strip_optim.py $(for g in $GS; do fin_of $c $g; done) >> $LOG/strip.log 2>&1
  touch $LOG/point_$s.done; say "point $s: evaluated; merged CE $(python -c "import json; print(round(json.load(open('$H/merged_match$s/none/rank0/meta.json' if __import__('os').path.exists('$H/merged_match$s/none/rank0/meta.json') else '$H/merged_match$s/none/meta.json'))['mean_ce'],4))" 2>/dev/null)"; }
probe() { [ -f $LOG/probe.done ] && return 0; [ -f $SQ/merged/probe_c3/merge_info.json ] || return 1   # CE-only reading of the merge after cycle 3, in this session's GPU job queue
  if [ ! -s $H/probe_c3/none/meta.json ]; then [ -f $LOG/probe.launched ] && return 1
    u=$(python3 scripts/sparse_experts/olmoe3_beaker_cmd_cli.py --name "rp100m-probe-$(date +%m%d%H%M%S)" --gpus 1 -- python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint $W/$SQN/merged/probe_c3 --instances $W/olmoe3_routing/$SAMPLE --out-dir $W/olmoe3_routing/$HR/probe_c3/none --restrict none --batch-size 8 --log-every 100 --ce-only 2>&1 | tail -n 1); say "probe after cycle 3: $u"; echo "$u" > $LOG/probe.launched; return 1; fi
  say "probe: merged CE after cycle 3 = $(python -c "import json; print(round(json.load(open('$H/probe_c3/none/meta.json'))['mean_ce'],4))") (start model 2.467)"
  rm -rf "$SQ/merged/probe_c3"; for g in $GS; do d=$(fin_of 3 $g); rm -rf "$(dirname $d)"; done; touch $LOG/probe.done; }
while true; do workers; probe; ok=1; point 57 30000 || ok=0; point 100 38148 || ok=0; point 200 57221 || ok=0; [ $ok = 1 ] && [ -f $SQ/DONE ] && break
  c=$(ls $SQ/cyc 2>/dev/null | sed 's/^c//' | sort -n | tail -1); [ "${c:-0}" != "${lastc:-}" ] && { say "cycle ${c:-0} of 200"; lastc=${c:-0}; }; sleep 600; done
for d in $H/*/none; do [ -f $d/rank0/DONE ] && [ ! -f $d/meta.json ] && python scripts/sparse_experts/olmoe3_routing/merge_routing.py $d 2>&1 | tail -1; done
rm -rf "${SQ:?}/cyc"; for c in 198 199; do for g in $GS; do d=$(fin_of $c $g); rm -rf "$(dirname $d)"; done; done
say done
