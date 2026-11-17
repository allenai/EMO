#!/usr/bin/env bash
# Mixed-pool EMO 512e (20% random pools) to the 61B matched point, three runs (user request 2026-11-17), all sub-models at LR 4e-4:
#   rp100m61  the 8-square arm re-partitioned every ~100M tokens (squares_repartition100m.sh, 10B -> 30B, 200 cycles) CONTINUED from its
#             30B merge (merged/match57221, Adam state carried) to 61B: cycles 201..510 in the same arm folder. After 30B there are no
#             document packs: square position g reads a contiguous eighth of the training stream (stream steps 57,221 + g x 23,842 ->
#             + 7,407), as the 4-square controls' third window does. 48 cycles to the 34.9B matched point (baseline step 66,481), 262 more
#             to 61B (116,479). Kept: the 61B merge (with Adam) + its squares (weights only); the 34.9B merge is evaluated and deleted;
#             a CE-only probe of the merge after cycle 203 checks the new stream path.
#   rp5b      NEW arm: 4 squares from the 10B start model, merged + cut into 4 new random squares (seed 2000 + cycle) every 5B tokens:
#             11 cycles ending at 15 / 20 / 25 / 30 / 35 / ... / 60 / 61B. Documents: the 4-square control's random packs to 30B
#             (olmoe3_squares_mix20rand4 pack / pack2, half a pack per cycle; square position g stays on pack g), then a contiguous
#             quarter of the stream (57,221 + g x 47,684 -> + 14,815, the control's window 3). Every merge is evaluated; merge + squares
#             are kept at 20B / 30B / 61B (squares weights only, also scored alone), deleted elsewhere once the next cycle is cut.
#   joint     the joint baseline 30B -> 61B (olmoe3_275m_mix20_61b, 4 nodes, keeper), scored at 66,481 and 116,479.
# Workers: repartition_worker.sh (symmetric 8-GPU jobs, mkdir locks); evaluations: one 8-GPU job per point (eval_batch_8gpu.sh).
#   bash scripts/sparse_experts/olmoe3_squares/squares_repartition_ext.sh rp100m61|rp5b|joint     (idempotent; detach; commit + push first)   NWORK=<jobs>
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
ARM="${1:?rp100m61|rp5b|joint}"
S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; FULL=olmoe3_275m_emo_mix20_10b; SAMPLE=sample_8k_300b.npz; LR=4e-4; B30=olmoe3_275m_mix20_30b/step57221
SP=/tmp/claude-0/-root-EMO/c7db74f2-bbe3-4a2c-9d37-93c64250d7c6/scratchpad; mkdir -p $SP
case $ARM in
  rp100m61) K=8; NWORK="${NWORK:-3}"; CTRL=olmoe3_squares_mix20rand8; CRP=olmoe3_275m_mix20rand8_square; SQN=olmoe3_squares_mix20rand8_rp100m; RP=olmoe3_275m_mix20rand8_rp100m; HR=runs_heldout300b_mix20rand8_rp100m; FIRST=201; SEED=1000; WANDB=false;;
  rp5b)     K=4; NWORK="${NWORK:-4}"; CTRL=olmoe3_squares_mix20rand4; CRP=olmoe3_275m_mix20rand4_square; SQN=olmoe3_squares_mix20rand4_rp5b;   RP=olmoe3_275m_mix20rand4_rp5b;   HR=runs_heldout300b_mix20rand4_rp5b;   FIRST=1;   SEED=2000; WANDB=true;;
  joint)    SQN=olmoe3_squares_mix20rand8; HR=runs_heldout300b_mix20; BRUN=olmoe3_275m_mix20_61b;;
  *) echo "rp100m61|rp5b|joint"; exit 1;;
esac
SQ=$S/$SQN; H=$S/olmoe3_routing/$HR; LOG=$SQ/logs; mkdir -p $LOG $H
say() { echo "$(date -u +%m-%d\ %H:%M) [$ARM] $*"; }
ended() { beaker experiment get "$1" --format json 2>/dev/null | grep -q '"exited"\|"canceled"'; }
have() { [ -s "$H/$1/none/meta.json" ] || [ -f "$H/$1/none/rank0/DONE" ]; }
hline() { echo "python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint $2 --instances $W/olmoe3_routing/$SAMPLE --out-dir $W/olmoe3_routing/$HR/$1/none/rank0 --restrict none --batch-size 8 --log-every 100"; }
pline() { echo "python scripts/debug_validation/eval_ppl_validation.py --checkpoints $1 --out-dir $W/$SQN/ppl_validation --batch-size 8"; }
# submit the commands of $CF (already rebuilt by the caller AFTER reading the job state) as one 8-GPU job; key = file stem. Returns 0 when nothing is left to run.
evaljob() { local key=$1 e tries u; [ -s $CF ] || return 0
  e=$(tail -n 1 $LOG/evals_$key.launched 2>/dev/null | grep -o '01[A-Z0-9]\{20,\}' | head -1)
  if [ -z "$e" ] || [ "$2" = ended ]; then tries=$(cat $LOG/evals_$key.launched 2>/dev/null | wc -l)
    if [ $tries -lt 4 ]; then cp $CF $LOG/evals_${key}_try$tries.txt   # a job reads its own snapshot of the list
      u=$(python3 scripts/sparse_experts/olmoe3_beaker_cmd_cli.py --name "$ARM-evals-$key-$(date +%m%d%H%M%S)" --gpus 8 -- bash scripts/sparse_experts/olmoe3_squares/eval_batch_8gpu.sh $W/$SQN/logs/evals_${key}_try$tries.txt 2>&1 | tail -n 1)
      say "evals $key ($(wc -l < $CF) commands): $u"; echo "$u" | grep -q beaker.org && echo "$u" >> $LOG/evals_$key.launched; fi; fi
  return 1; }
jobstate() { local e; e=$(tail -n 1 $LOG/evals_$1.launched 2>/dev/null | grep -o '01[A-Z0-9]\{20,\}' | head -1); if [ -z "$e" ]; then echo none; elif ended $e; then echo ended; else echo live; fi; }
fold() { local d; for d in $H/*/none; do [ -f $d/rank0/DONE ] && [ ! -f $d/meta.json ] && python scripts/sparse_experts/olmoe3_routing/merge_routing.py $d 2>&1 | tail -1; done; }

# =========================== joint baseline 30B -> 61B ===========================
if [ $ARM = joint ]; then KEEP=$S/olmoe3_routing/baseline_keeper; mkdir -p $KEEP
  if [ ! -f $S/$BRUN/step116479/train/rank0.pt ] && [ ! -f $KEEP/mix20_61b.url ]; then u=""
    for attempt in 1 2 3 4 5 6; do OLMOE3_FOLLOW=0 bash scripts/sparse_experts/model_scripts/olmoe3_275m_mix20_61b_baseline.sh launch > $SP/launch_mix20_61b.log 2>&1
      u=$(sed 's/\x1b\[[0-9;]*m//g' $SP/launch_mix20_61b.log | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 60; done
    say "joint 30B -> 61B: ${u:-LAUNCH FAILED}"; [ -n "$u" ] && echo "$u" > $KEEP/mix20_61b.url; fi
  pgrep -f "^bash scripts/sparse_experts/olmoe3_squares/baseline_keeper.sh" > /dev/null || { ( setsid nohup bash scripts/sparse_experts/olmoe3_squares/baseline_keeper.sh >> $KEEP/log.txt 2>&1 < /dev/null & ); say "baseline keeper started"; }
  jpoint() { local st=$1 js; [ -f $LOG/joint_$st.done ] && return 0; [ -f $S/$BRUN/step$st/train/rank0.pt ] || return 1; CF=$LOG/evals_joint${st}_cur.txt; js=$(jobstate joint$st); : > $CF
    have baseline_step$st || hline baseline_step$st $W/$BRUN/step$st >> $CF; [ -f $SQ/ppl_validation/$BRUN/step$st.json ] || pline $W/$BRUN/step$st >> $CF
    [ -s $CF ] && [ $js = live ] && return 1; evaljob joint$st $js || return 1; touch $LOG/joint_$st.done; say "joint step $st: evaluated"; }
  while true; do ok=1; jpoint 66481 || ok=0; jpoint 116479 || ok=0; [ $ok = 1 ] && break
    l=$(ls $S/$BRUN 2>/dev/null | grep -E '^step[0-9]+$' | sed 's/step//' | sort -n | tail -1); [ "${l:-}" != "${last:-}" ] && { say "latest checkpoint: step ${l:-none}"; last=${l:-}; }; sleep 600; done
  fold; say done; exit 0
fi

# =========================== the two re-partitioned arms ===========================
mkdir -p $SQ/templates $SQ/merged; GS=$(seq 0 $((K-1)))
for d in pack pack2; do [ -e $SQ/$d ] || ln -s ../$CTRL/$d $SQ/$d; done
if [ ! -f $LOG/ext61.planned ]; then python - <<PY || { say "ABORT: plan failed"; exit 1; }
import json, shutil, os
S, SQ, K, arm = "$S", "$SQ", $K, "$ARM"; rows = []; keep = []; points = []
def template(w, g, train_from, config_from):   # trainer state (data order / position) + the config of a square of this size
    d = f"{SQ}/templates/w{w}/group{g}"; shutil.rmtree(d, ignore_errors=True); os.makedirs(d); shutil.copytree(f"{train_from}/train", f"{d}/train")
    shutil.copy(f"{train_from}/.metadata.json", f"{d}/.metadata.json"); shutil.copy(f"{config_from}/config.json", f"{d}/config.json")
if arm == "rp100m61":
    for g in range(K):
        base = 57221 + g * 23842; b = [round(1157 * c / 48) for c in range(49)] + [1157 + round((7407 - 1157) * c / 262) for c in range(1, 263)]   # 1157 = the 34.9B fraction (926 / 5926) of 7407
        rows += [(201 + c, g, 3, base + b[c], base + b[c + 1]) for c in range(310)]; template(3, g, f"{S}/$B30", f"{SQ}/templates/w2/group{g}")
    old = [tuple(map(int, l.split())) for l in open(f"{SQ}/plan.tsv")]; assert max(r[0] for r in old) == 200, "plan already extended?"; rows = old + rows
    keep = [l.split() for l in open(f"{SQ}/keep.tsv")] + [["203", "probe_c203"], ["248", "match66481"], ["510", "match116479"]]
    points = [(203, "probe_c203", "probe"), (248, "match66481", "temp"), (510, "match116479", "perm")]
else:
    t1 = json.load(open(f"{SQ}/pack/stats.json"))["tokens_per_group"]; t2 = json.load(open(f"{SQ}/pack2/stats.json"))["tokens_per_group"]
    o = [0] + [round(5e9 / 524288 / K * c) for c in range(1, 7)] + [14815]   # window 3: 5B per cycle = 2,384 steps per square; 14,815 = the control's 61B point
    for g in range(K):
        fin = t1[g] // 524288; mid = max(1, round(fin * 10926 / 19074)); fin2 = t2[g] // 524288; base = 57221 + g * 47684
        rows += [(1, g, 1, 0, round(fin / 2)), (2, g, 1, round(fin / 2), fin), (3, g, 2, 0, round(fin2 / 2)), (4, g, 2, round(fin2 / 2), fin2)]
        rows += [(5 + i, g, 3, base + o[i], base + o[i + 1]) for i in range(7)]
        template(1, g, f"{S}/$CRP{g}/step{mid}", f"{S}/$CRP{g}/step{mid}"); template(2, g, f"{S}/$CRP{g}_w2/step{fin2}", f"{S}/$CRP{g}_w2/step{fin2}"); template(3, g, f"{S}/$B30", f"{S}/$CRP{g}_w2/step{fin2}")
    names = [28611, 38148, 47685, 57221, 66757, 76293, 85833, 95369, 104905, 114441, 116479]   # joint-equivalent step of each cycle end (19,074 + tokens trained / 524,288; the last = the control's matched 61B point)
    keep = [[str(c + 1), f"match{n}"] for c, n in enumerate(names)]; points = [(c + 1, f"match{n}", "perm" if n in (38148, 57221, 116479) else "temp") for c, n in enumerate(names)]
assert all(r[4] > r[3] for r in rows)
open(f"{SQ}/keep.tsv", "w").write("".join(" ".join(k) + "\n" for k in keep)); open(f"{SQ}/points.tsv", "w").write("".join("%d %s %s\n" % p for p in points))
open(f"{SQ}/plan.tsv.tmp", "w").write("".join("%d %d %d %d %d\n" % r for r in sorted(rows))); os.replace(f"{SQ}/plan.tsv.tmp", f"{SQ}/plan.tsv")
new = [r for r in rows if r[0] >= $FIRST]; print("plan: cycles", min(r[0] for r in new), "to", max(r[0] for r in new), "- steps per square and cycle", min(r[4] - r[3] for r in new), "to", max(r[4] - r[3] for r in new))
PY
  rm -f $SQ/DONE; rm -rf "${LOG:?}/locks"; touch $LOG/ext61.planned; fi
NC=$(awk '{ if ($1 > m) m = $1 } END { print m }' $SQ/plan.tsv)
fin_of() { awk -v c="$1" -v g="$2" -v p="$S/${RP}_c$1_g$2/step" '$1 == c && $2 == g { print p $5 }' $SQ/plan.tsv; }
workers() { local i e n u; [ -f $SQ/DONE ] && return 0
  [ "$(df --output=avail -BG /root | tail -n 1 | tr -dc 0-9)" -gt 500 ] || { [ "${fullsaid:-}" = 1 ] || say "less than 500 GB free on the shared filesystem: no worker launched until space returns"; fullsaid=1; return 0; }; fullsaid=0
  for i in $(seq 1 $NWORK); do e=$(tail -n 1 $LOG/ext_worker$i.launched 2>/dev/null | grep -o '01[A-Z0-9]\{20,\}' | head -1)
    if [ -n "$e" ]; then ended $e || continue; fi
    n=$(cat $LOG/ext_worker$i.launched 2>/dev/null | wc -l); [ $n -ge 12 ] && continue
    u=$(python3 scripts/sparse_experts/olmoe3_beaker_cmd_cli.py --name "$ARM-worker$i-$(date +%m%d%H%M%S)" --gpus 8 -- bash -c "K=$K FIRST_CYCLE=$FIRST SEED_BASE=$SEED REPART_WANDB=$WANDB OLMOE3_WANDB_TAGS=$SQN,square,repartition,$ARM SQUARES_NAME=$SQN RUN_PREFIX=$RP FULL=$FULL OLMOE3_LR=$LR OLMOE3_EMO_POOL_SELECT=mix:0.2 bash scripts/sparse_experts/olmoe3_squares/repartition_worker.sh" 2>&1 | tail -n 1)
    say "worker $i (launch $((n+1))): $u"; echo "$u" | grep -q beaker.org && echo "$u" >> $LOG/ext_worker$i.launched; done; }
later_cut() { local n; for n in $(ls $SQ/cyc 2>/dev/null | sed 's/^c//'); do [ "$n" -gt "$1" ] && [ -f $SQ/cyc/c$n/PREP_DONE ] && return 0; done; return 1; }
# a kept cycle end: perm = merged model + every square alone, both kept (squares cut down to weights); temp = merged model only, merge + squares deleted
# once evaluated and the next cycle is cut; probe = CE-only reading of the merge, then deleted like temp
point() { local c=$1 name=$2 kind=$3 g d js ce; [ -f $LOG/point_$name.done ] && return 0; [ -f $SQ/merged/$name/merge_info.json ] || return 1
  CF=$LOG/evals_${name}_cur.txt; js=$(jobstate $name); : > $CF
  if [ $kind = probe ]; then [ -s $H/$name/none/meta.json ] || echo "python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint $W/$SQN/merged/$name --instances $W/olmoe3_routing/$SAMPLE --out-dir $W/olmoe3_routing/$HR/$name/none --restrict none --batch-size 8 --log-every 100 --ce-only" >> $CF
  else have merged_$name || hline merged_$name $W/$SQN/merged/$name >> $CF; [ -f $SQ/ppl_validation/merged/$name.json ] || pline $W/$SQN/merged/$name >> $CF; fi
  if [ $kind = perm ]; then for g in $GS; do d=$(fin_of $c $g); have sub${g}_$name || hline sub${g}_$name ${d/$S/$W} >> $CF; [ -f $SQ/ppl_validation/$(basename $(dirname $d))/$(basename $d).json ] || pline ${d/$S/$W} >> $CF; done; fi
  [ -s $CF ] && [ $js = live ] && return 1; evaljob $name $js || return 1
  if [ $kind = perm ]; then PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/strip_optim.py $(for g in $GS; do fin_of $c $g; done) >> $LOG/strip.log 2>&1
    for g in $GS; do d=$(fin_of $c $g); for x in $(dirname $d)/step*; do [ "$x" = "$d" ] || rm -rf "$x"; done; done   # interval checkpoints of a long cycle
  else later_cut $c || [ -f $SQ/DONE ] || return 1   # the merge is the next cut's source until that cut is written
    rm -rf "${SQ:?}/merged/${name:?}"; for g in $GS; do rm -rf "$S/${RP:?}_c${c}_g$g"; done; fi
  d=$H/merged_$name/none; [ $kind = probe ] && d=$H/$name/none; ce=$(python -c "import json, os; f = '$d/meta.json' if os.path.exists('$d/meta.json') else '$d/rank0/meta.json'; print(round(json.load(open(f))['mean_ce'], 4))" 2>/dev/null)
  touch $LOG/point_$name.done; say "point $name ($kind): evaluated; merged CE $ce"; }
while true; do workers; ok=1
  while read -r c name kind <&3; do point $c $name $kind || ok=0; done 3< $SQ/points.tsv
  [ $ok = 1 ] && [ -f $SQ/DONE ] && break
  c=$(ls $SQ/cyc 2>/dev/null | sed 's/^c//' | sort -n | tail -1); [ "${c:-0}" != "${lastc:-}" ] && { say "cycle ${c:-0} of $NC"; lastc=${c:-0}; }; sleep 600; done
fold; rm -rf "${SQ:?}/cyc"
for c in $((NC-2)) $((NC-1)); do grep -q "^$c " $SQ/keep.tsv || for g in $GS; do rm -rf "$S/${RP:?}_c${c}_g$g"; done; done
say done
