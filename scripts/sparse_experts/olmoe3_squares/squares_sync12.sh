#!/usr/bin/env bash
# 12 sub-models with a sync every 2B tokens (user request 2026-11-23): mixed-pool EMO 512e (20% random pools) at 10B is cut into 12 random
# sub-models (43 / 42 experts per layer 2-9, layer 1 whole; ONE fixed partition, seed 3000). Every 2B tokens of the stream (about 318 steps of
# each sub-model) the 12 are merged (plain average of what they share, experts scattered back, Adam state carried) and cut again along the
# SAME partition, so a sync only averages the shared parameters. Documents: there are no 12-way packs, so sub-model g reads a contiguous
# twelfth of the 10B -> 30B training stream (stream steps 19,074 + g x 3,179 -> + 3,179; the stream is a global shuffle, so this is a random
# twelfth of the documents), as the earlier arms do after 30B.
#   sweep <lr> : the sub-model LR ablation, 10B -> 20B (5 syncs). Evaluated: the 20B merge on the SELECTION sample (sample_8k_300b_val.npz, as
#                the K = 4 / 8 sweeps of the EMO 512e model), the reporting sample and the ppl sets; the 15.7B matched point (not a sync:
#                an extra checkpoint inside the third cycle, merged for evaluation only) on the reporting sample.
#   choose     : waits for the four sweep arms, picks the LR with the lowest selection CE at 20B, deletes the other arms' checkpoints and
#                runs `final` for the winner.
#   final <lr> : the chosen arm continued 20B -> 30B (5 more syncs); 30B merge + every sub-model alone at 20B and 30B evaluated.
# Storage (user): only the final model is kept (30B merge with Adam + its 12 sub-models, weights only); every other merge / cycle is deleted
# as soon as it is evaluated and superseded. Workers: repartition_worker.sh (FIXED_GROUPS=1); evaluations: 8-GPU batch jobs.
# More layers kept whole (user request 2026-11-23): SYNC_K=<8|12> SYNC_WHOLE=<N> ... run <lr> does the whole 10B -> 30B run in one go (10 syncs, no
# ablation) with K sub-models that keep expert layers 1..N whole and split only layers N+1..9 (N = 4: layers 5-9 split; arm folders
# olmoe3_squares_mix20sync<K>w<N>_lr<lr>, sub-model g reads a contiguous 1/K of the stream). Same evaluations and storage rule as `final`.
#   SYNC_K=8 SYNC_WHOLE=4 bash scripts/sparse_experts/olmoe3_squares/squares_sync12.sh run 4e-4
#   bash scripts/sparse_experts/olmoe3_squares/squares_sync12.sh sweep <lr> | choose | final <lr>      (idempotent; detach; commit + push first)
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
MODE="${1:?sweep|choose|final|run}"; LR="${2:-}"; LRS="1e-4 2e-4 4e-4 8e-4"; K="${SYNC_K:-12}"; WH="${SYNC_WHOLE:-1}"
AT=mix20sync12; SEEDB=3000; [ "$K-$WH" = 12-1 ] || { AT=mix20sync${K}w$WH; SEEDB=3100; }   # arm tag; the first run (12 sub-models, layer 1 whole) keeps its names
S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; FULL=olmoe3_275m_emo_mix20_10b; SAMPLE=sample_8k_300b.npz; VAL=sample_8k_300b_val.npz
say() { echo "$(date -u +%m-%d\ %H:%M) [$AT $MODE ${LR}] $*"; }
ce_of() { python -c "import json, os; d = '$1/none/'; f = d + 'meta.json' if os.path.exists(d + 'meta.json') else d + 'rank0/meta.json'; print(round(json.load(open(f))['mean_ce'], 4))" 2>/dev/null; }
arm() { SQN=olmoe3_squares_${AT}_lr$1; RP=olmoe3_275m_${AT}_lr$1; HR=runs_heldout300b_${AT}_lr$1; SQ=$S/$SQN; H=$S/olmoe3_routing/$HR; HV=${H}_val; LOG=$SQ/logs; }

# =========================== choose: the LR with the lowest selection CE at 20B ===========================
if [ $MODE = choose ]; then CH=$S/olmoe3_squares_mix20sync12_choice.txt
  if [ ! -s $CH ]; then
    while true; do n=0; for lr in $LRS; do arm $lr; [ -n "$(ce_of $HV/merged_match38148)" ] && n=$((n+1)); done; [ $n = 4 ] && break; sleep 300; done
    best=""; bv=99; for lr in $LRS; do arm $lr; v=$(ce_of $HV/merged_match38148); say "LR $lr: selection CE at 20B $v, reporting $(ce_of $H/merged_match38148)"; [ "$(python -c "print(int($v < $bv))")" = 1 ] && { best=$lr; bv=$v; }; done
    echo "$best" > $CH; say "chosen sub-model LR: $best"; fi
  best=$(cat $CH)
  for lr in $LRS; do [ $lr = $best ] && continue; arm $lr; rm -rf "${SQ:?}/merged" "${SQ:?}/cyc" "${SQ:?}/templates"; for d in $S/${RP}_c*_g*; do [ -d "$d" ] && rm -rf "$d"; done; done; say "checkpoints of the other arms deleted"
  exec bash scripts/sparse_experts/olmoe3_squares/squares_sync12.sh final $best
fi

# =========================== sweep / final: one arm ===========================
[ -n "$LR" ] || { echo "sweep|final need an LR"; exit 1; }
arm $LR; mkdir -p $LOG $H $HV $SQ/templates $SQ/merged; GS=$(seq 0 $((K-1)))
NCYC=5; FIRST=1; FULLRUN=0; [ $MODE = final ] && { NCYC=10; FIRST=6; FULLRUN=1; }; [ $MODE = run ] && { NCYC=10; FULLRUN=1; }
if [ ! -f $SQ/plan.tsv ] || [ "$(awk '{ if ($1 > m) m = $1 } END { print m }' $SQ/plan.tsv)" -lt $NCYC ]; then python - <<PY || { say "ABORT: plan failed"; exit 1; }
import shutil, os
S, SQ, K, n = "$S", "$SQ", $K, $NCYC; rows = []
import math
per = 38147 / K; width = math.ceil(per); b = [round(per * c / 10) for c in range(11)]   # per-sub-model steps at each sync: 2B = 3,814.7 stream steps / K
for g in range(K):
    base = 19074 + g * width; mid = base + round((g + 1) * 10926 / K) - round(g * 10926 / K)   # 15.7B = joint step 30,000: 10,926 stream steps, split over the K sub-models
    rows += [(c, g, 3, base + b[c - 1], base + b[c], mid if c == 3 else 0) for c in range(1, n + 1)]
    d = f"{SQ}/templates/w3/group{g}"
    if not os.path.exists(d + "/train"):   # trainer state (stream position) of the joint 10B checkpoint + a sub-model config (the trainer rewrites config.json at every save)
        shutil.rmtree(d, ignore_errors=True); os.makedirs(d); shutil.copytree(f"{S}/$FULL/step19074/train", f"{d}/train"); shutil.copy(f"{S}/$FULL/step19074/.metadata.json", f"{d}/.metadata.json")
        shutil.copy(f"{S}/olmoe3_squares_mix20rand8_rp100m/templates/w2/group0/config.json", f"{d}/config.json")
assert all(r[4] > r[3] for r in rows) and all(r[3] < r[5] < r[4] for r in rows if r[5])
open(f"{SQ}/keep.tsv", "w").write("3 sync3\n5 match38148\n10 match57221\n")
open(f"{SQ}/plan.tsv.tmp", "w").write("".join("%d %d %d %d %d %d\n" % r for r in sorted(rows))); os.replace(f"{SQ}/plan.tsv.tmp", f"{SQ}/plan.tsv")
print("plan:", n, "cycles, steps per sub-model and cycle", min(r[4] - r[3] for r in rows), "to", max(r[4] - r[3] for r in rows))
PY
  rm -f $SQ/DONE; rm -rf "${LOG:?}/locks"; fi
NWORK="${NWORK:-2}"; [ $FULLRUN = 1 ] && NWORK="${NWORK_FINAL:-4}"
ended() { beaker experiment get "$1" --format json 2>/dev/null | grep -q '"exited"\|"canceled"'; }
row5() { awk -v c="$1" -v g="$2" -v k="$3" '$1 == c && $2 == g { print $k }' $SQ/plan.tsv; }
fin_of() { echo "$S/${RP}_c$1_g$2/step$(row5 $1 $2 5)"; }
mid_of() { echo "$S/${RP}_c3_g$1/step$(row5 3 $1 6)"; }
later_cut() { local n; [ -f $SQ/DONE ] && [ "$1" -lt "$(awk '{ if ($1 > m) m = $1 } END { print m }' $SQ/plan.tsv)" ] && return 0; for n in $(ls $SQ/cyc 2>/dev/null | sed 's/^c//'); do [ "$n" -gt "$1" ] && [ -f $SQ/cyc/c$n/PREP_DONE ] && return 0; done; return 1; }
workers() { local i e n u; [ -f $SQ/DONE ] && return 0
  [ "$(df --output=avail -BG /root | tail -n 1 | tr -dc 0-9)" -gt 500 ] || { say "less than 500 GB free on the shared filesystem: no worker launched"; return 0; }
  for i in $(seq 1 $NWORK); do e=$(tail -n 1 $LOG/${MODE}_worker$i.launched 2>/dev/null | grep -o '01[A-Z0-9]\{20,\}' | head -1)
    if [ -n "$e" ]; then ended $e || continue; fi
    n=$(cat $LOG/${MODE}_worker$i.launched 2>/dev/null | wc -l); [ $n -ge 10 ] && continue
    u=$(python3 scripts/sparse_experts/olmoe3_beaker_cmd_cli.py --name "$AT-lr$LR-$MODE-worker$i-$(date +%m%d%H%M%S)" --gpus 8 -- bash -c "K=$K WHOLE_LAYERS=$WH FIRST_CYCLE=$FIRST SEED_BASE=$SEEDB FIXED_GROUPS=1 REPART_WANDB=false SQUARES_NAME=$SQN RUN_PREFIX=$RP FULL=$FULL OLMOE3_LR=$LR OLMOE3_EMO_POOL_SELECT=mix:0.2 bash scripts/sparse_experts/olmoe3_squares/repartition_worker.sh" 2>&1 | tail -n 1)
    say "worker $i (launch $((n+1))): $u"; echo "$u" | grep -q beaker.org && echo "$u" >> $LOG/${MODE}_worker$i.launched; done; }
# ---- the 15.7B matched point: the 12 extra checkpoints of cycle 3, merged for evaluation only ----
midmerge() { local g; [ -f $LOG/mid.done ] && return 0; [ -f $SQ/merged/match30000/merge_info.json ] && return 0
  for g in $GS; do [ -f "$(mid_of $g)/train/rank0.pt" ] && [ -f "$(fin_of 3 $g)/train/rank0.pt" ] || return 1; done
  PYTHONPATH=external/OLMo-core/src OPENBLAS_NUM_THREADS=8 python scripts/sparse_experts/olmoe3_squares/merge_models.py --groups $SQ/groups/c3.json --full $S/$FULL/step19074 --subs "$(for g in $GS; do mid_of $g; done | paste -sd,)" \
    --weights "$(for g in $GS; do echo 1; done | paste -sd,)" --out $SQ/merged/match30000 --overwrite > $LOG/merge_mid.log 2>&1
  [ -f $SQ/merged/match30000/merge_info.json ] && say "15.7B point merged (evaluation only)" || say "15.7B merge FAILED (see $LOG/merge_mid.log)"; }
# ---- evaluations: everything that exists and is not scored yet, as one 8-GPU job reading its own snapshot ----
done_h() { [ -s "$1/none/meta.json" ] || [ -f "$1/none/rank0/DONE" ]; }
hl() { echo "python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint $2 --instances $W/olmoe3_routing/$3 --out-dir ${1/$S/$W}/none/rank0 --restrict none --batch-size 8 --log-every 100"; }
pl() { echo "python scripts/debug_validation/eval_ppl_validation.py --checkpoints $1 --out-dir $W/$SQN/ppl_validation --batch-size 8"; }
subs_eval() { local c=$1 s=$2 g d; for g in $GS; do d=$(fin_of $c $g); [ -f $d/train/rank0.pt ] || continue; done_h $H/sub${g}_match$s || hl $H/sub${g}_match$s ${d/$S/$W} $SAMPLE >> $CF; done; }
build() { local s; : > $CF
  if [ -f $SQ/merged/match30000/merge_info.json ]; then done_h $H/merged_match30000 || hl $H/merged_match30000 $W/$SQN/merged/match30000 $SAMPLE >> $CF; [ -f $SQ/ppl_validation/merged/match30000.json ] || pl $W/$SQN/merged/match30000 >> $CF; fi
  for s in 38148 57221; do [ -f $SQ/merged/match$s/merge_info.json ] || continue
    done_h $H/merged_match$s || hl $H/merged_match$s $W/$SQN/merged/match$s $SAMPLE >> $CF; done_h $HV/merged_match$s || hl $HV/merged_match$s $W/$SQN/merged/match$s $VAL >> $CF
    [ -f $SQ/ppl_validation/merged/match$s.json ] || pl $W/$SQN/merged/match$s >> $CF; done
  if [ $FULLRUN = 1 ]; then [ -f $LOG/c5.cleaned ] || subs_eval 5 38148; [ -f $SQ/merged/match57221/merge_info.json ] && subs_eval 10 57221; fi; }
evals() { local e tries u; CF=$LOG/evals_${MODE}_cur.txt
  e=$(tail -n 1 $LOG/evals_$MODE.launched 2>/dev/null | grep -o '01[A-Z0-9]\{20,\}' | head -1); if [ -n "$e" ] && ! ended $e; then return 0; fi
  build; [ -s $CF ] || return 0; tries=$(cat $LOG/evals_$MODE.launched 2>/dev/null | wc -l); [ $tries -ge 8 ] && return 0; cp $CF $LOG/evals_${MODE}_try$tries.txt
  u=$(python3 scripts/sparse_experts/olmoe3_beaker_cmd_cli.py --name "$AT-lr$LR-$MODE-evals-$(date +%m%d%H%M%S)" --gpus 8 -- bash scripts/sparse_experts/olmoe3_squares/eval_batch_8gpu.sh $W/$SQN/logs/evals_${MODE}_try$tries.txt 2>&1 | tail -n 1)
  say "evals ($(wc -l < $CF) commands): $u"; echo "$u" | grep -q beaker.org && echo "$u" >> $LOG/evals_$MODE.launched; }
# ---- storage: delete what is evaluated and superseded ----
cleanup() { local g
  if [ ! -f $LOG/mid.done ] && done_h $H/merged_match30000 && [ -f $SQ/ppl_validation/merged/match30000.json ] && later_cut 3; then
    rm -rf "${SQ:?}/merged/match30000" "${SQ:?}/merged/sync3"; for g in $GS; do rm -rf "$S/${RP:?}_c3_g$g"; done; touch $LOG/mid.done; say "15.7B point: evaluated (merged CE $(ce_of $H/merged_match30000)); its checkpoints deleted"; fi
  if [ $FULLRUN = 1 ] && [ ! -f $LOG/c5.cleaned ] && later_cut 5 && done_h $H/merged_match38148 && done_h $HV/merged_match38148; then
    for g in $GS; do done_h $H/sub${g}_match38148 || return 0; done
    rm -rf "${SQ:?}/merged/match38148"; for g in $GS; do rm -rf "$S/${RP:?}_c5_g$g"; done; touch $LOG/c5.cleaned; say "20B point: sub-models evaluated; its checkpoints deleted"; fi; }
scored() { local s=$1; done_h $H/merged_match$s && done_h $HV/merged_match$s && [ -f $SQ/ppl_validation/merged/match$s.json ]; }
LAST=38148; [ $FULLRUN = 1 ] && LAST=57221
while true; do workers; midmerge; evals; cleanup
  if [ -f $SQ/DONE ] && [ -f $LOG/mid.done ] && scored $LAST; then
    if [ $MODE = sweep ]; then break; fi
    ok=1; for g in $GS; do done_h $H/sub${g}_match57221 || ok=0; done; [ -f $LOG/c5.cleaned ] || ok=0; [ $ok = 1 ] && break; fi
  c=$(ls $SQ/cyc 2>/dev/null | sed 's/^c//' | sort -n | tail -1); [ "${c:-0}" != "${lastc:-}" ] && { say "cycle ${c:-0} of $NCYC"; lastc=${c:-0}; }; sleep 300; done
for d in $H/*/none $HV/*/none; do [ -f $d/rank0/DONE ] && [ ! -f $d/meta.json ] && python scripts/sparse_experts/olmoe3_routing/merge_routing.py $d 2>&1 | tail -1; done
if [ $FULLRUN = 1 ]; then PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_squares/strip_optim.py $(for g in $GS; do fin_of 10 $g; done) >> $LOG/strip.log 2>&1
  rm -rf "${SQ:?}/cyc" "${SQ:?}/templates"; for c in 8 9; do for g in $GS; do rm -rf "$S/${RP:?}_c${c}_g$g"; done; done
  say "merged held-out CE: 15.7B $(ce_of $H/merged_match30000), 20B $(ce_of $H/merged_match38148), 30B $(ce_of $H/merged_match57221)"; fi
say "selection CE at $LAST: $(ce_of $HV/merged_match$LAST); reporting $(ce_of $H/merged_match$LAST)"; say done
