#!/usr/bin/env bash
# The 10B / 0% point of every Q3 squares experiment (user request 2026-09-21): merge the UNTRAINED sliced start checkpoints of each
# variant (must reproduce the start model exactly) and score the merge and every slice on the variant's held-out sample, so every
# line of the Q3 charts starts at "no training done". Also links baseline_step19074 -> the start pass in each held-out dir, and runs
# the piecewise summary at match19074 where the variant has one.   bash scripts/sparse_experts/olmoe3_squares/init_point_evals.sh [variant ...]   (idempotent; detach)
set -u; cd "$(git rev-parse --show-toplevel)"; export PATH=/root/.conda/envs/emo/bin:$PATH
S=sparse_experts; W=/weka/oe-training-default/ryanwang/EMO/sparse_experts; SP=/tmp/claude-0/-root-EMO/c7db74f2-bbe3-4a2c-9d37-93c64250d7c6/scratchpad; mkdir -p $SP
# name | SQN | run prefix | K | full start run | held-out dir | sample | start tag (dir holding it) | piecewise out dir (- = none)
V_ALL="emo std noemo k8 pool learnedd stdrand emorand stdrand8 s128rand4 s128rand8"
spec() { case $1 in
  emo)      echo "olmoe3_squares|olmoe3_275m_emo_square|4|olmoe3_275m_emo_10b|runs_heldout20b|sample_8k_20b.npz|runs_heldout20b/emo_step19074|squares";;
  std)      echo "olmoe3_squares_std|olmoe3_275m_square|4|olmoe3_275m_10b|runs_heldout300b_std|sample_8k_300b.npz|runs_heldout300b_std/std_step19074|olmoe3_squares_std_w2";;
  noemo)    echo "olmoe3_squares_noemo|olmoe3_275m_noemo_square|4|olmoe3_275m_emo_10b|runs_heldout20b_noemo|sample_8k_20b.npz|runs_heldout20b_noemo/emo_step19074|-";;
  k8)       echo "olmoe3_squares_k8|olmoe3_275m_emo_k8_square|8|olmoe3_275m_emo_10b|runs_heldout20b_k8|sample_8k_20b.npz|runs_heldout20b_k8/emo_step19074|-";;
  pool)     echo "olmoe3_squares_pool64or512|olmoe3_275m_pool64or512_square|4|olmoe3_275m_emo_pool64or512_10b|runs_heldout20b_pool64or512|sample_8k_20b.npz|runs_heldout20b_pool64or512/pool64or512_step19074|olmoe3_squares_pool64or512";;
  learnedd) echo "olmoe3_squares_learnedd|olmoe3_275m_learnedd_square|4|olmoe3_275m_emo_learnedd_10b|runs_heldout20b_learnedd|sample_8k_20b.npz|runs_heldout20b_learnedd/learnedd_step19074|olmoe3_squares_learnedd";;
  stdrand)  echo "olmoe3_squares_stdrand|olmoe3_275m_stdrand_square|4|olmoe3_275m_10b|runs_heldout300b_stdrand|sample_8k_300b.npz|runs_heldout300b_std/std_step19074|-";;
  emorand)  echo "olmoe3_squares_emorand|olmoe3_275m_emorand_square|4|olmoe3_275m_emo_10b|runs_heldout300b_emorand|sample_8k_300b.npz|runs_heldout300b_emo/emo_step19074|-";;
  stdrand8) echo "olmoe3_squares_stdrand8|olmoe3_275m_stdrand8_square|8|olmoe3_275m_10b|runs_heldout300b_stdrand8|sample_8k_300b.npz|runs_heldout300b_std/std_step19074|-";;
  s128rand4) echo "olmoe3_squares_s128rand4|olmoe3_275m_s128rand4_square|4|olmoe3_275m_128e_10b|runs_heldout300b_s128rand4|sample_8k_300b.npz|runs_heldout300b_s128/s128_step19074|-";;
  s128rand8) echo "olmoe3_squares_s128rand8|olmoe3_275m_s128rand8_square|8|olmoe3_275m_128e_10b|runs_heldout300b_s128rand8|sample_8k_300b.npz|runs_heldout300b_s128/s128_step19074|-";;
  *) return 1;; esac; }
say() { echo "$(date -u +%m-%d\ %H:%M) [$V] $*"; }
launch() { local name=$1; shift; local log=$SP/launch_$name.log; local u=""
  for attempt in 1 2 3 4 5 6; do
    PYTHONPATH=external/OLMo-core/src python scripts/sparse_experts/olmoe3_beaker_cmd.py --name "$name" --gpus 1 --allocated -- "$@" > "$log" 2>&1
    u=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -aoE 'beaker.org/ex/[A-Z0-9]+' | head -1); [ -n "$u" ] && break; sleep 45; done
  say "$name: ${u:-LAUNCH FAILED}"; }
have() { [ -f "$1/meta.json" ] || [ -f "$1/rank0/DONE" ]; }
heldout() { local tag=$1 ckpt=$2; have $S/olmoe3_routing/$HR/$tag/none || launch "$SQN-init-$tag" python scripts/sparse_experts/olmoe3_routing/extract_routing.py --checkpoint "$ckpt" --instances "$W/olmoe3_routing/$SAMPLE" --out-dir "$W/olmoe3_routing/$HR/$tag/none/rank0" --restrict none --batch-size 8 --log-every 100; }
for V in ${*:-$V_ALL}; do IFS='|' read -r SQN RP K FULL HR SAMPLE STARTP PW <<< "$(spec $V)" || { echo "unknown $V"; continue; }
  SQ=$S/$SQN; [ -f $SQ/init/group0/.metadata ] || { say "no init slices, skipping"; continue; }
  # baseline_step19074 = the start pass (same checkpoint) in the held-out dir(s) the report reads
  for hd in $S/olmoe3_routing/$HR $S/olmoe3_routing/$(dirname $STARTP); do [ -e $hd/baseline_step19074 ] || ln -s "$(python -c "import os; print(os.path.relpath('$S/olmoe3_routing/$STARTP', '$hd'))")" $hd/baseline_step19074; done
  # sliced start checkpoints wrapped as evaluable checkpoint dirs (config.json of the trained square of the same group)
  ok=1; for g in $(seq 0 $((K-1))); do cfg=$(ls -d $S/${RP}$g/step*/config.json 2>/dev/null | head -1); [ -z "$cfg" ] && { say "group $g has no trained checkpoint yet (config.json needed); skipping variant for now"; ok=0; break; }
    d=$SQ/init_ckpt/group$g; mkdir -p $d; [ -f $d/config.json ] || cp $cfg $d/config.json; [ -e $d/model_and_optim ] || ln -s ../../init/group$g $d/model_and_optim; done
  [ $ok = 1 ] || continue
  SHARES=$(python -c "import json; print(','.join(f'{x:.4f}' for x in json.load(open('$SQ/pack/stats.json'))['token_share']))")
  subs=""; for g in $(seq 0 $((K-1))); do subs="$subs,$SQ/init/group$g"; done; subs=${subs#,}
  [ -f $SQ/merged/match19074/merge_info.json ] || { PYTHONPATH=external/OLMo-core/src OPENBLAS_NUM_THREADS=8 python scripts/sparse_experts/olmoe3_squares/merge_models.py --groups $SQ/groups.json --full $S/$FULL/step19074 --subs "$subs" --weights "$SHARES" --out $SQ/merged/match19074 --overwrite 2>&1 | tail -1 | cut -c1-120; say "init merge done"; }
  heldout merged_match19074 $W/$SQN/merged/match19074
  [ -f $SQ/ppl_validation/merged/match19074.json ] || launch "$SQN-init-ppl" python scripts/debug_validation/eval_ppl_validation.py --checkpoints "$W/$SQN/merged/match19074" --out-dir "$W/$SQN/ppl_validation" --batch-size 8
  for g in $(seq 0 $((K-1))); do heldout sub${g}_match19074 $W/$SQN/init_ckpt/group$g; done
done
# collect + piecewise at match19074
for V in ${*:-$V_ALL}; do IFS='|' read -r SQN RP K FULL HR SAMPLE STARTP PW <<< "$(spec $V)" || continue; SQ=$S/$SQN
  [ -f $SQ/merged/match19074/merge_info.json ] || continue
  for tag in merged_match19074 $(for g in $(seq 0 $((K-1))); do echo sub${g}_match19074; done); do until have $S/olmoe3_routing/$HR/$tag/none; do sleep 300; done; d=$S/olmoe3_routing/$HR/$tag/none; [ -f $d/meta.json ] || python scripts/sparse_experts/olmoe3_routing/merge_routing.py $d 2>&1 | tail -1 | cut -c1-80; done
  m=$(python -c "import json; print(round(json.load(open('$S/olmoe3_routing/$HR/merged_match19074/none/meta.json'))['mean_ce'],4))"); s0=$(python -c "import json; print(round(json.load(open('$S/olmoe3_routing/$STARTP/none/meta.json'))['mean_ce'],4))")
  say "SANITY merged init $m vs start model $s0"
  [ "$PW" = - ] || HELDOUT_DIR=$HR SQUARES_NAME=$SQN START_TAG=$(basename $STARTP) python scripts/sparse_experts/olmoe3_squares/piecewise_eval.py --name match19074 --out-dir claude_outputs/olmoe3_routing/$PW | head -2
done
echo "$(date -u +%m-%d\ %H:%M) init-point evals done"
