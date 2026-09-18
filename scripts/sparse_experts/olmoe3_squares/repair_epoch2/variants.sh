# shellcheck shell=bash
# Variant table of the olmoe3_squares epoch-2 repair (2026-09-17). Sourced by worker.sh / launch.sh / downstream.sh.
#
# Every square run was given ceil(tokens / 524288) steps while its re-packed documents hold only floor(...) full
# batches, so the last step of every square trained on one reshuffled epoch-2 batch (Beaker log: "Starting epoch 2"
# right before "Training complete"); all "100%" merges, finetunes, the router-only finetune and the standard-MoE
# window 2 derived from those finals. The repair: resume every square from its latest checkpoint before the final,
# train to floor(tokens / 524288) with the (now fixed) trainer, then redo everything downstream of the final.
variant_env() {
  V=$1; PACK=pack; K=4; EXTRA_ENV=""; HAS_FT2=1; HAS_SUB=1
  case $V in
    emo)         SQN=olmoe3_squares;             RP=olmoe3_275m_emo_square;         EMO=1; PFX=olmoe3_275m_emo_;         HR=runs_heldout20b;             FULL=olmoe3_275m_emo_10b;             BRUN=olmoe3_275m_emo_20b_1node;         START=emo_step19074;         FTTAG=emo;         SQO=squares;;
    noemo)       SQN=olmoe3_squares_noemo;       RP=olmoe3_275m_noemo_square;       EMO=0; PFX=olmoe3_275m_noemo_;       HR=runs_heldout20b_noemo;       FULL=olmoe3_275m_emo_10b;             BRUN=olmoe3_275m_emo_20b_1node;         START=emo_step19074;         FTTAG=noemo;       SQO=olmoe3_squares_noemo; HAS_FT2=0; HAS_SUB=0;;
    std)         SQN=olmoe3_squares_std;         RP=olmoe3_275m_square;             EMO=0; PFX=olmoe3_275m_;             HR=runs_heldout20b_std;         FULL=olmoe3_275m_10b;                 BRUN=olmoe3_275m_20b_1node;             START=std_step19074;         FTTAG=std;         SQO=olmoe3_squares_std;;
    k8)          SQN=olmoe3_squares_k8;          RP=olmoe3_275m_emo_k8_square;      EMO=1; PFX=olmoe3_275m_emo_k8_;      HR=runs_heldout20b_k8;          FULL=olmoe3_275m_emo_10b;             BRUN=olmoe3_275m_emo_20b_1node;         START=emo_step19074;         FTTAG=k8;          SQO=olmoe3_squares_k8; K=8; HAS_SUB=0;;
    pool64or512) SQN=olmoe3_squares_pool64or512; RP=olmoe3_275m_pool64or512_square; EMO=1; PFX=olmoe3_275m_pool64or512_; HR=runs_heldout20b_pool64or512; FULL=olmoe3_275m_emo_pool64or512_10b; BRUN=olmoe3_275m_pool64or512_20b_1node; START=pool64or512_step19074; FTTAG=pool64or512; SQO=olmoe3_squares_pool64or512
                 EXTRA_ENV="OLMOE3_EMO_POOL_DIST=choice:64,512";;
    learnedd)    SQN=olmoe3_squares_learnedd;    RP=olmoe3_275m_learnedd_square;    EMO=1; PFX=olmoe3_275m_learnedd_;    HR=runs_heldout20b_learnedd;    FULL=olmoe3_275m_emo_learnedd_10b;    BRUN=olmoe3_275m_learnedd_20b_1node;    START=learnedd_step19074;    FTTAG=learnedd;    SQO=olmoe3_squares_learnedd
                 EXTRA_ENV="OLMOE3_EMO_LEARNED_D=1 OLMOE3_LD_SIGNAL=coverage OLMOE3_LD_LAMBDA=1.0 OLMOE3_LD_LAMBDA_COV=1.0 OLMOE3_LD_TEMP=2.0 OLMOE3_LD_WARMUP=0 OLMOE3_LD_LR_MULT=10";;
    *) echo "unknown variant $V (emo|noemo|std|k8|pool64or512|learnedd)"; return 1;;
  esac
}
# fixed checkpoint steps of a square with s steps: the baseline's progress fractions + the final step
fixed_steps() { python -c "s=$1; print(','.join(str(max(1, round(s*f/19074))) for f in (926, 5926, 10926, 15926)) + f',{s}')"; }
# one "g:tokens:s_new:s_old:resume" spec per square of the variant, from the pack stats and the run dirs (local view)
variant_specs() {
  python - "$1" "$2" "$3" "$4" <<'PY'
import json, re, sys, os
sqn, rp, pack, k = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
tok = json.load(open(f"sparse_experts/{sqn}/{pack}/stats.json"))["tokens_per_group"]
for g in range(k):
    t = tok[g]; s = t // 524288; run = f"sparse_experts/{rp}{g}"
    steps = sorted(int(m.group(1)) for d in os.listdir(run) if (m := re.fullmatch(r"step(\d+)(_epoch2)?", d)))
    old = max(steps); assert old == s + 1, (run, old, s)   # the tainted final = ceil
    resume = max(x for x in steps if x < old and x != s)   # latest full checkpoint before the final (ephemeral or fixed)
    print(f"{g}:{t}:{s}:{old}:{resume}")
PY
}
