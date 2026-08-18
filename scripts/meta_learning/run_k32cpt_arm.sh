#!/usr/bin/env bash
# PARENT: "scripts/modular_extension/run_k32cpt_arm.sh"
# Sequential driver for ONE MODEL ARM of the meta_learning phase-2 k=32 CPT sweep
# (carry optimizer only). Runs LOCALLY on the CPU box; per cluster (fixed order 0..31):
#   1. extract    expert_subset_surgery.py extract  (arm pool -> 33-expert subset ckpt,
#                 selection from the arm's OWN expert_concentration.json)
#   2. train      submit k32cpt_stage.sh to Beaker (4 nodes), poll until the stage's
#                 final checkpoint appears and the experiment finalizes
#   3. writeback  expert_subset_surgery.py writeback (trained subset -> arm pool)
# Fully resumable: stages with a pool writeback marker are skipped; a partially trained
# stage auto-resumes from its own save folder on relaunch.
#
# The arm pool starts as a copy of the arm's 20B checkpoint (step4768) model_and_optim
# (created here on first run).
#
#   MODEL=sametok_ws_lam05 bash scripts/meta_learning/run_k32cpt_arm.sh
#   MODEL=vanilla          bash scripts/meta_learning/run_k32cpt_arm.sh
#   CLUSTERS="0" MODEL=sametok_ws_lam05 bash ...   # pilot: just cluster 0
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

: "${MODEL:?set MODEL=vanilla|sametok_ws_lam05|...}"
STEP="${STEP:-4768}"
CLUSTERS="${CLUSTERS:-$(seq -s' ' 0 31)}"

REPO="$(pwd)"
BASE_CKPT="${REPO}/meta_learning/meta128_${MODEL}/step${STEP}"
RUNS="${REPO}/meta_learning/k32_cpt_runs/${MODEL}"
POOL="${RUNS}/pool"
TOKENS_DIR="${REPO}/meta_learning/data/meta128_20B-40B/k32_cpt_tokens_${MODEL}"
CONC="${REPO}/meta_learning/cluster/${MODEL}_step${STEP}/k32_cpt/expert_concentration.json"
WEKA_RUNS="/weka/oe-training-default/ryanwang/EMO/meta_learning/k32_cpt_runs/${MODEL}"

[ -f "$CONC" ] || { echo "ERROR: missing ${CONC} (run cluster_expert_concentration.py first)" >&2; exit 1; }

mkdir -p "$RUNS"
echo "$CLUSTERS" >> "${RUNS}/order.txt"   # provenance: the stage order this driver ran
if [ ! -f "${POOL}/model_and_optim/.metadata" ]; then
    echo "=== initializing ${MODEL} pool from ${BASE_CKPT}"
    mkdir -p "$POOL"
    rm -rf "${POOL}/model_and_optim.copying" "${POOL}/model_and_optim"  # stale partial init
    cp -r "${BASE_CKPT}/model_and_optim" "${POOL}/model_and_optim.copying"
    mv "${POOL}/model_and_optim.copying" "${POOL}/model_and_optim"
    cp "${BASE_CKPT}/config.json" "$POOL/" 2>/dev/null || true
fi

for c in $CLUSTERS; do
    tag=$(printf 'c%02d' "$c")
    marker="${POOL}/wroteback_cluster$(printf '%02d' "$c").json"
    if [ -f "$marker" ]; then
        echo "=== stage ${tag}: already written back, skipping"
        continue
    fi
    subset="${RUNS}/subset_${tag}"
    save="${RUNS}/stage_${tag}"
    tokens=$(python -c "import json; print(json.load(open('${TOKENS_DIR}/cluster$(printf '%02d' "$c")/manifest.json'))['train_tokens_with_eos'])")

    echo "=== stage ${tag}: extract (${MODEL})"
    OPENBLAS_NUM_THREADS=16 PYTHONPATH=src python scripts/meta_learning/expert_subset_surgery.py extract \
        --pool "$POOL" --out "$subset" --cluster "$c" --conc "$CONC"

    submit_stage() {  # retries around the flaky GitPython/launcher path; sets exp_id
        local attempt
        for attempt in 1 2 3 4 5; do
            MODE=beaker MODEL=$MODEL CLUSTER=$c \
                SUBSET_DIR="${WEKA_RUNS}/subset_${tag}" \
                SAVE_DIR="${WEKA_RUNS}/stage_${tag}" \
                TOKENS=$tokens \
                bash scripts/meta_learning/k32cpt_stage.sh 2>&1 | tee "${RUNS}/launch_${tag}.log" || true
            exp_id=$(grep -oE 'beaker.org/ex/[A-Z0-9]+' "${RUNS}/launch_${tag}.log" 2>/dev/null | head -1 | sed 's|beaker.org/ex/||' || true)
            [ -n "$exp_id" ] && return 0
            echo "!!! stage ${tag}: submit attempt ${attempt} produced no experiment id; retrying in 60s"
            sleep 60
        done
        echo "!!! stage ${tag}: submit failed 5x, giving up"
        return 1
    }

    echo "=== stage ${tag}: train (${tokens} tokens)"
    # blocking submit-and-wait: submit no-follow, then poll for the final checkpoint.
    submit_stage
    echo "=== stage ${tag}: experiment ${exp_id}; polling"
    while true; do
        js=$(timeout 30 beaker experiment get "$exp_id" --format json 2>/dev/null || true)
        n_final=$(echo "$js" | jq '[.[0].jobs[].status | select(.finalized != null)] | length' 2>/dev/null || echo 0)
        n_bad=$(echo "$js" | jq '[.[0].jobs[].status | select(.exitCode != null and .exitCode != 0)] | length' 2>/dev/null || echo 0)
        if [ "${n_bad:-0}" -gt 0 ]; then
            echo "!!! stage ${tag}: replica failed (https://beaker.org/ex/${exp_id}); relaunching stage"
            submit_stage
        fi
        final=$(ls -d "${save}"/step*/model_and_optim/.metadata 2>/dev/null | sort -V | tail -1 || true)
        if [ -n "$final" ] && [ "${n_final:-0}" -ge 4 ] && [ "${n_bad:-0}" -eq 0 ]; then
            trained="${final%/model_and_optim/.metadata}"
            echo "=== stage ${tag}: trained checkpoint ${trained}"
            break
        fi
        sleep 180
    done

    echo "=== stage ${tag}: writeback"
    OPENBLAS_NUM_THREADS=16 PYTHONPATH=src python scripts/meta_learning/expert_subset_surgery.py writeback \
        --pool "$POOL" --trained "$trained" --selection "${subset}/selection.json"

    # disk hygiene: stage + subset checkpoints are reproducible from pool history; drop them
    rm -rf "$subset" "${save}"
    echo "=== stage ${tag}: DONE"
done
echo "=== MODEL ${MODEL}: all requested stages complete; pool at ${POOL}"
