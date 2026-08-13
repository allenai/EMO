#!/usr/bin/env bash
# Sequential driver for ONE arm (carry|fresh) of the k=32 CPT sweep. Runs LOCALLY on the
# CPU box; per cluster (fixed order 0..31):
#   1. extract    expert_subset_surgery.py extract  (arm pool -> 33-expert subset ckpt)
#   2. train      submit k32cpt_stage.sh to Beaker (8 nodes), poll until the stage's
#                 final checkpoint appears and the experiment finalizes
#   3. writeback  expert_subset_surgery.py writeback (trained subset -> arm pool)
# Fully resumable: stages with a pool writeback marker are skipped; a partially trained
# stage auto-resumes from its own save folder on relaunch.
#
# The arm pool starts as a copy of the 100B checkpoint's model_and_optim (created here on
# first run). Two arms = run this script twice (ARM=carry / ARM=fresh), concurrently.
#
#   ARM=carry  bash scripts/modular_extension/run_k32cpt_arm.sh
#   ARM=fresh  bash scripts/modular_extension/run_k32cpt_arm.sh
#   CLUSTERS="0" ARM=carry bash ...   # pilot: just cluster 0
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

: "${ARM:?set ARM=carry|fresh|carry_shuf}"
# carry_shuf = carried Adam moments but a FIXED shuffled cluster order (rng(0) permutation),
# to measure how much stage ordering matters.
SHUF_ORDER="2 11 25 21 10 4 29 16 23 6 18 26 3 30 8 0 19 12 20 13 7 5 17 14 27 22 9 28 24 1 15 31"
if [ "$ARM" = "carry_shuf" ]; then
    CLUSTERS="${CLUSTERS:-$SHUF_ORDER}"
else
    CLUSTERS="${CLUSTERS:-$(seq -s' ' 0 31)}"
fi

REPO="$(pwd)"
BASE_CKPT="${REPO}/models_v2/emo_64exp_50b_wsd_lr2e-3/step23842"
RUNS="${REPO}/modular_extension/k32_cpt_runs/${ARM}"
POOL="${RUNS}/pool"
TOKENS_DIR="${REPO}/modular_extension/data/emo_64exp_50b_wsd_lr2e-3_100B-130B/k32_cpt_tokens"
WEKA_RUNS="/weka/oe-training-default/ryanwang/EMO/modular_extension/k32_cpt_runs/${ARM}"

FRESH_FLAG=""
[ "$ARM" = "fresh" ] && FRESH_FLAG="--fresh-optim"

mkdir -p "$RUNS"
echo "$CLUSTERS" >> "${RUNS}/order.txt"   # provenance: the stage order this driver ran
if [ ! -f "${POOL}/model_and_optim/.metadata" ]; then
    echo "=== initializing ${ARM} pool from ${BASE_CKPT}"
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

    echo "=== stage ${tag}: extract (${ARM})"
    OPENBLAS_NUM_THREADS=16 PYTHONPATH=src python scripts/modular_extension/expert_subset_surgery.py extract \
        --pool "$POOL" --out "$subset" --cluster "$c" $FRESH_FLAG

    submit_stage() {  # retries around the flaky GitPython/launcher path; sets exp_id
        local attempt
        for attempt in 1 2 3 4 5; do
            MODE=beaker CLUSTER=$c ARM=$ARM \
                SUBSET_DIR="${WEKA_RUNS}/subset_${tag}" \
                SAVE_DIR="${WEKA_RUNS}/stage_${tag}" \
                TOKENS=$tokens \
                bash scripts/modular_extension/k32cpt_stage.sh 2>&1 | tee "${RUNS}/launch_${tag}.log" || true
            exp_id=$(grep -oE 'beaker.org/ex/[A-Z0-9]+' "${RUNS}/launch_${tag}.log" | head -1 | sed 's|beaker.org/ex/||')
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
        if [ -n "$final" ] && [ "${n_final:-0}" -ge 8 ] && [ "${n_bad:-0}" -eq 0 ]; then
            trained="${final%/model_and_optim/.metadata}"
            echo "=== stage ${tag}: trained checkpoint ${trained}"
            break
        fi
        sleep 180
    done

    echo "=== stage ${tag}: writeback"
    OPENBLAS_NUM_THREADS=16 PYTHONPATH=src python scripts/modular_extension/expert_subset_surgery.py writeback \
        --pool "$POOL" --trained "$trained" --selection "${subset}/selection.json"

    # disk hygiene: stage + subset checkpoints are reproducible from pool history; drop them
    rm -rf "$subset" "${save}"
    echo "=== stage ${tag}: DONE"
done
echo "=== ARM ${ARM}: all requested stages complete; pool at ${POOL}"
