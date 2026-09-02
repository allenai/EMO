#!/usr/bin/env bash
# PARENT: "scripts/meta_learning/eval_scripts/k32cpt_arm_worker.sh" (job structure) +
#         "scripts/meta_learning/eval_scripts/probe_eval_ce.py" (patch-and-eval).
# DESCRIPTION:
#     Step-granularity diagnostic: does ONE stage of selective cluster CPT hurt the full
#     128-expert model immediately (subset/routing shift) or only after many steps
#     (nonlocality / perturbation fragility)? For each PROBE cluster INDEPENDENTLY (all
#     starting from the arm's 20B base, NOT sequential): extract the 33-expert subset,
#     then train 1 step at a time up to STEPS, and after EVERY step evaluate full-model
#     CE on all 32 cluster held-out sets (4M tokens each). Evals patch the trained
#     subset experts into a bf16 base snapshot in memory (probe_eval_ce.py) -- no pool
#     copies, no writebacks. One 4-node job; per eval round each of the 32 GPUs handles
#     one cluster.
#
#     DISK HYGIENE: only the eval JSONs survive (kept in
#     meta_learning/k32_cpt_runs/evals_stepprobe/). The base snapshot, subsets, and all
#     per-step training checkpoints are deleted by the leader at the end.
#
#     Required env: MODEL (default sametok_ws_lam05). Tunables: STEPS (10),
#     PROBE_CLUSTERS ("0 1 2"), EVAL_TOKENS (4194304).
set -euo pipefail

MODEL="${MODEL:-sametok_ws_lam05}"
STEPS="${STEPS:-10}"
PROBE_CLUSTERS="${PROBE_CLUSTERS:-0 1 2}"
EVAL_TOKENS="${EVAL_TOKENS:-4194304}"
RANK="${BEAKER_REPLICA_RANK:-0}"
NREP="${BEAKER_REPLICA_COUNT:-1}"
LEADER="${BEAKER_LEADER_REPLICA_HOSTNAME:-127.0.0.1}"
NPROC="${BEAKER_ASSIGNED_GPU_COUNT:-8}"
RDZV_PORT=29768

WEKA_ROOT="/weka/oe-training-default/ryanwang/EMO"
RUNS="${WEKA_ROOT}/meta_learning/k32_cpt_runs/${MODEL}_stepprobe"   # deleted at end
EV="${WEKA_ROOT}/meta_learning/k32_cpt_runs/evals_stepprobe"        # kept
BASE_CKPT="${WEKA_ROOT}/meta_learning/meta128_${MODEL}/step4768"
BASE_SNAP="${RUNS}/base_bf16.pt"
TOKENS_DIR="${WEKA_ROOT}/meta_learning/data/meta128_20B-40B/k32_cpt_tokens_${MODEL}"
CONC="${WEKA_ROOT}/meta_learning/cluster/${MODEL}_step4768/k32_cpt/expert_concentration.json"
DATASET_CACHE="/weka/oe-training-default/ryanwang/dataset-cache"

wait_for() { local p=$1 poll=${2:-10}; until [ -e "$p" ]; do sleep "$poll"; done; }

mkdir -p "$RUNS" "$EV"
echo "stepprobe: MODEL=${MODEL} rank=${RANK}/${NREP} clusters='${PROBE_CLUSTERS}' steps=${STEPS}"

# --- base bf16 snapshot (leader; others wait) -------------------------------------
if [ "$RANK" = 0 ]; then
    PYTHONPATH=src python scripts/meta_learning/eval_scripts/probe_eval_ce.py make-base \
        --checkpoint "$BASE_CKPT" --out "$BASE_SNAP"
    touch "${RUNS}/.base_ready"
else
    wait_for "${RUNS}/.base_ready"
fi

# eval_round <tag> [<subset_ckpt> <selection>] -- each GPU evals cluster RANK*NPROC+g
eval_round() {
    local tag=$1 subset="${2:-}" sel="${3:-}"
    local pids=() g idx out
    for g in $(seq 0 $((NPROC - 1))); do
        idx=$((RANK * NPROC + g))
        [ "$idx" -lt 32 ] || continue
        out="${EV}/ce_${tag}_shard$(printf '%02d' "$idx").json"
        [ -f "$out" ] && continue
        CUDA_VISIBLE_DEVICES=$g PYTHONPATH=.:src python -u scripts/meta_learning/eval_scripts/probe_eval_ce.py eval \
            --base-snapshot "$BASE_SNAP" --config-from "$BASE_CKPT" \
            ${subset:+--subset-ckpt "$subset" --selection "$sel"} \
            --tokens-root "$TOKENS_DIR" --clusters "$idx" \
            --max-tokens-per-cluster "$EVAL_TOKENS" --batch-size 4 \
            --out "$out" > "${RUNS}/evallog_${tag}_g${g}.log" 2>&1 &
        pids+=($!)
    done
    local pid
    for pid in "${pids[@]:-}"; do
        if [ -n "$pid" ]; then wait "$pid"; fi
    done
    # explicit success: with every output already present the pids array is empty and
    # the loop's last test is falsy -- without this, set -e kills the worker.
    return 0
}

echo "=== eval round: base (step 0)"
eval_round "probe_${MODEL}_base_step00"

for c in $PROBE_CLUSTERS; do
    ctag=$(printf 'c%02d' "$c")
    subset="${RUNS}/subset_${ctag}"
    save="${RUNS}/stage_${ctag}"
    runname="k32cpt_probe_${MODEL}_${ctag}"

    if [ "$RANK" = 0 ]; then
        echo "=== probe ${ctag}: extract"
        OPENBLAS_NUM_THREADS=16 PYTHONPATH=src python scripts/meta_learning/eval_scripts/expert_subset_surgery.py extract \
            --pool "$BASE_CKPT" --out "$subset" --cluster "$c" --conc "$CONC"
        touch "${subset}/.ready"
    else
        wait_for "${subset}/.ready"
    fi

    # ONE training run to STEPS with per-step checkpoints (fixed_steps 1..STEPS-1 +
    # the final save at STEPS). NOTE: with WSD decay_steps=1, the LAST step (STEPS)
    # is the decay step (lr -> min), so the stepSTEPS row ~= step(STEPS-1); steps
    # 1..STEPS-1 train at the flat 2e-3. (An earlier incremental design set
    # max_duration=k per round, making EVERY trained step the decay step -> lr 0 ->
    # bitwise no-op training. Hence the single-run + fixed_steps shape.)
    if [ ! -f "${save}/step${STEPS}/model_and_optim/.metadata" ]; then
        fixed=$(seq -s', ' 1 $((STEPS - 1)))
        echo "=== probe ${ctag}: train ${STEPS} steps, per-step checkpoints (rank ${RANK})"
        torchrun \
            --nnodes="${NREP}:${NREP}" \
            --nproc-per-node="${NPROC}" \
            --rdzv-id="probe_${MODEL}_${ctag}" \
            --rdzv-backend=static \
            --rdzv-endpoint="${LEADER}:${RDZV_PORT}" \
            --node-rank="${RANK}" \
            --rdzv-conf="read_timeout=900" \
            src/scripts/train/olmoe-1B-7B_fsl.py "$runname" \
            --data-root="s3://ai2-llm" \
            --save-folder="${save}" \
            --load_path="${subset}/model_and_optim" \
            --load_trainer_state=false \
            --load_optim_state=true \
            --model.freeze_params='[embeddings.*, lm_head.*, blocks.*.attention.*, blocks.*.attention_norm.*, blocks.*.feed_forward_norm.*, blocks.*.feed_forward_moe.router.*]' \
            --dataset.mix=null \
            --dataset.paths="[${TOKENS_DIR}/cluster$(printf '%02d' "$c")/train/part-*.npy]" \
            --dataset.expand_glob=true \
            --dataset.dtype=uint32 \
            --work-dir="${DATASET_CACHE}" \
            --trainer.max_duration="{value: ${STEPS}, unit: steps}" \
            --scheduler=wsd \
            --warmup_steps=0 \
            --decay_steps=1 \
            --trainer.callbacks.checkpointer.save_interval=1000000 \
            --trainer.callbacks.checkpointer.fixed_steps="[${fixed}]" \
            --trainer.callbacks.checkpointer.ephemeral_save_interval=null \
            --trainer.callbacks.checkpointer.pre_train_checkpoint=false \
            --trainer.callbacks.downstream_evaluator.enabled=false \
            --trainer.callbacks.wandb.enabled=false \
            --model.block.feed_forward_moe.num_experts=33 \
            --dataset.generate_doc_lengths=true \
            --model.block.sequence_mixer.backend=flash_2 \
            --model-type="two-level_lb-batch_reduce-dp_sharedexp_randpool" \
            --min_document_expert_pool=33 \
            --max_document_expert_pool=33 \
            --eval_document_expert_pool=33 \
            --num_shared_experts=1 \
            --dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
            --model.block.name="moe" \
            --model.block.sequence_mixer.qk_norm=null \
            --lr=2e-3 \
            --model.block.feed_forward_moe.lb_loss_weight=1e-1
    fi
    wait_for "${save}/step${STEPS}/model_and_optim/.metadata"

    for k in $(seq 1 "$STEPS"); do
        echo "=== probe ${ctag}: eval after step ${k}"
        eval_round "probe_${MODEL}_${ctag}_step$(printf '%02d' "$k")" \
            "${save}/step${k}" "${subset}/selection.json"
    done

    if [ "$RANK" = 0 ]; then
        # delete artifacts only when every replica's eval shards for this cluster exist
        need=$((32 * STEPS)); t0=$(date +%s)
        while true; do
            have=$(ls "${EV}/ce_probe_${MODEL}_${ctag}_step"*_shard*.json 2>/dev/null | wc -l)
            [ "$have" -ge "$need" ] && break
            [ $(( $(date +%s) - t0 )) -gt 7200 ] && { echo "!!! ${ctag}: eval shards stuck at ${have}/${need}; keeping artifacts"; break; }
            sleep 30
        done
        [ "${have:-0}" -ge "$need" ] && rm -rf "$subset" "$save"
        echo "=== probe ${ctag}: DONE"
    fi
done

# --- final cleanup: leader deletes the whole probe tree once every replica is done ---
touch "${RUNS}/.done_rank${RANK}"
if [ "$RANK" = 0 ]; then
    for r in $(seq 0 $((NREP - 1))); do wait_for "${RUNS}/.done_rank${r}" 5; done
    rm -rf "$RUNS"
    echo "=== stepprobe COMPLETE: artifacts deleted; evals kept in ${EV}"
fi
