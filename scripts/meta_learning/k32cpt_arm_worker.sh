#!/usr/bin/env bash
# PARENT: "scripts/meta_learning/run_k32cpt_arm.sh" (sequential logic) +
#         "scripts/meta_learning/k32cpt_stage.sh" (training flags).
# DESCRIPTION:
#     Per-REPLICA worker for the single-job phase-2 k=32 CPT sweep: ONE 4-node Beaker
#     experiment per arm runs ALL 32 sequential stages internally (no per-stage
#     requeue). Submitted by launch_k32cpt_arm_job.sh with --no-torchrun; every replica
#     runs this script, which per cluster:
#       1. rank 0: expert_subset_surgery.py extract (idempotent) -> .ready flag
#       2. all replicas: multi-node torchrun of the stage training (gantry's own
#          static-rendezvous template, envs BEAKER_REPLICA_RANK / _COUNT /
#          BEAKER_LEADER_REPLICA_HOSTNAME from --nodes>1 leader selection)
#       3. rank 0: writeback --snapshot (bf16 pool snapshot) -> wroteback marker;
#          other replicas wait on the marker
#     Fully resumable via the wroteback markers + per-stage trainer auto-resume, so
#     Beaker preemption/restart of the whole experiment is safe. Per-stage evals are
#     NOT launched here -- the local run_snapshot_evals.sh watches the snapshots.
#
#     Required env: MODEL (vanilla|sametok_ws_lam05|...). Runs inside the gantry repo
#     clone; all data/pool paths are absolute weka paths.
set -euo pipefail

: "${MODEL:?set MODEL=vanilla|sametok_ws_lam05|...}"
RANK="${BEAKER_REPLICA_RANK:-0}"
NREP="${BEAKER_REPLICA_COUNT:-1}"
LEADER="${BEAKER_LEADER_REPLICA_HOSTNAME:-127.0.0.1}"
NPROC="${BEAKER_ASSIGNED_GPU_COUNT:-8}"
RDZV_PORT=29765

WEKA_ROOT="/weka/oe-training-default/ryanwang/EMO"
RUNS="${WEKA_ROOT}/meta_learning/k32_cpt_runs/${MODEL}"
POOL="${RUNS}/pool"
TOKENS_DIR="${WEKA_ROOT}/meta_learning/data/meta128_20B-40B/k32_cpt_tokens_${MODEL}"
CONC="${WEKA_ROOT}/meta_learning/cluster/${MODEL}_step4768/k32_cpt/expert_concentration.json"
BASE_CKPT="${WEKA_ROOT}/meta_learning/meta128_${MODEL}/step4768"
DATASET_CACHE="/weka/oe-training-default/ryanwang/dataset-cache"

wait_for() {  # path [poll_seconds]
    local p=$1 poll=${2:-15}
    until [ -e "$p" ]; do sleep "$poll"; done
}

echo "worker: MODEL=${MODEL} rank=${RANK}/${NREP} leader=${LEADER} nproc=${NPROC}"

# --- pool init (leader only; others wait) -----------------------------------------
if [ "$RANK" = 0 ] && [ ! -f "${POOL}/model_and_optim/.metadata" ]; then
    echo "=== initializing ${MODEL} pool from ${BASE_CKPT}"
    mkdir -p "$POOL"
    rm -rf "${POOL}/model_and_optim.copying" "${POOL}/model_and_optim"
    cp -r "${BASE_CKPT}/model_and_optim" "${POOL}/model_and_optim.copying"
    mv "${POOL}/model_and_optim.copying" "${POOL}/model_and_optim"
    cp "${BASE_CKPT}/config.json" "$POOL/" 2>/dev/null || true
fi
wait_for "${POOL}/model_and_optim/.metadata"

# --- trunk-matched training flags (see k32cpt_stage.sh) ---------------------------
lr=2e-3
lb=1e-1
num_shared_experts=1
num_experts=33              # 32 standard + 1 shared
min_document_expert_pool=33 # pinned full pool: standard training, no random pools
max_document_expert_pool=33
eval_document_expert_pool=33
warmup_steps=0              # carry arm: Adam moments carried; flat 2e-3
decay_steps=1

for c in $(seq 0 31); do
    tag=$(printf 'c%02d' "$c")
    marker="${POOL}/wroteback_cluster$(printf '%02d' "$c").json"
    if [ -f "$marker" ]; then
        [ "$RANK" = 0 ] && echo "=== stage ${tag}: already written back, skipping"
        continue
    fi
    subset="${RUNS}/subset_${tag}"
    save="${RUNS}/stage_${tag}"
    tokens=$(python -c "import json; print(json.load(open('${TOKENS_DIR}/cluster$(printf '%02d' "$c")/manifest.json'))['train_tokens_with_eos'])")
    runname="k32cpt_${MODEL}_${tag}"

    if [ "$RANK" = 0 ]; then
        echo "=== stage ${tag}: extract"
        OPENBLAS_NUM_THREADS=16 PYTHONPATH=src python scripts/meta_learning/expert_subset_surgery.py extract \
            --pool "$POOL" --out "$subset" --cluster "$c" --conc "$CONC"
        touch "${subset}/.ready"
    else
        wait_for "${subset}/.ready"
    fi

    echo "=== stage ${tag}: train (${tokens} tokens; rank ${RANK})"
    torchrun \
        --nnodes="${NREP}:${NREP}" \
        --nproc-per-node="${NPROC}" \
        --rdzv-id="k32cpt_${MODEL}_${tag}" \
        --rdzv-backend=static \
        --rdzv-endpoint="${LEADER}:${RDZV_PORT}" \
        --node-rank="${RANK}" \
        --rdzv-conf="read_timeout=420" \
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
        --trainer.max_duration="{value: ${tokens}, unit: tokens}" \
        --scheduler=wsd \
        --warmup_steps=${warmup_steps} \
        --decay_steps=${decay_steps} \
        --trainer.callbacks.checkpointer.save_interval=1000000 \
        --trainer.callbacks.checkpointer.ephemeral_save_interval=500 \
        --trainer.callbacks.checkpointer.keep_ephemeral=2 \
        --trainer.callbacks.checkpointer.pre_train_checkpoint=false \
        --trainer.callbacks.downstream_evaluator.enabled=false \
        --trainer.callbacks.wandb.enabled=true \
        --trainer.callbacks.wandb.entity=ryanyxw \
        --trainer.callbacks.wandb.project=emo-extension \
        --trainer.callbacks.wandb.name="${runname}" \
        --trainer.callbacks.wandb.tags="[pretraining, meta_learning, k32cpt, ${MODEL}]" \
        --model.block.feed_forward_moe.num_experts=${num_experts} \
        --dataset.generate_doc_lengths=true \
        --model.block.sequence_mixer.backend=flash_2 \
        --model-type="two-level_lb-batch_reduce-dp_sharedexp_randpool" \
        --min_document_expert_pool=${min_document_expert_pool} \
        --max_document_expert_pool=${max_document_expert_pool} \
        --eval_document_expert_pool=${eval_document_expert_pool} \
        --num_shared_experts=${num_shared_experts} \
        --dataset.instance_filter_config='{repetition_max_period: 13, repetition_min_period: 1, repetition_max_count: 32}' \
        --model.block.name="moe" \
        --model.block.sequence_mixer.qk_norm=null \
        --lr=${lr} \
        --model.block.feed_forward_moe.lb_loss_weight=${lb}

    if [ "$RANK" = 0 ]; then
        final=$(ls -d "${save}"/step*/model_and_optim/.metadata 2>/dev/null | sort -V | tail -1)
        [ -n "$final" ] || { echo "!!! stage ${tag}: training exited but no final checkpoint" >&2; exit 1; }
        trained="${final%/model_and_optim/.metadata}"
        echo "=== stage ${tag}: writeback (${trained})"
        OPENBLAS_NUM_THREADS=16 PYTHONPATH=src python scripts/meta_learning/expert_subset_surgery.py writeback \
            --pool "$POOL" --trained "$trained" --selection "${subset}/selection.json" --snapshot
        rm -rf "$subset" "$save"
        echo "=== stage ${tag}: DONE"
    else
        wait_for "$marker" 30
    fi
done
echo "=== MODEL ${MODEL}: all 32 stages complete (rank ${RANK}); pool at ${POOL}"
