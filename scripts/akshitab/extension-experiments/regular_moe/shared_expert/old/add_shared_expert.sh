BASE_MODEL_PATH="/weka/oe-training-default/ryanwang/phdbrainstorm/FlexMoE/models/moe_1b14b_128experts_olmoe-mix_130B_prenorm_noqknorm_1123/step30995"

# Top global expert as per python src/scripts/eval/router_analysis.py --router-files router_evals/moe_1b14b_128experts_olmoe-mix_130B_prenorm_noqknorm_1123_step30995-hf/task-hellaswag_rc_test-router.jsonl
EXPERT_INIT="56"

SAVE_PATH="/weka/oe-training-default/akshitab/FlexMoE/models/extensions/moe_1b14b_128experts_1shared_expert_init_${EXPERT_INIT// /_}"

echo $EXPERT_INIT
echo $SAVE_PATH

python src/scripts/akshitab/add_finegrained_expert/add_shared_expert.py \
    -c ${BASE_MODEL_PATH} \
    -o ${SAVE_PATH} \
    --shared-expert-init-idx ${EXPERT_INIT}