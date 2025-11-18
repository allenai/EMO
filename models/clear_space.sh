PARENT_MODELS=(
    "twolevelbatchlb-32_1b14b_stability_filter-true_zlossweight-1e-3_1115/step30995"
#    "moe_1b7b_128experts_olmoe-mix_130B_1103/step30995"
#    "twolevel-32_1b7b_128experts_olmoe-mix_130B_1110/step30995"
)

postfix="-keepk8"

FINETUNE_TASKS=(
    "task-arc_easy_rc_train_0shot_finetune${postfix}/step0"
    "task-arc_easy_rc_train_0shot_finetune${postfix}/step84"
    "task-arc_easy_rc_train_0shot_finetune${postfix}/step168"
    "task-arc_easy_rc_train_0shot_finetune${postfix}/step252"
    "task-arc_easy_rc_train_0shot_finetune${postfix}/step336"
    "task-arc_easy_rc_train_0shot_finetune${postfix}/step420"

    "task-arc_challenge_rc_train_0shot_finetune${postfix}/step0"
    "task-arc_challenge_rc_train_0shot_finetune${postfix}/step41"
    "task-arc_challenge_rc_train_0shot_finetune${postfix}/step82"
    "task-arc_challenge_rc_train_0shot_finetune${postfix}/step123"
    "task-arc_challenge_rc_train_0shot_finetune${postfix}/step164"
    "task-arc_challenge_rc_train_0shot_finetune${postfix}/step205"
    "task-arc_challenge_rc_train_0shot_finetune${postfix}/step207"

    "task-boolq_rc_train_0shot_finetune${postfix}/step0"
    "task-boolq_rc_train_0shot_finetune${postfix}/step315"
    "task-boolq_rc_train_0shot_finetune${postfix}/step630"
    "task-boolq_rc_train_0shot_finetune${postfix}/step945"
    "task-boolq_rc_train_0shot_finetune${postfix}/step1260"
    "task-boolq_rc_train_0shot_finetune${postfix}/step1575"
    "task-boolq_rc_train_0shot_finetune${postfix}/step1578"

    "task-csqa_rc_train_0shot_finetune${postfix}/step0"
    "task-csqa_rc_train_0shot_finetune${postfix}/step327"
    "task-csqa_rc_train_0shot_finetune${postfix}/step654"
    "task-csqa_rc_train_0shot_finetune${postfix}/step981"
    "task-csqa_rc_train_0shot_finetune${postfix}/step1308"
    "task-csqa_rc_train_0shot_finetune${postfix}/step1635"
    "task-csqa_rc_train_0shot_finetune${postfix}/step1638"

    "task-hellaswag_rc_train_0shot_finetune${postfix}/step0"
    "task-hellaswag_rc_train_0shot_finetune${postfix}/step1458"
    "task-hellaswag_rc_train_0shot_finetune${postfix}/step2916"
    "task-hellaswag_rc_train_0shot_finetune${postfix}/step4374"
    "task-hellaswag_rc_train_0shot_finetune${postfix}/step5832"
    "task-hellaswag_rc_train_0shot_finetune${postfix}/step7290"
    "task-hellaswag_rc_train_0shot_finetune${postfix}/step7293"

    "task-openbookqa_rc_train_0shot_finetune${postfix}/step0"
    "task-openbookqa_rc_train_0shot_finetune${postfix}/step185"
    "task-openbookqa_rc_train_0shot_finetune${postfix}/step370"
    "task-openbookqa_rc_train_0shot_finetune${postfix}/step555"
    "task-openbookqa_rc_train_0shot_finetune${postfix}/step740"
    "task-openbookqa_rc_train_0shot_finetune${postfix}/step925"
    "task-openbookqa_rc_train_0shot_finetune${postfix}/step927"

    "task-piqa_rc_train_0shot_finetune${postfix}/step0"
    "task-piqa_rc_train_0shot_finetune${postfix}/step566"
    "task-piqa_rc_train_0shot_finetune${postfix}/step1132"
    "task-piqa_rc_train_0shot_finetune${postfix}/step1698"
    "task-piqa_rc_train_0shot_finetune${postfix}/step2264"
    "task-piqa_rc_train_0shot_finetune${postfix}/step2830"
    "task-piqa_rc_train_0shot_finetune${postfix}/step2832"

    "task-socialiqa_rc_train_0shot_finetune${postfix}/step0"
    "task-socialiqa_rc_train_0shot_finetune${postfix}/step1215"
    "task-socialiqa_rc_train_0shot_finetune${postfix}/step2430"
    "task-socialiqa_rc_train_0shot_finetune${postfix}/step3645"
    "task-socialiqa_rc_train_0shot_finetune${postfix}/step4860"
    "task-socialiqa_rc_train_0shot_finetune${postfix}/step6075"

    "task-winogrande_rc_train_0shot_finetune${postfix}/step0"
    "task-winogrande_rc_train_0shot_finetune${postfix}/step1477"
    "task-winogrande_rc_train_0shot_finetune${postfix}/step2954"
    "task-winogrande_rc_train_0shot_finetune${postfix}/step4431"
    "task-winogrande_rc_train_0shot_finetune${postfix}/step5908"
    "task-winogrande_rc_train_0shot_finetune${postfix}/step7385"
    "task-winogrande_rc_train_0shot_finetune${postfix}/step7386"

)

# Do a two-level loop to get all combinations of parent model and finetune task
for PARENT_MODEL in "${PARENT_MODELS[@]}"; do
    for FINETUNE_TASK in "${FINETUNE_TASKS[@]}"; do
        # check if the path exists
        if [ -d "${PARENT_MODEL}/${FINETUNE_TASK}" ]; then
            echo "#### Directory exists: ${PARENT_MODEL}/${FINETUNE_TASK}"
            rm -rf "${PARENT_MODEL}/${FINETUNE_TASK}"
            echo "Removed directory: ${PARENT_MODEL}/${FINETUNE_TASK}"
        else
            echo "#### Directory does not exist: ${PARENT_MODEL}/${FINETUNE_TASK}"
        fi
    done
done