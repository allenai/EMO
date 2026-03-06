# python src/scripts/eval/print_evals.py \
#     -r \
#     -t \
#     -b /data/input/kevinf/eval_results/flexmoe/ \
#     --show-models "new-kevinf-olmo3-1b-130b-dolma3-0625-150Bsample_step30995-hf,olmo3-1b-the-pile-of-law-10B-lr5e-5-warmup0.1-ctd_step2385-hf" \
#     --hide-models "parse-float,updated" \
#     --show-tasks "legalbench" \
#     --hide-tasks "agi_eval,bbh" \
#     --avg-all \

# python src/scripts/eval/print_evals.py \
#     -r \
#     -t \
#     -b /data/input/kevinf/eval_results/flexmoe/ \
#     --show-models "chembenchfinal" \
#     --show-tasks "chembench" \
#     --hide-tasks "mc" \
#     --avg-all \
#     # --hide-models "new-kevinf-olmo3-1b-130b-dolma3-0625-150Bsample_step30995-hf-him-2,olmo3-1b-croissant-10B-lr5e-5-warmup0.1-ctd_step2385-hf-him-2" \

# python src/scripts/eval/print_evals.py \
#     -r \
#     -b /data/input/kevinf/eval_results/flexmoe/ \
#     --show-models "mimic-iv-note" \
#     --show-tasks "med,bio" \
#     --hide-tasks "math"

# python src/scripts/eval/print_evals.py \
#     -r \
#     -b /data/input/kevinf/eval_results/flexmoe/ \
#     --show-models "new-kevinf-olmo3-1b-130b-dolma3-0625-150Bsample_step30995-hf,dolma2-code" \
#     --show-tasks "mbpp,humaneval" \
#     --hide-tasks "mt_" \
#     --nicknames "new-kevinf:base_150B,fim_cpp:fim_cpp_2B,fim_java:fim_java_2B,fim_python:fim_python_2B,dolma2-code-50B:d2_50B,dolma2-code-cpp:d2_cpp_10B,dolma2-code-java:d2_java_10B,dolma2-code-python:d2_python_10B"
#     # --lower-is-better \

python src/scripts/eval/print_evals.py \
    -r \
    -b /data/input/kevinf/fix_old_flex/7x7B_1000_final_model/FlexOlmo-7x7B-1T-RT_step10900-hf/ \
    --avg-mmlu \
    --avg-mmlu-pro \ 