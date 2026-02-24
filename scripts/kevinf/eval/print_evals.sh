# python src/scripts/eval/print_evals.py \
#     -r \
#     -t \
#     -b /data/input/kevinf/flexmoe/eval/results/ \
#     --show-models "new-kevinf-olmo3-1b-130b-dolma3-0625-150Bsample_step30995-hf,olmo3-1b-the-pile-of-law-10B-lr5e-5-warmup0.1-ctd_step2385-hf" \
#     --hide-models "parse-float,updated" \
#     --show-tasks "legalbench" \
#     --hide-tasks "agi_eval,bbh" \
#     --avg-all \

python src/scripts/eval/print_evals.py \
    -r \
    -t \
    -b /data/input/kevinf/flexmoe/eval/results/ \
    --show-models "new-kevinf-olmo3-1b-130b-dolma3-0625-150Bsample_step30995-hf,olmo3-1b-code_fim_cpp-2B-lr5e-5-warmup0.1-pplx-raw-ctd_step477-hf,olmo3-1b-code_fim_java-2B-lr5e-5-warmup0.1-pplx-raw-ctd_step477-hf,olmo3-1b-code_fim_python-2B-lr5e-5-warmup0.1-pplx-raw-ctd_step477-hf" \
    --show-tasks "human,mbpp" \
    --hide-tasks "" \
    --avg-all \
    # --hide-models "new-kevinf-olmo3-1b-130b-dolma3-0625-150Bsample_step30995-hf-him-2,olmo3-1b-croissant-10B-lr5e-5-warmup0.1-ctd_step2385-hf-him-2" \
