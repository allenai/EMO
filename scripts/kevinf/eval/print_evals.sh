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
    --show-models "him-2" \
    --hide-models "olmo3-1b-chempile-10B-lr2e-4-warmup715-ctd_step2385-hf-him-2,olmo3-1b-chempile-10B-lr5e-5-warmup715-ctd_step2385-hf-him-2" \
    --show-tasks "chembench" \
    --hide-tasks "" \
    --avg-all \
