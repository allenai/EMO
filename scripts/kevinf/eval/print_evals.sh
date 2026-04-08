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

# Code evals: base + dolma2 + stack-v2 + fim (all python/java/cpp) + sponge
# rm -f cached_results.pkl
# python src/scripts/eval/print_evals.py \
#     -r \
#     -b /data/input/kevinf/eval_results/flexmoe/ \
#     --show-models "new-kevinf-olmo3-1b-130b-dolma3-0625-150Bsample_step30995-hf,olmo3-1b-dolma2-code-python-10B-lr5e-5-warmup0.1,olmo3-1b-dolma2-code-java-10B-lr5e-5-warmup0.1,olmo3-1b-dolma2-code-cpp-10B-lr5e-5-warmup0.1,train-olmo3-1b-stack-v2-python-p75,train-olmo3-1b-stack-v2-cpp-p75,train-olmo3-1b-stack-v2-java-p75,olmo3-1b-code_fim_python-2B,olmo3-1b-code_fim_cpp-2B,olmo3-1b-code_fim_java-2B,train-olmo3-1b-sponge-code-prose-p75" \
#     --hide-models "lr5e-6,lr5e-7,chembenchfinal,chembench-fixed,chembench-test,frenchbenchfinal,legalbenchfinal,him,newseed,parse-float,updated,eval-on,train-olmo3-1b-dolma2" \
#     --show-tasks "mt_mbpp" \
#     --hide-tasks "mmlu,codex" \
#     --nicknames "new-kevinf-olmo3-1b-130b-dolma3-0625-150Bsample:base_150B,dolma2-code-python-10B-lr5e-5-warmup:d2_py,dolma2-code-java-10B-lr5e-5-warmup:d2_java,dolma2-code-cpp-10B-lr5e-5-warmup:d2_cpp,stack-v2-python-p75:sv2_py,stack-v2-cpp-p75:sv2_cpp,stack-v2-java-p75:sv2_java,code_fim_python:fim_py,code_fim_cpp:fim_cpp,code_fim_java:fim_java,sponge-code-prose-p75:sponge_cp75" \
#     --lower-is-better

# python src/scripts/eval/print_evals.py \
#     -r \
#     -b /data/input/kevinf/fix_old_flex/7x7B_1000_final_model/FlexOlmo-7x7B-1T-RT_step10900-hf/ \
#     --avg-mmlu \
#     --avg-mmlu-pro \ 

python src/scripts/eval/print_evals.py \
    -r \
    -b /data/input/kevinf/eval_results/flexmoe/ \
    --show-models "olmo3_1b_5xc_50web_alldressed_v2_50spring2code_stack_edu_redux_all_step61007-hf,new-kevinf-olmo3-1b-130b-dolma3-0625-150Bsample_step30995-hf,train-olmo3-1b-dolma50-stackedu-python50-10B-lr5e-5-ctd" \
    --hide-models "lr5e-6,lr5e-7,chembenchfinal,chembench-fixed,chembench-test,frenchbenchfinal,legalbenchfinal,him,newseed,parse-float,updated,eval-on,train-olmo3-1b-dolma2" \
    --show-tasks "mt_mbpp_v2fix" \
    --hide-tasks "" \
    --nicknames "" \
    --lower-is-better
