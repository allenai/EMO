#! /bin/bash

## example use: bash src/scripts/eval/run_eval.sh allenai/Flex-public-7B-1T mc9 eval_results/ 1

MODEL=$1
TASK_NAME=$2
BASE_OUTPUT_DIR=$3
GPUS=$4


if [[ $TASK_NAME == "mc9" ]] ; then
	TASKS=(
		arc_easy:mc::olmes
		arc_challenge:mc::olmes
		boolq:mc::olmes
		csqa:mc::olmes
		hellaswag:mc::olmes
		openbookqa:mc::olmes
		piqa:mc::olmes
		socialiqa:mc::olmes
		winogrande:mc::olmes
	)
elif [[ $TASK_NAME == "gen5" ]] ; then
	TASKS=(
		coqa::olmes
		squad::olmes
		naturalqs::olmes
		triviaqa::olmes
		drop::olmes
	)
elif [[ $TASK_NAME == "mmlu" ]] ; then
	TASKS=(mmlu:mc::olmes)
elif [[ $TASK_NAME == "mmlu_pro" ]] ; then
	TASKS=(mmlu_pro_mc::none)
elif [[ $TASK_NAME == "agi_eval" ]] ; then
	TASKS=(agi_eval_english:1shot::olmes)
elif [[ $TASK_NAME == "bbh" ]] ; then
	TASKS=(bbh:cot-v1::olmes)
elif [[ $TASK_NAME == "math2" ]] ; then
	TASKS=(
		gsm8k::olmes
		minerva_math_algebra::olmes
		minerva_math_counting_and_probability::olmes
		minerva_math_geometry::olmes
		minerva_math_intermediate_algebra::olmes
		minerva_math_number_theory::olmes
		minerva_math_prealgebra::olmes
		minerva_math_precalculus::olmes
	)
elif [[ $TASK_NAME == "code4" ]] ; then
	TASKS=(
		codex_humaneval:temp0.8
		codex_humanevalplus:temp0.8
		mbpp::none
		mbppplus::none
	)
else
	TASKS=($TASK_NAME)
fi

for TASK in "${TASKS[@]}"; do
	# For setting the output_dir
	model=$(echo $MODEL | cut -d'/' -f2)
	# OOM with some tasks, so batch size to be 1
	if [[ $TASK == "minerva_math_"* || $TASK == "mbpp"* || $TASK == "bigcodebench"* || $TASK == "sciriff"* ]] ; then
		batch_size=1
	else
		batch_size=4
	fi

	PYTHONPATH=. python src/scripts/eval/launch_eval.py \
	--model $MODEL \
	--model-type hf \
	--task $TASK \
	--limit 1000 \
	--output-dir ${BASE_OUTPUT_DIR}/${MODEL} \
	--batch-size $batch_size \
	--gpus $GPUS
done
