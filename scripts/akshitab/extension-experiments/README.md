
# Summary of extension experiments

The goal is to experiment with expert modularity in MoEs, to enable cheaper continual learning.


Extension: Add new fine-grained experts to existing MoE specializing in new domains (eg. math, code, medical/biology, french, etc.)
Selective training: Train selected experts in existing MoEs, and train them further.


Extension and merging: Merge newly added experts in multiple domains, lightly train the router.

## Model paths and what they mean

Regular MoE base model - moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_0308/step30995-hf

Twolevel MoE base model - twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0301/step30995-hf

Selective training math regular moe - moereducedp512sharedexp1_1b14b_128experts_69_30_3_6_trained_math_10B_lr_4e-4/step2385-hf

Selective training code regular moe - moereducedp512sharedexp1_1b14b_128experts_123_1_76_6_trained_code_10B_lr_4e-4/step2385-hf

selective training code twolevel moe - twolevel_1b14b_128experts_63_26_6_19_trained_code_10B_lr_4e-4/step2385-hf

selective training math twolevel moe - twolevel_1b14b_128experts_99_0_42_88_trained_math_10B_lr_4e-4/step2385-hf

extension math twolevel moe - twolevel_132experts_4trained_forced_math_init_top2_average_noise_10B_lr_4e-4/step2385-hf

extension code twolevel moe - twolevel_132experts_4trained_forced_code_mix_init_top2_average_noise_10B_lr_4e-4/step2385-hf

extension math regular moe - moereducedp512sharedexp1_132experts_4trained_math_init_top2_average_10B_lr_4e-4/step2385-hf

extension code regular moe - moereducedp512sharedexp1_132experts_4trained_code_mix_init_top2_average_noise_10B_lr_4e-4/step2385-hf

## Evals that we care about

General evals: core9:rc, squad:5shot, triviaqa:5shot_rc.wikipedia.nocontext
Math evals: basic_skills:5shot_arithmetic, gsm8k:8shot_main
Code evals: humaneval:3shot:bpb, mbpp:3shot:bpb

Example run for getting evals:

python scripts/akshitab/extension-experiments/print_evals.py --base-dir s3://ai2-sewonm/akshitab/mose/evals/extensions --show-models twolevel_1b14b_128experts_63_26_6_19_trained_code_10B_lr_4e-4,moereducedp512sharedexp1_1b14b_128experts_123_1_76_6_trained_code_10B_lr_4e-4,moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_0308,twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0301  --hide-tasks basic_skills:mc,algebra,french,mmlu,generation,minerva --reset-cache --avg-core-rc