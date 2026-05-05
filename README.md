<div align="center">
  <img src="assets/Emo_Logo.png" alt="Emo Logo" style="width: 300px; margin-left:'auto' margin-right:'auto' display:'block'"/>
  <br>
  <br>
</div>

<p align="center">
  <a href="LICENSE">
    <img alt="GitHub License" src="https://img.shields.io/badge/license-TODO-lightgrey">
  </a>
  <a href="#">
    <img alt="Blog Post" src="https://img.shields.io/badge/Emo-blog-F0529C">
  </a>
  <a href="#">
    <img alt="Paper URL" src="https://img.shields.io/badge/Emo-arxiv-blue">
  </a>
  <a href="https://huggingface.co/collections/allenai/modmoe">
    <img alt="Model Checkpoints" src="https://img.shields.io/badge/%F0%9F%A4%97%20HF-Models-yellow">
  </a>
</p>

EMO is a new Mixture-of-Experts model trained so that modular structure emerges during pretraining without requiring human-defined priors. EMO enables selective expert use, down to 12.5% of total experts, with minimal performance degradation. We find that its expert groups specialize to higher-level topics and capabilities rather than low-level lexical patterns.

## Installation

First install [PyTorch](https://pytorch.org) according to the instructions specific to your operating system and hardware.

```bash
git clone https://github.com/allenai/EMO.git
cd EMO
conda create -n emo python==3.12
uv pip install -e .[all]
```

## Released Models

All checkpoints are available in the [ModMoE collection](https://huggingface.co/collections/allenai/modmoe) on the Hugging Face Hub.

### Main Release (1T tokens)

| Model | Active | Total | Pretraining (1T) | Annealing (50B) | Description |
|---|---|---|---|---|---|
| [`allenai/Emo_1b14b_1T`](https://huggingface.co/allenai/Emo_1b14b_1T) | 1B | 14B | EMO | EMO | Main EMO release |
| [`allenai/StdMoE_1b14b_1T`](https://huggingface.co/allenai/StdMoE_1b14b_1T) | 1B | 14B | standard | standard | Architecture-matched standard MoE baseline |

### Ablation Models (130B tokens)

Smaller-scale checkpoints used for memory-matched comparisons. These models were not midtrained. 

| Model | Active | Total | Pretraining (130B) | Description |
|---|---|---|---|---|
| [`allenai/Emo_1b14b_130B`](https://huggingface.co/allenai/Emo_1b14b_130B) | 1B | 14B | EMO | EMO at the 130B-token ablation scale |
| [`allenai/StdMoE_1b14b_130B`](https://huggingface.co/allenai/StdMoE_1b14b_130B) | 1B | 14B | standard | Standard MoE baseline at the 130B-token scale |
| [`allenai/StdMoE_1b4b_130B`](https://huggingface.co/allenai/StdMoE_1b4b_130B) | 1B | 4B | standard | Memory-matched standard MoE with 32 experts ("Reg. MoE @ 32" in Figure 4), used as a memory-matched baseline for EMO's 32-expert subsets |
| [`allenai/Dense_1b_130B`](https://huggingface.co/allenai/Dense_1b_130B) | 1B | 1B | dense LM | Dense baseline matched to active parameters ("Dense @ 8" in Figure 4), used as a memory-matched baseline for EMO's 8-expert subsets |

### Midtraining Ablation Models

Checkpoints used in Appendix B.3 to test whether modularity can be induced after pretraining via annealing alone, rather than during pretraining.

| Model | Active | Total | Pretraining (1T) | Annealing (50B) | Description                                                                                                             |
|---|---|---|---|---|-------------------------------------------------------------------------------------------------------------------------|
| [`allenai/StdMoE_1b14b_1T_Preanneal`](https://huggingface.co/allenai/StdMoE_1b14b_1T_Preanneal) | 1B | 14B | standard | — | Standard MoE checkpoint after 1T-token pretraining, before any annealing. Starting point for the EMO-anneal experiment |
| [`allenai/StdMoE_1b14b_1T_EmoAnnealed`](https://huggingface.co/allenai/StdMoE_1b14b_1T_EmoAnnealed) | 1B | 14B | standard | EMO | EMO-anneal: a standard MoE annealed under the document-level expert pool constraint for 50B tokens |

## Training scripts

Project-specific pretraining, continual-training, eval, and pruning recipes live in [`scripts/`](scripts/). Run scripts source [`scripts/launch_common.sh`](scripts/launch_common.sh), which exports shared paths and a `launch()` helper that dispatches to either `torchrun` (default, `MODE=local`) or `python -m olmo_core.launch.beaker` (`MODE=beaker`) if you have Beaker available.

Run a script locally:

```bash
bash scripts/models_0116/dense_1b_lr-4e-3_0213.sh
```

Submit it as a Beaker job:

```bash
MODE=beaker bash scripts/models_0116/dense_1b_lr-4e-3_0213.sh
```

Override paths via env vars before launching:

- `PREFIX` — output root
- `MODELS_DIR` — derived from `PREFIX` (`${PREFIX}/models`)
- `DATASET_CACHE` — tokenizer-mapped dataset cache

Override Beaker cluster sizing per script with `BEAKER_GPUS=8 BEAKER_NODES=4 ...`.

### Run templates

[`scripts/RUN_TEMPLATES.md`](scripts/RUN_TEMPLATES.md) has the full code for each recipe. Summary:

| Template | Description | Entry point |
|---|---|---|
| Dense 1B pretrain | Dense 1B on `OLMoE-mix-0824` | `src/scripts/train/olmo2-1B.py` |
| MoE 1B/14B single-level | 128-expert single-level MoE | `src/scripts/train/olmoe-1B-7B_fsl.py` |
| MoE 1B/14B two-level (shared experts) | Two-level router with `--document-expert-pool`, `--num_shared_experts*` | `src/scripts/train/olmoe-1B-7B_fsl.py` |
| Continual pretrain / extension | Resume from a checkpoint on a domain mix (e.g. `mj_finemath4plus`) | `src/scripts/train/olmoe-1B-7B_fsl_extension.py` |
| Eval | OLMES eval on HF checkpoints, results uploaded to S3 | via [`scripts/extensions/launch_eval.sh`](scripts/extensions/launch_eval.sh) |
| Pruning + task finetune | Compute router activations → prune to top-k experts → finetune | [`scripts/pruning_hf/hf_finetune_with_pruning.sh`](scripts/pruning_hf/hf_finetune_with_pruning.sh) |

Two-level model variants (`--model-type`):

- `moe` — single-level
- `two-level_lb-batch_reduce-dp` — two-level, no shared experts
- `two-level_lb-batch_reduce-dp_sharedexppool` — two-level with fixed shared-expert pool
- `two-level_lb-batch_reduce-dp_sharedexp_randpool` — two-level with random shared-expert pool sampling (used in anneal runs)

Pruning modes (`PRUNING_MODE`):

- `global` — single-pass activation collection + top-k prune across the whole model
- `layerwise` — greedy layer-by-layer pruning (each layer conditioned on earlier pruned layers)
- `layerwise_variable` — greedy layerwise with a per-layer keep-k schedule
- `easy_ep` — EASY-EP ([arXiv 2504.06792](https://arxiv.org/abs/2504.06792)): domain-specific one-shot prune on calibration data

The `runname` naming convention (size · router · LR · LB · date · phase) is documented in the cheatsheet at the bottom of `scripts/RUN_TEMPLATES.md`.

## Inference

See [Released Models](#released-models) for the available checkpoints. All inference snippets below require `trust_remote_code=True` since the models use custom modeling code from the [ryanyxw/transformers](https://github.com/ryanyxw/transformers/tree/flexmoe_v4_57_1) fork.

### With Hugging Face Transformers

You can use our Hugging Face [transformers](https://github.com/huggingface/transformers) integration to run inference on the released checkpoints:

```bash
pip install transformers>=4.57.0
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = "allenai/Emo_1b14b_1T"
olmo = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
message = ["Language modeling is "]
inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)
# inputs = {k: v.to('cuda') for k,v in inputs.items()} # optional verifying cuda
# olmo = olmo.to('cuda')
response = olmo.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=1.0, top_p=0.7)
print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])
```

Alternatively, with the Hugging Face pipeline abstraction:

```python
from transformers import pipeline
olmo_pipe = pipeline("text-generation", model="allenai/Emo_1b14b_1T", trust_remote_code=True)
print(olmo_pipe("Language modeling is"))
```

### With vLLM

[vLLM](https://docs.vllm.ai/en/latest/) provides high-throughput inference. You can use it for offline batched inference:

```bash
pip install vllm>=0.11.0
```

```python
from vllm import LLM, SamplingParams
llm = LLM(model="allenai/Emo_1b14b_1T", trust_remote_code=True)
sampling_params = SamplingParams(temperature=1.0, top_p=0.7)
prompts = ["Language modeling is"]
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

For more details, see the [vLLM documentation](https://docs.vllm.ai/en/latest/getting_started/quickstart/#offline-batched-inference).

## Citing

TODO
