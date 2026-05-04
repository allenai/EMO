# HF `trust_remote_code` scripts

Minimal `configuration_*.py` / `modeling_*.py` files for loading Emo-flavoured
checkpoints with stock `transformers` (no custom fork required) via
`AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`.

Two architectures are provided:

| Architecture                          | Folder                          | Used by                      |
|---------------------------------------|---------------------------------|------------------------------|
| `EmoNoQKNormPrenormForCausalLM`  | `emo_noqknorm_prenorm/`   | All MoE checkpoints          |
| `Olmo2NoQKNormPrenormForCausalLM`     | `olmo2_noqknorm_prenorm/`       | `dense_1b_lr-4e-3_0213/...`  |

The files were lifted from the
[`ryanyxw/transformers#flexmoe_v4_57_1`](https://github.com/ryanyxw/transformers/tree/flexmoe_v4_57_1/src/transformers/models)
fork, with relative imports (`from ...X`) rewritten to absolute imports
(`from transformers.X`) so they work outside the fork's source tree.

## Usage

Use `upload_to_hf.py` to detect the architecture, copy the matching remote-code
files into the checkpoint, patch `config.json` with the right `auto_map`, and
push the folder to the HF Hub:

```bash
python scripts/akshitab/hf_trust_remote_code/upload_to_hf.py \
    --model-path /path/to/<ckpt-dir> \
    --repo-id    allenai/<repo-name>
```

Useful flags:
- `--public` — create the repo as public. Default is **private**.
- `--no-upload` — only stage the `.py` files and patch `config.json` locally.
- `--token <hf_token>` — override the cached login / `HF_TOKEN` env var.

After the push, load with:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("allenai/<repo-name>", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("allenai/<repo-name>")
```

### Manual equivalent

If you need to do it by hand: copy the matching `configuration_*.py` /
`modeling_*.py` next to `config.json`, then add an `auto_map` block.

For `EmoNoQKNormPrenormForCausalLM`:
```json
"auto_map": {
  "AutoConfig": "configuration_emo_noqknorm_prenorm.EmoNoQKNormPrenormConfig",
  "AutoModelForCausalLM": "modeling_emo_noqknorm_prenorm.EmoNoQKNormPrenormForCausalLM"
}
```

For `Olmo2NoQKNormPrenormForCausalLM`:
```json
"auto_map": {
  "AutoConfig": "configuration_olmo2_noqknorm_prenorm.Olmo2NoQKNormPrenormConfig",
  "AutoModelForCausalLM": "modeling_olmo2_noqknorm_prenorm.Olmo2NoQKNormPrenormForCausalLM"
}
```

## Checkpoint → architecture mapping

| Checkpoint                                                                                                    | Architecture                          |
|---------------------------------------------------------------------------------------------------------------|---------------------------------------|
| `twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_1T_0313_anneal_from_step238419/step250339-hf` | `EmoNoQKNormPrenormForCausalLM` |
| `twolevelbatchlbreducedp512sharedexp1randpool-8-128eval32_1b14b_lr-4e-3_lb-1e-1_0301/step30995-hf`            | `EmoNoQKNormPrenormForCausalLM` |
| `dense_1b_lr-4e-3_0213/step30995-hf`                                                                          | `Olmo2NoQKNormPrenormForCausalLM`    |
| `moereducedp512sharedexp1_1b4b_lr-4e-3_lb-1e-1_0308/step30995-hf`                                             | `EmoNoQKNormPrenormForCausalLM` |
| `moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_0308/step30995-hf`                                            | `EmoNoQKNormPrenormForCausalLM` |
| `moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_1T_0322_anneal_from_step238419/step250339-hf`                 | `EmoNoQKNormPrenormForCausalLM` |
| `moereducedp512sharedexp1_1b14b_lr-4e-3_lb-1e-1_1T_0322_anneal_twolevel_randpool-8-128_from_step238419/step250339-hf` | `EmoNoQKNormPrenormForCausalLM` |

## Requirements

A `transformers` install that exposes the symbols imported at the top of the
modeling files (notably `transformers.utils.generic.check_model_inputs`,
`OutputRecorder`, `transformers.masking_utils.create_causal_mask`,
`transformers.integrations.use_kernel_forward_from_hub`). These are present in
`transformers >= 4.55` and the `4.57.x` family used by this project.
