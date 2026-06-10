# models_sizescaling

Size-scaling experiment: how does expert specialization change as the total
expert count grows, holding active parameters fixed?

Four EMO models, identical except for `num_experts` (top-k = 8 active either
way, 1 of the 8 shared), trained on the same 130B-token OLMoE-mix-0824 recipe:

| Run | Experts | Total params |
|---|---|---|
| `emo_1b4b_130b` | 32 | ~4B |
| `emo_1b7b_130b` | 64 | ~7B |
| `emo_1b11b_130b` | 96 | ~11B |
| `emo_1b14b_130b` | 128 | ~14B |

Checkpoints live at `models_sizescaling/<run>/step30995/` (OLMo-core) and
`step30995-hf/` (HF, float32, loadable with `trust_remote_code=True`).

Note: the four models do **not** share expert initialization — a single
sequential RNG stream initializes the whole model, and the per-block router
weight (whose size depends on `num_experts`) is drawn before the expert
weights, so the streams diverge at block 0. Cross-model expert correspondence
must be discovered functionally (see `analysis_4`), never by index.

## Scripts

### Pretraining + conversion

- `emo_1b{4,7,11,14}b_130b.sh` — pretraining launches (see top-level CLAUDE.md
  for `MODE=local|beaker` usage).
- `convert_to_hf.sh` — OLMo-core → HF conversion sweep (idempotent).

### Specialization analysis

The `analysis_N_` prefix encodes dependency order: analysis 1 produces the
embeddings that 2–4 consume; 4 also reads 3's profiles. Each wrapper pins the
four models and calls a generic tool (from `scripts/clustering/` or
`src/scripts/clustering/`); all model-agnostic logic lives in those tools.

1. `analysis_1_weborganizer_extract.sh` — run each model over the same ~20M
   tokens / ~29k docs of weborganizer-labeled data (24 topics, shuffle seed
   42); save per-doc per-layer per-expert usage (`embeddings_doc_probs.npy`,
   `embeddings_doc_topk_freq.npy`) + the stock per-model heatmaps.
   GPU, ~40 min/model. Everything downstream is CPU-cheap replays of these
   files.
2. `analysis_2_trends.sh` — cross-model trend curves: per-topic effective
   expert count / coverage vs total experts (constant number vs constant
   fraction?). Wraps `src/scripts/clustering/plot_expert_usage_trends.py`.
3. `analysis_3_expert_profiles.sh` — per-expert 24-dim topic profiles and
   specialization-score (normalized topic-entropy) distributions per model.
   Caches `expert_profiles_<emb>.npz` into each model's extraction dir. Wraps
   `src/scripts/clustering/expert_topic_profiles.py`.
4. `analysis_4_match_experts.sh` — cross-model expert matching on per-doc
   usage fingerprints (valid because all models saw identical docs):
   similarity matrices, Hungarian matching, splitting / redundancy / novelty
   statistics for consecutive pairs + 32↔128. Wraps
   `src/scripts/clustering/match_experts.py`.
5. `build_report.py` — assembles the figures + summary JSONs from analyses
   1–4 into a single self-contained tabbed HTML report (one tab per analysis:
   goal / method / results, all images base64-embedded):
   `python scripts/models_sizescaling/build_report.py` →
   `claude_outputs/models_sizescaling/report.html`.
6. `publish_report.sh` — rebuilds the report and force-pushes it to a secret
   GitHub gist, served rendered at a stable gist.githack.com URL. The gist ID
   is read from the untracked `claude_outputs/.report_gist_id` (this repo is
   public, so the unlisted URL must not be committed).

## Output layout

Outputs live under `claude_outputs/models_sizescaling/`, matching the
experiment name:

```
claude_outputs/models_sizescaling/weborganizer/<run>/ # analysis 1 (+ profiles from 3)
claude_outputs/models_sizescaling/trends/             # analysis 2
claude_outputs/models_sizescaling/profiles/           # analysis 3 plots
claude_outputs/models_sizescaling/matching/           # analysis 4
claude_outputs/models_sizescaling/report.html         # build_report.py
```

`.npy`/`.npz` files are excluded from the S3 push (regenerable); plots, JSONs,
and logs sync via `scripts/push_claude_outputs.sh`.
