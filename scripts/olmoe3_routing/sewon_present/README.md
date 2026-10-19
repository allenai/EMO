# Merged four-square sub-models vs joint training (EMO 512e / 1000e / 2000e, 10B -> 61B)

`plot_merge_vs_joint.py` (raw values embedded) -> `merge_vs_joint.png` / `.pdf`. Regenerate with `python plot_merge_vs_joint.py`.

Held-out cross-entropy on the 300B-stream sample (65.5M tokens). "Merged" = the 10B start model split into 4 random sub-models
(a quarter of the experts in layers 2-9, layer 1 whole), each trained at sub-model LR 4e-4 on a quarter of the documents, experts
put side by side and shared parameters averaged. "Baseline" = the same start model continued jointly (LR 8e-4).

| model | point | 10B (start) | 15.7B | 20B | 30B | 61B |
|---|---|---|---|---|---|---|
| EMO 512e | baseline | 2.465 | 2.420 | 2.403 | 2.376 | 2.331 |
| EMO 512e | merged 4 squares | 2.465 | 2.417 | 2.403 | 2.385 | 2.351 |
| EMO 1000e | baseline | 2.444 | 2.398 | 2.379 | 2.350 | 2.313 |
| EMO 1000e | merged 4 squares | 2.444 | 2.391 | 2.373 | 2.351 | 2.315 |
| EMO 2000e | baseline | 2.429 | 2.376 | 2.362 | 2.336 | 2.298 |
| EMO 2000e | merged 4 squares | 2.429 | 2.367 | 2.350 | 2.324 | 2.280 |

Sources: `sparse_experts/olmoe3_routing/runs_heldout300b_{emo,emo1000,emo2000}/baseline_step<N>/none/meta.json` (baselines, start models)
and `runs_heldout300b_{emorand_lr4e-4,emo1000rand4,emo2000rand4}/merged_match<N>/none/meta.json` (merges); full context in the
olmoe3_routing report, Q5 boxes 5A / 5E / 5G (https://emo-reports.pages.dev/olmoe3_routing.html).
