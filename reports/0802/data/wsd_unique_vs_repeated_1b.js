window.STEP2_REPORT_DATA = {
  "updated": "2026-08-02 16:44 PDT · 1B unique retry provenance and live progress refreshed; dense-153M probes active",
  "modelSizes": {
    "1B": {"label": "Dense 1B", "factory": "olmo2_1B_v2", "totalParameters": 1484916736, "nonEmbeddingParameters": 1279395840, "globalBatchTokens": 4194304, "warmupSteps": 24},
    "153M": {"label": "Dense 153M", "factory": "explicit current-architecture dimensions", "totalParameters": 153104896, "nonEmbeddingParameters": 101724672, "globalBatchTokens": 1048576, "warmupSteps": 95}
  },
  "manifest": {
    "status": "verified",
    "sourceTokens": 3684671294617,
    "sourceDocuments": 2949254346,
    "validationTokens": 100000118,
    "validationDocuments": 79690,
    "testTokens": 100002095,
    "testDocuments": 79680,
    "trainTokens": 3684471292404,
    "trainDocuments": 2949094976,
    "repeatedTokens": 1000031666,
    "repeatedDocuments": 801407,
    "uniqueTokens": 5000000058,
    "uniqueDocuments": 4002478,
    "sha256": "2f3984482b51e499b1c854706dcd64e9bdd8cc567ac52d6546d3325a7ef68fe9"
  },
  "statisticsPolicy": {
    "confidenceLevel": 0.95,
    "validationTokens": 100000118,
    "validationDocuments": 79690,
    "lossMethod": "paired document-level bootstrap on the identical DCLM validation documents",
    "accuracyMethod": "paired item-level bootstrap on identical downstream examples",
    "significanceRule": "statistically significant only when the two-sided 95% confidence interval for the paired difference excludes zero",
    "requiredReportFields": ["difference", "95% confidence interval", "statistically significant yes/no"]
  },
  "seedProbePolicy": {
    "status": "authorized",
    "epoch2Paused": true,
    "regimes": ["repeated", "unique"],
    "epoch": 1,
    "lr": "5e-4",
    "wd": "0.033",
    "trainingCorpus": "each probe uses its regime's exact existing training corpus and ordering source",
    "baselines": {"repeated": {"initSeed": 12536, "dataSeed": 0, "wandb": "v8uxyvfb"}, "unique": {"initSeed": 12536, "dataSeed": 0, "wandb": "y5rjtnx5"}},
    "purpose": "Separate exact-seed reproducibility from sensitivity to initialization and data-order seeds before interpreting independent-sample unique-vs-repeated gaps. These diagnostics are excluded from the LR coordinate grid."
  },
  "seedProbeConclusion": {
    "status": "interim",
    "summary": "The completed repeated-data probes do not yet justify a unique-vs-repeated conclusion. Relative to the original repeated 5e-4 baseline, the same-seed rerun matches rounded train CE (4.972), validation CE (5.000), and HellaSwag BPB (1.528), while HellaSwag accuracy differs by -0.24 percentage points and individual downstream tasks differ by as much as +1.97 points. The alternate-seed rerun changes train CE by +0.053, leaves validation CE equal at three-decimal precision, changes HellaSwag accuracy by +0.13 points and BPB by -0.001, but shifts some individual downstream accuracies substantially (BoolQ +7.83 points; Winogrande +3.63 points). This suggests the 100M-token validation CE is much more stable than the undertrained downstream metrics at the 1B-token endpoint. Exact confidence intervals remain pending paired artifacts, and the two unique-data probes are still active, so neither training-seed variance nor a regime effect is resolved."
  },
  "seedProbeRuns": [
    {"name": "same-seed reproducibility", "regime": "repeated", "status": "complete", "epoch": 1, "lr": "5e-4", "wd": "0.033", "initSeed": 12536, "dataSeed": 0, "runSuffix": "_seedprobe_same_seed", "beaker": "01KZ23D3W3H2FCPN2YRMN447MF", "job": "01KZ23D3ZHRY3NAFKC9M0Q4QYD", "wandb": "a9rdjrf9", "revision": "8e32e686", "output": "/weka/oe-training-default/sewonm/icsl/models/dense_1b_step2_0802_repeated_dclm1b_wsd_e1_lr5e-4_wd0.033_warmup24_seedprobe_same_seed", "train": 4.972, "c4": 5.000, "acc": 24.78, "bpb": 1.528, "downstream": {"arc_challenge": 24.08, "arc_easy": 29.47, "boolq": 38.01, "csqa": 24.16, "openbookqa": 23.00, "piqa": 52.07, "socialiqa": 38.33, "winogrande": 49.80}, "reason": "Exact same corpus and explicit original initialization/data-order seeds. Completed with full DCLM validation and nine tasks; fixed step214 preserved; eight H100; minRuntime 0; autoResume false; no experiment retries. Measures reproducibility and residual runtime nondeterminism; excluded from LR selection grid."},
    {"name": "alternate-seed sensitivity", "regime": "repeated", "status": "complete", "epoch": 1, "lr": "5e-4", "wd": "0.033", "initSeed": 12537, "dataSeed": 1, "runSuffix": "_seedprobe_alt_seed", "beaker": "01KZ23ED5W6NNE11CYSPKHH1YM", "job": "01KZ23ED9CDN9ACP1XV66JG1VK", "wandb": "9xljzi4u", "revision": "8e32e686", "output": "/weka/oe-training-default/sewonm/icsl/models/dense_1b_step2_0802_repeated_dclm1b_wsd_e1_lr5e-4_wd0.033_warmup24_seedprobe_alt_seed", "train": 5.025, "c4": 5.000, "acc": 25.15, "bpb": 1.527, "downstream": {"arc_challenge": 22.41, "arc_easy": 30.70, "boolq": 45.93, "csqa": 24.00, "openbookqa": 23.00, "piqa": 52.61, "socialiqa": 39.46, "winogrande": 51.46}, "reason": "Exact same corpus with alternate initialization and data-order seeds. Completed with full DCLM validation and nine tasks; fixed step214 preserved; eight H100; minRuntime 0; autoResume false; no experiment retries. Measures seed sensitivity; excluded from LR selection grid."},
    {"name": "same-seed reproducibility", "regime": "unique", "status": "active", "epoch": 1, "lr": "5e-4", "wd": "0.033", "initSeed": 12536, "dataSeed": 0, "runSuffix": "_seedprobe_same_seed", "beaker": "01KZ2CGJC01Q9869VPKD1Y6SWA", "job": "01KZ2CGJJZ4B4SP1H236VJB1SH", "wandb": "5h4gsedi", "revision": "0f48f982", "step": 115, "totalSteps": 239, "eta": "21m at 16:44 PDT", "output": "/weka/oe-training-default/sewonm/icsl/models/dense_1b_step2_0802_unique_dclm5b_wsd_e1_lr5e-4_wd0.033_warmup24_seedprobe_same_seed", "reason": "Exact unique corpus and original initialization/data-order seeds; measures reproducibility and residual runtime nondeterminism. Eight H100; minRuntime 0; autoResume false; no experiment retries; fixed step214 configured. Excluded from LR selection grid."},
    {"name": "alternate-seed sensitivity", "regime": "unique", "status": "active", "epoch": 1, "lr": "5e-4", "wd": "0.033", "initSeed": 12537, "dataSeed": 1, "runSuffix": "_seedprobe_alt_seed", "beaker": "01KZ2CM6HH6R2PPTQ9ZWBHH6XW", "job": "01KZ2CM6N4Q9JY8WWFBV5B9090", "wandb": "es3nxv54", "revision": "0f48f982", "step": 95, "totalSteps": 239, "eta": "24m at 16:44 PDT", "output": "/weka/oe-training-default/sewonm/icsl/models/dense_1b_step2_0802_unique_dclm5b_wsd_e1_lr5e-4_wd0.033_warmup24_seedprobe_alt_seed", "reason": "Exact unique corpus with alternate initialization/data-order seeds; measures seed sensitivity. Eight H100; minRuntime 0; autoResume false; no experiment retries; fixed step214 configured. Excluded from LR selection grid."}
  ],
  "smallUniqueRuns": [
    {"epoch": 1, "lr": "5e-4", "wd": "0.033", "status": "planned", "replicateCount": 3, "reason": "Matched epoch-1 aggregate of two identical-seed replicas and one alternate-seed replica on the exact repeated 1B corpus; excluded until all three runs complete."}
  ],
  "smallRepeatedRuns": [
    {"epoch": 1, "lr": "5e-4", "wd": "0.033", "status": "planned", "replicateCount": 3, "reason": "Matched epoch-1 aggregate of two identical-seed replicas and one alternate-seed replica on the exact repeated 1B corpus; excluded until all three runs complete."}
  ],
  "smallSeedProbeRuns": [
    {"name": "same-seed replicate A", "regime": "unique", "status": "active", "epoch": 1, "lr": "5e-4", "wd": "0.033", "initSeed": 12536, "dataSeed": 0, "runSuffix": "_seed0_a", "beaker": "01KZ2CWDFDA03V8HK2T243VDN5", "job": "01KZ2CWDK9JN3757S7YY6BT2CD", "wandb": "81uvwkin", "revision": "5e77c77a", "step": 255, "totalSteps": 954, "eta": "5m at 16:33 PDT", "output": "/weka/oe-training-default/sewonm/icsl/models/dense_153m_step2_0802_unique_wsd_e1_lr5e-4_wd0.033_warmup95_seed0_a", "reason": "Matched epoch-1 unique label on the exact repeated 1B corpus; explicit Choice-A architecture; eight H100; minRuntime 0; autoResume false; no experiment retries; fixed step858 configured."},
    {"name": "same-seed replicate B", "regime": "unique", "status": "active", "epoch": 1, "lr": "5e-4", "wd": "0.033", "initSeed": 12536, "dataSeed": 0, "runSuffix": "_seed0_b", "beaker": "01KZ2CXJ9FP30AFAQ6JYQBPQZ2", "job": "01KZ2CXJCZW5FAWNCE4SZYCHKP", "wandb": "zgogtxye", "revision": "5e77c77a", "step": 70, "totalSteps": 954, "eta": "6m at 16:33 PDT", "output": "/weka/oe-training-default/sewonm/icsl/models/dense_153m_step2_0802_unique_wsd_e1_lr5e-4_wd0.033_warmup95_seed0_b", "reason": "Matched epoch-1 unique label on the exact repeated 1B corpus; explicit Choice-A architecture; eight H100; minRuntime 0; autoResume false; no experiment retries; fixed step858 configured."},
    {"name": "alternate-seed replicate", "regime": "unique", "status": "active", "epoch": 1, "lr": "5e-4", "wd": "0.033", "initSeed": 12537, "dataSeed": 1, "runSuffix": "_seed1", "beaker": "01KZ2CYRRDTDRVX14SFX4DXBTS", "job": "01KZ2CYRW4DSDR4JB2NC54Z2DT", "wandb": "736ej39o", "revision": "5e77c77a", "step": 235, "totalSteps": 954, "eta": "5m at 16:33 PDT", "output": "/weka/oe-training-default/sewonm/icsl/models/dense_153m_step2_0802_unique_wsd_e1_lr5e-4_wd0.033_warmup95_seed1", "reason": "Matched epoch-1 unique label on the exact repeated 1B corpus with alternate initialization/data-order seeds; explicit Choice-A architecture; eight H100; minRuntime 0; autoResume false; no experiment retries; fixed step858 configured."},
    {"name": "same-seed replicate A", "regime": "repeated", "status": "active", "epoch": 1, "lr": "5e-4", "wd": "0.033", "initSeed": 12536, "dataSeed": 0, "runSuffix": "_seed0_a", "beaker": "01KZ2D00RYSREQZKJVW2MKPGMK", "job": "01KZ2D00WQEB9A2X1S63S0XYAR", "wandb": "wzagl12x", "revision": "5e77c77a", "output": "/weka/oe-training-default/sewonm/icsl/models/dense_153m_step2_0802_repeated_wsd_e1_lr5e-4_wd0.033_warmup95_seed0_a", "reason": "Matched epoch-1 repeated label on the exact repeated 1B corpus; explicit Choice-A architecture; eight H100; minRuntime 0; autoResume false; no experiment retries; fixed step858 configured."},
    {"name": "same-seed replicate B", "regime": "repeated", "status": "queued", "epoch": 1, "lr": "5e-4", "wd": "0.033", "initSeed": 12536, "dataSeed": 0, "runSuffix": "_seed0_b", "beaker": "01KZ2D1HKK9CQRK5K2F4MZG1PJ", "job": "01KZ2D1HW6DZCE2AZABCY0Y3QB", "revision": "5e77c77a", "output": "/weka/oe-training-default/sewonm/icsl/models/dense_153m_step2_0802_repeated_wsd_e1_lr5e-4_wd0.033_warmup95_seed0_b", "reason": "Matched epoch-1 repeated label on the exact repeated 1B corpus; explicit Choice-A architecture; eight H100; minRuntime 0; autoResume false; no experiment retries; fixed step858 configured."},
    {"name": "alternate-seed replicate", "regime": "repeated", "status": "queued", "epoch": 1, "lr": "5e-4", "wd": "0.033", "initSeed": 12537, "dataSeed": 1, "runSuffix": "_seed1", "beaker": "01KZ2D2VYD9QX4P3ZJVWZ6BCFE", "job": "01KZ2D2W26VKKS038N5SNRGFV4", "revision": "5e77c77a", "output": "/weka/oe-training-default/sewonm/icsl/models/dense_153m_step2_0802_repeated_wsd_e1_lr5e-4_wd0.033_warmup95_seed1", "reason": "Matched epoch-1 repeated label on the exact repeated 1B corpus with alternate initialization/data-order seeds; explicit Choice-A architecture; eight H100; minRuntime 0; autoResume false; no experiment retries; fixed step858 configured."}
  ],
  "smallEvaluationRuns": [],
  "dataGeneration": [
    {
      "beaker": "01KZ0PV76Y6JDEFWPHJXMFRJ5S",
      "resultDataset": "01KZ0PV72R27GKATHXP8SNTZ74",
      "revision": "0834a592",
      "status": "canceled",
      "reason": "Confirmed startup failure: full-history Git clone stalled for more than 14 minutes before any Weka output was created"
    },
    {
      "beaker": "01KZ0QPSJYSDP2T9BASQXFNC53",
      "resultDataset": "01KZ0QPSFVX4V4VVHH42K07GPH",
      "revision": "0834a592",
      "status": "canceled",
      "reason": "Confirmed startup failure: shallow Git clone also stalled before checkout or Weka output, isolating the problem to GitHub access from the image"
    },
    {
      "beaker": "01KZ0QXYG49R34RXMR1M79EWVE",
      "resultDataset": "01KZ0QXYCQX1VTYJH6W5RNCQNV",
      "sourceDataset": "01KZ0QXCDPRAZNDBEK437CXJ7D",
      "revision": "0834a592",
      "status": "preempted",
      "reason": "System-preempted during I/O-bound materialization by urgent job 01KZ0ZQ3GSH726K4J01MDNHHZV; exited 143 before validation and partial Weka paths were retained untouched"
    },
    {
      "beaker": "01KZ10SYHTRDAAE7QQ0HZYDX6J",
      "resultDataset": "01KZ10SYBBGJ72R0CX7W5RPX0E",
      "sourceDataset": "01KZ0QXCDPRAZNDBEK437CXJ7D",
      "revision": "0834a592",
      "status": "canceled",
      "reason": "Diagnosed non-completable implementation: after 5.5 hours the single-threaded writer had completed validation/test but only 1.85/4.0 GB of the 1B repeated artifact; stopped before the eight-hour timeout, partial paths retained"
    },
    {
      "beaker": "01KZ1KTFZ6KGG8Z44VF4DPRYT8",
      "resultDataset": "01KZ1KTFRQD1VWAVYDCMS4CQSQ",
      "sourceDataset": "01KZ1KSH835PMYNB0S6Z00Y59A",
      "revision": "0ee8c2d9",
      "status": "succeeded",
      "reason": "Generation and independent validation passed. Across 2,949,254,346 source documents and 3,684,671,294,617 source tokens, train/validation/test are exhaustive with zero document intersections; both training samples are training-only; all artifact and metadata hashes verified; and the 4,096-bucket key-uniformity sanity check passed."
    }
  ],
  "uniqueRuns": [
    {"epoch": 1, "lr": "5e-4", "wd": "0.033", "status": "complete", "beaker": "01KZ1XN29PN219QMSECNF5WGH4", "job": "01KZ1XN2DK2PTEJAQW5JGMF64M", "wandb": "y5rjtnx5", "revision": "a55cb850", "output": "/weka/oe-training-default/sewonm/icsl/models/dense_1b_step2_0802_unique_dclm5b_wsd_e1_lr5e-4_wd0.033_warmup24", "train": 5.013, "c4": 4.996, "acc": 25.03, "bpb": 1.534, "downstream": {"arc_challenge": 23.75, "arc_easy": 28.42, "boolq": 38.07, "csqa": 23.91, "openbookqa": 24.40, "piqa": 52.29, "socialiqa": 38.69, "winogrande": 48.38}, "reason": "Replacement lower-rung run completed successfully with DCLM validation and all nine downstream evaluations; fixed pre-decay step214 saved successfully; train CE is nearest final logged value; minRuntime 0; autoResume false; no experiment retries. LR comparison significance pending paired document statistics."},
    {"epoch": 1, "lr": "1e-3", "wd": "0.033", "status": "complete", "beaker": "01KZ1WQNDDBA1P8YTSRXG16A01", "job": "01KZ1WQNGWPA1YV4H71NXYCVH6", "wandb": "h269qykw", "revision": "04f7f19f", "output": "/weka/oe-training-default/sewonm/icsl/models/dense_1b_step2_0802_unique_dclm5b_wsd_e1_lr1e-3_wd0.033_warmup24_retry1", "train": 5.509, "c4": 5.490, "acc": 24.47, "bpb": 1.682, "downstream": {"arc_challenge": 22.41, "arc_easy": 27.72, "boolq": 37.95, "csqa": 23.83, "openbookqa": 25.80, "piqa": 52.83, "socialiqa": 39.97, "winogrande": 52.01}, "reason": "Confirmed configuration-failure retry completed successfully with DCLM validation and all nine downstream evaluations; fixed pre-decay step214 saved successfully; train CE is nearest final logged value; minRuntime 0; autoResume false; no experiment retries. LR comparison significance pending paired document statistics and the 5e-4 result."}
  ],
  "repeatedRuns": [
    {"epoch": 1, "lr": "5e-4", "wd": "0.033", "status": "complete", "beaker": "01KZ1XPEAY432MQ8NDJQR62ETB", "job": "01KZ1XPEEQS5F93JDMY2WQ9T12", "wandb": "v8uxyvfb", "revision": "a55cb850", "output": "/weka/oe-training-default/sewonm/icsl/models/dense_1b_step2_0802_repeated_dclm1b_wsd_e1_lr5e-4_wd0.033_warmup24", "train": 4.972, "c4": 5.000, "acc": 25.02, "bpb": 1.528, "downstream": {"arc_challenge": 23.75, "arc_easy": 29.82, "boolq": 38.10, "csqa": 25.31, "openbookqa": 24.00, "piqa": 52.29, "socialiqa": 39.10, "winogrande": 47.83}, "reason": "Replacement lower-rung run completed successfully with DCLM validation and all nine downstream evaluations; fixed pre-decay step214 saved successfully; train CE is nearest final logged value; minRuntime 0; autoResume false; no experiment retries. LR and unique-vs-repeated comparison significance pending paired statistics."},
    {"epoch": 1, "lr": "1e-3", "wd": "0.033", "status": "complete", "beaker": "01KZ1WXY3172DA828HQ47VACST", "job": "01KZ1WXY6GQC9BEGZ95BVGJTKN", "wandb": "g7dpaxjf", "revision": "04f7f19f", "output": "/weka/oe-training-default/sewonm/icsl/models/dense_1b_step2_0802_repeated_dclm1b_wsd_e1_lr1e-3_wd0.033_warmup24_retry1", "train": 5.380, "c4": 5.410, "acc": 25.10, "bpb": 1.656, "downstream": {"arc_challenge": 22.07, "arc_easy": 28.25, "boolq": 37.98, "csqa": 22.11, "openbookqa": 24.60, "piqa": 50.71, "socialiqa": 38.74, "winogrande": 51.93}, "reason": "Confirmed configuration-failure retry completed successfully with DCLM validation and all nine downstream evaluations; fixed pre-decay step214 saved successfully; train CE is nearest final logged value; minRuntime 0; autoResume false; no experiment retries. LR comparison significance pending paired document statistics and the 5e-4 result."}
  ],
  "oldRepeatedRuns": [],
  "codelionRuns": [],
  "evaluationRuns": [
    {"kind": "paired-document-loss", "regime": "repeated", "epoch": 1, "lr": "5e-4", "status": "canceled", "beaker": "01KZ213WK71JWSBJR5P3BEPRN8", "job": "01KZ213WQ8GHG7J4ENP3JBYW1Y", "revision": "3bdc6751", "reason": "Canceled after the shared one-GPU LOCAL_RANK configuration failure was diagnosed; no paired-stat artifact; excluded from comparisons."},
    {"kind": "paired-document-loss", "regime": "repeated", "epoch": 1, "lr": "1e-3", "status": "canceled", "beaker": "01KZ215D5DMXWJ4GD07ZWK6VCH", "job": "01KZ215D8THPB4ZEKC7WVZ8JHE", "revision": "3bdc6751", "reason": "Canceled after the shared one-GPU LOCAL_RANK configuration failure was diagnosed; no paired-stat artifact; excluded from comparisons."},
    {"kind": "paired-document-loss", "regime": "unique", "epoch": 1, "lr": "5e-4", "status": "complete", "beaker": "01KZ21VXQX8G1PDW25SDPKK88P", "job": "01KZ21VXWAM9YCM9VAT0ANQ52S", "wandb": "egyl90as", "revision": "31dbfc14", "output": "/weka/oe-training-default/sewonm/icsl/models/dense_1b_step2_0802_paired_unique_e1_lr5e-4_retry2", "pairedStats": "/weka/oe-training-default/sewonm/icsl/paired-stats/0802_e1_unique_lr5e-4_retry2.npz", "validationCE": 4.99563, "validationDocuments": 79690, "validationRealTokens": 100000118, "scoredTokens": 99975703, "reason": "Paired statistics completed on all 79,690 validation documents; 99,975,703 next-token targets scored from 100,000,118 real tokens. This checkpoint came from an independently sampled 1B training realization, so comparisons against repeated-data checkpoints are conditional fixed-model comparisons, not evidence of a unique-vs-repeated regime effect. One H100; minRuntime 0; autoResume false; no experiment retries."},
    {"kind": "paired-document-loss", "regime": "unique", "epoch": 1, "lr": "1e-3", "status": "complete", "beaker": "01KZ2268ZWWJ06NMAGVK3FY6M7", "job": "01KZ226953AA7DM79NCSMTS60R", "wandb": "c1vhtq62", "revision": "31dbfc14", "output": "/weka/oe-training-default/sewonm/icsl/models/dense_1b_step2_0802_paired_unique_e1_lr1e-3_retry2", "pairedStats": "/weka/oe-training-default/sewonm/icsl/paired-stats/0802_e1_unique_lr1e-3_retry2.npz", "validationCE": 5.48952, "validationDocuments": 79690, "validationRealTokens": 100000118, "scoredTokens": 99975703, "reason": "Paired statistics completed on all 79,690 validation documents and 99,975,703 scored next-token targets. One H100; minRuntime 0; autoResume false; no experiment retries."},
    {"kind": "paired-document-loss", "regime": "repeated", "epoch": 1, "lr": "5e-4", "status": "complete", "beaker": "01KZ229G6K7Z9Q4DESMFJV1FDC", "job": "01KZ229GA3P02HD1833G7X2TVV", "wandb": "q0qjckax", "revision": "31dbfc14", "output": "/weka/oe-training-default/sewonm/icsl/models/dense_1b_step2_0802_paired_repeated_e1_lr5e-4_retry2", "pairedStats": "/weka/oe-training-default/sewonm/icsl/paired-stats/0802_e1_repeated_lr5e-4_retry2.npz", "validationCE": 4.99960, "validationDocuments": 79690, "validationRealTokens": 100000118, "scoredTokens": 99975703, "reason": "Paired statistics completed on all 79,690 validation documents and 99,975,703 scored next-token targets. One H100; minRuntime 0; autoResume false; no experiment retries."},
    {"kind": "paired-document-loss", "regime": "repeated", "epoch": 1, "lr": "1e-3", "status": "complete", "beaker": "01KZ22BANKBYS60SCC0P6NJ1ZA", "job": "01KZ22BAT92PQEZP9RXXVWAM5S", "wandb": "htrzize0", "revision": "31dbfc14", "output": "/weka/oe-training-default/sewonm/icsl/models/dense_1b_step2_0802_paired_repeated_e1_lr1e-3_retry2", "pairedStats": "/weka/oe-training-default/sewonm/icsl/paired-stats/0802_e1_repeated_lr1e-3_retry2.npz", "validationCE": 5.41025, "validationDocuments": 79690, "validationRealTokens": 100000118, "scoredTokens": 99975703, "reason": "Paired statistics completed on all 79,690 validation documents and 99,975,703 scored next-token targets. One H100; minRuntime 0; autoResume false; no experiment retries."}
  ]
};
