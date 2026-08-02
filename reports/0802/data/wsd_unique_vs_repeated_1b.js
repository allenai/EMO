window.STEP2_REPORT_DATA = {
  "updated": "2026-08-02 12:46 PDT · unique epoch-1 5e-4 complete; repeated validation active",
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
    {"epoch": 1, "lr": "4e-4", "wd": "0.033", "status": "failed", "beaker": "01KZ1VH9WSTTJ9MY04WBKVX013", "job": "01KZ1VHA1R0MBNNAXSDVG7PHS7", "revision": "3b96fba5", "output": "/weka/oe-training-default/sewonm/icsl/models/dense_1b_step2_0802_unique_dclm5b_wsd_e1_lr4e-4_wd0.033_warmup24", "reason": "Configuration failure before W&B initialization or training: LMEvaluatorCallback rejected NumpyFSLSubsetDataset because it required NumpyPaddedFSLDataset. Exit 1; no checkpoint; excluded from coordinate grids."},
    {"epoch": 1, "lr": "1e-3", "wd": "0.033", "status": "failed", "beaker": "01KZ1VJXNRQNDNZN0J9G9F5STR", "job": "01KZ1VJXSCEK2X13RW6HFEJTN8", "revision": "3b96fba5", "output": "/weka/oe-training-default/sewonm/icsl/models/dense_1b_step2_0802_unique_dclm5b_wsd_e1_lr1e-3_wd0.033_warmup24", "reason": "Configuration failure before W&B initialization or training: LMEvaluatorCallback rejected NumpyFSLSubsetDataset because it required NumpyPaddedFSLDataset. Exit 1; no checkpoint; excluded from coordinate grids."},
    {"epoch": 1, "lr": "4e-4", "wd": "0.033", "status": "canceled", "beaker": "01KZ1WGXNDFXHP7FCJ33E1G5T1", "job": "01KZ1WGXZGRZJ14S2JTQ8MY3NK", "wandb": "lhvbbqtd", "revision": "ffaa4105", "output": "/weka/oe-training-default/sewonm/icsl/models/dense_1b_step2_0802_unique_dclm5b_wsd_e1_lr4e-4_wd0.033_warmup24_retry1", "reason": "Superseded by the user-requested 5e-4 lower LR rung and manually canceled during epoch-1 training at 11:58 PDT. Retained as provenance and excluded from coordinate grids; never resume this chain."},
    {"epoch": 1, "lr": "5e-4", "wd": "0.033", "status": "complete", "beaker": "01KZ1XN29PN219QMSECNF5WGH4", "job": "01KZ1XN2DK2PTEJAQW5JGMF64M", "wandb": "y5rjtnx5", "revision": "a55cb850", "output": "/weka/oe-training-default/sewonm/icsl/models/dense_1b_step2_0802_unique_dclm5b_wsd_e1_lr5e-4_wd0.033_warmup24", "train": 5.013, "c4": 4.996, "acc": 25.03, "bpb": 1.534, "downstream": {"arc_challenge": 23.75, "arc_easy": 28.42, "boolq": 38.07, "csqa": 23.91, "openbookqa": 24.40, "piqa": 52.29, "socialiqa": 38.69, "winogrande": 48.38}, "reason": "Replacement lower-rung run completed successfully with DCLM validation and all nine downstream evaluations; fixed pre-decay step214 saved successfully; train CE is nearest final logged value; minRuntime 0; autoResume false; no experiment retries. LR comparison significance pending paired document statistics."},
    {"epoch": 1, "lr": "1e-3", "wd": "0.033", "status": "complete", "beaker": "01KZ1WQNDDBA1P8YTSRXG16A01", "job": "01KZ1WQNGWPA1YV4H71NXYCVH6", "wandb": "h269qykw", "revision": "04f7f19f", "output": "/weka/oe-training-default/sewonm/icsl/models/dense_1b_step2_0802_unique_dclm5b_wsd_e1_lr1e-3_wd0.033_warmup24_retry1", "train": 5.509, "c4": 5.490, "acc": 24.47, "bpb": 1.682, "downstream": {"arc_challenge": 22.41, "arc_easy": 27.72, "boolq": 37.95, "csqa": 23.83, "openbookqa": 25.80, "piqa": 52.83, "socialiqa": 39.97, "winogrande": 52.01}, "reason": "Confirmed configuration-failure retry completed successfully with DCLM validation and all nine downstream evaluations; fixed pre-decay step214 saved successfully; train CE is nearest final logged value; minRuntime 0; autoResume false; no experiment retries. LR comparison significance pending paired document statistics and the 5e-4 result."}
  ],
  "repeatedRuns": [
    {"epoch": 1, "lr": "4e-4", "wd": "0.033", "status": "failed", "beaker": "01KZ1VMKTGWG74B7M5ZREXR4QQ", "job": "01KZ1VMKXY8EX0ER5X0AZFR5Y2", "revision": "3b96fba5", "output": "/weka/oe-training-default/sewonm/icsl/models/dense_1b_step2_0802_repeated_dclm1b_wsd_e1_lr4e-4_wd0.033_warmup24", "reason": "Configuration failure before W&B initialization or training: LMEvaluatorCallback rejected NumpyFSLSubsetDataset because it required NumpyPaddedFSLDataset. Exit 1; no checkpoint; excluded from coordinate grids."},
    {"epoch": 1, "lr": "1e-3", "wd": "0.033", "status": "failed", "beaker": "01KZ1VP92657VW72NQH88XW69X", "job": "01KZ1VP95N2D1KA997C7HJ51XC", "revision": "3b96fba5", "output": "/weka/oe-training-default/sewonm/icsl/models/dense_1b_step2_0802_repeated_dclm1b_wsd_e1_lr1e-3_wd0.033_warmup24", "reason": "Configuration failure before W&B initialization or training: LMEvaluatorCallback rejected NumpyFSLSubsetDataset because it required NumpyPaddedFSLDataset. Exit 1; no checkpoint; excluded from coordinate grids."},
    {"epoch": 1, "lr": "4e-4", "wd": "0.033", "status": "canceled", "beaker": "01KZ1WTE2CYC7JFCD142SFFDMQ", "job": "01KZ1WTE5Y2M8GHB28KWPXKCK5", "wandb": "s4rru520", "revision": "04f7f19f", "output": "/weka/oe-training-default/sewonm/icsl/models/dense_1b_step2_0802_repeated_dclm1b_wsd_e1_lr4e-4_wd0.033_warmup24_retry1", "reason": "Superseded by the user-requested 5e-4 lower LR rung and manually canceled during epoch-1 training at 11:58 PDT. Retained as provenance and excluded from coordinate grids; never resume this chain."},
    {"epoch": 1, "lr": "4e-4", "wd": "0.033", "status": "canceled", "duplicate": true, "beaker": "01KZ1WVS71WFJY6FYK18R4ZMY1", "job": "01KZ1WVSACH6TZVZHQ1XECJEE2", "revision": "04f7f19f", "output": "/weka/oe-training-default/sewonm/icsl/models/dense_1b_step2_0802_repeated_dclm1b_wsd_e1_lr4e-4_wd0.033_warmup24_retry1", "reason": "Duplicate launcher submission caused by delayed asynchronous experiment creation; detected in the next all-job audit and manually canceled two seconds after process start, before training initialization. Excluded from coordinate grids; never retry this experiment."},
    {"epoch": 1, "lr": "5e-4", "wd": "0.033", "status": "active", "beaker": "01KZ1XPEAY432MQ8NDJQR62ETB", "job": "01KZ1XPEEQS5F93JDMY2WQ9T12", "wandb": "v8uxyvfb", "revision": "a55cb850", "output": "/weka/oe-training-default/sewonm/icsl/models/dense_1b_step2_0802_repeated_dclm1b_wsd_e1_lr5e-4_wd0.033_warmup24", "reason": "Training and all nine downstream evaluations complete; DCLM validation active at 12:46 PDT; fixed pre-decay step214 saved successfully; 8 H100; minRuntime 0; autoResume false; no experiment retries."},
    {"epoch": 1, "lr": "1e-3", "wd": "0.033", "status": "complete", "beaker": "01KZ1WXY3172DA828HQ47VACST", "job": "01KZ1WXY6GQC9BEGZ95BVGJTKN", "wandb": "g7dpaxjf", "revision": "04f7f19f", "output": "/weka/oe-training-default/sewonm/icsl/models/dense_1b_step2_0802_repeated_dclm1b_wsd_e1_lr1e-3_wd0.033_warmup24_retry1", "train": 5.380, "c4": 5.410, "acc": 25.10, "bpb": 1.656, "downstream": {"arc_challenge": 22.07, "arc_easy": 28.25, "boolq": 37.98, "csqa": 22.11, "openbookqa": 24.60, "piqa": 50.71, "socialiqa": 38.74, "winogrande": 51.93}, "reason": "Confirmed configuration-failure retry completed successfully with DCLM validation and all nine downstream evaluations; fixed pre-decay step214 saved successfully; train CE is nearest final logged value; minRuntime 0; autoResume false; no experiment retries. LR comparison significance pending paired document statistics and the 5e-4 result."}
  ],
  "oldRepeatedRuns": [],
  "codelionRuns": [],
  "evaluationRuns": []
};
