window.STEP2_REPORT_DATA = {
  "updated": "2026-08-02 11:59 PDT · lower LR rung changed from 4e-4 to 5e-4; active 4e-4 runs canceled",
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
    {"epoch": 1, "lr": "1e-3", "wd": "0.033", "status": "active", "beaker": "01KZ1WQNDDBA1P8YTSRXG16A01", "job": "01KZ1WQNGWPA1YV4H71NXYCVH6", "wandb": "h269qykw", "revision": "04f7f19f", "output": "/weka/oe-training-default/sewonm/icsl/models/dense_1b_step2_0802_unique_dclm5b_wsd_e1_lr1e-3_wd0.033_warmup24_retry1", "reason": "Confirmed configuration-failure retry; passed evaluator construction; step25/239 with ETA 35m at 11:53 PDT; minRuntime 0; autoResume false; no experiment retries; preserve step214."}
  ],
  "repeatedRuns": [
    {"epoch": 1, "lr": "4e-4", "wd": "0.033", "status": "failed", "beaker": "01KZ1VMKTGWG74B7M5ZREXR4QQ", "job": "01KZ1VMKXY8EX0ER5X0AZFR5Y2", "revision": "3b96fba5", "output": "/weka/oe-training-default/sewonm/icsl/models/dense_1b_step2_0802_repeated_dclm1b_wsd_e1_lr4e-4_wd0.033_warmup24", "reason": "Configuration failure before W&B initialization or training: LMEvaluatorCallback rejected NumpyFSLSubsetDataset because it required NumpyPaddedFSLDataset. Exit 1; no checkpoint; excluded from coordinate grids."},
    {"epoch": 1, "lr": "1e-3", "wd": "0.033", "status": "failed", "beaker": "01KZ1VP92657VW72NQH88XW69X", "job": "01KZ1VP95N2D1KA997C7HJ51XC", "revision": "3b96fba5", "output": "/weka/oe-training-default/sewonm/icsl/models/dense_1b_step2_0802_repeated_dclm1b_wsd_e1_lr1e-3_wd0.033_warmup24", "reason": "Configuration failure before W&B initialization or training: LMEvaluatorCallback rejected NumpyFSLSubsetDataset because it required NumpyPaddedFSLDataset. Exit 1; no checkpoint; excluded from coordinate grids."},
    {"epoch": 1, "lr": "4e-4", "wd": "0.033", "status": "canceled", "beaker": "01KZ1WTE2CYC7JFCD142SFFDMQ", "job": "01KZ1WTE5Y2M8GHB28KWPXKCK5", "wandb": "s4rru520", "revision": "04f7f19f", "output": "/weka/oe-training-default/sewonm/icsl/models/dense_1b_step2_0802_repeated_dclm1b_wsd_e1_lr4e-4_wd0.033_warmup24_retry1", "reason": "Superseded by the user-requested 5e-4 lower LR rung and manually canceled during epoch-1 training at 11:58 PDT. Retained as provenance and excluded from coordinate grids; never resume this chain."},
    {"epoch": 1, "lr": "4e-4", "wd": "0.033", "status": "canceled", "duplicate": true, "beaker": "01KZ1WVS71WFJY6FYK18R4ZMY1", "job": "01KZ1WVSACH6TZVZHQ1XECJEE2", "revision": "04f7f19f", "output": "/weka/oe-training-default/sewonm/icsl/models/dense_1b_step2_0802_repeated_dclm1b_wsd_e1_lr4e-4_wd0.033_warmup24_retry1", "reason": "Duplicate launcher submission caused by delayed asynchronous experiment creation; detected in the next all-job audit and manually canceled two seconds after process start, before training initialization. Excluded from coordinate grids; never retry this experiment."},
    {"epoch": 1, "lr": "1e-3", "wd": "0.033", "status": "active", "beaker": "01KZ1WXY3172DA828HQ47VACST", "job": "01KZ1WXY6GQC9BEGZ95BVGJTKN", "wandb": "g7dpaxjf", "revision": "04f7f19f", "output": "/weka/oe-training-default/sewonm/icsl/models/dense_1b_step2_0802_repeated_dclm1b_wsd_e1_lr1e-3_wd0.033_warmup24_retry1", "reason": "Confirmed configuration-failure retry; passed evaluator construction; step10/239 with ETA 37m at 11:53 PDT; minRuntime 0; autoResume false; no experiment retries; preserve step214."}
  ],
  "oldRepeatedRuns": [],
  "codelionRuns": [],
  "evaluationRuns": []
};
