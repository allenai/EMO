window.ICSL_POOL111M_GRID={
  "policy": "dense_dclm111m_integrated_producer_eval_v1",
  "updatedAt": "2026-09-13T04:10:12.740199+00:00",
  "datasetManifest": "src/olmo_core/data/subsets/0802/dclm_0802_repeated_train_111m.json",
  "scheduling": "allocated",
  "minRuntimeOmitted": false,
  "trajectoryCount": 16,
  "lastPolledAt": "2026-09-13T04:02:10.000000Z",
  "liveSummary": {
    "queued": 0,
    "running": 1,
    "terminal": 14,
    "telemetryAvailable": true,
    "warning": "1B BS32 LR5e-4 WD1.0 exhausted all eight retries with reproducible NaN loss at exact step 403"
  },
  "crossWeightDecayPruning": {
    "enabled": true,
    "comparisonScope": "same_model_batch_learning_rate_and_evaluation_epoch",
    "criterion": "higher_weight_decay_strictly_lower_healthy_matched_post_validationExact",
    "action": "stop_lower_weight_decay_trajectory",
    "pairs": [
      {
        "model": "1b",
        "batchSequences": 32,
        "learningRate": "5e-4",
        "lowerWeightDecay": "1.0",
        "higherWeightDecay": "3.0"
      },
      {
        "model": "1b",
        "batchSequences": 64,
        "learningRate": "1e-3",
        "lowerWeightDecay": "1.0",
        "higherWeightDecay": "3.0"
      },
      {
        "model": "474m",
        "batchSequences": 32,
        "learningRate": "1e-3",
        "lowerWeightDecay": "0.3",
        "higherWeightDecay": "1.0"
      },
      {
        "model": "474m",
        "batchSequences": 64,
        "learningRate": "2e-3",
        "lowerWeightDecay": "0.3",
        "higherWeightDecay": "1.0"
      },
      {
        "model": "153m",
        "batchSequences": 32,
        "learningRate": "1e-3",
        "lowerWeightDecay": "0.3",
        "higherWeightDecay": "1.0"
      },
      {
        "model": "153m",
        "batchSequences": 64,
        "learningRate": "2e-3",
        "lowerWeightDecay": "0.3",
        "higherWeightDecay": "1.0"
      }
    ]
  },
  "jobAllowlist": [
    "01M2CER8CV4C68KDATR5Z3VWX1"
  ],
  "retainedCheckpointEvaluation": {
    "id": "dense-153m-dclm111m-bs64-lr2e-3-wd0.3-retained-post-e40-e48-e56-v1",
    "role": "evaluation_only_retained_checkpoint_wsd_and_heldout",
    "policy": "dense_dclm111m_retained_checkpoint_e40_e48_e56_v1",
    "producerId": "dense-153m-dclm111m-bs64-lr2e-3-wd0.3",
    "epochs": [
      40,
      48,
      56
    ],
    "sourceCheckpoints": {
      "40": "/weka/oe-training-default/sewonm/icsl/models/dense_153m_dclm111m/bs64_dr_wt_embwd_lr2e-3_wd0.3/step15243",
      "48": "/weka/oe-training-default/sewonm/icsl/models/dense_153m_dclm111m/bs64_dr_wt_embwd_lr2e-3_wd0.3/step18292",
      "56": "/weka/oe-training-default/sewonm/icsl/models/dense_153m_dclm111m/bs64_dr_wt_embwd_lr2e-3_wd0.3/step21341"
    },
    "status": "terminal_blocked_missing_sources_at_attempt_time",
    "beakerStatus": "stopped",
    "experiment": "01M2BRZ4C01Q7RJC0A6HEEVSJX",
    "job": "01M2BRZ4J3N6XNKWYV8X1RNG1W",
    "jobs": [
      "01M2BRZ4J3N6XNKWYV8X1RNG1W"
    ],
    "revision": "183ec0f924be551a870ffb317fc19d2912c7a269",
    "gpuCount": 4,
    "minRuntimeSeconds": 7200,
    "futureProducerStagesAuthorized": false,
    "terminalReason": "original producer stopped at the E32 saturation gate; exact logs contain no E40/E48/E56 PD retention markers and the recorded checkpoint paths are incomplete",
    "stoppedAt": "2026-09-12T21:44:00.000000+00:00",
    "supersededByContinuationExperiment": "01M2C0JWHVDAYY1WAT7BHA8PY4"
  },
  "replacedUnallocated": {
    "status": "canceled_after_protected_replacements_accepted",
    "experiments": [
      "01M28ZZR0E9TKWY15KB0W5JM2C",
      "01M28ZZS1KP5Y9MTHNR5DHCRJE",
      "01M28ZZT49733MYMB3ET0C71KR",
      "01M28ZZVA542H948XDZWCEHN6V",
      "01M28ZZWBDNJGA8YCEB06WYHST",
      "01M28ZZXFV61GV9EVY5VSR06SA",
      "01M28ZZYKTYCMQYH2P6AC00CJZ",
      "01M28ZZZRG8515FFJQ3NFE5ZS5",
      "01M29000S841H74ZVM20MH9B38",
      "01M29001TDVHJM6NWEZH1216G2",
      "01M29002TQ9REX2VCAY97VT1S7",
      "01M29003VAH94KHPYZXGJFCV05"
    ]
  },
  "trajectories": [
    {
      "id": "dense-1b-dclm111m-bs32-lr5e-4-wd1.0",
      "model": "1b",
      "pool": "dclm111m",
      "batchSequences": 32,
      "learningRate": "5e-4",
      "weightDecay": "1.0",
      "gpuCount": 8,
      "retainedCheckpointEpochs": [
        2,
        4,
        6,
        8,
        10,
        12,
        14,
        16,
        18,
        20,
        22,
        24
      ],
      "evaluationEpochs": [
        4,
        8,
        12,
        16,
        20,
        24
      ],
      "stopOnAdjacentPostNonImprovement": true,
      "status": "failed",
      "experiment": "01M293GQQV3Q0231SE2AZN7RTX",
      "job": "01M294GW6KQJ158BHEA5XVA0B2",
      "jobHistory": [
        {
          "job": "01M293GQY5MJQK65953FCDZB6V",
          "status": "failed",
          "failure": "nan loss at step 403"
        },
        {
          "job": "01M293ZS255H9RXYH177PFM84R",
          "status": "failed",
          "failure": "nan loss at step 403"
        },
        {
          "job": "01M2948P5JGKHBQBMFA2W2NZ8Y",
          "status": "failed",
          "failure": "nan loss at step 403"
        },
        {
          "job": "01M294GW6KQJ158BHEA5XVA0B2",
          "status": "failed",
          "failure": "nan loss at step 403"
        },
        {
          "job": "01M299AY0P824H6SK587NTWZMH",
          "status": "failed",
          "failure": "nan loss at step 403"
        },
        {
          "job": "01M299Q9HVY6BBCV9T53SYP1XC",
          "status": "failed",
          "failure": "nan loss at step 403"
        },
        {
          "job": "01M299Z26PGEC8D3CTX428SFAS",
          "status": "failed",
          "failure": "nan loss at step 403"
        },
        {
          "job": "01M29A5YJGGGJZ5N26A26HYD6W",
          "status": "failed",
          "failure": "nan loss at step 403"
        },
        {
          "job": "01M29AGE7DKMHHQMBAME92FQ5W",
          "status": "failed",
          "failure": "nan loss at step 403"
        }
      ],
      "resolvedCheckpointEpochs": [],
      "resolvedPostEpochs": [],
      "postDecayResults": {},
      "currentEpoch": 0,
      "currentPhase": "terminal",
      "terminalReason": "exhausted all eight retries with NaN loss at exact step 403",
      "revision": "4f29bf3a76f246876cc5bc57e54e86caf6629731",
      "output": "/weka/oe-training-default/sewonm/icsl/models/dense_1b_dclm111m/bs32_dr_wt_embwd_lr5e-4_wd1.0",
      "minRuntimeSeconds": 21600
    },
    {
      "id": "dense-1b-dclm111m-bs32-lr5e-4-wd3.0",
      "model": "1b",
      "pool": "dclm111m",
      "batchSequences": 32,
      "learningRate": "5e-4",
      "weightDecay": "3.0",
      "gpuCount": 8,
      "retainedCheckpointEpochs": [
        2,
        4,
        6,
        8,
        10,
        12,
        14,
        16,
        18,
        20,
        22,
        24
      ],
      "evaluationEpochs": [
        4,
        8,
        12,
        16,
        20,
        24
      ],
      "stopOnAdjacentPostNonImprovement": true,
      "status": "saturated",
      "experiment": "01M293GRKQG4WNN2F8DVMVVNQ1",
      "job": "01M293GRQERDPDWXZB22JECGKR",
      "resolvedCheckpointEpochs": [
        2,
        4,
        6,
        8,
        10,
        12,
        14,
        16,
        18,
        20
      ],
      "resolvedPostEpochs": [
        4,
        8,
        12,
        16,
        20
      ],
      "postDecayResults": {
        "4": {
          "status": "complete",
          "validationExact": 4.14644
        },
        "8": {
          "status": "complete",
          "validationExact": 4.07941
        },
        "12": {
          "status": "complete",
          "validationExact": 4.07021
        },
        "16": {
          "status": "complete",
          "validationExact": 4.06879
        },
        "20": {
          "status": "complete",
          "validationExact": 4.07484
        }
      },
      "currentEpoch": 20,
      "currentPhase": "terminal",
      "terminalReason": "adjacent POST non-improvement at E20 versus E16",
      "revision": "4f29bf3a76f246876cc5bc57e54e86caf6629731",
      "output": "/weka/oe-training-default/sewonm/icsl/models/dense_1b_dclm111m/bs32_dr_wt_embwd_lr5e-4_wd3.0",
      "minRuntimeSeconds": 21600
    },
    {
      "id": "dense-1b-dclm111m-bs64-lr1e-3-wd1.0",
      "model": "1b",
      "pool": "dclm111m",
      "batchSequences": 64,
      "learningRate": "1e-3",
      "weightDecay": "1.0",
      "gpuCount": 8,
      "retainedCheckpointEpochs": [
        2,
        4,
        6,
        8,
        10,
        12,
        14,
        16,
        18,
        20,
        22,
        24
      ],
      "evaluationEpochs": [
        4,
        8,
        12,
        16,
        20,
        24
      ],
      "stopOnAdjacentPostNonImprovement": true,
      "status": "saturated",
      "experiment": "01M293GSCT5TERP8583VVDRR38",
      "job": "01M293GSGEF4Z1BQBMTZSYBRZF",
      "resolvedCheckpointEpochs": [
        2,
        4,
        6,
        8,
        10,
        12
      ],
      "resolvedPostEpochs": [
        4,
        8,
        12
      ],
      "postDecayResults": {
        "4": {
          "status": "complete",
          "validationExact": 3.72232
        },
        "8": {
          "status": "complete",
          "validationExact": 3.60106
        },
        "12": {
          "status": "complete",
          "validationExact": 3.64293
        }
      },
      "currentEpoch": 12,
      "currentPhase": "terminal",
      "terminalReason": "adjacent POST non-improvement at E12 versus E8",
      "revision": "4f29bf3a76f246876cc5bc57e54e86caf6629731",
      "output": "/weka/oe-training-default/sewonm/icsl/models/dense_1b_dclm111m/bs64_dr_wt_embwd_lr1e-3_wd1.0",
      "minRuntimeSeconds": 21600
    },
    {
      "id": "dense-1b-dclm111m-bs64-lr1e-3-wd3.0",
      "model": "1b",
      "pool": "dclm111m",
      "batchSequences": 64,
      "learningRate": "1e-3",
      "weightDecay": "3.0",
      "gpuCount": 8,
      "retainedCheckpointEpochs": [
        2,
        4,
        6,
        8,
        10,
        12,
        14,
        16,
        18,
        20,
        22,
        24
      ],
      "evaluationEpochs": [
        4,
        8,
        12,
        16,
        20,
        24
      ],
      "stopOnAdjacentPostNonImprovement": true,
      "status": "complete",
      "experiment": "01M293GT4S5J5V6XAF9HHDT249",
      "job": "01M293GT8A8HJNB7H61K3B0P76",
      "resolvedCheckpointEpochs": [
        2,
        4,
        6,
        8,
        10,
        12,
        14,
        16,
        18,
        20,
        22,
        24
      ],
      "resolvedPostEpochs": [
        4,
        8,
        12,
        16,
        20,
        24
      ],
      "postDecayResults": {
        "4": {
          "status": "complete",
          "validationExact": 4.31064
        },
        "8": {
          "status": "complete",
          "validationExact": 4.15319
        },
        "12": {
          "status": "complete",
          "validationExact": 4.09019
        },
        "16": {
          "status": "complete",
          "validationExact": 4.0613
        },
        "20": {
          "status": "complete",
          "validationExact": 4.04639
        },
        "24": {
          "status": "complete",
          "validationExact": 4.03187
        }
      },
      "currentEpoch": 24,
      "currentPhase": "terminal",
      "terminalReason": "completed hard ceiling E24",
      "completedAt": "2026-09-12T01:21:58.355301Z",
      "revision": "4f29bf3a76f246876cc5bc57e54e86caf6629731",
      "output": "/weka/oe-training-default/sewonm/icsl/models/dense_1b_dclm111m/bs64_dr_wt_embwd_lr1e-3_wd3.0",
      "minRuntimeSeconds": 21600
    },
    {
      "id": "dense-474m-dclm111m-bs32-lr1e-3-wd0.3",
      "model": "474m",
      "pool": "dclm111m",
      "batchSequences": 32,
      "learningRate": "1e-3",
      "weightDecay": "0.3",
      "gpuCount": 2,
      "retainedCheckpointEpochs": [
        4,
        8,
        12,
        16,
        20,
        24,
        28,
        32
      ],
      "evaluationEpochs": [
        8,
        16,
        24,
        32
      ],
      "stopOnAdjacentPostNonImprovement": true,
      "status": "saturated",
      "experiment": "01M293GTX16M9QRY3WPSG8BFBQ",
      "job": "01M293GV0H055Q1W5BTDTB37ZA",
      "resolvedCheckpointEpochs": [
        4,
        8,
        12,
        16,
        20
      ],
      "resolvedPostEpochs": [
        8,
        16
      ],
      "postDecayResults": {
        "8": {
          "status": "complete",
          "validationExact": 3.69019
        },
        "16": {
          "status": "complete",
          "validationExact": 3.81034
        }
      },
      "currentEpoch": 16,
      "currentPhase": "terminal",
      "terminalReason": "adjacent POST non-improvement at E16 versus E8",
      "revision": "4f29bf3a76f246876cc5bc57e54e86caf6629731",
      "output": "/weka/oe-training-default/sewonm/icsl/models/dense_474m_dclm111m/bs32_dr_wt_embwd_lr1e-3_wd0.3",
      "minRuntimeSeconds": 23400
    },
    {
      "id": "dense-474m-dclm111m-bs32-lr1e-3-wd1.0",
      "model": "474m",
      "pool": "dclm111m",
      "batchSequences": 32,
      "learningRate": "1e-3",
      "weightDecay": "1.0",
      "gpuCount": 2,
      "retainedCheckpointEpochs": [
        4,
        8,
        12,
        16,
        20,
        24,
        28,
        32
      ],
      "evaluationEpochs": [
        8,
        16,
        24,
        32
      ],
      "stopOnAdjacentPostNonImprovement": true,
      "status": "saturated",
      "experiment": "01M293GVNRR123W9AZJMXB74VX",
      "job": "01M293GVSD3XJ0J617W0QRW447",
      "resolvedCheckpointEpochs": [
        4,
        8,
        12,
        16,
        20,
        24,
        28,
        32
      ],
      "resolvedPostEpochs": [
        8,
        16,
        24,
        32
      ],
      "postDecayResults": {
        "8": {
          "status": "complete",
          "validationExact": 3.73274
        },
        "16": {
          "status": "complete",
          "validationExact": 3.66639
        },
        "24": {
          "status": "complete",
          "validationExact": 3.66229
        },
        "32": {
          "status": "complete",
          "validationExact": 3.66676
        }
      },
      "currentEpoch": 32,
      "currentPhase": "terminal",
      "terminalReason": "adjacent POST non-improvement at E32 versus E24",
      "completedAt": "2026-09-12T01:40:10.009101Z",
      "revision": "4f29bf3a76f246876cc5bc57e54e86caf6629731",
      "output": "/weka/oe-training-default/sewonm/icsl/models/dense_474m_dclm111m/bs32_dr_wt_embwd_lr1e-3_wd1.0",
      "minRuntimeSeconds": 23400
    },
    {
      "id": "dense-474m-dclm111m-bs64-lr2e-3-wd0.3",
      "model": "474m",
      "pool": "dclm111m",
      "batchSequences": 64,
      "learningRate": "2e-3",
      "weightDecay": "0.3",
      "gpuCount": 4,
      "retainedCheckpointEpochs": [
        4,
        8,
        12,
        16,
        20,
        24,
        28,
        32
      ],
      "evaluationEpochs": [
        8,
        16,
        24,
        32
      ],
      "stopOnAdjacentPostNonImprovement": true,
      "status": "saturated",
      "experiment": "01M293GWE1VJC8ZJAZRAKC4M6J",
      "job": "01M293GWHHG3TRP6BAYW64787Q",
      "resolvedCheckpointEpochs": [
        4,
        8,
        12,
        16
      ],
      "resolvedPostEpochs": [
        8,
        16
      ],
      "postDecayResults": {
        "8": {
          "status": "complete",
          "validationExact": 3.72114
        },
        "16": {
          "status": "complete",
          "validationExact": 3.86078
        }
      },
      "currentEpoch": 16,
      "currentPhase": "terminal",
      "terminalReason": "adjacent POST non-improvement at E16 versus E8; matched E16 higher-WD result also dominates",
      "revision": "4f29bf3a76f246876cc5bc57e54e86caf6629731",
      "output": "/weka/oe-training-default/sewonm/icsl/models/dense_474m_dclm111m/bs64_dr_wt_embwd_lr2e-3_wd0.3",
      "minRuntimeSeconds": 14400
    },
    {
      "id": "dense-474m-dclm111m-bs64-lr2e-3-wd1.0",
      "model": "474m",
      "pool": "dclm111m",
      "batchSequences": 64,
      "learningRate": "2e-3",
      "weightDecay": "1.0",
      "gpuCount": 4,
      "retainedCheckpointEpochs": [
        4,
        8,
        12,
        16,
        20,
        24,
        28,
        32,
        36,
        40,
        44,
        48,
        52,
        56,
        60,
        64,
        68,
        72,
        76,
        80,
        84,
        88,
        92,
        96
      ],
      "evaluationEpochs": [
        8,
        16,
        24,
        32,
        40,
        48,
        56,
        64,
        72,
        80,
        88,
        96
      ],
      "stopOnAdjacentPostNonImprovement": true,
      "status": "saturated",
      "experiment": "01M29V87NMF2PFBMXFVZFKZKZP",
      "job": "01M29V881J5DGKM7QKV9GGX4NY",
      "previousExperiment": "01M293GX61H7EDQV8BK2DRBBJ2",
      "previousJob": "01M293GX9M097ZJMVCNN3KA8SF",
      "previousExperimentStatus": "completed_hard_ceiling_e32",
      "canceledContinuationExperiment": "01M29FHPHT0ECQM63ZEDPFNTVW",
      "canceledContinuationJob": "01M29FHPP74W513SVWMCH1JVRS",
      "continuationSourceEpoch": 32,
      "continuationTargetEpoch": 96,
      "resolvedCheckpointEpochs": [
        4,
        8,
        12,
        16,
        20,
        24,
        28,
        32,
        36,
        40,
        44,
        48,
        56,
        60,
        64
      ],
      "resolvedPostEpochs": [
        8,
        16,
        24,
        32,
        40,
        48,
        56,
        64
      ],
      "postDecayResults": {
        "8": {
          "status": "complete",
          "validationExact": 3.79271
        },
        "16": {
          "status": "complete",
          "validationExact": 3.68683
        },
        "24": {
          "status": "complete",
          "validationExact": 3.67063
        },
        "32": {
          "status": "complete",
          "validationExact": 3.65843
        },
        "40": {
          "status": "complete",
          "validationExact": 3.65725
        },
        "48": {
          "status": "complete",
          "validationExact": 3.65173,
          "source": "fast_recovered_heldout_eval_only",
          "checkpoint": "/weka/oe-training-default/sewonm/icsl/models/dense_474m_dclm111m/bs64_dr_wt_embwd_lr2e-3_wd1.0/.dclm111m_integrated_producer_eval_v1/post_decay_runs/e48/step20325",
          "wandb": "asmiwddm"
        },
        "56": {
          "status": "complete",
          "validationExact": 3.65099,
          "source": "integrated_isolated_wsd_decay_and_heldout_eval",
          "checkpoint": "/weka/oe-training-default/sewonm/icsl/models/dense_474m_dclm111m/bs64_dr_wt_embwd_lr2e-3_wd1.0/.dclm111m_integrated_producer_eval_v1/post_decay_runs/e56/step23713",
          "wandb": "o940x7wi"
        },
        "64": {
          "status": "complete",
          "validationExact": 3.66659,
          "source": "integrated_isolated_wsd_decay_and_heldout_eval",
          "checkpoint": "/weka/oe-training-default/sewonm/icsl/models/dense_474m_dclm111m/bs64_dr_wt_embwd_lr2e-3_wd1.0/.dclm111m_integrated_producer_eval_v1/post_decay_runs/e64/step27100",
          "wandb": "iw94shm1"
        }
      },
      "currentEpoch": 64,
      "currentPhase": "terminal",
      "terminalReason": "adjacent POST non-improvement at E64 versus E56",
      "completedAt": "2026-09-12T05:49:06.518000Z",
      "recoveredEvaluationExperiment": "01M29R81CMN6GSZXSBVY91BXXG",
      "recoveredEvaluationJob": "01M29R81GD6SG0QRTQQRG2M8BQ",
      "recoveredEvaluationCompletedAt": "2026-09-12T02:52:23.315848Z",
      "startedAt": "2026-09-12T03:41:26.407569Z",
      "revision": "a9716b82d196222a9e3a820c04be45e08e7d9062",
      "output": "/weka/oe-training-default/sewonm/icsl/models/dense_474m_dclm111m/bs64_dr_wt_embwd_lr2e-3_wd1.0",
      "minRuntimeSeconds": 28800
    },
    {
      "id": "dense-474m-dclm111m-bs64-lr1e-3-wd1.0",
      "model": "474m",
      "pool": "dclm111m",
      "batchSequences": 64,
      "learningRate": "1e-3",
      "weightDecay": "1.0",
      "gpuCount": 4,
      "retainedCheckpointEpochs": [
        4,
        8,
        12,
        16,
        20,
        24,
        28,
        32
      ],
      "evaluationEpochs": [
        8,
        16,
        24,
        32
      ],
      "stopOnAdjacentPostNonImprovement": true,
      "status": "saturated",
      "experiment": "01M2BS0822FD7ST6JDX9KTYKGG",
      "job": "01M2BS085X57EKWH97A63E0PP1",
      "jobs": [
        "01M2BS085X57EKWH97A63E0PP1"
      ],
      "beakerStatus": "finalized",
      "currentEpoch": 24,
      "currentPhase": "terminal",
      "resolvedCheckpointEpochs": [
        4,
        8,
        12,
        16,
        20,
        24
      ],
      "resolvedPostEpochs": [
        16,
        24
      ],
      "postDecayResults": {
        "16": {
          "status": "complete",
          "validationExact": 3.65649,
          "source": "integrated_isolated_wsd_decay_and_heldout_eval",
          "checkpoint": "/weka/oe-training-default/sewonm/icsl/models/dense_474m_dclm111m/bs64_dr_wt_embwd_lr1e-3_wd1.0/.dclm111m_integrated_producer_eval_v1/post_decay_runs/e16/step6775",
          "wandb": "u1481may"
        },
        "24": {
          "status": "complete",
          "validationExact": 3.68202,
          "source": "integrated_isolated_wsd_decay_and_heldout_eval",
          "checkpoint": "/weka/oe-training-default/sewonm/icsl/models/dense_474m_dclm111m/bs64_dr_wt_embwd_lr1e-3_wd1.0/.dclm111m_integrated_producer_eval_v1/post_decay_runs/e24/step10163",
          "wandb": "3ku71jiu"
        }
      },
      "startedAt": "2026-09-12T21:44:25.527650Z",
      "wandbHealth": {
        "status": "healthy",
        "checkedAt": "2026-09-12T23:29:02.183941Z",
        "beakerState": "finalized",
        "run": "3ku71jiu",
        "criticalSignals": []
      },
      "terminalReason": "adjacent POST non-improvement at E24 versus E16",
      "completedAt": "2026-09-12T23:29:02.183941Z",
      "revision": "183ec0f924be551a870ffb317fc19d2912c7a269",
      "output": "/weka/oe-training-default/sewonm/icsl/models/dense_474m_dclm111m/bs64_dr_wt_embwd_lr1e-3_wd1.0",
      "minRuntimeSeconds": 14400
    },
    {
      "id": "dense-474m-dclm111m-bs64-lr4e-3-wd1.0",
      "model": "474m",
      "pool": "dclm111m",
      "batchSequences": 64,
      "learningRate": "4e-3",
      "weightDecay": "1.0",
      "gpuCount": 8,
      "retainedCheckpointEpochs": [
        4,
        8,
        12,
        16,
        20,
        24,
        28,
        32,
        36,
        40,
        44,
        48,
        52,
        56,
        60,
        64
      ],
      "evaluationEpochs": [
        8,
        16,
        24,
        32,
        40,
        48,
        56,
        64
      ],
      "stopOnAdjacentPostNonImprovement": true,
      "status": "submitted",
      "experiment": "01M2CFAQ893RPBV8YXZD7WQVKK",
      "job": "01M2CFAQJEMSB3HXTGRZC59S1Q",
      "revision": "f97bc1190fb065c0e8edaf575a9a649cd0e627ca",
      "output": "/weka/oe-training-default/sewonm/icsl/models/dense_474m_dclm111m/bs64_dr_wt_embwd_lr4e-3_wd1.0",
      "minRuntimeSeconds": 19800
    },
    {
      "id": "dense-153m-dclm111m-bs32-lr1e-3-wd0.3",
      "model": "153m",
      "pool": "dclm111m",
      "batchSequences": 32,
      "learningRate": "1e-3",
      "weightDecay": "0.3",
      "gpuCount": 2,
      "retainedCheckpointEpochs": [
        8,
        16,
        24,
        32,
        40,
        48,
        56,
        64,
        72,
        80,
        88,
        96,
        104,
        112,
        120,
        128
      ],
      "evaluationEpochs": [
        16,
        32,
        48,
        64,
        80,
        96,
        112,
        128
      ],
      "stopOnAdjacentPostNonImprovement": true,
      "status": "saturated",
      "experiment": "01M293GY098DADZ2FHGK3QTWCY",
      "job": "01M293GY41MSMEQMCFM4CH0W8T",
      "resolvedCheckpointEpochs": [
        8,
        16,
        24,
        32,
        40,
        48
      ],
      "resolvedPostEpochs": [
        16,
        32,
        48
      ],
      "postDecayResults": {
        "16": {
          "status": "complete",
          "validationExact": 3.7701
        },
        "32": {
          "status": "complete",
          "validationExact": 3.76751
        },
        "48": {
          "status": "complete",
          "validationExact": 3.77021
        }
      },
      "currentEpoch": 48,
      "currentPhase": "terminal",
      "terminalReason": "adjacent POST non-improvement at E48 versus E32",
      "completedAt": "2026-09-12T01:22:36.836702Z",
      "revision": "4f29bf3a76f246876cc5bc57e54e86caf6629731",
      "output": "/weka/oe-training-default/sewonm/icsl/models/dense_153m_dclm111m/bs32_dr_wt_embwd_lr1e-3_wd0.3",
      "minRuntimeSeconds": 28800
    },
    {
      "id": "dense-153m-dclm111m-bs32-lr1e-3-wd1.0",
      "model": "153m",
      "pool": "dclm111m",
      "batchSequences": 32,
      "learningRate": "1e-3",
      "weightDecay": "1.0",
      "gpuCount": 2,
      "retainedCheckpointEpochs": [
        8,
        16,
        24,
        32,
        40,
        48,
        56,
        64,
        72,
        80,
        88,
        96,
        104,
        112,
        120,
        128
      ],
      "evaluationEpochs": [
        16,
        32,
        48,
        64,
        80,
        96,
        112,
        128
      ],
      "stopOnAdjacentPostNonImprovement": true,
      "status": "saturated",
      "experiment": "01M293GYR2MV1ZXQC3GXGMC70V",
      "job": "01M293GYVXPQF8K7SB27834CZ6",
      "resolvedCheckpointEpochs": [
        8,
        16,
        24,
        32
      ],
      "resolvedPostEpochs": [
        16,
        32,
        48,
        64
      ],
      "postDecayResults": {
        "16": {
          "status": "complete",
          "validationExact": 3.94581
        },
        "32": {
          "status": "complete",
          "validationExact": 3.92103
        },
        "48": {
          "status": "complete",
          "validationExact": 3.9156
        },
        "64": {
          "status": "complete",
          "validationExact": 3.92091
        }
      },
      "currentEpoch": 64,
      "currentPhase": "terminal",
      "terminalReason": "adjacent POST non-improvement at E64 versus E48",
      "completedAt": "2026-09-12T02:22:25.958704Z",
      "revision": "4f29bf3a76f246876cc5bc57e54e86caf6629731",
      "output": "/weka/oe-training-default/sewonm/icsl/models/dense_153m_dclm111m/bs32_dr_wt_embwd_lr1e-3_wd1.0",
      "minRuntimeSeconds": 28800
    },
    {
      "id": "dense-153m-dclm111m-bs64-lr2e-3-wd0.3",
      "model": "153m",
      "pool": "dclm111m",
      "batchSequences": 64,
      "learningRate": "2e-3",
      "weightDecay": "0.3",
      "gpuCount": 4,
      "retainedCheckpointEpochs": [
        8,
        16,
        24,
        32,
        40,
        48,
        56,
        64,
        72,
        80,
        88,
        96,
        104,
        112,
        120,
        128
      ],
      "evaluationEpochs": [
        16,
        32,
        48,
        64,
        80,
        96,
        112,
        128
      ],
      "stopOnAdjacentPostNonImprovement": true,
      "status": "saturated",
      "experiment": "01M2C0JWHVDAYY1WAT7BHA8PY4",
      "job": "01M2C0JWNMRB8PRXJS4EE7MAJ7",
      "jobs": [
        "01M2C0JWNMRB8PRXJS4EE7MAJ7"
      ],
      "resolvedCheckpointEpochs": [
        8,
        16,
        24,
        32,
        40,
        48
      ],
      "resolvedPostEpochs": [
        16,
        32,
        48
      ],
      "postDecayResults": {
        "16": {
          "status": "complete",
          "validationExact": 3.77252
        },
        "32": {
          "status": "complete",
          "validationExact": 3.77322
        },
        "48": {
          "status": "complete",
          "validationExact": 3.78031,
          "source": "integrated_isolated_wsd_decay_and_heldout_eval",
          "checkpoint": "/weka/oe-training-default/sewonm/icsl/models/dense_153m_dclm111m/bs64_dr_wt_embwd_lr2e-3_wd0.3/.dclm111m_integrated_producer_eval_v1/post_decay_runs/e48/step20325",
          "wandb": "joq8a2xm"
        }
      },
      "currentEpoch": 48,
      "currentPhase": "terminal",
      "terminalReason": "adjacent POST non-improvement at E48 versus E32",
      "completedAt": "2026-09-13T01:45:45.050225Z",
      "revision": "2b88de9f70bc242c28a7e36e52be6b702361f2ed",
      "output": "/weka/oe-training-default/sewonm/icsl/models/dense_153m_dclm111m/bs64_dr_wt_embwd_lr2e-3_wd0.3",
      "minRuntimeSeconds": 23400,
      "experimentHistory": [
        {
          "experiment": "01M293GZFRV32HRHN9W6QKDMR0",
          "job": "01M293GZM0MSQT2YJ5EB7HBEW2",
          "revision": "4f29bf3a76f246876cc5bc57e54e86caf6629731",
          "status": "saturated",
          "terminalReason": "adjacent POST non-improvement at E32 versus E16",
          "maxValidatedEpoch": 32,
          "output": "/weka/oe-training-default/sewonm/icsl/models/dense_153m_dclm111m/bs64_dr_wt_embwd_lr2e-3_wd0.3"
        },
        {
          "experiment": "01M2BTM2HWYH20T46A3HME7B05",
          "job": "01M2BTM2P1Q7ZNCAJAEDRYMYND",
          "revision": "aff02e457edcfc3266a6132d03c95a580cfda5e0",
          "status": "terminal_noop_prior_saturation_replayed",
          "terminalReason": "v1 exited cleanly before E40 because the CLI entrypoint replayed the prior E32 saturation gate; no checkpoint or output mutation",
          "maxValidatedEpoch": 32,
          "output": "/weka/oe-training-default/sewonm/icsl/models/dense_153m_dclm111m/bs64_dr_wt_embwd_lr2e-3_wd0.3"
        }
      ],
      "beakerStatus": "succeeded",
      "continuationSourceEpoch": 32,
      "continuationSourceCheckpoint": "/weka/oe-training-default/sewonm/icsl/models/dense_153m_dclm111m/bs64_dr_wt_embwd_lr2e-3_wd0.3/step12194",
      "continuationTargetEpoch": 48,
      "continuationCheckpointEpochs": [
        40,
        48
      ],
      "continuationEvaluationEpochs": [
        48
      ],
      "continuationIgnoresPriorE32Saturation": true,
      "continuationHardStopEpoch": 48,
      "submittedAt": "2026-09-12T23:52:31.418673+00:00",
      "minRuntime": "2h"
    },
    {
      "id": "dense-153m-dclm111m-bs64-lr1e-3-wd0.3",
      "model": "153m",
      "pool": "dclm111m",
      "batchSequences": 64,
      "learningRate": "1e-3",
      "weightDecay": "0.3",
      "gpuCount": 4,
      "retainedCheckpointEpochs": [
        8,
        16,
        24,
        32,
        40,
        48,
        56,
        64,
        72,
        80,
        88,
        96,
        104,
        112,
        120,
        128
      ],
      "evaluationEpochs": [
        16,
        32,
        48,
        64,
        80,
        96,
        112,
        128
      ],
      "stopOnAdjacentPostNonImprovement": true,
      "status": "saturated",
      "experiment": "01M29Q3YRQ7RTZ72NGADHM4TYP",
      "job": "01M29Q3Z0D6Z98X8NNKZM0VP7H",
      "resolvedCheckpointEpochs": [
        8,
        16,
        24,
        32
      ],
      "resolvedPostEpochs": [
        16,
        32
      ],
      "postDecayResults": {
        "16": {
          "status": "complete",
          "validationExact": 3.83343
        },
        "32": {
          "status": "complete",
          "validationExact": 3.89059
        }
      },
      "currentEpoch": 32,
      "currentPhase": "terminal",
      "terminalReason": "adjacent POST non-improvement at E32 versus E16",
      "completedAt": "2026-09-12T03:31:36.987091Z",
      "startedAt": "2026-09-12T02:30:21.000000Z",
      "revision": "c1425d6d0b0d212d7023842027fe35d2e81b61cb",
      "output": "/weka/oe-training-default/sewonm/icsl/models/dense_153m_dclm111m/bs64_dr_wt_embwd_lr1e-3_wd0.3",
      "minRuntimeSeconds": 23400
    },
    {
      "id": "dense-153m-dclm111m-bs64-lr2e-3-wd1.0",
      "model": "153m",
      "pool": "dclm111m",
      "batchSequences": 64,
      "learningRate": "2e-3",
      "weightDecay": "1.0",
      "gpuCount": 4,
      "retainedCheckpointEpochs": [
        8,
        16,
        24,
        32,
        40,
        48,
        56,
        64,
        72,
        80,
        88,
        96,
        104,
        112,
        120,
        128
      ],
      "evaluationEpochs": [
        16,
        32,
        48,
        64,
        80,
        96,
        112,
        128
      ],
      "stopOnAdjacentPostNonImprovement": true,
      "status": "saturated",
      "experiment": "01M293H08JR4PFDC8GR31D2GAY",
      "job": "01M293H0C6Y8NHJZ4D55E2SYF8",
      "resolvedCheckpointEpochs": [
        8,
        16,
        24,
        32,
        40,
        48,
        56,
        64,
        72,
        80
      ],
      "resolvedPostEpochs": [
        16,
        32,
        48,
        64,
        80
      ],
      "postDecayResults": {
        "16": {
          "status": "complete",
          "validationExact": 3.94414
        },
        "32": {
          "status": "complete",
          "validationExact": 3.89755
        },
        "48": {
          "status": "complete",
          "validationExact": 3.8816
        },
        "64": {
          "status": "complete",
          "validationExact": 3.87971
        },
        "80": {
          "status": "complete",
          "validationExact": 3.8834
        }
      },
      "currentEpoch": 80,
      "currentPhase": "terminal",
      "terminalReason": "adjacent POST non-improvement at E80 versus E64",
      "completedAt": "2026-09-12T01:50:46.128731Z",
      "revision": "4f29bf3a76f246876cc5bc57e54e86caf6629731",
      "output": "/weka/oe-training-default/sewonm/icsl/models/dense_153m_dclm111m/bs64_dr_wt_embwd_lr2e-3_wd1.0",
      "minRuntimeSeconds": 23400
    },
    {
      "id": "dense-153m-dclm111m-bs64-lr4e-3-wd0.3",
      "model": "153m",
      "pool": "dclm111m",
      "batchSequences": 64,
      "learningRate": "4e-3",
      "weightDecay": "0.3",
      "gpuCount": 4,
      "retainedCheckpointEpochs": [
        8,
        16,
        24,
        32,
        40,
        48,
        56,
        64,
        72,
        80,
        88,
        96,
        104,
        112,
        120,
        128
      ],
      "evaluationEpochs": [
        16,
        32,
        48,
        64,
        80,
        96,
        112,
        128
      ],
      "stopOnAdjacentPostNonImprovement": true,
      "status": "running",
      "experiment": "01M2CER896M9TQZ4V8S7KTZ1WJ",
      "job": "01M2CER8CV4C68KDATR5Z3VWX1",
      "jobs": [
        "01M2CER8CV4C68KDATR5Z3VWX1"
      ],
      "beakerStatus": "running",
      "revision": "eb3799b9d87ed905f62103d98d73286643775345",
      "output": "/weka/oe-training-default/sewonm/icsl/models/dense_153m_dclm111m/bs64_dr_wt_embwd_lr4e-3_wd0.3",
      "minRuntimeSeconds": 14400,
      "resolvedCheckpointEpochs": [
        8,
        16,
        24,
        32
      ],
      "resolvedPostEpochs": [
        16,
        32
      ],
      "postDecayResults": {
        "16": {
          "status": "complete",
          "validationExact": 3.78085,
          "source": "integrated_isolated_wsd_decay_and_heldout_eval",
          "checkpoint": "/weka/oe-training-default/sewonm/icsl/models/dense_153m_dclm111m/bs64_dr_wt_embwd_lr4e-3_wd0.3/.dclm111m_integrated_producer_eval_v1/post_decay_runs/e16/step6775",
          "wandb": "96hcb36g"
        },
        "32": {
          "status": "complete",
          "validationExact": 3.73145,
          "source": "integrated_isolated_wsd_decay_and_heldout_eval",
          "checkpoint": "/weka/oe-training-default/sewonm/icsl/models/dense_153m_dclm111m/bs64_dr_wt_embwd_lr4e-3_wd0.3/.dclm111m_integrated_producer_eval_v1/post_decay_runs/e32/step13550",
          "wandb": "8u9tilzl"
        }
      },
      "currentEpoch": 40,
      "currentPhase": "producer",
      "startedAt": "2026-09-13T04:01:07.478019Z",
      "experimentHistory": [
        {
          "experiment": "01M2C4JGE3T55KDTTFTCZNJBNT",
          "job": "01M2C4JGHSMB170B4CBH7DG0JT",
          "revision": "3f7e2e3819fc0d8c14578e768b72bf770d666217",
          "status": "complete",
          "terminalReason": "completed authorized E32 boundary after E32 POST improved over E16",
          "maxValidatedEpoch": 32,
          "output": "/weka/oe-training-default/sewonm/icsl/models/dense_153m_dclm111m/bs64_dr_wt_embwd_lr4e-3_wd0.3"
        }
      ],
      "continuationSourceEpoch": 32,
      "continuationSourceCheckpoint": "/weka/oe-training-default/sewonm/icsl/models/dense_153m_dclm111m/bs64_dr_wt_embwd_lr4e-3_wd0.3/step12194",
      "continuationTargetEpoch": 128,
      "continuationCheckpointEpochs": [
        40,
        48,
        56,
        64,
        72,
        80,
        88,
        96,
        104,
        112,
        120,
        128
      ],
      "continuationEvaluationEpochs": [
        48,
        64,
        80,
        96,
        112,
        128
      ],
      "continuationHardStopEpoch": 128,
      "submittedAt": "2026-09-13T04:00:07.327982+00:00",
      "minRuntime": "4h",
      "wandbHealth": {
        "status": "healthy_startup",
        "checkedAt": "2026-09-13T04:02:10.000000Z",
        "beakerState": "running",
        "latestStep": 12194,
        "totalSteps": 15243,
        "epoch": 32,
        "eta": "pending/no reliable ETA until E40 training telemetry",
        "criticalSignals": []
      }
    }
  ]
};
