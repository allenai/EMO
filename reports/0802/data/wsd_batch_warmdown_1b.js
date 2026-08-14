window.ICSL_BATCH_WARMDOWN_DATA = {
  "updated": "2026-08-13",
  "title": "Step 1-1 — Dense 1B Batch Warmdown",
  "setup": "Dense 1B; dynamic document repacking; LR 1e-3; WD 0.333. The chain starts from the BS1024 E2 checkpoint at optimizer step 477, runs BS256 for 1,907 more steps to step 2,384, then BS64 for 15,259 more steps to step 17,643.",
  "selection": "The two stages form one initialization chain. BS256 initializes from BS1024 E2; BS64 initializes from the BS256 E4 checkpoint, so its accumulated training history is E8: two epochs at BS1024, two at BS256, and four at BS64.",
  "revision": "315686c620bbea431e87a6ac22f238635fefc486",
  "beaker": "01KZZ8VPV5BD5144HHDVC05JDH",
  "timing": {
    "microbatchSequences": 8,
    "secondsPerMicrobatch": 0.6,
    "gpuCount": 8,
    "description": "Idealized one-node training time uses 0.6 seconds per rank microbatch: BS256 has accumulation 4 and BS64 has accumulation 1. The inherited BS1024 source used 32 GPUs with accumulation 4."
  },
  "recalibration": {
    "needed": true,
    "description": "At each 4x batch reduction, preserve Adam's bias-corrected first-moment signal estimate and multiply only the estimated stochastic-gradient variance inside the second moment by 4. The endpoint checkpoint stores current LR 0 after WSD decay, so loading deliberately restores both LR and base LR to the configured 1e-3 while requiring every non-LR optimizer hyperparameter to match exactly. Optimizer step counters, model weights, trainer token position, and WD are preserved. Distributed checkpoint loading reshards model and optimizer state; the loader recomputes its batch index from the saved token position under the new batch size. Rank-local RNG state is not restored across the 32-GPU to 8-GPU topology change."
  },
  "runs": [
    {
      "stage": "source",
      "batchSequences": 1024,
      "accumulatedEpoch": 2,
      "optimizerStep": 477,
      "addedSteps": 477,
      "status": "complete",
      "validation": 4.128,
      "train": 4.112,
      "wandb": "pp97ixxl",
      "beaker": "01KZRTKNSSX9H1DPZ7NHJSM28F",
      "checkpoint": "/weka/oe-training-default/sewonm/icsl/models/dense_1b_dclm1b/bs1024_dr_lr1e-3_wd0.333/step477",
      "idealizedTrainingSeconds": 1144.8
    },
    {
      "stage": "bs256",
      "batchSequences": 256,
      "accumulatedEpoch": 4,
      "optimizerStep": 2384,
      "addedSteps": 1907,
      "status": "queued",
      "validation": null,
      "train": null,
      "wandb": null,
      "beaker": "01KZZ8VPV5BD5144HHDVC05JDH",
      "output": "/weka/oe-training-default/sewonm/icsl/models/dense_1b_dclm1b/bs256_dr_init=bs1024e2_lr1e-03_wd0.333",
      "checkpoint": "/weka/oe-training-default/sewonm/icsl/models/dense_1b_dclm1b/bs256_dr_init=bs1024e2_lr1e-03_wd0.333/step2384",
      "idealizedTrainingSeconds": 5721.6
    },
    {
      "stage": "bs64",
      "batchSequences": 64,
      "accumulatedEpoch": 8,
      "optimizerStep": 17643,
      "addedSteps": 15259,
      "status": "waiting_for_bs256",
      "validation": null,
      "train": null,
      "wandb": null,
      "beaker": "01KZZ8VPV5BD5144HHDVC05JDH",
      "output": "/weka/oe-training-default/sewonm/icsl/models/dense_1b_dclm1b/bs64_dr_init=bs256e4_init=bs1024e2_lr1e-03_wd0.333",
      "checkpoint": "/weka/oe-training-default/sewonm/icsl/models/dense_1b_dclm1b/bs64_dr_init=bs256e4_init=bs1024e2_lr1e-03_wd0.333/step17643",
      "idealizedTrainingSeconds": 14877.0
    }
  ]
};
