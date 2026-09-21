window.ICSL_1B_REPACK_UNTIED_EMBEDWD_REVIEWER={
  "policy": "dense_1b_repack_untied_embedwd_reviewer_v1",
  "updatedAt": "2026-09-21T12:44:38+00:00",
  "model": "1.5B (repository Dense-1B / 1.279B parameters)",
  "pool": "dclm1b",
  "recipe": {
    "dynamicRepacking": true,
    "weightTying": false,
    "decayEmbeddings": true
  },
  "checkpointEpochs": [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32
  ],
  "evaluationEpochs": [
    8,
    12,
    16,
    20,
    24,
    28,
    32
  ],
  "hardTerminalEpoch": 32,
  "stopCriterion": "first adjacent POST validationExact non-improvement",
  "trajectories": [
    {
      "id": "dense-1b-dclm1b-bs64-lr1e-3-wd0.3-repack-untied-embedwd",
      "batchSequences": 64,
      "learningRate": "1e-3",
      "weightDecay": "0.3",
      "manifest": "scripts/models/manifests/dense-1b-repack-untied-embedwd-reviewer-bs64.json",
      "output": "/weka/oe-training-default/sewonm/icsl/models/dense_1b_dclm1b_repack_untied_embedwd_reviewer_v1/bs64_lr1e-3_wd0.3",
      "gpuCount": 8,
      "nodeCount": 1,
      "rankMicrobatchSequences": 8,
      "gradientAccumulation": 1,
      "minRuntime": "8h",
      "status": "running",
      "experiment": "01M30H9FZ9FCDJXE2ZRQ1Q4BY2",
      "job": "01M31CW73MWJ57M1HJR5HV89B0",
      "jobHistory": [
        {
          "job": "01M30H9G51TNHTMD7Z9YB5D105",
          "status": "preempted by system",
          "endedAt": "2026-09-21T07:11:28.526474+00:00",
          "reason": "preempted after exceeding the protected 8-hour minimum runtime"
        }
      ],
      "revision": "e8d4d7d4a739122d2029ec3d9c3e6e2dded6e3ad",
      "progress": {
        "currentEpoch": 16,
        "currentStep": 60225,
        "stageTotalSteps": 61036,
        "stageTargetEpoch": 16,
        "phase": "post_decay_train"
      },
      "completedPdEpochs": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
      "resolvedPostEpochs": [8, 12],
      "completeWorkflowEta": "2026-09-22T08:30:00+00:00",
      "health": {
        "status": "healthy; resumed successfully",
        "checkedAt": "2026-09-21T12:44:38+00:00",
        "criticalSignals": []
      },
      "postDecayResults": {}
    },
    {
      "id": "dense-1b-dclm1b-bs512-lr1e-3-wd1.0-repack-untied-embedwd",
      "batchSequences": 512,
      "learningRate": "1e-3",
      "weightDecay": "1.0",
      "manifest": "scripts/models/manifests/dense-1b-repack-untied-embedwd-reviewer-bs512.json",
      "output": "/weka/oe-training-default/sewonm/icsl/models/dense_1b_dclm1b_repack_untied_embedwd_reviewer_v1/bs512_lr1e-3_wd1.0",
      "gpuCount": 8,
      "nodeCount": 1,
      "rankMicrobatchSequences": 8,
      "gradientAccumulation": 8,
      "minRuntime": "8h",
      "status": "running",
      "experiment": "01M30H9HAPF77KK2V1X31CKPSW",
      "job": "01M31CW78KHB2MHYYQ7FEEGK1F",
      "jobHistory": [
        {
          "job": "01M30H9HEATE6MP42TEAFW7YQY",
          "status": "preempted by system",
          "endedAt": "2026-09-21T07:11:31.806449+00:00",
          "reason": "preempted after exceeding the protected 8-hour minimum runtime"
        }
      ],
      "revision": "e8d4d7d4a739122d2029ec3d9c3e6e2dded6e3ad",
      "progress": {
        "currentEpoch": 16,
        "currentStep": 7555,
        "stageTotalSteps": 7630,
        "stageTargetEpoch": 16,
        "phase": "post_decay_train"
      },
      "completedPdEpochs": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
      "resolvedPostEpochs": [8, 12],
      "completeWorkflowEta": "2026-09-22T06:30:00+00:00",
      "health": {
        "status": "healthy; resumed successfully",
        "checkedAt": "2026-09-21T12:44:38+00:00",
        "criticalSignals": []
      },
      "postDecayResults": {}
    }
  ]
};
