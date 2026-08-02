window.STEP2_REPORT_DATA = {
  "updated": "2026-08-02 11:09 PDT · generation complete; independent audit active",
  "manifest": {
    "status": "generated; independent audit active"
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
      "status": "validating",
      "reason": "Generation completed after parallel resumable per-shard materialization and sequential hash-preserving assembly. Generator checks report 2,949,254,346 source documents and 3,684,671,294,617 source tokens; train/validation/test document intersections are all zero, the split is exhaustive, and the 4,096-bucket key-uniformity sanity check passed. Independent audit is active before training."
    }
  ],
  "uniqueRuns": [
    {"epoch": 1, "lr": "4e-4", "wd": "0.033", "status": "planned", "reason": "Pending verified 0802 split"},
    {"epoch": 1, "lr": "1e-3", "wd": "0.033", "status": "planned", "reason": "Pending verified 0802 split"}
  ],
  "repeatedRuns": [
    {"epoch": 1, "lr": "4e-4", "wd": "0.033", "status": "planned", "reason": "Pending verified 0802 split"},
    {"epoch": 1, "lr": "1e-3", "wd": "0.033", "status": "planned", "reason": "Pending verified 0802 split"}
  ],
  "oldRepeatedRuns": [],
  "codelionRuns": [],
  "evaluationRuns": []
};
