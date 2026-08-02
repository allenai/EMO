window.STEP2_REPORT_DATA = {
  "updated": "2026-08-02 01:03 PDT · exhaustive partition scan active",
  "manifest": {
    "status": "pending exhaustive document scan and independent audit"
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
      "status": "active",
      "reason": "Confirmed retry using a minimal immutable Beaker source snapshot; no GitHub access is required and the same output paths were still absent"
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
