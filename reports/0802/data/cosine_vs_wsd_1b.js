window.ICSL_REPORT_DATA = {
  "updated": "2026-08-02 20:18 PDT",
  "title": "Step 0 — Cosine vs. Cosine (alpha_f=0) vs. WSD",
  "setup": "Dense 1B; verified 0802 Pool-A train-only 1B corpus; 100M-token DCLM validation; sealed test unused; WD 0.033; init seed 12536; data seed 0.",
  "selection": "Compare cosine, cosine alpha_f=0, and WSD at each 1B-token endpoint. Cosine endpoints train from scratch. WSD reuses the original Pool-A Step-2 chain and resumes only matching pre-decay checkpoints.",
  "runs": [
    {"series":"WSD","scheduler":"wsd","epoch":1,"lr":"5e-4","wd":"0.033","status":"complete","beaker":"01KZ1XPEAY432MQ8NDJQR62ETB","job":"01KZ1XPEEQS5F93JDMY2WQ9T12","wandb":"v8uxyvfb","revision":"a55cb850","train":4.972,"validation":5.000,"acc":25.02,"bpb":1.528,"downstream":{"arc_challenge":23.75,"arc_easy":29.82,"boolq":38.10,"csqa":25.31,"openbookqa":24.00,"piqa":52.29,"socialiqa":39.10,"winogrande":47.83}},
    {"series":"WSD","scheduler":"wsd","epoch":2,"lr":"5e-4","wd":"0.033","status":"active","beaker":"01KZ2RRCGB5CHHZRSS1V5B1QPK","job":"01KZ2RRCKXA6RS1CNC4223K4A6","wandb":"jrnkkho3","revision":"919253d4"},
    {"series":"Cosine","scheduler":"cosine","epoch":1,"lr":"5e-4","wd":"0.033","status":"active","beaker":"01KZ2SFGQEBQFEPTBJ4NRNC4P4","job":"01KZ2SFGV22TXN4XBE77TWSB1P","revision":"1cad5bd1"},
    {"series":"Cosine","scheduler":"cosine","epoch":2,"lr":"5e-4","wd":"0.033","status":"active","beaker":"01KZ2SGNQR7ZS9HGX78S4FHMS1","job":"01KZ2SGNW0YB123W2ETRZGZPMJ","revision":"1cad5bd1"},
    {"series":"Cosine","scheduler":"cosine","epoch":3,"lr":"5e-4","wd":"0.033","status":"active","beaker":"01KZ2SHZ1K86MRFANFSKCY8WBN","job":"01KZ2SHZ54W7PEX3AHEW3BCJEQ","revision":"1cad5bd1"},
    {"series":"Cosine","scheduler":"cosine","epoch":4,"lr":"5e-4","wd":"0.033","status":"active","beaker":"01KZ2SK6SSGA87VM2QDNDDF7MD","job":"01KZ2SK6ZC8E7RBVK0ZJD38PE4","revision":"1cad5bd1"},
    {"series":"Cosine","scheduler":"cosine","epoch":5,"lr":"5e-4","wd":"0.033","status":"active","beaker":"01KZ2SMCE9MAV3XCQEECWBDZEM","job":"01KZ2SMCJFTFXX4792WDCB1P79","revision":"1cad5bd1"},
    {"series":"Cosine (alpha_f=0)","scheduler":"cosine0","epoch":1,"lr":"5e-4","wd":"0.033","status":"active","beaker":"01KZ2SNKE4N67QZH3QH16FXJEM","job":"01KZ2SNKJF3Y3XZYFDBSXSYB0P","revision":"1cad5bd1"},
    {"series":"Cosine (alpha_f=0)","scheduler":"cosine0","epoch":2,"lr":"5e-4","wd":"0.033","status":"active","beaker":"01KZ2SPRVJM1TQ1FYQSDAVG32S","job":"01KZ2SPRZE9D8W7403ZFSYFED9","revision":"1cad5bd1"},
    {"series":"Cosine (alpha_f=0)","scheduler":"cosine0","epoch":3,"lr":"5e-4","wd":"0.033","status":"active","beaker":"01KZ2SQY5NJRTDP514KP0YB869","job":"01KZ2SQY9B5J1DQCZT44A4KZY6","revision":"1cad5bd1"},
    {"series":"Cosine (alpha_f=0)","scheduler":"cosine0","epoch":4,"lr":"5e-4","wd":"0.033","status":"queued","beaker":"01KZ2SS7PF1TXVQQ8FWVZNYD0A","job":"01KZ2SS7T1KWYKRSSJ43F4Z8MS","revision":"1cad5bd1"},
    {"series":"Cosine (alpha_f=0)","scheduler":"cosine0","epoch":5,"lr":"5e-4","wd":"0.033","status":"queued","beaker":"01KZ2STFV7G0ZW7538W70A3EAA","job":"01KZ2STGCRKB7VQ8WQEQFE2G4T","revision":"1cad5bd1"}
  ]
};
