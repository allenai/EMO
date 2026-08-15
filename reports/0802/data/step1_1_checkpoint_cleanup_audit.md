# ICSL checkpoint cleanup audit

Completed against the read-write Weka mount at:

`/weka/oe-training-default/sewonm/icsl/models`

## Retention rule

The cleanup retained:

- every checkpoint explicitly referenced by a report or launch configuration;
- every checkpoint named as preserved for a skipped epoch;
- integer-epoch WSD pre-decay and final/evaluation checkpoints;
- the latest checkpoint in every completed or partial output directory; and
- every checkpoint in report-marked active or recently modified output directories.

Only numeric `stepNNN` directories outside all of those sets were deleted.

## Result

| Category | Deleted checkpoints |
|---|---:|
| Dense 1B cosine | 50 |
| Dense 1B Step 1 | 233 |
| Dense 1B Step 2 | 60 |
| Dense 1B Step 2-1 | 13 |
| Dense 153M/474M | 117 |
| MoE | 163 |
| **Total** | **636** |

The frozen initial deletion list contained 635 unique directories in 422 outputs and measured 19,656,775,166,458 bytes (17.88 TiB apparent). One additional 1B periodic checkpoint became eligible as the recent-output safety window elapsed and was deleted after the same checks, bringing the total reclaimed apparent space to approximately 17.89 TiB.

The initial list was validated for exact scope, directory type, uniqueness, reference exclusion, and active-output exclusion before deletion. Its SHA-256 digest was:

`eb4324bd4316e326c7d4ba826ab5f5772b669962012d57592b68e8e44e720cc1`

## Post-delete verification

- Remaining numeric checkpoints: 1,187
- Remaining output directories containing numeric checkpoints: 447
- Existing report/config-referenced checkpoints retained: 181
- Remaining deletion candidates under the same policy: 0
- Deletion errors: 0

Weka directory deletion is permanent unless an external snapshot or backup exists.
