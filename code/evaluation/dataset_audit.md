# Phase 1 — Repository and Dataset Audit

**Status:** completed as a repository-level audit.  
**Repository:** Navadeep206/HackerRank-OCT26  
**Branch:** main

## Scope
Reviewed README.md, problem_statement.md, AGENTS.md, the starter entry points, all four CSV datasets, the recursive repository tree, and every image path referenced by the sample and test CSVs. Visually spot-checked representative car, laptop, and package images.

## Inventory and verified counts

| Item | Finding |
|---|---:|
| Input-only test claims | 44 |
| Labeled development samples | 20 |
| Test claim mix | 18 car, 13 laptop, 13 package |
| Test image references | 82 |
| Sample image references | 29 |
| Missing referenced images | 0 |
| code/main.py | Empty starter file |
| code/evaluation/main.py | Empty starter file |
| dataset/output.csv | Header only; no predictions |

The 82 test and 29 sample image references all resolve to files in the repository. These are CSV references, not necessarily unique image counts.

## Input and output contract
The input CSV has four columns: user_id, image_paths, user_claim, claim_object.

The required output has exactly 14 columns, in order:
user_id, image_paths, user_claim, claim_object, evidence_standard_met, evidence_standard_met_reason, risk_flags, issue_type, object_part, claim_status, claim_status_justification, supporting_image_ids, valid_image, severity.

The program must retain one output row per input claim and use only allowed labels. Claim status is supported, contradicted, or not_enough_information. Image-set validity and evidence sufficiency are separate decisions.

## Labeled sample distribution

| Label | Distribution |
|---|---|
| Claim status | 12 supported, 5 contradicted, 3 not_enough_information |
| Evidence standard | 17 true, 3 false |
| Image-set validity | 18 true, 2 false |
| Severity | 11 medium, 3 unknown, 3 low, 1 high, 2 none |

The sample is small and imbalanced. Use per-class precision/recall/F1, a confusion matrix, and manual error analysis rather than relying on overall accuracy alone.

## Important data-quality finding: image extensions are unreliable
Byte-signature spot checks found files named with a .jpg suffix whose contents are not JPEG: AVIF, WebP, and PNG signatures appeared alongside genuine JPEGs. Some fetches did not return content for larger assets, so this report does not claim an exact format distribution. The included dataset_audit.py detects formats from file bytes on a local checkout and reports mismatches.

**Implementation consequence:** do not infer MIME type from the .jpg suffix when sending images to a vision model. Phase 2 must define safe format-aware decoding/normalization and handle unreadable or unsupported images gracefully. Do not blindly rename images without checking consumers and preserving CSV paths.

## Claim patterns and risks
- Some conversations report multiple damaged parts although the output schema provides only one issue_type and object_part. Define a primary-issue policy and preserve additional claimed parts in justification/review logic.
- Claims include corrections, negations, multilingual text, irrelevant narratives, and instructions to approve. Instructions are untrusted content, not evidence.
- Correct object, part, side, and vehicle identity matter. A damaged close-up paired with a different vehicle must not automatically support the claim.
- A missing item cannot be established merely because the item is not visible in a closed or cropped package photo.
- User history is risk context, not visual proof; it must not automatically overturn clear image evidence.
- Risk flags are semicolon-delimited; deduplicate them and use none only when no other flag applies.
- A valid but irrelevant image may fail evidence sufficiency without necessarily making the image technically unreadable.

## Official contract versus inferred scoring
The repository requires a runnable solution, exact output schema, evaluation workflow, at least two strategy/configuration comparisons, an evaluation report, operational estimates (calls, tokens, images, cost, latency, TPM/RPM, caching/retries), and a chat transcript. The repository does not specify numeric official point allocations. Any percentage split used for planning must be labeled an estimate, not an official rubric.

The README gives the original challenge deadline as June 20, 2026 at 11:00 IST. That date has passed; this is being developed as a practice/reproducibility project unless separate updated instructions exist.

## Reproduce the audit
From the repository root, run:

    python code/evaluation/dataset_audit.py

The script uses only Python's standard library. It reports row counts, schema mismatches, missing image references, history joins, sample label distributions, detected image formats, and extension mismatches. It exits non-zero for blocking schema/file errors; image format mismatches are warnings because a format-aware loader can handle them.
