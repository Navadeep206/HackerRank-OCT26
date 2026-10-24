# Phase 10 — Evaluation and Quality Measurement

## Goal

Measure the pipeline against labeled reference rows without making model calls inside the evaluator. Never report a metric until predictions have actually been generated and compared to the labels.

## Run sequence

From the repository root, install runtime dependencies and configure Gemini in a local, untracked `.env`. To evaluate the labeled sample set, run the pipeline against the sample CSV first:

```bash
python -m code.run_pipeline --claims dataset/sample_claims.csv --output output_sample.csv
python -m code.evaluation.evaluate_predictions --predictions output_sample.csv --references dataset/sample_claims.csv --report evaluation_report.json
```

The first command requires a valid Gemini API key and can incur API usage. The second command is local-only and uses no model calls. Do not commit `.env`, API keys, or generated output unless explicitly requested.

## Report contents

- Row matching by the composite key `user_id + image_paths`, so reordered rows are supported.
- Missing and unexpected rows, duplicate-row validation, and field-level mismatches.
- Accuracy, per-class precision/recall/F1, macro F1 across classes present in the reference set, and a confusion matrix for `claim_status`.
- Exact-match accuracy for `issue_type`, `object_part`, `severity`, `valid_image`, and `evidence_standard_met`.
- Machine-readable JSON output suitable for comparing prompts or configurations.

## Interpretation cautions

- The sample is small and may be imbalanced. Treat scores as diagnostic, not proof of production performance.
- `not_enough_information` is a deliberate conservative class; do not optimize accuracy by forcing uncertain cases into supported/contradicted.
- A reference label can encode context such as vehicle identity across photos; per-image vision classification alone may not resolve all such cases.
- Compare false support and false contradiction errors separately. For insurance-like decisions, review errors and calibration, not just aggregate accuracy.
- Runtime, API calls, retry counts, tokens, and cost should be collected from actual provider telemetry. The current pipeline's model-call count is not token usage or dollar cost; do not fabricate those metrics.
- The evaluator does not tune thresholds automatically and does not use the reference labels to alter predictions.
