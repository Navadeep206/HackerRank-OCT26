# Phase 11 — Prompt and Configuration Experiments

## Purpose

Phase 11 adds a reproducible workflow for comparing a baseline model/prompt configuration with a candidate. It does not automatically change the production prompt or tune decision thresholds against reference labels.

## Experiment protocol

1. Freeze the current code commit and reference CSV.
2. Create baseline and candidate prediction CSVs from the same rows, image set, provider/model version, and retry policy. Change one variable at a time.
3. Save each prediction CSV, evaluation JSON, and manifest with commit SHA, date, model name, prompt version, changed variable, and notes. Never store API keys in the manifest.
4. Evaluate each prediction file, then compare the reports.
5. Inspect row-level errors, especially false supported predictions and false contradicted predictions. Confirm row counts and sample keys match.
6. Select a candidate only after manual review and a separate holdout check. Do not repeatedly tune on the same small sample and present that score as an unbiased estimate.

## Example

~~~bash
python -m code.evaluation.evaluate_predictions --predictions baseline.csv --references dataset/sample_claims.csv --report baseline_report.json
python -m code.evaluation.evaluate_predictions --predictions candidate.csv --references dataset/sample_claims.csv --report candidate_report.json
python -m code.evaluation.compare_runs --baseline baseline_report.json --candidate candidate_report.json --report comparison.json
~~~

The evaluation and comparison commands are local-only. Generating predictions with Gemini requires a configured API key and may incur API usage.

## Interpretation rules

- Overall accuracy is not enough if false support increases.
- Review per-class precision, recall, F1, and class support.
- Preserve not_enough_information as a valid conservative outcome.
- Record prompt version and exact commit SHA with every result.
- Do not select prompt text or thresholds on the same examples and treat the score as an independent test.
- No benchmark numbers are published until actual prediction files are evaluated.
