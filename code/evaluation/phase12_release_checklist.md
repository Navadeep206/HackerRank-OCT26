# Phase 12 — Final integration and release readiness

## Implemented

- Documented environment setup, Gemini credentials, one-row smoke run, full run, output validation, and sample evaluation in the README.
- Added `code/evaluation/validate_output.py` to check exact output headers, row count, row alignment, and allowed status/issue/severity/boolean values.
- Added validator unit tests.
- Updated CI to run unit tests, CLI validation, a one-row mock-provider smoke test, and the dataset audit.

## Local release sequence

1. Install dependencies and set `GEMINI_API_KEY` in an untracked local `.env`.
2. Run `python -m code.run_pipeline --limit 1 --output smoke_output.csv`.
3. Run `python -m code.evaluation.validate_output --predictions smoke_output.csv --claims dataset/claims.csv --limit 1`.
4. Run the full pipeline with `python -m code.run_pipeline --output output.csv`.
5. Validate all output rows with `python -m code.evaluation.validate_output --predictions output.csv --claims dataset/claims.csv`.
6. Run sample evaluation only after generating sample predictions:
   `python -m code.run_pipeline --claims dataset/sample_claims.csv --output sample_predictions.csv`, then `python -m code.evaluation.evaluate_predictions --predictions sample_predictions.csv --references dataset/sample_claims.csv --report evaluation_report.json`.
7. Save the exact commit SHA, model name/version, prompt version, and report with experiment records. Keep generated claim data and API keys out of commits.

## What CI proves

CI uses mock/fake providers, so it proves that imports, deterministic logic, output schema validation, and a no-key smoke run work. It does not prove live Gemini credentials, live provider behavior, model accuracy, token usage, or cost. Those must be recorded from a real run.

## Submission checklist

- [ ] Run the full pipeline against every row in `dataset/claims.csv`.
- [ ] Validate exact output columns, one-to-one row count, and input row alignment.
- [ ] Generate and inspect the labeled sample evaluation report.
- [ ] Compare at least two prompt/configuration runs using `code/evaluation/compare_runs.py`.
- [ ] Review false-support and false-contradiction cases manually.
- [ ] Record operational telemetry from actual provider responses; do not infer token counts or dollar costs from image-call counts.
- [ ] Prepare the code ZIP and required AI-tool transcript as requested by the hackathon.
- [ ] Confirm no `.env`, API keys, generated claim data, or private audit logs are committed.

## Remaining external verification

The live Gemini run and full-dataset output are intentionally not fabricated or triggered by CI. They require the owner's API credentials and can incur charges. Run them locally before submitting.
