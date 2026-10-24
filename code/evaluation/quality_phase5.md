# Code quality improvement — Phase 5: evaluation and release hardening

## Changes

- Evaluation comparison now validates report structure, matched-row counts, metric ranges, and error-list shape before producing a comparison.
- Non-finite metrics (NaN/infinity), booleans masquerading as numbers, invalid scores, and malformed JSON are rejected with actionable errors.
- Comparison reports now include both accuracy and macro-F1 deltas, while retaining added/removed error keys.
- Warnings explicitly state that equal row counts do not prove identical row alignment and that comparisons are descriptive, not proof of model quality.
- Added offline regression tests for malformed reports, non-finite metrics, invalid row counts, missing metrics, and macro-F1 changes.

## Release checklist

- [ ] Run `python -m unittest discover -s code/tests -v`.
- [ ] Run `python code/evaluation/dataset_audit.py`.
- [ ] Run the mock-provider pipeline and validate output schema/alignment.
- [ ] Evaluate at least two configurations on the same labeled examples.
- [ ] Review row-level false-support cases, false contradictions, and error-key changes.
- [ ] Record provider/model, prompt/config version, dataset version, timestamp, number of claims/images, request count, cache hits, runtime, and any observed token usage.
- [ ] Estimate API cost only from actual provider usage/pricing inputs; clearly label estimates and missing usage data.
- [ ] Keep secrets, claim-sensitive audit files, and generated predictions out of commits unless the task explicitly requires sharing them.

## Interpretation guardrails

The comparison utility does not establish causal improvement, and matching row counts alone do not prove row alignment. Use the same examples and labels, inspect per-row disagreements, and report limitations. CI uses offline tests and a mock provider; passing CI is not evidence of live-model accuracy or production latency.

## Commands

```bash
python -m unittest discover -s code/tests -v
python code/evaluation/dataset_audit.py
python -m code.evaluation.compare_runs --baseline baseline.json --candidate candidate.json --report comparison.json
```
