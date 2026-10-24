# Phase 8 — User-history risk intelligence

`code/history_risk.py` reads a user's row from `dataset/user_history.csv` and emits a small, explainable risk-context object.

## Guardrails
- History is never used to decide whether visible damage exists.
- Only explicitly allowed risk flags are propagated from the dataset.
- Deterministic review signals include a rejection ratio of at least 40% with at least five past claims, two or more prior manual reviews, and five or more claims in the last 90 days.
- New users are not automatically risky.
- Missing history triggers manual review because context is unavailable, not because the user is presumed dishonest.
- The free-text history summary is not parsed as instructions or treated as visual proof.

## Use
```python
from pathlib import Path
from code.history_risk import assess_user_history, history_flags_for_user

assessment = assess_user_history("user_005", Path("dataset/user_history.csv"))
print(assessment.to_dict())
flags = history_flags_for_user("user_005", Path("dataset/user_history.csv"))
# Pass flags to decide_claim(..., user_history_flags=flags)
```

## Tests
`python -m unittest code.tests.test_history_risk -v`

The tests use a temporary CSV and do not call Gemini.
