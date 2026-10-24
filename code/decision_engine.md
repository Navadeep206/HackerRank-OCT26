# Phase 7 — Claim Decision Engine

## Purpose
`code/decision_engine.py` combines the parsed user intent, validated per-image vision findings, and evidence checklist into the required claim-level decision fields.

## Decision policy
- A clear, relevant finding that matches both the described issue and part can support the claim.
- A clear, relevant finding that shows no issue or a different issue/part can contradict the claim.
- Underspecified claims, low confidence, unclear quality, or conflicts between clear images yield `not_enough_information` and manual-review risk where appropriate.
- Evidence sufficiency is separate from claim support: a clear image may contradict a claim while still providing enough evidence to evaluate it.
- User-history risk flags are included as context but do not change the visual status.
- Embedded instructions are flagged and never followed.

## Output
The `ClaimDecision` dataclass returns claim status and justification, evidence-standard fields, risk flags, issue/part, supporting image IDs, image validity, severity, matched requirement IDs, and decision notes.

## Tests

    python -m unittest code.tests.test_decision_engine -v

Tests cover supported, contradicted, uncertain, conflicting-image, injection, and history-nonoverride cases. No provider calls or API credits are needed.
