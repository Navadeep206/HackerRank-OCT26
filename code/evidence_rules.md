# Phase 4 — Evidence Requirements Engine

## Purpose
`code/evidence_rules.py` loads and validates the supplied evidence checklist and determines which requirements apply to a parsed claim. It assesses reviewability and evidence sufficiency separately from the final supported/contradicted/not-enough-information decision.

## Implemented behavior
- Loads the challenge's `evidence_requirements.csv` using the standard CSV parser.
- Validates required columns, non-empty unique requirement IDs, and non-empty evidence descriptions.
- Selects general rules and issue/object-specific rules for car, laptop, and package claims.
- Accepts original CSV-relative image paths and rejects path traversal.
- Reports missing/unusable paths, matched requirement IDs, evidence descriptions, and explanatory notes.
- Allows the later image-inspection phase to provide paths confirmed visually relevant.
- Fails closed: file existence alone never sets `evidence_standard_met=true`. A positive label requires `visual_assessment_complete=true` and at least one readable path explicitly confirmed relevant, as well as a sufficiently specific claim target.

## Boundary with later phases
A path can exist while showing the wrong object, wrong part, a blurry image, or no damage. Therefore this phase does not infer visual relevance from filenames or text. Phase 5 will supply per-image quality and relevance observations. Phase 6 will use the resulting evidence assessment alongside visual findings to determine claim status.

## Tests
Run from repository root:

    python -m unittest code.tests.test_evidence_rules -v

Tests cover rule loading and duplicate IDs, car and package requirement matching, missing paths, path traversal, and the rule that paths alone cannot establish visual sufficiency.
