# Phase 3 — Claim Conversation Extraction

`code/claim_parser.py` provides deterministic, testable extraction of the user's intended claim from a conversation. It is deliberately separate from visual evidence assessment: extracted text is a hypothesis about what to inspect, not proof that damage exists.

## Contract

`extract_claim_intent(user_claim, claim_object=None)` returns an immutable `ClaimIntent` object. Its `to_dict()` method converts the result into JSON-friendly data.

The parser extracts:
- authoritative object type when supplied by the input CSV;
- a primary issue type and object part;
- all distinct mentioned parts, so a single-field output schema does not erase conversation context;
- negated issue mentions and correction-language indicators;
- embedded approval/evidence-override instruction indicators;
- a limited multilingual hint signal and extraction notes.

## Safety and decision boundaries

- The structured `claim_object` field wins when it is valid. Text cannot silently change the dataset's object category.
- Embedded instructions such as "ignore previous instructions" are flagged as untrusted content. They must never change the evidence decision.
- The parser does not label a claim supported or contradicted and does not infer that damage is visually present.
- Negation/correction detection is heuristic. It is surfaced explicitly for later review rather than presented as a full natural-language understanding system.
- If several parts or issues are mentioned, the parser selects a deterministic primary mention and preserves the others for downstream reasoning. Phase 4+ should combine this with evidence requirements and visual results.

## Run tests

From the repository root:

    python -m unittest discover -s code/tests -v

The regression suite covers object/part aliases, corrections, negation, package damage wording, embedded instruction detection, empty conversations, and JSON serialization.
