# Phase 6 — Structured Vision Provider

## Overview
`code/vision_provider.py` defines a provider-neutral vision interface and implementations for Gemini and OpenAI Responses API. The provider inspects one normalized image per call and returns a validated `VisionFinding`.

## Structured result
Each finding includes the observed object and part, visible issue, damage visibility, relevance confirmation, quality adequacy, confidence in [0, 1], allow-listed risk flags, a concise visual summary, and an explicit semantic-completion marker.

## Safety and reliability
- Strict validation of JSON fields, enumerations, part/object combinations, booleans, finite confidence in [0, 1], and risk flags.
- Claim text and image text are explicitly treated as untrusted data, not instructions.
- A model finding for the wrong known object cannot be marked relevant to the claim.
- Errors are bounded and retried with capped exponential backoff plus jitter.
- Credential tokens matching API key patterns are automatically redacted from findings and logs.
- Provider exceptions avoid logging secrets or full request contents.

## Setup
For Gemini (default):

    python -m pip install google-genai
    export MODEL_PROVIDER=gemini
    export GEMINI_API_KEY=...

For OpenAI:

    python -m pip install openai
    export MODEL_PROVIDER=openai
    export OPENAI_API_KEY=...

Set `MODEL_NAME` if a different provider-supported vision model is desired (`gemini-2.5-flash` or `gpt-4.1-mini`).

## Tests

    python -m unittest code.tests.test_vision_provider -v

These unit tests cover strict schema checks, object mismatch handling, retry behavior, client stubs, and non-finite confidence rejection using local test doubles without making paid API calls. Live API calls are opt-in via `test_opt_in_live_gemini_call` when `GEMINI_API_KEY` is present.
