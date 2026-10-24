# Phase 6 — Structured Vision Provider

## Overview
`code/vision_provider.py` defines a provider-neutral vision interface and implementations for Gemini, OpenAI Responses API, and an offline mock. The provider inspects one normalized image per call and returns a validated `VisionFinding`.

## Structured result
Each finding includes the observed object and part, visible issue, damage visibility, relevance confirmation, quality adequacy, confidence in [0, 1], allow-listed risk flags, a concise visual summary, and an explicit semantic-completion marker.

## Safety and reliability
- Strict validation of JSON fields, enumerations, part/object combinations, booleans, confidence range, and risk flags.
- Claim text and image text are explicitly treated as untrusted data, not instructions.
- A model finding for the wrong known object cannot be marked relevant to the claim.
- Errors are bounded and retried with capped exponential backoff plus jitter.
- The mock provider returns unknown findings; it never pretends to inspect image semantics.
- Provider SDKs are imported lazily, so offline tests do not require network credentials.
- A visual summary is bounded in length, and provider exceptions avoid logging secrets or full request contents.

## Setup
For Gemini:

    python -m pip install google-genai
    export MODEL_PROVIDER=gemini
    export GEMINI_API_KEY=...

For OpenAI:

    python -m pip install openai
    export MODEL_PROVIDER=openai
    export OPENAI_API_KEY=...

For offline wiring/tests:

    export MODEL_PROVIDER=mock

Set `MODEL_NAME` if a different provider-supported vision model is desired. Provider SDK/model schema support can vary by version; validate against the provider SDK used in deployment.

## Tests

    python -m unittest code.tests.test_vision_provider -v

These tests cover strict schema checks, object mismatch handling, retry behavior, and the offline provider. They do not make paid API calls.
