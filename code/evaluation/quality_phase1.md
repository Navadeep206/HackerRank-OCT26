# Code quality improvement — Phase 1: reliability and observability

## Changes

- Track successful image analyses separately from total provider attempts, including retries.
- Report both counts in the CLI so transient provider failures do not make usage telemetry look lower than it was.
- Add a callback to the retry helper that emits one-based attempt numbers without changing retry behavior.
- Extend retry tests to assert attempt telemetry for a fail-then-succeed provider.

## Validation

GitHub Actions runs the complete offline unit suite and mock-provider smoke test. No live Gemini requests are required for these checks. This phase does not change prompts, decision thresholds, or model behavior; it improves diagnostics before optimization work.
