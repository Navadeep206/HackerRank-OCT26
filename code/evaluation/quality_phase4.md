# Code quality improvement — Phase 4: maintainability and error handling

## Changes

- Runtime numeric configuration rejects non-finite values, fractional values for integer settings, and extreme limits before processing begins.
- Added upper bounds for timeouts, retry counts, request rate, image bytes, pixel count, and dimensions to avoid accidental runaway configurations.
- CLI startup now reports invalid configuration and input/setup errors as concise messages and returns exit code 2 rather than exposing an avoidable traceback.
- Typed the retry-attempt callback as `Callable[[int], None] | None`.
- Added offline configuration regression tests for NaN/infinity, fractional integer values, and extreme retry counts.

## Operational bounds

- Model timeout: maximum 600 seconds.
- Retries: maximum 10.
- Requests per minute: maximum 600 per process.
- Image file size: maximum 200 MiB.
- Image pixels: maximum 200 million.
- Image dimension: maximum 8192 pixels.

These are guardrails, not recommended production values. Prefer lower limits appropriate to the dataset and provider quota.

## Validation

Run `python -m unittest discover -s code/tests -v` for the offline test suite. The GitHub Actions workflow also checks CLI help, the mock-provider pipeline smoke test, and repository/dataset contracts. No live API key is needed for these checks.
