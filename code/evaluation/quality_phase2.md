# Code quality improvement — Phase 2: API efficiency

## Changes

- Added a SQLite-backed local vision-result cache.
- Cache keys include a cache schema version, provider, model name, normalized source-image SHA-256, object type, and exact claim text. Changing any of these causes a cache miss.
- Only schema-valid model findings are cached. Malformed or transiently bad output can be retried instead of being replayed from cache.
- Added a per-process request pacer honoring `MODEL_REQUESTS_PER_MINUTE`. Cache hits do not consume the provider request budget.
- CLI reports successful image analyses, analysis attempts, actual provider requests, cache hits, and claim errors separately.
- Added offline tests for cache reuse, cache-key separation, invalid-response recovery, and request pacing.
- Added `.cache/` to `.gitignore`; the cache may contain claim-derived information and must stay local.

## Configuration

- `MODEL_CACHE_ENABLED=true` enables the cache; set it to `false` to force fresh provider calls.
- `MODEL_REQUESTS_PER_MINUTE=10` spaces live provider requests within one process. This is a conservative local limiter, not a distributed quota manager; coordinate limits if multiple workers or machines share one API key.

## Validation

All added tests use local fake providers and should not consume Gemini credits. CI is the authoritative validation source. The cache is an optimization only: cache read/write failures fall back to live inference and do not decide claim outcomes.
