# Phase 9 — End-to-end Gemini pipeline

## Configure
Create a local `.env` from `.env.example`, set `MODEL_PROVIDER=gemini` and your `GEMINI_API_KEY`. Never commit the key. Install dependencies with `pip install -r requirements.txt google-genai python-dotenv pillow-heif`.

## Smoke run
`python -m code.run_pipeline --limit 1 --output smoke_output.csv`

## Full run
`python -m code.run_pipeline --output output.csv`

The runner reads `dataset/claims.csv`, checks image safety and technical quality, calls Gemini for reviewable images, adds user-history risk context, applies the decision engine and evidence rules, then writes the exact 14-column output CSV. It also writes `audit.jsonl` next to the output file with per-image findings, runtime, model-call count, and errors. Keep audit data local when it contains claim information.

Use `--limit 1` first to control API usage. Calls are retried according to `MODEL_MAX_RETRIES`; this phase does not claim live accuracy or cost benchmarks. Pipeline tests inject a fake provider and do not consume Gemini credits:
`python -m unittest code.tests.test_pipeline -v`

## Request pacing and cache

`MODEL_REQUESTS_PER_MINUTE` paces real provider requests in the pipeline process. `MODEL_CACHE_ENABLED=true` enables the local SQLite vision cache at `.cache/vision_cache.sqlite3`; keys include provider/model, image digest, object type, and claim text. Cache entries are validated before reuse. The cache can contain claim-derived data, is excluded from git, and can be cleared by deleting `.cache/`. Set `MODEL_CACHE_ENABLED=false` to force fresh model requests for evaluation experiments. See [`evaluation/quality_phase2.md`](evaluation/quality_phase2.md).
