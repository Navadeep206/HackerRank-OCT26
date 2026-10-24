# Evidence Review Pipeline

This directory contains the implementation and evaluation code for the multimodal damage-claim task. Read the root [problem statement](../problem_statement.md) and [agent contract](../AGENTS.md) before changing entry points.

## Architecture

- `config.py` loads environment settings, validates provider selection and resource limits, and checks credentials without printing them.
- `image_utils.py` resolves CSV paths safely, applies size/pixel limits, detects image format from content rather than extension, applies EXIF orientation, composites transparency onto white, and normalizes supported images to bounded JPEG bytes.
- The model adapter will connect the selected provider in a later phase. Phase 2 makes no network calls and does not send claims to a provider.
- `main.py` and `evaluation/main.py` remain the expected entry points and will be implemented in later phases.

## Setup

Use Python 3.11 or newer. From the repository root:

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
cp .env.example .env
```

For live inference, install only the SDK for the selected provider and set its API key in the environment or untracked `.env` file:

```bash
python -m pip install google-genai  # Gemini
# or: python -m pip install openai
```

Never commit `.env`, API keys, or generated private data.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PROVIDER` | `gemini` | `gemini` or `openai` (real vision LLMs only) |
| `MODEL_NAME` | Provider-specific | Model identifier (`gemini-2.5-flash` or `gpt-4.1-mini`) |
| `MODEL_TIMEOUT_SECONDS` | `60` | Timeout for vision model requests |
| `MODEL_MAX_RETRIES` | `3` | Retry budget for retryable failures |
| `MODEL_REQUESTS_PER_MINUTE` | `10` | Conservative throttle target |
| `MODEL_CACHE_ENABLED` | `true` | Persistent response caching (SQLite) |
| `MAX_IMAGE_BYTES` | `20971520` | Source-file size limit (20 MiB) |
| `MAX_IMAGE_PIXELS` | `40000000` | Decoded pixel limit |
| `MAX_IMAGE_DIMENSION` | `2048` | Longest normalized edge |
| `DATASET_DIR` | `dataset` | Dataset root |
| `OUTPUT_CSV` | `output.csv` | Output CSV destination |

Production execution requires a real multimodal vision provider (`gemini` or `openai`) and corresponding credentials (`GEMINI_API_KEY` or `OPENAI_API_KEY`). Mock providers are completely removed from production runtime.
