"""CLI entry point for the Gemini-first claims pipeline."""
from __future__ import annotations
import argparse
from dataclasses import replace
from pathlib import Path
from code.config import AppConfig
from code.pipeline import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate damage claims with Gemini vision.")
    parser.add_argument("--claims", type=Path, help="Input CSV (default: dataset/claims.csv)")
    parser.add_argument("--limit", type=int, help="Process only first N claims; use 1 for smoke test")
    parser.add_argument("--output", type=Path, help="Output CSV path override")
    args = parser.parse_args()
    config = AppConfig.from_env()
    if args.output:
        config = replace(config, output_csv=args.output.resolve())
    results = run_pipeline(config, claims_csv=args.claims, limit=args.limit)
    calls = sum(result["model_calls"] for result in results)
    attempts = sum(result.get("model_attempts", result["model_calls"]) for result in results)
    requests = sum(result.get("provider_requests", 0) for result in results)
    cache_hits = sum(result.get("cache_hits", 0) for result in results)
    errors = sum(result["error"] is not None for result in results)
    print(f"Processed claims: {len(results)}")
    print(f"Successfully analyzed images: {calls}")
    print(f"Analysis attempts (including cache hits): {attempts}")
    print(f"Actual provider requests: {requests}")
    print(f"Vision cache hits: {cache_hits}")
    print(f"Claims with image/model errors: {errors}")
    print(f"Output CSV: {config.output_csv}")
    print(f"Audit JSONL: {config.output_csv.with_name('audit.jsonl')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
