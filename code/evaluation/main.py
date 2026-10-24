"""Evaluation entry point for HackerRank Orchestrate claim pipeline."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from code.config import AppConfig, ConfigurationError
from code.evaluation.evaluate_predictions import evaluate, read_rows
from code.evaluation.synthetic_edge_cases import run_synthetic_evaluation
from code.pipeline import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate damage claims against labeled ground truth.")
    parser.add_argument("--claims", type=Path, help="Input sample claims CSV (default: dataset/sample_claims.csv)")
    parser.add_argument("--output", type=Path, help="Output predictions CSV path override")
    parser.add_argument("--predictions", type=Path, help="Pre-computed predictions CSV to evaluate directly without running pipeline")
    parser.add_argument("--skip-synthetic", action="store_true", help="Skip synthetic edge case evaluation")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    claims_csv = args.claims or (repo_root / "dataset" / "sample_claims.csv")
    if not claims_csv.exists():
        print(f"Claims CSV not found: {claims_csv}", file=sys.stderr)
        return 2

    # Run synthetic edge case suite first
    synthetic_summary = None
    if not args.skip_synthetic:
        synthetic_summary = run_synthetic_evaluation(verbose=False)

    # Determine predictions file
    predictions_csv = args.predictions
    if not predictions_csv:
        try:
            config = AppConfig.from_env()
        except ConfigurationError as exc:
            print(f"Configuration error: {exc}", file=sys.stderr)
            return 2
        
        output_path = args.output or (repo_root / "code" / "evaluation" / "eval_output.csv")
        config = replace(config, output_csv=output_path.resolve())
        try:
            run_pipeline(config, claims_csv=claims_csv)
            predictions_csv = output_path
        except (OSError, ValueError, ConfigurationError) as exc:
            print(f"Pipeline evaluation failed: {exc}", file=sys.stderr)
            return 2

    # Evaluate predictions
    report = evaluate(read_rows(predictions_csv), read_rows(claims_csv))

    if args.json:
        combined = {
            "dataset_evaluation": report,
            "synthetic_evaluation": synthetic_summary,
        }
        print(json.dumps(combined, indent=2))
    else:
        print("=== EVALUATION REPORT ===")
        print(f"Matched Claims Evaluated: {report['matched_rows']}")
        print(f"Status Accuracy: {report['metrics']['claim_status']['accuracy']:.2%}")
        print(f"Status Macro F1: {report['metrics']['claim_status']['macro_f1_present_classes']:.4f}")
        print(f"Issue Type Accuracy: {report['exact_match']['issue_type']['accuracy']:.2%}")
        print(f"Object Part Accuracy: {report['exact_match']['object_part']['accuracy']:.2%}")
        print(f"Severity Accuracy: {report['exact_match']['severity']['accuracy']:.2%}")
        if synthetic_summary:
            print("\n=== SYNTHETIC EDGE CASES ===")
            print(f"Total Cases: {synthetic_summary['total_cases']}")
            print(f"Passed Cases: {synthetic_summary['passed_cases']} ({synthetic_summary['pass_rate']:.2%})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
