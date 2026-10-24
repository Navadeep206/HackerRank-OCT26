"""Validate a generated predictions CSV against the required output contract and input row count."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

INPUT_COLUMNS = ["user_id", "image_paths", "user_claim", "claim_object"]
OUTPUT_COLUMNS = [
    "user_id", "image_paths", "user_claim", "claim_object",
    "evidence_standard_met", "evidence_standard_met_reason", "risk_flags",
    "issue_type", "object_part", "claim_status", "claim_status_justification",
    "supporting_image_ids", "valid_image", "severity",
]
STATUSES = {"supported", "contradicted", "not_enough_information"}
ISSUES = {"dent", "scratch", "crack", "glass_shatter", "broken_part", "missing_part",
          "torn_packaging", "crushed_packaging", "water_damage", "stain", "none", "unknown"}
SEVERITIES = {"none", "low", "medium", "high", "unknown"}

def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames or [], list(reader)

def validate(predictions: Path, claims: Path, limit: int | None = None) -> dict:
    in_headers, in_rows = read_csv(claims)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        in_rows = in_rows[:limit]
    out_headers, out_rows = read_csv(predictions)
    errors = []
    if in_headers != INPUT_COLUMNS:
        errors.append(f"Input schema mismatch: {in_headers}")
    if out_headers != OUTPUT_COLUMNS:
        errors.append(f"Output schema mismatch: {out_headers}")
    if len(out_rows) != len(in_rows):
        errors.append(f"Row count mismatch: expected {len(in_rows)}, got {len(out_rows)}")
    for i, row in enumerate(out_rows, start=2):
        if row.get("claim_status") not in STATUSES:
            errors.append(f"Row {i}: invalid claim_status {row.get('claim_status')!r}")
        if row.get("issue_type") not in ISSUES:
            errors.append(f"Row {i}: invalid issue_type {row.get('issue_type')!r}")
        if row.get("severity") not in SEVERITIES:
            errors.append(f"Row {i}: invalid severity {row.get('severity')!r}")
        for field in ("evidence_standard_met", "valid_image"):
            if (row.get(field) or "").strip().lower() not in {"true", "false"}:
                errors.append(f"Row {i}: {field} must be true or false")
        for field in ("user_id", "image_paths", "user_claim", "claim_object"):
            if row.get(field, "") != (in_rows[i - 2].get(field, "") if i - 2 < len(in_rows) else None):
                errors.append(f"Row {i}: input field {field} does not match corresponding claim")
    return {"input_rows": len(in_rows), "prediction_rows": len(out_rows),
            "output_columns_exact": out_headers == OUTPUT_COLUMNS, "errors": errors, "valid": not errors}

def main() -> int:
    parser = argparse.ArgumentParser(description="Check prediction CSV row count, schema, allowed values and row alignment.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--claims", type=Path, default=Path("dataset/claims.csv"))
    parser.add_argument("--limit", type=int, help="Validate only the first N input rows for smoke tests")
    args = parser.parse_args()
    report = validate(args.predictions, args.claims, args.limit)
    print(json.dumps(report, indent=2))
    return 0 if report["valid"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
