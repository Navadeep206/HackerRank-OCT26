"""Evaluate pipeline predictions against the labeled sample claims CSV.

No model calls are made here. The predictions CSV must already have been
produced by running the pipeline on the same sample claim rows.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

STATUSES = ("supported", "contradicted", "not_enough_information")
METRIC_FIELDS = ("claim_status", "issue_type", "object_part", "severity", "valid_image", "evidence_standard_met")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no CSV header")
        return list(reader)


def _key(row: dict[str, str]) -> str:
    user_id = (row.get("user_id") or "").strip()
    paths = (row.get("image_paths") or "").strip()
    if not user_id:
        raise ValueError("Every row must have a non-empty user_id")
    return user_id + "|" + paths


def confusion_matrix(actual: Iterable[str], predicted: Iterable[str], labels: tuple[str, ...]) -> dict[str, dict[str, int]]:
    matrix = {label: {column: 0 for column in labels} for label in labels}
    for truth, guess in zip(actual, predicted):
        if truth in matrix and guess in matrix[truth]:
            matrix[truth][guess] += 1
    return matrix


def classification_metrics(actual: list[str], predicted: list[str], labels: tuple[str, ...]) -> dict[str, object]:
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted lists must have equal lengths")
    total = len(actual)
    correct = sum(a == p for a, p in zip(actual, predicted))
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for label in labels:
        tp = sum(a == label and p == label for a, p in zip(actual, predicted))
        fp = sum(a != label and p == label for a, p in zip(actual, predicted))
        fn = sum(a == label and p != label for a, p in zip(actual, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = sum(a == label for a in actual)
        per_class[label] = {
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "support": support,
        }
        if support:
            f1_values.append(f1)
    return {
        "count": total,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "correct": correct,
        "per_class": per_class,
        "macro_f1_present_classes": round(sum(f1_values) / len(f1_values), 4) if f1_values else 0.0,
        "confusion_matrix": confusion_matrix(actual, predicted, labels),
    }


def evaluate(predictions: list[dict[str, str]], references: list[dict[str, str]]) -> dict[str, object]:
    pred_map: dict[str, dict[str, str]] = {}
    for row in predictions:
        key = _key(row)
        if key in pred_map:
            raise ValueError(f"Duplicate prediction row key: {key}")
        pred_map[key] = row
    ref_map: dict[str, dict[str, str]] = {}
    for row in references:
        key = _key(row)
        if key in ref_map:
            raise ValueError(f"Duplicate reference row key: {key}")
        ref_map[key] = row

    missing = sorted(set(ref_map) - set(pred_map))
    unexpected = sorted(set(pred_map) - set(ref_map))
    common = sorted(set(ref_map) & set(pred_map))
    report: dict[str, object] = {
        "reference_rows": len(ref_map),
        "prediction_rows": len(pred_map),
        "matched_rows": len(common),
        "missing_predictions": missing,
        "unexpected_predictions": unexpected,
        "metrics": {},
        "exact_match": {},
        "errors": [],
    }
    metrics: dict[str, object] = {}
    exact_match: dict[str, object] = {}
    for field in METRIC_FIELDS:
        actual = [(ref_map[key].get(field) or "").strip().lower() for key in common]
        predicted = [(pred_map[key].get(field) or "").strip().lower() for key in common]
        matches = sum(a == p for a, p in zip(actual, predicted))
        exact_match[field] = {
            "correct": matches,
            "count": len(common),
            "accuracy": round(matches / len(common), 4) if common else 0.0,
        }
        if field == "claim_status":
            metrics[field] = classification_metrics(actual, predicted, STATUSES)
        else:
            labels = tuple(sorted(set(actual + predicted)))
            metrics[field] = classification_metrics(actual, predicted, labels)
    report["metrics"] = metrics
    report["exact_match"] = exact_match
    report["errors"] = [
        {"user_id": ref_map[key].get("user_id", ""), "image_paths": ref_map[key].get("image_paths", ""),
         "field": field, "expected": ref_map[key].get(field, ""), "predicted": pred_map[key].get(field, "")}
        for key in common for field in METRIC_FIELDS
        if (ref_map[key].get(field) or "").strip().lower() != (pred_map[key].get(field) or "").strip().lower()
    ]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare prediction CSV against labeled sample claims.")
    parser.add_argument("--predictions", type=Path, required=True, help="Pipeline output generated from sample claims")
    parser.add_argument("--references", type=Path, default=Path("dataset/sample_claims.csv"))
    parser.add_argument("--report", type=Path, default=Path("evaluation_report.json"))
    args = parser.parse_args()
    report = evaluate(read_rows(args.predictions), read_rows(args.references))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "matched_rows": report["matched_rows"],
        "missing_predictions": len(report["missing_predictions"]),
        "unexpected_predictions": len(report["unexpected_predictions"]),
        "claim_status": report["metrics"]["claim_status"],
        "report": str(args.report),
    }, indent=2))
    return 0 if report["matched_rows"] and not report["missing_predictions"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
