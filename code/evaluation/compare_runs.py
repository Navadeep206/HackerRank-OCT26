"""Compare two completed evaluation reports; this script makes no model calls."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path


def load_report(path: Path) -> dict:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read evaluation report {path}: {exc}") from exc
    if not isinstance(report, dict) or not isinstance(report.get("metrics"), dict):
        raise ValueError(f"{path} is not a valid evaluation report: expected an object with a metrics object")
    matched_rows = report.get("matched_rows")
    if isinstance(matched_rows, bool) or not isinstance(matched_rows, int) or matched_rows < 0:
        raise ValueError(f"{path} has invalid matched_rows; expected a non-negative integer")
    for metric_name, values in report["metrics"].items():
        if not isinstance(metric_name, str) or not isinstance(values, dict):
            raise ValueError(f"{path} has invalid metric entry {metric_name!r}")
        for key in ("accuracy", "macro_f1_present_classes"):
            value = values.get(key)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{path} metric {metric_name!r} field {key!r} must be between 0 and 1 or null")
    errors = report.get("errors", [])
    if not isinstance(errors, list) or any(not isinstance(item, dict) for item in errors):
        raise ValueError(f"{path} errors must be a list of objects")
    return report


def _delta(old, new):
    if isinstance(old, (int, float)) and not isinstance(old, bool) and isinstance(new, (int, float)) and not isinstance(new, bool):
        return round(new - old, 4)
    return None


def compare(baseline: dict, candidate: dict) -> dict:
    if baseline.get("matched_rows") != candidate.get("matched_rows"):
        note = "Matched-row counts differ; align evaluated examples before comparison."
    else:
        note = "Matched-row counts agree; verify row keys before comparison."
    fields = sorted(set(baseline["metrics"]) | set(candidate["metrics"]))
    deltas = {}
    for field in fields:
        old = baseline["metrics"].get(field, {})
        new = candidate["metrics"].get(field, {})
        old_acc, new_acc = old.get("accuracy"), new.get("accuracy")
        old_f1, new_f1 = old.get("macro_f1_present_classes"), new.get("macro_f1_present_classes")
        deltas[field] = {
            "accuracy_baseline": old_acc, "accuracy_candidate": new_acc,
            "accuracy_delta": _delta(old_acc, new_acc),
            "macro_f1_baseline": old_f1, "macro_f1_candidate": new_f1,
            "macro_f1_delta": _delta(old_f1, new_f1),
        }
    old_errors = {(e.get("user_id"), e.get("image_paths"), e.get("field")) for e in baseline.get("errors", [])}
    new_errors = {(e.get("user_id"), e.get("image_paths"), e.get("field")) for e in candidate.get("errors", [])}
    return {
        "baseline_matched_rows": baseline.get("matched_rows"),
        "candidate_matched_rows": candidate.get("matched_rows"),
        "row_count_comparison_note": note, "metrics": deltas,
        "error_keys_removed": [list(x) for x in sorted(old_errors - new_errors)],
        "error_keys_added": [list(x) for x in sorted(new_errors - old_errors)],
        "caution": "Descriptive comparison only; row counts are not proof of identical row alignment. Review row-level errors, false-support cases, and the same evaluation examples before selecting a candidate.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare baseline and candidate evaluation JSON reports.")
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=Path("evaluation_comparison.json"))
    args = parser.parse_args()
    try:
        result = compare(load_report(args.baseline), load_report(args.candidate))
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"report": str(args.report), "row_count_comparison_note": result["row_count_comparison_note"], "metrics": result["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
