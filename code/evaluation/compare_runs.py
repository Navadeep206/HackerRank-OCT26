"""Compare two completed evaluation reports; this script makes no model calls."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

def load_report(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or not isinstance(report.get("metrics"), dict):
        raise ValueError(f"{path} is not a valid evaluation report")
    return report

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
        deltas[field] = {
            "accuracy_baseline": old_acc, "accuracy_candidate": new_acc,
            "accuracy_delta": round(new_acc - old_acc, 4) if isinstance(old_acc, (int, float)) and isinstance(new_acc, (int, float)) else None,
            "macro_f1_baseline": old.get("macro_f1_present_classes"),
            "macro_f1_candidate": new.get("macro_f1_present_classes"),
        }
    old_errors = {(e.get("user_id"), e.get("image_paths"), e.get("field")) for e in baseline.get("errors", [])}
    new_errors = {(e.get("user_id"), e.get("image_paths"), e.get("field")) for e in candidate.get("errors", [])}
    return {
        "baseline_matched_rows": baseline.get("matched_rows"),
        "candidate_matched_rows": candidate.get("matched_rows"),
        "row_count_comparison_note": note, "metrics": deltas,
        "error_keys_removed": [list(x) for x in sorted(old_errors - new_errors)],
        "error_keys_added": [list(x) for x in sorted(new_errors - old_errors)],
        "caution": "Descriptive comparison only; review row-level errors and false-support cases.",
    }

def main() -> int:
    parser = argparse.ArgumentParser(description="Compare baseline and candidate evaluation JSON reports.")
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=Path("evaluation_comparison.json"))
    args = parser.parse_args()
    result = compare(load_report(args.baseline), load_report(args.candidate))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.report), "row_count_comparison_note": result["row_count_comparison_note"], "metrics": result["metrics"]}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
