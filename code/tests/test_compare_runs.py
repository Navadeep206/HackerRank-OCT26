"""Tests for report comparison; no external model calls."""
import json
import tempfile
import unittest
from pathlib import Path
from code.evaluation.compare_runs import compare, load_report

class CompareRunsTests(unittest.TestCase):
    def test_metric_delta_and_error_key_changes(self):
        baseline = {"matched_rows": 4, "metrics": {"claim_status": {"accuracy": 0.5, "macro_f1_present_classes": 0.4}}, "errors": [{"user_id": "u1", "image_paths": "a.jpg", "field": "claim_status"}]}
        candidate = {"matched_rows": 4, "metrics": {"claim_status": {"accuracy": 0.75, "macro_f1_present_classes": 0.7}}, "errors": [{"user_id": "u2", "image_paths": "b.jpg", "field": "issue_type"}]}
        result = compare(baseline, candidate)
        self.assertEqual(result["metrics"]["claim_status"]["accuracy_delta"], 0.25)
        self.assertEqual(result["metrics"]["claim_status"]["macro_f1_delta"], 0.3)
        self.assertEqual(len(result["error_keys_removed"]), 1)
        self.assertEqual(len(result["error_keys_added"]), 1)

    def test_rejects_non_finite_metric(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps({"matched_rows": 1, "metrics": {"claim_status": {"accuracy": float("nan")}}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                load_report(path)

    def test_rejects_invalid_matched_row_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps({"matched_rows": -1, "metrics": {}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "matched_rows"):
                load_report(path)

    def test_missing_metric_is_reported_as_unavailable_delta(self):
        baseline = {"matched_rows": 1, "metrics": {"claim_status": {"accuracy": 0.5}}, "errors": []}
        candidate = {"matched_rows": 1, "metrics": {}, "errors": []}
        result = compare(baseline, candidate)
        self.assertIsNone(result["metrics"]["claim_status"]["accuracy_delta"])

    def test_warns_when_matched_row_counts_differ(self):
        result = compare({"matched_rows": 2, "metrics": {}, "errors": []}, {"matched_rows": 3, "metrics": {}, "errors": []})
        self.assertIn("differ", result["row_count_comparison_note"])

if __name__ == "__main__":
    unittest.main()
