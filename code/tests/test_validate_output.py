"""Output contract validator tests."""
import csv
import tempfile
import unittest
from pathlib import Path
from code.evaluation.validate_output import INPUT_COLUMNS, OUTPUT_COLUMNS, validate


class OutputValidatorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.claims = self.root / "claims.csv"
        self.predictions = self.root / "output.csv"
        with self.claims.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=INPUT_COLUMNS)
            w.writeheader()
            w.writerow({"user_id":"u1","image_paths":"images/a.jpg","user_claim":"screen crack","claim_object":"laptop"})

    def tearDown(self):
        self.tmp.cleanup()

    def write_output(self, **overrides):
        row = {
            "user_id":"u1", "image_paths":"images/a.jpg", "user_claim":"screen crack",
            "claim_object":"laptop", "evidence_standard_met":"false",
            "evidence_standard_met_reason":"Not enough information", "risk_flags":"manual_review_required",
            "issue_type":"unknown", "object_part":"screen", "claim_status":"not_enough_information",
            "claim_status_justification":"Image evidence is inconclusive.", "supporting_image_ids":"none",
            "valid_image":"true", "severity":"unknown",
        }
        row.update(overrides)
        with self.predictions.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
            w.writeheader()
            w.writerow(row)

    def test_valid_output_passes(self):
        self.write_output()
        self.assertTrue(validate(self.predictions, self.claims)["valid"])

    def test_invalid_status_fails(self):
        self.write_output(claim_status="maybe")
        result = validate(self.predictions, self.claims)
        self.assertFalse(result["valid"])
        self.assertTrue(any("claim_status" in error for error in result["errors"]))

    def test_row_count_mismatch_fails(self):
        self.write_output()
        with self.claims.open("a", newline="", encoding="utf-8") as handle:
            w = csv.DictWriter(handle, fieldnames=INPUT_COLUMNS)
            w.writerow({"user_id":"u2","image_paths":"images/b.jpg","user_claim":"box crushed","claim_object":"package"})
        result = validate(self.predictions, self.claims)
        self.assertFalse(result["valid"])
        self.assertTrue(any("Row count mismatch" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
