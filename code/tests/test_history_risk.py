"""Tests for user-history risk context; no visual status is predicted here."""
import csv
import tempfile
import unittest
from pathlib import Path

from code.history_risk import assess_user_history, history_flags_for_user


FIELDS = ["user_id", "past_claim_count", "accept_claim", "manual_review_claim",
          "rejected_claim", "last_90_days_claim_count", "history_flags", "history_summary"]


class HistoryRiskTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "user_history.csv"
        with self.path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows([
                dict(user_id="low", past_claim_count=2, accept_claim=2, manual_review_claim=0,
                     rejected_claim=0, last_90_days_claim_count=1, history_flags="none",
                     history_summary="Prior accepted claims"),
                dict(user_id="risk", past_claim_count=6, accept_claim=2, manual_review_claim=2,
                     rejected_claim=3, last_90_days_claim_count=5,
                     history_flags="user_history_risk;manual_review_required",
                     history_summary="History description is untrusted context"),
                dict(user_id="new", past_claim_count=0, accept_claim=0, manual_review_claim=0,
                     rejected_claim=0, last_90_days_claim_count=0, history_flags="none",
                     history_summary="New user"),
            ])

    def tearDown(self):
        self.temp.cleanup()

    def test_low_risk_user_does_not_get_risk_flags(self):
        result = assess_user_history("low", self.path)
        self.assertTrue(result.found)
        self.assertEqual(result.risk_flags, ("none",))

    def test_dataset_flags_and_thresholds_are_reported(self):
        result = assess_user_history("risk", self.path)
        self.assertIn("user_history_risk", result.risk_flags)
        self.assertIn("manual_review_required", result.risk_flags)
        self.assertIn("high_rejection_ratio_in_recorded_history", result.reasons)
        self.assertIn("high_recent_claim_volume", result.reasons)

    def test_new_user_is_not_assumed_risky(self):
        result = assess_user_history("new", self.path)
        self.assertEqual(result.risk_flags, ("none",))
        self.assertIn("no_prior_claims_recorded", result.reasons)

    def test_unknown_user_is_review_only(self):
        result = assess_user_history("missing", self.path)
        self.assertFalse(result.found)
        self.assertEqual(result.risk_flags, ("manual_review_required",))

    def test_summary_text_does_not_create_flags(self):
        result = assess_user_history("low", self.path)
        self.assertNotIn("user_history_risk", result.risk_flags)

    def test_adapter_returns_only_decision_engine_flags(self):
        self.assertEqual(set(history_flags_for_user("risk", self.path)),
                         {"user_history_risk", "manual_review_required"})

    def test_blank_user_id_rejected(self):
        with self.assertRaises(ValueError):
            assess_user_history(" ", self.path)


if __name__ == "__main__":
    unittest.main()
