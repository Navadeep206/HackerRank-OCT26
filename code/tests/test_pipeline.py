"""Pipeline orchestration tests with a local fake provider (no Gemini calls)."""
import csv
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from code.config import AppConfig
from code.pipeline import OUTPUT_COLUMNS, process_claim, write_output
from code.evidence_rules import load_evidence_rules


class FakeProvider:
    def analyze(self, image, *, claim_object, user_claim):
        return {
            "observed_object": "laptop", "observed_part": "screen",
            "visible_issue": "crack", "damage_visibility": "clearly_visible",
            "relevance_confirmed": True, "quality_adequate": True,
            "confidence": 0.96, "risk_flags": [],
            "visual_summary": "A crack is clearly visible on the laptop screen.",
        }


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset = self.root / "dataset"
        (self.dataset / "images").mkdir(parents=True)
        Image.new("RGB", (640, 480), "white").save(self.dataset / "images/img_1.jpg")
        with (self.dataset / "evidence_requirements.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["requirement_id", "claim_object", "applies_to", "minimum_image_evidence"])
            w.writeheader()
            w.writerow({"requirement_id":"r1","claim_object":"all","applies_to":"all","minimum_image_evidence":"Show the claimed part"})
        with (self.dataset / "user_history.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["user_id","past_claim_count","accept_claim","manual_review_claim","rejected_claim","last_90_days_claim_count","history_flags","history_summary"])
            w.writeheader()
            w.writerow({"user_id":"u1","past_claim_count":"1","accept_claim":"1","manual_review_claim":"0","rejected_claim":"0","last_90_days_claim_count":"1","history_flags":"none","history_summary":"Prior accepted claim"})
        self.config = AppConfig(self.root, self.dataset, self.root/"output.csv", "gemini",
            "gemini-2.5-flash", 60, 0, 20*1024*1024, 40_000_000, 2048, 10, False)

    def tearDown(self):
        self.temp.cleanup()

    def test_claim_processes_without_live_api(self):
        result = process_claim({"user_id":"u1","image_paths":"images/img_1.jpg","user_claim":"The laptop screen has a crack","claim_object":"laptop"}, self.config, FakeProvider(), load_evidence_rules(self.dataset/"evidence_requirements.csv"))
        self.assertEqual(result["decision"]["claim_status"], "supported")
        self.assertEqual(result["model_calls"], 1)

    def test_csv_uses_exact_schema(self):
        result = process_claim({"user_id":"u1","image_paths":"images/img_1.jpg","user_claim":"The laptop screen has a crack","claim_object":"laptop"}, self.config, FakeProvider(), load_evidence_rules(self.dataset/"evidence_requirements.csv"))
        write_output([result], self.config.output_csv)
        with self.config.output_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, OUTPUT_COLUMNS)
            self.assertEqual(len(list(reader)), 1)

    def test_bad_image_is_isolated_and_reviewed(self):
        result = process_claim({"user_id":"u1","image_paths":"images/missing.jpg","user_claim":"The laptop screen has a crack","claim_object":"laptop"}, self.config, FakeProvider(), load_evidence_rules(self.dataset/"evidence_requirements.csv"))
        self.assertEqual(result["decision"]["claim_status"], "not_enough_information")
        self.assertIn("manual_review_required", result["decision"]["risk_flags"])


if __name__ == "__main__":
    unittest.main()
