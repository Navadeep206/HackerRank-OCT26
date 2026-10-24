"""Tests for evidence sufficiency rules and fail-closed behavior."""
import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from code.evidence_rules import assess_evidence, load_evidence_rules


class EvidenceRulesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        self.images = self.dataset / "images"
        self.images.mkdir()
        self.image = self.images / "img_1.jpg"
        Image.new("RGB", (40, 30), "white").save(self.image)
        self.rules_file = self.dataset / "evidence_requirements.csv"
        with self.rules_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "requirement_id", "claim_object", "applies_to", "minimum_image_evidence"
            ])
            writer.writeheader()
            writer.writerows([
                {"requirement_id":"REQ_GENERAL_OBJECT_PART","claim_object":"all","applies_to":"general claim review","minimum_image_evidence":"Object and part must be visible."},
                {"requirement_id":"REQ_CAR_BODY_PANEL","claim_object":"car","applies_to":"dent or scratch","minimum_image_evidence":"Panel must be visible."},
                {"requirement_id":"REQ_PACKAGE_CONTENTS","claim_object":"package","applies_to":"contents or inner item","minimum_image_evidence":"Contents must be visible."},
                {"requirement_id":"REQ_REVIEW_TRUST","claim_object":"all","applies_to":"reviewability","minimum_image_evidence":"Evidence must be relevant and reviewable."},
            ])
        self.rules = load_evidence_rules(self.rules_file)

    def tearDown(self):
        self.temp.cleanup()

    def test_loads_rules_and_rejects_duplicate_ids(self):
        self.assertEqual(len(self.rules), 4)
        with self.rules_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["requirement_id","claim_object","applies_to","minimum_image_evidence"])
            writer.writeheader()
            writer.writerows([
                {"requirement_id":"DUP","claim_object":"all","applies_to":"a","minimum_image_evidence":"one"},
                {"requirement_id":"DUP","claim_object":"all","applies_to":"b","minimum_image_evidence":"two"},
            ])
        with self.assertRaises(ValueError):
            load_evidence_rules(self.rules_file)

    def test_matches_car_surface_requirement(self):
        result = assess_evidence(
            claim_object="car", user_claim="A dent on the front bumper",
            image_paths=["images/img_1.jpg"], dataset_dir=self.dataset, rules=self.rules,
            readable_paths=["images/img_1.jpg"],
        )
        self.assertIn("REQ_CAR_BODY_PANEL", result.applicable_requirement_ids)

    def test_matches_package_contents_requirement(self):
        result = assess_evidence(
            claim_object="package", user_claim="The item inside the box is broken",
            image_paths=["images/img_1.jpg"], dataset_dir=self.dataset, rules=self.rules,
            readable_paths=["images/img_1.jpg"],
        )
        self.assertIn("REQ_PACKAGE_CONTENTS", result.applicable_requirement_ids)

    def test_missing_path_fails_closed(self):
        result = assess_evidence(
            claim_object="car", user_claim="Dent on the front bumper",
            image_paths=["images/missing.jpg"], dataset_dir=self.dataset, rules=self.rules,
        )
        self.assertFalse(result.evidence_standard_met)
        self.assertFalse(result.reviewable)
        self.assertIn("images/missing.jpg", result.missing_or_unusable_paths)

    def test_paths_alone_do_not_prove_visual_sufficiency(self):
        result = assess_evidence(
            claim_object="car", user_claim="Dent on the front bumper",
            image_paths=["images/img_1.jpg"], dataset_dir=self.dataset, rules=self.rules,
            readable_paths=["images/img_1.jpg"],
        )
        self.assertFalse(result.evidence_standard_met)
        self.assertIn("awaiting_visual_assessment", result.notes)

    def test_path_traversal_is_unusable(self):
        result = assess_evidence(
            claim_object="car", user_claim="Dent on the front bumper",
            image_paths=["../../outside.jpg"], dataset_dir=self.dataset, rules=self.rules,
        )
        self.assertFalse(result.evidence_standard_met)
        self.assertFalse(result.reviewable)

if __name__ == "__main__":
    unittest.main()
