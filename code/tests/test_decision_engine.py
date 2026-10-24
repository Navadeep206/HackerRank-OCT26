"""Unit tests for conservative claim-level decision policy."""
import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from code.decision_engine import decide_claim
from code.evidence_rules import load_evidence_rules
from code.vision_provider import validate_finding


def finding(image_id="img_1", **overrides):
    payload = {
        "observed_object": "laptop", "observed_part": "screen",
        "visible_issue": "crack", "damage_visibility": "clearly_visible",
        "relevance_confirmed": True, "quality_adequate": True,
        "confidence": 0.92, "risk_flags": [], "visual_summary": "A clear crack is visible on the screen.",
    }
    payload.update(overrides)
    return validate_finding(payload, expected_object="laptop", image_id=image_id)


class DecisionEngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset = self.root / "dataset"
        (self.dataset / "images").mkdir(parents=True)
        Image.new("RGB", (320, 240), "white").save(self.dataset / "images" / "img_1.jpg")
        Image.new("RGB", (320, 240), "white").save(self.dataset / "images" / "img_2.jpg")
        self.rules_file = self.dataset / "evidence_requirements.csv"
        with self.rules_file.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "requirement_id", "claim_object", "applies_to", "minimum_image_evidence"
            ])
            writer.writeheader()
            writer.writerow({
                "requirement_id": "REQ_GENERAL", "claim_object": "all",
                "applies_to": "general claim review",
                "minimum_image_evidence": "The relevant part must be visible.",
            })
        self.rules = load_evidence_rules(self.rules_file)

    def tearDown(self):
        self.temp.cleanup()

    def decide(self, claim, findings, history=()):
        return decide_claim(
            claim_object="laptop", user_claim=claim,
            image_paths=["images/img_1.jpg"], dataset_dir=self.dataset,
            findings=findings, rules=self.rules, user_history_flags=history,
        )

    def test_matching_clear_finding_supports_claim(self):
        result = self.decide("The laptop screen has a crack", [finding()])
        self.assertEqual(result.claim_status, "supported")
        self.assertEqual(result.issue_type, "crack")
        self.assertEqual(result.object_part, "screen")
        self.assertEqual(result.supporting_image_ids, ("img_1",))
        self.assertTrue(result.evidence_standard_met)

    def test_clear_absence_contradicts_claim(self):
        result = self.decide("The laptop screen has a crack", [
            finding(visible_issue="none", visual_summary="No visible damage on the screen.")
        ])
        self.assertEqual(result.claim_status, "contradicted")
        self.assertEqual(result.supporting_image_ids, ("img_1",))

    def test_low_confidence_does_not_force_decision(self):
        result = self.decide("The laptop screen has a crack", [finding(confidence=0.35)])
        self.assertEqual(result.claim_status, "not_enough_information")
        self.assertIn("manual_review_required", result.risk_flags)
        self.assertFalse(result.evidence_standard_met)

    def test_underspecified_claim_is_not_enough_information(self):
        result = self.decide("Something looks wrong", [finding()])
        self.assertEqual(result.claim_status, "not_enough_information")

    def test_history_flags_do_not_override_visual_result(self):
        ordinary = self.decide("The laptop screen has a crack", [finding()])
        flagged = self.decide("The laptop screen has a crack", [finding()], history=["user_history_risk"])
        self.assertEqual(ordinary.claim_status, flagged.claim_status)
        self.assertIn("user_history_risk", flagged.risk_flags)

    def test_conflicting_clear_findings_escalate(self):
        result = decide_claim(
            claim_object="laptop", user_claim="The laptop screen has a crack",
            image_paths=["images/img_1.jpg", "images/img_2.jpg"], dataset_dir=self.dataset,
            findings=[finding(), finding("img_2", visible_issue="none",
                                        visual_summary="No visible damage on the screen.")],
            rules=self.rules,
        )
        self.assertEqual(result.claim_status, "not_enough_information")
        self.assertIn("manual_review_required", result.risk_flags)
        self.assertIn("conflicting_clear_images", result.decision_notes)

    def test_history_risk_cannot_flip_a_contradicted_claim(self):
        result = self.decide(
            "The laptop screen has a crack",
            [finding(visible_issue="none", visual_summary="No visible damage.")],
            history=["user_history_risk", "manual_review_required"],
        )
        self.assertEqual(result.claim_status, "contradicted")
        self.assertIn("user_history_risk", result.risk_flags)

    def test_custom_confidence_threshold_is_respected(self):
        low_confidence = finding(confidence=0.68)
        default = self.decide("The laptop screen has a crack", [low_confidence])
        strict = decide_claim(
            claim_object="laptop", user_claim="The laptop screen has a crack",
            image_paths=["images/img_1.jpg"], dataset_dir=self.dataset,
            findings=[low_confidence], rules=self.rules, confidence_threshold=0.65,
        )
        self.assertEqual(default.claim_status, "not_enough_information")
        self.assertEqual(strict.claim_status, "supported")

    def test_different_part_does_not_falsely_contradict_claim(self):
        # Inspecting an undamaged door does not prove or disprove a claimed crack on the screen/headlight.
        door_finding = finding(observed_part="keyboard", visible_issue="none",
                               visual_summary="Keyboard area shows no damage.")
        result = self.decide("The laptop screen has a crack", [door_finding])
        self.assertEqual(result.claim_status, "not_enough_information")
        self.assertFalse(result.evidence_standard_met)
        self.assertIn("wrong_object_part", result.risk_flags)

    def test_multi_image_with_other_part_does_not_cancel_valid_damage(self):
        # A valid crack on the screen should not be cancelled by an additional context view of the keyboard.
        screen_crack = finding(observed_part="screen", visible_issue="crack")
        keyboard_clean = finding("img_2", observed_part="keyboard", visible_issue="none",
                                 visual_summary="Keyboard has no damage.")
        result = decide_claim(
            claim_object="laptop", user_claim="The laptop screen has a crack",
            image_paths=["images/img_1.jpg", "images/img_2.jpg"], dataset_dir=self.dataset,
            findings=[screen_crack, keyboard_clean], rules=self.rules,
        )
        self.assertEqual(result.claim_status, "supported")
        self.assertEqual(result.supporting_image_ids, ("img_1",))
        self.assertTrue(result.evidence_standard_met)

    def test_deterministic_quality_flags_are_mapped_and_reported(self):
        result = decide_claim(
            claim_object="laptop", user_claim="The laptop screen has a crack",
            image_paths=["images/img_1.jpg"], dataset_dir=self.dataset,
            findings=[finding()], rules=self.rules,
            quality_flags=["possible_blur_or_low_detail", "low_light"],
        )
        self.assertIn("blurry_image", result.risk_flags)
        self.assertIn("low_light_or_glare", result.risk_flags)


if __name__ == "__main__":
    unittest.main()
