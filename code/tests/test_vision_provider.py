"""Unit tests for vision schema validation, retries, and offline provider."""
import random
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from code.config import AppConfig
from code.image_utils import normalize_dataset_image
from code.vision_provider import (
    MockVisionProvider, VisionProviderError, analyze_with_retries, validate_finding,
)


def valid_payload(**overrides):
    payload = {
        "observed_object": "laptop", "observed_part": "screen",
        "visible_issue": "crack", "damage_visibility": "clearly_visible",
        "relevance_confirmed": True, "quality_adequate": True,
        "confidence": 0.91, "risk_flags": [], "visual_summary": "A crack is visible on the screen.",
    }
    payload.update(overrides)
    return payload


class VisionProviderTests(unittest.TestCase):
    def test_valid_finding(self):
        result = validate_finding(valid_payload(), expected_object="laptop", image_id="img_01")
        self.assertEqual(result.image_id, "img_01")
        self.assertTrue(result.relevance_confirmed)
        self.assertEqual(result.visible_issue, "crack")

    def test_mismatched_object_cannot_be_confirmed_relevant(self):
        result = validate_finding(valid_payload(observed_object="car", observed_part="hood"),
                                  expected_object="laptop", image_id="img_02")
        self.assertFalse(result.relevance_confirmed)
        self.assertIn("wrong_object", result.risk_flags)

    def test_invalid_part_for_object_is_rejected(self):
        with self.assertRaises(VisionProviderError):
            validate_finding(valid_payload(observed_part="windshield"), expected_object="laptop", image_id="x")

    def test_rejects_invalid_confidence_and_extra_risk_flag(self):
        with self.assertRaises(VisionProviderError):
            validate_finding(valid_payload(confidence=1.5), expected_object="laptop", image_id="x")
        with self.assertRaises(VisionProviderError):
            validate_finding(valid_payload(risk_flags=["fraud_confirmed"]), expected_object="laptop", image_id="x")

    def test_json_string_is_supported(self):
        import json
        finding = validate_finding(json.dumps(valid_payload()), expected_object="laptop", image_id="img")
        self.assertAlmostEqual(finding.confidence, 0.91)

    def test_retry_then_success(self):
        class Flaky:
            def __init__(self):
                self.calls = 0
            def analyze(self, image, *, claim_object, user_claim):
                self.calls += 1
                if self.calls == 1:
                    raise VisionProviderError("temporary")
                return valid_payload()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "img.png"
            Image.new("RGB", (100, 100), "white").save(path)
            image = normalize_dataset_image(root, "img.png")
            provider = Flaky()
            result = analyze_with_retries(provider, image, claim_object="laptop", user_claim="cracked screen",
                                          max_retries=2, base_delay_seconds=0, sleep=lambda _: None,
                                          rng=random.Random(1))
            self.assertTrue(result.semantic_assessment_complete)
            self.assertEqual(provider.calls, 2)

    def test_mock_provider_does_not_fabricate_damage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (100, 100), "white").save(root / "img.png")
            image = normalize_dataset_image(root, "img.png")
            finding = analyze_with_retries(MockVisionProvider(), image, claim_object="laptop",
                                           user_claim="cracked screen", max_retries=0)
            self.assertEqual(finding.visible_issue, "unknown")
            self.assertFalse(finding.relevance_confirmed)
            self.assertEqual(finding.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
