"""Unit tests for vision schema validation, retries, and offline provider."""
import json
import os
import random
import tempfile
import unittest
from pathlib import Path

from unittest.mock import MagicMock, patch
from PIL import Image

from code.config import AppConfig, ConfigurationError
from code.image_utils import normalize_dataset_image
from code.vision_provider import (
    GeminiVisionProvider, VisionProviderError, analyze_with_retries,
    build_vision_provider, validate_finding,
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
            attempts = []
            result = analyze_with_retries(provider, image, claim_object="laptop", user_claim="cracked screen",
                                          max_retries=2, base_delay_seconds=0, sleep=lambda _: None,
                                          rng=random.Random(1), on_attempt=attempts.append)
            self.assertTrue(result.semantic_assessment_complete)
            self.assertEqual(provider.calls, 2)
            self.assertEqual(attempts, [1, 2])

    def test_build_vision_provider_requires_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = AppConfig(
                root, root / "dataset", root / "output.csv", "gemini",
                "gemini-2.5-flash", 60.0, 3, 20*1024*1024, 40_000_000, 2048, 10, False,
            )
            with patch.dict("os.environ", {}, clear=True):
                with self.assertRaises(ConfigurationError):
                    build_vision_provider(config)

    def test_gemini_client_stub_parses_response(self):
        # SDK stub validates client invocation without calling external network APIs
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = AppConfig(
                root, root / "dataset", root / "output.csv", "gemini",
                "gemini-2.5-flash", 60.0, 3, 20*1024*1024, 40_000_000, 2048, 10, False,
            )
            Image.new("RGB", (100, 100), "white").save(root / "img.png")
            image = normalize_dataset_image(root, "img.png")

            with patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key-for-stub"}):
                provider = GeminiVisionProvider(config)
                stubbed_response = MagicMock()
                stubbed_response.text = json.dumps(valid_payload())
                provider._client.models.generate_content = MagicMock(return_value=stubbed_response)

                payload = provider.analyze(image, claim_object="laptop", user_claim="crack on screen")
                finding = validate_finding(payload, expected_object="laptop", image_id="img")
                self.assertEqual(finding.visible_issue, "crack")
                self.assertEqual(finding.observed_part, "screen")
                self.assertTrue(finding.relevance_confirmed)

    @unittest.skipUnless(os.getenv("GEMINI_API_KEY"), "Opt-in live Gemini test requires GEMINI_API_KEY in environment")
    def test_opt_in_live_gemini_call(self):
        # Only runs when credentials are intentionally provided by the user
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (128, 128), "white").save(root / "img.png")
            image = normalize_dataset_image(root, "img.png")
            config = AppConfig.from_env(root)
            provider = build_vision_provider(config)
            finding = analyze_with_retries(provider, image, claim_object="laptop", user_claim="cracked screen", max_retries=1)
            self.assertTrue(finding.semantic_assessment_complete)

    def test_rejects_non_finite_confidence(self):
        import math
        with self.assertRaises(VisionProviderError):
            validate_finding(valid_payload(confidence=float("nan")), expected_object="laptop", image_id="x")
        with self.assertRaises(VisionProviderError):
            validate_finding(valid_payload(confidence=float("inf")), expected_object="laptop", image_id="x")

    def test_redacts_credentials_in_summary(self):
        finding = validate_finding(
            valid_payload(visual_summary="Found API key sk-abcdef12345678901234567890 in photo"),
            expected_object="laptop",
            image_id="img",
        )
        self.assertNotIn("sk-abcdef", finding.visual_summary)
        self.assertIn("[REDACTED]", finding.visual_summary)


if __name__ == "__main__":
    unittest.main()
