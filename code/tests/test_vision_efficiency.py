"""Tests for provider caching and RPM pacing; all calls use a fake provider."""
import tempfile
import unittest
from pathlib import Path
from PIL import Image

from code.image_utils import normalize_dataset_image
from code.vision_efficiency import EfficientVisionProvider


class CountingProvider:
    def __init__(self):
        self.calls = 0
    def analyze(self, image, *, claim_object, user_claim):
        self.calls += 1
        return {
            "observed_object": "laptop", "observed_part": "screen",
            "visible_issue": "crack", "damage_visibility": "clearly_visible",
            "relevance_confirmed": True, "quality_adequate": True,
            "confidence": 0.9, "risk_flags": [], "visual_summary": "Visible crack.",
        }


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []
    def clock(self):
        return self.now
    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class VisionEfficiencyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        Image.new("RGB", (120, 120), "white").save(self.root / "img.png")
        self.image = normalize_dataset_image(self.root, "img.png")

    def tearDown(self):
        self.tmp.cleanup()

    def test_cache_avoids_duplicate_provider_call(self):
        base = CountingProvider()
        wrapped = EfficientVisionProvider(
            base, cache_path=self.root / "cache.sqlite3", cache_enabled=True,
            requests_per_minute=60, provider_name="fake", model_name="test-v1",
            sleep=lambda _: None,
        )
        first = wrapped.analyze(self.image, claim_object="laptop", user_claim="cracked screen")
        second = wrapped.analyze(self.image, claim_object="laptop", user_claim="cracked screen")
        self.assertEqual(first, second)
        self.assertEqual(base.calls, 1)
        self.assertEqual(wrapped.provider_requests, 1)
        self.assertEqual(wrapped.cache_hits, 1)

    def test_cache_key_includes_claim_and_model(self):
        base = CountingProvider()
        wrapped = EfficientVisionProvider(
            base, cache_path=self.root / "cache.sqlite3", cache_enabled=True,
            requests_per_minute=600, provider_name="fake", model_name="test-v1",
            sleep=lambda _: None,
        )
        wrapped.analyze(self.image, claim_object="laptop", user_claim="cracked screen")
        wrapped.analyze(self.image, claim_object="laptop", user_claim="broken hinge")
        self.assertEqual(base.calls, 2)
        self.assertEqual(wrapped.cache_misses, 2)

    def test_invalid_response_is_not_cached(self):
        class Flaky:
            def __init__(self):
                self.calls = 0
            def analyze(self, image, *, claim_object, user_claim):
                self.calls += 1
                if self.calls == 1:
                    return {"malformed": True}
                return {
                    "observed_object": "laptop", "observed_part": "screen",
                    "visible_issue": "crack", "damage_visibility": "clearly_visible",
                    "relevance_confirmed": True, "quality_adequate": True,
                    "confidence": 0.9, "risk_flags": [], "visual_summary": "Visible crack.",
                }
        base = Flaky()
        wrapped = EfficientVisionProvider(
            base, cache_path=self.root / "cache.sqlite3", cache_enabled=True,
            requests_per_minute=600, provider_name="fake", model_name="test-v1",
            sleep=lambda _: None,
        )
        with self.assertRaises(Exception):
            wrapped.analyze(self.image, claim_object="laptop", user_claim="cracked screen")
        result = wrapped.analyze(self.image, claim_object="laptop", user_claim="cracked screen")
        self.assertEqual(result["visible_issue"], "crack")
        self.assertEqual(base.calls, 2)

    def test_rate_limit_waits_between_real_requests(self):
        base = CountingProvider()
        clock = FakeClock()
        wrapped = EfficientVisionProvider(
            base, cache_path=self.root / "unused.sqlite3", cache_enabled=False,
            requests_per_minute=30, provider_name="fake", model_name="test-v1",
            clock=clock.clock, sleep=clock.sleep,
        )
        wrapped.analyze(self.image, claim_object="laptop", user_claim="first")
        wrapped.analyze(self.image, claim_object="laptop", user_claim="second")
        self.assertEqual(base.calls, 2)
        self.assertEqual(clock.sleeps, [2.0])


if __name__ == "__main__":
    unittest.main()
