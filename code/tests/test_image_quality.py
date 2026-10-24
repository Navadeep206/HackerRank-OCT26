"""Tests for deterministic image quality metrics and inspection failure isolation."""
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from code.config import AppConfig
from code.image_inspector import inspect_image_paths, inspect_one_image
from code.image_quality import QualityThresholds, assess_image_quality
from code.image_utils import normalize_dataset_image


def test_config(root: Path) -> AppConfig:
    return AppConfig(
        repo_root=root, dataset_dir=root / "dataset", output_csv=root / "output.csv",
        provider="mock", model_name="mock", request_timeout_seconds=5.0,
        max_retries=1, max_image_bytes=1024 * 1024, max_image_pixels=100_000,
        max_image_dimension=512, request_rate_per_minute=10, cache_enabled=True,
    )


class ImageQualityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset = self.root / "dataset"
        (self.dataset / "images").mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def test_quality_metrics_are_finite_and_deterministic(self):
        path = self.dataset / "images" / "detail.jpg"
        image = Image.new("RGB", (320, 240), "gray")
        for x in range(40, 280, 20):
            for y in range(40, 200, 20):
                image.paste("black", (x, y, x + 8, y + 8))
        image.save(path)
        normalized = normalize_dataset_image(self.dataset, "images/detail.jpg")
        first = assess_image_quality(normalized)
        second = assess_image_quality(normalized)
        self.assertEqual(first, second)
        for value in (first.brightness_mean, first.grayscale_contrast,
                      first.edge_variance, first.clipped_fraction):
            self.assertTrue(value >= 0)
        self.assertLessEqual(first.brightness_mean, 255)
        self.assertLessEqual(first.grayscale_contrast, 255)
        self.assertLessEqual(first.clipped_fraction, 1)

    def test_uniform_tiny_image_is_flagged_but_not_crashed(self):
        path = self.dataset / "images" / "tiny.jpg"
        Image.new("RGB", (32, 32), (128, 128, 128)).save(path)
        normalized = normalize_dataset_image(self.dataset, "images/tiny.jpg")
        quality = assess_image_quality(normalized)
        self.assertIn("low_resolution", quality.quality_flags)
        self.assertFalse(quality.usable_for_review)

    def test_missing_and_unsafe_paths_become_observations(self):
        cfg = test_config(self.root)
        observations = inspect_image_paths(
            self.dataset,
            ["images/missing.jpg", "../outside.jpg", "images/missing.jpg"],
            cfg,
        )
        self.assertEqual(len(observations), 2)
        self.assertTrue(all(not item.load_ok for item in observations))
        self.assertTrue(all("image_unusable" in item.risk_flags for item in observations))

    def test_success_does_not_claim_semantic_inspection(self):
        path = self.dataset / "images" / "valid.jpg"
        Image.new("RGB", (320, 240), (80, 90, 100)).save(path)
        observation = inspect_one_image(self.dataset, "images/valid.jpg", test_config(self.root))
        self.assertTrue(observation.load_ok)
        self.assertFalse(observation.semantic_assessment_complete)
        self.assertFalse(observation.relevance_confirmed)
        self.assertEqual(observation.visible_issue, "unknown")

    def test_threshold_validation(self):
        path = self.dataset / "images" / "valid.jpg"
        Image.new("RGB", (320, 240), "gray").save(path)
        normalized = normalize_dataset_image(self.dataset, "images/valid.jpg")
        with self.assertRaises(ValueError):
            assess_image_quality(normalized, QualityThresholds(min_width=0))


if __name__ == "__main__":
    unittest.main()
