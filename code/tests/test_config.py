"""Unit tests for validated configuration."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from code.config import AppConfig, ConfigurationError

class ConfigTests(unittest.TestCase):
    def test_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = AppConfig.from_env(Path(tempfile.gettempdir()))
        self.assertEqual(cfg.provider, "gemini")
        self.assertEqual(cfg.model_name, "gemini-2.5-flash")
        self.assertEqual(cfg.max_image_dimension, 2048)
        self.assertTrue(cfg.cache_enabled)

    def test_unknown_provider_rejected(self):
        with patch.dict(os.environ, {"MODEL_PROVIDER": "other"}, clear=True):
            with self.assertRaises(ConfigurationError):
                AppConfig.from_env(Path(tempfile.gettempdir()))

    def test_non_positive_timeout_rejected(self):
        with patch.dict(os.environ, {"MODEL_TIMEOUT_SECONDS": "0"}, clear=True):
            with self.assertRaises(ConfigurationError):
                AppConfig.from_env(Path(tempfile.gettempdir()))

    def test_non_finite_timeout_rejected(self):
        for raw in ("nan", "inf", "-inf"):
            with self.subTest(raw=raw), patch.dict(os.environ, {"MODEL_TIMEOUT_SECONDS": raw}, clear=True):
                with self.assertRaises(ConfigurationError):
                    AppConfig.from_env(Path(tempfile.gettempdir()))

    def test_integer_settings_reject_fractional_values(self):
        with patch.dict(os.environ, {"MODEL_MAX_RETRIES": "2.5"}, clear=True):
            with self.assertRaises(ConfigurationError):
                AppConfig.from_env(Path(tempfile.gettempdir()))

    def test_extreme_retry_count_rejected(self):
        with patch.dict(os.environ, {"MODEL_MAX_RETRIES": "1000"}, clear=True):
            with self.assertRaises(ConfigurationError):
                AppConfig.from_env(Path(tempfile.gettempdir()))

    def test_mock_provider_is_rejected(self):
        with patch.dict(os.environ, {"MODEL_PROVIDER": "mock"}, clear=True):
            with self.assertRaises(ConfigurationError):
                AppConfig.from_env(Path(tempfile.gettempdir()))

    def test_live_provider_requires_key(self):
        with patch.dict(os.environ, {"MODEL_PROVIDER": "gemini"}, clear=True):
            with self.assertRaises(ConfigurationError):
                AppConfig.from_env(Path(tempfile.gettempdir())).validate_credentials()

if __name__ == "__main__":
    unittest.main()
