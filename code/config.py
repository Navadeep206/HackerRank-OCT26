"""Validated runtime configuration for the evidence-review pipeline."""
from __future__ import annotations
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

Provider = Literal["gemini", "openai"]

class ConfigurationError(ValueError):
    """Raised when environment configuration is invalid."""

def _number(name: str, default: str, cast: type) -> int | float:
    raw = os.getenv(name, default).strip()
    try:
        value = cast(raw)
    except (ValueError, OverflowError) as exc:
        raise ConfigurationError(f"{name} must be a valid {cast.__name__}") from exc
    if isinstance(value, float) and not math.isfinite(value):
        raise ConfigurationError(f"{name} must be finite")
    if value <= 0:
        raise ConfigurationError(f"{name} must be greater than zero")
    if cast is int and str(value) != raw:
        raise ConfigurationError(f"{name} must be a whole number")
    return value

@dataclass(frozen=True, slots=True)
class AppConfig:
    repo_root: Path
    dataset_dir: Path
    output_csv: Path
    provider: Provider
    model_name: str
    request_timeout_seconds: float
    max_retries: int
    max_image_bytes: int
    max_image_pixels: int
    max_image_dimension: int
    request_rate_per_minute: int
    cache_enabled: bool

    @classmethod
    def from_env(cls, repo_root: Path | None = None) -> "AppConfig":
        root = (repo_root or Path(__file__).resolve().parents[1]).resolve()
        if load_dotenv is not None:
            load_dotenv(root / ".env", override=False)
        provider_raw = os.getenv("MODEL_PROVIDER", "gemini").strip().lower()
        if provider_raw not in {"gemini", "openai"}:
            raise ConfigurationError(f"MODEL_PROVIDER must be gemini or openai (received: {provider_raw!r})")
        provider: Provider = provider_raw  # type: ignore[assignment]
        defaults = {"gemini": "gemini-2.5-flash", "openai": "gpt-4.1-mini"}
        model = os.getenv("MODEL_NAME", defaults[provider]).strip()
        if not model:
            raise ConfigurationError("MODEL_NAME cannot be empty")
        cache_raw = os.getenv("MODEL_CACHE_ENABLED", "true").strip().lower()
        if cache_raw not in {"true", "false", "1", "0", "yes", "no"}:
            raise ConfigurationError("MODEL_CACHE_ENABLED must be true/false, 1/0, or yes/no")
        dataset = (root / os.getenv("DATASET_DIR", "dataset")).resolve()
        output = (root / os.getenv("OUTPUT_CSV", "output.csv")).resolve()
        timeout = float(_number("MODEL_TIMEOUT_SECONDS", "60", float))
        retries = int(_number("MODEL_MAX_RETRIES", "3", int))
        image_bytes = int(_number("MAX_IMAGE_BYTES", str(20 * 1024 * 1024), int))
        image_pixels = int(_number("MAX_IMAGE_PIXELS", "40000000", int))
        image_dimension = int(_number("MAX_IMAGE_DIMENSION", "2048", int))
        requests_per_minute = int(_number("MODEL_REQUESTS_PER_MINUTE", "10", int))
        bounds = (
            (timeout, 600, "MODEL_TIMEOUT_SECONDS"),
            (retries, 10, "MODEL_MAX_RETRIES"),
            (requests_per_minute, 600, "MODEL_REQUESTS_PER_MINUTE"),
            (image_bytes, 200 * 1024 * 1024, "MAX_IMAGE_BYTES"),
            (image_pixels, 200_000_000, "MAX_IMAGE_PIXELS"),
            (image_dimension, 8192, "MAX_IMAGE_DIMENSION"),
        )
        for value, maximum, name in bounds:
            if value > maximum:
                raise ConfigurationError(f"{name} must not exceed {maximum}")
        return cls(
            repo_root=root, dataset_dir=dataset, output_csv=output, provider=provider,
            model_name=model, request_timeout_seconds=timeout, max_retries=retries,
            max_image_bytes=image_bytes, max_image_pixels=image_pixels,
            max_image_dimension=image_dimension, request_rate_per_minute=requests_per_minute,
            cache_enabled=cache_raw in {"true", "1", "yes"},
        )

    def validate_credentials(self) -> None:
        """Check credentials for a live provider without revealing their values."""
        key_name = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY"}.get(self.provider)
        if key_name and not os.getenv(key_name, "").strip():
            raise ConfigurationError(
                f"{self.provider} selected but {key_name} is not set. "
                "Set it in the environment or local .env file; never commit secrets."
            )

def load_config(repo_root: Path | None = None) -> AppConfig:
    return AppConfig.from_env(repo_root)
