"""Deterministic image-quality checks for claim evidence.

These checks measure technical usability only. They cannot establish that an
image depicts the claimed object or that damage is present; those judgments
belong to a vision model or human reviewer.
"""
from __future__ import annotations

import io
from dataclasses import asdict, dataclass
from typing import Any

from PIL import Image, ImageFilter, ImageStat, UnidentifiedImageError

from code.image_utils import NormalizedImage


@dataclass(frozen=True, slots=True)
class QualityThresholds:
    min_width: int = 256
    min_height: int = 192
    min_brightness: float = 28.0
    max_brightness: float = 238.0
    min_contrast: float = 12.0
    min_edge_variance: float = 2.0
    max_clipped_fraction: float = 0.35


@dataclass(frozen=True, slots=True)
class ImageQuality:
    image_id: str
    source_path: str
    width: int
    height: int
    brightness_mean: float
    grayscale_contrast: float
    edge_variance: float
    clipped_fraction: float
    quality_flags: tuple[str, ...]
    usable_for_review: bool
    quality_summary: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["quality_flags"] = list(self.quality_flags)
        return result


def _edge_variance(image: Image.Image) -> float:
    """Return a cheap edge-energy proxy; not a calibrated blur classifier."""
    edges = image.convert("L").filter(ImageFilter.FIND_EDGES)
    return float(ImageStat.Stat(edges).var[0])


def _clipped_fraction(image: Image.Image) -> float:
    """Estimate over/under-exposed pixel share from a downsampled RGB image."""
    sample = image.convert("RGB").resize((min(128, image.width), min(128, image.height)))
    pixels = list(sample.getdata())
    if not pixels:
        return 1.0
    clipped = sum(
        1 for pixel in pixels
        if all(channel >= 250 for channel in pixel)
        or all(channel <= 5 for channel in pixel)
    )
    return clipped / len(pixels)


def assess_image_quality(
    normalized: NormalizedImage,
    thresholds: QualityThresholds | None = None,
) -> ImageQuality:
    """Assess resolution, brightness, contrast, edge detail, and clipping."""
    limits = thresholds or QualityThresholds()
    if min(limits.min_width, limits.min_height) <= 0:
        raise ValueError("Minimum image dimensions must be positive")
    if not 0 <= limits.min_brightness < limits.max_brightness <= 255:
        raise ValueError("Brightness thresholds must satisfy 0 <= min < max <= 255")
    if limits.min_contrast < 0 or limits.min_edge_variance < 0:
        raise ValueError("Contrast and edge thresholds cannot be negative")
    if not 0 <= limits.max_clipped_fraction <= 1:
        raise ValueError("max_clipped_fraction must be between 0 and 1")

    try:
        with Image.open(io.BytesIO(normalized.jpeg_bytes)) as opened:
            opened.load()
            image = opened.convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("Normalized image bytes could not be decoded") from exc

    gray_stats = ImageStat.Stat(image.convert("L"))
    brightness = float(gray_stats.mean[0])
    contrast = float(gray_stats.stddev[0])
    edge_variance = _edge_variance(image)
    clipped = _clipped_fraction(image)
    flags: list[str] = []

    if normalized.width < limits.min_width or normalized.height < limits.min_height:
        flags.append("low_resolution")
    if brightness < limits.min_brightness:
        flags.append("low_light")
    elif brightness > limits.max_brightness:
        flags.append("possible_overexposure")
    if contrast < limits.min_contrast:
        flags.append("low_contrast")
    if edge_variance < limits.min_edge_variance:
        flags.append("possible_blur_or_low_detail")
    if clipped > limits.max_clipped_fraction:
        flags.append("highlight_or_shadow_clipping")

    # Hard failures are limited to tiny images and extreme exposure. Blur-like
    # and contrast flags should trigger caution, not automatic rejection.
    hard_failure = (
        normalized.width < 64 or normalized.height < 64
        or brightness < 8 or brightness > 250
    )
    usable = not hard_failure
    summary = (
        "Technical quality appears usable, but relevance and damage visibility "
        "must be assessed separately."
        if usable and not flags else
        "Quality warning(s): " + ", ".join(flags) + ". "
        + ("Image is technically unusable under hard limits." if hard_failure
           else "Review may still be possible; inspect the image and claim context.")
    )
    return ImageQuality(
        image_id=normalized.image_id,
        source_path=str(normalized.source_path),
        width=normalized.width,
        height=normalized.height,
        brightness_mean=round(brightness, 2),
        grayscale_contrast=round(contrast, 2),
        edge_variance=round(edge_variance, 2),
        clipped_fraction=round(clipped, 4),
        quality_flags=tuple(flags),
        usable_for_review=usable,
        quality_summary=summary,
    )
