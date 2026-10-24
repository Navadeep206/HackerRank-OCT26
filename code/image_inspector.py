"""Per-image inspection orchestration and structured observations.

The current phase computes reproducible technical quality metrics. Semantic
fields deliberately remain unknown until a vision provider or human review
fills them in; filename and claim text are never treated as visual evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from code.config import AppConfig
from code.image_quality import ImageQuality, QualityThresholds, assess_image_quality
from code.image_utils import ImageLoadError, NormalizedImage, normalize_dataset_image


@dataclass(frozen=True, slots=True)
class ImageObservation:
    image_id: str
    image_path: str
    load_ok: bool
    source_format: str | None
    sha256: str | None
    quality: ImageQuality | None
    observed_object: str
    observed_part: str
    visible_issue: str
    damage_visibility: str
    relevance_confirmed: bool
    semantic_assessment_complete: bool
    risk_flags: tuple[str, ...]
    error_code: str | None
    error_message: str | None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.quality is not None:
            result["quality"] = self.quality.to_dict()
        result["risk_flags"] = list(self.risk_flags)
        return result


def inspect_one_image(
    dataset_dir: Path,
    relative_path: str,
    config: AppConfig,
    *,
    thresholds: QualityThresholds | None = None,
) -> ImageObservation:
    """Normalize and assess one image; convert per-image errors to observations."""
    image_id = Path(relative_path.strip()).stem if relative_path.strip() else "unknown"
    try:
        normalized: NormalizedImage = normalize_dataset_image(
            dataset_dir,
            relative_path,
            max_bytes=config.max_image_bytes,
            max_pixels=config.max_image_pixels,
            max_dimension=config.max_image_dimension,
        )
        quality = assess_image_quality(normalized, thresholds)
        flags = list(quality.quality_flags)
        if not quality.usable_for_review:
            flags.append("image_unusable")
        return ImageObservation(
            image_id=normalized.image_id,
            image_path=relative_path,
            load_ok=True,
            source_format=normalized.source_format,
            sha256=normalized.sha256,
            quality=quality,
            observed_object="unknown",
            observed_part="unknown",
            visible_issue="unknown",
            damage_visibility="not_assessed",
            relevance_confirmed=False,
            semantic_assessment_complete=False,
            risk_flags=tuple(dict.fromkeys(flags)),
            error_code=None,
            error_message=None,
        )
    except ImageLoadError as exc:
        message = str(exc)
        code = "unsafe_or_missing_path" if any(
            word in message.lower() for word in ("escapes dataset", "does not exist", "empty")
        ) else "image_decode_or_limit_error"
        return ImageObservation(
            image_id=image_id, image_path=relative_path, load_ok=False,
            source_format=None, sha256=None, quality=None,
            observed_object="unknown", observed_part="unknown",
            visible_issue="unknown", damage_visibility="not_assessed",
            relevance_confirmed=False, semantic_assessment_complete=False,
            risk_flags=("image_unusable",), error_code=code,
            error_message=message,
        )


def inspect_image_paths(
    dataset_dir: Path,
    image_paths: Iterable[str],
    config: AppConfig,
    *,
    thresholds: QualityThresholds | None = None,
) -> list[ImageObservation]:
    """Inspect each unique non-empty path in input order, without aborting a claim."""
    observations: list[ImageObservation] = []
    seen: set[str] = set()
    for path in image_paths:
        clean = path.strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        observations.append(
            inspect_one_image(dataset_dir, clean, config, thresholds=thresholds)
        )
    return observations
