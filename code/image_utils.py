"""Safe, format-aware image loading and normalization.

CSV image paths are relative to dataset/, and extensions are not trusted:
some assets have AVIF, WebP, or PNG bytes despite a .jpg suffix.
"""
from __future__ import annotations
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from PIL import Image, ImageOps, UnidentifiedImageError

try:
    from pillow_heif import register_heif_opener
except ImportError:
    register_heif_opener = None
if register_heif_opener is not None:
    register_heif_opener()

SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP", "GIF", "BMP", "TIFF", "AVIF", "HEIF", "HEIC"}

class ImageLoadError(ValueError):
    """An image is missing, unsafe, unsupported, or cannot be decoded."""

@dataclass(frozen=True, slots=True)
class NormalizedImage:
    source_path: Path
    source_format: str
    mime_type: str
    width: int
    height: int
    sha256: str
    jpeg_bytes: bytes

    @property
    def image_id(self) -> str:
        return self.source_path.stem

def resolve_image_path(dataset_dir: Path, relative_path: str) -> Path:
    """Resolve a dataset-relative path and reject path traversal."""
    raw = relative_path.strip()
    if not raw:
        raise ImageLoadError("Image path is empty")
    root = dataset_dir.resolve()
    candidate = (root / raw).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ImageLoadError(f"Image path escapes dataset directory: {raw!r}") from exc
    if not candidate.is_file():
        raise ImageLoadError(f"Image file does not exist: {raw!r}")
    return candidate

def _read_limited(path: Path, max_bytes: int) -> bytes:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ImageLoadError(f"Cannot inspect image file: {path.name}") from exc
    if size <= 0:
        raise ImageLoadError(f"Image file is empty: {path.name}")
    if size > max_bytes:
        raise ImageLoadError(f"Image is {size} bytes; limit is {max_bytes}")
    try:
        with path.open("rb") as handle:
            return handle.read(max_bytes + 1)
    except OSError as exc:
        raise ImageLoadError(f"Cannot read image file: {path.name}") from exc

def normalize_image(
    path: Path, *, max_bytes: int = 20 * 1024 * 1024,
    max_pixels: int = 40_000_000, max_dimension: int = 2048,
    jpeg_quality: int = 90,
) -> NormalizedImage:
    """Decode from bytes, apply EXIF orientation, and encode a bounded JPEG.

    The original asset is never modified. Errors use ImageLoadError so callers
    can flag one bad image and continue processing the remaining claim.
    """
    if min(max_bytes, max_pixels, max_dimension) <= 0:
        raise ValueError("Image limits must be positive")
    if not 1 <= jpeg_quality <= 95:
        raise ValueError("jpeg_quality must be between 1 and 95")
    raw = _read_limited(path, max_bytes)
    try:
        with Image.open(io.BytesIO(raw)) as opened:
            detected = (opened.format or "").upper()
            if detected not in SUPPORTED_FORMATS:
                raise ImageLoadError(f"Unsupported format {detected or 'unknown'}: {path.name}")
            width, height = opened.size
            if width <= 0 or height <= 0:
                raise ImageLoadError(f"Invalid image dimensions: {path.name}")
            if width * height > max_pixels:
                raise ImageLoadError(f"Image pixel count exceeds limit ({max_pixels})")
            opened.load()
            image = ImageOps.exif_transpose(opened)
            if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
                rgba = image.convert("RGBA")
                background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                image = Image.alpha_composite(background, rgba).convert("RGB")
            else:
                image = image.convert("RGB")
            image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=jpeg_quality, optimize=True)
            return NormalizedImage(
                source_path=path, source_format=detected, mime_type="image/jpeg",
                width=image.width, height=image.height,
                sha256=hashlib.sha256(raw).hexdigest(), jpeg_bytes=output.getvalue(),
            )
    except ImageLoadError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise ImageLoadError(f"Unable to decode image {path.name}: {exc}") from exc

def normalize_dataset_image(
    dataset_dir: Path, relative_path: str, *,
    max_bytes: int = 20 * 1024 * 1024, max_pixels: int = 40_000_000,
    max_dimension: int = 2048,
) -> NormalizedImage:
    path = resolve_image_path(dataset_dir, relative_path)
    return normalize_image(
        path, max_bytes=max_bytes, max_pixels=max_pixels,
        max_dimension=max_dimension,
    )
