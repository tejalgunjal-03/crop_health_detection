"""Validate uploaded images by CONTENT (not extension) and store them safely."""
from __future__ import annotations
import io, uuid
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from ..core.errors import InvalidImageError

ALLOWED = {"JPEG": ".jpg", "PNG": ".png"}
MIN_SIDE = 64
Image.MAX_IMAGE_PIXELS = 120_000_000    # PIL raises on decompression bombs beyond 2x this


def validate_image_bytes(data: bytes, max_bytes: int, max_pixels: int | None = None) -> tuple[Image.Image, str]:
    """Return (RGB PIL image, extension). Raises InvalidImageError with a friendly message."""
    if not data:
        raise InvalidImageError("The uploaded file is empty.")
    if len(data) > max_bytes:
        raise InvalidImageError(f"File too large. Maximum size is {max_bytes // (1024 * 1024)} MB.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            probe.verify()                      # structural check
        img = Image.open(io.BytesIO(data))
        img.load()                              # full decode catches truncated files
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)      # fix EXIF rotation
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError, ValueError):
        raise InvalidImageError("This file is not a valid image. Please upload a JPG or PNG photo.")
    if fmt not in ALLOWED:
        raise InvalidImageError("Unsupported image type. Please upload a JPG or PNG image.")
    w, h = img.size
    if min(w, h) < MIN_SIDE:
        raise InvalidImageError(f"Image is too small (minimum {MIN_SIDE}x{MIN_SIDE} pixels).")
    if max_pixels and w * h > max_pixels:
        raise InvalidImageError(f"Image is too large ({w}x{h}). Maximum is {max_pixels / 1e6:.0f} megapixels.")
    return img.convert("RGB"), ALLOWED[fmt]


def save_upload(data: bytes, folder: Path, ext: str) -> Path:
    """Save raw bytes under a random server-generated filename (user filename is never used)."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{uuid.uuid4().hex}{ext}"
    path.write_bytes(data)
    return path
