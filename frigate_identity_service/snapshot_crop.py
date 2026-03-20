"""Pure image-cropping helpers for ReID snapshot preparation.

These functions are intentionally free of MQTT/network dependencies so they
can be imported and unit-tested without triggering the service's module-level
side effects (broker connection, config validation, etc.).

All crop geometry is expressed in normalised [0, 1] coordinates matching
the relative coordinate system used in Frigate event payloads.
"""

from __future__ import annotations

import logging
from io import BytesIO

from PIL import Image

_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Target aspect ratio
# ---------------------------------------------------------------------------
# OSNet (and most Market-1501 / DukeMTMC-trained models) are designed for
# 256x128 (h:w = 2:1) inputs.  Producing crops at this ratio before handing
# off to torchreid avoids the distortion that occurs when the model's internal
# resize squashes an arbitrary-shaped crop.
REID_TARGET_RATIO: float = 2.0  # height / width

# ---------------------------------------------------------------------------
# Letterbox fill colour
# ---------------------------------------------------------------------------
# When we cannot pull more context from the original frame we pad missing
# pixels with the ImageNet mean colour so model normalisation is minimally
# disrupted.  Values are 8-bit RGB: (0.485 * 255, 0.456 * 255, 0.406 * 255).
IMAGENET_MEAN_RGB: tuple[int, int, int] = (124, 116, 104)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def build_local_crop_rect(
    crop_geometry: dict | None,
    padding_x: float = 0.05,
    padding_y: float = 0.20,
) -> tuple[float, float, float, float] | None:
    """Return a normalised (left, top, right, bottom) crop rectangle.

    Applies asymmetric padding (more vertical for head/feet context) and
    then expands the padded region to the 2:1 aspect ratio expected by
    ReID models.  The returned rectangle may exceed [0, 1] on any axis
    when the frame does not have enough room; callers should letterbox-pad
    in that case (see :func:`crop_snapshot_bytes`).

    Args:
        crop_geometry: Dict with optional ``"box"`` and ``"region"`` keys,
            each a normalised ``(x, y, width, height)`` tuple.  The first
            non-``None`` key is used.
        padding_x: Fractional horizontal padding relative to bbox width.
        padding_y: Fractional vertical padding relative to bbox height.

    Returns:
        ``(left, top, right, bottom)`` in normalised coordinates, or
        ``None`` if no usable geometry was found.
    """
    if not crop_geometry:
        return None

    rect = crop_geometry.get("box") or crop_geometry.get("region")
    if not rect:
        return None

    x_pos, y_pos, width, height = rect

    # Asymmetric padding: generous vertical context (head/feet),
    # minimal horizontal (persons are tall, not wide).
    left = x_pos - width * padding_x
    top = y_pos - height * padding_y
    right = x_pos + width + width * padding_x
    bottom = y_pos + height + height * padding_y

    # Expand to 2:1 (h:w) — prefer expanding the narrower dimension so we
    # pull as much real-pixel context as possible from the source frame.
    padded_w = right - left
    padded_h = bottom - top
    if padded_w > 0 and padded_h > 0:
        current_ratio = padded_h / padded_w
        if current_ratio < REID_TARGET_RATIO:
            # Too wide — expand height symmetrically
            needed_h = padded_w * REID_TARGET_RATIO
            extra = (needed_h - padded_h) / 2.0
            top -= extra
            bottom += extra
        elif current_ratio > REID_TARGET_RATIO:
            # Too tall — expand width symmetrically
            needed_w = padded_h / REID_TARGET_RATIO
            extra = (needed_w - padded_w) / 2.0
            left -= extra
            right += extra

    if right <= left or bottom <= top:
        return None

    return (left, top, right, bottom)


def letterbox_to_ratio(
    image: Image.Image,
    target_ratio: float = REID_TARGET_RATIO,
    fill_colour: tuple[int, int, int] = IMAGENET_MEAN_RGB,
) -> Image.Image:
    """Pad *image* to *target_ratio* (h/w) using *fill_colour*.

    Used when a crop region extends beyond the frame boundary and we cannot
    pull more real pixels from the source.

    Args:
        image: PIL RGB image to pad.
        target_ratio: Target height-to-width ratio (default 2.0).
        fill_colour: RGB fill tuple for padding regions.

    Returns:
        Padded PIL image with the requested aspect ratio.
    """
    w, h = image.size
    if w == 0 or h == 0:
        return image
    current_ratio = h / w
    if abs(current_ratio - target_ratio) < 0.01:
        return image
    if current_ratio < target_ratio:
        # Pad top and bottom
        new_h = int(round(w * target_ratio))
        pad = new_h - h
        pad_top = pad // 2
        padded = Image.new("RGB", (w, new_h), fill_colour)
        padded.paste(image, (0, pad_top))
        return padded
    else:
        # Pad left and right
        new_w = int(round(h / target_ratio))
        pad = new_w - w
        pad_left = pad // 2
        padded = Image.new("RGB", (new_w, h), fill_colour)
        padded.paste(image, (pad_left, 0))
        return padded


def crop_snapshot_bytes(
    image_bytes: bytes,
    crop_geometry: dict | None,
    quality: int = 85,
    padding_x: float = 0.05,
    padding_y: float = 0.20,
) -> bytes | None:
    """Crop a Frigate snapshot to a ReID-friendly 2:1 region and return JPEG bytes.

    The crop rect is expanded to the 2:1 aspect ratio expected by OSNet and
    similar models.  When the expanded region extends beyond the frame boundary
    the missing area is letterbox-padded with the ImageNet mean colour rather
    than distorting existing pixels.

    No intermediate resize is applied — the ReID model handles its own input
    scaling, avoiding a double-interpolation quality loss.

    Args:
        image_bytes: Raw image bytes (any PIL-readable format, e.g. JPEG/WebP).
        crop_geometry: Bounding box dict from
            :func:`identity_service._extract_snapshot_crop_geometry`.
        quality: JPEG output quality (1-95).
        padding_x: Horizontal padding fraction (relative to bbox width).
        padding_y: Vertical padding fraction (relative to bbox height).

    Returns:
        JPEG bytes of the cropped region, or ``None`` on failure.
    """
    crop_rect = build_local_crop_rect(crop_geometry, padding_x=padding_x, padding_y=padding_y)
    if not crop_rect:
        return None

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image = image.convert("RGB")
            iw, ih = image.size
            left, top, right, bottom = crop_rect

            # Clamp to valid pixel range
            px_left = max(0, int(left * iw))
            px_top = max(0, int(top * ih))
            px_right = min(iw, int(right * iw))
            px_bottom = min(ih, int(bottom * ih))

            if px_right <= px_left or px_bottom <= px_top:
                return None

            cropped = image.crop((px_left, px_top, px_right, px_bottom))

            # When the desired rect extended outside the frame we lost context.
            # Letterbox-pad to restore the 2:1 ratio so the ReID model receives
            # a geometrically correct crop image.
            if left < 0.0 or top < 0.0 or right > 1.0 or bottom > 1.0:
                cropped = letterbox_to_ratio(cropped)

            output = BytesIO()
            cropped.save(output, format="JPEG", quality=quality)
            return output.getvalue()
    except Exception as exc:
        _LOGGER.warning("Local snapshot crop failed: %s", exc)
        return None


def crop_snapshot_pil(
    image_bytes: bytes,
    crop_geometry: dict | None,
    padding_x: float = 0.05,
    padding_y: float = 0.20,
) -> Image.Image | None:
    """Crop a Frigate snapshot and return a PIL Image — *no* JPEG encode.

    Identical to :func:`crop_snapshot_bytes` but stops before the lossy JPEG
    encode step.  Use this for the ReID embedding path to avoid the DCT
    block artifacts introduced by a JPEG encode/decode round-trip.

    Args:
        image_bytes: Raw image bytes (any PIL-readable format, e.g. JPEG/WebP).
        crop_geometry: Bounding box dict from
            :func:`identity_service._extract_snapshot_crop_geometry`.
        padding_x: Horizontal padding fraction (relative to bbox width).
        padding_y: Vertical padding fraction (relative to bbox height).

    Returns:
        PIL RGB Image of the cropped region, or ``None`` on failure.
    """
    crop_rect = build_local_crop_rect(crop_geometry, padding_x=padding_x, padding_y=padding_y)
    if not crop_rect:
        return None

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image = image.convert("RGB")
            iw, ih = image.size
            left, top, right, bottom = crop_rect

            px_left = max(0, int(left * iw))
            px_top = max(0, int(top * ih))
            px_right = min(iw, int(right * iw))
            px_bottom = min(ih, int(bottom * ih))

            if px_right <= px_left or px_bottom <= px_top:
                return None

            cropped = image.crop((px_left, px_top, px_right, px_bottom))

            if left < 0.0 or top < 0.0 or right > 1.0 or bottom > 1.0:
                cropped = letterbox_to_ratio(cropped)

            # Return a copy so the file handle is safe to close
            return cropped.copy()
    except Exception as exc:
        _LOGGER.warning("Local snapshot crop (PIL) failed: %s", exc)
        return None


def pil_to_jpeg_bytes(image: Image.Image, quality: int = 85) -> bytes:
    """Encode a PIL Image to JPEG bytes.

    Convenience helper so callers that hold a PIL image do not need to
    import PIL/BytesIO directly.

    Args:
        image: PIL Image to encode.
        quality: JPEG quality (1-95).

    Returns:
        JPEG-encoded bytes.
    """
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
