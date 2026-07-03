"""Image processing utilities for RoboBench.

Handles image loading, resizing, and base64 encoding for API calls.
"""

import base64
from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np


def reshape_frame_to_512(image: np.ndarray) -> np.ndarray:
    """Resize image to 512x512."""
    return cv2.resize(image, (512, 512))


def image_to_frame(image_path: Union[str, Path]) -> np.ndarray:
    """Load an image from disk."""
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return frame


def encode_frame_cv(frame: np.ndarray, fmt: str = "jpeg", quality: int = 85) -> str:
    """Encode an OpenCV frame as base64.

    Defaults to JPEG @ Q85 because multi-image prompts (up to 30 images per
    request) blew up to >10MB with PNG and timed out on Gemini. JPEG is
    typically 5-10x smaller and visually indistinguishable for our task.
    """
    fmt = fmt.lower()
    if fmt == "jpeg" or fmt == "jpg":
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        _, buffer = cv2.imencode(".jpg", frame, params)
    else:
        _, buffer = cv2.imencode(f".{fmt}", frame)
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def encode_image_file(image_path: Union[str, Path]) -> str:
    """Encode an image file as base64 (raw bytes, no re-encoding)."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def process_image(
    image_path: Union[str, Path],
    resize: bool = True,
    fmt: str = "jpeg",
    quality: int = 85,
) -> Tuple[str, str]:
    """Load, optionally resize, and base64-encode an image.

    Args:
        image_path: Path to the image file.
        resize: Resize to 512x512 to bound payload size.
        fmt: "jpeg" (default, smaller) or "png" (lossless).
        quality: JPEG quality (1-100). Ignored for PNG.

    Returns:
        (base64_string, media_type) where media_type is e.g. "image/jpeg".
    """
    frame = image_to_frame(image_path)
    if resize:
        frame = reshape_frame_to_512(frame)
    b64 = encode_frame_cv(frame, fmt=fmt, quality=quality)
    media_type = f"image/{'jpeg' if fmt in ('jpeg', 'jpg') else fmt}"
    return b64, media_type
