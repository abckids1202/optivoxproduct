"""Small, deterministic face-image augmentations for sparse local enrollment.

These transforms improve tolerance to lighting and small camera changes. They
are deliberately conservative: they do not invent a new identity or claim to
replace real multi-angle enrollment images.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import cv2
import numpy as np


def _rotate(image: np.ndarray, degrees: float) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width * 0.5, height * 0.5), degrees, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _gamma(image: np.ndarray, value: float) -> np.ndarray:
    inverse = 1.0 / max(0.1, float(value))
    table = np.array(
        [((index / 255.0) ** inverse) * 255.0 for index in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image, table)


def _crop_resize(image: np.ndarray, ratio: float = 0.04) -> np.ndarray:
    height, width = image.shape[:2]
    dx = max(1, int(width * ratio))
    dy = max(1, int(height * ratio))
    if width - 2 * dx < 8 or height - 2 * dy < 8:
        return image.copy()
    cropped = image[dy:height - dy, dx:width - dx]
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)


def generate_variants(image: np.ndarray, max_variants: int = 3) -> List[Tuple[str, np.ndarray]]:
    """Return unique, bounded variants with stable transformation labels."""
    if image is None or getattr(image, "size", 0) == 0:
        return []
    candidates: Iterable[Tuple[str, np.ndarray]] = (
        ("brightness_contrast", cv2.convertScaleAbs(image, alpha=1.06, beta=10)),
        ("crop_in", _crop_resize(image)),
        ("rotate_left", _rotate(image, -4.0)),
        ("rotate_right", _rotate(image, 4.0)),
        ("gamma_low", _gamma(image, 0.92)),
        ("horizontal_flip", cv2.flip(image, 1)),
    )
    variants: List[Tuple[str, np.ndarray]] = []
    for label, variant in candidates:
        if len(variants) >= max(0, int(max_variants)):
            break
        if variant is None or variant.shape != image.shape or np.array_equal(variant, image):
            continue
        if any(np.array_equal(variant, previous) for _, previous in variants):
            continue
        variants.append((label, variant))
    return variants
