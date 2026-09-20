"""Privacy-aware demographic output helpers.

Age inferred from a face model is an approximate visual estimate. The public
runtime therefore exposes coarse bands rather than exact ages and never uses
the result for attendance, access, discipline, or security decisions.
"""

from __future__ import annotations

import math
from typing import Optional


AGE_BANDS = (
    (0, 12, "child"),
    (13, 17, "teen"),
    (18, 24, "young_adult"),
    (25, 44, "adult"),
    (45, 64, "middle_aged"),
    (65, 120, "older_adult"),
)


def age_band(value: object) -> Optional[str]:
    """Return a conservative age band or ``None`` for unusable estimates."""
    try:
        estimate = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(estimate) or estimate < 0 or estimate > 120:
        return None
    rounded = int(round(estimate))
    for minimum, maximum, label in AGE_BANDS:
        if minimum <= rounded <= maximum:
            return label
    return None
