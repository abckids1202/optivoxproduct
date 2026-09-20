"""Policy boundary for automatic external security notifications."""

from __future__ import annotations

from typing import Any, Dict


AUTO_SEND_EVENTS = frozenset({
    "DANGEROUS_OBJECT", "WEAPON_DETECTED", "FIRE_DETECTED", "SMOKE_DETECTED",
    "EVACUATION_ALERT", "FALL_DETECTED", "HANDS_RAISED", "SPOOF_DETECTED",
    "CONGESTION", "ZONE_INTRUSION",
    "LOITERING", "RUNNING", "PPE_VIOLATION", "SPEED_THRESHOLD_EXCEEDED",
    "CAMERA_DISCONNECTED", "RUNTIME_CRASH", "ALERT_DELIVERY_FAILURE",
})

SENSITIVE_TERMS = frozenset({
    "disciplinary", "law enforcement", "police", "accusation", "identity claim",
    "student misconduct", "biometric", "permission change", "delete record",
})


def notification_decision(event_type: str, details: Any = "") -> Dict[str, str]:
    """Return a deterministic external-notification decision.

    The runtime can automatically notify about bounded operational hazards.
    Natural-language or identity-sensitive messages remain review-only even if
    an upstream caller accidentally uses a familiar event name.
    """
    normalized_type = str(event_type or "").strip().upper()
    normalized_details = str(details or "").casefold()
    if any(term in normalized_details for term in SENSITIVE_TERMS):
        return {"decision": "APPROVAL_REQUIRED", "reason": "sensitive_external_message"}
    if normalized_type in AUTO_SEND_EVENTS:
        return {"decision": "AUTO_SEND", "reason": "allowlisted_operational_event"}
    return {"decision": "REVIEW_ONLY", "reason": "event_type_not_allowlisted"}
