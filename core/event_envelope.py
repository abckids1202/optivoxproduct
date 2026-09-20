"""Versioned operational event envelopes for OptiVox.

Perception modules can continue returning their small legacy tuples, while
the correlation boundary converts them into a stable, auditable contract for
attendance, security, evidence, and future edge synchronization.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional


def _stable_id(prefix: str, values: Iterable[Any]) -> str:
    payload = "|".join("" if value is None else str(value) for value in values)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{digest}"


@dataclass(frozen=True)
class EventEnvelope:
    """A privacy-aware event record shared by edge and platform layers."""

    event_type: str
    occurred_at: str
    camera_id: str = "cam_0"
    device_id: Optional[str] = None
    site_id: Optional[str] = None
    organization_id: Optional[str] = None
    source_frame_id: Optional[int] = None
    entity_id: Optional[str] = None
    presence_session_id: Optional[str] = None
    confidence: float = 0.0
    target: str = "SYSTEM"
    policy_version: str = "1"
    privacy_class: str = "operational"
    model_versions: Dict[str, str] = field(default_factory=dict)
    evidence_refs: tuple[str, ...] = ()
    correlation_id: str = ""
    event_id: str = ""
    schema_version: int = 1

    def __post_init__(self):
        if not self.correlation_id:
            object.__setattr__(
                self,
                "correlation_id",
                _stable_id(
                    "corr",
                    (self.camera_id, self.entity_id, self.presence_session_id,
                     self.site_id, self.organization_id,
                     self.event_type.split("_", 1)[0], self.policy_version),
                ),
            )
        if not self.event_id:
            object.__setattr__(
                self,
                "event_id",
                _stable_id(
                    "evt",
                    (self.correlation_id, self.event_type, self.target,
                     self.source_frame_id, self.occurred_at),
                ),
            )

    @classmethod
    def from_security_tuple(
        cls,
        event: tuple,
        *,
        camera_id: str,
        occurred_at: str,
        source_frame_id: Optional[int] = None,
        entity_id: Optional[str] = None,
        presence_session_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> "EventEnvelope":
        tuple_metadata = event[4] if len(event) > 4 and isinstance(event[4], dict) else {}
        merged_metadata = dict(tuple_metadata)
        merged_metadata.update(metadata or {})
        metadata = merged_metadata
        event_type = str(event[0]) if event else "SECURITY_SIGNAL"
        target = str(event[1]) if len(event) > 1 else "SYSTEM"
        try:
            confidence = float(event[2]) if len(event) > 2 else 0.0
        except (TypeError, ValueError):
            confidence = 0.0
        frame_id = metadata.get("source_frame_id", source_frame_id)
        try:
            frame_id = int(frame_id) if frame_id is not None else None
        except (TypeError, ValueError):
            frame_id = None
        normalized_occurred_at = str(occurred_at or "").strip()
        if not normalized_occurred_at:
            normalized_occurred_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        supplied_correlation_id = str(metadata.get("correlation_id") or "").strip()
        if not supplied_correlation_id:
            # Keep incident correlation aligned with the operational context.
            # In particular, two restricted zones must not share a context key
            # merely because the same entity visited them in one session.
            supplied_correlation_id = _stable_id(
                "corr",
                (camera_id, entity_id or metadata.get("entity_id"),
                 presence_session_id or metadata.get("presence_session_id"),
                 event_type.split("_", 1)[0], metadata.get("zone_id"),
                 metadata.get("policy_version") or "1"),
            )
        return cls(
            event_type=event_type,
            occurred_at=normalized_occurred_at,
            camera_id=str(camera_id or "cam_0"),
            device_id=metadata.get("device_id"),
            site_id=metadata.get("site_id"),
            organization_id=metadata.get("organization_id"),
            source_frame_id=frame_id,
            entity_id=entity_id or metadata.get("entity_id"),
            presence_session_id=presence_session_id or metadata.get("presence_session_id"),
            confidence=max(0.0, min(1.0, confidence)),
            target=target,
            policy_version=str(metadata.get("policy_version") or "1"),
            privacy_class=str(metadata.get("privacy_class") or "operational"),
            model_versions=dict(metadata.get("model_versions") or {}),
            evidence_refs=tuple(str(value) for value in (metadata.get("evidence_refs") or ())),
            correlation_id=supplied_correlation_id,
        )

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "occurred_at": self.occurred_at,
            "camera_id": self.camera_id,
            "device_id": self.device_id,
            "site_id": self.site_id,
            "organization_id": self.organization_id,
            "source_frame_id": self.source_frame_id,
            "entity_id": self.entity_id,
            "presence_session_id": self.presence_session_id,
            "correlation_id": self.correlation_id,
            "confidence": self.confidence,
            "target": self.target,
            "policy_version": self.policy_version,
            "privacy_class": self.privacy_class,
            "model_versions": dict(self.model_versions),
            "evidence_refs": list(self.evidence_refs),
        }


def attach_envelope_metadata(metadata: dict, envelope: EventEnvelope) -> dict:
    """Return metadata enriched without mutating the producer's dictionary."""
    enriched = dict(metadata or {})
    enriched.update({
        "event_id": envelope.event_id,
        "correlation_id": envelope.correlation_id,
        "occurred_at": envelope.occurred_at,
        "camera_id": envelope.camera_id,
        "device_id": envelope.device_id,
        "site_id": envelope.site_id,
        "organization_id": envelope.organization_id,
        "entity_id": envelope.entity_id,
        "presence_session_id": envelope.presence_session_id,
        "policy_version": envelope.policy_version,
        "privacy_class": envelope.privacy_class,
        "model_versions": dict(envelope.model_versions),
        "evidence_refs": list(envelope.evidence_refs),
    })
    return enriched
