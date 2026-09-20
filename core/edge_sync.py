"""Local-first Edge Agent outbox for minimized operational synchronization.

The outbox does not perform network I/O. It provides an idempotent, hash-
chained queue that a future control-plane connector can consume after device
authentication and site policy are configured.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Optional


_FORBIDDEN_KEYS = {
    "embedding", "embeddings", "face_embedding", "raw_frame", "frame_bytes",
    "image_bytes", "password", "password_hash", "api_key", "token",
    "access_token", "refresh_token", "plate_text_local",
}


class OutboxSecurityError(ValueError):
    """Raised when an event contains data that must remain on the edge."""


def verify_sync_batch(
    records: Iterable[dict],
    *,
    expected_device_id: str,
    signing_secret: Optional[str] = None,
    last_accepted_sequence: int = 0,
    expected_previous_hash: str = "",
    expected_site_id: Optional[str] = None,
    expected_organization_id: Optional[str] = None,
    require_signature: bool = False,
) -> dict:
    """Validate a contiguous batch before a control plane accepts it.

    This is deliberately transport-agnostic. A future HTTPS connector can use
    it after authenticating the device, while tests and local tooling can
    validate the exact same replay and provenance rules without networking.
    A batch is all-or-nothing: a gap, replay, device mismatch, or tampered
    record rejects the entire batch instead of partially applying it.
    """
    items = list(records)
    issues: list[str] = []
    seen_event_ids: set[str] = set()
    previous_hash = str(expected_previous_hash or "")
    try:
        accepted_sequence = max(0, int(last_accepted_sequence))
        expected_sequence = accepted_sequence + 1
    except (TypeError, ValueError):
        accepted_sequence = 0
        expected_sequence = 1
        issues.append("last accepted sequence is invalid")

    for index, record in enumerate(items):
        if not isinstance(record, dict):
            issues.append(f"record {index} is not an object")
            continue
        try:
            schema_version = int(record.get("schema_version", 0) or 0)
        except (TypeError, ValueError):
            schema_version = 0
        if schema_version != 1:
            issues.append(f"unsupported schema version at record {index}")
        if str(record.get("device_id") or "") != str(expected_device_id or ""):
            issues.append(f"device mismatch at record {index}")
        if expected_site_id is not None and str(record.get("site_id") or "") != str(expected_site_id):
            issues.append(f"site mismatch at record {index}")
        if expected_organization_id is not None and str(record.get("organization_id") or "") != str(expected_organization_id):
            issues.append(f"organization mismatch at record {index}")
        event_id = str(record.get("event_id") or "")
        if not event_id:
            issues.append(f"missing event id at record {index}")
        elif event_id in seen_event_ids:
            issues.append(f"duplicate event id at record {index}")
        else:
            seen_event_ids.add(event_id)
        try:
            sequence = int(record.get("sequence"))
        except (TypeError, ValueError):
            sequence = None
            issues.append(f"invalid sequence at record {index}")
        if sequence is not None:
            if sequence <= accepted_sequence:
                issues.append(f"sequence replay at record {index}")
            elif sequence != expected_sequence:
                issues.append(f"sequence gap at record {index}")
        if str(record.get("previous_hash") or "") != previous_hash:
            issues.append(f"previous hash mismatch at record {index}")
        unsigned = dict(record)
        unsigned.pop("record_hash", None)
        unsigned.pop("signature", None)
        actual_hash = hashlib.sha256(_canonical(unsigned).encode("utf-8")).hexdigest()
        if not hmac.compare_digest(str(record.get("record_hash") or ""), actual_hash):
            issues.append(f"record hash mismatch at record {index}")
        supplied_signature = str(record.get("signature") or "")
        if require_signature and not signing_secret:
            issues.append("signature verification secret is required")
        elif signing_secret:
            expected_signature = hmac.new(
                str(signing_secret).encode("utf-8"),
                _canonical({key: value for key, value in record.items() if key != "signature"}).encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            if not hmac.compare_digest(supplied_signature, expected_signature):
                issues.append(f"signature mismatch at record {index}")
        elif supplied_signature:
            issues.append(f"unverifiable signature at record {index}")
        if sequence is not None:
            expected_sequence = sequence + 1
        previous_hash = str(record.get("record_hash") or "")

    return {
        "valid": not issues,
        "records": len(items),
        "accepted_records": items if not issues else [],
        "last_sequence": expected_sequence - 1 if not issues else accepted_sequence,
        "last_hash": previous_hash if not issues else str(expected_previous_hash or ""),
        "issues": issues,
    }


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _contains_forbidden(value: Any, parent: str = "") -> Optional[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).casefold()
            if normalized in _FORBIDDEN_KEYS:
                return str(key)
            found = _contains_forbidden(child, normalized)
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for child in value:
            found = _contains_forbidden(child, parent)
            if found:
                return found
    return None


class EdgeEventOutbox:
    """Append-only, idempotent queue for minimized edge operational events."""

    def __init__(self, path: str | os.PathLike[str], device_id: str,
                 signing_secret: Optional[str] = None, *, site_id: str = "",
                 organization_id: str = ""):
        self.path = Path(path).expanduser().resolve()
        self.ack_path = self.path.with_suffix(self.path.suffix + ".acks")
        self.meta_path = self.path.with_suffix(self.path.suffix + ".meta")
        self.device_id = str(device_id or "edge-local")
        self.site_id = str(site_id or "")
        self.organization_id = str(organization_id or "")
        self.signing_secret = str(signing_secret) if signing_secret else None
        self._lock = threading.RLock()
        self._read_issues: list[str] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def security_mode(self) -> str:
        return "SIGNED" if self.signing_secret else "LOCAL_ONLY_UNSIGNED"

    def _load_meta(self) -> dict:
        try:
            value = json.loads(self.meta_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                return value
        except (OSError, ValueError, TypeError):
            pass
        return {
            "next_sequence": 1,
            "previous_hash": "",
            "device_id": self.device_id,
            "site_id": self.site_id,
            "organization_id": self.organization_id,
            "acknowledged_sequence": 0,
            "acknowledged_hash": "",
        }

    def _write_meta(self, value: dict) -> None:
        temporary = self.meta_path.with_suffix(self.meta_path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        # Windows can briefly deny a replace while the previous metadata
        # handle is being released (and while endpoint scanners inspect the
        # freshly-written file). Keep the operation atomic, but tolerate that
        # short-lived filesystem race instead of losing an acknowledgement.
        attempts = 5 if os.name == "nt" else 1
        try:
            for attempt in range(attempts):
                try:
                    os.replace(temporary, self.meta_path)
                    return
                except PermissionError:
                    if attempt + 1 >= attempts:
                        raise
                    time.sleep(0.01 * (attempt + 1))
        finally:
            if temporary.exists():
                temporary.unlink()

    def _signature(self, record: dict) -> Optional[str]:
        if not self.signing_secret:
            return None
        return hmac.new(
            self.signing_secret.encode("utf-8"),
            _canonical(record).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def enqueue(self, event: dict, event_id: Optional[str] = None) -> dict:
        if not isinstance(event, dict):
            raise OutboxSecurityError("Outbox events must be JSON objects.")
        forbidden = _contains_forbidden(event)
        if forbidden:
            raise OutboxSecurityError(
                f"Sensitive field '{forbidden}' is not allowed in the sync outbox.")
        payload = dict(event)
        stable_id = str(event_id or payload.get("event_id") or "").strip()
        if not stable_id:
            stable_id = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()[:24]
        with self._lock:
            existing = {item.get("event_id") for item in self.read_all()}
            if stable_id in existing:
                return next(item for item in self.read_all() if item.get("event_id") == stable_id)
            meta = self._load_meta()
            if meta.get("device_id") and str(meta["device_id"]) != self.device_id:
                raise OutboxSecurityError("Outbox metadata belongs to a different device.")
            if meta.get("site_id", "") != self.site_id or meta.get("organization_id", "") != self.organization_id:
                raise OutboxSecurityError("Outbox metadata belongs to a different site or organization.")
            sequence = max(1, int(meta.get("next_sequence", 1)))
            previous_hash = str(meta.get("previous_hash") or "")
            record = {
                "schema_version": 1,
                "event_id": stable_id,
                "device_id": self.device_id,
                "site_id": self.site_id,
                "organization_id": self.organization_id,
                "sequence": sequence,
                "queued_at": time.time(),
                "previous_hash": previous_hash,
                "payload": payload,
                "security_mode": self.security_mode,
            }
            record["record_hash"] = hashlib.sha256(
                _canonical(record).encode("utf-8")).hexdigest()
            signature = self._signature(record)
            if signature:
                record["signature"] = signature
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(_canonical(record) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._write_meta({
                "next_sequence": sequence + 1,
                "previous_hash": record["record_hash"],
                "device_id": self.device_id,
                "site_id": self.site_id,
                "organization_id": self.organization_id,
            })
            return record

    def read_all(self) -> list[dict]:
        records = []
        issues = []
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            self._read_issues = []
            return records
        for index, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except (ValueError, TypeError):
                issues.append(f"invalid JSON at line {index}")
                continue
            if isinstance(value, dict):
                records.append(value)
            else:
                issues.append(f"record is not an object at line {index}")
        self._read_issues = issues
        return records

    def acknowledged_ids(self) -> set[str]:
        try:
            return {
                line.strip() for line in self.ack_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
        except OSError:
            return set()

    def pending(self, limit: int = 100) -> list[dict]:
        acknowledged = self.acknowledged_ids()
        return [
            record for record in self.read_all()
            if record.get("event_id") not in acknowledged
        ][:max(1, int(limit))]

    def acknowledge(self, event_ids: Iterable[str]) -> int:
        values = [str(value).strip() for value in event_ids if str(value).strip()]
        if not values:
            return 0
        with self._lock:
            records = self.read_all()
            known = {item.get("event_id") for item in records}
            acknowledged = self.acknowledged_ids()
            fresh = [value for value in values if value in known and value not in acknowledged]
            if fresh:
                with self.ack_path.open("a", encoding="utf-8") as handle:
                    handle.write("\n".join(fresh) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            acknowledged.update(fresh)
            checkpoint_sequence = 0
            checkpoint_hash = ""
            for record in sorted(records, key=lambda item: int(item.get("sequence", 0) or 0)):
                sequence = int(record.get("sequence", 0) or 0)
                if sequence != checkpoint_sequence + 1 or record.get("event_id") not in acknowledged:
                    break
                checkpoint_sequence = sequence
                checkpoint_hash = str(record.get("record_hash") or "")
            meta = self._load_meta()
            meta.update({
                "acknowledged_sequence": checkpoint_sequence,
                "acknowledged_hash": checkpoint_hash,
            })
            self._write_meta(meta)
            return len(fresh)

    def checkpoint(self) -> dict:
        """Return the last contiguous acknowledged chain checkpoint."""
        meta = self._load_meta()
        try:
            sequence = max(0, int(meta.get("acknowledged_sequence", 0) or 0))
        except (TypeError, ValueError):
            sequence = 0
        return {
            "device_id": self.device_id,
            "site_id": self.site_id,
            "organization_id": self.organization_id,
            "sequence": sequence,
            "record_hash": str(meta.get("acknowledged_hash") or ""),
        }

    def verify_chain(self) -> dict:
        records = self.read_all()
        issues = list(self._read_issues)
        meta = self._load_meta()
        if meta.get("device_id") and str(meta["device_id"]) != self.device_id:
            issues.append("metadata device mismatch")
        if meta.get("site_id", "") != self.site_id or meta.get("organization_id", "") != self.organization_id:
            issues.append("metadata site or organization mismatch")
        if records:
            last_record = records[-1]
            try:
                expected_next = int(last_record.get("sequence")) + 1
            except (TypeError, ValueError):
                expected_next = None
            try:
                actual_next = int(meta.get("next_sequence", 0) or 0)
            except (TypeError, ValueError):
                actual_next = None
                issues.append("metadata next sequence is invalid")
            if expected_next is not None and actual_next != expected_next:
                issues.append("metadata next sequence mismatch")
            if str(meta.get("previous_hash") or "") != str(last_record.get("record_hash") or ""):
                issues.append("metadata previous hash mismatch")
        checkpoint = self.checkpoint()
        checkpoint_sequence = checkpoint["sequence"]
        if checkpoint_sequence:
            checkpoint_record = next(
                (item for item in records if int(item.get("sequence", 0) or 0) == checkpoint_sequence),
                None,
            )
            if checkpoint_record is None or str(checkpoint_record.get("record_hash") or "") != checkpoint["record_hash"]:
                issues.append("acknowledged checkpoint mismatch")
            acknowledged = self.acknowledged_ids()
            for record in records:
                sequence = int(record.get("sequence", 0) or 0)
                if sequence <= checkpoint_sequence and record.get("event_id") not in acknowledged:
                    issues.append("acknowledged checkpoint contains an unacknowledged record")
                    break
        result = verify_sync_batch(
            records,
            expected_device_id=self.device_id,
            signing_secret=self.signing_secret,
            expected_site_id=self.site_id,
            expected_organization_id=self.organization_id,
            require_signature=bool(self.signing_secret),
        )
        issues.extend(result["issues"])
        return {
            "valid": not issues,
            "records": len(records),
            "pending": len(self.pending()),
            "security_mode": self.security_mode,
            "last_sequence": result["last_sequence"],
            "acknowledged_sequence": checkpoint_sequence,
            "issues": issues,
        }

    def snapshot(self) -> dict:
        verification = self.verify_chain()
        return {
            "path": str(self.path),
            "device_id": self.device_id,
            "site_id": self.site_id,
            "organization_id": self.organization_id,
            "checkpoint": self.checkpoint(),
            "security_mode": self.security_mode,
            "records": verification["records"],
            "pending": verification["pending"],
            "integrity": "PASS" if verification["valid"] else "FAIL",
            "issues": verification["issues"],
        }
