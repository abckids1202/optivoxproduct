"""Control-plane receipt handling for authenticated edge synchronization."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Mapping

from core.edge_sync import _canonical
from core.sync_transport import verify_request_signature, verify_sync_request

from ..config import DEVICE_ID, ORGANIZATION_ID, SITE_ID, SYNC_SECRET
from ..database import fetch_one, transaction


class SyncIngestError(RuntimeError):
    def __init__(self, message: str, status_code: int = 400, details: dict | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.details = details or {}


def _device_registry() -> dict[str, dict[str, str]]:
    """Load active receiver credentials without exposing them in responses."""
    registry: dict[str, dict[str, str]] = {}
    if len(SYNC_SECRET) >= 32:
        registry[DEVICE_ID] = {
            "secret": SYNC_SECRET,
            "site_id": SITE_ID,
            "organization_id": ORGANIZATION_ID,
            "signature_algorithm": "hmac-sha256",
            "key_id": "",
            "public_key": "",
        }
    raw = os.getenv("OPTIVOX_SYNC_DEVICE_REGISTRY_JSON", "").strip()
    if not raw:
        return registry
    try:
        configured = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return registry
    if not isinstance(configured, dict):
        return registry
    for device_id, entry in configured.items():
        if not isinstance(entry, dict):
            continue
        secret = str(entry.get("secret") or "").strip()
        site_id = str(entry.get("site_id") or "").strip()
        organization_id = str(entry.get("organization_id") or "").strip()
        status = str(entry.get("status", "active") or "active").strip().lower()
        if len(secret) >= 32 and site_id and organization_id and status == "active":
            public_keys = entry.get("public_keys") if isinstance(entry.get("public_keys"), dict) else {}
            single_public_key = str(entry.get("public_key") or "").strip()
            configured_key_id = str(entry.get("key_id") or "").strip()
            if single_public_key and configured_key_id:
                public_keys = {**public_keys, configured_key_id: single_public_key}
            registry[str(device_id)] = {
                "secret": secret,
                "site_id": site_id,
                "organization_id": organization_id,
                "signature_algorithm": str(entry.get("signature_algorithm", "hmac-sha256") or "hmac-sha256").strip().lower(),
                "key_id": str(entry.get("key_id") or "").strip(),
                "public_key": single_public_key,
                "public_keys": json.dumps({str(key): str(value) for key, value in public_keys.items() if str(key).strip() and str(value).strip()}),
            }
    return registry


def _verification_key(entry: Mapping[str, Any], key_id: str) -> str:
    """Select one explicitly registered public key for this request."""
    try:
        key_ring = json.loads(str(entry.get("public_keys") or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        key_ring = {}
    if isinstance(key_ring, dict) and str(key_id) in key_ring:
        return str(key_ring[str(key_id)])
    return str(entry.get("public_key") or "")


def _registered_key_ids(entry: Mapping[str, Any]) -> set[str]:
    try:
        key_ring = json.loads(str(entry.get("public_keys") or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        key_ring = {}
    ids = {str(item) for item in key_ring} if isinstance(key_ring, dict) else set()
    configured = str(entry.get("key_id") or "").strip()
    if configured and entry.get("public_key"):
        ids.add(configured)
    return ids


def sync_status() -> dict[str, Any]:
    """Return bounded receiver health without exposing payloads or secrets."""
    devices: list[dict[str, Any]] = []
    for device_id, entry in _device_registry().items():
        device = fetch_one(
            """select device_id, organization_id, site_id, last_sequence,
                      last_received_at, status
               from edge_sync_devices where device_id=? and site_id=? and organization_id=?""",
            (device_id, entry["site_id"], entry["organization_id"]),
        )
        batch_count = fetch_one(
            """select count(*) as count from edge_sync_batches
               where device_id=? and site_id=? and organization_id=?""",
            (device_id, entry["site_id"], entry["organization_id"]),
        )
        event_count = fetch_one(
            """select count(*) as count from edge_sync_events
               where device_id=? and site_id=? and organization_id=?""",
            (device_id, entry["site_id"], entry["organization_id"]),
        )
        devices.append({
            "device_id": device_id,
            "site_id": entry["site_id"],
            "organization_id": entry["organization_id"],
            "signature_algorithm": entry.get("signature_algorithm", "hmac-sha256"),
            "key_id": entry.get("key_id", "") if entry.get("signature_algorithm") == "ed25519" else None,
            "registered_key_count": len(_registered_key_ids(entry)),
            "checkpoint": {
                "sequence": int((device or {}).get("last_sequence") or 0),
                "last_received_at": (device or {}).get("last_received_at"),
                "status": (device or {}).get("status", "never_connected"),
            },
            "batches_received": int((batch_count or {}).get("count") or 0),
            "events_received": int((event_count or {}).get("count") or 0),
        })
    primary = next((item for item in devices if item["device_id"] == DEVICE_ID), devices[0] if devices else {})
    return {
        "receiver_configured": bool(devices),
        "device_id": DEVICE_ID,
        "site_id": SITE_ID,
        "organization_id": ORGANIZATION_ID,
        "registered_devices": len(devices),
        "devices": devices,
        # Compatibility summary for the current single-site dashboard.
        "checkpoint": primary.get("checkpoint", {"status": "never_connected"}),
        "batches_received": primary.get("batches_received", 0),
        "events_received": primary.get("events_received", 0),
    }


def set_device_status(device_id: str, status: str, *, actor_id: str | None = None) -> dict[str, Any]:
    """Revoke or reactivate a registered device without resetting its chain."""
    normalized_device = str(device_id or "").strip()
    normalized_status = str(status or "").strip().lower()
    if normalized_status not in {"active", "revoked", "disabled"}:
        raise SyncIngestError("device status must be active, revoked, or disabled", 400)
    entry = _device_registry().get(normalized_device)
    if not entry:
        raise SyncIngestError("device is not registered for this receiver", 404)
    with transaction(immediate=True) as con:
        con.execute(
            """insert into edge_sync_devices
               (device_id, organization_id, site_id, status)
               values (?, ?, ?, ?)
               on conflict(device_id, organization_id, site_id) do update set status=excluded.status""",
            (normalized_device, entry["organization_id"], entry["site_id"], normalized_status),
        )
        from .audit_service import record_action_in_connection
        record_action_in_connection(
            con,
            "sync.device_status",
            "edge_sync_device",
            normalized_device,
            {"status": normalized_status, "site_id": entry["site_id"], "organization_id": entry["organization_id"]},
            actor_type="operator" if actor_id else "system",
            actor_id=actor_id,
        )
        return {
            "device_id": normalized_device,
            "site_id": entry["site_id"],
            "organization_id": entry["organization_id"],
            "status": normalized_status,
        }


def ingest_batch(
    body: Mapping[str, Any],
    signature: str,
    *,
    header_scope: Mapping[str, str] | None = None,
    signature_algorithm: str = "",
    key_id: str = "",
) -> dict[str, Any]:
    """Atomically verify and persist one edge batch.

    Receiver credentials are selected from the configured device registry.
    Every device still has an independent sequence checkpoint and deployment
    scope, so one device cannot advance or replay another device's chain.
    """
    if not isinstance(body, Mapping):
        raise SyncIngestError("synchronization body must be an object", 400)
    device_id = str(body.get("device_id") or "")
    entry = _device_registry().get(device_id)
    if not entry:
        raise SyncIngestError("device is not registered for this receiver", 403)
    if len(entry["secret"]) < 32:
        raise SyncIngestError("synchronization receiver is not configured", 503)
    header_scope = header_scope or {}
    for key, header in (("device_id", "device"), ("site_id", "site"), ("organization_id", "organization")):
        supplied_header = str(header_scope.get(header) or "")
        if supplied_header and supplied_header != str(body.get(key) or ""):
            raise SyncIngestError(f"{header} header does not match request body", 400)
    if str(body.get("site_id") or "") != entry["site_id"] or str(body.get("organization_id") or "") != entry["organization_id"]:
        raise SyncIngestError("synchronization scope is not registered for this receiver", 403)
    expected_algorithm = entry.get("signature_algorithm", "hmac-sha256")
    supplied_algorithm = str(signature_algorithm or "hmac-sha256").strip().lower()
    if supplied_algorithm != expected_algorithm:
        raise SyncIngestError("synchronization signature algorithm is not registered for this device", 401)
    expected_key_id = str(entry.get("key_id") or "")
    verification_key = _verification_key(entry, str(key_id or expected_key_id))
    if expected_algorithm == "ed25519" and (
        not str(key_id or "")
        or str(key_id) not in _registered_key_ids(entry)
        or not verification_key
    ):
        raise SyncIngestError("synchronization key id is not registered for this device", 401)

    with transaction(immediate=True) as con:
        state = con.execute(
            "select last_sequence, last_hash, status from edge_sync_devices where device_id=? and site_id=? and organization_id=?",
            (device_id, entry["site_id"], entry["organization_id"]),
        ).fetchone()
        if state and str(state["status"] or "active") in {"revoked", "disabled"}:
            raise SyncIngestError("device is not active", 403)
        # Idempotent retries must be recognized before continuity advances,
        # while still authenticating and binding the original body exactly.
        body_checksum = hashlib.sha256(_canonical(dict(body)).encode("utf-8")).hexdigest()
        if not verify_request_signature(
            body,
            signature,
            algorithm=expected_algorithm,
            signing_secret=entry["secret"],
            verification_key=verification_key,
        ):
            raise SyncIngestError("synchronization request signature mismatch", 401)
        existing = con.execute(
            "select body_checksum, last_sequence, last_hash from edge_sync_batches where batch_id=?",
            (str(body.get("batch_id") or ""),),
        ).fetchone()
        if existing:
            if str(existing["body_checksum"]) != body_checksum:
                raise SyncIngestError("batch id was reused with different content", 409)
            return {
                "accepted": True,
                "idempotent": True,
                "batch_id": str(body.get("batch_id") or ""),
                "accepted_records": int(body.get("last_sequence", 0) or 0) - int(body.get("first_sequence", 0) or 0) + 1,
                "last_sequence": int(existing["last_sequence"]),
                "last_hash": str(existing["last_hash"]),
            }
        last_sequence = int(state["last_sequence"] or 0) if state else 0
        last_hash = str(state["last_hash"] or "") if state else ""
        verified = verify_sync_request(
            body,
            signature,
            signing_secret=entry["secret"],
            expected_device_id=device_id,
            expected_site_id=entry["site_id"],
            expected_organization_id=entry["organization_id"],
            last_accepted_sequence=last_sequence,
            expected_previous_hash=last_hash,
            signature_algorithm=expected_algorithm,
            verification_key=verification_key,
        )
        if not verified["valid"]:
            raise SyncIngestError("synchronization batch rejected", 409, {"issues": verified["issues"]})
        records = verified["records"]
        first = records[0]
        last = records[-1]
        for record in records:
            payload = record.get("payload")
            payload_json = _canonical(payload if isinstance(payload, dict) else {})
            con.execute(
                """insert into edge_sync_events
                   (event_id, device_id, organization_id, site_id, sequence,
                    event_type, occurred_at, record_hash, payload_json)
                   values (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(record["event_id"]), device_id, entry["organization_id"], entry["site_id"],
                    int(record["sequence"]), str((payload or {}).get("event_type") or "EDGE_EVENT"),
                    str((payload or {}).get("occurred_at") or "") or None,
                    str(record["record_hash"]), payload_json,
                ),
            )
        con.execute(
            """insert into edge_sync_batches
               (batch_id, device_id, organization_id, site_id, first_sequence,
                last_sequence, last_hash, record_count, body_checksum)
               values (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                verified["batch_id"], device_id, entry["organization_id"], entry["site_id"],
                int(first["sequence"]), int(last["sequence"]), str(last["record_hash"]),
                len(records), body_checksum,
            ),
        )
        con.execute(
            """insert into edge_sync_devices
               (device_id, organization_id, site_id, last_sequence, last_hash,
                last_batch_id, last_received_at)
               values (?, ?, ?, ?, ?, ?, datetime('now'))
               on conflict(device_id, organization_id, site_id) do update set
                 last_sequence=excluded.last_sequence,
                 last_hash=excluded.last_hash,
                 last_batch_id=excluded.last_batch_id,
                 last_received_at=excluded.last_received_at""",
            (device_id, entry["organization_id"], entry["site_id"], int(last["sequence"]), str(last["record_hash"]), verified["batch_id"]),
        )
        return {
            "accepted": True,
            "idempotent": False,
            "batch_id": verified["batch_id"],
            "accepted_records": len(records),
            "last_sequence": int(last["sequence"]),
            "last_hash": str(last["record_hash"]),
        }
