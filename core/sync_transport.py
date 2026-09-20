"""Authenticated, bounded transport for the local-first edge outbox.

The edge agent can keep running without this module ever touching a network.
When an endpoint and injected transport are configured, it creates a signed
contiguous batch and acknowledges records only after the receiver confirms the
same sequence and hash.  The protocol is intentionally small so a future
control plane can implement it without importing the vision runtime.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any, Callable, Mapping, Optional

from .edge_sync import EdgeEventOutbox, _canonical, verify_sync_batch
from .device_auth import DeviceAuthError, sign_ed25519, verify_ed25519


MAX_RESPONSE_BYTES = 64 * 1024


class SyncTransportError(RuntimeError):
    """Raised when a delivery cannot be safely completed."""


def sign_request(body: Mapping[str, Any], secret: str) -> str:
    if not secret or len(str(secret)) < 32:
        raise SyncTransportError("a 32-character synchronization secret is required")
    return hmac.new(
        str(secret).encode("utf-8"),
        _canonical(dict(body)).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def sign_request_ed25519(body: Mapping[str, Any], private_key: Any) -> str:
    """Sign the canonical request body with an edge-only Ed25519 key."""
    try:
        return sign_ed25519(_canonical(dict(body)).encode("utf-8"), private_key)
    except DeviceAuthError as exc:
        raise SyncTransportError(str(exc)) from exc


def verify_request_signature(
    body: Mapping[str, Any],
    signature: str,
    *,
    algorithm: str,
    signing_secret: str = "",
    verification_key: Any = None,
) -> bool:
    """Verify a request using the configured algorithm without leaking key data."""
    normalized = str(algorithm or "hmac-sha256").strip().lower()
    canonical = _canonical(dict(body)).encode("utf-8")
    if normalized == "hmac-sha256":
        if not signing_secret or len(str(signing_secret)) < 32:
            return False
        expected = hmac.new(str(signing_secret).encode("utf-8"), canonical, hashlib.sha256).hexdigest()
        return hmac.compare_digest(str(signature or ""), expected)
    if normalized == "ed25519":
        return verify_ed25519(canonical, signature, verification_key)
    return False


def _batch_id(device_id: str, first_sequence: int, last_sequence: int, last_hash: str) -> str:
    value = f"{device_id}:{first_sequence}:{last_sequence}:{last_hash}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_sync_request(outbox: EdgeEventOutbox, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the stable, minimized body that a control plane receives."""
    if not records:
        raise SyncTransportError("cannot build an empty synchronization batch")
    checkpoint = outbox.checkpoint()
    first = records[0]
    last = records[-1]
    return {
        "schema_version": 1,
        "batch_id": _batch_id(
            outbox.device_id,
            int(first["sequence"]),
            int(last["sequence"]),
            str(last.get("record_hash") or ""),
        ),
        "device_id": outbox.device_id,
        "site_id": outbox.site_id,
        "organization_id": outbox.organization_id,
        "base_sequence": int(checkpoint["sequence"]),
        "base_hash": str(checkpoint.get("record_hash") or ""),
        "first_sequence": int(first["sequence"]),
        "last_sequence": int(last["sequence"]),
        "last_hash": str(last.get("record_hash") or ""),
        "records": records,
    }


def verify_sync_request(
    body: Mapping[str, Any],
    signature: str,
    *,
    signing_secret: str,
    expected_device_id: str,
    expected_site_id: str,
    expected_organization_id: str,
    last_accepted_sequence: int = 0,
    expected_previous_hash: str = "",
    signature_algorithm: str = "hmac-sha256",
    verification_key: Any = None,
) -> dict[str, Any]:
    """Verify request authentication, scope, batch metadata, and continuity."""
    if not isinstance(body, Mapping):
        return {"valid": False, "issues": ["request body must be an object"]}
    issues: list[str] = []
    try:
        if not verify_request_signature(
            body,
            signature,
            algorithm=signature_algorithm,
            signing_secret=signing_secret,
            verification_key=verification_key,
        ):
            issues.append("request signature mismatch")
    except (SyncTransportError, DeviceAuthError) as exc:
        issues.append(str(exc))
    if int(body.get("schema_version", 0) or 0) != 1:
        issues.append("unsupported request schema version")
    for key, expected_value in (
        ("device_id", expected_device_id),
        ("site_id", expected_site_id),
        ("organization_id", expected_organization_id),
    ):
        if str(body.get(key) or "") != str(expected_value or ""):
            issues.append(f"{key} mismatch")
    records = body.get("records")
    if not isinstance(records, list) or not records:
        issues.append("request records must be a non-empty list")
        records = []
    if records:
        first = records[0]
        last = records[-1]
        checks = (
            ("first_sequence", first.get("sequence"), body.get("first_sequence")),
            ("last_sequence", last.get("sequence"), body.get("last_sequence")),
            ("last_hash", last.get("record_hash"), body.get("last_hash")),
        )
        for label, actual, declared in checks:
            if str(actual) != str(declared):
                issues.append(f"{label} mismatch")
        expected_batch = _batch_id(
            str(body.get("device_id") or ""),
            int(first.get("sequence", 0) or 0),
            int(last.get("sequence", 0) or 0),
            str(last.get("record_hash") or ""),
        )
        if not hmac.compare_digest(str(body.get("batch_id") or ""), expected_batch):
            issues.append("batch id mismatch")
    batch = verify_sync_batch(
        records,
        expected_device_id=expected_device_id,
        signing_secret=signing_secret,
        last_accepted_sequence=last_accepted_sequence,
        expected_previous_hash=expected_previous_hash,
        expected_site_id=expected_site_id,
        expected_organization_id=expected_organization_id,
        require_signature=True,
    )
    issues.extend(batch["issues"])
    return {
        "valid": not issues,
        "issues": issues,
        "batch": batch,
        "records": records if not issues else [],
        "batch_id": str(body.get("batch_id") or ""),
    }


Transport = Callable[[str, dict[str, Any], dict[str, str], float], Mapping[str, Any]]


class EdgeSyncClient:
    """One-shot delivery client with no background thread or hidden network I/O."""

    def __init__(
        self,
        outbox: EdgeEventOutbox,
        endpoint: str = "",
        signing_secret: Optional[str] = None,
        *,
        batch_size: int = 50,
        timeout_seconds: float = 10.0,
        transport: Optional[Transport] = None,
        private_key: Any = None,
        key_id: str = "",
        signature_algorithm: str = "hmac-sha256",
    ):
        self.outbox = outbox
        self.endpoint = str(endpoint or "").strip()
        self.signing_secret = str(signing_secret or outbox.signing_secret or "")
        self.batch_size = max(1, min(int(batch_size), 500))
        self.timeout_seconds = max(1.0, min(float(timeout_seconds), 30.0))
        self.transport = transport
        self.private_key = private_key
        self.key_id = str(key_id or "").strip()
        self.signature_algorithm = str(signature_algorithm or "hmac-sha256").strip().lower()

    def sync_once(self) -> dict[str, Any]:
        """Attempt one contiguous delivery and acknowledge only verified success."""
        if not self.endpoint or not self.transport:
            return {"status": "DISABLED", "sent": 0, "reason": "transport_not_configured"}
        if not self.signing_secret or self.outbox.signing_secret != self.signing_secret:
            return {"status": "BLOCKED", "sent": 0, "reason": "outbox_signing_secret_mismatch"}
        chain = self.outbox.verify_chain()
        if not chain.get("valid"):
            return {"status": "BLOCKED", "sent": 0, "reason": "outbox_integrity_failure", "issues": chain.get("issues", [])}
        records = self.outbox.pending(self.batch_size)
        if not records:
            return {"status": "IDLE", "sent": 0}
        checkpoint = self.outbox.checkpoint()
        batch = verify_sync_batch(
            records,
            expected_device_id=self.outbox.device_id,
            signing_secret=self.signing_secret,
            last_accepted_sequence=checkpoint["sequence"],
            expected_previous_hash=checkpoint["record_hash"],
            expected_site_id=self.outbox.site_id,
            expected_organization_id=self.outbox.organization_id,
            require_signature=True,
        )
        if not batch["valid"]:
            return {"status": "BLOCKED", "sent": 0, "reason": "non_contiguous_batch", "issues": batch["issues"]}
        body = build_sync_request(self.outbox, records)
        if self.signature_algorithm == "ed25519":
            if self.private_key is None or not self.key_id:
                return {"status": "BLOCKED", "sent": 0, "reason": "ed25519_key_not_configured"}
            try:
                request_signature = sign_request_ed25519(body, self.private_key)
            except SyncTransportError as exc:
                return {"status": "BLOCKED", "sent": 0, "reason": str(exc)}
        elif self.signature_algorithm == "hmac-sha256":
            request_signature = sign_request(body, self.signing_secret)
        else:
            return {"status": "BLOCKED", "sent": 0, "reason": "unsupported_signature_algorithm"}
        headers = {
            "Content-Type": "application/json",
            "X-Optivox-Device": self.outbox.device_id,
            "X-Optivox-Site": self.outbox.site_id,
            "X-Optivox-Organization": self.outbox.organization_id,
            "X-Optivox-Signature": request_signature,
            "X-Optivox-Signature-Algorithm": self.signature_algorithm,
            "Idempotency-Key": str(body["batch_id"]),
        }
        if self.key_id:
            headers["X-Optivox-Key-Id"] = self.key_id
        try:
            response = dict(self.transport(self.endpoint, body, headers, self.timeout_seconds) or {})
        except Exception as exc:
            return {"status": "FAILED", "sent": 0, "reason": str(exc)[:300]}
        final = records[-1]
        if not response.get("accepted") or str(response.get("last_hash") or "") != str(final.get("record_hash") or ""):
            return {"status": "REJECTED", "sent": 0, "reason": "receiver_ack_mismatch", "response": response}
        if int(response.get("last_sequence", -1)) != int(final.get("sequence", -2)):
            return {"status": "REJECTED", "sent": 0, "reason": "receiver_sequence_mismatch", "response": response}
        sent = self.outbox.acknowledge([str(item["event_id"]) for item in records])
        return {
            "status": "SENT" if sent == len(records) else "PARTIAL_ACK",
            "sent": sent,
            "batch_id": body["batch_id"],
            "last_sequence": final["sequence"],
        }


def urllib_transport(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
    """Small stdlib transport for an explicitly validated HTTPS endpoint."""
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, build_opener, HTTPRedirectHandler

    class _NoRedirect(HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    encoded = _canonical(body).encode("utf-8")
    request = Request(url, data=encoded, headers=headers, method="POST")
    opener = build_opener(_NoRedirect())
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise SyncTransportError("sync response exceeds size limit")
            return json.loads(raw.decode("utf-8"))
    except HTTPError as exc:
        raise SyncTransportError(f"sync endpoint returned HTTP {exc.code}") from exc
    except (URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise SyncTransportError(f"sync transport failed: {exc}") from exc
