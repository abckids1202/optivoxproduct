from __future__ import annotations

import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from backend import database
from backend.platform_schema import ensure_platform_schema
from backend.services import sync_service
from backend.services.sync_service import SyncIngestError
from core.edge_sync import EdgeEventOutbox
from core.sync_transport import (
    EdgeSyncClient,
    build_sync_request,
    sign_request,
    sign_request_ed25519,
    verify_request_signature,
    verify_sync_request,
)


def test_sync_request_is_signed_contiguous_and_scope_bound(tmp_path):
    secret = "s" * 40
    outbox = EdgeEventOutbox(tmp_path / "edge.ndjson", "edge-1", secret, site_id="site-1", organization_id="org-1")
    outbox.enqueue({"event_id": "evt-1", "event_type": "ATTENDANCE_DECISION", "occurred_at": "2026-09-18T01:00:00Z"})
    body = build_sync_request(outbox, outbox.pending())
    signature = sign_request(body, secret)
    result = verify_sync_request(
        body, signature, signing_secret=secret,
        expected_device_id="edge-1", expected_site_id="site-1", expected_organization_id="org-1",
    )
    assert result["valid"] is True
    assert verify_sync_request(
        body, "bad", signing_secret=secret,
        expected_device_id="edge-1", expected_site_id="site-1", expected_organization_id="org-1",
    )["valid"] is False


def test_edge_sync_client_acknowledges_only_matching_receiver_checkpoint(tmp_path):
    secret = "s" * 40
    outbox = EdgeEventOutbox(tmp_path / "edge.ndjson", "edge-1", secret, site_id="site-1", organization_id="org-1")
    outbox.enqueue({"event_id": "evt-1", "event_type": "CAMERA_HEALTH"})
    calls = []

    def transport(url, body, headers, timeout):
        calls.append((url, headers, timeout))
        assert headers["X-Optivox-Signature"] == sign_request(body, secret)
        return {"accepted": True, "last_sequence": body["last_sequence"], "last_hash": body["last_hash"]}

    result = EdgeSyncClient(
        outbox, "https://control.example.test/api/sync/ingest", secret, transport=transport,
    ).sync_once()
    assert result["status"] == "SENT"
    assert result["sent"] == 1
    assert outbox.pending() == []
    assert calls and calls[0][0].startswith("https://")


def test_edge_sync_client_rejects_wrong_ack_without_acknowledging(tmp_path):
    secret = "s" * 40
    outbox = EdgeEventOutbox(tmp_path / "edge.ndjson", "edge-1", secret)
    outbox.enqueue({"event_id": "evt-1", "event_type": "CAMERA_HEALTH"})

    def transport(url, body, headers, timeout):
        return {"accepted": True, "last_sequence": body["last_sequence"], "last_hash": "forged"}

    result = EdgeSyncClient(outbox, "https://control.example.test/ingest", secret, transport=transport).sync_once()
    assert result["status"] == "REJECTED"
    assert len(outbox.pending()) == 1


def test_ed25519_request_authentication_and_client_headers(tmp_path):
    private = Ed25519PrivateKey.generate()
    private_pem = private.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    public_pem = private.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("ascii")
    rollover = Ed25519PrivateKey.generate()
    rollover_pem = rollover.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    rollover_public_pem = rollover.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("ascii")
    secret = "e" * 40
    outbox = EdgeEventOutbox(tmp_path / "edge-ed.ndjson", "edge-ed", secret, site_id="site-ed", organization_id="org-ed")
    outbox.enqueue({"event_id": "evt-ed", "event_type": "CAMERA_HEALTH"})

    def transport(_url, body, headers, _timeout):
        assert headers["X-Optivox-Signature-Algorithm"] == "ed25519"
        assert headers["X-Optivox-Key-Id"] == "ed-key-1"
        assert verify_request_signature(
            body,
            headers["X-Optivox-Signature"],
            algorithm="ed25519",
            verification_key=public_pem,
        )
        return {"accepted": True, "last_sequence": body["last_sequence"], "last_hash": body["last_hash"]}

    result = EdgeSyncClient(
        outbox,
        "https://control.example.test/ingest",
        secret,
        transport=transport,
        private_key=private_pem,
        key_id="ed-key-1",
        signature_algorithm="ed25519",
    ).sync_once()
    assert result["status"] == "SENT"
    assert outbox.pending() == []


def test_ed25519_receiver_accepts_registered_key_and_rejects_tampering(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "platform.db")
    ensure_platform_schema()
    private = Ed25519PrivateKey.generate()
    public_pem = private.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("ascii")
    private_pem = private.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    rollover = Ed25519PrivateKey.generate()
    rollover_pem = rollover.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    rollover_public_pem = rollover.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("ascii")
    secret = "k" * 40
    monkeypatch.setattr(sync_service, "SYNC_SECRET", "")
    monkeypatch.setenv(
        "OPTIVOX_SYNC_DEVICE_REGISTRY_JSON",
        json.dumps({
            "edge-ed": {
                "secret": secret,
                "site_id": "site-ed",
                "organization_id": "org-ed",
                "signature_algorithm": "ed25519",
                "key_id": "ed-key-1",
                "public_key": public_pem,
                "public_keys": {"ed-key-1": public_pem, "ed-key-2": rollover_public_pem},
            }
        }),
    )
    outbox = EdgeEventOutbox(tmp_path / "edge-ed.ndjson", "edge-ed", secret, site_id="site-ed", organization_id="org-ed")
    outbox.enqueue({"event_id": "evt-ed", "event_type": "CAMERA_HEALTH"})
    body = build_sync_request(outbox, outbox.pending())
    signature = sign_request_ed25519(body, private_pem)
    accepted = sync_service.ingest_batch(body, signature, signature_algorithm="ed25519", key_id="ed-key-1")
    assert accepted["accepted"] is True
    rollover_signature = sign_request_ed25519(body, rollover_pem)
    rotated_retry = sync_service.ingest_batch(body, rollover_signature, signature_algorithm="ed25519", key_id="ed-key-2")
    assert rotated_retry["idempotent"] is True

    forged = dict(body)
    forged["organization_id"] = "other-org"
    forged_signature = sign_request_ed25519(forged, private_pem)
    with pytest.raises(SyncIngestError, match="scope"):
        sync_service.ingest_batch(forged, forged_signature, signature_algorithm="ed25519", key_id="ed-key-1")

    with pytest.raises(SyncIngestError, match="key id"):
        sync_service.ingest_batch(body, signature, signature_algorithm="ed25519", key_id="wrong-key")


def test_backend_sync_ingest_is_idempotent_and_rejects_replay(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "platform.db")
    ensure_platform_schema()
    secret = "s" * 40
    monkeypatch.setattr(sync_service, "SYNC_SECRET", secret)
    monkeypatch.setattr(sync_service, "DEVICE_ID", "edge-1")
    monkeypatch.setattr(sync_service, "SITE_ID", "site-1")
    monkeypatch.setattr(sync_service, "ORGANIZATION_ID", "org-1")
    outbox = EdgeEventOutbox(tmp_path / "edge.ndjson", "edge-1", secret, site_id="site-1", organization_id="org-1")
    outbox.enqueue({"event_id": "evt-1", "event_type": "CAMERA_HEALTH", "occurred_at": "2026-09-18T01:00:00Z"})
    body = build_sync_request(outbox, outbox.pending())
    signature = sign_request(body, secret)
    first = sync_service.ingest_batch(
        body, signature, header_scope={"device": "edge-1", "site": "site-1", "organization": "org-1"},
    )
    second = sync_service.ingest_batch(
        body, signature, header_scope={"device": "edge-1", "site": "site-1", "organization": "org-1"},
    )
    assert first["accepted"] is True and first["idempotent"] is False
    assert second["idempotent"] is True
    assert database.fetch_one("select count(*) as count from edge_sync_events")["count"] == 1

    forged = dict(body)
    forged["records"] = [dict(body["records"][0], payload={"event_type": "FORGED"})]
    with pytest.raises(SyncIngestError):
        sync_service.ingest_batch(
            forged, signature, header_scope={"device": "edge-1", "site": "site-1", "organization": "org-1"}
        )


def test_backend_sync_ingest_rejects_non_object_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "platform.db")
    ensure_platform_schema()
    monkeypatch.setattr(sync_service, "SYNC_SECRET", "s" * 40)
    with pytest.raises(SyncIngestError, match="must be an object"):
        sync_service.ingest_batch([], "bad")


def test_backend_sync_registry_isolates_device_keys_and_scopes(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "platform.db")
    ensure_platform_schema()
    secret = "d" * 40
    monkeypatch.setattr(sync_service, "SYNC_SECRET", "")
    monkeypatch.setenv(
        "OPTIVOX_SYNC_DEVICE_REGISTRY_JSON",
        '{"edge-2":{"secret":"' + secret + '","site_id":"site-2","organization_id":"org-2"}}',
    )
    outbox = EdgeEventOutbox(tmp_path / "edge-2.ndjson", "edge-2", secret, site_id="site-2", organization_id="org-2")
    outbox.enqueue({"event_id": "evt-2", "event_type": "CAMERA_HEALTH"})
    body = build_sync_request(outbox, outbox.pending())
    result = sync_service.ingest_batch(body, sign_request(body, secret))
    assert result["accepted"] is True
    assert database.fetch_one(
        "select organization_id, site_id from edge_sync_events where event_id='evt-2'"
    ) == {"organization_id": "org-2", "site_id": "site-2"}

    forged = dict(body)
    forged["site_id"] = "site-1"
    with pytest.raises(SyncIngestError):
        sync_service.ingest_batch(forged, sign_request(forged, secret))


def test_revoked_device_cannot_ingest_and_status_change_is_audited(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "platform.db")
    ensure_platform_schema()
    secret = "r" * 40
    monkeypatch.setattr(sync_service, "SYNC_SECRET", "")
    monkeypatch.setenv(
        "OPTIVOX_SYNC_DEVICE_REGISTRY_JSON",
        '{"edge-revoked":{"secret":"' + secret + '","site_id":"site-r","organization_id":"org-r"}}',
    )
    outbox = EdgeEventOutbox(tmp_path / "edge-revoked.ndjson", "edge-revoked", secret, site_id="site-r", organization_id="org-r")
    outbox.enqueue({"event_id": "evt-r1", "event_type": "CAMERA_HEALTH"})
    body = build_sync_request(outbox, outbox.pending())
    sync_service.set_device_status("edge-revoked", "revoked", actor_id="admin")
    with pytest.raises(SyncIngestError, match="not active"):
        sync_service.ingest_batch(body, sign_request(body, secret))
    assert database.fetch_one("select status from edge_sync_devices where device_id='edge-revoked'")["status"] == "revoked"
    assert database.fetch_one("select action from platform_audit_log where action='sync.device_status'")
