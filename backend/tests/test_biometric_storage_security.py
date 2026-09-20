from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from fastapi import HTTPException

from backend import config, database
from backend.platform_schema import ensure_platform_schema
from backend.services import audit_service, backup_scheduler, event_service, retention_service, storage_service
from biometric_storage import BiometricStorageError, decode_embedding_blob, encode_embedding_blob, load_face_database, save_face_database, secure_path_for
from schema_migrations import audit_record_hash, ensure_schema_migrations, start_schema_migration_run, verify_audit_chain


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "security.db")
    ensure_platform_schema()
    return tmp_path


def test_malformed_biometric_file_is_rejected(tmp_path):
    source = tmp_path / "face_db.pkl"
    secure = secure_path_for(source)
    secure.write_text("not-json", encoding="utf-8")
    with pytest.raises(BiometricStorageError):
        load_face_database(source)


def test_tampered_biometric_file_is_rejected(tmp_path):
    source = tmp_path / "face_db.pkl"
    save_face_database(source, {"Ada": {"embeddings": [np.ones(512, dtype=np.float32)]}})
    secure = secure_path_for(source)
    envelope = json.loads(secure.read_text(encoding="utf-8"))
    envelope["payload"]["people"]["Ada"]["embeddings"][0][0] = 99
    secure.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(BiometricStorageError, match="checksum"):
        load_face_database(source)


def test_biometric_database_can_use_encrypted_envelope(tmp_path, monkeypatch):
    monkeypatch.setenv("OPTIVOX_BIOMETRIC_KEY", "phase-three-test-key")
    source = tmp_path / "face_db.pkl"
    save_face_database(source, {"Ada": {"embeddings": [np.ones(4, dtype=np.float32)]}})
    envelope = json.loads(secure_path_for(source).read_text(encoding="utf-8"))
    assert envelope["encrypted"] is True
    assert "payload" not in envelope
    assert load_face_database(source)["Ada"]["embeddings"][0].shape == (4,)


def test_legacy_embedding_blob_boundary_supports_encryption(monkeypatch):
    monkeypatch.setenv("OPTIVOX_BIOMETRIC_KEY", "phase-three-test-key")
    original = np.arange(4, dtype=np.float32)
    encoded = encode_embedding_blob(original)
    assert encoded.startswith(b"OVX2")
    np.testing.assert_array_equal(decode_embedding_blob(encoded), original)


def test_legacy_biometric_pickle_migrates_without_generic_unpickle(tmp_path):
    source = tmp_path / "face_db.pkl"
    import pickle
    with source.open("wb") as handle:
        pickle.dump({"Ada": {"embeddings": [np.ones(4, dtype=np.float32)]}}, handle)
    loaded = load_face_database(source)
    assert loaded["Ada"]["embeddings"][0].shape == (4,)
    assert secure_path_for(source).exists()
    assert not source.exists()


def test_people_responses_do_not_expose_embeddings(isolated_db):
    person_id = database.execute("insert into people(name, role) values (?, ?)", ["Ada", "Student"])
    from backend.routes.people import person
    response = person(person_id)
    assert "face_embedding" not in json.dumps(response)
    assert "embeddings" not in json.dumps(response)


def test_event_responses_do_not_expose_local_evidence_paths(isolated_db):
    event_id = database.execute(
        "insert into events(event_type, timestamp, snapshot_path, evidence_path) values (?, ?, ?, ?)",
        ["DANGER", "2026-01-01T00:00:00", "C:/private/snapshots/a.jpg", "C:/private/snapshots/a.jpg"],
    )
    response = event_service.get_event(event_id)
    assert response["snapshot_url"].endswith(f"/events/{event_id}/snapshot")
    assert response["evidence_available"] is True
    assert response["evidence_path"] is None
    assert "C:/private" not in json.dumps(response)


def test_evidence_path_traversal_is_rejected(isolated_db, tmp_path, monkeypatch):
    monkeypatch.setattr(event_service, "SNAPSHOTS_DIR", tmp_path)
    event_id = database.execute(
        "insert into events(event_type, timestamp, snapshot_path) values (?, ?, ?)",
        ["DANGER", "2026-01-01T00:00:00", "../outside.jpg"],
    )
    with pytest.raises(HTTPException) as error:
        event_service.snapshot_response(event_id)
    assert error.value.status_code == 400


def test_evidence_checksum_mismatch_is_rejected(isolated_db, tmp_path, monkeypatch):
    monkeypatch.setattr(event_service, "SNAPSHOTS_DIR", tmp_path)
    evidence = tmp_path / "evidence.jpg"
    evidence.write_bytes(b"original")
    event_id = database.execute(
        "insert into events(event_type, timestamp, snapshot_path, evidence_checksum) values (?, ?, ?, ?)",
        ["DANGER", "2026-01-01T00:00:00", str(evidence), "00" * 32],
    )
    with pytest.raises(HTTPException) as error:
        event_service.snapshot_response(event_id)
    assert error.value.status_code == 409


def test_evidence_envelope_round_trip_and_stored_checksum(tmp_path, monkeypatch):
    monkeypatch.setattr(storage_service, "SNAPSHOTS_DIR", tmp_path)
    monkeypatch.setattr(storage_service, "EVIDENCE_ENCRYPTION_ENABLED", True)
    monkeypatch.setattr(storage_service, "EVIDENCE_ENCRYPTION_KEY", "evidence-key-" + "x" * 32)
    monkeypatch.setattr(storage_service, "EVIDENCE_ENCRYPTION_KEY_ID", "primary")
    monkeypatch.setattr(storage_service, "EVIDENCE_ENCRYPTION_KEYS_JSON", "")
    path = storage_service.write_evidence_bytes(b"private-jpeg-bytes", tmp_path / "evidence.jpg")
    assert path.read_bytes().startswith(storage_service.EVIDENCE_MAGIC)
    checksum = storage_service.sha256_file(path)
    assert storage_service.read_evidence_bytes(path, expected_checksum=checksum) == b"private-jpeg-bytes"


def test_evidence_tampering_and_wrong_key_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(storage_service, "SNAPSHOTS_DIR", tmp_path)
    monkeypatch.setattr(storage_service, "EVIDENCE_ENCRYPTION_ENABLED", True)
    monkeypatch.setattr(storage_service, "EVIDENCE_ENCRYPTION_KEY", "evidence-key-" + "x" * 32)
    path = storage_service.write_evidence_bytes(b"secret", tmp_path / "evidence.jpg")
    original_checksum = storage_service.sha256_file(path)
    path.write_bytes(path.read_bytes()[:-1] + bytes([path.read_bytes()[-1] ^ 1]))
    with pytest.raises(storage_service.StorageSecurityError, match="checksum"):
        storage_service.read_evidence_bytes(path, expected_checksum=original_checksum)
    monkeypatch.setattr(storage_service, "EVIDENCE_ENCRYPTION_KEY", "different-key-" + "y" * 32)
    with pytest.raises(storage_service.StorageSecurityError, match="authentication"):
        storage_service.read_evidence_bytes(path)


def test_evidence_key_rotation_reads_previous_key(tmp_path, monkeypatch):
    monkeypatch.setattr(storage_service, "SNAPSHOTS_DIR", tmp_path)
    monkeypatch.setattr(storage_service, "EVIDENCE_ENCRYPTION_ENABLED", True)
    old_key = "old-evidence-key-" + "a" * 32
    new_key = "new-evidence-key-" + "b" * 32
    monkeypatch.setattr(storage_service, "EVIDENCE_ENCRYPTION_KEY", old_key)
    monkeypatch.setattr(storage_service, "EVIDENCE_ENCRYPTION_KEY_ID", "previous")
    monkeypatch.setattr(storage_service, "EVIDENCE_ENCRYPTION_KEYS_JSON", json.dumps({"previous": old_key}))
    path = storage_service.write_evidence_bytes(b"rotated-evidence", tmp_path / "rotated.jpg")
    monkeypatch.setattr(storage_service, "EVIDENCE_ENCRYPTION_KEY", new_key)
    monkeypatch.setattr(storage_service, "EVIDENCE_ENCRYPTION_KEY_ID", "primary")
    monkeypatch.setattr(storage_service, "EVIDENCE_ENCRYPTION_KEYS_JSON", json.dumps({"previous": old_key}))
    assert storage_service.read_evidence_bytes(path) == b"rotated-evidence"


def test_production_rejects_plaintext_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(storage_service, "SNAPSHOTS_DIR", tmp_path)
    monkeypatch.setattr(storage_service, "EVIDENCE_ENCRYPTION_ENABLED", True)
    monkeypatch.setattr(storage_service, "RUNTIME_MODE", "production")
    path = tmp_path / "legacy.jpg"
    path.write_bytes(b"plaintext")
    with pytest.raises(storage_service.StorageSecurityError, match="Plaintext evidence"):
        storage_service.read_evidence_bytes(path)


def test_evidence_retention_preserves_incident_references(tmp_path, monkeypatch):
    monkeypatch.setattr(storage_service, "SNAPSHOTS_DIR", tmp_path)
    protected = tmp_path / "incident.jpg"
    disposable = tmp_path / "disposable.jpg"
    protected.write_bytes(b"incident")
    disposable.write_bytes(b"old")
    old_timestamp = (datetime.now(timezone.utc) - timedelta(days=3)).timestamp()
    os.utime(protected, (old_timestamp, old_timestamp))
    os.utime(disposable, (old_timestamp, old_timestamp))
    result = storage_service.purge_expired_evidence(1, protected_paths=[protected])
    assert result == {"deleted": 1, "protected": 1, "failed": 0}
    assert protected.exists()
    assert not disposable.exists()


def test_backup_and_restore_validates_database(tmp_path, monkeypatch):
    source = tmp_path / "source.db"
    target = tmp_path / "restored.db"
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    monkeypatch.setattr(storage_service, "BACKUPS_DIR", backup_dir)
    with sqlite3.connect(source) as con:
        con.execute("create table sample(value text)")
        con.execute("insert into sample values ('before')")
    result = storage_service.backup_database(source, backup_dir / "sample.sqlite3")
    with sqlite3.connect(source) as con:
        con.execute("update sample set value='changed'")
    storage_service.restore_database(result["path"], target)
    with sqlite3.connect(target) as con:
        assert con.execute("select value from sample").fetchone()[0] == "before"


def test_encrypted_backup_round_trip_and_inventory_metadata(tmp_path, monkeypatch):
    source = tmp_path / "source.db"
    target = tmp_path / "restored.db"
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    monkeypatch.setattr(storage_service, "BACKUPS_DIR", backup_dir)
    monkeypatch.setattr(storage_service, "BACKUP_ENCRYPTION_ENABLED", True)
    monkeypatch.setattr(storage_service, "BACKUP_ENCRYPTION_KEY", "backup-encryption-test-key")
    with sqlite3.connect(source) as con:
        con.execute("create table sample(value text)")
        con.execute("insert into sample values ('confidential')")

    result = storage_service.backup_database(source, backup_dir / "encrypted.sqlite3")
    raw = (backup_dir / "encrypted.sqlite3").read_bytes()
    assert raw.startswith(storage_service.BACKUP_MAGIC)
    assert b"confidential" not in raw
    manifest = json.loads((backup_dir / "encrypted.sqlite3.manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == 3
    assert manifest["encrypted"] is True
    assert storage_service.list_backups()["items"][0]["encrypted"] is True

    storage_service.restore_database(result["path"], target)
    with sqlite3.connect(target) as con:
        assert con.execute("select value from sample").fetchone()[0] == "confidential"


def test_encrypted_backup_rejects_tampering_and_wrong_key(tmp_path, monkeypatch):
    source = tmp_path / "source.db"
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    monkeypatch.setattr(storage_service, "BACKUPS_DIR", backup_dir)
    monkeypatch.setattr(storage_service, "BACKUP_ENCRYPTION_ENABLED", True)
    monkeypatch.setattr(storage_service, "BACKUP_ENCRYPTION_KEY", "backup-encryption-test-key")
    with sqlite3.connect(source) as con:
        con.execute("create table sample(value text)")
        con.execute("insert into sample values ('secret')")
    result = storage_service.backup_database(source, backup_dir / "tamper.sqlite3")

    encrypted = bytearray((backup_dir / "tamper.sqlite3").read_bytes())
    encrypted[-1] ^= 0x01
    (backup_dir / "tamper.sqlite3").write_bytes(encrypted)
    with pytest.raises(storage_service.StorageSecurityError, match="checksum"):
        storage_service.restore_database(result["path"], tmp_path / "tampered.db")

    # Restore the authenticated bytes and prove the key is required to decrypt.
    storage_service.backup_database(source, backup_dir / "wrong-key.sqlite3")
    monkeypatch.setattr(storage_service, "BACKUP_ENCRYPTION_KEY", "a-different-key")
    with pytest.raises(storage_service.StorageSecurityError, match="authentication"):
        storage_service.restore_database(backup_dir / "wrong-key.sqlite3", tmp_path / "wrong-key.db")


def test_encrypted_backup_key_rotation_keeps_previous_backups_recoverable(tmp_path, monkeypatch):
    source = tmp_path / "source.db"
    restored = tmp_path / "restored.db"
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    monkeypatch.setattr(storage_service, "BACKUPS_DIR", backup_dir)
    monkeypatch.setattr(storage_service, "BACKUP_ENCRYPTION_ENABLED", True)
    old_key = "old-backup-encryption-key-" + "o" * 20
    new_key = "new-backup-encryption-key-" + "n" * 20
    monkeypatch.setattr(storage_service, "BACKUP_ENCRYPTION_KEY", old_key)
    monkeypatch.setattr(storage_service, "BACKUP_ENCRYPTION_KEY_ID", "previous")
    monkeypatch.setattr(storage_service, "BACKUP_ENCRYPTION_KEYS_JSON", json.dumps({"previous": old_key}))
    with sqlite3.connect(source) as con:
        con.execute("create table sample(value text)")
        con.execute("insert into sample values ('rotatable')")
    old_backup = storage_service.backup_database(source, backup_dir / "previous.sqlite3")

    monkeypatch.setattr(storage_service, "BACKUP_ENCRYPTION_KEY", new_key)
    monkeypatch.setattr(storage_service, "BACKUP_ENCRYPTION_KEY_ID", "primary")
    monkeypatch.setattr(
        storage_service,
        "BACKUP_ENCRYPTION_KEYS_JSON",
        json.dumps({"primary": new_key, "previous": old_key}),
    )
    storage_service.restore_database(old_backup["path"], restored)
    with sqlite3.connect(restored) as con:
        assert con.execute("select value from sample").fetchone()[0] == "rotatable"
    current = storage_service.backup_database(source, backup_dir / "primary.sqlite3")
    current_manifest = json.loads(Path(current["manifest"]).read_text(encoding="utf-8"))
    assert current_manifest["key_id"] == "primary"
    assert all(item["key_available"] for item in storage_service.list_backups()["items"])


def test_restore_creates_a_signed_safety_backup_of_existing_target(tmp_path, monkeypatch):
    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    monkeypatch.setattr(storage_service, "BACKUPS_DIR", backup_dir)
    monkeypatch.setattr(storage_service, "BACKUP_SIGNING_KEY", "r" * 40)
    with sqlite3.connect(source) as con:
        con.execute("create table sample(value text)")
        con.execute("insert into sample values ('new')")
    target_con = sqlite3.connect(target)
    try:
        target_con.execute("create table sample(value text)")
        target_con.execute("insert into sample values ('old')")
        target_con.commit()
    finally:
        target_con.close()
    source_backup = storage_service.backup_database(source, backup_dir / "source.sqlite3")
    restored = storage_service.restore_database(source_backup["path"], target)
    assert restored["rollback_backup"] is not None
    rollback = restored["rollback_backup"]["path"]
    assert storage_service.list_backups()["valid"] == 2
    with sqlite3.connect(rollback) as con:
        assert con.execute("select value from sample").fetchone()[0] == "old"


def test_backup_manifest_preserves_schema_state(tmp_path, monkeypatch):
    source = tmp_path / "source.db"
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    monkeypatch.setattr(storage_service, "BACKUPS_DIR", backup_dir)
    with sqlite3.connect(source) as con:
        con.execute("create table schema_migration_state(schema_name text, version integer, description text, checksum text, applied_at text)")
        con.execute("insert into schema_migration_state values ('platform', 2, 'scope', 'abc', 'now')")
    result = storage_service.backup_database(source, backup_dir / "state.sqlite3")
    manifest = json.loads((backup_dir / "state.sqlite3.manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == 2
    assert {item["schema_name"] for item in manifest["schema_states"]} == {"platform"}


def test_signed_backup_manifest_is_verified_and_tamper_evident(tmp_path, monkeypatch):
    source = tmp_path / "source.db"
    target = tmp_path / "restored.db"
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    monkeypatch.setattr(storage_service, "BACKUPS_DIR", backup_dir)
    monkeypatch.setattr(storage_service, "BACKUP_SIGNING_KEY", "x" * 40)
    with sqlite3.connect(source) as con:
        con.execute("create table sample(value text)")
        con.execute("insert into sample values ('signed')")
    result = storage_service.backup_database(source, backup_dir / "signed.sqlite3")
    manifest_path = backup_dir / "signed.sqlite3.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["integrity_mode"] == "signed"
    assert len(manifest["signature"]) == 64
    storage_service.restore_database(result["path"], target)

    manifest["integrity_mode"] = "tampered"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(storage_service.StorageSecurityError, match="signature"):
        storage_service.restore_database(result["path"], tmp_path / "tampered-restore.db")


def test_production_rejects_unsigned_backup(tmp_path, monkeypatch):
    source = tmp_path / "source.db"
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    monkeypatch.setattr(storage_service, "BACKUPS_DIR", backup_dir)
    monkeypatch.setattr(storage_service, "BACKUP_SIGNING_KEY", "")
    monkeypatch.setattr(storage_service, "RUNTIME_MODE", "development")
    with sqlite3.connect(source) as con:
        con.execute("create table sample(value text)")
    result = storage_service.backup_database(source, backup_dir / "unsigned.sqlite3")
    monkeypatch.setattr(storage_service, "RUNTIME_MODE", "production")
    with pytest.raises(storage_service.StorageSecurityError, match="Unsigned backups"):
        storage_service.restore_database(result["path"], tmp_path / "restore.db")


def test_backup_inventory_reports_integrity_without_exposing_paths(tmp_path, monkeypatch):
    source = tmp_path / "source.db"
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    monkeypatch.setattr(storage_service, "BACKUPS_DIR", backup_dir)
    monkeypatch.setattr(storage_service, "BACKUP_SIGNING_KEY", "x" * 40)
    with sqlite3.connect(source) as con:
        con.execute("create table sample(value text)")
    storage_service.backup_database(source, backup_dir / "inventory.sqlite3")
    inventory = storage_service.list_backups()
    assert inventory["valid"] == 1
    assert inventory["items"][0]["name"] == "inventory.sqlite3"
    assert inventory["items"][0]["signed"] is True
    assert "path" not in inventory["items"][0]


def test_backup_retention_is_dry_run_first_and_keeps_safety_floor(tmp_path, monkeypatch):
    source = tmp_path / "source.db"
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    monkeypatch.setattr(storage_service, "BACKUPS_DIR", backup_dir)
    monkeypatch.setattr(storage_service, "BACKUP_SIGNING_KEY", "b" * 40)
    with sqlite3.connect(source) as con:
        con.execute("create table sample(value text)")
        con.execute("insert into sample values ('backup')")
    paths = []
    for name in ("old.sqlite3", "middle.sqlite3", "new.sqlite3"):
        result = storage_service.backup_database(source, backup_dir / name)
        paths.append(result["path"])
    old_timestamp = 1_600_000_000
    os.utime(paths[0], (old_timestamp, old_timestamp))
    os.utime(str(paths[0]) + ".manifest.json", (old_timestamp, old_timestamp))
    preview = storage_service.purge_backups(1, 2, execute_delete=False)
    assert preview["dry_run"] is True
    assert preview["eligible_count"] == 1
    assert preview["eligible"][0]["name"] == "old.sqlite3"
    result = storage_service.purge_backups(1, 2, execute_delete=True)
    assert result["deleted"] == ["old.sqlite3"]
    assert (backup_dir / "middle.sqlite3").exists()
    assert (backup_dir / "new.sqlite3").exists()


def test_backup_scheduler_reports_success_without_leaking_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(
        backup_scheduler,
        "backup_database",
        lambda: {"path": str(tmp_path / "security.sqlite3"), "sha256": "a" * 64},
    )
    monkeypatch.setattr(
        backup_scheduler,
        "purge_backups",
        lambda *args, **kwargs: {"deleted_count": 2},
    )
    scheduler = backup_scheduler.BackupScheduler(
        interval_minutes=5,
        retention_days=30,
        retention_count=2,
        status_path=tmp_path / "backup_status.json",
    )
    result = scheduler.run_once()
    assert result["status"] == "success"
    assert result["backup_name"] == "security.sqlite3"
    assert scheduler.snapshot()["last_deleted_count"] == 2
    assert "security.sqlite3" in (tmp_path / "backup_status.json").read_text(encoding="utf-8")


def test_restore_rejects_backup_without_manifest(tmp_path, monkeypatch):
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    monkeypatch.setattr(storage_service, "BACKUPS_DIR", backup_dir)
    backup = backup_dir / "unverified.sqlite3"
    with sqlite3.connect(backup) as con:
        con.execute("create table sample(value text)")
    with pytest.raises(storage_service.StorageSecurityError, match="manifest"):
        storage_service.restore_database(backup, tmp_path / "restored.db")


def test_migration_compatibility_adds_audit_chain_to_legacy_database(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as con:
        con.execute("create table audit_log(id integer primary key, action text, target text, details_json text, timestamp text)")
        con.execute("insert into audit_log values(1, 'legacy', 'x', '{}', '2026-01-01T00:00:00')")
        ensure_schema_migrations(con)
        assert verify_audit_chain(con, "audit_log")["ok"]
        assert con.execute("select version from schema_migrations").fetchone()[0] == 1


def test_audit_integrity_failure_is_detected(isolated_db):
    audit_service.record_action("test.action", "test", "1", {"safe": True}, actor_id="tester")
    assert audit_service.integrity_report()["ok"]
    database.execute("drop trigger trg_platformauditlog_append_only_update")
    database.execute("drop trigger trg_scope_immutable_platform_audit_log")
    database.execute("update platform_audit_log set details_json='tampered' where action='test.action'")
    report = audit_service.integrity_report()
    assert report["ok"] is False
    assert report["reason"] == "record_hash_mismatch"


def test_legacy_audit_hash_contract_is_preserved_and_new_rows_are_scope_bound(isolated_db):
    # Simulate a row written before scope columns were introduced. The
    # migration must verify it without rewriting the historical hash.
    with database.get_connection() as con:
        con.execute(
            """insert into platform_audit_log
               (action, entity_type, actor_type, details_json, prev_hash, record_hash, hash_version)
               values (?, ?, ?, ?, ?, ?, ?)""",
            ["legacy.audit", "test", "system", "{}", "", None, 1],
        )
        row = dict(con.execute(
            "select * from platform_audit_log where action='legacy.audit'"
        ).fetchone())
        con.execute(
            "update platform_audit_log set record_hash=? where id=?",
            [audit_record_hash(row, ""), row["id"]],
        )
        con.commit()

    legacy = audit_service.integrity_report()
    assert legacy["ok"] is True
    assert legacy["integrity_level"] == "legacy_compatible"
    assert legacy["legacy_unscoped_records"] == 1

    audit_service.record_action("scope.bound", "test", "1", {"safe": True}, actor_id="tester")
    report = audit_service.integrity_report()
    assert report["ok"] is True
    assert report["hash_versions"]["1"] == 1
    assert report["hash_versions"]["3"] == 1

    database.execute("drop trigger trg_platformauditlog_append_only_update")
    database.execute("drop trigger trg_scope_immutable_platform_audit_log")
    database.execute(
        "update platform_audit_log set site_id='tampered' where action='scope.bound'"
    )
    tampered = audit_service.integrity_report()
    assert tampered["ok"] is False
    assert tampered["reason"] == "record_hash_mismatch"


def test_audit_records_are_append_only_at_database_boundary(isolated_db):
    audit_service.record_action("append.only", "test", "1", {"safe": True}, actor_id="tester")
    with pytest.raises(database.DatabaseError, match="append-only audit record"):
        database.execute("update platform_audit_log set details_json='tampered' where action='append.only'")
    with pytest.raises(database.DatabaseError, match="append-only audit record"):
        database.execute("delete from platform_audit_log where action='append.only'")


def test_operational_history_is_append_only_at_database_boundary(isolated_db):
    person_id = database.execute("insert into people(name, role) values (?, ?)", ["History Test", "Student"])
    attendance_id = database.execute(
        "insert into attendance(person_id, date) values (?, ?)", [person_id, "2026-09-18"]
    )
    correction_id = database.execute(
        """insert into attendance_corrections
           (attendance_id, person_id, attendance_date, before_json, after_json, reason)
           values (?, ?, ?, ?, ?, ?)""",
        [attendance_id, person_id, "2026-09-18", "{}", "{}", "test correction"],
    )
    incident_id = database.execute(
        """insert into incidents(status, category, severity, summary, first_event_at, last_event_at)
           values ('open', 'test', 1, 'history test', '2026-09-18', '2026-09-18')"""
    )
    review_id = database.execute(
        "insert into incident_review_actions(incident_id, action, note) values (?, ?, ?)",
        [incident_id, "acknowledge", "test"],
    )
    alert_id = database.execute(
        "insert into incident_alerts(incident_id, channel, status) values (?, ?, ?)",
        [incident_id, "test", "sent"],
    )
    for table, row_id in (
        ("attendance_corrections", correction_id),
        ("incident_review_actions", review_id),
        ("incident_alerts", alert_id),
    ):
        with pytest.raises(database.DatabaseError, match="append-only"):
            database.execute(f"update {table} set id=id where id=?", [row_id])
        with pytest.raises(database.DatabaseError, match="append-only"):
            database.execute(f"delete from {table} where id=?", [row_id])


def test_migration_history_is_append_only_and_auditable(isolated_db):
    history = database.fetch_one(
        "select schema_name, version, checksum from schema_migration_history order by id desc limit 1"
    )
    assert history["schema_name"] in {"platform", "outbox"}
    with pytest.raises(database.DatabaseError, match="append-only migration history"):
        database.execute(
            "update schema_migration_history set status='tampered' where schema_name=?",
            [history["schema_name"]],
        )
    with pytest.raises(database.DatabaseError, match="append-only migration history"):
        database.execute(
            "delete from schema_migration_history where schema_name=?",
            [history["schema_name"]],
        )


def test_concurrent_audit_writers_preserve_the_hash_chain(isolated_db):
    def write(index: int) -> None:
        audit_service.record_action(
            "concurrent.audit", "test", str(index), {"index": index}, actor_id=f"worker-{index}"
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write, range(24)))
    report = audit_service.integrity_report()
    assert report["ok"] is True
    assert database.fetch_one(
        "select count(*) as count from platform_audit_log where action='concurrent.audit'"
    )["count"] == 24


def test_signed_audit_checkpoint_detects_checkpoint_tampering(isolated_db, monkeypatch):
    monkeypatch.setenv("OPTIVOX_AUDIT_SIGNING_KEY", "audit-key-" + "x" * 32)
    audit_service.record_action("signed.audit", "test", "1", {"safe": True}, actor_id="tester")
    report = audit_service.integrity_report()
    assert report["ok"] is True
    assert report["checkpoint_status"] == "valid"
    database.execute("drop trigger trg_auditcheckpoints_append_only_update")
    database.execute(
        "update audit_checkpoints set signature=? where table_name='platform_audit_log'",
        ["tampered"],
    )
    report = audit_service.integrity_report()
    assert report["ok"] is False
    assert report["reason"] == "checkpoint_signature_mismatch"


def test_explicit_transaction_rolls_back_all_writes(tmp_path):
    path = tmp_path / "transaction.db"
    with pytest.raises(RuntimeError):
        with database.transaction(path, immediate=True) as con:
            con.execute("create table sample(value text)")
            con.execute("insert into sample values ('uncommitted')")
            raise RuntimeError("abort")
    with sqlite3.connect(path) as con:
        assert con.execute(
            "select name from sqlite_master where type='table' and name='sample'"
        ).fetchone() is None


def test_database_health_reports_integrity_and_migrations(isolated_db):
    health = database.database_health(include_counts=True)
    assert health["status"] == "healthy"
    assert health["quick_check"] == "ok"
    assert health["foreign_key_violations"] == 0
    assert health["journal_mode"].lower() == "wal"
    assert health["migration_version"] == 1
    states = {item["schema_name"]: item for item in health["migration_states"]}
    assert states["platform"]["version"] == 17
    assert states["outbox"]["version"] == 3
    assert len(states["platform"]["checksum"]) == 64
    assert "people" in health["tables"]
    assert health["schema_issues"] == []
    assert health["migration_state_issues"] == []
    assert health["migration_run_issues"] == []
    assert health["migration_runs"][0]["status"] == "applied"
    assert health["audit"]["integrity_level"] == "scope_bound"
    assert health["page_count"] > 0
    assert "wal_size_bytes" in health


def test_deployment_scope_is_immutable(isolated_db):
    person_id = database.execute("insert into people(name, role) values (?, ?)", ["Scoped Person", "Student"])
    with pytest.raises(database.DatabaseError, match="deployment scope is immutable"):
        database.execute("update people set site_id='other-site' where id=?", [person_id])

    incident_id = database.execute(
        """insert into incidents(status, category, severity, summary, first_event_at, last_event_at)
           values ('open', 'scope-test', 1, 'scope test', '2026-09-18', '2026-09-18')"""
    )
    with pytest.raises(database.DatabaseError, match="deployment scope is immutable"):
        database.execute("update incidents set organization_id='other-org' where id=?", [incident_id])


def test_database_domain_guards_reject_impossible_operational_states(isolated_db):
    person_id = database.execute("insert into people(name, role) values (?, ?)", ["Domain Person", "Student"])
    with pytest.raises(database.DatabaseError, match="invalid attendance.attendance_status"):
        database.execute(
            "insert into attendance(person_id, date, attendance_status) values (?, ?, ?)",
            [person_id, "2026-09-18", "not-a-status"],
        )
    with pytest.raises(database.DatabaseError, match="attendance durations cannot be negative"):
        database.execute(
            "insert into attendance(person_id, date, work_minutes) values (?, ?, ?)",
            [person_id, "2026-09-19", -1],
        )
    with pytest.raises(database.DatabaseError, match="attendance clock-out precedes clock-in"):
        database.execute(
            "insert into attendance(person_id, date, clock_in, clock_out) values (?, ?, ?, ?)",
            [person_id, "2026-09-20", "2026-09-20T10:00:00+00:00", "2026-09-20T09:00:00+00:00"],
        )
    with pytest.raises(database.DatabaseError, match="invalid presence_sessions.identity_state"):
        database.execute(
            "insert into presence_sessions(entity_id, started_at, last_seen_at, identity_state) values (?, ?, ?, ?)",
            ["entity-domain", "2026-09-18T10:00:00+00:00", "2026-09-18T10:00:01+00:00", "IMPOSSIBLE"],
        )


def test_migration_run_ledger_is_append_only(isolated_db):
    run = database.fetch_one(
        "select run_id from schema_migration_runs order by id desc limit 1"
    )
    with pytest.raises(database.DatabaseError, match="append-only migration run"):
        database.execute(
            "update schema_migration_runs set status='failed' where run_id=?",
            [run["run_id"]],
        )
    with pytest.raises(database.DatabaseError, match="append-only migration run"):
        database.execute(
            "delete from schema_migration_runs where run_id=?",
            [run["run_id"]],
        )


def test_incomplete_migration_run_degrades_health(isolated_db):
    with database.get_connection() as con:
        start_schema_migration_run(con, "platform", 999)
        con.commit()
    health = database.database_health()
    assert health["status"] == "degraded"
    assert any("is started" in issue for issue in health["migration_run_issues"])


def test_database_maintenance_runs_explicit_integrity_and_checkpoint(isolated_db):
    result = database.database_maintenance(full_integrity=True, checkpoint=True, optimize=True)
    assert result["integrity_check"] == "ok"
    assert result["wal_checkpoint"]["busy"] in (0, 1)


def test_database_maintenance_can_repair_missing_schema_protection(isolated_db):
    database.execute("drop trigger trg_scope_events_session")
    assert any("trg_scope_events_session" in issue for issue in database.database_health()["schema_object_issues"])
    result = database.database_maintenance(repair_schema=True, optimize=False)
    assert result["schema_repaired"] is True
    assert result["health"]["schema_object_issues"] == []
    assert result["health"]["status"] == "healthy"


def test_retention_preview_and_purge_protect_incident_linked_telemetry(isolated_db):
    old_unlinked = database.execute(
        "insert into events(event_type, timestamp) values (?, ?)",
        ["LOW_LEVEL", "2000-01-01T00:00:00+00:00"],
    )
    old_linked = database.execute(
        "insert into events(event_type, timestamp) values (?, ?)",
        ["SECURITY", "2000-01-01T00:00:00+00:00"],
    )
    incident_id = database.execute(
        """insert into incidents(status, category, severity, summary, first_event_at, last_event_at)
           values ('open', 'security', 1, 'retention test', '2000-01-01', '2000-01-01')""",
    )
    database.execute("insert into incident_events(incident_id, event_id) values (?, ?)", [incident_id, old_linked])
    preview = retention_service.preview(1)
    assert preview["dry_run"] is True
    assert any(item["table"] == "events" and item["eligible"] == 1 for item in preview["items"])
    result = retention_service.purge(1, actor_id="admin")
    assert result["deleted"]["events"] == 1
    assert database.fetch_one("select id from events where id=?", [old_unlinked]) is None
    assert database.fetch_one("select id from events where id=?", [old_linked]) is not None
    assert database.fetch_one("select action from platform_audit_log where action='database.retention_purge'")


def test_database_health_distinguishes_sqlite_open_from_schema_compatibility(tmp_path):
    path = tmp_path / "incomplete.db"
    with sqlite3.connect(path) as con:
        con.execute("create table people(id integer primary key, name text)")
    health = database.database_health(path)
    assert health["connected"] is True
    assert health["quick_check"] == "ok"
    assert health["status"] == "degraded"
    assert any("missing table" in issue for issue in health["schema_issues"])


def test_platform_schema_upgrade_refreshes_structural_fingerprint(isolated_db):
    database.execute(
        "update schema_migration_state set version=2, checksum=?, schema_fingerprint='legacy' where schema_name='platform'",
        ["old-platform-checksum"],
    )
    ensure_platform_schema()
    health = database.database_health()
    assert health["migration_state_issues"] == []
    state = database.fetch_one("select version, schema_fingerprint from schema_migration_state where schema_name='platform'")
    assert state["version"] == 17
    assert len(state["schema_fingerprint"]) == 64


def test_database_health_detects_semantic_duplicate_presence(isolated_db):
    # Bypass the production invariant only to simulate an already-corrupted
    # legacy database and verify that health diagnostics still detect it.
    database.execute("drop index idx_presence_one_open_per_entity")
    for _ in range(2):
        database.execute(
            """insert into presence_sessions
               (entity_id, camera_id, started_at, last_seen_at, status)
               values (?, ?, ?, ?, 'active')""",
            ["cam_0:entity:duplicate", "cam_0", "2026-09-18T10:00:00", "2026-09-18T10:00:01"],
        )
    health = database.database_health()
    assert health["status"] == "degraded"
    assert any("duplicate active presence sessions" in issue for issue in health["consistency_issues"])


def test_database_rejects_two_open_presence_sessions_for_one_entity(isolated_db):
    database.execute(
        """insert into presence_sessions
           (entity_id, camera_id, started_at, last_seen_at, status,
            organization_id, site_id, device_id)
           values (?, ?, ?, ?, 'active', ?, ?, ?)""",
        [
            "cam_0:entity:unique", "cam_0", "2026-09-18T10:00:00",
            "2026-09-18T10:00:01", "local-organization", "local-site", "local-edge-cam-0",
        ],
    )
    with pytest.raises(database.DatabaseError, match="UNIQUE constraint"):
        database.execute(
            """insert into presence_sessions
               (entity_id, camera_id, started_at, last_seen_at, status,
                organization_id, site_id, device_id)
               values (?, ?, ?, ?, 'occluded', ?, ?, ?)""",
            [
                "cam_0:entity:unique", "cam_0", "2026-09-18T10:00:02",
                "2026-09-18T10:00:03", "local-organization", "local-site", "local-edge-cam-0",
            ],
        )


def test_database_health_detects_schema_state_tampering(isolated_db):
    database.execute(
        "update schema_migration_state set checksum=? where schema_name='platform'",
        ["tampered"],
    )
    health = database.database_health()
    assert health["status"] == "degraded"
    assert "schema checksum mismatch: platform" in health["migration_state_issues"]


def test_database_health_detects_migration_ledger_tampering(isolated_db):
    database.execute(
        "update schema_migrations set checksum=? where version=1",
        ["tampered"],
    )
    health = database.database_health()
    assert health["status"] == "degraded"
    assert "migration checksum mismatch: 1" in health["migration_ledger_issues"]


def test_database_health_detects_outbox_ddl_drift(isolated_db):
    assert database.database_health()["migration_state_issues"] == []
    database.execute(
        "create index idx_test_unapproved_outbox_shape on platform_outbox(last_error)"
    )
    health = database.database_health()
    assert health["status"] == "degraded"
    assert "schema fingerprint mismatch: outbox" in health["migration_state_issues"]


def test_database_health_detects_cross_scope_attendance(isolated_db):
    person_id = database.execute(
        "insert into people(name, role) values (?, ?)", ["Scoped Ada", "Student"]
    )
    with database.get_connection() as con:
        con.execute("drop trigger trg_scope_attendance_person")
        con.execute(
            """insert into attendance
               (person_id, date, clock_in, organization_id, site_id, device_id)
               values (?, ?, datetime('now'), ?, ?, ?)""",
            [person_id, "2026-09-18", "other-org", "other-site", "other-device"],
        )
        con.commit()
    health = database.database_health()
    assert health["status"] == "degraded"
    assert any("attendance references a person from another scope" in issue for issue in health["consistency_issues"])


def test_database_health_detects_cross_scope_incident_event_link(isolated_db):
    event_id = database.execute(
        "insert into events(event_type, timestamp, organization_id, site_id, device_id) values (?, ?, ?, ?, ?)",
        ["ZONE_INTRUSION", "2026-09-18T10:00:00", "other-org", "other-site", "other-device"],
    )
    incident_id = database.execute(
        """insert into incidents(category, severity, summary, first_event_at, last_event_at,
           organization_id, site_id, device_id)
           values (?, ?, ?, ?, ?, ?, ?, ?)""",
        ["ZONE_INTRUSION", 2, "Scope mismatch", "2026-09-18T10:00:00", "2026-09-18T10:00:00", "local-organization", "local-site", "local-edge-cam-0"],
    )
    with database.get_connection() as con:
        con.execute("drop trigger trg_scope_incident_event")
        con.execute(
            "insert into incident_events(incident_id, event_id, organization_id, site_id, device_id) values (?, ?, ?, ?, ?)",
            [incident_id, event_id, "local-organization", "local-site", "local-edge-cam-0"],
        )
        con.commit()
    health = database.database_health()
    assert health["status"] == "degraded"
    assert any("incident event link crosses deployment scope" in issue for issue in health["consistency_issues"])


def test_database_health_detects_cross_scope_person_and_evidence_links(isolated_db):
    person_id = database.execute(
        "insert into people(name, role, metadata_json, organization_id, site_id, device_id) values (?, ?, ?, ?, ?, ?)",
        ["Scoped Student", "student", "{}", "org-a", "site-a", "edge-a"],
    )
    # Simulate a damaged legacy/import database by removing only the write
    # guards before inserting invalid rows; runtime health must still detect it.
    with database.get_connection() as con:
        con.execute("drop trigger trg_scope_absence_person")
        con.execute("drop trigger trg_scope_presence_person")
        con.execute(
            "insert into absence_records(person_id, absence_date, subject, organization_id, site_id, device_id) values (?, ?, ?, ?, ?, ?)",
            [person_id, "2026-09-18", "Math", "org-b", "site-b", "edge-b"],
        )
        con.execute(
            "insert into presence_sessions(entity_id, person_id, camera_id, started_at, last_seen_at, organization_id, site_id, device_id) values (?, ?, ?, ?, ?, ?, ?, ?)",
            ["cam:entity:scope", person_id, "cam-a", "2026-09-18T10:00:00", "2026-09-18T10:00:01", "org-b", "site-b", "edge-b"],
        )
        con.commit()
    health = database.database_health()
    assert health["status"] == "degraded"
    assert any("absence references a person from another scope" in issue for issue in health["consistency_issues"])
    assert any("presence session references a person from another scope" in issue for issue in health["consistency_issues"])


def test_scope_triggers_reject_cross_site_writes(isolated_db):
    person_id = database.execute(
        "insert into people(name, role, metadata_json, organization_id, site_id, device_id) values (?, ?, ?, ?, ?, ?)",
        ["Protected Student", "student", "{}", "org-a", "site-a", "edge-a"],
    )
    with pytest.raises(database.DatabaseError, match="scope mismatch"):
        database.execute(
            "insert into attendance(person_id, date, organization_id, site_id, device_id) values (?, ?, ?, ?, ?)",
            [person_id, "2026-09-18", "org-b", "site-b", "edge-b"],
        )


def test_database_health_detects_legacy_domain_corruption(isolated_db):
    person_id = database.execute("insert into people(name, role) values (?, ?)", ["Legacy Ada", "Student"])
    with database.get_connection() as con:
        con.execute("drop trigger trg_domain_attendance_attendance_status_insert")
        con.execute("drop trigger trg_domain_attendance_attendance_status_update")
        con.execute("drop trigger trg_domain_attendance_time_insert")
        con.execute("drop trigger trg_domain_attendance_time_update")
        con.execute("drop trigger trg_domain_attendance_nonnegative_insert")
        con.execute("drop trigger trg_domain_attendance_nonnegative_update")
        con.execute(
            """insert into attendance
               (person_id, date, clock_in, clock_out, work_minutes, attendance_status)
               values (?, ?, ?, ?, ?, ?)""",
            [
                person_id,
                "2026-09-18",
                "2026-09-18T10:00:00+00:00",
                "2026-09-18T09:00:00+00:00",
                -4,
                "corrupted",
            ],
        )
        con.commit()
    health = database.database_health()
    assert health["status"] == "degraded"
    assert any("attendance clock-out precedes clock-in" in issue for issue in health["consistency_issues"])
    assert any("attendance contains negative durations" in issue for issue in health["consistency_issues"])
    assert any("attendance contains an unsupported status" in issue for issue in health["consistency_issues"])
