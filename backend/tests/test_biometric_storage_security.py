from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest
from fastapi import HTTPException

from backend import config, database
from backend.platform_schema import ensure_platform_schema
from backend.services import audit_service, event_service, storage_service
from biometric_storage import BiometricStorageError, decode_embedding_blob, encode_embedding_blob, load_face_database, save_face_database, secure_path_for
from schema_migrations import ensure_schema_migrations, verify_audit_chain


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
    database.execute("update platform_audit_log set details_json='tampered' where action='test.action'")
    report = audit_service.integrity_report()
    assert report["ok"] is False
    assert report["reason"] == "record_hash_mismatch"
