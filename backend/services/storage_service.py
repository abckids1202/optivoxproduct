"""Protected local storage helpers for evidence and SQLite backups."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import tempfile
import base64
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from ..config import (
    BACKUP_RETENTION_COUNT,
    BACKUP_RETENTION_DAYS,
    BACKUP_MAX_AGE_MINUTES,
    RUNTIME_MODE,
    BACKUP_ENCRYPTION_ENABLED,
    BACKUP_ENCRYPTION_KEY,
    BACKUP_ENCRYPTION_KEY_ID,
    BACKUP_ENCRYPTION_KEYS_JSON,
    EVIDENCE_ENCRYPTION_ENABLED,
    EVIDENCE_ENCRYPTION_KEY,
    EVIDENCE_ENCRYPTION_KEY_ID,
    EVIDENCE_ENCRYPTION_KEYS_JSON,
    BACKUP_SIGNING_KEY,
    BACKUPS_DIR,
    DATABASE_PATH,
    PROJECT_ROOT,
    SNAPSHOTS_DIR,
)
from ..database import connect_database


class StorageSecurityError(RuntimeError):
    pass


BACKUP_MAGIC = b"OVXDB1"
BACKUP_AAD = b"optivox-sqlite-backup-v1"
EVIDENCE_MAGIC = b"OVXE1"
EVIDENCE_AAD = b"optivox-evidence-v1"
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except ImportError:  # pragma: no cover - validated by strict startup configuration
    AESGCM = None


def _inside(path: Path, roots: Iterable[Path]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def safe_storage_path(raw: str | os.PathLike[str], *, roots: Iterable[Path] | None = None, base: Path | None = None) -> Path:
    value = str(raw or "")
    if not value or "\x00" in value:
        raise StorageSecurityError("Storage path is empty or invalid.")
    base_path = Path(base or SNAPSHOTS_DIR)
    candidate = Path(value)
    resolved = (candidate if candidate.is_absolute() else base_path / candidate).resolve()
    resolved_roots = tuple(Path(root).resolve() for root in (roots if roots is not None else (base_path,)))
    if not _inside(resolved, resolved_roots) or resolved in resolved_roots:
        raise StorageSecurityError("Storage path is outside the allowed directory.")
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path: Path, expected: str | None) -> bool:
    return bool(expected) and path.is_file() and hmac.compare_digest(sha256_file(path), str(expected))


def evidence_is_encrypted(path: Path) -> bool:
    """Identify an encrypted evidence envelope without decrypting it."""
    try:
        with path.open("rb") as handle:
            return handle.read(len(EVIDENCE_MAGIC)) == EVIDENCE_MAGIC
    except OSError:
        return False


def write_evidence_bytes(
    data: bytes,
    target: str | os.PathLike[str],
    *,
    enabled: bool | None = None,
) -> Path:
    """Atomically persist evidence, encrypting it when the runtime requires it."""
    path = safe_storage_path(target, base=SNAPSHOTS_DIR)
    payload = _encrypt_evidence_bytes(data) if (EVIDENCE_ENCRYPTION_ENABLED if enabled is None else enabled) else bytes(data)
    _write_bytes_atomic(payload, path)
    return path


def read_evidence_bytes(
    raw_path: str | os.PathLike[str],
    *,
    expected_checksum: str | None = None,
    base: Path | None = None,
    roots: Iterable[Path] | None = None,
) -> bytes:
    """Read and authenticate evidence without materializing a decrypted file."""
    path = safe_storage_path(raw_path, base=base, roots=roots)
    if not path.is_file() or path.is_symlink():
        raise StorageSecurityError("Evidence file does not exist.")
    if expected_checksum and not verify_checksum(path, expected_checksum):
        raise StorageSecurityError("Evidence checksum mismatch.")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise StorageSecurityError("Evidence file could not be read.") from exc
    if payload.startswith(EVIDENCE_MAGIC):
        return _decrypt_evidence_bytes(payload)
    if EVIDENCE_ENCRYPTION_ENABLED and RUNTIME_MODE in {"pilot", "production"}:
        raise StorageSecurityError("Plaintext evidence is not accepted in this runtime mode.")
    return payload


def validate_evidence_file(
    path: Path,
    expected_checksum: str | None = None,
    *,
    roots: Iterable[Path] | None = None,
) -> bool:
    """Verify both stored-file integrity and authenticated evidence contents."""
    read_evidence_bytes(path, expected_checksum=expected_checksum, roots=roots)
    return True


def _manifest_bytes(manifest: dict) -> bytes:
    unsigned = {key: value for key, value in manifest.items() if key != "signature"}
    return json.dumps(unsigned, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _manifest_signature(manifest: dict, key: str | None = None) -> str | None:
    secret = str(key if key is not None else BACKUP_SIGNING_KEY).encode("utf-8")
    if not secret:
        return None
    return hmac.new(secret, _manifest_bytes(manifest), hashlib.sha256).hexdigest()


def _evidence_key(key_id: str | None = None) -> bytes | None:
    """Resolve the active or a previous evidence key without exposing it."""
    keys = {str(EVIDENCE_ENCRYPTION_KEY_ID or "primary"): str(EVIDENCE_ENCRYPTION_KEY or "").strip()}
    if EVIDENCE_ENCRYPTION_KEYS_JSON:
        try:
            configured = json.loads(EVIDENCE_ENCRYPTION_KEYS_JSON)
        except (TypeError, ValueError, json.JSONDecodeError):
            configured = {}
        if isinstance(configured, dict):
            keys.update({str(item): str(value or "").strip() for item, value in configured.items()})
    raw = keys.get(str(key_id or EVIDENCE_ENCRYPTION_KEY_ID or "primary"), "")
    if not raw:
        return None
    try:
        padded = raw + ("=" * (-len(raw) % 4))
        decoded = base64.urlsafe_b64decode(padded.encode("ascii"))
        if len(decoded) == 32:
            return decoded
    except (ValueError, UnicodeEncodeError):
        pass
    return hashlib.sha256(raw.encode("utf-8")).digest()


def _encrypt_evidence_bytes(plaintext: bytes) -> bytes:
    key_id = str(EVIDENCE_ENCRYPTION_KEY_ID or "primary").strip()
    key_id_bytes = key_id.encode("utf-8")
    if not key_id_bytes or len(key_id_bytes) > 255:
        raise StorageSecurityError("Evidence encryption key ID is invalid.")
    key = _evidence_key(key_id)
    if key is None:
        raise StorageSecurityError("Encrypted evidence requires OPTIVOX_EVIDENCE_ENCRYPTION_KEY.")
    if AESGCM is None:
        raise StorageSecurityError("Encrypted evidence requires the cryptography package.")
    nonce = os.urandom(12)
    ciphertext = AESGCM(key).encrypt(nonce, bytes(plaintext), EVIDENCE_AAD + key_id_bytes)
    return EVIDENCE_MAGIC + bytes([len(key_id_bytes)]) + key_id_bytes + nonce + ciphertext


def _decrypt_evidence_bytes(ciphertext: bytes) -> bytes:
    if not ciphertext.startswith(EVIDENCE_MAGIC):
        raise StorageSecurityError("Evidence envelope is missing or invalid.")
    key_length_offset = len(EVIDENCE_MAGIC)
    if len(ciphertext) <= key_length_offset:
        raise StorageSecurityError("Evidence envelope is truncated.")
    key_length = ciphertext[key_length_offset]
    key_start = key_length_offset + 1
    nonce_start = key_start + key_length
    if key_length == 0 or len(ciphertext) <= nonce_start + 12:
        raise StorageSecurityError("Evidence envelope is truncated.")
    try:
        key_id = ciphertext[key_start:nonce_start].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise StorageSecurityError("Evidence envelope key ID is invalid.") from exc
    key = _evidence_key(key_id)
    if key is None:
        raise StorageSecurityError("Evidence encryption key ID is not configured.")
    if AESGCM is None:
        raise StorageSecurityError("Encrypted evidence requires the cryptography package.")
    nonce = ciphertext[nonce_start:nonce_start + 12]
    try:
        return AESGCM(key).decrypt(nonce, ciphertext[nonce_start + 12:], EVIDENCE_AAD + key_id.encode("utf-8"))
    except Exception as exc:
        raise StorageSecurityError("Evidence authentication failed.") from exc


def _atomic_replace(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, target)
    try:
        os.chmod(target, 0o600)
    except OSError:
        pass


def _backup_key(key_id: str | None = None) -> bytes | None:
    """Derive a stable 256-bit backup key without exposing key material."""
    keys = {str(BACKUP_ENCRYPTION_KEY_ID or "primary"): str(BACKUP_ENCRYPTION_KEY or "").strip()}
    if BACKUP_ENCRYPTION_KEYS_JSON:
        try:
            configured = json.loads(BACKUP_ENCRYPTION_KEYS_JSON)
        except (TypeError, ValueError, json.JSONDecodeError):
            configured = {}
        if isinstance(configured, dict):
            keys.update({str(item): str(value or "").strip() for item, value in configured.items()})
    raw = keys.get(str(key_id or BACKUP_ENCRYPTION_KEY_ID or "primary"), "")
    if not raw:
        return None
    try:
        padded = raw + ("=" * (-len(raw) % 4))
        decoded = base64.urlsafe_b64decode(padded.encode("ascii"))
        if len(decoded) == 32:
            return decoded
    except (ValueError, UnicodeEncodeError):
        pass
    return hashlib.sha256(raw.encode("utf-8")).digest()


def _encrypt_backup_bytes(plaintext: bytes) -> bytes:
    key = _backup_key(BACKUP_ENCRYPTION_KEY_ID)
    if key is None:
        raise StorageSecurityError("Encrypted backups require OPTIVOX_BACKUP_ENCRYPTION_KEY.")
    if AESGCM is None:
        raise StorageSecurityError("Encrypted backups require the cryptography package.")
    nonce = os.urandom(12)
    return BACKUP_MAGIC + nonce + AESGCM(key).encrypt(nonce, plaintext, BACKUP_AAD)


def _decrypt_backup_bytes(ciphertext: bytes, key_id: str | None = None) -> bytes:
    if not ciphertext.startswith(BACKUP_MAGIC):
        raise StorageSecurityError("Encrypted backup envelope is missing or invalid.")
    key = _backup_key(key_id)
    if key is None:
        raise StorageSecurityError("Encrypted backup key ID is not configured.")
    if AESGCM is None:
        raise StorageSecurityError("Encrypted backups require the cryptography package.")
    nonce_start = len(BACKUP_MAGIC)
    if len(ciphertext) <= nonce_start + 12:
        raise StorageSecurityError("Encrypted backup envelope is truncated.")
    nonce = ciphertext[nonce_start:nonce_start + 12]
    try:
        return AESGCM(key).decrypt(nonce, ciphertext[nonce_start + 12:], BACKUP_AAD)
    except Exception as exc:
        raise StorageSecurityError("Encrypted backup authentication failed.") from exc


def _write_bytes_atomic(data: bytes, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        _atomic_replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def backup_database(source: str | os.PathLike[str] | None = None, destination: str | os.PathLike[str] | None = None) -> dict:
    source_path = Path(source or DATABASE_PATH).resolve()
    if not source_path.is_file():
        raise StorageSecurityError("Source database does not exist.")
    if destination is None:
        destination_path = BACKUPS_DIR / f"security_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')}_{os.getpid()}.sqlite3"
    else:
        destination_path = safe_storage_path(destination, roots=(BACKUPS_DIR,), base=BACKUPS_DIR)
    if destination_path == source_path:
        raise StorageSecurityError("Backup destination must differ from the source database.")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_suffix(destination_path.suffix + ".tmp")
    schema_states: list[dict] = []
    try:
        source_con = connect_database(source_path)
        target_con = connect_database(temporary)
        try:
            source_con.row_factory = sqlite3.Row
            source_con.execute("PRAGMA busy_timeout=10000")
            source_con.execute("PRAGMA foreign_keys=ON")
            source_integrity = source_con.execute("PRAGMA integrity_check").fetchone()[0]
            if source_integrity != "ok":
                raise StorageSecurityError("Source database integrity check failed.")
            if source_con.execute("PRAGMA foreign_key_check").fetchall():
                raise StorageSecurityError("Source database has foreign-key violations.")
            if source_con.execute(
                "select 1 from sqlite_master where type='table' and name='schema_migration_state'"
            ).fetchone():
                state_columns = {
                    row[1] for row in source_con.execute(
                        "pragma table_info(schema_migration_state)"
                    ).fetchall()
                }
                fingerprint_column = ", schema_fingerprint" if "schema_fingerprint" in state_columns else ""
                schema_states = [
                    dict(row) for row in source_con.execute(
                        f"select schema_name, version, description, checksum{fingerprint_column}, applied_at "
                        "from schema_migration_state order by schema_name"
                    ).fetchall()
                ]
            source_con.backup(target_con)
            target_con.execute("PRAGMA foreign_keys=ON")
            result = target_con.execute("pragma integrity_check").fetchone()[0]
            if result != "ok":
                raise StorageSecurityError("Backup integrity check failed.")
            if target_con.execute("PRAGMA foreign_key_check").fetchall():
                raise StorageSecurityError("Backup has foreign-key violations.")
        finally:
            target_con.close()
            source_con.close()
        encrypted = bool(BACKUP_ENCRYPTION_ENABLED)
        if encrypted:
            _write_bytes_atomic(_encrypt_backup_bytes(temporary.read_bytes()), destination_path)
            temporary.unlink()
        else:
            _atomic_replace(temporary, destination_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    manifest = destination_path.with_name(destination_path.name + ".manifest.json")
    data = {
        "format": "optivox.sqlite_backup",
        "version": 3 if BACKUP_ENCRYPTION_ENABLED else 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sha256": sha256_file(destination_path),
        "filename": destination_path.name,
        "schema_states": schema_states,
        "encrypted": bool(BACKUP_ENCRYPTION_ENABLED),
        "encryption": "aes-256-gcm" if BACKUP_ENCRYPTION_ENABLED else None,
        "key_id": str(BACKUP_ENCRYPTION_KEY_ID) if BACKUP_ENCRYPTION_ENABLED else None,
        "integrity_mode": (
            "encrypted_signed" if BACKUP_ENCRYPTION_ENABLED and BACKUP_SIGNING_KEY
            else "encrypted_checksum_only" if BACKUP_ENCRYPTION_ENABLED
            else "signed" if BACKUP_SIGNING_KEY
            else "checksum_only"
        ),
    }
    data["signature"] = _manifest_signature(data)
    _write_bytes_atomic(
        (json.dumps(data, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        manifest,
    )
    return {"path": str(destination_path), "manifest": str(manifest), "sha256": data["sha256"]}


def restore_database(
    backup: str | os.PathLike[str],
    target: str | os.PathLike[str] | None = None,
    *,
    create_safety_backup: bool = True,
) -> dict:
    backup_path = safe_storage_path(backup, roots=(BACKUPS_DIR,), base=BACKUPS_DIR)
    if not backup_path.is_file():
        raise StorageSecurityError("Backup file does not exist.")
    manifest = backup_path.with_name(backup_path.name + ".manifest.json")
    if not manifest.is_file():
        raise StorageSecurityError("Backup manifest is missing.")
    try:
        manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StorageSecurityError("Backup manifest is invalid.") from exc
    if not isinstance(manifest_data, dict) or manifest_data.get("format") != "optivox.sqlite_backup":
        raise StorageSecurityError("Backup manifest format is invalid.")
    manifest_version = int(manifest_data.get("version", 0) or 0)
    if manifest_version not in {1, 2, 3}:
        raise StorageSecurityError("Backup manifest version is unsupported.")
    if manifest_version == 2 and not isinstance(manifest_data.get("schema_states"), list):
        raise StorageSecurityError("Backup schema-state metadata is invalid.")
    encrypted = bool(manifest_data.get("encrypted", manifest_version >= 3))
    if manifest_version >= 3 and not encrypted:
        raise StorageSecurityError("Encrypted backup manifest is inconsistent.")
    if manifest_data.get("filename") != backup_path.name:
        raise StorageSecurityError("Backup manifest filename does not match the backup.")
    expected = str(manifest_data.get("sha256") or "")
    if len(expected) != 64 or sha256_file(backup_path) != expected:
        raise StorageSecurityError("Backup checksum mismatch.")
    signature = manifest_data.get("signature")
    if signature:
        if not BACKUP_SIGNING_KEY:
            raise StorageSecurityError("Backup signature cannot be verified without OPTIVOX_BACKUP_SIGNING_KEY.")
        expected_signature = _manifest_signature(manifest_data)
        if not expected_signature or not hmac.compare_digest(str(signature), expected_signature):
            raise StorageSecurityError("Backup manifest signature mismatch.")
    elif RUNTIME_MODE in {"pilot", "production"}:
        raise StorageSecurityError("Unsigned backups are not accepted in pilot or production mode.")
    if RUNTIME_MODE in {"pilot", "production"} and not encrypted:
        raise StorageSecurityError("Unencrypted backups are not accepted in pilot or production mode.")
    target_path = Path(target or DATABASE_PATH).resolve()
    if not _inside(target_path, (PROJECT_ROOT,)):
        raise StorageSecurityError("Restore target is outside the project storage boundary.")
    rollback_backup = None
    if create_safety_backup and target_path.is_file():
        if target_path == backup_path.resolve():
            raise StorageSecurityError("Restore source and target must be different files.")
        rollback_name = f"pre_restore_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{os.getpid()}.sqlite3"
        rollback_backup = backup_database(source=target_path, destination=BACKUPS_DIR / rollback_name)
    backup_data_path = backup_path
    decrypted_temporary: Path | None = None
    if encrypted:
        fd, decrypted_name = tempfile.mkstemp(
            prefix=f".{backup_path.stem}.", suffix=".decrypted", dir=backup_path.parent
        )
        decrypted_temporary = Path(decrypted_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(_decrypt_backup_bytes(backup_path.read_bytes(), str(manifest_data.get("key_id") or BACKUP_ENCRYPTION_KEY_ID)))
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.chmod(decrypted_temporary, 0o600)
            except OSError:
                pass
            backup_data_path = decrypted_temporary
        except Exception:
            if decrypted_temporary.exists():
                decrypted_temporary.unlink()
            raise
    target_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target_path.name}.", suffix=".restore", dir=target_path.parent)
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        source_con = connect_database(backup_data_path)
        target_con = connect_database(temporary)
        try:
            source_con.execute("PRAGMA foreign_keys=ON")
            if source_con.execute("pragma integrity_check").fetchone()[0] != "ok":
                raise StorageSecurityError("Backup integrity check failed.")
            if source_con.execute("PRAGMA foreign_key_check").fetchall():
                raise StorageSecurityError("Backup has foreign-key violations.")
            source_con.backup(target_con)
            target_con.execute("PRAGMA foreign_keys=ON")
            if target_con.execute("pragma integrity_check").fetchone()[0] != "ok":
                raise StorageSecurityError("Restored database integrity check failed.")
            if target_con.execute("PRAGMA foreign_key_check").fetchall():
                raise StorageSecurityError("Restored database has foreign-key violations.")
        finally:
            target_con.close()
            source_con.close()
        _atomic_replace(temporary, target_path)
    finally:
        if temporary.exists():
            temporary.unlink()
        if decrypted_temporary is not None and decrypted_temporary.exists():
            decrypted_temporary.unlink()
    return {
        "path": str(target_path),
        "sha256": sha256_file(target_path),
        "rollback_backup": rollback_backup,
    }


def list_backups(limit: int = 100) -> dict:
    """Return a bounded integrity inventory without exposing local paths."""
    limit = max(1, min(int(limit), 500))
    items: list[dict] = []
    if not BACKUPS_DIR.exists():
        return {"count": 0, "valid": 0, "invalid": 0, "items": items}
    candidates = sorted(
        (path for pattern in ("*.sqlite3", "*.sqlite3.enc") for path in BACKUPS_DIR.glob(pattern)
         if path.is_file() and not path.is_symlink()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:limit]
    for path in candidates:
        manifest_path = path.with_name(path.name + ".manifest.json")
        item = {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "modified_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            "status": "invalid",
            "signed": False,
            "encrypted": False,
            "manifest_version": None,
        }
        try:
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            item["manifest_version"] = manifest_data.get("version")
            item["signed"] = bool(manifest_data.get("signature"))
            item["encrypted"] = bool(manifest_data.get("encrypted", int(manifest_data.get("version", 0) or 0) >= 3))
            item["key_id"] = str(manifest_data.get("key_id") or "") or None
            item["key_available"] = bool(_backup_key(item["key_id"])) if item["encrypted"] else None
            if manifest_data.get("filename") != path.name:
                raise StorageSecurityError("filename mismatch")
            if RUNTIME_MODE in {"pilot", "production"} and not item["encrypted"]:
                raise StorageSecurityError("unencrypted backup")
            if item["encrypted"] and not path.read_bytes().startswith(BACKUP_MAGIC):
                raise StorageSecurityError("encrypted envelope missing")
            if item["encrypted"] and not item["key_available"]:
                raise StorageSecurityError("encryption key is unavailable")
            expected = str(manifest_data.get("sha256") or "")
            if len(expected) != 64 or not hmac.compare_digest(sha256_file(path), expected):
                raise StorageSecurityError("checksum mismatch")
            if item["signed"]:
                if not BACKUP_SIGNING_KEY or not hmac.compare_digest(
                    str(manifest_data.get("signature")), str(_manifest_signature(manifest_data))
                ):
                    raise StorageSecurityError("signature mismatch")
            elif RUNTIME_MODE in {"pilot", "production"}:
                raise StorageSecurityError("unsigned backup")
            item["status"] = "valid"
        except (OSError, TypeError, ValueError, json.JSONDecodeError, StorageSecurityError):
            item["status"] = "invalid"
        items.append(item)
    valid = sum(1 for item in items if item["status"] == "valid")
    return {"count": len(items), "valid": valid, "invalid": len(items) - valid, "items": items}


def verify_backup_contents(name: str) -> bool:
    """Run SQLite integrity checks against one already-inventoried backup.

    Encrypted backup envelopes are decrypted only into a 0600 temporary file,
    which is removed before returning. The helper returns a boolean so health
    responses do not leak storage paths or low-level database details.
    """
    backup_path = safe_storage_path(name, roots=(BACKUPS_DIR,), base=BACKUPS_DIR)
    if not backup_path.is_file() or backup_path.is_symlink():
        return False
    manifest_path = backup_path.with_name(backup_path.name + ".manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        encrypted = bool(manifest.get("encrypted", int(manifest.get("version", 0) or 0) >= 3))
        candidate = backup_path
        temporary: Path | None = None
        if encrypted:
            fd, temporary_name = tempfile.mkstemp(prefix=".backup-verify-", suffix=".sqlite3", dir=BACKUPS_DIR)
            temporary = Path(temporary_name)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(_decrypt_backup_bytes(backup_path.read_bytes(), str(manifest.get("key_id") or BACKUP_ENCRYPTION_KEY_ID)))
                    handle.flush()
                    os.fsync(handle.fileno())
                try:
                    os.chmod(temporary, 0o600)
                except OSError:
                    pass
                candidate = temporary
            except Exception:
                if temporary.exists():
                    temporary.unlink()
                return False
        try:
            con = connect_database(candidate)
            try:
                con.execute("PRAGMA foreign_keys=ON")
                integrity = str(con.execute("PRAGMA integrity_check").fetchone()[0]).lower()
                return integrity == "ok" and not con.execute("PRAGMA foreign_key_check").fetchall()
            finally:
                con.close()
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    except (OSError, TypeError, ValueError, json.JSONDecodeError, StorageSecurityError, sqlite3.Error):
        return False


def rehearse_latest_backup() -> dict:
    """Restore the newest verified backup into an isolated temporary database.

    This is a recovery drill, not a production restore: the live database and
    its path are never replaced. The temporary materialization is removed on
    every exit path and only bounded health metadata is returned.
    """
    inventory = list_backups(limit=1000)
    valid = [item for item in inventory.get("items", []) if item.get("status") == "valid"]
    valid.sort(key=lambda item: str(item.get("modified_at") or ""), reverse=True)
    latest = valid[0] if valid else None
    if not latest:
        return {"status": "failed", "reason": "no_valid_backup"}

    rehearsal_dir = Path(tempfile.mkdtemp(prefix=".restore-drill-", dir=BACKUPS_DIR))
    target = rehearsal_dir / "rehearsal.sqlite3"
    try:
        restored = restore_database(
            str(latest.get("name") or ""),
            target,
            create_safety_backup=False,
        )
        con = connect_database(target)
        try:
            con.execute("PRAGMA foreign_keys=ON")
            integrity = str(con.execute("PRAGMA integrity_check").fetchone()[0]).lower()
            foreign_keys = con.execute("PRAGMA foreign_key_check").fetchall()
            table_count = int(con.execute(
                "select count(*) from sqlite_master where type='table' and name not like 'sqlite_%'"
            ).fetchone()[0] or 0)
        finally:
            con.close()
        if integrity != "ok" or foreign_keys:
            return {
                "status": "failed",
                "reason": "restored_database_integrity_failure",
                "backup_name": latest.get("name"),
                "integrity_check": integrity,
                "foreign_key_violations": len(foreign_keys),
            }
        return {
            "status": "success",
            "backup_name": latest.get("name"),
            "backup_modified_at": latest.get("modified_at"),
            "restored_size_bytes": int(Path(restored["path"]).stat().st_size),
            "integrity_check": integrity,
            "foreign_key_violations": 0,
            "table_count": table_count,
        }
    except (OSError, TypeError, ValueError, StorageSecurityError, sqlite3.Error) as exc:
        return {
            "status": "failed",
            "reason": "restore_rehearsal_failed",
            "backup_name": latest.get("name"),
            "error_type": type(exc).__name__,
        }
    finally:
        shutil.rmtree(rehearsal_dir, ignore_errors=True)


def backup_recovery_status(
    max_age_minutes: int | None = None,
    *,
    now: datetime | None = None,
) -> dict:
    """Report whether a recent, usable recovery point is available.

    This is metadata-only: it returns backup names and times, never local
    filesystem paths or database contents. A backup is usable only when its
    manifest, checksum, encryption policy, and signature policy pass the same
    validation used by ``list_backups``.
    """
    threshold_minutes = max(
        1,
        int(max_age_minutes if max_age_minutes is not None else BACKUP_MAX_AGE_MINUTES),
    )
    current = now or datetime.now(timezone.utc)
    inventory = list_backups(limit=1000)
    valid = [item for item in inventory.get("items", []) if item.get("status") == "valid"]
    valid.sort(key=lambda item: str(item.get("modified_at") or ""), reverse=True)
    issues: list[str] = []
    latest = valid[0] if valid else None
    age_seconds = None
    if latest:
        try:
            modified = datetime.fromisoformat(str(latest.get("modified_at")))
            if modified.tzinfo is None:
                modified = modified.replace(tzinfo=timezone.utc)
            age_seconds = max(0.0, (current - modified.astimezone(timezone.utc)).total_seconds())
        except (TypeError, ValueError):
            issues.append("latest_backup_timestamp_invalid")
    if not latest:
        issues.append("no_valid_backup")
    elif age_seconds is None or age_seconds > threshold_minutes * 60:
        issues.append("latest_backup_stale")
    if int(inventory.get("invalid", 0) or 0):
        issues.append("invalid_backups_present")
    if latest and not verify_backup_contents(str(latest.get("name") or "")):
        issues.append("latest_backup_unreadable")
    critical_issues = {"no_valid_backup", "latest_backup_stale", "latest_backup_unreadable"}
    strict_mode = RUNTIME_MODE in {"pilot", "production"}
    status = "CRITICAL" if strict_mode and any(item in critical_issues for item in issues) else "WARN" if issues else "OK"
    return {
        "status": status,
        "max_age_minutes": threshold_minutes,
        "latest_backup_name": latest.get("name") if latest else None,
        "latest_backup_at": latest.get("modified_at") if latest else None,
        "latest_backup_age_seconds": age_seconds,
        "valid_count": int(inventory.get("valid", 0) or 0),
        "invalid_count": int(inventory.get("invalid", 0) or 0),
        "issues": issues,
    }


def backup_retention_preview(
    retention_days: int | None = None,
    retention_count: int | None = None,
    *,
    now: datetime | None = None,
) -> dict:
    """Plan conservative backup deletion without touching the filesystem.

    A backup is eligible only when it is valid, older than the time window,
    and outside the newest-count safety floor. Invalid backups are never
    silently removed by automated retention; they require investigation.
    """
    days = max(1, int(retention_days if retention_days is not None else BACKUP_RETENTION_DAYS))
    keep_count = max(2, int(retention_count if retention_count is not None else BACKUP_RETENTION_COUNT))
    current = now or datetime.now(timezone.utc)
    cutoff = current - timedelta(days=days)
    inventory = list_backups(limit=1000)
    valid = [item for item in inventory["items"] if item.get("status") == "valid"]
    valid.sort(key=lambda item: str(item.get("modified_at") or ""), reverse=True)
    protected = {str(item.get("name")) for item in valid[:keep_count]}
    eligible: list[dict] = []
    for item in valid:
        name = str(item.get("name") or "")
        try:
            modified = datetime.fromisoformat(str(item["modified_at"]).replace("Z", "+00:00"))
        except (KeyError, TypeError, ValueError):
            continue
        if name not in protected and modified < cutoff:
            eligible.append({"name": name, "modified_at": item.get("modified_at"), "size_bytes": item.get("size_bytes", 0)})
    return {
        "dry_run": True,
        "retention_days": days,
        "retention_count": keep_count,
        "cutoff": cutoff.isoformat(),
        "valid_backups": len(valid),
        "invalid_backups_preserved": int(inventory["invalid"]),
        "eligible": eligible,
        "eligible_count": len(eligible),
    }


def purge_backups(
    retention_days: int | None = None,
    retention_count: int | None = None,
    *,
    execute_delete: bool = False,
) -> dict:
    """Preview or execute bounded deletion of old, verified backup pairs."""
    preview = backup_retention_preview(retention_days, retention_count)
    if not execute_delete:
        return preview
    deleted: list[str] = []
    failures: list[dict] = []
    for item in preview["eligible"]:
        name = str(item["name"])
        try:
            backup_path = safe_storage_path(name, roots=(BACKUPS_DIR,), base=BACKUPS_DIR)
            manifest_path = backup_path.with_name(backup_path.name + ".manifest.json")
            try:
                manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise StorageSecurityError("backup manifest is invalid") from exc
            if (
                backup_path.name != name
                or backup_path.is_symlink()
                or manifest_path.is_symlink()
                or not backup_path.is_file()
                or not manifest_path.is_file()
                or manifest_data.get("filename") != name
                or not verify_checksum(backup_path, manifest_data.get("sha256"))
            ):
                raise StorageSecurityError("backup pair is no longer available")
            backup_path.unlink()
            manifest_path.unlink()
            deleted.append(name)
        except (OSError, StorageSecurityError) as exc:
            failures.append({"name": name, "error": str(exc)[:200]})
    return {
        **preview,
        "dry_run": False,
        "deleted": deleted,
        "deleted_count": len(deleted),
        "failures": failures,
    }


def delete_evidence(raw_path: str, expected_checksum: str | None = None) -> bool:
    path = safe_storage_path(raw_path)
    if not path.exists():
        return False
    # Authenticate the envelope before deletion so a corrupted evidence file
    # cannot be silently erased by retention or an administrative action.
    read_evidence_bytes(path, expected_checksum=expected_checksum)
    path.unlink()
    return True


def purge_expired_evidence(
    retention_days: int,
    *,
    protected_paths: Iterable[str | os.PathLike[str]] = (),
) -> dict[str, int]:
    """Delete only aged, unreferenced evidence files.

    The database remains the source of truth for incident references. Callers
    pass those paths explicitly so this low-level storage module does not
    create a circular database dependency.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, int(retention_days)))
    deleted = 0
    protected = set()
    for raw in protected_paths:
        try:
            protected.add(safe_storage_path(raw).resolve())
        except StorageSecurityError:
            continue
    if not SNAPSHOTS_DIR.exists():
        return {"deleted": 0, "protected": 0, "failed": 0}
    skipped = 0
    failed = 0
    for path in SNAPSHOTS_DIR.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        if path.resolve() in protected:
            skipped += 1
            continue
        try:
            if datetime.fromtimestamp(path.stat().st_mtime, timezone.utc) < cutoff:
                path.unlink()
                deleted += 1
        except OSError:
            failed += 1
    return {"deleted": deleted, "protected": skipped, "failed": failed}
