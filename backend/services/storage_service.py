"""Protected local storage helpers for evidence and SQLite backups."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from ..config import BACKUPS_DIR, DATABASE_PATH, PROJECT_ROOT, SNAPSHOTS_DIR


class StorageSecurityError(RuntimeError):
    pass


def _inside(path: Path, roots: Iterable[Path]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def safe_storage_path(raw: str | os.PathLike[str], *, roots: Iterable[Path] | None = None, base: Path = SNAPSHOTS_DIR) -> Path:
    value = str(raw or "")
    if not value or "\x00" in value:
        raise StorageSecurityError("Storage path is empty or invalid.")
    candidate = Path(value)
    resolved = (candidate if candidate.is_absolute() else base / candidate).resolve()
    resolved_roots = tuple(Path(root).resolve() for root in (roots if roots is not None else (base,)))
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


def _atomic_replace(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, target)
    try:
        os.chmod(target, 0o600)
    except OSError:
        pass


def backup_database(source: str | os.PathLike[str] | None = None, destination: str | os.PathLike[str] | None = None) -> dict:
    source_path = Path(source or DATABASE_PATH).resolve()
    if not source_path.is_file():
        raise StorageSecurityError("Source database does not exist.")
    if destination is None:
        destination_path = BACKUPS_DIR / f"security_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.sqlite3"
    else:
        destination_path = safe_storage_path(destination, roots=(BACKUPS_DIR,), base=BACKUPS_DIR)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_suffix(destination_path.suffix + ".tmp")
    try:
        source_con = sqlite3.connect(source_path)
        target_con = sqlite3.connect(temporary)
        try:
            source_con.backup(target_con)
            result = target_con.execute("pragma integrity_check").fetchone()[0]
            if result != "ok":
                raise StorageSecurityError("Backup integrity check failed.")
        finally:
            target_con.close()
            source_con.close()
        _atomic_replace(temporary, destination_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    manifest = destination_path.with_name(destination_path.name + ".manifest.json")
    data = {
        "format": "optivox.sqlite_backup",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sha256": sha256_file(destination_path),
        "filename": destination_path.name,
    }
    manifest.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        os.chmod(manifest, 0o600)
    except OSError:
        pass
    return {"path": str(destination_path), "manifest": str(manifest), "sha256": data["sha256"]}


def restore_database(backup: str | os.PathLike[str], target: str | os.PathLike[str] | None = None) -> dict:
    backup_path = safe_storage_path(backup, roots=(BACKUPS_DIR,), base=BACKUPS_DIR)
    if not backup_path.is_file():
        raise StorageSecurityError("Backup file does not exist.")
    manifest = backup_path.with_name(backup_path.name + ".manifest.json")
    if manifest.exists():
        expected = json.loads(manifest.read_text(encoding="utf-8")).get("sha256")
        if expected and sha256_file(backup_path) != expected:
            raise StorageSecurityError("Backup checksum mismatch.")
    target_path = Path(target or DATABASE_PATH).resolve()
    if not _inside(target_path, (PROJECT_ROOT,)):
        raise StorageSecurityError("Restore target is outside the project storage boundary.")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target_path.name}.", suffix=".restore", dir=target_path.parent)
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        source_con = sqlite3.connect(backup_path)
        target_con = sqlite3.connect(temporary)
        try:
            if source_con.execute("pragma integrity_check").fetchone()[0] != "ok":
                raise StorageSecurityError("Backup integrity check failed.")
            source_con.backup(target_con)
        finally:
            target_con.close()
            source_con.close()
        _atomic_replace(temporary, target_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"path": str(target_path), "sha256": sha256_file(target_path)}


def delete_evidence(raw_path: str, expected_checksum: str | None = None) -> bool:
    path = safe_storage_path(raw_path)
    if not path.exists():
        return False
    if expected_checksum and not verify_checksum(path, expected_checksum):
        raise StorageSecurityError("Evidence checksum mismatch; refusing deletion.")
    path.unlink()
    return True


def purge_expired_evidence(retention_days: int) -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, int(retention_days)))
    deleted = 0
    if not SNAPSHOTS_DIR.exists():
        return 0
    for path in SNAPSHOTS_DIR.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            if datetime.fromtimestamp(path.stat().st_mtime, timezone.utc) < cutoff:
                path.unlink()
                deleted += 1
        except OSError:
            continue
    return deleted
