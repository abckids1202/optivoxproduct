"""Versioned, integrity-checked local face embedding storage."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


FORMAT = "optivox.face_database"
VERSION = 2
AAD = b"optivox-face-database-v2"
EMBEDDING_BLOB_PREFIX = b"OVX2"
PLAINTEXT_BLOB_PREFIX = b"OVX1"

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except ImportError:  # pragma: no cover - exercised only on minimal installs
    AESGCM = None


class BiometricStorageError(RuntimeError):
    pass


def secure_path_for(path: str | os.PathLike[str]) -> Path:
    source = Path(path)
    if source.suffix.lower() == ".pkl":
        return source.with_name(f"{source.stem}.secure.json")
    return source.with_suffix(source.suffix + ".secure.json")


def _key() -> bytes | None:
    raw = os.getenv("OPTIVOX_BIOMETRIC_KEY", "").strip()
    if not raw:
        return None
    try:
        decoded = base64.urlsafe_b64decode(raw.encode("ascii"))
        if len(decoded) == 32:
            return decoded
    except (ValueError, UnicodeEncodeError):
        pass
    return hashlib.sha256(raw.encode("utf-8")).digest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _embedding(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1 or array.size < 2 or array.size > 4096:
        raise BiometricStorageError("Embedding has an invalid shape.")
    if not np.isfinite(array).all():
        raise BiometricStorageError("Embedding contains non-finite values.")
    return array.copy()


def encode_embedding_blob(value: Any) -> bytes:
    """Encode the legacy SQLite embedding column without changing its API."""
    raw = _embedding(value).tobytes()
    key = _key()
    if key is None:
        return PLAINTEXT_BLOB_PREFIX + raw
    if AESGCM is None:
        raise BiometricStorageError("OPTIVOX_BIOMETRIC_KEY is set but cryptography is unavailable.")
    nonce = os.urandom(12)
    return EMBEDDING_BLOB_PREFIX + nonce + AESGCM(key).encrypt(nonce, raw, AAD)


def decode_embedding_blob(value: bytes) -> np.ndarray:
    """Decode encrypted, compatibility-prefixed, or legacy raw BLOB values."""
    raw = bytes(value or b"")
    if raw.startswith(EMBEDDING_BLOB_PREFIX):
        key = _key()
        if key is None or AESGCM is None or len(raw) < len(EMBEDDING_BLOB_PREFIX) + 12:
            raise BiometricStorageError("Encrypted embedding requires OPTIVOX_BIOMETRIC_KEY.")
        nonce_start = len(EMBEDDING_BLOB_PREFIX)
        nonce = raw[nonce_start:nonce_start + 12]
        try:
            decoded = AESGCM(key).decrypt(nonce, raw[nonce_start + 12:], AAD)
        except Exception as exc:
            raise BiometricStorageError("Embedding BLOB integrity validation failed.") from exc
        return _embedding(np.frombuffer(decoded, dtype=np.float32))
    if raw.startswith(PLAINTEXT_BLOB_PREFIX):
        raw = raw[len(PLAINTEXT_BLOB_PREFIX):]
    return _embedding(np.frombuffer(raw, dtype=np.float32))


def _serialize(face_db: dict[str, Any]) -> dict[str, Any]:
    people: dict[str, Any] = {}
    for name, raw_record in (face_db or {}).items():
        if not isinstance(raw_record, dict):
            raise BiometricStorageError(f"Identity record for {name!r} is malformed.")
        embeddings = [_embedding(item).tolist() for item in raw_record.get("embeddings", []) if item is not None]
        record = {
            str(key): _json_safe(value)
            for key, value in raw_record.items()
            if key != "embeddings"
        }
        record["embeddings"] = embeddings
        people[str(name)] = record
    return {"format": FORMAT, "version": VERSION, "people": people}


def _deserialize(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("format") != FORMAT or int(payload.get("version", -1)) != VERSION:
        raise BiometricStorageError("Unsupported biometric database format or version.")
    people = payload.get("people")
    if not isinstance(people, dict):
        raise BiometricStorageError("Biometric database people collection is malformed.")
    output: dict[str, Any] = {}
    for name, raw_record in people.items():
        if not isinstance(raw_record, dict):
            raise BiometricStorageError(f"Identity record for {name!r} is malformed.")
        embeddings = [_embedding(item) for item in raw_record.get("embeddings", [])]
        output[str(name)] = {**raw_record, "embeddings": embeddings}
    return output


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_face_database(path: str | os.PathLike[str], face_db: dict[str, Any]) -> bool:
    target = secure_path_for(path)
    payload = _serialize(face_db)
    payload_bytes = _canonical(payload)
    key = _key()
    if key is not None:
        if AESGCM is None:
            raise BiometricStorageError("OPTIVOX_BIOMETRIC_KEY is set but cryptography is unavailable.")
        nonce = os.urandom(12)
        ciphertext = AESGCM(key).encrypt(nonce, payload_bytes, AAD)
        envelope = {
            "format": FORMAT,
            "version": VERSION,
            "encrypted": True,
            "nonce": base64.urlsafe_b64encode(nonce).decode("ascii"),
            "ciphertext": base64.urlsafe_b64encode(ciphertext).decode("ascii"),
            "sha256": hashlib.sha256(ciphertext).hexdigest(),
        }
    else:
        envelope = {
            "format": FORMAT,
            "version": VERSION,
            "encrypted": False,
            "payload": payload,
            "sha256": hashlib.sha256(payload_bytes).hexdigest(),
        }
    _atomic_write(target, json.dumps(envelope, sort_keys=True, indent=2) + "\n")
    # Once the integrity-checked replacement is durable, remove the old
    # executable-object container so deleted embeddings are not left behind.
    legacy = Path(path)
    if legacy.suffix.lower() == ".pkl" and legacy.exists() and legacy.resolve() != target.resolve():
        try:
            legacy.unlink()
        except OSError as exc:
            raise BiometricStorageError(f"Secure database saved but legacy biometric file could not be removed: {exc}") from exc
    return True


def _restricted_legacy_load(path: Path) -> Any:
    """Read only the numpy/container globals emitted by the old face DB.

    This is a one-time compatibility migration, not a general pickle loader.
    Arbitrary classes and functions are rejected before object construction.
    """
    allowed = {
        ("builtins", name) for name in ("dict", "list", "tuple", "set", "str", "int", "float", "bool")
    }
    allowed.update({
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("numpy", "float32"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
    })

    class RestrictedUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> Any:
            if (module, name) not in allowed:
                raise BiometricStorageError(f"Legacy biometric file contains blocked type {module}.{name}.")
            return super().find_class(module, name)

    with path.open("rb") as handle:
        return RestrictedUnpickler(handle).load()


def load_face_database(path: str | os.PathLike[str]) -> dict[str, Any]:
    legacy = Path(path)
    secure = secure_path_for(legacy)
    if secure.exists():
        try:
            envelope = json.loads(secure.read_text(encoding="utf-8"))
            if envelope.get("format") != FORMAT or int(envelope.get("version", -1)) != VERSION:
                raise BiometricStorageError("Unsupported biometric database envelope.")
            if envelope.get("encrypted"):
                key = _key()
                if key is None or AESGCM is None:
                    raise BiometricStorageError("Encrypted biometric database requires OPTIVOX_BIOMETRIC_KEY.")
                ciphertext = base64.urlsafe_b64decode(envelope["ciphertext"])
                if hashlib.sha256(ciphertext).hexdigest() != envelope.get("sha256"):
                    raise BiometricStorageError("Biometric database checksum mismatch.")
                nonce = base64.urlsafe_b64decode(envelope["nonce"])
                payload = json.loads(AESGCM(key).decrypt(nonce, ciphertext, AAD).decode("utf-8"))
            else:
                payload = envelope.get("payload")
                if hashlib.sha256(_canonical(payload)).hexdigest() != envelope.get("sha256"):
                    raise BiometricStorageError("Biometric database checksum mismatch.")
            return _deserialize(payload)
        except BiometricStorageError:
            raise
        except Exception as exc:
            raise BiometricStorageError(f"Biometric database cannot be read: {exc}") from exc

    if not legacy.exists():
        return {}
    try:
        migrated = _deserialize({**_serialize(_restricted_legacy_load(legacy)), "version": VERSION})
        save_face_database(legacy, migrated)
        print(f"[SECURITY] Migrated legacy face database to {secure.name}; legacy file is no longer loaded.")
        return migrated
    except (OSError, ValueError, TypeError, pickle.UnpicklingError, BiometricStorageError) as exc:
        raise BiometricStorageError(f"Legacy biometric database rejected: {exc}") from exc
