"""Small asymmetric authentication helpers for OptiVox edge devices.

Ed25519 is used for request authentication because a receiver can keep only a
public key. Private keys stay on the edge device and are never included in
configuration responses or logs. HMAC remains supported for local and legacy
deployments.
"""

from __future__ import annotations

import base64
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


class DeviceAuthError(ValueError):
    """Raised when a device signing key or signature is invalid."""


def _pem_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, str) and value.strip():
        return value.encode("utf-8")
    raise DeviceAuthError("an Ed25519 PEM key is required")


def load_ed25519_private_key(value: Any) -> Ed25519PrivateKey:
    try:
        key = serialization.load_pem_private_key(_pem_bytes(value), password=None)
    except Exception as exc:  # cryptography deliberately exposes several parse errors
        raise DeviceAuthError("invalid Ed25519 private key") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise DeviceAuthError("private key must be Ed25519")
    return key


def load_ed25519_public_key(value: Any) -> Ed25519PublicKey:
    try:
        key = serialization.load_pem_public_key(_pem_bytes(value))
    except Exception as exc:
        raise DeviceAuthError("invalid Ed25519 public key") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise DeviceAuthError("public key must be Ed25519")
    return key


def sign_ed25519(payload: bytes, private_key: Any) -> str:
    try:
        key = private_key if isinstance(private_key, Ed25519PrivateKey) else load_ed25519_private_key(private_key)
        return base64.b64encode(key.sign(bytes(payload))).decode("ascii")
    except DeviceAuthError:
        raise
    except Exception as exc:
        raise DeviceAuthError("could not sign device request") from exc


def verify_ed25519(payload: bytes, signature: str, public_key: Any) -> bool:
    try:
        key = public_key if isinstance(public_key, Ed25519PublicKey) else load_ed25519_public_key(public_key)
        decoded = base64.b64decode(str(signature or ""), validate=True)
        key.verify(decoded, bytes(payload))
        return True
    except (DeviceAuthError, ValueError, TypeError):
        return False
