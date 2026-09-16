"""Offline supply-chain checks for the OptiVox repository."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def scan(root: Path) -> list[str]:
    findings: list[str] = []
    lock = root / "requirements.lock.txt"
    if not lock.exists():
        findings.append("requirements.lock.txt is missing")
    else:
        for line_number, line in enumerate(lock.read_text(encoding="utf-8").splitlines(), 1):
            value = line.strip()
            if value and not value.startswith("#") and "==" not in value:
                findings.append(f"requirements.lock.txt:{line_number}:un-pinned requirement")
    for relative in ("requirements.txt", "backend/requirements.txt"):
        dependency_file = root / relative
        if not dependency_file.exists():
            findings.append(f"{relative} is missing")
            continue
        for line_number, line in enumerate(dependency_file.read_text(encoding="utf-8").splitlines(), 1):
            value = line.strip()
            if value and not value.startswith("#") and "==" not in value:
                findings.append(f"{relative}:{line_number}:un-pinned requirement")
    package = root / "frontend" / "package.json"
    if package.exists():
        data = json.loads(package.read_text(encoding="utf-8"))
        for section in ("dependencies", "devDependencies"):
            for name, version in (data.get(section) or {}).items():
                if re.search(r"[~^*]|[<>]=?", str(version)):
                    findings.append(f"frontend/package.json:{section}:{name}:range not exact")
    if not (root / "frontend" / "package-lock.json").exists():
        findings.append("frontend/package-lock.json is missing")
    return findings


def main() -> int:
    root = Path(__file__).resolve().parent
    findings = scan(root)
    if findings:
        print("Supply-chain scan findings:")
        print("\n".join(findings))
        return 1
    print("Supply-chain scan passed: pinned Python and npm dependency metadata found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
