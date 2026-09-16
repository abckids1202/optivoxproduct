"""Small, dependency-free scan for credentials in tracked repository files.

This is a guardrail, not a replacement for a hosted secret scanner. It only
prints file names, line numbers, and rule names; matched values are never
included in output.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


_RULES = (
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("private_key", re.compile(r"-----BEGIN [A-Z0-9 ]+ PRIVATE KEY-----")),
    ("credential_assignment", re.compile(
        r"(?im)^\s*(?:OPENAI_API_KEY|OPTIVOX_(?:API_KEY|ADMIN_KEY|OPERATOR_KEY|COMMAND_PIN|SECRET_KEY))"
        r"\s*=\s*(?!\s*(?:$|#|[\"']?\s*$|<|your[_-]?|placeholder))\S+"
    )),
)

_ALLOWED_EXAMPLES = {
    relative.as_posix()
    for relative in (
        Path(".env.example"),
        Path("frontend/.env.example"),
        Path("optivox-web/backend/.env.example"),
        Path("optivox-web/backend/legacy_monolith_backup/.env.example"),
    )
}


def tracked_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [root / item for item in result.stdout.decode("utf-8").split("\0") if item]


def scan_tracked_files(root: Path | str | None = None) -> list[str]:
    root_path = Path(root or Path(__file__).resolve().parent).resolve()
    findings: list[str] = []
    for path in tracked_files(root_path):
        relative = path.relative_to(root_path).as_posix()
        if relative in _ALLOWED_EXAMPLES or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line_number, line in enumerate(text.splitlines(), 1):
            for rule_name, rule in _RULES:
                if rule.search(line):
                    findings.append(f"{relative}:{line_number}:{rule_name}")
    return findings


def main() -> int:
    findings = scan_tracked_files()
    if findings:
        print("Potential credentials found in tracked files:")
        print("\n".join(findings))
        return 1
    print("Secret scan passed: no credential patterns found in tracked files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
