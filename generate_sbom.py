"""Generate a dependency SBOM from installed Python and frontend lock metadata."""

from __future__ import annotations

import argparse
import json
import re
from importlib import metadata
from pathlib import Path

from deployment_security import safe_local_path


def python_components() -> list[dict]:
    components = []
    for distribution in sorted(metadata.distributions(), key=lambda item: (item.metadata.get("Name") or "").lower()):
        name = distribution.metadata.get("Name")
        if not name:
            continue
        normalized = re.sub(r"[^a-z0-9.-]+", "-", name.lower())
        components.append({
            "type": "library", "bom-ref": f"pkg:pypi/{normalized}@{distribution.version}",
            "name": name, "version": distribution.version,
            "purl": f"pkg:pypi/{normalized}@{distribution.version}", "scope": "required",
        })
    return components


def npm_components(lock_path: Path) -> list[dict]:
    if not lock_path.exists():
        return []
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    components = []
    for package_path, item in (lock.get("packages") or {}).items():
        if package_path == "" or not isinstance(item, dict) or not item.get("version"):
            continue
        name = package_path.rsplit("/node_modules/", 1)[-1].removeprefix("node_modules/")
        version = item["version"]
        components.append({
            "type": "library", "bom-ref": f"pkg:npm/{name}@{version}",
            "name": name, "version": version, "purl": f"pkg:npm/{name}@{version}",
            "scope": "required",
        })
    return components


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=Path("reports/sbom.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    components = python_components() + npm_components(root / "frontend" / "package-lock.json")
    bom = {
        "bomFormat": "CycloneDX", "specVersion": "1.5", "serialNumber": "urn:uuid:optivox-local-sbom",
        "version": 1, "metadata": {"component": {"type": "application", "name": "optivox", "version": "1.0.0"}},
        "components": components,
    }
    output = safe_local_path(args.output, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bom, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(components)} component(s) to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
