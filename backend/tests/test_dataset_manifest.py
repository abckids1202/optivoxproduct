from build_dataset_manifest import build_records
from core.data_registry import DatasetRegistry


def test_manifest_builder_is_relative_and_provenance_explicit(tmp_path):
    person = tmp_path / "Ada"
    person.mkdir()
    (person / "forward.jpg").write_bytes(b"forward")
    (person / "notes.txt").write_text("ignore", encoding="utf-8")

    records = build_records(tmp_path, "face_verification", "holdout", "internal")

    assert len(records) == 1
    assert records[0].path == "Ada/forward.jpg"
    assert records[0].label == "Ada"
    assert records[0].split == "holdout"
    assert records[0].provenance == "internal"
    assert len(records[0].content_sha256) == 64
    assert not records[0].path.startswith(str(tmp_path))


def test_manifest_validation_catches_exact_duplicate_media(tmp_path):
    (tmp_path / "one.jpg").write_bytes(b"same-media")
    (tmp_path / "two.jpg").write_bytes(b"same-media")
    registry = DatasetRegistry(
        tmp_path,
        build_records(tmp_path, "face_verification", "validation", "internal"),
    )

    issues = registry.validate(require_files=True)

    assert any(issue.startswith("exact duplicate media:") for issue in issues)


def test_manifest_detects_media_tampering_after_registration(tmp_path):
    sample = tmp_path / "Ada" / "forward.jpg"
    sample.parent.mkdir()
    sample.write_bytes(b"original-media")
    record = build_records(tmp_path, "face_verification", "holdout", "internal")[0]
    sample.write_bytes(b"tampered-media")

    issues = DatasetRegistry(tmp_path, [record]).validate(require_files=True)

    assert any(issue.startswith("content hash mismatch:") for issue in issues)
