"""Focused tests for portable v4 configuration and file I/O."""

from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import telco_anomaly.io as io_helpers
from telco_anomaly.io import (
    authorise_holdout,
    acquire_public_dataset,
    file_sha256,
    find_project_root,
    immutable_output_directory,
    load_config,
    read_json,
    require_same,
    resolve_data_root,
    resolve_dataset_source,
    write_json,
)


def project_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "configs").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'fixture'\n")
    (root / "configs" / "project.yml").write_text("version: 1\n")
    return root


def test_project_discovery_and_safe_config_loading(tmp_path, monkeypatch):
    root = project_fixture(tmp_path)
    nested = root / "notebooks" / "run"
    nested.mkdir(parents=True)
    (root / "configs" / "example.yml").write_text(
        "version: 1\noptions:\n  enabled: true\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(nested)
    monkeypatch.delenv("TELCO_PROJECT_ROOT", raising=False)
    monkeypatch.delenv("ANOMALY_PROJECT_ROOT", raising=False)

    assert find_project_root() == root.resolve()
    assert load_config("example") == {
        "version": 1,
        "options": {"enabled": True},
    }

    other = project_fixture(tmp_path / "other")
    monkeypatch.setenv("TELCO_PROJECT_ROOT", str(other))
    assert find_project_root() == other.resolve()


def test_data_root_environment_precedence(tmp_path):
    preferred = tmp_path / "preferred"
    legacy = tmp_path / "legacy"

    assert resolve_data_root(
        environ={
            "TELCO_DATA_ROOT": str(preferred),
            "ANOMALY_DATA_ROOT": str(legacy),
        }
    ) == preferred.resolve()
    assert resolve_data_root(
        environ={"ANOMALY_DATA_ROOT": str(legacy)}
    ) == legacy.resolve()
    assert resolve_data_root(environ={}, home=tmp_path) == (
        tmp_path / "telco_anomaly_data"
    ).resolve()


def test_data_root_supports_existing_legacy_colab_folder(tmp_path, monkeypatch):
    drive = tmp_path / "drive" / "MyDrive"
    preferred = drive / "telco_anomaly_data"
    legacy = drive / "anomaly_detection"
    legacy.mkdir(parents=True)
    monkeypatch.setattr(io_helpers, "COLAB_DATA_ROOT", preferred)
    monkeypatch.setattr(io_helpers, "LEGACY_COLAB_DATA_ROOT", legacy)

    assert resolve_data_root(environ={}) == legacy
    preferred.mkdir()
    assert resolve_data_root(environ={}) == preferred


def test_dataset_source_uses_first_existing_candidate(tmp_path):
    project = project_fixture(tmp_path)
    data_root = tmp_path / "data"
    second = data_root / "correct-spelling"
    second.mkdir(parents=True)
    (project / "configs" / "dataset.yml").write_text(
        """\
version: 1
primary_dataset: fixture
datasets:
  fixture:
    source_candidates:
      - historical-spelling
      - correct-spelling
""",
        encoding="utf-8",
    )

    assert resolve_dataset_source(
        data_root=data_root, project_root=project
    ) == second.resolve()

    (data_root / "historical-spelling").mkdir()
    assert resolve_dataset_source(
        "fixture", data_root=data_root, project_root=project
    ) == (data_root / "historical-spelling").resolve()


def test_immutable_output_is_published_only_on_success(tmp_path):
    destination = tmp_path / "result"
    with immutable_output_directory(destination) as working:
        assert working != destination
        assert not destination.exists()
        (working / "value.txt").write_text("complete", encoding="utf-8")

    assert (destination / "value.txt").read_text(encoding="utf-8") == "complete"
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        with immutable_output_directory(destination):
            pass

    failed = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="stop"):
        with immutable_output_directory(failed) as working:
            (working / "partial.txt").write_text("partial", encoding="utf-8")
            raise RuntimeError("stop")
    assert not failed.exists()
    assert not list(tmp_path.glob(".failed.tmp-*"))


def test_atomic_json_round_trip_refuses_accidental_overwrite(tmp_path):
    path = tmp_path / "nested" / "manifest.json"
    write_json(path, {"z": 2, "a": [1, 2]})

    assert read_json(path) == {"a": [1, 2], "z": 2}
    assert path.read_text(encoding="utf-8").startswith('{\n  "a"')
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_json(path, {"replacement": True})
    assert read_json(path) == {"a": [1, 2], "z": 2}

    write_json(path, {"replacement": True}, overwrite=True)
    assert read_json(path) == {"replacement": True}
    assert not list(path.parent.glob(".manifest.json.tmp-*"))


def test_json_write_falls_back_when_drive_rejects_hard_links(tmp_path, monkeypatch):
    path = tmp_path / "drive" / "manifest.json"

    def no_hard_links(*_args, **_kwargs):
        raise PermissionError("Drive does not implement hard links")

    monkeypatch.setattr(io_helpers.os, "link", no_hard_links)
    write_json(path, {"status": "complete"})

    assert read_json(path) == {"status": "complete"}
    with pytest.raises(FileExistsError):
        write_json(path, {"status": "replacement"})


def test_public_dataset_acquisition_downloads_once_and_records_checksums(
    tmp_path, monkeypatch
):
    project = project_fixture(tmp_path)
    data_root = tmp_path / "data"
    (project / "configs" / "dataset.yml").write_text(
        """\
version: 1
primary_dataset: fixture
datasets:
  fixture:
    source_candidates:
      - sources/fixture/raw
    source_url: https://example.test/fixture
    download:
      destination: sources/fixture/raw
      resources:
        - filename: observations.zip
          url: https://example.test/observations.zip
          archive: zip
        - filename: README.md
          url: https://example.test/README.md
          archive: none
""",
        encoding="utf-8",
    )
    calls = []

    def fake_download(url, destination):
        calls.append(url)
        if destination.suffix == ".zip":
            with zipfile.ZipFile(destination, "w") as archive:
                archive.writestr("data/part.csv", "timestamp,value\n1,2\n")
        else:
            destination.write_text("fixture documentation\n", encoding="utf-8")
        payload = destination.read_bytes()
        return {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "md5": hashlib.md5(payload).hexdigest(),
        }

    monkeypatch.setattr(io_helpers, "_download", fake_download)
    source = acquire_public_dataset(
        "fixture", data_root=data_root, project_root=project
    )

    assert (source / "data" / "part.csv").is_file()
    assert (source / "README.md").is_file()
    manifest = read_json(source / "source_manifest.json")
    assert manifest["dataset"] == "fixture"
    assert len(manifest["resources"]) == 2

    acquire_public_dataset("fixture", data_root=data_root, project_root=project)
    assert len(calls) == 2


def test_file_sha256_streams_exact_bytes(tmp_path):
    payload = b"telecom\x00telemetry\n" * 17
    path = tmp_path / "payload.bin"
    path.write_bytes(payload)

    assert file_sha256(path, chunk_size=7) == hashlib.sha256(payload).hexdigest()
    with pytest.raises(ValueError, match="positive"):
        file_sha256(path, chunk_size=0)


def test_require_same_rejects_stale_lineage():
    manifest = {"core_fingerprint": "core-a", "policy_sha256": "policy-a"}
    require_same(
        manifest,
        core_fingerprint="core-a",
        policy_sha256="policy-a",
    )
    with pytest.raises(ValueError, match="different inputs"):
        require_same(manifest, core_fingerprint="core-b")


def test_holdout_ledger_allows_replay_but_not_a_different_model(tmp_path):
    ledger = tmp_path / "holdout_openings.jsonl"
    receipt = {
        "selected_configuration_sha256": "configuration-a",
        "partition": "holdout",
    }

    assert authorise_holdout(ledger, receipt) == 1
    assert authorise_holdout(ledger, receipt) == 2
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 2

    with pytest.raises(PermissionError, match="another configuration"):
        authorise_holdout(
            ledger,
            {
                "selected_configuration_sha256": "configuration-b",
                "partition": "holdout",
            },
        )


def test_holdout_ledger_fails_closed_on_an_incomplete_old_entry(tmp_path):
    ledger = tmp_path / "holdout_openings.jsonl"
    ledger.write_text('{"partition": "holdout"}\n', encoding="utf-8")

    with pytest.raises(PermissionError, match="without a frozen configuration"):
        authorise_holdout(
            ledger,
            {
                "selected_configuration_sha256": "configuration-a",
                "partition": "holdout",
            },
        )
