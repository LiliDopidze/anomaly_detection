"""Portable configuration, data-path, and small-file helpers.

The functions in this module deliberately resolve paths without creating data
directories.  Pipeline stages remain responsible for deciding what they are
allowed to materialise.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import urllib.request
import zipfile
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import yaml


PROJECT_ROOT_ENV_VARS = ("TELCO_PROJECT_ROOT", "ANOMALY_PROJECT_ROOT")
DATA_ROOT_ENV_VARS = ("TELCO_DATA_ROOT", "ANOMALY_DATA_ROOT")
COLAB_DATA_ROOT = Path("/content/drive/MyDrive/telco_anomaly_data")
LEGACY_COLAB_DATA_ROOT = Path("/content/drive/MyDrive/anomaly_detection")
LOCAL_DATA_ROOT_NAME = "telco_anomaly_data"


def _expanded(path: str | os.PathLike[str]) -> Path:
    """Return a user-expanded absolute path without requiring it to exist."""

    return Path(path).expanduser().resolve()


def _is_project_root(path: Path) -> bool:
    """Recognise both an installed source tree and a minimal config fixture."""

    config_dir = path / "configs"
    return config_dir.is_dir() and (
        (path / "pyproject.toml").is_file()
        or (config_dir / "project.yml").is_file()
    )


def find_project_root(
    start: str | os.PathLike[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Find the project root from an explicit environment setting or ``start``.

    ``TELCO_PROJECT_ROOT`` takes precedence over the legacy
    ``ANOMALY_PROJECT_ROOT``.  Without either setting, the current directory
    (or ``start``) and each of its parents are searched for ``configs`` plus a
    project marker.
    """

    environment = os.environ if environ is None else environ
    for variable in PROJECT_ROOT_ENV_VARS:
        value = environment.get(variable)
        if value:
            root = _expanded(value)
            if not _is_project_root(root):
                raise FileNotFoundError(
                    f"{variable} does not point to a project root: {root}"
                )
            return root

    origin = _expanded(Path.cwd() if start is None else start)
    if origin.is_file():
        origin = origin.parent
    for candidate in (origin, *origin.parents):
        if _is_project_root(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not find a project root from {origin}; set TELCO_PROJECT_ROOT"
    )


def load_config(
    name: str | os.PathLike[str],
    *,
    project_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Safely load one mapping-valued YAML file from the project's configs.

    ``name`` may be given as ``"dataset"``, ``"dataset.yml"``, or
    ``"configs/dataset.yml"``.  Absolute paths and paths escaping ``configs``
    are rejected so configuration always comes from the selected project.
    """

    root = find_project_root() if project_root is None else _expanded(project_root)
    config_root = (root / "configs").resolve()
    requested = Path(name)
    if requested.is_absolute():
        raise ValueError("Config names must be relative to the project root")
    if requested.parts and requested.parts[0] == "configs":
        requested = Path(*requested.parts[1:])
    if not requested.suffix:
        requested = requested.with_suffix(".yml")
    elif requested.suffix.lower() not in {".yml", ".yaml"}:
        raise ValueError(f"Config must be YAML, not {requested.suffix!r}")

    path = (config_root / requested).resolve()
    if not path.is_relative_to(config_root):
        raise ValueError(f"Config path escapes {config_root}: {name}")
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping in config file: {path}")
    return loaded


def resolve_data_root(
    *,
    environ: Mapping[str, str] | None = None,
    home: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve the external data root without creating it.

    Precedence is ``TELCO_DATA_ROOT``, ``ANOMALY_DATA_ROOT``, the mounted
    Colab Drive location, then ``~/telco_anomaly_data``.
    """

    environment = os.environ if environ is None else environ
    for variable in DATA_ROOT_ENV_VARS:
        value = environment.get(variable)
        if value:
            return _expanded(value)

    in_colab = "google.colab" in sys.modules
    if in_colab or COLAB_DATA_ROOT.parent.is_dir():
        if COLAB_DATA_ROOT.exists():
            return COLAB_DATA_ROOT
        if LEGACY_COLAB_DATA_ROOT.exists():
            return LEGACY_COLAB_DATA_ROOT
        return COLAB_DATA_ROOT

    home_path = Path.home() if home is None else Path(home).expanduser()
    return (home_path / LOCAL_DATA_ROOT_NAME).resolve()


def resolve_dataset_source(
    dataset_name: str | None = None,
    *,
    data_root: str | os.PathLike[str] | None = None,
    project_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Return the first existing configured source for a dataset.

    Relative candidates in ``configs/dataset.yml`` are interpreted beneath
    ``data_root``.  If ``dataset_name`` is omitted, ``primary_dataset`` is
    used.  Candidate order is significant and is preserved.
    """

    config = load_config("dataset", project_root=project_root)
    selected = dataset_name or config.get("primary_dataset")
    datasets = config.get("datasets")
    if not selected:
        raise ValueError("dataset.yml does not define primary_dataset")
    if not isinstance(datasets, dict) or selected not in datasets:
        raise KeyError(f"Unknown dataset {selected!r} in configs/dataset.yml")

    definition = datasets[selected]
    if not isinstance(definition, dict):
        raise ValueError(f"Dataset {selected!r} must be a mapping")
    candidates = definition.get("source_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"Dataset {selected!r} has no source_candidates")

    root = resolve_data_root() if data_root is None else _expanded(data_root)
    checked: list[Path] = []
    for value in candidates:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Dataset {selected!r} contains an invalid source candidate"
            )
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate = candidate.resolve()
        checked.append(candidate)
        if candidate.exists():
            return candidate

    locations = ", ".join(str(path) for path in checked)
    raise FileNotFoundError(
        f"No source found for dataset {selected!r}; checked: {locations}"
    )


@contextmanager
def immutable_output_directory(
    destination: str | os.PathLike[str],
) -> Iterator[Path]:
    """Build ``destination`` off to the side, then publish it on success.

    An existing destination is never overwritten.  If the body raises, its
    temporary directory is removed and no completed output is published.
    """

    destination_path = Path(destination)
    if os.path.lexists(destination_path):
        raise FileExistsError(f"Refusing to overwrite {destination_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            dir=destination_path.parent,
            prefix=f".{destination_path.name}.tmp-",
        )
    )
    try:
        yield temporary
        if os.path.lexists(destination_path):
            raise FileExistsError(f"Refusing to overwrite {destination_path}")
        temporary.rename(destination_path)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


# The v3 pipeline uses this name.  Keep it as a transparent compatibility
# alias while new code uses the more descriptive public name above.
new_output_directory = immutable_output_directory


def write_json(
    path: str | os.PathLike[str],
    payload: Any,
    overwrite: bool = False,
) -> None:
    """Atomically write deterministic, human-readable JSON.

    With the default ``overwrite=False``, publishing first uses a hard link so
    a concurrent writer cannot replace an existing file between the existence
    check and publication.  Google Drive does not support hard links, so the
    fallback uses exclusive file creation and removes any partial destination
    if copying fails.
    """

    destination = Path(path)
    content = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and os.path.lexists(destination):
        raise FileExistsError(f"Refusing to overwrite {destination}")

    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.tmp-",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())

        if overwrite:
            os.replace(temporary, destination)
        else:
            try:
                os.link(temporary, destination)
            except FileExistsError as error:
                raise FileExistsError(
                    f"Refusing to overwrite {destination}"
                ) from error
            except OSError as error:
                unsupported = {
                    errno.EPERM,
                    errno.EXDEV,
                    getattr(errno, "ENOTSUP", errno.EOPNOTSUPP),
                    errno.EOPNOTSUPP,
                }
                if not isinstance(error, PermissionError) and error.errno not in unsupported:
                    raise
                descriptor = os.open(
                    destination,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o666,
                )
                try:
                    with os.fdopen(descriptor, "wb") as target:
                        with temporary.open("rb") as source:
                            shutil.copyfileobj(source, target)
                        target.flush()
                        os.fsync(target.fileno())
                except BaseException:
                    destination.unlink(missing_ok=True)
                    raise
            finally:
                temporary.unlink(missing_ok=True)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _download(url: str, destination: Path) -> dict[str, Any]:
    """Stream one public resource and return its observed checksums."""

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "telco-anomaly-detection/4.0"},
    )
    sha256 = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    size = 0
    with urllib.request.urlopen(request) as response, destination.open("wb") as target:
        while chunk := response.read(1 << 20):
            target.write(chunk)
            sha256.update(chunk)
            md5.update(chunk)
            size += len(chunk)
    return {
        "bytes": size,
        "sha256": sha256.hexdigest(),
        "md5": md5.hexdigest(),
    }


def _extract_zip(archive_path: Path, destination: Path) -> None:
    """Extract a ZIP archive while rejecting links and path traversal."""

    root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Unsafe archive path: {member.filename}")
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"Archive links are not allowed: {member.filename}")
        archive.extractall(destination)


def acquire_public_dataset(
    dataset_name: str,
    *,
    data_root: str | os.PathLike[str] | None = None,
    project_root: str | os.PathLike[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Download and extract one configured public dataset exactly once.

    This removes the manual download/upload step, but it does not stream large
    archives on every notebook run.  The extracted source is cached outside
    Git under ``TELCO_DATA_ROOT`` and accompanied by a checksum manifest.
    """

    root = resolve_data_root() if data_root is None else _expanded(data_root)
    project = find_project_root() if project_root is None else _expanded(project_root)
    environment = os.environ if environ is None else environ
    datasets = load_config("dataset", project_root=project).get("datasets", {})
    if dataset_name not in datasets:
        raise KeyError(f"Unknown dataset {dataset_name!r}")

    definition = datasets[dataset_name]
    download = definition.get("download")
    if not isinstance(download, dict):
        raise ValueError(f"Dataset {dataset_name!r} has no download configuration")
    acknowledgement = download.get("acknowledgement_env")
    if acknowledgement and environment.get(str(acknowledgement)) != "1":
        raise PermissionError(
            f"Review the source terms, then set {acknowledgement}=1"
        )

    relative_destination = Path(str(download.get("destination", "")))
    if (
        str(relative_destination) in {"", "."}
        or relative_destination.is_absolute()
        or ".." in relative_destination.parts
    ):
        raise ValueError("Public download destination must stay beneath TELCO_DATA_ROOT")
    destination = (root / relative_destination).resolve()
    if destination.exists():
        return resolve_dataset_source(
            dataset_name, data_root=root, project_root=project
        )

    resources = download.get("resources")
    if not isinstance(resources, list) or not resources:
        raise ValueError(f"Dataset {dataset_name!r} has no download resources")

    records = []
    with immutable_output_directory(destination) as staging:
        with tempfile.TemporaryDirectory(prefix=f"{dataset_name}-download-") as temporary:
            temporary = Path(temporary)
            for item in resources:
                filename = Path(str(item.get("filename", ""))).name
                url = item.get("url")
                if not filename or not isinstance(url, str) or not url.startswith("https://"):
                    raise ValueError(f"Invalid download resource for {dataset_name!r}")
                local_file = temporary / filename
                observed = _download(url, local_file)
                expected_md5 = item.get("md5")
                expected_sha256 = item.get("sha256")
                expected_bytes = item.get("bytes")
                if expected_bytes is not None and observed["bytes"] != int(expected_bytes):
                    raise ValueError(f"Size mismatch for {filename}")
                if expected_md5 and observed["md5"] != str(expected_md5).lower():
                    raise ValueError(f"MD5 mismatch for {filename}")
                if expected_sha256 and observed["sha256"] != str(expected_sha256).lower():
                    raise ValueError(f"SHA-256 mismatch for {filename}")

                if item.get("archive") == "zip":
                    _extract_zip(local_file, staging)
                elif item.get("archive") in {None, "none"}:
                    shutil.copy2(local_file, staging / filename)
                else:
                    raise ValueError(f"Unsupported archive type for {filename}")
                records.append({"filename": filename, "url": url, **observed})

        write_json(staging / "source_manifest.json", {
            "dataset": dataset_name,
            "source_page": definition.get("source_url"),
            "resources": records,
        })

    return resolve_dataset_source(
        dataset_name, data_root=root, project_root=project
    )


def read_json(path: str | os.PathLike[str]) -> Any:
    """Read a UTF-8 JSON file."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def require_same(manifest: Mapping[str, Any], **expected: Any) -> None:
    """Refuse to reuse an artifact built from different recorded inputs."""

    stale = {
        key: {"recorded": manifest.get(key), "expected": value}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if stale:
        raise ValueError(f"Existing output was built from different inputs: {stale}")


def authorise_holdout(
    ledger_path: str | os.PathLike[str],
    receipt: Mapping[str, Any],
) -> int:
    """Record a holdout opening and reject a different selected model.

    Re-running the same frozen configuration is allowed so an interrupted
    evaluation can be completed. Once a different configuration hash has
    opened the holdout, however, the ledger refuses another choice.
    """

    selected_hash = receipt.get("selected_configuration_sha256")
    if not selected_hash:
        raise ValueError("A holdout receipt needs selected_configuration_sha256")

    try:
        import fcntl
    except ImportError as error:  # pragma: no cover - production target is Linux
        raise RuntimeError("Holdout ledger requires POSIX file locking") from error

    ledger = Path(ledger_path)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            earlier = []
            for line_number, line in enumerate(handle.read().splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    earlier.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid holdout ledger entry on line {line_number}"
                    ) from error

            incomplete = [
                index + 1
                for index, item in enumerate(earlier)
                if not item.get("selected_configuration_sha256")
            ]
            if incomplete:
                raise PermissionError(
                    "Holdout ledger contains an opening without a frozen "
                    f"configuration hash on entries {incomplete}"
                )

            previous_hashes = {
                item["selected_configuration_sha256"] for item in earlier
            } - {selected_hash}
            if previous_hashes:
                raise PermissionError(
                    "Holdout was already opened for another configuration: "
                    f"{sorted(previous_hashes)}"
                )

            handle.seek(0, os.SEEK_END)
            handle.write(json.dumps(dict(receipt), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            return len(earlier) + 1
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def file_sha256(
    path: str | os.PathLike[str],
    chunk_size: int = 1 << 20,
) -> str:
    """Return the lowercase SHA-256 digest of a file's bytes."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
