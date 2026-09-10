"""Portable configuration, data-path, and small-file helpers.

The functions in this module deliberately resolve paths without creating data
directories.  Pipeline stages remain responsible for deciding what they are
allowed to materialise.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
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

    With the default ``overwrite=False``, publishing uses a hard link so a
    concurrent writer cannot replace an existing file between the existence
    check and publication.
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
            temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def read_json(path: str | os.PathLike[str]) -> Any:
    """Read a UTF-8 JSON file."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


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
