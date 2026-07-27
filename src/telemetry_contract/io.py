"""Canonical hashing and immutable manifest helpers used by all adapters."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_frame_hash(frame, *, sort_by: Iterable[str] | None = None) -> str:
    """Hash logical table content without depending on Parquet file metadata."""

    import pandas as pd

    def normalise_object(value):
        if isinstance(value, (dict, list, tuple, set)):
            serialisable = sorted(value) if isinstance(value, set) else value
            return json.dumps(serialisable, sort_keys=True, default=str)
        if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
            converted = value.tolist()
            if isinstance(converted, (dict, list, tuple)):
                return json.dumps(converted, sort_keys=True, default=str)
        return value

    canonical = frame.copy()
    if sort_by:
        columns = [column for column in sort_by if column in canonical.columns]
        if columns:
            canonical = canonical.sort_values(columns, kind="stable")
    canonical = canonical.reset_index(drop=True)
    for column in canonical.columns:
        if pd.api.types.is_datetime64_any_dtype(canonical[column]):
            canonical[column] = canonical[column].astype("string")
        elif canonical[column].dtype == "object":
            canonical[column] = canonical[column].map(normalise_object)
    row_hashes = pd.util.hash_pandas_object(canonical, index=False).to_numpy()
    schema = "|".join(f"{name}:{dtype}" for name, dtype in canonical.dtypes.items())
    digest = hashlib.sha256(schema.encode("utf-8"))
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def write_json_immutable(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {destination}")
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
