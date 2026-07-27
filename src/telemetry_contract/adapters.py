"""Generic source-adapter interface.

Adapters own source mapping only. They do not own detector logic or fault truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, runtime_checkable


@dataclass(frozen=True)
class DatasetInventory:
    source_format: str
    source_version: str
    tables: tuple[str, ...]
    core_ready: bool
    evaluation_ready: bool
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class MaterialisationReport:
    adapter_id: str
    adapter_version: str
    source_format: str
    core_path: Path
    evaluation_path: Path | None
    row_counts: Mapping[str, int]
    truth_columns_removed: tuple[str, ...]


@runtime_checkable
class Adapter(Protocol):
    adapter_id: str
    adapter_version: str
    supported_source_formats: tuple[str, ...]

    def discover(self, source: Path) -> DatasetInventory: ...

    def materialise(
        self,
        source: Path,
        core_destination: Path,
        evaluation_destination: Path | None = None,
    ) -> MaterialisationReport: ...
