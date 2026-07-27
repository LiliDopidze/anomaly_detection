"""Translator from telemetry-synth native v4 to SPEC-CORE/SPEC-EVAL v0.3.

The adapter is the only component allowed to see both native operational data and
native truth. SPEC-CORE is built from explicit operational allowlists. Tickets and
all ground-truth inputs are routed only to SPEC-EVAL.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .core import (
    CONTRACT_VERSION,
    DatasetInventory,
    MaterialisationReport,
    SectorPack,
    canonical_frame_hash,
    validate_core_bundle,
)
from .evaluation import EVAL_CONTRACT_VERSION, validate_eval_bundle
from .packs import load_telecom_pack

NATIVE_FORMAT = "telemetry-synth native v4"
SOURCE_ID = "telemetry-synth-4.0.1"

CORE_REQUIRED_FILES = {
    "reference_dataset.parquet",
    "topology.csv",
    "entity_service_windows.csv",
}
CORE_OPTIONAL_FILES = {"engineering_events.csv"}
EVALUATION_REQUIRED_FILES = {
    "fault_entity_intervals.csv",
    "gt_fault_groups.csv",
    "gt_fault_registry.csv",
}
EVALUATION_OPTIONAL_FILES = {
    "tickets.csv",
    "gt_benign_anomalies.csv",
    "gt_collection_gaps.parquet",
}

ENTITY_ATTRIBUTE_ALLOWLIST = (
    "device_model",
    "vendor",
    "enclosure",
    "firmware_version",
    "geo_cluster",
    "lat",
    "lon",
    "distance_m",
    "distance_bucket",
    "splitter_ratio",
    "l2_splitter_capacity",
    "fibre_age_yr",
    "ont_age_yr",
    "expected_rx_power_dbm",
    "rx_sensitivity_dbm",
    "service_impact_weight",
    "customer_priority_weight",
)

TOPOLOGY_LEVELS = (
    ("olt_id", "olt"),
    ("pon_port", "pon_port"),
    ("splitter_l1", "splitter_l1"),
    ("splitter_l2", "splitter_l2"),
    ("geo_cluster", "geo_cluster"),
    ("ont_id", "ont"),
)

RELATION_MAPPINGS = (
    ("olt_id", "pon_port", "contains_pon", "network_topology"),
    ("pon_port", "splitter_l1", "contains_splitter_l1", "network_topology"),
    ("splitter_l1", "splitter_l2", "contains_splitter_l2", "network_topology"),
    ("splitter_l2", "ont_id", "serves_ont", "network_topology"),
    ("geo_cluster", "ont_id", "groups_ont", "geographic_membership"),
)

FEC_CEILING = 5_000_000


@dataclass(frozen=True)
class NativeSelection:
    sample_start: str | None = None
    sample_end: str | None = None
    entity_ids: tuple[str, ...] = ()
    batch_native_rows: int = 250_000


def _pandas():
    import pandas as pd

    return pd


def _normalise_json_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    try:
        if value != value:
            return None
    except (TypeError, ValueError):
        pass
    return value


def _attributes_json(row: Mapping[str, Any], fields: Iterable[str]) -> str:
    payload = {
        field: _normalise_json_value(row[field])
        for field in fields
        if field in row and not str(field).startswith("gt_")
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _stable_id(prefix: str, *parts: Any) -> str:
    raw = "|".join("" if part is None else str(part) for part in parts)
    return f"{prefix}-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


def _empty_frame(columns: Iterable[str]):
    return _pandas().DataFrame(columns=list(columns))


def _utc_timestamp(value):
    """Parse either naive or timezone-aware input without double-localising it."""

    return _pandas().to_datetime(value, utc=True)


class SyntheticGponAdapter:
    adapter_id = "synthetic-gpon-native-v4"
    adapter_version = "0.3.0"
    supported_source_formats = (NATIVE_FORMAT,)
    source_id = SOURCE_ID

    def __init__(
        self,
        pack: SectorPack | None = None,
        selection: NativeSelection | None = None,
    ):
        self.pack = pack or load_telecom_pack()
        self.selection = selection or NativeSelection()

    @staticmethod
    def _roots(source: Path, *, evaluation: bool) -> tuple[Path, ...]:
        roots: list[Path] = [source, source / "Data"]
        if evaluation:
            for candidate in source.iterdir() if source.is_dir() else ():
                if candidate.is_dir() and candidate.name.strip().lower() in {
                    "evaluation",
                    "spec-eval",
                    "spec_eval",
                }:
                    roots.extend((candidate, candidate / "Data"))
        unique: list[Path] = []
        for root in roots:
            if root not in unique:
                unique.append(root)
        return tuple(unique)

    @classmethod
    def _native_path(
        cls,
        source: Path,
        filename: str,
        *,
        evaluation: bool = False,
        required: bool = True,
    ) -> Path | None:
        matches = [
            root / filename
            for root in cls._roots(Path(source), evaluation=evaluation)
            if (root / filename).is_file()
        ]
        if len(matches) > 1:
            raise ValueError(f"ambiguous native file {filename!r}: {matches}")
        if matches:
            return matches[0]
        if required:
            raise FileNotFoundError(f"native file not found: {filename}")
        return None

    def discover(self, source: Path) -> DatasetInventory:
        root = Path(source)
        tables = tuple(
            sorted(
                str(path.relative_to(root))
                for path in root.rglob("*")
                if path.is_file() and not path.name.startswith(".")
            )
        )
        missing_core = sorted(
            name
            for name in CORE_REQUIRED_FILES
            if self._native_path(root, name, required=False) is None
        )
        missing_eval = sorted(
            name
            for name in EVALUATION_REQUIRED_FILES
            if self._native_path(root, name, evaluation=True, required=False) is None
        )
        notes = tuple(
            ([f"missing core files: {missing_core}"] if missing_core else [])
            + ([f"missing evaluation files: {missing_eval}"] if missing_eval else [])
        )
        return DatasetInventory(
            source_format=NATIVE_FORMAT,
            source_version="4.0.1",
            tables=tables,
            core_ready=not missing_core,
            evaluation_ready=not missing_eval,
            notes=notes,
        )

    def _metric_mapping(self) -> dict[str, str]:
        return {
            spec.native_field: metric_id
            for metric_id, spec in self.pack.metric_specs().items()
            if spec.native_field
        }

    def _filter_native(self, frame):
        pd = _pandas()
        selected = frame
        if self.selection.entity_ids:
            selected = selected.loc[
                selected["ont_id"].astype(str).isin(self.selection.entity_ids)
            ]
        if self.selection.sample_start is not None:
            start = _utc_timestamp(self.selection.sample_start)
            selected = selected.loc[
                pd.to_datetime(selected["timestamp_utc"], utc=True).ge(start)
            ]
        if self.selection.sample_end is not None:
            end = _utc_timestamp(self.selection.sample_end)
            selected = selected.loc[
                pd.to_datetime(selected["timestamp_utc"], utc=True).lt(end)
            ]
        return selected.reset_index(drop=True)

    @staticmethod
    def _median_cadence(panel):
        pd = _pandas()
        if panel.empty:
            raise ValueError("cannot infer cadence from an empty selection")
        first_entity = str(panel["ont_id"].astype(str).iloc[0])
        timestamps = (
            pd.to_datetime(
                panel.loc[panel["ont_id"].astype(str).eq(first_entity), "timestamp_utc"],
                utc=True,
            )
            .drop_duplicates()
            .sort_values()
        )
        positive = timestamps.diff().dropna()
        positive = positive[positive > pd.Timedelta(0)]
        if positive.empty:
            raise ValueError("could not infer a positive cadence")
        return positive.median()

    def adapt_telemetry(self, panel, *, cadence_seconds: float | None = None):
        pd = _pandas()
        mapping = self._metric_mapping()
        available = [field for field in mapping if field in panel.columns]
        if not available:
            raise ValueError("native panel contains none of the pack metric fields")
        missing_ids = [
            field for field in ("timestamp_utc", "ont_id") if field not in panel.columns
        ]
        if missing_ids:
            raise ValueError(f"native panel missing identifiers: {missing_ids}")
        if cadence_seconds is None:
            cadence_seconds = float(self._median_cadence(panel).total_seconds())

        working = panel[["timestamp_utc", "ont_id", *available]].copy()
        frame_size_bytes = float(
            self.pack.parameters()["crc_frame_size_bytes"]["value"]
        )
        if "throughput_mbps" in working:
            working["_crc_exposure"] = (
                pd.to_numeric(working["throughput_mbps"], errors="coerce")
                * 1_000_000.0
                * cadence_seconds
                / (frame_size_bytes * 8.0)
            )
        else:
            working["_crc_exposure"] = float("nan")

        long = working.melt(
            id_vars=["timestamp_utc", "ont_id", "_crc_exposure"],
            value_vars=available,
            var_name="native_metric",
            value_name="value",
        )
        long["event_ts"] = pd.to_datetime(long.pop("timestamp_utc"), utc=True)
        long["ingested_at"] = long["event_ts"]
        long["entity_id"] = long.pop("ont_id").astype(str)
        long["metric_id"] = long.pop("native_metric").map(mapping)

        is_null = long["value"].isna()
        is_fec_ceiling = long["metric_id"].eq("telecom.link.fec_count") & long[
            "value"
        ].ge(FEC_CEILING)
        long["quality_code"] = "measured"
        long.loc[is_null, "quality_code"] = "invalid"
        long.loc[is_fec_ceiling, "quality_code"] = "clipped"
        long["quality_detail"] = None
        long.loc[is_null, "quality_detail"] = "source_null"
        long.loc[is_fec_ceiling, "quality_detail"] = "upper_bound"

        long["exposure"] = float("nan")
        fec_line_rate_bps = float(
            self.pack.parameters()["generator_fec_line_rate_bps"]["value"]
        )
        long.loc[
            long["metric_id"].eq("telecom.link.fec_count"), "exposure"
        ] = fec_line_rate_bps * cadence_seconds
        long.loc[
            long["metric_id"].eq("telecom.link.crc_errors"), "exposure"
        ] = long.loc[
            long["metric_id"].eq("telecom.link.crc_errors"), "_crc_exposure"
        ]
        long = long.drop(columns=["_crc_exposure"])
        long["source_id"] = self.source_id
        columns = [
            "event_ts",
            "ingested_at",
            "entity_id",
            "metric_id",
            "value",
            "quality_code",
            "quality_detail",
            "exposure",
            "source_id",
        ]
        return (
            long[columns]
            .sort_values(["event_ts", "entity_id", "metric_id"], kind="stable")
            .reset_index(drop=True)
        )

    def metric_catalogue(self, available_native_fields: set[str] | None = None):
        pd = _pandas()
        rows = []
        for metric_id, spec in sorted(self.pack.metric_specs().items()):
            if (
                available_native_fields is not None
                and spec.native_field not in available_native_fields
            ):
                continue
            row = asdict(spec)
            row.pop("native_field", None)
            row["measurement_kind"] = spec.measurement_kind.value
            row["anomaly_direction"] = spec.anomaly_direction.value
            row["candidate_periods"] = list(spec.candidate_periods)
            row["context_keys"] = list(spec.context_keys)
            rows.append(row)
        return pd.DataFrame(rows)

    def entity_registry(self, topology, service_windows):
        pd = _pandas()
        service = service_windows.copy()
        service["entity_id"] = service["entity_id"].astype(str)
        service = service.set_index("entity_id")
        rows: list[dict[str, Any]] = []
        for level_field, entity_type in TOPOLOGY_LEVELS[:-1]:
            if level_field not in topology.columns:
                continue
            for entity_id in sorted(topology[level_field].dropna().astype(str).unique()):
                rows.append(
                    {
                        "entity_id": entity_id,
                        "entity_type": entity_type,
                        "valid_from": pd.NaT,
                        "valid_to": pd.NaT,
                        "attributes_json": "{}",
                        "source_id": self.source_id,
                    }
                )
        for record in topology.to_dict(orient="records"):
            entity_id = str(record["ont_id"])
            window = service.loc[entity_id] if entity_id in service.index else None
            rows.append(
                {
                    "entity_id": entity_id,
                    "entity_type": "ont",
                    "valid_from": pd.NaT if window is None else window.get("install_ts"),
                    "valid_to": pd.NaT if window is None else window.get("decommission_ts"),
                    "attributes_json": _attributes_json(
                        record, ENTITY_ATTRIBUTE_ALLOWLIST
                    ),
                    "source_id": self.source_id,
                }
            )
        return (
            pd.DataFrame(rows)
            .drop_duplicates("entity_id")
            .sort_values("entity_id")
            .reset_index(drop=True)
        )

    def entity_relations(self, topology, service_windows):
        pd = _pandas()
        service = service_windows.copy()
        service["entity_id"] = service["entity_id"].astype(str)
        service = service.set_index("entity_id")
        rows: list[dict[str, Any]] = []
        for parent_field, child_field, relation_type, family in RELATION_MAPPINGS:
            if parent_field not in topology.columns or child_field not in topology.columns:
                continue
            pairs = topology[[parent_field, child_field]].dropna().drop_duplicates()
            for parent, child in pairs.itertuples(index=False):
                child_id = str(child)
                window = (
                    service.loc[child_id]
                    if child_field == "ont_id" and child_id in service.index
                    else None
                )
                rows.append(
                    {
                        "parent_entity_id": str(parent),
                        "child_entity_id": child_id,
                        "relation_type": relation_type,
                        "relation_family": family,
                        "valid_from": pd.NaT if window is None else window.get("install_ts"),
                        "valid_to": pd.NaT if window is None else window.get("decommission_ts"),
                        "relation_confidence": 1.0,
                        "source": self.source_id,
                    }
                )
        return (
            pd.DataFrame(rows)
            .sort_values(
                ["relation_family", "relation_type", "parent_entity_id", "child_entity_id"]
            )
            .reset_index(drop=True)
        )

    def operational_events(self, engineering_events=None):
        pd = _pandas()
        columns = [
            "event_id",
            "entity_id",
            "event_type",
            "event_start",
            "event_end",
            "known_at",
            "attributes_json",
            "source",
        ]
        if engineering_events is None or not len(engineering_events):
            return pd.DataFrame(columns=columns)
        rows = []
        for record in engineering_events.to_dict(orient="records"):
            rows.append(
                {
                    "event_id": _stable_id(
                        "ENG",
                        record.get("entity_id"),
                        record.get("ts"),
                        record.get("event_type"),
                    ),
                    "entity_id": str(record["entity_id"]),
                    "event_type": str(record["event_type"]),
                    "event_start": record["ts"],
                    "event_end": pd.NaT,
                    "known_at": record["ts"],
                    "attributes_json": _attributes_json(
                        record, ("level_change_db", "detail")
                    ),
                    "source": self.source_id,
                }
            )
        return (
            pd.DataFrame(rows, columns=columns)
            .sort_values(["event_start", "event_id"], kind="stable")
            .reset_index(drop=True)
        )

    def collection_gaps(self, presence, service_windows, *, cadence=None):
        """Compress missing timestamp runs without a Cartesian product or Python sets."""

        pd = _pandas()
        observed = presence[["ont_id", "timestamp_utc"]].copy()
        observed["ont_id"] = observed["ont_id"].astype(str)
        observed["timestamp_utc"] = pd.to_datetime(
            observed["timestamp_utc"], utc=True
        )
        observed = observed.drop_duplicates().sort_values(
            ["ont_id", "timestamp_utc"], kind="stable"
        )
        if observed.empty:
            return _empty_frame(
                ["entity_id", "gap_start", "gap_end", "known_at", "source"]
            )
        cadence = cadence or self._median_cadence(observed)
        global_start = observed["timestamp_utc"].min()
        global_end = observed["timestamp_utc"].max() + cadence
        if self.selection.sample_start is not None:
            global_start = max(
                global_start, _utc_timestamp(self.selection.sample_start)
            )
        if self.selection.sample_end is not None:
            global_end = min(
                global_end, _utc_timestamp(self.selection.sample_end)
            )
        service = service_windows.copy()
        service["entity_id"] = service["entity_id"].astype(str)
        service = service.set_index("entity_id")
        rows: list[dict[str, Any]] = []

        def add_gap(entity_id, start, end):
            if start < end:
                rows.append(
                    {
                        "entity_id": entity_id,
                        "gap_start": start,
                        "gap_end": end,
                        "known_at": end,
                        "source": self.source_id,
                    }
                )

        for entity_id, group in observed.groupby("ont_id", sort=False):
            window = service.loc[entity_id] if entity_id in service.index else None
            valid_start = (
                global_start
                if window is None or pd.isna(window.get("install_ts"))
                else max(global_start, pd.to_datetime(window.get("install_ts"), utc=True))
            )
            valid_to = None if window is None else window.get("decommission_ts")
            valid_end = (
                global_end
                if valid_to is None or pd.isna(valid_to)
                else min(global_end, pd.to_datetime(valid_to, utc=True))
            )
            timestamps = group["timestamp_utc"].to_numpy()
            first = pd.Timestamp(timestamps[0])
            add_gap(entity_id, valid_start, first)
            previous = first
            for raw_timestamp in timestamps[1:]:
                timestamp = pd.Timestamp(raw_timestamp)
                if timestamp - previous > cadence:
                    add_gap(entity_id, previous + cadence, timestamp)
                previous = timestamp
            add_gap(entity_id, previous + cadence, valid_end)

        return pd.DataFrame(
            rows,
            columns=["entity_id", "gap_start", "gap_end", "known_at", "source"],
        )

    def adapt_core_frames(
        self,
        panel,
        topology,
        service_windows,
        tickets=None,
        engineering_events=None,
    ) -> dict[str, Any]:
        del tickets  # Tickets are evaluation-only for this source.
        selected_panel = self._filter_native(panel)
        selected_entities = set(selected_panel["ont_id"].astype(str))
        topology = topology.loc[topology["ont_id"].astype(str).isin(selected_entities)]
        service_windows = service_windows.loc[
            service_windows["entity_id"].astype(str).isin(selected_entities)
        ]
        cadence = self._median_cadence(selected_panel)
        bundle = {
            "telemetry": self.adapt_telemetry(
                selected_panel, cadence_seconds=float(cadence.total_seconds())
            ),
            "metric_catalogue": self.metric_catalogue(set(selected_panel.columns)),
            "entity_registry": self.entity_registry(topology, service_windows),
            "entity_relations": self.entity_relations(topology, service_windows),
            "operational_events": self.operational_events(engineering_events),
            "collection_gaps": self.collection_gaps(
                selected_panel, service_windows, cadence=cadence
            ),
        }
        errors = validate_core_bundle(bundle)
        if errors:
            raise ValueError("SPEC-CORE adaptation failed: " + "; ".join(errors))
        return bundle

    def adapt_eval_frames(
        self,
        fault_registry,
        fault_entity_intervals,
        fault_groups,
        *,
        tickets=None,
        benign_anomalies=None,
        collection_gap_truth=None,
    ) -> dict[str, Any]:
        pd = _pandas()
        registry = fault_registry.copy()
        fault_events = pd.DataFrame(
            {
                "fault_event_id": registry["gt_fault_id"].astype(str),
                "fault_type": registry["gt_fault_type"].astype(str),
                "fault_family": registry["family"].astype(str),
                "fault_domain_type": registry["scope"].astype(str),
                "fault_domain_id": registry["target"].astype(str),
                "onset_ts": registry.get("onset_ts"),
                "first_observable_ts": registry.get("first_observable_ts"),
                "impact_ts": registry.get("impact_ts"),
                "resolution_ts": registry.get("repair_ts"),
                "cause_group_id": registry.get(
                    "group_id", pd.Series([pd.NA] * len(registry))
                ),
                "left_censored": registry.get(
                    "left_censored", pd.Series([False] * len(registry))
                ).astype(bool),
                "label_source": "synthetic_fixture",
                "source_instance_id": pd.Series([pd.NA] * len(registry)),
            }
        )
        intervals = fault_entity_intervals.rename(
            columns={
                "fault_id": "fault_event_id",
                "entity_id": "affected_entity_id",
                "contribution_db": "contribution",
            }
        ).copy()
        interval_columns = [
            "fault_event_id",
            "affected_entity_id",
            "fault_family",
            "channel",
            "active_start_ts",
            "active_end_ts",
            "impact_ts",
            "contribution",
        ]
        for column in interval_columns:
            if column not in intervals:
                intervals[column] = pd.NA
        intervals = intervals[interval_columns]
        groups = fault_groups.copy()
        cause_groups = pd.DataFrame(
            {
                "cause_group_id": groups["group_id"].astype(str),
                "cause_type": "regional_storm",
                "start_ts": groups["start_ts"],
                "end_ts": groups["end_ts"],
                "footprint_json": groups.apply(
                    lambda row: json.dumps(
                        {"geo_clusters": _normalise_json_value(row.get("geo_clusters"))},
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    axis=1,
                ),
            }
        )
        condition_columns = [
            "entity_id",
            "condition_start_ts",
            "condition_end_ts",
            "condition_code",
            "condition_label",
            "label_source",
            "source_instance_id",
        ]
        bundle: dict[str, Any] = {
            "gt_fault_events": fault_events,
            "gt_fault_entity_intervals": intervals,
            "gt_cause_groups": cause_groups,
            "gt_condition_states": pd.DataFrame(columns=condition_columns),
        }
        if tickets is not None:
            bundle["gt_ticket_links"] = pd.DataFrame(
                {
                    "ticket_id": tickets["ticket_id"].astype(str),
                    "entity_id": tickets["ont_id"].astype(str),
                    "fault_event_id": tickets.get("gt_fault_id"),
                    "fault_type_label": tickets.get("gt_fault_type"),
                    "is_no_fault_found": tickets.get("gt_is_nff"),
                    "is_misattributed": tickets.get("gt_misattributed"),
                    "reported_ts": tickets.get("reported_ts"),
                    "resolved_ts": tickets.get("resolved_ts"),
                }
            )
        if benign_anomalies is not None:
            benign = benign_anomalies.rename(
                columns={"ts": "event_ts", "gt_benign_type": "benign_type"}
            ).copy()
            bundle["gt_benign_anomalies"] = benign[
                ["entity_id", "event_ts", "benign_type", "n_samples"]
            ]
        if collection_gap_truth is not None:
            gaps = collection_gap_truth.rename(
                columns={"ts": "event_ts", "gt_gap_reason": "gap_reason"}
            ).copy()
            bundle["gt_collection_gaps"] = gaps[
                ["entity_id", "event_ts", "gap_reason"]
            ]
        errors = validate_eval_bundle(bundle)
        if errors:
            raise ValueError("SPEC-EVAL adaptation failed: " + "; ".join(errors))
        return bundle

    @staticmethod
    def write_bundle(
        bundle: Mapping[str, Any], destination: Path, *, file_format: str = "parquet"
    ) -> None:
        destination = Path(destination)
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f"destination is not empty: {destination}")
        destination.mkdir(parents=True, exist_ok=True)
        row_counts = {}
        content_hashes = {}
        for table_name, table in sorted(bundle.items()):
            row_counts[table_name] = int(len(table))
            content_hashes[table_name] = canonical_frame_hash(table)
            if file_format == "parquet":
                table.to_parquet(destination / f"{table_name}.parquet", index=False)
            elif file_format == "csv":
                table.to_csv(destination / f"{table_name}.csv", index=False)
            else:
                raise ValueError("file_format must be 'parquet' or 'csv'")
        (destination / "manifest.json").write_text(
            json.dumps(
                {
                    "file_format": file_format,
                    "row_counts": row_counts,
                    "canonical_content_hashes": content_hashes,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def _load_presence(self, panel_file):
        pd = _pandas()
        frames = []
        for batch in panel_file.iter_batches(
            batch_size=max(250_000, self.selection.batch_native_rows),
            columns=["timestamp_utc", "ont_id"],
        ):
            selected = self._filter_native(batch.to_pandas())
            if len(selected):
                frames.append(selected)
        if not frames:
            raise ValueError("selection contains no native telemetry rows")
        return pd.concat(frames, ignore_index=True)

    def materialise(
        self,
        source: Path,
        core_destination: Path,
        evaluation_destination: Path | None = None,
    ) -> MaterialisationReport:
        pd = _pandas()
        import pyarrow as pa
        import pyarrow.parquet as pq

        source = Path(source)
        core_destination = Path(core_destination)
        evaluation_destination = (
            None if evaluation_destination is None else Path(evaluation_destination)
        )
        inventory = self.discover(source)
        if not inventory.core_ready:
            raise FileNotFoundError("; ".join(inventory.notes))
        if core_destination.exists():
            raise FileExistsError(f"refusing to overwrite: {core_destination}")
        if evaluation_destination is not None and evaluation_destination.exists():
            raise FileExistsError(f"refusing to overwrite: {evaluation_destination}")
        core_destination.mkdir(parents=True)

        topology = pd.read_csv(self._native_path(source, "topology.csv"))
        service_windows = pd.read_csv(
            self._native_path(source, "entity_service_windows.csv"),
            parse_dates=["install_ts", "decommission_ts"],
        )
        engineering_path = self._native_path(
            source, "engineering_events.csv", required=False
        )
        engineering = (
            pd.read_csv(engineering_path, parse_dates=["ts"])
            if engineering_path is not None
            else None
        )
        panel_path = self._native_path(source, "reference_dataset.parquet")
        panel_file = pq.ParquetFile(panel_path)
        native_columns = set(panel_file.schema_arrow.names)
        metric_fields = [
            field for field in self._metric_mapping() if field in native_columns
        ]
        presence = self._load_presence(panel_file)
        cadence = self._median_cadence(presence)
        selected_entities = set(presence["ont_id"].astype(str))
        topology = topology.loc[topology["ont_id"].astype(str).isin(selected_entities)]
        service_windows = service_windows.loc[
            service_windows["entity_id"].astype(str).isin(selected_entities)
        ]
        selected_graph_entities = set(selected_entities)
        for field, _ in TOPOLOGY_LEVELS:
            if field in topology:
                selected_graph_entities.update(topology[field].dropna().astype(str))
        if engineering is not None:
            engineering = engineering.loc[
                engineering["entity_id"].astype(str).isin(selected_graph_entities)
            ].copy()
            engineering["ts"] = pd.to_datetime(engineering["ts"], utc=True)
            if self.selection.sample_start is not None:
                engineering = engineering.loc[
                    engineering["ts"].ge(_utc_timestamp(self.selection.sample_start))
                ]
            if self.selection.sample_end is not None:
                engineering = engineering.loc[
                    engineering["ts"].lt(_utc_timestamp(self.selection.sample_end))
                ]
        sidecars = {
            "metric_catalogue": self.metric_catalogue(native_columns),
            "entity_registry": self.entity_registry(topology, service_windows),
            "entity_relations": self.entity_relations(topology, service_windows),
            "operational_events": self.operational_events(engineering),
            "collection_gaps": self.collection_gaps(
                presence, service_windows, cadence=cadence
            ),
        }
        validation_view = {
            "telemetry": _empty_frame(
                [
                    "event_ts",
                    "ingested_at",
                    "entity_id",
                    "metric_id",
                    "value",
                    "quality_code",
                    "quality_detail",
                    "exposure",
                    "source_id",
                ]
            ),
            **sidecars,
        }
        errors = validate_core_bundle(validation_view)
        if errors:
            raise ValueError("SPEC-CORE adaptation failed: " + "; ".join(errors))

        telemetry_dir = core_destination / "telemetry"
        telemetry_dir.mkdir()
        telemetry_rows = 0
        part_index = 0
        content_digest = hashlib.sha256()
        columns = ["timestamp_utc", "ont_id", *metric_fields]
        for batch in panel_file.iter_batches(
            batch_size=self.selection.batch_native_rows, columns=columns
        ):
            selected = self._filter_native(batch.to_pandas())
            if selected.empty:
                continue
            long = self.adapt_telemetry(
                selected, cadence_seconds=float(cadence.total_seconds())
            )
            long.to_parquet(
                telemetry_dir / f"part-{part_index:05d}.parquet",
                index=False,
                compression="zstd",
            )
            content_digest.update(
                canonical_frame_hash(
                    long, sort_by=["event_ts", "entity_id", "metric_id"]
                ).encode("ascii")
            )
            telemetry_rows += len(long)
            part_index += 1
        for table_name, table in sidecars.items():
            table.to_parquet(core_destination / f"{table_name}.parquet", index=False)

        core_hashes = {
            "telemetry": content_digest.hexdigest(),
            **{
                name: canonical_frame_hash(table)
                for name, table in sorted(sidecars.items())
            },
        }
        core_counts = {
            "telemetry": telemetry_rows,
            **{name: int(len(table)) for name, table in sidecars.items()},
        }
        (core_destination / "manifest.json").write_text(
            json.dumps(
                {
                    "contract_version": CONTRACT_VERSION,
                    "pack_version": self.pack.pack_version,
                    "adapter_version": self.adapter_version,
                    "row_counts": core_counts,
                    "canonical_content_hashes": core_hashes,
                    "selection": asdict(self.selection),
                    "cadence_seconds": float(cadence.total_seconds()),
                },
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )

        evaluation = None
        if evaluation_destination is not None:
            if not inventory.evaluation_ready:
                raise FileNotFoundError("; ".join(inventory.notes))
            registry = pd.read_csv(
                self._native_path(
                    source, "gt_fault_registry.csv", evaluation=True
                ),
                parse_dates=[
                    "onset_ts",
                    "impact_ts",
                    "repair_ts",
                    "first_observable_ts",
                ],
            )
            intervals = pd.read_csv(
                self._native_path(
                    source, "fault_entity_intervals.csv", evaluation=True
                ),
                parse_dates=["active_start_ts", "active_end_ts", "impact_ts"],
            )
            groups = pd.read_csv(
                self._native_path(source, "gt_fault_groups.csv", evaluation=True),
                parse_dates=["start_ts", "end_ts"],
            )
            tickets_path = self._native_path(
                source, "tickets.csv", evaluation=True, required=False
            )
            tickets = (
                pd.read_csv(
                    tickets_path, parse_dates=["reported_ts", "resolved_ts"]
                )
                if tickets_path is not None
                else None
            )
            benign_path = self._native_path(
                source, "gt_benign_anomalies.csv", evaluation=True, required=False
            )
            benign = (
                pd.read_csv(benign_path, parse_dates=["ts"])
                if benign_path is not None
                else None
            )
            gap_path = self._native_path(
                source,
                "gt_collection_gaps.parquet",
                evaluation=True,
                required=False,
            )
            gap_truth = pd.read_parquet(gap_path) if gap_path is not None else None

            intervals = intervals.loc[
                intervals["entity_id"].astype(str).isin(selected_entities)
            ].copy()
            if self.selection.sample_start is not None:
                start = _utc_timestamp(self.selection.sample_start)
                intervals = intervals.loc[
                    pd.to_datetime(intervals["active_end_ts"], utc=True).gt(start)
                ]
            if self.selection.sample_end is not None:
                end = _utc_timestamp(self.selection.sample_end)
                intervals = intervals.loc[
                    pd.to_datetime(intervals["active_start_ts"], utc=True).lt(end)
                ]
            selected_fault_ids = set(intervals["fault_id"].dropna().astype(str))
            registry = registry.loc[
                registry["gt_fault_id"].astype(str).isin(selected_fault_ids)
            ].copy()
            selected_group_ids = set(registry["group_id"].dropna().astype(str))
            groups = groups.loc[
                groups["group_id"].astype(str).isin(selected_group_ids)
            ].copy()
            if tickets is not None:
                tickets = tickets.loc[
                    tickets["ont_id"].astype(str).isin(selected_entities)
                ].copy()
                if "gt_fault_id" in tickets:
                    tickets = tickets.loc[
                        tickets["gt_fault_id"].isna()
                        | tickets["gt_fault_id"].astype(str).isin(selected_fault_ids)
                    ]
            if benign is not None:
                benign = benign.loc[
                    benign["entity_id"].astype(str).isin(selected_entities)
                ].copy()
            if gap_truth is not None:
                gap_truth = gap_truth.loc[
                    gap_truth["entity_id"].astype(str).isin(selected_entities)
                ].copy()
            evaluation = self.adapt_eval_frames(
                registry,
                intervals,
                groups,
                tickets=tickets,
                benign_anomalies=benign,
                collection_gap_truth=gap_truth,
            )
            self.write_bundle(evaluation, evaluation_destination)

        lineage = {
            "telemetry": {
                "source_file": "reference_dataset.parquet",
                "source_fields": ["timestamp_utc", "ont_id", *metric_fields],
                "evaluation_fields_loaded": [],
            },
            "entity_registry": {
                "source_files": ["topology.csv", "entity_service_windows.csv"],
                "validity_stored_once": True,
            },
            "entity_relations": {
                "source_file": "topology.csv",
                "relation_families": ["network_topology", "geographic_membership"],
            },
            "operational_events": {
                "source_file": "engineering_events.csv",
                "tickets_excluded": True,
            },
            "collection_gaps": {
                "source_fields": ["ont_id", "timestamp_utc"],
                "truth_reason_file_loaded": False,
                "algorithm": "sorted consecutive differences with run compression",
            },
        }
        (core_destination / "translation_lineage.json").write_text(
            json.dumps(lineage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        row_counts = {f"core.{name}": count for name, count in core_counts.items()}
        if evaluation is not None:
            row_counts.update(
                {f"eval.{name}": int(len(table)) for name, table in evaluation.items()}
            )
        truth_removed = tuple(
            sorted(name for name in native_columns if str(name).startswith("gt_"))
        )
        return MaterialisationReport(
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            source_format=NATIVE_FORMAT,
            core_path=core_destination,
            evaluation_path=evaluation_destination,
            row_counts=row_counts,
            truth_columns_removed=truth_removed,
        )
