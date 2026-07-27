"""Verified Week 1 mechanics shared by the three Drive research notebooks.

This is intentionally one flat file, not a package. It contains only behaviour where
silent divergence would invalidate the research: canonical hashing, telecom exposure
and quality semantics, truth routing, bounded Parquet materialisation, and the verified
Petrobras 3W subset/materialisation path.

Contracts, sector-pack decisions, acceptance assertions, modelling choices, and
operator ranking remain visible in notebook cells.
"""

from __future__ import annotations

import configparser
import gc
import hashlib
import itertools
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


CORE_VERSION = "0.3.0"
EVAL_VERSION = "0.3.0"
TELECOM_SOURCE_ID = "telemetry-synth-4.0.1"
THREEW_SOURCE_ID = "petrobras-3w-2.0.0"
FEC_CEILING = 5_000_000

CORE_TABLES = (
    "telemetry",
    "metric_catalogue",
    "entity_registry",
    "entity_relations",
    "operational_events",
    "collection_gaps",
)
EVAL_TABLES = (
    "gt_fault_events",
    "gt_fault_entity_intervals",
    "gt_cause_groups",
    "gt_condition_states",
    "gt_benign_anomalies",
    "gt_collection_gaps",
    "gt_ticket_links",
)

TELEMETRY_COLUMNS = (
    "event_ts",
    "ingested_at",
    "entity_id",
    "metric_id",
    "value",
    "quality_code",
    "quality_detail",
    "exposure",
    "source_id",
)

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


@dataclass(frozen=True)
class Selection:
    sample_start: str | None = None
    sample_end: str | None = None
    entity_ids: tuple[str, ...] = ()
    batch_native_rows: int = 100_000


@dataclass(frozen=True)
class InstanceSummary:
    path: Path
    relative_path: str
    instance_id: str
    entity_id: str
    event_code: int
    source_kind: str
    rows: int
    bytes: int
    has_state_transition: bool
    has_missing_measurement: bool
    coverage: frozenset[str]


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_frame_hash(
    frame: pd.DataFrame,
    *,
    sort_by: Iterable[str] | None = None,
) -> str:
    """Hash logical table content without depending on Parquet metadata."""

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


def write_json(path: str | Path, payload: Any, *, overwrite: bool = False) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite immutable artifact: {destination}")
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


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


def _empty(columns: Iterable[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _roots(source: Path, *, evaluation: bool) -> tuple[Path, ...]:
    source = Path(source)
    roots: list[Path] = [source, source / "Data"]
    if evaluation and source.is_dir():
        for candidate in source.iterdir():
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


def native_path(
    source: str | Path,
    filename: str,
    *,
    evaluation: bool = False,
    required: bool = True,
) -> Path | None:
    matches = [
        root / filename
        for root in _roots(Path(source), evaluation=evaluation)
        if (root / filename).is_file()
    ]
    if len(matches) > 1:
        raise ValueError(f"ambiguous native file {filename!r}: {matches}")
    if matches:
        return matches[0]
    if required:
        raise FileNotFoundError(f"native file not found: {filename}")
    return None


def discover_telecom(source: str | Path) -> dict[str, Any]:
    core_required = (
        "reference_dataset.parquet",
        "topology.csv",
        "entity_service_windows.csv",
    )
    eval_required = (
        "fault_entity_intervals.csv",
        "gt_fault_groups.csv",
        "gt_fault_registry.csv",
    )
    missing_core = [
        name
        for name in core_required
        if native_path(source, name, required=False) is None
    ]
    missing_eval = [
        name
        for name in eval_required
        if native_path(source, name, evaluation=True, required=False) is None
    ]
    return {
        "core_ready": not missing_core,
        "evaluation_ready": not missing_eval,
        "missing_core": missing_core,
        "missing_evaluation": missing_eval,
    }


def filter_native(frame: pd.DataFrame, selection: Selection) -> pd.DataFrame:
    selected = frame
    if selection.entity_ids:
        selected = selected.loc[
            selected["ont_id"].astype(str).isin(selection.entity_ids)
        ]
    timestamps = pd.to_datetime(selected["timestamp_utc"], utc=True)
    if selection.sample_start is not None:
        selected = selected.loc[timestamps.ge(pd.to_datetime(
            selection.sample_start, utc=True
        ))]
        timestamps = pd.to_datetime(selected["timestamp_utc"], utc=True)
    if selection.sample_end is not None:
        selected = selected.loc[timestamps.lt(pd.to_datetime(
            selection.sample_end, utc=True
        ))]
    return selected.reset_index(drop=True)


def infer_cadence(panel: pd.DataFrame) -> pd.Timedelta:
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


def transform_telecom_telemetry(
    panel: pd.DataFrame,
    metric_catalogue: pd.DataFrame,
    pack_parameters: Mapping[str, Any],
    *,
    cadence_seconds: float,
) -> pd.DataFrame:
    """Verified K1/M1 behaviour: throughput CRC exposure and >= FEC clipping."""

    mapping = dict(
        zip(metric_catalogue["native_field"], metric_catalogue["metric_id"])
    )
    available = [field for field in mapping if field in panel.columns]
    if not available:
        raise ValueError("native panel contains none of the pack metric fields")
    working = panel[["timestamp_utc", "ont_id", *available]].copy()
    frame_size_bytes = float(pack_parameters["crc_frame_size_bytes"]["value"])
    if "throughput_mbps" in working:
        working["_crc_exposure"] = (
            pd.to_numeric(working["throughput_mbps"], errors="coerce")
            * 1_000_000.0
            * cadence_seconds
            / (frame_size_bytes * 8.0)
        )
    else:
        working["_crc_exposure"] = np.nan

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
    is_fec_ceiling = long["metric_id"].eq("telecom.link.fec_count") & (
        pd.to_numeric(long["value"], errors="coerce").ge(FEC_CEILING)
    )
    long["quality_code"] = "measured"
    long.loc[is_null, "quality_code"] = "invalid"
    long.loc[is_fec_ceiling, "quality_code"] = "clipped"
    long["quality_detail"] = None
    long.loc[is_null, "quality_detail"] = "source_null"
    long.loc[is_fec_ceiling, "quality_detail"] = "upper_bound"

    long["exposure"] = np.nan
    fec_line_rate = float(pack_parameters["generator_fec_line_rate_bps"]["value"])
    long.loc[
        long["metric_id"].eq("telecom.link.fec_count"), "exposure"
    ] = fec_line_rate * cadence_seconds
    crc = long["metric_id"].eq("telecom.link.crc_errors")
    long.loc[crc, "exposure"] = long.loc[crc, "_crc_exposure"]
    long = long.drop(columns=["_crc_exposure"])
    long["source_id"] = TELECOM_SOURCE_ID
    return (
        long[list(TELEMETRY_COLUMNS)]
        .sort_values(["event_ts", "entity_id", "metric_id"], kind="stable")
        .reset_index(drop=True)
    )


def build_entity_registry(
    topology: pd.DataFrame,
    service_windows: pd.DataFrame,
) -> pd.DataFrame:
    service = service_windows.copy()
    service["entity_id"] = service["entity_id"].astype(str)
    service = service.set_index("entity_id")
    rows: list[dict[str, Any]] = []
    for level_field, entity_type in TOPOLOGY_LEVELS[:-1]:
        if level_field not in topology:
            continue
        for entity_id in sorted(topology[level_field].dropna().astype(str).unique()):
            rows.append(
                {
                    "entity_id": entity_id,
                    "entity_type": entity_type,
                    "valid_from": pd.NaT,
                    "valid_to": pd.NaT,
                    "attributes_json": "{}",
                    "source_id": TELECOM_SOURCE_ID,
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
                "source_id": TELECOM_SOURCE_ID,
            }
        )
    return (
        pd.DataFrame(rows)
        .drop_duplicates("entity_id")
        .sort_values("entity_id")
        .reset_index(drop=True)
    )


def build_entity_relations(
    topology: pd.DataFrame,
    service_windows: pd.DataFrame,
    relation_mappings: Iterable[Mapping[str, str]],
) -> pd.DataFrame:
    service = service_windows.copy()
    service["entity_id"] = service["entity_id"].astype(str)
    service = service.set_index("entity_id")
    rows: list[dict[str, Any]] = []
    for relation in relation_mappings:
        parent_field = relation["parent_field"]
        child_field = relation["child_field"]
        if parent_field not in topology or child_field not in topology:
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
                    "relation_type": relation["relation_type"],
                    "relation_family": relation["relation_family"],
                    "valid_from": pd.NaT if window is None else window.get("install_ts"),
                    "valid_to": pd.NaT if window is None else window.get("decommission_ts"),
                    "relation_confidence": 1.0,
                    "source": TELECOM_SOURCE_ID,
                }
            )
    columns = (
        "parent_entity_id",
        "child_entity_id",
        "relation_type",
        "relation_family",
        "valid_from",
        "valid_to",
        "relation_confidence",
        "source",
    )
    if not rows:
        return _empty(columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(
            ["relation_family", "relation_type", "parent_entity_id", "child_entity_id"]
        )
        .reset_index(drop=True)
    )


def build_operational_events(engineering_events: pd.DataFrame | None) -> pd.DataFrame:
    columns = (
        "event_id",
        "entity_id",
        "event_type",
        "event_start",
        "event_end",
        "known_at",
        "attributes_json",
        "source",
    )
    if engineering_events is None or not len(engineering_events):
        return _empty(columns)
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
                "source": TELECOM_SOURCE_ID,
            }
        )
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["event_start", "event_id"], kind="stable")
        .reset_index(drop=True)
    )


def build_collection_gaps(
    presence: pd.DataFrame,
    service_windows: pd.DataFrame,
    selection: Selection,
    *,
    cadence: pd.Timedelta,
) -> pd.DataFrame:
    """Compress gaps using consecutive differences, never a timestamp Cartesian set."""

    observed = presence[["ont_id", "timestamp_utc"]].copy()
    observed["ont_id"] = observed["ont_id"].astype(str)
    observed["timestamp_utc"] = pd.to_datetime(observed["timestamp_utc"], utc=True)
    observed = observed.drop_duplicates().sort_values(
        ["ont_id", "timestamp_utc"], kind="stable"
    )
    columns = ("entity_id", "gap_start", "gap_end", "known_at", "source")
    if observed.empty:
        return _empty(columns)
    global_start = observed["timestamp_utc"].min()
    global_end = observed["timestamp_utc"].max() + cadence
    if selection.sample_start is not None:
        global_start = max(
            global_start, pd.to_datetime(selection.sample_start, utc=True)
        )
    if selection.sample_end is not None:
        global_end = min(global_end, pd.to_datetime(selection.sample_end, utc=True))
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
                    "source": TELECOM_SOURCE_ID,
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
        timestamps = group["timestamp_utc"].array
        first = pd.Timestamp(timestamps[0])
        add_gap(entity_id, valid_start, first)
        previous = first
        for raw_timestamp in timestamps[1:]:
            timestamp = pd.Timestamp(raw_timestamp)
            if timestamp - previous > cadence:
                add_gap(entity_id, previous + cadence, timestamp)
            previous = timestamp
        add_gap(entity_id, previous + cadence, valid_end)
    return pd.DataFrame(rows, columns=columns)


def translate_telecom_eval(
    fault_registry: pd.DataFrame,
    fault_entity_intervals: pd.DataFrame,
    fault_groups: pd.DataFrame,
    *,
    tickets: pd.DataFrame | None = None,
    benign_anomalies: pd.DataFrame | None = None,
    collection_gap_truth: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Route all supplied labels and tickets exclusively into SPEC-EVAL."""

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
    interval_columns = (
        "fault_event_id",
        "affected_entity_id",
        "fault_family",
        "channel",
        "active_start_ts",
        "active_end_ts",
        "impact_ts",
        "contribution",
    )
    for column in interval_columns:
        if column not in intervals:
            intervals[column] = pd.NA
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
    condition_columns = (
        "entity_id",
        "condition_start_ts",
        "condition_end_ts",
        "condition_code",
        "condition_label",
        "label_source",
        "source_instance_id",
    )
    bundle = {
        "gt_fault_events": fault_events,
        "gt_fault_entity_intervals": intervals[list(interval_columns)],
        "gt_cause_groups": cause_groups,
        "gt_condition_states": _empty(condition_columns),
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
        )
        bundle["gt_benign_anomalies"] = benign[
            ["entity_id", "event_ts", "benign_type", "n_samples"]
        ]
    if collection_gap_truth is not None:
        gap_truth = collection_gap_truth.rename(
            columns={"ts": "event_ts", "gt_gap_reason": "gap_reason"}
        )
        bundle["gt_collection_gaps"] = gap_truth[
            ["entity_id", "event_ts", "gap_reason"]
        ]
    return bundle


def _write_bundle(
    bundle: Mapping[str, pd.DataFrame],
    destination: Path,
    *,
    contract_version: str,
) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite: {destination}")
    destination.mkdir(parents=True)
    counts, hashes = {}, {}
    for name, frame in bundle.items():
        counts[name] = int(len(frame))
        hashes[name] = canonical_frame_hash(frame)
        frame.to_parquet(destination / f"{name}.parquet", index=False)
    manifest = {
        "contract_version": contract_version,
        "row_counts": counts,
        "canonical_content_hashes": hashes,
    }
    write_json(destination / "manifest.json", manifest)
    return manifest


def materialise_telecom(
    source: str | Path,
    run_root: str | Path,
    metric_catalogue: pd.DataFrame,
    relation_mappings: Iterable[Mapping[str, str]],
    pack_parameters: Mapping[str, Any],
    *,
    selection: Selection | None = None,
    include_evaluation: bool = True,
) -> dict[str, Any]:
    """Materialise telecom data in bounded batches with truth physically separate."""

    selection = selection or Selection()
    source = Path(source)
    run_root = Path(run_root)
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite immutable run: {run_root}")
    inventory = discover_telecom(source)
    if not inventory["core_ready"]:
        raise FileNotFoundError(inventory)
    if include_evaluation and not inventory["evaluation_ready"]:
        raise FileNotFoundError(inventory)
    core = run_root / "SPEC-CORE"
    evaluation_root = run_root / "SPEC-EVAL"
    telemetry_dir = core / "telemetry"
    telemetry_dir.mkdir(parents=True)

    topology = pd.read_csv(native_path(source, "topology.csv"))
    windows = pd.read_csv(
        native_path(source, "entity_service_windows.csv"),
        parse_dates=["install_ts", "decommission_ts"],
    )
    engineering_file = native_path(
        source, "engineering_events.csv", required=False
    )
    engineering = (
        pd.read_csv(engineering_file, parse_dates=["ts"])
        if engineering_file is not None
        else None
    )
    panel_file = pq.ParquetFile(native_path(source, "reference_dataset.parquet"))
    native_columns = set(panel_file.schema_arrow.names)
    available_metrics = [
        field
        for field in metric_catalogue["native_field"]
        if field in native_columns
    ]

    presence_parts = []
    for batch in panel_file.iter_batches(
        batch_size=max(250_000, selection.batch_native_rows),
        columns=["timestamp_utc", "ont_id"],
    ):
        selected = filter_native(batch.to_pandas(), selection)
        if len(selected):
            presence_parts.append(selected)
    if not presence_parts:
        raise ValueError("selection contains no native telemetry rows")
    presence = pd.concat(presence_parts, ignore_index=True)
    cadence = infer_cadence(presence)
    selected_entities = set(presence["ont_id"].astype(str))
    topology = topology.loc[topology["ont_id"].astype(str).isin(selected_entities)]
    windows = windows.loc[windows["entity_id"].astype(str).isin(selected_entities)]

    graph_entities = set(selected_entities)
    for field, _ in TOPOLOGY_LEVELS:
        if field in topology:
            graph_entities.update(topology[field].dropna().astype(str))
    if engineering is not None:
        engineering = engineering.loc[
            engineering["entity_id"].astype(str).isin(graph_entities)
        ].copy()
    catalogue = metric_catalogue.loc[
        metric_catalogue["native_field"].isin(available_metrics)
    ].copy()
    sidecars = {
        "metric_catalogue": catalogue.drop(columns=["native_field"]).reset_index(drop=True),
        "entity_registry": build_entity_registry(topology, windows),
        "entity_relations": build_entity_relations(
            topology, windows, relation_mappings
        ),
        "operational_events": build_operational_events(engineering),
        "collection_gaps": build_collection_gaps(
            presence, windows, selection, cadence=cadence
        ),
    }
    for name, frame in sidecars.items():
        frame.to_parquet(core / f"{name}.parquet", index=False)

    telemetry_digest = hashlib.sha256()
    telemetry_rows = 0
    part_index = 0
    peak_rss = _rss_gib()
    columns = ["timestamp_utc", "ont_id", *available_metrics]
    for batch in panel_file.iter_batches(
        batch_size=selection.batch_native_rows,
        columns=columns,
    ):
        selected = filter_native(batch.to_pandas(), selection)
        if selected.empty:
            continue
        long = transform_telecom_telemetry(
            selected,
            catalogue,
            pack_parameters,
            cadence_seconds=float(cadence.total_seconds()),
        )
        long.to_parquet(
            telemetry_dir / f"part-{part_index:05d}.parquet",
            index=False,
            compression="zstd",
        )
        telemetry_digest.update(
            canonical_frame_hash(
                long, sort_by=["event_ts", "entity_id", "metric_id"]
            ).encode("ascii")
        )
        telemetry_rows += len(long)
        part_index += 1
        peak_rss = max(peak_rss, _rss_gib())
        del selected, long
        gc.collect()

    core_counts = {
        "telemetry": telemetry_rows,
        **{name: int(len(frame)) for name, frame in sidecars.items()},
    }
    core_hashes = {
        "telemetry": telemetry_digest.hexdigest(),
        **{name: canonical_frame_hash(frame) for name, frame in sidecars.items()},
    }
    core_manifest = {
        "contract_version": CORE_VERSION,
        "row_counts": core_counts,
        "canonical_content_hashes": core_hashes,
        "selection": asdict(selection),
        "cadence_seconds": float(cadence.total_seconds()),
        "truth_columns_removed": sorted(
            name for name in native_columns if str(name).startswith("gt_")
        ),
    }
    write_json(core / "manifest.json", core_manifest)
    write_json(
        core / "translation_lineage.json",
        {
            "telemetry": {
                "source_fields": ["timestamp_utc", "ont_id", *available_metrics],
                "evaluation_fields_loaded": [],
            },
            "entity_validity_stored_once": True,
            "tickets_excluded_from_core": True,
            "collection_gap_algorithm": "consecutive differences and run compression",
        },
    )

    eval_manifest = None
    if include_evaluation:
        registry = pd.read_csv(
            native_path(source, "gt_fault_registry.csv", evaluation=True),
            parse_dates=[
                "onset_ts",
                "first_observable_ts",
                "impact_ts",
                "repair_ts",
            ],
        )
        intervals = pd.read_csv(
            native_path(source, "fault_entity_intervals.csv", evaluation=True),
            parse_dates=["active_start_ts", "active_end_ts", "impact_ts"],
        )
        groups = pd.read_csv(
            native_path(source, "gt_fault_groups.csv", evaluation=True),
            parse_dates=["start_ts", "end_ts"],
        )
        intervals = intervals.loc[
            intervals["entity_id"].astype(str).isin(selected_entities)
        ].copy()
        selected_faults = set(intervals["fault_id"].dropna().astype(str))
        registry = registry.loc[
            registry["gt_fault_id"].astype(str).isin(selected_faults)
        ].copy()
        selected_groups = set(registry["group_id"].dropna().astype(str))
        groups = groups.loc[
            groups["group_id"].astype(str).isin(selected_groups)
        ].copy()
        tickets_file = native_path(
            source, "tickets.csv", evaluation=True, required=False
        )
        tickets = (
            pd.read_csv(tickets_file, parse_dates=["reported_ts", "resolved_ts"])
            if tickets_file is not None
            else None
        )
        if tickets is not None:
            tickets = tickets.loc[
                tickets["ont_id"].astype(str).isin(selected_entities)
            ].copy()
        benign_file = native_path(
            source, "gt_benign_anomalies.csv", evaluation=True, required=False
        )
        benign = (
            pd.read_csv(benign_file, parse_dates=["ts"])
            if benign_file is not None
            else None
        )
        gap_file = native_path(
            source,
            "gt_collection_gaps.parquet",
            evaluation=True,
            required=False,
        )
        gap_truth = pd.read_parquet(gap_file) if gap_file is not None else None
        eval_bundle = translate_telecom_eval(
            registry,
            intervals,
            groups,
            tickets=tickets,
            benign_anomalies=benign,
            collection_gap_truth=gap_truth,
        )
        eval_manifest = _write_bundle(
            eval_bundle, evaluation_root, contract_version=EVAL_VERSION
        )

    stored_bytes = sum(path.stat().st_size for path in telemetry_dir.glob("*.parquet"))
    report = {
        "workflow": "telecom_materialisation",
        "week1_core_sha256": sha256_file(__file__),
        "source": str(source),
        "run_root": str(run_root),
        "core_manifest": core_manifest,
        "evaluation_manifest": eval_manifest,
        "memory": {
            "peak_observed_rss_gib": peak_rss,
            "scope": "fresh notebook process sampled after each physical batch",
        },
        "representation": {
            "logical_shape": "long",
            "physical_shape": "partitioned_parquet",
            "native_rows": int(len(presence)),
            "canonical_rows": telemetry_rows,
            "row_multiplier": telemetry_rows / max(len(presence), 1),
            "stored_bytes": stored_bytes,
        },
    }
    write_json(run_root / "workflow_report.json", report)
    return report


def read_core_bundle(core: str | Path) -> dict[str, pd.DataFrame]:
    core = Path(core)
    telemetry_parts = sorted((core / "telemetry").glob("part-*.parquet"))
    return {
        "telemetry": pd.concat(
            (pd.read_parquet(path) for path in telemetry_parts),
            ignore_index=True,
        ),
        **{
            name: pd.read_parquet(core / f"{name}.parquet")
            for name in CORE_TABLES
            if name != "telemetry"
        },
    }


def timestamp_canary_leaks(
    bundle: Mapping[str, pd.DataFrame],
    canaries: Iterable[pd.Timestamp],
) -> list[str]:
    expected = {pd.Timestamp(value) for value in canaries}
    leaks = []
    for table_name, frame in bundle.items():
        for column in frame.columns:
            name = str(column).lower()
            timestamp_name = (
                "ts" in name
                or "time" in name
                or name.endswith(("_at", "_start", "_end", "_from", "_to"))
            )
            timestamp_dtype = pd.api.types.is_datetime64_any_dtype(frame[column])
            if not timestamp_name and not timestamp_dtype:
                continue
            values = pd.to_datetime(frame[column], utc=True, errors="coerce")
            if any(values.eq(value).any() for value in expected):
                leaks.append(f"{table_name}.{column}")
    return sorted(leaks)


def runtime_probe(core: str | Path) -> str:
    """Deterministic placeholder score that has no SPEC-EVAL path or argument."""

    core = Path(core)
    totals = []
    for part in sorted((core / "telemetry").glob("part-*.parquet")):
        frame = pd.read_parquet(
            part, columns=["entity_id", "metric_id", "value", "quality_code"]
        )
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        totals.append(
            frame.groupby(["entity_id", "metric_id"], as_index=False)
            .agg(value_sum=("value", "sum"), observed=("value", "count"))
        )
    combined = pd.concat(totals, ignore_index=True)
    combined = (
        combined.groupby(["entity_id", "metric_id"], as_index=False)
        .agg(value_sum=("value_sum", "sum"), observed=("observed", "sum"))
    )
    return canonical_frame_hash(combined, sort_by=["entity_id", "metric_id"])


def audit_telecom_output(core: str | Path) -> dict[str, Any]:
    core = Path(core)
    quality: dict[str, int] = {}
    exposure = {
        "telecom.link.fec_count": [float("inf"), float("-inf")],
        "telecom.link.crc_errors": [float("inf"), float("-inf")],
    }
    for part in sorted((core / "telemetry").glob("part-*.parquet")):
        frame = pd.read_parquet(
            part, columns=["metric_id", "quality_code", "exposure"]
        )
        for code, count in frame["quality_code"].value_counts().items():
            quality[str(code)] = quality.get(str(code), 0) + int(count)
        for metric, limits in exposure.items():
            values = pd.to_numeric(
                frame.loc[frame["metric_id"].eq(metric), "exposure"],
                errors="coerce",
            ).dropna()
            if len(values):
                limits[0] = min(limits[0], float(values.min()))
                limits[1] = max(limits[1], float(values.max()))
    return {"quality_counts": quality, "exposure_ranges": exposure}


def _rss_gib() -> float:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024**3)
    except ImportError:
        return float("nan")


# ---------------------------------------------------------------------------
# Petrobras 3W: verified real-source selector and bounded materialisation
# ---------------------------------------------------------------------------

THREEW_EXPECTED_VERSION = "2.0.0"
THREEW_EXPECTED_INSTANCE_COUNT = 2228
THREEW_TRANSIENT_CODES = {1, 2, 5, 6, 7, 8, 9}
THREEW_PERSISTENT_CODES = {3, 4}
THREEW_BATCH_NATIVE_ROWS = 5_000


def discover_threew(source: str | Path) -> dict[str, Any]:
    source = Path(source)
    parser = configparser.ConfigParser()
    config_path = source / "dataset.ini"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing 3W configuration: {config_path}")
    parser.read(config_path, encoding="utf-8")
    version = parser.get("VERSION", "DATASET")
    # Only directories 0–9 contain event instances. The public release can also
    # contain auxiliary Parquet files under folds/, which are not dataset instances.
    files = sorted(
        path
        for event_code in range(10)
        for path in (source / str(event_code)).glob("*.parquet")
    )
    all_direct_child_parquet = sorted(source.glob("*/*.parquet"))
    auxiliary_files = sorted(set(all_direct_child_parquet) - set(files))
    missing_dirs = [str(code) for code in range(10) if not (source / str(code)).is_dir()]
    return {
        "version": version,
        "file_count": len(files),
        "auxiliary_parquet_file_count": len(auxiliary_files),
        "auxiliary_parquet_directories": sorted(
            {path.parent.name for path in auxiliary_files}
        ),
        "missing_event_directories": missing_dirs,
        "ready": (
            version == THREEW_EXPECTED_VERSION
            and len(files) == THREEW_EXPECTED_INSTANCE_COUNT
            and not missing_dirs
        ),
    }


def _threew_source_kind(filename: str) -> str:
    if filename.startswith("WELL-"):
        return "real"
    if filename.startswith("SIMULATED_"):
        return "simulated"
    if filename.startswith("DRAWN_"):
        return "hand_drawn"
    raise ValueError(f"unrecognised 3W filename: {filename}")


def _threew_entity_id(filename: str) -> str:
    return filename.split("_", 1)[0] if filename.startswith("WELL-") else Path(filename).stem


def _threew_summary(
    path: Path,
    root: Path,
    native_metric_fields: Iterable[str],
) -> InstanceSummary:
    parquet = pq.ParquetFile(path)
    names = parquet.schema_arrow.names
    state_index = names.index("state")
    metric_indexes = [names.index(field) for field in native_metric_fields]
    state_min, state_max, missing = None, None, False
    for group_index in range(parquet.metadata.num_row_groups):
        row_group = parquet.metadata.row_group(group_index)
        state_stats = row_group.column(state_index).statistics
        if state_stats is not None and state_stats.has_min_max:
            state_min = (
                state_stats.min if state_min is None else min(state_min, state_stats.min)
            )
            state_max = (
                state_stats.max if state_max is None else max(state_max, state_stats.max)
            )
        for column_index in metric_indexes:
            stats = row_group.column(column_index).statistics
            if stats is not None and (stats.null_count or 0) > 0:
                missing = True
                break
    event_code = int(path.parent.name)
    coverage: set[str] = set()
    if event_code == 0:
        coverage.add("normal_instance")
    if event_code in THREEW_TRANSIENT_CODES:
        coverage.add("transient_event")
    if event_code in THREEW_PERSISTENT_CODES:
        coverage.add("persistent_condition")
    state_transition = (
        state_min is not None and state_max is not None and state_min != state_max
    )
    if state_transition:
        coverage.add("state_transition")
    if missing:
        coverage.add("missing_or_frozen_measurement")
    return InstanceSummary(
        path=path,
        relative_path=str(path.relative_to(root)),
        instance_id=path.stem,
        entity_id=_threew_entity_id(path.name),
        event_code=event_code,
        source_kind=_threew_source_kind(path.name),
        rows=parquet.metadata.num_rows,
        bytes=path.stat().st_size,
        has_state_transition=state_transition,
        has_missing_measurement=missing,
        coverage=frozenset(coverage),
    )


def select_threew_subset(
    source: str | Path,
    native_metric_fields: Iterable[str],
) -> list[InstanceSummary]:
    source = Path(source)
    candidates = []
    for code in range(10):
        candidates.extend(sorted((source / str(code)).glob("WELL-*.parquet"))[:8])
    summaries = sorted(
        (_threew_summary(path, source, native_metric_fields) for path in candidates),
        key=lambda item: item.relative_path,
    )
    required = {
        "normal_instance",
        "transient_event",
        "persistent_condition",
        "state_transition",
        "missing_or_frozen_measurement",
    }
    for size in range(1, 8):
        valid = []
        for combination in itertools.combinations(summaries, size):
            if len({item.entity_id for item in combination}) < 3:
                continue
            if len({item.entity_id for item in combination}) != size:
                continue
            covered = set().union(*(item.coverage for item in combination))
            if required <= covered:
                valid.append(
                    (
                        sum(item.bytes for item in combination),
                        tuple(item.relative_path for item in combination),
                        combination,
                    )
                )
        if valid:
            return list(min(valid, key=lambda item: (item[0], item[1]))[2])
    raise ValueError("no real-well subset satisfies all five contract criteria")


def _contiguous_runs(values) -> list[tuple[int, int, Any]]:
    normalised = values.astype("object").where(values.notna(), "unknown").tolist()
    if not normalised:
        return []
    runs, start, current = [], 0, normalised[0]
    for index, value in enumerate(normalised[1:], start=1):
        if value != current:
            runs.append((start, index, current))
            start, current = index, value
    runs.append((start, len(normalised), current))
    return runs


def _threew_condition_rows(frame, summary: InstanceSummary) -> list[dict[str, Any]]:
    timestamps = frame["timestamp"].reset_index(drop=True)
    states = frame["state"].reset_index(drop=True)
    rows = []
    for start, end, value in _contiguous_runs(states):
        end_ts = (
            timestamps.iloc[end]
            if end < len(timestamps)
            else timestamps.iloc[-1] + pd.Timedelta(seconds=1)
        )
        code = "unknown" if value == "unknown" else str(int(value))
        rows.append(
            {
                "entity_id": summary.entity_id,
                "condition_start_ts": timestamps.iloc[start],
                "condition_end_ts": end_ts,
                "condition_code": code,
                "condition_label": f"source_state_{code}",
                "label_source": "petrobras_3w_state",
                "source_instance_id": summary.instance_id,
            }
        )
    return rows


def _threew_event_rows(
    frame,
    summary: InstanceSummary,
    event_descriptions: Mapping[int, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if summary.event_code == 0:
        return [], []
    timestamps = frame["timestamp"].reset_index(drop=True)
    labels = frame["class"].reset_index(drop=True)
    relevant = labels.map(
        lambda value: (
            False
            if pd.isna(value)
            else int(value) in {summary.event_code, 100 + summary.event_code}
        )
    )
    events, intervals, event_index = [], [], 0
    for start, end, active in _contiguous_runs(relevant):
        if not bool(active):
            continue
        segment = labels.iloc[start:end]
        onset = timestamps.iloc[start]
        steady = segment.index[segment.eq(summary.event_code)]
        impact = timestamps.iloc[int(steady[0])] if len(steady) else pd.NaT
        resolution = timestamps.iloc[end] if end < len(timestamps) else pd.NaT
        event_id = f"3W-{summary.instance_id}-{summary.event_code}-{event_index:02d}"
        events.append(
            {
                "fault_event_id": event_id,
                "fault_type": event_descriptions[summary.event_code],
                "fault_family": f"3w_event_{summary.event_code}",
                "fault_domain_type": "oil_well",
                "fault_domain_id": summary.entity_id,
                "onset_ts": onset,
                "first_observable_ts": onset,
                "impact_ts": impact,
                "resolution_ts": resolution,
                "cause_group_id": pd.NA,
                "left_censored": start == 0,
                "label_source": "petrobras_3w_class",
                "source_instance_id": summary.instance_id,
            }
        )
        intervals.append(
            {
                "fault_event_id": event_id,
                "affected_entity_id": summary.entity_id,
                "fault_family": f"3w_event_{summary.event_code}",
                "channel": "multivariate_process",
                "active_start_ts": onset,
                "active_end_ts": (
                    resolution
                    if pd.notna(resolution)
                    else timestamps.iloc[-1] + pd.Timedelta(seconds=1)
                ),
                "impact_ts": impact,
                "contribution": pd.NA,
            }
        )
        event_index += 1
    return events, intervals


def _threew_event_descriptions(source: Path) -> dict[int, str]:
    parser = configparser.ConfigParser()
    parser.read(source / "dataset.ini", encoding="utf-8")
    names = [
        item.strip()
        for item in parser.get("EVENTS", "NAMES").replace("\n", "").split(",")
    ]
    return {
        parser.getint(name, "LABEL"): parser.get(name, "DESCRIPTION")
        for name in names
    }


def materialise_threew(
    source: str | Path,
    run_root: str | Path,
    metric_catalogue: pd.DataFrame,
    *,
    fixture_destination: str | Path | None = None,
) -> dict[str, Any]:
    source, run_root = Path(source), Path(run_root)
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite immutable run: {run_root}")
    inventory = discover_threew(source)
    if not inventory["ready"]:
        raise ValueError(inventory)
    native_fields = tuple(metric_catalogue["native_field"])
    selected = select_threew_subset(source, native_fields)
    core, evaluation = run_root / "SPEC-CORE", run_root / "SPEC-EVAL"
    telemetry_dir = core / "telemetry"
    telemetry_dir.mkdir(parents=True)

    fixture = None if fixture_destination is None else Path(fixture_destination)
    if fixture is not None:
        if fixture.exists():
            raise FileExistsError(f"refusing to overwrite fixture: {fixture}")
        fixture.mkdir(parents=True)
        for name in ("dataset.ini", "LICENSE-CC-BY", "README.md"):
            shutil.copy2(source / name, fixture / name)
        for summary in selected:
            destination = fixture / summary.relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(summary.path, destination)

    mapping = dict(zip(metric_catalogue["native_field"], metric_catalogue["metric_id"]))
    event_descriptions = _threew_event_descriptions(source)
    registry_rows, gap_rows, event_rows, interval_rows, condition_rows = [], [], [], [], []
    telemetry_digest, telemetry_rows, part_index = hashlib.sha256(), 0, 0
    peak_rss = _rss_gib()
    for summary in selected:
        frame = pd.read_parquet(summary.path).reset_index()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        for start in range(0, len(frame), THREEW_BATCH_NATIVE_ROWS):
            native_batch = frame.iloc[start : start + THREEW_BATCH_NATIVE_ROWS]
            long = native_batch[["timestamp", *mapping]].melt(
                id_vars=["timestamp"],
                value_vars=list(mapping),
                var_name="native_metric",
                value_name="value",
            )
            long["event_ts"] = long.pop("timestamp")
            long["ingested_at"] = long["event_ts"]
            long["entity_id"] = summary.entity_id
            long["metric_id"] = long.pop("native_metric").map(mapping)
            invalid = long["value"].isna()
            long["quality_code"] = invalid.map({True: "invalid", False: "measured"})
            long["quality_detail"] = invalid.map(
                {True: "source_null", False: None}
            )
            long["exposure"] = np.nan
            long["source_id"] = THREEW_SOURCE_ID
            long = long[list(TELEMETRY_COLUMNS)]
            long.to_parquet(
                telemetry_dir / f"part-{part_index:05d}.parquet",
                index=False,
                compression="zstd",
            )
            telemetry_digest.update(
                canonical_frame_hash(
                    long, sort_by=["event_ts", "entity_id", "metric_id"]
                ).encode("ascii")
            )
            telemetry_rows += len(long)
            part_index += 1
            peak_rss = max(peak_rss, _rss_gib())
            del native_batch, long
            gc.collect()
        registry_rows.append(
            {
                "entity_id": summary.entity_id,
                "entity_type": "oil_well",
                "valid_from": frame["timestamp"].min(),
                "valid_to": frame["timestamp"].max() + pd.Timedelta(seconds=1),
                "attributes_json": json.dumps(
                    {
                        "source_kind": summary.source_kind,
                        "source_instance_id": summary.instance_id,
                        "event_code": summary.event_code,
                    },
                    sort_keys=True,
                ),
                "source_id": THREEW_SOURCE_ID,
            }
        )
        timestamps = frame["timestamp"].drop_duplicates().sort_values()
        deltas = timestamps.diff()
        for index in deltas[deltas > pd.Timedelta(seconds=1)].index:
            previous = timestamps.iloc[timestamps.index.get_loc(index) - 1]
            current = timestamps.loc[index]
            gap_rows.append(
                {
                    "entity_id": summary.entity_id,
                    "gap_start": previous + pd.Timedelta(seconds=1),
                    "gap_end": current,
                    "known_at": current,
                    "source": THREEW_SOURCE_ID,
                }
            )
        condition_rows.extend(_threew_condition_rows(frame, summary))
        events, intervals = _threew_event_rows(
            frame, summary, event_descriptions
        )
        event_rows.extend(events)
        interval_rows.extend(intervals)
        del frame
        gc.collect()

    relation_columns = (
        "parent_entity_id",
        "child_entity_id",
        "relation_type",
        "relation_family",
        "valid_from",
        "valid_to",
        "relation_confidence",
        "source",
    )
    operational_columns = (
        "event_id",
        "entity_id",
        "event_type",
        "event_start",
        "event_end",
        "known_at",
        "attributes_json",
        "source",
    )
    sidecars = {
        "metric_catalogue": metric_catalogue.drop(columns=["native_field"]).copy(),
        "entity_registry": pd.DataFrame(registry_rows),
        "entity_relations": _empty(relation_columns),
        "operational_events": _empty(operational_columns),
        "collection_gaps": pd.DataFrame(
            gap_rows,
            columns=["entity_id", "gap_start", "gap_end", "known_at", "source"],
        ),
    }
    for name, frame in sidecars.items():
        frame.to_parquet(core / f"{name}.parquet", index=False)
    core_manifest = {
        "contract_version": CORE_VERSION,
        "row_counts": {
            "telemetry": telemetry_rows,
            **{name: int(len(frame)) for name, frame in sidecars.items()},
        },
        "canonical_content_hashes": {
            "telemetry": telemetry_digest.hexdigest(),
            **{name: canonical_frame_hash(frame) for name, frame in sidecars.items()},
        },
        "physical_telemetry_batch_native_rows": THREEW_BATCH_NATIVE_ROWS,
        "truth_columns_removed": ["class", "state"],
    }
    write_json(core / "manifest.json", core_manifest)

    eval_bundle = {
        "gt_fault_events": pd.DataFrame(event_rows),
        "gt_fault_entity_intervals": pd.DataFrame(interval_rows),
        "gt_cause_groups": _empty(
            ["cause_group_id", "cause_type", "start_ts", "end_ts", "footprint_json"]
        ),
        "gt_condition_states": pd.DataFrame(condition_rows),
    }
    eval_manifest = _write_bundle(
        eval_bundle, evaluation, contract_version=EVAL_VERSION
    )
    selected_manifest = {
        "dataset": "Petrobras 3W",
        "dataset_version": THREEW_EXPECTED_VERSION,
        "selection_rule": "smallest deterministic real-well subset satisfying all criteria",
        "selected": [
            {
                "relative_path": summary.relative_path,
                "entity_id": summary.entity_id,
                "coverage": sorted(summary.coverage),
                "rows": summary.rows,
                "bytes": summary.bytes,
                "sha256": sha256_file(summary.path),
            }
            for summary in selected
        ],
    }
    if fixture is not None:
        write_json(fixture / "SUBSET_MANIFEST.json", selected_manifest)
    report = {
        "workflow": "petrobras_3w_contract_challenge",
        "week1_core_sha256": sha256_file(__file__),
        "source": str(source),
        "run_root": str(run_root),
        "selected": selected_manifest["selected"],
        "core_manifest": core_manifest,
        "evaluation_manifest": eval_manifest,
        "peak_observed_rss_gib": peak_rss,
    }
    write_json(run_root / "workflow_report.json", report)
    return report


__all__ = [
    "CORE_TABLES",
    "CORE_VERSION",
    "EVAL_TABLES",
    "EVAL_VERSION",
    "FEC_CEILING",
    "Selection",
    "audit_telecom_output",
    "canonical_frame_hash",
    "discover_telecom",
    "discover_threew",
    "materialise_telecom",
    "materialise_threew",
    "native_path",
    "read_core_bundle",
    "runtime_probe",
    "select_threew_subset",
    "sha256_file",
    "timestamp_canary_leaks",
    "transform_telecom_telemetry",
    "write_json",
]
