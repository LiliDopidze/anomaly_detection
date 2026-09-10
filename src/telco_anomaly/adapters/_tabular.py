"""Small, explicit helpers for public tabular Telecom datasets.

The public adapters deliberately require a reviewed mapping.  Column names are
easy to discover; their engineering meaning, units and topology are not safe to
guess.  This module handles the mechanical conversion after those decisions
have been written down.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from telco_anomaly.contract import (
    OPTIONAL_CORE_SCHEMAS,
    PACK_SCHEMAS,
    SPLIT_SCHEMAS,
    save_pack,
    source_file,
    truth_like_columns,
)


def chronological_splits(entity_bounds, *, split_version):
    """Create label-free 50/25/25 chronological research partitions."""

    first = min(bounds[0] for bounds in entity_bounds.values())
    last = max(bounds[1] for bounds in entity_bounds.values())
    span = last - first
    if span <= pd.Timedelta(0):
        raise ValueError("Public telemetry needs more than one timestamp")
    edges = [first, first + span * 0.50, first + span * 0.75, last + pd.Timedelta(microseconds=1)]
    return pd.DataFrame([
        (name, edges[index], edges[index + 1], split_version)
        for index, name in enumerate(("calibration", "development", "holdout"))
    ], columns=SPLIT_SCHEMAS["time_partitions"])


def data_files(source, patterns):
    """Return unique CSV/Parquet files selected by explicit glob patterns."""

    source = Path(source)
    files = {
        path.resolve()
        for pattern in patterns
        for path in source.glob(pattern)
        if path.is_file() and path.suffix.lower() in {".csv", ".parquet"}
    }
    if not files:
        raise FileNotFoundError(
            f"No CSV or Parquet files under {source} matched {list(patterns)}"
        )
    return sorted(files)


def table_columns(path, *, csv_options=None):
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pq.ParquetFile(path).schema_arrow.names
    return pd.read_csv(path, nrows=0, **(csv_options or {})).columns.tolist()


def table_batches(path, columns, *, batch_rows, csv_options=None):
    """Stream selected source columns without loading a whole public dataset."""

    path = Path(path)
    if path.suffix.lower() == ".parquet":
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_rows, columns=columns):
            yield batch.to_pandas()
        return
    yield from pd.read_csv(
        path,
        usecols=columns,
        chunksize=batch_rows,
        **(csv_options or {}),
    )


def inspect_tables(source, patterns, *, csv_options=None, sample_rows=5):
    """Describe source structure without assigning any metric meaning."""

    files = data_files(source, patterns)
    inventories = []
    union = set()
    for path in files:
        columns = table_columns(path, csv_options=csv_options)
        union.update(columns)
        rows = pq.ParquetFile(path).metadata.num_rows if path.suffix.lower() == ".parquet" else None
        inventories.append({
            "relative_path": str(path.relative_to(Path(source).resolve())),
            "format": path.suffix.lower().lstrip("."),
            "rows": rows,
            "columns": columns,
        })

    first = files[0]
    if first.suffix.lower() == ".parquet":
        sample = pd.read_parquet(first).head(sample_rows)
    else:
        sample = pd.read_csv(first, nrows=sample_rows, **(csv_options or {}))
    return {
        "file_count": len(files),
        "files": inventories,
        "columns": sorted(union),
        "sample": sample,
    }


def _require_mapping(mapping):
    required = {
        "file_patterns", "timestamp_column", "entity_column", "entity_type",
        "source_system", "metrics",
    }
    missing = sorted(name for name in required if not mapping.get(name))
    if missing:
        raise ValueError(
            "The reviewed source mapping is incomplete. Set: " + ", ".join(missing)
        )
    if not isinstance(mapping["metrics"], list) or not mapping["metrics"]:
        raise ValueError("The reviewed source mapping needs at least one metric")

    metric_fields = {
        "native_field", "metric_id", "measurement_kind", "unit", "direction",
        "transform", "expected_cadence_seconds", "minimum_scale",
    }
    for number, metric in enumerate(mapping["metrics"], start=1):
        unresolved = sorted(name for name in metric_fields if metric.get(name) is None)
        if unresolved:
            raise ValueError(
                f"Metric mapping {number} is incomplete. Set: {', '.join(unresolved)}"
            )
    topology_fields = {"native_field", "group_type", "group_family"}
    for number, membership in enumerate(mapping.get("topology", []), start=1):
        unresolved = sorted(
            name for name in topology_fields if membership.get(name) is None
        )
        if unresolved:
            raise ValueError(
                f"Topology mapping {number} is incomplete. Set: {', '.join(unresolved)}"
            )


def catalogue_from_mapping(mapping):
    """Create the canonical catalogue from reviewed metric declarations."""

    _require_mapping(mapping)
    rows = []
    for metric in mapping["metrics"]:
        rows.append({
            "metric_id": metric["metric_id"],
            "entity_type": metric.get("entity_type", mapping["entity_type"]),
            "measurement_kind": metric["measurement_kind"],
            "unit": metric["unit"],
            "sampling_mode": metric.get("sampling_mode", "periodic"),
            "aggregation_semantics": metric.get("aggregation_semantics", "unknown"),
            "expected_cadence_seconds": metric["expected_cadence_seconds"],
            "direction": metric["direction"],
            "transform": metric["transform"],
            "valid_min": metric.get("valid_min"),
            "valid_max": metric.get("valid_max"),
            "censoring_type": metric.get("censoring_type", "none"),
            "reporting_lod": metric.get("reporting_lod"),
            "reset_policy": metric.get("reset_policy", "not_applicable"),
            "counter_modulus": metric.get("counter_modulus"),
            "minimum_scale": metric["minimum_scale"],
            "seasonality_candidate": bool(metric.get("seasonality_candidate", False)),
            "peer_eligible": bool(metric.get("peer_eligible", True)),
        })
    catalogue = pd.DataFrame(rows, columns=PACK_SCHEMAS["metric_catalogue"])
    if catalogue["metric_id"].duplicated().any():
        raise ValueError("Reviewed metric IDs must be unique")
    return catalogue


def _timestamps(values, mapping):
    unit = mapping.get("timestamp_unit")
    if unit:
        result = pd.to_datetime(values, unit=unit, utc=True, errors="coerce")
    else:
        result = pd.to_datetime(values, utc=True, errors="coerce")
    if result.isna().any():
        raise ValueError("Source contains timestamps that the reviewed mapping cannot parse")
    return result


def scan_tabular_source(source, mapping, *, batch_rows=100_000):
    """Validate a mapping and collect bounded entity/episode metadata."""

    _require_mapping(mapping)
    source = Path(source).resolve()
    csv_options = dict(mapping.get("csv_options", {}))
    files = data_files(source, mapping["file_patterns"])
    metric_columns = [item["native_field"] for item in mapping["metrics"]]
    topology_columns = [item["native_field"] for item in mapping.get("topology", [])]
    required = [mapping["timestamp_column"], mapping["entity_column"], *metric_columns, *topology_columns]
    episode_column = mapping.get("episode_column")
    if episode_column:
        required.append(episode_column)
    required = list(dict.fromkeys(required))

    available_metrics = set()
    available_pairs = set()
    entity_bounds = {}
    episode_bounds = {}
    topology_values = defaultdict(set)
    source_columns = set()

    for path in files:
        columns = table_columns(path, csv_options=csv_options)
        source_columns.update(columns)
        missing = sorted(set(required) - set(columns))
        if missing:
            raise ValueError(f"{path.name} is missing mapped fields: {missing}")
        leaks = truth_like_columns(columns)
        if leaks:
            raise ValueError(
                f"Model-visible source {path.name} contains truth-like fields: {leaks}"
            )

        for frame in table_batches(
            path, required, batch_rows=batch_rows, csv_options=csv_options
        ):
            frame = frame.reset_index(drop=True)
            times = _timestamps(frame[mapping["timestamp_column"]], mapping).reset_index(drop=True)
            entities = frame[mapping["entity_column"]].astype("string").reset_index(drop=True)
            if entities.isna().any():
                raise ValueError(f"{path.name} contains null entity identifiers")
            if episode_column:
                episodes = frame[episode_column].astype("string").reset_index(drop=True)
                if episodes.isna().any():
                    raise ValueError(f"{path.name} contains null episode identifiers")
            else:
                episodes = mapping["source_system"] + "::" + entities

            keys = pd.DataFrame({"entity_id": entities, "episode_id": episodes, "event_ts": times})
            for key, group in keys.groupby("entity_id", sort=False):
                low, high = group["event_ts"].min(), group["event_ts"].max()
                previous = entity_bounds.get(str(key))
                entity_bounds[str(key)] = (
                    min(low, previous[0]) if previous else low,
                    max(high, previous[1]) if previous else high,
                )
            for (episode, entity), group in keys.groupby(["episode_id", "entity_id"], sort=False):
                low, high = group["event_ts"].min(), group["event_ts"].max()
                previous = episode_bounds.get(str(episode))
                if previous and previous[0] != str(entity):
                    raise ValueError(f"Episode {episode} belongs to more than one entity")
                episode_bounds[str(episode)] = (
                    str(entity),
                    min(low, previous[1]) if previous else low,
                    max(high, previous[2]) if previous else high,
                )
            for column in metric_columns:
                numeric = pd.to_numeric(frame[column], errors="coerce")
                available = pd.DataFrame({"entity_id": entities, "value": numeric})
                observed_entities = available.loc[
                    available["value"].notna(), "entity_id"
                ].drop_duplicates()
                available_pairs.update((str(entity), column) for entity in observed_entities)
                if len(observed_entities):
                    available_metrics.add(column)
            for item in mapping.get("topology", []):
                pairs = frame[[mapping["entity_column"], item["native_field"]]].dropna().drop_duplicates()
                for entity, group in pairs.itertuples(index=False, name=None):
                    topology_values[(str(entity), item["group_type"])].add(str(group))

    unavailable = sorted(set(metric_columns) - available_metrics)
    if unavailable:
        raise ValueError(f"Mapped metrics contain no numeric observations: {unavailable}")
    return {
        "files": files,
        "columns": sorted(source_columns),
        "entity_bounds": entity_bounds,
        "episode_bounds": episode_bounds,
        "available_pairs": available_pairs,
        "topology_values": topology_values,
    }


def telemetry_batches(source, mapping, scan, *, batch_rows=100_000):
    """Stream canonical long telemetry; mapped truth fields are never accepted."""

    source = Path(source).resolve()
    csv_options = dict(mapping.get("csv_options", {}))
    metric_columns = [item["native_field"] for item in mapping["metrics"]]
    native_to_metric = {
        item["native_field"]: item["metric_id"] for item in mapping["metrics"]
    }
    available_pairs = pd.DataFrame(
        sorted(scan["available_pairs"]), columns=["entity_id", "native_field"]
    )
    columns = [mapping["timestamp_column"], mapping["entity_column"], *metric_columns]
    episode_column = mapping.get("episode_column")
    if episode_column:
        columns.append(episode_column)
    columns = list(dict.fromkeys(columns))

    for path in scan["files"]:
        for frame in table_batches(
            path, columns, batch_rows=batch_rows, csv_options=csv_options
        ):
            frame = frame.reset_index(drop=True)
            entities = frame[mapping["entity_column"]].astype("string")
            if episode_column:
                episodes = frame[episode_column].astype("string")
            else:
                episodes = mapping["source_system"] + "::" + entities
            fixed = pd.DataFrame({
                "event_ts": _timestamps(frame[mapping["timestamp_column"]], mapping),
                "entity_id": entities,
                "episode_id": episodes,
            }).reset_index(drop=True)
            wide = pd.concat([fixed, frame[metric_columns].reset_index(drop=True)], axis=1)
            long = wide.melt(
                id_vars=["event_ts", "entity_id", "episode_id"],
                value_vars=metric_columns,
                var_name="native_field",
                value_name="value",
            )
            # A metric that never reports for one entity is absent, not a run
            # of invalid values.  This matters for technology-specific RAN PM
            # counters.  Once a pair has reported, its null rows remain invalid.
            long = long.merge(
                available_pairs, on=["entity_id", "native_field"], how="inner"
            )
            long["metric_id"] = long["native_field"].map(native_to_metric)
            long["value"] = pd.to_numeric(long["value"], errors="coerce")
            finite = np.isfinite(long["value"].to_numpy(dtype=float, na_value=np.nan))
            long["quality_code"] = np.where(finite, "measured", "invalid")
            long["source_system"] = mapping["source_system"]
            long["ingestion_ts"] = pd.NaT
            yield long[PACK_SCHEMAS["telemetry"]]


def pack_tables(mapping, scan):
    """Create entity, episode and optional topology tables from the scan."""

    entities = pd.DataFrame([
        {
            "entity_id": entity,
            "entity_type": mapping["entity_type"],
            "vendor": pd.NA,
            "model": pd.NA,
            "source_system": mapping["source_system"],
            "valid_from": bounds[0],
            "valid_to": pd.NaT,
        }
        for entity, bounds in sorted(scan["entity_bounds"].items())
    ], columns=PACK_SCHEMAS["entity_registry"])
    episodes = pd.DataFrame([
        {
            "episode_id": episode,
            "entity_id": values[0],
            "episode_basis": mapping.get("episode_basis", "continuous_source_recording"),
        }
        for episode, values in sorted(scan["episode_bounds"].items())
    ], columns=PACK_SCHEMAS["observation_episodes"])

    topology_rows = []
    definitions = {item["group_type"]: item for item in mapping.get("topology", [])}
    for (entity, group_type), values in sorted(scan["topology_values"].items()):
        if len(values) != 1:
            raise ValueError(
                f"Entity {entity} maps to multiple {group_type} values: {sorted(values)}"
            )
        definition = definitions[group_type]
        topology_rows.append({
            "entity_id": entity,
            "group_type": group_type,
            "group_id": next(iter(values)),
            "hierarchy_level": definition.get("hierarchy_level"),
            "group_family": definition.get("group_family", "physical_topology"),
            "valid_from": scan["entity_bounds"][entity][0],
            "valid_to": pd.NaT,
        })
    topology = None
    if topology_rows:
        topology = pd.DataFrame(
            topology_rows, columns=OPTIONAL_CORE_SCHEMAS["topology_memberships"]
        )
    return entities, episodes, topology


def build_tabular_pack(
    source,
    destination,
    mapping,
    *,
    dataset_name,
    pack_version,
    source_url,
    evidence_role,
    licence,
    licence_note,
    batch_rows=100_000,
):
    """Build an unlabelled public-data pack from a reviewed mapping."""

    scan = scan_tabular_source(source, mapping, batch_rows=batch_rows)
    catalogue = catalogue_from_mapping(mapping)
    entities, episodes, topology = pack_tables(mapping, scan)
    source_root = Path(source).resolve()
    files = [source_file(path, source_root, "model_input") for path in scan["files"]]
    source_info = {
        "source_id": dataset_name,
        "source_url": source_url,
        "licence": licence,
        "licence_note": licence_note,
        "evidence_role": evidence_role,
        "redistributed_by_project": False,
        "files": files,
    }
    return save_pack(
        destination,
        sector="telecom",
        pack_version=pack_version,
        source_info=source_info,
        telemetry=telemetry_batches(source, mapping, scan, batch_rows=batch_rows),
        catalogue=catalogue,
        entities=entities,
        episodes=episodes,
        topology=topology,
        splits={
            "time_partitions": chronological_splits(
                scan["entity_bounds"],
                split_version=f"{dataset_name}_time_v1",
            )
        },
        notes=(
            "Raw public data is referenced, not redistributed.",
            "No fault labels are asserted for this source.",
        ),
    )
