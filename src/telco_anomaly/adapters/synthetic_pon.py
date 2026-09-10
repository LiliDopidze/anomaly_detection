"""Adapter for the synthetic PON engineering fixture.

The adapter translates declared measurements and topology. It never derives
model features and never reads truth while materialising PACK-CORE.
"""

from __future__ import annotations

from pathlib import Path
import hashlib

import duckdb
import numpy as np
import pandas as pd

from telco_anomaly.contract import (
    EVAL_SCHEMAS,
    OPTIONAL_CORE_SCHEMAS,
    PACK_SCHEMAS,
    SPLIT_SCHEMAS,
    save_pack,
    source_file,
    truth_like_columns,
)


SOURCE_SYSTEM = "telemetry_synth_4_1_0"
SOURCE_INSTANCE = "telemetry_synth_4_1_0_run_1"


def _find_file(root, name, *, required=True):
    """Find one named source file and reject ambiguous copies."""

    root = Path(root)
    direct = root / name
    matches = [direct] if direct.is_file() else sorted(root.rglob(name))
    if len(matches) > 1:
        raise ValueError(f"More than one {name} found under {root}")
    if matches:
        return matches[0]
    if required:
        raise FileNotFoundError(root / name)
    return None


def metric_catalogue(registry):
    """Return canonical metric metadata and the native-to-canonical map."""

    rows = []
    native = []
    for item in registry["metrics"]:
        native.append((item["native_field"], item["metric_id"]))
        rows.append({
            "metric_id": item["metric_id"],
            "entity_type": registry.get("entity_type", "ont"),
            "measurement_kind": item["measurement_kind"],
            "unit": item["unit"],
            "sampling_mode": "periodic",
            "aggregation_semantics": item.get(
                "aggregation_semantics",
                {
                    "interval_count": "sum_over_interval",
                    "cumulative_counter": "last_observation",
                    "discrete_state": "last_observation",
                }.get(item["measurement_kind"], "instantaneous"),
            ),
            "expected_cadence_seconds": item["expected_cadence_seconds"],
            "direction": item["direction"],
            "transform": item["transform"],
            "valid_min": item.get("valid_min"),
            "valid_max": item.get("valid_max"),
            "censoring_type": item.get("censoring_type", "none"),
            "reporting_lod": item.get("reporting_lod"),
            "reset_policy": item.get("reset_policy", "not_applicable"),
            "counter_modulus": item.get("counter_modulus"),
            "minimum_scale": item["minimum_scale"],
            "seasonality_candidate": bool(item["seasonality_candidate"]),
            "peer_eligible": bool(item.get("peer_eligible", True)),
        })
    return (
        pd.DataFrame(rows, columns=PACK_SCHEMAS["metric_catalogue"]),
        pd.DataFrame(native, columns=["native_field", "metric_id"]),
    )


def inspect_synthetic_pon(source, registry):
    """Inventory model-visible inputs without opening evaluation files."""

    source = Path(source)
    panel = _find_file(source, "reference_dataset.parquet")
    topology = _find_file(source, "topology.csv")
    catalogue, mapping = metric_catalogue(registry)
    native_columns = duckdb.sql(
        f"SELECT * FROM read_parquet('{str(panel).replace(chr(39), chr(39) * 2)}') LIMIT 0"
    ).df().columns.tolist()
    required = {"timestamp_utc", "ont_id", *mapping["native_field"]}
    missing = sorted(required - set(native_columns))
    leaks = truth_like_columns(native_columns)
    if missing:
        raise ValueError(f"Missing declared source fields: {missing}")
    if leaks:
        raise ValueError(
            "The observable panel still contains truth-like fields. Use the "
            f"redacted reference dataset. Found: {leaks}"
        )
    return {
        "panel": panel,
        "topology": topology,
        "catalogue": catalogue,
        "mapping": mapping,
        "native_columns": native_columns,
        "unmapped_columns": sorted(set(native_columns) - required),
    }


def _available_pairs(panel, mapping):
    expressions = ", ".join(
        f'count("{field}") AS "{field}"' for field in mapping["native_field"]
    )
    panel_sql = str(panel).replace("'", "''")
    with duckdb.connect() as connection:
        wide = connection.execute(f"""
            SELECT CAST(ont_id AS VARCHAR) AS entity_id, {expressions}
            FROM read_parquet('{panel_sql}') GROUP BY entity_id
        """).df()
    pairs = wide.melt(
        id_vars="entity_id", var_name="native_field", value_name="observed_rows"
    )
    pairs = pairs.loc[pairs["observed_rows"].gt(0)].merge(
        mapping, on="native_field", how="left", validate="many_to_one"
    )
    return pairs[["entity_id", "native_field", "metric_id"]]


def _record_batches(cursor, batch_rows):
    """Use the batch API available in the installed DuckDB release."""

    if hasattr(cursor, "to_arrow_reader"):
        return cursor.to_arrow_reader(batch_size=int(batch_rows))
    if hasattr(cursor, "fetch_record_batch"):
        return cursor.fetch_record_batch(rows_per_batch=int(batch_rows))
    if hasattr(cursor, "fetch_arrow_reader"):
        return cursor.fetch_arrow_reader(batch_size=int(batch_rows))
    raise RuntimeError("DuckDB needs an Arrow record-batch reader")


def _telemetry_batches(
    panel,
    pairs,
    mapping,
    *,
    clip_limits,
    batch_rows,
    source_instance,
):
    native_fields = mapping["native_field"].tolist()
    quoted = ", ".join(f'"{name}"' for name in native_fields)
    native_to_metric = dict(mapping.itertuples(index=False, name=None))
    panel_sql = str(panel).replace("'", "''")
    connection = duckdb.connect()
    try:
        cursor = connection.execute(f"""
            SELECT CAST(timestamp_utc AS TIMESTAMPTZ) AS event_ts,
                   CAST(ont_id AS VARCHAR) AS entity_id, {quoted}
            FROM read_parquet('{panel_sql}')
        """)
        for batch in _record_batches(cursor, batch_rows):
            long = batch.to_pandas().melt(
                id_vars=["event_ts", "entity_id"],
                value_vars=native_fields,
                var_name="native_field",
                value_name="value",
            )
            long["metric_id"] = long["native_field"].map(native_to_metric)
            long = long.merge(
                pairs[["entity_id", "metric_id"]],
                on=["entity_id", "metric_id"], how="inner",
            )
            value = pd.to_numeric(long["value"], errors="coerce")
            invalid = value.isna() | ~np.isfinite(value)
            long["value"] = value
            long["quality_code"] = np.where(invalid, "invalid", "measured")
            for metric_id, ceiling in clip_limits.items():
                clipped = long["metric_id"].eq(metric_id) & value.ge(float(ceiling))
                long.loc[clipped & ~invalid, "quality_code"] = "clipped"
            long["episode_id"] = source_instance + "::" + long["entity_id"]
            long["source_system"] = SOURCE_SYSTEM
            long["ingestion_ts"] = pd.NaT
            yield long[PACK_SCHEMAS["telemetry"]]
    finally:
        connection.close()


def _service_windows(source, entities, first_ts):
    path = _find_file(source, "entity_service_windows.csv", required=False)
    if path is None:
        return pd.DataFrame({
            "entity_id": entities,
            "valid_from": first_ts,
            "valid_to": pd.NaT,
        })
    windows = pd.read_csv(path)
    expected = {"entity_id", "install_ts", "decommission_ts"}
    if missing := expected - set(windows.columns):
        raise ValueError(f"Service-window fields missing: {sorted(missing)}")
    windows = windows.loc[windows["entity_id"].astype(str).isin(entities)].copy()
    windows = windows.rename(columns={
        "install_ts": "valid_from", "decommission_ts": "valid_to"
    })[["entity_id", "valid_from", "valid_to"]]
    windows["entity_id"] = windows["entity_id"].astype(str)
    for name in ("valid_from", "valid_to"):
        windows[name] = pd.to_datetime(windows[name], utc=True, errors="coerce")
    if set(entities) != set(windows["entity_id"]):
        raise ValueError("Service windows do not cover every observed ONT")
    return windows


def _topology_tables(source, entities, topology_config, first_ts):
    path = _find_file(source, "topology.csv")
    fields = ["ont_id", "vendor", "device_model", *topology_config["memberships"]]
    topology = pd.read_csv(path, usecols=lambda name: name in fields)
    topology["ont_id"] = topology["ont_id"].astype(str)
    topology = topology.loc[topology["ont_id"].isin(entities)].copy()
    if set(entities) != set(topology["ont_id"]):
        raise ValueError("Topology does not cover every observed ONT")
    required_groups = list(topology_config["memberships"])
    if topology[["ont_id", *required_groups]].isna().any().any():
        raise ValueError("Observed ONTs have incomplete topology memberships")

    windows = _service_windows(source, entities, first_ts)
    enriched = topology.merge(
        windows, left_on="ont_id", right_on="entity_id",
        how="left", validate="one_to_one",
    )
    rows = []
    for native_field, definition in topology_config["memberships"].items():
        part = enriched[["ont_id", native_field, "valid_from", "valid_to"]].copy()
        part = part.rename(columns={"ont_id": "entity_id", native_field: "group_id"})
        part["group_type"] = definition["scope_type"]
        part["hierarchy_level"] = definition["hierarchy_level"]
        part["group_family"] = definition["family"]
        rows.append(part)
    memberships = pd.concat(rows, ignore_index=True)
    memberships = memberships[OPTIONAL_CORE_SCHEMAS["topology_memberships"]]

    entities_table = enriched[[
        "entity_id", "vendor", "device_model", "valid_from", "valid_to"
    ]].rename(columns={"device_model": "model"})
    entities_table.insert(1, "entity_type", "ont")
    entities_table.insert(4, "source_system", SOURCE_SYSTEM)
    return entities_table[PACK_SCHEMAS["entity_registry"]], memberships


def _time_partitions(first_ts, last_ts, cadence_seconds):
    span = last_ts - first_ts + pd.Timedelta(seconds=float(cadence_seconds))
    edges = [first_ts, first_ts + span * 0.50, first_ts + span * 0.75, first_ts + span]
    return pd.DataFrame([
        (name, edges[index], edges[index + 1], "pon_time_v1")
        for index, name in enumerate(("calibration", "development", "holdout"))
    ], columns=SPLIT_SCHEMAS["time_partitions"])


def _infrastructure_partitions(memberships, group_type="olt"):
    """Create a deterministic secondary split with whole groups held out."""

    groups = memberships.loc[
        memberships["group_type"].eq(group_type), ["entity_id", "group_id"]
    ].drop_duplicates()
    if groups["entity_id"].duplicated().any():
        raise ValueError(f"An ONT belongs to more than one {group_type}")
    group_ids = sorted(
        groups["group_id"].astype(str).unique(),
        key=lambda value: hashlib.sha256(value.encode()).hexdigest(),
    )
    if len(group_ids) < 3:
        return pd.DataFrame(columns=SPLIT_SCHEMAS["entity_partitions"])
    calibration_end = max(1, round(len(group_ids) * 0.50))
    development_end = min(
        len(group_ids) - 1,
        max(calibration_end + 1, round(len(group_ids) * 0.75)),
    )
    partition = {
        group_id: (
            "calibration" if index < calibration_end
            else "development" if index < development_end
            else "holdout"
        )
        for index, group_id in enumerate(group_ids)
    }
    result = groups.assign(
        partition=groups["group_id"].astype(str).map(partition),
        split_version=f"pon_{group_type}_hash_v1",
    )
    return result[["entity_id", "partition", "split_version"]]


def _evaluation_tables(source, entities, memberships, source_instance):
    registry_path = _find_file(source, "gt_fault_registry.csv", required=False)
    interval_path = _find_file(source, "fault_entity_intervals.csv", required=False)
    if (registry_path is None) != (interval_path is None):
        raise FileNotFoundError("Evaluation requires both fault files or neither")
    if registry_path is None:
        raise FileNotFoundError(
            "Evaluation was requested but the fault registry and intervals are absent"
        )
    registry = pd.read_csv(registry_path)
    intervals = pd.read_csv(interval_path)
    required_registry = {
        "gt_fault_id", "gt_fault_type", "scope", "target", "onset_ts",
        "first_observable_ts", "impact_ts", "repair_ts", "group_id",
    }
    required_intervals = {
        "fault_id", "entity_id", "active_start_ts", "active_end_ts",
    }
    if missing := required_registry - set(registry.columns):
        raise ValueError(f"Fault-registry fields missing: {sorted(missing)}")
    if missing := required_intervals - set(intervals.columns):
        raise ValueError(f"Fault-interval fields missing: {sorted(missing)}")
    intervals = intervals.loc[intervals["entity_id"].astype(str).isin(entities)].copy()
    fault_ids = set(intervals["fault_id"].dropna().astype(str))
    registry = registry.loc[registry["gt_fault_id"].astype(str).isin(fault_ids)].copy()
    scope_map = {
        "ont": "entity", "l2": "splitter_l2", "l1": "splitter_l1",
        "pon": "pon_port", "olt": "olt",
    }
    domain_type = registry["scope"].astype(str).map(scope_map)
    if domain_type.isna().any():
        raise ValueError("Unknown fault scope in synthetic truth")
    known_groups = set(zip(memberships["group_type"], memberships["group_id"].astype(str)))
    known_entities = set(entities)
    for kind, identifier in zip(domain_type, registry["target"].astype(str)):
        if (kind == "entity" and identifier not in known_entities) or (
            kind != "entity" and (kind, identifier) not in known_groups
        ):
            raise ValueError(f"Fault truth references unknown scope {kind}:{identifier}")

    fault_events = pd.DataFrame({
        "fault_id": registry["gt_fault_id"].astype("string"),
        "fault_type": registry["gt_fault_type"].astype("string"),
        "domain_type": domain_type.astype("string"),
        "domain_id": registry["target"].astype("string"),
        "onset_ts": registry["onset_ts"],
        "observable_ts": registry["first_observable_ts"],
        "impact_ts": registry["impact_ts"],
        "end_ts": registry["repair_ts"],
        "group_id": registry["group_id"].astype("string"),
        "label_source": "synthetic_generator_truth",
        "source_instance_id": source_instance,
    })
    fault_intervals = pd.DataFrame({
        "fault_id": intervals["fault_id"].astype("string"),
        "entity_id": intervals["entity_id"].astype("string"),
        "start_ts": intervals["active_start_ts"],
        "end_ts": intervals["active_end_ts"],
        "label_source": "synthetic_generator_truth",
        "source_instance_id": source_instance,
    })
    tables = {
        "fault_events": fault_events,
        "fault_entity_intervals": fault_intervals,
    }
    ticket_path = _find_file(source, "tickets.csv", required=False)
    if ticket_path is not None:
        tickets = pd.read_csv(ticket_path)
        required = {
            "ticket_id", "ont_id", "reported_ts", "resolved_ts",
            "reported_symptom", "gt_fault_id", "gt_is_nff",
            "gt_misattributed",
        }
        if missing := required - set(tickets.columns):
            raise ValueError(f"Ticket fields missing: {sorted(missing)}")
        tickets = tickets.loc[tickets["ont_id"].astype(str).isin(entities)]
        tables["tickets"] = pd.DataFrame({
            "ticket_id": tickets["ticket_id"].astype("string"),
            "entity_id": tickets["ont_id"].astype("string"),
            "reported_ts": tickets["reported_ts"],
            "resolved_ts": tickets["resolved_ts"],
            "reported_symptom": tickets["reported_symptom"].astype("string"),
            "fault_id": tickets["gt_fault_id"].astype("string"),
            "is_no_fault_found": tickets["gt_is_nff"].astype("boolean"),
            "is_misattributed": tickets["gt_misattributed"].astype("boolean"),
            "label_source": "synthetic_generator_ticket",
            "source_instance_id": source_instance,
        })
    for name, frame in tables.items():
        for column in (item for item in frame if item.endswith("_ts")):
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
        tables[name] = frame[EVAL_SCHEMAS[name]]
    return tables


def _operational_events(source, entities):
    """Translate observable engineering changes; never infer fault alarms."""

    path = _find_file(source, "engineering_events.csv", required=False)
    if path is None:
        return None, None
    native = pd.read_csv(path)
    required = {"entity_id", "event_type", "ts"}
    if missing := required - set(native.columns):
        raise ValueError(f"Engineering-event fields missing: {sorted(missing)}")
    native = native.loc[native["entity_id"].astype(str).isin(entities)].copy()
    if native.empty:
        return None, path
    events = pd.DataFrame({
        "event_ts": pd.to_datetime(native["ts"], utc=True, errors="raise"),
        "entity_id": native["entity_id"].astype("string"),
        "event_code": native["event_type"].astype("string"),
        "event_family": "engineering_change",
        "event_state": "observed",
        "vendor_event_code": native["event_type"].astype("string"),
        "quality_code": "measured",
        "source_system": SOURCE_SYSTEM,
    })
    return events[OPTIONAL_CORE_SCHEMAS["operational_events"]], path


def build_synthetic_pon_pack(
    source,
    destination,
    *,
    metric_registry,
    topology_config,
    source_instance=SOURCE_INSTANCE,
    include_evaluation=True,
    batch_rows=200_000,
):
    """Build a versioned PON pack from the declared source files."""

    source = Path(source)
    inventory = inspect_synthetic_pon(source, metric_registry)
    panel = inventory["panel"]
    catalogue, mapping = inventory["catalogue"], inventory["mapping"]
    clip_limits = {
        item["metric_id"]: item["source_quality"]["clipped_at"]
        for item in metric_registry["metrics"]
        if item.get("source_quality", {}).get("clipped_at") is not None
    }
    pairs = _available_pairs(panel, mapping)
    entities = sorted(pairs["entity_id"].astype(str).unique())
    panel_sql = str(panel).replace("'", "''")
    bounds = duckdb.sql(f"""
        SELECT min(CAST(timestamp_utc AS TIMESTAMPTZ)) AS first_ts,
               max(CAST(timestamp_utc AS TIMESTAMPTZ)) AS last_ts
        FROM read_parquet('{panel_sql}')
    """).df().iloc[0]
    first_ts = pd.to_datetime(bounds["first_ts"], utc=True)
    last_ts = pd.to_datetime(bounds["last_ts"], utc=True)
    entity_table, memberships = _topology_tables(
        source, entities, topology_config, first_ts
    )
    cadence = float(catalogue["expected_cadence_seconds"].dropna().mode().iloc[0])
    splits = {"time_partitions": _time_partitions(first_ts, last_ts, cadence)}
    infrastructure_split = _infrastructure_partitions(memberships)
    if not infrastructure_split.empty:
        splits["entity_partitions"] = infrastructure_split
    evaluation = (
        _evaluation_tables(source, entities, memberships, source_instance)
        if include_evaluation else {}
    )
    operational_events, engineering_path = _operational_events(source, entities)

    files = [
        source_file(panel, source, "model_input"),
        source_file(inventory["topology"], source, "model_input_topology"),
    ]
    service_path = _find_file(source, "entity_service_windows.csv", required=False)
    if service_path:
        files.append(source_file(service_path, source, "model_input_validity"))
    if engineering_path:
        files.append(source_file(engineering_path, source, "model_input_event"))
    if include_evaluation:
        for name in ("gt_fault_registry.csv", "fault_entity_intervals.csv", "tickets.csv"):
            path = _find_file(source, name, required=False)
            if path:
                files.append(source_file(path, source, "evaluation_only"))

    episodes = pd.DataFrame({
        "episode_id": [source_instance + "::" + entity for entity in entities],
        "entity_id": entities,
        "episode_basis": "one_continuous_source_run_per_ont",
    })[PACK_SCHEMAS["observation_episodes"]]
    return save_pack(
        destination,
        sector="telecom_pon",
        pack_version="1.0.0",
        source_info={
            "source_id": SOURCE_SYSTEM,
            "source_instance_id": source_instance,
            "source_root": str(source),
            "files": files,
        },
        telemetry=_telemetry_batches(
            panel, pairs, mapping,
            clip_limits=clip_limits,
            batch_rows=batch_rows, source_instance=source_instance,
        ),
        catalogue=catalogue.loc[
            catalogue["metric_id"].isin(pairs["metric_id"].unique())
        ],
        entities=entity_table,
        episodes=episodes,
        topology=memberships,
        operational_events=operational_events,
        splits=splits,
        evaluation=evaluation,
        notes=[
            "Canonical metric identifiers are independent of native field names.",
            "The adapter reads only declared observable columns.",
            "Static source topology is represented with effective dates from service windows.",
            "Only source-observable engineering events are model-visible; fault truth and tickets are isolated.",
            "No dying-gasp or LOS events are invented when the source does not supply them.",
        ],
    )
