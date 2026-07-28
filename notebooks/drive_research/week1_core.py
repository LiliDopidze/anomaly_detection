"""Small, deterministic mechanics for the Week 1 telecom notebook."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


CORE_VERSION = "0.4.0"
EVAL_VERSION = "0.4.0"
SOURCE_ID = "telemetry-synth-4.1.0"

CORE_SCHEMAS = {
    "telemetry": ["event_ts", "entity_id", "metric_id", "value", "quality_code"],
    "metric_catalogue": ["metric_id", "entity_type", "measurement_kind", "unit"],
    "entity_registry": ["entity_id", "entity_type", "valid_from", "valid_to"],
    "entity_relations": ["parent_entity_id", "child_entity_id", "relation_type"],
    "collection_gaps": ["entity_id", "gap_start", "gap_end"],
}
EVAL_SCHEMAS = {
    "fault_events": [
        "fault_id",
        "fault_type",
        "domain_id",
        "onset_ts",
        "observable_ts",
        "impact_ts",
        "end_ts",
        "group_id",
    ],
    "fault_entity_intervals": ["fault_id", "entity_id", "start_ts", "end_ts"],
    "tickets": ["ticket_id", "entity_id", "reported_ts", "resolved_ts", "fault_id"],
}
METRIC_MAP_COLUMNS = [
    "native_field",
    "metric_id",
    "entity_type",
    "measurement_kind",
    "unit",
    "clip_at",
]
MEASUREMENT_KINDS = {
    "gauge",
    "bounded_fraction",
    "interval_count",
    "cumulative_counter",
    "discrete_state",
}
TOPOLOGY_LEVELS = [
    ("olt_id", "olt"),
    ("pon_port", "pon_port"),
    ("splitter_l1", "splitter_l1"),
    ("splitter_l2", "splitter_l2"),
    ("ont_id", "ont"),
]

CORE_TABLES = tuple(CORE_SCHEMAS)
EVAL_TABLES = tuple(EVAL_SCHEMAS)


@dataclass(frozen=True)
class Selection:
    """Optional development slice. Empty values mean the full source."""

    entity_ids: tuple[str, ...] = ()
    start: str | None = None
    end: str | None = None
    batch_rows: int = 100_000


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def canonical_frame_hash(frame, sort_by=()):
    """Hash logical table content without depending on Parquet metadata."""

    frame = frame.copy()
    order = [column for column in sort_by if column in frame]
    if order:
        frame = frame.sort_values(order, kind="stable")
    frame = frame.reset_index(drop=True)

    for column in frame:
        if pd.api.types.is_datetime64_any_dtype(frame[column]):
            frame[column] = frame[column].astype("string")

    schema = "|".join(f"{name}:{dtype}" for name, dtype in frame.dtypes.items())
    row_hashes = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    digest = hashlib.sha256(schema.encode())
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def _source_roots(source, evaluation):
    source = Path(source)
    roots = [source, source / "Data"]
    if evaluation:
        roots += [source / "evaluation", source / "SPEC-EVAL", source / "SPEC_EVAL"]
    return tuple(dict.fromkeys(roots))


def source_file(source, filename, evaluation=False, required=True):
    matches = [
        root / filename
        for root in _source_roots(source, evaluation)
        if (root / filename).is_file()
    ]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous source file {filename}: {matches}")
    if matches:
        return matches[0]
    if required:
        raise FileNotFoundError(f"Missing source file: {filename}")
    return None


def discover_telecom(source):
    required_core = [
        "reference_dataset.parquet",
        "topology.csv",
        "entity_service_windows.csv",
    ]
    required_eval = ["gt_fault_registry.csv", "fault_entity_intervals.csv"]
    missing_core = [
        name
        for name in required_core
        if source_file(source, name, required=False) is None
    ]
    missing_eval = [
        name
        for name in required_eval
        if source_file(source, name, evaluation=True, required=False) is None
    ]
    return {
        "core_ready": not missing_core,
        "evaluation_ready": not missing_eval,
        "missing_core": missing_core,
        "missing_evaluation": missing_eval,
        "tickets_available": source_file(
            source, "tickets.csv", evaluation=True, required=False
        )
        is not None,
    }


def validate_metric_map(metric_map):
    if list(metric_map.columns) != METRIC_MAP_COLUMNS:
        raise ValueError(f"Metric map columns must be {METRIC_MAP_COLUMNS}")
    if metric_map["native_field"].duplicated().any():
        raise ValueError("native_field must be unique")
    if metric_map["metric_id"].duplicated().any():
        raise ValueError("metric_id must be unique")
    unknown = set(metric_map["measurement_kind"]) - MEASUREMENT_KINDS
    if unknown:
        raise ValueError(f"Unknown measurement kinds: {sorted(unknown)}")


def _filter_panel(frame, selection):
    if selection.entity_ids:
        frame = frame.loc[frame["ont_id"].astype(str).isin(selection.entity_ids)]
    timestamps = pd.to_datetime(frame["timestamp_utc"], utc=True)
    if selection.start:
        frame = frame.loc[timestamps.ge(pd.to_datetime(selection.start, utc=True))]
        timestamps = pd.to_datetime(frame["timestamp_utc"], utc=True)
    if selection.end:
        frame = frame.loc[timestamps.lt(pd.to_datetime(selection.end, utc=True))]
    return frame.reset_index(drop=True)


def _infer_cadence(presence):
    entity_id = str(presence["ont_id"].iloc[0])
    timestamps = (
        pd.to_datetime(
            presence.loc[presence["ont_id"].astype(str).eq(entity_id), "timestamp_utc"],
            utc=True,
        )
        .drop_duplicates()
        .sort_values()
    )
    positive = timestamps.diff().dropna()
    positive = positive.loc[positive.gt(pd.Timedelta(0))]
    if positive.empty:
        raise ValueError("Could not infer a positive telemetry cadence")
    return positive.mode().iloc[0]


def translate_telemetry_batch(panel, metric_map):
    """Convert one native wide batch to canonical long telemetry."""

    available = [field for field in metric_map["native_field"] if field in panel]
    if not available:
        raise ValueError("No mapped metric fields found in the native panel")

    mapping = metric_map.set_index("native_field")
    long = panel[["timestamp_utc", "ont_id", *available]].melt(
        id_vars=["timestamp_utc", "ont_id"],
        value_vars=available,
        var_name="native_field",
        value_name="value",
    )
    long["event_ts"] = pd.to_datetime(long.pop("timestamp_utc"), utc=True)
    long["entity_id"] = long.pop("ont_id").astype(str)
    long["metric_id"] = long["native_field"].map(mapping["metric_id"])

    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    clip_at = long["native_field"].map(mapping["clip_at"])
    long["quality_code"] = "measured"
    long.loc[long["value"].isna(), "quality_code"] = "invalid"
    clipped = long["value"].notna() & clip_at.notna() & long["value"].ge(clip_at)
    long.loc[clipped, "quality_code"] = "clipped"

    return (
        long[CORE_SCHEMAS["telemetry"]]
        .sort_values(["event_ts", "entity_id", "metric_id"], kind="stable")
        .reset_index(drop=True)
    )


def build_entity_registry(topology, service_windows):
    windows = service_windows.copy()
    windows["entity_id"] = windows["entity_id"].astype(str)
    windows = windows.set_index("entity_id")

    rows = []
    for field, entity_type in TOPOLOGY_LEVELS:
        if field not in topology:
            continue
        for entity_id in sorted(topology[field].dropna().astype(str).unique()):
            window = (
                windows.loc[entity_id]
                if entity_type == "ont" and entity_id in windows.index
                else None
            )
            rows.append(
                {
                    "entity_id": entity_id,
                    "entity_type": entity_type,
                    "valid_from": pd.NaT if window is None else window["install_ts"],
                    "valid_to": (
                        pd.NaT if window is None else window["decommission_ts"]
                    ),
                }
            )
    return (
        pd.DataFrame(rows, columns=CORE_SCHEMAS["entity_registry"])
        .drop_duplicates("entity_id")
        .sort_values("entity_id")
        .reset_index(drop=True)
    )


def build_entity_relations(topology, relation_mappings):
    rows = []
    for parent_field, child_field, relation_type in relation_mappings:
        pairs = topology[[parent_field, child_field]].dropna().drop_duplicates()
        rows += [
            {
                "parent_entity_id": str(parent),
                "child_entity_id": str(child),
                "relation_type": relation_type,
            }
            for parent, child in pairs.itertuples(index=False)
        ]
    return (
        pd.DataFrame(rows, columns=CORE_SCHEMAS["entity_relations"])
        .drop_duplicates()
        .sort_values(CORE_SCHEMAS["entity_relations"])
        .reset_index(drop=True)
    )


def build_collection_gaps(presence, service_windows, cadence, selection):
    """Find absent timestamp runs only while each ONT is in service."""

    observed = presence.copy()
    observed["ont_id"] = observed["ont_id"].astype(str)
    observed["timestamp_utc"] = pd.to_datetime(observed["timestamp_utc"], utc=True)
    observed = observed.drop_duplicates().sort_values(
        ["ont_id", "timestamp_utc"], kind="stable"
    )

    windows = service_windows.copy()
    windows["entity_id"] = windows["entity_id"].astype(str)
    windows = windows.set_index("entity_id")
    data_start = observed["timestamp_utc"].min()
    data_end = observed["timestamp_utc"].max() + cadence
    if selection.start:
        data_start = max(data_start, pd.to_datetime(selection.start, utc=True))
    if selection.end:
        data_end = min(data_end, pd.to_datetime(selection.end, utc=True))

    gaps = []
    for entity_id, group in observed.groupby("ont_id", sort=False):
        service = windows.loc[entity_id] if entity_id in windows.index else None
        valid_from = (
            data_start
            if service is None or pd.isna(service["install_ts"])
            else max(data_start, pd.to_datetime(service["install_ts"], utc=True))
        )
        valid_to = (
            data_end
            if service is None or pd.isna(service["decommission_ts"])
            else min(data_end, pd.to_datetime(service["decommission_ts"], utc=True))
        )
        timestamps = group.loc[
            group["timestamp_utc"].between(valid_from, valid_to, inclusive="left"),
            "timestamp_utc",
        ].tolist()
        if not timestamps:
            gaps.append((entity_id, valid_from, valid_to))
            continue

        previous = valid_from - cadence
        for timestamp in timestamps:
            gap_start = previous + cadence
            if gap_start < timestamp:
                gaps.append((entity_id, gap_start, timestamp))
            previous = timestamp
        if previous + cadence < valid_to:
            gaps.append((entity_id, previous + cadence, valid_to))

    return pd.DataFrame(gaps, columns=CORE_SCHEMAS["collection_gaps"])


def _write_table(frame, path, sort_by=()):
    frame.to_parquet(path, index=False)
    return len(frame), canonical_frame_hash(frame, sort_by)


def _materialise_core(source, destination, metric_map, relation_mappings, selection):
    destination.mkdir(parents=True)
    telemetry_directory = destination / "telemetry"
    telemetry_directory.mkdir()

    panel = pq.ParquetFile(source_file(source, "reference_dataset.parquet"))
    panel_columns = set(panel.schema_arrow.names)
    if not {"timestamp_utc", "ont_id"} <= panel_columns:
        raise ValueError("Panel must contain timestamp_utc and ont_id")
    truth_columns = sorted(
        name for name in panel_columns if str(name).startswith("gt_")
    )
    if truth_columns:
        raise ValueError(f"Observable panel contains truth: {truth_columns}")

    metric_map = metric_map.loc[metric_map["native_field"].isin(panel_columns)].copy()
    if metric_map.empty:
        raise ValueError("None of the mapped metrics exist in the panel")

    presence_parts = []
    for batch in panel.iter_batches(
        batch_size=selection.batch_rows,
        columns=["timestamp_utc", "ont_id"],
    ):
        selected = _filter_panel(batch.to_pandas(), selection)
        if len(selected):
            presence_parts.append(selected)
    if not presence_parts:
        raise ValueError("The selected telemetry slice is empty")
    presence = pd.concat(presence_parts, ignore_index=True)
    cadence = _infer_cadence(presence)
    selected_entities = set(presence["ont_id"].astype(str))

    topology = pd.read_csv(source_file(source, "topology.csv"))
    topology = topology.loc[
        topology["ont_id"].astype(str).isin(selected_entities)
    ].copy()
    windows = pd.read_csv(
        source_file(source, "entity_service_windows.csv"),
        parse_dates=["install_ts", "decommission_ts"],
    )
    windows = windows.loc[
        windows["entity_id"].astype(str).isin(selected_entities)
    ].copy()

    tables = {
        "metric_catalogue": metric_map[CORE_SCHEMAS["metric_catalogue"]].reset_index(
            drop=True
        ),
        "entity_registry": build_entity_registry(topology, windows),
        "entity_relations": build_entity_relations(topology, relation_mappings),
        "collection_gaps": build_collection_gaps(presence, windows, cadence, selection),
    }
    row_counts, hashes = {}, {}
    for name, frame in tables.items():
        row_counts[name], hashes[name] = _write_table(
            frame, destination / f"{name}.parquet", frame.columns
        )

    telemetry_hash = hashlib.sha256()
    telemetry_rows = part_number = 0
    columns = ["timestamp_utc", "ont_id", *metric_map["native_field"]]
    for batch in panel.iter_batches(batch_size=selection.batch_rows, columns=columns):
        selected = _filter_panel(batch.to_pandas(), selection)
        if selected.empty:
            continue
        canonical = translate_telemetry_batch(selected, metric_map)
        canonical.to_parquet(
            telemetry_directory / f"part-{part_number:05d}.parquet",
            index=False,
            compression="zstd",
        )
        telemetry_hash.update(
            canonical_frame_hash(
                canonical, ["event_ts", "entity_id", "metric_id"]
            ).encode()
        )
        telemetry_rows += len(canonical)
        part_number += 1

    row_counts["telemetry"] = telemetry_rows
    hashes["telemetry"] = telemetry_hash.hexdigest()
    manifest = {
        "contract_version": CORE_VERSION,
        "source_id": SOURCE_ID,
        "cadence_seconds": cadence.total_seconds(),
        "selection": asdict(selection),
        "tables": CORE_SCHEMAS,
        "row_counts": row_counts,
        "canonical_content_hashes": hashes,
    }
    write_json(destination / "manifest.json", manifest)
    return manifest


def _column(frame, name, default=pd.NA):
    if name in frame:
        return frame[name]
    return pd.Series(default, index=frame.index)


def _materialise_evaluation(source, destination, selected_entities):
    registry = pd.read_csv(
        source_file(source, "gt_fault_registry.csv", evaluation=True)
    )
    intervals = pd.read_csv(
        source_file(source, "fault_entity_intervals.csv", evaluation=True)
    )
    intervals = intervals.loc[
        intervals["entity_id"].astype(str).isin(selected_entities)
    ].copy()
    fault_ids = set(intervals["fault_id"].astype(str))
    registry = registry.loc[registry["gt_fault_id"].astype(str).isin(fault_ids)].copy()

    fault_events = pd.DataFrame(
        {
            "fault_id": registry["gt_fault_id"].astype(str),
            "fault_type": registry["gt_fault_type"].astype(str),
            "domain_id": _column(registry, "target"),
            "onset_ts": _column(registry, "onset_ts"),
            "observable_ts": _column(registry, "first_observable_ts"),
            "impact_ts": _column(registry, "impact_ts"),
            "end_ts": _column(registry, "repair_ts"),
            "group_id": _column(registry, "group_id"),
        }
    )
    fault_intervals = pd.DataFrame(
        {
            "fault_id": intervals["fault_id"].astype(str),
            "entity_id": intervals["entity_id"].astype(str),
            "start_ts": _column(intervals, "active_start_ts"),
            "end_ts": _column(intervals, "active_end_ts"),
        }
    )

    ticket_path = source_file(source, "tickets.csv", evaluation=True, required=False)
    if ticket_path:
        native_tickets = pd.read_csv(ticket_path)
        native_tickets = native_tickets.loc[
            native_tickets["ont_id"].astype(str).isin(selected_entities)
        ]
        tickets = pd.DataFrame(
            {
                "ticket_id": native_tickets["ticket_id"].astype(str),
                "entity_id": native_tickets["ont_id"].astype(str),
                "reported_ts": _column(native_tickets, "reported_ts"),
                "resolved_ts": _column(native_tickets, "resolved_ts"),
                "fault_id": _column(native_tickets, "gt_fault_id"),
            }
        )
    else:
        tickets = pd.DataFrame(columns=EVAL_SCHEMAS["tickets"])

    tables = {
        "fault_events": fault_events,
        "fault_entity_intervals": fault_intervals,
        "tickets": tickets,
    }
    for frame in tables.values():
        for column in frame:
            if column.endswith("_ts"):
                frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")

    destination.mkdir(parents=True)
    row_counts, hashes = {}, {}
    for name, frame in tables.items():
        frame = frame[EVAL_SCHEMAS[name]]
        row_counts[name], hashes[name] = _write_table(
            frame, destination / f"{name}.parquet", frame.columns
        )
    manifest = {
        "contract_version": EVAL_VERSION,
        "tables": EVAL_SCHEMAS,
        "row_counts": row_counts,
        "canonical_content_hashes": hashes,
    }
    write_json(destination / "manifest.json", manifest)
    return manifest


def _source_manifest(source, include_evaluation):
    files = [
        ("reference_dataset.parquet", False, True),
        ("topology.csv", False, True),
        ("entity_service_windows.csv", False, True),
    ]
    if include_evaluation:
        files += [
            ("gt_fault_registry.csv", True, True),
            ("fault_entity_intervals.csv", True, True),
            ("tickets.csv", True, False),
        ]

    records = []
    for name, evaluation, required in files:
        path = source_file(source, name, evaluation=evaluation, required=required)
        if path:
            records.append(
                {
                    "relative_path": str(path.relative_to(source)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "evaluation_only": evaluation,
                }
            )
    return {"source_id": SOURCE_ID, "source_root": str(source), "files": records}


def materialise_telecom(
    source,
    run_root,
    metric_map,
    relation_mappings,
    selection=None,
    include_evaluation=True,
):
    """Create immutable SPEC-CORE and optional, separate SPEC-EVAL."""

    source, run_root = Path(source), Path(run_root)
    selection = selection or Selection()
    validate_metric_map(metric_map)
    inventory = discover_telecom(source)
    if not inventory["core_ready"]:
        raise FileNotFoundError(inventory)
    if include_evaluation and not inventory["evaluation_ready"]:
        raise FileNotFoundError(inventory)
    if run_root.exists():
        raise FileExistsError(f"Refusing to overwrite {run_root}")

    temporary = run_root.with_name(f".{run_root.name}.tmp-{uuid.uuid4().hex[:8]}")
    try:
        core_manifest = _materialise_core(
            source,
            temporary / "SPEC-CORE",
            metric_map,
            relation_mappings,
            selection,
        )
        entities = pd.read_parquet(temporary / "SPEC-CORE" / "entity_registry.parquet")
        selected_entities = set(
            entities.loc[entities["entity_type"].eq("ont"), "entity_id"].astype(str)
        )
        evaluation_manifest = (
            _materialise_evaluation(source, temporary / "SPEC-EVAL", selected_entities)
            if include_evaluation
            else None
        )
        write_json(
            temporary / "source_manifest.json",
            _source_manifest(source, include_evaluation),
        )
        report = {
            "contract_version": CORE_VERSION,
            "source": str(source),
            "run_root": str(run_root),
            "core_manifest": core_manifest,
            "evaluation_manifest": evaluation_manifest,
        }
        write_json(temporary / "workflow_report.json", report)
        run_root.parent.mkdir(parents=True, exist_ok=True)
        temporary.rename(run_root)
        return report
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def audit_core(core):
    """Validate the minimal schemas and count quality codes."""

    core = Path(core)
    manifest = json.loads((core / "manifest.json").read_text())
    quality_counts, telemetry_rows = {}, 0

    for part in sorted((core / "telemetry").glob("part-*.parquet")):
        frame = pd.read_parquet(part)
        if list(frame.columns) != CORE_SCHEMAS["telemetry"]:
            raise AssertionError(f"Unexpected telemetry schema in {part.name}")
        telemetry_rows += len(frame)
        for code, count in frame["quality_code"].value_counts().items():
            quality_counts[code] = quality_counts.get(code, 0) + int(count)

    for name in CORE_TABLES[1:]:
        frame = pd.read_parquet(core / f"{name}.parquet")
        if list(frame.columns) != CORE_SCHEMAS[name]:
            raise AssertionError(f"Unexpected {name} schema")
    if telemetry_rows != manifest["row_counts"]["telemetry"]:
        raise AssertionError("Telemetry row count does not match manifest")

    return {
        "telemetry_rows": telemetry_rows,
        "quality_counts": quality_counts,
        **{name: manifest["row_counts"][name] for name in CORE_TABLES[1:]},
    }
