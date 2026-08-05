"""Shared mechanics for the sector-pack research notebooks.

The module is intentionally flat.  Sector notebooks own native field names
and label meanings; this file owns only the versioned interfaces, validation,
canonical materialisation, hashing, and isolation helpers.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


CORE_VERSION = "0.6.0"
EVAL_VERSION = "0.6.0"
PACK_INTERFACE_VERSION = "0.3.0"

CORE_SCHEMAS = {
    "telemetry": [
        "event_ts",
        "entity_id",
        "metric_id",
        "value",
        "quality_code",
    ],
    "metric_catalogue": [
        "metric_id",
        "entity_type",
        "measurement_kind",
        "unit",
        "sampling_mode",
        "expected_cadence_seconds",
        "lower_bound",
        "upper_bound",
        "censoring_type",
        "clip_at",
    ],
    "entity_registry": [
        "entity_id",
        "entity_type",
        "observed_from",
        "observed_to",
        "validity_basis",
    ],
    "collection_gaps": [
        "entity_id",
        "metric_id",
        "gap_start",
        "gap_end",
        "expected_cadence_seconds",
        "coverage_basis",
    ],
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
        "label_source",
        "source_instance_id",
    ],
    "fault_entity_intervals": [
        "fault_id",
        "entity_id",
        "start_ts",
        "end_ts",
        "label_source",
        "source_instance_id",
    ],
    "condition_states": [
        "entity_id",
        "start_ts",
        "end_ts",
        "condition_code",
        "label_source",
        "source_instance_id",
    ],
    "tickets": [
        "ticket_id",
        "entity_id",
        "reported_ts",
        "resolved_ts",
        "fault_id",
        "reported_fault_type",
        "is_no_fault_found",
        "is_misattributed",
        "label_source",
    ],
}

SPLIT_SCHEMAS = {
    "entity_partitions": [
        "entity_id",
        "partition",
        "split_version",
    ],
}

PACK_METRIC_SCHEMA = CORE_SCHEMAS["metric_catalogue"]
PACK_ENTITY_SCHEMA = ["entity_id", "entity_type"]
PACK_OBSERVATION_KEYS = ["event_ts", "entity_id"]

MEASUREMENT_KINDS = {
    "gauge",
    "bounded_fraction",
    "interval_count",
    "cumulative_counter",
    "discrete_state",
}
SAMPLING_MODES = {
    "periodic",
    "recording",
    "irregular",
    "event_driven",
    "unknown",
}
QUALITY_CODES = {"measured", "invalid", "clipped"}

CORE_TABLES = tuple(CORE_SCHEMAS)
EVAL_TABLES = tuple(EVAL_SCHEMAS)


def sha256_file(path, chunk_size=1024 * 1024):
    """Return a streaming SHA-256 hash for one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload, overwrite=False):
    """Write stable JSON without silently replacing a completed artifact."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical_frame_hash(frame, sort_by=()):
    """Hash logical table content independently of Parquet metadata."""

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
    digest = hashlib.sha256(schema.encode("utf-8"))
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


@contextmanager
def immutable_directory(destination):
    """Build a directory atomically and refuse to overwrite completed runs."""

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{uuid.uuid4().hex[:8]}"
    )
    try:
        temporary.mkdir()
        yield temporary
        temporary.rename(destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def source_record(path, source_root, role="model_input"):
    """Describe one native file in a reproducible lineage manifest."""

    path = Path(path)
    source_root = Path(source_root)
    try:
        relative_path = str(path.relative_to(source_root))
    except ValueError:
        relative_path = str(path)
    return {
        "relative_path": relative_path,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "role": str(role),
    }


def _table_hash(path, schema):
    frame = pd.read_parquet(path)
    if list(frame.columns) != list(schema):
        raise ValueError(f"Unexpected schema in {path}: {list(frame.columns)}")
    return canonical_frame_hash(frame, schema), len(frame)


def _partitioned_hash(directory, sort_by):
    parts = sorted(Path(directory).glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No Parquet parts found in {directory}")
    digest = hashlib.sha256()
    rows = 0
    for part in parts:
        frame = pd.read_parquet(part)
        digest.update(canonical_frame_hash(frame, sort_by).encode("ascii"))
        rows += len(frame)
    return digest.hexdigest(), rows


def _truth_like_columns(columns):
    tokens = ("fault", "label", "anomaly", "root_cause", "ticket")
    reserved = {"class", "state", "condition_code"}
    leaks = []
    for column in map(str, columns):
        name = column.lower()
        if name in reserved or name.startswith("gt_") or any(
            token in name for token in tokens
        ):
            leaks.append(column)
    return sorted(leaks)


def _validate_catalogue(catalogue):
    if list(catalogue.columns) != PACK_METRIC_SCHEMA:
        raise ValueError(f"Metric catalogue must use {PACK_METRIC_SCHEMA}")
    if catalogue["metric_id"].duplicated().any():
        raise ValueError("metric_id values must be unique")

    unknown_kinds = set(catalogue["measurement_kind"]) - MEASUREMENT_KINDS
    if unknown_kinds:
        raise ValueError(f"Unknown measurement kinds: {sorted(unknown_kinds)}")
    unknown_sampling = set(catalogue["sampling_mode"]) - SAMPLING_MODES
    if unknown_sampling:
        raise ValueError(f"Unknown sampling modes: {sorted(unknown_sampling)}")

    cadence = pd.to_numeric(
        catalogue["expected_cadence_seconds"], errors="coerce"
    )
    periodic = catalogue["sampling_mode"].eq("periodic")
    if cadence.loc[periodic].isna().any() or cadence.loc[periodic].le(0).any():
        raise ValueError("Periodic metrics require a positive expected cadence")


def _validate_pack_tables(
    pack_root,
    *,
    evaluation_tables=(),
    split_tables=(),
):
    """Validate the physical pack interface without interpreting a sector."""

    pack_root = Path(pack_root)
    core = pack_root / "PACK-CORE"
    if (pack_root / "PACK-CONTEXT").exists():
        raise ValueError("Pack v0.3 is telemetry-only; PACK-CONTEXT is not supported")
    parts = sorted((core / "observations").glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No observation parts in {core / 'observations'}")

    catalogue = pd.read_parquet(core / "metric_catalogue.parquet")
    registry = pd.read_parquet(core / "entity_registry.parquet")
    _validate_catalogue(catalogue)
    if list(registry.columns) != PACK_ENTITY_SCHEMA:
        raise ValueError(f"Entity registry must use {PACK_ENTITY_SCHEMA}")
    if registry["entity_id"].duplicated().any():
        raise ValueError("entity_id values must be unique")

    metric_ids = catalogue["metric_id"].astype(str).tolist()
    expected_columns = [*PACK_OBSERVATION_KEYS, *metric_ids]
    registry_ids = set(registry["entity_id"].astype(str))
    observed_ids = set()
    for part in parts:
        columns = pq.ParquetFile(part).schema_arrow.names
        if columns != expected_columns:
            raise ValueError(f"Unexpected observation schema in {part.name}: {columns}")
        leaks = _truth_like_columns(columns)
        if leaks:
            raise ValueError(f"Evaluation fields found in PACK-CORE: {leaks}")
        observed_ids.update(
            pd.read_parquet(part, columns=["entity_id"])["entity_id"]
            .dropna()
            .astype(str)
            .unique()
        )
    missing_entities = observed_ids - registry_ids
    if missing_entities:
        raise ValueError(
            "Observed entities missing from the registry: "
            f"{sorted(missing_entities)[:5]}"
        )

    table_groups = [
        ("PACK-EVAL", evaluation_tables, EVAL_SCHEMAS),
        ("SPLITS", split_tables, SPLIT_SCHEMAS),
    ]
    for folder, table_names, schemas in table_groups:
        for table_name in table_names:
            if table_name not in schemas:
                raise ValueError(f"Unknown {folder} table: {table_name}")
            path = pack_root / folder / f"{table_name}.parquet"
            if not path.is_file():
                raise FileNotFoundError(path)
            frame = pd.read_parquet(path)
            if list(frame.columns) != schemas[table_name]:
                raise ValueError(f"Unexpected schema in {path}")

    return {
        "observation_parts": len(parts),
        "metrics": len(catalogue),
        "observed_entities": len(observed_ids),
        "registered_entities": len(registry),
    }


def pack_core_hashes(pack_root):
    """Return logical hashes used by the pack truth-isolation test."""

    core = Path(pack_root) / "PACK-CORE"
    observations_hash, _ = _partitioned_hash(
        core / "observations", PACK_OBSERVATION_KEYS
    )
    catalogue_hash, _ = _table_hash(
        core / "metric_catalogue.parquet", PACK_METRIC_SCHEMA
    )
    registry_hash, _ = _table_hash(
        core / "entity_registry.parquet", PACK_ENTITY_SCHEMA
    )
    return {
        "observations": observations_hash,
        "metric_catalogue": catalogue_hash,
        "entity_registry": registry_hash,
    }


def finalise_pack(
    pack_root,
    *,
    sector,
    pack_version,
    source_manifest,
    evaluation_tables=(),
    split_tables=(),
    notes=(),
):
    """Validate and seal a sector pack after its notebook writes the tables."""

    pack_root = Path(pack_root)
    validation = _validate_pack_tables(
        pack_root,
        evaluation_tables=evaluation_tables,
        split_tables=split_tables,
    )
    core = pack_root / "PACK-CORE"
    observation_hash, observation_rows = _partitioned_hash(
        core / "observations", PACK_OBSERVATION_KEYS
    )
    core_hashes = {"observations": observation_hash}
    core_rows = {"observations": observation_rows}
    for table_name, schema in {
        "metric_catalogue": PACK_METRIC_SCHEMA,
        "entity_registry": PACK_ENTITY_SCHEMA,
    }.items():
        table_hash, rows = _table_hash(core / f"{table_name}.parquet", schema)
        core_hashes[table_name] = table_hash
        core_rows[table_name] = rows

    auxiliary = {}
    for folder, table_names, schemas in [
        ("PACK-EVAL", evaluation_tables, EVAL_SCHEMAS),
        ("SPLITS", split_tables, SPLIT_SCHEMAS),
    ]:
        auxiliary[folder] = {"row_counts": {}, "content_hashes": {}}
        for table_name in table_names:
            table_hash, rows = _table_hash(
                pack_root / folder / f"{table_name}.parquet",
                schemas[table_name],
            )
            auxiliary[folder]["row_counts"][table_name] = rows
            auxiliary[folder]["content_hashes"][table_name] = table_hash

    manifest = {
        "pack_interface_version": PACK_INTERFACE_VERSION,
        "pack_version": str(pack_version),
        "sector": str(sector),
        "evaluation_tables": list(evaluation_tables),
        "split_tables": list(split_tables),
        "metric_ids": pd.read_parquet(
            core / "metric_catalogue.parquet", columns=["metric_id"]
        )["metric_id"].astype(str).tolist(),
        "core_row_counts": core_rows,
        "core_content_hashes": core_hashes,
        "auxiliary": auxiliary,
        "notes": list(notes),
        "validation": validation,
    }
    write_json(pack_root / "source_manifest.json", source_manifest)
    write_json(pack_root / "pack_manifest.json", manifest)
    return manifest


def validate_pack(pack_root):
    """Validate a completed pack and return its manifest."""

    pack_root = Path(pack_root)
    manifest = read_json(pack_root / "pack_manifest.json")
    if manifest["pack_interface_version"] != PACK_INTERFACE_VERSION:
        raise ValueError(
            f"Unsupported pack interface: {manifest['pack_interface_version']}"
        )
    _validate_pack_tables(
        pack_root,
        evaluation_tables=manifest["evaluation_tables"],
        split_tables=manifest["split_tables"],
    )
    if pack_core_hashes(pack_root) != manifest["core_content_hashes"]:
        raise ValueError("PACK-CORE content no longer matches its manifest")
    return manifest


def _quality_codes(long_frame, catalogue):
    values = pd.to_numeric(long_frame["value"], errors="coerce")
    metadata = catalogue.set_index("metric_id")
    lower = pd.to_numeric(
        long_frame["metric_id"].map(metadata["lower_bound"]), errors="coerce"
    )
    upper = pd.to_numeric(
        long_frame["metric_id"].map(metadata["upper_bound"]), errors="coerce"
    )
    clip_at = pd.to_numeric(
        long_frame["metric_id"].map(metadata["clip_at"]), errors="coerce"
    )

    invalid = values.isna()
    invalid |= lower.notna() & values.lt(lower)
    invalid |= upper.notna() & values.gt(upper)
    clipped = values.notna() & clip_at.notna() & values.ge(clip_at)

    long_frame = long_frame.copy()
    long_frame["value"] = values
    long_frame["quality_code"] = "measured"
    long_frame.loc[invalid, "quality_code"] = "invalid"
    long_frame.loc[clipped & ~invalid, "quality_code"] = "clipped"
    return long_frame


def _entity_registry(presence, pack_registry):
    bounds = (
        presence.groupby("entity_id", as_index=False)["event_ts"]
        .agg(observed_from="min", observed_to="max")
    )
    registry = pack_registry.copy()
    registry["entity_id"] = registry["entity_id"].astype(str)
    result = bounds.merge(registry, on="entity_id", how="left", validate="one_to_one")
    if result["entity_type"].isna().any():
        raise ValueError("Canonical entity type is missing")
    result["validity_basis"] = "derived_from_observations_as_of"
    return result[CORE_SCHEMAS["entity_registry"]]


def _collection_gaps(presence, registry, catalogue):
    """Find missing periodic observations inside observation-derived bounds."""

    periodic = catalogue.loc[
        catalogue["sampling_mode"].eq("periodic")
        & catalogue["expected_cadence_seconds"].notna()
    ]
    if presence.empty or periodic.empty:
        return pd.DataFrame(columns=CORE_SCHEMAS["collection_gaps"])

    by_entity = {
        entity_id: group["event_ts"].drop_duplicates().sort_values().tolist()
        for entity_id, group in presence.groupby("entity_id", sort=False)
    }
    bounds = registry.set_index("entity_id")
    rows = []
    for metric in periodic.itertuples(index=False):
        cadence_seconds = float(metric.expected_cadence_seconds)
        cadence = pd.Timedelta(seconds=cadence_seconds)
        for entity_id, timestamps in by_entity.items():
            if not timestamps:
                continue
            start = bounds.loc[entity_id, "observed_from"]
            end = bounds.loc[entity_id, "observed_to"] + cadence
            previous = start - cadence
            for timestamp in timestamps:
                gap_start = previous + cadence
                if gap_start < timestamp:
                    rows.append((
                        entity_id,
                        metric.metric_id,
                        gap_start,
                        timestamp,
                        cadence_seconds,
                        "observed_bounds",
                    ))
                previous = timestamp
            if previous + cadence < end:
                rows.append((
                    entity_id,
                    metric.metric_id,
                    previous + cadence,
                    end,
                    cadence_seconds,
                    "observed_bounds",
                ))
    return pd.DataFrame(rows, columns=CORE_SCHEMAS["collection_gaps"])


def _copy_tables(source_root, destination_root, table_names, schemas):
    if not table_names:
        return {}, {}
    destination_root.mkdir()
    hashes, rows = {}, {}
    for table_name in table_names:
        frame = pd.read_parquet(source_root / f"{table_name}.parquet")
        frame = frame[schemas[table_name]]
        frame.to_parquet(destination_root / f"{table_name}.parquet", index=False)
        hashes[table_name] = canonical_frame_hash(frame, schemas[table_name])
        rows[table_name] = len(frame)
    return hashes, rows


def materialise_canonical(
    pack_root,
    run_root,
    *,
    include_evaluation=True,
    as_of_ts=None,
):
    """Convert any valid sector pack into immutable canonical directories."""

    pack_root, run_root = Path(pack_root), Path(run_root)
    pack_manifest = validate_pack(pack_root)
    catalogue = pd.read_parquet(
        pack_root / "PACK-CORE" / "metric_catalogue.parquet"
    )
    pack_registry = pd.read_parquet(
        pack_root / "PACK-CORE" / "entity_registry.parquet"
    )
    metric_ids = catalogue["metric_id"].astype(str).tolist()
    as_of = pd.to_datetime(as_of_ts, utc=True) if as_of_ts is not None else None

    with immutable_directory(run_root) as temporary:
        core = temporary / "SPEC-CORE"
        telemetry_directory = core / "telemetry"
        telemetry_directory.mkdir(parents=True)

        telemetry_digest = hashlib.sha256()
        telemetry_rows = 0
        quality_counts = {}
        presence_parts = []
        output_part = 0

        observation_parts = sorted(
            (pack_root / "PACK-CORE" / "observations").glob("part-*.parquet")
        )
        for part in observation_parts:
            wide = pd.read_parquet(part)
            wide["event_ts"] = pd.to_datetime(wide["event_ts"], utc=True)
            wide["entity_id"] = wide["entity_id"].astype(str)
            if as_of is not None:
                wide = wide.loc[wide["event_ts"].le(as_of)]
            if wide.empty:
                continue
            presence_parts.append(wide[PACK_OBSERVATION_KEYS].drop_duplicates())

            long = wide.melt(
                id_vars=PACK_OBSERVATION_KEYS,
                value_vars=metric_ids,
                var_name="metric_id",
                value_name="value",
            )
            long = _quality_codes(long, catalogue)
            long = (
                long[CORE_SCHEMAS["telemetry"]]
                .sort_values(["event_ts", "entity_id", "metric_id"], kind="stable")
                .reset_index(drop=True)
            )
            long.to_parquet(
                telemetry_directory / f"part-{output_part:05d}.parquet",
                index=False,
                compression="zstd",
            )
            output_part += 1
            telemetry_digest.update(
                canonical_frame_hash(
                    long, ["event_ts", "entity_id", "metric_id"]
                ).encode("ascii")
            )
            telemetry_rows += len(long)
            for code, count in long["quality_code"].value_counts().items():
                quality_counts[str(code)] = quality_counts.get(str(code), 0) + int(count)

        if not presence_parts:
            raise ValueError("No observations are available at the requested as_of_ts")
        presence = pd.concat(presence_parts, ignore_index=True).drop_duplicates()
        registry = _entity_registry(presence, pack_registry)
        gaps = _collection_gaps(presence, registry, catalogue)

        sidecars = {
            "metric_catalogue": catalogue[CORE_SCHEMAS["metric_catalogue"]],
            "entity_registry": registry,
            "collection_gaps": gaps,
        }
        core_hashes = {"telemetry": telemetry_digest.hexdigest()}
        core_rows = {"telemetry": telemetry_rows}
        for table_name, frame in sidecars.items():
            frame.to_parquet(core / f"{table_name}.parquet", index=False)
            core_hashes[table_name] = canonical_frame_hash(
                frame, CORE_SCHEMAS[table_name]
            )
            core_rows[table_name] = len(frame)

        has_periodic_coverage = (
            catalogue["sampling_mode"].eq("periodic")
            & catalogue["expected_cadence_seconds"].notna()
        ).any()
        core_manifest = {
            "contract_version": CORE_VERSION,
            "sector": pack_manifest["sector"],
            "source_pack_version": pack_manifest["pack_version"],
            "as_of_ts": as_of,
            "tables": CORE_SCHEMAS,
            "row_counts": core_rows,
            "canonical_content_hashes": core_hashes,
            "quality_counts": quality_counts,
            "coverage_basis": (
                "observed_bounds"
                if has_periodic_coverage
                else "not_applicable_recordings"
            ),
        }
        write_json(core / "manifest.json", core_manifest)

        split_manifest = None
        if pack_manifest["split_tables"]:
            hashes, rows = _copy_tables(
                pack_root / "SPLITS",
                temporary / "SPLITS",
                pack_manifest["split_tables"],
                SPLIT_SCHEMAS,
            )
            split_manifest = {
                "tables": pack_manifest["split_tables"],
                "row_counts": rows,
                "canonical_content_hashes": hashes,
            }
            write_json(temporary / "SPLITS" / "manifest.json", split_manifest)

        eval_manifest = None
        if include_evaluation and pack_manifest["evaluation_tables"]:
            hashes, rows = _copy_tables(
                pack_root / "PACK-EVAL",
                temporary / "SPEC-EVAL",
                pack_manifest["evaluation_tables"],
                EVAL_SCHEMAS,
            )
            eval_manifest = {
                "contract_version": EVAL_VERSION,
                "tables": {
                    name: EVAL_SCHEMAS[name]
                    for name in pack_manifest["evaluation_tables"]
                },
                "row_counts": rows,
                "canonical_content_hashes": hashes,
            }
            write_json(temporary / "SPEC-EVAL" / "manifest.json", eval_manifest)

        lineage = {
            "pack_root": str(pack_root),
            "pack_manifest_sha256": sha256_file(pack_root / "pack_manifest.json"),
            "source_manifest_sha256": sha256_file(pack_root / "source_manifest.json"),
            "evaluation_mounted": bool(
                include_evaluation and pack_manifest["evaluation_tables"]
            ),
            "adapter_has_sector_branch": False,
            "as_of_ts": as_of,
        }
        write_json(temporary / "lineage.json", lineage)
        write_json(
            temporary / "workflow_report.json",
            {
                "run_root": str(run_root),
                "core_manifest": core_manifest,
                "split_manifest": split_manifest,
                "evaluation_manifest": eval_manifest,
                "lineage": lineage,
            },
        )

    return read_json(run_root / "workflow_report.json")


def audit_core(core_root):
    """Validate a completed SPEC-CORE and report its contents."""

    core_root = Path(core_root)
    manifest = read_json(core_root / "manifest.json")
    if manifest["contract_version"] != CORE_VERSION:
        raise ValueError("Unsupported SPEC-CORE version")

    telemetry_rows = 0
    quality_counts = {}
    parts = sorted((core_root / "telemetry").glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError("SPEC-CORE contains no telemetry parts")
    for part in parts:
        frame = pd.read_parquet(part)
        if list(frame.columns) != CORE_SCHEMAS["telemetry"]:
            raise ValueError(f"Unexpected telemetry schema in {part.name}")
        unknown_quality = set(frame["quality_code"].dropna()) - QUALITY_CODES
        if unknown_quality:
            raise ValueError(f"Unknown quality codes: {sorted(unknown_quality)}")
        telemetry_rows += len(frame)
        for code, count in frame["quality_code"].value_counts().items():
            quality_counts[str(code)] = quality_counts.get(str(code), 0) + int(count)

    for table_name in CORE_TABLES[1:]:
        frame = pd.read_parquet(core_root / f"{table_name}.parquet")
        if list(frame.columns) != CORE_SCHEMAS[table_name]:
            raise ValueError(f"Unexpected {table_name} schema")
    if telemetry_rows != manifest["row_counts"]["telemetry"]:
        raise ValueError("Telemetry row count does not match the manifest")

    return {
        "telemetry_rows": telemetry_rows,
        "quality_counts": quality_counts,
        **{
            name: int(manifest["row_counts"][name])
            for name in CORE_TABLES[1:]
        },
    }


def core_content_hashes(core_root):
    """Read sealed logical hashes without opening SPEC-EVAL."""

    return read_json(Path(core_root) / "manifest.json")[
        "canonical_content_hashes"
    ]


def runtime_probe(core_root):
    """Return a deterministic SPEC-CORE-only score for isolation tests."""

    totals = []
    for part in sorted((Path(core_root) / "telemetry").glob("part-*.parquet")):
        frame = pd.read_parquet(part, columns=["entity_id", "metric_id", "value"])
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
    return canonical_frame_hash(combined, ["entity_id", "metric_id"])


__all__ = [
    "CORE_SCHEMAS",
    "CORE_TABLES",
    "CORE_VERSION",
    "EVAL_SCHEMAS",
    "EVAL_TABLES",
    "EVAL_VERSION",
    "PACK_ENTITY_SCHEMA",
    "PACK_INTERFACE_VERSION",
    "PACK_METRIC_SCHEMA",
    "PACK_OBSERVATION_KEYS",
    "SPLIT_SCHEMAS",
    "audit_core",
    "canonical_frame_hash",
    "core_content_hashes",
    "finalise_pack",
    "immutable_directory",
    "materialise_canonical",
    "pack_core_hashes",
    "read_json",
    "runtime_probe",
    "sha256_file",
    "source_record",
    "validate_pack",
    "write_json",
]
