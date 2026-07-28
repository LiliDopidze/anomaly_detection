"""Shared, sector-neutral mechanics for the Week 1 research notebooks.

Sector notebooks create a small PACK-CORE/PACK-EVAL bundle.  This module
validates that interface and converts PACK-CORE observations to the canonical
long SPEC-CORE contract.  It deliberately contains no telecom or oil-well
field names, file names, entity types, or label meanings.
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


CORE_VERSION = "0.5.0"
EVAL_VERSION = "0.5.0"
PACK_INTERFACE_VERSION = "0.1.0"

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
    "condition_states": [
        "entity_id",
        "start_ts",
        "end_ts",
        "condition_code",
    ],
}

PACK_METRIC_SCHEMA = [
    "metric_id",
    "entity_type",
    "measurement_kind",
    "unit",
    "clip_at",
]
PACK_OBSERVATION_KEYS = ["event_ts", "entity_id"]

MEASUREMENT_KINDS = {
    "gauge",
    "bounded_fraction",
    "interval_count",
    "cumulative_counter",
    "discrete_state",
}
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
    """Write stable, human-readable JSON without silently replacing a file."""

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
    """Hash table values without depending on Parquet file metadata."""

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


def empty_table(table_name):
    """Create an empty canonical evaluation table with the correct columns."""

    return pd.DataFrame(columns=EVAL_SCHEMAS[table_name])


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


def source_record(path, source_root, evaluation_only=False):
    """Describe one source file for a reproducible lineage manifest."""

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
        "evaluation_only": bool(evaluation_only),
    }


def _partitioned_hash(directory, sort_by):
    digest = hashlib.sha256()
    rows = 0
    parts = sorted(Path(directory).glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No Parquet parts found in {directory}")
    for part in parts:
        frame = pd.read_parquet(part)
        digest.update(canonical_frame_hash(frame, sort_by).encode("ascii"))
        rows += len(frame)
    return digest.hexdigest(), rows


def _table_hash(path, schema):
    frame = pd.read_parquet(path)
    if list(frame.columns) != list(schema):
        raise ValueError(f"Unexpected schema in {path}: {list(frame.columns)}")
    return canonical_frame_hash(frame, schema), len(frame)


def _truth_like_columns(columns):
    reserved = {
        "class",
        "state",
        "label",
        "fault_id",
        "condition_code",
        "ticket_id",
    }
    return sorted(
        name
        for name in map(str, columns)
        if name.lower() in reserved or name.lower().startswith("gt_")
    )


def _validate_pack_tables(pack_root, evaluation_tables=()):
    """Validate the physical pack interface without interpreting the sector."""

    pack_root = Path(pack_root)
    core = pack_root / "PACK-CORE"
    observations = core / "observations"
    parts = sorted(observations.glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No observation parts in {observations}")

    catalogue = pd.read_parquet(core / "metric_catalogue.parquet")
    registry = pd.read_parquet(core / "entity_registry.parquet")
    relations = pd.read_parquet(core / "entity_relations.parquet")

    if list(catalogue.columns) != PACK_METRIC_SCHEMA:
        raise ValueError(f"Pack metric catalogue must use {PACK_METRIC_SCHEMA}")
    if list(registry.columns) != CORE_SCHEMAS["entity_registry"]:
        raise ValueError("Pack entity_registry schema is invalid")
    if list(relations.columns) != CORE_SCHEMAS["entity_relations"]:
        raise ValueError("Pack entity_relations schema is invalid")
    if catalogue["metric_id"].duplicated().any():
        raise ValueError("Pack metric_id values must be unique")
    if registry["entity_id"].duplicated().any():
        raise ValueError("Pack entity_id values must be unique")

    unknown_kinds = set(catalogue["measurement_kind"]) - MEASUREMENT_KINDS
    if unknown_kinds:
        raise ValueError(f"Unknown measurement kinds: {sorted(unknown_kinds)}")

    metric_ids = catalogue["metric_id"].astype(str).tolist()
    expected_observation_columns = [*PACK_OBSERVATION_KEYS, *metric_ids]
    for part in parts:
        columns = pq.ParquetFile(part).schema_arrow.names
        if columns != expected_observation_columns:
            raise ValueError(
                f"Unexpected observation schema in {part.name}: {columns}"
            )
        leaks = _truth_like_columns(columns)
        if leaks:
            raise ValueError(f"Evaluation fields found in PACK-CORE: {leaks}")

    registry_ids = set(registry["entity_id"].astype(str))
    relation_ids = set(relations["parent_entity_id"].dropna().astype(str))
    relation_ids |= set(relations["child_entity_id"].dropna().astype(str))
    missing_relation_entities = relation_ids - registry_ids
    if missing_relation_entities:
        raise ValueError(
            "Relations reference entities missing from the registry: "
            f"{sorted(missing_relation_entities)[:5]}"
        )

    for table_name in evaluation_tables:
        if table_name not in EVAL_SCHEMAS:
            raise ValueError(f"Unknown evaluation table: {table_name}")
        path = pack_root / "PACK-EVAL" / f"{table_name}.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        if list(frame.columns) != EVAL_SCHEMAS[table_name]:
            raise ValueError(f"Unexpected {table_name} schema")

    return {
        "observation_parts": len(parts),
        "metrics": len(catalogue),
        "entities": len(registry),
        "relations": len(relations),
    }


def pack_core_hashes(pack_root):
    """Return logical hashes used by the sector-pack isolation test."""

    core = Path(pack_root) / "PACK-CORE"
    observations_hash, _ = _partitioned_hash(
        core / "observations", PACK_OBSERVATION_KEYS
    )
    hashes = {"observations": observations_hash}
    schemas = {
        "metric_catalogue": PACK_METRIC_SCHEMA,
        "entity_registry": CORE_SCHEMAS["entity_registry"],
        "entity_relations": CORE_SCHEMAS["entity_relations"],
    }
    for table_name, schema in schemas.items():
        hashes[table_name], _ = _table_hash(
            core / f"{table_name}.parquet", schema
        )
    return hashes


def finalise_pack(
    pack_root,
    *,
    sector,
    pack_version,
    expected_cadence_seconds,
    source_manifest,
    evaluation_tables=(),
    notes=(),
):
    """Validate and seal a sector pack after its notebook writes the tables."""

    pack_root = Path(pack_root)
    cadence = float(expected_cadence_seconds)
    if cadence <= 0:
        raise ValueError("expected_cadence_seconds must be positive")

    validation = _validate_pack_tables(pack_root, evaluation_tables)
    core = pack_root / "PACK-CORE"
    observation_hash, observation_rows = _partitioned_hash(
        core / "observations", PACK_OBSERVATION_KEYS
    )
    core_hashes = {"observations": observation_hash}
    core_rows = {"observations": observation_rows}

    schemas = {
        "metric_catalogue": PACK_METRIC_SCHEMA,
        "entity_registry": CORE_SCHEMAS["entity_registry"],
        "entity_relations": CORE_SCHEMAS["entity_relations"],
    }
    for table_name, schema in schemas.items():
        table_hash, rows = _table_hash(core / f"{table_name}.parquet", schema)
        core_hashes[table_name] = table_hash
        core_rows[table_name] = rows

    eval_hashes, eval_rows = {}, {}
    for table_name in evaluation_tables:
        path = pack_root / "PACK-EVAL" / f"{table_name}.parquet"
        table_hash, rows = _table_hash(path, EVAL_SCHEMAS[table_name])
        eval_hashes[table_name] = table_hash
        eval_rows[table_name] = rows

    manifest = {
        "pack_interface_version": PACK_INTERFACE_VERSION,
        "pack_version": str(pack_version),
        "sector": str(sector),
        "expected_cadence_seconds": cadence,
        "metric_ids": pd.read_parquet(
            core / "metric_catalogue.parquet", columns=["metric_id"]
        )["metric_id"].astype(str).tolist(),
        "evaluation_tables": list(evaluation_tables),
        "core_row_counts": core_rows,
        "core_content_hashes": core_hashes,
        "evaluation_row_counts": eval_rows,
        "evaluation_content_hashes": eval_hashes,
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
            "Unsupported pack interface: "
            f"{manifest['pack_interface_version']}"
        )
    _validate_pack_tables(pack_root, manifest["evaluation_tables"])
    actual_hashes = pack_core_hashes(pack_root)
    if actual_hashes != manifest["core_content_hashes"]:
        raise ValueError("PACK-CORE content no longer matches its manifest")
    return manifest


def _quality_codes(long_frame, catalogue):
    long_frame["value"] = pd.to_numeric(long_frame["value"], errors="coerce")
    long_frame["quality_code"] = "measured"
    long_frame.loc[long_frame["value"].isna(), "quality_code"] = "invalid"

    clip_limits = catalogue.set_index("metric_id")["clip_at"]
    limit = pd.to_numeric(
        long_frame["metric_id"].map(clip_limits), errors="coerce"
    )
    clipped = (
        long_frame["value"].notna()
        & limit.notna()
        & long_frame["value"].ge(limit)
    )
    long_frame.loc[clipped, "quality_code"] = "clipped"
    return long_frame


def _collection_gaps(presence, registry, cadence_seconds):
    """Find absent expected samples inside observed service time."""

    presence = presence.copy()
    presence["event_ts"] = pd.to_datetime(
        presence["event_ts"], utc=True, errors="raise"
    )
    presence["entity_id"] = presence["entity_id"].astype(str)
    presence = presence.drop_duplicates().sort_values(
        ["entity_id", "event_ts"], kind="stable"
    )
    if presence.empty:
        return pd.DataFrame(columns=CORE_SCHEMAS["collection_gaps"])

    cadence = pd.Timedelta(seconds=float(cadence_seconds))
    data_start = presence["event_ts"].min()
    data_end = presence["event_ts"].max() + cadence
    registry = registry.copy()
    registry["entity_id"] = registry["entity_id"].astype(str)
    registry = registry.set_index("entity_id")

    gaps = []
    for entity_id, group in presence.groupby("entity_id", sort=False):
        if entity_id not in registry.index:
            raise ValueError(f"Observation entity missing from registry: {entity_id}")
        entity = registry.loc[entity_id]
        valid_from = pd.to_datetime(entity["valid_from"], utc=True, errors="coerce")
        valid_to = pd.to_datetime(entity["valid_to"], utc=True, errors="coerce")
        window_start = data_start if pd.isna(valid_from) else max(data_start, valid_from)
        window_end = data_end if pd.isna(valid_to) else min(data_end, valid_to)
        if window_start >= window_end:
            continue

        timestamps = (
            group.loc[
                group["event_ts"].ge(window_start)
                & group["event_ts"].lt(window_end),
                "event_ts",
            ]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
        if not timestamps:
            gaps.append((entity_id, window_start, window_end))
            continue

        previous = window_start - cadence
        for timestamp in timestamps:
            gap_start = previous + cadence
            if gap_start < timestamp:
                gaps.append((entity_id, gap_start, timestamp))
            previous = timestamp
        if previous + cadence < window_end:
            gaps.append((entity_id, previous + cadence, window_end))

    return pd.DataFrame(gaps, columns=CORE_SCHEMAS["collection_gaps"])


def materialise_canonical(pack_root, run_root, include_evaluation=True):
    """Convert any valid sector pack into immutable SPEC-CORE/SPEC-EVAL."""

    pack_root, run_root = Path(pack_root), Path(run_root)
    pack_manifest = validate_pack(pack_root)
    catalogue = pd.read_parquet(
        pack_root / "PACK-CORE" / "metric_catalogue.parquet"
    )
    registry = pd.read_parquet(
        pack_root / "PACK-CORE" / "entity_registry.parquet"
    )
    relations = pd.read_parquet(
        pack_root / "PACK-CORE" / "entity_relations.parquet"
    )
    metric_ids = catalogue["metric_id"].astype(str).tolist()

    with immutable_directory(run_root) as temporary:
        core = temporary / "SPEC-CORE"
        telemetry_directory = core / "telemetry"
        telemetry_directory.mkdir(parents=True)

        telemetry_digest = hashlib.sha256()
        telemetry_rows = 0
        quality_counts = {}
        presence_parts = []

        observation_parts = sorted(
            (pack_root / "PACK-CORE" / "observations").glob("part-*.parquet")
        )
        for part_number, part in enumerate(observation_parts):
            wide = pd.read_parquet(part)
            wide["event_ts"] = pd.to_datetime(
                wide["event_ts"], utc=True, errors="raise"
            )
            wide["entity_id"] = wide["entity_id"].astype(str)
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
                .sort_values(
                    ["event_ts", "entity_id", "metric_id"], kind="stable"
                )
                .reset_index(drop=True)
            )
            long.to_parquet(
                telemetry_directory / f"part-{part_number:05d}.parquet",
                index=False,
                compression="zstd",
            )
            telemetry_digest.update(
                canonical_frame_hash(
                    long, ["event_ts", "entity_id", "metric_id"]
                ).encode("ascii")
            )
            telemetry_rows += len(long)
            for code, count in long["quality_code"].value_counts().items():
                quality_counts[str(code)] = (
                    quality_counts.get(str(code), 0) + int(count)
                )

        canonical_catalogue = catalogue[
            CORE_SCHEMAS["metric_catalogue"]
        ].copy()
        presence = pd.concat(presence_parts, ignore_index=True)
        gaps = _collection_gaps(
            presence,
            registry,
            pack_manifest["expected_cadence_seconds"],
        )
        sidecars = {
            "metric_catalogue": canonical_catalogue,
            "entity_registry": registry[CORE_SCHEMAS["entity_registry"]],
            "entity_relations": relations[CORE_SCHEMAS["entity_relations"]],
            "collection_gaps": gaps,
        }
        for table_name, frame in sidecars.items():
            frame.to_parquet(core / f"{table_name}.parquet", index=False)

        core_hashes = {"telemetry": telemetry_digest.hexdigest()}
        core_rows = {"telemetry": telemetry_rows}
        for table_name, frame in sidecars.items():
            core_hashes[table_name] = canonical_frame_hash(
                frame, CORE_SCHEMAS[table_name]
            )
            core_rows[table_name] = len(frame)

        core_manifest = {
            "contract_version": CORE_VERSION,
            "sector": pack_manifest["sector"],
            "source_pack_version": pack_manifest["pack_version"],
            "expected_cadence_seconds": pack_manifest[
                "expected_cadence_seconds"
            ],
            "tables": CORE_SCHEMAS,
            "row_counts": core_rows,
            "canonical_content_hashes": core_hashes,
            "quality_counts": quality_counts,
        }
        write_json(core / "manifest.json", core_manifest)

        eval_manifest = None
        if include_evaluation and pack_manifest["evaluation_tables"]:
            evaluation = temporary / "SPEC-EVAL"
            evaluation.mkdir()
            eval_hashes, eval_rows = {}, {}
            for table_name in pack_manifest["evaluation_tables"]:
                frame = pd.read_parquet(
                    pack_root / "PACK-EVAL" / f"{table_name}.parquet"
                )
                frame = frame[EVAL_SCHEMAS[table_name]]
                frame.to_parquet(
                    evaluation / f"{table_name}.parquet", index=False
                )
                eval_hashes[table_name] = canonical_frame_hash(
                    frame, EVAL_SCHEMAS[table_name]
                )
                eval_rows[table_name] = len(frame)
            eval_manifest = {
                "contract_version": EVAL_VERSION,
                "tables": {
                    name: EVAL_SCHEMAS[name]
                    for name in pack_manifest["evaluation_tables"]
                },
                "row_counts": eval_rows,
                "canonical_content_hashes": eval_hashes,
            }
            write_json(evaluation / "manifest.json", eval_manifest)

        lineage = {
            "pack_root": str(pack_root),
            "pack_manifest_sha256": sha256_file(
                pack_root / "pack_manifest.json"
            ),
            "source_manifest_sha256": sha256_file(
                pack_root / "source_manifest.json"
            ),
            "evaluation_mounted": bool(
                include_evaluation and pack_manifest["evaluation_tables"]
            ),
            "adapter_has_sector_branch": False,
        }
        write_json(temporary / "lineage.json", lineage)
        write_json(
            temporary / "workflow_report.json",
            {
                "run_root": str(run_root),
                "core_manifest": core_manifest,
                "evaluation_manifest": eval_manifest,
                "lineage": lineage,
            },
        )

    return read_json(run_root / "workflow_report.json")


def audit_core(core_root):
    """Validate a completed SPEC-CORE and report its basic contents."""

    core_root = Path(core_root)
    manifest = read_json(core_root / "manifest.json")
    if manifest["contract_version"] != CORE_VERSION:
        raise ValueError("Unsupported SPEC-CORE version")

    telemetry_rows = 0
    quality_counts = {}
    for part in sorted((core_root / "telemetry").glob("part-*.parquet")):
        frame = pd.read_parquet(part)
        if list(frame.columns) != CORE_SCHEMAS["telemetry"]:
            raise ValueError(f"Unexpected telemetry schema in {part.name}")
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
    """Read the sealed logical hashes without opening SPEC-EVAL."""

    return read_json(Path(core_root) / "manifest.json")[
        "canonical_content_hashes"
    ]


def runtime_probe(core_root):
    """A deterministic SPEC-CORE-only score used as deployment evidence."""

    totals = []
    for part in sorted((Path(core_root) / "telemetry").glob("part-*.parquet")):
        frame = pd.read_parquet(
            part, columns=["entity_id", "metric_id", "value"]
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
    return canonical_frame_hash(
        combined, sort_by=["entity_id", "metric_id"]
    )


__all__ = [
    "CORE_SCHEMAS",
    "CORE_TABLES",
    "CORE_VERSION",
    "EVAL_SCHEMAS",
    "EVAL_TABLES",
    "EVAL_VERSION",
    "PACK_INTERFACE_VERSION",
    "PACK_METRIC_SCHEMA",
    "PACK_OBSERVATION_KEYS",
    "audit_core",
    "canonical_frame_hash",
    "core_content_hashes",
    "empty_table",
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
