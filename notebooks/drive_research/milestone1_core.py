"""Shared mechanics for the Milestone 1 research notebooks.

Sector notebooks own native field names and label meanings.  This module owns
the versioned interfaces, validation, canonical materialisation, one content
fingerprint, and the isolation helpers.  It contains no sector logic.

Layering
--------
    native source  ->  PACK  (sector notebook translates)
    PACK           ->  SPEC-CORE + SPEC-EVAL + SPLITS  (common adapter)

A detector reads SPEC-CORE only and must run with SPEC-EVAL absent.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

CORE_VERSION = "0.10.1"
EVAL_VERSION = "0.8.0"
PACK_INTERFACE_VERSION = "0.7.1"

GAP_TOLERANCE_FACTOR = 1.5
CANONICAL_BATCH_ROWS = 250_000

# --------------------------------------------------------------------------
# Schemas.  Pack and canonical share table names; canonical adds derived
# columns and one derived table, so the pack schema is the canonical schema
# minus what the adapter computes.
# --------------------------------------------------------------------------

CORE_SCHEMAS = {
    "telemetry": [
        "event_ts", "entity_id", "episode_id", "metric_id", "value", "quality_code",
    ],
    "metric_catalogue": [
        "metric_id", "entity_type", "measurement_kind", "unit",
        "sampling_mode", "expected_cadence_seconds",
    ],
    "entity_registry": [
        "entity_id", "entity_type", "observed_from", "observed_to", "validity_basis",
    ],
    "observation_episodes": [
        "episode_id", "entity_id", "observed_from", "observed_to", "episode_basis",
    ],
    "collection_gaps": [
        "entity_id", "episode_id", "metric_id",
        "gap_start", "gap_end", "expected_cadence_seconds", "coverage_basis",
    ],
}

DERIVED_COLUMNS = {
    "entity_registry": ("observed_from", "observed_to", "validity_basis"),
    "observation_episodes": ("observed_from", "observed_to"),
}

PACK_TABLES = ("telemetry", "metric_catalogue", "entity_registry", "observation_episodes")
PACK_SCHEMAS = {
    name: [c for c in CORE_SCHEMAS[name] if c not in DERIVED_COLUMNS.get(name, ())]
    for name in PACK_TABLES
}
TELEMETRY_KEYS = ["event_ts", "entity_id", "episode_id", "metric_id"]

EVAL_SCHEMAS = {
    "fault_events": [
        "fault_id", "fault_type", "domain_id",
        "onset_ts", "observable_ts", "impact_ts", "end_ts",
        "group_id", "label_source", "source_instance_id",
    ],
    "fault_entity_intervals": [
        "fault_id", "entity_id", "start_ts", "end_ts",
        "label_source", "source_instance_id",
    ],
    "condition_states": [
        "entity_id", "start_ts", "end_ts", "condition_code",
        "label_source", "source_instance_id",
    ],
}

SPLIT_SCHEMAS = {
    "entity_partitions": ["entity_id", "partition", "split_version"],
    "time_partitions": ["partition", "start_ts", "end_ts", "split_version"],
    "entity_groups": ["entity_id", "group_type", "group_id", "split_version"],
}

MEASUREMENT_KINDS = {
    "gauge", "bounded_fraction", "interval_count", "cumulative_counter", "discrete_state",
}
SAMPLING_MODES = {"periodic", "recording", "irregular", "event_driven", "unknown"}
QUALITY_CODES = {"measured", "invalid", "clipped"}

# Anchored truth detection.  Free substring matching rejected legitimate
# measurements such as ``ground_fault_current`` and ``fault_passage_indicator``.
TRUTH_NAMES = {"class", "state", "label", "target", "condition_code", "anomaly", "fault"}
TRUTH_PREFIXES = ("gt_", "truth_", "anomaly_")
TRUTH_SUFFIXES = ("_label", "_labels", "_anomaly", "_ground_truth")


def truth_like_columns(names):
    """Return the names that look like evaluation truth, not measurement."""

    found = []
    for name in map(str, names):
        lowered = name.lower()
        if (
            lowered in TRUTH_NAMES
            or lowered.startswith(TRUTH_PREFIXES)
            or lowered.endswith(TRUTH_SUFFIXES)
        ):
            found.append(name)
    return sorted(found)


# --------------------------------------------------------------------------
# Small file and hashing helpers
# --------------------------------------------------------------------------


def _duckdb():
    """Import DuckDB only where a canonical scan needs it."""

    try:
        import duckdb
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError("Install duckdb: pip install duckdb") from error
    return duckdb


def _record_batches(result, batch_size):
    """Return Arrow batches across supported DuckDB versions."""

    try:
        return result.to_arrow_reader(batch_size)
    except AttributeError:
        return result.fetch_record_batch(batch_size)


def file_sha256(path, chunk_size=1 << 20):
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
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def table_digest(frame, sort_by=()):
    """Hash logical table content, independent of Parquet file metadata."""

    frame = frame.copy()
    order = [column for column in sort_by if column in frame]
    if order:
        frame = frame.sort_values(order, kind="stable")
    frame = frame.reset_index(drop=True)
    for column in frame:
        if pd.api.types.is_datetime64_any_dtype(frame[column]):
            frame[column] = frame[column].astype("string")
    schema = "|".join(f"{name}:{dtype}" for name, dtype in frame.dtypes.items())
    digest = hashlib.sha256(schema.encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def _parts(directory):
    found = sorted(Path(directory).glob("part-*.parquet"))
    if not found:
        raise FileNotFoundError(f"No Parquet parts in {directory}")
    return found


def _directory_digest(directory, sort_by):
    digest = hashlib.sha256()
    rows = 0
    for part in _parts(directory):
        frame = pd.read_parquet(part)
        digest.update(table_digest(frame, sort_by).encode("ascii"))
        rows += len(frame)
    return digest.hexdigest(), rows


def _combine(table_hashes):
    return hashlib.sha256(
        json.dumps(table_hashes, sort_keys=True).encode("utf-8")
    ).hexdigest()


@contextmanager
def new_output_directory(destination):
    """Build a directory atomically and never overwrite a completed run."""

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex[:8]}")
    try:
        temporary.mkdir()
        yield temporary
        temporary.rename(destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def source_file(path, source_root, role="model_input"):
    """Describe one native file for the pack manifest."""

    path, source_root = Path(path), Path(source_root)
    try:
        relative_path = str(path.relative_to(source_root))
    except ValueError:
        relative_path = str(path)
    return {
        "relative_path": relative_path,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "role": str(role),
    }


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def validate_catalogue(catalogue):
    if list(catalogue.columns) != PACK_SCHEMAS["metric_catalogue"]:
        raise ValueError(f"Catalogue must use {PACK_SCHEMAS['metric_catalogue']}")
    if catalogue["metric_id"].duplicated().any():
        raise ValueError("metric_id values must be unique")
    leaks = truth_like_columns(catalogue["metric_id"])
    if leaks:
        raise ValueError(f"Evaluation-like metric IDs are not allowed: {leaks}")

    unknown = set(catalogue["measurement_kind"]) - MEASUREMENT_KINDS
    if unknown:
        raise ValueError(f"Unknown measurement kinds: {sorted(unknown)}")
    unknown = set(catalogue["sampling_mode"]) - SAMPLING_MODES
    if unknown:
        raise ValueError(f"Unknown sampling modes: {sorted(unknown)}")

    cadence = pd.to_numeric(catalogue["expected_cadence_seconds"], errors="coerce")
    declared = cadence.notna()
    if declared.any() and cadence.loc[declared].le(0).any():
        raise ValueError("A declared cadence must be positive")
    if cadence.loc[catalogue["sampling_mode"].eq("periodic")].isna().any():
        raise ValueError("Periodic metrics require an expected cadence")


def _validate_telemetry(frame, where):
    if list(frame.columns) != CORE_SCHEMAS["telemetry"]:
        raise ValueError(f"Unexpected telemetry schema in {where}: {list(frame.columns)}")
    if frame[TELEMETRY_KEYS].isna().any().any():
        raise ValueError(f"Null telemetry key in {where}")
    if frame.duplicated(TELEMETRY_KEYS).any():
        raise ValueError(f"Duplicate telemetry key in {where}")
    unknown = set(frame["quality_code"].dropna()) - QUALITY_CODES
    if unknown or frame["quality_code"].isna().any():
        raise ValueError(f"Invalid quality codes in {where}: {sorted(unknown)}")
    numeric = pd.to_numeric(frame["value"], errors="coerce")
    unusable = numeric.isna() | ~np.isfinite(numeric)
    if frame.loc[unusable, "quality_code"].ne("invalid").any():
        raise ValueError(
            f"Missing, non-numeric or non-finite values must be invalid in {where}"
        )


def _validate_references(catalogue, entities, episodes, observed):
    """Every declared object is observed and every observed object is declared."""

    if list(entities.columns) != PACK_SCHEMAS["entity_registry"]:
        raise ValueError(f"Entity registry must use {PACK_SCHEMAS['entity_registry']}")
    if list(episodes.columns) != PACK_SCHEMAS["observation_episodes"]:
        raise ValueError(f"Episodes must use {PACK_SCHEMAS['observation_episodes']}")
    if entities["entity_id"].duplicated().any():
        raise ValueError("entity_id values must be unique")
    if episodes["episode_id"].duplicated().any():
        raise ValueError("episode_id values must be unique")

    declared_entities = set(entities["entity_id"].astype(str))
    declared_episodes = set(episodes["episode_id"].astype(str))
    declared_metrics = set(catalogue["metric_id"].astype(str))
    episode_owner = dict(
        zip(episodes["episode_id"].astype(str), episodes["entity_id"].astype(str))
    )

    if set(episodes["entity_id"].astype(str)) - declared_entities:
        raise ValueError("An episode refers to an unregistered entity")

    checks = [
        ("entities", observed["entities"], declared_entities),
        ("episodes", observed["episodes"], declared_episodes),
        ("metrics", observed["metrics"], declared_metrics),
    ]
    for name, seen, declared in checks:
        if seen - declared:
            raise ValueError(f"Observed {name} missing from the pack: {sorted(seen - declared)[:5]}")
        if declared - seen:
            raise ValueError(f"Declared {name} have no observations: {sorted(declared - seen)[:5]}")

    wrong = {
        episode for episode, entity in observed["episode_owner"].items()
        if episode_owner.get(episode) != entity
    }
    if wrong:
        raise ValueError(f"Episodes attached to the wrong entity: {sorted(wrong)[:5]}")


# --------------------------------------------------------------------------
# Writing a pack
# --------------------------------------------------------------------------


def save_pack(
    pack_root,
    *,
    sector,
    pack_version,
    source_info,
    telemetry,
    catalogue,
    entities,
    episodes,
    splits=None,
    evaluation=None,
    notes=(),
):
    """Write one complete, immutable sector pack.

    ``telemetry`` is any iterable of long DataFrames using the telemetry
    schema.  Streaming it keeps memory bounded and keeps the sector notebooks
    free of file layout, part numbering and manifest logic.

    A ``(episode_id, metric_id)`` pair must be present only where the source
    attempted to observe it.  A metric that was never available in an episode
    is therefore absent, not a run of invalid rows.
    """

    catalogue = catalogue[PACK_SCHEMAS["metric_catalogue"]].reset_index(drop=True)
    entities = entities[PACK_SCHEMAS["entity_registry"]].reset_index(drop=True)
    episodes = episodes[PACK_SCHEMAS["observation_episodes"]].reset_index(drop=True)
    validate_catalogue(catalogue)

    splits = dict(splits or {})
    evaluation = dict(evaluation or {})
    for name in splits:
        if name not in SPLIT_SCHEMAS:
            raise ValueError(f"Unknown SPLITS table: {name}")
    for name in evaluation:
        if name not in EVAL_SCHEMAS:
            raise ValueError(f"Unknown PACK-EVAL table: {name}")

    observed = {
        "entities": set(), "episodes": set(), "metrics": set(),
        "episode_owner": {}, "pairs": set(),
    }

    with new_output_directory(pack_root) as pack:
        core = pack / "PACK-CORE"
        telemetry_directory = core / "telemetry"
        telemetry_directory.mkdir(parents=True)

        rows = 0
        for number, batch in enumerate(telemetry):
            if batch.empty:
                continue
            batch = batch[CORE_SCHEMAS["telemetry"]].reset_index(drop=True)
            batch["event_ts"] = pd.to_datetime(batch["event_ts"], utc=True)
            for column in ("entity_id", "episode_id", "metric_id", "quality_code"):
                batch[column] = batch[column].astype(str)
            _validate_telemetry(batch, f"telemetry part {number}")
            batch.to_parquet(
                telemetry_directory / f"part-{number:05d}.parquet",
                index=False, compression="zstd",
            )
            rows += len(batch)
            identities = batch[["entity_id", "episode_id", "metric_id"]].drop_duplicates()
            observed["entities"].update(identities["entity_id"])
            observed["episodes"].update(identities["episode_id"])
            observed["metrics"].update(identities["metric_id"])
            observed["pairs"].update(
                map(tuple, identities[["episode_id", "metric_id"]].to_numpy())
            )
            observed["episode_owner"].update(
                zip(identities["episode_id"], identities["entity_id"])
            )
        if rows == 0:
            raise ValueError("The pack contains no observations")

        _validate_references(catalogue, entities, episodes, observed)

        row_counts = {"telemetry": rows}
        table_hashes = {
            "telemetry": _directory_digest(telemetry_directory, TELEMETRY_KEYS)[0]
        }
        for name, frame in (
            ("metric_catalogue", catalogue),
            ("entity_registry", entities),
            ("observation_episodes", episodes),
        ):
            frame.to_parquet(core / f"{name}.parquet", index=False)
            table_hashes[name] = table_digest(frame, PACK_SCHEMAS[name])
            row_counts[name] = len(frame)

        for folder, tables, schemas in (
            ("SPLITS", splits, SPLIT_SCHEMAS),
            ("PACK-EVAL", evaluation, EVAL_SCHEMAS),
        ):
            if not tables:
                continue
            (pack / folder).mkdir()
            for name, frame in tables.items():
                frame[schemas[name]].to_parquet(
                    pack / folder / f"{name}.parquet", index=False
                )

        manifest = {
            "pack_interface_version": PACK_INTERFACE_VERSION,
            "pack_version": str(pack_version),
            "sector": str(sector),
            "observation_layout": "long_metric_level",
            "availability_rule": "episode_metric_pair_present_only_when_attempted",
            "split_tables": sorted(splits),
            "evaluation_tables": sorted(evaluation),
            "metric_ids": catalogue["metric_id"].astype(str).tolist(),
            "core_row_counts": row_counts,
            "episode_metric_pairs": len(observed["pairs"]),
            "fingerprint": _combine(table_hashes),
            "source": source_info,
            "notes": list(notes),
        }
        write_json(pack / "pack_manifest.json", manifest)

    return read_pack(pack_root)


def pack_fingerprint(pack_root):
    """One fingerprint over every model-visible pack table."""

    core = Path(pack_root) / "PACK-CORE"
    table_hashes = {
        "telemetry": _directory_digest(core / "telemetry", TELEMETRY_KEYS)[0]
    }
    for name in ("metric_catalogue", "entity_registry", "observation_episodes"):
        frame = pd.read_parquet(core / f"{name}.parquet")
        if list(frame.columns) != PACK_SCHEMAS[name]:
            raise ValueError(f"Unexpected schema in pack {name}")
        table_hashes[name] = table_digest(frame, PACK_SCHEMAS[name])
    return _combine(table_hashes)


def read_pack(pack_root):
    """Validate a completed pack and return its manifest."""

    pack_root = Path(pack_root)
    manifest = read_json(pack_root / "pack_manifest.json")
    if manifest["pack_interface_version"] != PACK_INTERFACE_VERSION:
        raise ValueError(f"Unsupported pack interface: {manifest['pack_interface_version']}")

    core = pack_root / "PACK-CORE"
    catalogue = pd.read_parquet(core / "metric_catalogue.parquet")
    entities = pd.read_parquet(core / "entity_registry.parquet")
    episodes = pd.read_parquet(core / "observation_episodes.parquet")
    validate_catalogue(catalogue)

    observed = {
        "entities": set(), "episodes": set(), "metrics": set(), "episode_owner": {},
    }
    for part in _parts(core / "telemetry"):
        frame = pd.read_parquet(part)
        _validate_telemetry(frame, part.name)
        identities = frame[["entity_id", "episode_id", "metric_id"]].astype(str).drop_duplicates()
        observed["entities"].update(identities["entity_id"])
        observed["episodes"].update(identities["episode_id"])
        observed["metrics"].update(identities["metric_id"])
        observed["episode_owner"].update(zip(identities["episode_id"], identities["entity_id"]))
    _validate_references(catalogue, entities, episodes, observed)

    for folder, names, schemas in (
        ("SPLITS", manifest["split_tables"], SPLIT_SCHEMAS),
        ("PACK-EVAL", manifest["evaluation_tables"], EVAL_SCHEMAS),
    ):
        for name in names:
            frame = pd.read_parquet(pack_root / folder / f"{name}.parquet")
            if list(frame.columns) != schemas[name]:
                raise ValueError(f"Unexpected schema in {folder}/{name}")

    if pack_fingerprint(pack_root) != manifest["fingerprint"]:
        raise ValueError("PACK-CORE content no longer matches its manifest")
    return manifest


# --------------------------------------------------------------------------
# Pack -> canonical
# --------------------------------------------------------------------------


def _collection_gaps(connection, tolerance_factor=GAP_TOLERANCE_FACTOR):
    """Internal gaps for any metric that declares a cadence.

    Gaps are found inside one ``(entity, episode, metric)`` series, so a
    recording never implies an obligation to the next recording, while a hole
    inside a single recording is still reported.
    """

    gaps = connection.execute(
        """
        WITH ordered AS (
            SELECT t.entity_id, t.episode_id, t.metric_id, t.event_ts,
                   lag(t.event_ts) OVER (
                       PARTITION BY t.entity_id, t.episode_id, t.metric_id
                       ORDER BY t.event_ts
                   ) AS previous_ts,
                   CAST(c.expected_cadence_seconds AS DOUBLE) AS expected_cadence_seconds
            FROM selected_observations AS t
            JOIN metric_catalogue AS c USING (metric_id)
            WHERE c.expected_cadence_seconds IS NOT NULL
        )
        SELECT entity_id, episode_id, metric_id,
               previous_ts + expected_cadence_seconds * INTERVAL '1 second' AS gap_start,
               event_ts AS gap_end,
               expected_cadence_seconds,
               'within_episode_declared_cadence' AS coverage_basis
        FROM ordered
        WHERE previous_ts IS NOT NULL
          AND epoch(event_ts - previous_ts) > expected_cadence_seconds * ?
        ORDER BY entity_id, episode_id, metric_id, gap_start
        """,
        [float(tolerance_factor)],
    ).df()
    for column in ("gap_start", "gap_end"):
        gaps[column] = pd.to_datetime(gaps[column], utc=True)
    return gaps[CORE_SCHEMAS["collection_gaps"]]


def _copy_tables(source_root, destination_root, names, schemas):
    if not names:
        return {}
    destination_root.mkdir()
    counts = {}
    for name in names:
        frame = pd.read_parquet(source_root / f"{name}.parquet")[schemas[name]]
        frame.to_parquet(destination_root / f"{name}.parquet", index=False)
        counts[name] = len(frame)
    return counts


def build_canonical(pack_root, run_root, *, include_evaluation=True, as_of_ts=None):
    """Convert any valid sector pack into immutable canonical directories."""

    pack_root, run_root = Path(pack_root), Path(run_root)
    pack_manifest = read_pack(pack_root)
    core_source = pack_root / "PACK-CORE"
    catalogue = pd.read_parquet(core_source / "metric_catalogue.parquet")
    pack_entities = pd.read_parquet(core_source / "entity_registry.parquet")
    pack_episodes = pd.read_parquet(core_source / "observation_episodes.parquet")
    as_of = pd.to_datetime(as_of_ts, utc=True) if as_of_ts is not None else None

    telemetry_glob = str(core_source / "telemetry" / "part-*.parquet").replace("'", "''")
    where = ""
    if as_of is not None:
        where = f"WHERE event_ts <= TIMESTAMPTZ '{as_of.isoformat()}'"

    with new_output_directory(run_root) as temporary, _duckdb().connect() as connection:
        core = temporary / "SPEC-CORE"
        telemetry_directory = core / "telemetry"
        telemetry_directory.mkdir(parents=True)

        connection.register("metric_catalogue", catalogue)
        connection.execute(f"""
            CREATE VIEW selected_observations AS
            SELECT CAST(event_ts AS TIMESTAMPTZ) AS event_ts,
                   CAST(entity_id AS VARCHAR)    AS entity_id,
                   CAST(episode_id AS VARCHAR)   AS episode_id,
                   CAST(metric_id AS VARCHAR)    AS metric_id,
                   value,
                   CAST(quality_code AS VARCHAR) AS quality_code
            FROM read_parquet('{telemetry_glob}') {where}
        """)

        audit = connection.execute("""
            SELECT count(*) AS rows,
                   count(*) - count(DISTINCT (event_ts, entity_id, episode_id, metric_id))
                       AS duplicate_keys
            FROM selected_observations
        """).df().iloc[0]
        if int(audit["rows"]) == 0:
            raise ValueError("No observations are available at the requested as_of_ts")
        if int(audit["duplicate_keys"]):
            raise ValueError("Duplicate telemetry keys exist across Pack parts")

        episode_bounds = connection.execute("""
            SELECT episode_id, entity_id,
                   min(event_ts) AS observed_from, max(event_ts) AS observed_to
            FROM selected_observations GROUP BY episode_id, entity_id
        """).df()
        for column in ("observed_from", "observed_to"):
            episode_bounds[column] = pd.to_datetime(episode_bounds[column], utc=True)

        episodes = episode_bounds.merge(
            pack_episodes.astype({"episode_id": str, "entity_id": str}),
            on=["episode_id", "entity_id"], how="left", validate="one_to_one",
        )
        if episodes["episode_basis"].isna().any():
            raise ValueError("Canonical episode basis is missing")
        episodes = episodes[CORE_SCHEMAS["observation_episodes"]]

        entities = (
            episode_bounds.groupby("entity_id", as_index=False)
            .agg(observed_from=("observed_from", "min"), observed_to=("observed_to", "max"))
            .merge(pack_entities.astype({"entity_id": str}), on="entity_id",
                   how="left", validate="one_to_one")
            .assign(validity_basis="derived_from_observations_as_of")
        )
        if entities["entity_type"].isna().any():
            raise ValueError("Canonical entity type is missing")
        entities = entities[CORE_SCHEMAS["entity_registry"]]

        telemetry_digest = hashlib.sha256()
        telemetry_rows = 0
        quality_counts = {}
        query = connection.execute(f"""
            SELECT {', '.join(CORE_SCHEMAS['telemetry'])} FROM selected_observations
        """)
        batches = _record_batches(query, CANONICAL_BATCH_ROWS)
        for number, batch in enumerate(batches):
            frame = batch.to_pandas()[CORE_SCHEMAS["telemetry"]]
            frame["event_ts"] = pd.to_datetime(frame["event_ts"], utc=True)
            frame = frame.sort_values(TELEMETRY_KEYS, kind="stable").reset_index(drop=True)
            frame.to_parquet(
                telemetry_directory / f"part-{number:05d}.parquet",
                index=False, compression="zstd",
            )
            telemetry_digest.update(table_digest(frame, TELEMETRY_KEYS).encode("ascii"))
            telemetry_rows += len(frame)
            for code, count in frame["quality_code"].value_counts().items():
                quality_counts[str(code)] = quality_counts.get(str(code), 0) + int(count)

        table_hashes = {"telemetry": telemetry_digest.hexdigest()}
        row_counts = {"telemetry": telemetry_rows}
        sidecars = {
            "metric_catalogue": catalogue[CORE_SCHEMAS["metric_catalogue"]],
            "entity_registry": entities,
            "observation_episodes": episodes,
            "collection_gaps": _collection_gaps(connection),
        }
        for name, frame in sidecars.items():
            frame.to_parquet(core / f"{name}.parquet", index=False)
            table_hashes[name] = table_digest(frame, CORE_SCHEMAS[name])
            row_counts[name] = len(frame)

        core_manifest = {
            "contract_version": CORE_VERSION,
            "sector": pack_manifest["sector"],
            "source_pack_version": pack_manifest["pack_version"],
            "as_of_ts": as_of,
            "tables": CORE_SCHEMAS,
            "row_counts": row_counts,
            "fingerprint": _combine(table_hashes),
            "quality_counts": quality_counts,
            "coverage_basis": "within_episode_declared_cadence",
            "gap_tolerance_factor": GAP_TOLERANCE_FACTOR,
            "availability_rule": pack_manifest["availability_rule"],
        }
        write_json(core / "manifest.json", core_manifest)

        split_rows = _copy_tables(
            pack_root / "SPLITS", temporary / "SPLITS",
            pack_manifest["split_tables"], SPLIT_SCHEMAS,
        )
        evaluation_rows = {}
        if include_evaluation:
            evaluation_rows = _copy_tables(
                pack_root / "PACK-EVAL", temporary / "SPEC-EVAL",
                pack_manifest["evaluation_tables"], EVAL_SCHEMAS,
            )

        write_json(temporary / "run_manifest.json", {
            "run_root": str(run_root),
            "pack_root": str(pack_root),
            "pack_manifest_sha256": file_sha256(pack_root / "pack_manifest.json"),
            "evaluation_mounted": bool(evaluation_rows),
            "as_of_ts": as_of,
            "core": core_manifest,
            "split_rows": split_rows,
            "evaluation_rows": evaluation_rows,
        })

    return read_json(run_root / "run_manifest.json")


def check_core(core_root):
    """Validate a completed SPEC-CORE and report its contents."""

    core_root = Path(core_root)
    manifest = read_json(core_root / "manifest.json")
    if manifest["contract_version"] != CORE_VERSION:
        raise ValueError("Unsupported SPEC-CORE version")

    telemetry_digest = hashlib.sha256()
    telemetry_rows = 0
    quality_counts = {}
    for part in _parts(core_root / "telemetry"):
        frame = pd.read_parquet(part)
        _validate_telemetry(frame, part.name)
        telemetry_rows += len(frame)
        telemetry_digest.update(table_digest(frame, TELEMETRY_KEYS).encode("ascii"))
        for code, count in frame["quality_code"].value_counts().items():
            quality_counts[str(code)] = quality_counts.get(str(code), 0) + int(count)

    table_hashes = {"telemetry": telemetry_digest.hexdigest()}
    row_counts = {"telemetry": telemetry_rows}
    for name in ("metric_catalogue", "entity_registry", "observation_episodes", "collection_gaps"):
        frame = pd.read_parquet(core_root / f"{name}.parquet")
        if list(frame.columns) != CORE_SCHEMAS[name]:
            raise ValueError(f"Unexpected {name} schema")
        table_hashes[name] = table_digest(frame, CORE_SCHEMAS[name])
        row_counts[name] = len(frame)

    if row_counts != manifest["row_counts"]:
        raise ValueError("SPEC-CORE row counts do not match the manifest")
    if quality_counts != manifest["quality_counts"]:
        raise ValueError("Telemetry quality counts do not match the manifest")
    if _combine(table_hashes) != manifest["fingerprint"]:
        raise ValueError("SPEC-CORE content does not match its manifest")

    paths = {
        name: str(core_root / f"{name}.parquet").replace("'", "''")
        for name in ("metric_catalogue", "entity_registry", "observation_episodes")
    }
    telemetry_glob = str(core_root / "telemetry" / "part-*.parquet").replace("'", "''")
    with _duckdb().connect() as connection:
        audit = connection.execute(f"""
            SELECT count(*) - count(DISTINCT (t.event_ts, t.entity_id, t.episode_id, t.metric_id))
                       AS duplicate_keys,
                   sum(CASE WHEN c.metric_id IS NULL THEN 1 ELSE 0 END) AS unknown_metrics,
                   sum(CASE WHEN r.entity_id IS NULL THEN 1 ELSE 0 END) AS unknown_entities,
                   sum(CASE WHEN e.episode_id IS NULL THEN 1 ELSE 0 END) AS unknown_episodes
            FROM read_parquet('{telemetry_glob}') AS t
            LEFT JOIN read_parquet('{paths["metric_catalogue"]}') AS c USING (metric_id)
            LEFT JOIN read_parquet('{paths["entity_registry"]}') AS r USING (entity_id)
            LEFT JOIN read_parquet('{paths["observation_episodes"]}') AS e
                   ON t.episode_id = e.episode_id AND t.entity_id = e.entity_id
        """).df().iloc[0]

    failures = {name: int(audit[name]) for name in audit.index if int(audit[name]) != 0}
    if failures:
        raise ValueError(f"SPEC-CORE key audit failed: {failures}")

    return {
        "telemetry_rows": telemetry_rows,
        "quality_counts": quality_counts,
        "duplicate_keys": 0,
        "foreign_key_failures": 0,
        "fingerprint_verified": True,
        **{name: row_counts[name] for name in list(CORE_SCHEMAS)[1:]},
    }


def core_fingerprint(core_root):
    """Read the model-visible fingerprint without opening SPEC-EVAL."""

    return read_json(Path(core_root) / "manifest.json")["fingerprint"]


def _evaluation_root(run_root):
    run_root = Path(run_root)
    return next(
        (run_root / name for name in ("SPEC-EVAL", "PACK-EVAL")
         if (run_root / name).is_dir()),
        run_root / "SPEC-EVAL",
    )


def _entity_windows(run_root):
    """Return observed entity bounds from a Pack or canonical run."""

    run_root = Path(run_root)
    for folder in ("SPEC-CORE", "PACK-CORE"):
        core = run_root / folder
        if not core.is_dir():
            continue
        registry = pd.read_parquet(core / "entity_registry.parquet")
        if {"observed_from", "observed_to"} <= set(registry.columns):
            windows = registry[["entity_id", "observed_from", "observed_to"]].copy()
        else:
            telemetry_glob = str(core / "telemetry" / "part-*.parquet").replace("'", "''")
            with _duckdb().connect() as connection:
                windows = connection.execute(f"""
                    SELECT CAST(entity_id AS VARCHAR) AS entity_id,
                           min(event_ts) AS observed_from,
                           max(event_ts) AS observed_to
                    FROM read_parquet('{telemetry_glob}')
                    GROUP BY entity_id
                """).df()
        windows["entity_id"] = windows["entity_id"].astype(str)
        for column in ("observed_from", "observed_to"):
            windows[column] = pd.to_datetime(windows[column], utc=True)
        return windows
    raise FileNotFoundError(f"No PACK-CORE or SPEC-CORE found in {run_root}")


def _fault_assignments(run_root, partition_table=None):
    """Assign each declared fault once, after checking telemetry overlap."""

    run_root = Path(run_root)
    evaluation_root = _evaluation_root(run_root)
    events_path = evaluation_root / "fault_events.parquet"
    intervals_path = evaluation_root / "fault_entity_intervals.parquet"
    columns = [
        "fault_id", "fault_type", "partition", "scoreable",
        "cross_partition", "scoreable_entity_count",
    ]
    if not (events_path.is_file() and intervals_path.is_file()):
        return pd.DataFrame(columns=columns)

    events = pd.read_parquet(events_path).astype({"fault_id": str})[
        ["fault_id", "fault_type"]
    ].drop_duplicates("fault_id")
    intervals = pd.read_parquet(intervals_path).astype({
        "fault_id": str, "entity_id": str,
    })
    windows = _entity_windows(run_root)
    joined = intervals.merge(windows, on="entity_id", how="left")
    starts = pd.to_datetime(joined["start_ts"], utc=True, errors="coerce")
    ends = pd.to_datetime(joined["end_ts"], utc=True, errors="coerce")
    joined["scoreable_interval"] = (
        joined["observed_from"].notna()
        & starts.notna()
        & starts.le(joined["observed_to"])
        & (ends.isna() | ends.ge(joined["observed_from"]))
    )

    if partition_table is not None:
        partitions_path = run_root / "SPLITS" / f"{partition_table}.parquet"
        if not partitions_path.is_file():
            return pd.DataFrame(columns=columns)
        partitions = pd.read_parquet(partitions_path).astype({"entity_id": str})[
            ["entity_id", "partition"]
        ]
        if partitions["entity_id"].duplicated().any():
            raise ValueError(f"{partition_table} assigns an entity more than once")
        joined = joined.merge(partitions, on="entity_id", how="left")
    else:
        joined["partition"] = "all"

    assignments = []
    for event in events.itertuples(index=False):
        rows = joined.loc[joined["fault_id"].eq(event.fault_id)]
        scoreable = rows.loc[rows["scoreable_interval"]]
        partitions = sorted(scoreable["partition"].dropna().astype(str).unique())
        has_unassigned = scoreable["partition"].isna().any()
        cross_partition = len(partitions) > 1
        if scoreable.empty:
            partition = "unscoreable"
        elif cross_partition:
            partition = "cross_partition"
        elif has_unassigned or not partitions:
            partition = "unassigned"
        else:
            partition = partitions[0]
        assignments.append({
            "fault_id": event.fault_id,
            "fault_type": event.fault_type,
            "partition": partition,
            "scoreable": not scoreable.empty,
            "cross_partition": cross_partition,
            "scoreable_entity_count": scoreable["entity_id"].nunique(),
        })
    return pd.DataFrame(assignments, columns=columns)


def check_evaluation(run_root):
    """Check evaluation truth against observable telemetry."""

    run_root = Path(run_root)
    evaluation_root = _evaluation_root(run_root)
    if not evaluation_root.is_dir():
        return {"evaluation_mounted": False}

    windows = _entity_windows(run_root)
    known = set(windows["entity_id"])
    report = {"evaluation_mounted": True}
    intervals_path = evaluation_root / "fault_entity_intervals.parquet"
    if intervals_path.is_file():
        intervals = pd.read_parquet(intervals_path)
        entity_ids = intervals["entity_id"].astype(str)
        unknown_rows = ~entity_ids.isin(known)
        unknown_entities = sorted(set(entity_ids.loc[unknown_rows]))
        assignments = _fault_assignments(run_root)
        report.update({
            "fault_intervals": len(intervals),
            "intervals_on_unknown_entities": int(unknown_rows.sum()),
            "unknown_entities": len(unknown_entities),
            "unknown_entity_examples": unknown_entities[:5],
            "faults_scoreable": int(assignments["scoreable"].sum()),
            "faults_unscoreable": int((~assignments["scoreable"]).sum()),
        })

        interval_windows = intervals.astype({"entity_id": str}).merge(
            windows, on="entity_id", how="left"
        )
        starts = pd.to_datetime(interval_windows["start_ts"], utc=True, errors="coerce")
        ends = pd.to_datetime(interval_windows["end_ts"], utc=True, errors="coerce")
        overlap = (
            interval_windows["observed_from"].notna()
            & starts.notna()
            & starts.le(interval_windows["observed_to"])
            & (ends.isna() | ends.ge(interval_windows["observed_from"]))
        )
        report["intervals_scoreable"] = int(overlap.sum())
        report["intervals_outside_observed_window"] = int((~overlap).sum())

    events_path = evaluation_root / "fault_events.parquet"
    if events_path.is_file() and intervals_path.is_file():
        events = pd.read_parquet(events_path)
        referenced = set(pd.read_parquet(intervals_path)["fault_id"].astype(str))
        declared = set(events["fault_id"].astype(str))
        report["fault_events"] = len(events)
        report["events_without_intervals"] = len(declared - referenced)
        report["intervals_without_events"] = len(referenced - declared)

    conditions_path = evaluation_root / "condition_states.parquet"
    if conditions_path.is_file():
        conditions = pd.read_parquet(conditions_path)
        report["condition_states"] = len(conditions)
        report["conditions_on_unknown_entities"] = int(
            (~conditions["entity_id"].astype(str).isin(known)).sum()
        )
    return report


def fault_coverage(run_root, partition_table="entity_partitions"):
    """Summarise declared and scoreable faults with one assignment per fault."""

    assignments = _fault_assignments(run_root, partition_table)
    columns = [
        "partition", "fault_type", "declared_faults", "scoreable_faults",
        "unscoreable_faults", "cross_partition_faults",
        "scoreable_entity_fault_pairs",
    ]
    if assignments.empty:
        return pd.DataFrame(columns=columns)
    assignments["unscoreable"] = ~assignments["scoreable"]
    return (
        assignments.groupby(["partition", "fault_type"], as_index=False, dropna=False)
        .agg(
            declared_faults=("fault_id", "nunique"),
            scoreable_faults=("scoreable", "sum"),
            unscoreable_faults=("unscoreable", "sum"),
            cross_partition_faults=("cross_partition", "sum"),
            scoreable_entity_fault_pairs=("scoreable_entity_count", "sum"),
        )[columns]
        .sort_values(["partition", "fault_type"])
        .reset_index(drop=True)
    )
