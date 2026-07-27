"""Readable, notebook-facing Milestone 1 workflows.

Notebooks call these functions; they do not reimplement translation or acceptance
logic. The functions return JSON-serialisable reports so every run leaves auditable
evidence beside its outputs.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tracemalloc
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .core import (
    CONTRACT_VERSION,
    ContractViolation,
    assert_event_as_of,
    canonical_frame_hash,
    schema_root,
)
from .evaluation import EVAL_CONTRACT_VERSION, eval_schema_root
from .oil_well import ThreeWAdapter
from .packs import (
    OIL_WELL_PACK_VERSION,
    TELECOM_PACK_VERSION,
    load_oil_well_pack,
    load_telecom_pack,
)
from .runtime import DetectorConfig, RobustHistoryDetector, score_core_directory
from .telecom import FEC_CEILING, NativeSelection, SyntheticGponAdapter


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def runtime_provenance() -> dict[str, str]:
    """Return the source revision recorded by the notebook bootstrap cell."""

    return {
        "repository": "https://github.com/LiliDopidze/anomaly_detection",
        "requested_ref": os.getenv("ANOMALY_RUNTIME_REF", "local"),
        "resolved_commit": os.getenv(
            "ANOMALY_RUNTIME_COMMIT", "local-working-tree"
        ),
    }


def contract_summary() -> dict[str, Any]:
    """Return the neutral contract and both Milestone 1 pack summaries."""

    telecom = load_telecom_pack()
    oil_well = load_oil_well_pack()
    table_name = lambda path: path.name.removesuffix(".schema.json")
    return {
        "spec_core_version": CONTRACT_VERSION,
        "spec_eval_version": EVAL_CONTRACT_VERSION,
        "spec_core_tables": sorted(
            table_name(path) for path in schema_root().glob("*.json")
        ),
        "spec_eval_tables": sorted(
            table_name(path) for path in eval_schema_root().glob("*.json")
        ),
        "packs": {
            "telecom": {
                "version": TELECOM_PACK_VERSION,
                "entities": list(telecom.entity_types()),
                "metric_count": len(telecom.metric_specs()),
                "relation_types": sorted(telecom.relation_specs()),
                "parameters": dict(telecom.parameters()),
            },
            "oil_well": {
                "version": OIL_WELL_PACK_VERSION,
                "entities": list(oil_well.entity_types()),
                "metric_count": len(oil_well.metric_specs()),
                "relation_types": sorted(oil_well.relation_specs()),
            },
        },
        "truth_boundary": (
            "runtime imports core only; translators alone may materialise SPEC-EVAL"
        ),
    }


def _child_environment() -> dict[str, str]:
    env = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[1])
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        source_root if not existing else source_root + os.pathsep + existing
    )
    return env


def _telecom_memory_probe(
    adapter: SyntheticGponAdapter,
    source: Path,
) -> dict[str, Any]:
    """Trace one complete native-to-long batch, scoped independently of the full run."""

    import pyarrow.parquet as pq

    panel_path = adapter._native_path(source, "reference_dataset.parquet")
    panel_file = pq.ParquetFile(panel_path)
    native_columns = set(panel_file.schema_arrow.names)
    metric_fields = [
        field for field in adapter._metric_mapping() if field in native_columns
    ]
    columns = ["timestamp_utc", "ont_id", *metric_fields]
    tracemalloc.start()
    native_rows = 0
    canonical_rows = 0
    for batch in panel_file.iter_batches(
        batch_size=adapter.selection.batch_native_rows,
        columns=columns,
    ):
        selected = adapter._filter_native(batch.to_pandas())
        if selected.empty:
            continue
        widened = adapter.adapt_telemetry(
            selected,
            cadence_seconds=float(
                next(
                    spec.expected_cadence_seconds
                    for spec in adapter.pack.metric_specs().values()
                    if spec.expected_cadence_seconds
                )
            ),
        )
        native_rows = len(selected)
        canonical_rows = len(widened)
        break
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if native_rows == 0:
        raise ValueError("configured telecom selection contains no probe rows")
    return {
        "tracemalloc_peak_gib": peak_bytes / (1024**3),
        "tracemalloc_scope": "one complete configured native-to-long batch",
        "probe_native_rows": native_rows,
        "probe_canonical_rows": canonical_rows,
    }


def _representation_report(core_directory: Path) -> dict[str, Any]:
    manifest = json.loads(
        (core_directory / "manifest.json").read_text(encoding="utf-8")
    )
    catalogue = pd.read_parquet(core_directory / "metric_catalogue.parquet")
    registry = pd.read_parquet(core_directory / "entity_registry.parquet")
    parts = sorted((core_directory / "telemetry").glob("part-*.parquet"))
    stored_bytes = sum(path.stat().st_size for path in parts)
    canonical_rows = int(manifest["row_counts"]["telemetry"])
    ont_count = int(registry["entity_type"].eq("ont").sum())
    row_multiplier = int(len(catalogue))
    bytes_per_ont_for_fixture = stored_bytes / max(ont_count, 1)
    return {
        "logical_shape": "long telemetry rows",
        "physical_shape": "partitioned Parquet batches",
        "canonical_rows": canonical_rows,
        "stored_bytes": stored_bytes,
        "bytes_per_canonical_row": stored_bytes / max(canonical_rows, 1),
        "native_to_canonical_row_multiplier": row_multiplier,
        "fixture_ont_count": ont_count,
        "same_duration_storage_extrapolation_bytes": {
            str(entity_count): int(bytes_per_ont_for_fixture * entity_count)
            for entity_count in (10_000, 100_000, 1_000_000)
        },
        "decision_note": (
            "The logical contract remains long; detector interfaces must consume "
            "partitioned batches rather than materialising the complete table."
        ),
    }


def materialise_telecom(
    source: str | Path,
    run_root: str | Path,
    *,
    sample_start: str | None = None,
    sample_end: str | None = None,
    entity_ids: Iterable[str] = (),
    batch_native_rows: int = 250_000,
    memory_budget_gib: float = 8.0,
) -> dict[str, Any]:
    """Materialise one immutable telecom run in a clean subprocess."""

    source = Path(source)
    run_root = Path(run_root)
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite run: {run_root}")
    run_root.mkdir(parents=True)
    core = run_root / "SPEC-CORE"
    evaluation = run_root / "SPEC-EVAL"
    child_report_path = run_root / "materialisation_report.json"

    selection = NativeSelection(
        sample_start=sample_start,
        sample_end=sample_end,
        entity_ids=tuple(str(item) for item in entity_ids),
        batch_native_rows=batch_native_rows,
    )
    adapter = SyntheticGponAdapter(selection=selection)
    probe = _telecom_memory_probe(adapter, source)

    command = [
        sys.executable,
        "-m",
        "anomaly_detection.cli",
        "telecom",
        "--native",
        str(source),
        "--core",
        str(core),
        "--eval",
        str(evaluation),
        "--batch-native-rows",
        str(batch_native_rows),
        "--report",
        str(child_report_path),
    ]
    if sample_start is not None:
        command.extend(["--sample-start", sample_start])
    if sample_end is not None:
        command.extend(["--sample-end", sample_end])
    for entity_id in selection.entity_ids:
        command.extend(["--entity-id", entity_id])
    subprocess.run(command, check=True, env=_child_environment())

    materialisation = json.loads(child_report_path.read_text(encoding="utf-8"))
    memory = {
        **probe,
        "peak_rss_gib": float(materialisation["peak_rss_gib"]),
        "rss_scope": "clean full or selected materialisation subprocess",
        "budget_gib": memory_budget_gib,
        "budget_pass": float(materialisation["peak_rss_gib"]) <= memory_budget_gib,
        "gate_metric": "clean_subprocess_peak_rss_gib",
    }
    _write_json(run_root / "memory_report.json", memory)
    if not memory["budget_pass"]:
        raise MemoryError(
            f"materialisation peak RSS {memory['peak_rss_gib']:.2f} GiB "
            f"exceeds {memory_budget_gib:.2f} GiB"
        )

    representation = _representation_report(core)
    _write_json(run_root / "representation_report.json", representation)
    report = {
        "workflow": "telecom_materialisation",
        "runtime": runtime_provenance(),
        "contract": contract_summary(),
        "source": str(source),
        "run_root": str(run_root),
        "selection": {
            "sample_start": sample_start,
            "sample_end": sample_end,
            "entity_ids": list(selection.entity_ids),
            "batch_native_rows": batch_native_rows,
        },
        "materialisation": materialisation,
        "memory": memory,
        "representation": representation,
    }
    _write_json(run_root / "workflow_report.json", report)
    return report


def _find_sensitive_native_sample(
    adapter: SyntheticGponAdapter,
    source: Path,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Find the smallest entity set that exercises clipped and invalid branches."""

    import pyarrow.parquet as pq

    panel_path = adapter._native_path(source, "reference_dataset.parquet")
    panel_file = pq.ParquetFile(panel_path)
    available = set(panel_file.schema_arrow.names)
    metric_fields = [
        field for field in adapter._metric_mapping() if field in available
    ]
    scan_fields = ["timestamp_utc", "ont_id", *metric_fields]
    clipped_entity: str | None = None
    invalid_entity: str | None = None
    for batch in panel_file.iter_batches(
        batch_size=adapter.selection.batch_native_rows,
        columns=scan_fields,
    ):
        frame = batch.to_pandas()
        if "fec_count" in frame:
            clipped = frame.loc[
                pd.to_numeric(frame["fec_count"], errors="coerce").ge(FEC_CEILING),
                "ont_id",
            ]
            if clipped_entity is None and len(clipped):
                clipped_entity = str(clipped.iloc[0])
        invalid = frame.loc[frame[metric_fields].isna().any(axis=1), "ont_id"]
        if invalid_entity is None and len(invalid):
            invalid_entity = str(invalid.iloc[0])
        if clipped_entity is not None and invalid_entity is not None:
            break
    if clipped_entity is None or invalid_entity is None:
        raise AssertionError(
            "could not construct a sample containing both clipped and invalid values"
        )
    entities = tuple(sorted({clipped_entity, invalid_entity}))

    frames = []
    for batch in panel_file.iter_batches(
        batch_size=adapter.selection.batch_native_rows,
    ):
        frame = batch.to_pandas()
        selected = frame.loc[frame["ont_id"].astype(str).isin(entities)]
        if len(selected):
            frames.append(selected)
    if not frames:
        raise AssertionError(f"could not load sensitive entities: {entities}")
    return pd.concat(frames, ignore_index=True), entities


def _timestamp_canary_leaks(
    bundle: dict[str, pd.DataFrame],
    canaries: Iterable[pd.Timestamp],
) -> list[str]:
    expected = {pd.Timestamp(value) for value in canaries}
    leaks: list[str] = []
    for table_name, frame in bundle.items():
        for column in frame.columns:
            name = str(column).lower()
            if "ts" not in name and "time" not in name:
                continue
            values = pd.to_datetime(frame[column], utc=True, errors="coerce")
            if any(values.eq(value).any() for value in expected):
                leaks.append(f"{table_name}.{column}")
    return sorted(leaks)


def _read_optional_csv(
    adapter: SyntheticGponAdapter,
    source: Path,
    filename: str,
    *,
    parse_dates: list[str] | None = None,
    evaluation: bool = False,
) -> pd.DataFrame | None:
    path = adapter._native_path(
        source,
        filename,
        evaluation=evaluation,
        required=False,
    )
    return None if path is None else pd.read_csv(path, parse_dates=parse_dates)


def _write_invariance_sources(
    adapter: SyntheticGponAdapter,
    source: Path,
    destination: Path,
    panel: pd.DataFrame,
    topology: pd.DataFrame,
    windows: pd.DataFrame,
    engineering: pd.DataFrame | None,
    tickets: pd.DataFrame | None,
) -> tuple[Path, Path, set[pd.Timestamp]]:
    """Create original/canary and physically redacted native source trees."""

    original = destination / "original_native"
    redacted = destination / "redacted_native"
    (original / "evaluation").mkdir(parents=True)
    redacted.mkdir(parents=True)

    truth_columns = lambda frame: [
        name for name in frame.columns if str(name).startswith("gt_")
    ]
    panel.to_parquet(original / "reference_dataset.parquet", index=False)
    panel.drop(columns=truth_columns(panel)).to_parquet(
        redacted / "reference_dataset.parquet", index=False
    )
    topology.to_csv(original / "topology.csv", index=False)
    topology.drop(columns=truth_columns(topology)).to_csv(
        redacted / "topology.csv", index=False
    )
    windows.to_csv(original / "entity_service_windows.csv", index=False)
    windows.to_csv(redacted / "entity_service_windows.csv", index=False)
    if engineering is not None:
        engineering.to_csv(original / "engineering_events.csv", index=False)
        engineering.to_csv(redacted / "engineering_events.csv", index=False)

    canaries: set[pd.Timestamp] = set()

    def shifted(frame: pd.DataFrame, timestamp_columns: Iterable[str]):
        result = frame.copy()
        for column in timestamp_columns:
            if column not in result:
                continue
            values = pd.to_datetime(result[column], utc=True, errors="coerce")
            values = values + pd.DateOffset(years=50)
            result[column] = values
            canaries.update(pd.Timestamp(value) for value in values.dropna())
        return result

    if tickets is not None:
        shifted(tickets, ("reported_ts", "resolved_ts")).to_csv(
            original / "tickets.csv", index=False
        )

    evaluation_files = {
        "gt_fault_registry.csv": (
            "onset_ts",
            "first_observable_ts",
            "impact_ts",
            "repair_ts",
        ),
        "fault_entity_intervals.csv": (
            "active_start_ts",
            "active_end_ts",
            "impact_ts",
        ),
        "gt_fault_groups.csv": ("start_ts", "end_ts"),
    }
    for filename, timestamp_columns in evaluation_files.items():
        path = adapter._native_path(
            source, filename, evaluation=True, required=True
        )
        shifted(pd.read_csv(path), timestamp_columns).to_csv(
            original / "evaluation" / filename, index=False
        )
    return original, redacted, canaries


def _read_materialised_core(core: Path) -> dict[str, pd.DataFrame]:
    telemetry_parts = sorted((core / "telemetry").glob("part-*.parquet"))
    telemetry = pd.concat(
        (pd.read_parquet(path) for path in telemetry_parts),
        ignore_index=True,
    )
    return {
        "telemetry": telemetry,
        **{
            name: pd.read_parquet(core / f"{name}.parquet")
            for name in (
                "metric_catalogue",
                "entity_registry",
                "entity_relations",
                "operational_events",
                "collection_gaps",
            )
        },
    }


def _audit_materialised_telemetry(core: Path) -> dict[str, Any]:
    quality_counts: dict[str, int] = {}
    exposure_ranges = {
        "telecom.link.fec_count": [float("inf"), float("-inf")],
        "telecom.link.crc_errors": [float("inf"), float("-inf")],
    }
    for part in sorted((core / "telemetry").glob("part-*.parquet")):
        frame = pd.read_parquet(
            part,
            columns=["metric_id", "quality_code", "exposure"],
        )
        for code, count in frame["quality_code"].value_counts().items():
            quality_counts[str(code)] = quality_counts.get(str(code), 0) + int(count)
        for metric_id, limits in exposure_ranges.items():
            values = pd.to_numeric(
                frame.loc[frame["metric_id"].eq(metric_id), "exposure"],
                errors="coerce",
            ).dropna()
            if len(values):
                limits[0] = min(limits[0], float(values.min()))
                limits[1] = max(limits[1], float(values.max()))
    fec = exposure_ranges["telecom.link.fec_count"]
    crc = exposure_ranges["telecom.link.crc_errors"]
    if not quality_counts.get("clipped"):
        raise AssertionError("materialised output does not exercise clipped quality")
    if not quality_counts.get("invalid"):
        raise AssertionError("materialised output does not exercise invalid quality")
    if fec[0] != fec[1]:
        raise AssertionError(f"FEC exposure should be constant, observed {fec}")
    if not crc[0] < crc[1]:
        raise AssertionError(f"CRC exposure is degenerate, observed {crc}")
    return {
        "quality_counts": quality_counts,
        "exposure_ranges": exposure_ranges,
        "fec_constant_expected": True,
        "crc_non_degenerate_expected": True,
    }


def _runtime_isolation_hashes(
    core: Path,
    evaluation: Path,
) -> dict[str, str]:
    detector = RobustHistoryDetector(
        DetectorConfig(history_window=4, minimum_history=2, threshold=4.0)
    )
    hashes = {
        "mounted": canonical_frame_hash(score_core_directory(core, detector))
    }
    renamed = evaluation.with_name("SPEC-EVAL-renamed")
    evaluation.rename(renamed)
    hashes["renamed"] = canonical_frame_hash(score_core_directory(core, detector))
    shutil.rmtree(renamed)
    hashes["removed"] = canonical_frame_hash(score_core_directory(core, detector))
    evaluation.mkdir()
    hashes["empty"] = canonical_frame_hash(score_core_directory(core, detector))
    if len(set(hashes.values())) != 1:
        raise AssertionError(f"runtime output changed with SPEC-EVAL state: {hashes}")
    return hashes


def run_week1_acceptance(
    source: str | Path,
    run_root: str | Path,
) -> dict[str, Any]:
    """Run the truth-boundary and value-semantics acceptance suite."""

    source = Path(source)
    run_root = Path(run_root)
    core = run_root / "SPEC-CORE"
    evaluation = run_root / "SPEC-EVAL"
    if not core.is_dir() or not evaluation.is_dir():
        raise FileNotFoundError(
            "Notebook 02 output is missing; expected SPEC-CORE and SPEC-EVAL "
            f"under {run_root}"
        )
    report_path = run_root / "acceptance_report.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite acceptance evidence: {report_path}")

    adapter = SyntheticGponAdapter()
    panel, entities = _find_sensitive_native_sample(adapter, source)
    topology = pd.read_csv(adapter._native_path(source, "topology.csv"))
    topology = topology.loc[topology["ont_id"].astype(str).isin(entities)]
    windows = pd.read_csv(
        adapter._native_path(source, "entity_service_windows.csv"),
        parse_dates=["install_ts", "decommission_ts"],
    )
    windows = windows.loc[windows["entity_id"].astype(str).isin(entities)]
    engineering = _read_optional_csv(
        adapter,
        source,
        "engineering_events.csv",
        parse_dates=["ts"],
    )
    tickets = _read_optional_csv(
        adapter,
        source,
        "tickets.csv",
        parse_dates=["reported_ts", "resolved_ts"],
        evaluation=True,
    )

    with tempfile.TemporaryDirectory() as temp_root:
        temp = Path(temp_root)
        original_source, redacted_source, canaries = _write_invariance_sources(
            adapter,
            source,
            temp,
            panel,
            topology,
            windows,
            engineering,
            tickets,
        )
        original_core = temp / "original_output" / "SPEC-CORE"
        original_evaluation = temp / "original_output" / "SPEC-EVAL"
        redacted_core = temp / "redacted_output" / "SPEC-CORE"
        adapter.materialise(
            original_source,
            original_core,
            original_evaluation,
        )
        adapter.materialise(redacted_source, redacted_core, None)

        original_manifest = json.loads(
            (original_core / "manifest.json").read_text(encoding="utf-8")
        )
        redacted_manifest = json.loads(
            (redacted_core / "manifest.json").read_text(encoding="utf-8")
        )
        original_hashes = original_manifest["canonical_content_hashes"]
        redacted_hashes = redacted_manifest["canonical_content_hashes"]
        if original_hashes != redacted_hashes:
            raise AssertionError(
                "SPEC-CORE changed after evaluation files, tickets, and truth "
                "columns were physically removed"
            )

        original = _read_materialised_core(original_core)
        telemetry = original["telemetry"]
        clipped_present = bool(telemetry["quality_code"].eq("clipped").any())
        invalid_present = bool(telemetry["quality_code"].eq("invalid").any())
        if not clipped_present or not invalid_present:
            raise AssertionError(
                "translator invariance sample must exercise clipped and invalid branches"
            )

        canary_leaks = _timestamp_canary_leaks(original, canaries)
        if canary_leaks:
            raise AssertionError(
                f"truth timestamp canary reached SPEC-CORE: {canary_leaks}"
            )

        negative_control = {name: frame.copy() for name, frame in original.items()}
        canary = min(canaries)
        negative_control["telemetry"]["event_ts"] = pd.to_datetime(
            negative_control["telemetry"]["event_ts"], utc=True
        )
        negative_control["telemetry"].loc[
            negative_control["telemetry"].index[0], "event_ts"
        ] = canary
        detected = _timestamp_canary_leaks(negative_control, canaries)
        if not detected:
            raise AssertionError("value-level negative control did not trigger")

        runtime_hashes = _runtime_isolation_hashes(
            original_core, original_evaluation
        )

    materialised_audit = _audit_materialised_telemetry(core)

    delayed_event = {
        "event_start": pd.Timestamp("2026-01-01T10:00:00Z"),
        "known_at": pd.Timestamp("2026-01-01T10:05:00Z"),
    }
    delayed_rejected = False
    try:
        assert_event_as_of(delayed_event, pd.Timestamp("2026-01-01T10:02:00Z"))
    except ContractViolation:
        delayed_rejected = True
    assert_event_as_of(delayed_event, pd.Timestamp("2026-01-01T10:06:00Z"))
    if not delayed_rejected:
        raise AssertionError("as-of guard accepted an event before known_at")

    runtime_tree = ast.parse(
        Path(sys.modules["anomaly_detection.runtime"].__file__).read_text(
            encoding="utf-8"
        )
    )
    runtime_imports = [
        node.module
        for node in ast.walk(runtime_tree)
        if isinstance(node, ast.ImportFrom) and node.module
    ]
    if any(name.endswith("evaluation") for name in runtime_imports):
        raise AssertionError(f"runtime imports evaluation: {runtime_imports}")

    report = {
        "workflow": "week1_acceptance",
        "runtime": runtime_provenance(),
        "primary_leakage_proof": "translator canonical content invariance",
        "runtime_isolation_is_secondary_deployment_evidence": True,
        "sensitive_sample": {
            "entities": list(entities),
            "native_rows": len(panel),
            "clipped_present": clipped_present,
            "invalid_present": invalid_present,
        },
        "translator_invariance": {
            "passed": True,
            "test_level": "native source tree to materialised canonical output",
            "original_hashes": original_hashes,
            "redacted_hashes": redacted_hashes,
            "tickets_removed_explicitly": True,
            "evaluation_files_removed_physically": True,
        },
        "value_level_leakage": {
            "passed": True,
            "canary_leaks": canary_leaks,
            "truth_timestamp_canary_count": len(canaries),
            "negative_control_detected": detected,
        },
        "runtime_isolation": {
            "passed": True,
            "hashes": runtime_hashes,
        },
        "as_of_guard": {
            "passed": True,
            "delayed_event_rejected_before_known_at": delayed_rejected,
        },
        "materialised_telemetry": materialised_audit,
        "dependency_boundary": {
            "passed": True,
            "runtime_imports": runtime_imports,
        },
    }
    _write_json(report_path, report)
    return report


def challenge_threew(
    source: str | Path,
    run_root: str | Path,
    *,
    fixture_destination: str | Path | None = None,
) -> dict[str, Any]:
    """Materialise and report the smallest real 3W contract-challenge subset."""

    source = Path(source)
    run_root = Path(run_root)
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite run: {run_root}")
    run_root.mkdir(parents=True)
    core = run_root / "SPEC-CORE"
    evaluation = run_root / "SPEC-EVAL"
    fixture = None if fixture_destination is None else Path(fixture_destination)
    adapter = ThreeWAdapter()
    selected = adapter.select_minimal_subset(source)
    required = {
        "normal_instance",
        "transient_event",
        "persistent_condition",
        "state_transition",
        "missing_or_frozen_measurement",
    }
    covered = set().union(*(item.coverage for item in selected))
    if not required <= covered:
        raise AssertionError(f"3W subset misses criteria: {sorted(required - covered)}")
    report = adapter.materialise(
        source,
        core,
        evaluation,
        fixture_destination=fixture,
    )
    fit_report = json.loads(
        (run_root / "threew_contract_fit_report.json").read_text(encoding="utf-8")
    )
    workflow_report = {
        "workflow": "petrobras_3w_contract_challenge",
        "runtime": runtime_provenance(),
        "contract": contract_summary(),
        "source": str(source),
        "run_root": str(run_root),
        "selected": [
            {
                "path": item.relative_path,
                "entity_id": item.entity_id,
                "coverage": sorted(item.coverage),
                "rows": item.rows,
                "bytes": item.bytes,
            }
            for item in selected
        ],
        "all_required_criteria_covered": True,
        "row_counts": dict(report.row_counts),
        "contract_fit_report": fit_report,
    }
    _write_json(run_root / "workflow_report.json", workflow_report)
    return workflow_report
