"""Adapter for the labelled optical-failure testbed dataset.

The source's ``Failure`` column is read only when PACK-EVAL is created.  It is
never copied into PACK-CORE.  The flag identifies a controlled testbed failure
window, not a production fault prevalence or an exact failed component, so the
pack is suitable for score-response and event-detection tests, not localisation
accuracy claims.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from telco_anomaly.contract import EVAL_SCHEMAS, PACK_SCHEMAS, save_pack, source_file

from ._tabular import table_batches, table_columns


SOURCE_URL = "https://github.com/Network-And-Services/optical-failure-dataset"
SOURCE_SYSTEM = "santanna_optical_failure_testbed"
CADENCE_SECONDS = 3.5
FILE_NAMES = ("HardFailure_dataset.csv", "SoftFailure_dataset.csv")

METRICS = (
    ("BER", "optical.ber", "optical_terminal", "bounded_fraction", "ratio", "high_bad", "hurdle_log10", 1e-3),
    ("OSNR", "optical.osnr", "optical_terminal", "gauge", "dB", "low_bad", "identity", 0.05),
    ("InputPower", "optical.input_power", "optical_amplifier", "gauge", "dBm", "low_bad", "identity", 0.05),
    ("OutputPower", "optical.output_power", "optical_amplifier", "gauge", "dBm", "two_sided", "identity", 0.05),
)


def _source_files(source):
    source = Path(source)
    files = [source / name for name in FILE_NAMES if (source / name).is_file()]
    if not files:
        raise FileNotFoundError(
            f"Expected at least one of {list(FILE_NAMES)} under {source}"
        )
    return files


def inspect_optical_failure(source):
    """Validate documented columns and report whether labels are present."""

    files = _source_files(source)
    required = {"Timestamp", "Type", "ID", *(item[0] for item in METRICS)}
    rows = []
    for path in files:
        columns = table_columns(path)
        missing = sorted(required - set(columns))
        if missing:
            raise ValueError(f"{path.name} is missing documented fields: {missing}")
        rows.append({
            "file": path.name,
            "columns": columns,
            "failure_label_present": "Failure" in columns,
        })
    return {
        "files": rows,
        "cadence_seconds": CADENCE_SECONDS,
        "label_scope": "controlled_testbed_failure_window",
        "localisation_truth": False,
    }


def _timestamps(values):
    numeric = pd.to_numeric(values, errors="coerce")
    timestamps = pd.to_datetime(numeric, unit="s", utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Optical-failure Timestamp must contain Unix seconds")
    return timestamps


def _entity_type(values):
    normalised = values.astype("string").str.strip().str.lower()
    mapped = normalised.map({
        "devices": "optical_terminal",
        "device": "optical_terminal",
        "infrastructure": "optical_amplifier",
    })
    if mapped.isna().any():
        unknown = sorted(normalised.loc[mapped.isna()].dropna().unique())
        raise ValueError(f"Unknown optical-failure Type values: {unknown}")
    return mapped


def _scan(source, *, batch_rows):
    files = _source_files(source)
    entity_types = {}
    entity_bounds = {}
    episodes = {}
    available_metrics = set()
    failure_present = {}

    for path in files:
        columns = table_columns(path)
        required = {"Timestamp", "Type", "ID", *(item[0] for item in METRICS)}
        missing = sorted(required - set(columns))
        if missing:
            raise ValueError(f"{path.name} is missing documented fields: {missing}")
        failure_present[path] = "Failure" in columns
        for frame in table_batches(path, list(required), batch_rows=batch_rows):
            frame = frame.reset_index(drop=True)
            times = _timestamps(frame["Timestamp"])
            ids = frame["ID"].astype("string")
            types = _entity_type(frame["Type"])
            if ids.isna().any():
                raise ValueError(f"{path.name} contains null device identifiers")
            for entity_id, entity_type in zip(ids, types):
                previous = entity_types.setdefault(str(entity_id), str(entity_type))
                if previous != str(entity_type):
                    raise ValueError(f"Entity {entity_id} changes Type across the source")
            keys = pd.DataFrame({"entity_id": ids, "event_ts": times})
            for entity_id, group in keys.groupby("entity_id", sort=False):
                entity_times = group["event_ts"]
                low, high = entity_times.min(), entity_times.max()
                previous = entity_bounds.get(str(entity_id))
                entity_bounds[str(entity_id)] = (
                    min(low, previous[0]) if previous else low,
                    max(high, previous[1]) if previous else high,
                )
                episode_id = f"{path.stem}::{entity_id}"
                episodes[episode_id] = str(entity_id)
            for native, metric_id, entity_type, *_ in METRICS:
                intended = types.eq(entity_type)
                if pd.to_numeric(frame.loc[intended, native], errors="coerce").notna().any():
                    available_metrics.add(metric_id)
    return {
        "files": files,
        "entity_types": entity_types,
        "entity_bounds": entity_bounds,
        "episodes": episodes,
        "available_metrics": available_metrics,
        "failure_present": failure_present,
    }


def _catalogue(available_metrics):
    rows = []
    for native, metric_id, entity_type, kind, unit, direction, transform, scale in METRICS:
        if metric_id not in available_metrics:
            continue
        rows.append({
            "metric_id": metric_id,
            "entity_type": entity_type,
            "measurement_kind": kind,
            "unit": unit,
            "sampling_mode": "periodic",
            "aggregation_semantics": "instantaneous",
            "expected_cadence_seconds": CADENCE_SECONDS,
            "direction": direction,
            "transform": transform,
            "valid_min": 0 if metric_id == "optical.ber" else None,
            "valid_max": 1 if metric_id == "optical.ber" else None,
            "censoring_type": "none",
            "reporting_lod": None,
            "reset_policy": "not_applicable",
            "counter_modulus": None,
            "minimum_scale": scale,
            "seasonality_candidate": False,
            "peer_eligible": True,
        })
    return pd.DataFrame(rows, columns=PACK_SCHEMAS["metric_catalogue"])


def _telemetry(source, scan, *, batch_rows):
    specs = {
        entity_type: [item for item in METRICS if item[2] == entity_type and item[1] in scan["available_metrics"]]
        for entity_type in {item[2] for item in METRICS}
    }
    columns = ["Timestamp", "Type", "ID", *(item[0] for item in METRICS)]
    for path in scan["files"]:
        for frame in table_batches(path, columns, batch_rows=batch_rows):
            types = _entity_type(frame["Type"])
            pieces = []
            for entity_type, metrics in specs.items():
                selected = frame.loc[types.eq(entity_type)].copy()
                if selected.empty:
                    continue
                long = selected.melt(
                    id_vars=["Timestamp", "ID"],
                    value_vars=[item[0] for item in metrics],
                    var_name="native_field",
                    value_name="value",
                )
                native_to_metric = {item[0]: item[1] for item in metrics}
                long["metric_id"] = long["native_field"].map(native_to_metric)
                pieces.append(long)
            if not pieces:
                continue
            long = pd.concat(pieces, ignore_index=True)
            long["event_ts"] = _timestamps(long["Timestamp"])
            long["entity_id"] = long["ID"].astype("string")
            long["episode_id"] = path.stem + "::" + long["entity_id"]
            long["value"] = pd.to_numeric(long["value"], errors="coerce")
            finite = np.isfinite(long["value"].to_numpy(dtype=float, na_value=np.nan))
            long["quality_code"] = np.where(finite, "measured", "invalid")
            long["source_system"] = SOURCE_SYSTEM
            long["ingestion_ts"] = pd.NaT
            yield long[PACK_SCHEMAS["telemetry"]]


def _failure_mask(values):
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    numeric = pd.to_numeric(values, errors="coerce")
    nonempty = values.notna() & values.astype("string").str.strip().ne("")
    if numeric.loc[nonempty].notna().all():
        return numeric.fillna(0).ne(0)
    text = values.astype("string").str.strip().str.lower()
    accepted = {
        "0": False, "false": False, "no": False, "normal": False,
        "1": True, "true": True, "yes": True, "failure": True, "fail": True,
    }
    unknown = sorted(set(text.dropna()) - set(accepted))
    if unknown:
        raise ValueError(f"Unknown Failure label values: {unknown}")
    return text.map(accepted).fillna(False).astype(bool)


def _truth(scan, *, batch_rows):
    events = []
    intervals = []
    cadence = pd.Timedelta(seconds=CADENCE_SECONDS)
    for path in scan["files"]:
        if not scan["failure_present"][path]:
            raise ValueError(
                f"Evaluation requested but {path.name} has no Failure field"
            )
        chunks = []
        for frame in table_batches(
            path, ["Timestamp", "ID", "Failure"], batch_rows=batch_rows
        ):
            chunks.append(pd.DataFrame({
                "event_ts": _timestamps(frame["Timestamp"]),
                "entity_id": frame["ID"].astype("string"),
                "active": _failure_mask(frame["Failure"]),
            }))
        labels = pd.concat(chunks, ignore_index=True)
        agreement = labels.groupby("event_ts")["active"].nunique()
        if agreement.gt(1).any():
            raise ValueError(f"Failure flags disagree across devices in {path.name}")
        timeline = labels.groupby("event_ts", as_index=False)["active"].first().sort_values("event_ts")
        changed = timeline["active"].ne(timeline["active"].shift())
        broken = timeline["event_ts"].diff().gt(cadence * 1.5)
        timeline["run"] = (changed | broken).cumsum()
        fault_number = 0
        for _, run in timeline.groupby("run", sort=True):
            if not bool(run["active"].iloc[0]):
                continue
            start = run["event_ts"].iloc[0]
            last = run["event_ts"].iloc[-1]
            next_rows = timeline.loc[timeline["event_ts"].gt(last), "event_ts"]
            end = next_rows.iloc[0] if len(next_rows) and next_rows.iloc[0] - last <= cadence * 1.5 else last + cadence
            fault_id = f"OPTICAL-{path.stem}-{fault_number:04d}"
            fault_type = "hard_optical_failure" if "hard" in path.name.lower() else "soft_optical_failure"
            events.append({
                "fault_id": fault_id,
                "fault_type": fault_type,
                "domain_type": "testbed_network",
                "domain_id": path.stem,
                "onset_ts": start,
                "observable_ts": start,
                "impact_ts": start,
                "end_ts": end,
                "group_id": pd.NA,
                "label_source": "controlled_failure_flag",
                "source_instance_id": path.stem,
            })
            affected = labels.loc[
                labels["event_ts"].ge(start) & labels["event_ts"].lt(end),
                "entity_id",
            ].dropna().unique()
            intervals.extend({
                "fault_id": fault_id,
                "entity_id": str(entity_id),
                "start_ts": start,
                "end_ts": end,
                "label_source": "recording_scope_not_localisation_truth",
                "source_instance_id": path.stem,
            } for entity_id in affected)
            fault_number += 1
    return {
        "fault_events": pd.DataFrame(events, columns=EVAL_SCHEMAS["fault_events"]),
        "fault_entity_intervals": pd.DataFrame(
            intervals, columns=EVAL_SCHEMAS["fault_entity_intervals"]
        ),
    }


def build_optical_failure_pack(
    source,
    destination,
    *,
    include_evaluation=True,
    pack_version="optical_failure_v1",
    batch_rows=100_000,
):
    """Build testbed telemetry and, optionally, isolated failure intervals."""

    source = Path(source)
    scan = _scan(source, batch_rows=batch_rows)
    catalogue = _catalogue(scan["available_metrics"])
    entities = pd.DataFrame([
        {
            "entity_id": entity,
            "entity_type": scan["entity_types"][entity],
            "vendor": pd.NA,
            "model": pd.NA,
            "source_system": SOURCE_SYSTEM,
            "valid_from": bounds[0],
            "valid_to": pd.NaT,
        }
        for entity, bounds in sorted(scan["entity_bounds"].items())
    ], columns=PACK_SCHEMAS["entity_registry"])
    episodes = pd.DataFrame([
        {
            "episode_id": episode,
            "entity_id": entity,
            "episode_basis": "separate_controlled_testbed_recording",
        }
        for episode, entity in sorted(scan["episodes"].items())
    ], columns=PACK_SCHEMAS["observation_episodes"])
    evaluation = _truth(scan, batch_rows=batch_rows) if include_evaluation else {}
    source_info = {
        "source_id": "santanna_optical_failure_testbed",
        "source_url": SOURCE_URL,
        "licence": "No licence file verified in the source repository",
        "licence_note": (
            "Confirm permission and citation requirements before external use; "
            "the project does not redistribute source data."
        ),
        "evidence_role": "controlled_physical_failure_score_response",
        "redistributed_by_project": False,
        "files": [
            source_file(path, source, "mixed_observable_and_evaluation_source")
            for path in scan["files"]
        ],
    }
    return save_pack(
        destination,
        sector="telecom",
        pack_version=pack_version,
        source_info=source_info,
        telemetry=_telemetry(source, scan, batch_rows=batch_rows),
        catalogue=catalogue,
        entities=entities,
        episodes=episodes,
        evaluation=evaluation,
        notes=(
            "Failure labels are isolated in PACK-EVAL.",
            "Failure intervals are recording-level evidence, not component localisation truth.",
            "Controlled testbed prevalence must not be presented as production prevalence.",
        ),
    )


__all__ = [
    "SOURCE_URL", "inspect_optical_failure", "build_optical_failure_pack"
]
