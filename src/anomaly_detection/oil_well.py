"""Petrobras 3W 2.0.0 adapter used as the non-telecom contract challenge."""

from __future__ import annotations

import configparser
import hashlib
import itertools
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .core import (
    CONTRACT_VERSION,
    DatasetInventory,
    MaterialisationReport,
    canonical_frame_hash,
    sha256_file,
    validate_core_bundle,
)
from .evaluation import EVAL_CONTRACT_VERSION, validate_eval_bundle
from .packs import NATIVE_METRIC_FIELDS, OilWellPack, load_oil_well_pack

NATIVE_FORMAT = "Petrobras 3W Dataset"
EXPECTED_VERSION = "2.0.0"
EXPECTED_INSTANCE_COUNT = 2228
TRANSIENT_EVENT_CODES = {1, 2, 5, 6, 7, 8, 9}
PERSISTENT_EVENT_CODES = {3, 4}
# Long-format widening multiplies each native row by the metric count. Keep the
# physical batch deliberately small so hashing and Parquet encoding stay bounded.
NATIVE_TELEMETRY_BATCH_ROWS = 5_000


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


def _pandas():
    import pandas as pd

    return pd


def _source_kind(filename: str) -> str:
    if filename.startswith("WELL-"):
        return "real"
    if filename.startswith("SIMULATED_"):
        return "simulated"
    if filename.startswith("DRAWN_"):
        return "hand_drawn"
    raise ValueError(f"unrecognised 3W filename: {filename}")


def _entity_id(filename: str) -> str:
    if filename.startswith("WELL-"):
        return filename.split("_", 1)[0]
    return Path(filename).stem


def _empty(columns: Iterable[str]):
    return _pandas().DataFrame(columns=list(columns))


def _contiguous_runs(values) -> list[tuple[int, int, Any]]:
    """Return inclusive start/exclusive end runs, retaining missing as ``unknown``."""

    pd = _pandas()
    normalised = values.astype("object").where(values.notna(), "unknown").tolist()
    if not normalised:
        return []
    runs: list[tuple[int, int, Any]] = []
    start = 0
    current = normalised[0]
    for index, value in enumerate(normalised[1:], start=1):
        equal = (pd.isna(value) and pd.isna(current)) or value == current
        if not equal:
            runs.append((start, index, current))
            start, current = index, value
    runs.append((start, len(normalised), current))
    return runs


class ThreeWAdapter:
    adapter_id = "petrobras-threew-v2"
    adapter_version = "0.1.0"
    supported_source_formats = (NATIVE_FORMAT,)
    source_id = "petrobras-3w-2.0.0"

    def __init__(self, pack: OilWellPack | None = None):
        self.pack = pack or load_oil_well_pack()

    @staticmethod
    def _config(source: Path) -> configparser.ConfigParser:
        parser = configparser.ConfigParser()
        path = Path(source) / "dataset.ini"
        if not path.is_file():
            raise FileNotFoundError(f"missing 3W configuration: {path}")
        parser.read(path, encoding="utf-8")
        return parser

    def discover(self, source: Path) -> DatasetInventory:
        root = Path(source)
        parser = self._config(root)
        version = parser.get("VERSION", "DATASET")
        files = tuple(
            sorted(str(path.relative_to(root)) for path in root.glob("*/*.parquet"))
        )
        missing_dirs = [str(code) for code in range(10) if not (root / str(code)).is_dir()]
        notes = []
        if version != EXPECTED_VERSION:
            notes.append(f"expected version {EXPECTED_VERSION}, found {version}")
        if len(files) != EXPECTED_INSTANCE_COUNT:
            notes.append(
                f"expected {EXPECTED_INSTANCE_COUNT} Parquet instances, found {len(files)}"
            )
        if missing_dirs:
            notes.append(f"missing event directories: {missing_dirs}")
        ready = not notes
        return DatasetInventory(
            source_format=NATIVE_FORMAT,
            source_version=version,
            tables=files,
            core_ready=ready,
            evaluation_ready=ready,
            notes=tuple(notes),
        )

    @staticmethod
    def _metadata_summary(path: Path, root: Path) -> InstanceSummary:
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        names = parquet.schema_arrow.names
        state_index = names.index("state")
        measurement_indexes = [names.index(field) for field in NATIVE_METRIC_FIELDS]
        state_min = None
        state_max = None
        missing = False
        for group_index in range(parquet.metadata.num_row_groups):
            row_group = parquet.metadata.row_group(group_index)
            state_stats = row_group.column(state_index).statistics
            if state_stats is not None and state_stats.has_min_max:
                state_min = (
                    state_stats.min
                    if state_min is None
                    else min(state_min, state_stats.min)
                )
                state_max = (
                    state_stats.max
                    if state_max is None
                    else max(state_max, state_stats.max)
                )
            for column_index in measurement_indexes:
                stats = row_group.column(column_index).statistics
                if stats is not None and (stats.null_count or 0) > 0:
                    missing = True
                    break
            if missing and state_min is not None and state_max is not None:
                continue
        event_code = int(path.parent.name)
        source_kind = _source_kind(path.name)
        coverage: set[str] = set()
        if event_code == 0:
            coverage.add("normal_instance")
        if event_code in TRANSIENT_EVENT_CODES:
            coverage.add("transient_event")
        if event_code in PERSISTENT_EVENT_CODES:
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
            entity_id=_entity_id(path.name),
            event_code=event_code,
            source_kind=source_kind,
            rows=parquet.metadata.num_rows,
            bytes=path.stat().st_size,
            has_state_transition=state_transition,
            has_missing_measurement=missing,
            coverage=frozenset(coverage),
        )

    def candidate_summaries(self, source: Path) -> list[InstanceSummary]:
        """Build a small deterministic candidate pool using Parquet metadata only."""

        root = Path(source)
        candidates: list[Path] = []
        for code in range(10):
            real_files = sorted((root / str(code)).glob("WELL-*.parquet"))
            candidates.extend(real_files[:8])
        summaries = [self._metadata_summary(path, root) for path in candidates]
        return sorted(summaries, key=lambda item: item.relative_path)

    def select_minimal_subset(self, source: Path) -> list[InstanceSummary]:
        required = {
            "normal_instance",
            "transient_event",
            "persistent_condition",
            "state_transition",
            "missing_or_frozen_measurement",
        }
        candidates = self.candidate_summaries(source)
        for size in range(1, 8):
            valid: list[tuple[int, tuple[str, ...], tuple[InstanceSummary, ...]]] = []
            for combination in itertools.combinations(candidates, size):
                if (
                    len({item.entity_id for item in combination}) < 3
                    or len({item.entity_id for item in combination}) != size
                ):
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
                _, _, selected = min(valid, key=lambda item: (item[0], item[1]))
                return list(selected)
        missing = sorted(
            required - set().union(*(item.coverage for item in candidates))
        )
        raise ValueError(
            "no deterministic 3W subset satisfies the contract challenge; "
            f"uncovered criteria: {missing}"
        )

    def metric_catalogue(self):
        pd = _pandas()
        rows = []
        for metric in self.pack.metric_specs().values():
            row = asdict(metric)
            row.pop("native_field", None)
            row["measurement_kind"] = metric.measurement_kind.value
            row["anomaly_direction"] = metric.anomaly_direction.value
            row["candidate_periods"] = list(metric.candidate_periods)
            row["context_keys"] = list(metric.context_keys)
            rows.append(row)
        return pd.DataFrame(rows).sort_values("metric_id").reset_index(drop=True)

    def _read_instance(self, summary: InstanceSummary):
        pd = _pandas()
        frame = pd.read_parquet(summary.path)
        frame = frame.reset_index()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        return frame

    def _telemetry(self, frame, summary: InstanceSummary):
        pd = _pandas()
        mapping = {
            metric.native_field: metric_id
            for metric_id, metric in self.pack.metric_specs().items()
        }
        long = frame[["timestamp", *mapping]].melt(
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
        long["exposure"] = float("nan")
        long["source_id"] = self.source_id
        return long[
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
        ]

    @staticmethod
    def _instance_gap_rows(frame, summary: InstanceSummary):
        pd = _pandas()
        timestamps = frame["timestamp"].drop_duplicates().sort_values()
        deltas = timestamps.diff()
        rows = []
        for index in deltas[deltas > pd.Timedelta(seconds=1)].index:
            previous_position = timestamps.index.get_loc(index) - 1
            previous = timestamps.iloc[previous_position]
            current = timestamps.loc[index]
            rows.append(
                {
                    "entity_id": summary.entity_id,
                    "gap_start": previous + pd.Timedelta(seconds=1),
                    "gap_end": current,
                    "known_at": current,
                    "source": "petrobras-3w-observed-timestamps",
                }
            )
        return rows

    @staticmethod
    def _condition_rows(frame, summary: InstanceSummary):
        pd = _pandas()
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

    @staticmethod
    def _event_rows(frame, summary: InstanceSummary, event_descriptions):
        pd = _pandas()
        timestamps = frame["timestamp"].reset_index(drop=True)
        labels = frame["class"].reset_index(drop=True)
        event_code = summary.event_code
        if event_code == 0:
            return [], []
        relevant = labels.map(
            lambda value: (
                False
                if pd.isna(value)
                else int(value) in {event_code, 100 + event_code}
            )
        )
        runs = _contiguous_runs(relevant)
        events, intervals = [], []
        event_index = 0
        for start, end, active in runs:
            if not bool(active):
                continue
            segment = labels.iloc[start:end]
            onset = timestamps.iloc[start]
            steady = segment.index[segment.eq(event_code)]
            impact = timestamps.iloc[int(steady[0])] if len(steady) else pd.NaT
            resolution = timestamps.iloc[end] if end < len(timestamps) else pd.NaT
            event_id = (
                f"3W-{summary.instance_id}-{event_code}-{event_index:02d}"
            )
            events.append(
                {
                    "fault_event_id": event_id,
                    "fault_type": event_descriptions[event_code],
                    "fault_family": f"3w_event_{event_code}",
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
                    "fault_family": f"3w_event_{event_code}",
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

    def _event_descriptions(self, source: Path) -> dict[int, str]:
        parser = self._config(source)
        descriptions = {}
        names = [
            item.strip()
            for item in parser.get("EVENTS", "NAMES").replace("\n", "").split(",")
        ]
        for name in names:
            descriptions[parser.getint(name, "LABEL")] = parser.get(
                name, "DESCRIPTION"
            )
        return descriptions

    def materialise(
        self,
        source: Path,
        core_destination: Path,
        evaluation_destination: Path | None = None,
        *,
        fixture_destination: Path | None = None,
    ) -> MaterialisationReport:
        pd = _pandas()
        source = Path(source)
        core_destination = Path(core_destination)
        evaluation_destination = (
            None if evaluation_destination is None else Path(evaluation_destination)
        )
        inventory = self.discover(source)
        if not inventory.core_ready:
            raise ValueError("; ".join(inventory.notes))
        if core_destination.exists():
            raise FileExistsError(f"refusing to overwrite: {core_destination}")
        if evaluation_destination is not None and evaluation_destination.exists():
            raise FileExistsError(f"refusing to overwrite: {evaluation_destination}")
        selected = self.select_minimal_subset(source)

        if fixture_destination is not None:
            fixture_destination = Path(fixture_destination)
            if fixture_destination.exists():
                raise FileExistsError(f"refusing to overwrite: {fixture_destination}")
            fixture_destination.mkdir(parents=True)
            for metadata_name in ("dataset.ini", "LICENSE-CC-BY", "README.md"):
                shutil.copy2(source / metadata_name, fixture_destination / metadata_name)
            for summary in selected:
                destination = fixture_destination / summary.relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(summary.path, destination)

        core_destination.mkdir(parents=True)
        telemetry_dir = core_destination / "telemetry"
        telemetry_dir.mkdir()
        registry_rows = []
        gap_rows = []
        event_rows = []
        interval_rows = []
        condition_rows = []
        telemetry_hash = hashlib.sha256()
        telemetry_count = 0
        telemetry_part_index = 0
        event_descriptions = self._event_descriptions(source)

        for summary in selected:
            frame = self._read_instance(summary)
            for batch_start in range(0, len(frame), NATIVE_TELEMETRY_BATCH_ROWS):
                native_batch = frame.iloc[
                    batch_start : batch_start + NATIVE_TELEMETRY_BATCH_ROWS
                ]
                telemetry = self._telemetry(native_batch, summary)
                telemetry.to_parquet(
                    telemetry_dir / f"part-{telemetry_part_index:05d}.parquet",
                    index=False,
                    compression="zstd",
                )
                telemetry_hash.update(
                    canonical_frame_hash(
                        telemetry, sort_by=["event_ts", "entity_id", "metric_id"]
                    ).encode("ascii")
                )
                telemetry_count += len(telemetry)
                telemetry_part_index += 1
                del telemetry, native_batch
            attributes = json.dumps(
                {
                    "source_kind": summary.source_kind,
                    "source_instance_id": summary.instance_id,
                    "event_code": summary.event_code,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            registry_rows.append(
                {
                    "entity_id": summary.entity_id,
                    "entity_type": "oil_well",
                    "valid_from": frame["timestamp"].min(),
                    "valid_to": frame["timestamp"].max() + pd.Timedelta(seconds=1),
                    "attributes_json": attributes,
                    "source_id": self.source_id,
                }
            )
            gap_rows.extend(self._instance_gap_rows(frame, summary))
            conditions = self._condition_rows(frame, summary)
            condition_rows.extend(conditions)
            events, intervals = self._event_rows(
                frame, summary, event_descriptions
            )
            event_rows.extend(events)
            interval_rows.extend(intervals)
            del frame

        metric_catalogue = self.metric_catalogue()
        entity_registry = (
            pd.DataFrame(registry_rows)
            .sort_values("entity_id")
            .reset_index(drop=True)
        )
        entity_relations = _empty(
            [
                "parent_entity_id",
                "child_entity_id",
                "relation_type",
                "relation_family",
                "valid_from",
                "valid_to",
                "relation_confidence",
                "source",
            ]
        )
        operational_events = _empty(
            [
                "event_id",
                "entity_id",
                "event_type",
                "event_start",
                "event_end",
                "known_at",
                "attributes_json",
                "source",
            ]
        )
        collection_gaps = pd.DataFrame(
            gap_rows,
            columns=["entity_id", "gap_start", "gap_end", "known_at", "source"],
        )
        sidecars = {
            "metric_catalogue": metric_catalogue,
            "entity_registry": entity_registry,
            "entity_relations": entity_relations,
            "operational_events": operational_events,
            "collection_gaps": collection_gaps,
        }
        validation_view = {
            "telemetry": _empty(
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
            raise ValueError("3W SPEC-CORE adaptation failed: " + "; ".join(errors))
        for name, frame in sidecars.items():
            frame.to_parquet(core_destination / f"{name}.parquet", index=False)

        cause_groups = _empty(
            ["cause_group_id", "cause_type", "start_ts", "end_ts", "footprint_json"]
        )
        fault_events = pd.DataFrame(
            event_rows,
            columns=[
                "fault_event_id",
                "fault_type",
                "fault_family",
                "fault_domain_type",
                "fault_domain_id",
                "onset_ts",
                "first_observable_ts",
                "impact_ts",
                "resolution_ts",
                "cause_group_id",
                "left_censored",
                "label_source",
                "source_instance_id",
            ],
        )
        fault_intervals = pd.DataFrame(
            interval_rows,
            columns=[
                "fault_event_id",
                "affected_entity_id",
                "fault_family",
                "channel",
                "active_start_ts",
                "active_end_ts",
                "impact_ts",
                "contribution",
            ],
        )
        condition_states = pd.DataFrame(
            condition_rows,
            columns=[
                "entity_id",
                "condition_start_ts",
                "condition_end_ts",
                "condition_code",
                "condition_label",
                "label_source",
                "source_instance_id",
            ],
        )
        evaluation = {
            "gt_fault_events": fault_events,
            "gt_fault_entity_intervals": fault_intervals,
            "gt_cause_groups": cause_groups,
            "gt_condition_states": condition_states,
        }
        errors = validate_eval_bundle(evaluation)
        if errors:
            raise ValueError("3W SPEC-EVAL adaptation failed: " + "; ".join(errors))
        if evaluation_destination is not None:
            evaluation_destination.mkdir(parents=True)
            for name, frame in evaluation.items():
                frame.to_parquet(
                    evaluation_destination / f"{name}.parquet", index=False
                )

        selected_manifest = {
            "dataset": "Petrobras 3W",
            "dataset_version": EXPECTED_VERSION,
            "selection_rule": "smallest deterministic metadata candidate subset satisfying all criteria",
            "criteria": [
                "normal_instance",
                "transient_event",
                "persistent_condition",
                "state_transition",
                "missing_or_frozen_measurement",
                "at_least_three_distinct_real_wells",
            ],
            "selected": [
                {
                    **{
                        key: value
                        for key, value in asdict(summary).items()
                        if key not in {"path", "coverage"}
                    },
                    "coverage": sorted(summary.coverage),
                    "sha256": sha256_file(summary.path),
                }
                for summary in selected
            ],
        }
        if fixture_destination is not None:
            (fixture_destination / "SUBSET_MANIFEST.json").write_text(
                json.dumps(selected_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        core_counts = {
            "telemetry": telemetry_count,
            **{name: int(len(frame)) for name, frame in sidecars.items()},
        }
        core_hashes = {
            "telemetry": telemetry_hash.hexdigest(),
            **{
                name: canonical_frame_hash(frame)
                for name, frame in sidecars.items()
            },
        }
        (core_destination / "manifest.json").write_text(
            json.dumps(
                {
                    "contract_version": CONTRACT_VERSION,
                    "pack_version": self.pack.pack_version,
                    "adapter_version": self.adapter_version,
                    "row_counts": core_counts,
                    "canonical_content_hashes": core_hashes,
                    "physical_telemetry_batch_native_rows":
                        NATIVE_TELEMETRY_BATCH_ROWS,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if evaluation_destination is not None:
            (evaluation_destination / "manifest.json").write_text(
                json.dumps(
                    {
                        "contract_version": EVAL_CONTRACT_VERSION,
                        "row_counts": {
                            name: int(len(frame))
                            for name, frame in evaluation.items()
                        },
                        "canonical_content_hashes": {
                            name: canonical_frame_hash(frame)
                            for name, frame in evaluation.items()
                        },
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        report = self.contract_fit_report(source, selected, condition_states)
        (core_destination.parent / "threew_contract_fit_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        row_counts = {f"core.{name}": count for name, count in core_counts.items()}
        row_counts.update(
            {f"eval.{name}": int(len(frame)) for name, frame in evaluation.items()}
        )
        return MaterialisationReport(
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            source_format=NATIVE_FORMAT,
            core_path=core_destination,
            evaluation_path=evaluation_destination,
            row_counts=row_counts,
            truth_columns_removed=("class", "state"),
        )

    @staticmethod
    def contract_fit_report(source, selected, condition_states):
        return {
            "dataset": "Petrobras 3W 2.0.0",
            "source_root": str(source),
            "selected_instances": [item.relative_path for item in selected],
            "represented_cleanly": [
                "multiple oil-well entities",
                "one-second nullable multivariate telemetry",
                "event class intervals",
                "operational state intervals via gt_condition_states",
                "source instance and real/simulated/hand-drawn provenance vocabulary",
            ],
            "not_representable_without_invention": [
                "shared manifold topology",
                "relationships between masked wells",
                "shared cause groups",
                "maintenance ticket links",
                "graded condition severity",
            ],
            "distortions_or_qualifications": [
                "state codes are retained as source_state_<code> because dataset.ini does not name them",
                "class-label interval start is used as event onset and first observable time",
                "resolution exists only when the label returns to a non-event state inside the instance",
                "each selected real well contributes one source instance so one registry validity interval remains honest",
            ],
            "information_deliberately_not_translated": [
                "class and state labels into SPEC-CORE",
                "manifold or cross-well relations",
                "severity ordinal",
            ],
            "generic_contract_changes_required": [
                "gt_condition_states added; severity field intentionally absent"
            ],
            "condition_state_rows": int(len(condition_states)),
            "telecom_shaped_assumptions_found": [
                "event-only evaluation truth was insufficient",
                "a relation table must be allowed to be empty",
                "entity validity must not imply observations outside a source instance",
            ],
        }
