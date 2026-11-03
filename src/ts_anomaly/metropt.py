"""UCI MetroPT-3 (2020): source reading, documented signals and incident records.

Definitions: https://doi.org/10.24432/C5VW3R and the accompanying data dictionary.
The CSV spellings are retained. The 2022 MetroPT release is a different dataset.
"""

from hashlib import file_digest
from pathlib import Path
import shutil
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

from .contracts import MetricDefinition, validate_schema


METRICS = {
    "TP2": MetricDefinition("Compressor pressure", "bar", "gauge", "measurement"),
    "TP3": MetricDefinition("Pneumatic-panel pressure", "bar", "gauge", "measurement"),
    "H1": MetricDefinition("Separator-related pressure", "bar", "gauge", "measurement"),
    "DV_pressure": MetricDefinition("Dryer pressure drop", "bar", "gauge", "measurement",
                                    note="Source associates near-zero readings with loading; verify empirically."),
    "Reservoirs": MetricDefinition("Reservoir pressure", "bar", "gauge", "measurement"),
    "Oil_temperature": MetricDefinition("Compressor oil temperature", "degC", "gauge", "measurement"),
    "Motor_current": MetricDefinition("One motor phase current", "A", "gauge", "measurement",
                                      note="Approximate operating levels in the source are not hard limits."),
    "COMP": MetricDefinition("Intake-valve control", None, "state", "context", (0, 1),
                             "Active means no air intake: off or unloaded."),
    "DV_eletric": MetricDefinition("Outlet-valve control", None, "state", "context", (0, 1),
                                   "Source: active during loading. CSV spelling retained."),
    "Towers": MetricDefinition("Drying-tower selection", None, "state", "context", (0, 1)),
    "MPG": MetricDefinition("Compressor loading control", None, "state", "context", (0, 1),
                            "Source says behaviour follows COMP; compare rather than assume equivalence."),
    "LPS": MetricDefinition("Low-pressure indicator", None, "state", "existing_alarm", (0, 1),
                            "Source describes activation below 7 bar; not independent ground truth."),
    "Pressure_switch": MetricDefinition("Dryer discharge indicator", None, "state", "context", (0, 1)),
    "Oil_level": MetricDefinition("Oil-level indicator", None, "state", "unresolved", (0, 1),
                                  "Source says active means low oil; encoding/polarity requires confirmation."),
    "Caudal_impulses": MetricDefinition("Air-flow pulse signal", None, "pulse", "unresolved",
                                        note=("Pulse encoding and volume per pulse are unspecified; "
                                              "do not sum or difference as a counter.")),
}


def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def fetch_metropt(source: dict, cache: Path, *, download=True) -> Path:
    """Cache the two official files and verify their pinned content on every use."""
    cache = Path(cache)
    resources = source["files"]
    if any(not (cache / name).exists() for name in resources):
        archive = cache / "metropt.zip"
        if not archive.exists():
            if not download:
                raise FileNotFoundError("MetroPT files are missing; run once with download enabled")
            cache.mkdir(parents=True, exist_ok=True)
            partial = cache / "metropt.download"
            with urlopen(source["url"], timeout=1200) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output)
            if sha256_file(partial) != source["archive_sha256"]:
                raise ValueError("Archive checksum mismatch; review the source version")
            partial.rename(archive)
        if sha256_file(archive) != source["archive_sha256"]:
            raise ValueError("Archive checksum mismatch")
        with ZipFile(archive) as bundle:
            for name in resources:
                target = cache / name
                if not target.exists():
                    with bundle.open(name) as input_file, target.open("wb") as output:
                        shutil.copyfileobj(input_file, output)
    for name, expected in resources.items():
        if sha256_file(cache / name) != expected:
            raise ValueError(f"Checksum mismatch: {name}; cached content was not overwritten")
    return cache / "MetroPT3(AirCompressor).csv"


def read_metropt(path: Path, *, before) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return development values and full-timeline availability, never test values.

    The full file is read for structural checks. After the boundary, only times,
    source indices and missing-cell counts leave this function. No UTC is assumed.
    """
    native = pd.read_csv(path)
    if list(native.columns) != ["Unnamed: 0", "timestamp", *METRICS]:
        raise ValueError("Unexpected MetroPT-3 columns; check the release")
    times = pd.to_datetime(native.timestamp, format="%Y-%m-%d %H:%M:%S", errors="raise")
    if times.isna().any() or times.dt.tz is not None:
        raise ValueError("Expected nonmissing, timezone-unspecified recorded times")
    timeline = pd.DataFrame({
        "timestamp": times, "entity_id": "metropt_apu_1",
        "source_index": native["Unnamed: 0"],
        "missing_measurements": native[list(METRICS)].isna().sum(axis=1),
    })
    selected = times.lt(pd.Timestamp(before))
    data = native.loc[selected, list(METRICS)].copy()
    data.insert(0, "entity_id", "metropt_apu_1")
    data.insert(0, "timestamp", times[selected])
    validate_schema(data, METRICS)
    return data.reset_index(drop=True), timeline


def operating_state(data: pd.DataFrame) -> pd.Series:
    """Documented valve-command combinations, not inferred physical machine states.

    Off and unloaded are deliberately combined. Current is not used to define
    states, so comparisons of current across valve commands are not circular.
    """
    state = pd.Series("unresolved", index=data.index, name="operating_state")
    state.loc[data.COMP.eq(0) & data.DV_eletric.eq(1)] = "load_command"
    state.loc[data.COMP.eq(1) & data.DV_eletric.eq(0)] = "off_or_unloaded_command"
    return state


def incident_records() -> pd.DataFrame:
    """Published inclusive report intervals, not verified physical failure onset.

    The second maintenance date is inconsistent in both UCI and its PDF. Preserve
    it as source text, with an explicit issue, rather than silently correcting it.
    """
    rows = [
        ("report_1", "2020-04-18 00:00", "2020-04-18 23:59", "", ""),
        ("report_2", "2020-05-29 23:30", "2020-05-30 06:00", "Maintenance on 30Apr at 12:00",
         "Maintenance precedes reported incident; duplicate source report number #1."),
        ("report_3", "2020-06-05 10:00", "2020-06-07 14:30", "Maintenance on 8Jun at 16:00", ""),
        ("report_4", "2020-07-15 14:30", "2020-07-15 19:00", "Maintenance on 16Jul at 00:00", ""),
    ]
    result = pd.DataFrame(rows, columns=["incident_id", "start", "end", "maintenance_text", "source_issue"])
    result["start"] = pd.to_datetime(result.start)
    result["end"] = pd.to_datetime(result.end)
    return result.assign(entity_id="metropt_apu_1", reported_type="air_leak")
