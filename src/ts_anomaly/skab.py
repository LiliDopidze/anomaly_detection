"""SKAB source reading and documented sensor semantics; no anomaly models."""

from hashlib import sha1
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from .contracts import MetricDefinition, validate_schema


FAMILIES = {
    "anomaly-free": "Separate recording described as normal operation; no point annotations.",
    "valve1": "Experiments restricting water entering the pump (inlet valve).",
    "valve2": "Experiments restricting water leaving the pump (outlet valve).",
    "other": "Other interventions: fluid changes, rotor imbalance, cavitation or warmer water.",
}


METRICS = {
    "Accelerometer1RMS": MetricDefinition(
        "Vibration acceleration, sensor 1 (RMS)", "g", "gauge", "measurement",
        note="Amplitude summary; raw vibration frequencies cannot be recovered."),
    "Accelerometer2RMS": MetricDefinition(
        "Vibration acceleration, sensor 2 (RMS)", "g", "gauge", "measurement",
        note="A second vibration channel; location and axis are not specified."),
    "Current": MetricDefinition(
        "Electric motor current", "A", "gauge", "measurement",
        note="Changes may reflect motor load or electrical behaviour."),
    "Pressure": MetricDefinition(
        "Loop pressure after the pump", "bar", "gauge", "measurement",
        note="Pressure reference and calibrated limits are unspecified; retain negatives."),
    "Temperature": MetricDefinition(
        "Motor body temperature", "degC", "gauge", "measurement",
        note="Thermal history and operating conditions can affect this signal."),
    "Thermocouple": MetricDefinition(
        "Circulating fluid temperature", "degC", "gauge", "measurement",
        note="A different thermal location from the motor body."),
    "Voltage": MetricDefinition(
        "Electric motor voltage", "V", "gauge", "measurement",
        note="Supply changes can accompany changes in current."),
    "Volume Flow RateRMS": MetricDefinition(
        "Circulating fluid flow rate", "L/min", "gauge", "measurement",
        note="Named RateRMS in the README; preserve the actual CSV column name."),
}


def fetch_recording(cache: Path, commit: str, record: dict, download=True) -> Path:
    """Verify original bytes against the pinned Git blob, including cached files."""
    path = Path(cache) / record["path"]
    if path.exists():
        content = path.read_bytes()
    else:
        if not download:
            raise FileNotFoundError(path)
        url = f'https://raw.githubusercontent.com/waico/SKAB/{commit}/{record["path"]}'
        with urlopen(url, timeout=60) as response:
            content = response.read()
    header = b"blob " + str(len(content)).encode() + b"\0"
    if sha1(header + content).hexdigest() != record["git_blob_sha"]:
        raise ValueError(f"Checksum mismatch: {record['path']}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return path


def read_recording(path: Path, recording_id: str):
    """Keep measurements separate from annotations; preserve order and timestamps.

    entity_id is an analytical recording key here, not a distinct physical asset.
    The source has one testbed, unknown timezone and no operating-command channels.
    """
    raw = pd.read_csv(path, sep=";")
    has_labels = recording_id != "anomaly-free/anomaly-free"
    expected = {"datetime", *METRICS}
    if has_labels:
        expected.update(("anomaly", "changepoint"))
    if set(raw.columns) != expected:
        raise ValueError(f"Unexpected source columns in {recording_id}")
    timestamps = pd.to_datetime(raw["datetime"], format="%Y-%m-%d %H:%M:%S", errors="raise")
    data = raw[list(METRICS)].copy()
    data.insert(0, "entity_id", recording_id)
    data.insert(0, "timestamp", timestamps)
    validate_schema(data, METRICS)

    labels = None
    if has_labels:
        labels = raw[["anomaly", "changepoint"]].copy()
        if not labels.isin([0, 1]).all().all():
            raise ValueError(f"Labels must be complete binary values in {recording_id}")
        labels = labels.astype("int8")
        labels.insert(0, "timestamp", timestamps)
    return data, labels


def load_development(cache: Path, settings: dict, download=True):
    """Load only declared reference/development recordings; never open test files."""
    recordings, annotations = {}, {}
    for record in settings["files"]:
        if record["partition"] == "reserved_test":
            continue
        if record["partition"] not in {"reference", "development"}:
            raise ValueError(f"Unknown partition: {record['partition']}")
        identifier = record["recording_id"]
        if identifier in recordings:
            raise ValueError(f"Duplicate recording identifier: {identifier}")
        path = fetch_recording(cache, settings["source_commit"], record, download)
        data, labels = read_recording(path, identifier)
        recordings[identifier] = data
        if labels is not None:
            annotations[identifier] = labels
    return recordings, annotations


def initial_normal(data: pd.DataFrame, labels: pd.DataFrame | None) -> pd.DataFrame:
    """Development diagnostic prefix before the first positive annotation.

    Labels make this a retrospective diagnostic, not an online selection rule.
    Later returns to label 0 are deliberately excluded.
    """
    if labels is None:
        return data.copy()
    if not data.timestamp.equals(labels.timestamp):
        raise ValueError("Annotations must align exactly with observations")
    positive = labels[["anomaly", "changepoint"]].any(axis=1)
    if not positive.any():
        return data.copy()
    first = positive.to_numpy().nonzero()[0][0]
    return data.iloc[:first].copy()
