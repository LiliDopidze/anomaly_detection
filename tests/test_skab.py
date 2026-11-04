"""Source contracts, label isolation and held-out experiment protection."""

from hashlib import sha1
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from ts_anomaly.skab import METRICS, fetch_recording, initial_normal, load_development, read_recording


def write_recording(tmp_path, identifier="valve1/1"):
    raw = pd.DataFrame({name: np.arange(4, dtype=float) for name in METRICS})
    raw.insert(0, "datetime", pd.date_range("2020-01-01", periods=4, freq="s"))
    if identifier != "anomaly-free/anomaly-free":
        raw["anomaly"] = [0, 0, 1, 0]
        raw["changepoint"] = [0, 0, 1, 1]
    path = tmp_path / f"data/{identifier}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(path, sep=";", index=False)
    return path, raw


def record_for(path, root, identifier="valve1/1", partition="development"):
    content = path.read_bytes()
    digest = sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
    return {"recording_id": identifier, "path": str(path.relative_to(root)),
            "git_blob_sha": digest, "partition": partition}


def test_source_separates_annotations_and_preserves_names_units_and_timezone(tmp_path):
    path, _ = write_recording(tmp_path)
    data, labels = read_recording(path, "valve1/1")
    assert set(data) == {"timestamp", "entity_id", *METRICS}
    assert set(labels) == {"timestamp", "anomaly", "changepoint"}
    assert data.timestamp.equals(labels.timestamp)
    assert data.timestamp.dt.tz is None
    assert "Volume Flow RateRMS" in data
    assert METRICS["Volume Flow RateRMS"].unit == "L/min"


def test_source_preserves_invalid_measurements_and_duplicate_acquisition_for_audit(tmp_path):
    path, raw = write_recording(tmp_path)
    raw.loc[0, "Pressure"] = -1
    raw.loc[1, "Pressure"] = np.nan
    raw.loc[2, "Pressure"] = np.inf
    raw.loc[3, "datetime"] = raw.datetime.iloc[0]
    raw.to_csv(path, sep=";", index=False)
    data, _ = read_recording(path, "valve1/1")
    assert data.Pressure.iloc[0] == -1
    assert pd.isna(data.Pressure.iloc[1])
    assert np.isinf(data.Pressure.iloc[2])
    assert data.timestamp.duplicated().sum() == 1


@pytest.mark.parametrize("value", [np.nan, 2, -1])
def test_invalid_or_missing_annotation_is_rejected(tmp_path, value):
    path, raw = write_recording(tmp_path)
    raw["anomaly"] = raw.anomaly.astype(float)
    raw.loc[1, "anomaly"] = value
    raw.to_csv(path, sep=";", index=False)
    with pytest.raises(ValueError, match="binary"):
        read_recording(path, "valve1/1")


def test_reference_does_not_invent_point_annotations(tmp_path):
    path, _ = write_recording(tmp_path, "anomaly-free/anomaly-free")
    data, labels = read_recording(path, "anomaly-free/anomaly-free")
    assert labels is None
    pd.testing.assert_frame_equal(initial_normal(data, labels), data)


def test_initial_prefix_excludes_later_label_zero_and_is_future_independent(tmp_path):
    path, _ = write_recording(tmp_path)
    data, labels = read_recording(path, "valve1/1")
    prefix = initial_normal(data, labels)
    assert len(prefix) == 2
    data.loc[2:, list(METRICS)] = 9999
    pd.testing.assert_frame_equal(prefix, initial_normal(data, labels))
    labels.loc[0, "changepoint"] = 1
    assert initial_normal(data, labels).empty
    labels["timestamp"] += pd.Timedelta(seconds=1)
    with pytest.raises(ValueError, match="align"):
        initial_normal(data, labels)


def test_cache_integrity_and_offline_missing_file(tmp_path):
    path, _ = write_recording(tmp_path)
    item = record_for(path, tmp_path)
    assert fetch_recording(tmp_path, "unused", item, False) == path
    path.write_text("corrupted")
    with pytest.raises(ValueError, match="Checksum"):
        fetch_recording(tmp_path, "unused", item, False)
    assert path.read_text() == "corrupted"
    path.unlink()
    with pytest.raises(FileNotFoundError):
        fetch_recording(tmp_path, "unused", item, False)


def test_reserved_files_are_never_downloaded_or_read(tmp_path):
    path, _ = write_recording(tmp_path)
    settings = {"source_commit": "unused", "files": [
        record_for(path, tmp_path),
        {"recording_id": "valve1/3", "partition": "reserved_test",
         "path": "nonexistent.csv", "git_blob_sha": "deliberately invalid"}]}
    before, labels = load_development(tmp_path, settings, download=False)
    (tmp_path / "nonexistent.csv").write_text("not even a CSV")
    after, after_labels = load_development(tmp_path, settings, download=False)
    assert list(before) == ["valve1/1"]
    pd.testing.assert_frame_equal(before["valve1/1"], after["valve1/1"])
    pd.testing.assert_frame_equal(labels["valve1/1"], after_labels["valve1/1"])


def test_manifest_covers_actual_source_tree_and_freezes_whole_experiments():
    settings = json.loads((Path(__file__).resolve().parents[1] / "configs/skab.json").read_text())
    files = settings["files"]
    identifiers = [record["recording_id"] for record in files]
    reserved = {record["recording_id"] for record in files if record["partition"] == "reserved_test"}
    assert len(files) == len(set(identifiers)) == 35
    assert reserved == {"valve1/3", "valve1/7", "valve1/11", "valve1/15",
                        "valve2/3", "other/4", "other/8", "other/12"}
    assert settings["example_recording"] not in reserved
    assert len(settings["source_commit"]) == 40


def test_initial_period_stops_at_changepoint_before_anomaly_label(tmp_path):
    path, _ = write_recording(tmp_path)
    data, labels = read_recording(path, 'valve1/1')
    labels.loc[1, 'changepoint'] = 1
    before = initial_normal(data, labels)
    assert len(before) == 1
    assert labels.anomaly.iloc[1] == 0
