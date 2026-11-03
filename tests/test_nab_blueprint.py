"""Data-integrity and elapsed-time tests, independent of NAB downloads."""

from hashlib import sha256
from io import BytesIO
import json

import numpy as np
import pandas as pd
import pytest

from ts_anomaly.assessment import assessment_report, diagnostic_series, lag_report
from ts_anomaly.contracts import MetricDefinition, validate_schema
from ts_anomaly.nab import fetch_files, load_nab


DEFINITION = MetricDefinition("Oven temperature", "C", lower=0, upper=500)


def observations(times, values):
    return pd.DataFrame({
        "timestamp": pd.to_datetime(times), "entity_id": "oven_1",
        "metric_name": "temperature", "value": values,
    })


def test_duplicate_conflicts_are_visible_and_never_averaged():
    data = observations(
        ["2024-01-01 00:05", "2024-01-01 00:00", "2024-01-01 00:05", "2024-01-01 00:15"],
        [100., 90., 300., 110.],
    )
    original = data.copy(deep=True)
    report = assessment_report(data, {"temperature": DEFINITION}, {("oven_1", "temperature"): "5min"}).iloc[0]
    assert report.duplicate_timestamps == 1
    assert report.duplicate_rows == 2
    assert report.backward_steps == 1
    assert report.absent_slots == 1
    assert report.usable_fraction_in_file_span == 0.5
    assert report.longest_unusable_minutes == 10
    result = diagnostic_series(data, DEFINITION, "5min")
    assert result.iloc[0] == 90
    assert result.iloc[1:3].isna().all()
    pd.testing.assert_frame_equal(data, original)


def test_grid_is_anchored_at_recorded_start_not_epoch():
    data = observations(pd.date_range("2024-01-01 00:02", periods=3, freq="5min"), [1., 2., 3.])
    result = diagnostic_series(data, DEFINITION, "5min")
    assert result.index.equals(pd.DatetimeIndex(data.timestamp))


def test_irregular_sampling_is_reported_and_not_rounded():
    data = observations(["2024-01-01 00:00", "2024-01-01 00:06"], [1., 2.])
    report = assessment_report(data, {"temperature": DEFINITION}, {("oven_1", "temperature"): "5min"})
    assert report.off_grid_rows.iloc[0] == 1
    assert pd.isna(report.usable_fraction_in_file_span.iloc[0])
    with pytest.raises(ValueError, match="Off-grid"):
        diagnostic_series(data, DEFINITION, "5min")


def test_missing_nonfinite_and_out_of_bounds_are_distinguished():
    data = observations(pd.date_range("2024-01-01", periods=5, freq="5min"), [np.nan, np.inf, -1., 501., 20.])
    report = assessment_report(data, {"temperature": DEFINITION}, {("oven_1", "temperature"): "5min"}).iloc[0]
    assert report.missing_values == 1
    assert report.invalid_nonmissing_values == 3
    assert report.absent_slots == 0
    assert report.usable_fraction_in_file_span == 0.2


def test_count_semantics_do_not_accept_fractional_or_negative_counts():
    data = observations(pd.date_range("2024-01-01", periods=4, freq="5min"), [0., 2., 1.5, -1.])
    definition = MetricDefinition("Completed units per interval", "units", "interval_count")
    result = diagnostic_series(data, definition, "5min")
    assert result.notna().tolist() == [True, True, False, False]


def test_lag_pairs_do_not_bridge_missing_slots():
    data = observations(pd.date_range("2024-01-01", periods=6, freq="5min"), [1., 4., np.nan, 2., 5., 9.])
    series = diagnostic_series(data, DEFINITION, "5min")
    report = lag_report(series, {"one step": "5min", "one day": "1D"})
    expected = np.corrcoef([4., 5., 9.], [1., 2., 5.])[0, 1]
    assert report.pairs.tolist() == [3, 0]
    assert report.correlation.iloc[0] == pytest.approx(expected)
    assert pd.isna(report.correlation.iloc[1])


@pytest.mark.parametrize("value", [np.nan, 2.0])
def test_constant_or_missing_series_has_undefined_correlation(value):
    series = pd.Series(value, index=pd.date_range("2024-01-01", periods=8, freq="5min"))
    assert pd.isna(lag_report(series, {"one step": "5min"}).correlation.iloc[0])


@pytest.mark.parametrize("cadence", ["0min", "-5min", "NaT"])
def test_invalid_cadence(cadence):
    data = observations(pd.date_range("2024-01-01", periods=2, freq="5min"), [1., 2.])
    with pytest.raises(ValueError, match="Cadence"):
        diagnostic_series(data, DEFINITION, cadence)


def test_schema_rejects_labels_as_model_observations():
    data = observations(["2024-01-01"], [1.])
    validate_schema(data, {"temperature": DEFINITION})
    with pytest.raises(ValueError, match="only"):
        validate_schema(data.assign(label=1), {"temperature": DEFINITION})
    with pytest.raises(ValueError, match="definitions"):
        validate_schema(data, {})


@pytest.fixture
def nab_fixture(tmp_path):
    content = {
        "data/example.csv": b"timestamp,value\n2024-01-01 00:00:00,1\n2024-01-01 00:05:00,2\n2024-01-01 00:05:00,3\n",
        "labels/combined_windows.json": json.dumps({"example.csv": [["2024-01-01 00:00:00", "2024-01-01 00:05:00"]]}).encode(),
        "LICENSE.txt": b"Test fixture, no external data.\n",
    }
    manifest = {
        "upstream_commit": "a" * 40,
        "files": [{"path": name, "sha256": sha256(data).hexdigest()} for name, data in content.items()],
        "series": [{"path": "example.csv", "entity_id": "device", "metric_name": "temperature"}],
    }
    for name, data in content.items():
        target = tmp_path / manifest["upstream_commit"] / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return manifest, tmp_path


def test_loader_keeps_duplicates_and_timezone_unknown_and_labels_separate(nab_fixture):
    manifest, cache = nab_fixture
    data, windows = load_nab(manifest, cache, download=False)
    assert data.value.tolist() == [1, 2, 3]
    assert data.timestamp.dt.tz is None
    assert list(data.columns) == ["timestamp", "entity_id", "metric_name", "value"]
    assert len(windows) == 1
    assert windows.window_end.iloc[0] == data.timestamp.iloc[-1]


def test_checksum_mismatch_is_not_silently_overwritten(nab_fixture):
    manifest, cache = nab_fixture
    target = cache / manifest["upstream_commit"] / "data/example.csv"
    target.write_text("corrupt")
    with pytest.raises(ValueError, match="Checksum"):
        load_nab(manifest, cache, download=False)
    assert target.read_text() == "corrupt"


def test_offline_mode_requires_cached_files(nab_fixture):
    manifest, cache = nab_fixture
    (cache / manifest["upstream_commit"] / "LICENSE.txt").unlink()
    with pytest.raises(FileNotFoundError):
        fetch_files(manifest, cache, download=False)


def test_source_version_must_be_immutable(nab_fixture):
    manifest, cache = nab_fixture
    manifest["upstream_commit"] = "master"
    with pytest.raises(ValueError, match="full commit SHA"):
        fetch_files(manifest, cache, download=False)


def test_download_uses_pinned_url_and_rejects_wrong_content(nab_fixture, monkeypatch):
    manifest, cache = nab_fixture
    target = cache / manifest["upstream_commit"] / "LICENSE.txt"
    expected = target.read_bytes()
    target.unlink()
    urls = []

    def download(url, timeout):
        urls.append(url)
        return BytesIO(expected)

    monkeypatch.setattr("ts_anomaly.nab.urlopen", download)
    fetch_files(manifest, cache)
    assert urls == [f"https://raw.githubusercontent.com/numenta/NAB/{'a' * 40}/LICENSE.txt"]
    assert target.read_bytes() == expected
    target.unlink()
    monkeypatch.setattr("ts_anomaly.nab.urlopen", lambda *args, **kwargs: BytesIO(b"wrong"))
    with pytest.raises(ValueError, match="Checksum"):
        fetch_files(manifest, cache)
    assert not target.exists()


def test_window_endpoints_must_exist(nab_fixture):
    manifest, cache = nab_fixture
    target = cache / manifest["upstream_commit"] / "labels/combined_windows.json"
    content = target.read_bytes().replace(b"00:05:00", b"00:06:00")
    target.write_bytes(content)
    manifest["files"][1]["sha256"] = sha256(content).hexdigest()
    with pytest.raises(ValueError, match="endpoints"):
        load_nab(manifest, cache, download=False)
