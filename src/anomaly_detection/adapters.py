"""Source-specific meanings, outside the generic numeric core."""
import numpy as np
import pandas as pd
from optical_anomaly.sources import synthetic_adapter
from .validation import CANONICAL, Metric, interval_ratio


def gpon_registry(cadence=5):
    return {name: Metric("dBm", cadence_minutes=cadence, direction="low", floor=0.05)
            for name in ["rx_power_dbm", "upstream_rx_power_dbm"]}


def gpon(native, available_column=None):
    """Reuse the unchanged checked GPON adapter; zero latency is explicit in simulation."""
    data = synthetic_adapter().transform(native).rename(columns={"timestamp": "event_time"})
    if available_column:
        times = native[["device", "time", available_column]].rename(columns={
            "device": "entity_id", "time": "event_time", available_column: "available_time"})
        data = data.merge(times, on=["entity_id", "event_time"], validate="many_to_one")
    else:
        data["available_time"] = data.event_time
    data["quality"] = np.where(data.value.notna(), "ok", "missing")
    # Counts remain counts. Sparse FEC predictors require a separately justified model.
    return data[CANONICAL]


def optical_loss(tx, rx, tx_time, rx_time):
    if not np.array_equal(tx_time, rx_time):
        raise ValueError("Tx/Rx intervals must agree")
    return np.asarray(tx, float) - np.asarray(rx, float)


def fec_fraction(corrected, uncorrectable, total, start, end):
    c, u, d = map(lambda x: np.asarray(x, float), (corrected, uncorrectable, total))
    if np.any(np.isfinite(c+u+d) & (c+u > d)):
        raise ValueError("FEC error codewords exceed all received codewords")
    return interval_ratio(c+u, d, start, start, end, end)


def non_optical_fixture(periods=1000):
    """Machine temperature/load example. Interface demonstration, not transfer evidence."""
    t = pd.date_range("2025-01-01", periods=periods, freq="5min", tz="UTC")
    rng = np.random.default_rng(17)
    load = 50 + 10*np.sin(np.arange(periods)/35) + rng.normal(0, 1, periods)
    temperature = 20 + .2*load + rng.normal(0, .2, periods)
    records = []
    for name, values in [("load", load), ("temperature", temperature)]:
        records.append(pd.DataFrame(dict(entity_id="machine-A", metric_name=name,
            event_time=t, available_time=t, value=values, quality="ok")))
    registry = {"load": Metric("percent", minimum=0, maximum=100, floor=.1),
                "temperature": Metric("C", floor=.05)}
    return pd.concat(records, ignore_index=True)[CANONICAL], registry
