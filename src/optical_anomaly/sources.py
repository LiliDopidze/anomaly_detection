"""Source-specific mappings, separate from the canonical measurement definitions."""

from collections.abc import Iterable
from .adapter import CANONICAL_METRICS, TelemetryAdapter


SYNTHETIC_METRICS = {
    "rx_dbm": "rx_power_dbm",
    "upstream_rx_dbm": "upstream_rx_power_dbm",
    "ont_tx_dbm": "ont_tx_power_dbm",
    "olt_tx_dbm": "olt_tx_power_dbm",
    **{name: name for name in CANONICAL_METRICS if not name.endswith("power_dbm")},
}


def synthetic_adapter(sources: Iterable[str] | None = None) -> TelemetryAdapter:
    """Choose a declared subset explicitly; missing selected columns still fail."""
    selected = list(SYNTHETIC_METRICS) if sources is None else list(sources)
    unknown = set(selected).difference(SYNTHETIC_METRICS)
    if unknown:
        raise ValueError(f"Unknown synthetic measurements: {sorted(unknown)}")
    metrics = {source: SYNTHETIC_METRICS[source] for source in selected}
    return TelemetryAdapter(
        timestamp_column="time",
        entity_column="device",
        metrics=metrics,
        units={name: CANONICAL_METRICS[name].unit for name in metrics.values()},
        kinds={name: CANONICAL_METRICS[name].kind for name in metrics.values()},
    )
