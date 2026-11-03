"""Right-labelled, past-only aggregation; no interpolation or forward filling."""

from dataclasses import dataclass
import pandas as pd
import numpy as np
from .contracts import MetricDefinition
from .pon import CANONICAL_METRICS


@dataclass(frozen=True)
class DataValidator:
    # Assumption: polling grid; ETSI F5G 011 does not mandate one universal cadence.
    interval: str = "5min"
    source_interval: str | None = None
    definitions: dict[str, MetricDefinition] | None = None

    def transform(self, telemetry: pd.DataFrame) -> pd.DataFrame:
        required = {"timestamp", "entity_id", "metric_name", "value"}
        if set(telemetry.columns) != required:
            raise ValueError(f"Expected exactly {sorted(required)}")
        keys = ["entity_id", "metric_name", "timestamp"]
        if telemetry.duplicated(keys).any():
            raise ValueError("Duplicate telemetry")
        definitions = (
            CANONICAL_METRICS if self.definitions is None else self.definitions
        )
        target = pd.Timedelta(self.interval)
        source = pd.Timedelta(self.source_interval or self.interval)
        if min(source, target) <= pd.Timedelta(0) or target % source:
            raise ValueError(
                "Target interval must be a positive multiple of source interval"
            )
        if telemetry.empty:
            raise ValueError(
                "No telemetry; supply a declared schedule for coverage assessment"
            )
        if telemetry[keys].isna().any().any():
            raise ValueError("Missing telemetry keys")
        if not isinstance(telemetry.timestamp.dtype, pd.DatetimeTZDtype):
            raise ValueError("Telemetry timestamps must be timezone-aware")
        unknown = set(telemetry.metric_name) - definitions.keys()
        if unknown:
            raise ValueError(f"Unsupported metrics: {sorted(unknown)}")
        telemetry = telemetry.copy()
        telemetry["timestamp"] = telemetry.timestamp.dt.tz_convert("UTC")
        telemetry["value"] = telemetry.value.where(np.isfinite(telemetry.value))
        # Ratios must describe the same source intervals. A missing component
        # invalidates the group; absent unmapped metrics are not manufactured.
        count_groups = {}
        for metric in telemetry.metric_name.unique():
            definition = definitions[metric]
            if definition.kind == "interval_count":
                count_groups.setdefault(definition.count_group or metric, []).append(
                    metric
                )
        for metrics in count_groups.values():
            mask = telemetry.metric_name.isin(metrics)
            counts = telemetry.loc[mask].pivot(
                index=["entity_id", "timestamp"], columns="metric_name", values="value"
            )
            complete = counts.notna().all(axis=1)
            rows = pd.MultiIndex.from_frame(
                telemetry.loc[mask, ["entity_id", "timestamp"]]
            )
            telemetry.loc[mask, "value"] = telemetry.loc[mask, "value"].where(
                complete.reindex(rows).to_numpy()
            )
        frames = []
        for (entity, metric), group in telemetry.groupby(keys[:2], sort=True):
            series = group.sort_values("timestamp").set_index("timestamp").value
            definition = definitions[metric]
            if definition.kind == "interval_count":
                if (series.index != series.index.floor(source)).any():
                    raise ValueError(
                        "Count timestamps must align to the declared source interval"
                    )
                finite = series.dropna()
                if ((finite < 0) | (finite != np.floor(finite))).any():
                    raise ValueError("Interval counts must be nonnegative integers")
            # Design: reports at t use only values timestamped at or before t.
            bins = series.resample(
                self.interval, closed="right", label="right", origin="epoch"
            )
            sampled = (
                bins.sum(min_count=int(target / source))
                if definition.kind == "interval_count"
                else bins.mean()
            )
            frame = sampled.rename("value").reset_index()
            frame["entity_id"], frame["metric_name"] = entity, metric
            frame["observed"] = sampled.notna().to_numpy()
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)


def require_downstream_rx(telemetry: pd.DataFrame) -> None:
    """Check model inputs after adaptation; history sufficiency is checked at fit."""
    entities = set(telemetry.entity_id)
    rx = telemetry.loc[telemetry.metric_name.eq("rx_power_dbm")]
    usable = set(rx.loc[np.isfinite(rx.value), "entity_id"])
    missing = entities - usable
    if not entities or missing:
        raise ValueError(
            "Downstream Rx baseline requires finite rx_power_dbm observations "
            f"for each entity; unavailable: {sorted(missing)}"
        )
