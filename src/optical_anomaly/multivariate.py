"""Nested telemetry feature sets with training-only, per-entity references."""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from .features import FeatureEngineer, FEATURES, daily_design, rolling_slope

FEATURE_SETS = ("rx_only", "both_rx", "tx_rx", "fec", "temperature")
CHANNELS = {
    "both_rx": ["upstream_rx"],
    "tx_rx": ["downstream_loss", "upstream_loss"],
    "fec": [
        f"{d}_fec_{k}"
        for d in ("downstream", "upstream")
        for k in ("corrected", "uncorrectable")
    ],
    "temperature": ["ont_temperature", "olt_temperature"],
}
SUMMARIES = ("level", "slope", "variability")


def channels_for(feature_set: str) -> list[str]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    return [
        channel
        for stage in FEATURE_SETS[1 : FEATURE_SETS.index(feature_set) + 1]
        for channel in CHANNELS[stage]
    ]


def summaries_for(channel: str) -> tuple[str, ...]:
    if channel in ("upstream_rx", "downstream_loss", "upstream_loss"):
        return SUMMARIES + ("regression_slope", "long_level", "short_minus_long")
    if "fec" in channel:
        return SUMMARIES + ("error_interval_fraction",)
    return SUMMARIES


def columns_for(feature_set: str) -> list[str]:
    return FEATURES + [
        f"{channel}_{summary}"
        for channel in channels_for(feature_set)
        for summary in summaries_for(channel)
    ]


def measurement_channels(telemetry: pd.DataFrame) -> pd.DataFrame:
    """Synchronous relationships only; no filling of missing measurements.

    FEC fractions are transformed as log(1 + fraction / 1e-6); 1e-6 is a fixed
    numerical reference, not a physical alarm threshold. A zero total is unknown.
    Loss is Tx(dBm) - Rx(dBm), an approximate link-loss proxy in dB.
    """
    wide = telemetry.pivot(
        index=["entity_id", "timestamp"], columns="metric_name", values="value"
    ).sort_index()
    result = pd.DataFrame(index=wide.index)
    definitions = {
        "upstream_rx": ("upstream_rx_power_dbm",),
        "downstream_loss": ("olt_tx_power_dbm", "rx_power_dbm"),
        "upstream_loss": ("ont_tx_power_dbm", "upstream_rx_power_dbm"),
        "ont_temperature": ("ont_temperature_c",),
        "olt_temperature": ("olt_temperature_c",),
    }
    for name, inputs in definitions.items():
        if all(column in wide for column in inputs):
            result[name] = wide[inputs[0]]
            if len(inputs) == 2:
                result[name] = result[name] - wide[inputs[1]]
    for direction in ("downstream", "upstream"):
        total = f"{direction}_fec_total_codewords"
        for kind in ("corrected", "uncorrectable"):
            numerator = f"{direction}_fec_{kind}_codewords"
            if total in wide and numerator in wide:
                fraction = wide[numerator] / wide[total].where(wide[total] > 0)
                fraction = fraction.where(fraction.between(0, 1))
                result[f"{direction}_fec_{kind}"] = np.log1p(fraction / 1e-6)
    return result.reset_index()


@dataclass
class MultivariateFeatures:
    """Baseline Rx mathematics plus compact summaries of added measurements."""

    baseline: FeatureEngineer
    feature_set: str = "temperature"
    references: dict = field(default_factory=dict, init=False)

    def fit(self, telemetry: pd.DataFrame) -> "MultivariateFeatures":
        self.baseline.fit(telemetry)
        channels = measurement_channels(telemetry)
        selected = channels_for(self.feature_set)
        missing = set(selected).difference(channels)
        if missing:
            raise ValueError(
                f"Missing channels for {self.feature_set}: {sorted(missing)}"
            )
        self.references.clear()
        for entity, group in channels.groupby("entity_id"):
            for name in selected:
                clean = group.dropna(subset=[name])
                if len(clean) < max(100, self.baseline.window * 3):
                    raise ValueError(f"Insufficient training history: {entity}/{name}")
                design = daily_design(clean.timestamp)
                coefficients = np.linalg.lstsq(design, clean[name], rcond=None)[0]
                if not self.baseline.seasonal:
                    coefficients = np.array([clean[name].median(), 0.0, 0.0])
                residual = clean[name].to_numpy() - design @ coefficients
                # Explicit precision floors, not learned from validation or faults.
                floor = 0.1 if "temperature" in name else 0.05
                scale = max(
                    float(1.4826 * np.median(np.abs(residual - np.median(residual)))),
                    floor,
                )
                self.references[(entity, name)] = coefficients, scale
        return self

    def transform(self, telemetry: pd.DataFrame) -> pd.DataFrame:
        features = self.baseline.transform(telemetry)
        channels = measurement_channels(telemetry)
        additions = []
        for entity, group in channels.groupby("entity_id", sort=True):
            group = group.sort_values("timestamp").reset_index(drop=True)
            result = group[["timestamp", "entity_id"]].copy()
            for name in channels_for(self.feature_set):
                reference = self.references.get((entity, name))
                if reference is None or name not in group:
                    for summary in summaries_for(name):
                        result[f"{name}_{summary}"] = np.nan
                    continue
                coefficients, scale = reference
                residual = (
                    group[name] - daily_design(group.timestamp) @ coefficients
                ) / scale
                result = pd.concat(
                    [
                        result,
                        self._summaries(residual, group.timestamp, name, group[name]),
                    ],
                    axis=1,
                )
            additions.append(result)
        extra = pd.concat(additions, ignore_index=True)
        return features.merge(
            extra, on=["entity_id", "timestamp"], how="left", validate="one_to_one"
        )

    def _summaries(
        self,
        residual: pd.Series,
        times: pd.Series,
        name: str,
        measurement: pd.Series | None = None,
    ) -> pd.DataFrame:
        output = pd.DataFrame(
            np.nan,
            index=residual.index,
            columns=[f"{name}_{s}" for s in summaries_for(name)],
        )
        step = pd.Timedelta(minutes=self.baseline.interval_minutes)
        gaps = times.diff().ne(step)
        segments = (gaps | residual.isna() | residual.shift().isna()).cumsum()
        alpha = -np.expm1(
            -self.baseline.interval_minutes / (60 * self.baseline.smoothing_hours)
        )
        grouped = residual.groupby(segments)
        rolling = grouped.rolling(
            self.baseline.window, min_periods=self.baseline.window
        )
        output[f"{name}_level"] = rolling.mean().droplevel(0).sort_index()
        output[f"{name}_variability"] = rolling.std(ddof=0).droplevel(0).sort_index()
        derivative = grouped.diff() / (self.baseline.interval_minutes / 60)
        output[f"{name}_slope"] = (
            derivative.groupby(segments)
            .ewm(alpha=alpha, adjust=False, min_periods=self.baseline.window)
            .mean()
            .droplevel(0)
            .sort_index()
        )
        if "long_level" in summaries_for(name):
            output[f"{name}_long_level"] = (
                grouped.rolling(self.baseline.long_window)
                .mean()
                .droplevel(0)
                .sort_index()
            )
            output[f"{name}_short_minus_long"] = (
                output[f"{name}_level"] - output[f"{name}_long_level"]
            )
            output[f"{name}_regression_slope"] = grouped.transform(
                lambda x: rolling_slope(
                    x, self.baseline.window, self.baseline.interval_minutes / 60
                )
            )
        if "error_interval_fraction" in summaries_for(name):
            if measurement is None:
                raise ValueError("FEC summaries require uncentred measurements")
            present = measurement.gt(0).astype(float).where(measurement.notna())
            output[f"{name}_error_interval_fraction"] = (
                present.groupby(segments)
                .rolling(self.baseline.window)
                .mean()
                .droplevel(0)
                .sort_index()
            )
        return output
