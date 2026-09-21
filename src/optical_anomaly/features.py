"""Causal shape features; source IDs resolve in METHOD.md, Sources."""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from .validation import require_downstream_rx
from .mathematics import coefficient_of_variation, lag_one, entropy, negative_cusum

EXPERIMENTAL_FEATURES = ["cov", "autocorrelation", "entropy", "acceleration"]
FEATURES = [
    "cov",
    "autocorrelation",
    "cusum",
    "entropy",
    "acceleration",
    "slope",
    "level",
    "variability",
    "regression_slope",
    "long_level",
    "short_minus_long",
    "below_baseline_fraction",
]


def daily_design(times: pd.Series) -> np.ndarray:
    hours = (times.dt.hour + times.dt.minute / 60 + times.dt.second / 3600).to_numpy()
    phase = 2 * np.pi * hours / 24  # Source: [harmonic]; daily period is a choice.
    return np.column_stack([np.ones(len(times)), np.sin(phase), np.cos(phase)])


def seasonal_history_available(times: pd.Series, interval_minutes: int) -> bool:
    """Two covered daily cycles; an onboarding minimum, not proof of seasonality."""
    # Assumption: require two calendar days before fitting a daily reference.
    duration = times.max() - times.min() + pd.Timedelta(minutes=interval_minutes)
    return bool(duration >= pd.Timedelta(days=2))


@dataclass
class FeatureEngineer:
    window: int = 12  # Assumption: one-hour summary at the default cadence.
    long_window_hours: float = 6.0  # Assumption: development comparison scale.
    interval_minutes: int = 5  # Source cadence, not a universal PON standard.
    smoothing_hours: float = 1.0  # Assumption: EWMA time constant; [ewma].
    allowance: float = 0.25  # Assumption: CUSUM allowance in MAD units; [cusum].
    seasonal: bool = True  # Assumption: assess daily seasonality in training EDA.
    references: dict[str, tuple[np.ndarray, float]] = field(
        default_factory=dict, init=False
    )

    @property
    def long_window(self) -> int:
        return int(np.ceil(self.long_window_hours * 60 / self.interval_minutes))

    def fit(self, telemetry: pd.DataFrame) -> "FeatureEngineer":
        if self.window < 3 or self.interval_minutes <= 0 or self.smoothing_hours <= 0:
            raise ValueError("Need window >=3 and positive cadence/smoothing")
        if self.long_window < self.window:
            raise ValueError("Long window must be at least the short window")
        if self.allowance < 0:
            raise ValueError("CUSUM allowance must be nonnegative")
        require_downstream_rx(telemetry)
        self.references.clear()
        for entity, group in telemetry.loc[
            telemetry.metric_name.eq("rx_power_dbm")
        ].groupby("entity_id"):
            valid = group.dropna(subset=["value"])
            if self.seasonal and not seasonal_history_available(
                group.timestamp, self.interval_minutes
            ):
                continue
            # Assumption: minimum reference size, not a statistical guarantee.
            if len(valid) < max(100, self.window * 3):
                continue
            design = daily_design(valid.timestamp)
            if self.seasonal and np.linalg.matrix_rank(design) < 3:
                continue
            # Source: [ols], [harmonic]; freeze coefficients after training.
            coefficients = np.linalg.lstsq(design, valid.value, rcond=None)[0]
            if not self.seasonal:
                coefficients = np.array([valid.value.median(), 0, 0])
            residual = valid.value.to_numpy() - design @ coefficients
            # Source: [mad], Gaussian-consistent MAD. Floor is assumed precision.
            scale = max(
                float(1.4826 * np.median(np.abs(residual - np.median(residual)))), 0.05
            )
            self.references[str(entity)] = coefficients, scale
        if not self.references:
            raise ValueError(
                "No device has sufficient training reference; daily seasonality "
                "requires at least two days and identifiable daily phases"
            )
        return self

    def transform(self, telemetry: pd.DataFrame) -> pd.DataFrame:
        """Batch replay with history; all rolling operations are backward-looking.

        Replaying only a new chunk resets feature state. Supply prior history for
        equivalent scores; IncidentManager itself supports consecutive chunks.
        """
        frames = []
        for entity, group in telemetry.loc[
            telemetry.metric_name.eq("rx_power_dbm")
        ].groupby("entity_id"):
            group = group.sort_values("timestamp").reset_index(drop=True)
            frame = group[["timestamp", "entity_id"]].copy()
            if entity not in self.references:
                frame[FEATURES] = np.nan
            else:
                frame[FEATURES] = self._features(group)
            frames.append(frame)
        if not frames:
            raise ValueError("No Rx power observations")
        return pd.concat(frames, ignore_index=True)

    def _features(self, group: pd.DataFrame) -> pd.DataFrame:
        coefficients, scale = self.references[str(group.entity_id.iloc[0])]
        residual = (group.value - daily_design(group.timestamp) @ coefficients) / scale
        # Missing rows split state, so windows and derivatives cannot bridge gaps.
        # Source: [acf] assumes equal spacing; derivatives use that same cadence.
        step = pd.Timedelta(minutes=self.interval_minutes)
        discontinuity = group.timestamp.diff().ne(step)
        segments = (residual.isna() | residual.shift().isna() | discontinuity).cumsum()
        output = pd.DataFrame(np.nan, index=group.index, columns=FEATURES)
        for _, indices in group.groupby(segments).groups.items():
            x = residual.loc[indices]
            if x.isna().any():
                continue
            window = x.rolling(self.window, min_periods=self.window)
            # Source: [optical] uses window mean/SD; our durations are assumptions.
            output.loc[indices, "level"] = window.mean()
            output.loc[indices, "variability"] = window.std(ddof=0)
            output.loc[indices, "regression_slope"] = rolling_slope(
                x, self.window, self.interval_minutes / 60
            )
            long_mean = x.rolling(
                self.long_window, min_periods=self.window
            ).mean()
            output.loc[indices, "long_level"] = long_mean
            # Assumption: causal multi-scale contrast; no PON-specific guarantee.
            output.loc[indices, "short_minus_long"] = window.mean() - long_mean
            # Assumption: persistence below expected power, not a fault cutoff.
            output.loc[indices, "below_baseline_fraction"] = (
                x.lt(0).astype(float).rolling(self.window).mean()
            )
            # Sources: [cov], [db]. CoV requires linear power, never dBm.
            linear = 10 ** ((group.loc[indices, "value"] - coefficients[0]) / 10)
            output.loc[indices, "cov"] = linear.rolling(self.window).apply(
                coefficient_of_variation, raw=True
            )
            # Source: [acf]; correlation alone does not imply degradation.
            output.loc[indices, "autocorrelation"] = window.apply(lag_one, raw=True)
            # Source: [shannon]; these fixed residual bins are a design assumption.
            bins = np.array([-np.inf, -3, -2, -1, 0, 1, 2, 3, np.inf])
            output.loc[indices, "entropy"] = window.apply(
                lambda a: entropy(a, bins), raw=True
            )
            # Source: [cusum], [page]. A frozen target cannot follow a fault down.
            means = np.zeros(len(x))  # Expected training-normalised residual is 0.
            means[: self.window] = np.nan  # Assumption: rewarm after each gap.
            output.loc[indices, "cusum"] = negative_cusum(
                x.to_numpy(), means, self.allowance
            )
            dt = self.interval_minutes / 60
            # Source: [ewma]; this time-constant parameterisation is our choice.
            alpha = -np.expm1(-dt / self.smoothing_hours)
            output.loc[indices, "slope"] = (
                x.diff().div(dt).ewm(alpha=alpha, adjust=False).mean()
            )
            output.loc[indices, "acceleration"] = (
                # Assumption: exploratory second difference; amplifies sensor noise.
                x.diff().diff().div(dt**2).ewm(alpha=alpha, adjust=False).mean()
            )
        return output


def rolling_slope(values: pd.Series, window: int, step_hours: float) -> pd.Series:
    """Least-squares slope per hour on a complete, equally spaced past window."""
    # Source: [ols]; centred design gives the ordinary least-squares slope.
    time = np.arange(window, dtype=float) * step_hours
    centred = time - time.mean()
    weights = centred / (centred @ centred)
    return values.rolling(window).apply(lambda x: float(x @ weights), raw=True)
