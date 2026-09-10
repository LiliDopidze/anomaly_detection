"""Metric-aware, causal feature mechanics for Telecom telemetry.

The sector catalogue decides how a measurement is interpreted.  This module
only applies those declarations; it does not contain vendor thresholds or
fault labels.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


IDENTITY_COLUMNS = ["event_ts", "entity_id", "episode_id"]
VALID_DIRECTIONS = {"high_bad", "low_bad", "two_sided", "contextual"}
VALID_TRANSFORMS = {
    "identity",
    "log1p",
    "hurdle_log10",
    "hurdle_log1p",
    "reset_safe_increment",
    "state_transition",
    "event",
}


def _catalogue(catalogue: pd.DataFrame) -> pd.DataFrame:
    required = {
        "metric_id",
        "measurement_kind",
        "expected_cadence_seconds",
    }
    missing = required - set(catalogue.columns)
    if missing:
        raise ValueError(f"Metric catalogue is missing {sorted(missing)}")
    if catalogue["metric_id"].astype(str).duplicated().any():
        raise ValueError("metric_id must be unique")
    return catalogue.assign(metric_id=catalogue["metric_id"].astype(str)).set_index(
        "metric_id", drop=False
    )


def _declared(row: pd.Series, name: str, default):
    value = row.get(name, default)
    return default if pd.isna(value) else value


def _transform_name(row: pd.Series) -> str:
    """Use an explicit transform, with a conservative legacy fallback."""

    value = str(_declared(row, "transform", "")).strip()
    if not value:
        kind = str(row["measurement_kind"])
        value = {
            "interval_count": "log1p",
            "cumulative_counter": "reset_safe_increment",
        }.get(kind, "identity")
    if value not in VALID_TRANSFORMS:
        raise ValueError(f"Unsupported transform {value!r} for {row['metric_id']}")
    return value


def _direction(row: pd.Series) -> str:
    value = str(_declared(row, "direction", "two_sided"))
    if value not in VALID_DIRECTIONS:
        raise ValueError(f"Unsupported direction {value!r} for {row['metric_id']}")
    return value


def _valid_values(
    panel: pd.DataFrame, metric_id: str, row: pd.Series
) -> tuple[pd.Series, pd.Series]:
    values = pd.to_numeric(panel[metric_id], errors="coerce").astype(float)
    clipped_name = f"{metric_id}__clipped"
    clipped = (
        pd.to_numeric(
            panel.get(clipped_name, pd.Series(0, index=panel.index)),
            errors="coerce",
        )
        .fillna(0)
        .astype(bool)
    )

    lower = pd.to_numeric(pd.Series([row.get("valid_min")]), errors="coerce").iloc[0]
    upper = pd.to_numeric(pd.Series([row.get("valid_max")]), errors="coerce").iloc[0]
    invalid = ~np.isfinite(values)
    if pd.notna(lower):
        invalid |= values.lt(float(lower))
    if pd.notna(upper):
        invalid |= values.gt(float(upper))
    return values.mask(invalid | clipped), clipped


def _safe_difference(
    values: pd.Series,
    timestamps: pd.Series,
    cadence_seconds,
    gap_tolerance: float,
) -> tuple[pd.Series, pd.Series]:
    difference = values.diff()
    elapsed = timestamps.diff().dt.total_seconds()
    bad_interval = elapsed.le(0)
    if pd.notna(cadence_seconds) and float(cadence_seconds) > 0:
        bad_interval |= elapsed.gt(float(cadence_seconds) * float(gap_tolerance))
    difference = difference.mask(bad_interval)
    return difference, bad_interval.fillna(True)


def transform_episode(
    panel: pd.DataFrame,
    catalogue: pd.DataFrame,
    *,
    gap_tolerance: float = 1.5,
) -> pd.DataFrame:
    """Transform one wide episode without looking into the future.

    A hurdle transform keeps zero occurrence separate from positive magnitude.
    Cumulative counters become a non-negative increment plus a reset flag.
    Differences are never calculated across a collection gap.
    """

    missing = set(IDENTITY_COLUMNS) - set(panel.columns)
    if missing:
        raise ValueError(f"Panel is missing {sorted(missing)}")
    if panel[["entity_id", "episode_id"]].drop_duplicates().shape[0] != 1:
        raise ValueError("transform_episode expects exactly one entity episode")
    if not 1 <= float(gap_tolerance):
        raise ValueError("gap_tolerance must be at least 1")

    metadata = _catalogue(catalogue)
    missing_metrics = set(metadata.index) - set(panel.columns)
    if missing_metrics:
        raise ValueError(f"Panel is missing metrics {sorted(missing_metrics)}")

    panel = panel.copy()
    panel["event_ts"] = pd.to_datetime(panel["event_ts"], utc=True, errors="raise")
    panel = panel.sort_values("event_ts", kind="stable").reset_index(drop=True)
    if panel["event_ts"].duplicated().any():
        raise ValueError("An episode cannot contain duplicate timestamps")

    output = panel[IDENTITY_COLUMNS].copy()
    timestamps = panel["event_ts"]

    for metric_id, row in metadata.iterrows():
        values, clipped = _valid_values(panel, metric_id, row)
        transform = _transform_name(row)
        kind = str(row["measurement_kind"])
        cadence = pd.to_numeric(
            pd.Series([row["expected_cadence_seconds"]]), errors="coerce"
        ).iloc[0]

        if transform == "reset_safe_increment":
            difference, bad_interval = _safe_difference(
                values, timestamps, cadence, gap_tolerance
            )
            reset = difference.lt(0).where(difference.notna())
            modulus = pd.to_numeric(
                pd.Series([row.get("counter_modulus")]), errors="coerce"
            ).iloc[0]
            policy = str(_declared(row, "reset_policy", "reset_to_unknown"))
            if policy == "unwrap_known_modulus" and pd.notna(modulus):
                previous = values.shift()
                wrapped = float(modulus) - previous + values
                increment = difference.where(difference.ge(0), wrapped)
            else:
                increment = difference.mask(difference.lt(0))
            increment = increment.mask(bad_interval)
            output[f"{metric_id}__increment"] = increment
            output[f"{metric_id}__reset"] = reset.astype(float)

        elif kind == "discrete_state" or transform == "state_transition":
            difference, _ = _safe_difference(values, timestamps, cadence, gap_tolerance)
            output[f"{metric_id}__state"] = values
            output[f"{metric_id}__transition"] = (
                difference.ne(0).where(difference.notna()).astype(float)
            )

        elif transform in {"hurdle_log10", "hurdle_log1p"}:
            nonzero = values.gt(0).where(values.notna()).astype(float)
            positive = values.where(values.gt(0))
            magnitude = (
                np.log10(positive)
                if transform == "hurdle_log10"
                else np.log1p(positive)
            )
            suffix = (
                "positive_log10" if transform == "hurdle_log10" else "positive_log1p"
            )
            difference, _ = _safe_difference(
                magnitude, timestamps, cadence, gap_tolerance
            )
            output[f"{metric_id}__nonzero"] = nonzero
            output[f"{metric_id}__{suffix}"] = magnitude
            output[f"{metric_id}__{suffix}_difference"] = difference

        else:
            if transform == "log1p":
                level = np.log1p(values.where(values.ge(0)))
            else:
                level = values
            difference, _ = _safe_difference(level, timestamps, cadence, gap_tolerance)
            output[f"{metric_id}__level"] = level
            output[f"{metric_id}__difference"] = difference

        # Censoring is observable data quality, not an asset-health feature.
        output[f"{metric_id}__clipped"] = clipped.astype(float)

    return output


def feature_policy(
    catalogue: pd.DataFrame,
    feature_names: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Return the direction, role and scale floor for generated features."""

    metadata = _catalogue(catalogue)
    rows = []
    for metric_id, row in metadata.iterrows():
        transform = _transform_name(row)
        kind = str(row["measurement_kind"])
        direction = _direction(row)
        minimum_scale = pd.to_numeric(
            pd.Series([row.get("minimum_scale")]), errors="coerce"
        ).iloc[0]
        minimum_scale = float(minimum_scale) if pd.notna(minimum_scale) else 1e-6

        if transform == "reset_safe_increment":
            definitions = [
                (
                    "increment",
                    direction if direction != "contextual" else "two_sided",
                    minimum_scale,
                ),
                ("reset", "high_bad", 0.25),
            ]
        elif kind == "discrete_state" or transform == "state_transition":
            definitions = [
                ("state", "contextual", minimum_scale),
                ("transition", "high_bad", 0.25),
            ]
        elif transform in {"hurdle_log10", "hurdle_log1p"}:
            suffix = (
                "positive_log10" if transform == "hurdle_log10" else "positive_log1p"
            )
            definitions = [
                ("nonzero", "high_bad" if direction != "low_bad" else "low_bad", 0.25),
                (suffix, direction, minimum_scale),
                (
                    f"{suffix}_difference",
                    direction if direction != "contextual" else "two_sided",
                    minimum_scale,
                ),
            ]
        else:
            definitions = [
                ("level", direction, minimum_scale),
                (
                    "difference",
                    direction if direction != "contextual" else "two_sided",
                    minimum_scale,
                ),
            ]

        if bool(_declared(row, "seasonality_candidate", False)):
            definitions.append(
                ("seasonal_difference", direction, minimum_scale)
            )

        for suffix, feature_direction, floor in definitions:
            rows.append(
                {
                    "feature": f"{metric_id}__{suffix}",
                    "metric_id": metric_id,
                    "direction": feature_direction,
                    "minimum_scale": floor,
                    "role": "asset_health",
                }
            )
        rows.append(
            {
                "feature": f"{metric_id}__clipped",
                "metric_id": metric_id,
                "direction": "high_bad",
                "minimum_scale": 0.25,
                "role": "data_quality",
            }
        )

    policy = pd.DataFrame(rows).set_index("feature", drop=False)
    if feature_names is None:
        return policy.reset_index(drop=True)

    requested = list(map(str, feature_names))
    resolved = []
    for name in requested:
        base = name.removesuffix("__history_z")
        if base not in policy.index:
            raise ValueError(f"No metric policy for feature {name!r}")
        item = policy.loc[base].to_dict()
        item["feature"] = name
        resolved.append(item)
    return pd.DataFrame(resolved)


def add_causal_seasonal_differences(
    features: pd.DataFrame,
    catalogue: pd.DataFrame,
    selected_periods: dict[str, int | float | None],
) -> pd.DataFrame:
    """Compare selected metrics with the same phase in an earlier cycle.

    The previous value must exist at the exact lagged timestamp.  Missing
    observations therefore remain missing; the function never interpolates
    across a collection gap or uses a future value.
    """

    output = features.copy()
    timestamps = pd.DatetimeIndex(pd.to_datetime(output["event_ts"], utc=True))
    metadata = _catalogue(catalogue)

    for metric_id, period_seconds in (selected_periods or {}).items():
        if period_seconds is None or metric_id not in metadata.index:
            continue
        period_seconds = float(period_seconds)
        if period_seconds <= 0:
            raise ValueError(f"Seasonal period must be positive for {metric_id}")

        row = metadata.loc[metric_id]
        transform = _transform_name(row)
        if transform in {"identity", "log1p"}:
            source_column = f"{metric_id}__level"
        elif transform in {"hurdle_log10", "hurdle_log1p"}:
            suffix = "positive_log10" if transform == "hurdle_log10" else "positive_log1p"
            source_column = f"{metric_id}__{suffix}"
        else:
            # Counter increments and state changes are already local changes;
            # a phase-on-phase level comparison is not well defined for them.
            continue
        if source_column not in output:
            continue

        values = pd.Series(
            pd.to_numeric(output[source_column], errors="coerce").to_numpy(),
            index=timestamps,
        )
        previous = values.reindex(
            timestamps - pd.Timedelta(seconds=period_seconds)
        ).to_numpy()
        output[f"{metric_id}__seasonal_difference"] = (
            values.to_numpy() - previous
        )

    return output


def add_causal_history(
    features: pd.DataFrame,
    catalogue: pd.DataFrame,
    *,
    history_seconds: float,
    minimum_history_seconds: float,
    gap_tolerance: float = 1.5,
) -> pd.DataFrame:
    """Append trailing robust deviations based only on earlier observations."""

    if history_seconds <= 0 or minimum_history_seconds <= 0:
        raise ValueError("History durations must be positive")
    if minimum_history_seconds > history_seconds:
        raise ValueError("minimum_history_seconds cannot exceed history_seconds")

    output = features.copy()
    timestamps = pd.Series(
        pd.to_datetime(output["event_ts"], utc=True), index=output.index
    )
    metadata = _catalogue(catalogue)
    policies = feature_policy(catalogue).set_index("feature")
    history_exclusions = ("__difference", "_difference", "__transition", "__reset")
    health_features = [
        name
        for name in output.columns
        if name in policies.index
        and policies.at[name, "role"] == "asset_health"
        and not name.endswith(history_exclusions)
    ]

    for name in health_features:
        metric_id = str(policies.at[name, "metric_id"])
        cadence = pd.to_numeric(
            pd.Series([metadata.at[metric_id, "expected_cadence_seconds"]]),
            errors="coerce",
        ).iloc[0]
        if pd.isna(cadence) or float(cadence) <= 0:
            continue
        minimum_rows = max(3, math.ceil(minimum_history_seconds / float(cadence)))
        gap = timestamps.diff().dt.total_seconds().gt(float(cadence) * gap_tolerance)
        segments = gap.cumsum()
        result = pd.Series(np.nan, index=output.index, dtype=float)
        floor = float(policies.at[name, "minimum_scale"])

        for _, indices in output.groupby(segments, sort=False).groups.items():
            values = pd.Series(
                pd.to_numeric(output.loc[indices, name], errors="coerce").to_numpy(),
                index=pd.DatetimeIndex(timestamps.loc[indices]),
                dtype=float,
            )
            trailing = values.rolling(
                pd.Timedelta(seconds=float(history_seconds)),
                closed="left",
                min_periods=minimum_rows,
            )
            centre = trailing.median()
            scale = (trailing.quantile(0.75) - trailing.quantile(0.25)) / 1.349
            scale = scale.where(scale.gt(floor), floor)
            result.loc[indices] = ((values - centre) / scale).to_numpy()

        output[f"{name}__history_z"] = result
    return output


def orient_residuals(
    residuals: pd.DataFrame,
    directions: dict[str, str],
) -> pd.DataFrame:
    """Orient residuals so larger always means more anomalous."""

    output = pd.DataFrame(index=residuals.index)
    for name in residuals.columns:
        direction = directions.get(name, "two_sided")
        if direction not in VALID_DIRECTIONS:
            raise ValueError(f"Unsupported direction {direction!r} for {name}")
        values = pd.to_numeric(residuals[name], errors="coerce")
        output[name] = (
            values
            if direction == "high_bad"
            else -values if direction == "low_bad" else values.abs()
        )
    return output


def fit_empirical_tail_reference(
    residuals: pd.DataFrame,
    directions: dict[str, str],
    *,
    minimum_observations: int = 30,
) -> dict[str, np.ndarray]:
    """Freeze sorted calibration tails for finite-sample empirical p-values."""

    oriented = orient_residuals(residuals, directions)
    reference = {}
    for name in oriented:
        values = oriented[name].to_numpy(dtype=float)
        values = np.sort(values[np.isfinite(values)])
        if len(values) >= int(minimum_observations):
            reference[name] = values
    if not reference:
        raise ValueError(
            "No feature has enough calibration observations for tail evidence"
        )
    return reference


def empirical_tail_evidence(
    residuals: pd.DataFrame,
    reference: dict[str, np.ndarray],
    directions: dict[str, str],
) -> pd.DataFrame:
    """Return ``-log10(p)`` evidence using calibration survival ranks.

    The add-one estimate prevents zero p-values.  Consequently evidence is
    naturally capped by the calibration sample size rather than by a chosen
    numerical constant.
    """

    oriented = orient_residuals(residuals, directions)
    evidence = pd.DataFrame(np.nan, index=residuals.index, columns=residuals.columns)
    for name, calibration in reference.items():
        if name not in oriented:
            continue
        calibration = np.asarray(calibration, dtype=float)
        values = oriented[name].to_numpy(dtype=float)
        valid = np.isfinite(values)
        exceedances = len(calibration) - np.searchsorted(
            calibration, values[valid], side="left"
        )
        p_value = (exceedances + 1.0) / (len(calibration) + 1.0)
        evidence.loc[valid, name] = -np.log10(p_value)
    return evidence.astype(float)


def directional_cusum(
    residuals: pd.DataFrame,
    directions: dict[str, str],
    *,
    allowance: float = 0.5,
    reset_before=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Score directional CUSUMs and reset state after gaps or missing values."""

    if allowance < 0:
        raise ValueError("allowance cannot be negative")
    names = list(residuals.columns)
    values = residuals.to_numpy(dtype=float)
    reset_before = (
        np.zeros(len(values), dtype=bool)
        if reset_before is None
        else np.asarray(reset_before, dtype=bool)
    )
    if len(reset_before) != len(values):
        raise ValueError("reset_before must align with residual rows")

    positive = np.zeros(len(names), dtype=float)
    negative = np.zeros(len(names), dtype=float)
    scores = np.full(len(values), np.nan)
    leading = np.full(len(values), None, dtype=object)

    for row_number, row in enumerate(values):
        if reset_before[row_number]:
            positive.fill(0)
            negative.fill(0)
        valid = np.isfinite(row)
        positive[~valid] = 0
        negative[~valid] = 0
        positive[valid] = np.maximum(0, positive[valid] + row[valid] - float(allowance))
        negative[valid] = np.maximum(0, negative[valid] - row[valid] - float(allowance))

        combined = np.zeros(len(names), dtype=float)
        for index, name in enumerate(names):
            direction = directions.get(name, "two_sided")
            if direction not in VALID_DIRECTIONS:
                raise ValueError(f"Unsupported direction {direction!r} for {name}")
            combined[index] = (
                positive[index]
                if direction == "high_bad"
                else (
                    negative[index]
                    if direction == "low_bad"
                    else max(positive[index], negative[index])
                )
            )
        if valid.any():
            position = int(np.argmax(np.where(valid, combined, -np.inf)))
            scores[row_number] = combined[position]
            leading[row_number] = names[position]
    return scores, leading
