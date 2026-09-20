"""Readable native-data diagnostics; no model fitting or canonical adaptation."""

import numpy as np
import pandas as pd
from .generator import GeneratorConfig


def missingness_report(native: pd.DataFrame) -> pd.DataFrame:
    """Per-device measurement counts and longest missing run on supplied rows.

    Call after checking cadence: absent timestamp rows must not be mistaken for
    complete coverage. FEC's first missing interval is expected by construction.
    """
    records = []
    metrics = native.columns.difference(["time", "device"])
    for entity, group in native.groupby("device", sort=True):
        group = group.sort_values("time")
        for metric in metrics:
            missing = group[metric].isna()
            runs = missing.groupby((~missing).cumsum()).sum()
            records.append(
                {
                    "entity_id": entity,
                    "measurement": metric,
                    "rows": len(group),
                    "observed": int((~missing).sum()),
                    "missing_fraction": float(missing.mean()),
                    "longest_missing_intervals": int(runs.max()),
                    "unique_observed_values": group[metric].nunique(),
                }
            )
    return pd.DataFrame(records)


def healthy_statistics(healthy: pd.DataFrame, config: GeneratorConfig) -> pd.DataFrame:
    """Compare path-loss residuals with the configured AR(1) + sensor process.

    Remove measured shared Tx and fit a daily harmonic separately per ONT.
    Missing rows remain in place when estimating adjacent-sample correlation.
    Tolerances are diagnostic bands, not hypothesis-test confidence intervals.
    """
    expected_var = config.noise_db**2 + config.sensor_noise_db**2
    phi = np.exp(-config.interval_minutes / (60 * config.correlation_hours))
    expected_rho = phi * config.noise_db**2 / expected_var
    records = []
    for entity, group in healthy.groupby("device", sort=True):
        group = group.sort_values("time")
        hours = (group.time - group.time.min()).dt.total_seconds() / 3600
        phase = 2 * np.pi * hours.to_numpy() / 24
        design = np.column_stack([np.ones(len(group)), np.sin(phase), np.cos(phase)])
        path_loss = (group.olt_tx_dbm - group.rx_dbm).to_numpy()
        valid = np.isfinite(path_loss)
        if valid.sum() < 100:
            continue
        coefficients = np.linalg.lstsq(design[valid], path_loss[valid], rcond=None)[0]
        residual = pd.Series(path_loss - design @ coefficients)
        adjacent = group.time.diff().eq(pd.Timedelta(minutes=config.interval_minutes))
        paired = residual.shift().where(adjacent.to_numpy())
        amplitude = float(np.hypot(*coefficients[1:]))
        variance = float(residual.var())
        rho = float(residual.corr(paired))
        records.append(
            {
                "entity_id": entity,
                "daily_amplitude_db": amplitude,
                "expected_amplitude_db": config.daily_amplitude_db,
                "residual_variance": variance,
                "expected_variance": expected_var,
                "residual_lag1": rho,
                "expected_lag1": expected_rho,
                "amplitude_in_band": abs(amplitude - config.daily_amplitude_db)
                <= max(0.05, config.daily_amplitude_db * 0.2),
                "variance_in_band": abs(variance / expected_var - 1) <= 0.25,
                "correlation_in_band": abs(rho - expected_rho) <= 0.1,
            }
        )
    return pd.DataFrame(records)


def development_faults(
    truth: pd.DataFrame,
    native: pd.DataFrame,
    end: pd.Timestamp,
    interval_minutes: int,
) -> pd.DataFrame:
    """Only faults fully inside development; report conservative warning windows.

    Opportunity proxy requires 3 consecutive observed Rx polls and 30 minutes
    before impact. This is a data diagnostic, not detector recall.
    """
    faults = truth.loc[(truth.onset_time < end) & (truth.end_time <= end)].copy()
    for column in ("onset_time", "end_time", "observable_onset_time", "impact_time"):
        faults[column] = pd.to_datetime(faults[column], utc=True)
    faults["duration_hours"] = (
        faults.end_time - faults.onset_time
    ).dt.total_seconds() / 3600
    faults["warning_minutes"] = (
        faults.impact_time - faults.observable_onset_time
    ).dt.total_seconds() / 60
    opportunities = []
    step = pd.Timedelta(minutes=interval_minutes)
    for fault in faults.itertuples():
        view = native.loc[
            native.device.eq(fault.entity_id)
            & native.time.ge(fault.observable_onset_time)
            & native.time.le(fault.impact_time - pd.Timedelta(minutes=30))
        ].sort_values("time")
        observed = view.rx_dbm.notna()
        consecutive = view.time.diff().eq(step)
        available = (
            observed
            & observed.shift(1, fill_value=False)
            & observed.shift(2, fill_value=False)
            & consecutive
            & consecutive.shift(1, fill_value=False)
        )
        opportunities.append(bool(available.any()))
    faults["warning_opportunity_proxy"] = opportunities
    return faults


def fault_contrasts(native: pd.DataFrame, faults: pd.DataFrame) -> pd.DataFrame:
    """Development-only effect and missingness screens, not causal estimates.

    Compare each fault with its preceding 24 hours. Daily phase and small samples
    can affect these contrasts, so review plots rather than imposing a pass mark.
    """
    records = []
    for fault in faults.itertuples():
        entity = native.loc[native.device.eq(fault.entity_id)]
        before = entity.loc[
            entity.time.ge(fault.onset_time - pd.Timedelta(days=1))
            & entity.time.lt(fault.onset_time),
            "rx_dbm",
        ]
        during = entity.loc[
            entity.time.ge(fault.onset_time) & entity.time.lt(fault.end_time),
            "rx_dbm",
        ]
        scale = before.std()
        records.append(
            {
                "fault_id": fault.fault_id,
                "fault_type": fault.fault_type,
                "median_change_db": during.median() - before.median(),
                "low_tail_change_db": during.quantile(0.05) - before.median(),
                "std_ratio": during.std() / scale if scale > 0 else np.nan,
                "missing_before": before.isna().mean(),
                "missing_during": during.isna().mean(),
            }
        )
    return pd.DataFrame(records)


def canonical_statistics(
    telemetry: pd.DataFrame, interval_minutes: int
) -> pd.DataFrame:
    """Healthy canonical EDA with a chronological seasonal holdout.

    Compare constant, daily, and daily+weekly regression within the supplied
    training period. R2 here measures holdout improvement over a fitted constant,
    not a significance test. No missing values are interpolated.
    """
    records = []
    for (entity, metric), group in telemetry.groupby(["entity_id", "metric_name"]):
        group = group.sort_values("timestamp")
        series = group.set_index("timestamp").value
        step = pd.Timedelta(minutes=interval_minutes)
        grid = pd.date_range(series.index.min(), series.index.max(), freq=step)
        series = series.reindex(grid)
        days = (series.index - series.index[0]).total_seconds().to_numpy() / 86400
        values = series.to_numpy()
        finite = np.isfinite(values)
        cut = days[-1] * 0.7
        train, test = finite & (days < cut), finite & (days >= cut)
        row = {
            "entity_id": entity,
            "metric_name": metric,
            "observed": int(finite.sum()),
            "missing_fraction": 1 - finite.mean(),
            "mean": series.mean(),
            "std": series.std(),
            "minimum": series.min(),
            "maximum": series.max(),
            "lag1": series.autocorr(1) if series.nunique() > 1 else np.nan,
            "lag_daily": (
                series.autocorr(max(1, round(1440 / interval_minutes)))
                if series.nunique() > 1
                else np.nan
            ),
            "lag_weekly": (
                series.autocorr(max(1, round(10080 / interval_minutes)))
                if days[-1] >= 14 and series.nunique() > 1
                else np.nan
            ),
        }
        row.update(
            daily_amplitude=np.nan,
            daily_holdout_r2=np.nan,
            daily_weekly_holdout_r2=np.nan,
        )
        if train.sum() >= 100 and test.sum() >= 30:
            phase = 2 * np.pi * days
            design = np.column_stack(
                [
                    np.ones(len(days)),
                    np.sin(phase),
                    np.cos(phase),
                    np.sin(phase / 7),
                    np.cos(phase / 7),
                ]
            )
            denominator = np.mean((values[test] - values[train].mean()) ** 2)
            for columns, key in [
                (3, "daily_holdout_r2"),
                (5, "daily_weekly_holdout_r2"),
            ]:
                if columns == 5 and cut < 21:
                    continue
                coefficients = np.linalg.lstsq(
                    design[train, :columns], values[train], rcond=None
                )[0]
                errors = values[test] - design[test, :columns] @ coefficients
                row[key] = (
                    1 - np.mean(errors**2) / denominator if denominator > 0 else np.nan
                )
                if columns == 3:
                    row["daily_amplitude"] = float(np.hypot(*coefficients[1:]))
        records.append(row)
    return pd.DataFrame(records)


def dependence_reports(
    telemetry: pd.DataFrame, interval_minutes: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training EDA: per-entity relationships before/after daily-pattern removal.

    Pairwise complete observations only; preserve the regular grid for time lags.
    Residual fits are descriptive, never passed to the detector. No p-value or
    automatic feature-selection rule is implied by a correlation coefficient.
    """
    from .features import daily_design

    correlations, autocorrelations = [], []
    for entity, group in telemetry.groupby("entity_id"):
        wide = group.pivot(index="timestamp", columns="metric_name", values="value")
        grid = pd.date_range(
            wide.index.min(), wide.index.max(), freq=f"{interval_minutes}min"
        )
        wide = wide.reindex(grid)
        design = daily_design(pd.Series(grid))
        residual = pd.DataFrame(np.nan, index=grid, columns=wide.columns)
        for metric in wide:
            valid = wide[metric].notna()
            if valid.sum() < 4 or wide.loc[valid, metric].nunique() < 2:
                continue
            fit = np.linalg.lstsq(design[valid], wide.loc[valid, metric], rcond=None)[0]
            values = wide[metric] - design @ fit
            tolerance = 100 * np.finfo(float).eps * max(1, wide[metric].abs().max())
            if values.std() > tolerance:
                residual[metric] = values
        for view, frame in (("raw", wide), ("daily_residual", residual)):
            counts = frame.notna().astype(int).T @ frame.notna().astype(int)
            pearson = frame.corr(method="pearson", min_periods=3)
            spearman = frame.corr(method="spearman", min_periods=3)
            for i, left in enumerate(frame.columns):
                for right in frame.columns[i + 1 :]:
                    correlations.append(
                        dict(
                            entity_id=entity,
                            view=view,
                            left=left,
                            right=right,
                            paired_observations=int(counts.loc[left, right]),
                            pearson=pearson.loc[left, right],
                            spearman=spearman.loc[left, right],
                        )
                    )
            lags = sorted(
                {
                    1,
                    *[
                        max(1, round(m / interval_minutes))
                        for m in (30, 60, 360, 720, 1440, 2880, 10080)
                    ],
                }
            )
            for metric in frame:
                for lag in lags:
                    pair = pd.concat([frame[metric], frame[metric].shift(lag)], axis=1)
                    pair = pair.dropna()
                    coefficient = np.nan
                    if len(pair) >= 3 and (pair.nunique() > 1).all():
                        coefficient = pair.iloc[:, 0].corr(pair.iloc[:, 1])
                    autocorrelations.append(
                        dict(
                            entity_id=entity,
                            view=view,
                            metric_name=metric,
                            lag_intervals=lag,
                            lag_minutes=lag * interval_minutes,
                            paired_observations=len(pair),
                            autocorrelation=coefficient,
                        )
                    )
    return pd.DataFrame(correlations), pd.DataFrame(autocorrelations)
