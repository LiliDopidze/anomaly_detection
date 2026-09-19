"""Cadence, gap, seasonal and topology features from observable inputs only."""

import numpy as np
import pandas as pd


def regularize(data, inventory, start, end, cadence_seconds):
    """Explicit missing polls and effective-dated topology; never interpolate gauges."""
    inventory = inventory.copy()
    for key in ("valid_from", "valid_to"):
        inventory[key] = inventory[key].astype("datetime64[ns, UTC]")
    outputs = []
    for entity, memberships in inventory.groupby("ont_id"):
        times = pd.date_range(start, end, freq=f"{cadence_seconds}s", inclusive="left")
        grid = pd.DataFrame({"timestamp_utc": times, "ont_id": entity})
        group = data.loc[data.ont_id.eq(entity)].copy()
        group["reported"] = True
        grid = grid.merge(group, how="left", on=["ont_id", "timestamp_utc"])
        grid["reported"] = grid.reported.eq(True)
        grid = pd.merge_asof(
            grid.sort_values("timestamp_utc"),
            memberships.drop(columns="ont_id").sort_values("valid_from"),
            left_on="timestamp_utc",
            right_on="valid_from",
            direction="backward",
        )
        grid = grid.loc[grid.timestamp_utc.lt(grid.valid_to)].copy()
        observed = grid.loc[grid.reported]
        elapsed = observed.timestamp_utc.diff().dt.total_seconds()
        uptime_delta = observed.uptime_s.diff()
        reboot = observed.reboot_count.diff().gt(0) | uptime_delta.lt(0)
        gaps = elapsed.gt(cadence_seconds * 1.5)
        kinds = pd.Series("continuous", index=observed.index)
        kinds[gaps] = "indeterminate"
        kinds[gaps & (uptime_delta - elapsed).abs().le(cadence_seconds)] = "collection"
        kinds[reboot] = "device_restart"
        grid["gap_kind"] = "unobserved"
        grid.loc[observed.index, "gap_kind"] = kinds
        grid["episode"] = grid.gap_kind.eq("device_restart").cumsum()
        outputs.append(grid)
    if not outputs:
        raise ValueError("No active inventory")
    return pd.concat(outputs, ignore_index=True)


def fit_seasonality(data, train_end, timezone):
    """Approve daily profiles using only fit data, with minimum repeated-day support."""
    fit = data.loc[data.timestamp_utc.lt(train_end)].copy()
    fit["hour"] = fit.timestamp_utc.dt.tz_convert(timezone).dt.hour
    rows, templates = [], {}
    for metric in (
        "temperature_c",
        "throughput_mbps",
        "rx_power_dbm",
        "olt_rx_power_dbm",
    ):
        evidence = []
        for entity, group in fit.groupby("ont_id"):
            valid = group.dropna(subset=[metric])
            profile = valid.groupby("hour")[metric].mean()
            variance = valid[metric].var()
            residual = valid[metric] - valid.hour.map(profile)
            share = max(0.0, 1 - residual.var() / variance) if variance > 0 else 0.0
            days = valid.timestamp_utc.dt.floor("D").nunique()
            supported = days >= 7 and len(profile) == 24
            evidence.append((entity, share, supported, profile))
            rows.append(
                {
                    "ont_id": entity,
                    "metric": metric,
                    "daily_share": share,
                    "days": days,
                    "supported": supported,
                }
            )
        supported = [row for row in evidence if row[2]]
        approved = (
            len(supported) >= 0.5 * max(1, len(evidence))
            and np.mean([row[1] >= 0.2 for row in supported]) >= 0.5
        )
        if approved:
            templates[metric] = {
                entity: profile.to_dict() for entity, _, ok, profile in evidence if ok
            }
    decisions = {
        "timezone": timezone,
        "templates": templates,
        "approved_daily_metrics": list(templates),
        "fit_end": str(train_end),
        "weekly": "not_fitted",
        "annual": "insufficient_cycles",
    }
    return decisions, pd.DataFrame(rows)


def extended_features(grid, base_features, decisions):
    """Add transparent residual, hurdle, directional and availability evidence."""
    features = base_features.copy()
    for metric in decisions["approved_daily_metrics"]:
        hours = grid.timestamp_utc.dt.tz_convert(decisions["timezone"]).dt.hour
        adjustment = pd.Series(0.0, index=grid.index)
        for entity, profiles in decisions["templates"][metric].items():
            mask = grid.ont_id.eq(entity)
            adjustment.loc[mask] = (
                hours.loc[mask].map({int(k): v for k, v in profiles.items()}).fillna(0)
            )
        features[metric + "__seasonal_residual"] = grid[metric] - adjustment
    features["fec__nonzero"] = grid.fec_count.gt(0).where(grid.fec_count.notna())
    # All directional changes use past medians; current row is excluded.
    for metric in ("rx_power_dbm", "olt_rx_power_dbm"):
        values = []
        for _, group in grid.groupby("ont_id", sort=False):
            series = group.set_index("timestamp_utc")[metric]
            values.extend(
                (
                    series
                    - series.rolling("24h", closed="left", min_periods=12).median()
                ).to_numpy()
            )
        features[metric + "__delta"] = values
    features["optical__asymmetry"] = (
        features.rx_power_dbm__delta - features.olt_rx_power_dbm__delta
    )
    return features


def channel_scores(grid, features, baseline_scores, policy):
    scores = baseline_scores.copy()
    # Restart-aware one-sided CUSUM. Missing samples hold state, emit no score.
    drift = np.full(len(grid), np.nan)
    for _, indices in grid.groupby("ont_id", sort=False).groups.items():
        state = 0.0
        for index in indices:
            if grid.at[index, "gap_kind"] == "device_restart":
                state = 0.0
            value = features.at[index, "rx_power_dbm__z24h"]
            if np.isfinite(value):
                state = max(0.0, state + value - policy["cusum_allowance"])
                drift[index] = state
    scores["drift"] = drift
    scratch = grid[
        ["timestamp_utc", "ont_id", "splitter_l2", "olt_id", "reported"]
    ].copy()
    scratch["residual"] = features.rx_power_dbm__delta
    group_keys = ["splitter_l2", "timestamp_utc"]
    peers = scratch.groupby(group_keys)
    count = peers.residual.transform("count")
    median = peers.residual.transform("median")
    scores["peer"] = (
        ((median - scratch.residual) / 0.15).clip(lower=0).where(count >= 3)
    )
    scores["common_mode"] = (-median / 0.15).clip(lower=0).where(count >= 3)
    expected = peers.reported.transform("size")
    reporting = peers.reported.transform("sum")
    olt = scratch.groupby(["olt_id", "timestamp_utc"]).reported
    rest_total = olt.transform("size") - expected
    rest_reporting = olt.transform("sum") - reporting
    rest_fraction = rest_reporting / rest_total.replace(0, np.nan)
    scores["silence"] = (1 - reporting / expected).where(
        (expected >= policy["minimum_group_size"])
        & rest_fraction.ge(policy["rest_collector_reporting"]),
        0.0,
    )
    sensitivity = grid.get("rx_sensitivity_dbm", pd.Series(np.nan, index=grid.index))
    scores["margin"] = (
        sensitivity + policy["margin_warning_db"] - grid.rx_power_dbm
    ).clip(lower=0)
    scores["rules"] = (-policy["static_rx_limit_abs_dbm"] - grid.rx_power_dbm).clip(
        lower=0
    )
    return scores
