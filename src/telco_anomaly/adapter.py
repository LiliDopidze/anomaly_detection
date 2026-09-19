"""Map only the fields the optical-change model needs; do not infer units."""

import numpy as np
import pandas as pd

DEFAULT_COLUMNS = {
    name: name
    for name in (
        "timestamp_utc",
        "ont_id",
        "rx_power_dbm",
        "olt_rx_power_dbm",
    )
}


def adapt(native, columns=None, units=None, timezone="UTC"):
    """Return timestamps, identifiers and two optical powers in dBm.

    `columns` maps canonical names to source names. Upstream power is optional.
    Extra source columns are never passed to the model. Missing/invalid readings
    remain missing; observations are never forward-filled.
    """
    columns = {**DEFAULT_COLUMNS, **(columns or {})}
    units = units or {"rx_power_dbm": "dBm", "olt_rx_power_dbm": "dBm"}
    if set(columns) != set(DEFAULT_COLUMNS):
        raise ValueError("Mapping must contain only the four canonical fields")
    data = pd.DataFrame(index=native.index)
    timestamps = pd.to_datetime(native[columns["timestamp_utc"]], errors="raise")
    if timestamps.isna().any():
        raise ValueError("Missing timestamps")
    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize(
            timezone, ambiguous="raise", nonexistent="raise"
        )
    data["timestamp_utc"] = timestamps.dt.tz_convert("UTC")
    ids = native[columns["ont_id"]]
    if ids.isna().any() or ids.astype(str).str.strip().eq("").any():
        raise ValueError("Missing device identifiers")
    data["ont_id"] = ids.astype(str)
    for metric in ("rx_power_dbm", "olt_rx_power_dbm"):
        source = columns[metric]
        if source not in native and metric == "olt_rx_power_dbm":
            data[metric] = np.nan
            continue
        values = pd.to_numeric(native[source], errors="raise")
        unit = units.get(metric)
        if unit in {"mW", "W"}:
            values = values * (1000 if unit == "W" else 1)
            values = 10 * np.log10(values.where(values > 0))
        elif unit != "dBm":
            raise ValueError(f"Declare dBm, mW or W for {metric}")
        data[metric] = values.where(np.isfinite(values))
    if data.duplicated(["ont_id", "timestamp_utc"]).any():
        raise ValueError("Duplicate device/timestamp observations")
    return data.sort_values(["ont_id", "timestamp_utc"]).reset_index(drop=True)


def quality_report(data, cadence_minutes=15):
    """Plain diagnostics; missing telemetry is not a healthy observation."""
    rows = []
    for entity, group in data.groupby("ont_id"):
        gaps = group.timestamp_utc.diff().dt.total_seconds() / 60
        rows.append(
            {
                "ont_id": entity,
                "rows": len(group),
                "downstream_missing": float(group.rx_power_dbm.isna().mean()),
                "upstream_missing": float(group.olt_rx_power_dbm.isna().mean()),
                "gaps": int(gaps.gt(1.5 * cadence_minutes).sum()),
                "largest_gap_minutes": float(gaps.max()) if len(group) > 1 else 0,
            }
        )
    return pd.DataFrame(rows)
