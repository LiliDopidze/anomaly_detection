"""Reviewed source mappings into a small, model-safe PON data contract.

No directory discovery, inferred units, fault-label reads, or silent resampling.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import yaml

from .synthetic import METRICS, sha256

PACK_FILES = {
    "telemetry.parquet",
    "inventory.parquet",
    "events.parquet",
    "quality.parquet",
    "manifest.json",
    "adapter_audit.json",
}
UNITS = {
    "rx_power_dbm": "dBm",
    "olt_rx_power_dbm": "dBm",
    "tx_power_dbm": "dBm",
    "temperature_c": "C",
    "bias_current_ma": "mA",
    "ber": "ratio",
    "fec_count": "codewords",
    "throughput_mbps": "Mbps",
    "uptime_s": "s",
    "reboot_count": "count",
}


def utc(values, timezone):
    parsed = pd.to_datetime(values, errors="raise")
    if parsed.isna().any():
        raise ValueError("Missing timestamps are not allowed")
    if parsed.dt.tz is None:
        parsed = parsed.dt.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
    return parsed.dt.tz_convert("UTC")


def convert_units(values, source, target):
    if source == target:
        return values
    factors = {
        ("W", "mW"): 1000,
        ("A", "mA"): 1000,
        ("bps", "Mbps"): 1e-6,
        ("ms", "s"): 0.001,
        ("percent", "ratio"): 0.01,
    }
    if (source, target) in factors:
        return values * factors[source, target]
    if target == "dBm" and source in {"mW", "W"}:
        power = values * (1000 if source == "W" else 1)
        return 10 * np.log10(power.where(power > 0))
    if source == "K" and target == "C":
        return values - 273.15
    raise ValueError(f"Unreviewed unit conversion: {source} -> {target}")


def adapt_frame(native, mapping, cadence_seconds):
    """Return canonical wide telemetry and explicit per-cell quality codes."""
    columns = mapping["columns"]
    classified = {mapping["timestamp"], mapping["entity"], *mapping.get("ignored", [])}
    classified.update(spec["source"] for spec in columns.values())
    vendor_column = mapping.get("vendor_column")
    if vendor_column:
        classified.add(vendor_column)
    if unknown := set(native) - classified:
        raise ValueError(f"Unreviewed source columns: {sorted(unknown)}")
    if set(columns) != set(METRICS):
        raise ValueError("Mapping must declare every canonical metric")
    if native[mapping["entity"]].isna().any():
        raise ValueError("Missing entity identifiers")
    data = pd.DataFrame(
        {
            "timestamp_utc": utc(native[mapping["timestamp"]], mapping["timezone"]),
            "ont_id": native[mapping["entity"]].astype(str),
        }
    )
    if data.ont_id.str.strip().eq("").any():
        raise ValueError("Empty entity identifiers")
    if data.duplicated(["ont_id", "timestamp_utc"]).any():
        raise ValueError("Duplicate entity/timestamp observations")
    quality = data.copy()
    for metric, base in columns.items():
        if base["source"] not in native:
            if base.get("required", True):
                raise ValueError(f"Missing required source field: {base['source']}")
            data[metric], quality[metric] = np.nan, "unavailable"
            continue
        raw = pd.to_numeric(native[base["source"]], errors="raise")
        values = pd.Series(np.nan, index=native.index)
        kinds = pd.Series(base["kind"], index=native.index)
        groups = [(None, native.index)]
        if vendor_column:
            groups = native.groupby(vendor_column, dropna=False).groups.items()
        for vendor, index in groups:
            spec = {**base, **base.get("overrides", {}).get(str(vendor), {})}
            values.loc[index] = convert_units(
                raw.loc[index], spec["unit"], UNITS[metric]
            ) * spec.get("scale", 1)
            kinds.loc[index] = spec["kind"]
        if not kinds.isin({"gauge", "interval", "cumulative"}).all():
            raise ValueError(f"Invalid measurement kind for {metric}")
        invalid = values.notna() & ~np.isfinite(values)
        invalid |= raw.notna() & values.isna()
        if metric not in {
            "rx_power_dbm",
            "olt_rx_power_dbm",
            "tx_power_dbm",
            "temperature_c",
        }:
            invalid |= values < 0
        if metric == "ber":
            invalid |= values > 1
        if metric in {"fec_count", "reboot_count"}:
            invalid |= values.notna() & values.ne(np.floor(values))
        values = values.mask(invalid)
        codes = pd.Series("valid", index=native.index)
        codes[raw.isna()] = "missing"
        codes[invalid] = "invalid"
        # FEC's canonical meaning is interval corrected codewords. Never infer it.
        if metric == "fec_count":
            temporary = data.assign(value=values, kind=kinds).sort_values(
                ["ont_id", "timestamp_utc"]
            )
            grouped = temporary.groupby("ont_id")
            delta = grouped.value.diff()
            elapsed = grouped.timestamp_utc.diff().dt.total_seconds()
            cumulative = temporary.kind.eq("cumulative")
            discontinuity = elapsed.ne(cadence_seconds) | delta.lt(0) | delta.isna()
            discontinuity |= grouped.kind.shift().ne(temporary.kind)
            values.loc[temporary.index[cumulative]] = delta[cumulative].mask(
                discontinuity[cumulative]
            )
            codes.loc[temporary.index[cumulative & discontinuity]] = (
                "counter_discontinuity"
            )
        data[metric], quality[metric] = values, codes
    ordered = data.sort_values(["ont_id", "timestamp_utc"])
    grouped = ordered.groupby("ont_id")
    restarted = grouped.uptime_s.diff().lt(0) | grouped.reboot_count.diff().gt(0)
    fec_spec = columns["fec_count"]
    cumulative = pd.Series(fec_spec["kind"] == "cumulative", index=native.index)
    if vendor_column:
        for vendor, override in fec_spec.get("overrides", {}).items():
            cumulative.loc[native[vendor_column].astype(str).eq(vendor)] = (
                override.get("kind", fec_spec["kind"]) == "cumulative"
            )
    reset_indices = ordered.index[restarted & cumulative.loc[ordered.index]]
    data.loc[reset_indices, "fec_count"] = np.nan
    quality.loc[reset_indices, "fec_count"] = "counter_discontinuity"
    order = data.sort_values(["ont_id", "timestamp_utc"]).index
    return data.loc[order].reset_index(drop=True), quality.loc[order].reset_index(
        drop=True
    )


def adapt_inventory(native, config, start, end):
    spec = config["inventory"]
    fields = spec["columns"]
    if unknown := set(native) - set(fields.values()) - set(spec.get("ignored", [])):
        raise ValueError(f"Unreviewed inventory columns: {sorted(unknown)}")
    inventory = native[list(fields.values())].rename(
        columns={v: k for k, v in fields.items()}
    )
    required = ["ont_id", "olt_id", "pon_port", "splitter_l1", "splitter_l2"]
    if inventory[required].isna().any().any():
        raise ValueError("Missing topology memberships")
    for key in required:
        inventory[key] = inventory[key].astype(str)
    for key, default in [("valid_from", start), ("valid_to", end)]:
        inventory[key] = (
            utc(inventory[key], config["timezone"]) if key in inventory else default
        )
    if (inventory.valid_from >= inventory.valid_to).any():
        raise ValueError("Invalid membership interval")
    ordered = inventory.sort_values(["ont_id", "valid_from"])
    prior_end = ordered.groupby("ont_id").valid_to.shift()
    if (ordered.valid_from < prior_end).any():
        raise ValueError("Overlapping effective-dated memberships")
    return inventory


def adapt_events(native, config, entities):
    fields = ["timestamp_utc", "ont_id", "family"]
    if native is None:
        return pd.DataFrame(
            {
                "timestamp_utc": pd.Series(dtype="datetime64[ns, UTC]"),
                "ont_id": pd.Series(dtype=str),
                "family": pd.Series(dtype=str),
            }
        )
    spec = config["events"]
    if set(native) != {spec["timestamp"], spec["entity"], spec["code"]}:
        raise ValueError("Unreviewed event columns")
    family = native[spec["code"]].map(spec["codes"])
    if (
        family.isna().any()
        or not family.isin({"onu_power_loss", "loss_of_signal"}).all()
    ):
        raise ValueError("Unreviewed alarm code or canonical family")
    events = pd.DataFrame(
        {
            "timestamp_utc": utc(native[spec["timestamp"]], config["timezone"]),
            "ont_id": native[spec["entity"]].astype(str),
            "family": family,
        }
    )
    if not set(events.ont_id) <= set(entities):
        raise ValueError("Events reference unknown entities")
    return events[fields].sort_values("timestamp_utc").reset_index(drop=True)


def write_pack(telemetry, inventory, output, mapping, metadata, events=None):
    """Adapt explicitly supplied tables. Truth is neither an argument nor a dependency."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    for name in ("days", "sample_minutes", "n_onts"):
        if type(metadata[name]) is not int or metadata[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if 1440 % metadata["sample_minutes"]:
        raise ValueError("Cadence must divide one day")
    start = pd.Timestamp(metadata["start"]).tz_convert("UTC")
    end = start + pd.Timedelta(days=metadata["days"])
    cadence = metadata["sample_minutes"] * 60
    data, quality = adapt_frame(telemetry, mapping, cadence)
    topology = adapt_inventory(inventory, mapping, start, end)
    if not set(data.ont_id) <= set(topology.ont_id):
        raise ValueError("Measurements reference unknown entities")
    if not data.timestamp_utc.between(start, end, inclusive="left").all():
        raise ValueError("Measurements outside declared observation window")
    offset = (data.timestamp_utc - start).dt.total_seconds()
    if offset.mod(cadence).ne(0).any():
        raise ValueError(
            "Off-grid measurements: declare cadence; do not silently resample"
        )
    if topology.ont_id.nunique() != metadata["n_onts"]:
        raise ValueError("Declared population differs from inventory")
    for entity, rows in data.groupby("ont_id"):
        memberships = topology.loc[topology.ont_id.eq(entity)].copy()
        memberships["valid_from"] = memberships.valid_from.astype("datetime64[ns, UTC]")
        joined = pd.merge_asof(
            rows[["timestamp_utc"]].sort_values("timestamp_utc"),
            memberships[["valid_from", "valid_to"]].sort_values("valid_from"),
            left_on="timestamp_utc",
            right_on="valid_from",
            direction="backward",
        )
        if not joined.timestamp_utc.lt(joined.valid_to).all():
            raise ValueError("Observation outside effective inventory membership")
    canonical_events = adapt_events(events, mapping, topology.ont_id)
    output.mkdir(parents=True)
    for name, table in [
        ("telemetry", data),
        ("inventory", topology),
        ("quality", quality),
        ("events", canonical_events),
    ]:
        table.to_parquet(output / f"{name}.parquet", index=False)
    audit = {metric: quality[metric].value_counts().to_dict() for metric in METRICS}
    (output / "adapter_audit.json").write_text(json.dumps(audit, indent=2))
    manifest = {
        "contract": "pon-observations-v1",
        "config": metadata,
        "mapping": mapping,
        "adapter_sha256": sha256(__file__),
        "truth_in_pack": False,
        "files": {p.name: sha256(p) for p in output.iterdir()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return output


def verify_pack(path):
    path = Path(path)
    if any(p.is_symlink() for p in path.rglob("*")):
        raise ValueError("Symlinks are not allowed in model packs")
    if {p.name for p in path.iterdir()} != PACK_FILES:
        raise ValueError("Unexpected files in model pack; truth must be separate")
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["contract"] != "pon-observations-v1":
        raise ValueError("Unknown canonical contract")
    if set(manifest["files"]) != PACK_FILES - {"manifest.json"}:
        raise ValueError("Incomplete pack manifest")
    for name, digest in manifest["files"].items():
        if name not in PACK_FILES or sha256(path / name) != digest:
            raise ValueError(f"Model input changed: {name}")
    return manifest


def load_observations(dataset, include_holdout=False):
    import duckdb

    dataset = Path(dataset)
    manifest = verify_pack(dataset)
    cfg = manifest["config"]
    cutoff = pd.Timestamp(cfg["start"]) + pd.Timedelta(days=cfg["days"] * 0.85)
    with duckdb.connect() as connection:
        query = "SELECT * FROM read_parquet(?)"
        args = [str(dataset / "telemetry.parquet")]
        if not include_holdout:
            query += " WHERE timestamp_utc < ?"
            args.append(cutoff)
        query += " ORDER BY ont_id, timestamp_utc"
        data = connection.execute(query, args).df()
    return data, manifest


def load_mapping(path):
    return yaml.safe_load(Path(path).read_text())
