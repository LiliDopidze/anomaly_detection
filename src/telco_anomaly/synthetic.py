"""Small, explicit PON simulator. See SYNTHETIC_REVIEW.md for evidence limits.

Truth is written separately. Rates/noise are assumptions; this is not a digital
twin, a calibrated receiver simulator, or a claimed operator failure population.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.signal import lfilter
from scipy.stats import binom
import yaml

VERSION = "5.0.0"
METRICS = [
    "rx_power_dbm",
    "olt_rx_power_dbm",
    "tx_power_dbm",
    "temperature_c",
    "bias_current_ma",
    "ber",
    "fec_count",
    "throughput_mbps",
    "uptime_s",
    "reboot_count",
]
FAULT_COLUMNS = [
    "gt_fault_id",
    "gt_fault_type",
    "scope",
    "target",
    "onset_ts",
    "repair_ts",
    "magnitude_db",
    "rise_hours",
    "group_id",
    "first_observable_ts",
    "impact_ts",
]
INTERVAL_COLUMNS = [
    "fault_id",
    "entity_id",
    "active_start_ts",
    "active_end_ts",
    "observable_ts",
    "impact_ts",
    "latent_max_effect_db",
    "observed_samples",
    "visibility_basis",
]


@dataclass(frozen=True)
class SyntheticConfig:
    seed: int = 20260919
    days: int = 90
    start: str = "2025-01-01T00:00:00Z"
    sample_minutes: int = 15
    n_onts: int = 96
    n_olts: int = 3
    ports_per_olt: int = 2
    splitters_per_port: int = 2
    faults_per_ont_year: float = 12.0
    faults_per_splitter_year: float = 6.0
    fault_duration_median_hours: float = 36.0
    fault_duration_log_sd: float = 0.7
    loss_median_db: float = 3.0
    loss_log_sd: float = 0.7
    poll_loss_probability: float = 0.02
    field_loss_probability: float = 0.003
    collector_outages_per_month: float = 1.0
    sensor_noise_db: float = 0.12
    seasonal_temperature_amplitude_c: float = 4.0
    daily_temperature_amplitude_c: float = 2.0

    def validate(self):
        integers = [
            "days",
            "sample_minutes",
            "n_onts",
            "n_olts",
            "ports_per_olt",
            "splitters_per_port",
        ]
        for name in integers:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if 1440 % self.sample_minutes:
            raise ValueError("sample_minutes must divide one day")
        capacity = self.n_olts * self.ports_per_olt
        capacity *= self.splitters_per_port * 16
        if self.splitters_per_port > 4 or self.n_onts > capacity:
            raise ValueError("Topology exceeds the assumed splitter capacity")
        for name in ("poll_loss_probability", "field_loss_probability"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        for name in (
            "faults_per_ont_year",
            "faults_per_splitter_year",
            "collector_outages_per_month",
            "fault_duration_log_sd",
            "loss_log_sd",
            "seasonal_temperature_amplitude_c",
            "daily_temperature_amplitude_c",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in (
            "fault_duration_median_hours",
            "loss_median_db",
            "sensor_noise_db",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        pd.Timestamp(self.start).tz_convert("UTC")


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    allowed = {f.name for f in fields(SyntheticConfig)} | {"output"}
    if unknown := set(config) - allowed:
        raise ValueError(f"Unknown synthetic settings: {sorted(unknown)}")
    output = config.pop("output", "data/synthetic_pon_v5")
    result = SyntheticConfig(**config)
    result.validate()
    return result, Path(output)


def random_stream(seed, name):
    """Stable named streams; changing collection settings cannot redraw faults."""
    key = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
    return np.random.default_rng(np.random.SeedSequence([seed, key]))


def ar1(n, tau_hours, sd, dt_hours, rng):
    """Stationary OU discretisation: variance sd², correlation exp(-dt/tau)."""
    phi = np.exp(-dt_hours / tau_hours)
    noise = rng.normal(0, sd * np.sqrt(1 - phi**2), n)
    result, _ = lfilter([1.0], [1.0, -phi], noise, zi=[phi * rng.normal(0, sd)])
    return result


def make_topology(config):
    rng = random_stream(config.seed, "topology")
    slots = [
        (o, p, s)
        for o in range(config.n_olts)
        for p in range(config.ports_per_olt)
        for s in range(config.splitters_per_port)
        for _ in range(16)
    ]
    chosen = rng.choice(len(slots), config.n_onts, replace=False)
    feeder = rng.uniform(1, 8, (config.n_olts, config.ports_per_olt))
    distribution = rng.uniform(
        0.2,
        2,
        (
            config.n_olts,
            config.ports_per_olt,
            config.splitters_per_port,
        ),
    )
    rows = []
    for i, slot in enumerate(chosen):
        o, p, s = slots[slot]
        distance = feeder[o, p] + distribution[o, p, s] + rng.uniform(0.01, 0.3)
        # 1:4 primary and 1:16 secondary; ideal division plus excess loss.
        split_loss = 10 * np.log10(64) + rng.uniform(1, 2)
        connector_loss = rng.uniform(0.5, 3)
        rows.append(
            {
                "ont_id": f"ONT-{i:05d}",
                "olt_id": f"OLT-{o:02d}",
                "pon_port": f"PON-{o:02d}-{p:02d}",
                "splitter_l1": f"L1-{o:02d}-{p:02d}",
                "splitter_l2": f"L2-{o:02d}-{p:02d}-{s:02d}",
                "vendor": "synthetic",
                "device_model": f"cohort-{rng.integers(2)}",
                "firmware_version": "simulated-1",
                "distance_km": distance,
                "loss_ds_db": split_loss + connector_loss + 0.24 * distance,
                "loss_us_db": split_loss + connector_loss + 0.35 * distance,
                "ont_tx_dbm": rng.uniform(1.5, 3),
                "rx_sensitivity_dbm": rng.uniform(-28, -27),
                "temperature_offset_c": rng.normal(40, 3),
                "bias_base_ma": rng.uniform(15, 30),
            }
        )
    topology = pd.DataFrame(rows)
    # Conservative equal capacity share; no dynamic bandwidth allocation model.
    members = topology.groupby("pon_port").ont_id.transform("count")
    topology["capacity_mbps"] = 2488.32 / members
    return topology


def sample_faults(config, topology):
    rng = random_stream(config.seed, "faults")
    start = pd.Timestamp(config.start)
    rows = []
    for scope, column, rate in [
        ("ont", "ont_id", config.faults_per_ont_year),
        ("l2", "splitter_l2", config.faults_per_splitter_year),
    ]:
        for target in sorted(topology[column].unique()):
            for _ in range(rng.poisson(rate * config.days / 365.25)):
                kind = rng.choice(["gradual_loss", "intermittent_loss", "abrupt_loss"])
                onset = start + pd.Timedelta(hours=rng.uniform(0, config.days * 24))
                duration = rng.lognormal(
                    np.log(config.fault_duration_median_hours),
                    config.fault_duration_log_sd,
                )
                rows.append(
                    {
                        "gt_fault_id": f"F-{len(rows):05d}",
                        "gt_fault_type": str(kind),
                        "scope": scope,
                        "target": target,
                        "onset_ts": onset,
                        "repair_ts": onset + pd.Timedelta(hours=duration),
                        "magnitude_db": rng.lognormal(
                            np.log(config.loss_median_db),
                            config.loss_log_sd,
                        ),
                        "rise_hours": duration * rng.uniform(0.2, 0.8),
                        "group_id": None,
                        "first_observable_ts": pd.NaT,
                        "impact_ts": pd.NaT,
                    }
                )
    return pd.DataFrame(rows, columns=FAULT_COLUMNS)


def fault_curve(event, times, config):
    hours = (times - event.onset_ts).total_seconds().to_numpy() / 3600
    active = (times >= event.onset_ts) & (times < event.repair_ts)
    if event.gt_fault_type == "gradual_loss":
        shape = np.clip(hours / event.rise_hours, 0, 1)
    elif event.gt_fault_type == "abrupt_loss":
        shape = np.ones(len(times))
    else:
        rng = random_stream(config.seed, event.gt_fault_id + ":intermittency")
        shape = np.zeros(len(times))
        state = 0
        dt = config.sample_minutes / 60
        for i in np.flatnonzero(active):
            mean_dwell = 2 if state else 4
            if rng.random() < -np.expm1(-dt / mean_dwell):
                state = 1 - state
            shape[i] = state
    return event.magnitude_db * shape * active


def corrected_codeword_probability(bit_error_probability):
    """Ideal independent-bit RS(255,239): 1..8 erroneous symbols corrected.

    This is a transparent approximation, not a measured FEC transfer curve.
    """
    symbol_error = -np.expm1(8 * np.log1p(-bit_error_probability))
    return np.clip(
        binom.sf(0, 255, symbol_error) - binom.sf(8, 255, symbol_error), 0, 1
    )


def _first_time(times, mask):
    found = np.flatnonzero(mask)
    return times[found[0]] if len(found) else pd.NaT


def _emit_entity(config, row, times, shared, events):
    n = len(times)
    dt = config.sample_minutes / 60
    rng = random_stream(config.seed, row.ont_id + ":healthy")
    sensor = random_stream(config.seed, row.ont_id + ":sensor")
    collection = random_stream(config.seed, row.ont_id + ":collection")
    errors = random_stream(config.seed, row.ont_id + ":errors")
    local_hour = times.hour.to_numpy() + times.minute.to_numpy() / 60
    daily = np.cos(2 * np.pi * (local_hour - 16) / 24)
    annual = np.cos(2 * np.pi * (times.dayofyear.to_numpy() - 200) / 365.25)
    temperature = (
        row.temperature_offset_c
        + config.daily_temperature_amplitude_c * daily
        + config.seasonal_temperature_amplitude_c * annual
        + shared["weather"]
        + ar1(n, 4, 0.8, dt, rng)
    )
    path_drift = (
        shared[row.splitter_l2] + shared[row.pon_port] + ar1(n, 24, 0.08, dt, rng)
    )
    tx = row.ont_tx_dbm + 0.005 * (temperature - 40) + ar1(n, 12, 0.06, dt, rng)
    # Benign workload variation is independent of physical faults.
    usage = np.exp(ar1(n, 3, 0.6, dt, rng))
    demand = rng.uniform(15, 70) * (1 + 0.5 * daily) * usage
    demand *= np.where(times.dayofweek.to_numpy() >= 5, 1.15, 1)
    bias = row.bias_base_ma + 0.12 * (temperature - 40) + ar1(n, 4, 0.4, dt, rng)
    base_ds = 3.0 - row.loss_ds_db + path_drift
    base_us = tx - row.loss_us_db + path_drift
    curves = [(event, fault_curve(event, times, config)) for event in events]
    loss = sum((curve for _, curve in curves), np.zeros(n))
    ds = base_ds - loss
    us = base_us - loss
    margin = np.minimum(ds - row.rx_sensitivity_dbm, us + 28)
    # Smooth assumed service response; same latent state drives impact labels.
    capacity_fraction = 1 / (1 + np.exp(-np.clip(margin, -50, 50)))
    throughput = np.minimum(demand, row.capacity_mbps) * capacity_fraction
    error_log_noise = ar1(n, 0.5, 0.3, dt, errors)
    ds_margin = ds - row.rx_sensitivity_dbm
    bit_probability = 10 ** np.clip(-3 - ds_margin + error_log_noise, -12, -1)
    opportunities = int(2.48832e9 * config.sample_minutes * 60 / (255 * 8))
    fec = errors.binomial(
        opportunities, corrected_codeword_probability(bit_probability)
    )
    measured_ber = bit_probability * np.exp(errors.normal(0, 0.2, n))
    measured_ber = np.where(measured_ber < 1e-9, 0, measured_ber)
    measured_ber = np.minimum(measured_ber, 1)
    reboot = rng.random(n) < -np.expm1(-0.05 * dt / 24)
    last_reset = -rng.uniform(24, 240)
    uptime = np.empty(n)
    for i in range(n):
        if reboot[i]:
            last_reset = i * dt
        uptime[i] = (i * dt - last_reset) * 3600
    data = pd.DataFrame(
        {
            "timestamp_utc": times,
            "ont_id": row.ont_id,
            "rx_power_dbm": ds + sensor.normal(0, config.sensor_noise_db, n),
            "olt_rx_power_dbm": us + sensor.normal(0, config.sensor_noise_db * 1.5, n),
            "tx_power_dbm": tx + sensor.normal(0, 0.05, n),
            "temperature_c": temperature + sensor.normal(0, 0.25, n),
            "bias_current_ma": bias + sensor.normal(0, 0.1, n),
            "ber": measured_ber,
            "fec_count": fec.astype(float),
            "throughput_mbps": throughput,
            "uptime_s": uptime,
            "reboot_count": np.cumsum(reboot).astype(float),
        }
    )
    # Readout errors must not propagate into the physical BER/service process.
    glitch = sensor.random(n) < -np.expm1(-0.04 * dt / 24)
    data.loc[glitch, "rx_power_dbm"] += sensor.normal(0, 1.5, glitch.sum())
    for column, step in {
        "rx_power_dbm": 0.1,
        "olt_rx_power_dbm": 0.1,
        "tx_power_dbm": 0.1,
        "temperature_c": 0.5,
        "bias_current_ma": 0.1,
        "throughput_mbps": 0.1,
    }.items():
        data[column] = np.round(data[column] / step) * step
    # Availability depends on both collection and physical impairment.
    polling = collection.random(n) >= config.poll_loss_probability
    reporting = collection.random(n) >= 0.8 * (margin < 0)
    keep = polling & reporting & ~shared[row.olt_id + ":collector"]
    field_missing = collection.random((n, len(METRICS))) < config.field_loss_probability
    data[METRICS] = data[METRICS].mask(field_missing)
    optical_available = data[["rx_power_dbm", "olt_rx_power_dbm"]].notna().any(axis=1)
    intervals = []
    for event, effect in curves:
        active = (times >= event.onset_ts) & (times < event.repair_ts)
        # A declared measurement-scale proxy, never a detector-success definition.
        visible = active & (effect >= max(0.2, 2 * config.sensor_noise_db))
        visible &= keep & optical_available.to_numpy()
        without_margin = np.minimum(
            ds + effect - row.rx_sensitivity_dbm,
            us + effect + 28,
        )
        without_capacity = 1 / (1 + np.exp(-np.clip(without_margin, -50, 50)))
        impact = active & (capacity_fraction < 0.5) & (without_capacity >= 0.5)
        intervals.append(
            {
                "fault_id": event.gt_fault_id,
                "entity_id": row.ont_id,
                "active_start_ts": event.onset_ts,
                "active_end_ts": event.repair_ts,
                "observable_ts": _first_time(times, visible),
                "impact_ts": _first_time(times, impact),
                "latent_max_effect_db": float(effect.max()),
                "observed_samples": int((active & keep & optical_available).sum()),
                "visibility_basis": "effect_above_resolution_proxy_not_detectability",
            }
        )
    # Aggregate paired physical diagnostics stay outside model-visible columns.
    audit = {
        "ont_id": row.ont_id,
        "expected_rows": n,
        "observed_rows": int(keep.sum()),
        "temperature_bias_correlation": float(np.corrcoef(temperature, bias)[0, 1]),
        "max_service_fraction": float(capacity_fraction.max()),
        "min_service_fraction": float(capacity_fraction.min()),
        "max_corrected_codewords": int(fec.max()),
        "codeword_opportunities": opportunities,
        "maximum_capacity_increase_from_loss": float(
            np.max(
                capacity_fraction
                - 1
                / (
                    1
                    + np.exp(
                        -np.clip(
                            np.minimum(base_ds - row.rx_sensitivity_dbm, base_us + 28),
                            -50,
                            50,
                        )
                    )
                )
            )
        ),
    }
    return data.loc[keep].reset_index(drop=True), intervals, audit


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _generate(config, destination):
    topology = make_topology(config)
    faults = sample_faults(config, topology)
    n = config.days * 1440 // config.sample_minutes
    times = pd.date_range(config.start, periods=n, freq=f"{config.sample_minutes}min")
    times = times.tz_convert("UTC")
    dt = config.sample_minutes / 60
    shared = {"weather": ar1(n, 48, 1, dt, random_stream(config.seed, "weather"))}
    for column, sd in [("pon_port", 0.08), ("splitter_l2", 0.06)]:
        for group in topology[column].unique():
            shared[group] = ar1(n, 24, sd, dt, random_stream(config.seed, group))
    for olt in topology.olt_id.unique():
        rng = random_stream(config.seed, olt + ":collector")
        gaps = np.zeros(n, dtype=bool)
        for _ in range(
            rng.poisson(config.collector_outages_per_month * config.days / 30)
        ):
            start = rng.integers(n)
            length = max(1, round(rng.exponential(2) / dt))
            gaps[start : start + length] = True
        shared[olt + ":collector"] = gaps
    intervals, audits = [], []
    writer = None
    try:
        for row in topology.itertuples(index=False):
            relevant = faults.loc[
                ((faults.scope == "ont") & (faults.target == row.ont_id))
                | ((faults.scope == "l2") & (faults.target == row.splitter_l2))
            ]
            panel, truth, audit = _emit_entity(
                config,
                row,
                times,
                shared,
                list(relevant.itertuples(index=False)),
            )
            table = pa.Table.from_pandas(panel, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    destination / "reference_dataset.parquet", table.schema
                )
            writer.write_table(table)
            intervals.extend(truth)
            audits.append(audit)
    finally:
        if writer is not None:
            writer.close()
    intervals = pd.DataFrame(intervals, columns=INTERVAL_COLUMNS)
    if len(intervals):
        first = intervals.groupby("fault_id")[["observable_ts", "impact_ts"]].min()
        faults["first_observable_ts"] = faults.gt_fault_id.map(first.observable_ts)
        faults["impact_ts"] = faults.gt_fault_id.map(first.impact_ts)
    faults.to_csv(destination / "gt_fault_registry.csv", index=False)
    intervals.to_csv(destination / "fault_entity_intervals.csv", index=False)
    topology.to_csv(destination / "topology.csv", index=False)
    pd.DataFrame(audits).to_csv(destination / "simulation_audit.csv", index=False)
    pd.DataFrame(
        {
            "entity_id": topology.ont_id,
            "install_ts": times[0],
            "decommission_ts": times[-1] + pd.Timedelta(minutes=config.sample_minutes),
        }
    ).to_csv(destination / "entity_service_windows.csv", index=False)
    payload = {
        "generator_version": VERSION,
        "evidence_level": "synthetic_uncalibrated",
        "config": asdict(config),
        "timestamp_semantics": (
            "State at timestamp; counts approximate the preceding interval "
            "using its end state. First interval assumes pre-start steady state."
        ),
        "limitations": [
            "Fault prevalence, durations, noise and dependencies are uncalibrated.",
            "RS independent-error approximation is not a vendor FEC model.",
            "Visibility is an effect-size proxy, not demonstrated detectability.",
            "No tickets, repair economy, exact physical localisation or real vendor profiles.",
        ],
        "python": platform.python_version(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in [
                "numpy",
                "pandas",
                "scipy",
                "pyarrow",
            ]
        },
        "generator_sha256": sha256(__file__),
        "files": {p.name: sha256(p) for p in sorted(destination.iterdir())},
    }
    try:
        payload["git_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        payload["git_revision"] = None
    (destination / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


def generate_dataset(config, output):
    """Generate atomically; never overwrite an existing dataset."""
    config.validate()
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".synthetic-", dir=output.parent))
    try:
        _generate(config, stage)
        stage.rename(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/synthetic.yml")
    parser.add_argument("--output")
    args = parser.parse_args()
    config, output = load_config(args.config)
    print(generate_dataset(config, args.output or output))


if __name__ == "__main__":
    main()
