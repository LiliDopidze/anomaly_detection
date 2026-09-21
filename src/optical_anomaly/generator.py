"""Controlled GPON scenarios, not a calibrated model of an operator network.

Source keys refer to METHOD.md: [etsi] identifies monitored measurements;
[ar] supports autoregressive noise; [db] defines logarithmic power/loss.
These sources do not establish our numerical distributions or fault laws.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from .optics import error_telemetry


@dataclass(frozen=True)
class GeneratorConfig:
    seed: int = 42  # Reproducibility choice; not evidence for one realisation.
    entities: int = 96  # Scenario size, not a representative fleet sample.
    days: int = 90  # Scenario duration; no annual-cycle claim.
    interval_minutes: int = 5  # Assumed poll cadence; verify source capability.
    noise_db: float = 0.08  # Assumed stationary path-loss SD, not sensor accuracy.
    sensor_noise_db: float = 0.04  # Assumed independent readout-noise SD.
    correlation_hours: float = 0.5  # Assumed decay time; [ar] supports form only.
    daily_amplitude_db: float = 0.25  # Assumed daily loss amplitude, not a rule.
    missing_probability: float = 0.02  # Assumed independent missed-poll rate.
    impact_threshold_dbm: float = -27.0  # Hypothetical impact, not receiver spec.
    faults_per_entity: int = 2  # Controlled cases, not an empirical fault rate.
    impact_intervals: int = 3  # Assumed persistence for the impact proxy.
    downstream_receiver_reference_dbm: float = -27.0  # Assumed BER-curve anchor.
    upstream_receiver_reference_dbm: float = -28.0  # Assumed BER-curve anchor.
    receiver_offset_halfwidth_db: float = 1.5  # Assumed device heterogeneity.
    fec_log10_noise_sd: float = 0.15  # Assumed error dispersion in log10 space.
    upstream_impact_threshold_dbm: float = -28.0  # Hypothetical impact only.
    fault_duration_median_hours: float = 36.0  # Assumed repair/duration median.
    onts_per_splitter: int = 8  # Scenario membership, not inferred optical loss.
    splitters_per_port: int = 2  # Scenario topology, not a mandated split ratio.
    ports_per_olt: int = 4  # Scenario capacity; last OLT may be partly populated.

    def __post_init__(self) -> None:
        integer_fields = (
            self.seed, self.entities, self.days, self.interval_minutes,
            self.faults_per_entity, self.impact_intervals, self.onts_per_splitter,
            self.splitters_per_port, self.ports_per_olt,
        )
        if any(type(value) is not int for value in integer_fields):
            raise ValueError("Seed, sizes, cadence and interval counts must be integers")
        if self.seed < 0:
            raise ValueError("Seed must be nonnegative")
        if not isinstance(self.impact_intervals, int) or self.impact_intervals < 1:
            raise ValueError("impact_intervals must be a positive integer")
        if min(self.receiver_offset_halfwidth_db, self.fec_log10_noise_sd) < 0:
            raise ValueError("Receiver spread and FEC dispersion must be nonnegative")
        if not np.isfinite(
            [
                self.downstream_receiver_reference_dbm,
                self.upstream_receiver_reference_dbm,
                self.receiver_offset_halfwidth_db,
                self.fec_log10_noise_sd,
                self.noise_db,
                self.correlation_hours,
                self.daily_amplitude_db,
                self.impact_threshold_dbm,
                self.upstream_impact_threshold_dbm,
                self.fault_duration_median_hours,
            ]
        ).all():
            raise ValueError("Generator parameters must be finite")
        if self.days < 8 or self.entities < 1 or self.interval_minutes < 1:
            raise ValueError("Need >=8 days, >=1 entity and a positive interval")
        if self.noise_db <= 0 or self.correlation_hours <= 0:
            raise ValueError("Noise and correlation time must be positive")
        if self.daily_amplitude_db < 0:
            raise ValueError("Daily amplitude must be nonnegative")
        if self.sensor_noise_db < 0 or not np.isfinite(self.sensor_noise_db):
            raise ValueError("Sensor noise must be finite and nonnegative")
        if not 0 <= self.missing_probability < 1:
            raise ValueError("Missing probability must be in [0, 1)")
        if min(self.onts_per_splitter, self.splitters_per_port, self.ports_per_olt) < 1:
            raise ValueError("Topology fan-outs must be positive")
        if self.fault_duration_median_hours <= 0:
            raise ValueError("Fault duration must be positive")
        if self.faults_per_entity not in (0, 1, 2):
            raise ValueError("Use zero, one or two separated faults per entity")


def stationary_noise(
    size: int, sigma: float, phi: float, rng: np.random.Generator
) -> np.ndarray:
    """Stationary Gaussian AR(1), including initial variance; source: [ar]."""
    if not isinstance(size, int) or size < 0:
        raise ValueError("Noise size must be a nonnegative integer")
    if not np.isfinite([sigma, phi]).all() or sigma < 0 or abs(phi) >= 1:
        raise ValueError("Need finite sigma >= 0 and stationary abs(phi) < 1")
    values = np.empty(size)
    if not size:
        return values
    values[0] = rng.normal(0, sigma)
    # Exact variance identity, not an empirically fitted PON noise level.
    innovation = sigma * np.sqrt(1 - phi**2)
    for i in range(1, size):
        values[i] = phi * values[i - 1] + rng.normal(0, innovation)
    return values


def fault_signature(
    kind: str, size: int, step_hours: float, rng: np.random.Generator
) -> np.ndarray:
    """Signed dB perturbation; distribution changes at the first fault sample."""
    elapsed = np.arange(size) * step_hours
    if kind == "random_walk":
        # Assumed drift -0.15 dB/hour and diffusion 0.06 dB/sqrt(hour).
        increments = rng.normal(-0.15 * step_hours, 0.06 * np.sqrt(step_hours), size)
        return -0.2 + np.cumsum(increments)  # Assumed 0.2 dB initial loss.
    if kind == "exponential":
        # Assumed 0.2/0.35 dB scales, 12-hour growth, cap after 36 hours.
        # This is exponential attenuation in dB, not a fibre ageing law.
        return -0.2 - 0.35 * np.expm1(np.minimum(elapsed / 12, 3))
    if kind == "variance_shift":
        # Assumed added 0.5 dB SD and 0.5-hour decay; includes both signs.
        return stationary_noise(size, 0.5, np.exp(-step_hours / 0.5), rng)
    raise ValueError(f"Unknown fault signature: {kind}")


def make_topology(config: GeneratorConfig) -> pd.DataFrame:
    """Static lookup only: no topology features, event tables or incident grouping."""
    entity = np.arange(config.entities)
    splitter = entity // config.onts_per_splitter
    port = splitter // config.splitters_per_port
    olt = port // config.ports_per_olt
    return pd.DataFrame(
        {
            "entity_id": [f"ONT-{i:03d}" for i in entity],
            "olt_id": [f"OLT-{i:03d}" for i in olt],
            "pon_port_id": [f"PON-{i:03d}" for i in port],
            "splitter_id": [f"SPLITTER-{i:03d}" for i in splitter],
        }
    )


def _port_optics(
    config: GeneratorConfig, entity: int, size: int
) -> tuple[np.ndarray, np.ndarray]:
    port = entity // (config.onts_per_splitter * config.splitters_per_port)
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, port, 100]))
    hours = np.arange(size) * config.interval_minutes / 60
    # [etsi] specifies module temperature monitoring, not this waveform.
    temperature = 35 + 2 * np.sin(2 * np.pi * hours / 24)  # Assumed 35 +/- 2 C.
    temperature += stationary_noise(
        # Assumed 0.5 C SD and six-hour correlation time (360 minutes).
        size, 0.5, np.exp(-config.interval_minutes / 360), rng
    )
    transmit = 3 + 0.005 * (temperature - 35)  # Assumed 3 dBm, 0.005 dB/C.
    # Assumed 0.03 dB SD and one-hour correlation time.
    transmit += stationary_noise(size, 0.03, np.exp(-config.interval_minutes / 60), rng)
    return transmit, temperature


def _normal_optics(
    config: GeneratorConfig, entity: int, size: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    dt = config.interval_minutes / 60
    hours = np.arange(size) * dt
    phase = rng.uniform(0, 2 * np.pi)  # Assumed device-specific daily phase.
    olt_tx, olt_temperature = _port_optics(config, entity, size)
    ont_temperature = 38 + 3 * np.sin(2 * np.pi * hours / 24 + phase)
    # Assumed 38 +/- 3 C daily cycle, 0.6 C SD and three-hour correlation.
    ont_temperature += stationary_noise(size, 0.6, np.exp(-dt / 3), rng)
    ont_tx = 2 + 0.008 * (ont_temperature - 38)  # Assumed 2 dBm, 0.008 dB/C.
    # Assumed 0.03 dB SD and one-hour correlation time.
    ont_tx += stationary_noise(size, 0.03, np.exp(-dt), rng)
    # Assumed aggregate loss, including splitter/fibre/connectors. [db] gives
    # Rx = Tx - loss; the standard does not specify a Uniform(23, 27) fleet.
    path_loss = rng.uniform(23, 27) + config.daily_amplitude_db * np.sin(
        2 * np.pi * hours / 24 + phase
    )
    path_loss += stationary_noise(
        size, config.noise_db, np.exp(-dt / config.correlation_hours), rng
    )
    return {
        "rx_dbm": olt_tx - path_loss,
        # Assumed shared fluctuations and 0.4--1.2 dB directional offset.
        # Real wavelengths need not have this nearly identical loss trajectory.
        "upstream_rx_dbm": ont_tx - path_loss - rng.uniform(0.4, 1.2),
        "ont_tx_dbm": ont_tx,
        "olt_tx_dbm": olt_tx,
        "ont_temperature_c": ont_temperature,
        "olt_temperature_c": olt_temperature,
    }


def first_persistent_crossing(crosses: np.ndarray, intervals: int) -> int | None:
    """Return the confirming sample, never backdate to the start of the run."""
    if intervals < 1:
        raise ValueError("Need at least one confirming interval")
    if len(crosses) < intervals:
        return None
    runs = np.convolve(crosses.astype(int), np.ones(intervals, dtype=int), "valid")
    starts = np.flatnonzero(runs == intervals)
    return int(starts[0] + intervals - 1) if len(starts) else None


def _inject_fault(
    config: GeneratorConfig,
    entity: int,
    number: int,
    times: pd.DatetimeIndex,
    optics: dict[str, np.ndarray],
    missing: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    dt = config.interval_minutes / 60
    # Assumed separated development/test scenarios after a healthy baseline.
    left, right = [(0.57, 0.72), (0.78, 0.96)][number]
    onset = int(len(times) * rng.uniform(left, left + 0.02))
    # Assumed lognormal duration with log-SD 0.5; not fitted repair statistics.
    duration = rng.lognormal(np.log(config.fault_duration_median_hours), 0.5)
    stop = min(int(len(times) * right), onset + max(2, int(duration / dt)))
    # Balanced experimental cases; this does not estimate fault prevalence.
    kind = ("random_walk", "exponential", "variance_shift")[(entity + number) % 3]
    signature = fault_signature(kind, stop - onset, dt, rng)
    severity = rng.uniform(0.4, 1.4)  # Assumed multiplicative fault severity.
    delta = signature * severity
    optics["rx_dbm"][onset:stop] += delta
    directional_rng = np.random.default_rng(
        np.random.SeedSequence([config.seed, entity, 4])
    )
    upstream_multiplier = directional_rng.uniform(0.8, 1.4)  # Assumed coupling.
    optics["upstream_rx_dbm"][onset:stop] += upstream_multiplier * delta
    crosses = (optics["rx_dbm"][onset:stop] < config.impact_threshold_dbm) | (
        optics["upstream_rx_dbm"][onset:stop] < config.upstream_impact_threshold_dbm
    )
    confirmed = first_persistent_crossing(crosses, config.impact_intervals)
    # Assumed 2*physical-SD visibility proxy, not a statistical power guarantee.
    visible = np.flatnonzero(
        (np.abs(delta) >= 2 * config.noise_db) & ~missing[onset:stop]
    )
    if kind == "variance_shift":
        visible = np.array([], dtype=int)
    return {
        "fault_id": f"F-{entity:03d}-{number}",
        "entity_id": f"ONT-{entity:03d}",
        "fault_type": kind,
        "onset_time": times[onset],
        "observable_onset_time": times[onset + visible[0]] if len(visible) else pd.NaT,
        "impact_time": times[onset + confirmed] if confirmed is not None else pd.NaT,
        "end_time": times[stop],
    }


def _simulate_entity(
    config: GeneratorConfig, entity: int, times: pd.DatetimeIndex
) -> tuple[pd.DataFrame, list[dict]]:
    physics = np.random.default_rng(np.random.SeedSequence([config.seed, entity, 0]))
    sensor = np.random.default_rng(np.random.SeedSequence([config.seed, entity, 1]))
    collection = np.random.default_rng(np.random.SeedSequence([config.seed, entity, 2]))
    optics = _normal_optics(config, entity, len(times), physics)
    # Assumed missing completely at random; no outage-related missingness.
    missing = collection.random(len(times)) < config.missing_probability
    faults = [
        _inject_fault(config, entity, number, times, optics, missing, physics)
        for number in range(config.faults_per_entity)
    ]
    per_port = config.onts_per_splitter * config.splitters_per_port
    port_start = entity // per_port * per_port
    members = min(per_port, config.entities - port_start)
    errors = error_telemetry(
        optics["rx_dbm"],
        optics["upstream_rx_dbm"],
        (
            config.downstream_receiver_reference_dbm,
            config.upstream_receiver_reference_dbm,
        ),
        config.interval_minutes,
        members,
        np.random.SeedSequence([config.seed, entity, 3]),
        receiver_offset_halfwidth_db=config.receiver_offset_halfwidth_db,
        log10_noise_sd=config.fec_log10_noise_sd,
    )
    # Assumed independent Gaussian readout noise and 0.001 dB quantisation.
    # These readings, rather than the latent powers, are used by the detector.
    for name in ("rx_dbm", "upstream_rx_dbm", "ont_tx_dbm"):
        optics[name] = np.round(
            optics[name] + sensor.normal(0, config.sensor_noise_db, len(times)), 3
        )
    # Port Tx/temperature are shared measurements, repeated by ONT for convenience.
    for values in (*optics.values(), *errors.values()):
        values[missing] = np.nan
    frame = pd.DataFrame(
        {"time": times, "device": f"ONT-{entity:03d}", **optics, **errors}
    )
    return frame, faults


def generate(config: GeneratorConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate measurements/truth; first 55% is a declared fault-free baseline.

    Fault timings are controlled scenarios, not empirical arrival distributions.
    Impact is a persistent latent Rx crossing, not measured customer-service loss.
    """
    times = pd.date_range(
        "2025-01-01",  # Arbitrary UTC origin; no calendar/event realism claimed.
        periods=int(config.days * 1440 / config.interval_minutes),
        freq=f"{config.interval_minutes}min",
        tz="UTC",
    )
    frames, faults = [], []
    for entity in range(config.entities):
        frame, labels = _simulate_entity(config, entity, times)
        frames.append(frame)
        faults.extend(labels)
    columns = [
        "fault_id",
        "entity_id",
        "fault_type",
        "onset_time",
        "observable_onset_time",
        "impact_time",
        "end_time",
    ]
    return pd.concat(frames, ignore_index=True), pd.DataFrame(faults, columns=columns)
