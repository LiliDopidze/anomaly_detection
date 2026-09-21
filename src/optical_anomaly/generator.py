"""Controlled optical telemetry; numerical defaults are simulation assumptions."""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from .optics import error_telemetry


@dataclass(frozen=True)
class GeneratorConfig:
    seed: int = 42
    entities: int = 96
    days: int = 90
    interval_minutes: int = 5
    noise_db: float = 0.08
    sensor_noise_db: float = 0.04
    correlation_hours: float = 0.5
    daily_amplitude_db: float = 0.25
    missing_probability: float = 0.02
    impact_threshold_dbm: float = -27.0
    faults_per_entity: int = 2
    impact_intervals: int = 3
    downstream_receiver_reference_dbm: float = -27.0
    upstream_receiver_reference_dbm: float = -28.0
    receiver_offset_halfwidth_db: float = 1.5
    fec_log10_noise_sd: float = 0.15
    upstream_impact_threshold_dbm: float = -28.0
    fault_duration_median_hours: float = 36.0
    onts_per_splitter: int = 8
    splitters_per_port: int = 2
    ports_per_olt: int = 4

    def __post_init__(self) -> None:
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
            ]
        ).all():
            raise ValueError("Receiver parameters must be finite")
        if self.days < 8 or self.entities < 1 or self.interval_minutes < 1:
            raise ValueError("Need >=8 days, >=1 entity and a positive interval")
        if self.noise_db <= 0 or self.correlation_hours <= 0:
            raise ValueError("Noise and correlation time must be positive")
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
    """Exact stationary Gaussian AR(1), including its initial distribution."""
    values = np.empty(size)
    values[0] = rng.normal(0, sigma)
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
        # Drift -0.15 dB/hour, diffusion 0.06 dB/sqrt(hour).
        increments = rng.normal(-0.15 * step_hours, 0.06 * np.sqrt(step_hours), size)
        return -0.2 + np.cumsum(increments)
    if kind == "exponential":
        # Accelerating attenuation in dB; zero-free initial loss fixes onset.
        return -0.2 - 0.35 * np.expm1(np.minimum(elapsed / 12, 3))
    if kind == "variance_shift":
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
    temperature = 35 + 2 * np.sin(2 * np.pi * hours / 24)
    temperature += stationary_noise(
        size, 0.5, np.exp(-config.interval_minutes / 360), rng
    )
    transmit = 3 + 0.005 * (temperature - 35)
    transmit += stationary_noise(size, 0.03, np.exp(-config.interval_minutes / 60), rng)
    return transmit, temperature


def _normal_optics(
    config: GeneratorConfig, entity: int, size: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    dt = config.interval_minutes / 60
    hours = np.arange(size) * dt
    phase = rng.uniform(0, 2 * np.pi)
    olt_tx, olt_temperature = _port_optics(config, entity, size)
    ont_temperature = 38 + 3 * np.sin(2 * np.pi * hours / 24 + phase)
    ont_temperature += stationary_noise(size, 0.6, np.exp(-dt / 3), rng)
    ont_tx = 2 + 0.008 * (ont_temperature - 38)
    ont_tx += stationary_noise(size, 0.03, np.exp(-dt), rng)
    path_loss = rng.uniform(23, 27) + config.daily_amplitude_db * np.sin(
        2 * np.pi * hours / 24 + phase
    )
    path_loss += stationary_noise(
        size, config.noise_db, np.exp(-dt / config.correlation_hours), rng
    )
    return {
        "rx_dbm": olt_tx - path_loss,
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
    left, right = [(0.57, 0.72), (0.78, 0.96)][number]
    onset = int(len(times) * rng.uniform(left, left + 0.02))
    duration = rng.lognormal(np.log(config.fault_duration_median_hours), 0.5)
    stop = min(int(len(times) * right), onset + max(2, int(duration / dt)))
    kind = ("random_walk", "exponential", "variance_shift")[(entity + number) % 3]
    signature = fault_signature(kind, stop - onset, dt, rng)
    severity = rng.uniform(0.4, 1.4)
    delta = signature * severity
    optics["rx_dbm"][onset:stop] += delta
    directional_rng = np.random.default_rng(
        np.random.SeedSequence([config.seed, entity, 4])
    )
    upstream_multiplier = directional_rng.uniform(0.8, 1.4)
    optics["upstream_rx_dbm"][onset:stop] += upstream_multiplier * delta
    crosses = (optics["rx_dbm"][onset:stop] < config.impact_threshold_dbm) | (
        optics["upstream_rx_dbm"][onset:stop] < config.upstream_impact_threshold_dbm
    )
    confirmed = first_persistent_crossing(crosses, config.impact_intervals)
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
        "2025-01-01",
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
