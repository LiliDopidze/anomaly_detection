"""Controlled optical telemetry; numerical defaults are simulation assumptions."""

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GeneratorConfig:
    seed: int = 42
    entities: int = 24
    days: int = 28
    interval_minutes: int = 5
    noise_db: float = 0.08
    correlation_hours: float = 0.5
    daily_amplitude_db: float = 0.25
    missing_probability: float = 0.02
    impact_threshold_dbm: float = -27.0
    faults_per_entity: int = 2

    def __post_init__(self) -> None:
        if self.days < 8 or self.entities < 1 or self.interval_minutes < 1:
            raise ValueError("Need >=8 days, >=1 entity and a positive interval")
        if self.noise_db <= 0 or self.correlation_hours <= 0:
            raise ValueError("Noise and correlation time must be positive")
        if not 0 <= self.missing_probability < 1:
            raise ValueError("Missing probability must be in [0, 1)")
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


def _inject_fault(
    config: GeneratorConfig,
    entity: int,
    number: int,
    times: pd.DatetimeIndex,
    latent: np.ndarray,
    missing: np.ndarray,
    physics: np.random.Generator,
) -> dict:
    dt = config.interval_minutes / 60
    left, right = [(0.57, 0.72), (0.78, 0.96)][number]
    onset = int(len(times) * physics.uniform(left, left + 0.02))
    stop = int(len(times) * right)
    kind = ("random_walk", "exponential", "variance_shift")[(entity + number) % 3]
    delta = fault_signature(kind, stop - onset, dt, physics)
    latent[onset:stop] += delta
    crossings = np.flatnonzero(latent[onset:stop] < config.impact_threshold_dbm)
    # A declared visibility proxy, independent of detector outcomes.
    effect = np.abs(delta) if kind != "variance_shift" else np.full(len(delta), 0.5)
    visible = np.flatnonzero((effect >= 2 * config.noise_db) & ~missing[onset:stop])
    return {
        "fault_id": f"F-{entity:03d}-{number}",
        "entity_id": f"ONT-{entity:03d}",
        "fault_type": kind,
        "onset_time": times[onset],
        "observable_onset_time": (
            times[onset + visible[0]] if len(visible) else pd.NaT
        ),
        "impact_time": (times[onset + crossings[0]] if len(crossings) else pd.NaT),
        "end_time": times[stop],
    }


def _simulate_entity(
    config: GeneratorConfig,
    entity: int,
    times: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, list[dict]]:
    dt = config.interval_minutes / 60
    physics = np.random.default_rng(np.random.SeedSequence([config.seed, entity, 0]))
    sensor = np.random.default_rng(np.random.SeedSequence([config.seed, entity, 1]))
    collection = np.random.default_rng(np.random.SeedSequence([config.seed, entity, 2]))
    level = physics.uniform(-24, -20)
    phase = physics.uniform(0, 2 * np.pi)
    latent = level + config.daily_amplitude_db * np.sin(
        2 * np.pi * np.arange(len(times)) * dt / 24 + phase
    )
    latent += stationary_noise(
        len(times),
        config.noise_db,
        np.exp(-dt / config.correlation_hours),
        physics,
    )
    missing = collection.random(len(times)) < config.missing_probability
    faults = []
    for number in range(config.faults_per_entity):
        faults.append(
            _inject_fault(config, entity, number, times, latent, missing, physics)
        )
    measured = latent + sensor.normal(0, config.noise_db / 2, len(times))
    # Illustrative monotone link-margin response, not a calibrated receiver.
    ber = np.clip(10 ** (-9 - (latent - config.impact_threshold_dbm)), 1e-12, 0.1)
    measured[missing], ber[missing] = np.nan, np.nan
    return (
        pd.DataFrame(
            {
                "time": times,
                "device": f"ONT-{entity:03d}",
                "rx_dbm": measured,
                "ber": ber,
            }
        ),
        faults,
    )


def generate(config: GeneratorConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate measurements/truth; first 55% is a declared fault-free baseline.

    Fault timings are controlled scenarios, not empirical arrival distributions.
    Impact is a latent Rx crossing, independent of measurement/collection noise.
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
