"""GPON-inspired optics and FEC, with explicitly assumed receiver behaviour."""

import numpy as np
import pandas as pd
from scipy.stats import binom


def fec_probabilities(ber: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ideal RS(255,239): correct 1..8 erroneous byte symbols.

    Independent bit/symbol errors are a simplification, not a burst-error model.
    """
    symbol_error = -np.expm1(8 * np.log1p(-ber))
    uncorrectable = binom.sf(8, 255, symbol_error)
    corrected = np.clip(binom.sf(0, 255, symbol_error) - uncorrectable, 0, 1)
    return corrected, uncorrectable


def fec_counts(
    ber: np.ndarray, total: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Joint categorical counts; corrected + uncorrectable never exceeds total."""
    corrected_p, uncorrectable_p = fec_probabilities(ber)
    # Independent streams prevent future corrected counts affecting earlier
    # uncorrectable counts through variable random-number consumption.
    seeds = rng.integers(0, 2**63, size=2)
    corrected_rng, uncorrectable_rng = [np.random.default_rng(s) for s in seeds]
    corrected = corrected_rng.binomial(total, corrected_p)
    conditional = np.divide(
        uncorrectable_p,
        1 - corrected_p,
        out=np.zeros_like(ber),
        where=corrected_p < 1,
    )
    uncorrectable = uncorrectable_rng.binomial(
        total - corrected, np.clip(conditional, 0, 1)
    )
    return corrected, uncorrectable


def error_telemetry(
    downstream: np.ndarray,
    upstream: np.ndarray,
    thresholds: tuple[float, float],
    interval_minutes: int,
    port_entities: int,
    seed: np.random.SeedSequence,
) -> dict[str, np.ndarray]:
    """Interval-ending counts; full downstream coding, equal upstream allocations.

    Ignore frame/burst overhead and shortening. These are counts per observation
    interval, NOT cumulative device counters. No interval exists at the first row.
    """
    rng = np.random.default_rng(seed)
    n = len(downstream)
    hour = (np.arange(n) - 1) * interval_minutes / 60
    utilisation = 0.5 + 0.2 * np.sin(2 * np.pi * hour / 24)
    result = {}
    for direction, power, threshold, rate, allocation in (
        ("downstream", downstream, thresholds[0], 2.48832e9, np.ones(n)),
        ("upstream", upstream, thresholds[1], 1.24416e9, utilisation / port_entities),
    ):
        # Assumed pre-FEC response: not a standard-mandated sensitivity curve.
        ber = 10 ** np.clip(-5 - (power - threshold), -12, -1)
        total = np.floor(rate * interval_minutes * 60 * allocation / (255 * 8)).astype(
            np.int64
        )
        # Piecewise-constant physical state over the PRECEDING interval.
        # A change at t first affects the interval count reported at t + dt.
        interval_ber = np.r_[ber[0], ber[:-1]]
        corrected, uncorrectable = fec_counts(interval_ber, total, rng)
        result["ber" if direction == "downstream" else "upstream_ber"] = ber
        for name, values in (
            ("corrected", corrected),
            ("uncorrectable", uncorrectable),
            ("total", total),
        ):
            values = values.astype(float)
            values[0] = np.nan
            result[f"{direction}_fec_{name}_codewords"] = values
    return result


def validate_generated(
    native: pd.DataFrame, truth: pd.DataFrame, topology: pd.DataFrame
) -> dict:
    """Structural/physical invariants, not claims of empirical network realism."""
    checks = {
        "unique_samples": not native.duplicated(["device", "time"]).any(),
        "unique_topology": not topology.entity_id.duplicated().any(),
        "topology_covers_devices": set(native.device) == set(topology.entity_id),
        "truth_has_topology": set(truth.entity_id).issubset(set(topology.entity_id)),
        "fault_end_after_onset": bool((truth.end_time > truth.onset_time).all()),
    }
    impacts = truth.dropna(subset=["impact_time"])
    checks["impact_inside_fault"] = bool(
        (
            (impacts.impact_time >= impacts.onset_time)
            & (impacts.impact_time < impacts.end_time)
        ).all()
    )
    for direction in ("downstream", "upstream"):
        prefix = f"{direction}_fec_"
        counts = native[
            [prefix + x + "_codewords" for x in ("corrected", "uncorrectable", "total")]
        ].dropna()
        checks[f"{direction}_count_conservation"] = bool(
            ((counts.iloc[:, 0] + counts.iloc[:, 1]) <= counts.iloc[:, 2]).all()
        )
        checks[f"{direction}_nonnegative_integer_counts"] = bool(
            ((counts >= 0) & (counts == np.floor(counts))).all().all()
        )
    checks["bounded_ber"] = all(
        native[column].dropna().between(0, 1).all()
        for column in ("ber", "upstream_ber")
    )
    if not all(checks.values()):
        raise ValueError(
            f"Synthetic invariant failures: {[k for k, ok in checks.items() if not ok]}"
        )
    return {
        "checks": checks,
        "rows": len(native),
        "entities": native.device.nunique(),
        "faults": len(truth),
        "note": "Consistency checks, not field calibration.",
    }
