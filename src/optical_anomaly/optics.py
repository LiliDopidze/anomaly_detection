"""GPON-inspired optics and FEC, with explicitly assumed receiver behaviour.

Source [g984] in METHOD.md: ITU-T G.984.3 (2014), section 6.2 (rates),
13.1.1 and Annex A.3 (RS coding), 13.1.3 (counter meanings).
https://www.itu.int/rec/T-REC-G.984.3-201401-I
This supports a GPON scenario, not XGS-PON or an equipment BER calibration.
"""

import numpy as np
import pandas as pd
from scipy.stats import binom


# Source: G.984.3 (2014), Annex A.3; full-length RS(255,239) byte symbols.
CODEWORD_BYTES = 255
CORRECTABLE_SYMBOLS = (255 - 239) // 2
# Source: G.984.3 (2014), section 6.2; one supported GPON rate pair.
DOWNSTREAM_BITS_PER_SECOND = 2.48832e9
UPSTREAM_BITS_PER_SECOND = 1.24416e9


def fec_probabilities(ber: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ideal RS(255,239): correct 1..8 erroneous byte symbols.

    Independent bit/symbol errors are a simplification, not a burst-error model.
    """
    ber = np.asarray(ber, dtype=float)
    if not np.isfinite(ber).all() or ((ber < 0) | (ber > 1)).any():
        raise ValueError("BER must contain finite probabilities in [0, 1]")
    # Independent-error assumption: a byte fails if any of its eight bits do.
    with np.errstate(divide="ignore"):
        symbol_error = -np.expm1(8 * np.log1p(-ber))
    uncorrectable = binom.sf(CORRECTABLE_SYMBOLS, CODEWORD_BYTES, symbol_error)
    corrected = np.clip(
        binom.sf(0, CODEWORD_BYTES, symbol_error) - uncorrectable, 0, 1
    )
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
    receiver_references: tuple[float, float],
    interval_minutes: int,
    port_entities: int,
    seed: np.random.SeedSequence,
    receiver_offset_halfwidth_db: float = 1.5,  # Assumed device spread, not spec.
    log10_noise_sd: float = 0.15,  # Assumed stochastic BER dispersion.
) -> dict[str, np.ndarray]:
    """Interval-ending counts; full downstream coding, equal upstream allocations.

    Assume FEC is enabled in both directions (optional in GPON, [g984] 13.1.2).
    Ignore frame/burst overhead and shortened last words ([g984] 13.2/13.3).
    These are counts per observation
    interval, NOT cumulative device counters. No interval exists at the first row.
    """
    rng = np.random.default_rng(seed)
    n = len(downstream)
    hour = (np.arange(n) - 1) * interval_minutes / 60
    # Assumed daily 30--70% aggregate upstream use, not measured traffic.
    utilisation = 0.5 + 0.2 * np.sin(2 * np.pi * hour / 24)
    result = {}
    for direction, power, threshold, rate, allocation in (
        (
            "downstream", downstream, receiver_references[0],
            DOWNSTREAM_BITS_PER_SECOND, np.ones(n),
        ),
        (
            "upstream",
            upstream,
            receiver_references[1],
            UPSTREAM_BITS_PER_SECOND,
            # Assumed equal grants; no dynamic bandwidth allocation model.
            utilisation / port_entities,
        ),
    ):
        # Assumed uniform receiver offset, independent of impact labels.
        reference = threshold + rng.uniform(
            -receiver_offset_halfwidth_db, receiver_offset_halfwidth_db
        )
        # Assumed 1e-5 at the anchor, tenfold BER per dB of loss, bounds
        # 1e-12..0.1. No standard or paper establishes this receiver curve.
        ber = 10 ** np.clip(-5 - (power - reference), -12, -1)
        total = np.floor(
            rate * interval_minutes * 60 * allocation / (CODEWORD_BYTES * 8)
        ).astype(np.int64)
        # Piecewise-constant physical state over the PRECEDING interval.
        # A change at t first affects the interval count reported at t + dt.
        # Assumed lognormal multiplicative jitter, median 1, capped BER 0.1.
        interval_ber = np.clip(
            np.r_[ber[0], ber[:-1]] * 10 ** rng.normal(0, log10_noise_sd, n),
            0,
            0.1,
        )
        corrected, uncorrectable = fec_counts(interval_ber, total, rng)
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
    # Schema check only: excluding BER names cannot prove absence of leakage.
    checks["no_label_leaking_columns"] = not bool(
        {"ber", "upstream_ber"}.intersection(native.columns)
    )
    ordered = truth.sort_values(["entity_id", "onset_time"])
    previous_end = ordered.groupby("entity_id").end_time.shift()
    checks["faults_do_not_overlap"] = bool(
        (ordered.onset_time.loc[previous_end.notna()] >= previous_end.dropna()).all()
    )
    visible = truth.dropna(subset=["observable_onset_time"])
    checks["observable_inside_fault"] = bool(
        (
            (visible.observable_onset_time >= visible.onset_time)
            & (visible.observable_onset_time < visible.end_time)
        ).all()
    )
    step = native.time.sort_values().drop_duplicates().diff().dropna().min()
    baseline_end = native.time.min() + 0.55 * (
        native.time.max() + step - native.time.min()
    )
    checks["baseline_fault_free"] = bool((truth.onset_time >= baseline_end).all())
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
