"""
telemetry_synth.generator
=========================

Synthetic GPON access-network telemetry generator, **v4.0.0**.

v4.0 responds to an independent empirical audit of v3.1.0 (reference run: 400 ONTs
x 180 days, seed 20250717, both v3.1 gates green). Every item in the E-series below
is a *measured* defect on that run, not a realism preference. Measurements are quoted
so that the fix can be checked against the number it was meant to move, and each has a
gate check (E1-E14) so it is guarded by the harness rather than by a comment.

The v3.1 corrections C1-C11 and the v2/v3 design goals D1-D18 are retained in full.
Where v4 changes a v3.1 mechanism it says so explicitly.

v4.0 corrections
----------------

  E1  TOPOLOGY IS DEGENERATE. v3.1 filled populated L2 splitters round-robin
      (`fill[splitter_order[i % len]] += 1`), so on the reference run every one of the
      50 populated splitters carried EXACTLY 8 ONTs -- fan-out standard deviation 0.00.
      Two gates ("no L2 exceeds capacity", "every peer group has >= 4 ONTs") passed
      vacuously, and the Week-5 node-level exceedance test, which must condition on
      cohort size, had no cohort-size variation to condition on. v4 draws a per-splitter
      capacity (1:8 / 1:16 / 1:32) and a Beta take-up fraction, giving a realistic and
      variable fill.

  E2  NO SHARED STRUCTURE AT THE SPLITTER LEVELS. `build_shared_hierarchy` emitted
      factors at fleet, geo, OLT and PON only. Measured on healthy hourly rx, median
      pairwise correlation was 0.575 same-L2 against 0.564 same-PON-different-L2: the
      L2 level added 0.011. The plan's leave-one-out cohort ladder starts at the L2
      splitter, so its finest and most specific cohort had nothing to remove.
      Furthermore the whole gradient was carried by a single binary OLT factor
      (same-OLT 0.473-0.575 vs cross-OLT 0.157) across only TWO OLTs. v4 adds L1 and L2
      shared AR(1) factors, a splitter-enclosure micro-weather term, and raises the
      default OLT count so cross-OLT is not a two-group comparison.

  E3  DISTANCE IS INCOHERENT WITHIN A SPLITTER. `dist_m` was drawn i.i.d. per ONT, so
      eight ONTs fed from one street cabinet had a median route-length spread of 7,080 m
      (within-L2 ICC 0.144). Physically impossible: they share a feeder. v4 composes
      distance as feeder (per PON port) + distribution (per L1) + drop (per ONT).

  E4  THE LASER FAMILY IS TRIVIALLY DETECTABLE, AND THE PUBLISHED DIFFICULTY FLOOR
      MEASURES A BASELINE BLIND SPOT RATHER THAN FIXTURE DIFFICULTY. v3.1 emitted
      `tx_power_dbm` as 3.05 + 0.004 dB/degC + N(0, 0.045) for every device -- measured
      between-entity standard deviation of healthy per-entity medians 0.0497 dB across
      SIX distinct quantised values -- and `voltage_v` as 3.30 - 0.05*lam + N(0, 0.006),
      measured between-entity standard deviation 0.0000 V. With no device-to-device
      heterogeneity and no benign dynamics on either channel, a per-entity 6*MAD rule on
      `bias_current_ma` detected 50 of 50 laser faults with 0 false positives among the
      105 clean ONTs; the same rule on `tx_power_dbm` also scored 50/50 with 0 false
      positives; `voltage_v <= 3.28 V` sustained 2 h scored 50/50 with 0 false
      positives. The union of the published naive rx rule and a 6*MAD bias rule reached
      0.87 event recall at 0/105 false positives, against a published floor of 0.74.
      v4 gives tx and voltage per-device offsets, real temperature and load coefficients,
      ageing, and benign excursions.

  E5  THE HEALTHY CLASS IS TOO CLEAN. Across the 105 clean ONTs and 180 days,
      per-entity max |robust z| on `bias_current_ma` reached 5.2 -- not one clean entity
      exceeded 6 sigma at any sample. On rx, 96.2% never exceeded 6 sigma. Healthy
      telemetry was Gaussian noise on a smooth deterministic mean, with no glitches,
      stale values, re-ranging steps or collection artefacts. v4 adds an explicit benign
      anomaly layer (sensor glitches, stuck values, transient bursts, re-ranging steps,
      CPE power cycles) recorded in `gt_benign_anomalies` so false alerts can be
      attributed rather than merely counted.

  E6  THE ERROR CASCADE IS A NOISELESS SECOND COPY OF OPTICAL MARGIN. A single global
      curve mapped margin to pre-FEC BER with no per-device, per-temperature or
      per-vendor variation, so `log10(fec_count)` recovered `gt_margin_db` with r =
      -0.988 and residual scatter of 0.30 decades, i.e. margin to +/- 0.33 dB (1 sd) --
      comparable to, or better than, the rx observable itself after its 0.1 dB
      quantisation. Separately, `ber` was documented as "heavily censored at low
      margins" but was exactly zero on only 0.48% of rows, so the `continuous_censored`
      signal class had no censoring to exercise. v4 adds a per-device implementation
      penalty, a temperature term, vendor counter scaling, negative-binomial burst
      overdispersion and a realistic reporting floor.

  E7  TICKET IDENTIFIERS LEAK THE FAULT REGISTRY. Tickets were emitted as
      `TKT-{fault_id[2:]}-{k:02d}` and no-fault-found tickets as `TKT-NFF{j:04d}`, so
      the canonical `service_tickets` table -- explicitly promoted to a *label channel*
      for the Week-2 ticket-proxy evaluation and the Week-5 localisation check -- grants
      exact fault grouping, exact cross-entity shared-fault grouping and exact NFF
      identification from a string prefix. Every ticket for a fault also shared one
      `resolved_ts` (260 of 260 faults had exactly one distinct value). v4 issues opaque
      identifiers, per-ticket resolution, duplicates and mis-attributions.

  E8  NO FAULT IS WEAK, AND ALMOST NONE IS INVISIBLE. Optical magnitudes were uniform
      with a floor of 1.5 dB against a median practical wander anchor near 0.14 dB, so
      only 1 of 293 faults (0.3%) never became observable. v4 draws magnitudes from
      lognormals with genuine mass below the anchor and adds intermittent, progressive
      and partially-self-healing trajectories.

  E9  NO RECURRENCE AND NO SUSCEPTIBILITY. Targets were drawn uniformly at random, so
      per-entity fault counts were exactly Poisson (variance/mean 0.925) and no static
      attribute predicted them (|r| < 0.10 for distance, fibre age, ONT age, connector
      loss and as-built excess loss). The plan's static-attribute susceptibility
      baseline therefore scores at chance by construction, and the Release-1.1 hazard
      model has nothing to learn. v4 adds per-entity gamma frailty, covariate-driven
      hazard and post-repair recurrence.

  E10 MISSINGNESS IS HOMOGENEOUS AND FAULT-EXCLUSIVE. Per-entity missing rate ran
      2.10% (p05) to 2.65% (p95) -- a 1.26x spread -- so there is no such thing as a
      badly-polled entity, and the plan's coverage-adjusted denominators have nothing to
      adjust. Loss-of-signal gaps occurred on faulty entities only, and no benign process
      produced dense per-entity gaps, so a gap burst is a fault oracle. v4 draws
      per-entity poll reliability, adds benign outage bursts (power cuts, CPE swaps,
      customer absence) and moves collector outages to a collector tier.

  E11 SHARED FAULTS EXIST AT ONE LEVEL ONLY, AND `group_id` IS NEVER POPULATED. All 34
      shared faults attached to an L2 splitter and each affected exactly 8 entities.
      Topology roll-up therefore has a single level to discover and no cohort-size
      variation, and the storm/grouped-cause mechanism the incident metrics require does
      not exist. v4 adds L1, PON-feeder and OLT-card mechanisms and a regional storm
      process populating `group_id`.

  E12 BENIGN OPERATIONAL CHANGE IS THIN. Plant steps existed (zero-mean, so half of
      them *improved* rx) but firmware cohorts were a placeholder equal to
      `device_model`, provisioning changes did not exist, and proactive maintenance
      repaired faults without appearing in telemetry at all. v4 adds firmware rollout
      cohorts with a step effect, provisioning/plan changes, planned-maintenance windows
      that suppress collection, and makes plant rework predominantly improving.

  E13 ENTITY IDENTIFIERS ENCODE THE TOPOLOGY. ONTs were numbered in nested
      OLT/port/splitter loop order: ONT index was monotone within every splitter block
      and correlated 0.866 with OLT index. Any entity-disjoint evaluation split taken in
      identifier order is a topology split, and any model consuming an entity index
      receives topology for free. v4 permutes identifiers.

  E14 NO CHURN AND NO IRREGULAR COLLECTION. Every entity was present for the whole
      window, so the plan's cold-entity generalisation protocol has no genuinely new
      entity to test; and every timestamp sat exactly on the cadence grid with no
      jitter, duplicates or late arrival, so the transform layer's resampling and gap
      handling are exercised against a no-op. v4 adds install/decommission churn and an
      optional poll-jitter mode (default off -- it requires a contract amendment, see
      OPEN DECISIONS).

Open decisions carried, not resolved
------------------------------------
  OD1  Poll jitter and duplicate polls (`poll_jitter_s`, `p_duplicate_poll`) default to
       zero because both break the SPEC-01 invariant "present polls plus recorded gaps
       reconstruct the configured grid". Enabling them by default requires a contract
       amendment. Scenario `irregular_polling` exercises them.
  OD2  Churn changes the same invariant to "present polls plus gaps reconstruct the
       grid *intersected with the entity's service window*". Implemented and gated, but
       downstream consumers must be told.
  OD3  Storm rate, footprint and hazard multipliers are uncalibrated.
  OD4  Frailty shape, covariate coefficients and recurrence multiplier are uncalibrated
       and deliberately weak; they are a *susceptibility signal to be discovered*, not a
       claim about any real fleet.
  OD5  Benign-anomaly rates are set to give a 6*MAD rule a non-zero clean-fleet false
       alarm rate. They are benchmark tuning and are recorded as such.

Units: dBm optical power, dB loss/margin, degC, mA, V, Mbps.
Ground truth: every column a real operator could not observe carries the `gt_` prefix.

ARCHITECTURAL NOTE (v4.0.1)
--------------------------
This module emits the generator's OWN NATIVE FORM only: a wide panel plus
sidecar tables. It does not know that a canonical schema exists, and it must
not. Mapping native to SPEC-01 is the job of
`telemetry_synth.adapter.SyntheticGponAdapter`, and materialising canonical
Parquet is the job of `telemetry_contract.materialise`. The generator is one
adapter among several, not the source of truth (plan section 6.3). v4.0.0 put
`emit_canonical_tables` inside this module, which inverted that dependency and
made the contract generator-specific; that is corrected here.
"""
from .catalogues import (DEVICE_MODELS, ENCLOSURES, FAULT_TYPES, FIRMWARE_BY_VENDOR,
                         SHARED_SCOPES, THROUGHPUT_PROFILES)
from .emit import generate_telecom_reference_data
from .faults import (apply_repair_to_fault_curve, build_fault_degradation_curve,
                     sample_fault_events, sample_storms, simulate_error_cascade)
from .provenance import build_reference_configs, build_scenario_grid, parameter_provenance
from .settings import TelecomSimulationSettings
from .signals import build_shared_hierarchy, simulate_healthy_ont_signals
from .topology import build_network_topology

FEATURE_BLOCKLIST_PREFIX = "gt_"
GENERATOR_VERSION = "4.0.1"
NATIVE_FORMAT_VERSION = "telemetry-synth native v4"

__all__ = ["FEATURE_BLOCKLIST_PREFIX", "GENERATOR_VERSION", "NATIVE_FORMAT_VERSION",
           "TelecomSimulationSettings", "FAULT_TYPES", "SHARED_SCOPES", "DEVICE_MODELS",
           "ENCLOSURES", "FIRMWARE_BY_VENDOR", "THROUGHPUT_PROFILES",
           "build_network_topology", "build_shared_hierarchy",
           "simulate_healthy_ont_signals", "sample_storms", "sample_fault_events",
           "build_fault_degradation_curve", "apply_repair_to_fault_curve",
           "simulate_error_cascade", "generate_telecom_reference_data",
           "parameter_provenance", "build_reference_configs", "build_scenario_grid"]
