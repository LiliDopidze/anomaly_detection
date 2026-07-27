"""Simulation settings.

One dataclass. Everything that changes the data lives here, so a config fingerprint
over this object is a complete statement of what a run was.
"""
from __future__ import annotations

from dataclasses import dataclass


# ======================================================================================
# Configuration
# ======================================================================================


@dataclass
class TelecomSimulationSettings:
    # --- scale -----------------------------------------------------------------------
    n_onts: int = 400
    days: int = 180
    sample_minutes: int = 15
    start: str = "2025-01-01"
    seed: int = 20250717
    pre_window_days: int = 28          # left-censoring horizon

    # --- topology (E1, E2, E3) --------------------------------------------------------
    # v4: four OLTs, not two. With two, "cross-OLT correlation" is a comparison between
    # two groups and the topological gradient collapses to one binary factor.
    n_olts: int = 4
    pon_ports_per_olt: int = 8
    l1_split: int = 4
    # v4 (E1): capacity is a property of the installed splitter, not a global constant,
    # and take-up is partial. v3.1 filled every populated splitter to exactly 8.
    l2_capacity_choices: tuple = (8, 16, 32)
    l2_capacity_weights: tuple = (0.45, 0.35, 0.20)
    l2_takeup_beta_a: float = 4.0      # Beta(a, b) take-up fraction of capacity
    l2_takeup_beta_b: float = 2.2
    l2_min_fill: int = 2
    # v4 (E3): route length composes as feeder (PON) + distribution (L1) + drop (ONT),
    # so entities under one splitter share most of their loss budget.
    feeder_km_shape: float = 2.2
    feeder_km_scale: float = 1.5
    distribution_km_shape: float = 1.8
    distribution_km_scale: float = 0.45
    drop_m_shape: float = 2.0
    drop_m_scale: float = 55.0
    n_geo_clusters: int = 8

    # --- churn (E14) ------------------------------------------------------------------
    p_late_install: float = 0.06       # provisioned after t0 -> genuinely cold entities
    p_decommission: float = 0.035      # churned away before the window ends

    # --- optical budget (ITU-T G.984 class B+) ----------------------------------------
    olt_launch_dbm: float = 3.0
    l1_loss_db: float = 7.2
    fibre_loss_db_per_km: float = 0.35
    connector_loss_mean_db: float = 1.4
    connector_loss_sd_db: float = 0.45
    splices_per_km: float = 0.5
    splice_loss_mean_db: float = 0.12
    splice_loss_sd_db: float = 0.04
    n_extra_connectors: int = 2
    extra_connector_loss_mean_db: float = 0.55
    extra_connector_loss_sd_db: float = 0.20
    as_built_excess_shape: float = 2.0
    as_built_excess_scale: float = 1.3
    extra_plant_loss_min_db: float = 0.5
    extra_plant_loss_max_db: float = 7.0
    turnup_min_margin_db: float = 2.5

    # --- fault process ----------------------------------------------------------------
    fault_rate_per_ont_year: float = 1.0
    shared_fault_rate_per_node_year: float = 0.80   # per L2 node-year; scaled per level
    # Coarser plant is more reliable AND affects far more customers, so a flat per-node
    # rate at every level saturates the fleet: on the first v4 run, 5 OLT-scope faults
    # left only 1 of 400 entities fault-free. Scaled per level here.
    shared_scope_rate_scale: tuple = (1.0, 0.45, 0.30, 0.12)   # l2, l1, pon, olt
    # v4 (E9): gamma frailty and covariate hazard. Deliberately weak: susceptibility is
    # something a model should have to work to find, not a giveaway.
    frailty_shape: float = 1.6
    beta_fibre_age: float = 0.045      # per year
    beta_outdoor: float = 0.30         # outdoor enclosure log-hazard uplift
    beta_distance_km: float = 0.035
    beta_excess_loss_db: float = 0.10
    p_recurrence: float = 0.22         # repeat fault on a repaired entity
    recurrence_window_days: float = 45.0

    # --- storms / grouped causes (E11) ------------------------------------------------
    storms_per_year: float = 24.0
    storm_duration_h_mean: float = 20.0
    storm_geo_clusters_mean: float = 1.8
    storm_hazard_multiplier: float = 15.0
    storm_collector_outage_multiplier: float = 4.0

    # --- impact -----------------------------------------------------------------------
    impact_margin_db: float = 2.0
    impact_sustain_samples: int = 4
    impact_attribution_min_db: float = 0.8

    # --- reporting / ticketing --------------------------------------------------------
    report_depth_floor_db: float = 1.0
    report_hazard_per_db_day: float = 0.012
    report_hazard_cap_per_day: float = 0.35
    laser_report_lam_floor: float = 0.30
    laser_report_hazard_per_day: float = 0.06
    p_ticket_given_impact: float = 0.72     # v4 (E7): was 0.85; ticket coverage is worse
    nff_ticket_rate: float = 0.14           # v4 (E7): was 0.08
    p_duplicate_ticket: float = 0.10        # same customer reports twice
    p_misattributed_ticket: float = 0.04    # raised against the wrong line
    p_ticket_unresolved: float = 0.09       # closed without a resolution timestamp
    ticket_resolution_jitter_h: float = 6.0
    report_delay_mean_h: float = 3.5
    report_delay_heavy_tail_p: float = 0.15  # a minority report days later
    report_delay_heavy_mean_h: float = 60.0
    mttr_median_h: float = 16.0
    mttr_sigma: float = 0.62

    # --- proactive maintenance --------------------------------------------------------
    proactive_ont_visit_rate_per_year: float = 0.10
    proactive_splitter_visit_rate_per_year: float = 0.6
    proactive_onsite_hours: float = 2.0

    # --- label semantics --------------------------------------------------------------
    repaired_convalescence_days: int = 7

    # --- missingness (E10) ------------------------------------------------------------
    # v4: per-entity reliability, so "badly polled entity" exists as a nuisance class.
    poll_reliability_beta_a: float = 60.0
    poll_reliability_beta_b: float = 1.1
    p_field_nan: float = 0.004
    collector_outage_per_collector_per_month: float = 2.0
    collector_outage_hours_mean: float = 2.5
    n_collectors: int = 3
    # benign dense-gap processes, so a gap burst is not a fault oracle
    benign_outage_rate_per_ont_year: float = 3.0
    benign_outage_hours_mean: float = 7.0
    holiday_absence_rate_per_ont_year: float = 0.7
    holiday_absence_days_mean: float = 6.0
    # OD1: default off; they break the grid-reconstruction invariant
    poll_jitter_s: float = 0.0
    p_duplicate_poll: float = 0.0

    # --- quantisation at emission -----------------------------------------------------
    quant_rx_db: float = 0.1
    quant_tx_db: float = 0.1
    quant_temp_c: float = 1.0
    quant_bias_ma: float = 0.1
    quant_volt_v: float = 0.01
    quant_throughput_mbps: float = 0.1

    # --- benign operational change (E12) ----------------------------------------------
    benign_plant_step_rate: float = 0.6
    plant_step_improve_share: float = 0.75   # rework usually improves the line
    plant_step_mean_db: float = 0.35
    provisioning_change_rate_per_ont_year: float = 0.55
    firmware_rollouts: int = 3
    firmware_rx_step_db: float = 0.0         # firmware does not move optics...
    firmware_bias_step_ma: float = 0.55      # ...but does move the reported bias
    firmware_temp_step_c: float = 0.8
    planned_maintenance_per_pon_per_year: float = 1.2
    planned_maintenance_hours_mean: float = 3.5

    # --- benign anomaly layer (E5) ----------------------------------------------------
    p_sensor_glitch: float = 6.0e-5          # isolated implausible reading
    p_stuck_start: float = 2.0e-5            # stale value run begins
    stuck_run_mean_samples: float = 9.0
    p_transient_burst: float = 1.2e-5        # short multi-sample excursion
    transient_burst_mean_samples: float = 5.0
    transient_burst_rx_db: float = 0.9
    transient_burst_bias_ma: float = 1.4
    reranging_rate_per_ont_year: float = 2.2  # ONT re-ranges: short rx/bias step
    p_cpe_power_cycle_per_day: float = 0.012

    # --- healthy CRC floor ------------------------------------------------------------
    crc_burst_prob_per_sample: float = 2.5e-4
    crc_burst_lognorm_mu: float = 1.2
    crc_burst_lognorm_sigma: float = 1.1
    crc_noisy_plant_share: float = 0.08       # entities with chronically noisy plant
    crc_noisy_plant_multiplier: float = 12.0

    # --- explicit chronic cohort (E5/E12) ---------------------------------------------
    chronic_share: float = 0.035
    chronic_extra_loss_db: float = 1.1
    chronic_noise_multiplier: float = 2.2

    # --- shared hierarchy (E2) --------------------------------------------------------
    fleet_weather_tau_h: float = 72.0
    fleet_weather_sd_c: float = 1.4
    geo_weather_sd_c: float = 1.8
    fleet_demand_tau_h: float = 96.0
    fleet_demand_sd: float = 0.10
    olt_launch_tau_h: float = 168.0
    olt_launch_sd_db: float = 0.22
    pon_feeder_tau_h: float = 120.0
    pon_feeder_sd_db: float = 0.10
    pon_congestion_tau_h: float = 12.0
    pon_congestion_sd: float = 0.18
    # v4 (E2): the two levels that did not exist
    l1_loss_tau_h: float = 96.0
    l1_loss_sd_db: float = 0.075
    l2_loss_tau_h: float = 60.0
    l2_loss_sd_db: float = 0.085
    l2_micro_weather_sd_c: float = 1.0        # cabinet-level thermal environment
    l2_micro_weather_tau_h: float = 30.0

    # --- thermal fault modulation -----------------------------------------------------
    thermal_base_frac: float = 0.60
    thermal_temp_gain: float = 0.35

    # --- microclimate -----------------------------------------------------------------
    microclimate_tau_h: float = 36.0
    solar_exposure_low: float = 0.55
    solar_exposure_high: float = 1.45
    sensor_noise_scale: float = 1.0

    # --- laser / device channels (E4) -------------------------------------------------
    tx_device_sd_db: float = 0.65             # device-to-device transmit spread
    tx_temp_coeff_db_per_c: float = -0.012
    tx_age_db_per_yr: float = -0.045
    tx_ar_tau_h: float = 48.0
    tx_ar_sd_db: float = 0.16
    volt_device_sd_v: float = 0.031
    volt_temp_coeff_v_per_c: float = -0.0016
    volt_load_coeff_v: float = -0.022         # rail sags under traffic load
    volt_ar_tau_h: float = 30.0
    volt_ar_sd_v: float = 0.007
    bias_device_noise_shape: float = 3.0
    bias_device_noise_scale: float = 0.085

    # --- error cascade (E6) -----------------------------------------------------------
    ber_implementation_penalty_sd_db: float = 0.85
    ber_temp_penalty_db_per_c: float = 0.018
    ber_slope_sd: float = 0.10                # per-device waterfall slope spread
    fec_overdispersion_k: float = 6.0         # negative-binomial shape (burstiness)
    fec_report_floor: int = 0
    crc_overdispersion_k: float = 2.0

    # --- demand metric (E4/E12) -------------------------------------------------------
    thr_base_log_mean: float = 4.79
    thr_base_log_sd: float = 0.50
    thr_base_min_mbps: float = 20.0
    thr_base_max_mbps: float = 800.0
    thr_floor: float = 0.35
    thr_morning_amp: float = 0.25
    thr_morning_peak_h: float = 9.0
    thr_morning_sd_h: float = 1.8
    thr_evening_amp: float = 0.90
    thr_evening_peak_h: float = 20.5
    thr_evening_sd_h: float = 2.2
    thr_weekend_factor: float = 1.15
    thr_ar_tau_h: float = 6.0
    thr_ar_sd: float = 0.22
    thr_impair_floor: float = 0.25
    thr_laser_impair: float = 0.70
    # v4: household archetypes, so the diurnal profile is not one fleet-wide shape
    thr_profile_weights: tuple = (0.45, 0.25, 0.20, 0.10)   # evening / day / night / flat

    # --- provenance / regime ----------------------------------------------------------
    config_role: str = "benchmark_enriched"
    # The scenario is part of the dataset's IDENTITY, not a comment. Without it, two
    # scenario-grid runs publish to directories distinguishable only by seed, and a
    # downstream result cannot name the environment it was computed under.
    scenario: str = "default"

    # --- output -----------------------------------------------------------------------
    out_path: str = "reference_dataset.parquet"
    batch_onts: int = 25
