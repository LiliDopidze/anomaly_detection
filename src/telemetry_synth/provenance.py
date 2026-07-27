"""Parameter provenance and the named configurations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .settings import TelecomSimulationSettings


# ======================================================================================
# Provenance and named configurations
# ======================================================================================


def parameter_provenance(cfg=None) -> pd.DataFrame:
    """Status and evidence for every material parameter.

    Statuses: vendor_specification | engineering_estimate | uncalibrated_assumption |
    benchmark_tuning. `benchmark_tuning` means the value was set to hit a gate target on
    THIS fixture. Those values are properties of the benchmark and must never be quoted
    as operator facts.
    """
    cfg = cfg or TelecomSimulationSettings()
    enriched = cfg.config_role == "benchmark_enriched"
    rows = [
        ("olt_launch_dbm", cfg.olt_launch_dbm, "vendor_specification", "ITU-T G.984 class B+ OLT transmit power"),
        ("l1_loss_db", cfg.l1_loss_db, "vendor_specification", "Typical 1:4 splitter insertion loss"),
        ("l2 insertion loss (8/16/32)", "10.5 / 13.8 / 17.0", "vendor_specification",
         "Typical insertion loss by secondary split ratio"),
        ("fibre_loss_db_per_km", cfg.fibre_loss_db_per_km, "vendor_specification", "G.652 attenuation at 1490 nm"),
        ("rx_sensitivity_dbm (per model)", "-26.5 to -28.5", "vendor_specification", "Class B+ ONT receiver sensitivity"),
        ("tx_nominal_dbm (per model)", "2.2 to 3.4", "vendor_specification", "Class B+ ONT launch power range"),
        ("quant_rx_db / quant_temp_c", f"{cfg.quant_rx_db} / {cfg.quant_temp_c}", "engineering_estimate",
         "OMCI/OLT reporting granularity"),
        ("impact_margin_db", cfg.impact_margin_db, "engineering_estimate", "Headroom at which service degrades"),
        ("cascade waterfall exponent (9.0)", 9.0, "engineering_estimate", "RS(255,239) t=8: t+1 decades per input decade"),
        ("l2_capacity_choices / weights", f"{cfg.l2_capacity_choices} / {cfg.l2_capacity_weights}",
         "engineering_estimate", "v4 (E1): mixed 1:8/1:16/1:32 secondary splitters"),
        ("l2_takeup_beta_a/b", f"{cfg.l2_takeup_beta_a}/{cfg.l2_takeup_beta_b}", "uncalibrated_assumption",
         "v4 (E1): partial take-up per installed splitter; v3.1 filled every splitter to exactly 8"),
        ("feeder/distribution/drop length", "gamma composition", "engineering_estimate",
         "v4 (E3): route length shared down the tree; v3.1 drew it i.i.d. per ONT"),
        ("tx_device_sd_db / volt_device_sd_v", f"{cfg.tx_device_sd_db} / {cfg.volt_device_sd_v}",
         "engineering_estimate", "v4 (E4): device-to-device spread; v3.1 emitted both channels as fleet constants"),
        ("tx_temp_coeff_db_per_c", cfg.tx_temp_coeff_db_per_c, "engineering_estimate", "APC-controlled laser thermal drift"),
        ("volt_temp_coeff_v_per_c / volt_load_coeff_v",
         f"{cfg.volt_temp_coeff_v_per_c} / {cfg.volt_load_coeff_v}", "uncalibrated_assumption",
         "v4 (E4): rail behaviour under temperature and traffic load"),
        ("ber_implementation_penalty_sd_db", cfg.ber_implementation_penalty_sd_db, "engineering_estimate",
         "v4 (E6): receiver-to-receiver implementation penalty; v3.1 used one global curve"),
        ("fec_overdispersion_k / crc_overdispersion_k", f"{cfg.fec_overdispersion_k} / {cfg.crc_overdispersion_k}",
         "benchmark_tuning", "v4 (E6): set so margin is not recoverable from fec to better than ~1 dB"),
        ("fec_scale (per vendor)", "0.55 / 1.00 / 2.10", "uncalibrated_assumption",
         "v4 (E6): vendors do not scale FEC counters alike; magnitudes are a guess"),
        ("l1_loss_sd_db / l2_loss_sd_db", f"{cfg.l1_loss_sd_db} / {cfg.l2_loss_sd_db}", "uncalibrated_assumption",
         "v4 (E2): the two shared levels v3.1 omitted; amplitudes uncalibrated"),
        ("l2_micro_weather_sd_c", cfg.l2_micro_weather_sd_c, "uncalibrated_assumption", "v4 (E2): cabinet thermal environment"),
        ("olt_launch_sd_db / pon_feeder_sd_db", f"{cfg.olt_launch_sd_db} / {cfg.pon_feeder_sd_db}",
         "uncalibrated_assumption", "Shared drift amplitudes"),
        ("frailty_shape", cfg.frailty_shape, "benchmark_tuning",
         "v4 (E9): set to give per-entity fault counts variance/mean ~1.5; v3.1 measured 0.93 (exactly Poisson)"),
        ("beta_fibre_age / beta_outdoor / beta_distance_km / beta_excess_loss_db",
         f"{cfg.beta_fibre_age}/{cfg.beta_outdoor}/{cfg.beta_distance_km}/{cfg.beta_excess_loss_db}",
         "uncalibrated_assumption", "v4 (E9): deliberately weak susceptibility signal (OD4)"),
        ("p_recurrence / recurrence_window_days", f"{cfg.p_recurrence}/{cfg.recurrence_window_days}",
         "uncalibrated_assumption", "v4 (E9): repeat-offender behaviour; no operator data behind it"),
        ("storms_per_year / duration / footprint",
         f"{cfg.storms_per_year}/{cfg.storm_duration_h_mean}h/{cfg.storm_geo_clusters_mean}",
         "benchmark_tuning", "v4 (E11): set to populate group_id on >=10% of faults with >=3 multi-node groups (OD3)"),
        ("storm_hazard_multiplier", cfg.storm_hazard_multiplier, "benchmark_tuning", "As above (OD3)"),
        ("FAULT_TYPES magnitude lognormals", "mag_mu/mag_sigma per type", "benchmark_tuning",
         "v4 (E8): set so ~15% of optical faults sit below 0.5 dB; v3.1 used uniforms floored at 1.5 dB"),
        ("intermittent duty cycle", "on ~1.5 h / off ~7.5 h", "uncalibrated_assumption", "v4 (E8): no field basis"),
        ("poll_reliability_beta_a/b", f"{cfg.poll_reliability_beta_a}/{cfg.poll_reliability_beta_b}",
         "benchmark_tuning", "v4 (E10): set so per-entity missingness spans an order of magnitude; v3.1 spanned 1.26x"),
        ("benign_outage / holiday_absence rates",
         f"{cfg.benign_outage_rate_per_ont_year}/{cfg.holiday_absence_rate_per_ont_year}",
         "benchmark_tuning", "v4 (E10): set so dense gap bursts are not exclusive to faulty entities (OD5)"),
        ("p_sensor_glitch / p_stuck_start / p_transient_burst / reranging_rate",
         f"{cfg.p_sensor_glitch}/{cfg.p_stuck_start}/{cfg.p_transient_burst}/{cfg.reranging_rate_per_ont_year}",
         "benchmark_tuning",
         "v4 (E5): set so a 6*MAD rule has a non-zero clean-fleet false-alarm rate; on v3.1 it was exactly zero on bias (OD5)"),
        ("chronic_share / chronic_extra_loss_db", f"{cfg.chronic_share}/{cfg.chronic_extra_loss_db}",
         "uncalibrated_assumption", "v4: chronic cohort is now declared, not emergent"),
        ("crc_noisy_plant_share / multiplier", f"{cfg.crc_noisy_plant_share}/{cfg.crc_noisy_plant_multiplier}",
         "uncalibrated_assumption", "Entities with chronically noisy plant"),
        ("p_late_install / p_decommission", f"{cfg.p_late_install}/{cfg.p_decommission}",
         "uncalibrated_assumption", "v4 (E14): churn, so the cold-entity protocol has unseen entities"),
        ("nff_ticket_rate", cfg.nff_ticket_rate, "uncalibrated_assumption", "v4 (E7): raised from 0.08; still a guess"),
        ("p_ticket_given_impact", cfg.p_ticket_given_impact, "uncalibrated_assumption", "v4 (E7): lowered from 0.85"),
        ("p_duplicate_ticket / p_misattributed_ticket / p_ticket_unresolved",
         f"{cfg.p_duplicate_ticket}/{cfg.p_misattributed_ticket}/{cfg.p_ticket_unresolved}",
         "uncalibrated_assumption", "v4 (E7): ticket-channel noise; no operator data behind it"),
        ("mttr_median_h / mttr_sigma", f"{cfg.mttr_median_h}/{cfg.mttr_sigma}", "uncalibrated_assumption",
         "Engineering assumption; no operator data behind it"),
        ("report_delay heavy tail", f"{cfg.report_delay_heavy_tail_p} at {cfg.report_delay_heavy_mean_h}h",
         "uncalibrated_assumption", "v4 (E7): a minority of customers report days late"),
        ("repaired_convalescence_days", cfg.repaired_convalescence_days, "uncalibrated_assumption",
         "OPEN DECISION pending sign-off"),
        ("benign_plant_step_rate / improve_share",
         f"{cfg.benign_plant_step_rate}/{cfg.plant_step_improve_share}", "uncalibrated_assumption",
         "v4 (E12): rework now predominantly improves the line; v3.1 steps were zero-mean"),
        ("firmware_bias_step_ma / firmware_temp_step_c",
         f"{cfg.firmware_bias_step_ma}/{cfg.firmware_temp_step_c}", "uncalibrated_assumption",
         "v4 (E12): firmware rebases reported counters without moving the optics"),
        ("planned_maintenance_per_pon_per_year", cfg.planned_maintenance_per_pon_per_year,
         "uncalibrated_assumption", "v4 (E12): maintenance now suppresses collection"),
        ("poll_jitter_s / p_duplicate_poll", f"{cfg.poll_jitter_s}/{cfg.p_duplicate_poll}",
         "uncalibrated_assumption", "v4 (E14): OFF by default -- enabling requires a contract amendment (OD1)"),
        ("fault_rate_per_ont_year", cfg.fault_rate_per_ont_year,
         "benchmark_tuning" if enriched else "uncalibrated_assumption",
         "Enriched above field prevalence for per-family statistics; field regime is a guess"),
        ("shared_fault_rate_per_node_year", cfg.shared_fault_rate_per_node_year,
         "benchmark_tuning" if enriched else "uncalibrated_assumption", "As above, for shared-mode faults"),
    ]
    prov = pd.DataFrame(rows, columns=["parameter", "value", "status", "evidence"])
    prov.insert(0, "config_role", cfg.config_role)
    return prov


def build_reference_configs(base_seed: int = 20250717) -> dict:
    return {
        "benchmark_enriched": TelecomSimulationSettings(seed=base_seed, config_role="benchmark_enriched"),
        "field_prevalence": TelecomSimulationSettings(
            scenario="field_prevalence", seed=base_seed, config_role="field_prevalence",
            fault_rate_per_ont_year=0.35, shared_fault_rate_per_node_year=0.30,
            storms_per_year=10.0),
    }


def build_scenario_grid(base_seed: int = 20250717) -> dict:
    return {
        "default_reference": TelecomSimulationSettings(seed=base_seed, scenario="default_reference"),
        "field_prevalence": TelecomSimulationSettings(
            scenario="field_prevalence", seed=base_seed + 1, config_role="field_prevalence",
            fault_rate_per_ont_year=0.35, shared_fault_rate_per_node_year=0.30,
            storms_per_year=10.0),
        "tight_margins": TelecomSimulationSettings(
            scenario="tight_margins", seed=base_seed + 2, extra_connector_loss_mean_db=0.85, splices_per_km=0.8),
        "noisy_sensors": TelecomSimulationSettings(scenario="noisy_sensors", seed=base_seed + 3, sensor_noise_scale=2.0),
        "hourly_polling": TelecomSimulationSettings(scenario="hourly_polling", seed=base_seed + 4, sample_minutes=60),
        # v4: poor data quality is a first-class environment, not a nuisance
        "poor_collection": TelecomSimulationSettings(
            scenario="poor_collection", seed=base_seed + 5, poll_reliability_beta_a=25.0, poll_reliability_beta_b=2.2,
            collector_outage_per_collector_per_month=5.0, benign_outage_rate_per_ont_year=6.0),
        "storm_season": TelecomSimulationSettings(scenario="storm_season", seed=base_seed + 6, storms_per_year=45.0),
    }
