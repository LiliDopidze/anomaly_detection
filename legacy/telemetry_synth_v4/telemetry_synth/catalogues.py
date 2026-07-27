"""Fault taxonomy, device models, enclosures, firmware and demand archetypes.

Data, not behaviour. Kept separate so the taxonomy can be reviewed and argued
with on its own -- it is the part a telecoms engineer will want to challenge.
"""
from __future__ import annotations


# ======================================================================================
# Catalogues
# ======================================================================================

# v4 (E8): magnitudes are lognormal, not uniform, so every optical family carries real
# mass BELOW the practical wander anchor (~0.14 dB day-over-day on this fixture). A
# fixture in which every fault is eventually visible cannot measure missed detection.
# `mag_mu`/`mag_sigma` are the lognormal parameters in dB; `mag_max` clips the tail.
FAULT_TYPES = {
    "connector_contamination": dict(
        weight=0.14, scope="ont", channel="optical", family="optical_ramp",
        onset_lead_h=(72, 240), shape="exp_ramp", mag_mu=0.20, mag_sigma=1.15, mag_max=9.0,
        recovery="full", scar_db=(0.0, 0.15), p_natural=0.25, natural_dwell_days=20.0,
        storm_sensitive=False),
    "cable_damage": dict(
        weight=0.08, scope="ont", channel="optical", family="optical_step",
        onset_lead_h=(0.5, 4.0), shape="step", mag_mu=1.10, mag_sigma=1.00, mag_max=16.0,
        recovery="full", scar_db=(0.05, 0.30), p_natural=0.0, natural_dwell_days=0.0,
        storm_sensitive=True),
    "fibre_bend": dict(
        weight=0.11, scope="ont", channel="optical", family="optical_ramp",
        onset_lead_h=(24, 96), shape="lin_ramp", mag_mu=-0.10, mag_sigma=1.15, mag_max=6.0,
        recovery="partial", scar_db=(0.1, 0.5), p_natural=0.50, natural_dwell_days=10.0,
        storm_sensitive=False),
    # v4 (E8): a genuinely intermittent mechanism. Loose or moisture-affected connectors
    # come and go; a persistence-based detector must not be able to assume monotonicity.
    "intermittent_connector": dict(
        weight=0.09, scope="ont", channel="optical", family="optical_intermittent",
        onset_lead_h=(12, 96), shape="intermittent", mag_mu=0.35, mag_sigma=1.05,
        mag_max=8.0, recovery="full", scar_db=(0.0, 0.2), p_natural=0.35,
        natural_dwell_days=14.0, storm_sensitive=False),
    "water_ingress": dict(
        weight=0.07, scope="ont", channel="optical", family="optical_ramp",
        onset_lead_h=(36, 240), shape="partial_recovery", mag_mu=0.55, mag_sigma=1.00,
        mag_max=10.0, recovery="partial", scar_db=(0.15, 0.8), p_natural=0.30,
        natural_dwell_days=18.0, storm_sensitive=True),
    # v4 (E4): the laser mechanism is no longer a single 22 mA rail. On v3.1 every
    # hardware failure drove bias by 22*lam^2 mA against a fleet-constant 0.22 mA noise
    # floor, so a 6*MAD rule on bias caught 50 of 50 with zero false positives. Gain is
    # now drawn per fault, with a soft mode in which the APC barely moves and the
    # signature sits on tx instead.
    "ont_hardware_failure": dict(
        weight=0.14, scope="ont", channel="laser", family="laser",
        onset_lead_h=(24, 144), shape="lin_ramp", mag_mu=None, mag_sigma=None, mag_max=0.0,
        recovery="full", scar_db=(0.0, 0.0), p_natural=0.0, natural_dwell_days=0.0,
        storm_sensitive=False,
        laser_gain_log_mu=1.35, laser_gain_log_sigma=1.15, laser_gain_max_ma=30.0,
        p_soft_mode=0.35, soft_tx_drop_db=(0.4, 2.2)),
    "thermal_instability": dict(
        weight=0.19, scope="ont", channel="optical_thermal", family="thermal",
        onset_lead_h=(48, 192), shape="thermal", mag_mu=0.30, mag_sigma=1.05, mag_max=7.0,
        recovery="partial", scar_db=(0.1, 0.4), p_natural=0.0, natural_dwell_days=0.0,
        storm_sensitive=False),
    "aging_fibre": dict(
        weight=0.07, scope="ont", channel="optical", family="optical_ramp",
        onset_lead_h=(480, 1440), shape="progressive", mag_mu=0.45, mag_sigma=0.95,
        mag_max=7.0, recovery="partial", scar_db=(0.3, 1.0), p_natural=0.0,
        natural_dwell_days=0.0, storm_sensitive=False),
    # ---- shared-scope mechanisms, now at FOUR levels (E11) --------------------------
    "splitter_degradation": dict(
        weight=0.05, scope="l2", channel="optical", family="optical_ramp",
        onset_lead_h=(24, 120), shape="exp_ramp", mag_mu=0.15, mag_sigma=1.05, mag_max=6.5,
        recovery="full", scar_db=(0.0, 0.2), p_natural=0.0, natural_dwell_days=0.0,
        storm_sensitive=True),
    "distribution_damage": dict(
        weight=0.02, scope="l1", channel="optical", family="optical_step",
        onset_lead_h=(0.5, 6.0), shape="step", mag_mu=1.30, mag_sigma=0.80, mag_max=14.0,
        recovery="full", scar_db=(0.05, 0.35), p_natural=0.0, natural_dwell_days=0.0,
        storm_sensitive=True),
    "feeder_degradation": dict(
        weight=0.015, scope="pon", channel="optical", family="optical_ramp",
        onset_lead_h=(48, 300), shape="lin_ramp", mag_mu=0.25, mag_sigma=0.80, mag_max=5.0,
        recovery="partial", scar_db=(0.1, 0.6), p_natural=0.0, natural_dwell_days=0.0,
        storm_sensitive=True),
    "olt_card_degradation": dict(
        weight=0.005, scope="olt", channel="optical", family="optical_ramp",
        onset_lead_h=(24, 240), shape="lin_ramp", mag_mu=0.10, mag_sigma=0.70, mag_max=3.5,
        recovery="full", scar_db=(0.0, 0.15), p_natural=0.0, natural_dwell_days=0.0,
        storm_sensitive=False),
}

SHARED_SCOPES = {"l2": "splitter_l2", "l1": "splitter_l1", "pon": "pon_port", "olt": "olt_id"}

# v4 (E4): tx and voltage are now device properties, not fleet constants. `tx_nominal`
# and `volt_nominal` vary by model; the per-device draw around them is what v3.1 lacked.
DEVICE_MODELS = {
    "HG8245H":  dict(vendor="Huawei", weight=0.26, rx_sensitivity_dbm=-27.0,
                     bias_nominal_ma=34.0, bias_thermal_coeff=0.16, bias_age_ma_per_yr=0.30,
                     tx_nominal_dbm=2.60, volt_nominal_v=3.300, fec_scale=1.00,
                     impl_penalty_db=0.00),
    "HG8546M":  dict(vendor="Huawei", weight=0.18, rx_sensitivity_dbm=-28.0,
                     bias_nominal_ma=36.5, bias_thermal_coeff=0.14, bias_age_ma_per_yr=0.26,
                     tx_nominal_dbm=2.85, volt_nominal_v=3.295, fec_scale=1.00,
                     impl_penalty_db=0.15),
    "EG8145V5": dict(vendor="Huawei", weight=0.16, rx_sensitivity_dbm=-27.5,
                     bias_nominal_ma=32.0, bias_thermal_coeff=0.18, bias_age_ma_per_yr=0.34,
                     tx_nominal_dbm=3.10, volt_nominal_v=3.310, fec_scale=1.00,
                     impl_penalty_db=-0.10),
    "G-240W-B": dict(vendor="Nokia", weight=0.15, rx_sensitivity_dbm=-28.5,
                     bias_nominal_ma=30.5, bias_thermal_coeff=0.12, bias_age_ma_per_yr=0.22,
                     tx_nominal_dbm=2.20, volt_nominal_v=3.280, fec_scale=0.55,
                     impl_penalty_db=0.35),
    "G-010S-A": dict(vendor="Nokia", weight=0.10, rx_sensitivity_dbm=-27.8,
                     bias_nominal_ma=29.0, bias_thermal_coeff=0.13, bias_age_ma_per_yr=0.25,
                     tx_nominal_dbm=2.45, volt_nominal_v=3.285, fec_scale=0.55,
                     impl_penalty_db=0.25),
    "F660":     dict(vendor="ZTE", weight=0.09, rx_sensitivity_dbm=-26.5,
                     bias_nominal_ma=35.5, bias_thermal_coeff=0.19, bias_age_ma_per_yr=0.38,
                     tx_nominal_dbm=3.40, volt_nominal_v=3.320, fec_scale=2.10,
                     impl_penalty_db=0.45),
    "F670L":    dict(vendor="ZTE", weight=0.06, rx_sensitivity_dbm=-27.2,
                     bias_nominal_ma=33.0, bias_thermal_coeff=0.17, bias_age_ma_per_yr=0.32,
                     tx_nominal_dbm=3.15, volt_nominal_v=3.315, fec_scale=2.10,
                     impl_penalty_db=0.30),
}

ENCLOSURES = {
    "indoor_wall": dict(weight=0.55, offset_c=14.0, diurnal_factor=0.45,
                        seasonal_factor=0.70, sensor_noise_sd_c=0.35, micro_sd_c=0.8,
                        phase_sd_h=1.1),
    "indoor_cabinet": dict(weight=0.25, offset_c=17.0, diurnal_factor=0.28,
                           seasonal_factor=0.60, sensor_noise_sd_c=0.30, micro_sd_c=0.6,
                           phase_sd_h=1.1),
    "outdoor_cabinet": dict(weight=0.20, offset_c=8.0, diurnal_factor=1.60,
                            seasonal_factor=1.00, sensor_noise_sd_c=0.80, micro_sd_c=2.4,
                            phase_sd_h=1.8),
}

# v4 (E12): firmware is a real cohort with a rollout schedule, not a copy of the model.
FIRMWARE_BY_VENDOR = {
    "Huawei": ["V5R019C10", "V5R020C00", "V5R020C10"],
    "Nokia":  ["3FE-4.6.02", "3FE-4.7.01", "3FE-4.7.04"],
    "ZTE":    ["V2.1.3P4", "V2.2.0P1", "V2.2.1P2"],
}

THROUGHPUT_PROFILES = {
    "evening_peak": dict(morning=0.25, evening=0.90, night=0.05, floor=0.35),
    "day_worker":   dict(morning=0.70, evening=0.45, night=0.04, floor=0.40),
    "night_owl":    dict(morning=0.10, evening=0.55, night=0.65, floor=0.30),
    "flat_light":   dict(morning=0.15, evening=0.20, night=0.10, floor=0.55),
}
