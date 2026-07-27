"""Shared hierarchy and healthy per-entity channels, including the benign-anomaly layer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scipy.signal import lfilter


# ======================================================================================
# Shared hierarchy (E2) and healthy channels (E4, E5, E12)
# ======================================================================================


def _ar1(n, tau_h, sd, sample_minutes, rng):
    """AR(1) started from its stationary distribution (v3.1 C7, retained)."""
    phi = float(np.exp(-(sample_minutes / 60.0) / max(tau_h, 1e-6)))
    eps = rng.normal(0, sd * np.sqrt(max(1 - phi ** 2, 1e-12)), n)
    zi = np.array([phi * rng.normal(0, sd)])
    out, _ = lfilter([1.0], [1.0, -phi], eps, zi=zi)
    return out


def _weather_components(ts_h, day_frac):
    diurnal = 5.5 * np.cos(2 * np.pi * (ts_h - 15.0) / 24.0)
    seasonal = 4.0 * day_frac + 2.2 * np.sin(2 * np.pi * day_frac * 1.5)
    return seasonal, diurnal


def build_shared_hierarchy(cfg, topo, ts, rng):
    """Effects shared by every entity beneath a node.

    v4 (E2) adds the two levels v3.1 omitted. Measured on the v3.1 reference run, median
    healthy hourly rx correlation was 0.575 within an L2 splitter against 0.564 within a
    PON port but a different L2 -- the splitter level contributed 0.011, so the finest
    and most specific cohort in the leave-one-out ladder had no shared component to
    remove. The C5 gate compared same-PON against cross-OLT only, and so could not see
    this. `splitter_l1` gets a distribution-segment loss factor and `splitter_l2` gets
    both a splitter-loss factor and a cabinet micro-weather factor.
    """
    n = len(ts)
    fleet = dict(
        weather_anomaly_c=_ar1(n, cfg.fleet_weather_tau_h, cfg.fleet_weather_sd_c,
                               cfg.sample_minutes, rng),
        demand_shock=_ar1(n, cfg.fleet_demand_tau_h, cfg.fleet_demand_sd,
                          cfg.sample_minutes, rng),
    )
    geo = {g: dict(weather_anomaly_c=_ar1(n, cfg.fleet_weather_tau_h, cfg.geo_weather_sd_c,
                                          cfg.sample_minutes, rng))
           for g in sorted(topo.geo_cluster.unique())}
    olt = {o: dict(launch_drift_db=_ar1(n, cfg.olt_launch_tau_h, cfg.olt_launch_sd_db,
                                        cfg.sample_minutes, rng))
           for o in sorted(topo.olt_id.unique())}
    pon = {p: dict(
        feeder_loss_db=_ar1(n, cfg.pon_feeder_tau_h, cfg.pon_feeder_sd_db, cfg.sample_minutes, rng),
        congestion=_ar1(n, cfg.pon_congestion_tau_h, cfg.pon_congestion_sd, cfg.sample_minutes, rng))
        for p in sorted(topo.pon_port.unique())}
    l1 = {s: dict(dist_loss_db=_ar1(n, cfg.l1_loss_tau_h, cfg.l1_loss_sd_db,
                                    cfg.sample_minutes, rng))
          for s in sorted(topo.splitter_l1.unique())}
    l2 = {s: dict(spl_loss_db=_ar1(n, cfg.l2_loss_tau_h, cfg.l2_loss_sd_db,
                                   cfg.sample_minutes, rng),
                  micro_weather_c=_ar1(n, cfg.l2_micro_weather_tau_h, cfg.l2_micro_weather_sd_c,
                                       cfg.sample_minutes, rng))
          for s in sorted(topo.splitter_l2.unique())}
    return dict(fleet=fleet, geo=geo, olt=olt, pon=pon, l1=l1, l2=l2)


def simulate_healthy_ont_signals(cfg, row, ts, ts_h, day_frac, shared, rng):
    """Per-ONT healthy trajectories: physical temperature, rx, bias base, tx base,
    voltage base, plus the log of benign operational changes and benign anomalies.

    v4 additions, all measured defects in v3.1:

      E4  `tx_power_dbm` was 3.05 + 0.004 dB/degC + N(0, 0.045) for every device in the
          fleet (measured between-entity sd of healthy medians: 0.0497 dB over six
          distinct quantised values), and `voltage_v` was 3.30 + N(0, 0.006) with a
          measured between-entity sd of 0.0000 V. Both are now device properties with
          real temperature, load and ageing behaviour.
      E5  Healthy data was Gaussian noise on a smooth mean: across 105 clean ONTs and
          180 days, no entity ever exceeded 6 robust sigma on bias, and 96.2% never did
          on rx. Sensor glitches, stuck values, transient bursts and re-ranging steps
          are added here and logged to `gt_benign_anomalies`.
      E12 Firmware rollouts, provisioning changes and predominantly-improving plant
          rework replace v3.1's zero-mean plant step as the benign-change layer.
    """
    n = len(ts)
    seasonal, diurnal = _weather_components(ts_h + row.temp_phase_shift_h, day_frac)
    events, benign = [], []

    # --- temperature: fleet -> region -> cabinet -> enclosure -> ONT microclimate ------
    micro = _ar1(n, cfg.microclimate_tau_h, row.microclimate_sd_c, cfg.sample_minutes, rng)
    temp_phys = (row.enclosure_offset_c
                 + row.enclosure_seasonal_factor * seasonal
                 + row.enclosure_diurnal_factor * diurnal * row.solar_factor
                 + row.enclosure_seasonal_factor * (shared["fleet"]["weather_anomaly_c"]
                                                    + shared["geo"][row.geo_cluster]["weather_anomaly_c"])
                 + shared["l2"][row.splitter_l2]["micro_weather_c"]
                 + micro)

    # --- rx ----------------------------------------------------------------------------
    span_loss = (cfg.l1_loss_db + row.l2_loss_db
                 + row.distance_m / 1000.0 * cfg.fibre_loss_db_per_km
                 + row.connector_loss_db + row.extra_plant_loss_db)
    rx0 = cfg.olt_launch_dbm - span_loss
    ageing = -(0.02 / 30.0 / 24.0) * (1 + row.fibre_age_yr / 12.0) * np.arange(n) * (cfg.sample_minutes / 60.0)
    thermal = -0.018 * (temp_phys - temp_phys.mean())
    shared_optics = (shared["olt"][row.olt_id]["launch_drift_db"]
                     + shared["pon"][row.pon_port]["feeder_loss_db"]
                     + shared["l1"][row.splitter_l1]["dist_loss_db"]
                     + shared["l2"][row.splitter_l2]["spl_loss_db"])
    pink = np.cumsum(rng.normal(0, 0.004, n))
    pink -= pink.mean()

    # --- benign operational change (E12) ------------------------------------------------
    plant = np.zeros(n)
    for _ in range(rng.poisson(cfg.benign_plant_step_rate)):
        k = int(rng.integers(int(0.05 * n), n))
        improving = rng.random() < cfg.plant_step_improve_share
        delta = abs(rng.normal(0, cfg.plant_step_mean_db)) * (1.0 if improving else -1.0)
        plant[k:] += delta
        events.append(dict(entity_id=row.ont_id, ts_i=k, event_type="plant_rework",
                           level_change_db=round(float(delta), 3), detail=""))

    prov = np.zeros(n)
    for _ in range(rng.poisson(cfg.provisioning_change_rate_per_ont_year * cfg.days / 365.0)):
        k = int(rng.integers(int(0.05 * n), n))
        delta = float(rng.normal(0, 0.22))
        prov[k:] += delta
        events.append(dict(entity_id=row.ont_id, ts_i=k, event_type="provisioning_change",
                           level_change_db=round(delta, 3), detail="profile_change"))

    rx_healthy = rx0 + ageing + thermal + shared_optics + pink + plant + prov

    # --- bias: ONT age drives laser ageing ---------------------------------------------
    # v4: v3.1 wrote `bias_age_ma_per_yr * ont_age_yr / 12.0`, dividing an annual rate by
    # twelve, so a five-year-old ONT accumulated one twelfth of its ageing and `ont_age_yr`
    # had essentially no observable footprint.
    in_window_yr = np.arange(n) * (cfg.sample_minutes / 60.0) / (24.0 * 365.0)
    bias_base = (row.bias_nominal_ma
                 + row.bias_thermal_coeff * (temp_phys - 25.0)
                 + row.bias_age_ma_per_yr * row.ont_age_yr
                 + row.bias_age_ma_per_yr * in_window_yr)

    # --- tx: device offset, temperature coefficient, ageing, slow APC wander (E4) -------
    tx_base = (row.tx_nominal_dbm
               + cfg.tx_temp_coeff_db_per_c * (temp_phys - 25.0)
               + cfg.tx_age_db_per_yr * (row.ont_age_yr + in_window_yr)
               + _ar1(n, cfg.tx_ar_tau_h, cfg.tx_ar_sd_db * row.tx_noise_scale,
                      cfg.sample_minutes, rng))

    # --- voltage: device offset, temperature, load, slow wander (E4) --------------------
    volt_base = (row.volt_nominal_v
                 + cfg.volt_temp_coeff_v_per_c * (temp_phys - 25.0)
                 + _ar1(n, cfg.volt_ar_tau_h, cfg.volt_ar_sd_v * row.volt_noise_scale,
                        cfg.sample_minutes, rng))

    # --- firmware rollout: a step in reported bias and temperature, not in optics -------
    fw_step_i = None
    if rng.random() < 0.55:
        fw_step_i = int(rng.integers(int(0.15 * n), int(0.9 * n)))
        bias_base[fw_step_i:] += float(rng.normal(cfg.firmware_bias_step_ma, 0.15))
        temp_phys[fw_step_i:] += float(rng.normal(cfg.firmware_temp_step_c, 0.25))
        events.append(dict(entity_id=row.ont_id, ts_i=fw_step_i, event_type="firmware_upgrade",
                           level_change_db=0.0, detail="reported_counters_rebased"))

    # --- benign anomalies: the reason a 6-sigma rule must have a false-alarm rate (E5) --
    # Re-ranging: the ONT re-acquires the PON and its reported rx/bias step for a while.
    for _ in range(rng.poisson(cfg.reranging_rate_per_ont_year * cfg.days / 365.0)):
        k = int(rng.integers(0, n - 1))
        ln = int(np.clip(rng.exponential(20.0), 2, 300))
        rx_healthy[k:k + ln] += float(rng.normal(0, 0.55))
        bias_base[k:k + ln] += float(rng.normal(0, 1.1))
        tx_base[k:k + ln] += float(rng.normal(0, 0.45))
        benign.append(dict(entity_id=row.ont_id, ts_i=k, n_samples=int(ln),
                           gt_benign_type="re_ranging"))
    # Transient bursts: reflection or interference events on one or both channels.
    for k in np.flatnonzero(rng.random(n) < cfg.p_transient_burst):
        ln = int(np.clip(rng.exponential(cfg.transient_burst_mean_samples), 2, 60))
        sgn = 1.0 if rng.random() < 0.5 else -1.0
        rx_healthy[k:k + ln] += sgn * abs(rng.normal(0, cfg.transient_burst_rx_db))
        bias_base[k:k + ln] += sgn * abs(rng.normal(0, cfg.transient_burst_bias_ma))
        benign.append(dict(entity_id=row.ont_id, ts_i=int(k), n_samples=int(ln),
                           gt_benign_type="transient_burst"))

    return (temp_phys.astype(np.float32), rx_healthy.astype(np.float32),
            bias_base.astype(np.float32), tx_base.astype(np.float32),
            volt_base.astype(np.float32), events, benign)
