"""Orchestration and native emission.

Writes the wide panel and its sidecars. It does not know that a canonical schema
exists: mapping native to canonical is an adapter concern, built later.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pathlib import Path
from dataclasses import asdict

import pyarrow as pa
import pyarrow.parquet as pq
from scipy.signal import lfilter

from .catalogues import FAULT_TYPES, SHARED_SCOPES, THROUGHPUT_PROFILES
from .faults import (apply_repair_to_fault_curve, build_fault_degradation_curve,
                     sample_fault_events, sample_storms, simulate_error_cascade)
from .provenance import parameter_provenance
from .settings import TelecomSimulationSettings
from .signals import build_shared_hierarchy, simulate_healthy_ont_signals
from .topology import build_network_topology


# ======================================================================================
# Main generation
# ======================================================================================


def generate_telecom_reference_data(cfg: TelecomSimulationSettings | None = None,
                                    out_dir: str | Path = "."):
    cfg = cfg or TelecomSimulationSettings()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    master = np.random.default_rng(cfg.seed)

    topo = build_network_topology(cfg, master)
    storms = sample_storms(cfg, topo, master)
    faults = sample_fault_events(cfg, topo, storms, master)

    n = int(cfg.days * 24 * 60 / cfg.sample_minutes)
    t0 = pd.Timestamp(cfg.start, tz="UTC")
    ts = t0 + pd.to_timedelta(np.arange(n) * cfg.sample_minutes, unit="m")
    ts_h = (ts.hour + ts.minute / 60.0).to_numpy()
    day_frac = np.arange(n) / n
    sph = 60 / cfg.sample_minutes
    spd = int(24 * sph)
    weekend = np.asarray(ts.dayofweek >= 5)

    shared = build_shared_hierarchy(cfg, topo, ts, np.random.default_rng([cfg.seed, 999_999]))

    # ---- Phase A: healthy channels ----------------------------------------------------
    chan = {}
    eng_log, benign_log = [], []
    for i, row in enumerate(topo.itertuples()):
        rng = np.random.default_rng([cfg.seed, i])
        temp_p, rxh, biasb, txb, voltb, events, benign = simulate_healthy_ont_signals(
            cfg, row, ts, ts_h, day_frac, shared, rng)
        chan[row.ont_id] = (temp_p, rxh, biasb, txb, voltb)
        eng_log.extend(events)
        benign_log.extend(benign)

    # ---- Phase A2: robust per-entity baseline wander (practical anchor) ---------------
    wander_rx, wander_bias = {}, {}
    for o, (temp_c, rxh_c, bias_c, _tx, _v) in chan.items():
        dm = rxh_c[: cfg.days * spd].astype(np.float64).reshape(cfg.days, spd).mean(1)
        wander_rx[o] = float(1.4826 * np.median(np.abs(np.diff(dm))))
        db = bias_c[: cfg.days * spd].astype(np.float64).reshape(cfg.days, spd).mean(1)
        wander_bias[o] = float(1.4826 * np.median(np.abs(np.diff(db))))

    # ---- service windows (E14) --------------------------------------------------------
    install_i = dict(zip(topo.ont_id, topo.install_i))
    decom_i = dict(zip(topo.ont_id, topo.decommission_i))

    # ---- Phase B: unrepaired degradation per entity -----------------------------------
    episodes = []
    curve_rng = np.random.default_rng(cfg.seed + 31)
    for f in faults.itertuples():
        if f.scope == "ont":
            targets = [f.target]
        else:
            targets = topo.loc[topo[SHARED_SCOPES[f.scope]] == f.target, "ont_id"].tolist()
        onset_i = int(round(f.onset_day * 24 * sph))
        if onset_i >= n:
            continue
        for o in targets:
            start = max(onset_i, install_i[o], 0)
            if start >= min(n, decom_i[o]):
                continue
            m = n - start
            temp = chan[o][0]
            tz_full = (temp - temp.mean()) / max(temp.std(), 1e-6)
            offset = max(0, start - onset_i)
            tz_curve = np.empty(m + offset, dtype=np.float64)
            tz_curve[offset:] = tz_full[start:start + m]
            if offset:
                tz_curve[:offset] = tz_full[start]
            curve = build_fault_degradation_curve(
                f.shape, m + offset, f.onset_lead_h * sph, f.magnitude_db, tz_curve,
                cfg.thermal_base_frac, cfg.thermal_temp_gain, curve_rng, sph)[offset:]
            episodes.append(dict(gt_fault_id=f.gt_fault_id, ont_id=o,
                                 gt_fault_type=f.gt_fault_type, scope=f.scope,
                                 target=f.target, onset_i=onset_i, start=start,
                                 curve=curve, channel=f.channel, recovery=f.recovery,
                                 scar_db=f.scar_db, onset_lead_h=f.onset_lead_h,
                                 left_censored=onset_i < 0, group_id=f.group_id,
                                 laser_gain_ma=f.laser_gain_ma,
                                 laser_tx_drop_db=f.laser_tx_drop_db))

    ep_by_fault = {}
    for ep in episodes:
        ep_by_fault.setdefault(ep["gt_fault_id"], []).append(ep)

    sens = topo.set_index("ont_id").rx_sensitivity_dbm.to_dict()
    noise_sd = topo.set_index("ont_id").noise_sd_db.to_dict()
    rng2 = np.random.default_rng(cfg.seed + 7)

    ont_visits, node_visits = {}, {}
    for o in topo.ont_id:
        k = rng2.poisson(cfg.proactive_ont_visit_rate_per_year * cfg.days / 365.0)
        ont_visits[o] = np.sort(rng2.integers(0, n, k)) if k else np.array([], dtype=int)
    for col in SHARED_SCOPES.values():
        for nd in topo[col].unique():
            k = rng2.poisson(cfg.proactive_splitter_visit_rate_per_year * cfg.days / 365.0)
            node_visits[nd] = np.sort(rng2.integers(0, n, k)) if k else np.array([], dtype=int)
    onsite_samples = int(round(cfg.proactive_onsite_hours * sph))

    # ---- Phase C: impact, reporting, repair economy -----------------------------------
    fault_rows, ticket_records = [], []
    for f in faults.itertuples():
        eps = ep_by_fault.get(f.gt_fault_id, [])
        if not eps:
            continue
        spec = FAULT_TYPES[f.gt_fault_type]

        impacts_cf = {}
        for ep in eps:
            o = ep["ont_id"]
            if ep["channel"] == "laser":
                k = int(ep["onset_i"] + f.onset_lead_h * sph * 0.8)
                k = max(k, ep["start"])
                impacts_cf[o] = k if 0 <= k < n else None
                continue
            healthy_from_start = chan[o][1].astype(np.float64)[ep["start"]:]
            episode_margin = healthy_from_start + ep["curve"] - sens[o]
            max_search = int(np.ceil(f.onset_lead_h * sph * 1.5)) + cfg.impact_sustain_samples
            episode_margin = episode_margin[:max_search]
            below = episode_margin < cfg.impact_margin_db
            if below.sum() < cfg.impact_sustain_samples:
                impacts_cf[o] = None
                continue
            c = np.convolve(below.astype(int), np.ones(cfg.impact_sustain_samples), "valid")
            hit = np.flatnonzero(c >= cfg.impact_sustain_samples)
            attribution_min = max(cfg.impact_attribution_min_db, 2.2 * noise_sd[o])
            k_attr = None
            for h in hit:
                if abs(ep["curve"][int(h)]) >= attribution_min:
                    k_attr = int(h)
                    break
            impacts_cf[o] = ep["start"] + k_attr if k_attr is not None else None

        cf_valid = [v for v in impacts_cf.values() if v is not None]
        cf_first_impact = min(cf_valid) if cf_valid else None

        obs_sensor, obs_practical, lam_u_by_ont = [], [], {}
        for ep in eps:
            o, s = ep["ont_id"], ep["start"]
            if ep["channel"] == "laser":
                lead_s = max(1.0, ep["onset_lead_h"] * sph)
                lam_u = np.clip((np.arange(len(ep["curve"])) + (s - ep["onset_i"])) / lead_s, 0, 1)
                lam_u_by_ont[o] = lam_u
                k_s = max(s, int(np.ceil(ep["onset_i"] + 0.1414 * ep["onset_lead_h"] * sph)))
                thr_b = max(2 * noise_sd[o], 4 * wander_bias[o])
                gain = max(ep.get("laser_gain_ma", 22.0), 1e-6)
                lam_thr = float(np.sqrt(min(thr_b / gain, 1.0)))
                k_p = max(s, int(np.ceil(ep["onset_i"] + lam_thr * ep["onset_lead_h"] * sph)))
                if k_s < n:
                    obs_sensor.append(k_s)
                if k_p < n:
                    obs_practical.append(k_p)
            else:
                thr_s = 2.0 * noise_sd[o]
                thr_p = max(thr_s, 4.0 * wander_rx[o])
                hits_s = np.flatnonzero(np.abs(ep["curve"]) > thr_s)
                hits_p = np.flatnonzero(np.abs(ep["curve"]) > thr_p)
                if len(hits_s) and s + int(hits_s[0]) < n:
                    obs_sensor.append(s + int(hits_s[0]))
                if len(hits_p) and s + int(hits_p[0]) < n:
                    obs_practical.append(s + int(hits_p[0]))
        first_obs_sensor = min(obs_sensor) if obs_sensor else None
        first_obs_practical = min(obs_practical) if obs_practical else None

        sev_reports = []
        for ep in eps:
            o, s = ep["ont_id"], ep["start"]
            if ep["channel"] == "laser":
                lam_u = lam_u_by_ont[o]
                frac = np.clip((lam_u - cfg.laser_report_lam_floor)
                               / max(1.0 - cfg.laser_report_lam_floor, 1e-6), 0, 1)
                p_day = cfg.laser_report_hazard_per_day * frac
            else:
                depth = np.abs(ep["curve"])
                p_day = np.clip(cfg.report_hazard_per_db_day * (depth - cfg.report_depth_floor_db),
                                0, cfg.report_hazard_cap_per_day)
            hit = np.flatnonzero(rng2.random(len(p_day)) < p_day / spd)
            if len(hit):
                sev_reports.append((s + int(hit[0]), o))
        first_sev = min(t for t, _ in sev_reports) if sev_reports else None

        visits = ont_visits[f.target] if f.scope == "ont" else node_visits.get(f.target, np.array([], dtype=int))
        vs = visits[visits >= max(int(round(f.onset_day * 24 * sph)), 0)]
        proactive_t = int(vs[0]) + onsite_samples if len(vs) else None
        if proactive_t is not None and proactive_t >= n:
            proactive_t = None
        natural_t = None
        if spec["p_natural"] > 0 and rng2.random() < spec["p_natural"]:
            dwell = rng2.exponential(spec["natural_dwell_days"]) * spd
            natural_t = int(max(int(round(f.onset_day * 24 * sph)), -10 ** 9)
                            + f.onset_lead_h * sph + dwell)
            natural_t = natural_t if 0 <= natural_t < n else None
        mttr_samples = int(rng2.lognormal(np.log(cfg.mttr_median_h), cfg.mttr_sigma) * sph)

        cand = [t for t in [(first_sev + mttr_samples) if first_sev is not None else None,
                            proactive_t, natural_t] if t is not None and t < n]
        cand_repair = min(cand) if cand else None
        impact_realised = cf_first_impact is not None and (cand_repair is None or cf_first_impact < cand_repair)

        imp_reports = []
        if impact_realised:
            for ep in eps:
                k_o = impacts_cf[ep["ont_id"]]
                if k_o is None or (cand_repair is not None and k_o >= cand_repair):
                    continue
                if rng2.random() < cfg.p_ticket_given_impact:
                    if rng2.random() < cfg.report_delay_heavy_tail_p:
                        delay = rng2.exponential(cfg.report_delay_heavy_mean_h)
                    else:
                        delay = rng2.exponential(cfg.report_delay_mean_h)
                    imp_reports.append((k_o + int(delay * sph), ep["ont_id"]))

        all_reports = sev_reports + imp_reports
        first_report = min(t for t, _ in all_reports) if all_reports else None
        repair_options = []
        if first_report is not None and first_report + mttr_samples < n:
            repair_options.append((first_report + mttr_samples, "ticket"))
        if proactive_t is not None:
            repair_options.append((proactive_t, "proactive"))
        if natural_t is not None:
            repair_options.append((natural_t, "natural"))
        repair_i, repair_source = min(repair_options, key=lambda x: x[0]) if repair_options else (None, None)

        impacts_real = {o: (k if (k is not None and (repair_i is None or k < repair_i)) else None)
                        for o, k in impacts_cf.items()}
        real_valid = [v for v in impacts_real.values() if v is not None]
        first_impact = min(real_valid) if (impact_realised and real_valid) else None
        averted = bool(cf_first_impact is not None and first_impact is None)

        valid_reports = sorted([(t, o) for t, o in all_reports
                                if t < n and (repair_i is None or t < repair_i)])
        # v4 (E7): duplicates and mis-attribution
        extra = []
        for t_rep, o_rep in valid_reports:
            if rng2.random() < cfg.p_duplicate_ticket:
                extra.append((min(n - 1, t_rep + int(rng2.exponential(8.0) * sph)), o_rep))
        valid_reports = sorted(valid_reports + extra)
        n_tickets = len(valid_reports)

        for t_rep, o_rep in valid_reports:
            k_o = impacts_real.get(o_rep)
            if k_o is not None and k_o <= t_rep:
                symptom = "no_service"
            elif f.gt_fault_type == "ont_hardware_failure" and o_rep in lam_u_by_ont and \
                    lam_u_by_ont[o_rep][min(max(t_rep - max(int(round(f.onset_day * 24 * sph)), 0), 0),
                                            len(lam_u_by_ont[o_rep]) - 1)] > 0.6:
                symptom = "intermittent"
            else:
                symptom = "slow_service"
            reported_on = o_rep
            misattributed = False
            if rng2.random() < cfg.p_misattributed_ticket:
                peers = topo.loc[topo.splitter_l2 == topo.set_index("ont_id").loc[o_rep, "splitter_l2"],
                                 "ont_id"].tolist()
                if len(peers) > 1:
                    reported_on = str(rng2.choice([p for p in peers if p != o_rep]))
                    misattributed = True
            ticket_records.append(dict(
                ont_id=reported_on, reported_i=t_rep, repair_i=repair_i,
                reported_symptom=symptom, gt_fault_id=f.gt_fault_id,
                gt_fault_type=f.gt_fault_type, gt_is_nff=False,
                gt_misattributed=misattributed))

        prodromal = bool(first_obs_sensor is not None and first_impact is not None
                         and (first_impact - first_obs_sensor) >= 12 * sph)
        prodromal_practical = bool(first_obs_practical is not None and first_impact is not None
                                   and (first_impact - first_obs_practical) >= 12 * sph)

        fault_rows.append(dict(
            gt_fault_id=f.gt_fault_id, gt_fault_type=f.gt_fault_type, family=spec["family"],
            scope=f.scope, target=f.target, group_id=f.group_id,
            gt_recurrence_of=f.gt_recurrence_of,
            onset_i=int(round(f.onset_day * 24 * sph)),
            impact_i=first_impact, counterfactual_impact_i=cf_first_impact, averted=averted,
            repair_i=repair_i, repair_source=repair_source, n_tickets=n_tickets,
            first_observable_i=first_obs_sensor, first_observable_practical_i=first_obs_practical,
            prodromal=prodromal, prodromal_practical=prodromal_practical,
            magnitude_db=f.magnitude_db, onset_lead_h=f.onset_lead_h,
            gt_laser_gain_ma=round(float(f.laser_gain_ma), 3),
            gt_laser_tx_drop_db=round(float(f.laser_tx_drop_db), 3),
            n_onts_affected=len(eps), left_censored=bool(f.onset_day < 0)))

        for ep in eps:
            ep["impact_i"] = impacts_real[ep["ont_id"]]
            ep["n_tickets"] = n_tickets
            ep["repair_i"] = repair_i

    faults_out = pd.DataFrame(fault_rows)
    return _emit_panel(cfg, topo, faults, faults_out, episodes, storms, chan, shared,
                       eng_log, benign_log, ticket_records, ts, ts_h, day_frac, weekend,
                       n, sph, spd, t0, master, out_dir, install_i, decom_i)


def _q(x, q):
    return np.round(np.round(x / q) * q, 6)


def _emit_panel(cfg, topo, faults, faults_out, episodes, storms, chan, shared,
                eng_log, benign_log, ticket_records, ts, ts_h, day_frac, weekend,
                n, sph, spd, t0, master, out_dir, install_i, decom_i):
    """Phase D: apply repair, emit observables, apply missingness, write everything."""
    # ---- degradation with repair ------------------------------------------------------
    deg = {o: np.zeros(n, dtype=np.float64) for o in topo.ont_id}
    laser_lead = {o: np.zeros(n, dtype=np.float64) for o in topo.ont_id}
    laser_bias = {o: np.zeros(n, dtype=np.float64) for o in topo.ont_id}
    laser_txdrop = {o: np.zeros(n, dtype=np.float64) for o in topo.ont_id}
    for ep in episodes:
        o, s = ep["ont_id"], ep["start"]
        curve = ep["curve"].copy()
        rep = ep.get("repair_i")
        rel_rep = None if rep is None else rep - s
        if rel_rep is not None and 0 <= rel_rep < len(curve):
            curve = apply_repair_to_fault_curve(curve, 0, rel_rep, ep["recovery"],
                                                ep["scar_db"], sph)
        if ep["channel"] == "laser":
            lead = max(1.0, ep["onset_lead_h"] * sph)
            lam = np.clip((np.arange(len(curve)) + (s - ep["onset_i"])) / lead, 0, 1)
            if rel_rep is not None and 0 <= rel_rep < len(curve):
                lam[rel_rep:] = 0.0
            laser_lead[o][s:] = np.maximum(laser_lead[o][s:], lam)
            laser_bias[o][s:] += ep["laser_gain_ma"] * lam ** 2
            laser_txdrop[o][s:] += ep["laser_tx_drop_db"] * lam ** 2
        else:
            deg[o][s:] += curve

    ep_by_ont = {}
    for ep in episodes:
        ep_by_ont.setdefault(ep["ont_id"], []).append(ep)

    # ---- collector outages, now at a COLLECTOR tier and storm-correlated (E10/E11) ----
    # v3.1 tied outages to the OLT, so a collector problem was indistinguishable from an
    # OLT problem and always removed exactly one OLT's worth of the fleet.
    coll_of_olt = {o: f"COLL-{i % max(cfg.n_collectors, 1) + 1:02d}"
                   for i, o in enumerate(sorted(topo.olt_id.unique()))}
    storm_windows = [(int(round(s.start_day * 24 * sph)),
                      int(round((s.start_day + s.duration_h / 24.0) * 24 * sph)))
                     for s in storms.itertuples()]
    outage = {}
    for coll in sorted(set(coll_of_olt.values())):
        mask = np.zeros(n, dtype=bool)
        k = master.poisson(cfg.collector_outage_per_collector_per_month * cfg.days / 30.0)
        for _ in range(int(k)):
            st = int(master.integers(0, n))
            ln = int(master.exponential(cfg.collector_outage_hours_mean * sph)) + 1
            mask[st:st + ln] = True
        for a, b in storm_windows:
            extra = master.poisson(max(0.0, (cfg.storm_collector_outage_multiplier - 1.0)
                                       * cfg.collector_outage_per_collector_per_month / 30.0
                                       * max(b - a, 1) / spd))
            for _ in range(int(extra)):
                st = int(master.integers(max(a, 0), max(min(b, n), max(a, 0) + 1)))
                ln = int(master.exponential(cfg.collector_outage_hours_mean * sph)) + 1
                mask[st:st + ln] = True
        outage[coll] = mask

    # ---- planned maintenance windows per PON port (E12) --------------------------------
    maint = {}
    for port in topo.pon_port.unique():
        mask = np.zeros(n, dtype=bool)
        k = master.poisson(cfg.planned_maintenance_per_pon_per_year * cfg.days / 365.0)
        for _ in range(int(k)):
            st = int(master.integers(0, n))
            ln = int(master.exponential(cfg.planned_maintenance_hours_mean * sph)) + 1
            mask[st:st + ln] = True
            eng_log.append(dict(entity_id=port, ts_i=st, event_type="planned_maintenance",
                                level_change_db=0.0, detail=f"{ln} samples"))
        maint[port] = mask

    conv_samples = int(cfg.repaired_convalescence_days * spd)
    gap_reason_log, writer, batch, schema = [], None, [], None
    topo_idx = topo.set_index("ont_id")
    hh_e = ((ts_h - cfg.thr_evening_peak_h + 12) % 24) - 12
    hh_m = ((ts_h - cfg.thr_morning_peak_h + 12) % 24) - 12
    hh_n = ((ts_h - 2.5 + 12) % 24) - 12

    for i, row in enumerate(topo.itertuples()):
        o = row.ont_id
        rng = np.random.default_rng([cfg.seed, 10_000 + i])
        temp, rxh, bias_base, tx_base, volt_base = [c.astype(np.float64) for c in chan[o]]
        d = deg[o]
        lam = laser_lead[o]
        lbias = laser_bias[o]
        ltx = laser_txdrop[o]

        rx = rxh + d + rng.normal(0, row.noise_sd_db, n)
        margin = rx - row.rx_sensitivity_dbm

        bias = bias_base + lbias + rng.normal(0, row.bias_noise_sd_ma, n)
        tx = (tx_base - ltx - 4.5 * np.clip(lam - 0.85, 0, 1) / 0.15
              + rng.normal(0, 0.075 * row.tx_noise_scale, n))

        prof = THROUGHPUT_PROFILES[row.thr_profile]
        shape_d = (prof["floor"]
                   + prof["morning"] * np.exp(-hh_m ** 2 / (2 * cfg.thr_morning_sd_h ** 2))
                   + prof["evening"] * np.exp(-hh_e ** 2 / (2 * cfg.thr_evening_sd_h ** 2))
                   + prof["night"] * np.exp(-hh_n ** 2 / (2 * 2.0 ** 2)))
        wk = np.where(weekend, cfg.thr_weekend_factor, 1.0)
        phi_t = float(np.exp(-(cfg.sample_minutes / 60.0) / cfg.thr_ar_tau_h))
        ar = lfilter([1.0], [1.0, -phi_t], rng.normal(0, cfg.thr_ar_sd * np.sqrt(1 - phi_t ** 2), n))
        avail = np.clip(cfg.thr_impair_floor + (1 - cfg.thr_impair_floor)
                        * np.clip(margin / cfg.impact_margin_db, 0, 1), cfg.thr_impair_floor, 1.0)
        avail = avail * (1 - cfg.thr_laser_impair * np.clip(lam - 0.6, 0, 0.4) / 0.4)
        contention = np.clip(shared["pon"][row.pon_port]["congestion"], 0, None)
        shared_demand = np.exp(shared["fleet"]["demand_shock"]) / (1.0 + contention)
        throughput = row.thr_base_mbps * shape_d * wk * np.exp(ar) * shared_demand * avail
        load = np.clip(throughput / max(row.thr_base_mbps, 1e-6), 0, 3.0)

        volt = (volt_base + cfg.volt_load_coeff_v * load - 0.05 * lam
                + rng.normal(0, 0.006 * row.volt_noise_scale, n))
        temp_obs = (temp + 0.06 * (bias - bias_base)
                    + rng.normal(0, row.temp_sensor_noise_sd_c * cfg.sensor_noise_scale, n))

        ber, fec, crc = simulate_error_cascade(margin, rng, cfg, row.impl_penalty_db,
                                               row.ber_slope, row.fec_scale, temp,
                                               bool(row.gt_noisy_plant))
        burst_mask = rng.random(n) < cfg.crc_burst_prob_per_sample
        if burst_mask.any():
            sizes = np.maximum(1, np.round(rng.lognormal(cfg.crc_burst_lognorm_mu,
                                                         cfg.crc_burst_lognorm_sigma,
                                                         int(burst_mask.sum())))).astype(np.int64)
            crc = crc.copy()
            crc[burst_mask] += sizes

        reboot_p = 0.0006 + 0.05 * lam ** 2 + 0.02 * (margin < 0.5)
        reboot = rng.random(n) < reboot_p
        # v4 (E5/E10): benign CPE power cycles look exactly like a fault-driven reboot
        cpe_cycle = rng.random(n) < (cfg.p_cpe_power_cycle_per_day / spd)
        reboot = reboot | cpe_cycle
        uptime = np.zeros(n)
        acc = int(rng.integers(0, 200_000))
        for k in range(n):
            acc = 0 if reboot[k] else acc + cfg.sample_minutes * 60
            uptime[k] = acc
        reboot_count = np.cumsum(reboot)

        # ---- benign anomaly layer applied to the OBSERVABLES (E5) ---------------------
        ent_benign = []
        gl = np.flatnonzero(rng.random(n) < cfg.p_sensor_glitch)
        for k in gl:
            which = rng.integers(0, 3)
            if which == 0:
                rx[k] += float(rng.normal(0, 3.0))
            elif which == 1:
                bias[k] += float(rng.normal(0, 6.0))
            else:
                temp_obs[k] += float(rng.normal(0, 9.0))
            ent_benign.append((int(k), 1, "sensor_glitch"))
        for k in np.flatnonzero(rng.random(n) < cfg.p_stuck_start):
            ln = int(np.clip(rng.exponential(cfg.stuck_run_mean_samples), 2, 200))
            sl = slice(int(k), min(n, int(k) + ln))
            for arr in (rx, bias, tx, volt, temp_obs, throughput):
                arr[sl] = arr[int(k)]
            ent_benign.append((int(k), int(sl.stop - sl.start), "stuck_value"))
        for k, ln, kind in ent_benign:
            benign_log.append(dict(entity_id=o, ts_i=k, n_samples=ln, gt_benign_type=kind))

        df = pd.DataFrame({
            "timestamp_utc": ts,
            "ont_id": o,
            "rx_power_dbm": _q(rx, cfg.quant_rx_db),
            "tx_power_dbm": _q(tx, cfg.quant_tx_db),
            "temperature_c": _q(temp_obs, cfg.quant_temp_c),
            "bias_current_ma": _q(bias, cfg.quant_bias_ma),
            "voltage_v": _q(volt, cfg.quant_volt_v),
            "ber": ber,
            "fec_count": fec,
            "crc_errors": crc,
            "uptime_s": uptime,
            "reboot_count": reboot_count,
            "throughput_mbps": _q(throughput, cfg.quant_throughput_mbps),
        })
        for c in ["olt_id", "pon_port", "splitter_l1", "splitter_l2", "geo_cluster",
                  "device_model", "vendor", "enclosure", "firmware_version", "distance_m",
                  "distance_bucket", "splitter_ratio", "l2_splitter_capacity", "fibre_age_yr",
                  "expected_rx_power_dbm", "rx_sensitivity_dbm", "service_impact_weight",
                  "customer_priority_weight"]:
            df[c] = topo_idx.loc[o, c]

        df["gt_physical_temperature_c"] = np.round(temp, 3)
        df["gt_optical_degradation_db"] = np.round(d, 4)
        df["gt_laser_degradation"] = np.round(lam, 4)
        df["gt_rx_healthy_dbm"] = np.round(rxh, 3)
        df["gt_margin_db"] = np.round(margin, 3)

        state = np.array(["healthy"] * n, dtype=object)
        fid_a = np.array([pd.NA] * n, dtype=object)
        ftype_a = np.array([pd.NA] * n, dtype=object)
        onset_a = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
        impact_a = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
        repair_a = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
        lc_a = np.zeros(n, dtype=bool)
        sh_a = np.zeros(n, dtype=bool)
        active_ct = np.zeros(n, dtype=np.int16)
        ts_ns = ts.tz_convert("UTC").tz_localize(None).to_numpy()
        for ep in sorted(ep_by_ont.get(o, []), key=lambda e: e["start"]):
            s = ep["start"]
            imp, rep = ep.get("impact_i"), ep.get("repair_i")
            end = rep if rep is not None else n
            fid_a[s:end] = ep["gt_fault_id"]
            ftype_a[s:end] = ep["gt_fault_type"]
            onset_a[s:end] = ts_ns[max(ep["onset_i"], 0)]
            if imp is not None:
                impact_a[s:end] = ts_ns[imp]
            if rep is not None:
                repair_a[s:end] = ts_ns[rep]
            lc_a[s:end] = ep["left_censored"]
            sh_a[s:end] = ep["scope"] != "ont"
            active_ct[s:end] += 1
            imp_end = imp if imp is not None else end
            state[s:imp_end] = "degrading"
            if imp is not None:
                state[imp:end] = "impaired"
            if rep is not None:
                state[rep:min(rep + conv_samples, n)] = "repaired"
        df["gt_state"] = state
        df["gt_fault_id"] = fid_a
        df["gt_fault_type"] = ftype_a
        df["gt_onset_ts"] = pd.to_datetime(onset_a, utc=True)
        df["gt_impact_ts"] = pd.to_datetime(impact_a, utc=True)
        df["gt_repair_ts"] = pd.to_datetime(repair_a, utc=True)
        df["gt_left_censored"] = lc_a
        df["gt_shared_fault"] = sh_a
        df["gt_active_fault_count"] = active_ct

        # ---- missingness (E10) --------------------------------------------------------
        p_drop = 1.0 - row.poll_reliability
        m_dropout = rng.random(n) < p_drop
        m_los = rng.random(n) < np.clip(0.35 * (margin < 0.0), 0, 1)
        m_outage = outage[coll_of_olt[row.olt_id]]
        m_maint = maint[row.pon_port]
        m_reboot = reboot
        # benign dense-gap processes: a power cut or a CPE swap looks like loss of signal
        m_benign = np.zeros(n, dtype=bool)
        for _ in range(int(rng.poisson(cfg.benign_outage_rate_per_ont_year * cfg.days / 365.0))):
            st = int(rng.integers(0, n))
            ln = int(rng.exponential(cfg.benign_outage_hours_mean * sph)) + 1
            m_benign[st:st + ln] = True
        for _ in range(int(rng.poisson(cfg.holiday_absence_rate_per_ont_year * cfg.days / 365.0))):
            st = int(rng.integers(0, n))
            ln = int(rng.exponential(cfg.holiday_absence_days_mean * spd)) + 1
            m_benign[st:st + ln] = True
        m_churn = np.ones(n, dtype=bool)
        m_churn[install_i[o]:decom_i[o]] = False

        drop = m_dropout | m_los | m_outage | m_reboot | m_benign | m_maint | m_churn
        if (drop & ~m_churn).any():
            reason = np.empty(n, dtype=object)
            reason[m_dropout] = "random_dropout"
            reason[m_reboot] = "cpe_restart"
            reason[m_benign] = "premises_outage"
            reason[m_los] = "loss_of_signal"
            reason[m_maint] = "planned_maintenance"
            reason[m_outage] = "collector_outage"
            gi = np.flatnonzero(drop & ~m_churn)
            gap_reason_log.append(pd.DataFrame({"entity_id": o, "ts": ts[gi],
                                                "gt_gap_reason": reason[gi]}))
        df = df.loc[~drop].copy()

        for c in ["rx_power_dbm", "tx_power_dbm", "temperature_c", "bias_current_ma",
                  "voltage_v", "ber", "fec_count", "crc_errors", "throughput_mbps"]:
            m = rng.random(len(df)) < cfg.p_field_nan
            df.loc[m, c] = np.nan

        batch.append(df)
        if len(batch) >= cfg.batch_onts or i == len(topo) - 1:
            out = pd.concat(batch, ignore_index=True)
            for c in ["ont_id", "olt_id", "pon_port", "splitter_l1", "splitter_l2",
                      "geo_cluster", "device_model", "vendor", "enclosure", "firmware_version",
                      "distance_bucket", "splitter_ratio", "gt_state", "gt_fault_id",
                      "gt_fault_type"]:
                out[c] = out[c].astype("string")
            tbl = pa.Table.from_pandas(out, preserve_index=False)
            if writer is None:
                schema = tbl.schema
                writer = pq.ParquetWriter(out_dir / cfg.out_path, schema, compression="zstd")
            writer.write_table(tbl.cast(schema))
            batch = []
    if writer is not None:
        writer.close()

    return _write_sidecars(cfg, topo, faults_out, episodes, storms, eng_log, benign_log,
                           ticket_records, gap_reason_log, ts, n, sph, t0, master, out_dir,
                           install_i, decom_i)


def _write_sidecars(cfg, topo, faults_out, episodes, storms, eng_log, benign_log,
                    ticket_records, gap_reason_log, ts, n, sph, t0, master, out_dir,
                    install_i, decom_i):
    """Registry, tickets, intervals, events and provenance.

    v4 (E7): ticket identifiers are OPAQUE. v3.1 issued `TKT-{fault_id[2:]}-{k:02d}` for
    customer tickets and `TKT-NFF{j:04d}` for no-fault-found, so the canonical
    `service_tickets` table -- the table the Week-2 ticket-proxy evaluation and the
    Week-5 localisation check are scored against -- handed over exact fault grouping,
    exact cross-entity shared-fault grouping and exact NFF identification from a string
    prefix. Resolution timestamps are now per ticket, some tickets never resolve, and
    duplicates and mis-attributions are present.
    """
    # ---- opaque, order-shuffled ticket identifiers ------------------------------------
    n_nff = int(round(len(ticket_records) * cfg.nff_ticket_rate
                      / max(1 - cfg.nff_ticket_rate, 1e-6)))
    nff_rows = []
    marginal = topo.sort_values("extra_plant_loss_db", ascending=False).ont_id.tolist()
    for _ in range(n_nff):
        # NFF is not uniform over the fleet: it concentrates on marginal and chronic lines
        # and on customers who have recently been in trouble.
        if master.random() < 0.45 and len(marginal):
            ent = str(master.choice(marginal[: max(10, len(marginal) // 4)]))
        else:
            ent = str(master.choice(topo.ont_id.values))
        nff_rows.append(dict(ont_id=ent, reported_i=int(master.integers(0, n)),
                             repair_i=None, reported_symptom=str(master.choice(
                                 ["intermittent", "slow_service", "no_service"],
                                 p=[0.5, 0.35, 0.15])),
                             gt_fault_id=pd.NA, gt_fault_type="nff", gt_is_nff=True,
                             gt_misattributed=False))
    all_tk = ticket_records + nff_rows
    # Opaque alphanumeric references, shuffled so issue order carries nothing either.
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    order = master.permutation(len(all_tk))
    ids, used = {}, set()
    for idx in order:
        while True:
            tok = "".join(alphabet[k] for k in master.integers(0, len(alphabet), 8))
            if tok not in used:
                used.add(tok)
                break
        ids[int(idx)] = f"TKT-{tok}"

    tick = []
    for j, r in enumerate(all_tk):
        rep_i = r["repair_i"]
        if rep_i is None or master.random() < cfg.p_ticket_unresolved:
            resolved = pd.NaT
        else:
            jitter = int(master.normal(0, cfg.ticket_resolution_jitter_h * sph))
            k = int(np.clip(rep_i + jitter, r["reported_i"] + 1, n - 1))
            resolved = ts[k]
        tick.append(dict(
            ticket_id=ids[j], ont_id=r["ont_id"], reported_ts=ts[int(r["reported_i"])],
            resolved_ts=resolved, reported_symptom=r["reported_symptom"],
            gt_fault_id=r["gt_fault_id"], gt_fault_type=r["gt_fault_type"],
            gt_is_nff=r["gt_is_nff"], gt_misattributed=r["gt_misattributed"]))
    tickets = pd.DataFrame(tick).sort_values("reported_ts").reset_index(drop=True)
    tickets["resolved_ts"] = pd.to_datetime(tickets.resolved_ts, utc=True)

    # ---- fault registry -----------------------------------------------------------------
    faults_out["onset_ts"] = t0 + pd.to_timedelta(faults_out.onset_i * cfg.sample_minutes, unit="m")
    for col_i, col_ts in [("impact_i", "impact_ts"),
                          ("counterfactual_impact_i", "counterfactual_impact_ts"),
                          ("repair_i", "repair_ts"),
                          ("first_observable_i", "first_observable_ts"),
                          ("first_observable_practical_i", "first_observable_practical_ts")]:
        faults_out[col_ts] = [ts[int(v)] if pd.notna(v) else pd.NaT for v in faults_out[col_i]]
    tk_by_fault = (tickets.loc[~tickets.gt_is_nff].groupby("gt_fault_id").ticket_id
                   .apply(lambda s: sorted(s)[0]))
    faults_out["ticket_id"] = faults_out.gt_fault_id.map(tk_by_fault)

    # ---- per-entity intervals -----------------------------------------------------------
    interval_rows = []
    for ep in episodes:
        s = ep["start"]
        rep_i = ep.get("repair_i")
        end_i = min(rep_i if rep_i is not None else n - 1, n - 1, decom_i[ep["ont_id"]] - 1)
        end_i = max(end_i, s)
        imp_i = ep.get("impact_i")
        interval_rows.append(dict(
            fault_id=ep["gt_fault_id"], entity_id=ep["ont_id"],
            fault_family=FAULT_TYPES[ep["gt_fault_type"]]["family"], channel=ep["channel"],
            active_start_ts=ts[s], active_end_ts=ts[end_i],
            impact_ts=ts[imp_i] if imp_i is not None else pd.NaT,
            contribution_db=round(float(np.max(np.abs(ep["curve"]))) if len(ep["curve"]) else 0.0, 3)))
    fault_entity_intervals = pd.DataFrame(interval_rows)

    engineering_events = pd.DataFrame(eng_log) if eng_log else pd.DataFrame(
        columns=["entity_id", "ts_i", "event_type", "level_change_db", "detail"])
    if len(engineering_events):
        engineering_events["ts"] = [ts[int(k)] for k in engineering_events.ts_i]
        engineering_events = engineering_events.drop(columns=["ts_i"])
    benign_anomalies = pd.DataFrame(benign_log) if benign_log else pd.DataFrame(
        columns=["entity_id", "ts_i", "n_samples", "gt_benign_type"])
    if len(benign_anomalies):
        benign_anomalies["ts"] = [ts[int(min(k, n - 1))] for k in benign_anomalies.ts_i]
        benign_anomalies = benign_anomalies.drop(columns=["ts_i"])

    gaps_gt = (pd.concat(gap_reason_log, ignore_index=True) if gap_reason_log
               else pd.DataFrame({"entity_id": pd.Series(dtype="object"),
                                  "ts": pd.Series(dtype="datetime64[ns, UTC]"),
                                  "gt_gap_reason": pd.Series(dtype="object")}))

    registry_service = pd.DataFrame({
        "entity_id": topo.ont_id,
        "install_ts": [ts[int(k)] for k in topo.install_i],
        "decommission_ts": [ts[int(k)] if k < n else pd.NaT for k in topo.decommission_i]})

    storms_out = storms.copy()
    if len(storms_out):
        storms_out["start_ts"] = t0 + pd.to_timedelta(storms_out.start_day * 24, unit="h")
        storms_out["end_ts"] = storms_out.start_ts + pd.to_timedelta(storms_out.duration_h, unit="h")

    out_dir = Path(out_dir)
    tickets.to_csv(out_dir / "tickets.csv", index=False)
    fault_entity_intervals.to_csv(out_dir / "fault_entity_intervals.csv", index=False)
    engineering_events.to_csv(out_dir / "engineering_events.csv", index=False)
    benign_anomalies.to_csv(out_dir / "gt_benign_anomalies.csv", index=False)
    storms_out.to_csv(out_dir / "gt_fault_groups.csv", index=False)
    registry_service.to_csv(out_dir / "entity_service_windows.csv", index=False)
    gaps_gt.to_parquet(out_dir / "gt_collection_gaps.parquet", index=False)
    parameter_provenance(cfg).to_csv(out_dir / "parameter_provenance.csv", index=False)
    faults_out.drop(columns=["onset_i", "impact_i", "counterfactual_impact_i", "repair_i",
                             "first_observable_i", "first_observable_practical_i"]).to_csv(
        out_dir / "gt_fault_registry.csv", index=False)
    topo.to_csv(out_dir / "topology.csv", index=False)
    pd.Series(asdict(cfg)).to_json(out_dir / "generator_config.json", indent=2)
    return dict(topology=topo, faults=faults_out, tickets=tickets, storms=storms_out,
                fault_entity_intervals=fault_entity_intervals,
                engineering_events=engineering_events, benign_anomalies=benign_anomalies,
                gap_reasons=gaps_gt, panel_path=str(out_dir / cfg.out_path))
