"""Network construction: the GPON tree, plant, geography, churn and identifiers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .catalogues import DEVICE_MODELS, ENCLOSURES, FIRMWARE_BY_VENDOR, THROUGHPUT_PROFILES
from .settings import TelecomSimulationSettings


# ======================================================================================
# Topology (E1, E2, E3, E13, E14)
# ======================================================================================


def _weighted_choice(rng, mapping, size=None):
    keys = list(mapping)
    p = np.array([mapping[k]["weight"] for k in keys], dtype=float)
    return rng.choice(keys, p=p / p.sum(), size=size)


def build_network_topology(cfg: TelecomSimulationSettings, rng: np.random.Generator) -> pd.DataFrame:
    """One row per ONT: static device, topology, geography and commercial attributes.

    v4 changes against v3.1, all measured defects:

      E1  Fan-out. v3.1 assigned subscribers round-robin, so every populated L2 splitter
          held exactly `l2_split` ONTs (measured sd 0.00 across 50 splitters). Here each
          installed splitter has its own capacity and a Beta take-up fraction.
      E3  Route length. v3.1 drew `distance_m` i.i.d. per ONT, giving a median within-
          splitter spread of 7,080 m for entities that physically share a feeder
          (within-L2 ICC 0.144). Length now composes as feeder + distribution + drop.
      E13 Identifiers. v3.1 numbered ONTs in nested loop order, so the identifier was a
          monotone function of topological position (r = 0.866 against OLT index) and an
          entity-disjoint split taken in identifier order was a topology split. The
          numbering is now a permutation.
      E14 Churn. A minority of entities are provisioned after t0 or churn away before
          the window ends, so the cold-entity protocol has genuinely unseen entities.
    """
    # ---- the physical tree ----------------------------------------------------------
    slots = []
    for o in range(1, cfg.n_olts + 1):
        for p in range(1, cfg.pon_ports_per_olt + 1):
            for a in range(1, cfg.l1_split + 1):
                slots.append((f"OLT-{o:02d}", f"PON-{o:02d}-{p:02d}",
                              f"SPL1-{o:02d}-{p:02d}", f"SPL2-{o:02d}-{p:02d}-{a:02d}"))

    # ---- geography: cluster centres, then nodes placed near their centre -------------
    # Geography is CORRELATED with, but not determined by, the OLT: an operator's
    # serving areas overlap at their edges. v3.1 assigned one letter per PON port, so
    # `geo_cluster` was a deterministic coarsening of `pon_port` and carried no
    # information a topology-aware model did not already have.
    n_geo = cfg.n_geo_clusters
    geo_names = [f"GEO-{i:02d}" for i in range(1, n_geo + 1)]
    geo_centre = {g: (float(rng.uniform(51.0, 55.5)), float(rng.uniform(-4.5, 0.5)))
                  for g in geo_names}
    olt_geo_prior = {}
    for o in sorted({s[0] for s in slots}):
        alpha = np.full(n_geo, 0.25)
        alpha[rng.integers(0, n_geo)] += 3.0          # a dominant serving area
        alpha[rng.integers(0, n_geo)] += 1.5          # and a secondary one
        olt_geo_prior[o] = rng.dirichlet(alpha)

    l1_pos = {}
    for olt, port, l1, _l2 in slots:
        if l1 in l1_pos:
            continue
        g = geo_names[int(rng.choice(n_geo, p=olt_geo_prior[olt]))]
        lat, lon = geo_centre[g]
        l1_pos[l1] = (lat + float(rng.normal(0, 0.13)), lon + float(rng.normal(0, 0.18)))

    # The L2 cabinet sits a short distance from its primary splitter, and its geographic
    # cluster follows POSITION, not the tree. v3.1 assigned one letter per PON port, so
    # `geo_cluster` was a deterministic function of `pon_port` -- it could not disagree
    # with the topology, and a storm footprint was therefore indistinguishable from a
    # PON-port fault. Here a cabinet near a serving-area boundary lands in the
    # neighbouring cluster, so geography and topology cross.
    centres = np.array([geo_centre[g] for g in geo_names])
    l2_geo, l2_pos = {}, {}
    for _o, _p, l1, l2 in slots:
        lat, lon = l1_pos[l1]
        pos = (lat + float(rng.normal(0, 0.05)), lon + float(rng.normal(0, 0.07)))
        d = np.hypot(centres[:, 0] - pos[0], centres[:, 1] - pos[1])
        l2_geo[l2] = geo_names[int(np.argmin(d))]
        l2_pos[l2] = pos

    # ---- route-length components, shared down the tree (E3) -------------------------
    feeder_km = {port: float(np.clip(rng.gamma(cfg.feeder_km_shape, cfg.feeder_km_scale),
                                     0.2, 14.0))
                 for _o, port, _l1, _l2 in slots}
    distribution_km = {l1: float(np.clip(rng.gamma(cfg.distribution_km_shape,
                                                   cfg.distribution_km_scale), 0.02, 3.5))
                       for _o, _p, l1, _l2 in slots}

    # ---- which splitters are installed, and how full (E1) ---------------------------
    # Deployment is contiguous: an operator brings a PON port into service and populates
    # its cabinets, rather than scattering single splitters across the whole estate.
    # Ports are taken round-robin across OLTs so no OLT is left with a handful of lines.
    by_olt = {}
    for _o, port, l1, l2 in slots:
        by_olt.setdefault(_o, {}).setdefault(port, []).append(l2)
    port_queue = []
    olt_ports = {o: list(rng.permutation(list(d))) for o, d in by_olt.items()}
    while any(olt_ports.values()):
        for o in sorted(olt_ports):
            if olt_ports[o]:
                port_queue.append(olt_ports[o].pop())
    slot_by_l2 = {s[3]: s for s in slots}

    caps = np.array(cfg.l2_capacity_choices)
    cap_w = np.array(cfg.l2_capacity_weights, dtype=float)
    cap_w = cap_w / cap_w.sum()
    chosen, remaining = [], cfg.n_onts
    for port in port_queue:
        if remaining <= 0:
            break
        olt = port.split("-")[1]
        l2s = by_olt[f"OLT-{olt}"][port]
        n_install = int(rng.integers(max(1, len(l2s) - 2), len(l2s) + 1))
        for l2 in rng.permutation(l2s)[:n_install]:
            if remaining <= 0:
                break
            cap = int(rng.choice(caps, p=cap_w))
            take = int(np.clip(round(cap * rng.beta(cfg.l2_takeup_beta_a, cfg.l2_takeup_beta_b)),
                               cfg.l2_min_fill, cap))
            take = min(take, remaining)
            chosen.append((slot_by_l2[l2], cap, take))
            remaining -= take
    if remaining > 0:
        raise ValueError(f"topology cannot host {cfg.n_onts} ONTs; {remaining} unplaced")

    # ---- per-ONT attributes ----------------------------------------------------------
    rows = []
    for (olt, port, l1, l2), cap, take in chosen:
        for _ in range(take):
            model = str(_weighted_choice(rng, DEVICE_MODELS))
            spec_m = DEVICE_MODELS[model]
            enclosure = str(_weighted_choice(rng, ENCLOSURES))
            spec_e = ENCLOSURES[enclosure]
            drop_m = float(np.clip(rng.gamma(cfg.drop_m_shape, cfg.drop_m_scale), 8.0, 400.0))
            dist_m = (feeder_km[port] + distribution_km[l1]) * 1000.0 + drop_m
            sens = spec_m["rx_sensitivity_dbm"]
            l2_loss = {8: 10.5, 16: 13.8, 32: 17.0}[cap]

            n_splices = int(rng.poisson(cfg.splices_per_km * dist_m / 1000.0)) + 1
            splice_loss = float(np.clip(rng.normal(cfg.splice_loss_mean_db,
                                                   cfg.splice_loss_sd_db, n_splices),
                                        0.02, 0.30).sum())
            extra_conn_loss = float(np.clip(
                rng.normal(cfg.extra_connector_loss_mean_db, cfg.extra_connector_loss_sd_db,
                           cfg.n_extra_connectors), 0.10, 1.20).sum())
            as_built_excess = float(rng.gamma(cfg.as_built_excess_shape, cfg.as_built_excess_scale))
            extra_plant = float(np.clip(splice_loss + extra_conn_loss + as_built_excess,
                                        cfg.extra_plant_loss_min_db, cfg.extra_plant_loss_max_db))
            conn_loss_i = float(np.clip(rng.normal(cfg.connector_loss_mean_db,
                                                   cfg.connector_loss_sd_db), 0.3, 3.5))
            margin_wo_extra = (cfg.olt_launch_dbm
                               - (cfg.l1_loss_db + l2_loss
                                  + dist_m / 1000.0 * cfg.fibre_loss_db_per_km + conn_loss_i)
                               - sens)
            extra_plant = float(min(extra_plant,
                                    max(margin_wo_extra - cfg.turnup_min_margin_db,
                                        cfg.extra_plant_loss_min_db)))

            fibre_age = float(np.round(rng.uniform(1, 15), 1))
            ont_age = float(np.round(np.clip(rng.gamma(2.2, 1.7), 0.1, min(fibre_age, 12.0)), 1))
            vendor = spec_m["vendor"]
            lat, lon = l2_pos[l2]
            profile = str(rng.choice(list(THROUGHPUT_PROFILES), p=np.array(cfg.thr_profile_weights)))

            rows.append(dict(
                olt_id=olt, pon_port=port, splitter_l1=l1, splitter_l2=l2,
                l2_splitter_capacity=cap, geo_cluster=l2_geo[l2],
                lat=round(lat + float(rng.normal(0, 0.012)), 5),
                lon=round(lon + float(rng.normal(0, 0.016)), 5),
                device_model=model, vendor=vendor, enclosure=enclosure,
                feeder_km=round(feeder_km[port], 3),
                distribution_km=round(distribution_km[l1], 3),
                drop_m=round(drop_m, 1),
                distance_m=round(dist_m),
                fibre_age_yr=fibre_age, ont_age_yr=ont_age,
                rx_sensitivity_dbm=sens,
                l2_loss_db=l2_loss,
                bias_nominal_ma=float(spec_m["bias_nominal_ma"] + rng.normal(0, 0.8)),
                bias_thermal_coeff=spec_m["bias_thermal_coeff"],
                bias_age_ma_per_yr=spec_m["bias_age_ma_per_yr"],
                # v4 (E4): the device channels that v3.1 emitted as fleet constants
                tx_nominal_dbm=float(spec_m["tx_nominal_dbm"] + rng.normal(0, cfg.tx_device_sd_db)),
                volt_nominal_v=float(spec_m["volt_nominal_v"] + rng.normal(0, cfg.volt_device_sd_v)),
                # v4 (E6): per-device receiver implementation penalty and waterfall slope
                impl_penalty_db=float(spec_m["impl_penalty_db"]
                                      + rng.normal(0, cfg.ber_implementation_penalty_sd_db)),
                ber_slope=float(np.clip(rng.normal(1.0, cfg.ber_slope_sd), 0.7, 1.35)),
                fec_scale=float(spec_m["fec_scale"] * np.exp(rng.normal(0, 0.25))),
                temp_sensor_noise_sd_c=spec_e["sensor_noise_sd_c"],
                connector_loss_db=conn_loss_i,
                extra_plant_loss_db=round(extra_plant, 3),
                noise_sd_db=float(np.clip(rng.gamma(4.0, 0.028), 0.03, 0.35) * cfg.sensor_noise_scale),
                bias_noise_sd_ma=float(np.clip(rng.gamma(cfg.bias_device_noise_shape,
                                                         cfg.bias_device_noise_scale), 0.08, 0.7)),
                tx_noise_scale=float(np.clip(rng.gamma(2.2, 0.45), 0.25, 3.0)),
                volt_noise_scale=float(np.clip(rng.gamma(2.2, 0.45), 0.25, 3.0)),
                solar_factor=float(np.round(rng.uniform(cfg.solar_exposure_low,
                                                        cfg.solar_exposure_high), 3)),
                service_impact_weight=float(np.round(rng.choice([0.5, 0.8, 1.0, 2.0, 3.0],
                                                                p=[.25, .30, .25, .15, .05]), 2)),
                customer_priority_weight=float(np.round(rng.choice([1.0, 1.5, 2.0, 4.0],
                                                                   p=[.60, .22, .13, .05]), 2)),
                temp_phase_shift_h=float(rng.normal(0, spec_e["phase_sd_h"])),
                enclosure_offset_c=spec_e["offset_c"],
                enclosure_diurnal_factor=spec_e["diurnal_factor"],
                enclosure_seasonal_factor=spec_e["seasonal_factor"],
                microclimate_sd_c=spec_e["micro_sd_c"],
                thr_base_mbps=float(np.round(np.clip(rng.lognormal(cfg.thr_base_log_mean,
                                                                   cfg.thr_base_log_sd),
                                                     cfg.thr_base_min_mbps,
                                                     cfg.thr_base_max_mbps), 1)),
                thr_profile=profile,
                firmware_version=str(rng.choice(FIRMWARE_BY_VENDOR[vendor][:2], p=[0.62, 0.38])),
                # v4 (E10): "badly polled entity" is a real nuisance class
                poll_reliability=float(rng.beta(cfg.poll_reliability_beta_a,
                                                cfg.poll_reliability_beta_b)),
                # v4 (E9): unobservable frailty driving the fault hazard
                gt_frailty=float(rng.gamma(cfg.frailty_shape, 1.0 / cfg.frailty_shape)),
            ))

    t = pd.DataFrame(rows)

    # ---- E13: identifiers carry no topological information ---------------------------
    perm = rng.permutation(len(t))
    t["ont_id"] = [f"ONT-{i + 1:05d}" for i in perm]

    # ---- explicit chronic cohort (E5) -------------------------------------------------
    chronic = rng.random(len(t)) < cfg.chronic_share
    t["gt_chronic"] = chronic
    t.loc[chronic, "extra_plant_loss_db"] = (t.loc[chronic, "extra_plant_loss_db"]
                                             + cfg.chronic_extra_loss_db)
    t.loc[chronic, "noise_sd_db"] = t.loc[chronic, "noise_sd_db"] * cfg.chronic_noise_multiplier
    t["gt_noisy_plant"] = rng.random(len(t)) < cfg.crc_noisy_plant_share

    # ---- churn (E14) -------------------------------------------------------------------
    n = int(cfg.days * 24 * 60 / cfg.sample_minutes)
    t["install_i"] = 0
    late = rng.random(len(t)) < cfg.p_late_install
    t.loc[late, "install_i"] = rng.integers(int(0.08 * n), int(0.75 * n), int(late.sum()))
    t["decommission_i"] = n
    gone = (rng.random(len(t)) < cfg.p_decommission) & (~late)
    t.loc[gone, "decommission_i"] = rng.integers(int(0.30 * n), n, int(gone.sum()))

    t["distance_bucket"] = pd.cut(t.distance_m, [0, 2500, 6000, 1e9],
                                  labels=["near", "mid", "far"]).astype(str)
    t["splitter_ratio"] = "1:" + t.l2_splitter_capacity.astype(str)

    # Vendor-published expected RX from plant records: right on average, wrong on any
    # individual line, because as-built connector, splice and excess loss is not recorded.
    mean_extra = ((cfg.splices_per_km * t.distance_m / 1000.0 + 1.0) * cfg.splice_loss_mean_db
                  + cfg.n_extra_connectors * cfg.extra_connector_loss_mean_db
                  + cfg.as_built_excess_shape * cfg.as_built_excess_scale)
    span_loss = (cfg.l1_loss_db + t.l2_loss_db + t.distance_m / 1000.0 * cfg.fibre_loss_db_per_km
                 + cfg.connector_loss_mean_db + mean_extra)
    t["expected_rx_power_dbm"] = np.round(cfg.olt_launch_dbm - span_loss
                                          + rng.normal(0, 0.9, len(t)), 2)
    return t.reset_index(drop=True)
