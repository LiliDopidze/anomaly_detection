"""Storms, fault arrivals, degradation trajectories, repair and the error cascade.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .catalogues import FAULT_TYPES, SHARED_SCOPES


# ======================================================================================
# Storms, fault arrivals, trajectories (E8, E9, E11)
# ======================================================================================


def sample_storms(cfg, topo, rng) -> pd.DataFrame:
    """Regional weather events producing GROUPED faults (E11).

    v3.1 had no grouped-cause mechanism at all: `group_id` was null on all 293 faults and
    the only shared-cause pattern in existence was a single L2 splitter fault affecting
    exactly eight entities. Incident compression, root-cause top-k and shared-fault
    recall per domain are degenerate on that fixture. A storm raises the hazard of
    storm-sensitive mechanisms across one or two GEOGRAPHIC clusters -- which cut across
    the topology tree -- and also raises the collector-outage rate, so a correlation
    engine must separate "one region in trouble" from "one PON port in trouble".

    Rates and footprint are uncalibrated (OD3) and recorded as such in the provenance.
    """
    total_days = cfg.days + cfg.pre_window_days
    k = rng.poisson(cfg.storms_per_year / 365.0 * total_days)
    geos = sorted(topo.geo_cluster.unique())
    rows = []
    for i in range(int(k)):
        n_geo = int(np.clip(rng.poisson(cfg.storm_geo_clusters_mean), 1, max(1, len(geos))))
        rows.append(dict(
            group_id=f"STORM-{i + 1:03d}",
            start_day=float(rng.uniform(-cfg.pre_window_days, cfg.days)),
            duration_h=float(np.clip(rng.exponential(cfg.storm_duration_h_mean), 2.0, 96.0)),
            geo_clusters=",".join(sorted(rng.choice(geos, size=n_geo, replace=False))),
        ))
    return pd.DataFrame(rows, columns=["group_id", "start_day", "duration_h", "geo_clusters"])


def _entity_hazard(cfg, topo):
    """Relative per-entity fault hazard (E9).

    v3.1 drew fault targets uniformly at random, so per-entity counts were exactly
    Poisson (measured variance/mean 0.925) and no static attribute predicted them
    (|r| < 0.10 for distance, fibre age, ONT age, connector loss and excess plant loss).
    The plan's static-attribute susceptibility baseline therefore scores at chance by
    construction, and the post-MVP hazard model has nothing to learn. The signal here is
    deliberately weak: unobservable gamma frailty dominates the observable covariates, so
    susceptibility is discoverable but far from determined.
    """
    lin = (cfg.beta_fibre_age * topo.fibre_age_yr.to_numpy()
           + cfg.beta_outdoor * (topo.enclosure.to_numpy() == "outdoor_cabinet")
           + cfg.beta_distance_km * (topo.distance_m.to_numpy() / 1000.0)
           + cfg.beta_excess_loss_db * topo.extra_plant_loss_db.to_numpy())
    h = topo.gt_frailty.to_numpy() * np.exp(lin - lin.mean())
    return h / h.mean()


def sample_fault_events(cfg, topo, storms, rng) -> pd.DataFrame:
    """Inhomogeneous arrivals with frailty, covariates, storms and recurrence."""
    names = list(FAULT_TYPES)
    w = np.array([FAULT_TYPES[k]["weight"] for k in names], dtype=float)
    ont_mask = np.array([FAULT_TYPES[k]["scope"] == "ont" for k in names], dtype=float)
    ont_w = w * ont_mask
    ont_w = ont_w / ont_w.sum()

    total_days = cfg.days + cfg.pre_window_days
    haz = _entity_hazard(cfg, topo)
    ont_ids = topo.ont_id.to_numpy()
    ent_geo = dict(zip(topo.ont_id, topo.geo_cluster))
    rows, fid = [], 0

    def _storm_for(day, geo):
        for s in storms.itertuples():
            if s.start_day <= day <= s.start_day + s.duration_h / 24.0 and geo in s.geo_clusters.split(","):
                return s.group_id
        return None

    def _magnitude(spec, rng):
        if spec["mag_mu"] is None:
            return 0.0
        return float(np.clip(rng.lognormal(spec["mag_mu"], spec["mag_sigma"]), 0.02, spec["mag_max"]))

    def _emit(ft, scope, target, day, group_id=None, parent=None):
        nonlocal fid
        spec = FAULT_TYPES[ft]
        fid += 1
        gain_ma, tx_drop_db = 0.0, 0.0
        if spec["channel"] == "laser":
            soft = rng.random() < spec["p_soft_mode"]
            gain_ma = float(np.clip(rng.lognormal(spec["laser_gain_log_mu"],
                                                  spec["laser_gain_log_sigma"]),
                                    0.4, spec["laser_gain_max_ma"]))
            if soft:
                gain_ma = min(gain_ma, 3.5)
                tx_drop_db = float(rng.uniform(*spec["soft_tx_drop_db"]))
        return dict(
            gt_fault_id=f"F-{fid:05d}", gt_fault_type=ft, scope=scope, target=target,
            onset_day=day, onset_lead_h=rng.uniform(*spec["onset_lead_h"]),
            magnitude_db=_magnitude(spec, rng), scar_db=rng.uniform(*spec["scar_db"]),
            shape=spec["shape"], channel=spec["channel"], recovery=spec["recovery"],
            group_id=group_id, gt_recurrence_of=parent,
            laser_gain_ma=gain_ma, laser_tx_drop_db=tx_drop_db,
        )

    # ---- entity-scope arrivals ---------------------------------------------------------
    n_ont_faults = rng.poisson(cfg.fault_rate_per_ont_year / 365.0 * total_days * cfg.n_onts)
    p_ent = haz / haz.sum()
    for _ in range(int(n_ont_faults)):
        j = int(rng.choice(len(ont_ids), p=p_ent))
        ft = str(rng.choice(names, p=ont_w))
        day = float(rng.uniform(-cfg.pre_window_days, cfg.days))
        gid = _storm_for(day, ent_geo[ont_ids[j]]) if FAULT_TYPES[ft]["storm_sensitive"] else None
        rows.append(_emit(ft, "ont", ont_ids[j], day, gid))

    # ---- storm-driven extra entity faults ----------------------------------------------
    storm_types = [k for k in names if FAULT_TYPES[k]["storm_sensitive"] and FAULT_TYPES[k]["scope"] == "ont"]
    sw = np.array([FAULT_TYPES[k]["weight"] for k in storm_types], dtype=float)
    sw = sw / sw.sum()
    for s in storms.itertuples():
        gl = s.geo_clusters.split(",")
        idx = np.flatnonzero(topo.geo_cluster.isin(gl).to_numpy())
        if not len(idx):
            continue
        base = cfg.fault_rate_per_ont_year / 365.0 / 24.0 * s.duration_h * len(idx)
        for _ in range(int(rng.poisson(base * (cfg.storm_hazard_multiplier - 1.0)))):
            j = int(rng.choice(idx, p=haz[idx] / haz[idx].sum()))
            ft = str(rng.choice(storm_types, p=sw))
            day = float(s.start_day + rng.uniform(0, s.duration_h / 24.0))
            rows.append(_emit(ft, "ont", ont_ids[j], day, s.group_id))

    # ---- shared-scope arrivals at four levels (E11) --------------------------------------
    scope_scale = dict(zip(["l2", "l1", "pon", "olt"], cfg.shared_scope_rate_scale))
    for scope, col in SHARED_SCOPES.items():
        rate = cfg.shared_fault_rate_per_node_year * scope_scale.get(scope, 1.0)
        types = [k for k in names if FAULT_TYPES[k]["scope"] == scope]
        if not types:
            continue
        tw = np.array([FAULT_TYPES[k]["weight"] for k in types], dtype=float)
        tw = tw / tw.sum()
        nodes = topo[col].unique()
        node_geo = topo.groupby(col).geo_cluster.first().to_dict()
        k_nodes = rng.poisson(rate / 365.0 * total_days * len(nodes))
        for _ in range(int(k_nodes)):
            node = str(rng.choice(nodes))
            ft = str(rng.choice(types, p=tw))
            day = float(rng.uniform(-cfg.pre_window_days, cfg.days))
            gid = _storm_for(day, node_geo[node]) if FAULT_TYPES[ft]["storm_sensitive"] else None
            rows.append(_emit(ft, scope, node, day, gid))
        # storm uplift on shared plant
        for s in storms.itertuples():
            gl = s.geo_clusters.split(",")
            cand = [nd for nd in nodes if node_geo[nd] in gl]
            if not cand:
                continue
            base = rate / 365.0 / 24.0 * s.duration_h * len(cand)
            for _ in range(int(rng.poisson(base * (cfg.storm_hazard_multiplier - 1.0)))):
                ft = str(rng.choice(types, p=tw))
                rows.append(_emit(ft, scope, str(rng.choice(cand)),
                                  float(s.start_day + rng.uniform(0, s.duration_h / 24.0)),
                                  s.group_id))

    # ---- recurrence: a repaired line is more likely to fail again (E9) -------------------
    base_rows = list(rows)
    for r in base_rows:
        if r["scope"] != "ont" or rng.random() >= cfg.p_recurrence:
            continue
        day = r["onset_day"] + float(rng.uniform(2.0, cfg.recurrence_window_days))
        if day >= cfg.days:
            continue
        rows.append(_emit(r["gt_fault_type"], "ont", r["target"], day,
                          None, r["gt_fault_id"]))

    f = pd.DataFrame(rows)
    if len(f):
        f["onset_ts"] = pd.Timestamp(cfg.start, tz="UTC") + pd.to_timedelta(f.onset_day * 24, unit="h")
    return f


def build_fault_degradation_curve(shape, n_from_onset, lead_samples, magnitude, temp_z=None,
                                  thermal_base_frac=0.60, thermal_temp_gain=0.35,
                                  rng=None, samples_per_h=4.0):
    """Unrepaired degradation trajectory in dB (negative = optical loss).

    v4 (E8) adds three trajectories v3.1 lacked. In v3.1 every mechanism rose
    monotonically to a fixed magnitude and held there until repair, so persistence was a
    free win for any detector and a "did it get better on its own" hypothesis never
    needed testing.

      intermittent      on/off duty cycle -- a loose or damp connector
      progressive       ramp, plateau, then a second acceleration
      partial_recovery  worsens, then partially self-heals and re-worsens
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    x = np.arange(n_from_onset, dtype=np.float64)
    u = np.clip(x / max(lead_samples, 1.0), 0.0, 1.0)
    if shape == "step":
        d = np.where(u >= 1.0, 1.0, u ** 3)
    elif shape == "lin_ramp":
        d = u
    elif shape == "exp_ramp":
        d = (np.exp(2.6 * u) - 1.0) / (np.exp(2.6) - 1.0)
    elif shape == "thermal":
        tz = np.asarray(temp_z) if temp_z is not None else np.zeros(n_from_onset)
        if len(tz) < n_from_onset:
            tz = np.pad(tz, (0, n_from_onset - len(tz)), mode="edge") if len(tz) else np.zeros(n_from_onset)
        tz = tz[:n_from_onset]
        d = np.clip(u * (thermal_base_frac + thermal_temp_gain * np.clip(tz, -2.0, 3.0)), 0.0, 1.8)
    elif shape == "progressive":
        # ramp to a plateau at ~45% depth, hold, then accelerate
        d = np.where(u < 0.45, u / 0.45 * 0.45,
                     np.where(u < 0.75, 0.45, 0.45 + (u - 0.75) / 0.25 * 0.55))
        d = np.clip(d, 0, 1)
    elif shape == "partial_recovery":
        d = np.clip(u * 1.35, 0, 1.35)
        heal_start = int(0.55 * max(lead_samples, 1.0))
        heal_len = int(max(1, 0.5 * max(lead_samples, 1.0)))
        if heal_start < n_from_onset:
            seg = slice(heal_start, min(n_from_onset, heal_start + heal_len))
            d[seg] = d[seg] * np.linspace(1.0, 0.35, max(0, seg.stop - seg.start))
            if seg.stop < n_from_onset:
                tail = np.clip(np.arange(n_from_onset - seg.stop) / max(lead_samples, 1.0), 0, 1)
                d[seg.stop:] = d[seg.stop - 1] + tail * (1.0 - d[seg.stop - 1])
        d = np.clip(d, 0, 1.35)
    elif shape == "intermittent":
        d = np.zeros(n_from_onset)
        env = np.clip(x / max(lead_samples, 1.0), 0.15, 1.0)     # episodes deepen over time
        on_mean = max(2.0, 0.25 * samples_per_h * 6.0)
        off_mean = max(4.0, 0.25 * samples_per_h * 30.0)
        i = 0
        while i < n_from_onset:
            off = int(np.clip(rng.exponential(off_mean), 1, n_from_onset))
            i += off
            if i >= n_from_onset:
                break
            on = int(np.clip(rng.exponential(on_mean), 1, n_from_onset - i))
            d[i:i + on] = env[i:i + on] * float(np.clip(rng.normal(0.85, 0.2), 0.2, 1.2))
            i += on
    else:
        raise ValueError(shape)
    return -magnitude * d


def apply_repair_to_fault_curve(deg, onset_i, repair_i, recovery, scar_db, samples_per_h):
    """Repair recovers the loss over a truck-roll window, leaving a residual scar.
    v3.1 (C3): the repair may never deepen the loss. Retained unchanged."""
    if repair_i is None or repair_i >= len(deg):
        return deg
    level = deg[repair_i]
    end_level = -scar_db if recovery == "full" else min(-scar_db, level * 0.35)
    end_level = max(end_level, level)
    ramp = int(max(1, 0.75 * samples_per_h))
    stop = min(len(deg), repair_i + ramp)
    deg[repair_i:stop] = np.linspace(level, end_level, stop - repair_i)
    deg[stop:] = end_level
    return deg


def simulate_error_cascade(margin_db, rng, cfg, impl_penalty_db, ber_slope, fec_scale,
                           temp_c, is_noisy_plant):
    """margin -> pre-FEC BER -> FEC corrections -> post-FEC BER -> CRC errors.

    v4 (E6). In v3.1 a single global curve mapped margin to BER with no device,
    temperature or vendor variation and Poisson counting noise only, so `log10(fec_count)`
    recovered `gt_margin_db` with r = -0.988 and residual scatter of 0.30 decades --
    margin to +/- 0.33 dB, comparable to or better than the 0.1 dB-quantised rx
    observable. The cascade was a cleaner second copy of the fault-carrying variable
    rather than an independent channel. Here each receiver carries its own implementation
    penalty and waterfall slope, temperature adds a penalty, the vendor scales the
    reported counter, and errors arrive in bursts (negative binomial), not as a Poisson
    readout of the mean.
    """
    n = len(margin_db)
    secs = cfg.sample_minutes * 60
    eff_margin = (margin_db - impl_penalty_db
                  - cfg.ber_temp_penalty_db_per_c * np.maximum(temp_c - 35.0, 0.0))
    log_ber = np.clip(-3.0 - ber_slope * eff_margin, -12.0, -1.0)
    ber_true = 10.0 ** log_ber

    n_codewords = 2.488e9 * secs / 1904.0
    fec_mu = np.clip(n_codewords * 1904.0 * ber_true * fec_scale, 0, 5e6)
    k = cfg.fec_overdispersion_k
    p = k / (k + np.maximum(fec_mu, 1e-9))
    fec = rng.negative_binomial(k, np.clip(p, 1e-9, 1 - 1e-12)).astype(np.int64)
    fec = np.minimum(fec, np.int64(5e6))

    log_post = np.clip(9.0 * log_ber + 34.4, -16.0, -1.0)
    frames = 78e6 * secs / (1500 * 8)
    crc_mu = np.clip(frames * 12000.0 * (10.0 ** log_post), 0, 5e6)
    if is_noisy_plant:
        crc_mu = crc_mu * cfg.crc_noisy_plant_multiplier
    kc = cfg.crc_overdispersion_k
    pc = kc / (kc + np.maximum(crc_mu, 1e-9))
    crc = rng.negative_binomial(kc, np.clip(pc, 1e-9, 1 - 1e-12)).astype(np.int64)
    crc = np.minimum(crc, np.int64(5e6))

    # The ONT estimates pre-FEC BER from observed corrections; below a reporting floor it
    # returns zero, so `ber` really is censored (in v3.1 it was zero on 0.48% of rows and
    # the `continuous_censored` signal class had nothing to exercise).
    ber_obs = np.where(fec > 0, ber_true * np.exp(rng.normal(0, 0.30, n)), 0.0)
    ber_obs = np.where(ber_obs < 1e-9, 0.0, ber_obs)
    return ber_obs, fec, crc
