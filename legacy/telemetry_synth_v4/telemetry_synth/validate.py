"""Fixture-specific validation.

Deliberately separate from `telemetry_contract.validate`, which holds only checks that
are true of ANY telemetry source (plan section 6.4). Everything here needs knowledge the
contract must never have: what the generator was configured with, what v3.1 measured,
and what "too easy" means for this fixture.

  contract (universal)              synth (fixture-specific, here)
  schema, dtypes, nullability       long/wide row conservation against the panel
  referential integrity             gaps reconstruct the CONFIGURED grid
  temporal ordering                 registry size against configured n_onts
  gaps vs telemetry itself          the C1-C11 and E1-E14 correction gates
  catalogue covers metrics          naive difficulty-floor probe
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .generator import FAULT_TYPES, SHARED_SCOPES


def validate_reference_data(panel_path, registry_path, topology_path, tickets_path, stage_dir=None):
    """Pass/fail table for the reference-data contract.

    Retains the v3.1 invariants and adds one check per v4 correction (E1-E14). Several
    v3.1 checks were VACUOUS on v3.1's own output -- "no L2 splitter exceeds capacity"
    and "every peer group has at least four ONTs" both passed on a fixture where every
    splitter held exactly eight -- so they are replaced by checks with content.
    """
    import pyarrow.parquet as parquet
    import json as _json_local

    stage_dir = Path(stage_dir) if stage_dir is not None else Path(panel_path).parent
    registry = pd.read_csv(registry_path, parse_dates=[
        "onset_ts", "impact_ts", "repair_ts", "counterfactual_impact_ts",
        "first_observable_ts", "first_observable_practical_ts"])
    topology = pd.read_csv(topology_path)
    tickets = pd.read_csv(tickets_path, parse_dates=["reported_ts", "resolved_ts"])
    cols = parquet.ParquetFile(panel_path).schema_arrow.names
    cfg_d = _json_local.loads((stage_dir / "generator_config.json").read_text())
    n = int(cfg_d["days"] * 24 * 60 / cfg_d["sample_minutes"])

    role = cfg_d.get("config_role", "unspecified")
    enriched = role == "benchmark_enriched"
    checks = []
    def rec(name, ok, details, enriched_only=False):
        # Two checks measure statistical richness (overdispersion, grouped causes) that
        # the field-prevalence regime is DESIGNED not to have: its per-family and
        # per-group counts are thin by construction. Enforcing them there would either
        # fail a correct fixture or push the field rates up to satisfy a gate written for
        # the development regime. They are enforced on enriched and REPORTED on field.
        if enriched_only and not enriched:
            checks.append(dict(check_name=name + " [reported, not gated on field regime]",
                               passed=True, details=str(details)))
            return
        checks.append(dict(check_name=name, passed=bool(ok), details=str(details)))

    # ---- retained structural invariants ------------------------------------------------
    gt_cols = [c for c in cols if c.startswith("gt_")]
    rec("Ground-truth columns use the gt_ prefix", len(gt_cols) >= 10, f"{len(gt_cols)} columns")
    need = {"timestamp_utc", "ont_id", "rx_power_dbm", "tx_power_dbm", "temperature_c",
            "bias_current_ma", "voltage_v", "fec_count", "crc_errors", "reboot_count",
            "splitter_l2", "throughput_mbps", "firmware_version"}
    rec("Required observable telemetry is present", not (need - set(cols)),
        f"missing={sorted(need - set(cols))}")

    dated = registry.dropna(subset=["impact_ts"]).copy()
    dated["lead_h"] = (dated.impact_ts - dated.onset_ts).dt.total_seconds() / 3600
    rec("Every realised impact occurs after its true onset", dated.lead_h.ge(0).all(),
        f"violations={int((~dated.lead_h.ge(0)).sum())}")

    shared_reg = registry.loc[registry.scope.ne("ont")]
    rec("Shared faults reference a resolvable topology node",
        bool(shared_reg.empty or shared_reg.apply(
            lambda r: r.target in set(topology[SHARED_SCOPES[r.scope]]), axis=1).all()),
        f"n_shared={len(shared_reg)}, scopes={shared_reg.scope.value_counts().to_dict()}")

    both = registry.dropna(subset=["first_observable_ts", "first_observable_practical_ts"])
    rec("Practical observability anchor never precedes the sensor anchor",
        (both.first_observable_practical_ts >= both.first_observable_ts).all(),
        f"n_with_both={len(both)}")
    av = registry.loc[registry.averted.astype(bool)]
    rec("Averted faults have no realised impact and repair precedes the counterfactual crossing",
        bool(av.empty or (av.impact_ts.isna() & av.counterfactual_impact_ts.notna()
                          & av.repair_ts.notna() & (av.repair_ts <= av.counterfactual_impact_ts)).all()),
        f"n_averted={len(av)}")
    repaired = registry.dropna(subset=["repair_ts"])
    rec("Every repaired fault records a repair_source",
        repaired.repair_source.isin(["ticket", "proactive", "natural"]).all(),
        f"n_repaired={len(repaired)}, sources={repaired.repair_source.value_counts().to_dict()}")
    rx = pd.read_parquet(panel_path, columns=["rx_power_dbm"]).rx_power_dbm.dropna()
    rec("rx_power_dbm lies on the 0.1 dB reporting grid",
        (rx / 0.1 - (rx / 0.1).round()).abs().max() < 1e-6, "grid check")
    del rx
    pairs = topology.groupby("device_model").vendor.nunique()
    rec("C1 Each device model maps to exactly one vendor", bool((pairs == 1).all()),
        f"models={len(pairs)}")
    rec("C6 Fibre age and ONT age are separate attributes",
        bool((topology.ont_age_yr <= topology.fibre_age_yr + 1e-9).all()
             and topology.fibre_age_yr.corr(topology.ont_age_yr) < 0.99),
        f"corr={topology.fibre_age_yr.corr(topology.ont_age_yr):.3f}")

    # ---- E1 fan-out has real variance ---------------------------------------------------
    fan = topology.groupby("splitter_l2").ont_id.nunique()
    caps = topology.groupby("splitter_l2").l2_splitter_capacity.first()
    rec("E1 Splitter fan-out varies and take-up is partial",
        bool(fan.std() > 1.5 and fan.nunique() >= 4 and (fan <= caps).all()),
        f"n_splitters={len(fan)}, fill min/med/max={fan.min()}/{int(fan.median())}/{fan.max()}, "
        f"sd={fan.std():.2f}, distinct sizes={fan.nunique()} (v3.1 measured sd=0.00, all 8)")

    # ---- E3 route length is coherent within a splitter ----------------------------------
    within = topology.groupby("splitter_l2").distance_m.std().median()
    icc = 1 - (within ** 2) / max(topology.distance_m.var(), 1e-9)
    rec("E3 Route length is shared down the tree",
        bool(icc > 0.75),
        f"within-L2 ICC={icc:.3f}, median within-splitter sd={within:.0f} m "
        f"(v3.1 measured ICC 0.144, spread 7,080 m)")

    # ---- E13 identifiers carry no topology ----------------------------------------------
    idx = topology.ont_id.str[-5:].astype(int)
    olt_i = topology.olt_id.str[-2:].astype(int)
    blocks = topology.groupby("splitter_l2").apply(
        lambda g: (g.ont_id.str[-5:].astype(int).max() - g.ont_id.str[-5:].astype(int).min()) == len(g) - 1)
    tol = max(0.15, 3.0 / np.sqrt(max(len(topology), 4)))
    rec("E13 Entity identifiers are not ordered by topology",
        bool(abs(np.corrcoef(idx, olt_i)[0, 1]) < tol and blocks.mean() < 0.2),
        f"corr(id, OLT)={np.corrcoef(idx, olt_i)[0, 1]:+.3f}, contiguous splitter blocks="
        f"{blocks.mean():.2f} (v3.1 measured +0.866 and 1.00)")

    # ---- E4 no observable channel is a fleet constant -------------------------------
    het = {}
    for m in ["rx_power_dbm", "tx_power_dbm", "bias_current_ma", "voltage_v", "temperature_c"]:
        s = pd.read_parquet(panel_path, columns=["ont_id", m, "gt_state"])
        s = s.loc[s.gt_state.eq("healthy")].dropna()
        het[m] = float(s.groupby("ont_id")[m].median().std())
        del s
    rec("E4 Every observable channel has device-to-device heterogeneity",
        bool(het["tx_power_dbm"] > 0.25 and het["voltage_v"] > 0.010),
        f"between-entity sd of healthy medians: {({k: round(v, 4) for k, v in het.items()})} "
        f"(v3.1 measured tx 0.0497 dB and voltage 0.0000 V)")

    # ---- E8 weak faults exist -----------------------------------------------------------
    optical = registry.loc[registry.family.ne("laser")]
    weak = float((optical.magnitude_db < 0.5).mean()) if len(optical) else 0.0
    invisible = float(registry.first_observable_ts.isna().mean())
    rec("E8 A material share of faults is weak, and some never become observable",
        bool(weak > 0.06 and invisible > 0.02),
        f"optical faults below 0.5 dB={weak:.1%}, faults with no sensor anchor={invisible:.1%} "
        f"(v3.1 measured 0.0% and 0.3%)")

    # ---- E9 overdispersion and susceptibility -------------------------------------------
    fei_p = stage_dir / "fault_entity_intervals.csv"
    if fei_p.exists():
        fei = pd.read_csv(fei_p)
        cnt = topology.ont_id.map(fei.groupby("entity_id").fault_id.nunique()).fillna(0)
        # A point threshold on variance/mean is the wrong instrument: at 400 entities
        # with roughly one fault each, its sampling spread is wide enough that a correct
        # fixture fails by chance -- `tight_margins` measured 1.104 against a 1.15
        # threshold while running the identical arrival process. Test it instead.
        # Under Poisson, T = sum((x - xbar)^2) / xbar ~ chi2(n - 1).
        from scipy import stats as _st
        vm = float(cnt.var() / max(cnt.mean(), 1e-9))
        T = float(((cnt - cnt.mean()) ** 2).sum() / max(cnt.mean(), 1e-9))
        dof = int(len(cnt) - 1)
        p_over = float(_st.chi2.sf(T, dof))
        # Deliberately lenient at alpha=0.10: at this fleet size the test guards against
        # Poisson-EXACTNESS (v3.1 measured 0.925, p=0.86) rather than measuring the
        # frailty precisely. Tightening it needs a larger fleet, not a smaller alpha.
        rec("E9 Per-entity fault counts are overdispersed relative to Poisson",
            bool(p_over < 0.10),
            f"variance/mean={vm:.3f}, dispersion test chi2({dof})={T:.0f}, p={p_over:.3f} "
            f"(v3.1 measured 0.925, p=0.86 -- indistinguishable from Poisson)",
            enriched_only=True)
    else:
        rec("E9 Per-entity fault counts are overdispersed relative to Poisson", False, "intervals missing")

    # ---- E11 shared faults at several levels, and grouped causes exist -------------------
    lvl = registry.loc[registry.scope.ne("ont"), "scope"].nunique()
    grouped = float(registry.group_id.notna().mean())
    multi = registry.dropna(subset=["group_id"]).groupby("group_id").target.nunique()
    rec("E11 Shared faults attach at several topology levels and groups are populated",
        bool(lvl >= 3 and grouped >= 0.08 and int((multi >= 3).sum()) >= 3),
        f"shared scopes={lvl}, group_id populated={grouped:.1%}, groups spanning >=3 nodes="
        f"{int((multi >= 3).sum())} (v3.1: one level, 0% grouped)", enriched_only=True)

    # ---- E7 ticket identifiers carry no fault information --------------------------------
    real = tickets.loc[~tickets.gt_is_nff.astype(bool)]
    leak = 0
    if len(real):
        leak = int(sum(str(r.gt_fault_id).replace("F-", "") in str(r.ticket_id)
                       for r in real.itertuples()))
    nff_pref = tickets.loc[tickets.gt_is_nff.astype(bool), "ticket_id"].astype(str)
    nff_sep = bool(len(nff_pref) and not nff_pref.str.contains("NFF").any())
    res_var = real.groupby("gt_fault_id").resolved_ts.nunique()
    n_multi = int((real.groupby("gt_fault_id").size() > 1).sum())
    rec("E7 Ticket identifiers carry no fault information",
        bool(leak == 0 and nff_sep and (n_multi == 0 or (res_var > 1).any())),
        f"ids embedding a fault id={leak}, NFF distinguishable by id={not nff_sep}, "
        f"faults whose tickets resolve at different times={int((res_var > 1).sum())} "
        f"(v3.1: all ids embedded the fault id; NFF carried an NFF prefix; 260/260 shared one resolution)")

    # ---- E10 missingness is heterogeneous and not fault-exclusive -------------------------
    gp = stage_dir / "gt_collection_gaps.parquet"
    if gp.exists():
        gg = pd.read_parquet(gp)
        sw = pd.read_csv(stage_dir / "entity_service_windows.csv", parse_dates=["install_ts", "decommission_ts"])
        per = (gg.groupby("entity_id").size().reindex(topology.ont_id).fillna(0) / n)
        spread = float(per.quantile(0.95) / max(per.quantile(0.05), 1e-9))
        faulty = set(pd.read_csv(fei_p).entity_id) if fei_p.exists() else set()
        clean_ents = [e for e in topology.ont_id if e not in faulty]
        # A dense gap must be defined relative to the CADENCE, not to a clock hour.
        # The first form of this check counted ">=3 of 4 polls lost in an hour", which is
        # unsatisfiable at hourly cadence and failed `hourly_polling` for a property the
        # fixture does not have -- a gate bug, not a fixture defect.
        # Compare Timedeltas, not raw int64. A tz-aware timestamp[us] column casts to
        # MICROSECONDS, so an assumed-nanosecond step silently matches nothing and every
        # entity reports a maximum gap run of 1.
        step = pd.Timedelta(minutes=int(cfg_d["sample_minutes"]))
        gg2 = gg.copy()
        gg2["ts"] = pd.to_datetime(gg2.ts, utc=True)
        burst_ents = set()
        for ent, grp in gg2.groupby("entity_id"):
            a = grp.ts.sort_values()
            if len(a) < 3:
                continue
            consecutive = a.diff() == step
            run = 0
            for flag in consecutive.to_numpy():
                run = run + 1 if flag else 0
                if run >= 2:            # two consecutive steps == three polls in a row
                    burst_ents.add(ent)
                    break
        clean_bursts = len(burst_ents & set(clean_ents))
        rec("E10 Missingness varies by entity and dense gaps are not fault-exclusive",
            bool(spread > 2.0 and clean_bursts >= max(1, int(0.05 * len(clean_ents)))),
            f"per-entity p95/p05 rate ratio={spread:.2f}, clean entities losing >=3 consecutive polls="
            f"{clean_bursts}/{len(clean_ents)} "
            f"(v3.1 measured 1.26 and LOS on faulty entities only)")
    else:
        rec("E10 Missingness varies by entity and dense gaps are not fault-exclusive", False, "gaps missing")

    # ---- E5 the healthy class is not perfectly clean ---------------------------------------
    ben_p = stage_dir / "gt_benign_anomalies.csv"
    ben = pd.read_csv(ben_p) if ben_p.exists() else pd.DataFrame()
    rec("E5 A benign anomaly layer exists and is labelled",
        bool(len(ben) > 0 and ben.gt_benign_type.nunique() >= 3),
        f"benign anomalies={len(ben)}, types={ben.gt_benign_type.value_counts().to_dict() if len(ben) else {}}")

    # ---- E6 the cascade is not a noiseless copy of margin ------------------------------------
    fc = pd.read_parquet(panel_path, columns=["fec_count", "gt_margin_db"]).dropna()
    fc = fc.sample(min(300_000, len(fc)), random_state=0)
    lf = np.log10(fc.fec_count + 1)
    fit = np.polyfit(fc.gt_margin_db, lf, 1)
    resid_db = float((lf - np.polyval(fit, fc.gt_margin_db)).std() / max(abs(fit[0]), 1e-9))
    rec("E6 Optical margin is not recoverable from FEC counts to better than ~1 dB",
        bool(resid_db > 0.8),
        f"margin recoverable from log10(fec) to +/-{resid_db:.2f} dB (1 sd); v3.1 measured +/-0.33 dB")
    del fc

    # ---- E12/E14 benign change and churn ------------------------------------------------------
    ep = stage_dir / "engineering_events.csv"
    eng = pd.read_csv(ep) if ep.exists() else pd.DataFrame()
    rec("E12 Benign operational change is logged and varied",
        bool(len(eng) and eng.event_type.nunique() >= 3),
        f"events={len(eng)}, types={eng.event_type.value_counts().to_dict() if len(eng) else {}}")
    sw_p = stage_dir / "entity_service_windows.csv"
    if sw_p.exists():
        sw = pd.read_csv(sw_p, parse_dates=["install_ts", "decommission_ts"])
        late = int((sw.install_ts > sw.install_ts.min()).sum())
        gone = int(sw.decommission_ts.notna().sum())
        rec("E14 Entities are provisioned and decommissioned inside the window",
            bool(late >= 3 and gone >= 3), f"late installs={late}, decommissions={gone}")
    else:
        rec("E14 Entities are provisioned and decommissioned inside the window", False, "service windows missing")

    return pd.DataFrame(checks)
