# Milestone 1 acceptance record

Date: 24 July 2026

- SPEC-CORE: 0.3.0
- SPEC-EVAL: 0.3.0
- Telecom Pack: 0.1.0
- Synthetic GPON adapter: 0.3.0
- OilWell Pack: 0.1.0

## Deliverables

| ID | Deliverable | Status | Evidence |
|---:|---|---|---|
| 335 | SPEC-CORE and SPEC-EVAL split | Complete | Separate packages and schema roots |
| 336 | Generic adapter and sector-pack interfaces | Complete | `telemetry_contract/adapters.py` and `models.py` |
| 337 | Telecom Pack v0.1 | Complete | Full metric and relation semantics under `telemetry_packs/telecom` |
| 338 | Generator adapter | Complete | Bounded translator in `telemetry_adapters/synthetic_gpon.py` |
| 339 | Grouped-fault mechanism | Complete | Cause-group mapping and multi-fault integrity test |
| 340 | Ground-truth isolation tests | Complete | Negative control, translator invariance and runtime mount variants |
| 341 | Existing pipeline frozen as a legacy baseline | Qualified | Generator release is hash-described; executable legacy detector source was not supplied |

## Exit criteria

| ID | Exit criterion | Result |
|---:|---|---|
| 342 | Detector runs with SPEC-EVAL removed | Pass |
| 343 | Outputs are identical with and without truth mounted | Pass |
| 344 | Contract and pack contain no hidden circular dependency | Pass |

Translator invariance is the primary leakage proof. Runtime mount equality is
secondary deployment evidence because the Week 1 detector is intentionally a
placeholder.

## Material contract decisions

- `entity_service_windows.csv` remains a native input; canonical validity is stored
  once in `entity_registry.valid_from` and `valid_to`.
- Null-valued telemetry rows are retained with `quality_code = invalid`.
- FEC values at the generator ceiling use `quality_code = clipped`.
- The anomaly-direction vocabulary is `decrease`, `increase`, `both`, `change`.
- Telecom relations exercise both network topology and geographic membership.
- `tickets.csv` is explicitly evaluation-only.
- Petrobras 3W adds `gt_condition_states` without `severity_ordinal`.
- No cross-well topology, shared manifold, cause group or ticket is invented for 3W.

## Full-panel verification

- Native telecom rows: 6,288,215.
- Canonical telemetry rows: 69,170,365.
- Collection-gap intervals: 169,501.
- Clipped rows: 1,626,252.
- Invalid rows: 226,384.
- Process high-water memory: 3.72 GiB against an 8 GiB budget.
- Traced 250,000-row widening probe: 0.97 GiB.

## Cross-sector challenge

The smallest deterministic 3W subset satisfying all criteria used three distinct real
wells and produced:

- 8,034,930 canonical telemetry observations;
- two fault-event intervals;
- ten condition-state intervals;
- an intentionally empty relation table.

The contract-fit report records what could not be represented without invention.
