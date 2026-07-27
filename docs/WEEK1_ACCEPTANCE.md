# Milestone 1 v0.3 acceptance map

- SPEC-CORE: 0.3.0
- SPEC-EVAL: 0.3.0
- Telecom Pack: 0.2.0
- Synthetic GPON adapter: 0.3.0
- OilWell Pack: 0.1.0

## Deliverables

| ID | Deliverable | Implementation evidence |
|---:|---|---|
| 335 | SPEC-CORE and SPEC-EVAL split | `core.py` and `evaluation.py`, with separate schema roots |
| 336 | Generic adapter and sector-pack interfaces | Neutral protocols and declarative pack loader |
| 337 | Telecom Pack v0.1+ | Complete metric, exposure, relation, quality, and behaviour metadata |
| 338 | Generator adapter | Bounded, batch-writing translator in `telecom.py` |
| 339 | Grouped-fault mechanism | Cause-group translation and multi-fault integrity test |
| 340 | Ground-truth isolation tests | Invariance, canary, negative-control, and runtime-mount tests |
| 341 | Legacy baseline frozen | Supplied generator frozen; unavailable detector source explicitly recorded |

## Exit criteria

| ID | Exit criterion | Primary evidence |
|---:|---|---|
| 342 | Detector runs with SPEC-EVAL removed | Notebook 03 runtime variants |
| 343 | Output is identical with and without truth | Notebook 03 translator content hashes |
| 344 | No hidden circular dependency | Automated import-graph tests |

Translator canonical-content invariance is the primary leakage proof. Runtime mount
equality is secondary deployment evidence because the Week 1 detector is a
placeholder.

## Acceptance tests implemented in Notebook 03

1. Find a native sample that definitely exercises both `clipped` and `invalid`.
2. Translate the original source with evaluation truth available.
3. Translate a redacted source with truth columns, evaluation files, and
   `tickets.csv` removed.
4. Compare canonical table content hashes—not Parquet bytes.
5. Inject a truth timestamp canary and prove no timestamp reaches SPEC-CORE.
6. Deliberately inject that canary into SPEC-CORE and prove the harness detects it.
7. Verify the materialised full/selected run contains clipped and invalid rows.
8. Verify FEC exposure is constant and CRC exposure is non-degenerate.
9. Exercise an event whose `known_at` is later than its event time.
10. Score with SPEC-EVAL mounted, renamed, removed, and empty.
11. Inspect the runtime import boundary.

## Memory gate

The 8 GiB gate uses peak RSS from a clean materialisation subprocess, which includes
native Arrow/NumPy allocations. A scoped `tracemalloc` widening probe is recorded
separately for diagnostic detail. Notebook-process `ru_maxrss` is not used as the
gate because it is a lifetime high-water mark.

## Cross-sector challenge

Notebook 04 finds the smallest Petrobras 3W subset satisfying all of:

- a normal instance;
- a transient event;
- a persistent condition;
- a state transition;
- a missing or frozen measurement;
- at least three distinct real wells.

It fails if any criterion is absent and records what could not be represented without
inventing source facts.

The authoritative run results live beside the immutable outputs as
`workflow_report.json`, `acceptance_report.json`, `memory_report.json`, and
`threew_contract_fit_report.json`.
