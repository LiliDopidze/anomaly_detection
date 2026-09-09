# Notebook run guide

The workflow is common after the sector Pack. Run Telecom as the primary
product path and Petrobras 3W as a real-data qualification path.

## Run order

| Order | Notebook | Purpose |
|---:|---|---|
| 0 | `00_TELECOM_LOCALISATION_AUDIT.ipynb` | Audits group sizes, live peer availability, topology equivalence and localisation denominators. Telecom only. |
| 1 | `01A_TELECOM_PACK.ipynb` | Maps Telecom telemetry, topology and isolated generator truth into the Pack interface. |
| 1 | `01A_PETROBRAS_3W_PACK.ipynb` | Maps real 3W well recordings and isolated labels. No topology is invented. |
| 2 | `01B_COMMON_CANONICAL_ADAPTER.ipynb` | Builds and validates canonical `SPEC-CORE`, `SPEC-EVAL` and `SPLITS` without sector-specific branches. |
| 3 | `02_CANONICAL_EDA.ipynb` | Calibration-only structural, quality, distribution, time-series and reference-stability assessment. |
| 4 | `03_EVALUATION_HARNESS.ipynb` | Freezes event matching, workload, uncertainty, localisation and holdout rules before modelling. |
| 5 | `04_SIMPLE_ANOMALY_MODELS.ipynb` | Fits calibration-frozen channels and selects a small predeclared portfolio on development data. |
| 6 | `05_INCIDENT_RANKING_AND_HOLDOUT.ipynb` | Produces ranked development incidents or performs one explicitly enabled sealed-holdout run. |
| 7 | `06_PRODUCT_DEMO.ipynb` | Read-only interactive summary for review and client discussion. |

`03A_OUTPUT_REVIEW.ipynb` is an optional inspection notebook. It does not alter
analytical results.

## Paths and sector switch

The notebooks use the first configured data root:

1. `ANOMALY_DATA_ROOT`;
2. `ANOMALY_DRIVE_ROOT`;
3. `/content/drive/MyDrive/anomaly_detection` in Colab;
4. `~/anomaly_detection_data` locally.

In Notebooks 02–06, change only:

```python
SECTOR = "telecom"          # or "petrobras_3w"
```

Notebook 01B uses `ADAPTER_SECTOR`. Environment variables remain available for
headless runs. Completed output directories are immutable; change the run ID
only when an input or method changes.

Large canonical builds use one DuckDB thread and a 1 GiB memory limit by
default. Gap detection processes complete episode-metric series in batches of
at most two million long-form observations. These controls can be changed with
`ANOMALY_DUCKDB_THREADS`, `ANOMALY_DUCKDB_MEMORY_LIMIT` and
`ANOMALY_GAP_BATCH_ROWS`; increasing them in a small Colab runtime can cause an
out-of-memory failure.

## What is common and what changes by sector

Each 01A notebook owns native field names, units, measurement kinds, cadence,
entity/episode identity, source-backed quality rules, genuine topology, split
construction and label translation. It creates no model features.

The common adapter and later notebooks own the versioned contract, canonical
materialisation, EDA, modelling mechanics, incident formation and evaluation.
They read capabilities rather than Telecom field names. Operational policies
such as decision horizon and workload budget are explicit by sector because
the common code must not guess them.

## Model and evaluation rules

- Calibration fits transformations, robust references and score thresholds;
  it does not use truth.
- Telecom thresholds use maxima within entity-days; group common-mode scores
  use group-days. Petrobras uses whole-recording maxima.
- Development selects among a small declared set of channel portfolios.
- Results are compared by event recall and the upper confidence bound on false
  incidents at the same operational exposure.
- Cross-entity incident consolidation requires common-mode evidence. Merely
  sharing an OLT is not enough.
- Localisation reports exact, top-two, topology-equivalence, footprint and
  different-branch evidence. It is probable observable scope, not proven root
  cause.
- Telecom is synthetic, so it validates injected mechanisms rather than
  real-fleet effectiveness. Petrobras is real but does not validate Telecom
  topology.

## Main output directories

```text
outputs/audits/v1.0.0/telecom/<run>/
outputs/packs/<sector>/<run>/
outputs/canonical/v0.11.0/<sector>/<run>/
outputs/eda/v3.0.0/<sector>/<run>/
outputs/evaluation/v3.0.0/<sector>/<run>/
outputs/models/v3.0.0/<sector>/<run>/
outputs/cases/v3.0.0/<sector>/<run>/
```

The durable artefacts are intentionally compact: manifests, audit summaries,
calibration thresholds, development comparisons, selected alerts/incidents,
event/localisation metrics and evidence traces. Large intermediate wide
matrices and residual files are temporary.

## Holdout rule

Run Notebook 05 normally with `RUN_HOLDOUT=0`. Set `RUN_HOLDOUT=1` only after
the contract, features, channels, thresholds, consolidation and evaluation
policy are frozen and the development workload gate passes. The notebook
writes `HOLDOUT_USED.json` to prevent silent reuse.

## Validation before release

From the repository root:

```bash
python -m unittest tests/test_pipeline_v3.py -v
```

These small fixtures check truth isolation, persisted-content hashes,
entity-day block maxima, peer exclusion, topology score calibration,
cross-entity consolidation and equivalence-aware localisation. Full-data runs
remain necessary to validate runtime and empirical results.
