# Sector-agnostic anomaly detection research workflow

The maintained workflow is deliberately small. Sector-specific logic ends in
Notebook 01A; every later notebook uses the same canonical contract.

The current deliverable is a **defensible analytical product core and client
demonstration**, not a production monitoring service. Production ingestion,
security, workflow integration and service operations remain separate work.

```text
native data
  -> 01A sector pack
  -> 01B common canonical adapter
  -> 02 calibration-only EDA
  -> 03 evaluation harness
  -> 04 frozen-residual models
  -> 05 ranked cases and optional one-time holdout
  -> 06 product demonstration
```

`SPEC-CORE` contains model-visible telemetry. `SPEC-EVAL` contains labels and
is physically separate. Detector code never reads `SPEC-EVAL`; only the
evaluation functions do.

## Run order

| Order | Notebook | Result |
|---:|---|---|
| 1 | `01A_TELECOM_PACK.ipynb` or `01A_PETROBRAS_3W_PACK.ipynb` | Sector-specific Pack |
| 2 | `01B_COMMON_CANONICAL_ADAPTER.ipynb` | Canonical `SPEC-CORE`, `SPEC-EVAL`, splits |
| 3 | `02_CANONICAL_EDA.ipynb` | Calibration EDA, readiness and modelling hand-off |
| 4 | `03_EVALUATION_HARNESS.ipynb` | Frozen matching, case and holdout policy |
| 5 | `04_SIMPLE_ANOMALY_MODELS.ipynb` | Development comparison and gated reference portfolio |
| 6 | `05_INCIDENT_RANKING_AND_HOLDOUT.ipynb` | Ranked development cases, or one sealed holdout run |
| 7 | `06_PRODUCT_DEMO.ipynb` | Read-only interactive client view |

`03A_OUTPUT_REVIEW.ipynb` is an optional read-only inspection notebook. It is
not a pipeline stage and does not create or change analytical results.

## What remains common and what changes by sector

Only Notebook 01A changes for a new sector. It maps native fields to:

- metric ID, measurement kind, unit, sampling mode and cadence;
- entity and episode identity;
- source quality rules backed by evidence;
- sector labels routed only to Pack evaluation;
- available topology and the correct split unit.

The canonical adapter, EDA, evaluation, modelling, case formation and demo are
common. A new sector must pass the same Pack validation and truth-isolation
tests before those notebooks run.

## Modelling approach

Notebook 04 no longer uses a rolling centre as the normal reference. A rolling
reference can absorb the slow degradation the detector is meant to find. The
new V1 sequence is:

1. transform each metric from its declared `measurement_kind`;
2. freeze median and robust scale on calibration data only;
3. calculate residuals against that frozen reference;
4. score rapid deviations and residual CUSUM drift;
5. calibrate the two channel thresholds independently;
6. consolidate channel alerts into cases;
7. select using the upper confidence bound on case workload.

Dispersion, PCA SPE and Isolation Forest are deferred from the primary
selection surface. They return only if a later controlled comparison shows
incremental recall or delay improvement at the same case workload.

Clipped values are not treated as ordinary measurements. They are withheld
from asset-health features and retained as data-quality indicators. Petrobras
3W additionally invalidates only three empirically verified repeated historian
sentinels; other finite extremes remain visible for review.

Case grouping uses a finite time gap and requires all entities in a multi-entity
case to share one resolvable topology group. This prevents uncontrolled A-B-C
transitive chains. Cases are ranked by anomaly evidence. Operational priority
is intentionally blank until validated criticality and impact data exist.

## Runtime and data paths

The same notebooks run in Colab or locally.

- Colab default: `/content/drive/MyDrive/anomaly_detection`
- Linux/WSL default: `~/anomaly_detection_data`
- Override: `ANOMALY_DATA_ROOT`

In Colab, keep the notebooks and the three helper files in
`MyDrive/anomaly_detection/research/milestone1/`. Locally, select the repository
virtual environment as the notebook kernel.

For Notebooks 02–06 set:

```python
SECTOR = "telecom"          # or "petrobras_3w"
```

Headless execution can set `ANOMALY_SECTOR`. Notebook 01B uses
`ADAPTER_SECTOR`. Output directories are immutable; use a new run ID when an
input or method changes.

## Durable outputs

The pipeline intentionally writes a small number of understandable artefacts:

```text
outputs/eda/v2.1.0/<sector>/<run>/
  metric_summary.csv
  temporal_summary.csv
  baseline_review.parquet
  readiness.parquet
  eda_decisions.json

outputs/evaluation/v2.1.0/<sector>/<run>/
  evaluation_policy.json
  truth_partition_audit.csv
  development/*.parquet
  holdout_sealed/*.parquet

outputs/models/v2.1.0/<sector>/<run>/
  residual_bundle.joblib
  selected_configuration.json
  calibration_thresholds.csv
  development_comparison.csv
  development_readiness.parquet
  selected_alerts.parquet
  selected_cases.parquet
  selected_case_members.parquet
  development_case_trace.parquet
  development_metrics.csv
  fault_type_results.csv
  legacy_workload_comparison.csv      # when frozen v2.0 outputs are available
  legacy_fault_type_comparison.csv    # when frozen v2.0 outputs are available

outputs/cases/v2.1.0/<sector>/<run>/
  ranked_cases.csv
  alerts.parquet
  case_members.parquet
  evaluation_metrics.csv
  fault_type_results.csv
  case_score_trace.parquet
  case_run.json
```

Development results are selection evidence, not final performance claims.
If no configuration meets the predeclared case-workload budget, Notebook 04
records `no_configuration_within_budget`. Notebook 05 may still rank those
development cases for diagnosis, but it will refuse to open holdout.
Run Notebook 05 with `RUN_HOLDOUT=1` only after the feature, detector, case and
evaluation policy is frozen. The holdout receipt then blocks reuse.

Petrobras 3W is the primary real-data validation sector. Telecom is a
synthetic methodological testbed: it is useful for controlled mechanism and
leakage tests, but it does not establish real-world fibre-fault realism.
