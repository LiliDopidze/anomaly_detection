# Run the pipeline

Open notebooks from this repository. Restart the kernel before each notebook
and choose **Run All**. Calculations live in `src/telco_anomaly`; notebooks
configure stages and display their results.

## Main workflow

| Order | Notebook | Purpose |
|---|---|---|
| 00 | `00_PROJECT_CONTRACT.ipynb` | Review dataset and evaluation policy |
| 01 | `01_DATA_QUALITY_AUDIT.ipynb` | Audit the source data |
| 02 | `02_CANONICAL_DATA_MODEL.ipynb` | Build canonical telemetry and topology |
| 03 | `03_SPLITS_AND_TRUTH_LOCK.ipynb` | Freeze chronological splits and truth |
| 04 | `04_CALIBRATION_EDA.ipynb` | Review calibration data and references |
| 05 | `05_FEATURE_ENGINEERING.ipynb` | Build causal features |
| 06 | `06_PRIMARY_UNSUPERVISED_MODEL.ipynb` | Fit detectors and calibration thresholds |
| 07 | `07_CHALLENGER_MODELS.ipynb` | Compare eligible candidates on development |
| 08 | `08_ALERTS_INCIDENTS_AND_DYING_GASP.ipynb` | Produce alerts and incidents |
| 09 | `09_TOPOLOGY_LOCALISATION.ipynb` | Estimate affected topology scopes |
| 10 | `10_LOCKED_EVALUATION.ipynb` | Evaluate development; holdout is sealed by default |
| 10A | `10A_DEVELOPMENT_DIAGNOSTICS.ipynb` | Investigate misses, latency and nuisance incidents |

Optional notebooks are not required for each run:

- `06A_CALIBRATION_SAMPLING_SENSITIVITY.ipynb`: compare calibration sampling
  policies before fitting Notebook 06; never reads fault labels.
- `11_PUBLIC_DATASET_VALIDATION.ipynb`: acquire and map a separate public source.
- `12_INFERENCE_DEMO_AND_MODEL_CARD.ipynb`: present a frozen run and its limits.

## Current modelling run

For the September update, reuse stages 00–04 if their lineage checks pass.
Run **05 → 06 → 07 → 08 → 09 → 10 (development) → 10A**.
A run already using these defaults does not need restarting for repository
cleanup: runtime source, configuration and notebook code are unchanged by it.

| Stage | Default run ID |
|---|---|
| Core | `synthetic_pon_core_v2` |
| Truth | `synthetic_pon_truth_v3` |
| Features | `synthetic_pon_features_v6` |
| Model | `synthetic_pon_models_v12` |
| Selection | `synthetic_pon_selection_v14` |
| Incidents | `synthetic_pon_incidents_v14` |
| Localisation | `synthetic_pon_localisation_v14` |
| Development evaluation | `synthetic_pon_development_v14` |

Clear old `TELCO_*_RUN_ID` and `PON_*_RUN_ID` environment overrides or explicitly
set them to the intended IDs. The paths printed by each notebook are the
resolved inputs. Completed outputs are never overwritten; use fresh run IDs
for subsequent experiments. A lineage mismatch means the inputs need fixing,
not bypassing the check.

The current model retains historical baselines across gaps of at most six
hours, subject to observed coverage. Exact lags, differences, counter totals
and alert persistence remain strict. Isolation Forest uses frozen calibration
medians and missing-input indicators, requiring at least half its selected
measurements. Old fitted bundles require the earlier code revision.

Selection ranks eligible candidates by 48-hour recall. Workload, coverage and
recall-confidence gates remain mandatory. Shared-network portfolios are
candidates, not presumed improvements. If no candidate passes, Notebook 07
writes diagnostics without a selected configuration. Do not relax gates or
open holdout to rescue a development experiment. Early warning and
localisation require their own evidence; detection qualification alone does
not establish either.

## Development diagnostics

Notebook 10A has two modes:

- `FULL_MODE=True`: verify frozen lineage and replay alerts/evaluation, then
  inspect per-fault scores, availability, incident matches and candidate results.
- `FULL_MODE=False`: summarise an extracted results directory without model data.

`DEEP_DATA=True` also scans raw quality, collection gaps and feature shift.
Use `TRACE_FAULT` for one fault, or set it to `None` and use `TRACE_CASE` for
one incident. `TRACE_LIMIT` bounds exports; check the trace manifest for
truncation. Outputs include `report.md`, CSV/Parquet tables and input hashes.

The same diagnostic is available from the repository root:

```bash
PYTHONPATH=src python -m telco_anomaly.diagnostics --help
```

These outputs identify pipeline evidence, not physical causes. Missing history
and an observed neutral measurement are different. Unmatched incidents can
include unlabelled anomalies; duplicates are counted separately. Latency
statistics among detections exclude misses, so read them beside event recall.

## Runtime and data location

Set `TELCO_DATA_ROOT` for data and final artifacts. In Colab the existing
`MyDrive/anomaly_detection` folder is recognised automatically. Large stages
use local scratch space under `/content`; do not use Drive for DuckDB spill.
Set `TELCO_WORK_ROOT` if a different local scratch directory is needed.
Notebook 06 expects at least 4 GB of free scratch space. If the runtime fills
up, restart it and reuse completed immutable feature outputs.

## Optional public-data qualification

In Notebook 11, choose one source and enable acquisition:

```python
%env PUBLIC_DATASET=ran_pm
%env DOWNLOAD_PUBLIC_DATA=1
```

Microsoft Optical also requires `ACKNOWLEDGE_MICROSOFT_DATA_TERMS=1`; the
optical-failure testbed requires `ACKNOWLEDGE_OPTICAL_FAILURE_TERMS=1` after
reviewing the applicable terms. Acquisition does not grant additional rights.

Review the mapping draft's timestamps, entities, units, cadence and topology
before setting `mapping_review_status: approved`. RAN and Microsoft optical
packs can then use stages 04–06 with their own `TELCO_DATASET`. Do not run
label-based selection on unlabelled sources or combine their raw metrics with
PON telemetry.
