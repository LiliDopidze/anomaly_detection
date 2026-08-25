# Sector-agnostic anomaly detection research workflow

This folder keeps the research pipeline deliberately small. Sector-specific
code ends at the pack notebooks; every notebook after that runs unchanged for
Telecom and Petrobras 3W.

```text
native data
  -> 01A sector pack
  -> 01B common canonical adapter
  -> 02 canonical EDA
  -> 03 evaluation contract
  -> 04 simple anomaly models
  -> 05 incident ranking and sealed holdout
```

The detector and all modelling code read `SPEC-CORE` only. Labels remain in
the physically separate `SPEC-EVAL` directory and are read only by evaluation.

## Maintained notebooks

| Order | Notebook | Responsibility |
|---:|---|---|
| 1 | `01A_TELECOM_PACK.ipynb` or `01A_PETROBRAS_3W_PACK.ipynb` | Translate one native sector into the pack interface. |
| 2 | `01B_COMMON_CANONICAL_ADAPTER.ipynb` | Validate and materialise canonical `SPEC-CORE`, `SPEC-EVAL` and splits. |
| 3 | `02_CANONICAL_EDA.ipynb` | Describe calibration data only and record modelling decisions. |
| 4 | `03_EVALUATION_CONTRACT.ipynb` | Freeze alert matching, metrics, false-alert exposure and sealed truth. |
| 5 | `04_SIMPLE_ANOMALY_MODELS.ipynb` | Compare four understandable unsupervised baselines on development truth. |
| 6 | `05_INCIDENT_RANKING_AND_HOLDOUT.ipynb` | Freeze the selected model, form incidents, rank them and optionally score holdout once. |

The earlier `03_EVALUATION_HARNESS.ipynb`, `03A_OUTPUT_REVIEW.ipynb`,
`04_GENERIC_BASELINE_MODELS.ipynb` and `04A_CLIENT_PROGRESS_REVIEW.ipynb` are
retained only as experimental history. Do not include them in the final run.

## Shared files

- `milestone1_core.py` contains the already-frozen canonical contract and
  adapter mechanics.
- `evaluation_core.py` contains the settled alert matching and metric
  calculations used by Notebooks 03–05.
- `simple_model_core.py` contains the repeated, settled feature and scoring
  mechanics used by Notebooks 04–05.

These are flat helper files, not a package or application. Research decisions
remain visible in the notebooks.

## Runtime and data location

The same notebooks run in Colab or locally.

- Colab default: `/content/drive/MyDrive/anomaly_detection`
- Linux/WSL default: `~/anomaly_detection_data`
- Override either with `ANOMALY_DATA_ROOT`.

In VS Code, select the repository `.venv` Python kernel. In Colab, upload this
folder to `<data_root>/research/milestone1/`. Each notebook prints its resolved
data, input and output paths before doing work.

Set one sector before running:

```python
SECTOR = "telecom"          # or "petrobras_3w"
```

Notebooks 02–05 also accept `ANOMALY_SECTOR` for headless runs. Notebook 01B
uses `ADAPTER_SECTOR` instead; setting only `ANOMALY_SECTOR` does not control
the canonical adapter. Output folders are immutable. Change the relevant run
ID before intentionally rebuilding an existing result.

## What each final stage does

### 02 — Canonical EDA

EDA reads calibration telemetry only. This prevents development and holdout
behaviour from influencing feature or model choices. It checks:

- entities, episodes, metrics, time span and duplicate keys;
- invalid/clipped rates, including variation between entity-series;
- robust distributions and representative time-series plots;
- autocorrelation and a limited daily-seasonality check where enough cycles exist;
- level and differenced cross-metric dependence.

Its only durable outputs are:

```text
outputs/eda/v1.3.0/<sector>/<eda_run_id>/
├── eda_summary.csv
└── eda_decisions.json
```

`eda_decisions.json` records short and long cadence-based rolling windows plus
one persistence and recovery duration per sector. The long window prevents a
sustained shift from immediately becoming its own reference. Clipped values are
retained as observed saturated measurements, while their rate is reported
separately. Seasonality is described but not automatically removed: it must be
stable and beneficial on development data before becoming a transformation.

### 03 — Evaluation contract

This notebook is not a model. It freezes how model output will be judged before
model selection begins. It defines:

- the alert-to-fault decision horizon;
- duplicate-alert treatment;
- event recall, pre-impact recall, precision, delay and false-alert exposure;
- development and sealed-holdout truth directories.

Nine controls verify perfect alerts, constant-low and constant-high scores,
recovery behaviour, late and duplicate alerts, malformed alerts, a deliberately
leaky reader and the absence of point-adjusted evaluation. The leaky reader must
fail when truth is unmounted.

```text
outputs/evaluation/v1.3.0/<sector>/<evaluation_run_id>/
├── evaluation_policy.json
├── development/{fault_events,fault_entity_intervals}.parquet
└── holdout_sealed/{fault_events,fault_entity_intervals}.parquet
```

### 04 — Simple anomaly models

All models use the same causal, measurement-aware feature table. Rolling
history is measured in elapsed time and converted using each metric's declared
cadence, so the same code remains valid when a future source mixes cadences:

- gauges: long-run-scaled change plus short- and long-window robust deviations;
- interval counts: `log1p`, then the same causal change and deviation features;
- cumulative counters: non-negative increment plus reset flag;
- discrete states: current state plus change flag.

Raw gauge and count levels are deliberately excluded: legitimate asset-level
offsets should not become anomalies merely because a new well or device has a
different operating level. Bounded fractions and discrete state codes remain
directly comparable. The first multivariate baseline requires aligned metric
cadence and fails clearly instead of silently resampling a mixed-cadence source.

Only calibration data fits imputers, scaling, models and operating thresholds.
Development labels compare exactly four baselines:

1. robust statistical score — maximum `log1p` absolute robust feature score;
2. Isolation Forest — multivariate nonlinear baseline;
3. PCA reconstruction error (SPE) — detects broken correlation structure;
4. PCA Hotelling T-squared — detects movement along retained PCA directions.

PCA retains approximately 90% of calibration variance while leaving at least
one direction to reconstruct. A calibration-time degeneracy check stops a
collapsed SPE baseline from being reported as a working detector.

Each model uses three calibration thresholds and one EDA-declared persistence
setting: exactly 12 operating candidates. A threshold is the median of
within-block calibration quantiles (entity blocks for Telecom, episode blocks
for 3W), so one long or contaminated block cannot dominate it. This controls a
point-level exceedance quantile, not the operational alert rate; the latter is
measured empirically.

VUS-PR over development thresholds and decision tolerances is the primary
threshold-free model comparison. The 12-row operating table then shows what a
frozen policy does. If VUS-PR and the operating-point winner disagree, the
notebook reports both. False-alert point rates use raw alerts, while their 95%
Poisson bounds use topology/time clusters to avoid pretending correlated alarms
are independent. Recall confidence bounds, clipped rates and results by fault
type are retained. Robust-scaled features are capped at ±50 before Isolation
Forest and PCA to prevent numerical overflow from corrupt but finite values.

```text
outputs/models/v1.3.0/<sector>/<model_run_id>/
├── model_comparison.csv
├── model_comparison_by_fault_type.csv
├── vus_pr_points.csv
├── vus_pr_summary.csv
├── seed_stability.csv
├── warmup_bias.csv
├── coverage_heterogeneity.csv
├── development_alerts.parquet
├── selected_model.joblib
├── selected_configuration.json
└── holdout_prediction_template.json
```

### 05 — Incident ranking and holdout

The development run reuses the selected alerts written by Notebook 04, avoiding
a second full scoring pass. The frozen model scores holdout only when explicitly
requested. Consecutive anomalous scores form alerts; sustained recovery closes
them. Telecom topology groups simultaneous
alerts into shared incidents without entering the detector. 3W incidents remain
recording/well based because no shared topology is supplied.

Incidents are ranked lexicographically by operational evidence: affected
entities, number of anomalous metrics, peak score divided by the frozen
threshold, and start time. Group priority is derived from observed group
cardinality rather than a Telecom-specific list.

The default is development review. Before `RUN_HOLDOUT=1`, copy
`holdout_prediction_template.json` to `holdout_prediction.json` and pre-register
an expected value and confirmation range for every primary metric. Opening
holdout writes `holdout_receipt.json` beside sealed truth. An existing receipt
blocks every later attempt, making holdout mechanically single-use for that
evaluation run.

```text
outputs/final/v1.3.0/<sector>/<final_run_id>/<development-or-holdout>/
├── ranked_incidents.csv
├── evaluation_summary.csv
├── fault_type_results.csv
├── model_comparison_by_fault_type.csv
└── holdout_prediction_comparison.csv   # holdout only
```

## Recommended execution

For each sector:

1. Run the relevant 01A pack notebook.
2. Run 01B for that sector.
3. Run 02 and review its compact summary and plots.
4. Run 03 and confirm all nine evaluator controls pass.
5. Run 04 and review VUS-PR plus the 12-row operating comparison.
6. Run 05 with `RUN_HOLDOUT=0`; inspect the ranked development incidents.
7. Freeze the policy and model, complete `holdout_prediction.json`, then run 05
   once with `RUN_HOLDOUT=1`.

Do not compare raw metric values or alert counts across sectors. Compare the
same operational metrics under each sector's declared exposure unit and split
design.
