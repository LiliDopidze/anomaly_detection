# Notebook run guide

Use the `Python (telco-anomaly)` kernel and run the notebooks in numerical
order. Restart the kernel before each notebook and choose **Run All**. The
notebooks contain orchestration, visible checks, tables, and plots; tested
calculations live in `src/telco_anomaly` so the same calculation is not copied
into several notebooks.

## Primary PON workflow

| Notebook | Reads | Produces | Opens labels? |
|---|---|---|---|
| `00_PROJECT_CONTRACT` | YAML configuration | Printed, versioned product contract | No |
| `01_DATA_QUALITY_AUDIT` | Native synthetic PON files | Source audit tables | No |
| `02_CANONICAL_DATA_MODEL` | Native telemetry, topology, service windows and separately staged truth | A vendor-neutral pack and truth-unmounted `SPEC-CORE` | Stages them, but the canonical detector run does not mount them |
| `03_SPLITS_AND_TRUTH_LOCK` | Pack plus split definitions | Physical calibration, development and locked-holdout truth directories | Yes, evaluator setup only |
| `04_CALIBRATION_EDA` | Calibration `SPEC-CORE` | Time-series plots, robust profiles and frozen EDA decisions | No |
| `05_FEATURE_ENGINEERING` | Calibration/development `SPEC-CORE` plus EDA decisions | Causal features, with calibration split into separate fit and threshold slices | No |
| `06_PRIMARY_UNSUPERVISED_MODEL` | Calibration/development features and topology | Early-calibration model, late-calibration thresholds, and frozen scoring policy | No |
| `07_CHALLENGER_MODELS` | Late-calibration workload plus development truth | Label-free operating points, labelled comparisons and, only if gates pass, a frozen selection | Development only |
| `08_ALERTS_INCIDENTS_AND_DYING_GASP` | Frozen selection, scores and observable operational events | Persistent alerts and consolidated incidents | No |
| `09_TOPOLOGY_LOCALISATION` | Incidents and observable topology | Ranked location candidates with ambiguity | No |
| `10_LOCKED_EVALUATION` | Frozen model plus requested truth partition | Detection, workload and localisation metrics with uncertainty | Yes, after freeze |
| `11_PUBLIC_DATASET_VALIDATION` | Optional public Telecom sources | Reviewed adapter drafts and separate qualification packs | Only for the controlled failure benchmark |
| `12_INFERENCE_DEMO_AND_MODEL_CARD` | Frozen incidents, localisation and available results | Interactive chronological replay and model card | No |

Notebook 07 deliberately fails closed. If no candidate satisfies the declared
development gates, it writes `best_diagnostic_configuration.json` but not a
deployable selection. To inspect the product flow without changing that fact,
set `ALLOW_DIAGNOSTIC_DEMO=1` for Notebook 08. Never use this switch to open
holdout or claim deployment readiness.

## Data location

Set one environment variable before starting Jupyter:

```bash
export TELCO_DATA_ROOT="$HOME/telco_anomaly_data"
jupyter lab
```

In Colab, the default is
`/content/drive/MyDrive/telco_anomaly_data`; the existing
`/content/drive/MyDrive/anomaly_detection` location is also recognised. To use
a different repository checkout, set `TELCO_PROJECT_ROOT` explicitly.

## Re-running a completed stage

Outputs are immutable. A completed run is validated and reused; it is never
silently overwritten. After changing data, configuration, or code, set a new
stage run ID—for example:

```bash
export PON_FEATURE_RUN_ID=synthetic_pon_features_v4
export PON_MODEL_RUN_ID=synthetic_pon_models_v5
```

Keep downstream run IDs aligned with the inputs printed at the top of each
notebook. Delete an old output only when you intentionally want to discard it;
normally, retain both runs so their manifests can be compared.

Notebook 05 deliberately processes the full calibration and development
partitions. In Colab it uses `/content` for large temporary wide tables and
copies only final feature files to Drive. It prints progress after each major
step and every 25 entity episodes. Set `TELCO_WORK_ROOT` only when a different
local scratch disk is required; do not point it at Google Drive.

Notebook 06 reuses the immutable v4 feature run. It needs at least 4 GB of
local scratch space by default, keeps DuckDB spill files under that scratch
directory, and deletes residual and topology intermediates immediately after
use. If a previous Colab attempt filled `/content`, restart the runtime before
rerunning Notebook 06; Notebook 05 does not need to be rerun.

## Public Telecom qualification

Notebook 11 can acquire one selected public source directly from its publisher
and cache the extracted files outside Git:

```python
%env PUBLIC_DATASET=ran_pm
%env DOWNLOAD_PUBLIC_DATA=1
```

Do not combine `PUBLIC_DATASET=all` with downloading: select one source so the
size and terms are explicit. Microsoft Optical additionally requires
`ACKNOWLEDGE_MICROSOFT_DATA_TERMS=1`; the optical-failure testbed requires
`ACKNOWLEDGE_OPTICAL_FAILURE_TERMS=1`.

Notebook 11 then creates a mapping draft and stops. A human must verify native
timestamp, entity, topology, metric meaning, units, cadence, and source terms,
then set `mapping_review_status: approved`. After a RAN or Microsoft optical
pack is ready, run Notebooks 04–06 with `TELCO_DATASET=ran_pm` or
`TELCO_DATASET=microsoft_optical`. Each dataset receives its own fitted
reference and thresholds. Do not run the label-based selection notebook on
these unlabelled sources and do not pool their raw metrics with PON.
