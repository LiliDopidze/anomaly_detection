# Week 1 research workflow

Keep these five maintained files together in Google Drive:

```text
MyDrive/anomaly_detection/research/week1/
├── 01A_TELECOM_PACK.ipynb
├── 01A_PETROBRAS_3W_PACK.ipynb
├── 01B_COMMON_CANONICAL_ADAPTER.ipynb
├── 02_CANONICAL_EDA.ipynb
└── week1_core.py
```

The notebooks contain the visible research decisions. `week1_core.py` is one
flat helper file for the small amount of settled logic that must not be copied
between notebooks: schemas, validation, hashing, quality coding, canonical
materialisation and isolation probes.

## Current contracts

- Pack interface: `0.2.0`
- SPEC-CORE: `0.6.0`
- SPEC-EVAL: `0.6.0`
- Canonical EDA: `0.2.0`

`SPEC-CORE v0.6.0` is the telemetry-only modelling floor:

```text
telemetry
metric_catalogue
entity_registry
collection_gaps
```

The catalogue declares measurement kind, unit, sampling mode, expected cadence
when known, bounds and censoring. Entity bounds are derived from observations
available at `as_of_ts`; they are not presented as contractual service windows.

Optional topology is physically separated in `SPEC-CONTEXT`. Faults, condition
states and tickets are physically separated in `SPEC-EVAL`. A detector must be
able to run with both directories absent.

## Native data locations

Telecom:

```text
MyDrive/anomaly_detection/telco_syntetic_data/
├── reference_dataset.parquet
├── topology.csv
├── entity_service_windows.csv
├── engineering_events.csv
├── gt_fault_registry.csv
├── fault_entity_intervals.csv
└── tickets.csv
```

The reference dataset must be the updated observable-only file. It must not
contain `gt_*`, `class`, `state`, fault, anomaly or label fields. Parquet is
recommended for the complete panel.

Petrobras 3W:

```text
MyDrive/anomaly_detection/sources/petrobras_3w/2.0.0/raw/
└── 3w_dataset_2.0.0/
    ├── dataset.ini
    ├── README.md
    ├── LICENSE-CC-BY
    └── 0/ ... 9/
```

The 3W notebook verifies the official 2,228-file inventory. By default it
selects one deterministic real recording for every available `(well, event
class)` pair. It excludes simulated, drawn and duplicate download variants.
This is the modelling-development population; the old three-file subset is no
longer used for population statistics.

## Run order

### Telecom

1. Run `01A_TELECOM_PACK.ipynb`.
2. In `01B_COMMON_CANONICAL_ADAPTER.ipynb`, set `SECTOR = "telecom"`.
3. Run Notebook 01B.
4. In `02_CANONICAL_EDA.ipynb`, set `SECTOR = "telecom"`.
5. Run Notebook 02.

### Petrobras 3W

1. Run `01A_PETROBRAS_3W_PACK.ipynb`.
2. In `01B_COMMON_CANONICAL_ADAPTER.ipynb`, set
   `SECTOR = "petrobras_3w"`.
3. Run the same adapter without changing translation code.
4. In `02_CANONICAL_EDA.ipynb`, set `SECTOR = "petrobras_3w"`.
5. Run the same EDA notebook.

In Colab use **Runtime → Run all**. Output directories are immutable. Change
the relevant run ID before rebuilding a completed stage.

## Notebook responsibilities

### 01A Telecom pack

- maps Telecom measurements into the authored catalogue;
- validates the topology mapping;
- writes observable ONT telemetry to `PACK-CORE`;
- writes topology to optional `PACK-CONTEXT`;
- writes faults, entity intervals and tickets to `PACK-EVAL`;
- proves that deleting evaluation files does not change `PACK-CORE`;
- records service windows and engineering events as unused source capabilities.

### 01A Petrobras 3W pack

- maps all 27 official measurements;
- identifies real `WELL-*` recordings;
- creates a deterministic expanded development population;
- assigns entire wells to calibration, development or holdout;
- keeps `class` and `state` in `PACK-EVAL` only;
- records that phase timestamps are derived from published class labels;
- proves that redacting `class` and `state` does not change `PACK-CORE`.

### 01B common canonical adapter

- validates either pack through the same code path;
- creates immutable `SPEC-CORE v0.6.0`;
- copies optional context, splits and evaluation into separate directories;
- marks invalid and clipped values;
- derives observation bounds and periodic collection gaps;
- supports an optional `as_of_ts` boundary;
- tests truth isolation, a deliberately leaky negative control and temporal
  isolation;
- prints all compact outputs and manifests for inspection.

### 02 canonical EDA

- reads `SPEC-CORE` only;
- performs a full structural audit with a global duplicate check;
- separates value validity from expected-observation coverage;
- computes all difference statistics within one entity-metric series;
- reports rate distributions across entities rather than misleading pooled
  percentages;
- plots distributions, missingness, representative series, rolling robust
  statistics, ACF/PACF and cross-metric correlations;
- detects candidate periodicity on irregular observations before attempting a
  guarded short-gap-filled STL decomposition;
- suppresses population claims when too few entity series are available;
- saves compact, hash-pinned evidence for Notebook 03.

## Outputs

Sector packs:

```text
outputs/packs/<sector>/<pack_run_id>/
├── PACK-CORE/
├── PACK-CONTEXT/        # optional
├── PACK-EVAL/           # optional and never read by detector code
├── SPLITS/              # 3W whole-well partitions
├── pack_manifest.json
└── source_manifest.json
```

Canonical runs:

```text
outputs/canonical/v0.6.0/<sector>/<canonical_run_id>/
├── SPEC-CORE/
├── SPEC-CONTEXT/        # optional
├── SPEC-EVAL/           # optional
├── SPLITS/              # orchestration metadata, not model features
├── lineage.json
├── workflow_report.json
└── acceptance_report.json
```

EDA evidence:

```text
outputs/eda/v0.2.0/<sector>/<eda_run_id>/
├── analysis_windows.parquet
├── full_series_profile.parquet
├── series_profile.parquet
├── metric_profile.parquet
├── temporal_evidence.parquet
├── seasonality_evidence.parquet
├── dependence_evidence.parquet
├── figures/
└── eda_manifest.json
```

## Adding another sector

Create one new `01A_<SECTOR>_PACK.ipynb`. It must map native telemetry into the
Pack v0.2 interface, route labels to `PACK-EVAL`, declare unavailable
capabilities honestly and pass the original-versus-redacted isolation test.

Do not add a sector branch to `week1_core.py`, Notebook 01B or Notebook 02. If
a genuine source concept cannot be represented without distortion, record the
failure before changing the versioned interface.

## Next stage

After both canonical runs and EDA evidence are reviewed, build
`03_EVALUATION_HARNESS.ipynb`. The harness comes before feature engineering and
models so random, constant and leakage-prone detectors can test the evaluation
rules first.
