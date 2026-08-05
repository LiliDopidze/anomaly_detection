# Milestone 1 research workflow

Keep these five maintained files together in Google Drive:

```text
MyDrive/anomaly_detection/research/milestone1/
├── 01A_TELECOM_PACK.ipynb
├── 01A_PETROBRAS_3W_PACK.ipynb
├── 01B_COMMON_CANONICAL_ADAPTER.ipynb
├── 02_CANONICAL_EDA.ipynb
└── milestone1_core.py
```

The notebooks contain the visible research decisions. `milestone1_core.py` is one
flat helper file for the small amount of settled logic that must not be copied
between notebooks: schemas, validation, one content fingerprint, canonical
materialisation and isolation probes.

## Current contracts

- Pack interface: `0.6.0`
- SPEC-CORE: `0.9.0`
- SPEC-EVAL: `0.7.0`
- Canonical EDA: `0.3.0`

`SPEC-CORE v0.9.0` is the telemetry-only modelling floor:

```text
telemetry
metric_catalogue
entity_registry
observation_episodes
collection_gaps
```

The catalogue declares only measurement kind, unit, sampling mode and expected
cadence when known. Each sector pack supplies `quality_code` directly. Entity bounds are derived from observations
available at `as_of_ts`; they are not presented as contractual service windows.
Every telemetry row also carries an `episode_id`. An episode is one source-
declared observation run across which time-series differences may be computed.
It is one continuous stream per Telecom ONT and one source recording per 3W
file.

Pack observations are long and metric-level: one row means that one metric
was observed or attempted at that timestamp. Different metrics may therefore
have different timestamp grids and cadences without a sector branch in the
common adapter. A null row is an invalid observation; an absent row is not an
observation and may become an internal collection gap only between that metric's
first and last observation when the catalogue declares a periodic obligation.

Faults and condition states are physically separated in `SPEC-EVAL`.
A detector reads only `SPEC-CORE` and must run with `SPEC-EVAL` absent. There
is no `SPEC-CONTEXT` layer in the active telemetry-only workflow.

## Native data locations

Telecom:

```text
MyDrive/anomaly_detection/telco_syntetic_data/
├── reference_dataset.parquet
├── gt_fault_registry.csv
├── fault_entity_intervals.csv
├── tickets.csv                 # optional; leakage test only
└── topology.csv                # optional split metadata
```

The reference dataset must be the updated observable-only file. It must not
contain `gt_*`, `class`, `state`, fault, anomaly or label fields. Parquet is
recommended for the complete panel.

Tickets are not translated; when present, the leakage test verifies that removing
them cannot change model input. `entity_service_windows.csv` and
`engineering_events.csv` are not read. Topology may only derive infrastructure
groups in `SPLITS/`; it never enters SPEC-CORE or the detector.

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
- declares one continuous observation episode per ONT;
- writes observable ONT telemetry to `PACK-CORE`;
- writes fault events and affected-entity intervals to `PACK-EVAL`;
- writes calibration/development/holdout time ranges to `SPLITS`;
- optionally derives PON/splitter evaluation groups from topology;
- proves that deleting evaluation files does not change `PACK-CORE`.

### 01A Petrobras 3W pack

- maps official measurements that contain at least one value in the selected
  real-well population;
- omits an all-null metric from an episode instead of fabricating invalid
  observations for a sensor that was unavailable;
- preserves every selected source recording as a distinct episode;
- identifies real `WELL-*` recordings;
- creates a deterministic expanded development population;
- assigns entire wells to calibration, development or holdout;
- keeps `class` and `state` in `PACK-EVAL` only;
- records that phase timestamps are derived from published class labels;
- proves that redacting `class` and `state` does not change `PACK-CORE`.

### 01B common canonical adapter

- validates either pack through the same code path;
- creates immutable `SPEC-CORE v0.9.0`;
- copies splits and evaluation into separate directories;
- preserves sector-supplied invalid and clipped quality codes;
- derives observation bounds and per-metric periodic collection gaps;
- tests one-second and five-second metrics in the same episode, including a
  deliberately missing slow observation;
- supports an optional `as_of_ts` boundary;
- tests truth isolation, a deliberately leaky negative control and temporal
  isolation;
- prints all compact outputs and manifests for inspection.

### 02 canonical EDA

- reads `SPEC-CORE` only;
- performs a full structural audit with a global duplicate check;
- separates value validity from expected-observation coverage;
- computes all difference statistics within one entity-episode-metric series;
- prevents differences and rolling statistics from crossing episode boundaries;
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
├── PACK-EVAL/           # optional and never read by detector code
├── SPLITS/              # 3W whole-well partitions
└── pack_manifest.json    # source files, row counts and one fingerprint
```

Canonical runs:

```text
outputs/canonical/v0.9.0/<sector>/<canonical_run_id>/
├── SPEC-CORE/
├── SPEC-EVAL/           # optional
├── SPLITS/              # orchestration metadata, not model features
└── run_manifest.json     # source pack and canonical run summary
```

EDA evidence:

```text
outputs/eda/v0.3.0/<sector>/<eda_run_id>/
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
Pack v0.6 long metric-level interface, declare observation episodes, assign
simple source-quality codes, route labels to `PACK-EVAL`
and pass the
original-versus-redacted isolation test.

Do not add a sector branch to `milestone1_core.py`, Notebook 01B or Notebook 02. If
a genuine source concept cannot be represented without distortion, record the
failure before changing the versioned interface.

## Next stage

Before building `03_EVALUATION_HARNESS.ipynb`, freeze the alert-to-fault
matching policy and physically separate development truth from final holdout
truth. The existing `SPLITS` tables provide whole-well, temporal and optional
infrastructure-group boundaries; they are orchestration metadata, not model
features. The harness comes before feature engineering and models so random,
constant and leakage-prone detectors can test the evaluation rules first.
