# Drive research workflow

## What to upload

Create this folder in Google Drive:

```text
MyDrive/anomaly_detection/
├── research/week1/
│   ├── 00_NATIVE_DATA_EXPLORATION.ipynb
│   ├── 01_TELECOM_WEEK1_END_TO_END.ipynb
│   ├── 02_PETROBRAS_3W_CONTRACT_CHALLENGE.ipynb
│   ├── 03_SECTOR_AGNOSTIC_MODELLING_AND_RANKING.ipynb
│   └── week1_core.py
├── Full dataset/
│   ├── Data/
│   │   ├── reference_dataset.parquet
│   │   ├── topology.csv
│   │   ├── entity_service_windows.csv
│   │   └── engineering_events.csv              # optional
│   └── evaluation/
│       ├── gt_fault_registry.csv
│       ├── fault_entity_intervals.csv
│       ├── gt_fault_groups.csv
│       └── tickets.csv                         # evaluation only
└── sources/petrobras_3w/2.0.0/raw/3w_dataset_2.0.0/
    ├── dataset.ini
    ├── README.md
    ├── LICENSE-CC-BY
    └── 0/ ... 9/
```

The telecom source resolver also accepts the files directly under `Full dataset/`.
It tolerates an evaluation folder whose name has accidental leading/trailing spaces,
but renaming it to `evaluation` is clearer.

Do not upload datasets to GitHub. The notebooks read data from Drive and write all
materialised data, reports, model scores, and ranked incidents back to Drive.

## Why the large data stays Parquet

Canonical telemetry and anomaly scores are written only as Parquet because Parquet
preserves types, compresses efficiently, and supports column-level reads. The telecom
panel can reach roughly 69 million canonical rows, while the three-file 3W challenge
produces more than eight million. A duplicate CSV export would add substantial Drive
space and I/O without helping the model.

Large outputs therefore look like:

```text
SPEC-CORE/
├── telemetry/
│   ├── part-00000.parquet
│   └── part-00001.parquet
├── metric_catalogue.parquet
├── entity_registry.parquet
└── ...
```

Notebook 03 additionally writes the small operator-facing
`ranked_incidents.csv`. This is intentionally a summary rather than a second copy of
the underlying dataset.

## Run order

### 0. Explore the native data

Open `00_NATIVE_DATA_EXPLORATION.ipynb` first and choose
**Runtime → Run all**. It defaults to telecom.

The notebook acts as a statistical intake audit:

1. Counts source files and Parquet rows from exact metadata.
2. Identifies the row grain, operational measurements, context and embedded truth.
3. Takes a reproducible bounded sample from every telecom row group.
4. Calculates count, missingness, zero share, robust quantiles, range, skewness and
   boundary concentration for every operational measurement.
5. Compares missingness across entities so a global average cannot hide an unusable
   ONT or well.
6. Plots bulk distributions and a Spearman dependence matrix.
7. Loads complete histories for a few selected entities and measures cadence,
   duplicate timestamps, gaps and frozen values.
8. Summarises vendor, firmware, topology and other available context.
9. Opens labels only in a clearly marked evaluation-only section.
10. Writes a small statistical report, tables and figures to Drive. It does not copy
    the sampled telemetry or create `SPEC-CORE`.

Default telecom output:

```text
MyDrive/anomaly_detection/outputs/exploration/telecom/telecom_native_eda_v1/
├── eda_report.json
├── source_inventory.csv
├── native_schema.csv
├── numeric_summary_sample.csv
├── entity_missingness_summary_sample.csv
├── quality_flags_sample.csv
├── temporal_quality_selected_series.csv
├── evaluation_only_summary.csv
└── figures/
```

The exact shape and file composition are not estimates. Distributions,
correlations, missingness prevalence and label prevalence are sample estimates.
Complete-series temporal checks are exact only for the selected example entities.

Useful telecom controls:

```python
%env EDA_SECTOR=telecom
%env EDA_RUN_ID=telecom_native_eda_v1
%env EDA_SAMPLE_ROWS=200000
%env EDA_SERIES_ENTITY_COUNT=2
%env EDA_SERIES_DAYS=7
```

To inspect Petrobras 3W instead:

```python
%env EDA_SECTOR=petrobras_3w
%env EDA_RUN_ID=threew_real_wells_eda_v1
%env EDA_THREEW_FILE_COUNT=20
%env EDA_THREEW_ROWS_PER_FILE=10000
```

The 3W exact inventory counts all real, simulated and hand-drawn files separately.
Its default descriptive sample draws 20 files uniformly from filenames beginning
with `WELL-`. It does not use event labels to choose the operational sample and does
not use `SIMULATED_` or `DRAWN_` files. Event-directory and row-label composition
are opened only in the final evaluation-only section.

Use `EDA_INCLUDE_TRUTH=0` when you want a purely operational EDA run. Use a new
`EDA_RUN_ID` if you want to retain two sets of exploration outputs.

### 1. Telecom contract and locked truth

Open `01_TELECOM_WEEK1_END_TO_END.ipynb` in Colab and choose **Runtime → Run all**.

Step by step, it:

1. Mounts Drive and imports only the settled mechanics from `week1_core.py`.
2. Defines the neutral `SPEC-CORE` and `SPEC-EVAL` table contracts in visible cells.
3. Defines the Telecom Pack: metric meanings, entity hierarchy, geographic relation,
   measurement kinds, directions, bounds, censoring and exposure provenance.
4. Discovers the native panel, topology, service windows and evaluation files.
5. Converts native wide telemetry to canonical long telemetry in bounded batches.
6. Writes every canonical table as partitioned Parquet.
7. Applies value-quality rules: native null becomes `invalid`; FEC at or above
   5,000,000 becomes `clipped`.
8. Calculates constant FEC bit opportunities and throughput-derived, varying CRC
   frame opportunities.
9. Stores tickets, fault IDs, true intervals and causes only in `SPEC-EVAL`.
10. Translates an original and a truth-redacted fixture and proves their canonical
    `SPEC-CORE` hashes are identical.
11. Injects a deliberate timestamp leak and proves the negative control detects it.
12. Saves the workflow, lineage and acceptance reports.

Example translation:

```text
Native:
timestamp_utc=2026-01-01T00:00Z
ont_id=ONT-00001
rx_power_dbm=-22.4

Canonical telemetry:
event_ts=2026-01-01T00:00Z
entity_id=ONT-00001
metric_id=telecom.optical.rx_power
value=-22.4
quality_code=measured
```

A native `gt_fault_id` on the same source row is not copied into canonical telemetry;
it is routed to `SPEC-EVAL`.

Default output:

```text
MyDrive/anomaly_detection/outputs/research/v0.3.0/telecom/telecom_full_v1/
```

Outputs are immutable. Change `TELECOM_RUN_ID` before rerunning a completed run.

For development, set environment variables before running:

```python
%env TELECOM_ENTITY_IDS=ONT-00001,ONT-00002
%env TELECOM_SAMPLE_START=2026-01-01
%env TELECOM_SAMPLE_END=2026-01-08
%env TELECOM_RUN_ID=telecom_debug_v1
```

Leave entity IDs and time limits empty for the final materialisation.

### 2. Petrobras 3W contract challenge

Open `02_PETROBRAS_3W_CONTRACT_CHALLENGE.ipynb` and choose **Run all**.

Step by step, it:

1. Verifies dataset version 2.0.0, directories `0`–`9`, and 2,228 event instances.
2. Defines the OilWell Pack for 27 state, opening, pressure, flow and temperature
   measurements.
3. Selects a deterministic three-real-well fixture covering normal operation, a
   persistent condition, a transient event, a state transition and missing values.
4. Hash-pins the selected source files.
5. Converts the 297,590 selected native wide rows into 8,034,930 canonical long
   telemetry rows in 5,000-row native batches.
6. Writes all canonical tables as partitioned Parquet.
7. Removes native `class` and `state` from `SPEC-CORE`.
8. Converts `class` to event truth and `state` to condition-interval truth in
   `SPEC-EVAL`.
9. Proves empty relations, tickets and operational events are accepted instead of
   inventing facts absent from 3W.
10. Writes `contract_fit_report.json`, recording what the source cannot represent.

Example translation:

```text
Native:
entity inferred from file = WELL-00014
P-ANULAR = 7,100,000
class = 3

SPEC-CORE:
entity_id=WELL-00014
metric_id=oil_well.pressure.p_anular
value=7100000

SPEC-EVAL:
event code 3 = Severe Slugging
```

Default output:

```text
MyDrive/anomaly_detection/outputs/research/v0.3.0/petrobras_3w/contract_challenge_v1/
```

It also copies the selected, hash-pinned fixture by default. Set
`COPY_PINNED_FIXTURE=0` if Drive space is tight.

### 3. Model and rank incidents

Open `03_SECTOR_AGNOSTIC_MODELLING_AND_RANKING.ipynb` and choose **Run all**.

Step by step, it:

1. Reads only `SPEC-CORE`; evaluation truth is not loaded.
2. Selects a small entity slice for rapid model iteration.
3. Transforms values from neutral catalogue semantics: gauges remain values,
   interval counts become log rates when exposure exists, cumulative counters
   become reset-safe increments, and zero-inflated bounded values use a log hurdle.
4. Builds a shifted, history-only rolling median and robust scale for every
   entity-metric series. The current observation is never in its own baseline.
5. Gives every entity its own early calibration window. This matters for historical
   fixtures such as 3W whose well records occur in different years.
6. Learns a truth-free raw-score threshold per metric from those calibration
   windows, then converts raw deviations to comparable calibrated scores.
7. Saves scores, calibration windows, thresholds, and metric diagnostics before
   evaluation truth is read.
8. Groups point alerts into adjacent episodes at their operational domain. For
   telecom, two anomalous ONTs served by the same L2 splitter can become one
   shared-domain episode.
9. Rejects weak isolated episodes unless they are persistent, affect multiple
   entities, affect multiple metrics, or are far above their calibrated threshold.
10. Ranks eligible episodes and applies an explicit daily incident budget. Rejected
    candidates remain inspectable rather than disappearing.
11. Saves `ranked_incidents.parquet` and the small operator-friendly
    `ranked_incidents.csv`.
12. Only after all model outputs are frozen, reads `SPEC-EVAL` and compares incident
    recall before and after the daily budget, truth-overlap fraction, condition
    coverage, and lead time.

Example:

```text
ONT-00001: abnormal receive power at 10:00
ONT-00002: abnormal receive power at 10:00
Both are served by L2-001

Result:
one ranked incident for L2-001
affected_entity_count=2
anomalous_metric_count=1
```

It defaults to the telecom core run and 20 entities for quick iteration. The output
is:

```text
MyDrive/anomaly_detection/outputs/research/v0.3.0/models/telecom/
└── telecom_episode_baseline_v2/
    ├── anomaly_scores.parquet
    ├── calibration_windows.parquet
    ├── calibration_thresholds.parquet
    ├── model_diagnostics.parquet
    ├── candidate_episodes.parquet
    ├── ranked_incidents.csv
    ├── ranked_incidents.parquet
    ├── modelling_report.json
    └── offline_evaluation.json
```

Do not start with `MODEL_ENTITY_LIMIT=0` on the full telecom run: the current research
model combines selected telemetry in memory. Increase the entity count gradually
until the ranking logic is accepted; full-population scoring is the next streaming
engineering step.

The most useful controls are:

- `MODEL_ENTITY_LIMIT` — defaults to 20; `0` means every entity.
- `MODEL_HISTORY` and `MODEL_MIN_HISTORY` — rolling history lengths in native
  observations.
- `MODEL_CALIBRATION_FRACTION` — early fraction of each entity's history used only
  to set thresholds.
- `MODEL_POINT_QUANTILE` — desired high calibration-score quantile.
- `MODEL_MIN_RAW_THRESHOLD` — safety floor when a calibration distribution is
  degenerate.
- `MODEL_MIN_EPISODE_BUCKETS` and `MODEL_HIGH_CONFIDENCE_RATIO` — evidence required
  for an episode to become an incident.
- `MODEL_MAX_INCIDENTS_PER_DAY` — the explicit operator workload budget; `0`
  disables it.

Use a new `MODEL_RUN_ID` whenever a control changes. Existing output directories are
immutable by design.

To model Petrobras instead:

```python
%env MODEL_SECTOR=petrobras_3w
%env MODEL_CORE_RUN_ROOT=/content/drive/MyDrive/anomaly_detection/outputs/research/v0.3.0/petrobras_3w/contract_challenge_v1
%env MODEL_RUN_ID=petrobras_episode_baseline_v2
```

The scoring and ranking cells read only `SPEC-CORE` and save every model artifact
before the evaluation cell reads `SPEC-EVAL`. `SPEC-EVAL` is therefore a scorecard,
not a feature source or threshold-tuning source.

`ranked_incidents.csv` is the review table. `candidate_episodes.parquet` explains
what was rejected or removed by the daily budget. `model_diagnostics.parquet`
reveals which metrics dominate the alert stream. These two diagnostic files are
usually where you look first when the ranked list is noisy.

## What happens after the four notebooks

The immediate next step is not packaging. It is to validate whether the ranked list
matches real operational decisions.

1. **Review the top 20–50 telecom incidents.** For each row, record: investigate,
   suppress, merge with another incident, or missing context.
2. **Fix the unit of work.** Decide whether an operator acts on an ONT, splitter,
   service, geographic cluster or a correlated group.
3. **Freeze the first evaluation protocol.** Use chronological train/calibration/test
   periods and read `SPEC-EVAL` only after scores are frozen.
4. **Compare honest baselines.** Include persistence, robust univariate rules and
   topology grouping. The generator is sufficient when a competent baseline is
   neither trivially perfect nor useless.
5. **Improve the model only where operator review identifies value.** Likely areas
   are seasonal baselines, cross-metric evidence, topology propagation and incident
   merging.
6. **Scale scoring after the ranking definition stabilises.** Replace Notebook 03's
   in-memory entity slice with partition-by-entity or partition-by-time scoring.
7. **Bring in engineering support later.** Package, schedule, monitor and deploy only
   after the incident list is useful and another person needs to operate the workflow.

The exit artifact for this research stage is therefore not “a trained model.” It is
a reproducible ranked incident list plus documented operator feedback explaining
which rankings are useful and why.

## What you should edit

Edit notebook cells when you are still reasoning about:

- contract fields;
- sector mappings;
- acceptance assertions;
- measurement transformations;
- model design;
- incident grouping and ranking.

Do not copy the exposure, FEC clipping, canonical hashing, truth routing, batch
materialisation, or 3W selection functions into notebook cells. Those settled
mechanics have one implementation in `week1_core.py`.

## When to bring back engineering infrastructure

Add a package, CLI, orchestration, CI, or release process only after the ranked
incident output is useful to an operator and more than one person needs to change or
run the pipeline. Until then, the five Drive files are the maintained research
surface.
