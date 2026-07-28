# Week 1 research workflow

This is the maintained research surface for the sector-agnostic anomaly
detection project. It contains four notebooks and one small Python file:

```text
research/week1/
├── 01_TELECOM_WEEK1_END_TO_END.ipynb
├── 02_PETROBRAS_3W_CONTRACT_CHALLENGE.ipynb
├── 00_CANONICAL_PROFILE.ipynb
├── 03_SECTOR_AGNOSTIC_MODELLING_AND_RANKING.ipynb
└── week1_core.py
```

The notebook numbers describe responsibilities, not execution order. Run `00`
after a sector notebook has created canonical data.

## Drive layout

Upload the five maintained files above to:

```text
MyDrive/anomaly_detection/research/week1/
```

Keep the telecom source here:

```text
MyDrive/anomaly_detection/telco_syntetic_data/
├── reference_dataset.parquet
├── topology.csv
├── entity_service_windows.csv
├── engineering_events.csv
├── alarms.csv                         # optional operational input
├── tickets.csv                        # evaluation only
├── gt_panel.parquet                   # evaluation only
├── gt_fault_registry.csv              # evaluation only
├── fault_entity_intervals.csv         # evaluation only
├── gt_fault_groups.csv                # evaluation only
├── gt_benign_anomalies.csv            # evaluation only
└── gt_collection_gaps.parquet         # evaluation only
```

The updated `reference_dataset.parquet` is the observable panel. It must not
contain `gt_` columns. The notebooks also accept the corrected folder spelling
`telco_synthetic_data` and an optional `native/` subfolder.

Keep Petrobras 3W here:

```text
MyDrive/anomaly_detection/sources/petrobras_3w/2.0.0/raw/
└── 3w_dataset_2.0.0/
    ├── dataset.ini
    ├── README.md
    ├── LICENSE-CC-BY
    └── 0/ ... 9/
```

Do not upload the datasets to GitHub. Data and materialised outputs stay on
Drive. GitHub is for the notebooks, `week1_core.py`, documentation and tests.

## Run order

### Telecom

1. `01_TELECOM_WEEK1_END_TO_END.ipynb`
2. `00_CANONICAL_PROFILE.ipynb`
3. `03_SECTOR_AGNOSTIC_MODELLING_AND_RANKING.ipynb`

### Petrobras 3W

1. `02_PETROBRAS_3W_CONTRACT_CHALLENGE.ipynb`
2. Set `PROFILE_SECTOR=petrobras_3w`, then run
   `00_CANONICAL_PROFILE.ipynb`
3. Set `MODEL_SECTOR=petrobras_3w`, then run
   `03_SECTOR_AGNOSTIC_MODELLING_AND_RANKING.ipynb`

In Colab, open a notebook and choose **Runtime → Run all**. Existing run
directories are immutable. Change the relevant run ID before repeating a
completed run.

## What each notebook does

### 01 — Telecom contract, translation and truth lock

Notebook 01 is the Telecom Pack and adapter workbench. It:

- defines the neutral `SPEC-CORE` and `SPEC-EVAL` contracts;
- defines telecom metric meanings, entity types and relations;
- converts the v4.1 wide panel into canonical long telemetry in batches;
- maps native nulls to `quality_code=invalid`;
- maps FEC values at the generator ceiling to `quality_code=clipped`;
- calculates source-faithful FEC and CRC exposure;
- converts engineering events and alarms into operational events;
- routes tickets and all ground truth to `SPEC-EVAL`;
- hash-pins every source file used in `source_manifest.json`;
- proves that original and truth-redacted native inputs produce identical
  canonical content;
- proves that a deliberately leaky translator fails the isolation harness.

Example:

```text
native:
timestamp_utc=2026-01-01T00:00Z
ont_id=ONT-00001
rx_power_dbm=-22.4

canonical:
event_ts=2026-01-01T00:00Z
entity_id=ONT-00001
metric_id=telecom.optical.rx_power
value=-22.4
quality_code=measured
```

Default output:

```text
outputs/research/v0.3.0/telecom/telecom_v4_1_full_v1/
├── SPEC-CORE/
├── SPEC-EVAL/
├── source_manifest.json
├── workflow_report.json
└── acceptance_report.json
```

For a quick development run, set a small entity/time slice and a new run ID:

```python
%env TELECOM_ENTITY_IDS=ONT-00001,ONT-00002
%env TELECOM_SAMPLE_START=2026-01-01
%env TELECOM_SAMPLE_END=2026-01-08
%env TELECOM_RUN_ID=telecom_debug_v1
```

Leave the filters empty for the final materialisation.

### 02 — Petrobras 3W contract challenge

Notebook 02 is not a second model. It tests whether the supposedly neutral
contract works outside telecom. It:

- verifies the official 3W 2.0.0 source inventory;
- defines the OilWell Pack;
- selects the smallest real-well fixture satisfying the coverage criteria;
- hash-pins the selected source files;
- converts native measurements into the same canonical telemetry schema;
- keeps native `class` and `state` out of `SPEC-CORE`;
- writes event truth and condition-state truth to `SPEC-EVAL`;
- records facts the adapter could not express without invention.

It does not invent topology, tickets, operating events or severity. Empty
canonical tables are valid when the source does not contain those facts.

Default output:

```text
outputs/research/v0.3.0/petrobras_3w/contract_challenge_v1/
```

### 00 — Canonical statistical profile

Notebook 00 runs only after canonical materialisation. The same notebook
profiles telecom or Petrobras because it sees canonical names and measurement
kinds, not native source columns.

It:

- opens `SPEC-CORE` only and verifies the truth box was not read;
- chooses a deterministic bounded entity sample;
- freezes the first 40% of each selected history as pre-label evidence;
- displays canonical fields, units, sampling semantics and catalogue meaning;
- quantifies coverage, missing values, clipping, constant runs and degenerate
  series;
- computes robust distributions and between/within-entity variation;
- recommends only measurement-kind transformations;
- plots representative series, rolling median/IQR and level/difference
  correlations;
- tests pack-declared candidate periods on complete regular segments;
- uses robust STL and ACF/PACF as descriptive evidence.

It deliberately does not choose anomaly thresholds, remove seasonality
automatically, inspect labels or decide that an extreme value is a fault.

Default profile output:

```text
outputs/research/v0.3.0/profiles/<sector>/<core_run_id>/
├── metric_profile.parquet
├── series_quality.parquet
├── decomposition_evidence.parquet
├── profile_windows.parquet
├── profile_manifest.json
└── figures/
```

The default profile uses eight entities. Change `PROFILE_ENTITY_LIMIT` for
broader evidence; do not change `PROFILE_FRACTION` without also changing the
model calibration fraction.

### 03 — Baseline model and ranked incidents

Notebook 03 is real baseline modelling. It:

- verifies that Notebook 00 profiled the same immutable `SPEC-CORE`;
- verifies the profile was label-free;
- applies the profile's neutral, measurement-kind transformations;
- builds shifted rolling-median and robust-scale baselines;
- calibrates metric thresholds on the early 40% of each entity's history;
- scores the later period without reading truth;
- groups point alerts into persistent operational episodes;
- ranks incidents and applies an explicit daily workload budget;
- freezes all model artifacts;
- only then opens `SPEC-EVAL` for offline evaluation.

The profile does not tune the model. It provides auditable statistical
evidence. The baseline remains intentionally simple so later seasonal,
multivariate or topology-aware challengers have an honest reference.

Default model output:

```text
outputs/research/v0.3.0/models/telecom/
└── telecom_episode_baseline_v3/
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

Start with the default 20 entities. The notebook currently combines the
selected slice in memory, so increase `MODEL_ENTITY_LIMIT` gradually rather
than starting with `0` (all entities).

## What is unified and what changes by sector

Unified:

- canonical telemetry and entity schemas;
- measurement-kind vocabulary;
- quality codes and truth isolation;
- profile calculations;
- baseline scoring, calibration, episode formation and ranking;
- evaluation boundary.

Sector-specific:

- native file discovery and field mapping;
- metric catalogue, units, bounds and sampling semantics;
- entity types and known relations;
- exposure formulas;
- interpretation of native labels into evaluation truth.

For a new sector, create a new adapter/pack notebook like Notebook 02. Do not
copy or edit the canonical profiler or model unless the new sector exposes a
real contract limitation. Record anything the source cannot express instead
of manufacturing facts to make the adapter look complete.

## Why `week1_core.py` remains

The notebooks contain decisions that are still under research. The one flat
Python file contains settled mechanics where duplication can create silent
scientific errors: batch translation, exposure arithmetic, clipping, canonical
hashing, truth routing and deterministic 3W selection.

There is no package, CLI, release process or notebook generator. Those belong
later, when the ranked incident list is useful and more than one person needs
to operate the workflow.

## Next step after Week 1

Review the top 20–50 rows of `ranked_incidents.csv` with someone who understands
the operational setting. Record whether each row should be investigated,
suppressed, merged or enriched with missing context. This determines the true
unit of work and evaluation cost.

Then compare focused challengers against the frozen baseline: seasonal
baselines where the canonical profile supports them, count models for interval
counts, multivariate evidence, and topology-aware propagation. Judge them at
incident level using recall, lead time and operator workload—not point-level
accuracy alone.
