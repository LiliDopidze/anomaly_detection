# Drive research workflow

## What to upload

Create this folder in Google Drive:

```text
MyDrive/anomaly_detection/
├── research/week1/
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

## Run order

### 1. Telecom contract and locked truth

Open `01_TELECOM_WEEK1_END_TO_END.ipynb` in Colab and choose **Runtime → Run all**.

It:

- defines `SPEC-CORE` and `SPEC-EVAL`;
- defines the complete Telecom Pack in visible cells;
- translates the native synthetic data in bounded batches;
- retains null rows as `quality_code=invalid`;
- marks FEC values at or above 5,000,000 as `clipped`;
- calculates constant FEC bit opportunity and varying throughput-derived CRC frame
  opportunity;
- keeps tickets and all truth exclusively in `SPEC-EVAL`;
- runs the primary original-versus-redacted translator invariance test;
- proves the lock test catches a deliberately leaky negative control;
- records the secondary runtime mount/rename/remove/empty check.

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

It selects the smallest deterministic real-well subset satisfying all challenge
criteria, records SHA-256 hashes, translates it through the same neutral contract,
adds condition-state truth, and writes `contract_fit_report.json`.

Default output:

```text
MyDrive/anomaly_detection/outputs/research/v0.3.0/petrobras_3w/contract_challenge_v1/
```

It also copies the selected, hash-pinned fixture by default. Set
`COPY_PINNED_FIXTURE=0` if Drive space is tight.

### 3. Model and rank incidents

Open `03_SECTOR_AGNOSTIC_MODELLING_AND_RANKING.ipynb` and choose **Run all**.

It defaults to the telecom core run and 20 entities for quick iteration. The critical
output is:

```text
MyDrive/anomaly_detection/outputs/research/v0.3.0/models/telecom/
└── telecom_robust_baseline_v1/
    ├── anomaly_scores.parquet
    ├── ranked_incidents.csv
    ├── ranked_incidents.parquet
    ├── modelling_report.json
    └── offline_evaluation.json
```

Set `MODEL_ENTITY_LIMIT=0` to use all entities. To model Petrobras instead:

```python
%env MODEL_SECTOR=petrobras_3w
%env MODEL_CORE_RUN_ROOT=/content/drive/MyDrive/anomaly_detection/outputs/research/v0.3.0/petrobras_3w/contract_challenge_v1
%env MODEL_RUN_ID=petrobras_robust_baseline_v1
```

The scoring and ranking cells read only `SPEC-CORE` and save both output files before
the evaluation cell reads `SPEC-EVAL`.

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
run the pipeline. Until then, the four Drive files are the maintained research
surface.
