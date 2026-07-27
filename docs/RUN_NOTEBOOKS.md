# How to run the Milestone 1 notebooks

## 1. Put only data in Google Drive

Use this layout:

```text
MyDrive/anomaly_detection/
├── reference_dataset.parquet
├── topology.csv
├── entity_service_windows.csv
├── engineering_events.csv              # optional operational context
├── tickets.csv                         # evaluation-only
├── evaluation/                         # a trailing space is also detected
│   ├── gt_fault_registry.csv
│   ├── fault_entity_intervals.csv
│   ├── gt_fault_groups.csv
│   ├── gt_benign_anomalies.csv         # optional
│   └── gt_collection_gaps.parquet      # optional
└── sources/
    └── petrobras_3w/
        └── 2.0.0/
            └── raw/
                └── 3w_dataset_2.0.0/
                    ├── 0/ ... 9/
                    ├── folds/
                    ├── dataset.ini
                    ├── LICENSE-CC-BY
                    └── README.md
```

If the telecom files are inside `anomaly_detection/Data/`, that is also supported.
Do not upload generated outputs or Python source into the input folders.

## 2. Download or open the notebooks

In GitHub, open
`notebooks/google_drive/milestone1_v0_3/`, select a notebook, then use **Download raw
file**. Upload the `.ipynb` to Google Colab, or open it from Google Drive.

Use a fresh Colab runtime for the first notebook. Choose **Runtime → Run all**.
Authorize the Drive mount when Colab asks.

## 3. Notebook 02 — telecom materialisation

This notebook:

- reads the native synthetic telecom fixture;
- applies the Telecom Pack phrasebook;
- writes runtime-safe tables to `SPEC-CORE`;
- writes labels and truth to the separate `SPEC-EVAL`;
- records clean-process peak memory;
- records the cost of the long telemetry representation.

The default run is:

```text
outputs/milestone_1/v0.3/telecom/telecom_full_v1/
```

For a quick smoke test, set `SAMPLE_START` / `SAMPLE_END` or `ENTITY_IDS` in the
configuration cell. For acceptance, leave them empty and use a new full-run ID.

Runs are immutable. If the folder already exists, the workflow stops instead of
overwriting evidence. Either inspect the existing result or choose a new `RUN_ID`.

Successful output contains:

```text
<run_id>/
├── SPEC-CORE/
│   ├── telemetry/part-*.parquet
│   ├── metric_catalogue.parquet
│   ├── entity_registry.parquet
│   ├── entity_relations.parquet
│   ├── operational_events.parquet
│   ├── collection_gaps.parquet
│   └── manifest.json
├── SPEC-EVAL/
│   ├── gt_*.parquet
│   └── manifest.json
├── materialisation_report.json
├── memory_report.json
├── representation_report.json
└── workflow_report.json
```

## 4. Notebook 03 — truth-lock acceptance

Use exactly the same telecom source and `RUN_ID` as Notebook 02. This notebook does
not translate the full panel again. It:

- compares translations with truth present and redacted;
- explicitly removes `tickets.csv` in the redacted case;
- proves clipped and invalid value branches were tested;
- tests a timestamp canary and a deliberately leaky negative control;
- checks FEC and CRC exposure behaviour;
- tests delayed `known_at`;
- verifies runtime output with SPEC-EVAL mounted, renamed, removed, and empty.

The most important line is:

```text
translator_invariance: True
```

The runtime variants are supporting evidence; the translator comparison is the
actual leakage exit criterion. Results are written to `acceptance_report.json`.

## 5. Notebook 04 — Petrobras 3W challenge

This notebook is independent of Notebooks 02–03. It:

- validates the public 3W 2.0.0 source;
- selects the smallest real-well subset satisfying every challenge criterion;
- translates it with the same neutral contract;
- introduces condition-state truth without inventing severity;
- records what the contract could not express without distortion.

Outputs go to:

```text
outputs/milestone_1/v0.3/petrobras_3w/<run_id>/
fixtures/petrobras_3w/2.0.0/v0.3/<run_id>/
```

Check `threew_contract_fit_report.json`, especially
`not_representable_without_invention`.

## 6. GitHub runtime version

Colab installs the package from this repository. `main` is convenient while
developing, but an acceptance run should use a release tag or exact commit:

```python
RUNTIME_REF = "v0.3.1"
```

The notebook resolves that reference to a commit and the workflow report stores both.
This replaces the old runtime ZIP bundle and keeps the notebook readable.

## Common errors

**“native file not found”** means `TELECOM_SOURCE` does not point at the folder
containing the files (or its parent when the files are in `Data/`).

**“missing 3W configuration”** means `THREEW_SOURCE` must be one level deeper, at the
folder containing `dataset.ini`.

**“refusing to overwrite”** is intentional. Choose a new `RUN_ID`; do not delete
signed-off evidence.

**Notebook 03 says Notebook 02 output is missing** means its `RUN_ID` or output root
does not match Notebook 02.

**Memory budget failed** means the clean translation subprocess exceeded 8 GiB.
Reduce the sample or batch size for diagnosis, but keep the 8 GiB full-run acceptance
gate.

**GitHub install fails**: confirm the repository is public (or authenticate Colab),
restart the runtime, and rerun the setup cell.
