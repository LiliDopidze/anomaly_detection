# Week 1 Drive workflow

Upload these five maintained files to:

```text
MyDrive/anomaly_detection/research/week1/
├── 01A_TELECOM_PACK.ipynb
├── 01A_PETROBRAS_3W_PACK.ipynb
├── 01B_COMMON_CANONICAL_ADAPTER.ipynb
├── 02_CANONICAL_EDA.ipynb
└── week1_core.py
```

The four notebooks are the visible research flow. `week1_core.py` is one flat
helper file containing only shared mechanics whose silent duplication would
invalidate the experiment.

## Data locations

Telecom:

```text
MyDrive/anomaly_detection/telco_syntetic_data/
├── reference_dataset.parquet          # observable data only
├── topology.csv
├── entity_service_windows.csv
├── gt_fault_registry.csv
├── fault_entity_intervals.csv
├── gt_fault_groups.csv                 # retained source; not needed by v0.1 pack
├── tickets.csv
└── ...                                 # other evaluation files may remain here
```

The notebook also accepts `reference_dataset.csv`, but Parquet is recommended
for the complete panel. The observable reference dataset must not contain
`gt_*`, `class`, or `state` columns.

Petrobras 3W:

```text
MyDrive/anomaly_detection/sources/petrobras_3w/2.0.0/raw/
└── 3w_dataset_2.0.0/
    ├── dataset.ini
    ├── README.md
    ├── LICENSE-CC-BY
    └── 0/ ... 9/
```

The 3W notebook verifies exactly 2,228 event-instance files and uses three
hash-pinned real-well files. It does not use simulated files.

## Run order

### Telecom

1. Run `01A_TELECOM_PACK.ipynb`.
2. Open the **Choose the sector here** cell in
   `01B_COMMON_CANONICAL_ADAPTER.ipynb`.
3. Select Telecom:

   ```python
   SECTOR = "telecom"
   ```

4. Run `01B_COMMON_CANONICAL_ADAPTER.ipynb`.
5. In `02_CANONICAL_EDA.ipynb`, choose `SECTOR = "telecom"` and enter the
   canonical run ID written by `01B`.
6. Run `02_CANONICAL_EDA.ipynb`.

### Petrobras 3W

1. Run `01A_PETROBRAS_3W_PACK.ipynb`.
2. Open the **Choose the sector here** cell in
   `01B_COMMON_CANONICAL_ADAPTER.ipynb`.
3. Select Petrobras 3W:

   ```python
   SECTOR = "petrobras_3w"
   ```

4. Run the same adapter notebook. No translation code changes.
5. In `02_CANONICAL_EDA.ipynb`, choose `SECTOR = "petrobras_3w"` and enter the
   canonical run ID written by `01B`.
6. Run the same EDA notebook. No analysis code changes.

In Colab choose **Runtime → Run all**. Output directories are immutable. Change
the relevant pack, canonical, or EDA run ID before repeating a completed stage.

The final `01B` section inventories every generated file, prints every JSON
manifest/report, and previews every Parquet output. Partitioned observations
and telemetry are summarized by schema, total rows, and first/last samples
instead of printing millions of rows. This inspection is Week 1 contract QA;
the canonical EDA still reads `SPEC-CORE` only.

If the canonical run already exists and you only want to inspect it, use the
sector-switch cell:

```python
SECTOR = "telecom"       # or "petrobras_3w"
BUILD_CANONICAL = False  # read the existing canonical run
```

## What the notebooks produce

Each sector notebook writes:

```text
outputs/packs/<sector>/<pack_run_id>/
├── PACK-CORE/
│   ├── observations/part-*.parquet
│   ├── metric_catalogue.parquet
│   ├── entity_registry.parquet
│   └── entity_relations.parquet
├── PACK-EVAL/
├── pack_manifest.json
└── source_manifest.json
```

The common adapter writes:

```text
outputs/canonical/v0.5.0/<sector>/<canonical_run_id>/
├── SPEC-CORE/
│   ├── telemetry/part-*.parquet
│   ├── metric_catalogue.parquet
│   ├── entity_registry.parquet
│   ├── entity_relations.parquet
│   ├── collection_gaps.parquet
│   └── manifest.json
├── SPEC-EVAL/
├── lineage.json
├── workflow_report.json
└── acceptance_report.json
```

Later EDA, feature engineering, and models receive only `SPEC-CORE`.
`SPEC-EVAL` is opened only after model outputs are frozen.

The canonical EDA writes:

```text
outputs/eda/v0.5.0/<sector>/<eda_run_id>/
├── analysis_windows.parquet
├── metric_profile.parquet
├── series_profile.parquet
├── temporal_evidence.parquet
├── seasonality_evidence.parquet
├── dependence_evidence.parquet
├── figures/
└── eda_manifest.json
```

`02_CANONICAL_EDA.ipynb` first scans all canonical telemetry for structural
quality. Statistical exploration then uses a deterministic, label-blind
calibration window: the first 40% of each selected entity's observed time
range. This is not assumed to be normal data. The notebook profiles
distributions, missingness, rolling median and IQR, autocorrelation,
stationarity evidence, gated robust STL decomposition, and cross-metric
dependence. Its file-access guard records every input and proves that all reads
remain inside `SPEC-CORE`.

## What changes for a new sector

Create one new `01A_<SECTOR>_PACK.ipynb`. In that notebook:

1. locate and inspect the sector's native files;
2. map native metric names to standardized `metric_id`, entity type,
   measurement kind, and unit;
3. build the entity registry and only relationships present in the source;
4. write standardized wide observations to `PACK-CORE`;
5. route native labels to the appropriate optional `PACK-EVAL` table;
6. run the original-versus-redacted pack isolation test;
7. record facts the pack could not express without invention.

Do not change `01B_COMMON_CANONICAL_ADAPTER.ipynb` or add a sector branch to
`week1_core.py`. If a genuine source concept cannot fit the pack, document the
failure first and change the versioned interface deliberately.

## Next stage

After the canonical EDA evidence is reviewed, the intended order is:

```text
review frozen EDA evidence
    ↓
feature engineering
    ↓
baseline modelling and incident ranking
```

The previous EDA and modelling drafts remain available in Git history. They
were removed from the active folder because they target the earlier contract.
