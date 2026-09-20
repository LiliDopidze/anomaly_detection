# Run locally

The default experiment generates **96 ONTs × 90 days × five-minute samples**:
2,488,320 rows. No dataset download is needed. The generator emits 14 measurements,
separate ground truth and a simple topology table. The pipeline compares five cumulative telemetry feature sets, retaining
downstream Rx alone as the baseline.

## Local environment

1. Download/extract the current Branch 2 ZIP, or clone:

   ```bash
   git clone --branch codex/branch-2 --single-branch https://github.com/LiliDopidze/anomaly_detection.git
   cd anomaly_detection
   ```

   For a ZIP, open a terminal inside the extracted folder containing `pyproject.toml`.

2. Create an environment (Python 3.10+; tested with 3.12).

   Windows PowerShell:

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If activation is blocked, use `.\.venv\Scripts\python.exe` instead of `python`
   below; changing PowerShell execution policy is unnecessary.

   macOS/Linux:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install and check:

   ```bash
   python -m pip install -e ".[dev]"
   python -m pytest
   ```

4. Open `configs/config.yaml`. Defaults are 96 entities, 90 days, five-minute
   intervals and `output: outputs/optical_v8`. To run another experiment, use a
   fresh output folder, such as `outputs/optical_v8_run02`.

5. Start Jupyter with the installed environment:

   ```bash
   python -m ipykernel install --user --name optical-anomaly --display-name "Python (optical anomaly)"
   python -m notebook
   ```

   Select **Python (optical anomaly)** inside each notebook. Run 00–05 in order:

   | Notebook | Purpose |
   |---|---|
   | `00_generate_and_diagnose.ipynb` | Generate; inspect native data, missingness, structural checks and development faults |
   | `01_canonical_eda.ipynb` | Adapt and validate; inspect canonical distributions, correlations and seasonality |
   | `02_feature_distributions.ipynb` | Inspect all five cumulative feature sets and their coverage |
   | `03_detector_tuning.ipynb` | Compare five telemetry sets; fit/calibrate detectors and tune incident persistence |
   | `04_evaluation.ipynb` | Review validation errors; final assessment defaults off |
   | `05_end_to_end_demo.ipynb` | Verify saved-model replay; optional stress tests |

Without notebooks, the complete development run is:

```bash
python -c "from optical_anomaly.pipeline import develop; print(develop('configs/config.yaml'))"
```

Run this from the repository root. It refuses to overwrite a fitted model.

**Where local outputs go:** `<repository>/outputs/optical_v8/` by default.
For the current local checkout, that is:
`/Users/lilidopidze/Documents/Anomaly Detection/outputs/optical_v8/`.
On Windows or another machine, the prefix is wherever you cloned/extracted the repo.
An absolute `output` path in the YAML saves directly to that path instead.

## Output files

| File | Contents |
|---|---|
| `telemetry.parquet` | Time/device plus downstream/upstream Rx, ONT/OLT Tx, ONT/OLT temperature, two BER proxies and six FEC interval counts |
| `topology.parquet` | `entity_id`, `olt_id`, `pon_port_id`, `splitter_id`; static mapping only |
| `ground_truth.parquet` | Fault ID/type, entity, onset, observable onset, impact and repair/end |
| `generation_checks.json` | Structural and FEC consistency checks |
| `settings.json`, `manifest.json` | Settings, time splits and frozen fingerprints |
| `canonical_development.parquet` | Adapted and validated long telemetry; final period excluded |
| `development_features.parquet` | All 33 candidate features, excluding final-period observations |
| `feature_set_comparison.csv` | Best validation policy per feature set |
| `validation_scores.parquet` | Normalised anomaly scores for validation |
| `validation_comparison.csv` | Candidate incident policies and operational metrics |
| `validation_incidents.csv`, `validation_faults.csv` | Discrete incidents and matched/missed faults |
| `model.joblib` | Fitted feature references, detectors and chosen policy |

`test_metrics.json`, `test_incidents.csv`, `test_faults.csv` and `FINAL_OPENED.json`
are created only when you explicitly open final assessment. Generated files are
ignored by Git; cloning the repository does not download earlier results.

Topology is extra context only: no operational-event table, automatic localisation
or topology-based incident grouping is added. FEC fields are interval counts, not
cumulative counters. BER and receiver thresholds are modelling assumptions, not
validated vendor measurements. Synthetic checks do not certify field performance.

## Reruns and updates

Use a new run name/output folder after changing configuration or code. Matching
prepared data can be reused; completed models are not overwritten. Frozen old
models deliberately reject changed code at final assessment. Do not edit their
fingerprints to bypass this protection. Inspecting the same final data under a
new folder name does not make it independent evidence.

## Native-data qualification and source mappings

Run notebook 00 first for native-data diagnostics, then notebook 01 for canonical
adaptation and statistical EDA. Reports are saved under `<configured output>/eda/`.
The adapted and validated development data is saved once as
`canonical_development.parquet`; processing and replay work one ONT at a time.
No final-period rows appear in that file. The canonical EDA uses training data;
its seasonal holdout is inside training, not the model-validation or test period.

For company data, follow the explicit mapping example in README: both `units` and
`kinds` are required and keyed by canonical measurement name. Optional measurements
are omitted from the mapping, not silently ignored when an expected column is absent.
Gauge-only or temperature-only inputs can adapt successfully, but cannot run the
current detector without usable downstream Rx. Training also needs sufficient healthy
history. The five-set synthetic comparison requires the full mapping. Standalone Rx-only
company experiments can still use the original FeatureEngineer and detector classes.

For an existing run made before this multivariate change, set a fresh `output` in
`configs/config.yaml` before running notebooks 00–05.
Preserve old run directories; their frozen code fingerprints intentionally differ.
Old completed runs cannot be upgraded in place. The new default is `outputs/optical_v8`.

## Reading the comparison

Notebook 03 saves `validation_comparison.csv` (every candidate) and
`feature_set_comparison.csv` (best policy per telemetry set). Compare pre-impact
recall, missed faults, nuisance workload, delay and score coverage together.
Different feature sets may abstain on different rows; missing scores are not normal
scores. Opportunity denominators remain based on observed downstream Rx for every
candidate. The `feature_set` and exact feature columns are stored in `manifest.json`.

The expanded canonical table and five model comparisons take more disk space and
runtime than v7. Keep the standard run for development; use fewer ONTs for an
installation smoke test, clearly separated from performance experiments.
