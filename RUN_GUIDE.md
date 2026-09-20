# Run locally

The default experiment generates **96 ONTs × 90 days × five-minute samples**:
2,488,320 rows. No dataset download is needed. The generator emits 14 measurements,
separate ground truth and a simple topology table. The current detector remains
the downstream-Rx baseline; added signals are available for EDA.

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
   intervals and `output: outputs/optical_v7`. To run another experiment, use a
   fresh output folder, such as `outputs/optical_v7_run02`.

5. Start Jupyter with the installed environment:

   ```bash
   python -m ipykernel install --user --name optical-anomaly --display-name "Python (optical anomaly)"
   python -m notebook
   ```

   Select **Python (optical anomaly)** inside each notebook. Run 01–05 in order:

   | Notebook | Purpose |
   |---|---|
   | `01_generator_eda.ipynb` | Inspect native data and qualification checks; review development faults; then preview canonical adaptation |
   | `02_feature_distributions.ipynb` | Inspect the baseline detector's causal Rx features and coverage |
   | `03_detector_tuning.ipynb` | Fit/calibrate detectors and tune incident persistence |
   | `04_evaluation.ipynb` | Review validation errors; final assessment defaults off |
   | `05_end_to_end_demo.ipynb` | Verify saved-model replay; optional stress tests |

Without notebooks, the complete development run is:

```bash
python -c "from optical_anomaly.pipeline import develop; print(develop('configs/config.yaml'))"
```

Run this from the repository root. It refuses to overwrite a fitted model.

**Where local outputs go:** `<repository>/outputs/optical_v7/` by default.
For the current local checkout, that is:
`/Users/lilidopidze/Documents/Anomaly Detection/outputs/optical_v7/`.
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
| `development_features.parquet` | Baseline Rx features, excluding final-period observations |
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

Run notebook 01 before feature engineering. It writes reports to
`<configured output>/eda/`: measurement dictionary, development missingness and
distributions, healthy statistical diagnostics, development fault summaries and
contrasts, structural checks, and a small canonical preview. It does not write a
second full copy of the long-format dataset.

For company data, follow the explicit mapping example in README: both `units` and
`kinds` are required and keyed by canonical measurement name. Optional measurements
are omitted from the mapping, not silently ignored when an expected column is absent.
Gauge-only or temperature-only inputs can adapt successfully, but cannot run the
current detector without usable downstream Rx. Training also needs sufficient healthy
history. The synthetic convenience pipeline uses `synthetic_adapter(["rx_dbm"])`.

For an existing run made before this adapter change, set a fresh `output` in
`configs/config.yaml` before running notebooks 02–05.
Preserve old run directories; their frozen code fingerprints intentionally differ.
Notebook 01 may inspect the existing generated data without retraining its model.
