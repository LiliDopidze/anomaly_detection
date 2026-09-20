# Run locally or in Google Colab

The default experiment generates **96 ONTs × 90 days × five-minute samples**:
2,488,320 rows. No dataset download is needed. The generator emits 14 measurements,
separate ground truth and a simple topology table. The current detector remains
the downstream-Rx baseline; added signals are available for EDA.

## Google Colab: the simplest route

1. Open `notebooks/00_colab_start.ipynb` from Branch 2 in Colab. You can use Colab's
   **File → Open notebook → GitHub**, paste the repository URL below, select
   `codex/branch-2`, and choose that notebook. Alternatively upload the downloaded
   `.ipynb` file to Colab.

   Repository: https://github.com/LiliDopidze/anomaly_detection

2. Use a **CPU runtime**. GPU hardware is not used by this pipeline.

3. Run the first code cell. It clones Branch 2 into `/content/anomaly_detection`
   and installs the project. It reuses an existing checkout rather than silently
   overwriting it; start a fresh runtime to obtain newer code cleanly.

4. In the output-location cell, set:

   ```python
   SAVE_TO_DRIVE = True
   RUN_NAME = "optical_v7_run01"
   SMALL_SMOKE_RUN = False
   ```

   Approve the Drive mount when Colab asks. Set `SMALL_SMOKE_RUN=True` only for a
   quick installation check (8 ONTs, 14 days). Leave it False for the full dataset.

5. Run the remaining cells in order. They generate/check data, fit the model,
   evaluate validation candidates, show training examples and list output files.
   The full run takes several minutes; actual time depends on runtime resources.

6. Keep `OPEN_FINAL_TEST=False` while developing. Only enable it once the model
   and settings are fixed and you deliberately want to inspect final performance.

**Where Colab outputs go:**

| Choice | Exact output folder |
|---|---|
| `SAVE_TO_DRIVE=True` | `/content/drive/MyDrive/anomaly_detection/optical_v7_run01/` |
| `SAVE_TO_DRIVE=False` | `/content/anomaly_detection/outputs/optical_v7_run01/` |
| Smoke run | Same location, with `_smoke` appended to the run name |

With Drive enabled, open Google Drive → My Drive → anomaly_detection → your run
folder to find the files. The configuration used by the notebook is temporarily
written to `/content/optical_run.yaml`; a copy of its settings is saved permanently
in the run folder as `settings.json`.

Files under `/content` can disappear when Colab recycles the runtime. Drive-mounted
files persist independently of that runtime. Colab resources are not guaranteed:
see [Google's Colab FAQ](https://research.google.com/colaboratory/faq.html).
If memory is constrained, reduce entities/days explicitly; do not silently
interpret a smoke run as the full experiment.

To download a run without Drive, execute this additional Colab cell:

```python
import shutil
from google.colab import files

archive = shutil.make_archive("/content/optical_results", "zip", root_dir=RUN)
files.download(archive)
```

The Colab starter is an end-to-end route. Notebooks 01–05 provide the more detailed
local workflow below. No Drive connection or authorisation is needed locally.

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
   | `01_generator_eda.ipynb` | Generate data; inspect topology, powers, temperatures, FEC ratios, missingness and seasonality |
   | `02_feature_distributions.ipynb` | Inspect the baseline detector's causal Rx features and coverage |
   | `03_detector_tuning.ipynb` | Fit/calibrate detectors and tune incident persistence |
   | `04_evaluation.ipynb` | Review validation errors; final assessment defaults off |
   | `05_end_to_end_demo.ipynb` | Verify saved-model replay; optional stress tests |

   **Do not run notebook 00 locally**: its setup is specifically for Colab.

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

## Output files in either environment

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
