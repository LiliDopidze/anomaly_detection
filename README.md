# Telecom anomaly detection

Detect abnormal PON/ONT telemetry, consolidate alerts into incidents, and
estimate the affected network scope. Models are calibrated separately for
each dataset or operator. Synthetic results validate injected mechanisms;
they do not establish production performance.

## Start here

- **Run the pipeline:** follow the [notebook guide](notebooks/README.md).
- **Change settings:** edit the YAML files in `configs/` and use new run IDs.
- **Change calculations:** edit `src/telco_anomaly/`, then run `pytest`.

## Repository layout

| Folder | Purpose |
|---|---|
| `notebooks/` | Ordered workflow, results inspection and optional experiments |
| `configs/` | Dataset mappings, feature settings, model and alert policies |
| `src/telco_anomaly/` | Reusable implementation imported by notebooks |
| `tests/` | Checks for leakage, causality, scoring and evaluation correctness |

Keep calculations in `src`, not copied into notebooks. Keep tests with the
implementation: a change to time windows or event matching can silently
change the reported performance.

## Local setup

```bash
git clone https://github.com/LiliDopidze/anomaly_detection.git
cd anomaly_detection
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
export TELCO_DATA_ROOT="$HOME/telco_anomaly_data"
pytest
jupyter lab
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1`.
In Colab, use a repository checkout with its dependencies installed. The
notebooks mount Drive and recognise `MyDrive/telco_anomaly_data` and the
existing `MyDrive/anomaly_detection` data folder. Set `TELCO_PROJECT_ROOT`
if the repository is separate from the notebook working directory.

## Data and outputs

Keep data and generated artifacts outside Git. The primary source directory is:

```text
$TELCO_DATA_ROOT/telco_synthetic_data/
    reference_dataset.parquet
    topology.csv
    entity_service_windows.csv
    gt_fault_registry.csv
    fault_entity_intervals.csv
```

The legacy spelling `telco_syntetic_data` is also supported. Ground truth is
reserved for evaluation; detectors do not use fault labels. Generated stage
outputs are immutable and stay under `TELCO_DATA_ROOT`, with hashes linking
their inputs. Change a run ID to create a new result rather than overwrite one.

## Evaluation rules

Calibration, development and holdout are chronological. Thresholds and
workload verification use separate calibration slices. Development compares
eligible candidates; holdout stays sealed until the frozen configuration is
qualified. Report timely detection, incident workload and localisation
separately. Shared-network localisation is not causal root-cause proof.

Public Telecom adapters are optional. They qualify portability or controlled
failures according to their source; they do not replace real PON validation.
Never pool incompatible datasets into a single fitted model.
