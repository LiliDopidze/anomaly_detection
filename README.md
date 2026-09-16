# Company-agnostic Telecom anomaly detection

This repository builds a Telecom telemetry detector for anomaly
detection, incident consolidation, and topology-aware localisation. The
primary product path is PON/ONT. 

## Evidence strategy

| Dataset | Role | Claim it can support |
|---|---|---|
| Synthetic PON/ONT fixture | Primary engineering and labelled development data | Software, leakage controls, injected-fault detection, and localisation mechanics |
| Commercial RAN PM counters | Open real-operator qualification | Portability, seasonality, heterogeneity, gaps, peer behaviour, and alert workload |
| Microsoft optical telemetry | Optional restricted research qualification | Real optical drift and alert stability; not recall |
| Optical failure testbed | Optional labelled testbed evaluation | Response to controlled physical failures; not production prevalence |

Raw datasets are never pooled into one fitted model. Each dataset has its own
adapter and label-free calibration; the detector, evidence calibration,
incident policy, and evaluation definitions remain shared.

## Project layout

```text
configs/                  Scientific and operational policy
notebooks/                Numbered, restart-and-run orchestration
src/telco_anomaly/        Tested reusable calculations
├── pipeline/             Partition and causal feature materialisation
├── scoring/              References, detectors, topology and thresholds
└── detectors.py          Backward-compatible public facade
tests/                    Contract, leakage, causality, and adapter tests
data/                     Git-ignored raw and generated data (optional local root)
```

The notebooks run in order:

```text
00 contract
01 source audit
02 canonical data
03 splits + truth lock
04 calibration-only EDA
05 leakage-safe features
06A optional calibration-sampling sensitivity
06 calibration-only statistical and Isolation Forest detectors
07 independent workload check + development selection
08 alerts, incidents + observable events
09 topology localisation
10 locked evaluation
11 public-data qualification
```


## Set up

```bash
git clone https://github.com/LiliDopidze/anomaly_detection.git
cd anomaly_detection
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m ipykernel install --user --name telco-anomaly --display-name "Python (telco-anomaly)"
pytest
```

Put data outside Git and select it with one environment variable:

```bash
export TELCO_DATA_ROOT="$HOME/telco_anomaly_data"
```

The same notebooks also work in Colab. They resolve
`/content/drive/MyDrive/telco_anomaly_data` first and support the existing
`/content/drive/MyDrive/anomaly_detection` folder as a legacy fallback.

Expected primary source layout:

```text
$TELCO_DATA_ROOT/
├── telco_synthetic_data/      # preferred; legacy telco_syntetic_data is recognised
│   ├── reference_dataset.parquet
│   ├── topology.csv
│   ├── entity_service_windows.csv
│   ├── gt_fault_registry.csv          # evaluation only
│   └── fault_entity_intervals.csv     # evaluation only
└── sources/
    ├── ran_pm/17815388/raw/
    ├── microsoft_optical/raw/
    └── optical_failure/raw/
```

Public sources do not need to be downloaded and uploaded by hand. In Notebook
11, choose exactly one source and enable its one-time acquisition:

```python
%env PUBLIC_DATASET=ran_pm
%env DOWNLOAD_PUBLIC_DATA=1
```

The notebook downloads from the publisher link, verifies publisher checksums
when available, extracts the files under `TELCO_DATA_ROOT`, and writes a source
manifest. Microsoft Optical additionally requires
`ACKNOWLEDGE_MICROSOFT_DATA_TERMS=1`; the optical-failure testbed requires
`ACKNOWLEDGE_OPTICAL_FAILURE_TERMS=1`. These acknowledgements confirm that the
user reviewed the applicable terms; they do not grant additional rights.

Generated outputs are immutable, stage-named directories beneath
`$TELCO_DATA_ROOT` (`audits/`, `prepared/`, `core/`, `eda/`, `features/`,
`models/`, `selection/`, `incidents/`, `localisation/`, and `results/`). Change
a run ID to create another run; completed runs are never overwritten.

GitHub stores code, configuration, tests, documentation, and optionally a few
curated small reports. Raw data, canonical telemetry, feature tables, fitted
artefacts, and full run outputs remain outside Git because they are large and
may be restricted. Their manifests and hashes provide reproducibility without
committing the data itself.

For exact run instructions and the purpose of every notebook, see
[notebooks/README.md](notebooks/README.md).

## Scientific guardrails

- Models read `SPEC-CORE` only. Faults, tickets, and labels live in
  `SPEC-EVAL` and are mounted only by the evaluator.
- Calibration, development, and holdout are chronological. Development labels
  may compare declared candidates; locked holdout cannot change the model.
- Global model rows are selected deterministically across time-of-day strata.
  The optional 06A diagnostic compares sampling density and seed stability
  using score rankings and label-free incident workload.
- Score thresholds use one late-calibration half that was not used to fit the
  detector. The other half verifies consolidated workload without labels.
  Empirical tail thresholds also require minimum block support. Only then are
  frozen operating points compared on development truth; holdout remains
  sealed until one configuration passes every gate.
- Detection credit requires an actually alerting entity to overlap an affected
  entity; a predicted topology footprint is localisation evidence only.
- False-incident workload uses scoreable portfolio exposure and reports the
  calendar denominator beside it.
- Detection selection fails closed unless the 95% Wilson lower recall bound,
  Garwood workload upper bound, sample-size, and availability gates all pass.
- Localisation reports the smallest supported observable scope and preserves
  topology ambiguity. It is qualified separately from detection and remains a
  probable location, not causal root cause.
- Synthetic results do not establish real-fleet effectiveness. Operator data
  is required before production claims.


