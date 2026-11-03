# Time-series anomaly detection blueprint

Start with **[01_data_assessment.ipynb](notebooks/01_data_assessment.ipynb)** on the
`time-series` branch. It assesses the **2020 UCI MetroPT-3** recording before any
expected-behaviour model or detector is fitted. It replaces the earlier NAB pilot;
that work remains on `codex/nab-blueprint` and in Git history. `main` is unchanged.

## Run in WSL / VS Code

Use Python 3.11 or newer (the clean execution was verified with Python 3.13).
From your WSL terminal, clone into a new directory so an existing checkout is safe:

```bash
git clone --branch time-series --single-branch https://github.com/LiliDopidze/anomaly_detection.git ~/projects/anomaly_detection_metropt
cd ~/projects/anomaly_detection_metropt
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m ipykernel install --user --name time-series --display-name "Time-series assessment"
code .
```

Open Notebook 01 and select **Time-series assessment**. Run cells in order. The
first load downloads the official archive (218 MB); allow several minutes.
The CSV and supplied PDF are cached under ignored `data/metropt/` and their SHA-256
hashes are verified on every run. Cached runs do not need network access.
Budget roughly 1 GB of free disk and a few GB of RAM for a comfortable notebook run.
On Python.org macOS, complete its certificate-installation step if HTTPS reports a
certificate error; do not disable verification.

```bash
python -m pytest
python tools/execute_assessment.py
```

The runner executes with the calling Python interpreter and saves notebook outputs
only after success. Jupyter requires local kernel communication. Assessment CSVs
and a source/configuration/version manifest are written to ignored
`outputs/metropt_assessment/`. Compact figures and tables remain in the committed
notebook; raw data, archives, caches and generated CSVs are not committed.

Verified environment: Python 3.13.2, NumPy 2.3.5, pandas 2.3.3, Matplotlib 3.11.2,
statsmodels 0.14.6 and nbclient 0.11.0. Dependency ranges in `pyproject.toml` keep
the tested NumPy/pandas/statsmodels compatibility boundaries. The executed notebook
also displays its versions; each local run writes them to its manifest.

## What Notebook 01 establishes

1. The meaning, units, representation and analytical role of every signal.
2. Timestamp order, duplicates, irregular intervals and collection gaps.
3. Coverage under an explicitly assumed continuous collection schedule.
4. Continuous distributions and digital values, transitions and run durations.
5. Dependence on contiguous minute aggregates, aggregation sensitivity, changes
   across development and exploratory hour/weekday summaries.
6. Valve-command context and relationships among pressure, current and temperature.
7. A measurement-by-measurement handover to Notebook 02.

The source has seven analogue and eight digital channels. Digital source names
are not silently corrected (for example, `DV_eletric`). Pulse encoding is not
inferred to be a cumulative counter. Negative near-zero pressures remain recorded
values, not automatically invalid measurements.

## Scientific boundaries

| Period (recorded time; timezone unknown) | Use |
| --- | --- |
| 2020-02-01 inclusive to 2020-03-01 exclusive | Reference / later model fitting |
| 2020-03-01 inclusive to 2020-06-01 exclusive | Development / later calibration |
| 2020-06-01 inclusive to 2020-09-01 04:00 exclusive | Reserved final evaluation |

The source actually ends at 2020-09-01 03:59:50. The last four hours are retained
for the availability audit rather than silently discarded or reinterpreted as a
timezone offset. The loader returns **no sensor values from the reserved period**;
its full-timeline output contains only timestamps, entity/source identifiers and
missing-cell counts. The final period has not been used for distributions,
correlations, operating-state inspection or model selection.

A 15-second continuity threshold accommodates approximately 10-second native
sampling. State runs stop at larger intervals and missing values. A run's observed
span and upper duration bound assume no hidden switches between samples; they are
not exact physical dwell times. Boundary/gap-censored runs are identified, and
excluded from complete-run duration summaries.

Continuous diagnostics use one-minute observation means with at least five finite
samples per bin; a 4/5/6-sample sensitivity check shows the effect of this choice.
This is diagnostic aggregation, not a finished causal feature pipeline. No bins
are interpolated, and digital signals remain at native timestamps. ACF uses the
longest uninterrupted valid segment, no more than one quarter of its length in
lags, and no IID significance bands. Hourly profiles do not prove seasonality.

Valve states use COMP and DV_eletric only. Off and unloaded operation remain pooled;
current does not define the states used to compare current. LPS is an existing
alarm comparison, not an independent failure label. Maintenance incidents are
separate from observations, with source errors preserved rather than guessed away.

## Files and reuse

| File | Responsibility |
| --- | --- |
| `notebooks/01_data_assessment.ipynb` | Analysis, figures, actual findings and handover |
| `src/ts_anomaly/metropt.py` | Source reader, sensor dictionary, valve interpretation and reports |
| `src/ts_anomaly/contracts.py` | Small wide-table contract and measurement definitions |
| `src/ts_anomaly/assessment.py` | Generic sampling, coverage, distributions, runs, ACF and relationships |
| `configs/metropt.json` | Source hashes, boundaries and explicit diagnostic assumptions |
| `tests/test_assessment.py` | Hand-checkable statistical and integrity examples |
| `tools/execute_assessment.py` | Execute and save Notebook 01 with the active interpreter |

For another dataset, supply a reader, metric definitions, operating context and
boundaries. Use the generic assessment functions on those inputs; no MetroPT names
appear in them. Functions needing one entity explicitly say so or check it. The
wide contract suits aligned sensors; asynchronous sources need explicit alignment
before using joint diagnostics. Gauge, state and unresolved pulse representations
are supported here. Additional count/counter semantics require source-specific
verification rather than automatic conversion. There is no registry, plugin system
or automatic model selector.

The prior four-column NAB contract is intentionally replaced on this development
branch by aligned wide measurements. No downstream notebooks exist on this branch.
The earlier package version is preserved in history; this is not a drop-in change
for the NAB loader.

## Evidence and limits

[UCI MetroPT-3, Davari et al., DOI 10.24432/C5VW3R](https://doi.org/10.24432/C5VW3R)
is released under **CC BY 4.0**. Definitions and incident records are paraphrased
from that source and its accompanying PDF; the exported CSV remains unchanged.
This is the 2020 recording, not the 2022 MetroPT release. The source PDF and web
metadata differ in row count and sampling-rate descriptions; the assessment uses
the pinned file and actual timestamps.

Method references: [statsmodels ACF](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html),
[scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage),
and [pandas datetime semantics](https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html).

This dataset supports development of an analytical blueprint, not a claim of
cross-company performance. Unknown timezone, oil-level polarity, pulse encoding,
sensor calibration, incomplete incident labels and one inconsistent maintenance
date remain limitations. There are only four reported incidents from one asset.
See the executed notebook for measured findings and the proposed next step:
interpretable expected-behaviour baselines, with and without valve-command context.
