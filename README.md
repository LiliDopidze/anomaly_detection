# Time-series anomaly detection blueprint — SKAB

The active `time-series` branch contains **Notebook 01: data assessment and exploratory analysis**.
It uses SKAB's physical water-circulation testbed recordings. No detector or alert policy is fitted yet.

MetroPT-specific code, configuration and the previous modelling notebook have been removed from the
active tree. That work remains in Git history at `d8ffd5aaacb5c71a554dd4636e5b73dcef41adb2`.
Generic measurement contracts and assessment functions are retained.

## Run in WSL

Use a new directory to keep an existing checkout unchanged:

```bash
cd ~/projects
git clone --branch time-series --single-branch https://github.com/LiliDopidze/anomaly_detection.git anomaly_detection_skab
cd anomaly_detection_skab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
code .
```

Python 3.11 or newer is required. In VS Code, open `notebooks/01_data_assessment.ipynb`,
select this project's `.venv` Python interpreter, and run the cells in order.
Install the VS Code Python and Jupyter extensions if needed.

The loading cell downloads only development/reference CSVs from the official repository and verifies
their pinned Git blob identities. Later runs validate the cache. Initial download requires internet
access and working system TLS certificates, but no Kaggle credentials.

To execute without the editor:

```bash
python -m pytest
python tools/execute_assessment.py
```

The runner uses the active interpreter and saves the notebook only after successful execution.

## Notebook contents

1. Original source folders, file counts and plain explanations of the testbed and all eight signals,
   before introducing our project-specific partitions.
2. Measurement and annotation examples, with the population of each analysis stated explicitly.
3. Sampling, gaps, occupied seconds and measurement quality, including their different denominators.
4. Annotation timeline, initial support and incomplete event boundaries caused by gaps or file edges.
5. Native time plots and distributions for a declared example, `valve1/1`.
6. Initial medians and within-recording interquartile ranges, compared with the separate normal file.
7. Gap-aware autocorrelation, aggregation-support sensitivity and a linear-trend sensitivity check
   across all 26 development recordings.
8. Sensor relationships with exact comparison periods and a measurement-by-measurement handover.

Figures and tables are embedded in the executed notebook. Small diagnostic tables and a provenance
manifest go to `outputs/skab_assessment/`; source files are cached under `data/skab/`.
Both directories are Git-ignored. No raw datasets are committed.

## Architecture

| File | Responsibility |
| --- | --- |
| `notebooks/01_data_assessment.ipynb` | Questions, orchestration, outputs, interpretation and handover |
| `src/ts_anomaly/contracts.py` | Generic measurement definitions and table validation |
| `src/ts_anomaly/assessment.py` | Generic sampling, distributions, runs, aggregation, ACF and relationships |
| `src/ts_anomaly/skab.py` | SKAB meanings, units, source loading and separate annotations |
| `configs/skab.json` | Pinned source, file identities, partitions and timing choices |
| `tests/test_assessment.py` | Hand-checkable statistical and temporal calculations |
| `tests/test_skab.py` | Source contracts, annotation isolation and reserved-file protection |
| `tools/execute_assessment.py` | Notebook execution using the selected Python interpreter |

Generic methods contain no SKAB sensor names. Here, `entity_id` is an analytical recording key,
not a separate physical machine; notebook tables display it as `recording_id`.
For another dataset, supply its definitions and reader and reuse
the assessment functions where their assumptions hold. No plugin framework is needed.

## Boundaries

The source has 34 labelled experiments and one separate normal recording. Eight complete experiments
are reserved before value exploration:

- `valve1/3`, `valve1/7`, `valve1/11`, `valve1/15`
- `valve2/3`
- `other/4`, `other/8`, `other/12`

These are every fourth file in numeric filename order within each family. The actual valve files
start at zero, unlike the example numbering in the source data README.
The loader neither downloads nor opens the reserved files. Keep this reservation in Notebook 02.

This is our experiment holdout, not a chronological future test or the official SKAB protocol.
Development annotations describe initial prefixes and intervention responses retrospectively.
Future test labels must not select training windows. Later evaluation must distinguish transfer
of a fixed model from a model allowed a predeclared initial calibration period.

Annotations are not predictors. Multiple changepoint markers can describe one intervention; gaps
can split one positive-label run. Neither automatically indicates a new failure.
The separate normal file has no point labels, and the reader does not invent them.

## Verified findings and limits

The opened reference and development recordings contain 37,892 rows across 27 files:

- No duplicate timestamps within recordings, backwards steps, missing measurement cells or non-finite values.
- 35 intervals longer than two seconds; the largest is 247 seconds.
  Two-second intervals reduce one-second occupancy without exceeding the gap threshold.
  Occupancy describes the observed file span, not equipment uptime.
- Initial annotated-development prefixes contain 104–577 observations.
  A prefix ends before either annotation first becomes positive; later label-zero readings are excluded.
  Gap-separated positive pieces are not necessarily distinct incidents.
- Initial flow medians span 32–128 L/min; the separate normal file's median is 125.677 L/min.
  These reference populations are not automatically interchangeable.
- In the declared `valve1/1` prefix, the longest uninterrupted one-second segment has only 24 samples.
  Aggregating the whole prefix supplies 119 contiguous five-second means with at least four readings
  per bin. Their timestamps denote completed-window availability; no values are interpolated.
- Across 26 development recordings, median lag-five-second ACF is 0.953 for motor temperature
  and 0.954 for fluid temperature. After removing a straight-line trend within each selected segment,
  these medians are 0.909 and 0.940. Simple linear drift does not fully explain this persistence;
  neither this check nor high ACF establishes stationarity or predictive performance.

These are descriptive findings, not detector performance. One testbed and short experiments cannot
validate cross-industry generalisation, daily/weekly seasonality, long-term drift or field alert burden.

Execution was checked with Python 3.13.2, NumPy 2.3.5, pandas 2.3.3, Matplotlib 3.11.2,
statsmodels 0.14.6 and nbclient 0.11.0. The run manifest records versions and source identities.
Dependency ranges in `pyproject.toml` are compatibility ranges, not a fully locked environment.
The reviewed notebook executes 17 code cells and embeds seven figures. All 35 focused tests pass,
including occupancy, gap censoring, aggregation boundaries, annotation isolation and detrended ACF.

## Sources

- Katser & Kozitsin (2020), [Skoltech Anomaly Benchmark](https://doi.org/10.34740/KAGGLE/DSV/1693952).
- [Pinned SKAB repository](https://github.com/waico/SKAB/tree/b2c0d46c2971dcbfe71e26087b6d231998bb91c2):
  sensor definitions and source licensing.
- [Experiment descriptions](https://github.com/waico/SKAB/blob/b2c0d46c2971dcbfe71e26087b6d231998bb91c2/data/README.md).
- [statsmodels ACF documentation](https://www.statsmodels.org/v0.14.6/generated/statsmodels.tsa.stattools.acf.html).

Our split, gap threshold and aggregation support are explicit analytical choices, not official mandates.
