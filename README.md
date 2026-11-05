# Time-series anomaly detection blueprint — SKAB

The active `time-series` branch contains **Notebook 01: data assessment**,
**Notebook 02: expected behaviour and time-series baselines**,
**Notebook 03: feature engineering and validation**, and
**Notebook 04: anomaly detection and development evaluation**. All use SKAB's physical
water-circulation testbed recordings. Notebook 04 compares frozen detectors and alert
policies; its results do not establish deployment readiness.

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
python tools/execute_assessment.py 02_expected_behaviour.ipynb
python tools/execute_assessment.py 03_feature_engineering.ipynb
python tools/execute_assessment.py 04_detection_and_validation.ipynb
```

The runner uses the active interpreter and saves the notebook only after successful execution.

## Notebook 01 contents

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

## Notebook 02: expected behaviour

Run `notebooks/02_expected_behaviour.ipynb` after installing the same dependencies; it can also
load its permitted source files independently. No additional dependencies are required.

The notebook checks the normal file chronologically, compares separate-normal and local median
references for all eight signals, and audits fixed 60/180-second calibrations. Native calibration
requires 80% nominal-second occupancy. Failed support remains unavailable; annotations never
choose or repair fitting windows. Thermal persistence uses completed five-second means with at
least four finite readings. Its issued forecast uses past values only; scoring eligibility is
confirmed afterward from target support and raw continuity. Unexpected gaps leave the forecast
in the audit output but invalidate its residual. All comparisons use common eligible observations.

**Source overlap discovered during review:** the normal file contains 572 rows identical to the
beginning of `other/5`, including all eight sensor values. Consequently, that whole recording is
excluded from independent external-reference transfer scores. Its local-calibration and persistence
comparisons remain valid and are retained. All 26 development experiments still have numerical
outputs; the eight reserved experiments remain unopened. The full normal reference is retained
as specified. This correction prevents a partially seen recording being reported as unseen transfer.

Observed results:

- Normal-file fluid temperature drifts: its forward empirical range coverage is 0.17% and 0% in
  the two chronological checks. An anomaly-free designation does not establish stationarity.
- Sixty-second calibration supports 25/26 recordings; 180 seconds supports 24/26. Longer calibration
  improves some references but is not uniformly better. No observed positive annotation enters either
  fitting duration; that retrospective audit does not prove healthy calibration.
- Local centres remove substantial cross-recording level mismatch. Thermal fixed ranges still
  have poor coverage; they must not be used directly as alert thresholds.
- Persistence reduces thermal MAE in all 25 available local-60 comparisons, but can follow a
  sustained departure. Both fixed-level and recent-history residuals are retained.
- Current shows approximately 80–95-second repetition in valve experiments. The permitted fitting
  windows provide too few cycles, and the normal-file pattern does not transfer. No seasonal model
  is fitted. A future longer-calibration experiment could assess one.
- A local vibration line is investigated only in two predeclared examples. Association and
  extrapolation vary; no universal relationship or detector superiority is claimed.

The executed notebook contains 32 code cells and five figures. Detailed calibration audits,
per-recording/annotation comparisons, fitted references, compressed residual tables, plots and
provenance are written to `outputs/skab_baselines/` (ignored by Git). The output fields distinguish
sample time from completed-window availability, forecast issuance from scoring eligibility, and
local calibration from independent external transfer. Residuals remain in original units.

At Notebook 02 delivery, Notebook 01 was executed again in memory without changing its tracked bytes. Its 52-test suite passed,
including hand-calculated references, boundary/future/annotation independence, support failures,
window timing, raw-gap resets, comparison exclusions and reserved-file protection.

## Notebook 03: feature engineering and validation

`notebooks/03_feature_engineering.ipynb` produces 19 unscaled candidate features:
8 level departures, 4 native-time slopes, 5 native-window IQRs and 2 thermal history errors.
Set A contains the eight level departures; Set B contains all 19. No detector, anomaly score,
threshold, scaler or label-based selection is fitted. The optional vibration relationship is
omitted because Notebook 02 did not establish a generally stable relationship.

Level departures subtract the median of **eligible five-second calibration means**, matching the
representation of the subsequent target. Calibration uses fixed half-open 60/180-second windows,
80% nominal-second occupancy and at least one eligible aggregate; counts remain visible for audit.
Primary supported calibrations contain 11–12 means, not 11–12 independent observations.
Levels begin on wholly post-calibration bins. Slopes and IQRs use native readings in `[t-30s,t)`;
slopes use actual elapsed minutes. Five-second means require four finite readings. All features
are timestamped at completed endpoints, with no interpolation or cross-recording history.

The existing baseline functions are reused unchanged. Additional feature-level checks reject
internal or left-edge gaps over two seconds and stale right edges, using no future reading.
The thermal history error also requires consecutive supported means and raw continuity. Valid zeros
remain distinct from unavailable values. Support rows include counts, occupancy, edge ages,
gap sizes and reasons; history rows describe target support plus separate previous-bin and
connection fields.

Observed development results on the common post-180-second population:

- Sixty-second calibration supports 25/26 recordings; 180 seconds supports 24/26. `other/13`
  fails both occupancy rules and `other/2` additionally fails at 180 seconds. Temporal features
  remain available where their own support permits, even without a calibrated median.
- Of 5,150 requested endpoints, Set A is complete at 4,885 and Set B at 4,864. Exported tables
  preserve all endpoints and individual feature availability; complete cases are not imposed.
- After the 247-second gap in `other/2`, illustrated level, history and 30-second features recover
  5, 10 and 30 seconds after the first returning reading, respectively.
- Current slope is sensitive to time scale: median within-recording rank correlations with the
  30-second version are 0.294 for 10 seconds and 0.417 for 60 seconds. This supports reporting
  sensitivity, not choosing a window by its response to labelled events.
- In `valve1/1`, median post-calibration temperature departure is −1.775 °C while median history
  error is +0.0037 °C. Recent stability can coexist with a sustained departure from the initial level.

The notebook includes five inspected figures, the 19-feature dictionary, per-recording summaries,
calibration/window sensitivity, within-recording block correlations and an eight-signal handover.
Generated values and availability tables (compressed CSV), detailed support, calibration metadata,
source identities, settings and checksums are written to `outputs/skab_features/`, which is ignored
by Git. No raw data or generated feature tables are committed. Run Notebook 03 independently after
installing the existing dependencies; it does not require generated outputs from Notebook 02.

Notebook 04 must fit scaling and data-dependent selection inside training partitions, split whole
recordings, handle the normal/`other/5` source overlap, and compare candidate sets on stated common
populations while reporting unavailable cases. Reliability here is not evidence of detection value.

Validation: all 67 tests pass. Notebook 03 executes end to end; Notebooks 01 and 02 also executed
again in memory without changing their tracked bytes. All eight reserved source files remain
unrequested by the loader. No baseline calculations or existing notebook behaviour changed.

## Notebook 04: anomaly detection and development evaluation

`notebooks/04_detection_and_validation.ipynb` rebuilds Notebook 03 features with resolution-independent
gap checks, then compares Set A (eight levels) and Set B (all 19 features). It requires scikit-learn,
now included in the project dependencies. No scaler, PCA or automatic feature selection is fitted.

Three predeclared whole-recording groups rotate separate detector-fitting, cutoff-calibration and
assessment roles. Candidate-normal fitting/calibration vectors are after elapsed 180 seconds and
before the first positive anomaly or changepoint annotation. This is annotation-assisted reference
training. Local 60/180-second sensor calibration remains fixed and label-free. Later label-zero
recovery intervals are not silently added to the normal training population. The overlapping separate
normal file is unused by the models. The eight reserved files remain unopened.

Methods and evaluation:

- Maximum absolute feature departure from its fitted median, divided by its fitted IQR.
  Medians and spreads use only the detector-fitting recordings. Scores grow beyond fitted extremes;
  they are not probabilities or Gaussian z-scores. Zero IQR (including nonconstant tied data) makes
  the component and complete-vector score unavailable, with no epsilon or fallback scale.
  All current fitted references have positive IQRs; their medians, spreads and support are exported.
- Isolation Forest with 200 trees, 256 samples per tree and seed 42; seeds 17/91 check stability.
  Its continuous score is calibrated externally rather than using its default binary prediction.
- Separate candidate-normal cutoff percentiles of 95% and 99%; strict exceedance. Immediate and
  three-consecutive-window alerts issue at actual five-second endpoints and reset on missingness.
- The primary comparison uses identical complete-vector rows for fitting, cutoff calibration,
  scoring and **alert history**. Own-availability outcomes are also preserved. Unavailable is never zero.
- Per-recording average precision, point precision/recall without point adjustment, fully observed
  episode detection/delay, label-negative alert burden, actual observed exposure and monitoring loss.
  Gap-censored positive pieces are not invented as complete incidents. A standing alarm cannot
  claim a later episode. Negative-label alerts are not proven false alarms.

Executed findings with primary 60-second calibration:

- Mean per-recording AP is **0.811 / 0.771** for unchanged Isolation Forest on Set A / B, and
  **0.856 / 0.847** for median/IQR departures. AP is available for 25/26 recordings; a constant-score
  reference averages 0.412. Expanded features do not improve mean ranking for either method.
- Common input availability is **4,864/5,150 (94.45%)**. `other/13` remains unmonitored; 180-second
  calibration additionally loses `other/2`. These cases remain in system-coverage reporting.
- At the 99th-percentile cutoff with three-window persistence, both Isolation Forest sets detect
  **7/20 fully observed episodes**, all in `other`; all 12 fully observed valve episodes are missed.
  There are also 12 censored positive pieces in six recordings, outside that episode denominator.
- Set A produces 10 label-negative alert starts over **3.923 observed, monitored label-negative hours**
  (2.55/hour); Set B produces 9 (2.29/hour). These short exposures cannot validate field alert burden.
- Median/IQR departures detect **6/20 / 8/20** complete episodes with the same 99th-percentile,
  three-window policy. Set B detects two valve1 episodes. Both sets produce **7 label-negative alert
  starts (1.78/hour)**, but spend **4.09% / 4.48%** of labelled-normal exposure in alert, versus
  **1.95% / 1.61%** for Isolation Forest. Fewer starts do not mean shorter alert duration.
- The new-alert-start episode metric is boundary-sensitive: in `other/5`, delaying a departure alert
  from 16:16:40 to 16:16:50 moves it inside the annotation starting at 16:16:48. The resulting event
  credit is not evidence that waiting improves early warning. Point coverage remains separately reported.
- The former bounded empirical-rarity score produced no alerts because its cutoffs reached its
  ceiling. It is replaced, not retuned by changing strict `>` to `>=`; the old result remains in Git
  history. Current cutoffs range from 11.34 to 21.66 IQRs, and reference transfer remains a limitation.
- On the 24 recordings and 4,689 pure-label bins shared by both calibration durations, level-only
  Isolation Forest AP changes from 0.807 to 0.851 with 180 seconds. The extra information and support
  loss are reported explicitly. No best seed, duration or cutoff is automatically selected.

The notebook contains five inspected figures and complete per-recording outputs under
`outputs/skab_detection/` (ignored): fitted-population/fold audits, local references, feature-support
reasons, fitted `departure_references.csv`, cutoffs, compressed scores/decisions and
`departure_components.csv.gz`, episode results, sensitivity and provenance.
There are no committed datasets, generated tables or model binaries. It runs independently of earlier
notebooks' output files. Notebooks 01–03 retain their tracked bytes. The only shared feature change computes gaps from
elapsed timedeltas rather than assuming nanosecond storage; native SKAB feature support and
Isolation Forest results were verified unchanged against the preceding execution.

**Readiness:** keep the level-only comparison and the transparent failure evidence. Investigate
normal-population and cutoff transfer before selecting a policy or opening the reserved test. These
results support neither a production detector nor cross-industry effectiveness.

Validation: **97 tests pass**, including median/IQR calculations, increasing scores beyond both
fitted extremes, zero-spread unavailability, frozen references, timestamp-resolution invariance,
gap recovery, causal alert timing, no point adjustment and observed exposure. Notebook 04 executes
end to end; all five figures were inspected. A file-access guard allowed only the 26 development
recordings and the separate normal file, and blocked any other source CSV. The normal file is
loaded but unused by the models. No reserved recording was requested or opened.

Interface change: `fit_departure_reference` and `departure_scores` replace `fit_rarity` and
`rarity_scores`; exported method names now use `departure`. No dependencies, data splits, feature
settings, Isolation Forest parameters or alert rules changed.

## Architecture

| File | Responsibility |
| --- | --- |
| `notebooks/01_data_assessment.ipynb` | Questions, orchestration, outputs, interpretation and handover |
| `notebooks/02_expected_behaviour.ipynb` | Fixed-reference, transfer and limited temporal experiments; interpretations and handover |
| `notebooks/03_feature_engineering.ipynb` | Feature definitions, diagnostics, sensitivity and unscaled handover |
| `notebooks/04_detection_and_validation.ipynb` | Whole-recording detector comparison, alert policies, honest evaluation and interpretation |
| `src/ts_anomaly/detection.py` | Generic median/IQR departures, cutoff calibration and causal consecutive-window alerts |
| `src/ts_anomaly/evaluation.py` | Generic label-window, point, episode and observed-exposure evaluation |
| `tests/test_detection.py` | Hand-calculated scores, alert timing, unavailable cases and event/exposure integrity |
| `src/ts_anomaly/features.py` | Generic trailing statistics, single-recording feature assembly and descriptive summaries |
| `tests/test_features.py` | Mathematical examples, availability timing, gap resets and source protection |
| `src/ts_anomaly/baselines.py` | Generic empirical references, support, completed windows, persistence and fair comparisons |
| `tests/test_baselines.py` | Statistical calculations, calibration and forecast information boundaries |
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
The loader neither downloads nor opens the reserved files. Keep this reservation throughout Notebooks 02–04 and subsequent development.

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
The reviewed notebook executes 17 code cells and embeds seven figures. The original 35 focused tests pass alongside the new baseline tests,
including occupancy, gap censoring, aggregation boundaries, annotation isolation and detrended ACF.

## Sources

- Katser & Kozitsin (2020), [Skoltech Anomaly Benchmark](https://doi.org/10.34740/KAGGLE/DSV/1693952).
- [Pinned SKAB repository](https://github.com/waico/SKAB/tree/b2c0d46c2971dcbfe71e26087b6d231998bb91c2):
  sensor definitions and source licensing.
- [Experiment descriptions](https://github.com/waico/SKAB/blob/b2c0d46c2971dcbfe71e26087b6d231998bb91c2/data/README.md).
- [statsmodels ACF documentation](https://www.statsmodels.org/v0.14.6/generated/statsmodels.tsa.stattools.acf.html).

Our split, gap threshold and aggregation support are explicit analytical choices, not official mandates.
