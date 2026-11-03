# Statistical foundation: notebooks 00 and 01

## Decision record (research before implementation)

This experiment tests software and detection assumptions. It is not a calibrated
GPON digital twin, a failure-probability model, or evidence of field performance.
The original main revision reviewed was `3333e789590ded14dd313ef42ffd32fd073b68a7`.

| Finding | Decision and alternative | Limitation |
|---|---|---|
| Missing numerator intervals could be divided by complete denominators. | Require complete, aligned source intervals when summing counts; align declared count groups before aggregation. Partial sums would require explicit exposure and a different contract. | Conservative rejection loses usable partial exposure; it never invents a full-interval rate. |
| A registry called canonical contained only optical metrics. | Keep a small generic metric definition and allow caller-supplied registries; put PON definitions in a domain module. | Source semantics, units, topology and actionable anomalies still need domain configuration. |
| Row-based coverage conceals absent polls and missing boundary rows. | Assess coverage against a declared active schedule, including absent entities/metrics. Preserve gaps in lag calculations. | A wrong schedule produces a wrong denominator; irregular event streams need a different observation model. |
| Every entity had faults with narrowly fixed onset windows and abrupt recovery. | Add negative controls, broader randomized onset, separate random streams, gradual recovery and shared-port scenarios. Retain separated development/test windows as an explicit experiment design. | These are controlled interventions, not estimated incident prevalence or repair laws. Overlapping causes are excluded. |
| One stationary daily model cannot establish robustness. | Retain the interpretable AR(1) reference and add separate weekly, drift, variance and collection-gap stress scenarios. Compare across seeds. | More simulated complexity is not more realism without field calibration. |
| Large injected faults can dominate an apparently successful experiment. | Compare predeclared fault strengths with identical timing/noise streams; retain events without impact. | Oracle visibility and impact counts are not detection performance. |
| Optical/FEC relationships were almost deterministic in places. | Preserve physical power accounting and conditional count conservation; disclose the assumed receiver curve and shared paths. | Ideal independent-bit coding is not burst-error or packet-loss physics. |
| One fit and a plot cannot justify a seasonal baseline. | Compare constant, daily and daily+weekly predictions on a chronological inner holdout of training gauges. Show remaining dependence and pair counts. | This is diagnostic model comparison, not a calibrated predictive interval or a stationarity test. |

## Mathematical assumptions

For the reference path-loss residual, `u[t] = phi*u[t-1] + epsilon[t]`,
`phi = exp(-delta/tau)` and `Var(epsilon) = sigma²(1-phi²)`.
The initial state has variance `sigma²`, avoiding a quiet warm-up artefact.
Independent readout noise of variance `s²` gives observed residual variance
`sigma²+s²` and adjacent correlation `phi*sigma²/(sigma²+s²)`.
These identities apply to the stationary reference, not to stress scenarios.
Fixed numerical diagnostic bands are engineering screens, not confidence intervals.

Power accounting uses `Rx[dBm] = Tx[dBm] - loss[dB]`. The loss distribution,
temperature coupling, seasonal amplitudes, receiver offsets and fault signatures
are simulation choices. No cited standard establishes those distributions.
An arithmetic mean of dBm samples is a mean log-power reading, not mean physical
power; convert to linear units first if physical energy averaging is required.

FEC counts describe the preceding interval `(t-delta, t]`, using the simulated
state at its start. The first interval is unknown. Counts are conditional
multinomial outcomes (implemented through binomial draws), so corrected plus
uncorrectable codewords cannot exceed received codewords. The simplified
RS(255,239) model ignores burst errors, shortening, protocol overhead and
direction-specific FEC enablement. The assumed power-to-BER curve is uncalibrated.
Lognormal receiver jitter adds conditional-rate variation; a fitted beta-binomial
or negative-binomial model is an alternative when real count dispersion supports it.
Do not add a zero-inflation component merely because rare events produce many zeros.

## Labels and separation

* `onset_time`: first affected latent sample; it need not be detectable.
* `observable_onset_time`: first collected sample with counterfactual mean effect
  at least twice the configured physical SD. This is an oracle visibility proxy,
  **not** an estimated detection time or statistical-power guarantee. It is undefined
  for variance-only faults. A visible effect can still be obscured by readout noise.
* `impact_time`: confirming sample of a sustained latent optical threshold crossing;
  it is never backdated and is not customer-service failure ground truth.
* `recovery_start_time` to exclusive `end_time`: simulated repair transition.
* `cause_id`: shared physical intervention; entity-level `fault_id` remains unique.

Ground truth stays in a separate table. Sensor/collection changes cannot move
physical impact. The first 55% is deliberately fault-free. This is a supervised
experimental convenience, not a defensible assumption for an unreviewed real feed.
Notebooks use training data for baseline decisions and complete development events
for fault diagnostics. They do not select a detector or tune against the final segment.

## Architecture and boundaries

1. `contracts.py`, `adapter.py`, `validation.py`, `assessment.py`: explicit metric
   definitions, source mapping, schedule and exposure checks. The old PON default
   remains for compatibility; pass definitions to use another domain.
2. `pon.py`, `sources.py`, `generator.py`, `optics.py`: optical measurements,
   topology, physical relationships and simulation assumptions.
3. Notebooks, configs, `diagnostics.py`, generation manifests and tests: controlled
   experiments, descriptive analyses and reproducibility. Optical diagnostics are
   deliberately separate from generic coverage checks.

Only gauges and interval counts are implemented. Cumulative counters require
reset/wrap-aware differencing with known measurement intervals upstream. Fractions
need numerator/denominator exposure; event streams are not automatically resampled
as gauges. No universal stationarity test, automatic imputation, plugin hierarchy,
adaptive baseline, or deep model is introduced.
The synthetic poll cadence must divide a UTC day and provide at least 150 samples;
this makes its calendar-origin experiment and separated event windows explicit.
The generic validator independently supports other fixed cadences using UTC epoch
alignment; it does not silently interpret irregular interval counts.

Before field modelling, obtain source semantics and a reviewed active schedule,
audit real telemetry, agree an alert budget/actionable duration, and evaluate
time-blocked and held-out-entity data. Synthetic tests cannot validate cross-company
transfer, failure prediction, missing-not-at-random outages, burst errors, maintenance
context, overlapping causes, or the relation between optical impact and service loss.
The later evaluation code also needs scheduled exposure and shared-cause accounting
before operational false-alert/recall claims. That redesign is outside 00/01 scope.

## Sources and their evidential role

* **[ar], [harmonic], [leakage] Statistical textbook:** Hyndman & Athanasopoulos,
  [Forecasting: Principles and Practice](https://otexts.com/fpp3/), especially
  [time-series cross-validation](https://otexts.com/fpp3/tscv.html) and
  [dynamic harmonic regression](https://otexts.com/fpp3/dhr.html). These support
  chronological comparisons and harmonic baselines, not a mandatory daily PON cycle.
* **[etsi] Engineering specification:**
  [ETSI GS F5G 011, §§8.3–8.4](https://www.etsi.org/deliver/etsi_gs/F5G/001_099/011/01.01.01_60/gs_F5G011v010101p.pdf).
  It defines telemetry. Its upstream FEC total-block count can mean errored blocks,
  whereas this simulator's total is all received codewords: mappings must verify this.
* **[db] Engineering standard:** [ITU-T G.984.2](https://www.itu.int/rec/T-REC-G.984.2-201908-P/en)
  specifies optical interfaces/budgets; [G.984.3](https://www.itu.int/rec/T-REC-G.984.3-201401-I)
  specifies transmission convergence including FEC. Neither calibrates this simulator.
* **Workshop paper, vendor-authored:** Toledano et al. (2018),
  [Anodot system design](https://proceedings.mlr.press/v71/toledano18a.html).
  Signal-dependent treatment is useful; commercial automation claims are not proof
  that arbitrary metrics share a statistical process.
* **Framework technical report:** [Merlion](https://arxiv.org/abs/2109.09265) and
  [official documentation](https://opensource.salesforce.com/Merlion/).
  Useful separation of preprocessing, scores and postprocessing; its defaults are
  not substitutes for measurement semantics.
* **Vendor documentation:** [MathWorks time-series anomaly detection](https://uk.mathworks.com/products/time-series-anomaly-detection.html).
  Useful tooling and synthetic anomaly workflows, not independent realism evidence.
* **Benchmark research:** Wu & Keogh,
  [Current Time Series Anomaly Detection Benchmarks are Flawed and are Creating the Illusion of Progress (extended abstract)](https://wu.renjie.im/research/anomaly-benchmarks-are-flawed/ICDE2022_TSAD.pdf).
  Trivial patterns and unrealistic anomaly density motivate controls and stress tests.
* **Scientific software documentation:** [statsmodels ACF](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html)
  documents that dropping missing values can treat remaining samples as contiguous.
  Here lag pairs remain at their actual grid offsets; no iid confidence band is shown.
* **Random-number engineering:** [NumPy independent streams](https://numpy.org/doc/stable/reference/random/parallel.html)
  supports separated simulation mechanisms and paired counterfactual experiments.
* **[ewma], [cusum] Statistical engineering guidance:**
  [NIST monitoring handbook](https://www.itl.nist.gov/div898/handbook/pmc/pmc.htm).
  Downstream control-chart implementations remain outside this review's change scope.
