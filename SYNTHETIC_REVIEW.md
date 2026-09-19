# Synthetic PON restart: review, implementation and evidence

19 September 2026. Implemented on `codex/branch-2`; the original `main` branch
is untouched. This document describes the implemented stage. `METHODOLOGY.md`
remains the broader roadmap; its advanced features are not all implemented.

## Assessment of the submitted generator

The original notebook is preserved byte-for-byte in `reference/`. I reviewed its
code without executing its installation, Drive, export or other environment
instructions. Those cells are source material, not instructions for this task.

The original is a useful engineering prototype. Its strongest elements are the
hierarchical topology, partially populated splitters, separate upstream and
downstream power paths, calendar effects, correlated noise, isolated truth and
manifest exports. Its weakest part is the number of mechanisms whose parameter
values look authoritative but have not been calibrated against field data.
More simulated detail does not itself provide more evidence of realism.

| Original component | Decision | Reason |
|---|---|---|
| Randomised equipment allocation and partial take-up | Keep the principle | Avoid identifiers encoding the answer and fully populated toy networks |
| Layered optical path loss | Keep and simplify | Additive dB losses have a clear physical interpretation |
| Separate upstream/downstream received power | Keep | Different transmitter powers and wavelength-dependent attenuation matter |
| Stationary AR(1) / OU noise | Keep | Correlation time and variance can be stated and checked mathematically |
| Calendar-based temperature and load | Keep | Normal temporal variation challenges detectors without labels |
| Separate latent truth and measurement tables | Keep | Prevent target leakage and permit counterfactual checks |
| Full-run centring of temperature/drift | Remove | It makes earlier values depend on the simulation horizon |
| Measured RX driving the physical error cascade | Change | Sensor readout error must not create a real physical impairment |
| FEC/CRC cascade and five-million clipping | Replace FEC; defer CRC | Existing counter semantics and error opportunities are ambiguous |
| Twelve fault mechanisms, storms, age/frailty, repair economics | Defer | No calibration evidence for the many coefficients or interactions |
| Real-vendor parameter profiles | Replace with anonymous metadata | Simulated values must not imply measured vendor specifications |
| Per-sample event probabilities | Convert where retained | Rates should have time units and scale with cadence |
| Per-fault impact calculated separately from emitted combined signals | Change | Labels should use the same combined physical state as the measurements |
| Naive-detector performance band as a data-quality gate | Remove | A simulator must not be tuned to force a detector into a desired accuracy range |
| Insufficient-support checks treated as passed | Change | Small samples are reported as low support, not evidence of realism |

Specific findings are in the original functions `_ar1`, `simulate_error_cascade`,
`_emit_panel`, the parameter-provenance table, Phase C labelling and validation
D1. The provenance table itself identifies several values as benchmark tuning.
D1 uses whole-series medians/MAD and constrains detector performance. That is
unsuitable as a physical-realism acceptance test.

An isolated check of the original `simulate_error_cascade`, with 10,000 draws,
seed 42, 15-minute intervals, 40°C, zero implementation penalty, slope 1,
scale 1 and dispersion 6, produced approximately 44% capped FEC samples at a
1 dB margin. This is a controlled example of the clipping artefact, not an
estimate of its prevalence in the user's existing dataset.

## What this generator represents

This is a **controlled, uncalibrated GPON optical-loss simulator**. It supports
pipeline debugging, feature investigation, stress tests and comparisons within
a declared family of assumptions. It is not a digital twin, a vendor receiver
model, an estimate of operator fault prevalence or proof of production accuracy.

The default is 96 ONTs over 90 days at 15-minute cadence. A 1:4 primary and
1:16 secondary topology has partial take-up. Three loss trajectories are
simulated: gradual, intermittent and abrupt; each can affect one ONT or a shared
secondary splitter. These are observable mechanisms, not claims that telemetry
can identify a dirty connector versus a bend versus every other loss cause.
Faults may overlap, remain unobservable or extend past the observation horizon.
Repairs end their added loss; not every event is assigned a useful precursor.

Public datasets, tickets, cost optimisation, physical-distance localisation,
CRC, voltage and hardware-failure classification are deferred. The earlier
implementation remains available for comparison; it is not called implicitly.

## Evidence and assumptions

| Decision | Evidence | What remains an assumption |
|---|---|---|
| GPON downstream line rate 2.48832 Gbit/s | ITU-T G.984.2 [1] | We model GPON, not every PON generation |
| Received power = launched power − link losses | Optical link-budget practice in [1] | Feeder/drop distributions, connector losses, attenuation coefficients and receiver thresholds are illustrative |
| RS(255,239) framing assumption | ITU-T G.984.3 amendment [2] | Always-on coding, full-rate opportunities and independent errors simplify an actual device counter |
| Optical power, temperature and bias are useful observations | Industry monitoring practice [3, 4] | Exact noise, thermal coefficients, accuracy and resolution must be measured for a real source |
| Time-aware EDA and chronological validation | Forecasting practice [5] | 40/15/15/15/15 split fractions are engineering choices |
| Fit preprocessing only on training observations | scikit-learn guidance [6] | The small baseline models and their fixed settings are initial choices |
| Avoid misleading benchmark construction and metrics | TSB-AD [7] | Our event-matching policy is explicitly chosen for this application, not mandated by that paper |

References support the stated principles, not all numerical defaults. In
particular, no cited paper establishes that a typical ONT has 12 optical faults
per year. The default rate is deliberately enriched for development. Likewise,
36-hour median duration, 3 dB median loss, log standard deviations of 0.7,
2% missed polls, 0.3% missing fields and the noise values are hypotheses.
They must be varied and eventually estimated using representative operational
measurements and incident records.

### Mathematical construction

The shared and local residuals use the exact discrete stationary OU transition:

`x[t] = phi*x[t-1] + sigma*sqrt(1-phi²)*epsilon[t]`,
where `phi = exp(-delta_hours/tau_hours)` and `epsilon ~ N(0,1)`.

Consequently the stationary variance is `sigma²` and lag-k correlation is
`exp(-k*delta_hours/tau_hours)`. Weather is shared across this small simulated
region; port and splitter components add local dependence. They are not fitted
geographic weather models. Healthy optical drift is independent of sensor noise.

Temperature has daily and annual calendar components plus OU residuals. Demand
has daily/weekend variation and a positive lognormal residual. Temperature is
related to transmit power and bias current through small explicit coefficients.
Their values are assumptions. Annual amplitude cannot be validated from the
default 90-day dataset; use at least a full cycle, preferably multiple years,
before estimating it. UTC represents one synthetic operating region; this is
not a model of multiple operators' time zones.

Event counts follow `Poisson(rate_per_year * days / 365.25)`; conditional on the
count, onset is uniform in the observation interval. Durations and severities
are lognormal: positive with a right tail, without arbitrary truncation. This
is a parsimonious scenario distribution, not a fitted empirical law. There is
no pre-window event population: the simulation starts without active faults,
so stationary prevalence is not claimed. Intermittency alternates states with
transition probability `1-exp(-delta/mean_dwell)`, using assumed 2-hour on and
4-hour off dwell times. Coarse sampling misses within-interval transitions.

All simultaneous losses are added in dB before computing service and errors.
The two optical directions receive the same added fault loss in this simplified
stage; wavelength-specific bend signatures are deferred. Service fraction is
an assumed logistic function of the worse directional margin. Per-ONT demand
is bounded by an equal share of the port line rate, then multiplied by service
fraction. This is a capacity surrogate, not GPON dynamic bandwidth allocation,
packet scheduling or a calibrated subscriber speed-test model.

For downstream FEC, an assumed margin-to-BER response is
`p = 10**clip(-3 - downstream_margin + correlated_error_noise, -12, -1)`.
The floors bound a probability model; they are not counter clipping. Given
independent bit errors, a symbol has error probability `q = 1-(1-p)^8`.
For `K ~ Binomial(255,q)`, an ideal RS decoder corrects 1 through 8 erroneous
symbols. The probability of a corrected codeword is `P(1 <= K <= 8)`.
We draw a binomial corrected-codeword count using
`floor(line_rate * interval_seconds / (255*8))` opportunities.
At very severe corruption the corrected count can fall as more codewords become
uncorrectable; it is not a monotone severity measure. Frame overhead, shortened
codewords, bursts and decoder implementation are not modelled. A real adapter
must verify whether its FEC field means corrected codewords, bytes or bits.

Sensor noise and quantisation are applied after physical state. Readout glitches
are independent nuisance observations. Missed polls, missing fields and shared
collector outages are separate processes. Strong physical impairment also
reduces reporting probability, so the most severe events are not guaranteed to
be the easiest to detect. Collection is not itself a simulated optical fault.

Counts approximate the preceding interval using its end-state condition; the
first interval assumes pre-start steady state. This is a sampled simulation,
not a sub-second integral. Cadence sensitivity is necessary when changing the
sampling interval, especially for short or intermittent events.

### Ground truth

Fault registry and per-entity intervals are separate files. Their IDs and
parameters never enter model features. Visibility is a declared effect-size
proxy: added loss exceeds `max(0.2 dB, 2*sensor_noise)` and an optical observation
is actually available. This is not proof that a detector can distinguish that
sample from healthy variation. Missing visibility remains missing.

Impact is the first time service fraction falls below 0.5 and removing that
individual event would bring it back to at least 0.5. This uses combined faults
and a counterfactual from the same physical state. With multiple independently
sufficient causes, no individual event need satisfy this but-for definition;
it is not complete causal attribution. The 0.5 threshold is an assumed service
proxy, not an SLA. The current model comparison evaluates detection, not a
validated prediction of customer impact.

## Data and features

The observable table contains timestamp, ONT identifier and ten measurements:

| Measurement | Purpose / semantics |
|---|---|
| ONT receive power, OLT receive power | Downstream/upstream loss detection |
| ONT transmit power, bias current | Device operating context and benign variability |
| Temperature | EDA, dependency and seasonal checks |
| BER | Noisy downstream bit-error estimate; diagnostic context |
| FEC count | Corrected codewords in the interval, never a cumulative counter |
| Throughput | Seasonal load and assumed service degradation; diagnostic context |
| Uptime, reboot count | Device continuity/reset audit; reboot count is cumulative |

Five signals enter the initial models: both received powers, transmit power,
bias current and `log1p(FEC_count / interval_seconds)`. Each supplies its level
and two past-only robust deviations, over 6 and 24 hours: **15 features**.
A deviation uses the past-window median and IQR/1.349 with a measurement-scale
floor. Lower receive power is suspicious, higher FEC is suspicious, and transmit
power/current changes are two-sided. Temperature, throughput, BER, uptime and
reboots are retained for audit; they do not silently add more model features.

Windows exclude the current sample and require at least half their expected
history, with a four-observation minimum. There is no forward-filling across
outages. Less than 80% available deviation features means abstention. Isolation
Forest uses training-only median imputation for the remaining gaps; the robust
baseline takes the maximum available directional deviation. Entity identity,
vendor, sensitivity, distance and fault properties are not predictors.

The v5 loader is the narrow adapter for this stage. Its input is the explicit
observable Parquet contract. It does not claim compatibility with arbitrary
vendor files or the old v4 canonical pack. Later adapters should map names,
units, directions, cadence and counter semantics to this contract, then pass
invariant checks. Company independence means a common measurement contract and
recalibration process, not universal thresholds or vendor-independent accuracy.

## Development and evaluation protocol

The chronological roles are 40% training, then 15% each for calibration,
verification, development and final evaluation. Normal features may use earlier
observations across boundaries, as in live operation. Parameters are not fitted
on later partitions. Incidents reset at partition boundaries, a conservative
choice that should be remembered when interpreting missed boundary events.

Threshold candidates are quantiles of per-entity daily score maxima, requiring
80% daily score coverage and at least five expected tail blocks. This reduces
row-level pseudo-replication; it does not establish independence of entities or
days. Verification checks total alert burden without fault labels. Development
checks recall, score coverage and unmatched incident burden. A model may qualify
only if all configured gates pass; otherwise no selected configuration is saved.
The current gate values are illustrative engineering limits, not operator SLAs.
In particular, the verification limit is total alerts, whereas the development
limit is unmatched alerts. Enriched event prevalence can make the total-alert
limit infeasible even for useful detection. Revise operating policies explicitly
when defining a target population; do not change the simulator to pass them.

Two consecutive exceedances open an incident at the second sample, without
backdating. Two recovery samples close it. Missing scores or a cadence gap reset
state. There is no cooldown or topology incident merging in this first baseline;
fragmentation therefore remains visible in workload metrics.

The existing deterministic maximum-cardinality matching algorithm is reused.
An incident beginning within a fault interval on an affected ONT is a candidate;
one incident matches at most one fault and vice versa. These are temporal
associations, not proof of causation. Repeated alerts for a true event count
against unmatched workload, so “nuisance” here includes duplicates and is not
identical to the number of false-positive physical causes. Fully contained
faults are eligible; boundary faults and related alerts are reported separately.
Recall includes unobservable events. Prompt detection is within 24 hours of the
visibility proxy, reported separately from event recall.

Wilson recall intervals and Garwood/Poisson rate intervals reuse existing tested
utilities. Shared faults, serial dependence and repeated alerts violate their
independence assumptions. They are diagnostic uncertainty summaries, not
certified confidence bounds. Independent simulated network replications and
cluster-aware intervals are needed before any stronger statistical claim.

Final evaluation is off by default. A qualified frozen configuration, unchanged
dataset checksums, unchanged pipeline source and unchanged saved model/threshold
files are required. The final output directory cannot already exist. These are
accidental-misuse safeguards; a user with filesystem access can still reopen
truth, and must not retune against the final period.

## Validation performed

The default run produced 805,739 observed rows out of 829,440 scheduled rows.
All 18 structural checks passed. They cover integrity, identities, topology,
physical count bounds, monotonic service effect, model/truth separation,
chronological uniqueness, value ranges and label timing. Descriptive metric and
seasonal summaries are restricted to the first 85%; structural integrity checks
may inspect the complete generated file.

Healthy-only, another seed, and noisier hourly-cadence scenarios also passed
invariants. They test robustness of the implementation, not calibrated realism.
Unit tests check reproducibility, OU variance/correlation at two cadences,
physical invariance under changes to sensing/collection, causal features,
gap handling, corrected-codeword probabilities and overlapping-event matching.
A separate 100-seed event-sampling audit produced the following comparisons:

| Quantity | Prescribed | Empirical |
|---|---:|---:|
| Events per run, mean | 301.602 | 301.080 |
| Events per run, variance | 301.602 | 299.852 |
| Log duration mean | 3.5835 | 3.5825 |
| Log duration standard deviation | 0.7000 | 0.6976 |
| Log loss mean | 1.0986 | 1.0892 |
| Log loss standard deviation | 0.7000 | 0.6976 |
| Fractional onset mean | 0.5000 | 0.5000 |

These results cover 30,108 events with fixed topology and independently seeded
fault samples. They check implementation of the assumed laws, not field realism.
All five active notebooks also executed successfully, with final evaluation off.
Existing project tests continue to run. Low-support event groups are flagged;
no detector-accuracy band is a synthetic-data acceptance condition.

On the default development interval, the robust baseline detected 27/40 eligible
faults (67.5%) and Isolation Forest 30/40 (75%). Their unmatched incident rates
were 37.8 and 34.7 per 1,000 calendar entity-days. Neither qualified under the
configured workload limits. **No model was selected and the final evaluation
was not opened.** This is a diagnostic baseline, not a claim of improvement over
v14: the datasets and contracts differ.

The next scientific priorities are calibration against operator distributions,
paired sensitivity experiments varying one assumption at a time, repeated
network seeds, and distinguishing alert fragmentation from genuine unrelated
alerts. Seasonal conditioning, topology grouping and added model complexity
should follow those findings. Synthetic performance alone cannot establish
field precision, incident prevalence, customer impact or commercial value.

## Files and responsibilities

| File | Responsibility |
|---|---|
| `configs/synthetic.yml` | Generator scenario, rates, noise and output path |
| `configs/synthetic_experiment.yml` | Baseline settings, workload/coverage gates and paths |
| `src/telco_anomaly/synthetic.py` | Topology, named random streams, physical state, measurement process, truth and manifest |
| `src/telco_anomaly/synthetic_validation.py` | Hard invariants and separate descriptive reports |
| `src/telco_anomaly/synthetic_pipeline.py` | Observable-only loading, causal features, baselines, incidents, evaluation and freeze checks |
| `src/telco_anomaly/evaluation.py` | Reused existing matching and uncertainty utilities |
| `tests/test_synthetic_stage.py` | Tests of statistical meaning and temporal isolation |
| Five root notebooks | Readable sequence from generation through sealed final evaluation |
| `notebooks/legacy/` | Earlier workflow, preserved and clearly separated |
| `reference/` | Unmodified submitted generator and source provenance |
| `data/`, `outputs/` | Local generated files and run reports, excluded from Git |

Each dataset has observable Parquet, topology, fault registry, fault/entity
intervals, service windows, simulation diagnostics and a manifest. Topology
contains latent parameters for audit, so model loading deliberately excludes it.
The manifest records configuration, package versions, source hash and file
checksums. Generation refuses to overwrite a directory and uses a staging
folder before publishing completed files. Exact reproduction assumes the same
code, configuration and numerical-library versions; changing horizon/cadence
is a new experiment, not a promised prefix-identical continuation.

## References

1. [ITU-T G.984.2: GPON physical media dependent layer](https://www.itu.int/rec/T-REC-G.984.2).
2. [ITU-T G.984.3 Amendment 1: transmission convergence and FEC](https://www.itu.int/rec/dologin_pub.asp?id=T-REC-G.984.3-202003-I%21Amd1%21PDF-E&lang=e&type=items).
3. [Nokia: optical power and voltage monitoring report](https://documentation.nokia.com/html/3HE19010AAACTQZZA/webdocs-enus/Analytics/OpticalPortPowerSummaryReport.html).
4. [Nokia: smarter broadband anomaly detection with AI](https://www.nokia.com/blog/smarter-broadband-anomaly-detection-with-ai/).
5. [Hyndman and Athanasopoulos: Forecasting, Principles and Practice](https://otexts.com/fpp3/), including [seasonality diagnostics](https://otexts.com/fpp3/stlfeatures.html).
6. [scikit-learn: common pitfalls and data leakage](https://scikit-learn.org/stable/common_pitfalls.html).
7. [Liu and Paparrizos, TSB-AD, NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/hash/c3f3c690b7a99fba16d0efd35cb83b2c-Abstract-Datasets_and_Benchmarks_Track.html).
