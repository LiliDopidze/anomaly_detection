# A pragmatic optical-loss detection workflow

19 September 2026. Branch `codex/branch-2`; `main` is unchanged.

## Decision: fewer mechanisms, better measurement of errors

The goal is useful early warning of sustained optical deterioration. This is a
narrower, testable target than detecting every telecom anomaly. The primary model
is a robust per-device reference plus EWMA. Isolation Forest remains a challenger,
not a mandatory extra production model. No algorithm can reliably predict a
sudden failure without an observable precursor.

| Keep | Simplify or remove | Why |
|---|---|---|
| Reproducible generator and physical checks | Do not rebuild the simulator again | Its separation of physics, measurement and truth is useful |
| Explicit adapter | Four canonical fields, no mapping YAML | The model needs time, identity and two optical powers |
| EDA, missingness and seasonality inspection | Three notebooks instead of five | Follow the actual data-science decisions |
| Per-device normalisation and causal smoothing | Remove the separate feature framework | The few features fit directly beside the scoring calculation |
| Static and Isolation Forest benchmarks | Remove combined channels and portfolio selection | No demonstrated incremental benefit justifies the extra layers |
| Chronological evaluation and frozen final assessment | Small saved JSON and one opening marker | Prevent accidental leakage without an experiment-ledger framework |
| Warning persistence and recovery | Remove topology localisation and queue ranking | These require operational evidence and are separate from detection |
| Focused statistical tests | Remove tests of deleted infrastructure | Protect meaningful behaviour without maintaining unused abstractions |

The previous expanded implementation is preserved in
[Git history](https://github.com/LiliDopidze/anomaly_detection/tree/74ae8a86dcd24afb43170bc7379eab6f0f0e6eac).
Generated local datasets and outputs are preserved too.

## Model and features

For each ONT and supported optical signal, fit a reference median `m` and robust
spread `s = max(1.4826 * MAD, 0.15 dB)`. Use only the reference period. At time t:

- Signed deterioration: `z[t] = (m - power[t]) / s`.
- Smoothed deterioration: `u[t] = (1-alpha)*u[t-1] + alpha*z[t]`.
- `alpha = 1 - exp(-poll_interval / smoothing_duration)`; default duration one hour.
- Main score: maximum of the available downstream and upstream smoothed scores.

These are four derived features: downstream/upstream signed deviation and their
two EWMAs. Only the two EWMAs determine the main score. Isolation Forest uses all
four with training-only median imputation. Device IDs, topology, fault labels,
impact times and other simulator fields are never model features.

The reference stays fixed so slow degradation is not learned away. At least 100
readings support a reference signal; an unseen device or two missing signals
produce no score. A missing reading or a gap greater than 1.5 polling intervals
resets smoothing. The next valid value starts the new smooth; warnings still need
two successive valid high scores. Longer history is needed to judge whether the
reference covers seasonal regimes; 100 readings is only a support safeguard.

Threshold = calibration-score median + six robust standard deviations, with a
minimum score of three for EWMA. This is an empirical operating-point heuristic,
not a six-sigma false-alarm guarantee. Serial dependence, mixed devices, faults
in calibration and taking a maximum all prevent that interpretation. The normal
reference and calibration periods must be mostly healthy. Contamination and long
seasonal changes require operator review and additional stress tests.

Two highs open a warning at the second reading, never backdated. Two lows below
the midpoint of calibration median and threshold close it. Short missing periods
preserve an open warning but clear pending confirmation counts. After six hours
without a valid score it closes at the last valid observation, labelled `gap`;
this is administrative closure, not evidence of recovery. Silence itself is not
an optical-loss detection. Coverage and missingness must remain visible.

EWMA's sensitivity to small persistent shifts is established statistical process
control practice: [NIST EWMA guidance](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc324.htm).
The smoothing duration, noise floor, confirmation counts and recovery policy are
engineering choices, not values established for this operator. Fit-only-on-past
preprocessing follows [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html).

## Seasonality and company transfer

Notebook 1 plots training-only time traces and device-centred daily profiles.
We do not automatically remove a fitted daily curve: that extra model must earn
its place by reducing false alarms on later periods while preserving fault
sensitivity. The current spread absorbs some normal variation, but it cannot
guarantee protection from annual change or a new operating regime. The 90-day
simulation cannot establish annual realism.

The adapter makes field names, dBm/mW/W conversion and local timezone explicit.
Local references remove fixed link-budget offsets; tests verify offset invariance
after refitting. This is portability of a method, not evidence of zero-shot transfer.
Cadence, reference dates, measurement resolution, receiver classes and acceptable
alert workload remain company-specific. The illustrative static −27 dBm limit
must be replaced by actual receiver specifications when evaluating an operator.

The batch API accepts mapped measurements directly. It needs historical context
for smoothing and does not persist live warning state across separate calls.
Live scheduling, notifications and ticket integration are intentionally absent.

## Evaluation that answers the operational question

The chronological split is 40% reference fit, 30% threshold calibration, 15%
development, 15% final assessment. Settings are written before development labels
are scored. Code/data/model fingerprints and an explicit one-opening marker guard
against accidental final-test reuse; they are not security controls. Generator
structural validation may inspect the complete fixture, but held-out detector
performance is not used for development.

Faults fully contained in the evaluation interval are eligible. An alert must
start during the physical fault interval on an affected device. Maximum-cardinality
one-to-one matching avoids counting one warning as several detected faults.
Overlapping faults can make attribution ambiguous; this is association, not proof
of the physical cause. Boundary faults and warnings starting within them are
reported separately. A warning starting before a later fault is not excused just
because its duration overlaps that fault.

Report detected/missed events, onset delay, warnings before simulated impact,
false alarms per monitored device-day, duplicate warnings, total warnings and
score coverage. Positive lead-time medians describe successes only; early recall
uses all eligible impacting faults in its denominator. Duplicates include multiple
ONT warnings for one shared fault and repeated warnings during the same fault.
They remain real workload. Poisson/Wilson intervals are descriptive: shared faults
violate independent-event assumptions. We do not inflate point-level recall by
crediting every timestamp in an event after one detection.

## Observed development evidence

All results below use unchanged one-hour smoothing and sensitivity six. All three
detectors share warning/recovery logic, but their operating points are not matched
to identical workload and the comparators are not exhaustively tuned.

| Default scenario | Faults detected | False alarms | Duplicate warnings | Total warnings | Before impact |
|---|---:|---:|---:|---:|---:|
| Robust EWMA | 36/40 | 2 | 62 | 161 | 3/9 |
| Isolation Forest | 36/40 | 18 | 63 | 174 | 3/9 |
| Static power limit | 11/40 | 0 | 27 | 54 | 3/9 |

EWMA detected 9/12 abrupt, 13/13 gradual and 14/15 intermittent faults. It warned
before impact for all three impacting gradual events, but not the other six
impacting faults. Median lead time among its three successful early warnings was
17.75 hours. Three examples do not establish reliable prediction. Remaining
warnings outside the table's matched/false/duplicate categories concern boundary
faults; see the comparison CSV for their counts.

| Additional scenario | EWMA detected | EWMA false alarms | IF detected | IF false alarms |
|---|---:|---:|---:|---:|
| New seed, 48 ONTs | 17/19 | 2 | 19/19 | 19 |
| More noise and missing polls | 20/23 | 1 | 19/23 | 1 |
| Hourly observations | 17/19 | 2 | 17/19 | 1 |
| No injected faults | 0/0 | 0 | 0/0 | 3 |

These runs support retaining EWMA as an understandable starting point; IF remains
worth comparing and sometimes detects more events. Zero false alarms in one
healthy synthetic run does not imply a zero field rate. Shared-event duplicates
are still substantial. The final performance period has not been opened.
The changed features and warning policy mean these figures are not a direct
algorithmic improvement claim over the earlier expanded pipeline.

Next evidence to obtain: normal-regime changes, contaminated calibration and
seasonal shifts; more independent seeds and weaker gradual loss; then representative
operator measurements and incident records. Fix the false-alarm budget and minimum
useful lead time with the operator before claiming success. Add FEC, explicit
seasonal correction or topology grouping only when error analysis demonstrates a
specific missing capability and a later-period comparison shows benefit.

## Assessment of the submitted generator

The original notebook is preserved byte-for-byte in [the pre-cleanup commit](https://github.com/LiliDopidze/anomaly_detection/tree/a4f37dd0e5d5702e863ca6ae9883c4173a72bcaa/reference). I reviewed its
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
implementation remains available in Git history for comparison.

## Evidence and assumptions

| Decision | Evidence | What remains an assumption |
|---|---|---|
| GPON downstream line rate 2.48832 Gbit/s | ITU-T G.984.2 [1] | We model GPON, not every PON generation |
| Received power = launched power − link losses | Optical link-budget practice in [1] | Feeder/drop distributions, connector losses, attenuation coefficients and receiver thresholds are illustrative |
| RS(255,239) framing assumption | ITU-T G.984.3 amendment [2] | Always-on coding, full-rate opportunities and independent errors simplify an actual device counter |
| Optical power, temperature and bias are useful observations | Industry monitoring practice [3, 4] | Exact noise, thermal coefficients, accuracy and resolution must be measured for a real source |
| Time-aware EDA and chronological validation | Forecasting practice [5] | 40/30/15/15 split fractions are engineering choices |
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


## References

1. [ITU-T G.984.2: GPON physical media dependent layer](https://www.itu.int/rec/T-REC-G.984.2).
2. [ITU-T G.984.3 Amendment 1: transmission convergence and FEC](https://www.itu.int/rec/dologin_pub.asp?id=T-REC-G.984.3-202003-I%21Amd1%21PDF-E&lang=e&type=items).
3. [Nokia: optical power and voltage monitoring report](https://documentation.nokia.com/html/3HE19010AAACTQZZA/webdocs-enus/Analytics/OpticalPortPowerSummaryReport.html).
4. [Nokia: smarter broadband anomaly detection with AI](https://www.nokia.com/blog/smarter-broadband-anomaly-detection-with-ai/).
5. [Hyndman and Athanasopoulos: Forecasting, Principles and Practice](https://otexts.com/fpp3/), including [seasonality diagnostics](https://otexts.com/fpp3/stlfeatures.html).
6. [scikit-learn: common pitfalls and data leakage](https://scikit-learn.org/stable/common_pitfalls.html).
7. [Liu and Paparrizos, TSB-AD, NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/hash/c3f3c690b7a99fba16d0efd35cb83b2c-Abstract-Datasets_and_Benchmarks_Track.html).
