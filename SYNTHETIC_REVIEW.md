# Synthetic PON pipeline: method, evidence and implementation

19 September 2026. Implemented on `codex/branch-2`; the original `main` branch
is untouched. The implementation section below supersedes the initial v5 baseline
workflow described later in this document. This document describes the implemented stage. Branch 2 contains only
this workflow; the earlier roadmap and implementation remain in Git history.


## Current implementation: the explicit adapter and connected workflow

The supplied September 10 approach document describes a research programme through
operator deployment. The implemented path is generation → validation → adapter →
canonical pack → training-only EDA → causal features → evidence channels → frozen
calibration → development assessment → incident queue → qualification → locked
final assessment or frozen future inference. Public integration remains out of scope.

### Mapping the document to executable components

| Document requirement | Implementation and remaining limit |
|---|---|
| Reviewed native mappings | `adapter.py` and `configs/adapter.yml`; explicit fields, units, kinds and vendor overrides |
| Quality codes and counter resets | Cell-level quality table; no cumulative FEC difference across a gap, reset or kind change |
| Time and topology | UTC normalisation with explicit local timezone; effective-dated memberships; unknown entities and overlapping memberships rejected |
| Operational alarms | Reviewed codes map to `onu_power_loss` or `loss_of_signal`; unknown codes fail; no free-text matching |
| Separate truth | Generator output is transient; only observables enter `data/`; evaluation labels go to the separate `evaluation/` root |
| Blind reconstruction | Adapter test rebuilds identical packs after removing source labels |
| Cadence, episodes and gaps | Explicit missing rows; return-time classification using uptime and reboot continuity; restart episodes |
| EDA decisions | Daily seasonal approval and profiles use the training slice only; timezone, support and decisions saved |
| Causal features | Past-only rolling deviations, interval FEC rate/nonzero indicator, directional power changes and asymmetry, frozen daily residuals |
| Independent evidence channels | Robust residual, drift, peer, common mode, silence and inventory margin; residual-feature Isolation Forest challenger |
| Status-quo comparator | Declared static receive-power limit with the same persistence/recovery; no claim of reproducing an actual operator's rules |
| Thresholds and portfolio | Empirical daily block maxima with support checks; fixed rules for margin, silence and comparator; combined thresholded-evidence portfolio |
| Label-free operating points | Choices are written and hashed before the first development truth read |
| Single scoring implementation | Saved-model development replay must equal original scores exactly; inference and final scoring use the same function |
| Incidents | Port/quiet-period consolidation, explicit membership and actual last-member decision timestamp |
| Disposition | Canonical power-event share across the scope; available even when some events belong to non-alerting members |
| Scope ranking | Conditional single-fault likelihoods over unique observable footprints, top candidates and structural ambiguity |
| Development selection | Workload/recall/coverage gates; simplest adequate declared candidate; STOP if none qualifies |
| Capability reports | Per-mechanism recall with Wilson intervals and estimability flag; current queues and matching outcomes |
| Governance | Frozen code/configuration/package/input fingerprints, artifact checksums, selection ledger and one-opening holdout ledger |
| Future inference | A qualified frozen model consumes a new canonical pack without opening truth; overlap with the experiment window is refused |

### Deliberate differences from the older approach

The document's twenty-one steps are not all software acceptance tests. Several
require independent data, a PON engineer or operator participation. The following
are explicitly **not completed evidence gates**:

- Physical realism calibrated to operator measurements, four simulated operator
  populations, directional mechanism expansions, power-fault labels and re-homing
  scenarios in the generator. The adapter accepts effective-dated topology and
  reviewed events; the current generator does not establish their field accuracy.
- Independent sealed-generator qualification, leave-one-operator-out capability,
  graded-injection power curves and comparisons with externally tuned algorithms.
- Calibrated localisation probabilities, inferred topology corrections and
  performance against labelled power-loss events. Current rankings use declared
  hit/background assumptions, not fitted field probabilities.
- Public-data and partner shadow-deployment gates.

Sparse-tail generalised Pareto fitting is not enabled: small support does not
justify inventing a stable tail estimate. Unsupported thresholds fail closed.
Daily maxima reduce row-level pseudo-replication but do not establish independent
blocks or a conformal coverage guarantee. Weekly profiles and temperature
compensation are not fitted by default. These are optional refinements needing
support and incremental-value evidence, not requirements for running the pipeline.

A filesystem allowlist and separate roots protect against accidental label reads.
They are **not** an OS security boundary: this local process still has permissions
to other folders. A strict truth-isolated modelling environment requires a separate
account/container or unmounted storage. The blind-rebuild test proves adapter
independence from labels, not infrastructure access control. Ledgers similarly
prevent accidental reuse; an administrator could edit their files.

### Data contract and adapter behaviour

A pack has exactly six files: `telemetry.parquet`, `inventory.parquet`,
`events.parquet`, `quality.parquet`, `adapter_audit.json`, and `manifest.json`.
The observable telemetry keeps the ten declared metrics in their canonical units.
Topology carries only mapped inventory attributes, including simulated receiver
sensitivity; it does not carry latent path loss or injected fault parameters.
The absence of events produces an empty typed event table, not fabricated alarms.

Mappings do not guess units or semantics. For example, mW power becomes
`10*log10(mW)` dBm, watts first become mW, and cumulative corrected-codeword
counts become adjacent-interval differences. Negative deltas, missing predecessors,
resets and gaps invalidate that interval. Unknown columns, duplicate observations,
unknown alarm codes, ambiguous local timestamps and off-cadence readings are refused.
Missing or invalid numeric observations retain explicit quality codes.

Company independence comes from this stable contract and local label-free
calibration. It is not a promise that one model's thresholds or seasonal profiles
work unchanged at another operator. The supplied mapping must be reviewed against
each actual source, particularly FEC units and alarm semantics.

### Features, decisions and operational semantics

The baseline is the prior median/IQR feature calculation. Additional features are
frozen daily residuals where population evidence supports seasonality, an FEC
nonzero indicator, downstream/upstream trailing changes and their asymmetry.
Isolation Forest uses residual features instead of raw level features. The
classification of a preceding gap is available only when a reading returns;
silence itself is available immediately from the current cadence grid.

CUSUM holds its state across missing reports and resets on observed device restarts.
Peer and common-mode evidence use contemporaneous splitter measurements with a
minimum group size. Silence compares the target splitter's missing fraction with
reporting by the rest of its OLT, suppressing whole-collector outages. Optical margin
uses explicitly supplied inventory sensitivity. Fixed-rule thresholds are not
selected using fault labels.

Channel thresholds are independent; the full portfolio combines their exceedance
indicators before persistence/recovery. Threshold verification conservatively
counts entity alerts. Development metrics count consolidated queue cases, so these
are different workload quantities, explicitly reported. Case matching uses the
last member's decision time: later membership never earns backdated detection
credit. Cases matching no fault, including duplicates, count against workload.
Boundary events are excluded and reported as in the original evaluator.

Incident consolidation is limited to a PON port and a quiet window. Ranking is a
single-fault hypothesis model over observed alert membership, not causal diagnosis.
Identical topology footprints share one hypothesis with equivalent scope names,
so duplicated inventory levels do not multiply evidence. The reported probability
belongs to that footprint equivalence class and is not calibrated confidence.
Simultaneous independent faults, missing topology and collector ambiguity can
invalidate that simple interpretation. Power disposition is a routing heuristic;
it is not counted as validated power-loss detection.

### Reproducibility and acceptance

`configs/pipeline.yml` resolves all paths. Preparation and modelling refuse changed
upstream inputs or existing run outputs. A run saves the resolved policy, package
versions, source hashes, canonical manifest hash, model, seasonal decisions,
label-free threshold choices and development scores. Replaying the saved model
must reproduce those scores exactly before evaluation proceeds.

The selection ledger is written before reading development truth. The holdout
ledger is written before reading final truth and remains outside the result
folder, so deleting results does not reopen the test. The current holdout evaluates
only the frozen selected candidate, not the older document's three-configuration
research comparison. No holdout is opened automatically by the CLI or notebooks.
Future inference refuses observations overlapping the experiment window.

Tests include unit conversion, vendor overrides, counter gaps/resets, DST ambiguity,
blind rebuilding, unknown-field/alarm rejection, dated memberships, collector-vs-group
silence, topology ambiguity, power disposition, 30 informative causal prefix checks,
saved-score replay, stale-artifact rejection, ledger reuse prevention and inference
while the evaluation-truth directory is unavailable. Small integration fixtures use
explicitly permissive gates only to exercise positive selection and holdout paths;
the default scientific experiment does not inherit those permissive settings.

### Expanded-pipeline development result

The 90-day synthetic run has 40 fully contained development faults. On consolidated
case metrics, robust residuals detect 24/40, Isolation Forest 30/40, static rules
9/40, and the combined portfolio 19/40. Their unmatched case rates are respectively
15.43, 20.06, 6.94 and 21.60 per 1,000 calendar entity-days. No candidate passes all
configured gates; the final test remains unopened.

Adding channels did not automatically improve the combined portfolio. Persistent
channels can merge a long sequence of alerts into a broad case, moving its final
membership decision later and reducing matching credit. That trade-off is visible
rather than hidden by backdating. These are development diagnostics, not evidence
that the combined system is better than the simpler detector. The next modelling
question is incident fragmentation versus over-consolidation, alongside whether
the assumed total-alert budget is feasible under enriched synthetic prevalence.

## Historical v5 generator review and first baseline

The remaining sections record the generator audit and initial v5 baseline, before
the explicit adapter and operational extensions above. Their numerical assumptions
still describe the generator. Their five-signal model results are historical and
must not be presented as results of the expanded pipeline.

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
The retained pipeline and evaluation-helper tests pass. Tests specific to
the removed workflow have been removed too. Low-support event groups are flagged;
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
| `src/telco_anomaly/evaluation.py` | Only the matching and uncertainty functions used by this pipeline |
| `tests/test_synthetic_stage.py` | Tests of statistical meaning and temporal isolation |
| Five root notebooks | Readable sequence from generation through sealed final evaluation |
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
