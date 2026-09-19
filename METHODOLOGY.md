# PON incident detection: methodology and technical design

Version: 1.0 — 19 September 2026

Status: proposed design for the project restart. This document describes the
intended implementation, not a claim that the current repository implements it.
Illustrative settings are experiments to validate, not approved operating limits.

## 1. Objective and scope

Build a reproducible system that identifies sustained optical degradation and
access-network incidents, groups related evidence, and prioritises investigation
by likely customer impact. The first user is a fibre-access operations engineer.

The output is an incident with an affected scope, start time, supporting
measurements, confidence limitations and suggested investigation. An anomaly
score is intermediate evidence, not a diagnosis or a probability of failure.

Evaluate four separate capabilities:

1. Detection: did the system identify an observable fault?
2. Timeliness: how soon after observability, or before impact, did it alert?
3. Localisation: did it identify the correct affected entity or shared segment?
4. Operational usefulness: was the alert actionable at an acceptable workload?

Do not require a precursor for every fault. An abrupt cut and a gradual optical
degradation have different prediction opportunities. Do not claim exact physical
fault location from topology correlations alone. Published PON localisation work
using OLT and OTDR data has additional sensing that our current table lacks. [R1]

Initial scope excludes autonomous remediation, exact cable-distance localisation,
general cybersecurity detection and complete home Wi-Fi diagnosis.

## 2. Evidence, assumptions and the existing baseline

The inspected `reference_dataset.parquet` contains 6,304,913 rows, 400 ONTs,
4 OLT identifiers and 22 PON-port identifiers. It spans approximately six months
from January through June 2025. Median within-ONT spacing is 900 seconds.
Measurements include optical power, temperature, bias current, voltage, BER,
FEC/CRC, uptime, reboot count and throughput. It also includes equipment and
topology attributes. The file itself contains no fault registry.

These observations establish structure, not physical provenance or realism.
The original generator has now been reviewed. See `SYNTHETIC_REVIEW.md` for
the audit, implemented synthetic subset and evidence limits. The remaining
sections of this document are the broader proposed roadmap.

The current `configs/metric_registry.yml` declares FEC and CRC as interval counts.
It declares uptime and reboot count as cumulative counters, and specifies a
source-specific FEC clipping value. Treat these as existing contracts requiring
confirmation against the generator, not reasons to infer different semantics from
column names. A decreasing interval count is not a counter reset.

The v14 selection run did not qualify a configuration. Changes to history
handling and missing-value treatment were introduced together, so their individual
effects have not been isolated. Preserve that run and the previous implementation
as reproducible baselines. Passing code tests is not evidence of improved recall.

There are three evidence levels:

| Level | Evidence available | Permitted claim |
|---|---|---|
| Synthetic | Known generator, independent scenarios, internal checks | Performance on the documented generated scenarios |
| Physical testbed | Measured equipment responses to controlled faults | Performance under those experimental conditions |
| Operator | Real telemetry and reviewed operational outcomes | Performance on the evaluated operator population and period |

A public wireless or traffic dataset can test a transferable technique, but cannot
replace PON validation. Never pool unrelated sources into a single result merely
because they share column names.

## 3. Proposed repository and responsibilities

```text
anomaly_detection/
├── README.md
├── METHODOLOGY.md
├── pyproject.toml
├── .gitignore
├── configs/
│   ├── synthetic.yml
│   ├── experiment.yml
│   └── metrics.yml
├── notebooks/
│   ├── 01_data_quality_and_eda.ipynb
│   └── 02_model_evaluation.ipynb
├── src/
│   └── telco_anomaly/
│       ├── __init__.py
│       ├── adapters.py
│       ├── data.py
│       ├── synthetic.py
│       ├── diagnostics.py
│       ├── features.py
│       ├── models.py
│       ├── incidents.py
│       └── evaluation.py
└── tests/
    ├── test_adapters.py
    ├── test_data.py
    ├── test_synthetic.py
    ├── test_features.py
    ├── test_models.py
    ├── test_incidents.py
    └── test_evaluation.py
```

This is a target structure, not a deletion instruction. Keep useful existing
modules until their behaviour has been migrated and verified. A module should
have one coherent responsibility; the target filenames must not force thousands
of unrelated lines into a single file. Split only when actual complexity warrants.

### Root files

| File | Responsibility |
|---|---|
| `README.md` | Quick start, installation, data generation/loading, notebook order, expected outputs and limitations. Link here for detailed methodology. |
| `METHODOLOGY.md` | This maintained design and its rationale. Avoid competing methodology documents. |
| `pyproject.toml` | Package metadata, dependencies, optional development tools and test configuration. Record exact installed versions in run manifests. |
| `.gitignore` | Exclude raw/generated data, model binaries, run outputs, environments and notebook checkpoints. Do not hide source files or configuration. |

### Configuration files

| File | Responsibility | Exclusions |
|---|---|---|
| `synthetic.yml` | Topology, period, cadence, seeds, equipment variation, normal dynamics, missingness and fault distributions | Detector thresholds and model performance targets |
| `metrics.yml` | Measurement definitions, units, location/direction, count semantics, missing codes, resolution and documented bounds | Learned baselines or guessed vendor limits |
| `experiment.yml` | Input path, split boundaries, features, candidates, calibration policy, incident policy and evaluation settings | Generator internals and credentials |

Check required fields and reject unknown configuration keys. Save the fully
resolved configuration with each run. Use explicit seeds and paths; avoid notebook
globals that silently select the newest directory.

### Python modules

The following function names are proposed interfaces, not existing APIs.

| Module | Main responsibilities | Example public functions |
|---|---|---|
| `__init__.py` | Package version and small public API; no work at import time | Version only initially |
| `adapters.py` | Documented, source-specific conversion to the shared contract | `adapt_synthetic`, later `adapt_operator_export` |
| `data.py` | File loading, schema checks, split assignment, manifests and dataset persistence | `load_dataset`, `validate_dataset`, `assign_splits`, `write_manifest` |
| `synthetic.py` | Reproducible measurements, topology and separate ground truth | `generate_dataset` |
| `diagnostics.py` | Quality summaries, seasonality analysis, generator audits and fault case studies | `quality_report`, `seasonality_report`, `audit_synthetic`, `inspect_fault` |
| `features.py` | Causal metric transforms, historical summaries and eligible peer evidence | `build_features` |
| `models.py` | Small detector implementations, fitted preprocessing, calibration and candidate comparison | `fit_detector`, `score_detector`, `calibrate_threshold`, `select_candidate` |
| `incidents.py` | Persistence, recovery, alert suppression, grouping and localisation hypotheses | `build_incidents`, `group_incidents` |
| `evaluation.py` | Deterministic event matching, exposure, uncertainty and stratified results | `match_faults`, `evaluate_incidents` |

Start with explicit functions and small result objects. Avoid a plugin system,
large inheritance trees, a model registry service or a workflow framework.

### Notebook 01: data quality and EDA

Run in this order: load/generate; validate schema; define and persist splits;
summarise development-accessible data; inspect time coverage and missingness;
review distributions and physical relationships; inspect seasonality; audit
generator mechanisms; examine labelled development faults; record decisions.

Its deliverable is a data-readiness report, with unresolved semantics and eligible
modelling tasks. It must not train a final detector or open the holdout for EDA.

### Notebook 02: model evaluation

Load the frozen data contract and splits; build causal features; fit baseline
models; calibrate incident policies; compare candidates on development; inspect
errors; freeze a qualified candidate; explicitly enable the final evaluation.

The default execution stops after development. Final evaluation requires an
explicit setting and an existing selection artifact; running all cells must not
accidentally evaluate the holdout. If nothing qualifies, write a STOP report.

## 4. Data contract and adapter design

The shared format is an analytical contract informed by telecom measurement
standards. It is not a claim of full TR-385, TR-181 or YANG compliance. Broadband
Forum's collection/translation architecture supports separating native telemetry
from downstream analysis. [R2, R3]

### Required and optional artifacts

| Artifact | Grain and essential fields |
|---|---|
| `measurements.parquet` | Initially one ONT/observation time; UTC event time, stable ONT ID, available metrics and quality information |
| `topology.parquet` | ONT-to-parent mappings with known OLT, PON port, splitter IDs, equipment metadata and validity intervals where applicable |
| `faults.parquet` | Optional event registry: fault ID, affected scope, onset, end, type, label source and confidence; impact/observability times when defensible |
| `manifest.json` | Dataset ID, schema version, source, hashes, code/config versions, seeds, environment, time semantics and audit counts |

Also retain `service_windows.parquet` when needed to describe when entities were
commissioned and expected to report. This prevents pre-installation time from
being counted as missing telemetry. Unknown commissioning dates remain unknown.

For a fault spanning several entities, use an explicit fault-to-entity mapping or
affected-entity list. Keep a single event ID so the event is not counted once per
ONT. Separate the failed component from the entities experiencing its symptoms.

### Identity and time

Use stable identifiers scoped to the operator/source. A PON port number without
its parent OLT is not necessarily unique. Do not use IDs as predictive numeric
features. Preserve the source IDs for traceability.

Record whether a timestamp describes an instant, interval start or interval end.
An interval measurement becomes available after its interval closes. Preserve
ingestion/availability time when supplied; do not use late measurements before
they would have arrived in an operational replay. The synthetic generator should
declare its assumed reporting delay rather than fabricate evidence of real delay.

Convert timezones only from documented source information. For daily seasonal
features retain the site's timezone and handle daylight-saving changes explicitly;
UTC remains the storage and ordering convention.

### Adapter boundaries

Adapters rename fields, convert documented units, resolve IDs and timestamps,
decode known sentinel values and preserve source metadata. They do not infer
fault labels, fill gaps, smooth series, learn scales or choose thresholds.

Map ONT downstream received power and OLT upstream received power separately.
They measure different directions and cannot be subtracted to claim an optical
path-loss estimate without compatible transmit power, wavelength and measurement
semantics. BER must specify pre/post-FEC if known; unknown semantics are flagged.

Keep the current wide table for the synchronous synthetic measurements. If real
sources have independent metric times, retain those times, possibly in a long
table, before an explicitly configured causal alignment. Do not force apparent
synchrony or treat every source as an ONT dataset.

### Validation responses

- Stop on ambiguous required units, unresolvable time, conflicting identities or
  conflicting duplicates that cannot be resolved by documented revision rules.
- Report exact duplicate removal, rejected records and every unit conversion.
- Flag gaps and missing optional metrics; do not substitute zero.
- Preserve plausible extremes. A healthy operating range is not a validity range.
- Flag impossible readings according to documented bounds while preserving raw
  records for inspection; report resulting scoreability losses.
- Disable group localisation when topology is missing or untrustworthy.

## 5. Synthetic generator methodology

Audit the supplied original code before replacing it. Archive its exact inputs,
configuration, environment and baseline output hashes. Fix generator defects in
a new dataset version so existing results remain interpretable.

### Generation order

1. Generate topology and equipment assignments with documented assumptions.
2. Generate shared environmental and usage processes, plus ONT-specific variation.
3. Generate healthy metrics with appropriate units, support and cross-metric
   relationships; include legitimate operational changes.
4. Sample fault events independently of the detector and attach them to physical
   scopes. Separate mechanism onset, measurable symptoms and customer impact.
5. Apply fault mechanisms to the relevant underlying processes and affected ONTs.
6. Apply measurement effects: quantisation, censoring, reporting delay, missing
   samples, counter behaviour and collector interruptions.
7. Export observed measurements and separate latent truth, metadata and provenance.

Maintain separate random streams for topology, normal dynamics, faults and
collection effects. A change to missingness should not unnecessarily redraw the
entire fault population. Record the random-number implementation and environment;
a seed alone does not guarantee byte-identical output across versions.

### Mechanisms to represent and review

| Scenario | Evidence to consider | Important limitation |
|---|---|---|
| Gradual optical attenuation | Received-power trend, possible error response | Error changes depend on operating margin; not every power change produces errors |
| Abrupt link loss | Alarm/state change and telemetry interruption | No guaranteed pre-failure signal |
| Intermittent connectivity | Repeated valid interruptions/recovery, error bursts | Must distinguish collection loss and customer power events |
| Device instability | Resets and associated health/optical changes | A reset alone does not establish hardware failure |
| Shared segment degradation | Coherent effects across downstream entities | Magnitude and visibility need not be identical across all ONTs |
| Normal traffic variation | Throughput changes without physical degradation | Low demand is not inherently a fault |
| Maintenance/firmware activity | Documented changes, restarts, temporary gaps | Keep these as difficult normal or separately classified operational events |

These are proposed mechanism families, not validated descriptions of the current
generator. Equipment specifications and operator evidence must constrain them.

### Five required audits

| Audit | Procedure | Output and decision |
|---|---|---|
| Observability | Compare pre-event and event windows; inspect mechanism and affected metrics; use paired fault-free replay when the generator supports it | Per-fault visible/no precursor/unobservable/uncertain status. Do not define visibility by whether our detector succeeds. |
| Physical consistency | Check bounds, dependencies, resets, direction and shared-scope effects | Violations and assumptions needing domain review |
| Normal diversity | Include legitimate variation, benign extremes, maintenance, gaps and device cohorts | False-alert stress scenarios without injected faults |
| Fault diversity | Vary severity, duration, onset, recovery, affected fraction and overlapping events | Coverage by mechanism and difficulty; identify repeated templates |
| Independence | Track parent simulations, templates and parameter families across splits | No reused base trajectories or copied fault windows across evaluation partitions |

Compare generated and measured data using distributions, quantiles, correlations,
lagged relationships, gap lengths, event durations and equipment variation. A
two-sample classifier can reveal source differences but cannot certify realism
when it fails to distinguish sources. Matching marginal histograms is insufficient.

Do not tune injected faults until the detector passes. Freeze a scenario suite
before comparison, document any label repair, and rerun all baselines when data
changes. Include held-out parameter ranges and mechanisms as explicit stress
tests, not as replacements for representative operating-condition evaluation.

## 6. EDA and seasonal analysis

### Data quality and exposure

Report entity counts, service windows, expected cadence, observed intervals,
duplicates, missing cells, missing rows and valid measurement coverage. Break down
by time, metric, vendor, model, firmware and topology group. Plot gap lengths and
calendar heatmaps, not just one overall missingness percentage.

Distinguish missing telemetry, invalid values and legitimately offline equipment.
Track two denominators: expected monitored time and time actually scoreable by a
given detector. A detector cannot appear reliable merely by refusing difficult
periods. Use common-exposure comparisons as a sensitivity analysis.

### Distribution and relationship analysis

For each metric inspect quantiles, zero mass, tails, clipping, quantisation,
cohort differences and drift. Inspect same-device and within-cohort relationships
before pooled correlations: device offsets can create misleading global patterns.

Plot optical power alongside error measures, temperature alongside bias current,
and reboots alongside uptime and coverage. Distinguish downstream/upstream optical
paths. Treat correlations as evidence to investigate, not causal diagnoses.

For faults, plot pre-onset history, observable onset, impact and recovery with
measurement availability. Include randomly sampled healthy windows and challenging
normal cases to avoid reviewing only obvious faults.

### Seasonality workflow

1. Plot time-of-day and day-of-week median/quantile profiles on training data.
2. Inspect autocorrelation at daily and weekly lags using valid aligned pairs;
   report pair support and do not zero-fill absent intervals.
3. Compare profiles across successive training blocks and equipment cohorts.
4. Check whether missingness or operating schedules create the apparent pattern.
5. Test whether seasonality improves future prediction/residual stability on
   development data and reduces nuisance alerts at a fixed detection objective.

At 15-minute cadence there are 96 nominal samples per day and 672 per week, but
use elapsed timestamps rather than positional shifts when gaps occur. A weekly
baseline needs several independent weeks, not merely many observations from one
week. Six months cannot establish annual seasonality.

Throughput and temperature are candidates, not guaranteed seasonal metrics.
Optical power and bias current may have environmental patterns but their removal
must not hide meaningful degradation. Retain physical levels alongside residuals.

STL decomposition can help inspect training-only seasonal/trend components. A
decomposition over a complete train-and-test series is not a causal production
feature. For live scoring use a frozen seasonal profile or explicitly one-sided
updates. The forecasting reference explains STL; the causal restriction is our
operational requirement. [R4]

Deliver a feature decision table: metric, pattern, history support, stability,
proposed adjustment and evidence for inclusion or exclusion.

## 7. Feature engineering specification

Let x(i,m,t) denote metric m for entity i available at decision time t. Current
observations can be used once available. Historical reference windows must end
strictly before t. Every feature has a name, formula, unit, required inputs,
minimum history, missingness policy and time-availability rule.

### Initial feature families

| Family | Candidates | Interpretation |
|---|---|---|
| Physical levels | ONT/OLT received power, transmit power, temperature, bias current, voltage | Absolute state and operating margin |
| Recent change | Valid elapsed-time differences and short-window slopes | Abrupt shifts or gradual movement |
| Historical deviation | Difference from causal robust reference | Change relative to the same equipment |
| Errors | Transformed BER, FEC/CRC intensity and zero/nonzero indicators | Error activity with correct denominator |
| Stability | Uptime reset, reboot increment, recent reset frequency | Repeated interruption evidence |
| Seasonal deviation | Residual from supported historical seasonal reference | Departure from recurring normal behaviour |
| Peer evidence | Self-normalised peer median and fraction affected | Isolated versus shared changes |
| Availability | Feature validity, coverage and observation age | Whether a score is supported; not automatically a physical fault |

Begin with a compact subset covering optical level/change, error activity and
reset evidence. Add seasonal and peer families through ablations. Do not generate
every possible window/metric interaction just because computation permits it.

### Robust historical deviation

For valid observations in [t-W, t):

```text
baseline(i,m,t) = median(past valid observations)
scale(i,m,t) = max(1.4826 × MAD(past observations), metric_scale_floor)
z(i,m,t) = (x(i,m,t) - baseline(i,m,t)) / scale(i,m,t)
```

MAD is the median absolute deviation from the median. The 1.4826 factor gives a
normal-reference scaling; it does not make the residual Gaussian. Scale floors
must reflect measurement resolution or training evidence. Without them, almost
constant channels can produce arbitrarily large scores.

Use `max(0, -z)` for low-direction evidence, `max(0, z)` for high-direction
evidence, and `abs(z)` for two-sided change where justified. Preserve signed z
for interpretation. Operational limits and residual scores are separate signals.

For history coverage, count supported observation slots or covered intervals,
not raw rows that duplicates can inflate. Do not let an isolated old point imply
continuous support. Short/long windows such as 6 and 24 hours are initial
hypotheses; 7-day windows require evidence and enough usable warm-up history.

### Differences and slopes

Use a lag only when a past observation exists within a documented tolerance of
the requested time. Return missing when support is absent. Fit slopes against
actual elapsed hours with minimum point count and time span; report units such
as dB/hour. Never treat two points separated by a day as adjacent 15-minute data.

Evaluate gap-tolerant history separately from gap-tolerant alert persistence.
Allowing older observations into a historical summary does not justify connecting
alerts through an unobserved outage. The prior six-hour gap setting is an ablation
candidate, not an established physical rule.

### BER and count transformations

BER must stay within its documented support. Separate a true zero from missing,
below-detection-limit and vendor sentinel values. A proposed representation is a
zero indicator plus log10(max(BER, epsilon)) for valid values, with epsilon tied
to known resolution or a frozen training choice. Do not call BER a packet-loss
rate or combine pre- and post-FEC values without distinction.

For nonnegative interval counts, `log1p(count)` is suitable only when reporting
durations are comparable. If the interval duration is known, also consider
count/interval_seconds. Errors/traffic is an error fraction only with a valid,
matching denominator; throughput cannot automatically supply that denominator.

For a cumulative counter C:

```text
rate(t) = (C(t) - C(previous)) / elapsed_seconds
```

Compute this only across compatible continuous observations. If there is a reset,
unknown discontinuity or unsupported rollover, mark the rate unavailable. Apply
wraparound correction only with a documented modulus and plausible increment.
IETF interface models explicitly provide counter discontinuity information. [R5]

### Seasonally adjusted features

Estimate a median reference for the entity/cohort and comparable local-time bucket
using training history or permitted past updates. Store bucket sample counts and
independent-week support. Use a documented broader reference when fine buckets
are sparse, with a fallback flag. Do not manufacture zero residuals for cold starts.

Compare raw, nonseasonal historical and seasonal residual features. Calendar
encoding alone does not guarantee a detector learns seasonality. Keep the simpler
representation unless the additional family helps on development.

### Topology and peer features

Normalise each ONT against its own valid history before comparing peers; raw power
differences can reflect different path losses. At time t, use only peers available
by t and topology valid at t. Exclude the focal ONT from its peer reference.

Candidate group evidence includes the median signed residual, dispersion, number
of eligible peers and fraction exceeding a frozen evidence threshold. Require
both minimum peer count and coverage; two of two observed ONTs need not represent
two of twenty expected ONTs. Keep peer coverage in incident explanations.

Peer-relative residuals can hide a fault affecting an entire group. Preserve the
absolute self-deviation and group common change as separate features. Compare
sibling groups before claiming an upstream shared cause.

### Missing inputs and baseline adaptation

Do not replace missing inputs with a value meaning healthy. Start by comparing:
(a) a compact complete-input feature set; (b) training-median imputation with
explicit validity handling. Add missing indicators only as an ablation: they can
cause a model to detect collection regimes instead of faults.

Define minimum available inputs per detector and record abstentions. Fit all
learned transformations on training data only. Fitted preprocessing and the
model must be reused together at inference. [R6]

Initially freeze reference parameters. Later test causal adaptation with a stated
update schedule and protection against absorbing sustained degradation. Adaptation
must depend only on evidence available then, never future fault labels.

## 8. Model development and calibration

Use three baseline approaches first:

1. Documented operational limits plus persistence. No invented universal optical
   thresholds; source equipment limits or label the policy as exploratory.
2. Robust deviation/trend detector with a small, explicit aggregation of evidence.
3. Isolation Forest on the same compact causal feature set, with reproducible
   training samples and frozen missing-input treatment.

Consider supervised logistic regression or gradient boosting only after enough
independent reviewed incidents exist. Consider forecasting/deep models only after
the simpler baselines expose a specific limitation. Benchmark research supports
testing simpler methods seriously rather than presuming architectural complexity
will improve performance. [R7]

Weight or stratify training samples so long-lived/high-volume entities do not
dominate unintentionally. Record samples by entity, time, equipment group and
operating regime. Compare sensitivity to seed and sampling; preserve a fixed
evaluation population.

### Split roles

| Partition | Permitted use |
|---|---|
| Training | Fit model, imputation, scales and frozen references |
| Calibration fit | Set score thresholds using the declared workload policy |
| Calibration verification | Check the complete frozen incident policy on later data |
| Development | Inspect errors, compare candidates and choose the final approach |
| Holdout | One final evaluation after selection is frozen |

Use chronological boundaries. Historical input preceding a boundary may supply
causal warm-up context; fitting and future labels may not cross it. Explicitly
handle faults spanning boundaries, avoiding shared event families in training and
evaluation. For an unseen-group test, hold out complete relevant topology groups
rather than randomly selected sibling ONTs.

Within synthetic evaluation, time splitting is necessary but insufficient. Keep
parent trajectories and derived variants together, and add independent generator
seeds and scenario families. Repeated development tuning eventually requires a
new locked test set; reopening the same holdout does not create independent evidence.

### Threshold selection

Calibrate the full incident policy, not just a pointwise score quantile. Adjacent
measurements are dependent and a tail probability is not an incident rate.

Set the operational alert budget with the intended user. Estimate workload on
separate verification time, report uncertainty and reject thresholds with inadequate
tail support. With n calibration blocks, n × (1-q) describes expected empirical
tail support at quantile q; it is a diagnostic, not a guarantee of independence
or reliable generalisation. Define blocks by the process, not to maximise n.

For unlabelled calibration periods report total alert workload, not verified
false-positive rate. Select a candidate on development only after its required
coverage and workload checks pass. Keep an explicit diagnostic candidate when
none qualifies, but never export it as a selected deployment configuration.

## 9. Alerts, incidents and localisation

Implement a small state machine: inactive, pending, active and recovering. Specify
entry evidence, persistence, recovery threshold/duration, maximum permissible
observation gap and quiet period. Evaluate these settings jointly with the score.

At 15-minute cadence, two qualifying observations span at least 15 minutes between
the first and second readings, before reporting and processing delays. An online
incident begins when the policy can actually emit it. Store an estimated symptom
onset separately; do not backdate the alert to obtain better latency scores.

Group overlapping alerts by time and validated topology. Preserve child evidence
when producing a group incident. Avoid emitting both the parent and every child
as independent operator tasks unless that is the defined workflow. Apply quiet
periods carefully so a new event is not suppressed merely because the entity had
a previous fault.

Each incident should contain:

- Unique ID, emission time, estimated onset, latest evidence and recovery state.
- Affected ONTs, suspected scope and number of potentially impacted subscribers.
- Model/policy version, score and trigger conditions.
- Observed versus reference measurements, availability and group support.
- Localisation confidence limitations and alternative explanations.
- Suggested investigation grounded in evidence, rather than an unsupported diagnosis.

Large residuals explain observed deviations; they are not automatically feature
attributions for Isolation Forest and are not proof of a physical cause.

## 10. Evaluation and acceptance

### Ground truth and matching

Separate mechanism onset, independently defined observability and customer impact.
For real data, ticket-open time can lag actual onset; preserve this uncertainty.
For synthetic data, record how visibility and impact were defined. Do not derive
ground truth from the detector being evaluated.

Predeclare eligible temporal overlap, entity/scope compatibility and treatment of
early warnings. Use deterministic one-to-one matching where appropriate: one
long incident must not receive unlimited credit for unrelated repeated faults.
For overlapping faults, resolve matches consistently and retain the alternatives
for audit. Do not choose matching tolerances after seeing which candidate wins.

### Required metrics

| Metric | Definition and interpretation |
|---|---|
| Event recall | Matched faults / target faults; report both all-fault and prespecified observable-subset results |
| Prompt recall | Faults alerted within a declared deadline / eligible target faults |
| Pre-impact recall | Eligible faults with an alert before impact / faults with a defensible warning opportunity |
| Incident precision | Credited incidents / emitted incidents when labels are sufficiently complete |
| Nuisance workload | Unmatched plus duplicate incidents under the declared matching policy, per 1,000 monitored entity-days |
| Delay | Alert emission minus observable onset; report detected-only quantiles alongside recall and misses |
| Joint localisation recall | Correctly detected and localised shared faults / all eligible shared faults |
| Coverage | Valid measurement time and scoreable time / expected monitored time |
| Operator usefulness | Reviewed actionable proportion, review time and useful investigation changes |

Also report workload per scoreable entity-day, per port and as total queue volume.
Fix exposure definitions before comparison. A scoreable-only denominator must not
conceal low coverage. In incomplete real labels, unmatched means unverified, not
necessarily false; use a reviewed sample and report its sampling procedure.

Break results down by mechanism, severity, equipment group, topology level,
coverage and history support. Report confidence intervals. Wilson intervals are
a useful simple event-proportion summary under independence assumptions; shared
scenarios and temporal dependence require group/block resampling sensitivity.
With few independent groups, state uncertainty rather than resampling rows and
claiming enormous effective sample size.

Avoid point adjustment that credits an entire fault interval after one detection.
Published work shows that it can inflate performance. Recent metric research also
shows that evaluation properties differ; our operational metrics remain primary,
with any benchmark score clearly secondary. [R8, R9]

### Acceptance policy

Agree target faults, review budget, prompt deadline, minimum coverage and confidence
requirements with the operator. Do not carry the old numerical gates into the
restart without revisiting their business meaning. Do not relax them merely to
force selection. A failed gate can trigger investigation or an explicit scope
change, not silent deployment.

Run in shadow mode before operational use. Ask engineers to review a representative
sample of alerts and non-alert periods; reviewing only alerts cannot estimate
missed faults. Compare against the existing alarm workflow, not just another model.

## 11. Controlled experiment plan

| Experiment | Change | Question |
|---|---|---|
| E0 | Reproduce frozen previous runs | Can we recover the recorded behaviour? |
| E1 | Operational limits and persistence | What can a transparent baseline achieve? |
| E2 | Causal robust history | Does relative change improve on absolute limits? |
| E3 | Compact Isolation Forest on matched inputs | Does multivariate modelling add value? |
| E4 | History-gap treatment only | Is better history coverage worth its altered baseline behaviour? |
| E5 | Missing-input treatment only | Are improvements physical or driven by missingness patterns? |
| E6 | Seasonal features only | Do they reduce nuisance alerts without hiding faults? |
| E7 | Peer/group evidence only | Does shared-scope detection/localisation improve? |
| E8 | Combine only supported improvements | Do the gains survive interactions? |
| E9 | Independent scenarios and operator replay | Do gains extend beyond development templates? |

Use identical eligible periods, labels and matching rules for comparisons. Calibrate
each candidate by the same policy, not necessarily the same numerical threshold.
Record training/runtime cost and uncertainty. Prefer the simpler model when the
observed difference is not operationally meaningful.

## 12. Technical execution and reproducibility

Use Parquet for data, DuckDB for selective scans/aggregations and pandas/NumPy for
bounded feature/model operations. Reuse current dependencies; introduce another
engine only for a measured bottleneck. Six million wide rows are manageable with
care, but converting every metric into a long table can multiply row counts.

Process by entity or time with the required history overlap. Preserve necessary
group context for peer features. Score each observation once after overlapping
partitions are combined. Assert partitioned and unpartitioned equivalence on a
small fixture, including warm-up and boundary cases.

Sort explicitly, seed stochastic steps and avoid implicit local timezone changes.
Keep raw inputs immutable. Save fitted preprocessing, feature order, model,
threshold and incident policy together with their versions. Load model artifacts
only from trusted runs.

Suggested local outputs, excluded from Git:

```text
data/<dataset_id>/
    measurements.parquet
    topology.parquet
    faults.parquet             # when available
    service_windows.parquet    # when available/required
    manifest.json

outputs/<run_id>/
    resolved_config.yml
    manifest.json
    data_audit.json
    feature_audit.parquet
    candidate_comparison.parquet
    incidents.parquet
    fault_results.parquet
    metrics.json
    model.joblib
    selection_status.json
    report.html
```

Only successful selection writes `selected_configuration.json`. Names above are
proposed; compatibility with current frozen run formats should be handled explicitly.
Record Git revision, dirty-working-tree state, source/config hashes, dependency
versions, seeds and time boundaries. If code is modified locally, retain its diff
or source snapshot rather than claiming the commit hash fully identifies the run.

A local generated HTML report is a portable output, not a separate application.
Each notebook must run top-to-bottom in a fresh kernel. Thin orchestration cells
call package functions; no repeated modelling implementations in notebooks.

## 13. Focused verification

| Test file | Consequential behaviours to protect |
|---|---|
| `test_adapters.py` | Unit/time conversion, missing sentinels, scoped IDs, semantic mismatch rejection and provenance |
| `test_data.py` | Keys, duplicates, interval boundaries, topology validity, service exposure and split assignment |
| `test_synthetic.py` | Fixed-environment reproducibility, label/mechanism alignment, physically constrained outputs and independent scenarios |
| `test_features.py` | Causality, reset-safe rates, interval counts, gap handling, seasonal cold starts and peer contamination |
| `test_models.py` | Train-only fitting, feature order, scoreability, frozen inference and reload equivalence |
| `test_incidents.py` | Persistence, gaps, recovery, duplicates, shared grouping and true emission time |
| `test_evaluation.py` | Overlapping faults, duplicate alerts, exposure, abstentions, latency and localisation denominators |

The strongest causal test changes or appends future observations and verifies
that earlier features and scores do not change. A corresponding label-isolation
test changes evaluation labels and verifies that fitting/scoring inputs remain
unchanged. Compare event matching to manually calculated small examples.

Keep tests small and explicit. Avoid tests that only restate implementation or
assert arbitrary model accuracy on a conveniently generated fixture.

## 14. Implementation stages and review deliverables

1. Preserve the baseline and audit the original generator. Deliver documented
   semantics, reproducibility information and a list of faults in the data itself.
2. Establish the contract and adapters. Deliver validated small fixtures and a
   manifest for the existing dataset, without changing its values silently.
3. Implement notebook 01. Deliver EDA, missingness/seasonality evidence and the
   generator audit, with a decision on which tasks are observable.
4. Implement compact features and baseline detectors. Deliver causal tests,
   coverage reports and controlled comparisons in notebook 02.
5. Freeze the incident and evaluation contract. Deliver reviewed example matches
   and acceptance settings tied to an intended operational workflow.
6. Run ablations, qualify or stop, and perform final evaluation only after freeze.
7. Validate on real operator data in shadow mode. Document domain differences
   rather than treating synthetic accuracy as a deployment result.

For migration, map existing I/O/contract/adapters into data/adapter responsibilities;
EDA and diagnostics into diagnostic functions; scoring/detectors/selection into
the model workflow; alerts/localisation into incident handling; retain useful
evaluation functions. Keep existing tests until replacement coverage exists.
Reduce the notebook sequence only after replay parity checks on a representative
fixture. Do not delete frozen artifacts or source data as repository cleanup.

## 15. Decisions still requiring evidence

- Original generator implementation, version and provenance of the supplied data.
- Confirmed metric definitions, including FEC/CRC intervals and clipping.
- Intended operator, actionable fault families and existing alarm baseline.
- Actual reporting delay, maintenance records and topology reliability.
- Fault observability/impact definitions and completeness of labels.
- Appropriate history windows, seasonal support and gap policies.
- Meaningful alert budget, prompt deadline and acceptance uncertainty.
- Availability of real PON telemetry and engineering review.

These are explicit research/operational decisions, not configuration values to
guess until a selection gate passes.

## 16. Configuration examples and the execution contract

These examples explain ownership of settings. They are not executable contracts
for the current repository and are not validated operating choices. Dates, limits,
history support and generator parameters must be settled through the earlier stages.

### `experiment.yml`: one reproducible comparison

```yaml
run_id: development_baselines_001
data:
  path: data/synthetic_pon_frozen
  schema_version: 1

splits:
  timezone: UTC
  train_end: "2025-03-01T00:00:00Z"
  calibration_fit_end: "2025-03-15T00:00:00Z"
  calibration_verify_end: "2025-04-01T00:00:00Z"
  development_end: "2025-05-01T00:00:00Z"

features:
  history_hours: [6, 24]
  minimum_coverage: 0.8
  seasonal_features: false
  peer_features: false

models:
  candidates: [robust_deviation, isolation_forest]
  random_seed: 42

calibration:
  incident_budget_per_1000_entity_days: null
  # Must be explicitly set before candidate qualification.

incidents:
  consecutive_observations: 2
  quiet_period_minutes: 60

evaluation:
  prompt_deadline_minutes: null
  # Must be explicitly defined for prompt-recall evaluation.
  allow_holdout: false
```

Intervals are left-inclusive and right-exclusive. The first interval begins at
the eligible dataset start; the holdout begins at `development_end`. Service
windows, warm-up and boundary-spanning events need the separate policies described
above. Null acceptance settings mean incomplete design, not unlimited workload.

Operational limits are omitted from this example because their source is not yet
confirmed. When available, add the operational-limit baseline to the candidates.

### `metrics.yml`: meaning before transformation

```yaml
schema_version: 1
metrics:
  rx_power_dbm:
    unit: dBm
    kind: gauge
    measurement_location: ont_receiver
    aggregation: instantaneous
    adverse_direction: low

  fec_count:
    unit: count
    kind: interval_count
    interval_seconds: 900
    aggregation: sum_over_interval
    semantic_status: awaiting_generator_confirmation
```

The FEC entry illustrates the existing source assumption. Do not apply its duration
or semantics to a different vendor/source without verification. A measurement
dictionary may need source-specific definitions where native meanings differ.
Detector-specific transforms and learned scales are recorded with the experiment.

### `synthetic.yml`: generated-world assumptions

Record start/end, reporting cadence, topology sizes, equipment groups, separate
random seeds, normal-process parameters, fault distributions and measurement
effects. Do not invent numerical distributions before inspecting the original
generator. Each parameter should have a unit, explanation and evidence status:
operator-derived, equipment specification, published experiment or assumption.

### End-to-end execution

```text
load configuration and verify it is complete for the requested stage
generate or adapt source data
validate data and persist the dataset manifest
assign frozen chronological and scenario partitions
run EDA and synthetic audits on permitted partitions
build causal features and audit availability
fit each candidate on training data
calibrate each incident policy on calibration-fit data
verify its workload on calibration-verification data
compare eligible candidates on development data
write selected configuration or an explicit STOP report
evaluate a frozen selection on holdout only when explicitly enabled
```

Before a run begins, check that its output directory does not contain another run.
Resume only through an explicit compatibility check; otherwise use a new run ID.
The report should identify the dataset, model and policy versions on its first
page and distinguish results from proposed settings and unresolved assumptions.

## 17. References and how they support this design

R1. Sica et al. (2026), **Detection, identification, and localization of faults in
PONs using joint OLT and OTDR telemetry data**, JOCN 18, D44–D55.
Supports comparing interpretable heuristics and ML with the sensing actually
available; does not validate our telemetry-only localisation.
https://opg.optica.org/jocn/abstract.cfm?uri=jocn-18-9-D44

R2. Broadband Forum, **OB-BAA Performance Monitoring Data Collection**.
Supports translating source-specific measurements into shared definitions before
analysis. It does not prescribe our Python package or file layout.
https://obbaa.broadband-forum.org/architecture/pm_collector/

R3. Broadband Forum, **TR-385: YANG Modules for PON Management**.
Reference for PON management semantics and identity; actual vendor exports still
require verification.
https://www.broadband-forum.org/technical-library/?number=TR-385

R4. Hyndman and Athanasopoulos, **Forecasting: Principles and Practice**, third
edition, section on STL decomposition. Reference for seasonal/trend exploration;
our online causal implementation must be designed separately.
https://otexts.com/fpp3/stl.html

R5. Bjorklund (2018), **RFC 8343: A YANG Data Model for Interface Management**.
Reference for interface statistics and counter discontinuity semantics.
https://www.rfc-editor.org/rfc/rfc8343.html

R6. scikit-learn, **Common pitfalls and recommended practices**.
Supports fitting learned preprocessing only on training data and reusing the
same transformations at inference. Our chronological/scenario split details are
additional requirements of this project.
https://scikit-learn.org/stable/common_pitfalls.html

R7. Liu and Paparrizos (2024), **The Elephant in the Room: Towards A Reliable
Time-Series Anomaly Detection Benchmark**, NeurIPS Datasets and Benchmarks.
Supports benchmark integrity and serious comparison with simple models, not a
claim that a particular baseline will win on PON data.
https://papers.nips.cc/paper_files/paper/2024/hash/c3f3c690b7a99fba16d0efd35cb83b2c-Abstract-Datasets_and_Benchmarks_Track.html

R8. Kim et al. (2022), **Towards a Rigorous Evaluation of Time-Series Anomaly
Detection**, AAAI. Supports caution about point-adjusted performance.
https://ojs.aaai.org/index.php/AAAI/article/view/20680

R9. Wagner et al. (2026), **Formally Exploring Time-Series Anomaly Detection
Evaluation Metrics**, AISTATS. Supports making desired evaluation properties
explicit; does not supply our operational budgets or confidence gates.
https://proceedings.mlr.press/v300/wagner26a.html

R10. Silva et al. (2022), **Learning Long-and Short-Term Temporal Patterns for
ML-driven Fault Management in Optical Communication Networks**. Domain evidence
for temporal modelling, without establishing our chosen windows or gap policy.
https://www.iris.sssup.it/handle/11382/544012

R11. Liu, Ting and Zhou (2008), **Isolation Forest**, ICDM. Foundational method
reference for the multivariate baseline, not evidence of PON deployment accuracy.
https://research.monash.edu/en/publications/isolation-forest/

R12. Feng et al. (2025), **TelecomTS: A Multi-Modal Observability Dataset for Time
Series and Language Analysis**. Useful context for normal burstiness and absolute
scale; the 5G dataset is not PON ground truth.
https://arxiv.org/abs/2510.06063

The repository structure, output schemas, experiment sequence and example feature
families are proposed engineering choices. References justify individual principles;
only controlled experiments and operational review can justify the final settings.
