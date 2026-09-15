# Company-agnostic Telecom anomaly detection methodology

| Document field | Value |
|---|---|
| Revision | 14 September 2026 |
| Implementation baseline | `SPEC-CORE 1.0.0`, `SPEC-EVAL 1.0.0`, pack interface `1.0.0`, detector core `4.5.0`, evaluator core `4.3.0` |
| Primary domain | Fixed-access PON/ONT telemetry |
| Audience | Data scientists, statisticians, ML engineers, data engineers, Telecom SMEs, and technical reviewers |
| Status | Research-grade implementation with production-oriented controls; an operator pilot is still required before a production-performance claim |

This document describes notebooks `00`–`12` and the shared implementation
under `src/telco_anomaly`. It deliberately distinguishes current behaviour
from recommended production extensions. **Implemented** means present in the
current pipeline. **Planned** or **required before production** means that the
capability is not part of current performance results.

---

## 1. Executive summary

The system answers three operational questions in sequence:

1. Is current telemetry unusual relative to the asset's own history, its
   contemporaneous peers, or a shared physical group?
2. Do repeated unusual scores form one operational incident rather than many
   point alerts?
3. What is the smallest observable physical scope supported by the incident
   footprint: one ONT, a splitter, a PON port, or an OLT?

It is an unsupervised anomaly-detection system. It is not yet a fault-type
classifier, failure-prediction model, or causal root-cause engine. Normal
references are learned without fault labels. Alert thresholds are calibrated
against an explicit incident-workload budget. Development labels compare a
small predeclared portfolio only after thresholds are fixed, and holdout truth
remains locked until the complete configuration is frozen.

```mermaid
flowchart LR
    A[Native operator or public data] --> B[Source audit]
    B --> C[Dataset-specific adapter]
    C --> D[Vendor-neutral SPEC-CORE]
    C --> E[Physically separate SPEC-EVAL]
    D --> F[Calibration-only EDA]
    F --> G[Causal feature engineering]
    G --> H[Label-free model fitting]
    H --> I[Late-calibration thresholds]
    I --> J[Alerts and incident consolidation]
    J --> K[Topology localisation]
    E --> L[Development or locked-holdout evaluator]
    K --> L
    L --> M[Metrics, uncertainty, model card]
```

“Company-agnostic” has a precise, limited meaning:

- operators map native fields to shared semantic metric identifiers;
- measurement behaviour is declared rather than guessed from a name;
- reference levels, variability, score distributions, and thresholds are
  fitted locally;
- the canonical contract, detector mechanics, incident logic, and evaluator
  remain shared.

It does **not** mean that raw PON, cellular RAN, and backbone-optical values can
be pooled into one fitted distribution. Those domains require separate metric
packs, reviewed feature policies, and local calibration.

---

# Part I — Product and evidence contract

## 2. Product boundary

### 2.1 Implemented capabilities

The implemented path provides:

- source-schema and observable-data-quality auditing;
- vendor-neutral long-form telemetry;
- explicit quality, clipping, collection gaps, episodes, and topology;
- causal, metric-aware feature engineering;
- rapid self-history, persistent-drift, peer-relative, and physical-group
  common-mode scoring;
- statistical and multivariate challenger models;
- workload-calibrated alert thresholds;
- persistence, recovery, and alert-to-incident consolidation;
- topology-footprint localisation with explicit ambiguity;
- event, workload, delay, and localisation evaluation;
- immutable manifests, fingerprints, and holdout-opening evidence.

### 2.2 Explicitly out of scope

The project does not currently claim:

- supervised fault-family classification;
- future failure probability, remaining useful life, or prediction horizon;
- causal root-cause proof;
- customer-impact or monetary-risk optimisation;
- universal calibration across operators or Telecom domains;
- production effectiveness established from synthetic data alone.

An anomaly score is evidence that an observation is surprising under the
frozen reference. It is not automatically a fault probability or a causal
label.

### 2.3 Operational unit and principal estimands

The primary unit is a consolidated incident, not an anomalous row. For
evaluation exposure \(E\), false incident count \(K\), scoreable fault count
\(N_F\), and detected fault count \(D_F\):

$$
\widehat{\lambda}_{\mathrm{false}}=\frac{K}{E},
\qquad
\widehat R_{\mathrm{event}}=\frac{D_F}{N_F}.
$$

The current research workload target is

$$
\lambda_0=0.01\ \text{false incidents per entity-day}
=10\ \text{per 1,000 entity-days}.
$$

For 400 continuously monitored ONTs, this corresponds to four false incidents
per fleet-day. It is a predeclared research operating point, not an assumed
production SLA. Operations staff must set the real budget in a pilot.

The principal development and holdout estimands are:

1. event recall at the declared workload;
2. false incidents per entity-day;
3. incident precision where trustworthy labels exist;
4. time from first observable fault evidence to operational alert firing;
5. joint detection-and-localisation recall;
6. predicted-versus-affected footprint overlap;
7. score availability and unscoreable exposure.

Anomalous-row rate, raw alert count, in-sample reconstruction loss, and
timestamp-level accuracy are diagnostics. None answers the operator workload
and fault-event question by itself.

## 3. Evidence hierarchy and allowed claims

| Evidence source | Role | Supported claim | Unsupported claim |
|---|---|---|---|
| Synthetic PON/ONT fixture | Primary engineering and labelled development fixture | Software correctness, injected-mechanism response, truth isolation, topology mechanics | Real prevalence, real workload, operator effectiveness |
| Commercial RAN PM counters | Real-operator portability qualification | Real cadence, heterogeneity, seasonality, gaps, scalability, unlabelled workload | PON recall or PON localisation |
| Microsoft optical telemetry | Real optical-behaviour qualification | Slow drift, channel reference stability, common movement, unlabelled workload | Recall, because outage days were removed |
| Optical-failure testbed | Controlled labelled response test | Response to induced physical degradation | Field prevalence, fleet workload, field localisation |
| Future operator pilot | Production evidence | Prospective workload, usefulness, failure coverage, workflow fit | Claims outside the pilot population without further validation |

Raw datasets are never concatenated into one training table. Each source is
adapted, calibrated, and reported separately. Cross-source evidence tests
portability of the method, not equality of the distributions.

### 3.1 Statistical notation

| Symbol | Meaning |
|---|---|
| \(e\) | monitored entity, currently an ONT in the primary PON experiment |
| \(g\) | physical topology group such as splitter, PON port, or OLT |
| \(j\) | semantic metric or derived feature |
| \(t\) | event time; all model windows are causal in event time |
| \(x_{e,t,j}\) | observed or transformed value |
| \(r_{e,t,j}\) | value standardized by the frozen entity/global reference |
| \(S_{c,e,t}\) | anomaly score from channel \(c\) |
| \(\tau_{c,q}\) | channel threshold at candidate calibration quantile \(q\) |
| \(\Delta_j\) | expected cadence for metric \(j\) |
| \(E\) | monitored exposure in entity-days |
| \(F\) | scoreable labelled fault set |
| \(C\) | consolidated incident set |

Unless stated otherwise, intervals are half-open: the start is included and
the end excluded. Robust medians and quantiles are empirical sample estimates.
Missing values do not contribute numerical evidence; availability is measured
and reported separately.

---

# Part II — Data architecture and truth isolation

## 4. Three data layers

### 4.1 Native source

The native source is retained as supplied by the generator, operator, or
publisher. It may be wide, use vendor-specific names, mix identifiers with
measurements, and store labels beside observations. Loadability does not make
it model-ready.

### 4.2 Prepared dataset pack

The dataset-specific adapter produces:

```text
prepared/<dataset>/<pack_run>/
├── PACK-CORE/     # model-observable adapter output
├── PACK-EVAL/     # source truth, if available
├── SPLITS/        # time and infrastructure partitions
└── pack_manifest.json
```

The prepared pack answers: *what did this adapter understand, preserve,
exclude, and declare?*

### 4.3 Canonical model run

The shared canonical builder produces:

```text
core/<dataset>/<core_run>/
├── SPEC-CORE/     # detector-readable canonical data
├── SPLITS/        # experimental partitions, not predictors
└── run_manifest.json

evaluation/<dataset>/<truth_run>/
├── development/
└── holdout_locked/
```

The detector runtime receives `SPEC-CORE`, never `SPEC-EVAL`. This layer
answers: *what is the model permitted to see?*

## 5. Canonical contract

### 5.1 Mandatory `SPEC-CORE` tables

| Table | Purpose | Key fields |
|---|---|---|
| `telemetry` | Long-form observations | `event_ts`, `entity_id`, `episode_id`, `metric_id`, `value`, `quality_code` |
| `metric_catalogue` | Meaning and allowed analytical treatment | kind, unit, cadence, direction, transform, bounds, censoring, reset policy, scale floor |
| `entity_registry` | Entity identity and observed validity | type, vendor/model if known, validity and observed range |
| `observation_episodes` | Recording contexts stateful calculations may not bridge | episode start/end and basis |
| `collection_gaps` | Expected observations absent inside an active periodic episode | entity, episode, metric, gap start/end, cadence |

Optional detector-visible tables are:

- `topology_memberships`, required for peer and localisation channels;
- `operational_events`, for genuinely observable events such as dying gasp or
  LOS. Such events are context and are never fabricated from continuous
  telemetry.

### 5.2 `SPEC-EVAL` tables

| Table | Evaluator purpose |
|---|---|
| `fault_events` | One row per fault, including timing and declared domain |
| `fault_entity_intervals` | Affected entities and active intervals |
| `condition_states` | Interval condition labels where the source is state-labelled |
| `tickets` | Optional operational reports and links |

No evaluator field is a detector feature. A ticket can support evaluation or
qualitative review, but its presence does not make it safe at inference time.

## 6. Measurement semantics

Each metric declares:

- semantic `metric_id` and `entity_type`;
- measurement kind, unit, aggregation semantics, and sampling mode;
- expected cadence where scheduled;
- direction: `high_bad`, `low_bad`, `two_sided`, or `contextual`;
- transform;
- defensible validity bounds only when known;
- censoring and source-specific clipping;
- counter reset policy where relevant;
- minimum numerical scale;
- seasonality candidacy and peer eligibility.

The PON registry contains gauges, bounded fractions, interval counts, and
cumulative counters. All current PON metrics have 900-second cadence, but the
contract permits metric-specific cadence.

The source adapter may carry broad `peer_eligible` metadata for compatibility,
but the model does not treat that as sufficient permission. Peer-relative
scoring uses the explicit metric allow-list in `configs/topology.yml`. A new
operator must review that list against its own measurement semantics before a
metric can enter peer or group evidence.

### 6.1 Why measurement kind matters

For an instantaneous gauge, a difference approximates a level change. For an
interval count, the value already aggregates opportunities during an interval.
For a cumulative counter, the level is usually less informative than a
reset-safe increment. One generic transform would change the physical meaning.

### 6.2 Censoring and clipping

If a source reports

$$
y_t=\min(x_t,c),
$$

then an observation at \(c\) means \(x_t\ge c\), not \(x_t=c\). The synthetic
PON `optical.fec_count` ceiling is source-specific at 5,000,000. Those rows are
marked `clipped` and excluded from numerical health-feature fitting. The exact
underlying value is not invented.

Unknown physical limits remain null in the catalogue rather than being added
to make validation look complete.

## 7. Missingness, episodes, and expected reporting

Three cases remain distinct:

1. **Row present, value null or invalid.** The source emitted an observation,
   but the measurement is unusable.
2. **Expected timestamp absent inside an active episode.** This is a collection
   gap for a periodic metric.
3. **Entity outside its valid or observed episode.** No report is expected and
   no collection gap is created.

For consecutive valid observation times \(t_{i-1},t_i\) and cadence \(\Delta\),
a gap is recognised when

$$
t_i-t_{i-1}>1.5\Delta.
$$

The factor 1.5 permits modest timing jitter but prevents long absence from
being interpreted as continuous monitoring. For irregular or event-driven
sources, expected gaps cannot be inferred from cadence alone.

A sensor that never reports cannot be distinguished from “not installed” by
telemetry alone. The adapter requires source capability metadata when it
exists.

## 8. Truth-isolation controls

Truth isolation has three layers.

### 8.1 Schema and routing control

Model-facing columns are checked against prohibited identifiers such as
`fault_id`, `fault_type`, `label`, `class`, `root_cause`, `gt_*`, and `truth_*`.
Generator expectations and business weights are also excluded from health
features.

### 8.2 Translator invariance

Two native fixtures are built: an original source with truth and a redacted
source with truth columns and files removed. Logical hashes of model-visible
content must match:

$$
H(\mathrm{CORE}_{\mathrm{original}})
=H(\mathrm{CORE}_{\mathrm{redacted}}).
$$

Logical content hashes are preferred to raw Parquet byte equality because
innocent metadata can change file bytes.

### 8.3 Runtime negative control

A deliberately leaky reader must work when evaluation is mounted and fail
when it is absent. This proves that the isolation harness can detect the
relevant leak. It is deployment evidence; translator invariance is the primary
leakage proof.

This is a governance and software boundary, not an OS security claim. A person
with storage access can deliberately open truth; the system prevents accidental
analytical leakage and records holdout access.

## 9. Topology model

The current physical hierarchy is:

```text
OLT
└── PON port
    └── splitter L1
        └── splitter L2
            └── ONT
```

`geo_cluster` is geographic context, not physical containment, and is excluded
from physical localisation.

Membership rows are effective-dated in the contract. Current scoring fails
closed if an entity has more than one effective membership at one group type
within the scored data, but it does not yet time-join membership at every event
timestamp. **Required before an operator claim:** event-time topology joins so
re-homing cannot be interpreted using a later inventory.

---

# Part III — Experimental design and phase map

## 10. Chronological partitions

The primary experiment is chronological:

- **calibration:** label-free reference fitting and threshold calibration;
- **development:** controlled comparison of predeclared portfolios;
- **holdout:** one final evaluation after configuration freeze.

No random row split is permitted. Random rows would put adjacent,
autocorrelated measurements from the same entity on both sides and create an
optimistic estimate.

Calibration is divided by elapsed time. The early 70% fits references and
models. The remaining 30% is then split equally into threshold-estimation and
workload-verification slices:

$$
\mathcal C=\mathcal C_{\mathrm{fit}}\cup
\mathcal C_{\mathrm{threshold}}\cup\mathcal C_{\mathrm{verify}},
\qquad
\mathcal C_i\cap\mathcal C_j=\varnothing\quad(i\ne j),
$$

where the slices remain chronological. Threshold values are estimated on
\(\mathcal C_{\mathrm{threshold}}\); label-free incident workload is checked on
\(\mathcal C_{\mathrm{verify}}\).

Notebook 04 and Notebook 05 use the same timestamp cutoff, derived from the
observed calibration span. EDA stops before that cutoff; feature fitting ends
there, and threshold calibration begins there. The two late-calibration halves
are therefore unseen by both model fitting and EDA design work before they
estimate thresholds and verify workload.

A secondary whole-infrastructure split can hold out complete OLT groups. It
tests transfer across infrastructure but does not replace the chronological
primary evaluation.

## 11. Notebook and phase map

| Phase | Notebook | Label access | Main output | Gate |
|---|---|---:|---|---|
| Contract | `00_PROJECT_CONTRACT` | None | Frozen configs and claims | Versions and prohibited fields valid |
| Source audit | `01_DATA_QUALITY_AUDIT` | None | Source inventory and audit evidence | No unreviewed model column |
| Canonicalisation | `02_CANONICAL_DATA_MODEL` | Adapter may stage labels separately | Truth-unmounted `SPEC-CORE` | Contract and content audit pass |
| Split/truth lock | `03_SPLITS_AND_TRUTH_LOCK` | Evaluator builder only | Development and locked holdout stores | Invariance and negative controls pass |
| EDA | `04_CALIBRATION_EDA` | None | Calibration evidence and EDA decisions | Limitations recorded |
| Features | `05_FEATURE_ENGINEERING` | None | Fit, late-calibration, and development features | Causality and schema checks pass |
| Primary fit | `06_PRIMARY_UNSUPERVISED_MODEL` | None | Model bundle, threshold/workload/development scores, thresholds | Calibration artifacts complete |
| Development selection | `07_CHALLENGER_MODELS` | Development only | Comparison and deployable/STOP decision | All gates pass or fail closed |
| Incidents | `08_ALERTS_INCIDENTS_AND_DYING_GASP` | None | Persistent alerts and incidents | Frozen selection required unless diagnostic |
| Localisation audit | `09_TOPOLOGY_LOCALISATION` | None | Location, alternatives, ambiguity | Physical/equivalence checks pass |
| Locked evaluation | `10_LOCKED_EVALUATION` | Development or explicitly opened holdout | Metrics, matches, intervals, receipt | Hash/replay/ledger checks pass |
| External qualification | `11_PUBLIC_DATASET_VALIDATION` | Source-dependent | Separate public-source evidence | Claim restricted to source capability |
| Demonstration | `12_INFERENCE_DEMO_AND_MODEL_CARD` | Frozen outputs only | Incident queue, replay, model card | Deployability status remains visible |

## 12. Phases 00–03 in detail

### Phase 00 — project contract

**Purpose:** freeze the question before seeing performance.
**Actions:** load version-controlled product, dataset, metric, topology,
feature, and alert policies; verify claim boundaries and prohibited fields;
record hashes.
**Output:** machine-readable contract receipt. No data or truth is opened.

### Phase 01 — observable source audit

**Purpose:** establish what the source actually contains.
**Actions:** inventory files and columns; classify measurement, identity/time,
topology, excluded generator, and prohibited truth fields; assess timestamp
validity, cadence, duplicates, nulls, sentinels, clipping, coverage, and
topology completeness.
**Output:** compact evidence tables and source manifest. An unreviewed column
stops the run.

### Phase 02 — canonical data model

**Purpose:** translate native data into semantic, detector-readable data.
**Actions:** apply reviewed native-to-semantic mappings; convert units only
when justified; preserve invalid/clipped quality; construct entities, episodes,
gaps, and topology; write immutable Parquet and content fingerprints.
**Output:** prepared pack, splits, and truth-unmounted `SPEC-CORE`.

### Phase 03 — splits and truth lock

For fault \(f\), define its reference time as

$$
r_f=\begin{cases}
t_{\mathrm{observable},f},&\text{if known},\\
t_{\mathrm{onset},f},&\text{otherwise}.
\end{cases}
$$

For affected entity \(e\), scoreability begins and ends at

$$
a_{fe}=\max(r_f,t^{\mathrm{interval\ start}}_{fe},t^{\mathrm{observed\ from}}_e),
$$

$$
b_{fe}=\min(t^{\mathrm{interval\ end}}_{fe},t^{\mathrm{observed\ to}}_e,
t^{\mathrm{event\ end}}_f).
$$

The pair is scoreable only when \(b_{fe}>a_{fe}\). A fault crossing time
partitions is marked cross-partition and published into none. The phase writes
development truth, sealed holdout truth, split manifests, and isolation-test
evidence.

---

## 12.1 Phase execution cards for notebooks 04–12

### Phase 04 — calibration-only EDA

**Reads:** calibration `SPEC-CORE` only.
**Computes:** cadence and coverage, quality rates, robust distributions,
chronological behaviour, dependence, ACF/PACF, stationarity diagnostics,
seasonality evidence, and peer-group sufficiency.
**Writes:** compact tables, plots, EDA decisions, and an input fingerprint.
**Gate:** limitations and unsupported analyses remain explicit; no label or
holdout path is opened.

### Phase 05 — feature engineering

**Reads:** calibration and development `SPEC-CORE`, metric catalogue, feature
policy, and EDA decisions.
**Computes:** quality-masked base transformations, gap-safe differences,
exact lags, trailing robust histories, activity summaries, and approved
seasonal differences.
**Writes:** separate early-fit, late-calibration, and development feature files
plus schema/provenance metadata. Phase 06 divides late-calibration scores into
disjoint threshold-estimation and workload-verification halves.
**Gate:** prefix-invariance, partition boundaries, and prohibited-field checks
must pass.

### Phase 06 — unsupervised fitting and score calibration

**Reads:** feature files, topology, feature/alert policies; no fault truth.
**Computes:** frozen robust references, self residuals, rapid and CUSUM scores,
peer and group statistics, dispersion, PCA, five Isolation Forest variants,
and late-calibration block-max threshold candidates.
**Writes:** model bundle, score files, threshold provenance, score-availability
evidence, and run manifest.
**Gate:** fitted artifacts and every requested partition must be complete; a
channel without adequate support is disabled rather than imputed into success.

### Phase 07 — matched-workload development selection

**Reads:** frozen scores and thresholds, late-calibration exposure, and
development truth only.
**Computes:** a label-free workload audit for every threshold, persistent
alerts, consolidated incidents, one-to-one development matching, uncertainty,
ablations, and selection gates across calibration-admissible operating points.
**Writes:** candidate comparison, diagnostics, and either one frozen selection
or an explicit STOP result.
**Gate:** no research detection configuration is selected unless all workload,
availability, fault-count, and recall-evidence gates pass.

Every threshold candidate is retained in the label-free calibration workload
audit. Only thresholds with adequate tail support and a passing independent
workload bound reach development. Development may select among those operating
points because it is the validation set; the locked holdout remains the sole
final performance test. The least strict passing threshold is also retained as
a separate label-free fallback for an operator with no development labels.

### Phase 08 — alerts, incidents, and observable event context

**Reads:** frozen development scores, selected thresholds/policy, topology, and
observable operational events; no fault labels.
**Computes:** persistence/recovery state, alert intervals, incident
consolidation, threshold-relative evidence, and optional dying-gasp context.
**Writes:** alerts, incidents, incident membership, and evidence provenance.
**Gate:** a deployable selection is required unless the run is explicitly
labelled diagnostic-only.

### Phase 09 — topology localisation audit

**Reads:** frozen incidents and physical topology; no fault truth.
**Computes:** native common-mode scope, smallest common physical footprint,
alternatives, observable descendants, and topology equivalence.
**Writes:** localised incident view and ambiguity evidence.
**Gate:** geographic context cannot become a physical root, and ambiguity must
not be silently resolved.

### Phase 10 — locked evaluation

**Reads:** frozen selection and hashes plus development truth by default; it
opens holdout only under the explicit two-flag procedure.
**Computes:** replay verification, one-to-one matches, event/workload/delay and
localisation metrics, confidence intervals, and fault/domain breakdowns.
**Writes:** immutable evaluation tables, receipt, and holdout ledger entry.
**Gate:** any changed configuration, failed replay, mismatched hash, or
unauthorized holdout request stops the run.

### Phase 11 — public dataset qualification

**Reads:** one separately acquired public source and its reviewed adapter
policy.
**Computes:** only analyses supported by that source—unlabelled workload for
RAN/Microsoft optical, controlled response for the optical testbed.
**Writes:** a distinct qualification report; it never augments PON training.
**Gate:** source terms and mapping approval must be satisfied, and the report
cannot claim labels or topology that the source lacks.

### Phase 12 — inference demonstration and model card

**Reads:** frozen outputs only.
**Computes:** operator incident queue, chronological evidence replay, channel
and feature explanation, data-quality status, and model limitations.
**Writes:** demonstration artifacts and model card.
**Gate:** diagnostic/nondeployable status remains visible in every display.

---

# Part IV — Calibration-only EDA

## 13. Purpose and boundary

Notebook 04 studies model-visible behaviour using calibration only. It never
mounts faults or tickets. Its purpose is to justify reference behaviour and a
compact feature policy, not to search visually for features that reproduce
known fault intervals.

The analysis remains time ordered. Long gaps are not silently filled, and
series are summarized at entity–episode–metric level before population
aggregation.

## 14. Coverage and value quality

For a periodic entity–episode–metric series \(s\):

$$
N_{\mathrm{expected},s}
=1+\left\lfloor\frac{t_{\max,s}-t_{\min,s}}{\Delta_s}\right\rfloor,
$$

$$
\mathrm{coverage}_s=
\frac{N_{\mathrm{observed},s}}{N_{\mathrm{expected},s}}.
$$

Value validity and clipping are separate:

$$
\mathrm{valid\ rate}_s
=\frac{N_{\mathrm{measured},s}}{N_{\mathrm{rows},s}},
\qquad
\mathrm{clipped\ rate}_s
=\frac{N_{\mathrm{clipped},s}}{N_{\mathrm{rows},s}}.
$$

This distinction matters. High timestamp coverage with invalid values is a
sensor-quality problem; low timestamp coverage is a collection problem.

The EDA displays coverage/missingness heatmaps and chronological examples. It
does not label all unobserved time as healthy or anomalous.

## 15. Robust distribution and drift summaries

For valid values \(x_1,\ldots,x_n\):

$$
\widetilde x=\operatorname{median}(x_i),
\qquad
\operatorname{MAD}=\operatorname{median}|x_i-\widetilde x|,
$$

and the IQR-based Gaussian-equivalent scale is

$$
s_{\mathrm{IQR}}
=\frac{Q_{0.75}-Q_{0.25}}{1.349}.
$$

Histograms and robust boxplots are paired with series plots, rolling medians,
and rolling IQRs. A static histogram alone can hide regime changes and serial
dependence.

The EDA also examines:

- zero inflation and discrete mass points;
- frozen or nearly constant signals;
- negative differences in cumulative counters;
- first-half versus second-half robust-level change;
- association between local level and local scale;
- vendor/model heterogeneity when metadata is available.

These are evidence summaries. They do not automatically remove a metric or
declare a fault mechanism.

## 16. Dependence, stationarity, and seasonality

### 16.1 Autocorrelation and stationarity

ACF and PACF are calculated only on sufficiently long regular segments.
Augmented Dickey–Fuller and KPSS results are reported as diagnostics, not
automatic feature-selection rules. Structural breaks, missingness, and large
sample sizes can make simple p-values misleading.

Level and first-difference Spearman correlations are reported separately.
High level correlation may reflect common baseline or drift; correlation of
differences is stronger evidence of synchronous change.

### 16.2 Seasonal decomposition

For a sufficiently regular candidate series, robust STL represents

$$
x_t=T_t+S_t+R_t.
$$

If scale grows with level and a log transform is physically defensible, STL is
applied to \(\log(1+x_t)\), approximating multiplicative structure on the
original scale.

Seasonal strength is summarized as

$$
F_S=\max\left(0,1-
\frac{\operatorname{Var}(R_t)}
{\operatorname{Var}(S_t+R_t)}\right).
$$

A seasonal feature is approved only when enough complete cycles exist, the
pattern repeats beyond one series, and an exact causal lag can be computed. A
plot is supporting evidence, not a selection rule.

### 16.3 EDA independence qualification

EDA decisions are saved with the canonical fingerprint and never use labels.
The current design uses only the early calibration-fit interval and records its
exact cutoff. Feature engineering reuses that cutoff, so EDA cannot inspect
the later threshold-estimation or independent workload-verification slices.
Repeated manual EDA changes after seeing development results remain a residual
source of selection bias and must be versioned as a new research cycle.

---

# Part V — Causal feature engineering

## 17. Causal construction rule

Every feature at time \(t\) must be measurable from observations available no
later than \(t\):

$$
\phi_t=f\{x_u:u\le t\}.
$$

Centered windows, backward filling, whole-series normalization, and future
interpolation are prohibited. Features are built separately per entity and
episode. Stateful operations restart at episode boundaries and, where
implemented, across long collection gaps.

## 18. Quality mask and safe difference

Non-finite, out-of-bound, and clipped values are unavailable to numerical
health features. Their quality remains visible for data-quality monitoring.

For two consecutive observations:

$$
\Delta x_t=\begin{cases}
x_t-x_{t-1},&0<t-t_{t-1}\le1.5\Delta,\\
\mathrm{NA},&\text{otherwise}.
\end{cases}
$$

A value after a long outage is therefore not treated as an instantaneous
equipment jump.

## 19. Metric-aware base transformations

### 19.1 Identity gauge

For gauges declared on their original scale:

$$
z_t=x_t,
\qquad d_t=\Delta x_t.
$$

Examples include optical power, temperature, bias current, and voltage where
their declared semantics justify the identity scale.

### 19.2 Non-negative skewed gauge or interval count

$$
z_t=\log(1+x_t),
\qquad d_t=\Delta z_t.
$$

`log1p` retains zero and compresses a long right tail. It does not turn a count
into a rate. Exposure normalization is used only if a defensible opportunity
count exists.

### 19.3 Zero-inflated error metric

For BER:

$$
h_t=\mathbf1(x_t>0),
\qquad
m_t=\begin{cases}
\log_{10}(x_t),&x_t>0,\\
\mathrm{NA},&x_t=0.
\end{cases}
$$

For CRC, positive magnitude uses \(\log(1+x_t)\). The hurdle separates two
questions: did any error occur, and how large was the positive magnitude?

### 19.4 Cumulative counter

For monotone counter \(c_t\):

$$
i_t=\begin{cases}
c_t-c_{t-1},&c_t\ge c_{t-1}\text{ and cadence is valid},\\
\mathrm{NA},&\text{otherwise},
\end{cases}
$$

$$
r_t=\mathbf1(c_t<c_{t-1}).
$$

The current `restart_at_zero` policy does not attempt modulus unwrapping.

### 19.5 Discrete state

State features retain the current code and a transition indicator

$$
q_t=\mathbf1(x_t\ne x_{t-1})
$$

when consecutive observations are valid.

## 20. Temporal features

### 20.1 Exact elapsed-time lag

For lag \(L\):

$$
\ell_{t,L}=x_t-x_{t-L}.
$$

The earlier value must exist at the exact timestamp. A row-count lag would
change meaning under missingness or a different cadence. Current reviewed lags
are 1 hour and 6 hours for selected PON metrics.

### 20.2 Trailing robust-history deviation

For closed-left window \(W_t=[t-W,t)\):

$$
m_{t,W}=\operatorname{median}\{x_u:u\in W_t\},
$$

$$
s_{t,W}=\max\left(
\frac{Q_{0.75,W}-Q_{0.25,W}}{1.349},s_{\min}
\right),
$$

$$
h_{t,W}=\frac{x_t-m_{t,W}}{s_{t,W}}.
$$

At least half of the expected window observations are required. Current
windows are 24 hours and 7 days for a reviewed metric subset. The active path
uses the declared minimum scale; it does **not** currently apply an additional
fraction of long-window variability as a floor.

### 20.3 Activity summaries

For a trailing window, zero-inflated error occurrence is summarized by a mean,
while non-negative counter/error increments are summarized by a sum. Current
durations are 6 hours and 24 hours.

### 20.4 Seasonal difference

For EDA-approved period \(P\):

$$
s_{t,P}=x_t-x_{t-P}.
$$

The lag must be exact and causal. A seasonal-versus-non-seasonal ablation was
described in an earlier design, but it is not yet a separately implemented
comparison and is not claimed as a result.

## 21. Current PON feature policy and portability

The active PON policy retains approximately 64–67 model features depending on
availability and approved seasonality:

- transformed current levels or magnitudes;
- safe first differences;
- exact 1-hour and 6-hour changes;
- 24-hour and 7-day robust-history deviations;
- 6-hour and 24-hour activity summaries;
- approved seasonal differences where available.

Clipping flags remain data-quality evidence and do not enter model-health
fitting.

The mechanics are generic, but `configs/features.yml` currently names PON
semantic metric IDs. RAN and Microsoft optical data require their own reviewed
feature-policy configuration—or an explicit catalogue-intersection resolver—
before notebooks `05`–`07` are portable. Shared mechanics plus dataset-specific
policy is the correct portability boundary.

## 22. Feature causality test

For cutoff \(T\), features are computed twice: once from observations through
\(T\), and once from the full source followed by restriction through \(T\).
Logical outputs must match:

$$
\Phi(X_{t\le T})
=\left.\Phi(X_{\mathrm{all}})\right|_{t\le T}.
$$

This prefix-invariance test detects centered rolling statistics, backward fill,
whole-series normalization, and other future leakage.

---

# Part VI — Frozen references and anomaly scores

## 23. Early-calibration robust reference

The implementation first retains at most eight seeded observations per
entity-day, then applies a global cap of 150,000 early-calibration rows for
pooled reference and challenger fitting. This reduces domination by dense,
serially correlated regimes while preserving broad entity and day coverage.
Candidate features have at least 20% availability and more than one distinct
value.

For feature \(j\):

$$
\mu_j=\operatorname{median}(x_{ij}),
\qquad
\sigma_j=\frac{Q_{0.75,j}-Q_{0.25,j}}{1.349}.
$$

If the scale degenerates, the implementation falls back to the median nonzero
absolute deviation from centre and applies the declared feature-scale floor.
An entity-specific centre and scale are fitted with at least 30 valid
observations; otherwise the pooled reference is used.

The standardized residual is

$$
r_{e,t,j}=\frac{x_{e,t,j}-\mu_{e,j}}{\sigma_{e,j}}.
$$

Unseen entities also fall back to the pooled reference.

Notebook 04 flags every observed entity-metric baseline with less than 80% timestamp
coverage, less than 80% valid values, or more than 20% clipped values. Notebook
06 passes those reviewed exclusions into reference fitting: the entity remains
monitored, but the affected metric falls back to its pooled reference.
Multivariate training first retains at most eight deterministic observations
per entity-day and then applies the global row cap. Other entity baselines use
the complete early-calibration slice.

## 24. Rapid self deviation

Residuals are oriented by adverse direction:

$$
a(r)=\begin{cases}
r,&\text{high is adverse},\\
-r,&\text{low is adverse},\\
|r|,&\text{two-sided}.
\end{cases}
$$

For \(n_j\) early-calibration oriented residuals:

$$
\widehat p_{e,t,j}
=\frac{1+\sum_{i=1}^{n_j}\mathbf1(a_{ij}\ge a_{e,t,j})}{n_j+1},
$$

$$
S^{\mathrm{rapid}}_{e,t}
=\max_j[-\log_{10}(\widehat p_{e,t,j})].
$$

The add-one term prevents a zero tail value. Because the same finite calibration
reservoir supplies the tail, this is empirical rank evidence—not a formal
independent p-value.

If only 4 of 1,000 calibration residuals are at least as adverse as the current
value, then

$$
\widehat p=\frac{5}{1001}\approx0.005,
\qquad -\log_{10}(\widehat p)\approx2.30.
$$

### 24.1 Coherent multi-metric evidence

Features belonging to the same semantic metric are first reduced to their
largest empirical-tail evidence. If (M_{(1)}(t)\ge M_{(2)}(t)) are the two
largest values across **distinct** metrics, the implementation compares two
label-free channels:

$$
S^{\mathrm{second}}_t=M_{(2)}(t),
\qquad
S^{\mathrm{mean2}}_t=\frac{M_{(1)}(t)+M_{(2)}(t)}{2}.
$$

The first requires the weaker corroborating metric itself to be extreme. The
second can retain one very strong metric plus one moderate corroborating
metric. Neither names a PON-specific metric pair, and both receive independent
late-calibration block-max thresholds and workload verification.

## 25. Persistent drift

With allowance \(k=0.5\), the two-sided CUSUM recursion is

$$
C_t^+=\max(0,C_{t-1}^+ + r_t-k),
$$

$$
C_t^-=\max(0,C_{t-1}^- - r_t-k).
$$

The channel selects \(C^+\), \(C^-\), or \(\max(C^+,C^-)\) by metric direction
and retains the strongest feature. State resets across missing values, episode
boundaries, and long gaps.

CUSUM is applied to interpretable level, increment, nonzero, positive-log, and
state features. It does not accumulate already-windowed history, lag, rate, or
sum features.

## 26. Peer deviation

For entity \(e\), feature \(j\), time \(t\), and contemporaneous eligible peers
\(\mathcal P(e,t)\) excluding the entity itself, let \(a_j(r)\) denote the
adverse-direction residual defined above. Peer evidence is

$$
P_{e,t,j}=\frac{\max\left\{0,
a_j(r_{e,t,j})-\operatorname{median}_{u\in\mathcal P(e,t)}a_j(r_{u,t,j})
\right\}}
{\max\left\{\operatorname{IQR}_{u\in\mathcal P(e,t)}a_j(r_{u,t,j})/1.349,
0.25\right\}}.
$$

The peer hierarchy is derived from the canonical numeric hierarchy level,
deepest to coarsest. Membership coverage first identifies candidate levels;
the finest level with a stable calibration reference from actual valid
residuals is chosen, falling back to the next coarser level when necessary.
This removes the former duplicate manual preference list. Self-standardization
first removes legitimate entity offsets.

The explicit topology allow-list contains like-for-like optical power,
temperature, bias-current, and voltage measurements. Traffic, error counters,
uptime, and reboot counters do not enter peer comparison merely because they
exist in the catalogue.

## 27. Group common mode

Peer deviation can be small when most descendants move together. For physical
group \(g\), feature \(j\), and available descendants \(\mathcal E_g(t)\):

$$
G_{g,t,j}=\max\left\{0,
\operatorname{median}_{e\in\mathcal E_g(t)}a_j(r_{e,t,j})
\right\},
$$

$$
A_{g,t,j}=\frac{1}{n_{g,t}}
\sum_{e\in\mathcal E_g(t)}\mathbf1(a_j(r_{e,t,j})\ge3).
$$

The group is scoreable with at least three available descendants and

$$
\frac{n_{g,t}}{N_g}\ge0.50,
$$

where \(N_g\) is declared group size.

Instead of a fixed 25% affected rule, the observed affected fraction must
strictly exceed the 0.99 early-calibration quantile for the same physical group type, feature,
and available-count band. This empirical null retains within-group dependence
and group-size effects present in calibration. It is shared-scope anomaly
evidence, not proof that the corresponding network component caused a fault.

Peer and group raw statistics are calibrated by channel, group type, feature,
and count band (`<7`, `7–14`, `15–29`, `>=30`). For stratum median \(m\) and
upper empirical quantile \(q_{0.995}\):

$$
S_{\mathrm{topology}}
=\max\left(0,\frac{S_{\mathrm{raw}}-m}{q_{0.995}-m}\right),
$$

provided at least 100 reference observations exist and the scale is not
degenerate.

## 28. Score readiness

At a timestamp, the primary self detector requires at least

$$
\left\lceil0.2p\right\rceil
$$

of \(p\) retained features. Otherwise its channels are missing. Availability is
reported explicitly so selective scoring cannot resemble high performance.
Topology channels degrade safely to self-only scoring if topology is absent or
no group meets the peer/common-mode requirements.

## 29. Challenger models

Challengers test whether additional statistical structure improves detection at
the same operational workload. They are not promoted merely because they are
more complex.

### 29.1 Dispersion change

For residual standard deviation \(s_{t,W}\) over a rolling window and frozen
calibration spread \(s_0\):

$$
S^{\mathrm{disp}}_t
=\max_j\left|
\log\left(\max\left(\frac{s_{t,W,j}}{s_{0,j}},0.05\right)\right)
\right|.
$$

This detects volatility change without a mean shift. The rolling calculation
is segmented at episode boundaries and internal gaps longer than 1.5 declared
cadences, matching the reset policy used by history and CUSUM.

### 29.2 PCA squared prediction error

Standardized residuals are median-imputed, clipped to \([-50,50]\), and fitted
with PCA on early calibration. The smallest component count explaining at
least 90% of calibration variance is used, capped at 12 and below feature
dimension.

For loading matrix \(V_K\):

$$
\widehat{\mathbf r}_t=V_KV_K^\top\mathbf r_t,
$$

$$
S^{\mathrm{PCA}}_t
=\frac{1}{p}\lVert\mathbf r_t-\widehat{\mathbf r}_t\rVert_2^2.
$$

The score detects departures from normal cross-metric relationships. It is
two-sided and is not a fault probability.

### 29.3 Isolation Forest

Isolation Forest recursively partitions feature space. Unusual observations
typically require fewer splits. Conceptually, its normalized anomaly score is

$$
s(x,n)=2^{-E[h(x)]/c(n)},
$$

where \(E[h(x)]\) is expected isolation path length and \(c(n)\) normalizes for
sample size. The implementation reports `-decision_function`, so larger values
are more anomalous. Alert rate is set by the later calibration threshold—not
by the estimator's contamination parameter.

Current hyperparameters are:

- 300 trees;
- `max_samples = min(2048, n)`;
- 75% of features per tree;
- adverse-direction residuals clipped to \([0,50]\);
- at most five retained inputs per metric;
- at most eight deterministic rows per entity-day before the global cap;
- fixed random seed 42.

Five variants are compared:

1. **base:** the non-window feature set; current classification still includes
   safe first and seasonal differences, so it is not literally level-only;
2. **temporal:** the metric-balanced current and causal temporal feature set;
3. **confirmed temporal:** temporal Isolation Forest tail evidence intersected
   with interpretable rapid-tail evidence using their minimum;
4. **entity calibrated:** the temporal score is robustly standardized against
   the entity's frozen early-calibration score median and IQR, with a pooled
   fallback when fewer than 30 rows exist; adjusted scores are mapped back to
   finite-sample empirical-tail evidence;
5. **contextual:** configured self/topology score context, including rapid,
   CUSUM, dispersion, peer, group, and group affected fraction.

At least half of contextual inputs must be available and nonconstant. The
reported leading feature is the largest adverse-direction standardized
residual—a human-readable proxy, not tree-level attribution.

### 29.4 Why the interpretable detector remains primary

The primary channels map to operational hypotheses: sudden self deviation,
sustained drift, isolated peer deviation, and coherent shared-scope movement.
Isolation Forest and PCA may detect multivariate combinations that these
channels miss, but their scores are less directly linked to a network scope or
failure mechanism. They are evaluated as challengers at the same workload.

Raw scores do **not** share one probability scale:

- rapid self is explicit finite-sample empirical-tail evidence;
- peer and group scores are calibration-quantile normalized;
- CUSUM, dispersion, PCA, and Isolation Forest retain native channel scales.

Channel-specific block-max thresholds make alert decisions operationally
comparable. Incident ranking later uses threshold-relative evidence.

---

# Part VII — Thresholds, alerts, and incidents

## 30. Calibration block maxima

For channel \(c\), entity or scope \(u\), and UTC-day block \(b\):

$$
M_{c,u,b}=\max_{t\in b}S_{c,u,t}.
$$

Candidate thresholds are continuous empirical quantiles

$$
\tau_{c,q}=\widehat Q_q\{M_{c,u,b}\},
\qquad q\in\{0.99,0.995,0.9975,0.999\}.
$$

At 900-second cadence, a block must contain at least 48 observations. Scope
channels count distinct timestamps because one scope score may be repeated on
descendant entity rows. A selectable threshold must have at least five
expected calibration blocks beyond its quantile,
\(B(1-q)\ge5\), where \(B\) is the number of usable block maxima. This is a
minimum empirical-support rule, not proof that the tail blocks are independent.

Block maxima account for repeated within-day scoring opportunities more
honestly than a pointwise quantile. They are empirical held-out-calibration
thresholds, not a formal conformal guarantee.

## 31. Label-free workload operating point

For each portfolio, the independent verification scores are converted to
alerts and cases at every candidate quantile. Every case is conservatively
counted as workload because labels are not opened. Let
\(U_{0.95}(K,E)\) be the exact Garwood upper rate bound. The chosen operating
point is

$$
q_p^*=\min\left\{q:
U_{0.95}\left(K_{\mathcal C_{\mathrm{verify}}}(p,q),
E_{\mathcal C_{\mathrm{verify}}}\right)
\le0.009\right\}.
$$

On an ascending grid, the minimum admissible \(q\) is the most sensitive point
meeting the budget. All late-calibration cases count toward workload because
truth is not read there; they are not verified false positives.

The 0.009 limit is the declared 0.01 research budget multiplied by the frozen
0.90 safety factor. Threshold estimation and workload verification are
therefore disjoint and use the same uncertainty rule as development selection.

## 32. Alert state machine

For score threshold \(\tau_c\):

$$
H_t=\mathbf1(S_c(t)\ge\tau_c).
$$

Persistence duration \(P_c\) becomes

$$
m_c=\left\lceil\frac{P_c}{\Delta}\right\rceil
$$

consecutive high observations. At the current 15-minute PON cadence:

| Channel | Persistence | Typical observations |
|---|---:|---:|
| rapid self | 1,800 s | 2 |
| persistent drift | 1 observation | 1 |
| peer deviation | 1,800 s | 2 |
| group common mode | 1,800 s | 2 |
| challenger channel | 1,800 s | 2 |

Recovery requires 3,600 seconds below 80% of the firing threshold. A missing
score or a gap beyond 1.5 times declared cadence closes any open alert and
resets persistence. `alert_start` is the observation at which persistence is
actually confirmed; it is never backdated to the first breach.

## 33. Incident consolidation

Alerts join an open incident when they are within the 3,600-second quiet period
and either:

- concern the same entity; or
- share a physical topology group and at least one side carries explicit
  group-common-mode evidence.

The incident retains the intersection of common memberships. This prevents an
unrestricted A–B–C chain from merging A and C when they have no common
defensible scope.

For alert score \(S_a\) and its channel threshold \(\tau_{c(a)}\), ranking
evidence is

$$
E(a)=1+
\frac{S_a-\tau_{c(a)}}{\max(|\tau_{c(a)}|,\epsilon)},
\qquad
E(C)=\max_{a\in C}E(a).
$$

This is threshold-relative anomaly evidence. It is not a calibrated incident
probability, expected loss, or customer-impact priority.

Observable dying-gasp or power-loss events may be attached when they occur on
an alerted entity within one hour of the incident. They do not alter model
scores, thresholds, or rank.

## 34. Exposure denominator

Calendar exposure sums observed episode spans plus one cadence:

$$
E_{\mathrm{calendar}}
=\sum_e\frac{t_{e,\max}-t_{e,\min}+\Delta_e}{86400}.
$$

For portfolio (P), the implemented scoreable exposure is

$$
E_{\mathrm{scoreable}}(P)
=\frac{\Delta}{86400}
\left|\left\{(e,t):\exists c\in P,\ S_{e,t,c}\text{ is finite}\right\}\right|.
$$

The union is taken before counting, so a timestamp with two available channels
is counted once. Internal gaps and rows where every selected channel is null or
non-finite contribute no scoreable time. Workload rates and Garwood intervals
use (E_{\mathrm{scoreable}}); Notebook 07 and locked evaluation publish both
denominators and the availability ratio
(E_{\mathrm{scoreable}}/E_{\mathrm{calendar}}). This is detector opportunity,
not an assertion that the entity was contractually obliged to report.

---

# Part VIII — Topology localisation

## 35. Detection and localisation are different claims

Detection asks whether model-visible telemetry is abnormal. Localisation asks
which observable physical footprint best explains incident members. The
output is a probable affected scope, not causal root cause.

## 36. Current implemented localiser

The current rule is transparent:

1. retain the strongest declared group-common-mode scope when present;
2. otherwise return the entity for a single-entity incident;
3. otherwise return the smallest common physical footprint of incident members;
4. report an equivalence class when scopes have identical observable
   descendants.

Notebook 09 audits and presents the location already formed with the incident;
it is not a second learned localiser.

Earlier methodology described scoring candidates with out-of-scope spill,
branch breadth, and direction agreement. That richer candidate-ranking model
is **not implemented** and is a planned extension, not part of current results.

## 37. Identifiability and footprint overlap

If two physical nodes have identical observable descendant sets, telemetry
alone cannot distinguish them. Reporting one as certain would create false
precision.

For predicted descendant footprint \(P\) and true affected footprint \(T\):

$$
\mathrm{precision}_{\mathrm{footprint}}=\frac{|P\cap T|}{|P|},
$$

$$
\mathrm{recall}_{\mathrm{footprint}}=\frac{|P\cap T|}{|T|},
$$

$$
J(P,T)=\frac{|P\cap T|}{|P\cup T|}.
$$

Jaccard penalizes a broad OLT prediction that covers the fault but also includes
much of the network.

---

# Part IX — Evaluation, uncertainty, and selection

## 38. Detection, early-warning, and promptness windows

For fault-entity interval \((f,e)\), matching begins at

$$
L_{fe}=\max(t_{\mathrm{observable/onset},f},
t^{\mathrm{interval\ start}}_{fe}),
$$

The active-fault detection window ends at

$$
U^{\mathrm{active}}_{fe}=\min\left(
t^{\mathrm{event\ end}}_f,
t^{\mathrm{interval\ end}}_{fe}
\right).
$$

The interval is half-open \([L_{fe},U^{\mathrm{active}}_{fe})\). This answers
whether observable anomalous behaviour was detected while the fault was active.
It is not, by itself, evidence of an early warning.

Early warning is credited only when the first matched incident occurs before
the recorded impact time:

$$
L_{fe}\le t^{\mathrm{first\ detection}}_f<t^{\mathrm{impact}}_f.
$$

Faults without a known impact time are excluded from this estimand rather than
silently treated as successes or failures. Promptness is a third estimand. Its
window ends at

$$
U^{\mathrm{prompt}}_{fe}=\min\left(
U^{\mathrm{active}}_{fe}, L_{fe}+H
\right),
$$

where the registered common horizon is currently
\(H=172{,}800\) seconds (48 hours). A pre-impact detection can therefore be a
valid early warning yet miss the 48-hour service objective for a very slow
fault. Conversely, a prompt detection after impact is not an early warning.
Prediction would require forecasting before observable fault evidence; that is
outside the present scope.

## 39. One-to-one event credit

Candidate incident–fault pairs are resolved by deterministic maximum-
cardinality bipartite matching. Each incident receives at most one primary
fault, and each fault receives at most one primary incident. Unassigned
incidents with a valid candidate are duplicates; incidents without a candidate
are false cases.

The matcher maximizes match count, not temporal plausibility or minimum total
delay. A delay-aware sensitivity analysis is appropriate when fault windows
overlap heavily.

Let (M_C) be the entities that actually contributed alerts to incident (C),
and let (A_f(t)) be the affected entities in fault (f)'s valid matching
window. A candidate match now requires

$$
M_C\cap A_f(t_C)\ne\varnothing.
$$

The predicted topology scope is never expanded into (M_C). Its descendants
are used only for localisation precision, recall, Jaccard, equivalence, and
hierarchy-distance evidence. Thus a broad OLT prediction cannot manufacture
detection credit for a concurrent fault on a non-alerting ONT.

## 40. Primary metrics

With credited incidents \(K_M\), total incidents \(K\), active-window
detections \(D_F\), prompt detections \(D_H\), impact-known faults \(N_I\),
and pre-impact detections \(D_I\):

$$
\mathrm{event\ recall}=\frac{D_F}{N_F},
$$

$$
\mathrm{incident\ precision}=\frac{K_M}{K},
$$

$$
\mathrm{false\ incident\ rate}=\frac{K-K_M}{E},
$$

$$
\mathrm{preimpact\ recall}=\frac{D_I}{N_I}.
$$

$$
\mathrm{prompt\ recall}_{H}=\frac{D_H}{N_F}.
$$

The evaluator also reports:

- median detection delay;
- exact, top-two, and equivalence-aware localisation accuracy;
- joint detection-and-localisation recall;
- different-branch rate and hierarchy distance;
- footprint precision, recall, and Jaccard;
- raw alerts, incidents, alerts per incident, duplicates, and volume reduction;
- stratification by fault type and declared domain;
- score availability and scoreable denominators.

Localisation metrics describe affected-scope agreement. They do not prove
causal root cause.

## 41. Statistical uncertainty

### 41.1 Wilson interval for proportions

For \(x\) successes among \(n\), \(\widehat p=x/n\), and \(z=1.96\), the 95%
Wilson interval is

$$
\frac{
\widehat p+\frac{z^2}{2n}
\pm z\sqrt{\frac{\widehat p(1-\widehat p)}{n}
+\frac{z^2}{4n^2}}
}{1+\frac{z^2}{n}}.
$$

It is preferred to the Wald interval for small samples and proportions near
zero or one.

### 41.2 Garwood interval for incident rates

For count \(k\), exposure \(E\), and \(\alpha=0.05\):

$$
\lambda_L=\frac{\chi^2_{\alpha/2,2k}}{2E},
\qquad
\lambda_U=\frac{\chi^2_{1-\alpha/2,2(k+1)}}{2E},
$$

with lower bound zero when \(k=0\).

Wilson and Garwood intervals assume more independence than a network often
provides. Faults and nuisance incidents can cluster by OLT, vendor, and time.
Production evidence needs topology/time-block bootstrap intervals. Current
delay, footprint averages, hierarchy distance, and domain-level results also
lack uncertainty intervals.

The configured confidence level is passed through event, localisation, and
workload interval calculations. The current frozen value is two-sided 95%.

## 42. Development selection gate

A candidate passes the current **research detection gate** only when all
implemented gates pass:

1. at least 30 development faults;
2. lower endpoint of the two-sided 95% Wilson event-recall interval at least
   0.20;
3. missing-score fraction at most 0.20;
4. upper 95% false-incident-rate bound no greater than

$$
0.01\times0.90=0.009
$$

per entity-day.

For 64 scoreable faults, the recall rule requires at least 20 detections:
the observed recall is then 31.25% and its Wilson lower bound is approximately
21.2%. The 20% value is a conservative research floor, not a production SLA;
an operator pilot must replace it with a cost- and criticality-based target.
At the minimum allowed sample of 30 faults, 11 detections—not six—are needed;
Notebook 07 prints this power table for the actual denominator on every run.

Among candidates within 0.02 point recall of the best eligible result, a frozen
simplicity preference is applied, followed by lower upper-bound workload and
higher recall. If no candidate passes, Notebook 07 writes diagnostic evidence
and **no selected detection configuration**.

This fail-closed result is scientifically valid. It prevents the least-bad
experiment from being mislabeled production-ready.

Detection selection and localisation qualification are deliberately separate.
For localisation, Notebook 07 reports the number of faults affecting more than
one unique entity and the point estimate plus Wilson interval for joint
detection-and-equivalence-aware-localisation recall. Fewer than 20 such faults
is insufficient. Even with 20 or more, localisation remains
`not_established_unregistered_performance_gate` until an explicit minimum joint
recall lower bound is registered; the code does not silently reuse the
detection target.

Remaining gate limitations are:

- no stakeholder-approved localisation performance target is registered;
- channel pruning can make nominal portfolios operationally identical;
- matched workload means the same budget constraint, not identical realized
  incident rates.

## 43. Locked holdout procedure

Notebook 10 requires a development-selected configuration and verifies hashes
of the model, features, policies, truth, and relevant code. Development is the
default partition.

Opening holdout requires both:

```text
EVALUATION_PARTITION=holdout
OPEN_HOLDOUT=1
```

Before truth is read, a ledger records the frozen configuration. Development
is replayed through the same scoring path and must produce byte-identical
scores. Repeat evaluation is permitted only for the same selection.

The ledger is editable local evidence, not tamper-proof governance. A
production evaluation should place truth and the opening record under
independent access control.

---

# Part X — Reproducibility, scaling, and operationalization

## 44. Immutable analytical stages

Every material stage writes to a new run directory and records:

- input file identities and content hashes;
- contract, dataset, feature, topology, and alert-policy versions;
- code version or source hashes;
- schemas, row counts, partitions, and timestamps;
- fitted reference and model parameters;
- threshold provenance and selected operating point;
- deployability or diagnostic-only status.

Completed outputs are not silently overwritten. Parquet is the analytical
storage format. Large source data and generated outputs remain outside Git;
Git stores code, configuration, notebooks, documentation, and small test
fixtures.

## 45. Resource-aware execution

The full synthetic PON fixture is large enough that convenient in-memory
notebook patterns can exhaust Colab memory or disk. The implementation uses:

- Parquet partitioning and column projection;
- DuckDB external execution and bounded memory;
- entity/episode batching;
- a bounded model-fit reservoir;
- decomposed topology passes rather than one wide global join;
- separate disk-backed join and ordered-write stages for the final
  self/topology score merge;
- temporary workspaces with explicit cleanup;
- compact evidence outputs rather than repeated full score copies.

These controls do not change the statistical estimand. They implement the same
frozen computation under a resource bound. Runtime, memory limit, row counts,
and truncation/sampling settings belong in every execution manifest.

## 46. Inference lifecycle

A production service would process each new telemetry interval as follows:

1. validate native schema and map to semantic metrics;
2. apply quality, bounds, clipping, episode, and gap logic;
3. update causal feature state;
4. standardize with the frozen reference;
5. score available self and topology channels;
6. apply frozen thresholds, persistence, and recovery;
7. consolidate alerts into incidents;
8. localise the incident footprint;
9. attach genuinely observable operational context;
10. publish evidence, data-quality status, and model version;
11. monitor score availability, drift, workload, and operator disposition.

The notebooks demonstrate this lifecycle in batch. They are not themselves the
final online serving architecture.

## 47. Adapting to another operator or Telecom dataset

### 47.1 What changes

- native file, table, API, or stream reader;
- native-to-semantic metric mapping;
- verified unit, cadence, aggregation, bounds, clipping, and reset policy;
- entity, service-window, and episode construction;
- topology identifiers and effective-time memberships;
- observable alarm/event mapping;
- dataset-specific feature policy;
- local calibration population;
- operational incident budget and reporting workflow.

### 47.2 What remains shared

- canonical schemas and truth boundary;
- causal feature primitives;
- robust-reference mechanics;
- primary and challenger algorithms;
- block-max threshold mechanism;
- alert and incident state machines;
- localisation interface;
- evaluation definitions and holdout controls.

### 47.3 Source-onboarding gate

A new source is accepted only after:

1. every native column is classified;
2. units and aggregation are verified from source documentation;
3. observed cadence is reconciled with declared schedule;
4. sentinels, clipping, counter resets, and missing capability are documented;
5. entity validity and episode logic are justified;
6. topology uniqueness and temporal validity are checked;
7. adapter output is truth-invariant;
8. calibration-only EDA supports the feature policy;
9. label-free workload is reported before recall;
10. labelled claims use controlled development and locked holdout.

## 48. Public-data qualification

Public datasets are evidence tests, not extra rows for one universal model.

- **RAN PM counters:** use a separate RAN pack and feature policy to test real
  operator telemetry, hierarchy, cadence, heterogeneity, scale, and workload.
  The source does not validate PON fault recall.
- **Microsoft optical:** use a separate optical pack to test channel baselines,
  slow drift, common movement, and unlabelled workload. Removed outage days
  prevent recall claims.
- **Optical failure testbed:** use as evaluator-only controlled-response
  evidence. Short induced failures and source terms limit operational claims.

Synthetic, public real, controlled testbed, and future operator evidence remain
separate in every report.

---

# Part XI — Limitations and production-hardening plan

## 49. Current limitations that must remain visible

1. The primary labelled evidence is synthetic.
2. Synthetic quiet periods do not estimate real operator nuisance workload.
3. Some holdout fault families have very small denominators.
4. Some declared faults are unscoreable because they do not overlap observable
   telemetry.
5. The localiser is footprint-rule based, not learned or causal.
6. No stakeholder-approved localisation performance gate is registered.
7. Promptness uses one fixed 48-hour value; operator-specific service objectives
   have not yet been validated.
8. Topology is not joined at every event's effective time.
9. Basic confidence intervals do not model temporal/topology clustering.
10. Public-source feature policies are not yet independently reviewed.
11. A sensor that never reports needs capability metadata to distinguish
   failure from non-installation.
12. Incident evidence is anomaly strength, not business risk.

## 50. Priority corrections

### Completed in evaluator/selection revision 4.3/v9

- detection credit requires observed alert-member overlap;
- predicted-footprint overlap is localisation-only evidence;
- workload uses portfolio-scoreable exposure and reports calendar exposure;
- event recall is gated on its two-sided 95% Wilson lower bound;
- localisation sample sufficiency and qualification status are reported
  separately from detection selection.
- active-fault detection, pre-impact early warning, and 48-hour prompt recall
  are distinct estimands;
- development selects only among operating points that independently passed
  calibration workload and empirical-tail support checks.

### Completed in detector/selection revision 4.5/v10

- EDA is restricted to early calibration and passes population-wide unstable
  entity-metric exclusions into reference fitting;
- physical peer fallback order is derived from the canonical hierarchy;
- common-mode breadth is calibrated by topology-size band instead of fixed
  affected-count/fraction rules;
- dispersion resets at internal gaps;
- entity-calibrated Isolation Forest and coherent two-metric evidence are
  predeclared candidates;
- the FEC clipping exception fails loudly when its source-run identity does
  not match; and
- the nominal workload budget, safety factor, effective gate, and confidence
  statistic are published together.

### Priority 1 — selection validity

- register an operator-meaningful joint-localisation performance target before
  claiming that capability;
- validate the common 48-hour promptness objective with operators before
  replacing it or registering fault-family objectives.

### Priority 2 — dependence and topology

- add time/topology-cluster bootstrap uncertainty;
- implement effective-time topology joins;
- recognize entity/single-descendant equivalence;
- compare the bounded entity-day sample with a cluster-weighted sensitivity
  analysis;

### Priority 3 — operational evidence

- add the operator's existing threshold/alarm rules as a transparent baseline;
- run a real PON shadow-mode pilot;
- have operators adjudicate incident usefulness and duplicate burden;
- estimate drift and define a scheduled recalibration policy;
- only then consider fault classification or prediction.

## 51. How to interpret a negative result

Failure to meet the development gate does not mean the experiment failed. It
means that, at the declared workload and uncertainty margin, no tested
configuration earned deployable status. Appropriate next work is diagnostic:
inspect channel and fault-family coverage, score availability, timing, feature
response, workload sources, and transparent operator-rule baselines. Opening
holdout or relaxing the gate after seeing development performance would not
solve the scientific problem.

---

# Appendix A — Current decision register

| Decision | Current implementation | Rationale or qualification |
|---|---|---|
| Primary domain | Fixed-access PON/ONT | Topology localisation is the product driver |
| PON cadence | 900 seconds | Declared schedule for current synthetic source |
| Main split | Chronological calibration/development/holdout | Prevent adjacent-row leakage |
| Calibration split | 70% fit, 15% threshold, 15% workload verification | Separate model fit, threshold estimation, and label-free workload evidence |
| Gap tolerance | 1.5 × cadence | Permit jitter, stop state across longer absence |
| Fit reservoir | 8 seeded rows per entity-day, then at most 150,000 rows | Limit serial and entity imbalance in multivariate fitting |
| Minimum entity reference | 30 observations | Avoid very small local estimates |
| Threshold blocks | Entity/scope UTC-day maxima | Account for repeated scoring opportunities |
| Threshold grid | 0.99, 0.995, 0.9975, 0.999 | Resolve the low-alert-rate operating region with at least five expected tail blocks |
| Workload target | 10 incidents per 1,000 entity-days | Research target; pilot must validate |
| Gate safety factor | 0.90 | Require upper interval below 9 per 1,000 |
| Rapid persistence | 1,800 seconds | Two observations at current cadence |
| Drift persistence | 1 observation | CUSUM already accumulates sustained evidence; avoid a second delay layer |
| Peer/group persistence | 1,800 seconds | Reduce isolated score spikes |
| Incident quiet period | 3,600 seconds | Consolidate nearby evidence |
| CUSUM allowance | 0.5 standardized units | Ignore small deviations while accumulating drift |
| Minimum valid peers | 7 | Disable unstable tiny peer comparison |
| Peer coverage | 80% | Require meaningful comparison population |
| Minimum group entities | 3 | Require nontrivial common-mode footprint |
| Group availability | 50% | Avoid inference from a small observed fraction |
| Group affected support | 0.99 early-calibration affected-fraction quantile within topology-size band | Scale shared-scope evidence to group size and observed dependence |
| Recovery | 3,600 seconds below 80% of firing threshold | Hysteresis without backdated closure |
| Detection window | Complete active fault interval | Measures anomaly detection without relabelling late true detections as false incidents |
| Early-warning window | Observable evidence to recorded impact | Separates warning from post-impact diagnosis |
| Promptness horizon | 172,800 seconds | Current common 48-hour service objective |
| Development faults | At least 30 | Avoid selection on a tiny denominator |
| Recall gate | 95% Wilson lower bound at least 0.20 | Applied to active detection; pre-impact and prompt recall are separately qualified at the same provisional research floor |
| Missing-score gate | At most 0.20 | Prevent selective availability looking successful |

# Appendix B — Current PON metric policy

All metrics below currently declare 900-second cadence. The table is a
source-policy summary, not a universal Telecom standard. A new operator must
verify its own cadence, units, aggregation, and source-quality behaviour.

| Semantic metric | Kind | Unit | Direction | Base transform | Important qualification |
|---|---|---|---|---|---|
| `optical.rx_power` | gauge | dBm | low bad | identity | entity baseline is important |
| `optical.olt_rx_power` | gauge | dBm | low bad | identity | distinct source-side optical view |
| `optical.tx_power` | gauge | dBm | two sided | identity | both high and low deviation may matter |
| `equipment.temperature` | gauge | °C | high bad | identity | seasonality candidate |
| `optical.bias_current` | gauge | mA | high bad | identity | seasonality candidate |
| `equipment.voltage` | gauge | V | two sided | identity | small declared minimum scale |
| `optical.ber` | bounded fraction | ratio | high bad | hurdle + positive `log10` | valid range 0–1 |
| `optical.fec_count` | interval count | count | high bad | `log1p` | right-censored at source-specific 5,000,000 |
| `ethernet.crc_errors` | interval count | count | high bad | hurdle + positive `log1p` | no exposure rate is invented |
| `equipment.uptime` | cumulative counter | seconds | contextual | reset-safe increment | negative difference marks reset |
| `equipment.reboot_count` | cumulative counter | count | high bad | reset-safe increment | negative difference marks reset |
| `service.throughput` | gauge | Mbps | low bad | `log1p` | interval average; seasonality candidate |

The declared scale floor is a numerical guard, not a normality claim or an
alert threshold. Current floors are 0.10 for most transformed optical/error
features, 0.25 °C for temperature, 0.05 dBm for transmit power, 0.005 V for
voltage, and 1.0 for counter increments.

# Appendix C — Notebook outputs

| Notebook | Principal outputs | Consumer |
|---|---|---|
| 00 | Contract/config receipt | All stages |
| 01 | Source inventory, column classification, quality report | Adapter review and 02 |
| 02 | Prepared pack and truth-unmounted canonical core | 03–06 |
| 03 | Development truth, locked holdout, split/isolation manifests | 07 and 10 only |
| 04 | Structural, temporal, seasonal, dependence, sufficiency evidence | Feature review and 05 |
| 05 | Early-fit, late-calibration, and development features | 06 and 07 |
| 06 | Frozen reference, threshold/workload/development scores, thresholds | 07, 08, and 10 |
| 07 | Workload points, development comparison, selection status | 08 and model freeze |
| 08 | Persistent alerts, incidents, observable event context | 09 and operator view |
| 09 | Location, alternatives, equivalence, footprint evidence | 10 and operator view |
| 10 | Matches, metrics, uncertainty, evaluation receipt | Model card and review |
| 11 | Separate public-source qualification evidence | External-validity report |
| 12 | Ranked queue, chronological replay, model card | Client/engineering handoff |

# Appendix D — Threats to validity

| Threat | Current control | Residual risk |
|---|---|---|
| Label leakage | Physical separation, schema guard, invariance, negative control | Deliberate manual access remains possible |
| Temporal leakage | Chronological split, early-calibration EDA, causal windows, prefix test | Manual policy changes can still overfit repeated development cycles |
| In-sample tail optimism | Early fit/EDA plus disjoint threshold and workload slices | Calibration blocks remain dependent within network events |
| Repeated alert opportunities | Daily maxima and incident consolidation | Block dependence/nonstationarity remain |
| Duplicate event credit | One-to-one matching | Matching does not optimize delay |
| Broad location credit | Alert-member-only detection match; footprint metrics and ambiguity | Broad predictions can still have poor localisation utility |
| Small samples | Counts, Wilson/Garwood intervals, fault gate | Dependence can make intervals optimistic |
| Topology changes | Effective-dated contract and ambiguity checks | No event-time join yet |
| Missing telemetry | Quality, episodes, gaps, dual calendar/scoreable exposure | Source capability metadata is still needed for never-seen sensors |
| Synthetic-to-real shift | Separate public qualification and pilot requirement | No real PON performance evidence yet |

# Appendix E — Glossary

- **Adapter:** dataset-specific translation from native storage and names into
  the prepared pack interface.
- **Alert:** a persistent threshold exceedance on one score channel.
- **Calibration:** label-free interval used for normal references and thresholds.
- **Canonical data:** vendor-neutral detector-facing data under `SPEC-CORE`.
- **Case/incident:** operational object consolidated from related alerts.
- **Clipping:** reporting saturation; the exact underlying value is unknown.
- **Collection gap:** an expected periodic observation absent inside an active
  episode.
- **Common mode:** coherent movement among descendants of one physical scope.
- **Development:** labelled period used to compare predeclared candidates at
  calibration-admissible operating points.
- **Early warning:** detection after fault evidence becomes observable but
  before recorded impact; it is not pre-evidence prediction.
- **Entity-day:** one entity observed for one day; the current workload unit.
- **Episode:** recording context across which stateful calculations may operate.
- **Evidence score:** anomaly strength under one channel; not necessarily a
  probability.
- **Holdout:** sealed final evaluation interval opened after configuration
  freeze.
- **Peer:** contemporaneously observed comparable entity in a justified
  physical group.
- **Residual:** transformed value minus frozen centre, divided by frozen scale.
- **Scoreability:** availability of model evidence and a valid evaluation
  window.
- **Topology equivalence:** scopes with identical observable descendant sets.
- **Truth:** fault, ticket, condition, or cause information reserved for
  evaluation.
