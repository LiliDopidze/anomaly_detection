# Company-agnostic Telecom anomaly detection methodology

## 1. Product boundary

The product detects unusual Telecom telemetry, consolidates repeated signals
into operational incidents, and estimates the smallest topology scope
supported by the evidence. PON/ONT operations drive development. Supervised
fault classification, failure prediction, and causal root-cause analysis are
outside the current scope.

“Company-agnostic” is an interface and calibration property:

1. Native vendor fields map to semantic metric identifiers.
2. Metric behaviour is declared rather than inferred from a field name.
3. Normal references and empirical score tails are fitted locally without
   fault labels.
4. Detection, consolidation, localisation, and evaluation algorithms are
   shared.

It is not a claim that raw PON, RAN, and backbone-optical distributions can be
pooled into one model.

## 2. Evidence and claim levels

The synthetic PON fixture is the primary engineering fixture because it
contains the required ONT metrics, topology, and injected fault scopes. It can
validate software and recovery of generator mechanisms, but not production
performance.

The commercial RAN PM dataset is the primary external real-data qualification.
It tests real seasonality, entity heterogeneity, missingness, peer behaviour,
scalability, and achievable alert workload through a separate RAN metric pack.
It has no verified incident truth, so quiet periods are not labelled negatives
and recall is not reported.

Microsoft optical telemetry is an optional research-only qualification for
slow optical drift and cross-channel behaviour. Outage days were removed, so
it cannot support recall. Its source agreement is restrictive; neither raw nor
derived data is redistributed by this repository.

The public optical-failure testbed is an optional labelled response test. Its
short controlled failures are not representative of production prevalence.
The repository currently has no explicit reuse licence, so users must obtain
legal clearance and explicitly acknowledge the source terms before the pinned
upstream files are acquired. The acquisition mechanism is not a grant of reuse
or redistribution rights.

## 3. Data contract and truth boundary

`SPEC-CORE` contains only information available to the detector:

- long-form telemetry and quality code;
- metric catalogue;
- entities and observation episodes;
- collection gaps;
- effective-dated topology when supplied;
- observable operational events when supplied.

`SPEC-EVAL` contains faults, affected-entity intervals, tickets, and labels.
It is physically separable. Tests compare logical `SPEC-CORE` fingerprints
with evaluation mounted and absent, and a deliberately leaky scorer must fail
when truth is absent.

The distinction between an absent observation, a present null value, and an
entity outside its service interval is preserved. Dying gasp, LOS, and similar
alarms are events—not synthetic continuous features.

## 4. Splits

The primary PON experiment is chronological:

- calibration: fit transformations, normal references, scales, score tails,
  and workload thresholds;
- development: compare the small predeclared model portfolio and incident
  policy;
- holdout: one final locked evaluation after configuration freeze.

A secondary whole-infrastructure split holds out complete OLT groups. It tests
portability across infrastructure but does not replace the chronological
primary evaluation. No random row split is permitted.

## 5. Measurement semantics

The metric registry declares measurement kind, direction, aggregation,
transform, cadence, censoring, counter reset policy, and a physical scale
floor. Important transformations are:

- gauges: identity or `log1p` only where justified;
- BER: a hurdle representation (zero occurrence plus positive log magnitude)
  unless a valid bit exposure supports a count likelihood;
- CRC/FEC interval counts: zero-aware `log1p`; exposure normalisation only when
  a defensible opportunity count exists;
- cumulative counters: gap-safe positive increments plus a reset indicator;
- states: level, transition, and persistence features.

Right-censored FEC values remain marked `clipped` and are not treated as exact
measurements. Unknown lower or upper physical bounds remain null; the adapter
does not invent them.

## 6. Calibration-only EDA

EDA never reads evaluation truth and only sees the calibration interval. It
establishes cadence, coverage, valid-value rate, clipping, zero inflation,
counter behaviour, robust level and scale, temporal dependence, seasonality
repeatability, vendor/model heterogeneity, cross-entity dependence, and peer
group availability.

Seasonality is enabled only when enough complete cycles exist and the pattern
is repeatable. A plot is evidence, not a selection rule. EDA decisions are
saved with the canonical fingerprint and become frozen inputs to features.

## 7. Leakage-safe features

All windows use elapsed time, restart after collection gaps, and exclude
future values. Calibration-frozen self residuals remove legitimate entity
offsets before any peer comparison. The compact feature set contains current
levels or transformed magnitudes, first differences, approved seasonal
differences, 1-hour and 6-hour changes, 24-hour and 7-day robust history
deviations, and 6-hour and 24-hour error/reset activity summaries. Feature
families are enabled only for metrics where they have a clear interpretation.
Missingness and clipping remain explicit data-quality evidence; they are not
silently converted into equipment-health values.

Peer residuals compare one entity against contemporaneous self residuals of
its eligible peers. Only current-state features from metrics explicitly marked
``peer_eligible`` by the dataset adapter enter cross-entity scoring; temporal
lags stay in the self-history and multivariate channels. Their null scale is
calibrated by topology level and group-size band, and very small peer groups
are disabled. This lets each operator supply its own identifiers and hierarchy
without changing the detector.

Group features describe common movement through the median descendant
residual, affected fraction, available fraction, and the physical scope that
supports the evidence. A future-data perturbation test proves feature
causality.

## 8. Primary detector

The primary model is an interpretable four-channel detector:

1. **Rapid self deviation** — sudden direction-aware departures from the
   calibration-frozen entity reference.
2. **Persistent drift** — gap-reset CUSUM or sustained residual accumulation.
3. **Peer deviation** — one entity departs from contemporaneous peers after
   its own normal offset is removed.
4. **Group common mode** — multiple descendants move coherently, covering the
   shared-fault regime where peer comparison loses sensitivity.

Raw channel scores become comparable empirical calibration-tail evidence:

`tail_score = -log10(max(P_calibration(S >= observed_score), epsilon))`

Metrics are not blindly averaged. The strongest credible evidence is retained
with persistence and coverage. Rapid/peer/group channels divide the fault
space by affected fraction, so channel-level recall is reported by fault type.

## 9. Challengers

Isolation Forest is the main challenger, trained on robust semantic residuals
and cross-metric consistency—not raw vendor fields or identifiers. Three
variants isolate what adds value: current-state features only, current plus
causal temporal features, and self scores plus topology context. They are
compared at the same consolidated incident workload. The model's
``contamination`` setting does not set the production alert rate; late
calibration thresholds do.

Robust PCA is a smaller correlated-change challenger. Autoencoders are deferred
until substantially more real labelled and unlabelled operator telemetry is
available. Matrix profile is limited to diagnostic univariate experiments.

Required ablations are rapid only; rapid plus drift; self plus peer; self plus
peer plus group; the three Isolation Forest variants; and approved seasonal
versus non-seasonal references.

## 10. Thresholds and incidents

The early calibration slice fits the reference model. A disjoint late
calibration slice supplies empirical daily-block maxima for candidate channel
thresholds, so in-sample residuals cannot make the tails look artificially
light. For each portfolio, a label-free operating point is the most sensitive
candidate whose complete consolidated incident rate on late calibration stays
inside the declared workload budget. Development labels may choose among
these pre-calibrated portfolio operating points, but never choose the
threshold quantile. Persistence is expressed in elapsed time. Consolidation
requires temporal overlap plus shared entity or a
justified topology relation; unrestricted transitive chaining is forbidden.

The pipeline fails closed when no development candidate meets the workload
budget, denominator, stability, and minimum-detection gates. It writes a
diagnostic comparison but no deployable selection.

## 11. Localisation

Localisation is a second stage over incident evidence. It scores entity, L2
splitter, L1 splitter, PON port, and OLT candidates using in-scope evidence,
out-of-scope spill, affected and observable fractions, branch breadth,
direction agreement, and channel provenance. Common-mode evidence is owned by
its native scope rather than duplicated onto every descendant.

If two nodes have the same observable descendants, the result is an
equivalence class. The output is a probable affected scope with alternatives
and limitations, not a causal diagnosis.

## 12. Locked evaluation

The evaluator matches predicted incident footprints to observed fault
footprints using deterministic one-to-one assignment. It reports event recall,
detection delay, incident precision, false incidents per entity-day, alerts per
incident, footprint precision/recall/Jaccard, exact and top-two scope accuracy,
and joint detection-and-localisation recall. Counts and uncertainty intervals
accompany every rate; clustered or block bootstrap intervals are preferred
where dependence matters.

Development labels may select among the declared portfolio. Holdout truth is
then opened once and cannot change transformations, features, thresholds,
persistence, consolidation, or localisation.

## 13. Production boundary

A production handoff requires an operator pilot with vendor mappings, topology
history, alarm semantics, service windows, monitoring delay, ticket quality,
and an agreed incident budget. The replay notebook demonstrates causal state,
missing-data resilience, incident lifecycle, explanations, and a model card.
Synthetic and public-data evidence remain separate sections in every report.
