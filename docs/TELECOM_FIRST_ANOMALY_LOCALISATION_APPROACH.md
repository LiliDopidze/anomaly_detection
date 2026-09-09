# Telecom-First Cross-Sector Anomaly Detection and Localisation

## Implemented analytical methodology

**Status:** implemented in contract/model version 3.0 and verified with controlled fixtures; full-data reruns required  
**Primary sector:** Telecom access telemetry  
**Secondary qualification sector:** Petrobras 3W  
**Current scope:** anomaly detection, alert consolidation, incident ranking and hierarchical localisation  
**Deferred scope:** fault-horizon prediction, supervised fault classification and causal root-cause diagnosis

---

## 1. Executive decision

The product will be developed for Telecom first. The generic layer will remain portable, but the roadmap will be driven by the capabilities that matter for Telecom:

1. detect unusual behaviour from telemetry;
2. combine repeated metric alerts into operational incidents;
3. use peer and topology structure to distinguish local from shared faults;
4. rank incidents by statistical evidence;
5. localise the most likely network scope and report uncertainty.

Petrobras 3W remains in the project because it is real industrial data and provides a valuable portability and sensor-quality test. It will not determine the main detector architecture, thresholds or product workflow.

The target is not a single universal algorithm. It is a common analytical framework with:

- one canonical telemetry contract;
- capability-aware generic modelling;
- small sector packs that supply meanings and available structure;
- sector-specific evaluation truth that is physically isolated from the detector.

---

## 2. What the product will and will not claim

### 2.1 Product outputs

For each run, the detector should produce:

- entity-metric anomaly scores;
- entity-level alerts;
- group-level common-mode alerts where topology is available;
- consolidated incidents;
- ranked candidate locations with evidence and confidence limitations;
- evaluation results when hidden truth is mounted separately.

### 2.2 Interpretation of the outputs

An anomaly means that observed telemetry is inconsistent with an estimated reference behaviour. It does **not** automatically mean a physical fault.

Localisation means identifying the topology scope whose descendant telemetry best explains the observed anomaly footprint. With ONT-only telemetry, this is footprint inference rather than direct observation of the upstream component.

The system will not initially claim to:

- predict future faults;
- diagnose an engineering root cause from telemetry alone;
- assign a reliable business-impact score without service or customer-impact data;
- prove real Telecom performance using synthetic Telecom data;
- provide a universal best model for every sector.

### 2.3 Evidence language

Synthetic Telecom data can validate mechanics, leakage controls and recovery of known injected patterns. It cannot establish real-world detection effectiveness.

Petrobras 3W can validate portability to real sensor recordings and real data-quality pathologies. It cannot validate Telecom peer or topology performance because its structure is materially different.

Final claims about Telecom effectiveness require real Telecom telemetry and operator review.

---

## 3. Roles of the two sectors

| Area | Telecom | Petrobras 3W |
|---|---|---|
| Product role | Primary development sector | Real-data qualification sector |
| Data status | Synthetic | Real industrial recordings |
| Main unit | ONT | Well recording episode |
| Cross-entity peers | Yes | Not reliably available |
| Topology | ONT to splitter, PON, OLT and geographic group | None supplied for modelling |
| Time structure | Long periodic panel | Bounded one-second recordings |
| Main value | Peer detection, common-mode detection and localisation | Portability, missingness, frozen sensors, sentinels and multiple event timescales |
| Permitted model path | Full self, peer and group hierarchy | Generic self-history fallback |
| Prohibited adaptation | None beyond honest source limitations | Do not invent peer groups or topology |

Petrobras will be run at release checkpoints and after changes to common code. It is not necessary to rerun it for every Telecom-only experiment.

---

## 4. Architecture and separation of responsibilities

### 4.1 Sector pack

The sector pack translates native source data into a small common input interface. It owns:

- native field names;
- metric identifiers, units and measurement kinds;
- sampling mode and expected cadence;
- entity and episode construction;
- quality-code rules justified by the source;
- topology memberships when genuinely available;
- evaluation-label interpretation;
- split construction appropriate to the source.

It does not create model features or tune detector thresholds.

### 4.2 Common canonical adapter

The common adapter performs the same operations for every valid pack:

- validates required schemas;
- materialises canonical telemetry;
- calculates collection gaps for periodic metrics;
- preserves topology without interpreting Telecom names;
- copies evaluation truth only when explicitly requested;
- proves that canonical model input is unchanged when truth is removed;
- records manifests and content fingerprints.

The adapter must not contain branches such as `if sector == "telecom"`.

### 4.3 Generic analytical engine

The analytical engine reads capabilities, not sector names. Examples:

- if only telemetry exists, run self-history channels;
- if topology memberships exist and pass eligibility checks, add peer and group channels;
- if discrete-state metrics exist, use state-appropriate transformations;
- if periodicity is supported by calibration-only evidence, use seasonal references;
- otherwise retain a simpler robust reference.

### 4.4 Evaluation harness

The evaluation harness is separate from the detector. It reads alerts and SPEC-EVAL only after scoring is complete. It owns:

- alert-to-fault matching;
- time horizons;
- duplicate treatment;
- grouped-fault evaluation;
- localisation scoring;
- workload and uncertainty reporting.

---

## 5. Canonical contract correction

### 5.1 Mandatory SPEC-CORE

The minimum model-visible core remains:

- `telemetry`;
- `metric_catalogue`;
- `entity_registry`;
- `observation_episodes`;
- `collection_gaps`.

Telemetry remains long-form, keyed by:

`event_ts, entity_id, episode_id, metric_id`

and contains:

`value, quality_code`.

### 5.2 Optional model-visible topology

Topology is a detector capability, not an experimental split. It should therefore move from `SPLITS/entity_groups.parquet` to an optional SPEC-CORE table:

`topology_memberships`

with the initial fields:

| Field | Meaning |
|---|---|
| `entity_id` | Telemetry-reporting entity, currently an ONT |
| `group_type` | Canonical topology level such as `splitter_l2`, `splitter_l1`, `pon_port` or `olt` |
| `group_id` | Identifier of the group at that level |
| `hierarchy_level` | Ordered depth used for hierarchical reasoning |
| `group_family` | `physical_topology` or `geographic_context` |

`geo_cluster` is contextual grouping, not a proven physical failure domain. It must not be treated as interchangeable with a splitter, PON or OLT.

For the present static synthetic fixture, validity dates are not needed. When real inventory history is introduced, memberships will require `valid_from` and `valid_to`.

### 5.3 SPLITS

SPLITS should contain experimental partitions only:

- `time_partitions` for Telecom;
- `entity_partitions` for Petrobras;
- any later infrastructure holdout definition.

The detector may use topology. It must never use partition labels as features.

### 5.4 SPEC-EVAL localisation truth

`fault_events` currently carries `domain_id`, but the identifier is ambiguous without its type. Add:

- `domain_type`;
- `domain_id`.

The Telecom pack should map native fault scope explicitly, for example:

| Native scope | Canonical `domain_type` |
|---|---|
| ONT | `entity` |
| L2 splitter | `splitter_l2` |
| L1 splitter | `splitter_l1` |
| PON | `pon_port` |
| OLT, if present | `olt` |

This truth is evaluation-only and is never read by the detector.

---

## 6. Phase 0: localisation feasibility audit

This audit must run before model changes because it determines what the available telemetry can actually identify.

It should be a short, Telecom-only notebook or a clearly separated first section. Its outputs are evidence tables, not model results.

### 6.1 Topology audit

Produce `topology_group_audit` with, for every topology level:

- number of groups;
- entities per group: minimum, median, upper quantile and maximum;
- number and proportion of singleton groups;
- missing memberships;
- entities mapping to multiple groups at the same level.

Current data already indicates approximately:

- 400 ONTs;
- 65 L2 splitter groups, averaging about 6.2 ONTs;
- 22 PON groups, averaging about 18.2 ONTs.

The average is not enough. Eligibility must use the per-group distribution and time-varying valid peer counts.

### 6.2 Effective peer availability

Produce `effective_peer_availability` by metric and topology level:

- eligible observations;
- median valid peers;
- lower-decile valid peers;
- proportion of observations with at least 7 valid peers;
- proportion with at least 15 valid peers;
- fallback level used.

Initial eligibility policy:

- fewer than 7 valid peers: do not emit a peer score at that level;
- 7–14 valid peers: allow peer centring but use a calibration-pooled scale;
- 15 or more valid peers: group-specific or size-banded calibration scale may be used.

These are operational starting rules. The audit must report sensitivity to the boundaries rather than presenting them as physical constants.

### 6.3 Identifiability audit

Produce `topology_identifiability` by comparing the descendant ONT sets of every candidate scope.

If two nodes have identical observable descendants, ONT telemetry alone cannot distinguish them. They must be treated as an equivalence class for scoring. For example, if an L1 splitter and a PON have exactly the same ONTs, the output should report a joint ambiguous scope rather than awarding false exact-location accuracy.

### 6.4 Truth and denominator audit

Produce `localisation_truth_audit` and `localisation_denominators` containing:

- fault counts by partition, fault type and domain type;
- entity-level versus multi-entity faults;
- affected-entity counts per fault;
- affected fraction within the true domain;
- scoreable, unscoreable and cross-partition faults;
- identifiable versus topology-equivalent locations;
- development and holdout denominators;
- confidence intervals or explicit low-sample warnings.

If development contains fewer than 20 multi-entity faults, localisation comparisons are descriptive and must not drive detailed label-based tuning. This is a reporting safeguard, not a universal statistical law.

### 6.5 Phase 0 exit gate

Proceed only when:

- every scored ONT has complete required physical-topology membership;
- every localisation truth identifier resolves to a known entity or group;
- ambiguous topology scopes are recorded as equivalence classes;
- an eligible peer level is chosen from observed peer availability;
- the number of multi-entity development and holdout faults is explicitly reported.

If no level has adequate peers, peer detection is disabled. Group common-mode detection may still operate when groups have adequate descendant coverage.

---

## 7. Detection model

The model will have four interpretable channel families. Additional algorithms are introduced only through controlled experiments against these channels.

### 7.1 Channel A: rapid self-history deviation

Purpose: detect abrupt change in one entity-metric series.

For entity `e`, metric `m` and time `t`, estimate a calibration-frozen expected value and scale:

`self_residual(e,m,t) = (transformed_value - expected_value) / robust_scale`

The transformation depends on measurement kind:

- gauges: robust level or seasonal residual;
- bounded fractions: stable bounded transformation when required;
- interval counts: variance-stabilised count transform;
- cumulative counters: increments with reset handling;
- discrete states: transition or persistence features, not continuous z-scores.

References must be fitted using calibration data only. Development and holdout values cannot update a frozen evaluation model.

### 7.2 Channel B: persistent self-history drift

Purpose: detect small sustained shifts that do not cross a rapid threshold.

Use a simple sequential accumulator such as two-sided CUSUM or sustained exceedance over self residuals. Parameters are selected in development and frozen before holdout.

### 7.3 Channel C: peer-relative deviation

Purpose: find one unusual entity relative to comparable entities at the same time.

Peer comparison must operate on self-history residuals, not raw metric values:

`peer_deviation(e,m,t) = self_residual(e,m,t) - median(self_residual of eligible peers)`

This removes persistent entity baselines before comparison. It prevents legitimate distance, hardware or installation differences from being treated as anomalies.

The scale of peer deviation is fitted from calibration observations for the relevant metric and eligible group-size band. It is **not** estimated from a handful of contemporaneous siblings at every timestamp.

The initial primary peer level should be PON because current L2 groups appear small. L2 peer scores are enabled only if the Phase 0 availability audit supports them. A parent-level fallback is explicit and recorded.

### 7.4 Channel D: group common-mode anomaly

Purpose: detect a fault affecting many siblings, where ordinary peer comparison loses contrast.

For every physical topology group and metric, calculate robust summaries of descendant self residuals:

- group median residual;
- affected-descendant fraction;
- available-descendant fraction.

The implemented v3 score is the absolute group median multiplied by the square
root of the available descendant count. It is normalised against calibration
groups in the same metric and group-size band. A score is emitted only when at
least three descendants and at least half of the known group are observed; the
affected-descendant fraction is retained as localisation evidence.

Directional agreement and immediate-child branch breadth are useful challenger
summaries, but they are deliberately deferred until the full v3 run establishes
where median common-mode evidence succeeds and fails. This avoids adding two
untested policy dimensions to a small development denominator.

This channel complements peer deviation:

- low affected fraction: peer deviation should be strong;
- broad common-mode shift: group score should be strong;
- intermediate fraction: both may contribute.

### 7.5 Optional dispersion channel

A group-dispersion change can help detect partial or heterogeneous group faults. It remains an ablation candidate rather than a mandatory fifth channel. It is retained only if it adds development recall at the same incident workload.

### 7.6 Seasonality and cadence

Seasonal modelling is conditional, not automatic.

- Candidate periods come from domain cadence and calibration-only EDA.
- A seasonal reference is used only when adequate cycles, stable coverage and repeatable seasonal strength exist.
- Otherwise the metric uses a robust non-seasonal reference.
- Window lengths are expressed in elapsed time and converted using each metric's cadence.

The shared model must not impose the Petrobras ten-second aggregation on 900-second Telecom telemetry.

### 7.7 Missing, invalid and clipped observations

- Invalid observations are not scored as numerical values.
- Clipped observations remain visible as censored evidence and are not treated as ordinary extremes.
- Expected missing observations are handled through collection-gap evidence, not interpolation across long gaps.
- Short interpolation, when required for a specific transformation, is limited, recorded and never used to manufacture evaluation truth.

---

## 8. Multiplicity and threshold calibration

The detector evaluates many metrics, entities, topology levels and channel families. Uncontrolled pointwise thresholds would create an unacceptable alert rate.

The primary calibration unit will therefore be a time block, not an individual score.

Within calibration data:

1. calculate all eligible scores in a channel family;
2. take the maximum relevant score per entity-time block or group-time block;
3. fit the empirical tail of these block maxima;
4. choose candidate thresholds by the desired alert workload;
5. evaluate configurations in development at matched false-incident burden.

Multiple topology levels inside peer or group detection are treated as one family for calibration. They are not counted as independent free channels.

This keeps the false-alarm comparison fair as capabilities are added.

---

## 9. From scores to incidents

### 9.1 Alert persistence

Point scores become alerts only after simple persistence rules appropriate to the channel:

- rapid channel: short consecutive evidence;
- drift channel: accumulator threshold;
- peer channel: repeated or strong isolated deviation;
- group channel: minimum descendant coverage and sustained common-mode evidence.

### 9.2 Consolidation

Alerts are consolidated when they overlap in time and share an entity or a justified containment relation. Consolidation should use a fixed quiet-period closure rule.

Transitive chaining across unrelated alerts is prohibited. One long incident must not be created merely because alert A overlaps B and B overlaps C.

### 9.3 Incident evidence retained

Each incident should retain:

- start, peak and end time;
- entities and metrics involved;
- contributing channel families;
- maximum and sustained normalised evidence;
- raw-alert count;
- candidate topology scopes;
- observed footprint;
- data-quality limitations;
- model and threshold version.

Ranking is based on statistical evidence and persistence. Business severity is added only when genuine impact data becomes available.

---

## 10. Hierarchical localisation

Localisation is a second stage over detected incident evidence, not a label classifier.

### 10.1 Candidate generation

Candidates include:

- individual ONTs with peer or self evidence;
- L2, L1/PON and OLT scopes with common-mode evidence;
- topology-equivalence classes where exact distinction is impossible.

Geographic clusters may be shown as context but are not physical fault candidates unless a use case independently justifies them.

### 10.2 Hierarchical selection logic

For a candidate parent scope:

- require adequate descendant availability before interpreting missing coverage as unaffected;
- penalise evidence that spills substantially outside the candidate footprint;
- return the top two candidates when evidence is ambiguous.

The implemented first pass selects the strongest calibration-normalised scope,
records its affected fraction and reports an equivalent second scope when two
topology nodes have the same observable descendants. Parent-versus-child branch
breadth and spill penalties remain predeclared localisation challengers. They
should be introduced only if full-run errors show that the current rule confuses
nested scopes, because the development localisation denominator is likely too
small for a broad rule search.

### 10.3 Output

The incident table should include:

- `scope_type` and `scope_id` for the primary predicted domain;
- `scope_type_2` and `scope_id_2` when an alternative is justified;
- `anomaly_evidence_score` for ranking;
- `footprint_size`;
- `affected_fraction_estimate`;
- `identifiability_status`;
- `location_explanation`.

The explanation should say whether the result is driven by an isolated entity, a coherent group shift or an indistinguishable topology scope.

---

## 11. Evaluation design

### 11.1 Detection metrics

Primary detection reporting:

- event recall;
- pre-impact event recall where `impact_ts` exists;
- detection delay distribution;
- false incidents per entity-day;
- alert precision as a secondary measure;
- bootstrap or Wilson uncertainty intervals as appropriate.

No configuration is called better without comparing it at the same false-incident workload.

### 11.2 Consolidation metrics

- raw alerts per incident;
- duplicate incidents per fault;
- alert-volume reduction;
- incident duration and fragmentation;
- proportion of incidents with multiple contributing metrics.

### 11.3 Localisation metrics

- exact-scope accuracy among detected, identifiable faults;
- top-two scope accuracy;
- hierarchy distance from the true scope;
- affected-footprint precision, recall and Jaccard similarity;
- accuracy after collapsing topology-equivalent scopes;
- joint detected-and-correctly-localised rate over all scoreable faults.

The joint metric prevents localisation quality from appearing strong merely because difficult missed faults were excluded.

### 11.4 Required stratification

Report performance by:

- fault type;
- domain type;
- affected fraction;
- group size;
- lead-time horizon;
- channel family contribution;
- data-quality status.

### 11.5 Required ablations

At matched workload compare:

1. self rapid only;
2. self rapid plus drift;
3. self channels plus peer deviation;
4. self, peer and group common mode;
5. optional dispersion contribution;
6. seasonal versus non-seasonal reference only for eligible metrics.

Isolation Forest and other multivariate detectors may remain challenger models. They are not automatically promoted because they are more complex. They must add stable development value at the same operational workload and preserve interpretability of the incident evidence.

### 11.6 Evaluation-harness controls

The harness must still prove:

- a deliberately leaky scorer fails when SPEC-EVAL is absent;
- canonical model input is identical with truth mounted and unmounted;
- matching handles overlapping fault windows safely;
- duplicate alerts are not rewarded as additional detections;
- low denominators are visible rather than hidden by point estimates.

---

## 12. Calibration, development and holdout policy

### Calibration

May be used to:

- fit transformations and robust references;
- estimate residual and peer scales;
- estimate seasonal components;
- form block-maximum threshold candidates;
- fit unsupervised challenger models.

Truth is not used.

### Development

May be used to:

- compare a small predeclared set of channel combinations;
- choose thresholds at an operational workload;
- choose persistence and consolidation settings;
- assess localisation rules and report uncertainty.

The number of configurations must remain small relative to the number of faults. A large hyperparameter search would overfit the synthetic fixture.

### Holdout

Used once after the pipeline and selection rule are frozen. No threshold, model, horizon or consolidation change may be justified using holdout results.

Any later design change creates a new model version and requires a new untouched holdout or an explicitly labelled exploratory status.

---

## 13. Petrobras qualification protocol

Petrobras will use the same canonical adapter, self-history model interface, incident builder and evaluation harness, but without peer or topology channels.

The default should include enough real recordings to support meaningful evaluation. `FILES_PER_WELL_EVENT=5` is a practical current starting point because it is a deterministic superset of the earlier one-file sample. Results must report:

- recordings;
- wells;
- event instances;
- event counts by class;
- clustering of recordings within wells;
- confidence intervals and minimum detectable burden.

Per-class evaluation horizons should be justified from the 3W definitions and literature rather than chosen after seeing model results.

The Petrobras pass criteria are:

- the common contract can represent the data without invented structure;
- recording episodes remain isolated;
- sentinel, frozen and missing sensor behaviour is handled honestly;
- the generic self-history fallback runs and produces evaluable incidents;
- Telecom-only topology changes do not break topology-absent sectors.

Poor Petrobras recall does not automatically reject a Telecom capability. A failure of the common interface, leakage controls or data-quality handling does.

---

## 14. Implemented notebook and code changes

### New, short feasibility notebook

`00_TELECOM_LOCALISATION_AUDIT.ipynb`

- audits topology, peer availability, identifiability and truth denominators;
- reads truth only in a clearly labelled evaluation section;
- produces the five Phase 0 evidence tables;
- makes no detector-performance claim.

### `01A_TELECOM_PACK.ipynb`

- emit model-visible topology memberships;
- preserve geographic groups as contextual rather than physical;
- add `domain_type` to evaluation truth;
- remove topology from experimental split tables;
- retain source and truth-isolation checks.

### `01A_PETROBRAS_3W_PACK.ipynb`

- remain topology-free;
- retain expanded real-well sampling and well-level partitions;
- update only for the revised shared interface and evaluation schema.

### `milestone1_core.py` and `01B_COMMON_CANONICAL_ADAPTER.ipynb`

- add optional `topology_memberships` support;
- validate it generically;
- copy it into SPEC-CORE;
- keep the adapter free of sector-name branches;
- retain truth and temporal isolation tests.

### `02_CANONICAL_EDA.ipynb`

- keep concise canonical data-quality and time-series EDA;
- restrict model-shaping conclusions to calibration data;
- add peer-availability and residual comparability plots for Telecom;
- avoid duplicating the Phase 0 truth audit;
- make plots descriptive rather than auto-generating model policy.

### `03_EVALUATION_HARNESS.ipynb` and `evaluation_core.py`

- add `domain_type` handling;
- add topology-equivalence-aware localisation metrics;
- add footprint and joint detection-localisation metrics;
- add affected-fraction stratification;
- keep random, constant, perfect, late and deliberately leaky controls.

### `04_SIMPLE_ANOMALY_MODELS.ipynb` and model core

- retain robust self rapid and drift baselines;
- add calibration-frozen residual construction;
- add eligible peer-deviation scoring;
- add hierarchical group common-mode scoring;
- calibrate family-level multiplicity with block maxima;
- compare only a small, interpretable ablation set;
- leave advanced black-box methods as challengers.

### `05_INCIDENT_RANKING_AND_HOLDOUT.ipynb`

- consolidate entity and group alerts;
- generate hierarchical location candidates;
- report top-two and ambiguity;
- select on development, freeze, then run holdout once;
- rank by statistical evidence, not invented business severity.

### `06_PRODUCT_DEMO.ipynb`

- show a small number of valuable views:
  - representative telemetry series;
  - self, peer and group evidence through time;
  - alert-to-incident consolidation;
  - topology footprint and predicted scope;
  - ranked incident list;
  - detection, workload and localisation scorecard;
- state clearly whether results are synthetic validation or real-data qualification.

`03A_OUTPUT_REVIEW.ipynb` remains an optional inspection utility rather than a pipeline stage.

---

## 15. Implementation sequence

1. Run the Telecom localisation feasibility audit.
2. Confirm the primary peer level, fallback level and topology equivalence classes.
3. Revise the canonical topology and evaluation contracts.
4. Rebuild Telecom and Petrobras packs and canonical outputs.
5. Re-run truth isolation and contract tests.
6. Update calibration-only EDA.
7. Extend the evaluation harness and validate it with controls.
8. Implement self-residual foundations.
9. Add peer deviation and group common-mode channels.
10. Compare the small development ablation set at matched workload.
11. Freeze the selected configuration.
12. Run Telecom holdout once.
13. Run the Petrobras qualification checkpoint.
14. Update the product demonstration and methodology.

---

## 16. Acceptance criteria

### Contract and leakage

- Detector-visible data is identical with and without SPEC-EVAL mounted.
- No model or threshold code reads truth, partition labels or ticket fields.
- Common code branches on capabilities and measurement kinds, not sector names.
- Topology is optional and absent sectors still run.

### Statistical validity

- All transformations, references and score scales are fitted on calibration only.
- All development comparisons use the same false-incident workload.
- Multiplicity across metrics and levels is calibrated.
- Uncertainty and evaluation denominators are reported.
- Holdout is untouched until selection is frozen.

### Detection and operations

- Self-only remains a working fallback.
- Peer scoring is emitted only with adequate eligible peers.
- Common-mode group faults have a dedicated detection path.
- Repeated alerts are consolidated without unsafe transitive chaining.
- Incident output retains traceable contributing evidence.

### Localisation

- Truth contains both domain type and identifier.
- All scored truth resolves to known topology.
- Indistinguishable scopes are reported as such.
- Exact, top-two, hierarchy-distance, footprint and joint metrics are reported.
- Entity-level and multi-entity fault results are separated.

### Claims

- Synthetic Telecom results are labelled mechanism validation.
- Petrobras results are labelled real-data qualification.
- No real Telecom performance claim is made until real Telecom data is assessed.

---

## 17. Deliberately deferred work

The following are valuable but not part of the present implementation:

- fault-horizon prediction;
- supervised fault-type classification;
- causal root-cause graphs;
- dynamic topology membership intervals;
- streaming deployment and online state stores;
- operator-feedback learning;
- business-impact severity;
- deep sequence models;
- a large hyperparameter optimisation framework.

They should be introduced only when the required data, evaluation denominator and operational consumer exist.

---

## 18. Frozen design decisions

The implemented decisions are:

1. Telecom is the primary product-development sector.
2. Petrobras is a topology-free real-data qualification test.
3. Topology becomes an optional model-visible SPEC-CORE capability.
4. Evaluation truth gains an explicit `domain_type`.
5. Peer comparison uses self-history residuals rather than raw values.
6. Peer scale is fitted from calibration, not instantaneous small-group MAD.
7. Peer and group common-mode detection are complementary channels.
8. PON is the provisional primary peer level; Phase 0 may revise it.
9. Hierarchical localisation reports ambiguity and top-two candidates.
10. Development comparisons are made at matched false-incident workload.
11. Holdout remains untouched until selection is frozen.
12. Prediction remains outside the current scope.

Phase 0 findings may alter the chosen topology level, but must not silently
alter the statistical or leakage principles above. Full Telecom results remain
synthetic-mechanism evidence until the same frozen workflow is evaluated on
real access telemetry and reviewed with operators.

---

## 19. Statistical references

- Tatbul et al., *Precision and Recall for Time Series*, NeurIPS 2018: range-
  based anomaly evaluation and explicit treatment of temporal anomaly windows.
  https://papers.neurips.cc/paper_files/paper/2018/hash/8f468c873a32bb0619eaeb2050ba45d1-Abstract.html
- Garg et al., *On the Evaluation of Time Series Anomaly Detection*, 2021:
  evidence that evaluation protocol choices and trivial baselines can dominate
  apparent multivariate time-series anomaly-detection performance.
  https://arxiv.org/abs/2109.11428
