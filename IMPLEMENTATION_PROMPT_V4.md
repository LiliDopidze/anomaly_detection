# Implementation prompt for company agnostic time series anomaly detection

Version 4 — 5 October 2026

## Role and governing specification

Act as a lead data scientist, statistician and practical Python developer with time-series expertise. Implement the accompanying **Company Agnostic Time Series Anomaly Detection**, version 4 dated 5 October 2026, in https://github.com/LiliDopidze/anomaly_detection. Read the actual methodology and repository instructions first. If the document or repository cannot be accessed, state the limitation and complete useful work that does not require inventing their contents.

Build a compact, understandable framework. Explain modelling choices in ordinary language and connect each addition to a measurable failure or benefit. Mathematical correctness, causal evaluation and readable code take priority over adding algorithms. Do not claim equivalence to a commercial product or cross-industry efficacy without evidence.

This prompt and the version 4 methodology supersede earlier conflicting designs. **Implement and independently evaluate three main routes: statistical residual detection, feature-based Isolation Forest, and matrix-profile-based detection.** Matrix-profile implementation is required; deployment is conditional on evidence. Use four features per channel as the initial forest representation, retaining the previous eight-feature profile as a bounded historical comparison. Mahalanobis and clustering are conditional extensions. A separately evaluated event-level hybrid and causal incident grouping are authorised; raw-score averaging and arbitrary cross-model maxima are not.

This is an implementation instruction, not a request for another plan. Inspect, reproduce, implement, compare and deliver in verified milestones. Failed operational targets are valid findings; do not fabricate success, change truth or tune against final test.

## Repository boundaries

Inspect actual status, current branch, remotes, source SHA, history, dependencies, configuration and tests before editing. Keep main unchanged locally and remotely. Create or resume development/industry-agnostic without overwriting unrelated work. Reproduce the historical source version in an isolated checkout or detached worktree; do not invent a benchmark revision. Do not force push, reset unrelated work, create an unnecessary benchmark branch or merge into main.

Commit passing milestones on the development branch. Push only when authorised in the active environment. Record actual local and remote status. On restart, inspect saved commits, outputs and manifests and resume the last verified stage. Do not regenerate or overwrite completed experiments unnecessarily.

Keep immutable experiment outputs outside tracked code unless they are intentional small fixtures. Record source and dependency versions, data/configuration hashes, feature order, selected training row identities, seeds, split boundaries and enabled routes. Retain failed runs with clear status.

## Architecture and scope

Implement the following responsibilities separately while reusing data, time, state and evaluation helpers:

1. **Statistical:** U-level directional residual baseline, with U-shift CUSUM as a separately evaluated candidate for sustained change.
2. **Isolation Forest:** independent feature-based route using compact causal features per declared channel group. Do not gate it on univariate alerts.
3. **Matrix-profile-based:** independent query-to-normal-bank subsequence route with shape and residual-level variants, independently evaluated. Use established distance/profile primitives and exact eligibility rules.
4. **Hybrid:** only after separate-route comparison, evaluate a small named set of event-level combinations. Any enabled route may trigger; agreement is not required.
5. **Grouping:** separate causal consolidation of confirmed events into investigations, initially within entity.

Keep scores, states, events, coverage and metrics identifiable by route. Select deployment on development evidence; implementing three candidates does not require deploying all three. No model or grouping stage establishes root cause or a fault probability. Do not add a plugin registry, web service, dashboard, workflow engine, deep model or automatic pooled-reference system.

Use the explicit mathematics below. MATLAB's timeSeriesIforestAD, timeSeriesSpcAD and distance-method documentation support modular workflows; they do not define our four features, CUSUM or frozen-bank policy. Do not claim Python/MATLAB parity without an actual matched-input test and verified versions. Do not implement an extra MATLAB codebase unless explicitly requested.

## Data and causal time contract

Canonical fields: entity_id, metric_name, event_time, available_time, value, quality. Store a monitoring schedule independently of observations. The registry specifies units, gauge/count/counter semantics, cadence, interval aggregation, transformation, adverse direction, support requirements, validity and positive scale floors. IDs are keys, not numeric features.

At decision time a, use only inputs available by a. Preserve window event-time bounds separately from decision availability. Never backdate alerts or rewrite earlier emitted notifications when late data arrives. Retrospective recomputations need separate labels.

Validate duplicates, units, counter resets/wraparound, rate denominators and compatible intervals. Never impute unknown error counts as zero. No centred windows or future interpolation. Keep truth, simulator latent states and diagnostic alarms out of predictors. Metadata/topology may support grouping but must not enter a detector silently.

Reuse the current synthetic generator and GPON adapter, verifying fixed-seed numeric telemetry/truth parity. Keep optical semantics in the adapter. Do not alter equations or random draws to improve detection. Add isolated fixtures for unsupported scenarios and a non-optical portability example, labelled separately from the benchmark.

Require complete short windows for windowed and distance routes. Gaps reset affected recursive state and require re-warm-up. Point scoring may resume on a valid observation with a fitted reference. Continue valid univariate channels when a joint route lacks a required channel. Never change a fitted vector's dimension at scoring time. Unknown entities initially abstain without a fitted local reference. Data-quality incidents remain separate from behaviour anomalies.

## Normal reference and residuals

For each entity/metric, transform y=g(x) using a documented domain-compatible transformation, identity by default. Fit m(t,u) on representative eligible normal training history. Begin with a median; compare a small seasonal or contextual reference only when training evidence supports it. Use known operating state before proposing learned regime clustering. Contemporaneous context must be available at scoring; assess whether conditioning could remove the fault signal.

Seasonal references may use a bounded set of sine/cosine harmonics with periods in the same units as time. OLS is not robust to contamination. Inspect residual seasonality, autocorrelation, tails and variability by operating condition. Do not automatically fit a Gaussian residual interpretation to sparse counts, intermittent activity or small-denominator proportions. Mark unsupported metrics or evaluate a justified extension.

Use training residuals r_i=y_i-m(t_i,u_i), residual centre b=median(r), scale s=max(1.4826*median(abs(r-b)), metric_floor), and z_t=(y_t-m(t,u_t)-b)/s. Freeze all fitted objects within a fold. The expected transformed value is m+b. MAD scaling does not make residuals normal; score thresholds and bands are not calibrated future tail probabilities.

For constant metrics, permit a univariate change check only with a defensible scale floor. Reject all-constant joint training. Minimum duration, observations, independent cycles and mode support must be declared; overlapping rows do not imply independent evidence.

## Univariate scores

U-level score is max(0,z) for high direction, max(0,-z) for low direction and abs(z) for two-sided monitoring. Its policy grid must include one-decision opening when point spikes are in scope.

U-shift uses Cplus=max(0, previous_Cplus+z-k) and Cminus=max(0, previous_Cminus-z-k), initialised at zero at segment boundaries. Use the configured direction or max(Cplus,Cminus) for one explicitly two-sided CUSUM. This maximum is internal to that detector and does not authorise cross-model score fusion. Fix k>0 and reset rules before scoring. Initially continue state through alerts; reset at gaps/segment boundaries, not automatically at every trigger. Report slow recovery and time in alarm. Cadence changes require reselection.

Classical k=delta/2 intuition applies under a normal unit-variance shift model, not as an automatic guarantee for the observed residual process. Empirically validate the actual policy. Keep U-level and U-shift scores, events and reason codes independently identifiable.

## Joint feature profiles and estimators

Fit each initial joint model per entity and declared channel group. Require local training support. Pooling across entities is a separate candidate, not a silent fallback. Use training residual relationships and operational meaning to choose groups; common raw seasonality alone is insufficient. Do not perform automatic lag searches by default.

Implement the compact default first: short mean, short population standard deviation, OLS slope and short-minus-long. Retain the following eight-feature representation as a historical ablation where applicable, not a mandatory production vector:

1. Short-window residual arithmetic mean.
2. Short-window population standard deviation, ddof=0.
3. EWMA of residual differences per elapsed hour: d_i=(z_i-z_previous)/delta_hours, alpha_i=1-2**(-delta_hours/H), E_i=alpha_i*d_i+(1-alpha_i)*E_previous. Initialise with the first valid difference; H>0 is a declared half-life.
4. OLS slope against elapsed hours using centred time and residual values.
5. Mean of permitted contiguous long history, at least the short-window minimum.
6. Short mean minus long mean.
7. Fraction with z>b0, z<-b0 or abs(z)>b0 by direction, using strict inequalities and b0>0.
8. Directional CUSUM as above.

Use right-closed (t-W,t] windows with a regular grid and declared stride. Require W_long>W_short. Record effective long history. Replay full recursive history or exact persisted state; a short buffer is not equivalent.

Preserve excluded summaries for diagnostics when useful. Compare compact and historical eight-feature profiles under matched channels, row sampling and tuning budget. No automatic feature elimination or SHAP selection. A relationship fixture may motivate a later declared instantaneous residual-vector ablation if summary windows lose the signal; do not silently add it.

Fit feature mean/std scaling on eligible training vectors, with positive floors for near-constant columns, distinct from metric residual scaling. Do not clip future extremes by default. If the optional covariance challenger is justified, it and the forest receive the same ordered/scaled vectors and sampled row identities in their controlled comparison.

Only add M-Gaussian after a documented relationship failure motivates a targeted challenger. If implemented, use Ledoit-Wolf covariance Sigma=(1-lambda)*S+lambda*trace(S)/p*I and q=(v-mu)' solve(Sigma, v-mu). Require positive definiteness and finite distances; persist any tiny scale-relative ridge. Test exact dependence of short-minus-long on the two means, constant columns and conditioning. Shrinkage is not contamination robustness and does not remove redundant weighting. No chi-square false-alarm claim. Under a fixed Gaussian, negative log-density and q rank identically.

For M-IF, inspect and record actual repository settings. If there are no applicable settings, start with 100 trees and min(256, eligible rows) samples per tree as an explicit candidate, not a claimed optimum. Use a fixed seed and -score_samples. Record training stride and deterministic row identities; overlapping windows can overweight persistent modes. Do not assume contamination estimates real fault prevalence. Positive affine feature scaling is not intrinsically required by axis-aligned isolation trees; keep any scaler frozen for consistency. A training-constant feature cannot furnish useful isolation splits; retain residual detection for changes in constant metrics. Temporal information comes from features. Joint scores are generally two-sided novelty measures; adverse direction and business severity are separate interpretation fields.

## Required separate matrix-profile-based route

Implement and evaluate the route independently, even if it is not deployed. Keep distance-profile and matrix-profile operations as established numerical primitives or thin wrappers; motifs and retrospective discords may support diagnostics. Do not implement a second large framework. Operational D_t=min distance(query, eligible frozen normal-training reference window), directed query-to-bank, per entity/metric.

Compare separately: (a) shape RMS Euclidean distance after per-window population z-normalisation of the transformed measurement, and (b) level-preserving RMS Euclidean distance on frozen metric residuals without window recentering. For nonconstant windows, shape distance equals sqrt(2*(1-correlation)); it is invariant to positive affine copies and can miss offsets/amplitude changes. Exclude near-constant shape references and abstain on near-constant shape queries under an explicit floor; level mode can score constants. Convert Euclidean library distances to RMS by dividing by sqrt(m). Different window lengths still need separate calibration.

Use m=W/cadence integer >=3, complete windows and a bounded duration shortlist. Freeze the bank before development; no development/test windows enter it. Exclude all shared raw observations, including boundary overlap. Require start separation >=m within one segment; the remaining eligible bank must contain at least two mutually non-overlapping references as a conservative project support floor. Other overlapping eligible references can remain candidates. This floor is not sufficient evidence of representative normality. Recompute the next eligible nearest match when a library's returned match is invalid. Do not calibrate using self-match zeros.

Never create windows across gaps or concatenation seams. Record bank hashes, source times, exclusions, support and nearest references. Cap support only by a deterministic predeclared training-only rule when needed. Verify against a direct oracle. Full-series self-joins are retrospective and cannot establish causal lead. Frequent motifs are not necessarily healthy, and repeated faults must not automatically become normal.

Evaluate standalone performance against both other routes first. Select deployment alone or in a combination using the common workload, coverage and resource criteria; do not require it to beat an already selected hybrid just to receive an experiment. Do not fuse shape and level scores. A standard self-join, a causal left profile and a frozen normal-bank AB comparison have different eligible neighbours: label them accurately. This implementation is not DAMP. A growing past-only bank can still absorb repeated faults; no automatic bank updates are allowed. Emit only after all query inputs are available, never at window start. Measure delay rather than assuming it equals window length. Use direct O(Q*B*m) distance only as a small oracle; measure the established implementation on the pilot before considering bank reduction or a streaming algorithm.

## Detector state and hybrid grouping

Each route opens after N_open consecutive valid decisions strictly above theta_open, and recovers after N_close strictly below theta_close, where theta_close<theta_open. Between thresholds retain state but reset the relevant confirmation run. Equality does not qualify. N decisions at stride delta span (N-1)*delta; use decision stride rather than raw sample cadence. Emit at confirmation and never backdate. Missing scores interrupt confirmation; they are not recovery. Extended gaps may cause a separately labelled administrative closure.

Hybrid inclusion permits events from any selected statistical, forest or matrix-profile route. Select the whole enabled policy on development data, including total workload. Do not demand both detectors agree. Do not claim independent false alarms or infer system error rates from training quantiles. Preserve raw scores and events; do not hide duplicate workload.

Initial grouping is within entity using configured symptom compatibility and a maximum span G measured from the group's first confirmation. Process in availability-time order. Attach only to an eligible active group; select earliest anchor then stable ID on ties. Otherwise create a group. Never extend G using the last arrival; prevent transitive chaining. Emit a new investigation immediately on its first constituent confirmation; later alerts are updates, not retrospective changes. Preserve membership history and constituent times.

The group recovers when all members recover. Any administrative member closure prevents labelling the whole closure as confirmed recovery. Events beyond G form new groups even if an older group remains active. Test this explicitly. Cross-entity grouping is disabled initially and needs separate group-level truth, exposure and evaluation if later introduced. Association is not root cause.

## Splits and staged comparisons

Use chronological train/development/test boundaries. Proposed 55/20/25 shares apply only if the current generator's early 55 percent is confirmed healthy; they are not defaults for real data. Keep original splits and Combined behaviour only inside isolated historical reproduction. Previously viewed test periods are not untouched.

All fitting uses training. Development selects a bounded shortlist of thresholds, persistence, baselines, features and grouping policies; log every trial. Deduplicate quantile ties and predeclare bounded edge extension. Permit causal past context across splits without refitting; replay state rather than resetting it arbitrarily. Final test remains closed until models, policies and primary deployment selection are frozen.

Run sequentially, avoiding a full Cartesian search:

1. Historical reproduction, preserving original behaviour in isolation.
2. U-level versus U-level plus U-shift, using the same reference candidates.
3. Compact Isolation Forest; bounded historical eight-feature comparison where applicable.
4. Separate shape and residual-level matrix-profile candidates at a short duration shortlist justified by expected anomaly duration.
5. Independent three-route comparison: workload/recall curves, coverage, delay and complementary misses.
6. A small predeclared set of route combinations, followed by before/after causal grouping.
7. Covariance, instantaneous/lagged features or clustering only if a diagnosed failure justifies an extension.

Compare each route under the same investigation budget, not the same numeric threshold. Keep matched-channel and matched-opportunity comparisons alongside full operational results. The initial selection rule retains configurations meeting workload/coverage requirements, optimises the predeclared primary recall objective, and resolves ties using delay and complexity with stated tolerances. Record a Pareto table when no configuration dominates. Do not run a full Cartesian hyperparameter search.

Predeclare recall, required-lead recall, availability, workload, delay and time-in-alarm requirements and a simplest-feasible tie rule. Carry historical numerical targets forward only as explicitly labelled proposals: recall >=95%, recall/lead-recall loss <=2 percentage points, coverage loss <=1 point, nuisance reduction >=50%, and workload <=5/1,000 entity-days. Clarify their baselines and applicability before results. No feasible solution is a result. Do not select a new winner from test and present its selected performance as unbiased.

## Evaluation contract

Schedule exposure is fixed independently of score availability and metric count. Timestamp confusion matrices use known truth and valid scores only; report exclusions, common-support comparisons and full coverage. Distinguish raw threshold decisions from active-event state. Hybrid schedule is the predeclared union of route decision times. Report any-route coverage and full-required-capability coverage separately.

Fault truth has independent observable onset o, impact p when known and end e. Use half-open [o,e) intervals. New alerts are eligible only when confirmed in the same-entity fault interval. Pre-existing active events and boundary faults receive a separate ongoing-event report. Do not equate simulator injection time with observable onset without justification.

Use deterministic maximum-cardinality one-to-one event matching, with earlier confirmation then stable IDs for ties. Report TP/FP/FN, precision, recall and F1; no natural event TN. Duplicates remain unmatched workload. One grouped investigation cannot get credit for several independent faults. With incomplete truth, call unmatched investigations unverified rather than proven false positives.

Lead=p-confirmation and delay=confirmation-o. For minimum lead L, run a separate matching restricted to confirmation<=p-L. Report early recall across all impacting faults and separately across the fixed opportunity subset with known times and p-o>=L. Misses and score-unavailable cases stay in the applicable denominator. Show exclusions, detected-only sample sizes and unknown times. No point adjustment.

Nuisance/1,000 entity-days = 1,000*unmatched investigations/scheduled entity-days, using the union of scheduled monitoring intervals per entity. Do not shrink exposure for missing scores or multiply it for channels. Also report total investigations, raw events, time in alarm, recovery delay, longest gaps and administrative closures. Inspect incorrect merges and fragmentation. Changing to cross-entity grouping changes the evaluation unit and requires separate group truth/exposure.

Use paired resampling over appropriate independent site/entity/time blocks and raw counts; no independent bootstrap of overlapping rows. Be explicit when few events limit precision. Test spikes, subtle shifts, gradual changes, relationship-only failures with normal marginals, benign modes, missing channels, repeated anomalies and supported contamination scenarios (0.5%, 1%, 2%). Keep unsupported fixtures separate from generator results. Add independent real evidence before transfer claims.

If labels are incomplete, propose a monitored pilot with operator review; do not report reliable recall without sufficient event ascertainment. No observable precursor means no achievable positive warning lead for that event.

## Verification and delivery milestones

1. **Inspect and reproduce.** Record source, dependencies and repository state; reproduce historical results. Deliver a run manifest and decision contract. Gate: input and truth counts reconcile.
2. **Validate data and schedule.** Implement canonical tables, adapter rules, segmentation and a non-optical fixture. Gate: units, counters, gaps, availability times and exposure reconcile.
3. **Freeze experimental boundaries.** Save chronological splits, eligible normal training, truth/boundary rules, candidate budget and held-out access status. Gate: selection is isolated from test data.
4. **Fit references.** Fit median or justified seasonal/contextual references and frozen residual scales; inspect diagnostics. Gate: arithmetic, floors and causal invariance hold.
5. **Implement statistical detection.** Produce U-level/U-shift scores and lifecycle traces. Gate: spike, shift, ramp, equality, stride and gap cases behave as specified.
6. **Implement Isolation Forest.** Build compact vectors, fit deterministic model and compare historical features where applicable. Gate: feature arithmetic, persistence and independent relationship-only scoring are checked; missed fixtures remain reported.
7. **Implement matrix-profile detection.** Build normal bank and separate shape/level candidates, retaining nearest-match provenance. Gate: direct-distance agreement, overlap/constant/repeated-fault fixtures and causal-prefix invariance pass.
8. **Select on development.** Produce separate-route operating curves, then bounded combinations/grouping. Gate: all trials logged; every added component has a measurable justification or is rejected.
9. **Freeze and assess.** Save artifacts, policy and primary decision before final assessment. Deliver event counts, uncertainty, workload, coverage, delay/lead and limitations. Gate: no post-test tuning is called unbiased evaluation.
10. **Prepare pilot and maintenance.** Deliver operator review, independent fault ascertainment, drift review, shadow replacement and rollback procedures. Do not claim the pilot was run unless it actually was.

Continue autonomously through authorised implementation milestones. Stop tuning to resolve a failed correctness gate; do not use model complexity to hide data leakage or incorrect evaluation.

Required focused tests: generator parity; adapter semantics; residual centring/scaling; constant/ramp/step and feature boundaries; directional CUSUM; covariance singularity/dependence if implemented; distance RMS/Euclidean conversion, positive affine shape invariance, constant-window validity, overlaps, bank identity and repeated-fault exclusion; future mutations with fitted artifacts fixed; fitting/selection isolation; save/load; state independence and reversed route execution order; threshold equality, N=1 and decision stride; gaps; grouping anchor-span/chaining and notification time; ordinary/early event matching including a greedy-failure case; undefined metrics, duplicate alerts and schedule denominators. Use suitable numeric tolerances and exact keys/masks/counts. Stop tuning on unexplained parity, leakage or exposure errors.

Keep the final package close to existing structure: adapters, validation, references/features, detectors, incidents, grouping, evaluation, diagnostics/plots and pipeline. Include a small distance_profiles module for the required separate route. Use four notebooks: 00_data_exploration, 01_references_and_features, 02_fit_and_select, 03_evaluate_and_explain. A simple fit/score/detect/group/evaluate/save/load interface suffices. Scoring never refits and plotting never mutates.

Define versioned maintenance: monitor quality, residuals, support, coverage, workload and outcomes; review drift; select eligible refit data; replay and shadow-test the replacement; record activation and rollback. Do not automatically absorb alerts into normal history. Carry ongoing investigation identity through version changes. Evaluate a future adaptive policy as a whole rather than citing frozen-model evidence.

Deliver runnable commands, actual changed-file map, test results, source/candidate SHAs, experiment artifacts, configuration provenance, migration notes, plots of representative successes/misses/nuisance cases, and a concise decision record. Provide separate route and complete-system metrics. Do not label unimplemented work as complete or synthetic performance as production validation.

## References and source discipline

Read the version 4 methodology's numbered references and verify software against the installed versions. Cite primary papers/documentation for technical claims and label Anodot as vendor architecture material. Core sources:

- Anodot normal modelling: https://www.anodot.com/blog/closer-look-time-series-anomaly-detection/
- Anodot alert grouping: https://www.anodot.com/blog/anomaly-detection-techniques-focus-multivariate-univariate/
- Schmidl et al., PVLDB 2022: https://www.vldb.org/pvldb/vol15/p1779-wenig.pdf
- Garg et al., TNNLS 2022: https://arxiv.org/abs/2109.11428
- Hyndman and Athanasopoulos, seasonal representations: https://otexts.com/fpp3/dhr.html
- NIST CUSUM: https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc323.htm
- LedoitWolf: https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html
- IsolationForest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
- Matrix Profile I: https://sites.google.com/site/icdmstamp
- STUMPY directed AB joins: https://stumpy.readthedocs.io/en/latest/Tutorial_AB_Joins.html
- Kim et al., AAAI 2022 evaluation: https://ojs.aaai.org/index.php/AAAI/article/view/20680
- MathWorks Isolation Forest workflow: https://www.mathworks.com/help/predmaint/ref/timeseriesiforestad.html
- MathWorks statistical process control: https://www.mathworks.com/help/predmaint/ref/timeseriesspcad.html
- MathWorks distance methods: https://www.mathworks.com/help/predmaint/ug/detecting-anomalies-in-time-series-using-distance-based-methods.html
- Liu, Ting and Zhou, original Isolation Forest, ICDM 2008: https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf
- Lu et al., DAMP / Matrix Profile XXIV, KDD 2022: https://www.cs.ucr.edu/~eamonn/DAMP_long_version.pdf

These references support components, not a guarantee about this complete pipeline. Numerical budgets, feature profiles, causal grouping and acceptance criteria are project choices. Keep claims proportional to the evidence actually produced.
