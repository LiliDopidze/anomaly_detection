# Version 4 results and decision

The requested framework is implemented and evaluated. **Deployment remains disabled.** None of the 48 standalone policy trials or 39 system/grouping candidates met the predeclared development requirements. The frozen primary is `level_forest:g120`: U-level plus compact Isolation Forest, with a 120-minute grouping anchor. It is a diagnostic comparator chosen under the documented no-feasible-policy rule, not an approved deployment. No new winner is selected from final results.

## Evidence and limits

The unchanged generator produced 96,768 native rows, 24 physical faults and 12 entities across 28 days. The first 55% is checked healthy; 55–75% selects policies; 75–100% is the frozen assessment. All 24 metric references qualified for one daily harmonic under the training-only rule. The median within-entity training residual correlation between the two power channels is 0.720, supporting the declared physical channel pair; this is not evidence that every relationship fault will be detected. Window lengths, tree settings, row selection, seeds, policies and budget are recorded in the decision contract and manifests.

The final period is previously explored seed-42 synthetic evidence. It is not a fresh independent test or field validation. There is one independent OLT block, so the uncertainty report correctly declines to give block confidence intervals. Variance faults lack independently defensible observable onset; they are shown separately. Precision is the fraction of investigations matched to known observable faults, with unverified investigations retained in the denominator. Reported FP excludes those unverified cases. Nuisance workload conservatively includes every unmatched investigation.

## Frozen system results

| period                     |   TP |   FP |   unverified |   FN |   investigations |   recall |   early_recall |   unmatched_per_1000_days |   any_coverage |   full_coverage |   alarm_fraction |
|:---------------------------|-----:|-----:|-------------:|-----:|-----------------:|---------:|---------------:|--------------------------:|---------------:|----------------:|-----------------:|
| Development                |    8 |   52 |           83 |    0 |              143 |    1.000 |          1.000 |                  2008.929 |          0.982 |           0.805 |            0.370 |
| Frozen temporal assessment |    8 |   70 |           65 |    0 |              143 |    1.000 |          1.000 |                  1607.143 |          0.979 |           0.778 |            0.266 |

The proposed limits were recall and required-lead recall at least 0.95, unmatched workload at most 5 per 1,000 scheduled entity-days, any-route coverage at least 0.95, full-capability coverage at least 0.75, median detected delay at most 60 minutes, and time in alarm at most 0.20. Required lead is 30 minutes. These are research proposals, not operator-approved limits.

Development has 8 faults with known observable onset, 4 unknown-onset faults and 5 known-impact opportunities. Final has 8 known-onset faults, 4 unknown-onset faults and 4 known-impact faults. Scheduled exposure is 67.2 and 84.0 entity-days respectively, independent of metric count or score availability. Final median detected delay is 12.5 minutes over 8 matches; median lead is 1527.5 minutes over 4 known-impact matches. Ongoing and boundary events receive separate reports; neither gains new-event credit.

## Separate route assessment

Each row uses its own development-selected thresholds and persistence; all rows share fault opportunities, monitored power channels and scheduled exposure. Distance modes remain separate. Common-support event replays and timestamp confusion matrices accompany these operational results.

| candidate            |   tp |   fp |   unverified |   fn |   recall |   early_recall |   nuisance_per_1000_entity_days |   any_coverage |   full_coverage |
|:---------------------|-----:|-----:|-------------:|-----:|---------:|---------------:|--------------------------------:|---------------:|----------------:|
| u_level:q1-n1        |    8 |   49 |          126 |    0 |    1.000 |          1.000 |                        2083.333 |          0.979 |           0.979 |
| u_shift:q0-n1        |    8 |   39 |           35 |    0 |    1.000 |          1.000 |                         880.952 |          0.979 |           0.979 |
| if_compact:q0-n1     |    8 |   43 |            5 |    0 |    1.000 |          1.000 |                         571.429 |          0.778 |           0.778 |
| if_historical8:q1-n1 |    7 |    5 |            5 |    1 |    0.875 |          1.000 |                         119.048 |          0.778 |           0.778 |
| mp_shape_6:q0-n1     |    8 |  334 |           24 |    0 |    1.000 |          1.000 |                        4261.905 |          0.882 |           0.882 |
| mp_level_6:q0-n1     |    8 |  148 |           17 |    0 |    1.000 |          1.000 |                        1964.286 |          0.882 |           0.882 |
| mp_shape_12:q0-n1    |    7 |  206 |            6 |    1 |    0.875 |          1.000 |                        2523.810 |          0.778 |           0.778 |
| mp_level_12:q2-n1    |    8 |   13 |            9 |    0 |    1.000 |          1.000 |                         261.905 |          0.778 |           0.778 |

Development operating curves and the Pareto table retain all rejected trials. Statistical, forest and distance candidates were compared before the five named combinations. No raw scores were fused. Under the declared 95% any-route availability requirement, standalone windowed routes are often ineligible because gaps require complete-window re-warm-up. This is a coverage cost, not a reason to remove missing intervals from the denominator.

## Grouping and failure diagnosis

For the frozen primary, development grouping reduced unmatched workload by 42.3%, below the proposed 50% requirement. Ordinary and early recall losses were both zero. Final ungrouped workload has 239 investigations and 231 unmatched; grouping has 143 and 135, a 41.6% reduction. Final member-level audit identifies 0 merge-risk groups and 7 fragmented faults. These are diagnostic associations, not root-cause assignments.

Eleven isolated diagnostic cases cover spikes, subtle shifts, steps, ramps, relationship-only change, benign modes, a missing channel, repeated faults and 0.5%, 1%, 2% contaminated training. They are separate from generator truth and use a fixed training-q99 policy without fixture tuning. Both forest profiles receive zero new-event credit on the relationship-only fixture: the compact route opened at 07:35 before the 08:00 fault onset and stayed active, while the eight-feature route did not open. This diagnoses both lifecycle masking and a representation/threshold miss; it does not establish that the forest score is wholly insensitive. Marginal detector matches inside that interval can be incidental and do not prove dependence learning. Compact forest detects one of two repeated faults; historical eight features detect both in this fixture. Shape distance misses step/ramp/repeated offsets; the level route retains that information. Benign modes still generate unmatched workload. Contamination sensitivities are measured, not a robustness guarantee.

Covariance, instantaneous features and clustering remain deferred. A focused challenger could investigate the recorded relationship failure in a new declared experiment, but no added model is justified as a deployed component by the present no-go evidence.

## Runtime and verification

Development score computation (shared residual/feature construction excluded; original verified timings reused):

| route          |   seconds |   scheduled_scores |   valid_scores |   frame_bytes |
|:---------------|----------:|-------------------:|---------------:|--------------:|
| mp_level_12    |    39.285 |             145152 |         114812 |      81667776 |
| if_compact     |     0.368 |              72576 |          57406 |      30468082 |
| u_shift        |     0.016 |             145152 |         142342 |      57276646 |
| if_historical8 |     0.351 |              72576 |          57406 |      30758386 |
| u_level        |     0.015 |             145152 |         142342 |      57276646 |
| mp_shape_12    |    58.123 |             145152 |         114812 |      81667776 |
| mp_shape_6     |    97.635 |             145152 |         128996 |      81309864 |
| mp_level_6     |    79.861 |             145152 |         128996 |      81309864 |

Final full-history replay cost:

| route          |   seconds |   scheduled_scores |   valid_scores |
|:---------------|----------:|-------------------:|---------------:|
| u_level        |     0.017 |             193536 |         189718 |
| u_shift        |     0.018 |             193536 |         189718 |
| if_compact     |     0.474 |              96768 |          76219 |
| if_historical8 |     0.491 |              96768 |          76219 |
| mp_shape_6     |   111.617 |             193536 |         171674 |
| mp_level_6     |    86.687 |             193536 |         171674 |
| mp_shape_12    |    70.405 |             193536 |         152438 |
| mp_level_12    |    44.591 |             193536 |         152438 |

These are local batch timings, not online latency guarantees. `frame_bytes` is the score table footprint, not process peak memory. The full banks were retained with no bank reduction. Matrix profiles use STUMPY 1.14.1 MASS and exact eligibility masks, checked against a direct RMS oracle.

138 tests passed; one legacy optional SHAP test was skipped because SHAP is not installed. Tests cover generator/adapter parity, scale arithmetic, constants, ramps, gaps, strict equality, N=1, decision stride, direct distance agreement, affine shape invariance, overlap exclusion, frozen bank identity, repeated-fault exclusion, future mutations, fitting/selection boundaries, save/load, reversed route execution order, grouping anchors and member closures, maximum-cardinality/early matching, missing score rows and independent exposure. The package wheel builds and contains both the v4 package and unchanged optical generator. All four notebooks are executed against these artifacts; executed copies are stored with the run.

The historical verification_v11 source `ec89fb14d38b890c3a3d5040df853c7b6305b075` reconciles all 17 source hashes, 6,912 observations and six truth events. Telemetry/truth are exact; derived floats agree at absolute tolerance 1e-10 with unchanged policies and counts. The separately interrupted larger historical attempt remains incomplete. See the migration note.

## Artifacts and provenance

Candidate source: `5110a8f501ac554b27e9d77308405bdd507f59e7`. Numerical cache lineage records snapshot `9120d67` and verifies source functions, dependencies, configuration and data hashes before reuse. The earlier development result remains untouched. The audited run was frozen before `FINAL_OPENED.json`; `FINAL_RESULT.json` records `selection_changed=false`.

The local artifact root is `/Users/lilidopidze/Documents/Anomaly Detection/outputs/industry_agnostic_v4`. `verified_run/` contains input/fit/freeze manifests, both trial tables, final route/system tables, model and banks, scores, lifecycle traces, notifications, grouping membership, quality report, complementary misses, ordinary/early matches, unknown-onset/boundary/ongoing reports, common-support comparisons and plots. `diagnostic_fixtures/` contains the separate sensitivity counts, exact contamination row IDs and failure plots. `historical_v11/` and the reproduction/revalidation records contain the retained historical evidence. `verification_tests.txt` records the complete suite output. Experiment outputs are ignored by Git.

Runnable commands are in README.md; `CHANGED_FILES_V4.txt` is the actual branch change map. `PILOT_AND_MAINTENANCE.md` and empty forms under `pilot/` prepare operator review, independent fault ascertainment, shadow replacement, activation and rollback. No field pilot, service, deployment or GitHub push has been performed. Local and remote main remain unchanged; work is committed only on `development/industry-agnostic`.
