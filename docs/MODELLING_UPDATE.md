# Modelling update: gap-tolerant history and explicit missing inputs

This change responds to the frozen development diagnostics: 25 faults stayed
below threshold, 10 had isolated crossings, long-history features were mostly
unavailable, and no shared-network fault was correctly localised.

## Changes

1. Historical baselines and occurrence rates retain observations across gaps
   of at most six hours. They use real elapsed-time windows, require at least
   50% observed support, and reset after longer gaps in valid measurements.
   History excludes the current observation. Exact lags, differences,
   counter totals and alert persistence keep their existing strict gap rules.
2. Base and temporal Isolation Forest use calibration residual medians plus
   explicit missing-input indicators. A missing measurement is no longer
   represented identically to an observed neutral residual. Training and
   scoring require at least half the selected inputs, with a minimum of two.
   The medians are frozen in the fitted bundle; scoring never refits them.
3. Development selection ranks eligible candidates by 48-hour recall, with
   the existing simplicity tie-break. Recall-confidence, workload-confidence,
   coverage and early-warning qualification gates remain unchanged.
4. Two small portfolios combine group common-mode evidence with temporal IF
   or the existing fast confirmed IF. Both must independently pass calibration
   workload verification. Existing two-observation persistence is not relaxed
   globally, and shared-scope detection is not assumed to be qualified.

No device-specific suppression rules were introduced for the two noisy ONTs.
Those cases still require trace review. No threshold was tuned to the missed
fault examples, and no increase in performance is claimed before rerunning.

The six-hour history tolerance is a declared starting policy, not a fitted
optimum. Compare feature coverage and calibration workload before considering
a different tolerance. Missing-input indicators can themselves respond to
collection regimes, so their effect on nuisance workload must be measured.

## Run in Colab

Update the repository checkout before opening the notebooks. Clear old
`TELCO_FEATURE_RUN_ID`, `PON_FEATURE_RUN_ID`, `TELCO_MODEL_RUN_ID` and downstream
`PON_*_RUN_ID` environment overrides, or set them explicitly to the new IDs.

| Stage | New default |
|---|---|
| Features | `synthetic_pon_features_v6` |
| Model | `synthetic_pon_models_v12` |
| Selection | `synthetic_pon_selection_v14` |
| Incidents | `synthetic_pon_incidents_v14` |
| Localisation | `synthetic_pon_localisation_v14` |
| Development evaluation | `synthetic_pon_development_v14` |

Reuse the existing core, truth split and calibration EDA when their lineage
checks pass. Run **05 → 06 → 07 → 08 → 09 → 10 (development) → 10A**.
Notebook 06A remains an optional calibration-only sampling check.

Notebook 05 freezes the history-gap policy in its feature manifest. Notebook
10 uses that frozen policy for any subsequently authorised holdout features.
Old fitted bundles are deliberately not supported by the new IF scorer:
their input representation differs. Existing results and Drive artifacts
remain intact. Use the earlier Git revision to reproduce an older model.

## Acceptance checks

- Compare 24-hour and seven-day feature coverage with the old diagnostic.
- Compare prompt and pre-impact recall at the same incident budget.
- Inspect the 77 previously unmatched incidents and the new nuisance set;
  improved aggregate recall must not hide a workload increase.
- Revisit the hardware-failure trace and isolated-crossing faults.
- Evaluate shared-scope joint recall separately from entity localisation.
- If no candidate passes the gates, keep the failure result and inspect it.
  Do not relax the gates or open holdout to rescue a development experiment.

These changes improve the representation and selection objective. They do
not establish operator performance; that requires new measured results.

The supplied F-00023 feature trace was recomputed with the new history policy:
RX-power 24-hour history became available on 2,372 of 2,432 rows (previously
41), and seven-day history on 2,090 rows (previously zero). This checks feature
availability on that trace only. It is not a rerun of model training or a
measurement of improved detection recall.

## Repository cleanup

The unused `baseline/task0_telecom` exports and `legacy/baseline-freeze.json`
were removed from the current tree. They remain recoverable from Git history.
Active adapters, notebooks, tests and evaluation safeguards are retained.
Raw data, fitted artifacts, generated reports and local reference documents
are excluded from the distributable code archive.
