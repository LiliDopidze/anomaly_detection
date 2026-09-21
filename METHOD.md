# Method

The task is early warning of sustained optical degradation. Synthetic data supports
controlled development and software verification; it does not establish field
accuracy. The current model retains all 52 configured features. Telemetry-set
comparisons and SHAP are diagnostic reports, not automatic feature-selection rules.

References below distinguish support for a measurement or mathematical method
from evidence for a particular parameter. Numerical simulation parameters are
assumptions unless explicitly identified as standard-defined. No cited paper
validates this complete generator or exact feature combination.

Inline `Source` comments identify a measurement definition or mathematical method;
`Assumption` comments identify a value that needs operator data or a declared
experimental policy. For example, `correlation_hours: 0.5` is not an ITU limit.
Each generated run saves the original commented `config.yaml` alongside
`settings.json`, so the numerical settings retain their rationale.

## Generated data

Defaults are 96 ONTs, 90 days and five-minute samples. Static membership gives
12 splitters, six PON ports and two OLTs. These sizes and fan-outs are scenario
choices. The output is native wide telemetry, a topology lookup and a separate
fault log; labels and topology identifiers never enter the feature matrix.

| Generated fields | Construction and purpose | Basis and limits |
|---|---|---|
| `rx_dbm`, `upstream_rx_dbm` | Directional receive power: transmitter power minus path loss. Downstream path loss starts uniformly between 23 and 27 dB; upstream adds an assumed 0.4–1.2 dB offset. Both paths share normal fluctuations and injected loss. | Optical measurement categories: [ETSI F5G 011, §§8.3–8.4][etsi]. Subtraction follows the logarithmic power definition [NIST][db]. Ranges and cross-direction dependence are simulation assumptions. |
| `ont_tx_dbm`, `olt_tx_dbm` | Nominal 2/3 dBm; small thermal response and stationary residuals. OLT Tx is shared by ONTs on a port. Tx does not change in response to a simulated path fault. | Monitored quantities: [ETSI][etsi]. Nominal levels and thermal coefficients 0.008/0.005 dB/°C are uncalibrated assumptions, not vendor specifications. |
| `ont_temperature_c`, `olt_temperature_c` | Baselines 38/35°C, daily amplitudes 3/2°C and correlated residuals. OLT temperature is shared per port with no annual ramp. | Temperature monitoring: [ETSI][etsi]. Waveform, ranges and time constants are assumptions. Ninety days cannot identify an annual cycle. |
| Internal BER (not exported) | Assumed probability `10**clip(-5-(latent_rx-receiver_reference), -12, -1)`. Receiver references are separate from impact thresholds, with per-ONT/direction offsets Uniform(-1.5, 1.5) dB. Prior-interval BER is multiplied by lognormal dispersion with log10 SD 0.15, then clipped to [0, 0.1]. | Error monitoring context: [G.988][g988]. Curve, offsets and dispersion are uncalibrated assumptions; receiver behaviour is independent of label thresholds but remains synthetic. |
| `{downstream,upstream}_fec_total_codewords` | Received codeword opportunities during the preceding interval, using 255-byte codewords and GPON line rates. Upstream capacity is divided among port members. | [G.984.3 (2014)][g984] §6.2 supports the rate pair; §13 and Annex A.3 define coding. Always-on coding, equal upstream allocation and omitted framing overhead are simplifications. |
| `{downstream,upstream}_fec_corrected_codewords` | For bit error probability p, symbol error probability is `q=1-(1-p)^8`. Sample codewords with 1–8 erroneous symbols. | RS(255,239) correction capability: [G.984.3][g984]; binomial probability calculation [SciPy][binomial]. Independent bit errors and ideal decoding are assumptions. |
| `{downstream,upstream}_fec_uncorrectable_codewords` | Sample codewords with more than eight erroneous symbols jointly with corrected/clean categories. Corrected + uncorrectable cannot exceed total. | Same coding basis. These are illustrative interval counters, not measured BER, cumulative counters or a model of another PON generation. |

Normal residual noise is stationary Gaussian AR(1):
`u[t] = phi*u[t-1] + sigma*sqrt(1-phi²)*epsilon[t]`, with stationary initial
variance and `phi=exp(-dt/tau)`. This is a standard autoregressive construction
([statsmodels time-series methods][ar]); its applicability and numerical time
constants are assumptions. Independent optical sensor noise is added after the
latent physical state. It therefore cannot itself create FEC errors or impact.
Daily sinusoidal structure is a controlled seasonal scenario, not an assertion
that every PON has daily optical seasonality.

Missed polls are independent Bernoulli draws with default probability 0.02. They
remove an ONT's telemetry row measurements together. This is a stress scenario,
not an empirical model of outages. Correlated missing runs and severity-dependent
telemetry loss remain limitations.

Faults use three deliberately distinct scenario families: negative-drift random
walks (-0.15 dB/hour, diffusion 0.06 dB/sqrt(hour)); accelerating attenuation
`-0.2-0.35*expm1(min(hours/12,3))`; and a stationary variance increase to 0.5 dB.
Severity is multiplied by Uniform(0.4,1.4); duration is lognormal with median
36 hours and log-SD 0.5, truncated at scenario boundaries. All signatures,
parameters and timings are **project assumptions**. A separate seeded per-ONT
Uniform(0.8, 1.4) multiplier relates upstream fault effect to downstream effect;
this avoids one fleet-wide fixed ratio but still shares the same fault trajectory.
Optical soft failures motivate the task [San Martín et al., 2026][optical], but
that study does not establish these PON degradation laws.

The first 55% is deliberately healthy. Up to two individual-ONT faults are placed
later, rather than sampled from a claimed real failure arrival process. Shared
port faults, independent Tx failures and field prevalence are not reproduced.
FEC counters describe the preceding interval: a state change at t affects the
report at t+dt. First-interval counts are missing.

Truth records distribution-change onset, observable-onset proxy, hypothetical
impact and fault end. For mean-shift scenarios, observable onset requires an injected effect at least
twice the configured physical-noise SD and an observed Rx reading. Variance shifts
have no asserted observable-onset proxy. Impact is confirmed at the third
consecutive latent reading below -27 dBm downstream or -28 dBm upstream
(either direction at each reading); it is never backdated to the start of the run. These are declared
scenario thresholds, not universal receiver limits or verified customer impact.

## Observation, adaptation and EDA

Notebook 00 checks native fields, counts, missing runs, distributions, topology,
noise/seasonality diagnostics, FEC conservation and development fault scenarios.
Structural checks establish consistency; comparisons against intended noise and
seasonality establish whether the implementation reproduces its assumptions.
Neither demonstrates resemblance to an operator network.

Notebook 01 maps names, units and gauge/interval-count semantics to
`timestamp, entity_id, metric_name, value`. Company mappings are explicit; the
synthetic source has its own mapping. Cumulative counters require reset-aware
conversion before this adapter. The detector checks required measurements after
adaptation. Validation resamples into right-closed, right-labelled intervals;
there is no interpolation or forward filling. Invalid readings remain missing.

Notebook labels call FEC codewords **blocks**: received, corrected or uncorrectable,
with upstream/downstream explicit. Stored names retain the precise codeword unit.

Canonical training EDA reports per-ONT Pearson/Spearman measurement correlations
and selected-lag autocorrelations, before and after daily-pattern removal. Pair
counts accompany coefficients; constants/insufficient pairs remain NaN. These are
descriptive diagnostics, not significance tests or automatic feature filters.
It also reports distributions, missingness and
chronological daily/weekly seasonal comparisons. Its holdout is inside training.
The model uses one daily sine/cosine pair fitted on healthy training only, or a
median baseline when seasonality is disabled. Harmonic regression is established
([Hyndman and Athanasopoulos][harmonic]); its necessity must be assessed per source.
Weekly terms are examined in EDA, not automatically fitted by the model.

## Features

For each ONT/channel, `z=(measurement-fitted_daily_baseline)/scale`. The scale is
`1.4826*median(abs(residual-median(residual)))` on training, with floors of 0.05
for dB/transformed-FEC channels and 0.1°C for temperature. MAD is a robust spread
measure ([NIST][mad]); these floors are numerical assumptions, not alarm limits.
Ordinary least-squares seasonal fitting itself is not robust to contaminated
training. A representative reviewed local baseline remains necessary.
Seasonal fitting requires at least two calendar days of reference history and
identifiable daily phases, as well as the existing finite-sample guard. Two days
is a minimum onboarding assumption, not sufficient evidence that a daily model
will generalize; the training-only seasonal holdout remains the diagnostic.

Short windows contain 12 readings (one hour at default cadence). Long optical
windows contain up to `ceil(6 hours/cadence)` readings. Six hours is a development
choice, not a standard. Windows end at the current decision; CUSUM uses the
frozen training-normalized target zero. Missing values or time gaps reset state.
Long-window features require at least 12 contiguous readings and use available
history up to the six-hour limit. Report coverage and misses, particularly with
frequent missing polls.

| Feature suffix/name | Definition and application | Source or assumption |
|---|---|---|
| `level` | Short-window mean of z. Downstream/upstream Rx, both losses, four FEC channels and both temperatures. | Window summaries have optical precedent [San Martín et al.][optical]; baseline normalisation/window duration are our choices. |
| `variability` | Short-window population SD of z on the same channels. | Same optical precedent; no standard optical warning threshold is implied. |
| `slope` | EWMA of first differences of z divided by elapsed hours. Same ten channels. Alpha=`1-exp(-dt/smoothing_hours)`, default one hour. | [NIST EWMA][ewma] supplies the smoothing basis. Applying it to optical derivatives is our modelling choice. |
| `regression_slope` | Least-squares slope of z against elapsed hours inside the short window. Downstream/upstream Rx and both loss channels. | [NIST least squares][ols]. Its early-warning value must be measured locally. |
| `long_level` | Mean of z over up to six hours of contiguous history, requiring the short-window count for the four optical channels above. | Established mean statistic; chosen time scale is an assumption. |
| `short_minus_long` | Short-window mean minus the available long-window mean for those four channels. | Proposed multi-scale contrast. No claim that this exact formulation is an established PON standard. |
| `below_baseline_fraction` | Fraction of short-window downstream Rx residuals below zero. Zero means the fitted expected value, not a fault threshold. | Proposed persistence summary; no feature-selection threshold. |
| `cusum` | Downstream Rx `max(0, previous - z - 0.25)`, after short-window warm-up. | [Page (1954)][page] and [NIST CUSUM][cusum] support a fixed in-control reference. Zero is the training-normalized target; allowance and post-gap warm-up are assumptions. A rolling target would follow sustained degradation and lose evidence. |
| `cov` | Short-window SD/absolute mean of downstream **linear** optical power. | [NIST CoV][cov] requires a ratio scale. Retained experimental feature; never computed on dBm or Celsius. |
| `autocorrelation` | Downstream residual lag-1 centred product sum divided by full-window centred squared sum. | [NIST autocorrelation][acf]. Retained experimental feature; not specific evidence of a fault. |
| `entropy` | `-sum(p*log(p))` for downstream z in fixed bins `[-inf,-3,-2,-1,0,1,2,3,inf]`. | [Shannon (1948)][shannon]; implementation context [SciPy][entropy]. Bin edges and optical application are assumptions. Extreme loss can reduce entropy; 12-point estimates are exploratory. |
| `acceleration` | EWMA of downstream second difference divided by dt². | Finite-difference definition plus [EWMA][ewma]; exploratory optical application. Differentiation amplifies noise. |
| `*_error_interval_fraction` | Short-window fraction with a positive corrected/uncorrectable FEC numerator; four directional error channels. | Proposed persistence statistic based on FEC counters [G.988][g988]. Zero is exact absence of recorded errors, not a learned cutoff. |

The downstream names above are unprefixed. Other prefixes are `upstream_rx`,
`downstream_loss`, `upstream_loss`, `{downstream,upstream}_fec_{corrected,uncorrectable}`,
`ont_temperature` and `olt_temperature`. This covers every generated feature column.

Loss channels are synchronised Tx(dBm) minus Rx(dBm): OLT Tx–ONT Rx downstream,
ONT Tx–OLT Rx upstream. They are link-loss proxies, affected by measurement error
and location. FEC channels are matching corrected/uncorrectable interval counts
divided by received codeword totals, then `log1p(fraction/1e-6)`. Zero totals are
unknown, not healthy zero. The numerical log reference is a scaling choice.
Intervals with corrected plus uncorrectable counts above received counts are
invalid for both fractions. An independently missing numerator does not erase a
valid sibling category. Feature merges reject duplicate keys and name collisions.
Vendor counters may have different denominators: [ETSI][etsi] supports checking
semantics, not assuming every field called FEC total counts all received codewords.

There are 12 downstream features; upstream adds six; both losses add twelve;
FEC adds sixteen; temperature adds six: **52 total**. The five cumulative sets
therefore contain 12, 18, 30, 46 and 52 features. No IDs, ground truth or simulated
BER are model inputs. Temperature is currently a separate feature group, not a
regressor used to remove thermal power variation. Receiver-margin and peer-port
features are deferred until reliable equipment limits/shared-fault scenarios exist.

Company independence comes from explicit schema mapping, physical relationships
and per-device reference fitting. It is not zero-shot transfer across equipment,
sensor precision, polling cadence or operator conditions. Unknown devices abstain.

## Modelling and operational evaluation

Chronological partitions are 40% training, 15% score calibration, 20% validation
and 25% final test. Training fits references and forests; later healthy calibration
fits empirical score ranks. Causal past history may cross a split boundary.
No future data enters fitting ([scikit-learn leakage guidance][leakage]).

The statistical comparator scores negative downstream EWMA slope. Each telemetry
set has an Isolation Forest with 100 trees, max_samples=256 and at most 20,000
training rows. The first two are established starting values from
[Liu, Ting and Zhou (2008), §4.1][iforest-paper]; the extra 20,000-row cap is a
runtime choice. Raw anomaly score is negative `score_samples` ([scikit-learn][iforest]).
Missing vectors abstain. Calibration maps raw scores to empirical ranks in [0,1],
not fault probabilities. The maximum of two ranks is an additional comparison,
requiring both scores; it is not itself a calibrated probability.

`model.feature_set` explicitly selects telemetry scope; default `temperature`
retains all 52 features. All five forests remain saved. Comparisons never remove
features or switch to a smaller set. Within that configured scope, validation
chooses detector and debounce duration using the declared nuisance budget, then
`minimum_lead_recall` and workload. Failure to meet the budget remains explicit.

Incident thresholds remain necessary operational decisions: N consecutive ranks
above high open at the Nth decision; M below low close. Equality triggers neither.
Current high/low 0.99/0.8 and N={2,3,6}, M={3,6} are development choices requiring
operator calibration. They are not feature-retention thresholds. Missing scores
reset confirmation; extended gaps close administratively, not as confirmed repair.
The independent incident gap timeout defaults to six hours. Validation tables
report score coverage and telemetry-gap closures alongside workload.

Evaluation deterministically matches incidents one-to-one with faults by entity
and emission inside the physical fault interval. No point adjustment is used;
[Kim et al. (2022)][point-adjustment] show how it can inflate apparent performance.
Pre-impact recall counts early matched faults among those with a declared warning
opportunity: observable onset and three observed Rx intervals before impact minus
30 minutes. Variance shifts use physical onset as an explicit synthetic opportunity
proxy. Results include separate observable-onset and physical-onset denominators,
numerators and recall; the pooled recall includes both. Physical onset is not proof
of immediate detectability. Nuisance counts unmatched and duplicate incidents per 1,000 scheduled
entity-days. Mean-shift delay uses alert emission minus observable onset. Variance-shift delay
uses physical onset with an explicit `delay_reference` and a separate aggregate
metric; it does not enter the observable-delay median. Misses and coverage are
reported separately. These opportunity/matching definitions are
project evaluation decisions, not claims of an industry-wide metric standard.
`evaluation.minimum_lead_minutes` (default 30) and `opportunity_intervals`
(default 3) are declared once for every candidate. `pre_impact_recall` credits
any strictly pre-impact alert; `minimum_lead_recall` additionally requires the
declared lead time. Validation ranks the latter so an alert one minute before
impact cannot satisfy a 30-minute warning target. `lead_minutes` is recorded for
each matched impacted fault, including negative values for late detection.
`pre_impact_recall_all_impacting` includes impacted faults without a declared
warning opportunity, making the conditional denominator's exclusions visible.
Final assessment is explicit, frozen and one-time; notebook defaults leave it shut.

The holdout is future data from the same simulated entities. It does not test
unseen companies, vendors, fault mechanisms or acquisition systems. Scores can
depend on generator shortcuts even with correct chronological splits.

## Feature importance

Notebook 06 uses [SHAP PermutationExplainer][shap] on the actual negative forest
`score_samples` function. It uses training-only background rows and complete
validation rows. Global mean absolute SHAP covers a uniform sample (default 64);
separate local explanations cover eight highest-score rows. Background size 16,
two permutation cycles and seed 42 are exploratory runtime choices. Larger/repeated
samples are needed to judge ranking stability. Saved local values must reconstruct
the raw score from the background expectation within numerical tolerance.

Positive SHAP increases the anomaly score. Attributions are not probabilities,
causes, incident explanations or measurements of pre-impact utility. Correlated
features can share credit, and independent masking can create unlikely feature
combinations. The notebook also reports Spearman correlations and complete-score
coverage. No SHAP cutoff removes features. Operational evaluation stays separate;
any future removal needs a reviewed, reproducible comparison, not a small ranking.

The combination of optical power, temperature and error measurements also has
field optical-transport precedent in [Zhang et al. (2025)](https://opg.optica.org/jocn/abstract.cfm?uri=jocn-17-2-81),
which uses SHAP for model interpretation. That is support for examining model
contributions, not evidence that SHAP identifies physical causes or that results
transfer directly to PON early warning. For the 2026 optical study, the public
abstract and tables support the cited window statistics; no inaccessible methods
are assumed here.

## Review of the supplied feature assessment

The 21 September 2026 assessment supports the existing causal, per-entity
architecture. Its EDA values describe a supplied project run; they have not been
independently reproduced on the revised generator and must not be presented as
new validation results. They demonstrate internal simulated patterns, not field
calibration. Daily seasonality is deliberately injected, so detecting it is an
implementation check rather than independent evidence about real PON networks.

| Recommendation | Decision and reason |
|---|---|
| Require a complete six-hour window | Retain the explicitly documented available-history mean: at least 12, at most 72 contiguous readings at five-minute cadence. Under independent 2% missingness, only `0.98**72 ≈ 23%` of decision times have a complete six-hour history. A strict window is a useful later controlled comparison, not an automatic correction. |
| Validate feature combination | Adopted: one-to-one entity/time merge and explicit rejection of overlapping non-key names. |
| Require multiple seasonal cycles | Adopted: two calendar days and identifiable daily phases are minimum guards. Longer representative baselines and held-out seasonal performance are still needed. |
| Add missingness features | Added observation-quality diagnostics in notebook 02: observed fraction, gap length, time since last poll and contiguous history. They stay outside the optical detector: this simulator's missingness is independent of faults, and adding it could turn management-plane failures into optical alerts. |
| Add short/long variance and MAD ratios | Defer to an explicit ablation. Variability already exists. A short window contained in the long window contaminates the denominator during a fault; near-zero long variance destabilizes ratios, and exact six-hour requirements reduce coverage. A disjoint prior reference or frozen training MAD is preferable for a future experiment. |
| Rename the temperature set | Reports now describe it as the full cumulative set. Keep the stored `temperature` key so existing settings remain understandable; it contains all 52 features, not temperature alone. |
| Add directional interactions and temperature-adjusted Tx | Candidate experiments, not established requirements. Existing two-direction levels and Tx−Rx losses already encode related information. Temperature correction could remove a true thermal fault unless trained and evaluated carefully. |
| Add margin to the impact threshold | Do not use the simulator's label threshold as a model feature. Real, independently supplied equipment limits could support a separate operational-margin report later. |
| Add topology-aware features | Defer until shared faults exist in generation and labels. Current static topology does not demonstrate shared-fault localization. |
| Remove low-importance or redundant features | No automatic removal. Correlations and SHAP remain descriptive; any later change needs a reviewed comparison using warning time, misses, workload and coverage. |

Several statements need qualification. Ordinary least-squares slope averages over
several samples but is not robust to outliers in the statistical sense. Zero
autocorrelation for a constant window is an implementation convention: theoretical
correlation is undefined when variance is zero. Below-baseline fraction uses zero
as the expected level, not an alarm threshold. The former rolling-reference CUSUM
was causal but could follow the fault; a fixed healthy target is preferable for
sustained-shift monitoring. Abrupt synthetic recovery still limits conclusions
about closing times and alert fragmentation in operations.

The reference list also spans distinct tasks:

- [Abdelli et al., ICTON 2023](https://arxiv.org/abs/2307.03945) use experimental
  OTDR data for PON fault monitoring. This supports the problem domain, not our
  daily optical-power distribution or warning horizon.
- [Abdelli et al., JOCN 2022](https://arxiv.org/abs/2204.07059) concern optical-fibre
  anomaly detection and localization. This is the precise venue for the report's
  loosely described 2022 reference; its neural architecture is not adopted.
- [Zegdou and Garadi, 2025](https://doi.org/10.63620/MKJGPSCD.2025.1015) simulate
  attenuation in OptiSystem and perform K-NN classification. This does not
  establish real GPON telemetry distributions or prospective warning performance,
  and is not used to set our numerical assumptions.
- [PONData](https://github.com/linglesloggia/PONData) describes a 16-ONU testbed
  with traffic/configuration experiments, 12-minute profiles and QoS measurements.
  Those experiments are not continuous optical-fault histories; they cannot
  directly calibrate our Rx-noise or optical-degradation laws. Public-data
  integration remains a separate stage.

## Technical decisions and next evidence needed

| Stage | Credible basis retained | Improvement or evidence still needed |
|---|---|---|
| Generator | GPON rates/coding [G.984.3][g984]; monitored quantities [ETSI][etsi] | Fit noise, daily amplitude, receiver response and missingness to actual equipment. Passing the current checks only confirms the stated simulation. |
| Features | Harmonic regression, MAD, least squares, EWMA, CUSUM and valid linear-power CoV | Compare feature groups on validation; examine sensitivity of short-window entropy/ACF. No paper establishes this exact 52-feature set as optimal. |
| Detector | Original [Isolation Forest][iforest-paper] and [EWMA][ewma] | Keep statistical and forest baselines. Slope alone may stop flagging an established low level; the forest also receives level and fixed-target CUSUM. A standalone CUSUM comparator is a useful next statistical experiment. |
| Protocol | Training-only fitted transformations [leakage guidance][leakage] | Add truly held-out devices or equipment classes once onboarding rules/data permit. Chronological same-ONT testing does not establish company transfer. |
| Incidents/evaluation | Emission-time, one-to-one matching; no point adjustment [Kim et al.][point-adjustment] | Calibrate operator warning/workload requirements and test realistic gradual recovery, shared faults and block missingness. |
| Interpretation | Actual-score [SHAP][shap] with training background | Repeat explanations across samples and seeds. Attribution is neither causality nor proof of operational utility. |

To calibrate the synthetic model, collect timestamped, technology-specific healthy
and fault histories. Estimate residual spread and lag dependence after testing
seasonality on a separate healthy period; compare their distributions across ONTs,
ports and hardware. For an AR(1) approximation, `tau=-dt/log(rho)` is meaningful
only when the residual lag-one correlation is between zero and one and a single
decay model is appropriate. Separate sensor resolution from physical variability.
Estimate missing-run distributions and error fractions jointly with power,
utilization and FEC mode. Use maintenance records to distinguish observed symptoms,
service impact and repair; do not infer all three from one power threshold.

Before claiming robustness, run multiple seeds and stress assumptions such as
noise, seasonality, missing polls and receiver offsets. Notebook 05 supports these
development comparisons with a fixed incident policy while refitting local
references; that tests adaptation to new simulated conditions, not zero-shot
transfer. Report each run and a median/range, plus results by fault type and
observation coverage. Reserve an untouched final scenario before selecting
features or policies. Real telemetry is required to substantiate field claims.

## Sources

[etsi]: https://www.etsi.org/deliver/etsi_gs/F5G/001_099/011/01.01.01_60/gs_F5G011v010101p.pdf
[db]: https://www.nist.gov/pml/special-publication-811/nist-guide-si-chapter-5-units-outside-si
[g988]: https://www.itu.int/rec/T-REC-G.988
[g984]: https://www.itu.int/rec/T-REC-G.984.3-201401-I
[binomial]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html
[ar]: https://www.statsmodels.org/stable/tsa.html
[optical]: https://opg.optica.org/jocn/abstract.cfm?uri=jocn-18-7-674
[harmonic]: https://otexts.com/fpp3/useful-predictors.html
[mad]: https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
[ols]: https://www.itl.nist.gov/div898/handbook/pmd/section1/pmd141.htm
[ewma]: https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc324.htm
[cusum]: https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc323.htm
[cov]: https://itl.nist.gov/div898/software/dataplot/refman2/auxillar/coefvari.htm
[acf]: https://www.itl.nist.gov/div898/handbook/eda/section3/eda35c.htm
[entropy]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html
[leakage]: https://scikit-learn.org/stable/common_pitfalls.html
[iforest]: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
[shap]: https://shap.readthedocs.io/en/latest/generated/shap.PermutationExplainer.html
[page]: https://doi.org/10.1093/biomet/41.1-2.100
[shannon]: https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf
[iforest-paper]: https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf
[point-adjustment]: https://doi.org/10.1609/aaai.v36i7.20680
