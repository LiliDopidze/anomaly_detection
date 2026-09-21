# Method

The task is early warning of sustained optical degradation. Synthetic data supports
controlled development and software verification; it does not establish field
accuracy. The current model retains all 52 configured features. Telemetry-set
comparisons and SHAP are diagnostic reports, not automatic feature-selection rules.

References below distinguish support for a measurement or mathematical method
from evidence for a particular parameter. Numerical simulation parameters are
assumptions unless explicitly identified as standard-defined. No cited paper
validates this complete generator or exact feature combination.

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
| Internal BER (not exported) | Assumed probability `10**clip(-5-(latent_rx-receiver_reference), -12, -1)`. Receiver references are separate from impact thresholds, with per-ONT/direction offsets Uniform(-1.5, 1.5) dB. Prior-interval BER is multiplied by lognormal dispersion with log10 SD 0.15, then clipped to [0, 0.1]. | Error monitoring context: [G.988][g988]. Curve, offsets and dispersion are uncalibrated assumptions; they remove direct threshold coupling, not all synthetic shortcuts. |
| `{downstream,upstream}_fec_total_codewords` | Received codeword opportunities during the preceding interval, using 255-byte codewords and GPON line rates. Upstream capacity is divided among port members. | Coding/rate context: [G.984.3][g984]. Always-on coding, equal upstream allocation and omitted framing overhead are simplifications. |
| `{downstream,upstream}_fec_corrected_codewords` | For bit error probability p, symbol error probability is `q=1-(1-p)^8`. Sample codewords with 1–8 erroneous symbols. | RS(255,239) correction capability: [G.984.3][g984]; binomial probability calculation [SciPy][binomial]. Independent bit errors and ideal decoding are assumptions. |
| `{downstream,upstream}_fec_uncorrectable_codewords` | Sample codewords with more than eight erroneous symbols jointly with corrected/clean categories. Corrected + uncorrectable cannot exceed total. | Same coding basis. These are illustrative interval counters, not measured BER, cumulative counters or an XGS-PON LDPC model. |

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

Short windows contain 12 readings (one hour at default cadence). Long optical
windows contain `ceil(6 hours/cadence)` readings. Six hours is a development
choice, not a standard. Windows end at the current decision; CUSUM's reference
ends one reading earlier. Missing values or time gaps reset temporal state.
Long-window features require complete uninterrupted history: report the resulting
coverage and misses, particularly with frequent missing polls.

| Feature suffix/name | Definition and application | Source or assumption |
|---|---|---|
| `level` | Short-window mean of z. Downstream/upstream Rx, both losses, four FEC channels and both temperatures. | Window summaries have optical precedent [San Martín et al.][optical]; baseline normalisation/window duration are our choices. |
| `variability` | Short-window population SD of z on the same channels. | Same optical precedent; no standard optical warning threshold is implied. |
| `slope` | EWMA of first differences of z divided by elapsed hours. Same ten channels. Alpha=`1-exp(-dt/smoothing_hours)`, default one hour. | [NIST EWMA][ewma] supplies the smoothing basis. Applying it to optical derivatives is our modelling choice. |
| `regression_slope` | Least-squares slope of z against elapsed hours inside the short window. Downstream/upstream Rx and both loss channels. | [NIST least squares][ols]. Its early-warning value must be measured locally. |
| `long_level` | Six-hour mean of z for the four optical channels above. | Established mean statistic; chosen time scale is an assumption. |
| `short_minus_long` | Short-window mean minus six-hour mean for those four channels. | Proposed multi-scale contrast. No claim that this exact formulation is an established PON standard. |
| `below_baseline_fraction` | Fraction of short-window downstream Rx residuals below zero. Zero means the fitted expected value, not a fault threshold. | Proposed persistence summary; no feature-selection threshold. |
| `cusum` | Downstream Rx `max(0, previous + prior_short_mean - z - 0.25)`. | [NIST CUSUM][cusum] motivates accumulation. Rolling reference and allowance are adaptations; the reference can absorb slow degradation. |
| `cov` | Short-window SD/absolute mean of downstream **linear** optical power. | [NIST CoV][cov] requires a ratio scale. Retained experimental feature; never computed on dBm or Celsius. |
| `autocorrelation` | Downstream residual lag-1 centred product sum divided by full-window centred squared sum. | [NIST autocorrelation][acf]. Retained experimental feature; not specific evidence of a fault. |
| `entropy` | `-sum(p*log(p))` for downstream z in fixed bins `[-inf,-3,-2,-1,0,1,2,3,inf]`. | [Shannon entropy definition][entropy]. Bin edges and optical application are assumptions. Extreme loss can reduce entropy. |
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
training rows. Raw anomaly score is negative `score_samples` ([scikit-learn][iforest]).
Missing vectors abstain. Calibration maps raw scores to empirical ranks in [0,1],
not fault probabilities. The maximum of two ranks is an additional comparison,
requiring both scores; it is not itself a calibrated probability.

`model.feature_set` explicitly selects telemetry scope; default `temperature`
retains all 52 features. All five forests remain saved. Comparisons never remove
features or switch to a smaller set. Within that configured scope, validation
chooses detector and debounce duration using the declared nuisance budget, then
early recall and workload. Failure to meet the budget remains explicit.

Incident thresholds remain necessary operational decisions: N consecutive ranks
above high open at the Nth decision; M below low close. Equality triggers neither.
Current high/low 0.99/0.8 and N={2,3,6}, M={3,6} are development choices requiring
operator calibration. They are not feature-retention thresholds. Missing scores
reset confirmation; extended gaps close administratively, not as confirmed repair.

Evaluation deterministically matches incidents one-to-one with faults by entity
and emission inside the physical fault interval. No point adjustment is used.
Pre-impact recall counts early matched faults among those with a declared warning
opportunity: observable onset and three observed Rx intervals before impact minus
30 minutes. Nuisance counts unmatched and duplicate incidents per 1,000 scheduled
entity-days. Mean-shift delay uses alert emission minus observable onset. Variance-shift delay
uses physical onset with an explicit `delay_reference` and a separate aggregate
metric; it does not enter the observable-delay median. Misses and coverage are
reported separately. These opportunity/matching definitions are
project evaluation decisions, not claims of an industry-wide metric standard.
Final assessment is explicit, frozen and one-time; notebook defaults leave it shut.

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

## Sources

[etsi]: https://www.etsi.org/deliver/etsi_gs/F5G/001_099/011/01.01.01_60/gs_F5G011v010101p.pdf
[db]: https://www.nist.gov/pml/special-publication-811/nist-guide-si-chapter-5-units-outside-si
[g988]: https://www.itu.int/rec/T-REC-G.988
[g984]: https://www.itu.int/rec/T-REC-G.984.3
[binomial]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html
[ar]: https://www.statsmodels.org/stable/tsa.html
[optical]: https://opg.optica.org/jocn/abstract.cfm?uri=jocn-18-7-674
[harmonic]: https://otexts.com/fpp3/useful-predictors.html
[mad]: https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
[ols]: https://www.itl.nist.gov/div898/handbook/pmd/section1/pmd141.htm
[ewma]: https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc314.htm
[cusum]: https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc323.htm
[cov]: https://itl.nist.gov/div898/software/dataplot/refman2/auxillar/coefvari.htm
[acf]: https://www.itl.nist.gov/div898/handbook/eda/section3/eda35c.htm
[entropy]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html
[leakage]: https://scikit-learn.org/stable/common_pitfalls.html
[iforest]: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
[shap]: https://shap.readthedocs.io/en/latest/generated/shap.PermutationExplainer.html


