# Company agnostic time series anomaly detection

Version 4 implements independent statistical residual, feature-based Isolation Forest, and frozen-normal-bank matrix-profile routes. It is a batch research framework. The synthetic experiment does not establish production effectiveness. Read `RESULTS_V4.md` for the measured decision and `DECISION_CONTRACT.md` for the predeclared requirements.

## Run the experiment

Use Python 3.12. The exact tested numerical environment is recorded in `requirements-reproduction.txt`. From this development checkout:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-reproduction.txt
python -m pip install -e . --no-deps
python -m pytest
python -m anomaly_detection.pipeline develop --config configs/industry_agnostic.yaml --output outputs/v4_run
python -m anomaly_detection.pipeline assess --output outputs/v4_run
python -m anomaly_detection.fixtures --output outputs/v4_fixtures
```

`develop` generates inputs once, fits on eligible training history, selects on development, writes all trials and freezes the primary policy. Repeating a completed development command reuses its recorded result. `assess` verifies frozen hashes, opens the final temporal period once, and evaluates the fixed primary and comparators. A previously opened final assessment cannot be repeated or used to select a new winner. Outputs and failed runs remain outside tracked code. A stopped or failed assessment remains labelled as opened.

The existing seed-42 data distribution has already been explored. The final period is a frozen temporal assessment, not untouched independent evidence. No deployment or field pilot is executed by these commands. Use a new output directory for a genuinely new declared experiment; changing only its name does not restore held-out status.

## Four notebooks

Run the notebooks in order from `notebooks/`: `00_data_exploration`, `01_references_and_features`, `02_fit_and_select`, `03_evaluate_and_explain`. Set `ANOMALY_RUN` to the absolute path of an existing result directory to review the delivered experiment. Otherwise they use `outputs/v4_run`. The last notebook reads completed assessment artifacts; the assessment command is deliberately explicit. Previous notebooks remain under `notebooks/historical/` and implement the older protocol.

## Interface and files

| Module | Responsibility |
| --- | --- |
| `validation.py`, `adapters.py` | Observation/availability contract, schedule, quality, counters, GPON and non-optical fixture |
| `references.py`, `features.py` | Frozen median/daily harmonic reference, centred MAD scale, causal four/eight features |
| `detectors.py` | Separate U-level, U-shift and local forests |
| `distance_profiles.py` | STUMPY MASS, complete queries, full-overlap exclusions and frozen bank provenance |
| `incidents.py`, `grouping.py` | Strict confirmation/recovery, gap closure, fixed-anchor investigation grouping |
| `evaluation.py`, `selection.py` | One-to-one ordinary/early matching, independent exposure, finite development search |
| `diagnostics.py`, `fixtures.py` | Read-only diagnostic plots, common support, distinct sensitivity fixtures |
| `pipeline.py` | `fit`, `score`, `save`, `load`; CLI `prepare`, `develop`, `assess` |

Use `incidents.detect`, `grouping.group` and `evaluation.evaluate` for the remaining interface operations. Load only trusted joblib artifacts. Scoring does not refit. Each model is local to an entity and declared group; there is no pooled fallback. The supplied experiment monitors the two received-power channels. Sparse FEC counts are not silently treated as Gaussian residuals.

Scores have distinct route and scope, event and decision times, validity reason, window bounds and version. Distance scores also identify eligible support and nearest normal windows. Configuration and source/dependency/data hashes, feature order and selected training rows are saved. `FROZEN.json` binds model, thresholds, enabled system and deployment decision. Event notifications and membership updates retain their original confirmation times.

The frozen-bank route compares shape and residual level separately at 30/60 minutes. It uses the established [STUMPY MASS primitive](https://stumpy.readthedocs.io/en/latest/api.html#stumpy.mass); our support and overlap rules are additional project policies. This is a directed normal-bank comparison, not DAMP or a retrospective unrestricted self-join. Isolation Forest uses `-score_samples`, verified against [scikit-learn documentation](https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.IsolationForest.html).

## Boundaries

The initial scope is regular numeric telemetry. Schedules and observation timestamps must be timezone-aware, with one immutable decision deadline per scheduled observation. Late inputs abstain at that deadline. There is no arbitrary streaming-chunk API, live service, automatic bank update, fault-probability interpretation, root-cause inference, MATLAB parity claim, covariance challenger or learned clustering.

`MIGRATION_V4.md` records the historical source and audit corrections. `PILOT_AND_MAINTENANCE.md` specifies operator review, independent fault ascertainment, shadow replacement and rollback. `METHODOLOGY_V4.txt` and `IMPLEMENTATION_PROMPT_V4.md` retain the requested specifications. The legacy `optical_anomaly` package remains unchanged for generator parity and historical reproduction.
