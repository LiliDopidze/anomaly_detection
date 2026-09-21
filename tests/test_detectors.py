"""Invalid features remain unknown instead of producing alarm scores."""

import numpy as np
import pandas as pd
from optical_anomaly.detectors import IsolationForestDetector, ScoreNormalizer


def test_percentile_scores_preserve_unknown_values():
    normalizer = ScoreNormalizer().fit(pd.Series(np.arange(100, dtype=float)))
    scores = normalizer.transform(pd.Series([-1, 100, np.inf, -np.inf, np.nan]))
    assert scores.iloc[0] == 0
    assert scores.iloc[1] == 1
    assert scores.iloc[2:].isna().all()


def test_forest_abstains_on_nonfinite_features():
    training = pd.DataFrame({"level": np.linspace(-1, 1, 300)})
    forest = IsolationForestDetector(feature_columns=["level"]).fit(training)
    scored = forest.raw_score(pd.DataFrame({"level": [0.0, np.nan, np.inf]}))
    assert np.isfinite(scored.iloc[0])
    assert scored.iloc[1:].isna().all()
