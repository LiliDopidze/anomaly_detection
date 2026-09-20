"""Explanations reconstruct the scored quantity without selecting features."""

import numpy as np
import pandas as pd
import pytest
from optical_anomaly.detectors import IsolationForestDetector
from optical_anomaly.explanations import explain_scores


def test_shap_reconstructs_actual_anomaly_score():
    pytest.importorskip("shap")
    rng = np.random.default_rng(12)
    data = pd.DataFrame(rng.normal(size=(140, 3)), columns=["a", "b", "c"])
    data["timestamp"] = pd.date_range("2025-01-01", periods=len(data), freq="h")
    data["entity_id"] = "A"
    model = IsolationForestDetector(feature_columns=["a", "b", "c"]).fit(data)
    before = model.raw_score(data)
    values, checks = explain_scores(model, data.iloc[:8], data.iloc[-3:])
    np.testing.assert_allclose(
        checks.background_score + values.sum(axis=1),
        model.raw_score(data.iloc[-3:]),
        atol=1e-7,
    )
    assert list(values) == model.feature_columns == ["a", "b", "c"]
    pd.testing.assert_series_equal(before, model.raw_score(data))
