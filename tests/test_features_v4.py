"""Focused checks for the v4 Telecom feature semantics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from telco_anomaly.features import (  # noqa: E402
    add_causal_history,
    add_causal_seasonal_differences,
    add_causal_temporal_features,
    directional_cusum,
    empirical_tail_evidence,
    feature_policy,
    fit_empirical_tail_reference,
    transform_episode,
)
from telco_anomaly.detectors import score_residual_episode  # noqa: E402


BASE = pd.Timestamp("2025-01-01", tz="UTC")


def catalogue(*rows):
    return pd.DataFrame(
        rows,
        columns=[
            "metric_id",
            "measurement_kind",
            "expected_cadence_seconds",
            "direction",
            "transform",
            "minimum_scale",
            "valid_min",
            "valid_max",
            "reset_policy",
            "counter_modulus",
        ],
    )


def panel(times, **metrics):
    return pd.DataFrame(
        {
            "event_ts": [BASE + pd.Timedelta(seconds=value) for value in times],
            "entity_id": "ont-1",
            "episode_id": "ont-1::episode-1",
            **metrics,
        }
    )


class TransformTests(unittest.TestCase):
    def test_seasonal_difference_uses_only_the_exact_past_phase(self):
        metrics = catalogue(
            (
                "service.throughput",
                "gauge",
                1,
                "low_bad",
                "log1p",
                0.1,
                0.0,
                None,
                None,
                None,
            )
        )
        metrics["seasonality_candidate"] = True
        transformed = transform_episode(
            panel(
                [0, 1, 2, 3, 4],
                **{"service.throughput": [1.0, 2.0, 3.0, 4.0, 9.0]},
            ),
            metrics,
        )
        result = add_causal_seasonal_differences(
            transformed, metrics, {"service.throughput": 4}
        )

        seasonal = result["service.throughput__seasonal_difference"]
        self.assertTrue(seasonal.iloc[:4].isna().all())
        self.assertAlmostEqual(seasonal.iloc[4], np.log1p(9.0) - np.log1p(1.0))

        prefix = add_causal_seasonal_differences(
            transformed.iloc[:4].copy(), metrics, {"service.throughput": 4}
        )
        pd.testing.assert_series_equal(
            result.iloc[:4]["service.throughput__seasonal_difference"].reset_index(drop=True),
            prefix["service.throughput__seasonal_difference"].reset_index(drop=True),
        )

    def test_ber_hurdle_does_not_replace_zero_with_an_arbitrary_floor(self):
        metrics = catalogue(
            (
                "optical.ber",
                "bounded_fraction",
                1,
                "high_bad",
                "hurdle_log10",
                0.1,
                0.0,
                1.0,
                None,
                None,
            )
        )
        result = transform_episode(
            panel([0, 1, 2], **{"optical.ber": [0.0, 1e-9, 1e-6]}),
            metrics,
        )

        self.assertEqual(result["optical.ber__nonzero"].tolist(), [0.0, 1.0, 1.0])
        self.assertTrue(pd.isna(result.loc[0, "optical.ber__positive_log10"]))
        self.assertEqual(result.loc[1, "optical.ber__positive_log10"], -9.0)
        self.assertEqual(result.loc[2, "optical.ber__positive_log10"], -6.0)

    def test_zero_inflated_count_separates_occurrence_and_magnitude(self):
        metrics = catalogue(
            (
                "ethernet.crc_errors",
                "interval_count",
                1,
                "high_bad",
                "hurdle_log1p",
                0.1,
                0.0,
                None,
                None,
                None,
            )
        )
        result = transform_episode(
            panel([0, 1, 2], **{"ethernet.crc_errors": [0.0, 3.0, 0.0]}),
            metrics,
        )

        self.assertEqual(
            result["ethernet.crc_errors__nonzero"].tolist(), [0.0, 1.0, 0.0]
        )
        self.assertTrue(pd.isna(result.loc[0, "ethernet.crc_errors__positive_log1p"]))
        self.assertAlmostEqual(
            result.loc[1, "ethernet.crc_errors__positive_log1p"], np.log1p(3)
        )

    def test_counter_differences_do_not_cross_resets_or_gaps(self):
        metrics = catalogue(
            (
                "equipment.uptime",
                "cumulative_counter",
                1,
                "contextual",
                "reset_safe_increment",
                1.0,
                0.0,
                None,
                "reset_to_unknown",
                None,
            )
        )
        result = transform_episode(
            panel(
                [0, 1, 2, 10, 11],
                **{"equipment.uptime": [100.0, 101.0, 5.0, 20.0, 22.0]},
            ),
            metrics,
        )

        increments = result["equipment.uptime__increment"]
        resets = result["equipment.uptime__reset"]
        self.assertTrue(pd.isna(increments.iloc[0]))
        self.assertEqual(increments.iloc[1], 1.0)
        self.assertTrue(pd.isna(increments.iloc[2]))
        self.assertTrue(pd.isna(increments.iloc[3]))
        self.assertEqual(increments.iloc[4], 2.0)
        self.assertEqual(resets.iloc[2], 1.0)
        self.assertTrue(pd.isna(resets.iloc[3]))


class CausalHistoryTests(unittest.TestCase):
    def test_current_and_future_values_do_not_enter_the_trailing_reference(self):
        metrics = catalogue(
            (
                "signal",
                "gauge",
                1,
                "high_bad",
                "identity",
                0.1,
                None,
                None,
                None,
                None,
            )
        )
        source = panel(range(8), signal=[1, 1, 1, 1, 10, 1, 1, 1])
        transformed = transform_episode(source, metrics)
        first = add_causal_history(
            transformed, metrics, history_seconds=4, minimum_history_seconds=3
        )

        changed = source.copy()
        changed.loc[6:, "signal"] = 1_000
        second = add_causal_history(
            transform_episode(changed, metrics),
            metrics,
            history_seconds=4,
            minimum_history_seconds=3,
        )
        name = "signal__level__history_z"
        self.assertGreater(first.loc[4, name], 80)
        pd.testing.assert_series_equal(first.loc[:5, name], second.loc[:5, name])

    def test_history_restarts_after_a_collection_gap(self):
        metrics = catalogue(
            (
                "signal",
                "gauge",
                1,
                "two_sided",
                "identity",
                0.1,
                None,
                None,
                None,
                None,
            )
        )
        transformed = transform_episode(
            panel([0, 1, 2, 3, 20, 21, 22, 23], signal=range(8)), metrics
        )
        result = add_causal_history(
            transformed, metrics, history_seconds=10, minimum_history_seconds=3
        )
        name = "signal__level__history_z"
        self.assertTrue(result.loc[4:6, name].isna().all())
        self.assertTrue(pd.notna(result.loc[7, name]))

    def test_multi_timescale_features_are_causal_and_gap_safe(self):
        metrics = catalogue(
            (
                "signal", "gauge", 1, "high_bad", "identity", 0.1,
                None, None, None, None,
            )
        )
        source = panel([0, 1, 2, 3, 4, 5, 20, 21, 22], signal=range(9))
        transformed = transform_episode(source, metrics)
        result = add_causal_temporal_features(
            transformed,
            metrics,
            history_windows_seconds={"4s": 4},
            lag_windows_seconds={"2s": 2},
            minimum_window_fraction=0.50,
        )

        lag = "signal__level__lag_2s"
        history = "signal__level__history_4s_z"
        self.assertEqual(result.loc[4, lag], 2.0)
        self.assertTrue(result.loc[6:7, lag].isna().all())
        self.assertTrue(pd.notna(result.loc[3, history]))
        self.assertTrue(result.loc[6:7, history].isna().all())

        prefix = add_causal_temporal_features(
            transform_episode(source.iloc[:5].copy(), metrics),
            metrics,
            history_windows_seconds={"4s": 4},
            lag_windows_seconds={"2s": 2},
            minimum_window_fraction=0.50,
        )
        pd.testing.assert_frame_equal(
            result.iloc[:5].reset_index(drop=True),
            prefix.reset_index(drop=True),
        )

    def test_error_activity_is_a_trailing_rate(self):
        metrics = catalogue(
            (
                "errors", "interval_count", 1, "high_bad", "hurdle_log1p",
                0.1, 0.0, None, None, None,
            )
        )
        transformed = transform_episode(
            panel([0, 1, 2, 3], errors=[0.0, 2.0, 0.0, 4.0]), metrics
        )
        result = add_causal_temporal_features(
            transformed,
            metrics,
            activity_windows_seconds={"4s": 4},
            minimum_window_fraction=0.50,
        )
        self.assertEqual(result.loc[3, "errors__nonzero__rate_4s"], 0.5)

    def test_temporal_feature_families_respect_metric_selection(self):
        metrics = catalogue(
            (
                "selected", "gauge", 1, "high_bad", "identity", 0.1,
                None, None, None, None,
            ),
            (
                "excluded", "gauge", 1, "high_bad", "identity", 0.1,
                None, None, None, None,
            ),
        )
        transformed = transform_episode(
            panel(
                [0, 1, 2, 3],
                selected=[1.0, 2.0, 3.0, 4.0],
                excluded=[4.0, 3.0, 2.0, 1.0],
            ),
            metrics,
        )
        result = add_causal_temporal_features(
            transformed,
            metrics,
            lag_windows_seconds={"1s": 1},
            lag_metric_ids=["selected"],
        )

        self.assertIn("selected__level__lag_1s", result)
        self.assertNotIn("excluded__level__lag_1s", result)


class EvidenceTests(unittest.TestCase):
    def test_empirical_tail_evidence_obeys_metric_direction(self):
        calibration = pd.DataFrame(
            {
                "high": [-2, -1, 0, 1, 2],
                "low": [-2, -1, 0, 1, 2],
                "both": [-2, -1, 0, 1, 2],
            }
        )
        directions = {"high": "high_bad", "low": "low_bad", "both": "two_sided"}
        reference = fit_empirical_tail_reference(
            calibration, directions, minimum_observations=5
        )
        observed = pd.DataFrame(
            {
                "high": [3.0, -3.0],
                "low": [3.0, -3.0],
                "both": [3.0, -3.0],
            }
        )
        result = empirical_tail_evidence(observed, reference, directions)

        self.assertGreater(result.loc[0, "high"], result.loc[1, "high"])
        self.assertLess(result.loc[0, "low"], result.loc[1, "low"])
        self.assertAlmostEqual(result.loc[0, "both"], result.loc[1, "both"])
        self.assertAlmostEqual(result.loc[0, "high"], np.log10(6))

    def test_directional_cusum_resets_at_a_gap(self):
        residuals = pd.DataFrame({"high": [1.0, 1.0, 1.0, 1.0]})
        scores, _ = directional_cusum(
            residuals,
            {"high": "high_bad"},
            allowance=0,
            reset_before=[True, False, True, False],
        )
        np.testing.assert_allclose(scores, [1, 2, 1, 2])

    def test_feature_policy_keeps_clipping_out_of_asset_health(self):
        metrics = catalogue(
            (
                "signal",
                "gauge",
                1,
                "low_bad",
                "identity",
                0.2,
                None,
                None,
                None,
                None,
            )
        )
        policy = feature_policy(metrics).set_index("feature")
        self.assertEqual(policy.at["signal__level", "direction"], "low_bad")
        self.assertEqual(policy.at["signal__clipped", "role"], "data_quality")

    def test_temporal_features_inherit_the_metric_direction(self):
        metrics = catalogue(
            (
                "signal", "gauge", 1, "low_bad", "identity", 0.2,
                None, None, None, None,
            )
        )
        names = [
            "signal__level__history_24h_z",
            "signal__level__lag_1h",
        ]
        policy = feature_policy(metrics, names).set_index("feature")

        self.assertEqual(policy.at[names[0], "direction"], "low_bad")
        self.assertEqual(policy.at[names[1], "direction"], "low_bad")


class DetectorIntegrationTests(unittest.TestCase):
    @staticmethod
    def bundle(direction="high_bad"):
        reference = np.sort(np.linspace(-2, 2, 101))
        return {
            "feature_columns": ["signal__level"],
            "global_centre": pd.Series({"signal__level": 0.0}),
            "global_scale": pd.Series({"signal__level": 1.0}),
            "entity_centre": None,
            "entity_scale": None,
            "feature_directions": {"signal__level": direction},
            "tail_reference": {"signal__level": reference},
            "residual_scale": pd.Series({"signal__level": 1.0}),
            "pca": None,
            "isolation_forest": None,
        }

    def test_rapid_channel_uses_directional_empirical_evidence(self):
        features = panel([0, 1], **{"signal__level": [-3.0, 3.0]})
        result = score_residual_episode(
            self.bundle("high_bad"),
            features,
            cadence_seconds=1,
            dispersion_window_seconds=4,
            cusum_allowance=0.5,
        )
        self.assertLess(
            result.loc[0, "rapid_residual"], result.loc[1, "rapid_residual"]
        )

    def test_detector_cusum_restarts_after_a_timestamp_gap(self):
        features = panel([0, 1, 10, 11], **{"signal__level": [1.0] * 4})
        result = score_residual_episode(
            self.bundle("high_bad"),
            features,
            cadence_seconds=1,
            dispersion_window_seconds=4,
            cusum_allowance=0.0,
        )
        np.testing.assert_allclose(result["drift_cusum"], [1, 2, 1, 2])


if __name__ == "__main__":
    unittest.main()
