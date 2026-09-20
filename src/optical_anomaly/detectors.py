"""Two detector tiers; calibration ranks are scores, never fault probabilities."""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from .features import FEATURES


@dataclass
class ScoreNormalizer:
    values: np.ndarray = field(default_factory=lambda: np.array([]))
    ranks: np.ndarray = field(default_factory=lambda: np.array([]))

    def fit(self, scores: pd.Series) -> "ScoreNormalizer":
        clean = scores.loc[np.isfinite(scores)].to_numpy()
        if len(clean) < 100:
            raise ValueError("Need at least 100 finite calibration scores")
        self.values, counts = np.unique(clean, return_counts=True)
        if len(self.values) < 2:
            raise ValueError("Degenerate calibration score distribution")
        self.ranks = (np.cumsum(counts) - counts / 2) / len(clean)
        return self

    def transform(self, scores: pd.Series) -> pd.Series:
        if len(self.values) < 2:
            raise ValueError("Fit score normalizer first")
        return pd.Series(
            np.interp(scores, self.values, self.ranks, left=0, right=1),
            index=scores.index,
        )


@dataclass
class StatisticalDetector:
    normalizer: ScoreNormalizer = field(default_factory=ScoreNormalizer)

    def raw_score(self, features: pd.DataFrame) -> pd.Series:
        # Slope already uses EWMA in FeatureEngineer; negative means deterioration.
        return -features.slope.where(features.slope.notna())

    def calibrate(self, features: pd.DataFrame) -> "StatisticalDetector":
        self.normalizer.fit(self.raw_score(features))
        return self

    def score(self, features: pd.DataFrame) -> pd.Series:
        return self.normalizer.transform(self.raw_score(features))


@dataclass
class IsolationForestDetector:
    seed: int = 42
    feature_columns: list[str] = field(default_factory=lambda: FEATURES.copy())
    forest: IsolationForest = field(init=False)
    normalizer: ScoreNormalizer = field(default_factory=ScoreNormalizer)

    def fit(self, features: pd.DataFrame) -> "IsolationForestDetector":
        complete = features[self.feature_columns].dropna()
        if len(complete) < 100:
            raise ValueError("Need 100 complete training rows")
        self.forest = IsolationForest(
            n_estimators=100, max_samples=256, random_state=self.seed, n_jobs=1
        )
        self.forest.fit(
            complete.sample(min(len(complete), 20000), random_state=self.seed)
        )
        return self

    def raw_score(self, features: pd.DataFrame) -> pd.Series:
        valid = features[self.feature_columns].notna().all(axis=1)
        scores = pd.Series(np.nan, index=features.index)
        if valid.any():
            scores.loc[valid] = -self.forest.score_samples(
                features.loc[valid, self.feature_columns]
            )
        return scores

    def calibrate(self, features: pd.DataFrame) -> "IsolationForestDetector":
        self.normalizer.fit(self.raw_score(features))
        return self

    def score(self, features: pd.DataFrame) -> pd.Series:
        return self.normalizer.transform(self.raw_score(features))
