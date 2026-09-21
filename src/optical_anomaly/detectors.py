"""Two detector tiers; calibration ranks are scores, never fault probabilities."""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from .features import FEATURES


@dataclass
class ScoreNormalizer:
    """Held-out baseline mid-ranks; neither probabilities nor conformal p-values."""

    values: np.ndarray = field(default_factory=lambda: np.array([]))
    ranks: np.ndarray = field(default_factory=lambda: np.array([]))

    def fit(self, scores: pd.Series) -> "ScoreNormalizer":
        clean = scores.loc[np.isfinite(scores)].to_numpy()
        # Assumption: a numerical guard, not 100 independent time-series samples.
        if len(clean) < 100:
            raise ValueError("Need at least 100 finite calibration scores")
        self.values, counts = np.unique(clean, return_counts=True)
        if len(self.values) < 2:
            raise ValueError("Degenerate calibration score distribution")
        # Convention: ties get their midpoint rank; interpolate between ranks.
        self.ranks = (np.cumsum(counts) - counts / 2) / len(clean)
        return self

    def transform(self, scores: pd.Series) -> pd.Series:
        if len(self.values) < 2:
            raise ValueError("Fit score normalizer first")
        normalized = pd.Series(
            np.interp(scores, self.values, self.ranks, left=0, right=1),
            index=scores.index,
        )
        # Invalid computations cannot become a confident anomaly or healthy score.
        return normalized.where(np.isfinite(scores))


@dataclass
class StatisticalDetector:
    normalizer: ScoreNormalizer = field(default_factory=ScoreNormalizer)

    def raw_score(self, features: pd.DataFrame) -> pd.Series:
        # Slope already uses EWMA in FeatureEngineer; negative means deterioration.
        # Source: NIST EWMA chart; using derivatives is our modelling choice.
        # https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc324.htm
        return -features.slope.where(features.slope.notna())

    def calibrate(self, features: pd.DataFrame) -> "StatisticalDetector":
        self.normalizer.fit(self.raw_score(features))
        return self

    def score(self, features: pd.DataFrame) -> pd.Series:
        return self.normalizer.transform(self.raw_score(features))


@dataclass
class IsolationForestDetector:
    seed: int = 42  # Assumption: reproducibility seed, not a tuned model parameter.
    feature_columns: list[str] = field(default_factory=lambda: FEATURES.copy())
    forest: IsolationForest = field(init=False)
    normalizer: ScoreNormalizer = field(default_factory=ScoreNormalizer)

    def fit(self, features: pd.DataFrame) -> "IsolationForestDetector":
        values = features[self.feature_columns]
        complete = values.loc[np.isfinite(values).all(axis=1)]
        # Assumption: fail on tiny baselines; this does not prove sample adequacy.
        if len(complete) < 100:
            raise ValueError("Need 100 complete training rows")
        # Source: Liu et al. (2008), section 4.1: 100 trees, 256-row subsamples.
        # https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf
        self.forest = IsolationForest(
            n_estimators=100,
            max_samples=min(256, len(complete)),
            random_state=self.seed,
            n_jobs=1,
        )
        # Assumption: this extra cap limits notebook memory, not a paper optimum.
        self.forest.fit(
            complete.sample(min(len(complete), 20000), random_state=self.seed)
        )
        return self

    def raw_score(self, features: pd.DataFrame) -> pd.Series:
        valid = np.isfinite(features[self.feature_columns]).all(axis=1)
        scores = pd.Series(np.nan, index=features.index)
        if valid.any():
            # sklearn score_samples is lower for anomalies, so reverse its sign.
            # Source: sklearn.ensemble.IsolationForest.score_samples documentation.
            scores.loc[valid] = -self.forest.score_samples(
                features.loc[valid, self.feature_columns]
            )
        return scores

    def calibrate(self, features: pd.DataFrame) -> "IsolationForestDetector":
        self.normalizer.fit(self.raw_score(features))
        return self

    def score(self, features: pd.DataFrame) -> pd.Series:
        return self.normalizer.transform(self.raw_score(features))
