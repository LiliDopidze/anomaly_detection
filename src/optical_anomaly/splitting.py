"""Timestamp boundaries rather than copied train/validation/test datasets."""

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    train_end: pd.Timestamp
    calibration_end: pd.Timestamp
    validation_end: pd.Timestamp
    test_end: pd.Timestamp

    def __post_init__(self) -> None:
        boundaries = [
            self.train_end,
            self.calibration_end,
            self.validation_end,
            self.test_end,
        ]
        if any(t.tzinfo is None for t in boundaries) or not all(
            a < b for a, b in zip(boundaries, boundaries[1:])
        ):
            raise ValueError(
                "Boundaries must be timezone-aware and strictly increasing"
            )

    def masks(self, times: pd.Series) -> dict[str, pd.Series]:
        # Half-open partitions share no observations; past feature history is valid.
        # Fit all learned references on train only (scikit-learn leakage guidance):
        # https://scikit-learn.org/stable/common_pitfalls.html#data-leakage
        return {
            "train": times < self.train_end,
            "calibration": (times >= self.train_end) & (times < self.calibration_end),
            "validation": (times >= self.calibration_end)
            & (times < self.validation_end),
            "test": (times >= self.validation_end) & (times < self.test_end),
        }
