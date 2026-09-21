"""Stateful debouncing; unknown telemetry never counts as recovery."""

from dataclasses import dataclass, field
from typing import Literal
import numpy as np
import pandas as pd


@dataclass
class EntityState:
    status: Literal["CLOSED", "OPEN"] = "CLOSED"
    high_count: int = 0
    low_count: int = 0
    previous: pd.Timestamp | None = None
    last_valid: pd.Timestamp | None = None
    start: pd.Timestamp | None = None
    max_score: float = 0.0
    number: int = 0


@dataclass
class IncidentManager:
    # Assumptions: operational policy candidates, not telecom-standard limits.
    high: float = 0.99
    low: float = 0.8
    opening_intervals: int = 3
    closing_intervals: int = 3
    interval_minutes: int = 5
    max_gap_hours: float = 6.0
    states: dict[str, EntityState] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if not 0 <= self.low < self.high <= 1:
            raise ValueError("Require 0 <= low < high <= 1")
        if (
            min(self.opening_intervals, self.closing_intervals, self.interval_minutes)
            < 1
        ):
            raise ValueError("Intervals/counts must be positive")
        if self.max_gap_hours <= 0:
            raise ValueError("Maximum gap must be positive")

    def _record(
        self, entity: str, state: EntityState, end: pd.Timestamp | None, reason: str
    ) -> dict:
        return {
            "incident_id": f"{entity}-{state.number}",
            "entity_id": entity,
            "start_time": state.start,
            "end_time": end,
            "max_score": state.max_score,
            "status": "OPEN" if end is None else "CLOSED",
            "reason": reason,
        }

    def process(self, scores: pd.DataFrame) -> pd.DataFrame:
        """Return new closures plus active snapshots; upsert by incident_id across calls."""
        records = []
        for entity, group in scores.groupby("entity_id", sort=True):
            state = self.states.setdefault(str(entity), EntityState())
            for row in group.sort_values("timestamp").itertuples(index=False):
                time, score = row.timestamp, row.score
                if state.previous is not None:
                    if time <= state.previous:
                        raise ValueError("Replay/overlapping chunks are not allowed")
                    # Assumption: allow half an interval of timestamp jitter.
                    if time - state.previous > pd.Timedelta(
                        minutes=self.interval_minutes * 1.5
                    ):
                        state.high_count = state.low_count = 0
                if (
                    state.last_valid is not None
                    and time - state.last_valid > pd.Timedelta(hours=self.max_gap_hours)
                ):
                    if state.status == "OPEN":
                        records.append(
                            self._record(
                                entity, state, state.last_valid, "telemetry_gap"
                            )
                        )
                    state.status, state.high_count, state.low_count = "CLOSED", 0, 0
                state.previous = time
                if not np.isfinite(score):
                    state.high_count = state.low_count = 0
                    continue
                if not 0 <= score <= 1:
                    raise ValueError("Scores must lie in [0, 1] or be missing")
                state.last_valid = time
                if state.status == "CLOSED":
                    state.high_count = state.high_count + 1 if score > self.high else 0
                    if state.high_count >= self.opening_intervals:
                        # Timestamp the actual alert; never backdate by N intervals.
                        state.status, state.start, state.max_score = "OPEN", time, score
                        state.number += 1
                else:
                    state.max_score = max(state.max_score, score)
                    state.low_count = state.low_count + 1 if score < self.low else 0
                    if state.low_count >= self.closing_intervals:
                        records.append(self._record(entity, state, time, "recovery"))
                        state.status, state.high_count, state.low_count = "CLOSED", 0, 0
        for entity, state in self.states.items():
            if state.status == "OPEN":
                records.append(self._record(entity, state, None, "active"))
        columns = [
            "incident_id",
            "entity_id",
            "start_time",
            "end_time",
            "max_score",
            "status",
            "reason",
        ]
        return pd.DataFrame(records, columns=columns)
