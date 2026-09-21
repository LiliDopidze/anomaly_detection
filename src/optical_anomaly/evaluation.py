"""Event-level operational evaluation. Point adjustment is never performed."""

from dataclasses import dataclass
import numpy as np
import pandas as pd


def match_events(incidents: pd.DataFrame, faults: pd.DataFrame) -> dict[str, str]:
    """Deterministic maximum-cardinality one-to-one interval/entity matching."""
    choices = {}
    for alert in incidents.sort_values(["start_time", "incident_id"]).itertuples():
        eligible = faults.loc[
            faults.entity_id.eq(alert.entity_id)
            & faults.onset_time.le(alert.start_time)
            & faults.end_time.gt(alert.start_time)
        ].sort_values(["end_time", "onset_time", "fault_id"])
        choices[alert.incident_id] = eligible.fault_id.tolist()
    assigned: dict[str, str] = {}

    def assign(alert: str, visited: set[str]) -> bool:
        for fault in choices[alert]:
            if fault in visited:
                continue
            visited.add(fault)
            previous = assigned.get(fault)
            if previous is None or assign(previous, visited):
                assigned[fault] = alert
                return True
        return False

    for alert in choices:
        assign(alert, set())
    return assigned


@dataclass(frozen=True)
class Evaluator:
    interval_minutes: int = 5
    minimum_lead_minutes: int = 30
    opportunity_intervals: int = 3

    def _opportunity(self, fault: pd.Series, available: pd.DataFrame) -> bool:
        """Observable precursor and consecutive telemetry; never detector success."""
        if pd.isna(fault.impact_time) or pd.isna(fault.observable_onset_time):
            return False
        deadline = fault.impact_time - pd.Timedelta(minutes=self.minimum_lead_minutes)
        rows = available.loc[
            available.entity_id.eq(fault.entity_id)
            & available.timestamp.ge(fault.observable_onset_time)
            & available.timestamp.le(deadline)
        ].sort_values("timestamp")
        count, previous = 0, None
        for row in rows.itertuples():
            gap = previous is not None and row.timestamp - previous > pd.Timedelta(
                minutes=self.interval_minutes * 1.5
            )
            count = 0 if gap or pd.isna(row.value) else count
            if pd.notna(row.value):
                count += 1
            if count >= self.opportunity_intervals:
                return True
            previous = row.timestamp
        return False

    def _outcomes(
        self,
        faults: pd.DataFrame,
        by_id: pd.DataFrame,
        matches: dict[str, str],
        available: pd.DataFrame,
    ) -> pd.DataFrame:
        outcomes = []
        for _, fault in faults.iterrows():
            alert = (
                by_id.loc[matches[fault.fault_id]]
                if fault.fault_id in matches
                else None
            )
            opportunity = self._opportunity(fault, available)
            early = (
                alert is not None
                and pd.notna(fault.impact_time)
                and alert.start_time < fault.impact_time
            )
            variance_shift = fault.fault_type == "variance_shift"
            delay_reference = (
                fault.onset_time if variance_shift else fault.observable_onset_time
            )
            delay = (
                (alert.start_time - delay_reference).total_seconds() / 60
                if alert is not None and pd.notna(delay_reference)
                else np.nan
            )
            outcomes.append(
                {
                    "fault_id": fault.fault_id,
                    "fault_type": fault.fault_type,
                    "detected": alert is not None,
                    "opportunity": opportunity,
                    "pre_impact": early,
                    "delay_minutes": delay,
                    "delay_reference": (
                        "onset_time" if variance_shift else "observable_onset_time"
                    ),
                }
            )
        outcomes = pd.DataFrame(
            outcomes,
            columns=[
                "fault_id",
                "fault_type",
                "detected",
                "opportunity",
                "pre_impact",
                "delay_minutes",
                "delay_reference",
            ],
        )
        return outcomes

    def _counts(
        self,
        alerts: pd.DataFrame,
        faults: pd.DataFrame,
        boundary: pd.DataFrame,
        matches: dict[str, str],
    ) -> tuple[int, int, int]:
        used = set(matches.values())
        duplicate, unmatched, excluded = 0, 0, 0
        for alert in alerts.itertuples():
            if alert.incident_id in used:
                continue

            def overlaps(table: pd.DataFrame) -> bool:
                return bool(
                    (
                        table.entity_id.eq(alert.entity_id)
                        & table.onset_time.le(alert.start_time)
                        & table.end_time.gt(alert.start_time)
                    ).any()
                )

            if overlaps(faults):
                duplicate += 1
            elif overlaps(boundary):
                excluded += 1
            else:
                unmatched += 1
        return duplicate, unmatched, excluded

    def _metrics(
        self,
        outcomes: pd.DataFrame,
        available: pd.DataFrame,
        faults: pd.DataFrame,
        alerts: pd.DataFrame,
        boundary: pd.DataFrame,
        matches: dict[str, str],
    ) -> dict:
        duplicate, unmatched, excluded = self._counts(alerts, faults, boundary, matches)
        # Scheduled monitoring exposure, explicitly accompanied by observed coverage.
        days = len(available) * self.interval_minutes / 1440
        opportunities = int(outcomes.opportunity.sum())
        early = int((outcomes.opportunity & outcomes.pre_impact).sum())
        observable_delays = outcomes.loc[
            outcomes.delay_reference.eq("observable_onset_time"), "delay_minutes"
        ]
        variance_delays = outcomes.loc[
            outcomes.delay_reference.eq("onset_time"), "delay_minutes"
        ]
        metrics = {
            "faults": len(faults),
            "detected": len(matches),
            "missed": len(faults) - len(matches),
            "warning_opportunities": opportunities,
            "pre_impact_detected": early,
            "pre_impact_recall": early / opportunities if opportunities else None,
            "unmatched_incidents": unmatched,
            "duplicate_incidents": duplicate,
            "nuisance_per_1000_entity_days": (
                1000 * (unmatched + duplicate) / days if days else None
            ),
            "monitored_entity_days": days,
            "observation_coverage": float(available.value.notna().mean()),
            "median_detection_delay_minutes": (
                float(observable_delays.median())
                if observable_delays.notna().any()
                else None
            ),
            "median_variance_onset_delay_minutes": (
                float(variance_delays.median())
                if variance_delays.notna().any()
                else None
            ),
            "incidents": len(alerts),
            "boundary_incidents": excluded,
            "boundary_faults": len(boundary),
        }
        return metrics

    def evaluate(
        self,
        incidents: pd.DataFrame,
        truth: pd.DataFrame,
        telemetry: pd.DataFrame,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> tuple[dict, pd.DataFrame]:
        alerts = incidents.loc[
            incidents.start_time.ge(start) & incidents.start_time.lt(end)
        ].copy()
        faults = truth.loc[truth.onset_time.ge(start) & truth.end_time.le(end)].copy()
        boundary = truth.loc[
            truth.onset_time.lt(end)
            & truth.end_time.gt(start)
            & ~truth.fault_id.isin(faults.fault_id)
        ]
        matches = match_events(alerts, faults)
        by_id = alerts.set_index("incident_id")
        available = telemetry.loc[
            telemetry.metric_name.eq("rx_power_dbm")
            & telemetry.timestamp.ge(start)
            & telemetry.timestamp.lt(end)
        ]
        outcomes = self._outcomes(faults, by_id, matches, available)
        metrics = self._metrics(outcomes, available, faults, alerts, boundary, matches)
        return metrics, outcomes
