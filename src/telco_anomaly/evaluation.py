"""Event matching and uncertainty intervals used by the synthetic pipeline."""

import numpy as np
from scipy.stats import chi2, norm


def _maximum_event_matches(candidates, ordered_alerts):
    """Return a deterministic maximum-cardinality alert-to-fault match."""

    if candidates.empty:
        return {}

    ranked = candidates.copy()
    ranked["alert_key"] = ranked["alert_id"].astype(str)
    ranked["fault_key"] = ranked["fault_id"].astype(str)
    ranked = ranked.sort_values(
        [
            "alert_key",
            "match_end",
            "match_start",
            "fault_key",
        ]
    ).drop_duplicates(["alert_key", "fault_key"])
    choices = {
        alert_key: group["fault_key"].tolist()
        for alert_key, group in ranked.groupby("alert_key", sort=False)
    }

    fault_to_alert = {}
    alert_to_fault = {}

    def assign(alert_key, visited_faults):
        for fault_key in choices.get(alert_key, []):
            if fault_key in visited_faults:
                continue
            visited_faults.add(fault_key)
            previous_alert = fault_to_alert.get(fault_key)
            if previous_alert is None or assign(previous_alert, visited_faults):
                fault_to_alert[fault_key] = alert_key
                alert_to_fault[alert_key] = fault_key
                return True
        return False

    for alert_id in ordered_alerts["alert_id"].astype(str):
        assign(alert_id, set())
    return alert_to_fault


def _wilson_interval(successes, total, z=1.96):
    if total == 0:
        return np.nan, np.nan
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * np.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return centre - margin, centre + margin


def wilson_interval(successes, total, confidence_level=0.95):
    """Return a two-sided Wilson interval for a binomial proportion."""

    confidence_level = float(confidence_level)
    if not np.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly between zero and one")
    if int(successes) != successes or int(total) != total:
        raise ValueError("successes and total must be integers")
    successes, total = int(successes), int(total)
    if successes < 0 or total < 0 or successes > total:
        raise ValueError("Require 0 <= successes <= total")
    z = norm.ppf(1 - (1 - confidence_level) / 2)
    return _wilson_interval(successes, total, z=z)


def poisson_rate_interval(count, exposure, confidence_level=0.95):
    """Return an exact two-sided Garwood interval for a Poisson rate."""

    confidence_level = float(confidence_level)
    if not np.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly between zero and one")

    count_value = float(count)
    if not np.isfinite(count_value) or count_value < 0 or not count_value.is_integer():
        raise ValueError("count must be a non-negative integer")
    count = int(count_value)

    exposure = float(exposure)
    if not np.isfinite(exposure) or exposure <= 0:
        return np.nan, np.nan

    alpha = 1 - confidence_level
    lower = 0.0 if count == 0 else chi2.ppf(alpha / 2, 2 * count) / 2
    upper = chi2.ppf(1 - alpha / 2, 2 * (count + 1)) / 2
    return lower / exposure, upper / exposure
