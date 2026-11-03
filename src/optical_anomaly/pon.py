"""GPON measurement definitions; numerical simulation assumptions live elsewhere."""

from .contracts import MetricDefinition


# Source: ETSI GS F5G 011 §§8.3–8.4; definitions, not synthetic distributions.
CANONICAL_METRICS = {
    "rx_power_dbm": MetricDefinition("dBm", "gauge", "Downstream Rx at ONT"),
    "upstream_rx_power_dbm": MetricDefinition(
        "dBm", "gauge", "Upstream Rx at OLT per ONT"
    ),
    "ont_tx_power_dbm": MetricDefinition("dBm", "gauge", "ONT transmit power"),
    "olt_tx_power_dbm": MetricDefinition(
        "dBm", "gauge", "Shared OLT port transmit power"
    ),
    "ont_temperature_c": MetricDefinition("C", "gauge", "ONT temperature"),
    "olt_temperature_c": MetricDefinition("C", "gauge", "Shared OLT port temperature"),
    "ber": MetricDefinition(
        "ratio", "gauge", "Downstream pre-FEC BER", lower=0, upper=1
    ),
    "upstream_ber": MetricDefinition(
        "ratio", "gauge", "Upstream pre-FEC BER", lower=0, upper=1
    ),
}
for direction in ("downstream", "upstream"):
    for kind in ("corrected", "uncorrectable", "total"):
        # Source: ITU-T G.989.3 Table 14-1. Total means ALL received codewords.
        # ETSI's upstream-FEC-total-block instead counts errored blocks only.
        name = f"{direction}_fec_{kind}_codewords"
        CANONICAL_METRICS[name] = MetricDefinition(
            "interval_count",
            "interval_count",
            f"{direction.title()} {kind} codewords in preceding reporting interval",
            count_group=f"{direction}_fec",
        )
CANONICAL_UNITS = {name: item.unit for name, item in CANONICAL_METRICS.items()}
