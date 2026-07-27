"""Small command-line surface used by notebooks and automation."""

from __future__ import annotations

import argparse
import json
import resource
import sys
from pathlib import Path

from .oil_well import ThreeWAdapter
from .telecom import NativeSelection, SyntheticGponAdapter


def _rss_gib() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024**3) if sys.platform == "darwin" else raw * 1024 / (1024**3)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialise sector-neutral telemetry contracts"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    telecom = commands.add_parser("telecom", help="translate telemetry-synth v4")
    telecom.add_argument("--native", type=Path, required=True)
    telecom.add_argument("--core", type=Path, required=True)
    telecom.add_argument("--eval", dest="evaluation", type=Path)
    telecom.add_argument("--sample-start")
    telecom.add_argument("--sample-end")
    telecom.add_argument("--entity-id", action="append", default=[])
    telecom.add_argument("--batch-native-rows", type=int, default=250_000)
    telecom.add_argument("--report", type=Path)

    threew = commands.add_parser("threew", help="translate Petrobras 3W 2.0.0")
    threew.add_argument("--native", type=Path, required=True)
    threew.add_argument("--core", type=Path, required=True)
    threew.add_argument("--eval", dest="evaluation", type=Path)
    threew.add_argument("--fixture", type=Path)
    threew.add_argument("--report", type=Path)
    return parser


def _write_report(path: Path | None, report, *, peak_rss_gib: float) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "adapter_id": report.adapter_id,
                "adapter_version": report.adapter_version,
                "source_format": report.source_format,
                "core_path": str(report.core_path),
                "evaluation_path": (
                    None
                    if report.evaluation_path is None
                    else str(report.evaluation_path)
                ),
                "row_counts": dict(report.row_counts),
                "truth_columns_removed": list(report.truth_columns_removed),
                "peak_rss_gib": peak_rss_gib,
                "memory_scope": "clean materialisation subprocess",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "telecom":
        selection = NativeSelection(
            sample_start=args.sample_start,
            sample_end=args.sample_end,
            entity_ids=tuple(args.entity_id),
            batch_native_rows=args.batch_native_rows,
        )
        report = SyntheticGponAdapter(selection=selection).materialise(
            args.native, args.core, args.evaluation
        )
    else:
        report = ThreeWAdapter().materialise(
            args.native,
            args.core,
            args.evaluation,
            fixture_destination=args.fixture,
        )
    _write_report(args.report, report, peak_rss_gib=_rss_gib())
    print(
        f"{report.adapter_id} {report.adapter_version}: "
        f"{len(report.row_counts)} tables materialised"
    )


if __name__ == "__main__":
    main()
