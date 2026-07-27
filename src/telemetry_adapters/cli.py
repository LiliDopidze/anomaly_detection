"""Command-line entry point for adapter materialisation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .synthetic_gpon import SyntheticGponAdapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialise canonical telemetry contracts")
    parser.add_argument("--native", type=Path, required=True, help="native dataset directory")
    parser.add_argument("--core", type=Path, required=True, help="SPEC-CORE output directory")
    parser.add_argument(
        "--eval",
        dest="evaluation",
        type=Path,
        help="optional, physically separate SPEC-EVAL output directory",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = SyntheticGponAdapter().materialise(
        args.native, args.core, args.evaluation
    )
    print(
        f"{report.adapter_id} {report.adapter_version}: "
        f"{len(report.row_counts)} tables materialised"
    )


if __name__ == "__main__":
    main()
