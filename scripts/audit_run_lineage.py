"""Check that one EDA, feature, and model run share the same inputs.

This diagnostic reads only the four paths supplied on the command line. It
does not read evaluation labels or holdout data and never changes run files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        content = json.load(source)
    if not isinstance(content, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return content


def audit(eda_path: Path, feature_path: Path, model_path: Path,
          config_path: Path) -> int:
    eda = read_manifest(eda_path)
    features = read_manifest(feature_path)
    model = read_manifest(model_path)

    checks = [
        ("EDA decisions used by features",
         features.get("eda_decisions_sha256"), sha256(eda_path)),
        ("EDA decisions cited by model",
         model.get("eda_decisions_sha256"), sha256(eda_path)),
        ("Feature manifest used by model",
         model.get("feature_manifest_sha256"), sha256(feature_path)),
        ("Feature config used by features",
         features.get("feature_config_sha256"), sha256(config_path)),
        ("EDA and features canonical core",
         eda.get("canonical_fingerprint"), features.get("core_fingerprint")),
        ("Features and model canonical core",
         features.get("core_fingerprint"), model.get("core_fingerprint")),
    ]

    failed = 0
    for name, recorded, actual in checks:
        if recorded and actual and recorded == actual:
            print(f"PASS  {name}")
        else:
            failed += 1
            print(f"FAIL  {name}")
            print(f"      recorded: {recorded or '<missing>'}")
            print(f"      actual:   {actual or '<missing>'}")

    print(f"\n{len(checks) - failed}/{len(checks)} lineage checks passed")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eda-decisions", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--feature-config", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        return audit(arguments.eda_decisions, arguments.feature_manifest,
                     arguments.model_manifest, arguments.feature_config)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
