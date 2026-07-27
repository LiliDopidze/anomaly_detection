"""Build the readable, Google Drive-ready Milestone 1 v0.3 notebooks."""

from __future__ import annotations

import json
import hashlib
import zipfile
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "google_drive" / "milestone1_v0_3"
CONTRACT_VERSION = "v0.3"
RUNTIME_BUNDLE_NAME = "M1_v0_3_runtime_bundle.zip"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def notebook(title: str, cells: list[object]):
    document = nbf.v4.new_notebook(cells=cells)
    document.metadata = {
        "colab": {"name": title, "provenance": []},
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    }
    return document


def source_payload() -> dict[str, str]:
    payload = {}
    package_names = (
        "telemetry_contract",
        "telemetry_eval_contract",
        "telemetry_packs",
        "telemetry_adapters",
        "telemetry_runtime",
    )
    for package_name in package_names:
        package_root = ROOT / "src" / package_name
        for path in sorted(package_root.rglob("*")):
            if (
                path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix in {".py", ".json"}
            ):
                payload[str(path.relative_to(ROOT / "src"))] = path.read_text(
                    encoding="utf-8"
                )
    return payload


def build_runtime_bundle(payload: dict[str, str]) -> tuple[str, dict[str, str]]:
    """Create a deterministic source archive instead of embedding code in Notebook 02."""

    hashes = {
        relative_path: hashlib.sha256(text.encode("utf-8")).hexdigest()
        for relative_path, text in sorted(payload.items())
    }
    manifest = {
        "bundle_id": "milestone1-v0.3-runtime",
        "contract_tag": CONTRACT_VERSION,
        "hash_algorithm": "sha256",
        "files": hashes,
        "legacy_baseline_descriptor": json.loads(
            (ROOT / "legacy" / "baseline-freeze.json").read_text(encoding="utf-8")
        ),
        "legacy_attachment_manifest": json.loads(
            (ROOT / "legacy" / "attachment-manifest.json").read_text(
                encoding="utf-8"
            )
        ),
    }
    destination = OUTPUT / RUNTIME_BUNDLE_NAME

    def write_member(archive, relative_path: str, content: bytes):
        info = zipfile.ZipInfo(relative_path, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        archive.writestr(info, content)

    with zipfile.ZipFile(destination, "w") as archive:
        write_member(
            archive,
            "runtime_manifest.json",
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        for relative_path, text in sorted(payload.items()):
            write_member(archive, relative_path, text.encode("utf-8"))

    bundle_hash = hashlib.sha256(destination.read_bytes()).hexdigest()
    return bundle_hash, hashes


DRIVE_SETUP = r'''
from pathlib import Path
import json
import os
import sys

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    DEFAULT_DRIVE_ROOT = Path("/content/drive/MyDrive/anomaly_detection")
else:
    DEFAULT_DRIVE_ROOT = Path.cwd() / "anomaly_detection"

DRIVE_ROOT = Path(
    os.environ.get("ANOMALY_DETECTION_DRIVE_ROOT", str(DEFAULT_DRIVE_ROOT))
).expanduser()
CONTRACT_TAG = "v0.3"
CONTRACT_ROOT = DRIVE_ROOT / "contracts" / CONTRACT_TAG
PYTHON_SOURCE_ROOT = CONTRACT_ROOT / "python_src"
OUTPUT_ROOT = DRIVE_ROOT / "outputs" / "milestone_1" / CONTRACT_TAG

print("Drive root:   ", DRIVE_ROOT)
print("Contract root:", CONTRACT_ROOT)
print("Output root:  ", OUTPUT_ROOT)
'''


IMPORT_CONTRACT = DRIVE_SETUP + r'''
if not PYTHON_SOURCE_ROOT.is_dir():
    raise FileNotFoundError(
        f"Contract source not found at {PYTHON_SOURCE_ROOT}. Run Notebook 02 first."
    )
if str(PYTHON_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_SOURCE_ROOT))
'''


def build_notebook_02(runtime_bundle_hash: str):
    return notebook(
        "02_M1_V03_CONTRACT_AND_PACK.ipynb",
        [
            md(
                """
# Notebook 02 — Authoritative neutral contract and Telecom Pack

This notebook creates the **locked vocabulary** for Milestone 1:

- **SPEC-CORE v0.3**: information a detector may observe at runtime.
- **SPEC-EVAL v0.3**: fault and condition truth used only after scoring.
- **Telecom Pack v0.1**: telecom metric, hierarchy, quality and exposure meaning.
- A hash-pinned Python source directory used by Notebooks 03–05.

It does not translate data. Run it once before the other notebooks.
"""
            ),
            code(DRIVE_SETUP),
            md(
                """
## 1. Publish the frozen implementation

The implementation is supplied as the separate, reviewable
`M1_v0_3_runtime_bundle.zip` file included with these notebooks. Put that file at:

`MyDrive/anomaly_detection/bootstrap/M1_v0_3_runtime_bundle.zip`

The short installer below verifies both the bundle and every extracted file.
Re-running is safe when contents are identical; changed content is never overwritten.
"""
            ),
            code(
                f'''
import hashlib
import zipfile
from pathlib import PurePosixPath

RUNTIME_BUNDLE_NAME = "{RUNTIME_BUNDLE_NAME}"
EXPECTED_BUNDLE_SHA256 = "{runtime_bundle_hash}"

configured_bundle = os.environ.get("ANOMALY_DETECTION_RUNTIME_BUNDLE")
bundle_candidates = []
if configured_bundle:
    bundle_candidates.append(Path(configured_bundle).expanduser())
bundle_candidates.extend([
    DRIVE_ROOT / "bootstrap" / RUNTIME_BUNDLE_NAME,
    DRIVE_ROOT / RUNTIME_BUNDLE_NAME,
    Path("/content") / RUNTIME_BUNDLE_NAME,
])
RUNTIME_BUNDLE = next(
    (path for path in bundle_candidates if path.is_file()),
    None,
)
if RUNTIME_BUNDLE is None:
    raise FileNotFoundError(
        "Runtime bundle not found. Copy M1_v0_3_runtime_bundle.zip to "
        "MyDrive/anomaly_detection/bootstrap/ and rerun this cell."
    )

def write_immutable_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != text:
            raise FileExistsError(f"Immutable artifact differs: {{path}}")
        return "verified"
    path.write_text(text, encoding="utf-8")
    return "created"

actual_bundle_hash = hashlib.sha256(RUNTIME_BUNDLE.read_bytes()).hexdigest()
assert actual_bundle_hash == EXPECTED_BUNDLE_SHA256, (
    "Runtime bundle hash mismatch. Use the bundle supplied with this notebook."
)

statuses = {{}}
with zipfile.ZipFile(RUNTIME_BUNDLE) as archive:
    runtime_manifest = json.loads(
        archive.read("runtime_manifest.json").decode("utf-8")
    )
    assert runtime_manifest["contract_tag"] == CONTRACT_TAG
    SOURCE_HASHES = runtime_manifest["files"]

    for relative_path, expected_hash in sorted(SOURCE_HASHES.items()):
        member = PurePosixPath(relative_path)
        if member.is_absolute() or ".." in member.parts:
            raise ValueError(f"Unsafe bundle member: {{relative_path}}")
        content = archive.read(relative_path)
        assert hashlib.sha256(content).hexdigest() == expected_hash

        destination = PYTHON_SOURCE_ROOT / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.read_bytes() != content:
                raise FileExistsError(
                    f"Immutable artifact differs: {{destination}}"
                )
            statuses[relative_path] = "verified"
        else:
            destination.write_bytes(content)
            statuses[relative_path] = "created"

print("Runtime bundle:", RUNTIME_BUNDLE)
print("Files created:", sum(value == "created" for value in statuses.values()))
print("Files verified:", sum(value == "verified" for value in statuses.values()))
'''
            ),
            md(
                """
## 2. Load and inspect the contracts

Validity appears **once**, in `entity_registry.valid_from` / `valid_to`.
There is deliberately no second `entity_service_windows` canonical table.

`gt_condition_states` supports state-labelled datasets such as Petrobras 3W.
It deliberately has no `severity_ordinal`, because 3W does not exercise graded severity.
"""
            ),
            code(
                r'''
if str(PYTHON_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_SOURCE_ROOT))

from telemetry_contract import CONTRACT_VERSION
from telemetry_eval_contract import EVAL_CONTRACT_VERSION

schema_root = PYTHON_SOURCE_ROOT / "telemetry_contract" / "schemas" / "spec-core"
eval_schema_root = (
    PYTHON_SOURCE_ROOT / "telemetry_eval_contract" / "schemas" / "spec-eval"
)
SPEC_CORE_TABLES = sorted(path.stem.replace(".schema", "") for path in schema_root.glob("*.schema.json"))
SPEC_EVAL_TABLES = sorted(path.stem.replace(".schema", "") for path in eval_schema_root.glob("*.schema.json"))

assert CONTRACT_VERSION == "0.3.0"
assert EVAL_CONTRACT_VERSION == "0.3.0"
assert "entity_service_windows" not in SPEC_CORE_TABLES

condition_schema = json.loads(
    (eval_schema_root / "gt_condition_states.schema.json").read_text(encoding="utf-8")
)
assert "severity_ordinal" not in condition_schema["properties"]

print("SPEC-CORE:", SPEC_CORE_TABLES)
print("SPEC-EVAL:", SPEC_EVAL_TABLES)
'''
            ),
            md(
                """
## 3. Inspect the Telecom Pack phrasebook

The catalogue is intentionally rich enough to drive profiling and resampling:
sampling and aggregation semantics, nullability, censoring, bounds, candidate periods,
context keys, and exposure semantics are part of the interface.
"""
            ),
            code(
                r'''
from dataclasses import asdict
import pandas as pd
from telemetry_packs.telecom import load_pack

pack = load_pack()
metric_rows = []
for metric in pack.metric_specs().values():
    row = asdict(metric)
    row["measurement_kind"] = metric.measurement_kind.value
    row["anomaly_direction"] = metric.anomaly_direction.value
    metric_rows.append(row)

relation_rows = [asdict(item) for item in pack.relation_specs().values()]
METRIC_CATALOGUE = pd.DataFrame(metric_rows).sort_values("metric_id")
RELATION_CATALOGUE = pd.DataFrame(relation_rows).sort_values("relation_type")

display(METRIC_CATALOGUE)
display(RELATION_CATALOGUE)

assert set(METRIC_CATALOGUE["anomaly_direction"]) <= {
    "decrease", "increase", "both", "change"
}
assert "groups_ont" in set(RELATION_CATALOGUE["relation_type"])
assert set(RELATION_CATALOGUE["relation_family"]) == {
    "network_topology", "geographic_membership"
}
'''
            ),
            md(
                """
## 4. Exposure provenance — physical standard versus frozen generator

These facts must not be silently reconciled:

- Physical GPON reference: 2,488,320,000 downstream bits/s and RS(255,239),
  with 2040 transmitted and 1912 payload bits per codeword.
- Frozen generator mechanism: 2,488,000,000 bits/s and an internal divisor of 1904,
  which cancels when its FEC mean is formed.
- Frozen generator CRC mechanism: a fixed 78,000,000 bits/s reference.

The translator reports **generator opportunity counts**, not a claim that
`fec_count` is a standards-defined corrected-codeword counter. Constant exposure is
validated by formula and dimensions, not by variance.
"""
            ),
            code(
                r'''
EXPOSURE_PROVENANCE = {
    "physical_reference": {
        "standard": "ITU-T G.984.3",
        "downstream_line_rate_bps": 2_488_320_000,
        "reed_solomon": "RS(255,239)",
        "transmitted_bits_per_codeword": 2040,
        "payload_bits_per_codeword": 1912,
    },
    "frozen_generator": {
        "release": "telemetry-synth-4.0.1",
        "fec_line_rate_bps": 2_488_000_000,
        "fec_internal_divisor": 1904,
        "crc_reference_rate_bps": 78_000_000,
        "interpretation": (
            "opportunity counts under the generator mechanism; "
            "not a standards-defined corrected-codeword counter"
        ),
    },
    "translation_rule": {
        "fec_exposure": "2_488_000_000 * cadence_seconds",
        "crc_exposure": "78_000_000 * cadence_seconds",
        "mismatch_policy": "report; do not reconcile",
    },
}
print(json.dumps(EXPOSURE_PROVENANCE, indent=2))
'''
            ),
            md(
                """
## 5. Boundary and dependency checks

The generic contract contains no telecom branch. The pack contains no truth fields.
Only an adapter may import both the observable and evaluation contracts.
"""
            ),
            code(
                r'''
import ast

generic_text = "\n".join(
    path.read_text(encoding="utf-8").lower()
    for path in (PYTHON_SOURCE_ROOT / "telemetry_contract").rglob("*")
    if path.is_file() and path.suffix in {".py", ".json"}
)
forbidden_sector_terms = ("telecom", "telco", "gpon", "ont_id", "rx_power")
assert not [term for term in forbidden_sector_terms if term in generic_text]

pack_text = "\n".join(
    path.read_text(encoding="utf-8").lower()
    for path in (PYTHON_SOURCE_ROOT / "telemetry_packs").rglob("*")
    if path.is_file() and path.suffix in {".py", ".json"}
)
assert "gt_" not in pack_text

for package in ("telemetry_contract", "telemetry_packs", "telemetry_runtime"):
    imports = []
    for path in (PYTHON_SOURCE_ROOT / package).rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
    assert not any(name.startswith("telemetry_eval_contract") for name in imports), (
        package, imports
    )

print("Boundary checks passed.")
'''
            ),
            md("## 6. Write the immutable contract manifest"),
            code(
                r'''
LEGACY_BASELINE = runtime_manifest["legacy_baseline_descriptor"]
LEGACY_ATTACHMENT_MANIFEST = runtime_manifest["legacy_attachment_manifest"]
CONTRACT_MANIFEST = {
    "contract_tag": CONTRACT_TAG,
    "spec_core_version": CONTRACT_VERSION,
    "spec_eval_version": EVAL_CONTRACT_VERSION,
    "telecom_pack_version": pack.pack_version,
    "source_hashes": SOURCE_HASHES,
    "spec_core_tables": SPEC_CORE_TABLES,
    "spec_eval_tables": SPEC_EVAL_TABLES,
    "exposure_provenance": EXPOSURE_PROVENANCE,
    "legacy_baseline_descriptor": LEGACY_BASELINE,
    "legacy_attachment_manifest": LEGACY_ATTACHMENT_MANIFEST,
}
manifest_text = json.dumps(CONTRACT_MANIFEST, indent=2, sort_keys=True) + "\n"
status = write_immutable_text(CONTRACT_ROOT / "contract_manifest.json", manifest_text)
print(status, CONTRACT_ROOT / "contract_manifest.json")
print("Notebook 02 complete.")
'''
            ),
        ],
    )


def build_notebook_03():
    return notebook(
        "03_M1_V03_TELECOM_TRANSLATOR.ipynb",
        [
            md(
                """
# Notebook 03 — Telecom translator and materialisation

This notebook translates the native synthetic telecom dataset into two physically
separable outputs:

- `SPEC-CORE/`: runtime-safe canonical telemetry and context.
- `SPEC-EVAL/`: fault, cause, ticket and gap truth.

The output path includes the contract version and a run ID, so a later contract
revision cannot overwrite an earlier translation.
"""
            ),
            code(IMPORT_CONTRACT),
            md(
                """
## 1. Configuration

Leave the sampling controls empty for the full panel. For a development run, set a
time interval and/or a list of ONT IDs. Row-count truncation is deliberately not
supported because it produces partial entity histories.
"""
            ),
            code(
                r'''
import resource
import tracemalloc
from datetime import datetime, timezone

from telemetry_adapters import NativeSelection, SyntheticGponAdapter
from telemetry_contract import canonical_frame_hash, sha256_file

TELECOM_SOURCE = Path(
    os.environ.get("ANOMALY_DETECTION_TELECOM_SOURCE", str(DRIVE_ROOT))
).expanduser()
RUN_ID = os.environ.get("ANOMALY_DETECTION_TELECOM_RUN_ID", "telecom_full_v1")

SAMPLE_START = os.environ.get("ANOMALY_DETECTION_SAMPLE_START") or None
SAMPLE_END = os.environ.get("ANOMALY_DETECTION_SAMPLE_END") or None
ENTITY_IDS = tuple(
    value.strip()
    for value in os.environ.get("ANOMALY_DETECTION_ENTITY_IDS", "").split(",")
    if value.strip()
)
BATCH_NATIVE_ROWS = int(os.environ.get("ANOMALY_DETECTION_BATCH_ROWS", "250000"))
MEMORY_BUDGET_GIB = float(os.environ.get("ANOMALY_DETECTION_MEMORY_BUDGET_GIB", "8"))

RUN_ROOT = OUTPUT_ROOT / "telecom" / RUN_ID
CORE_OUTPUT = RUN_ROOT / "SPEC-CORE"
EVAL_OUTPUT = RUN_ROOT / "SPEC-EVAL"

if RUN_ROOT.exists():
    raise FileExistsError(
        f"Refusing to overwrite immutable run {RUN_ROOT}. Choose a new RUN_ID."
    )
if not TELECOM_SOURCE.is_dir():
    raise FileNotFoundError(f"Telecom source not found: {TELECOM_SOURCE}")

selection = NativeSelection(
    sample_start=SAMPLE_START,
    sample_end=SAMPLE_END,
    entity_ids=ENTITY_IDS,
    batch_native_rows=BATCH_NATIVE_ROWS,
)
adapter = SyntheticGponAdapter(selection=selection)
inventory = adapter.discover(TELECOM_SOURCE)
print(inventory)
if not inventory.core_ready or not inventory.evaluation_ready:
    raise ValueError(inventory.notes)
'''
            ),
            md(
                """
## 2. Translate with bounded-memory instrumentation

`tracemalloc` measures a complete configured batch through widening and canonical
hashing. Tracing every allocation across all 69 million output observations would make
the full run needlessly slow, so full-run resident memory is measured by `ru_maxrss`.
That is the process high-water mark and never falls during a notebook session. Both
are recorded, with their scopes, because neither number alone tells the whole story.
"""
            ),
            code(
                r'''
import gc
import pyarrow.parquet as pq

def ru_maxrss_gib():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux (including Colab) reports KiB.
    return raw / (1024 ** 3) if sys.platform == "darwin" else raw * 1024 / (1024 ** 3)

# Trace one complete configured batch through the memory-heavy widening path.
rss_before_probe = ru_maxrss_gib()
panel_path = adapter._native_path(
    TELECOM_SOURCE, "reference_dataset.parquet"
)
panel_file = pq.ParquetFile(panel_path)
native_columns = set(panel_file.schema_arrow.names)
metric_fields = [
    field for field in adapter._metric_mapping() if field in native_columns
]
probe_columns = ["timestamp_utc", "ont_id", *metric_fields]
tracemalloc.start()
probe_native_rows = 0
probe_canonical_rows = 0
for batch in panel_file.iter_batches(
    batch_size=BATCH_NATIVE_ROWS, columns=probe_columns
):
    selected_probe = adapter._filter_native(batch.to_pandas())
    if selected_probe.empty:
        continue
    widened_probe = adapter.adapt_telemetry(
        selected_probe, cadence_seconds=900.0
    )
    canonical_frame_hash(
        widened_probe, sort_by=["event_ts", "entity_id", "metric_id"]
    )
    probe_native_rows = len(selected_probe)
    probe_canonical_rows = len(widened_probe)
    break
if probe_native_rows == 0:
    raise ValueError("The configured selection contains no probe rows.")
_, traced_peak_bytes = tracemalloc.get_traced_memory()
tracemalloc.stop()
del selected_probe, widened_probe, panel_file
gc.collect()

rss_before_full = ru_maxrss_gib()
report = adapter.materialise(TELECOM_SOURCE, CORE_OUTPUT, EVAL_OUTPUT)
rss_after = ru_maxrss_gib()

MEMORY_REPORT = {
    "measured_at_utc": datetime.now(timezone.utc).isoformat(),
    "tracemalloc_peak_gib": traced_peak_bytes / (1024 ** 3),
    "tracemalloc_scope": (
        "one configured native batch through long-format widening and "
        "canonical content hashing"
    ),
    "tracemalloc_probe_native_rows": probe_native_rows,
    "tracemalloc_probe_canonical_rows": probe_canonical_rows,
    "ru_maxrss_before_probe_gib": rss_before_probe,
    "ru_maxrss_before_full_gib": rss_before_full,
    "ru_maxrss_after_gib": rss_after,
    "ru_maxrss_scope": "entire full or selected materialisation",
    "ru_maxrss_semantics": "process high-water mark; it never falls in this kernel",
    "budget_gib": MEMORY_BUDGET_GIB,
    "budget_pass": rss_after <= MEMORY_BUDGET_GIB,
}
(RUN_ROOT / "memory_report.json").write_text(
    json.dumps(MEMORY_REPORT, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(MEMORY_REPORT, indent=2))
if not MEMORY_REPORT["budget_pass"]:
    raise MemoryError(
        "Peak resident memory exceeded the configured budget. "
        "Use a fresh kernel to distinguish current work from an earlier high-water mark."
    )
print(json.dumps(dict(report.row_counts), indent=2, sort_keys=True))
'''
            ),
            md(
                """
## 3. Source lineage manifest

The manifest pins every input file actually discovered. `tickets.csv` is explicitly
classified as evaluation-only.
"""
            ),
            code(
                r'''
source_files = []
for relative_path in inventory.tables:
    path = TELECOM_SOURCE / relative_path
    source_files.append(
        {
            "path": relative_path,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "boundary": (
                "SPEC-EVAL only"
                if path.name == "tickets.csv"
                or path.name.startswith("gt_")
                or path.name == "fault_entity_intervals.csv"
                else "SPEC-CORE input"
            ),
        }
    )
SOURCE_MANIFEST = {
    "source_format": inventory.source_format,
    "source_version": inventory.source_version,
    "selection": {
        "sample_start": SAMPLE_START,
        "sample_end": SAMPLE_END,
        "entity_ids": list(ENTITY_IDS),
        "batch_native_rows": BATCH_NATIVE_ROWS,
    },
    "files": source_files,
}
(RUN_ROOT / "source_manifest.json").write_text(
    json.dumps(SOURCE_MANIFEST, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print("Pinned", len(source_files), "source files.")
'''
            ),
            md(
                """
## 4. Acceptance checks on values and table boundaries

This distinguishes:

1. a present row with a null value (`quality_code = invalid`);
2. an expected observation that is absent (`collection_gaps`);
3. an entity outside its valid service window (not a collection gap).

FEC values at the generator ceiling are marked `clipped`, not `measured`.
"""
            ),
            code(
                r'''
import pandas as pd

core_tables = {
    path.stem for path in CORE_OUTPUT.glob("*.parquet")
} | ({"telemetry"} if (CORE_OUTPUT / "telemetry").is_dir() else set())
eval_tables = {path.stem for path in EVAL_OUTPUT.glob("*.parquet")}

assert "entity_service_windows" not in core_tables
assert "gt_ticket_links" in eval_tables
assert not any(name.startswith("gt_") for name in core_tables)

quality_counts = {}
exposure_values = {
    "telecom.link.fec_count": set(),
    "telecom.link.crc_errors": set(),
}
for part in sorted((CORE_OUTPUT / "telemetry").glob("part-*.parquet")):
    frame = pd.read_parquet(
        part, columns=["metric_id", "value", "quality_code", "exposure"]
    )
    for quality_code, count in frame["quality_code"].value_counts().items():
        quality_counts[str(quality_code)] = quality_counts.get(str(quality_code), 0) + int(count)
    for metric_id in exposure_values:
        values = frame.loc[frame["metric_id"].eq(metric_id), "exposure"].dropna().unique()
        exposure_values[metric_id].update(map(float, values))
    fec = frame.loc[frame["metric_id"].eq("telecom.link.fec_count")]
    assert fec.loc[fec["value"].eq(5_000_000), "quality_code"].eq("clipped").all()
    assert frame.loc[frame["value"].isna(), "quality_code"].eq("invalid").all()

core_manifest = json.loads((CORE_OUTPUT / "manifest.json").read_text(encoding="utf-8"))
cadence = float(core_manifest["cadence_seconds"])
assert exposure_values["telecom.link.fec_count"] == {2_488_000_000.0 * cadence}
assert exposure_values["telecom.link.crc_errors"] == {78_000_000.0 * cadence}

CHECK_REPORT = {
    "core_tables": sorted(core_tables),
    "eval_tables": sorted(eval_tables),
    "quality_counts": quality_counts,
    "exposure_unique_values": {
        key: sorted(values) for key, values in exposure_values.items()
    },
    "fec_constant_exposure_validated_by": "generator formula and dimensions",
    "tickets_boundary": "SPEC-EVAL only",
    "service_validity_storage": "entity_registry.valid_from/valid_to only",
}
(RUN_ROOT / "translation_checks.json").write_text(
    json.dumps(CHECK_REPORT, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(CHECK_REPORT, indent=2))
print("Notebook 03 complete:", RUN_ROOT)
'''
            ),
        ],
    )


def build_notebook_04():
    return notebook(
        "04_M1_V03_LOCK_AND_ACCEPTANCE_TESTS.ipynb",
        [
            md(
                """
# Notebook 04 — Ground-truth lock and Milestone 1 acceptance tests

This notebook proves the boundary rather than relying on directory names.

**Test 2, translator invariance, is the primary leakage proof and the Week 1 exit
criterion.** Runtime isolation is useful deployment evidence, but it runs against a
placeholder baseline detector that cannot establish translator safety by itself.
"""
            ),
            code(IMPORT_CONTRACT),
            code(
                r'''
import ast
import hashlib
import shutil
import tempfile

import pandas as pd
from pandas.testing import assert_frame_equal

from telemetry_adapters import NativeSelection, SyntheticGponAdapter
from telemetry_contract import canonical_frame_hash
from telemetry_runtime import DetectorConfig, RobustHistoryDetector, score_core_directory

TELECOM_SOURCE = Path(
    os.environ.get("ANOMALY_DETECTION_TELECOM_SOURCE", str(DRIVE_ROOT))
).expanduser()
TELECOM_RUN_ID = os.environ.get(
    "ANOMALY_DETECTION_TELECOM_RUN_ID", "telecom_full_v1"
)
VALIDATION_ID = os.environ.get(
    "ANOMALY_DETECTION_VALIDATION_ID", "week1_acceptance_v1"
)
RUN_ROOT = OUTPUT_ROOT / "telecom" / TELECOM_RUN_ID
CORE_OUTPUT = RUN_ROOT / "SPEC-CORE"
EVAL_OUTPUT = RUN_ROOT / "SPEC-EVAL"
VALIDATION_ROOT = RUN_ROOT / "validation" / VALIDATION_ID

if not CORE_OUTPUT.is_dir() or not EVAL_OUTPUT.is_dir():
    raise FileNotFoundError("Run Notebook 03 before Notebook 04.")
if VALIDATION_ROOT.exists():
    raise FileExistsError(
        f"Refusing to overwrite {VALIDATION_ROOT}; choose a new VALIDATION_ID."
    )
VALIDATION_ROOT.mkdir(parents=True)

RESULTS = {}
'''
            ),
            md(
                """
## Test 1 — Negative control

A deliberately leaky scorer searches for sibling `SPEC-EVAL`. The harness must allow
it to run while truth is mounted and must break it when truth is removed. This proves
the test is capable of detecting the relevant leak.
"""
            ),
            code(
                r'''
def deliberately_leaky_scorer(core_directory):
    eval_manifest = Path(core_directory).parent / "SPEC-EVAL" / "manifest.json"
    payload = json.loads(eval_manifest.read_text(encoding="utf-8"))
    return sum(payload["row_counts"].values())

with tempfile.TemporaryDirectory(prefix="negative-control-") as temp:
    mount = Path(temp)
    shutil.copytree(CORE_OUTPUT, mount / "SPEC-CORE")
    shutil.copytree(EVAL_OUTPUT, mount / "SPEC-EVAL")
    mounted_value = deliberately_leaky_scorer(mount / "SPEC-CORE")
    shutil.rmtree(mount / "SPEC-EVAL")
    broke_when_removed = False
    try:
        deliberately_leaky_scorer(mount / "SPEC-CORE")
    except FileNotFoundError:
        broke_when_removed = True

assert broke_when_removed
RESULTS["negative_control"] = {
    "pass": True,
    "value_with_truth_mounted": mounted_value,
    "result_without_truth": "FileNotFoundError",
}
print(RESULTS["negative_control"])
'''
            ),
            md(
                """
## Test 2 — Translator invariance (primary exit proof)

Two native fixtures are created:

- original: native truth columns, evaluation files, and `tickets.csv` present;
- redacted: all truth columns removed, all evaluation files removed, and
  **`tickets.csv` explicitly removed**.

Both are translated independently. Canonical table content hashes—not Parquet file
bytes—must be identical.
"""
            ),
            code(
                r'''
def copy_if_present(source, destination):
    if source is not None and source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

probe_adapter = SyntheticGponAdapter()
panel_path = probe_adapter._native_path(
    TELECOM_SOURCE, "reference_dataset.parquet"
)
topology_path = probe_adapter._native_path(TELECOM_SOURCE, "topology.csv")
service_path = probe_adapter._native_path(
    TELECOM_SOURCE, "entity_service_windows.csv"
)
engineering_path = probe_adapter._native_path(
    TELECOM_SOURCE, "engineering_events.csv", required=False
)

topology = pd.read_csv(topology_path)
entity_ids = tuple(topology["ont_id"].astype(str).drop_duplicates().head(3))
panel = pd.read_parquet(
    panel_path,
    filters=[("ont_id", "in", list(entity_ids))],
)
panel["timestamp_utc"] = pd.to_datetime(panel["timestamp_utc"], utc=True)
sample_start = panel["timestamp_utc"].min()
sample_end = sample_start + pd.Timedelta(days=2)
panel = panel.loc[
    panel["timestamp_utc"].ge(sample_start)
    & panel["timestamp_utc"].lt(sample_end)
].reset_index(drop=True)
assert len(panel)

with tempfile.TemporaryDirectory(prefix="translator-invariance-") as temp:
    root = Path(temp)
    original = root / "native_original"
    redacted = root / "native_redacted"
    original.mkdir()
    redacted.mkdir()

    panel.to_parquet(original / "reference_dataset.parquet", index=False)
    panel.drop(
        columns=[column for column in panel if str(column).startswith("gt_")]
    ).to_parquet(redacted / "reference_dataset.parquet", index=False)

    topology.to_csv(original / "topology.csv", index=False)
    topology.drop(
        columns=[column for column in topology if str(column).startswith("gt_")]
    ).to_csv(redacted / "topology.csv", index=False)
    shutil.copy2(service_path, original / "entity_service_windows.csv")
    shutil.copy2(service_path, redacted / "entity_service_windows.csv")
    copy_if_present(engineering_path, original / "engineering_events.csv")
    copy_if_present(engineering_path, redacted / "engineering_events.csv")

    # Original contains every supplied evaluation input, including tickets.csv.
    evaluation_names = (
        "fault_entity_intervals.csv",
        "gt_fault_groups.csv",
        "gt_fault_registry.csv",
        "gt_benign_anomalies.csv",
        "gt_collection_gaps.parquet",
        "tickets.csv",
    )
    for name in evaluation_names:
        path = probe_adapter._native_path(
            TELECOM_SOURCE, name, evaluation=True, required=False
        )
        copy_if_present(path, original / "evaluation" / name)
    assert any(path.name == "tickets.csv" for path in original.rglob("*"))
    assert not any(path.name == "tickets.csv" for path in redacted.rglob("*"))

    selection = NativeSelection(
        sample_start=sample_start.isoformat(),
        sample_end=sample_end.isoformat(),
        entity_ids=entity_ids,
        batch_native_rows=10_000,
    )
    original_core = root / "translated_original" / "SPEC-CORE"
    redacted_core = root / "translated_redacted" / "SPEC-CORE"
    SyntheticGponAdapter(selection=selection).materialise(
        original, original_core, None
    )
    SyntheticGponAdapter(selection=selection).materialise(
        redacted, redacted_core, None
    )

    def read_canonical_bundle(core_root):
        bundle = {}
        for path in sorted(core_root.glob("*.parquet")):
            bundle[path.stem] = pd.read_parquet(path)
        parts = sorted((core_root / "telemetry").glob("part-*.parquet"))
        bundle["telemetry"] = pd.concat(
            [pd.read_parquet(path) for path in parts], ignore_index=True
        )
        return bundle

    original_bundle = read_canonical_bundle(original_core)
    redacted_bundle = read_canonical_bundle(redacted_core)
    assert set(original_bundle) == set(redacted_bundle)
    original_hashes = {}
    redacted_hashes = {}
    for table_name in sorted(original_bundle):
        sort_by = (
            ["event_ts", "entity_id", "metric_id"]
            if table_name == "telemetry"
            else None
        )
        original_hashes[table_name] = canonical_frame_hash(
            original_bundle[table_name], sort_by=sort_by
        )
        redacted_hashes[table_name] = canonical_frame_hash(
            redacted_bundle[table_name], sort_by=sort_by
        )
    assert original_hashes == redacted_hashes

    # Preserve the small core for the deployment-evidence tests below.
    invariant_core = VALIDATION_ROOT / "invariance_SPEC-CORE"
    shutil.copytree(original_core, invariant_core)

RESULTS["translator_invariance"] = {
    "pass": True,
    "role": "primary Week 1 leakage exit criterion",
    "redaction": [
        "all native gt_* columns",
        "all evaluation files",
        "tickets.csv explicitly",
    ],
    "canonical_content_hashes": original_hashes,
    "sample_entities": list(entity_ids),
    "sample_start": sample_start.isoformat(),
    "sample_end": sample_end.isoformat(),
}
print(json.dumps(RESULTS["translator_invariance"], indent=2))
'''
            ),
            md(
                """
## Test 3 — Runtime isolation (secondary deployment evidence)

The same placeholder detector is run under four mount layouts. This is useful
deployment evidence, **not the primary leakage proof**, because modelling is deferred
and the placeholder only accepts a SPEC-CORE path by construction.
"""
            ),
            code(
                r'''
detector = RobustHistoryDetector(
    DetectorConfig(history_window=12, minimum_history=4, threshold=6.0)
)

with tempfile.TemporaryDirectory(prefix="runtime-isolation-") as temp:
    root = Path(temp)
    shutil.copytree(VALIDATION_ROOT / "invariance_SPEC-CORE", root / "SPEC-CORE")
    shutil.copytree(EVAL_OUTPUT, root / "SPEC-EVAL")
    hashes = {}

    def score_hash():
        scored = score_core_directory(root / "SPEC-CORE", detector)
        return canonical_frame_hash(
            scored, sort_by=["event_ts", "entity_id", "metric_id"]
        )

    hashes["eval_mounted"] = score_hash()
    shutil.move(root / "SPEC-EVAL", root / "TRUTH_RENAMED")
    hashes["eval_renamed"] = score_hash()
    shutil.rmtree(root / "TRUTH_RENAMED")
    hashes["eval_removed"] = score_hash()
    (root / "SPEC-EVAL").mkdir()
    hashes["empty_eval_directory"] = score_hash()

assert len(set(hashes.values())) == 1
RESULTS["runtime_isolation"] = {
    "pass": True,
    "role": "secondary deployment evidence; not the primary leakage proof",
    "output_hashes": hashes,
}
print(RESULTS["runtime_isolation"])
'''
            ),
            md(
                """
## Tests 4–6 — lineage, relation model, and dependency graph

These checks verify that tickets remain evaluation-only, validity is stored once,
the geographic edge makes telecom relations non-tree, an empty relation table remains
legal for 3W, and the generic packages have no circular dependency.
"""
            ),
            code(
                r'''
lineage = json.loads(
    (CORE_OUTPUT / "translation_lineage.json").read_text(encoding="utf-8")
)
assert lineage["operational_events"]["tickets_excluded"] is True
assert lineage["entity_registry"]["validity_stored_once"] is True
assert not (CORE_OUTPUT / "entity_service_windows.parquet").exists()

relations = pd.read_parquet(CORE_OUTPUT / "entity_relations.parquet")
assert "groups_ont" in set(relations["relation_type"])
assert set(relations["relation_family"]) == {
    "network_topology", "geographic_membership"
}
child_family_counts = relations.groupby("child_entity_id")["relation_family"].nunique()
assert child_family_counts.max() >= 2

# The supplied native truth must genuinely exercise grouped faults.
native_fault_registry = pd.read_csv(
    probe_adapter._native_path(
        TELECOM_SOURCE, "gt_fault_registry.csv", evaluation=True
    )
)
native_fault_groups = pd.read_csv(
    probe_adapter._native_path(
        TELECOM_SOURCE, "gt_fault_groups.csv", evaluation=True
    )
)
linked = native_fault_registry["group_id"].dropna().astype(str)
assert len(linked) > 0
assert set(linked) <= set(native_fault_groups["group_id"].astype(str))
assert linked.value_counts().max() >= 2

def package_dependencies(source_root):
    packages = {
        "telemetry_contract",
        "telemetry_eval_contract",
        "telemetry_packs",
        "telemetry_adapters",
        "telemetry_runtime",
    }
    graph = {name: set() for name in packages}
    for package in packages:
        for path in (source_root / package).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    target = name.split(".", 1)[0]
                    if target in packages and target != package:
                        graph[package].add(target)
    return graph

graph = package_dependencies(PYTHON_SOURCE_ROOT)
assert not graph["telemetry_contract"]
assert "telemetry_eval_contract" not in graph["telemetry_packs"]
assert "telemetry_eval_contract" not in graph["telemetry_runtime"]

def has_cycle(graph):
    visiting, visited = set(), set()
    def visit(node):
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        if any(visit(child) for child in graph[node]):
            return True
        visiting.remove(node)
        visited.add(node)
        return False
    return any(visit(node) for node in graph)

assert not has_cycle(graph)
RESULTS["lineage_and_values"] = {"pass": True, "lineage": lineage}
RESULTS["relation_model"] = {
    "pass": True,
    "relation_families": sorted(set(relations["relation_family"])),
    "non_tree_exercised": True,
    "empty_relations_allowed_for_3w": True,
}
RESULTS["grouped_fault_mechanism"] = {
    "pass": True,
    "native_cause_groups": int(len(native_fault_groups)),
    "fault_events_linked_to_groups": int(len(linked)),
    "largest_group_fault_count": int(linked.value_counts().max()),
}
RESULTS["dependency_graph"] = {
    "pass": True,
    "graph": {key: sorted(value) for key, value in graph.items()},
    "cycle": False,
}
'''
            ),
            md(
                """
## Legacy baseline ownership

The supplied generator release and methodology descriptor are frozen by hashes.
No executable legacy detector pipeline was supplied, so the report must not claim
that an unavailable detector is reproducible.
"""
            ),
            code(
                r'''
contract_manifest = json.loads(
    (CONTRACT_ROOT / "contract_manifest.json").read_text(encoding="utf-8")
)
legacy = contract_manifest["legacy_baseline_descriptor"]
attachments = contract_manifest["legacy_attachment_manifest"]
assert legacy["immutable_fixture_release"]["sha256"] == next(
    asset["sha256"]
    for asset in attachments["assets"]
    if asset["role"] == "immutable_generator_release"
)
RESULTS["legacy_baseline"] = {
    "pass": True,
    "owned_by": "Notebook 04",
    "frozen_generator_release": legacy["immutable_fixture_release"],
    "detector_pipeline_status": legacy["status"],
    "qualification": legacy["reason"],
}

assert all(result["pass"] for result in RESULTS.values())
(VALIDATION_ROOT / "week1_acceptance_report.json").write_text(
    json.dumps(RESULTS, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
display(pd.DataFrame(
    [{"test": name, "pass": result["pass"], "role": result.get("role", "")}
     for name, result in RESULTS.items()]
))
print("Notebook 04 complete:", VALIDATION_ROOT)
'''
            ),
        ],
    )


def build_notebook_05():
    return notebook(
        "05_M1_V03_PETROBRAS_3W_CONTRACT_CHALLENGE.ipynb",
        [
            md(
                """
# Notebook 05 — Petrobras 3W sector-agnostic contract challenge

This notebook asks the question that one telecom fixture cannot answer:

> Can a second sector be represented by a new pack and adapter without changing
> SPEC-CORE or adding telecom-specific branches?

It uses Petrobras 3W 2.0.0 from Drive. It has no public-download fallback.
The selection is the **smallest deterministic subset satisfying all criteria**, not
a fixed number of instances.
"""
            ),
            code(IMPORT_CONTRACT),
            code(
                r'''
import hashlib
import resource
import tracemalloc
from datetime import datetime, timezone

import pandas as pd

from telemetry_adapters import ThreeWAdapter
from telemetry_contract import sha256_file

THREEW_SOURCE = Path(
    os.environ.get(
        "ANOMALY_DETECTION_3W_SOURCE",
        str(
            DRIVE_ROOT
            / "sources"
            / "petrobras_3w"
            / "2.0.0"
            / "raw"
            / "3w_dataset_2.0.0"
        ),
    )
).expanduser()
RUN_ID = os.environ.get("ANOMALY_DETECTION_3W_RUN_ID", "threew_contract_challenge_v1")
HASH_FULL_SOURCE = os.environ.get(
    "ANOMALY_DETECTION_HASH_FULL_3W_SOURCE", "1"
) not in {"0", "false", "False"}

RUN_ROOT = OUTPUT_ROOT / "petrobras_3w" / RUN_ID
CORE_OUTPUT = RUN_ROOT / "SPEC-CORE"
EVAL_OUTPUT = RUN_ROOT / "SPEC-EVAL"
FIXTURE_OUTPUT = (
    DRIVE_ROOT
    / "fixtures"
    / "petrobras_3w"
    / "2.0.0"
    / CONTRACT_TAG
    / RUN_ID
)

if not THREEW_SOURCE.is_dir():
    raise FileNotFoundError(
        f"3W source not found: {THREEW_SOURCE}\n"
        "Expected Drive path: anomaly_detection/sources/petrobras_3w/"
        "2.0.0/raw/3w_dataset_2.0.0"
    )
if RUN_ROOT.exists() or FIXTURE_OUTPUT.exists():
    raise FileExistsError("Choose a new RUN_ID; immutable output already exists.")

adapter = ThreeWAdapter()
inventory = adapter.discover(THREEW_SOURCE)
print(inventory.source_version, len(inventory.tables), inventory.notes)
assert inventory.core_ready and inventory.evaluation_ready
assert inventory.source_version == "2.0.0"
assert len(inventory.tables) == 2228
'''
            ),
            md(
                """
## 1. Pin the source

By default all 2,228 Parquet files are SHA-256 hashed. For a quick local smoke test
only, set `ANOMALY_DETECTION_HASH_FULL_3W_SOURCE=0`; the selected fixture files are
always hashed regardless.
"""
            ),
            code(
                r'''
source_records = []
paths_to_hash = (
    [THREEW_SOURCE / relative for relative in inventory.tables]
    if HASH_FULL_SOURCE
    else []
)
for path in paths_to_hash:
    source_records.append(
        {
            "path": str(path.relative_to(THREEW_SOURCE)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    )
for metadata_name in ("dataset.ini", "LICENSE-CC-BY", "README.md"):
    path = THREEW_SOURCE / metadata_name
    source_records.append(
        {
            "path": metadata_name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    )
print("Hashed source files:", len(source_records))
'''
            ),
            md(
                """
## 2. Select the smallest subset satisfying every challenge criterion

The subset must include normal data, a transient event, a persistent condition,
a state transition, a missing/frozen measurement case, and at least three distinct
real wells. The adapter searches deterministic Parquet-metadata candidates and
minimises the number of instances first, then total bytes.
"""
            ),
            code(
                r'''
selected = adapter.select_minimal_subset(THREEW_SOURCE)
selection_table = pd.DataFrame(
    [
        {
            "path": item.relative_path,
            "entity_id": item.entity_id,
            "event_code": item.event_code,
            "rows": item.rows,
            "bytes": item.bytes,
            "coverage": ", ".join(sorted(item.coverage)),
        }
        for item in selected
    ]
)
display(selection_table)
assert len({item.entity_id for item in selected}) >= 3
assert all(item.source_kind == "real" for item in selected)
'''
            ),
            md(
                """
## 3. Translate and measure memory

No manifold topology, shared cause, ticket, or severity is invented. Native `class`
and `state` labels go only to SPEC-EVAL.
"""
            ),
            code(
                r'''
def ru_maxrss_gib():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 ** 3) if sys.platform == "darwin" else raw * 1024 / (1024 ** 3)

rss_before = ru_maxrss_gib()
tracemalloc.start()
report = adapter.materialise(
    THREEW_SOURCE,
    CORE_OUTPUT,
    EVAL_OUTPUT,
    fixture_destination=FIXTURE_OUTPUT,
)
_, traced_peak_bytes = tracemalloc.get_traced_memory()
tracemalloc.stop()
rss_after = ru_maxrss_gib()

memory_report = {
    "tracemalloc_peak_gib": traced_peak_bytes / (1024 ** 3),
    "ru_maxrss_before_gib": rss_before,
    "ru_maxrss_after_gib": rss_after,
    "ru_maxrss_semantics": "process high-water mark; never falls in this kernel",
}
(RUN_ROOT / "memory_report.json").write_text(
    json.dumps(memory_report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(dict(report.row_counts), indent=2, sort_keys=True))
print(json.dumps(memory_report, indent=2))
'''
            ),
            md(
                """
## 4. Inspect the cross-sector finding

`gt_condition_states` is the one contract addition forced by the second sector.
`severity_ordinal` remains absent because 3W supplies class/state codes, not graded
component severity.
"""
            ),
            code(
                r'''
condition_states = pd.read_parquet(
    EVAL_OUTPUT / "gt_condition_states.parquet"
)
relations = pd.read_parquet(CORE_OUTPUT / "entity_relations.parquet")
fit_report = json.loads(
    (RUN_ROOT / "threew_contract_fit_report.json").read_text(encoding="utf-8")
)
assert len(condition_states) > 0
assert "severity_ordinal" not in condition_states.columns
assert relations.empty
import pyarrow.parquet as pq
telemetry_columns = set()
for path in (CORE_OUTPUT / "telemetry").glob("part-*.parquet"):
    telemetry_columns.update(pq.ParquetFile(path).schema_arrow.names)
assert "class" not in telemetry_columns
assert "state" not in telemetry_columns
assert not any(str(column).startswith("gt_") for column in telemetry_columns)
display(condition_states)
print(json.dumps(fit_report, indent=2))
'''
            ),
            md("## 5. Write the versioned source and challenge manifest"),
            code(
                r'''
selected_manifest = json.loads(
    (FIXTURE_OUTPUT / "SUBSET_MANIFEST.json").read_text(encoding="utf-8")
)
challenge_manifest = {
    "contract_tag": CONTRACT_TAG,
    "source_version": inventory.source_version,
    "full_source_hashing_enabled": HASH_FULL_SOURCE,
    "source_files": source_records,
    "selected_fixture": selected_manifest,
    "contract_fit_report": fit_report,
    "truth_boundary": {
        "native_truth_fields": ["class", "state"],
        "destination": "SPEC-EVAL only",
    },
}
(RUN_ROOT / "source_and_challenge_manifest.json").write_text(
    json.dumps(challenge_manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print("Notebook 05 complete:", RUN_ROOT)
print("Pinned fixture:", FIXTURE_OUTPUT)
'''
            ),
        ],
    )


def build_readme():
    return """# Milestone 1 v0.3 — Google Drive notebooks

Run in this order:

1. `02_M1_V03_CONTRACT_AND_PACK.ipynb`
2. `03_M1_V03_TELECOM_TRANSLATOR.ipynb`
3. `04_M1_V03_LOCK_AND_ACCEPTANCE_TESTS.ipynb`
4. `05_M1_V03_PETROBRAS_3W_CONTRACT_CHALLENGE.ipynb`

Expected Drive data:

```text
MyDrive/anomaly_detection/
├── bootstrap/
│   └── M1_v0_3_runtime_bundle.zip
├── reference_dataset.parquet
├── topology.csv
├── entity_service_windows.csv
├── engineering_events.csv
├── tickets.csv
├── evaluation/                     # the supplied name may contain trailing space
│   └── ...
└── sources/
    └── petrobras_3w/
        └── 2.0.0/
            └── raw/
                └── 3w_dataset_2.0.0/
                    ├── 0/ ... 9/
                    ├── folds/
                    ├── dataset.ini
                    ├── LICENSE-CC-BY
                    └── README.md
```

Outputs are immutable and contract-versioned:

```text
MyDrive/anomaly_detection/
├── contracts/v0.3/
├── outputs/milestone_1/v0.3/
│   ├── telecom/<run_id>/{SPEC-CORE,SPEC-EVAL,validation}/
│   └── petrobras_3w/<run_id>/{SPEC-CORE,SPEC-EVAL}/
└── fixtures/petrobras_3w/2.0.0/v0.3/<run_id>/
```

Choose a new run ID to rerun. Existing artifacts are never overwritten.

Before running Notebook 02, copy the included `M1_v0_3_runtime_bundle.zip` file to
`MyDrive/anomaly_detection/bootstrap/`. It is kept separate so the notebook remains
readable and the implementation remains independently hash-verifiable.
"""


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    payload = source_payload()
    runtime_bundle_hash, _ = build_runtime_bundle(payload)
    documents = {
        "02_M1_V03_CONTRACT_AND_PACK.ipynb": build_notebook_02(
            runtime_bundle_hash
        ),
        "03_M1_V03_TELECOM_TRANSLATOR.ipynb": build_notebook_03(),
        "04_M1_V03_LOCK_AND_ACCEPTANCE_TESTS.ipynb": build_notebook_04(),
        "05_M1_V03_PETROBRAS_3W_CONTRACT_CHALLENGE.ipynb": build_notebook_05(),
    }
    for filename, document in documents.items():
        nbf.write(document, OUTPUT / filename)
    (OUTPUT / "README.md").write_text(build_readme(), encoding="utf-8")
    print(f"Wrote {len(documents)} notebooks to {OUTPUT}")


if __name__ == "__main__":
    main()
