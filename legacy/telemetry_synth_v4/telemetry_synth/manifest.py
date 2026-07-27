"""Versioned persistence, manifests and reproducibility verification.

Scope discipline: this module makes generator runs **reproducible, versioned and
discoverable**. It does not map anything to a canonical schema, and it must not. That
mapping comes later, as a separate adapter layer, once several sources exist to
generalise over. What it does instead is write down everything an adapter author will
need — a native-format description, a config, a seed, an environment capture and content
digests — so that mapping is a small job when it is time to do it.

Three things are versioned independently and all three are recorded:

  generator version   the simulator's semantics. Changing it changes the data.
  native format       the on-disk shape. Adapters bind to THIS, not to a Python version.
  dataset version     a specific run: generator version + config + seed + environment.

Reproducibility is *verified*, not asserted. `verify_reproducibility` regenerates from
the stored config and compares content digests row for row.
"""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

NATIVE_FORMAT_VERSION = "telemetry-synth native v4"

# Written by the validation or publication step, not by the generator. They travel with a
# published dataset but are not part of it, and must not be expected to regenerate.
DERIVED_FILES = {"fixture_validation_report.csv", "data_validation_report.csv",
                 "stage_01_state.json", "dataset_manifest.json"}

# The native format declaration. An adapter author reads THIS, not the generator source.
NATIVE_TABLES = {
    "reference_dataset.parquet": dict(
        role="telemetry_panel", grain="one row per entity per poll",
        note="wide panel. Observable columns are operator-visible; every column prefixed "
             "gt_ is ground truth and must never reach a feature or a model."),
    "topology.csv": dict(
        role="entity_attributes", grain="one row per ONT",
        note="static device, plant, geography and commercial attributes, plus the "
             "OLT/PON/L1/L2 parentage. gt_frailty, gt_chronic and gt_noisy_plant are "
             "ground truth."),
    "entity_service_windows.csv": dict(
        role="entity_lifecycle", grain="one row per ONT",
        note="install_ts and decommission_ts. Entities are provisioned and decommissioned "
             "inside the observation window, so any grid reconstruction must intersect "
             "with these bounds."),
    "gt_fault_registry.csv": dict(
        role="ground_truth_faults", grain="one row per fault",
        note="onset, observability anchors, impact, counterfactual impact, repair, "
             "group membership and recurrence parentage."),
    "fault_entity_intervals.csv": dict(
        role="ground_truth_faults", grain="one row per fault per affected entity",
        note="a shared fault appears once per entity beneath the failed node."),
    "gt_fault_groups.csv": dict(
        role="ground_truth_faults", grain="one row per grouped cause",
        note="storm windows and their geographic footprint."),
    "tickets.csv": dict(
        role="operational_labels", grain="one row per ticket",
        note="an OPERATIONAL channel, not ground truth: incomplete, delayed, duplicated, "
             "sometimes raised against the wrong line, and sometimes no-fault-found. "
             "Identifiers are opaque and carry no fault information."),
    "engineering_events.csv": dict(
        role="operational_context", grain="one row per planned change",
        note="firmware upgrades, plant rework, provisioning changes, planned maintenance. "
             "Known to the operator, so legitimately available to a detector."),
    "gt_benign_anomalies.csv": dict(
        role="ground_truth_negatives", grain="one row per benign excursion",
        note="labelled NON-fault excursions: sensor glitches, stuck values, transient "
             "bursts, re-ranging, CPE power cycles. Lets a false alert be attributed to a "
             "cause rather than merely counted."),
    "gt_collection_gaps.parquet": dict(
        role="ground_truth_missingness", grain="one row per missing poll",
        note="cause of every absent observation. The cause is ground truth; the absence "
             "itself is observable."),
    "parameter_provenance.csv": dict(
        role="provenance", grain="one row per material parameter",
        note="status is one of vendor_specification, engineering_estimate, "
             "uncalibrated_assumption, benchmark_tuning. Benchmark-tuned values are "
             "properties of this fixture and must never be quoted as operator facts."),
    "generator_config.json": dict(
        role="configuration", grain="single object",
        note="the complete settings object. Together with the generator version and the "
             "environment capture, this is sufficient to regenerate the dataset."),
}


# ======================================================================================
# Digests and environment
# ======================================================================================


def capture_environment() -> dict:
    """Library versions matter: the generator's output depends on NumPy's RNG stream."""
    import pyarrow
    import scipy
    return dict(
        python=sys.version.split()[0],
        platform=platform.platform(),
        numpy=np.__version__, pandas=pd.__version__,
        pyarrow=pyarrow.__version__, scipy=scipy.__version__,
    )


def _row_hashes(path: Path, batch_rows: int = 200_000):
    """Order-independent per-row hashes, streamed so a multi-million-row panel never
    has to be held in memory at once."""
    import pyarrow.parquet as pq
    if path.suffix == ".parquet":
        pf = pq.ParquetFile(path)
        cols = sorted(pf.schema_arrow.names)
        parts, n = [], 0
        for batch in pf.iter_batches(batch_size=batch_rows):
            df = batch.to_pandas().reindex(columns=cols)
            parts.append(pd.util.hash_pandas_object(df, index=False).to_numpy())
            n += len(df)
            del df
        h = np.concatenate(parts) if parts else np.zeros(0, dtype="uint64")
        return h, n, cols
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        cols = sorted(df.columns)
        df = df.reindex(columns=cols)
        return pd.util.hash_pandas_object(df, index=False).to_numpy(), len(df), cols
    return None, None, None


def content_digest(path: Path) -> tuple:
    """(content_sha256, n_rows). Hashes the DATA, not the file bytes.

    File bytes are not a fair reproducibility test: Parquet embeds writer metadata, and
    CSV float formatting is not stable across platforms. Hashing the values means the
    check answers the question that actually matters -- is this the same dataset?
    """
    h, n, cols = _row_hashes(path)
    if h is None:
        return hashlib.sha256(path.read_bytes()).hexdigest(), None
    if not n:
        return hashlib.sha256(b"empty").hexdigest(), 0
    h = np.sort(h)
    digest = hashlib.sha256(np.ascontiguousarray(h).tobytes())
    digest.update(",".join(map(str, cols)).encode())
    del h
    return digest.hexdigest(), int(n)


def file_digest(path: Path) -> dict:
    file_hash = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            file_hash.update(chunk)
    c_hash, n_rows = content_digest(path)
    return dict(file=path.name, bytes=path.stat().st_size,
                file_sha256=file_hash.hexdigest()[:32],
                content_sha256=c_hash[:32], n_rows=n_rows)


# ======================================================================================
# Dataset identity, manifest and versioned write
# ======================================================================================


def config_fingerprint(cfg) -> str:
    payload = json.dumps(asdict(cfg), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def dataset_id(cfg, generator_version: str) -> str:
    scenario = getattr(cfg, "scenario", "default")
    return (f"telco_gpon__{scenario}__{cfg.config_role}__gen{generator_version}"
            f"__seed{cfg.seed}__cfg{config_fingerprint(cfg)}")


def build_manifest(stage_dir, cfg, generator_version: str,
                   validation_report: pd.DataFrame | None = None,
                   notes: str = "") -> dict:
    stage_dir = Path(stage_dir)
    files = [file_digest(p) for p in sorted(stage_dir.iterdir())
             if p.is_file() and p.suffix in {".parquet", ".csv", ".json"}
             and p.name not in DERIVED_FILES]
    declared = set(NATIVE_TABLES)
    written = {f["file"] for f in files}
    manifest = dict(
        dataset_id=dataset_id(cfg, generator_version),
        sector="telco", domain="gpon_ftth", scenario=getattr(cfg, "scenario", "default"),
        generator_version=generator_version,
        native_format_version=NATIVE_FORMAT_VERSION,
        config_role=cfg.config_role,
        seed=int(cfg.seed),
        config_fingerprint=config_fingerprint(cfg),
        scale=dict(n_onts=int(cfg.n_onts), days=int(cfg.days),
                   sample_minutes=int(cfg.sample_minutes), start=str(cfg.start)),
        environment=capture_environment(),
        files=files,
        total_bytes=int(sum(f["bytes"] for f in files)),
        expected_files_missing=sorted(declared - written),
        undeclared_files_present=sorted(written - declared),
        notes=notes,
        created_ts=pd.Timestamp.now(tz="UTC").isoformat(),
    )
    if validation_report is not None and len(validation_report):
        manifest["fixture_validation"] = dict(
            n_checks=int(len(validation_report)),
            n_passed=int(validation_report.passed.sum()),
            all_passed=bool(validation_report.passed.all()),
            failed=validation_report.loc[~validation_report.passed, "check_name"].tolist())
    # identity of the dataset excludes anything that varies between identical runs
    stable = {k: manifest[k] for k in
              ("dataset_id", "generator_version", "native_format_version",
               "config_fingerprint", "seed", "scale", "scenario")}
    stable["content"] = {f["file"]: f["content_sha256"] for f in files}
    manifest["dataset_sha256"] = hashlib.sha256(
        json.dumps(stable, sort_keys=True).encode()).hexdigest()[:32]
    return manifest


def publish_dataset(stage_dir, datasets_root, cfg, generator_version: str,
                    validation_report: pd.DataFrame | None = None,
                    notes: str = "", overwrite: bool = False) -> dict:
    """Copy a completed run into the versioned dataset store and write its manifest.

    Published datasets are IMMUTABLE. Re-running with the same generator version, config
    and seed should produce the same bytes; if you want different data, change the seed or
    the config, which changes the dataset id. Overwriting in place is how a downstream
    result silently stops matching the data it was computed from.
    """
    stage_dir, datasets_root = Path(stage_dir), Path(datasets_root)
    manifest = build_manifest(stage_dir, cfg, generator_version, validation_report, notes)
    scenario = getattr(cfg, "scenario", "default")
    target = (datasets_root / "telco_gpon" / f"v{generator_version}"
              / f"{scenario}__{cfg.config_role}__seed{cfg.seed}"
                f"__{manifest['config_fingerprint']}")
    if target.exists():
        existing = target / "dataset_manifest.json"
        if existing.exists() and not overwrite:
            prev = json.loads(existing.read_text())
            same = prev.get("dataset_sha256") == manifest["dataset_sha256"]
            raise FileExistsError(
                f"{target} already exists and published datasets are immutable.\n"
                f"  content identical to the existing publication: {same}\n"
                f"  pass overwrite=True only if you are deliberately republishing.")
        if overwrite:
            shutil.rmtree(target)
    native = target / "native"
    native.mkdir(parents=True, exist_ok=True)
    for p in sorted(stage_dir.iterdir()):
        if p.is_file() and p.suffix in {".parquet", ".csv", ".json"} \
                and p.name not in DERIVED_FILES:
            shutil.copy2(p, native / p.name)
    (target / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))
    (target / "NATIVE_FORMAT.md").write_text(native_format_doc(generator_version))
    if validation_report is not None:
        validation_report.to_csv(target / "fixture_validation_report.csv", index=False)
    manifest["published_to"] = str(target)
    register_dataset(datasets_root, manifest)
    return manifest


def native_format_doc(generator_version: str) -> str:
    lines = [f"# Native format — `{NATIVE_FORMAT_VERSION}` (generator {generator_version})",
             "",
             "The on-disk shape of a telemetry-synth run. **Adapters bind to this document, "
             "not to the generator source.** It is versioned independently of the "
             "generator: a change in simulator semantics does not necessarily change the "
             "format, and a change in format is a breaking change for every adapter.",
             "",
             "Two rules hold across every file.",
             "",
             "1. **`gt_` prefix means ground truth.** Any column so prefixed is "
             "unobservable to a real operator and must never reach a feature, a model, or "
             "a production code path. It exists for evaluation only.",
             "2. **Tickets are not ground truth.** `tickets.csv` is an operational channel "
             "with the defects a real ticket system has. Its identifiers are opaque by "
             "design and carry no fault information.",
             "", "## Files", ""]
    for name, meta in NATIVE_TABLES.items():
        lines += [f"### `{name}`", "",
                  f"- **role** — {meta['role']}",
                  f"- **grain** — {meta['grain']}",
                  f"- {meta['note']}", ""]
    lines += ["## Regenerating a published dataset", "",
              "`dataset_manifest.json` carries the generator version, the full "
              "configuration, the seed and the library versions the run was made under. "
              "The generator's RNG streams are seeded per entity and per stage, so the "
              "same three inputs reproduce the dataset. `verify_reproducibility` performs "
              "that regeneration and compares content digests table by table.", ""]
    return "\n".join(lines)


# ======================================================================================
# Index and verification
# ======================================================================================

_INDEX_COLUMNS = ["dataset_id", "sector", "domain", "scenario", "generator_version",
                  "native_format_version", "config_role", "seed", "config_fingerprint",
                  "n_onts", "days", "sample_minutes", "total_bytes",
                  "fixture_checks_passed", "all_checks_passed", "dataset_sha256",
                  "created_ts", "path"]


def register_dataset(datasets_root, manifest: dict) -> pd.DataFrame:
    datasets_root = Path(datasets_root)
    datasets_root.mkdir(parents=True, exist_ok=True)
    index_path = datasets_root / "DATASET_INDEX.csv"
    fv = manifest.get("fixture_validation", {})
    row = {
        "dataset_id": manifest["dataset_id"], "sector": manifest["sector"],
        "domain": manifest["domain"], "scenario": manifest["scenario"],
        "generator_version": manifest["generator_version"],
        "native_format_version": manifest["native_format_version"],
        "config_role": manifest["config_role"], "seed": manifest["seed"],
        "config_fingerprint": manifest["config_fingerprint"],
        "n_onts": manifest["scale"]["n_onts"], "days": manifest["scale"]["days"],
        "sample_minutes": manifest["scale"]["sample_minutes"],
        "total_bytes": manifest["total_bytes"],
        "fixture_checks_passed": fv.get("n_passed"),
        "all_checks_passed": fv.get("all_passed"),
        "dataset_sha256": manifest["dataset_sha256"],
        "created_ts": manifest["created_ts"],
        "path": manifest.get("published_to", ""),
    }
    idx = pd.read_csv(index_path) if index_path.exists() else pd.DataFrame(columns=_INDEX_COLUMNS)
    idx = idx.loc[idx.dataset_id != row["dataset_id"]]
    idx = pd.concat([idx, pd.DataFrame([row])], ignore_index=True)
    idx = idx.reindex(columns=_INDEX_COLUMNS).sort_values("created_ts")
    idx.to_csv(index_path, index=False)
    return idx


def load_dataset_index(datasets_root) -> pd.DataFrame:
    p = Path(datasets_root) / "DATASET_INDEX.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=_INDEX_COLUMNS)


def verify_manifest(dataset_dir) -> pd.DataFrame:
    """Recompute every digest and compare against the manifest. Integrity, not identity."""
    dataset_dir = Path(dataset_dir)
    manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text())
    native = dataset_dir / "native"
    rows = []
    for rec in manifest["files"]:
        p = native / rec["file"]
        if not p.exists():
            rows.append(dict(file=rec["file"], status="MISSING", matches=False, detail=""))
            continue
        now = file_digest(p)
        ok = (now["file_sha256"] == rec["file_sha256"]
              and now["content_sha256"] == rec["content_sha256"])
        rows.append(dict(file=rec["file"], status="ok" if ok else "ALTERED", matches=ok,
                         detail="" if ok else
                         f"content {rec['content_sha256'][:8]} -> {now['content_sha256'][:8]}"))
    return pd.DataFrame(rows)


def verify_reproducibility(dataset_dir, generate_fn, settings_cls, work_dir=None,
                           ) -> pd.DataFrame:
    """Regenerate from the stored config and compare content digests table by table.

    Reproducibility is a claim about the generator, so it is tested by running the
    generator, not by trusting it. A mismatch here with a matching config fingerprint
    means either an undeclared source of randomness or a library-version dependency --
    both worth knowing before anything downstream is built on the data.
    """
    dataset_dir = Path(dataset_dir)
    manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text())
    cfg_d = json.loads((dataset_dir / "native" / "generator_config.json").read_text())
    cfg = settings_cls(**{k: (tuple(v) if isinstance(v, list) else v)
                          for k, v in cfg_d.items()})
    work_dir = Path(work_dir or (dataset_dir.parent / "_repro_check"))
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    generate_fn(cfg, out_dir=work_dir)

    env_now, env_then = capture_environment(), manifest["environment"]
    drifted = {k: (env_then.get(k), v) for k, v in env_now.items()
               if k != "platform" and env_then.get(k) != v}
    rows = []
    for rec in manifest["files"]:
        if rec["file"] == "generator_config.json" or rec["file"] in DERIVED_FILES:
            continue
        p = work_dir / rec["file"]
        if not p.exists():
            rows.append(dict(file=rec["file"], reproduced=False, detail="not regenerated"))
            continue
        c_hash, n_rows = content_digest(p)
        ok = c_hash[:32] == rec["content_sha256"]
        rows.append(dict(file=rec["file"], reproduced=ok,
                         detail="" if ok else f"rows {rec['n_rows']} -> {n_rows}"))
    out = pd.DataFrame(rows)
    out.attrs["environment_drift"] = drifted
    shutil.rmtree(work_dir, ignore_errors=True)
    return out
