"""Read pinned NAB files; keep source order, original values and labels separate."""

from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.request import urlopen

import pandas as pd


def fetch_files(manifest: dict, cache: Path, *, download: bool = True) -> Path:
    """Cache only declared files and verify SHA-256 on every use.

    A corrupt existing file raises; it is not silently replaced. No NAB runtime
    or detector dependencies are installed. The upstream license is cached too.
    """
    if not re.fullmatch(r"[0-9a-f]{40}", manifest["upstream_commit"]):
        raise ValueError("Pin NAB to a full commit SHA")
    root = Path(cache) / manifest["upstream_commit"]
    for resource in manifest["files"]:
        relative = Path(resource["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("NAB resource paths must stay within the cache")
        target = root / relative
        if target.exists():
            content = target.read_bytes()
        else:
            if not download:
                raise FileNotFoundError(target)
            url = (
                "https://raw.githubusercontent.com/numenta/NAB/"
                f"{manifest['upstream_commit']}/{relative.as_posix()}"
            )
            with urlopen(url, timeout=60) as response:
                content = response.read()
        if sha256(content).hexdigest() != resource["sha256"]:
            raise ValueError(f"Checksum mismatch: {target}")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    return root


def load_nab(
    manifest: dict, cache: Path, *, download: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return observations and benchmark windows as separate DataFrames.

    These files have no timezone offsets. Preserve their recorded times without
    inventing UTC or localising an unknown timezone. Windows are inclusive at
    both endpoints, following NAB; they are not verified fault durations.
    """
    root = fetch_files(manifest, cache, download=download)
    windows = json.loads((root / "labels/combined_windows.json").read_text())
    frames, annotations = [], []
    for source in manifest["series"]:
        native = pd.read_csv(root / "data" / source["path"])
        if list(native.columns) != ["timestamp", "value"]:
            raise ValueError(f"Unexpected NAB columns: {source['path']}")
        times = pd.to_datetime(native.timestamp, errors="raise")
        if times.isna().any() or times.dt.tz is not None:
            raise ValueError("Expected nonmissing, timezone-unspecified NAB timestamps")
        frames.append(pd.DataFrame({
            "timestamp": times, "entity_id": source["entity_id"],
            "metric_name": source["metric_name"],
            "value": pd.to_numeric(native.value, errors="raise"),
        }))
        previous_end = None
        recorded_times = set(times)
        for number, (start, end) in enumerate(windows[source["path"]], start=1):
            start, end = pd.Timestamp(start), pd.Timestamp(end)
            if start > end or start not in recorded_times or end not in recorded_times:
                raise ValueError(f"Invalid annotation endpoints: {source['path']}")
            if previous_end is not None and start <= previous_end:
                raise ValueError(f"Overlapping annotation windows: {source['path']}")
            annotations.append({
                "entity_id": source["entity_id"], "metric_name": source["metric_name"],
                "window_id": number, "window_start": start, "window_end": end,
            })
            previous_end = end
    columns = ["entity_id", "metric_name", "window_id", "window_start", "window_end"]
    return pd.concat(frames, ignore_index=True), pd.DataFrame(annotations, columns=columns)
