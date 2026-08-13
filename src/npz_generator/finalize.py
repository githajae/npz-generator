"""Single-writer finalization after independent artifact jobs complete."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def finalize_manifest(output_root: Path, workload: str, sf: int | float) -> Path:
    dataset_root = output_root / workload / f"sf{sf}"
    sidecars = sorted(dataset_root.glob("**/*.meta.json"))
    if not sidecars:
        raise FileNotFoundError(f"no artifact sidecars found under {dataset_root}")
    artifacts = []
    snapshots: set[str] = set()
    for sidecar in sidecars:
        payload = json.loads(sidecar.read_text())
        artifact = sidecar.with_suffix("").with_suffix(".npz")
        if not artifact.is_file():
            raise FileNotFoundError(f"metadata has no NPZ artifact: {sidecar}")
        payload["path"] = artifact.relative_to(dataset_root).as_posix()
        artifacts.append(payload)
        snapshots.add(payload["snapshot_id"])
    manifest = {
        "format_version": 1,
        "workload": workload,
        "scale_factor": sf,
        "artifact_count": len(artifacts),
        "snapshot_ids": sorted(snapshots),
        "artifacts": artifacts,
    }
    output = dataset_root / "manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".manifest.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as destination:
            json.dump(manifest, destination, indent=2, sort_keys=True)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output
