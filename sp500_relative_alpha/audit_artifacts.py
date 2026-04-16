from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd

ARTIFACT_NAMES = (
    "universe_proxy_manifest",
    "trading_calendar_manifest",
    "symbol_day_audit_table",
    "daily_breadth_summary",
    "symbol_coverage_summary",
    "spy_coverage_summary",
    "adjustment_support_summary",
    "sample_boundary_decision",
)


@dataclass(frozen=True)
class AuditArtifactContext:
    base_dir: Path
    audit_run_id: str
    data_snapshot_id: str
    preregistration_id: str = "SP500RA-V1-R1"
    file_format: str = "csv"

    def artifact_root(self) -> Path:
        return (
            Path(self.base_dir)
            / self.preregistration_id
            / self.data_snapshot_id
            / self.audit_run_id
        )

    def artifact_path(self, artifact_name: str) -> Path:
        if artifact_name not in ARTIFACT_NAMES:
            raise ValueError(
                f"Unknown artifact_name={artifact_name!r}. Expected one of {ARTIFACT_NAMES}."
            )
        if self.file_format != "csv":
            raise ValueError(
                f"Unsupported file_format={self.file_format!r}. v1 materialization uses csv only."
            )
        return self.artifact_root() / f"{artifact_name}.{self.file_format}"


def materialize_output_bundle(
    outputs: Mapping[str, pd.DataFrame],
    context: AuditArtifactContext,
) -> dict[str, Path]:
    """Write audit outputs to a deterministic artifact layout.

    v1 intentionally uses `csv` for the first executable scaffold because it
    avoids optional parquet dependencies. The object graph and schema are still
    format-agnostic, so a later L4 revision can swap the sink without changing
    the audit logic.
    """

    root = context.artifact_root()
    root.mkdir(parents=True, exist_ok=True)

    artifact_paths: dict[str, Path] = {}
    for artifact_name in ARTIFACT_NAMES:
        if artifact_name not in outputs:
            raise ValueError(
                f"Missing required output artifact {artifact_name!r} for materialization."
            )
        path = context.artifact_path(artifact_name)
        outputs[artifact_name].to_csv(path, index=False)
        artifact_paths[artifact_name] = path

    manifest = {
        "audit_run_id": context.audit_run_id,
        "data_snapshot_id": context.data_snapshot_id,
        "preregistration_id": context.preregistration_id,
        "file_format": context.file_format,
        "artifacts": {
            name: str(path.relative_to(root)) for name, path in artifact_paths.items()
        },
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    artifact_paths["manifest"] = manifest_path
    return artifact_paths
