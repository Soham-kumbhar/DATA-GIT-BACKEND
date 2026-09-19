from datetime import datetime
from typing import Any

from pydantic import BaseModel


class VersionReportResponse(BaseModel):
    id: int
    project_id: int
    version_number: int
    git_commit: str
    dvc_state: dict[str, Any] | None
    description: str | None
    ml_run_id: int | None
    created_at: datetime | None

    project: dict[str, Any]
    dataset: dict[str, Any]
    data_quality: dict[str, Any]
    preparation: dict[str, Any]

    # New version-centric result evidence.
    result_evidence: dict[str, Any]

    # Compatibility fields for the current frontend.
    model: dict[str, Any] | None
    training: dict[str, Any]
    performance: dict[str, Any]
    evaluation: dict[str, Any]

    git: dict[str, Any]
    dvc: dict[str, Any] | None
    lineage: dict[str, Any]
    evidence_completeness: dict[str, Any]

    report_metadata: dict[str, Any]
    ai_report: dict[str, Any] | None = None