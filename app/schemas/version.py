from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.version_result_evidence import (
    VersionResultEvidenceCreate,
    VersionResultEvidenceResponse,
)


class VersionCreate(BaseModel):
    description: str

    preparation_operations: list[dict] = Field(
        default_factory=list,
    )

    result_evidence: VersionResultEvidenceCreate | None = None


class VersionResponse(BaseModel):
    # ========================================================
    # INTERNAL DATABASE IDENTIFIERS
    # ========================================================

    id: int
    project_id: int

    # ========================================================
    # USER-FACING PROJECT IDENTIFIER
    # ========================================================

    project_number: int

    # ========================================================
    # USER-FACING VERSION IDENTIFIER
    #
    # Scoped to the project.
    # ========================================================

    version_number: int

    # ========================================================
    # VERSION EVIDENCE
    # ========================================================

    git_commit: str

    dvc_state: dict | None

    description: str | None

    # Kept for compatibility with existing data/API consumers.
    ml_run_id: int | None

    created_at: datetime

    result_evidence: (
        VersionResultEvidenceResponse | None
    ) = None

    model_config = ConfigDict(
        from_attributes=True,
    )