from typing import Any

from pydantic import BaseModel


class VersionComparisonResponse(BaseModel):
    project_id: int
    version_1: int
    version_2: int

    git_changed: bool
    dvc_changed: bool
    code_changed: bool

    git_commit_before: str | None
    git_commit_after: str | None

    dvc_state_before: dict[str, Any] | None
    dvc_state_after: dict[str, Any] | None

    changed_files: list[str]
    code_changed_files: list[str]
    code_patch: str

    dataset_diff: dict[str, Any]
    dataset_profiles: dict[str, Any]
    dataset_analysis: dict[str, Any]

    preparation: dict[str, Any]

    evidence_chain: list[str]
    changes: list[str]

    ai_insights: dict[str, Any] | None = None