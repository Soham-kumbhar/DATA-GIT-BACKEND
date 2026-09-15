from datetime import datetime

from pydantic import BaseModel, ConfigDict


class VersionHistoryItem(BaseModel):
    id: int
    version_number: int
    git_commit: str
    dvc_state: dict | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class VersionHistoryResponse(BaseModel):
    project_id: int
    total_versions: int
    versions: list[VersionHistoryItem]


class VersionDetailResponse(BaseModel):
    id: int
    project_id: int
    version_number: int
    git_commit: str
    dvc_state: dict | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)