from datetime import datetime

from pydantic import BaseModel, ConfigDict


class VersionCreate(BaseModel):
    description: str


class VersionResponse(BaseModel):
    id: int
    project_id: int
    version_number: int
    git_commit: str
    dvc_state: dict | None
    description: str | None
    ml_run_id: int | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)