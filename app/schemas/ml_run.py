from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class MLRunCreate(BaseModel):
    project_id: int
    model_name: str
    features: list[str] | None = None
    parameters: dict | None = None
    metrics: dict | None = None
    evaluation: dict[str, Any] | None = None


class MLRunResponse(BaseModel):
    id: int
    project_id: int
    git_commit: str
    dvc_state: dict | None

    model_name: str
    features: list[str] | None
    parameters: dict | None
    metrics: dict | None
    evaluation: dict[str, Any] | None

    created_at: datetime

    model_config = ConfigDict(from_attributes=True)