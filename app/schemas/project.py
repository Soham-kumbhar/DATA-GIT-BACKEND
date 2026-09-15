from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ProjectCreate(BaseModel):
    name: str = Field(
        min_length=1,
        max_length=200,
    )

    path: str = Field(
        min_length=1,
        max_length=1000,
    )

    description: str | None = None


class ProjectResponse(BaseModel):
    id: int
    name: str
    path: str
    description: str | None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)