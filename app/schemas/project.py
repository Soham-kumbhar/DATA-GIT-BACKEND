from datetime import datetime

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class ProjectCreate(BaseModel):
    name: str = Field(
        min_length=1,
        max_length=200,
    )

    path: str | None = Field(
        default=None,
        max_length=1000,
    )

    description: str | None = None


class ClaimLegacyProjectRequest(BaseModel):
    path: str = Field(
        min_length=1,
        max_length=1000,
    )


class ProjectResponse(BaseModel):
    # Internal database primary key.
    id: int

    # User-facing project number.
    project_number: int

    name: str
    path: str
    description: str | None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
    )