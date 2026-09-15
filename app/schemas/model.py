from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ModelCreate(BaseModel):
    name: str = Field(
        min_length=1,
        max_length=200,
    )

    path: str = Field(
        min_length=1,
        max_length=1000,
    )


class ModelResponse(BaseModel):
    id: int
    project_id: int
    name: str
    path: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)