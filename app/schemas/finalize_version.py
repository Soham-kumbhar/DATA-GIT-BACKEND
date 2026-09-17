from datetime import datetime

from pydantic import BaseModel, Field


class FinalizeVersionRequest(BaseModel):
    message: str | None = Field(default=None, max_length=2000)
    notes: str | None = Field(default=None, max_length=5000)


class FinalizeVersionEvidence(BaseModel):
    captured_at: datetime
    git: dict
    dvc: dict
    user_context: dict
