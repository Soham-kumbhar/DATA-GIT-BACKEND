import re
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator


class VersionResultEvidenceCreate(BaseModel):
    model_name: str | None = None
    model_path: str | None = None
    model_sha256: str | None = None

    framework: str | None = None
    framework_version: str | None = None

    metrics: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None

    notes: str | None = None

    @field_validator("model_sha256")
    @classmethod
    def validate_model_sha256(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        value = value.strip().lower()

        if not re.fullmatch(
            r"[0-9a-f]{64}",
            value,
        ):
            raise ValueError(
                "model_sha256 must be a 64-character SHA-256 hex digest."
            )

        return value


class VersionResultEvidenceResponse(BaseModel):
    id: int
    version_id: int

    model_name: str | None
    model_path: str | None
    model_sha256: str | None

    framework: str | None
    framework_version: str | None

    metrics: dict[str, Any] | None
    evaluation: dict[str, Any] | None

    notes: str | None

    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
    )