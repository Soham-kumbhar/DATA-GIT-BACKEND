from typing import Any

from pydantic import BaseModel, Field


class PreparationSelectionRequest(BaseModel):
    filename: str = Field(min_length=1)

    operations: list[str] = Field(
        default_factory=list,
        description="Preparation operations selected by the user.",
    )


class PreparationSelectionResponse(BaseModel):
    filename: str
    selected_operations: list[str]
    operation_count: int
    status: str
    message: str


class PreparationConfigurationRequest(BaseModel):
    filename: str = Field(min_length=1)

    operations: list[dict[str, Any]] = Field(
        min_length=1,
        description="Configured preparation operations.",
    )


class PreparationConfigurationResponse(BaseModel):
    filename: str
    operations: list[dict[str, Any]]
    operation_count: int
    status: str
    message: str


class PreparationProcessRequest(BaseModel):
    filename: str = Field(min_length=1)

    output_format: str = Field(
        default="same",
        description="Output format: same, csv, or json.",
    )

    operations: list[dict[str, Any]] = Field(
        min_length=1,
        description="Configured preparation operations.",
    )


class PreparationProceedRequest(BaseModel):
    filename: str = Field(
        min_length=1,
        description="Uploaded dataset filename.",
    )

    output_format: str = Field(
        default="csv",
        description="Output format: csv or json.",
    )

    operations: list[dict[str, Any]] = Field(
        min_length=1,
        description=(
            "Complete list of user-selected and configured "
            "preparation operations."
        ),
    )


class PreparationReportRequest(BaseModel):
    filename: str = Field(
        min_length=1,
        description="Original uploaded dataset filename.",
    )

    output_file: str = Field(
        min_length=1,
        description="Prepared dataset filename.",
    )

    operations: list[dict[str, Any]] = Field(
        min_length=1,
        description="Preparation operations used for the run.",
    )


class PreparationMissingValuesRequest(BaseModel):
    filename: str = Field(
        min_length=1,
        description="Uploaded CSV dataset filename.",
    )

    columns: dict[str, str] = Field(
        min_length=1,
        description=(
            "Mapping of column names to missing-value strategies."
        ),
    )

    constant_values: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Constant replacement values used when strategy is "
            "'constant'."
        ),
    )