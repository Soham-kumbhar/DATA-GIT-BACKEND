from typing import Any

from pydantic import BaseModel, Field


class MLRunComparison(BaseModel):
    run_id_before: int | None = None
    run_id_after: int | None = None

    model_name_before: str | None = None
    model_name_after: str | None = None

    features_before: list[str] = Field(
        default_factory=list
    )

    features_after: list[str] = Field(
        default_factory=list
    )

    features_added: list[str] = Field(
        default_factory=list
    )

    features_removed: list[str] = Field(
        default_factory=list
    )

    parameters_before: dict[str, Any] = Field(
        default_factory=dict
    )

    parameters_after: dict[str, Any] = Field(
        default_factory=dict
    )

    parameter_changes: dict[str, Any] = Field(
        default_factory=dict
    )

    performance_before: dict[str, Any] = Field(
        default_factory=dict
    )

    performance_after: dict[str, Any] = Field(
        default_factory=dict
    )

    performance_changes: dict[str, Any] = Field(
        default_factory=dict
    )

    other_metrics_before: dict[str, Any] = Field(
        default_factory=dict
    )

    other_metrics_after: dict[str, Any] = Field(
        default_factory=dict
    )

    other_metric_changes: dict[str, Any] = Field(
        default_factory=dict
    )

    # Evaluation evidence comparison
    evaluation: dict[str, Any] | None = None


class PerformanceComparison(BaseModel):
    metrics_before: dict[str, Any] = Field(
        default_factory=dict
    )

    metrics_after: dict[str, Any] = Field(
        default_factory=dict
    )

    metric_changes: dict[str, Any] = Field(
        default_factory=dict
    )

    performance_changed: bool = False


class VersionComparisonResponse(BaseModel):
    project_id: int

    version_1: int
    version_2: int

    git_changed: bool
    dvc_changed: bool
    code_changed: bool

    git_commit_before: str
    git_commit_after: str

    dvc_state_before: dict[str, Any] | None = None
    dvc_state_after: dict[str, Any] | None = None

    changed_files: list[str] = Field(
        default_factory=list
    )

    code_changed_files: list[str] = Field(
        default_factory=list
    )

    code_patch: str = ""

    ml_run_before: dict[str, Any] | None = None
    ml_run_after: dict[str, Any] | None = None

    ml_comparison: MLRunComparison | None = None

    performance: PerformanceComparison | None = None

    dataset_diff: dict[str, Any] | None = None

    dataset_profiles: dict[str, Any] | None = None

    dataset_analysis: dict[str, Any] | None = None

    evidence_chain: list[str] = Field(
        default_factory=list
    )

    changes: list[str] = Field(
        default_factory=list
    )