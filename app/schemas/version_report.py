from typing import Any

from pydantic import BaseModel, Field


class ReportSection(BaseModel):
    title: str
    summary: str
    details: dict[str, Any] = Field(
        default_factory=dict
    )


class VersionReportResponse(BaseModel):
    project_id: int

    version_1: int
    version_2: int

    executive_summary: ReportSection

    version_overview: ReportSection

    what_changed: ReportSection

    dataset_analysis: ReportSection

    feature_analysis: ReportSection

    code_analysis: ReportSection

    model_parameters: ReportSection

    performance_comparison: ReportSection

    error_analysis: ReportSection

    change_chain: ReportSection

    git_dvc_evidence: ReportSection

    ai_root_cause_analysis: ReportSection

    ai_recommendations: ReportSection

    reproducibility: ReportSection

    technical_evidence: ReportSection