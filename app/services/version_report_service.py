from typing import Any

from sqlalchemy.orm import Session

from app.services.groq_report_service import (
    GroqReportService,
)

from app.services.version_comparison_service import (
    VersionComparisonService,
)


class VersionReportService:

    # ============================================================
    # BUILD COMPLETE REPORT
    # ============================================================

    @staticmethod
    def build_report(
        db: Session,
        project_id: int,
        version_1: int,
        version_2: int,
    ) -> dict[str, Any]:

        # --------------------------------------------------------
        # BUILD DETERMINISTIC COMPARISON
        # --------------------------------------------------------

        comparison = (
            VersionComparisonService.compare_versions(
                db=db,
                project_id=project_id,
                version_1=version_1,
                version_2=version_2,
            )
        )

        # --------------------------------------------------------
        # DETERMINISTIC REPORT SECTIONS
        # --------------------------------------------------------

        executive_summary = (
            VersionReportService._build_executive_summary(
                comparison
            )
        )

        version_overview = (
            VersionReportService._build_version_overview(
                comparison
            )
        )

        what_changed = (
            VersionReportService._build_what_changed(
                comparison
            )
        )

        dataset_analysis = (
            VersionReportService._build_dataset_analysis(
                comparison
            )
        )

        feature_analysis = (
            VersionReportService._build_feature_analysis(
                comparison
            )
        )

        code_analysis = (
            VersionReportService._build_code_analysis(
                comparison
            )
        )

        model_parameters = (
            VersionReportService._build_model_parameters(
                comparison
            )
        )

        performance_comparison = (
            VersionReportService._build_performance_comparison(
                comparison
            )
        )

        error_analysis = (
            VersionReportService._build_error_analysis(
                comparison
            )
        )

        change_chain = (
            VersionReportService._build_change_chain(
                comparison
            )
        )

        git_dvc_evidence = (
            VersionReportService._build_git_dvc_evidence(
                comparison
            )
        )

        reproducibility = (
            VersionReportService._build_reproducibility(
                comparison
            )
        )

        technical_evidence = (
            VersionReportService._build_technical_evidence(
                comparison
            )
        )

        # --------------------------------------------------------
        # GROQ AI ANALYSIS
        # --------------------------------------------------------

        ai_result = (
            GroqReportService.analyze_comparison(
                comparison
            )
        )

        # --------------------------------------------------------
        # ROOT CAUSE ANALYSIS
        # --------------------------------------------------------

        if (
            ai_result.get("status")
            == "success"
        ):

            root_cause = (
                ai_result.get(
                    "root_cause",
                    {},
                )
            )

            ai_root_cause_analysis = {
                "title":
                    "AI Root-Cause Analysis",

                "summary":
                    root_cause.get(
                        "summary",
                        "No AI root-cause summary was returned.",
                    ),

                "details": {
                    "status":
                        "success",

                    "contributors":
                        root_cause.get(
                            "contributors",
                            [],
                        ),

                    "overall_confidence":
                        root_cause.get(
                            "overall_confidence"
                        ),

                    "limitations":
                        root_cause.get(
                            "limitations",
                            [],
                        ),

                    "alternative_explanations":
                        root_cause.get(
                            "alternative_explanations",
                            [],
                        ),
                },
            }

        else:

            ai_root_cause_analysis = {
                "title":
                    "AI Root-Cause Analysis",

                "summary":
                    "AI root-cause analysis could not be generated.",

                "details": {
                    "status":
                        ai_result.get(
                            "status",
                            "error",
                        ),

                    "error":
                        ai_result.get(
                            "reason",
                            "Unknown Groq error.",
                        ),

                    "contributors":
                        [],

                    "overall_confidence":
                        None,

                    "limitations":
                        [
                            "Groq analysis was unavailable."
                        ],

                    "alternative_explanations":
                        [],
                },
            }

        # --------------------------------------------------------
        # AI RECOMMENDATIONS
        # --------------------------------------------------------

        recommendations = (
            ai_result.get(
                "recommendations",
                []
            )
            if ai_result.get(
                "status"
            ) == "success"
            else []
        )

        if (
            ai_result.get("status")
            == "success"
        ):

            ai_recommendations = {
                "title":
                    "AI Recommendations",

                "summary":
                    (
                        f"{len(recommendations)} AI "
                        "recommendation(s) generated "
                        "from deterministic evidence."
                    ),

                "details": {
                    "status":
                        "success",

                    "recommendations":
                        recommendations,
                },
            }

        else:

            ai_recommendations = {
                "title":
                    "AI Recommendations",

                "summary":
                    "AI recommendations could not be generated.",

                "details": {
                    "status":
                        ai_result.get(
                            "status",
                            "error",
                        ),

                    "recommendations":
                        [],

                    "error":
                        ai_result.get(
                            "reason",
                            "Unknown Groq error.",
                        ),
                },
            }

        # --------------------------------------------------------
        # FINAL REPORT
        # --------------------------------------------------------

        return {
            "project_id":
                project_id,

            "version_1":
                version_1,

            "version_2":
                version_2,

            "executive_summary":
                executive_summary,

            "version_overview":
                version_overview,

            "what_changed":
                what_changed,

            "dataset_analysis":
                dataset_analysis,

            "feature_analysis":
                feature_analysis,

            "code_analysis":
                code_analysis,

            "model_parameters":
                model_parameters,

            "performance_comparison":
                performance_comparison,

            "error_analysis":
                error_analysis,

            "change_chain":
                change_chain,

            "git_dvc_evidence":
                git_dvc_evidence,

            "ai_root_cause_analysis":
                ai_root_cause_analysis,

            "ai_recommendations":
                ai_recommendations,

            "reproducibility":
                reproducibility,

            "technical_evidence":
                technical_evidence,
        }

    # ============================================================
    # EXECUTIVE SUMMARY
    # ============================================================

    @staticmethod
    def _build_executive_summary(
        comparison: dict[str, Any],
    ) -> dict[str, Any]:

        dataset_analysis = (
            comparison.get(
                "dataset_analysis"
            )
            or {}
        )

        performance = (
            comparison.get(
                "performance"
            )
            or {}
        )

        code_changed = comparison.get(
            "code_changed",
            False,
        )

        performance_changed = performance.get(
            "performance_changed",
            False,
        )

        row_delta = dataset_analysis.get(
            "row_delta",
            0,
        )

        if performance_changed:

            performance_summary = (
                "Recorded ML performance metrics changed."
            )

        else:

            performance_summary = (
                "No recorded ML performance metric changed."
            )

        if row_delta > 0:

            dataset_summary = (
                f"The dataset increased by {row_delta} rows."
            )

        elif row_delta < 0:

            dataset_summary = (
                f"The dataset decreased by {abs(row_delta)} rows."
            )

        else:

            dataset_summary = (
                "The dataset row count did not change."
            )

        code_summary = (
            "Training/model code changed."
            if code_changed
            else "Training/model code did not change."
        )

        summary = (
            f"Comparison of Version "
            f"{comparison['version_1']} to Version "
            f"{comparison['version_2']}: "
            f"{dataset_summary} "
            f"{code_summary} "
            f"{performance_summary}"
        )

        return {
            "title":
                "Executive Summary",

            "summary":
                summary,

            "details": {
                "dataset_summary":
                    dataset_summary,

                "code_summary":
                    code_summary,

                "performance_summary":
                    performance_summary,

                "evidence":
                    comparison.get(
                        "evidence_chain",
                        [],
                    ),
            },
        }

    # ============================================================
    # VERSION OVERVIEW
    # ============================================================

    @staticmethod
    def _build_version_overview(
        comparison: dict[str, Any],
    ) -> dict[str, Any]:

        return {
            "title":
                "Version Overview",

            "summary":
                f"Comparing Version "
                f"{comparison['version_1']} with Version "
                f"{comparison['version_2']}.",

            "details": {
                "version_1":
                    comparison["version_1"],

                "version_2":
                    comparison["version_2"],

                "git_commit_before":
                    comparison[
                        "git_commit_before"
                    ],

                "git_commit_after":
                    comparison[
                        "git_commit_after"
                    ],

                "git_changed":
                    comparison[
                        "git_changed"
                    ],

                "dvc_changed":
                    comparison[
                        "dvc_changed"
                    ],

                "code_changed":
                    comparison[
                        "code_changed"
                    ],
            },
        }

    # ============================================================
    # WHAT CHANGED
    # ============================================================

    @staticmethod
    def _build_what_changed(
        comparison: dict[str, Any],
    ) -> dict[str, Any]:

        return {
            "title":
                "What Changed",

            "summary":
                "Deterministic changes detected between the two versions.",

            "details": {
                "changes":
                    comparison.get(
                        "changes",
                        [],
                    ),

                "changed_files":
                    comparison.get(
                        "changed_files",
                        [],
                    ),

                "code_changed_files":
                    comparison.get(
                        "code_changed_files",
                        [],
                    ),
            },
        }

    # ============================================================
    # DATASET ANALYSIS
    # ============================================================

    @staticmethod
    def _build_dataset_analysis(
        comparison: dict[str, Any],
    ) -> dict[str, Any]:

        analysis = (
            comparison.get(
                "dataset_analysis"
            )
            or {}
        )

        diff = (
            comparison.get(
                "dataset_diff"
            )
            or {}
        )

        return {
            "title":
                "Dataset Analysis",

            "summary":
                "Detailed DVC-backed dataset analysis.",

            "details": {
                "analysis":
                    analysis,

                "dataset_diff":
                    diff,

                "profiles":
                    comparison.get(
                        "dataset_profiles"
                    ),
            },
        }

    # ============================================================
    # FEATURE ANALYSIS
    # ============================================================

    @staticmethod
    def _build_feature_analysis(
        comparison: dict[str, Any],
    ) -> dict[str, Any]:

        ml = (
            comparison.get(
                "ml_comparison"
            )
            or {}
        )

        dataset = (
            comparison.get(
                "dataset_analysis"
            )
            or {}
        )

        return {
            "title":
                "Feature Analysis",

            "summary":
                "Comparison of recorded model features and dataset feature statistics.",

            "details": {
                "model_features_before":
                    ml.get(
                        "features_before",
                        [],
                    ),

                "model_features_after":
                    ml.get(
                        "features_after",
                        [],
                    ),

                "model_features_added":
                    ml.get(
                        "features_added",
                        [],
                    ),

                "model_features_removed":
                    ml.get(
                        "features_removed",
                        [],
                    ),

                "dataset_feature_changes":
                    dataset.get(
                        "feature_changes",
                        {},
                    ),
            },
        }

    # ============================================================
    # CODE ANALYSIS
    # ============================================================

    @staticmethod
    def _build_code_analysis(
        comparison: dict[str, Any],
    ) -> dict[str, Any]:

        return {
            "title":
                "Code / Git Analysis",

            "summary":
                (
                    "Training and model code changes are determined from Git."
                ),

            "details": {
                "code_changed":
                    comparison.get(
                        "code_changed"
                    ),

                "changed_files":
                    comparison.get(
                        "code_changed_files",
                        [],
                    ),

                "patch":
                    comparison.get(
                        "code_patch",
                        "",
                    ),
            },
        }

    # ============================================================
    # MODEL PARAMETERS
    # ============================================================

    @staticmethod
    def _build_model_parameters(
        comparison: dict[str, Any],
    ) -> dict[str, Any]:

        ml = (
            comparison.get(
                "ml_comparison"
            )
            or {}
        )

        return {
            "title":
                "Model & Parameters",

            "summary":
                "Recorded model identity, parameters, and changes.",

            "details": {
                "model_before":
                    ml.get(
                        "model_name_before"
                    ),

                "model_after":
                    ml.get(
                        "model_name_after"
                    ),

                "parameters_before":
                    ml.get(
                        "parameters_before",
                        {},
                    ),

                "parameters_after":
                    ml.get(
                        "parameters_after",
                        {},
                    ),

                "parameter_changes":
                    ml.get(
                        "parameter_changes",
                        {},
                    ),
            },
        }

    # ============================================================
    # PERFORMANCE
    # ============================================================

    @staticmethod
    def _build_performance_comparison(
        comparison: dict[str, Any],
    ) -> dict[str, Any]:

        performance = (
            comparison.get(
                "performance"
            )
            or {}
        )

        return {
            "title":
                "Performance Comparison",

            "summary":
                (
                    "Comparison of recorded ML performance metrics."
                ),

            "details": {
                "before":
                    performance.get(
                        "metrics_before",
                        {},
                    ),

                "after":
                    performance.get(
                        "metrics_after",
                        {},
                    ),

                "changes":
                    performance.get(
                        "metric_changes",
                        {},
                    ),

                "performance_changed":
                    performance.get(
                        "performance_changed",
                        False,
                    ),
            },
        }

    # ============================================================
    # ERROR ANALYSIS
    # ============================================================

    @staticmethod
    def _build_error_analysis(
        comparison: dict[str, Any],
    ) -> dict[str, Any]:

        ml = (
            comparison.get(
                "ml_comparison"
            )
            or {}
        )

        evaluation = (
            ml.get(
                "evaluation"
            )
            or {}
        )

        # --------------------------------------------------------
        # NO EVALUATION EVIDENCE
        # --------------------------------------------------------

        if not evaluation.get(
            "available",
            False,
        ):

            return {
                "title":
                    "Error Analysis",

                "summary":
                    (
                        "No evaluation evidence was recorded "
                        "for the compared ML runs."
                    ),

                "details": {
                    "status":
                        "not_available",

                    "available":
                        False,

                    "reason":
                        evaluation.get(
                            "reason",
                            "No evaluation evidence is available.",
                        ),
                },
            }

        status = evaluation.get(
            "status"
        )

        before = evaluation.get(
            "before"
        )

        after = evaluation.get(
            "after"
        )

        # --------------------------------------------------------
        # AFTER-ONLY
        # --------------------------------------------------------

        if status == "after_only":

            return {
                "title":
                    "Error Analysis",

                "summary":
                    (
                        "Evaluation evidence is available for "
                        "the newer ML run, but not for the older run."
                    ),

                "details": {
                    "status":
                        "after_only",

                    "available":
                        True,

                    "comparison_possible":
                        False,

                    "task_type":
                        (
                            after.get(
                                "task_type"
                            )
                            if after
                            else None
                        ),

                    "evaluation_before":
                        None,

                    "evaluation_after":
                        after,

                    "confusion_matrix_after":
                        (
                            after.get(
                                "confusion_matrix"
                            )
                            if after
                            else None
                        ),

                    "precision_after":
                        (
                            after.get(
                                "precision"
                            )
                            if after
                            else None
                        ),

                    "recall_after":
                        (
                            after.get(
                                "recall"
                            )
                            if after
                            else None
                        ),

                    "f1_after":
                        (
                            after.get(
                                "f1"
                            )
                            if after
                            else None
                        ),

                    "prediction_count_after":
                        (
                            after.get(
                                "prediction_count"
                            )
                            if after
                            else None
                        ),

                    "incorrect_count_after":
                        (
                            after.get(
                                "incorrect_count"
                            )
                            if after
                            else None
                        ),

                    "error_rate_after":
                        (
                            after.get(
                                "error_rate"
                            )
                            if after
                            else None
                        ),

                    "misclassified_samples_after":
                        (
                            after.get(
                                "misclassified_samples",
                                [],
                            )
                            if after
                            else []
                        ),

                    "changes":
                        evaluation.get(
                            "changes",
                            [],
                        ),

                    "limitation":
                        (
                            "A before/after error change cannot "
                            "be calculated because the older "
                            "ML run has no evaluation evidence."
                        ),
                },
            }

        # --------------------------------------------------------
        # BEFORE-ONLY
        # --------------------------------------------------------

        if status == "before_only":

            return {
                "title":
                    "Error Analysis",

                "summary":
                    (
                        "Evaluation evidence exists for the "
                        "older ML run, but not for the newer run."
                    ),

                "details": {
                    "status":
                        "before_only",

                    "available":
                        True,

                    "comparison_possible":
                        False,

                    "task_type":
                        (
                            before.get(
                                "task_type"
                            )
                            if before
                            else None
                        ),

                    "evaluation_before":
                        before,

                    "evaluation_after":
                        None,

                    "confusion_matrix_before":
                        (
                            before.get(
                                "confusion_matrix"
                            )
                            if before
                            else None
                        ),

                    "precision_before":
                        (
                            before.get(
                                "precision"
                            )
                            if before
                            else None
                        ),

                    "recall_before":
                        (
                            before.get(
                                "recall"
                            )
                            if before
                            else None
                        ),

                    "f1_before":
                        (
                            before.get(
                                "f1"
                            )
                            if before
                            else None
                        ),

                    "prediction_count_before":
                        (
                            before.get(
                                "prediction_count"
                            )
                            if before
                            else None
                        ),

                    "incorrect_count_before":
                        (
                            before.get(
                                "incorrect_count"
                            )
                            if before
                            else None
                        ),

                    "error_rate_before":
                        (
                            before.get(
                                "error_rate"
                            )
                            if before
                            else None
                        ),

                    "misclassified_samples_before":
                        (
                            before.get(
                                "misclassified_samples",
                                [],
                            )
                            if before
                            else []
                        ),

                    "changes":
                        evaluation.get(
                            "changes",
                            [],
                        ),

                    "limitation":
                        (
                            "A before/after error change cannot "
                            "be calculated because the newer "
                            "ML run has no evaluation evidence."
                        ),
                },
            }

        # --------------------------------------------------------
        # BOTH RUNS
        # --------------------------------------------------------

        confusion = evaluation.get(
            "confusion_matrix"
        ) or {}

        metrics = evaluation.get(
            "metrics"
        ) or {}

        misclassified = evaluation.get(
            "misclassified_samples"
        ) or {}

        return {
            "title":
                "Error Analysis",

            "summary":
                (
                    "Recorded evaluation evidence is available "
                    "for both compared ML runs."
                    if evaluation.get(
                        "changed"
                    ) is not False
                    else
                    "Recorded evaluation evidence did not change "
                    "between the two compared ML runs."
                ),

            "details": {
                "status":
                    "both",

                "available":
                    True,

                "comparison_possible":
                    True,

                "task_type":
                    evaluation.get(
                        "task_type"
                    ),

                "confusion_matrix":
                    confusion,

                "metrics":
                    metrics,

                "misclassified_samples":
                    misclassified,

                "changes":
                    evaluation.get(
                        "changes",
                        [],
                    ),

                "changed":
                    evaluation.get(
                        "changed",
                        False,
                    ),

                "summary":
                    evaluation.get(
                        "summary"
                    ),
            },
        }

    # ============================================================
    # CHANGE CHAIN
    # ============================================================

    @staticmethod
    def _build_change_chain(
        comparison: dict[str, Any],
    ) -> dict[str, Any]:

        dataset_analysis = (
            comparison.get(
                "dataset_analysis"
            )
            or {}
        )

        ml = (
            comparison.get(
                "ml_comparison"
            )
            or {}
        )

        performance = (
            comparison.get(
                "performance"
            )
            or {}
        )

        return {
            "title":
                "Data → Code → Model → Performance",

            "summary":
                (
                    "Evidence chain connecting project changes to measured ML performance."
                ),

            "details": {
                "data": {
                    "changed":
                        comparison.get(
                            "dvc_changed",
                            False,
                        ),

                    "analysis":
                        dataset_analysis,
                },

                "code": {
                    "changed":
                        comparison.get(
                            "code_changed",
                            False,
                        ),

                    "files":
                        comparison.get(
                            "code_changed_files",
                            [],
                        ),
                },

                "model": {
                    "features_added":
                        ml.get(
                            "features_added",
                            [],
                        ),

                    "features_removed":
                        ml.get(
                            "features_removed",
                            [],
                        ),

                    "parameter_changes":
                        ml.get(
                            "parameter_changes",
                            {},
                        ),
                },

                "performance": {
                    "changed":
                        performance.get(
                            "performance_changed",
                            False,
                        ),

                    "metrics":
                        performance.get(
                            "metric_changes",
                            {},
                        ),
                },

                "evaluation": {
                    "available":
                        (
                            ml.get(
                                "evaluation"
                            )
                            or {}
                        ).get(
                            "available",
                            False,
                        ),

                    "status":
                        (
                            ml.get(
                                "evaluation"
                            )
                            or {}
                        ).get(
                            "status"
                        ),

                    "changes":
                        (
                            ml.get(
                                "evaluation"
                            )
                            or {}
                        ).get(
                            "changes",
                            [],
                        ),
                },
            },
        }

    # ============================================================
    # GIT + DVC EVIDENCE
    # ============================================================

    @staticmethod
    def _build_git_dvc_evidence(
        comparison: dict[str, Any],
    ) -> dict[str, Any]:

        return {
            "title":
                "Git + DVC Evidence",

            "summary":
                "Low-level reproducibility evidence captured by DataGit.",

            "details": {
                "git_commit_before":
                    comparison[
                        "git_commit_before"
                    ],

                "git_commit_after":
                    comparison[
                        "git_commit_after"
                    ],

                "dvc_state_before":
                    comparison[
                        "dvc_state_before"
                    ],

                "dvc_state_after":
                    comparison[
                        "dvc_state_after"
                    ],
            },
        }

    # ============================================================
    # REPRODUCIBILITY
    # ============================================================

    @staticmethod
    def _build_reproducibility(
        comparison: dict[str, Any],
    ) -> dict[str, Any]:

        ml_before = (
            comparison.get(
                "ml_run_before"
            )
            or {}
        )

        ml_after = (
            comparison.get(
                "ml_run_after"
            )
            or {}
        )

        return {
            "title":
                "Reproducibility",

            "summary":
                (
                    "Git, DVC, and ML Run identities provide the reproducibility anchors for both versions."
                ),

            "details": {
                "version_1": {
                    "git_commit":
                        comparison[
                            "git_commit_before"
                        ],

                    "dvc_state":
                        comparison[
                            "dvc_state_before"
                        ],

                    "ml_run_id":
                        ml_before.get(
                            "id"
                        ),
                },

                "version_2": {
                    "git_commit":
                        comparison[
                            "git_commit_after"
                        ],

                    "dvc_state":
                        comparison[
                            "dvc_state_after"
                        ],

                    "ml_run_id":
                        ml_after.get(
                            "id"
                        ),
                },
            },
        }

    # ============================================================
    # TECHNICAL EVIDENCE
    # ============================================================

    @staticmethod
    def _build_technical_evidence(
        comparison: dict[str, Any],
    ) -> dict[str, Any]:

        return {
            "title":
                "Technical Evidence",

            "summary":
                (
                    "Raw deterministic evidence used to build the report."
            ),

            "details": {
                "evidence_chain":
                    comparison.get(
                        "evidence_chain",
                        [],
                    ),

                "raw_comparison":
                    comparison,
            },
        }