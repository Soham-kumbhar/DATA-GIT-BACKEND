from typing import Any

from sqlalchemy.orm import Session

from app.db.models import MLRun, Project, Version, VersionPreparationEvidence
from app.services.dataset_analysis_service import (
    DatasetAnalysisService,
)
from app.services.dvc_service import DVCService
from app.services.evaluation_comparison_service import (
    EvaluationComparisonService,
)
from app.services.git_service import GitService


class VersionComparisonService:

    # ============================================================
    # PERFORMANCE METRICS
    # ============================================================

    PERFORMANCE_METRICS = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "f1_score",
        "roc_auc",
        "auc",
        "log_loss",
        "mse",
        "rmse",
        "mae",
        "r2",
        "r_squared",
        "mape",
        "balanced_accuracy",
        "specificity",
        "sensitivity",
    }

    # ============================================================
    # MULTI-VERSION COMPARISON
    # ============================================================

    @staticmethod
    def compare_multiple_versions(
        db: Session,
        project_id: int,
        version_ids: list[int],
        mode: str = "evolution",
    ) -> dict[str, Any]:

        project = (
            db.query(Project)
            .filter(
                Project.id == project_id
            )
            .first()
        )

        if project is None:
            raise ValueError(
                "Project not found."
            )

        if len(version_ids) < 2:
            raise ValueError(
                "At least two versions are required."
            )

        if mode not in {
            "evolution",
            "baseline",
        }:
            raise ValueError(
                "Invalid comparison mode. "
                "Use 'evolution' or 'baseline'."
            )

        # --------------------------------------------------------
        # LOAD SELECTED VERSIONS
        # --------------------------------------------------------

        versions = (
            db.query(Version)
            .filter(
                Version.project_id == project_id,
                Version.id.in_(version_ids),
            )
            .all()
        )

        version_by_id = {
            version.id: version
            for version in versions
        }

        missing_ids = [
            version_id
            for version_id in version_ids
            if version_id not in version_by_id
        ]

        if missing_ids:
            raise ValueError(
                "Version(s) not found: "
                + ", ".join(
                    str(version_id)
                    for version_id in missing_ids
                )
            )

        # --------------------------------------------------------
        # SORT BY DATAGIT VERSION NUMBER
        # --------------------------------------------------------

        ordered_versions = sorted(
            (
                version_by_id[version_id]
                for version_id in version_ids
            ),
            key=lambda item: item.version_number,
        )

        # --------------------------------------------------------
        # BUILD COMPARISON PAIRS
        # --------------------------------------------------------

        pairs: list[
            tuple[Version, Version]
        ] = []

        if mode == "evolution":

            for index in range(
                len(ordered_versions) - 1
            ):

                before = ordered_versions[index]
                after = ordered_versions[index + 1]

                pairs.append(
                    (
                        before,
                        after,
                    )
                )

        else:

            baseline = ordered_versions[0]

            for after in ordered_versions[1:]:

                pairs.append(
                    (
                        baseline,
                        after,
                    )
                )

        # --------------------------------------------------------
        # RUN EXISTING DETERMINISTIC COMPARISON
        # --------------------------------------------------------

        comparisons = []

        for before, after in pairs:

            report = (
                VersionComparisonService.compare_versions(
                    db=db,
                    project_id=project_id,
                    version_1=before.id,
                    version_2=after.id,
                )
            )

            comparisons.append(
                report
            )

        # --------------------------------------------------------
        # BUILD SUMMARY
        # --------------------------------------------------------

        summary = (
            VersionComparisonService
            ._build_multi_version_summary(
                comparisons
            )
        )

        return {
            "project_id":
                project_id,

            "mode":
                mode,

            "selected_versions": [
                {
                    "id":
                        version.id,

                    "version_number":
                        version.version_number,

                    "git_commit":
                        version.git_commit,

                    "dvc_state":
                        version.dvc_state,

                    "ml_run_id":
                        version.ml_run_id,

                    "created_at":
                        version.created_at,
                }
                for version in ordered_versions
            ],

            "comparison_count":
                len(comparisons),

            "comparisons":
                comparisons,

            "summary":
                summary,
        }

    # ============================================================
    # MULTI-VERSION SUMMARY
    # ============================================================

    @staticmethod
    def _build_multi_version_summary(
        comparisons: list[dict[str, Any]],
    ) -> dict[str, Any]:

        summary = {
            "performance_changes": [],
            "dataset_changes": [],
            "code_changes": [],
            "parameter_changes": [],
            "evaluation_changes": [],
            "regressions": [],
        }

        for comparison in comparisons:

            version_1 = comparison.get(
                "version_1"
            )

            version_2 = comparison.get(
                "version_2"
            )

            label = (
                f"V{version_1} → V{version_2}"
            )

            # ----------------------------------------------------
            # PERFORMANCE
            # ----------------------------------------------------

            performance = (
                comparison.get(
                    "performance"
                )
                or {}
            )

            if performance.get(
                "performance_changed",
                False,
            ):

                metric_changes = (
                    performance.get(
                        "metric_changes",
                        {},
                    )
                )

                summary[
                    "performance_changes"
                ].append(
                    {
                        "comparison":
                            label,

                        "metrics":
                            metric_changes,
                    }
                )

                # ------------------------------------------------
                # REGRESSIONS
                # ------------------------------------------------

                for (
                    metric,
                    change,
                ) in metric_changes.items():

                    if not isinstance(
                        change,
                        dict,
                    ):
                        continue

                    delta = change.get(
                        "delta"
                    )

                    if (
                        isinstance(
                            delta,
                            (int, float),
                        )
                        and delta < 0
                    ):

                        summary[
                            "regressions"
                        ].append(
                            {
                                "comparison":
                                    label,

                                "metric":
                                    metric,

                                "before":
                                    change.get(
                                        "before"
                                    ),

                                "after":
                                    change.get(
                                        "after"
                                    ),

                                "delta":
                                    delta,
                            }
                        )

            # ----------------------------------------------------
            # DATASET
            # ----------------------------------------------------

            if comparison.get(
                "dvc_changed",
                False,
            ):

                summary[
                    "dataset_changes"
                ].append(
                    {
                        "comparison":
                            label,

                        "dataset_diff":
                            comparison.get(
                                "dataset_diff"
                            ),

                        "dataset_analysis":
                            comparison.get(
                                "dataset_analysis"
                            ),
                    }
                )

            # ----------------------------------------------------
            # CODE
            # ----------------------------------------------------

            if comparison.get(
                "code_changed",
                False,
            ):

                summary[
                    "code_changes"
                ].append(
                    {
                        "comparison":
                            label,

                        "changed_files":
                            comparison.get(
                                "code_changed_files",
                                [],
                            ),
                    }
                )

            # ----------------------------------------------------
            # PARAMETERS
            # ----------------------------------------------------

            ml_comparison = (
                comparison.get(
                    "ml_comparison"
                )
                or {}
            )

            parameter_changes = (
                ml_comparison.get(
                    "parameter_changes",
                    {},
                )
            )

            if parameter_changes:

                summary[
                    "parameter_changes"
                ].append(
                    {
                        "comparison":
                            label,

                        "changes":
                            parameter_changes,
                    }
                )

            # ----------------------------------------------------
            # EVALUATION
            # ----------------------------------------------------

            evaluation = (
                ml_comparison.get(
                    "evaluation"
                )
                or {}
            )

            if (
                evaluation
                and evaluation.get(
                    "available",
                    False,
                )
                and evaluation.get(
                    "changed",
                    False,
                )
            ):

                summary[
                    "evaluation_changes"
                ].append(
                    {
                        "comparison":
                            label,

                        "changes":
                            evaluation.get(
                                "changes",
                                [],
                            ),
                    }
                )

        return summary

    # ============================================================
    # MAIN SINGLE VERSION COMPARISON
    # ============================================================

    @staticmethod
    def compare_versions(
        db: Session,
        project_id: int,
        version_1: int,
        version_2: int,
    ) -> dict[str, Any]:

        project = (
            db.query(Project)
            .filter(
                Project.id == project_id
            )
            .first()
        )

        if project is None:
            raise ValueError(
                "Project not found."
            )

        before = (
            db.query(Version)
            .filter(
                Version.id == version_1,
                Version.project_id == project_id,
            )
            .first()
        )

        if before is None:
            raise ValueError(
                f"Version {version_1} not found."
            )

        after = (
            db.query(Version)
            .filter(
                Version.id == version_2,
                Version.project_id == project_id,
            )
            .first()
        )

        if after is None:
            raise ValueError(
                f"Version {version_2} not found."
            )

        # --------------------------------------------------------
        # GIT
        # --------------------------------------------------------

        git_changed = (
            before.git_commit
            != after.git_commit
        )

        # --------------------------------------------------------
        # DVC
        # --------------------------------------------------------

        dvc_changed = (
            before.dvc_state
            != after.dvc_state
        )

        # --------------------------------------------------------
        # CHANGED FILES
        # --------------------------------------------------------

        changed_files = (
            GitService.get_changed_files(
                project.path,
                before.git_commit,
                after.git_commit,
            )
        )

        # --------------------------------------------------------
        # CODE FILES
        # --------------------------------------------------------

        code_changed_files = []

        for file_path in changed_files:

            normalized = (
                file_path
                .replace("\\", "/")
                .lower()
            )

            if normalized.endswith(".dvc"):
                continue

            if (
                normalized.startswith("src/")
                or normalized.endswith(".py")
                or normalized.endswith(".ipynb")
                or normalized.endswith(".pkl")
                or normalized.endswith(".joblib")
                or normalized.endswith(".onnx")
            ):
                code_changed_files.append(
                    file_path
                )

        code_changed = (
            len(code_changed_files) > 0
        )

        # --------------------------------------------------------
        # CODE PATCH
        # --------------------------------------------------------

        code_patch = ""

        if code_changed:

            try:

                code_patch = (
                    GitService.get_commit_patch(
                        project.path,
                        before.git_commit,
                        after.git_commit,
                    )
                )

            except RuntimeError:

                code_patch = ""

        # --------------------------------------------------------
        # ML RUNS
        # --------------------------------------------------------

        run_before = None

        if before.ml_run_id is not None:

            run_before = (
                db.query(MLRun)
                .filter(
                    MLRun.id
                    == before.ml_run_id,
                    MLRun.project_id
                    == project_id,
                )
                .first()
            )

        run_after = None

        if after.ml_run_id is not None:

            run_after = (
                db.query(MLRun)
                .filter(
                    MLRun.id
                    == after.ml_run_id,
                    MLRun.project_id
                    == project_id,
                )
                .first()
            )

        # --------------------------------------------------------
        # ML COMPARISON
        # --------------------------------------------------------

        ml_comparison = (
            VersionComparisonService._compare_ml_runs(
                run_before,
                run_after,
            )
        )

        # --------------------------------------------------------
        # PERFORMANCE
        # --------------------------------------------------------

        performance = (
            VersionComparisonService._compare_performance(
                run_before,
                run_after,
            )
        )

        # --------------------------------------------------------
        # DATASET DIFF
        # --------------------------------------------------------

        dataset_diff = (
            VersionComparisonService._compare_dataset(
                project.path,
                before,
                after,
            )
        )

        # --------------------------------------------------------
        # DATASET PROFILES
        # --------------------------------------------------------

        dataset_profiles = (
            VersionComparisonService._build_dataset_profiles(
                project.path,
                dataset_diff,
            )
        )

        # --------------------------------------------------------
        # DATASET ANALYSIS
        # --------------------------------------------------------

        dataset_analysis = (
            VersionComparisonService._compare_dataset_analysis(
                dataset_profiles.get("before"),
                dataset_profiles.get("after"),
            )
        )

        # --------------------------------------------------------
        # PREPARATION EVIDENCE
        # --------------------------------------------------------

        preparation = (
            VersionComparisonService._compare_preparation(
                db=db,
                before=before,
                after=after,
            )
        )

        # --------------------------------------------------------
        # EVIDENCE CHAIN
        # --------------------------------------------------------

        evidence_chain = (
            VersionComparisonService._build_evidence_chain(
                git_changed=git_changed,
                dvc_changed=dvc_changed,
                code_changed=code_changed,
                dataset_diff=dataset_diff,
                dataset_analysis=dataset_analysis,
                ml_comparison=ml_comparison,
                performance=performance,
            )
        )

        # --------------------------------------------------------
        # HUMAN-READABLE CHANGES
        # --------------------------------------------------------

        changes = []

        if git_changed:
            changes.append(
                "Git commit changed"
            )

        if dvc_changed:
            changes.append(
                "dataset changed"
            )

        if code_changed:
            changes.append(
                "model/code changed"
            )

        if ml_comparison is not None:

            if ml_comparison.get(
                "features_added",
                [],
            ):
                changes.append(
                    "features added"
                )

            if ml_comparison.get(
                "features_removed",
                [],
            ):
                changes.append(
                    "features removed"
                )

            if ml_comparison.get(
                "parameter_changes",
                {},
            ):
                changes.append(
                    "model parameters changed"
                )

            if ml_comparison.get(
                "performance_changes",
                {},
            ):
                changes.append(
                    "performance metrics changed"
                )

            evaluation = ml_comparison.get(
                "evaluation"
            )

            if (
                evaluation
                and evaluation.get("available")
                and evaluation.get("changed")
            ):
                changes.append(
                    "evaluation evidence changed"
                )

        if not changes:
            changes.append(
                "No changes detected."
            )

        # --------------------------------------------------------
        # FINAL RESULT
        # --------------------------------------------------------

        return {
            "project_id":
                project_id,

            "version_1":
                before.version_number,

            "version_2":
                after.version_number,

            "git_changed":
                git_changed,

            "dvc_changed":
                dvc_changed,

            "code_changed":
                code_changed,

            "git_commit_before":
                before.git_commit,

            "git_commit_after":
                after.git_commit,

            "dvc_state_before":
                before.dvc_state,

            "dvc_state_after":
                after.dvc_state,

            "changed_files":
                changed_files,

            "code_changed_files":
                code_changed_files,

            "code_patch":
                code_patch,

            "ml_run_before":
                VersionComparisonService._run_to_dict(
                    run_before
                ),

            "ml_run_after":
                VersionComparisonService._run_to_dict(
                    run_after
                ),

            "ml_comparison":
                ml_comparison,

            "performance":
                performance,

            "dataset_diff":
                dataset_diff,

            "dataset_profiles":
                dataset_profiles,

            "dataset_analysis":
                dataset_analysis,

            "preparation":
                preparation,

            "evidence_chain":
                evidence_chain,

            "changes":
                changes,
        }

    # ============================================================
    # PREPARATION COMPARISON
    # ============================================================

    @staticmethod
    def _compare_preparation(
        db: Session,
        before: Version,
        after: Version,
    ) -> dict[str, Any]:
        evidence_before = (
            db.query(VersionPreparationEvidence)
            .filter(
                VersionPreparationEvidence.version_id == before.id
            )
            .first()
        )

        evidence_after = (
            db.query(VersionPreparationEvidence)
            .filter(
                VersionPreparationEvidence.version_id == after.id
            )
            .first()
        )

        if evidence_before is None and evidence_after is None:
            return {
                "available": False,
                "status": "unavailable",
                "changed": False,
                "baseline": [],
                "target": [],
                "added": [],
                "removed": [],
                "modified": [],
                "message": (
                    "Preparation history is not recorded for "
                    "the selected versions."
                ),
            }

        baseline = (
            evidence_before.operations
            if evidence_before is not None
            and isinstance(evidence_before.operations, list)
            else []
        )

        target = (
            evidence_after.operations
            if evidence_after is not None
            and isinstance(evidence_after.operations, list)
            else []
        )

        baseline = [
            item for item in baseline
            if isinstance(item, dict)
        ]

        target = [
            item for item in target
            if isinstance(item, dict)
        ]

        baseline_by_name = {
            str(item.get("operation", "unknown")): item
            for item in baseline
        }

        target_by_name = {
            str(item.get("operation", "unknown")): item
            for item in target
        }

        baseline_names = set(baseline_by_name)
        target_names = set(target_by_name)

        added = [
            target_by_name[name]
            for name in sorted(target_names - baseline_names)
        ]

        removed = [
            baseline_by_name[name]
            for name in sorted(baseline_names - target_names)
        ]

        modified = []

        for name in sorted(
            baseline_names & target_names
        ):
            old_value = baseline_by_name[name]
            new_value = target_by_name[name]

            if old_value != new_value:
                modified.append(
                    {
                        "operation": name,
                        "baseline": old_value,
                        "target": new_value,
                    }
                )

        return {
            "available": True,
            "status": (
                "recorded"
                if evidence_before is not None
                and evidence_after is not None
                else "partial"
            ),
            "changed": bool(
                added or removed or modified
            ),
            "baseline": baseline,
            "target": target,
            "added": added,
            "removed": removed,
            "modified": modified,
            "message": (
                "Preparation operations are recorded for "
                "both selected versions."
                if evidence_before is not None
                and evidence_after is not None
                else (
                    "Preparation evidence is recorded for "
                    "only one selected version."
                )
            ),
        }

    # ============================================================
    # DATASET PROFILES
    # ============================================================

    @staticmethod
    def _build_dataset_profiles(
        project_path: str,
        dataset_diff: dict[str, Any] | None,
    ) -> dict[str, Any]:

        if not dataset_diff:
            return {
                "before": None,
                "after": None,
            }

        datasets = dataset_diff.get(
            "datasets",
            [],
        )

        before_profile = None
        after_profile = None

        for dataset in datasets:

            old_hash = dataset.get(
                "old_dvc_hash"
            )

            new_hash = dataset.get(
                "new_dvc_hash"
            )

            dataset_name = dataset.get(
                "dataset"
            )

            if (
                not old_hash
                or not new_hash
                or not dataset_name
            ):
                continue

            try:

                before_profile = (
                    DatasetAnalysisService
                    .analyze_dataset(
                        project_path=project_path,
                        dvc_hash=old_hash,
                        dataset_name=dataset_name,
                    )
                )

                after_profile = (
                    DatasetAnalysisService
                    .analyze_dataset(
                        project_path=project_path,
                        dvc_hash=new_hash,
                        dataset_name=dataset_name,
                    )
                )

            except Exception as exc:

                return {
                    "before": None,
                    "after": None,
                    "error": str(exc),
                }

            break

        return {
            "before": before_profile,
            "after": after_profile,
        }

    # ============================================================
    # DATASET ANALYSIS COMPARISON
    # ============================================================

    @staticmethod
    def _compare_dataset_analysis(
        before: dict[str, Any] | None,
        after: dict[str, Any] | None,
    ) -> dict[str, Any] | None:

        if before is None and after is None:
            return None

        if before is None or after is None:
            return {
                "available":
                    False,

                "reason":
                    "A complete dataset profile was not available.",
            }

        missing_before = before.get(
            "missing_values",
            {},
        )

        missing_after = after.get(
            "missing_values",
            {},
        )

        missing_changes = (
            VersionComparisonService
            ._compare_dict_values(
                missing_before,
                missing_after,
            )
        )

        duplicate_before = before.get(
            "duplicate_rows",
            0,
        )

        duplicate_after = after.get(
            "duplicate_rows",
            0,
        )

        duplicates_changed = (
            duplicate_before
            != duplicate_after
        )

        columns_before = before.get(
            "columns",
            [],
        )

        columns_after = after.get(
            "columns",
            [],
        )

        columns_added = [
            column
            for column in columns_after
            if column not in columns_before
        ]

        columns_removed = [
            column
            for column in columns_before
            if column not in columns_after
        ]

        target_distribution_before = (
            before.get(
                "target_distribution",
                {},
            )
        )

        target_distribution_after = (
            after.get(
                "target_distribution",
                {},
            )
        )

        target_distribution_changes = (
            VersionComparisonService
            ._compare_dict_values(
                target_distribution_before,
                target_distribution_after,
            )
        )

        feature_changes = (
            VersionComparisonService
            ._compare_feature_profiles(
                before.get(
                    "feature_profile",
                    {},
                ),
                after.get(
                    "feature_profile",
                    {},
                ),
            )
        )

        return {
            "available":
                True,

            "rows_before":
                before.get("rows"),

            "rows_after":
                after.get("rows"),

            "row_delta":
                (
                    after.get("rows", 0)
                    - before.get("rows", 0)
                ),

            "columns_before":
                columns_before,

            "columns_after":
                columns_after,

            "columns_added":
                columns_added,

            "columns_removed":
                columns_removed,

            "missing_values_before":
                missing_before,

            "missing_values_after":
                missing_after,

            "missing_value_changes":
                missing_changes,

            "duplicate_rows_before":
                duplicate_before,

            "duplicate_rows_after":
                duplicate_after,

            "duplicates_changed":
                duplicates_changed,

            "target_column_before":
                before.get(
                    "target_column"
                ),

            "target_column_after":
                after.get(
                    "target_column"
                ),

            "target_distribution_before":
                target_distribution_before,

            "target_distribution_after":
                target_distribution_after,

            "target_distribution_changes":
                target_distribution_changes,

            "feature_changes":
                feature_changes,
        }

    # ============================================================
    # FEATURE PROFILE COMPARISON
    # ============================================================

    @staticmethod
    def _compare_feature_profiles(
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> dict[str, Any]:

        features_added = [
            feature
            for feature in after
            if feature not in before
        ]

        features_removed = [
            feature
            for feature in before
            if feature not in after
        ]

        changed = {}

        shared_features = [
            feature
            for feature in before
            if feature in after
        ]

        for feature in shared_features:

            old_profile = before[feature]
            new_profile = after[feature]

            differences = (
                VersionComparisonService
                ._compare_dict_values(
                    old_profile,
                    new_profile,
                    calculate_numeric_delta=True,
                )
            )

            if differences:
                changed[feature] = differences

        return {
            "features_added":
                features_added,

            "features_removed":
                features_removed,

            "features_changed":
                changed,
        }

    # ============================================================
    # PERFORMANCE COMPARISON
    # ============================================================

    @staticmethod
    def _compare_performance(
        before: MLRun | None,
        after: MLRun | None,
    ) -> dict[str, Any] | None:

        if before is None and after is None:
            return None

        before_metrics = (
            before.metrics
            if before is not None
            and before.metrics is not None
            else {}
        )

        after_metrics = (
            after.metrics
            if after is not None
            and after.metrics is not None
            else {}
        )

        (
            before_performance,
            _,
        ) = VersionComparisonService._split_metrics(
            before_metrics
        )

        (
            after_performance,
            _,
        ) = VersionComparisonService._split_metrics(
            after_metrics
        )

        metric_changes = (
            VersionComparisonService
            ._compare_dict_values(
                before_performance,
                after_performance,
                calculate_numeric_delta=True,
            )
        )

        return {
            "metrics_before":
                before_performance,

            "metrics_after":
                after_performance,

            "metric_changes":
                metric_changes,

            "performance_changed":
                bool(metric_changes),
        }

    # ============================================================
    # SPLIT PERFORMANCE FROM OTHER METRICS
    # ============================================================

    @staticmethod
    def _split_metrics(
        metrics: dict[str, Any],
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
    ]:

        performance = {}
        other = {}

        for key, value in metrics.items():

            normalized_key = (
                str(key).strip().lower()
            )

            if normalized_key in (
                VersionComparisonService
                .PERFORMANCE_METRICS
            ):
                performance[key] = value

            else:
                other[key] = value

        return performance, other

    # ============================================================
    # ML RUN COMPARISON
    # ============================================================

    @staticmethod
    def _compare_ml_runs(
        before: MLRun | None,
        after: MLRun | None,
    ) -> dict[str, Any] | None:

        if before is None and after is None:
            return None

        before_features = (
            before.features
            if before is not None
            and before.features is not None
            else []
        )

        after_features = (
            after.features
            if after is not None
            and after.features is not None
            else []
        )

        features_added = [
            feature
            for feature in after_features
            if feature not in before_features
        ]

        features_removed = [
            feature
            for feature in before_features
            if feature not in after_features
        ]

        before_parameters = (
            before.parameters
            if before is not None
            and before.parameters is not None
            else {}
        )

        after_parameters = (
            after.parameters
            if after is not None
            and after.parameters is not None
            else {}
        )

        parameter_changes = (
            VersionComparisonService
            ._compare_dict_values(
                before_parameters,
                after_parameters,
            )
        )

        before_metrics = (
            before.metrics
            if before is not None
            and before.metrics is not None
            else {}
        )

        after_metrics = (
            after.metrics
            if after is not None
            and after.metrics is not None
            else {}
        )

        (
            before_performance,
            before_other,
        ) = VersionComparisonService._split_metrics(
            before_metrics
        )

        (
            after_performance,
            after_other,
        ) = VersionComparisonService._split_metrics(
            after_metrics
        )

        performance_changes = (
            VersionComparisonService
            ._compare_dict_values(
                before_performance,
                after_performance,
                calculate_numeric_delta=True,
            )
        )

        other_metric_changes = (
            VersionComparisonService
            ._compare_dict_values(
                before_other,
                after_other,
                calculate_numeric_delta=True,
            )
        )

        before_evaluation = (
            before.evaluation
            if before is not None
            and before.evaluation is not None
            else None
        )

        after_evaluation = (
            after.evaluation
            if after is not None
            and after.evaluation is not None
            else None
        )

        evaluation_comparison = (
            EvaluationComparisonService.compare(
                evaluation_before=before_evaluation,
                evaluation_after=after_evaluation,
            )
        )

        return {
            "run_id_before":
                before.id
                if before is not None
                else None,

            "run_id_after":
                after.id
                if after is not None
                else None,

            "model_name_before":
                before.model_name
                if before is not None
                else None,

            "model_name_after":
                after.model_name
                if after is not None
                else None,

            "features_before":
                before_features,

            "features_after":
                after_features,

            "features_added":
                features_added,

            "features_removed":
                features_removed,

            "parameters_before":
                before_parameters,

            "parameters_after":
                after_parameters,

            "parameter_changes":
                parameter_changes,

            "performance_before":
                before_performance,

            "performance_after":
                after_performance,

            "performance_changes":
                performance_changes,

            "other_metrics_before":
                before_other,

            "other_metrics_after":
                after_other,

            "other_metric_changes":
                other_metric_changes,

            "evaluation":
                evaluation_comparison,
        }

    # ============================================================
    # DATASET COMPARISON
    # ============================================================

    @staticmethod
    def _compare_dataset(
        project_path: str,
        before: Version,
        after: Version,
    ) -> dict[str, Any] | None:

        before_files = (
            before.dvc_state.get(
                "tracked_files",
                [],
            )
            if before.dvc_state
            else []
        )

        after_files = (
            after.dvc_state.get(
                "tracked_files",
                [],
            )
            if after.dvc_state
            else []
        )

        if not before_files and not after_files:
            return None

        before_by_dvc = {
            item.get("dvc_file"):
                item
            for item in before_files
            if item.get("dvc_file")
        }

        after_by_dvc = {
            item.get("dvc_file"):
                item
            for item in after_files
            if item.get("dvc_file")
        }

        shared_files = [
            dvc_file
            for dvc_file in before_by_dvc
            if dvc_file in after_by_dvc
        ]

        if not shared_files:
            return {
                "dataset_changed":
                    True,

                "datasets":
                    [],

                "error":
                    "No common DVC dataset was found between the two versions.",
            }

        dataset_results = []

        for dvc_file in shared_files:

            try:

                diff = (
                    DVCService.get_dataset_diff(
                        project_path=project_path,
                        old_commit=before.git_commit,
                        new_commit=after.git_commit,
                        data_path=dvc_file,
                    )
                )

                dataset_results.append(
                    diff
                )

            except Exception as exc:

                dataset_results.append(
                    {
                        "dataset":
                            dvc_file,

                        "dataset_changed":
                            True,

                        "error":
                            str(exc),
                    }
                )

        return {
            "dataset_changed":
                any(
                    item.get(
                        "dataset_changed",
                        False,
                    )
                    for item in dataset_results
                ),

            "datasets":
                dataset_results,
        }

    # ============================================================
    # EVIDENCE CHAIN
    # ============================================================

    @staticmethod
    def _build_evidence_chain(
        git_changed: bool,
        dvc_changed: bool,
        code_changed: bool,
        dataset_diff: dict[str, Any] | None,
        dataset_analysis: dict[str, Any] | None,
        ml_comparison: dict[str, Any] | None,
        performance: dict[str, Any] | None,
    ) -> list[str]:

        evidence = []

        if git_changed:
            evidence.append(
                "The Git commit changed between the two versions."
            )

        if dvc_changed:
            evidence.append(
                "The DVC-tracked dataset state changed."
            )

        if code_changed:
            evidence.append(
                "Training/model code files changed."
            )

        else:
            evidence.append(
                "No training/model code files changed."
            )

        if dataset_analysis:

            if dataset_analysis.get(
                "available",
                False,
            ):

                row_delta = dataset_analysis.get(
                    "row_delta",
                    0,
                )

                if row_delta > 0:

                    evidence.append(
                        f"The dataset gained {row_delta} rows."
                    )

                elif row_delta < 0:

                    evidence.append(
                        f"The dataset lost {abs(row_delta)} rows."
                    )

                else:

                    evidence.append(
                        "Dataset row count did not change."
                    )

                if (
                    not dataset_analysis.get(
                        "columns_added",
                        [],
                    )
                    and not dataset_analysis.get(
                        "columns_removed",
                        [],
                    )
                ):

                    evidence.append(
                        "Dataset schema columns remained unchanged."
                    )

                if not dataset_analysis.get(
                    "missing_value_changes",
                    {},
                ):

                    evidence.append(
                        "Missing-value counts did not change."
                    )

                if not dataset_analysis.get(
                    "duplicates_changed",
                    False,
                ):

                    evidence.append(
                        "Duplicate-row count did not change."
                    )

                if not dataset_analysis.get(
                    "target_distribution_changes",
                    {},
                ):

                    evidence.append(
                        "Target/class distribution counts did not change."
                    )

                feature_changes = (
                    dataset_analysis.get(
                        "feature_changes",
                        {},
                    )
                )

                if (
                    not feature_changes.get(
                        "features_added"
                    )
                    and not feature_changes.get(
                        "features_removed"
                    )
                    and not feature_changes.get(
                        "features_changed"
                    )
                ):

                    evidence.append(
                        "Deterministic feature-level statistics did not change."
                    )

        elif dataset_diff:

            evidence.append(
                "Detailed dataset profiling was unavailable."
            )

        if ml_comparison:

            if (
                not ml_comparison.get(
                    "features_added"
                )
                and not ml_comparison.get(
                    "features_removed"
                )
            ):

                evidence.append(
                    "The recorded feature set did not change."
                )

            if not ml_comparison.get(
                "parameter_changes"
            ):

                evidence.append(
                    "The recorded model parameters did not change."
                )

            evaluation = ml_comparison.get(
                "evaluation"
            )

            if evaluation:

                status = evaluation.get(
                    "status"
                )

                if status == "both":

                    if evaluation.get(
                        "changed"
                    ):

                        evidence.append(
                            "Recorded evaluation evidence changed."
                        )

                    else:

                        evidence.append(
                            "Recorded evaluation evidence did not change."
                        )

                elif status == "after_only":

                    evidence.append(
                        "Evaluation evidence is available only for the newer run."
                    )

                elif status == "before_only":

                    evidence.append(
                        "Evaluation evidence is available only for the older run."
                    )

                elif not evaluation.get(
                    "available",
                    False,
                ):

                    evidence.append(
                        "No evaluation evidence was recorded."
                    )

        if performance:

            if performance.get(
                "performance_changed"
            ):

                evidence.append(
                    "At least one recorded performance metric changed."
                )

            else:

                evidence.append(
                    "No recorded performance metric changed."
                )

        return evidence

    # ============================================================
    # DICTIONARY DIFFERENCE
    # ============================================================

    @staticmethod
    def _compare_dict_values(
        before: dict[str, Any],
        after: dict[str, Any],
        calculate_numeric_delta: bool = False,
    ) -> dict[str, Any]:

        changes = {}

        keys = sorted(
            set(before.keys())
            | set(after.keys())
        )

        for key in keys:

            old_value = before.get(
                key
            )

            new_value = after.get(
                key
            )

            if old_value == new_value:
                continue

            result = {
                "before":
                    old_value,

                "after":
                    new_value,
            }

            if (
                calculate_numeric_delta
                and isinstance(
                    old_value,
                    (int, float),
                )
                and isinstance(
                    new_value,
                    (int, float),
                )
            ):

                result["delta"] = (
                    new_value
                    - old_value
                )

            changes[key] = result

        return changes

    # ============================================================
    # RUN SERIALIZATION
    # ============================================================

    @staticmethod
    def _run_to_dict(
        run: MLRun | None,
    ) -> dict[str, Any] | None:

        if run is None:
            return None

        return {
            "id":
                run.id,

            "project_id":
                run.project_id,

            "git_commit":
                run.git_commit,

            "dvc_state":
                run.dvc_state,

            "model_name":
                run.model_name,

            "features":
                run.features,

            "parameters":
                run.parameters,

            "metrics":
                run.metrics,

            "evaluation":
                run.evaluation,

            "created_at":
                run.created_at,
        }