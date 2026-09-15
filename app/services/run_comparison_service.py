from sqlalchemy.orm import Session

from app.db.models import MLRun
from app.services.dvc_service import DVCService
from app.services.git_service import GitService
from app.services.code_analysis_service import CodeAnalysisService
from app.services.ll_analysis_service import LLMAnalysisService


class RunComparisonService:

    @staticmethod
    def compare_runs(
        db: Session,
        run_id_1: int,
        run_id_2: int,
    ) -> dict:

        # ---------------------------------------------------------
        # 1. Get both runs
        # ---------------------------------------------------------

        run_1 = (
            db.query(MLRun)
            .filter(MLRun.id == run_id_1)
            .first()
        )

        run_2 = (
            db.query(MLRun)
            .filter(MLRun.id == run_id_2)
            .first()
        )

        if not run_1 or not run_2:
            raise ValueError(
                "One or both ML runs were not found."
            )

        if run_1.project_id != run_2.project_id:
            raise ValueError(
                "Runs must belong to the same project."
            )

        # ---------------------------------------------------------
        # 2. Basic changes
        # ---------------------------------------------------------

        dataset_changed = (
            run_1.dvc_state != run_2.dvc_state
        )

        model_changed = (
            run_1.model_name != run_2.model_name
        )

        features_changed = (
            run_1.features != run_2.features
        )

        parameters_changed = (
            run_1.parameters != run_2.parameters
        )

        code_changed = (
            run_1.git_commit != run_2.git_commit
        )

        changes = {
            "dataset_changed": dataset_changed,
            "model_changed": model_changed,
            "features_changed": features_changed,
            "parameters_changed": parameters_changed,
            "code_changed": code_changed,
        }

        # ---------------------------------------------------------
        # 3. Metric comparison
        # ---------------------------------------------------------

        metric_changes = {}

        metrics_1 = run_1.metrics or {}
        metrics_2 = run_2.metrics or {}

        metric_names = set(metrics_1) | set(metrics_2)

        for metric in metric_names:

            before = metrics_1.get(metric)
            after = metrics_2.get(metric)

            change = None
            percentage_change = None

            if (
                isinstance(before, (int, float))
                and isinstance(after, (int, float))
            ):
                change = after - before

                if before != 0:
                    percentage_change = (
                        (after - before) / before
                    ) * 100

            metric_changes[metric] = {
                "before": before,
                "after": after,
                "change": change,
                "percentage_change": percentage_change,
            }

        # ---------------------------------------------------------
        # 4. Get project path
        # ---------------------------------------------------------

        from app.db.models import Project

        project_record = (
            db.query(Project)
            .filter(Project.id == run_1.project_id)
            .first()
        )

        if not project_record:
            raise ValueError("Project not found.")

        project_path = project_record.path

        # ---------------------------------------------------------
        # 5. Dataset comparison
        # ---------------------------------------------------------

        dataset_details = None

        if dataset_changed:

            try:
                # Get DVC state stored with the second run
                dvc_state = run_2.dvc_state or {}

                tracked_files = dvc_state.get(
                    "tracked_files",
                    [],
                )

                if tracked_files:

                    tracked = tracked_files[0]

                    data_path = tracked.get("data_path")

                    if data_path:

                        # DVC metadata paths are relative
                        # to the .dvc file location.
                        dvc_file = tracked.get("dvc_file")

                        if dvc_file:

                            if "/" in dvc_file:

                                dvc_directory = (
                                    dvc_file.rsplit(
                                        "/",
                                        1,
                                    )[0]
                                )

                            elif "\\" in dvc_file:

                                dvc_directory = (
                                    dvc_file.rsplit(
                                        "\\",
                                        1,
                                    )[0]
                                )

                            else:

                                dvc_directory = ""

                            if dvc_directory:

                                data_path = (
                                    f"{dvc_directory}/"
                                    f"{data_path}"
                                )

                        dataset_details = (
                            DVCService.get_dataset_diff(
                                project_path=project_path,
                                old_commit=run_1.git_commit,
                                new_commit=run_2.git_commit,
                                data_path=data_path,
                            )
                        )

            except Exception as e:

                dataset_details = {
                    "dataset_changed": True,
                    "error": str(e),
                }

        # ---------------------------------------------------------
        # 6. Code comparison
        # ---------------------------------------------------------

        code_details = None

        if code_changed:

            try:

                patch = GitService.get_commit_patch(
                    project_path=project_path,
                    old_commit=run_1.git_commit,
                    new_commit=run_2.git_commit,
                )

                code_details = (
                    CodeAnalysisService.analyze_patch(
                        patch
                    )
                )

            except Exception as e:

                code_details = {
                    "error": str(e),
                }

        # ---------------------------------------------------------
        # 7. Generic explanation
        # ---------------------------------------------------------

        explanation = (
            RunComparisonService._build_explanation(
                changes=changes,
                metrics=metric_changes,
                dataset_details=dataset_details,
                code_details=code_details,
                run_1=run_1,
                run_2=run_2,
            )
        )

        # ---------------------------------------------------------
        # 8. Prepare evidence for LLM
        # ---------------------------------------------------------

        comparison_data = {
            "run_1": run_1.id,
            "run_2": run_2.id,

            "changes": changes,

            "metrics": metric_changes,

            "dataset_details": dataset_details,

            "code_details": code_details,
        }

        # ---------------------------------------------------------
        # 9. AI analysis using Groq
        # ---------------------------------------------------------

        try:

            ai_analysis = (
                LLMAnalysisService.analyze_comparison(
                    comparison_data
                )
            )

        except Exception as e:

            ai_analysis = {
                "error": (
                    "AI analysis could not be completed."
                ),
                "details": str(e),
            }

        # ---------------------------------------------------------
        # 10. Final response
        # ---------------------------------------------------------

        return {
            "run_1": run_1.id,
            "run_2": run_2.id,

            "changes": changes,

            "metrics": metric_changes,

            "dataset_details": dataset_details,

            "code_details": code_details,

            "explanation": explanation,

            "ai_analysis": ai_analysis,
        }

    # =============================================================
    # GENERIC EXPLANATION ENGINE
    # =============================================================

    @staticmethod
    def _build_explanation(
        changes: dict,
        metrics: dict,
        dataset_details: dict | None,
        code_details: dict | None,
        run_1: MLRun,
        run_2: MLRun,
    ) -> dict:

        # ---------------------------------------------------------
        # Determine performance
        # ---------------------------------------------------------

        r2_change = None
        rmse_change = None

        if "r2" in metrics:

            r2_change = metrics["r2"].get(
                "change"
            )

        if "rmse" in metrics:

            rmse_change = metrics["rmse"].get(
                "change"
            )

        performance_decreased = False
        performance_improved = False

        if (
            r2_change is not None
            and r2_change < 0
        ):

            performance_decreased = True

        if (
            rmse_change is not None
            and rmse_change > 0
        ):

            performance_decreased = True

        if (
            r2_change is not None
            and r2_change > 0
        ):

            performance_improved = True

        if (
            rmse_change is not None
            and rmse_change < 0
        ):

            performance_improved = True

        # ---------------------------------------------------------
        # Overall explanation
        # ---------------------------------------------------------

        if performance_decreased:

            overall = (
                "Model performance decreased between "
                "the two runs."
            )

        elif performance_improved:

            overall = (
                "Model performance improved between "
                "the two runs."
            )

        else:

            overall = (
                "Model performance did not show a clear "
                "improvement or degradation."
            )

        # ---------------------------------------------------------
        # Metric summary
        # ---------------------------------------------------------

        summary_parts = []

        if "rmse" in metrics:

            item = metrics["rmse"]

            before = item.get("before")
            after = item.get("after")
            change = item.get("change")

            if (
                before is not None
                and after is not None
                and change is not None
            ):

                sign = (
                    "+"
                    if change >= 0
                    else ""
                )

                summary_parts.append(
                    f"RMSE changed from {before} "
                    f"to {after}, a change of "
                    f"{sign}{change}."
                )

        if "r2" in metrics:

            item = metrics["r2"]

            before = item.get("before")
            after = item.get("after")
            change = item.get("change")

            if (
                before is not None
                and after is not None
                and change is not None
            ):

                sign = (
                    "+"
                    if change >= 0
                    else ""
                )

                summary_parts.append(
                    f"R² changed from {before} "
                    f"to {after}, a change of "
                    f"{sign}{change}."
                )

        summary = " ".join(summary_parts)

        # ---------------------------------------------------------
        # Identify possible causes
        # ---------------------------------------------------------

        possible_causes = []

        if changes["dataset_changed"]:

            possible_causes.append(
                "dataset changes"
            )

        if changes["features_changed"]:

            possible_causes.append(
                "feature changes"
            )

        if changes["parameters_changed"]:

            possible_causes.append(
                "parameter changes"
            )

        if changes["model_changed"]:

            possible_causes.append(
                "model changes"
            )

        if changes["code_changed"]:

            possible_causes.append(
                "code changes"
            )

        # ---------------------------------------------------------
        # Cause analysis
        # ---------------------------------------------------------

        cause_parts = []

        # ---------------------------------------------------------
        # Dataset
        # ---------------------------------------------------------

        if dataset_details:

            if dataset_details.get("error"):

                cause_parts.append(
                    "The dataset version changed, but "
                    "detailed dataset analysis could not "
                    "be completed."
                )

            else:

                old_rows = dataset_details.get(
                    "old_rows"
                )

                new_rows = dataset_details.get(
                    "new_rows"
                )

                rows_added = dataset_details.get(
                    "rows_added",
                    0,
                )

                rows_removed = dataset_details.get(
                    "rows_removed",
                    0,
                )

                columns_added = dataset_details.get(
                    "columns_added",
                    [],
                )

                columns_removed = dataset_details.get(
                    "columns_removed",
                    [],
                )

                dataset_messages = []

                if (
                    old_rows is not None
                    and new_rows is not None
                    and old_rows != new_rows
                ):

                    dataset_messages.append(
                        f"Dataset size changed from "
                        f"{old_rows} rows to "
                        f"{new_rows} rows."
                    )

                if rows_added:

                    dataset_messages.append(
                        f"{rows_added} row(s) were added."
                    )

                if rows_removed:

                    dataset_messages.append(
                        f"{rows_removed} row(s) were removed."
                    )

                if columns_added:

                    dataset_messages.append(
                        "Columns added: "
                        + ", ".join(columns_added)
                        + "."
                    )

                if columns_removed:

                    dataset_messages.append(
                        "Columns removed: "
                        + ", ".join(columns_removed)
                        + "."
                    )

                if dataset_messages:

                    cause_parts.append(
                        "Dataset changes: "
                        + " ".join(
                            dataset_messages
                        )
                    )

        # ---------------------------------------------------------
        # Code
        # ---------------------------------------------------------

        if code_details:

            if code_details.get("error"):

                cause_parts.append(
                    "Code changed, but detailed code "
                    "analysis could not be completed."
                )

            else:

                code_messages = []

                feature_details = (
                    code_details.get(
                        "feature_details"
                    )
                )

                if feature_details:

                    added = feature_details.get(
                        "added",
                        [],
                    )

                    removed = feature_details.get(
                        "removed",
                        [],
                    )

                    if added:

                        code_messages.append(
                            "Features added: "
                            + ", ".join(added)
                            + "."
                        )

                    if removed:

                        code_messages.append(
                            "Features removed: "
                            + ", ".join(removed)
                            + "."
                        )

                if code_details.get(
                    "preprocessing_changed"
                ):

                    code_messages.append(
                        "Preprocessing logic changed."
                    )

                if code_details.get(
                    "training_changed"
                ):

                    code_messages.append(
                        "Training logic changed."
                    )

                if code_details.get(
                    "model_changed"
                ):

                    code_messages.append(
                        "Model selection or model "
                        "configuration changed."
                    )

                if code_details.get(
                    "parameters_changed"
                ):

                    code_messages.append(
                        "Training parameters changed."
                    )

                if code_messages:

                    cause_parts.append(
                        "Code changes: "
                        + " ".join(
                            code_messages
                        )
                    )

        # ---------------------------------------------------------
        # Feature comparison from stored runs
        # ---------------------------------------------------------

        if changes["features_changed"]:

            before = run_1.features or []
            after = run_2.features or []

            added = [
                feature
                for feature in after
                if feature not in before
            ]

            removed = [
                feature
                for feature in before
                if feature not in after
            ]

            feature_message = []

            if added:

                feature_message.append(
                    "Added features: "
                    + ", ".join(added)
                    + "."
                )

            if removed:

                feature_message.append(
                    "Removed features: "
                    + ", ".join(removed)
                    + "."
                )

            if feature_message:

                cause_parts.append(
                    "Feature configuration changed. "
                    + " ".join(
                        feature_message
                    )
                )

        # ---------------------------------------------------------
        # Parameters
        # ---------------------------------------------------------

        if changes["parameters_changed"]:

            before = run_1.parameters or {}
            after = run_2.parameters or {}

            parameter_changes = []

            parameter_names = (
                set(before) | set(after)
            )

            for parameter in parameter_names:

                old_value = before.get(
                    parameter
                )

                new_value = after.get(
                    parameter
                )

                if old_value != new_value:

                    parameter_changes.append(
                        f"{parameter}: "
                        f"{old_value} → "
                        f"{new_value}"
                    )

            if parameter_changes:

                cause_parts.append(
                    "Parameter changes: "
                    + "; ".join(
                        parameter_changes
                    )
                    + "."
                )

        # ---------------------------------------------------------
        # Model
        # ---------------------------------------------------------

        if changes["model_changed"]:

            cause_parts.append(
                f"Model changed from "
                f"{run_1.model_name} to "
                f"{run_2.model_name}."
            )

        # ---------------------------------------------------------
        # Performance
        # ---------------------------------------------------------

        if summary:

            cause_parts.append(
                "Performance: "
                + summary
            )

        # ---------------------------------------------------------
        # Final cause analysis
        # ---------------------------------------------------------

        if cause_parts:

            cause_analysis = " ".join(
                cause_parts
            )

        else:

            cause_analysis = (
                "No significant differences were "
                "identified from the available run metadata."
            )

        # ---------------------------------------------------------
        # Detailed explanation list
        # ---------------------------------------------------------

        details = []

        if dataset_details:

            if dataset_details.get("error"):

                details.append({
                    "category": "dataset",
                    "message": (
                        "Dataset changed, but detailed "
                        "dataset information could not "
                        "be retrieved."
                    ),
                })

            else:

                old_rows = dataset_details.get(
                    "old_rows"
                )

                new_rows = dataset_details.get(
                    "new_rows"
                )

                rows_added = dataset_details.get(
                    "rows_added",
                    0,
                )

                rows_removed = dataset_details.get(
                    "rows_removed",
                    0,
                )

                columns_added = dataset_details.get(
                    "columns_added",
                    [],
                )

                columns_removed = dataset_details.get(
                    "columns_removed",
                    [],
                )

                message = (
                    f"The dataset changed from "
                    f"{old_rows} rows to "
                    f"{new_rows} rows."
                )

                if rows_added:

                    message += (
                        f" {rows_added} row(s) added."
                    )

                if rows_removed:

                    message += (
                        f" {rows_removed} row(s) removed."
                    )

                if columns_added:

                    message += (
                        " Columns added: "
                        + ", ".join(columns_added)
                        + "."
                    )

                if columns_removed:

                    message += (
                        " Columns removed: "
                        + ", ".join(columns_removed)
                        + "."
                    )

                details.append({
                    "category": "dataset",
                    "message": message,
                })

        if code_details:

            if code_details.get("error"):

                details.append({
                    "category": "code",
                    "message": (
                        "Code changed between the two "
                        "commits, but detailed code analysis "
                        "could not be completed."
                    ),
                })

            else:

                feature_details = (
                    code_details.get(
                        "feature_details"
                    )
                )

                if feature_details:

                    added = feature_details.get(
                        "added",
                        [],
                    )

                    removed = feature_details.get(
                        "removed",
                        [],
                    )

                    if added:

                        details.append({
                            "category": "features",
                            "message": (
                                "The following features "
                                "were added: "
                                + ", ".join(added)
                                + "."
                            ),
                        })

                    if removed:

                        details.append({
                            "category": "features",
                            "message": (
                                "The following features "
                                "were removed: "
                                + ", ".join(removed)
                                + "."
                            ),
                        })

        if changes["parameters_changed"]:

            details.append({
                "category": "parameters",
                "message": (
                    "Training parameters changed "
                    "between the two runs."
                ),
            })

        if changes["model_changed"]:

            details.append({
                "category": "model",
                "message": (
                    f"The model changed from "
                    f"{run_1.model_name} to "
                    f"{run_2.model_name}."
                ),
            })

        if summary:

            details.append({
                "category": "performance",
                "message": summary,
            })

        # ---------------------------------------------------------
        # Return explanation
        # ---------------------------------------------------------

        return {
            "overall": overall,
            "summary": summary,
            "possible_causes": possible_causes,
            "cause_analysis": cause_analysis,
            "details": details,
        }