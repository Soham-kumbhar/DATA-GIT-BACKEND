class SmartExplanationService:

    @staticmethod
    def generate(
        comparison: dict,
        code_analysis: dict | None = None,
        dataset_diff: dict | None = None,
    ) -> dict:

        changes = comparison.get("changes", {})
        metrics = comparison.get("metrics", {})

        explanations = []
        summary_parts = []

        # --------------------------------------------------
        # 1. Dataset explanation
        # --------------------------------------------------

        if changes.get("dataset_changed"):
            if dataset_diff:
                rows_added = dataset_diff.get("rows_added", 0)
                rows_removed = dataset_diff.get("rows_removed", 0)
                columns_added = dataset_diff.get("columns_added", [])
                columns_removed = dataset_diff.get("columns_removed", [])

                dataset_text = "The dataset changed."

                if rows_added:
                    dataset_text += (
                        f" {rows_added} row(s) were added."
                    )

                if rows_removed:
                    dataset_text += (
                        f" {rows_removed} row(s) were removed."
                    )

                if columns_added:
                    dataset_text += (
                        f" New columns were added: "
                        f"{', '.join(columns_added)}."
                    )

                if columns_removed:
                    dataset_text += (
                        f" Columns were removed: "
                        f"{', '.join(columns_removed)}."
                    )

                explanations.append({
                    "category": "dataset",
                    "message": dataset_text,
                    "details": dataset_diff,
                })

                summary_parts.append(dataset_text)

            else:
                explanations.append({
                    "category": "dataset",
                    "message": (
                        "The dataset version changed, but detailed "
                        "dataset information is not available."
                    ),
                })

        # --------------------------------------------------
        # 2. Feature explanation
        # --------------------------------------------------

        feature_details = None

        if code_analysis:
            feature_details = code_analysis.get(
                "feature_details"
            )

        if changes.get("features_changed") and feature_details:

            added = feature_details.get("added", [])
            removed = feature_details.get("removed", [])

            feature_messages = []

            if added:
                feature_messages.append(
                    f"Added feature(s): {', '.join(added)}."
                )

            if removed:
                feature_messages.append(
                    f"Removed feature(s): {', '.join(removed)}."
                )

            feature_message = " ".join(feature_messages)

            explanations.append({
                "category": "features",
                "message": feature_message,
                "details": feature_details,
            })

            summary_parts.append(feature_message)

        # --------------------------------------------------
        # 3. Code explanation
        # --------------------------------------------------

        if changes.get("code_changed") and code_analysis:

            code_changes = code_analysis.get("changes", [])

            changed_files = code_analysis.get(
                "files_changed",
                [],
            )

            code_message = (
                "Code changes were detected in: "
                + ", ".join(changed_files)
                + "."
            )

            if code_changes:
                categories = sorted(
                    set(
                        change.get("category")
                        for change in code_changes
                        if change.get("category")
                    )
                )

                if categories:
                    code_message += (
                        " The changes affect: "
                        + ", ".join(categories)
                        + "."
                    )

            explanations.append({
                "category": "code",
                "message": code_message,
                "details": code_analysis,
            })

            summary_parts.append(code_message)

        # --------------------------------------------------
        # 4. Model explanation
        # --------------------------------------------------

        if changes.get("model_changed"):
            model_message = (
                "The machine learning model or model "
                "configuration changed."
            )

            explanations.append({
                "category": "model",
                "message": model_message,
            })

            summary_parts.append(model_message)

        # --------------------------------------------------
        # 5. Parameter explanation
        # --------------------------------------------------

        if changes.get("parameters_changed"):
            parameter_message = (
                "Training parameters changed between "
                "the two runs."
            )

            explanations.append({
                "category": "parameters",
                "message": parameter_message,
            })

            summary_parts.append(parameter_message)

        # --------------------------------------------------
        # 6. Preprocessing explanation
        # --------------------------------------------------

        if (
            code_analysis
            and code_analysis.get("preprocessing_changed")
        ):
            preprocessing_message = (
                "Preprocessing logic changed between "
                "the two runs."
            )

            explanations.append({
                "category": "preprocessing",
                "message": preprocessing_message,
            })

            summary_parts.append(preprocessing_message)

        # --------------------------------------------------
        # 7. Training explanation
        # --------------------------------------------------

        if (
            code_analysis
            and code_analysis.get("training_changed")
        ):
            training_message = (
                "The model training logic changed between "
                "the two runs."
            )

            explanations.append({
                "category": "training",
                "message": training_message,
            })

            summary_parts.append(training_message)

        # --------------------------------------------------
        # 8. Metric explanation
        # --------------------------------------------------

        metric_explanations = []

        rmse = metrics.get("rmse")

        if rmse:
            before = rmse.get("before")
            after = rmse.get("after")
            change = rmse.get("change")

            if before is not None and after is not None:

                if change > 0:
                    message = (
                        f"RMSE increased from {before} "
                        f"to {after}, a change of +{change}."
                    )
                elif change < 0:
                    message = (
                        f"RMSE decreased from {before} "
                        f"to {after}, a change of {change}."
                    )
                else:
                    message = (
                        f"RMSE remained unchanged at {after}."
                    )

                metric_explanations.append(message)

        r2 = metrics.get("r2")

        if r2:
            before = r2.get("before")
            after = r2.get("after")
            change = r2.get("change")

            if before is not None and after is not None:

                if change > 0:
                    message = (
                        f"R² increased from {before} "
                        f"to {after}, a change of +{change}."
                    )
                elif change < 0:
                    message = (
                        f"R² decreased from {before} "
                        f"to {after}, a change of {change}."
                    )
                else:
                    message = (
                        f"R² remained unchanged at {after}."
                    )

                metric_explanations.append(message)

        if metric_explanations:
            explanations.append({
                "category": "performance",
                "message": " ".join(metric_explanations),
            })

            summary_parts.extend(metric_explanations)

        # --------------------------------------------------
        # 9. Overall assessment
        # --------------------------------------------------

        performance_direction = "unchanged"

        if rmse:
            rmse_change = rmse.get("change", 0)

            if rmse_change > 0:
                performance_direction = "worse"
            elif rmse_change < 0:
                performance_direction = "better"

        if r2:
            r2_change = r2.get("change", 0)

            if r2_change < 0:
                performance_direction = "worse"
            elif (
                r2_change > 0
                and performance_direction == "unchanged"
            ):
                performance_direction = "better"

        if performance_direction == "worse":
            overall = (
                "Model performance decreased between the "
                "two runs."
            )
        elif performance_direction == "better":
            overall = (
                "Model performance improved between the "
                "two runs."
            )
        else:
            overall = (
                "Model performance remained approximately "
                "unchanged between the two runs."
            )

        # --------------------------------------------------
        # 10. Possible relationship
        # --------------------------------------------------

        possible_causes = []

        if changes.get("dataset_changed"):
            possible_causes.append("dataset changes")

        if changes.get("features_changed"):
            possible_causes.append("feature changes")

        if changes.get("model_changed"):
            possible_causes.append("model changes")

        if changes.get("parameters_changed"):
            possible_causes.append("parameter changes")

        if (
            code_analysis
            and code_analysis.get("preprocessing_changed")
        ):
            possible_causes.append("preprocessing changes")

        if (
            code_analysis
            and code_analysis.get("training_changed")
        ):
            possible_causes.append("training logic changes")

        if possible_causes:
            cause_text = (
                "The performance difference occurred alongside "
                + ", ".join(possible_causes)
                + ". These changes should be investigated as "
                  "possible contributors to the performance difference."
            )
        else:
            cause_text = (
                "No specific code, dataset, or configuration "
                "change was identified as a possible cause."
            )

        return {
            "overall": overall,
            "summary": " ".join(summary_parts),
            "possible_causes": possible_causes,
            "cause_analysis": cause_text,
            "details": explanations,
        }