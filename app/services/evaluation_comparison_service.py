from typing import Any


class EvaluationComparisonService:

    @staticmethod
    def compare(
        evaluation_before: dict[str, Any] | None,
        evaluation_after: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not evaluation_before and not evaluation_after:
            return {
                "available": False,
                "reason": "No evaluation evidence recorded for either ML run.",
            }

        if not evaluation_before:
            return {
                "available": True,
                "before": None,
                "after": evaluation_after,
                "changes": [],
                "changed": False,
                "status": "after_only",
                "summary": (
                    "Evaluation evidence is available for the newer run, "
                    "but not for the older run."
                ),
            }

        if not evaluation_after:
            return {
                "available": True,
                "before": evaluation_before,
                "after": None,
                "changes": [],
                "changed": False,
                "status": "before_only",
                "summary": (
                    "Evaluation evidence is available for the older run, "
                    "but not for the newer run."
                ),
            }

        task_type_before = evaluation_before.get("task_type")
        task_type_after = evaluation_after.get("task_type")

        confusion_before = EvaluationComparisonService._confusion_matrix(
            evaluation_before
        )
        confusion_after = EvaluationComparisonService._confusion_matrix(
            evaluation_after
        )

        metric_names = [
            "precision",
            "recall",
            "f1",
            "f1_score",
            "error_rate",
            "prediction_count",
            "incorrect_count",
        ]

        metric_changes = {}

        for metric in metric_names:
            before = evaluation_before.get(metric)
            after = evaluation_after.get(metric)

            if before is None and after is None:
                continue

            delta = None

            if (
                isinstance(before, (int, float))
                and isinstance(after, (int, float))
            ):
                delta = after - before

            metric_changes[metric] = {
                "before": before,
                "after": after,
                "delta": delta,
            }

        confusion_changes = {}

        for key in ["tp", "tn", "fp", "fn"]:
            before = confusion_before.get(key)
            after = confusion_after.get(key)

            if before is None and after is None:
                continue

            delta = None

            if (
                isinstance(before, (int, float))
                and isinstance(after, (int, float))
            ):
                delta = after - before

            confusion_changes[key] = {
                "before": before,
                "after": after,
                "delta": delta,
            }

        misclassified_before = evaluation_before.get(
            "misclassified_samples",
            [],
        )

        misclassified_after = evaluation_after.get(
            "misclassified_samples",
            [],
        )

        changed = (
            task_type_before != task_type_after
            or confusion_changes != {}
            and any(
                item["delta"] not in (None, 0)
                for item in confusion_changes.values()
            )
            or any(
                item["delta"] not in (None, 0)
                for item in metric_changes.values()
            )
            or misclassified_before != misclassified_after
        )

        changes = EvaluationComparisonService._build_changes(
            metric_changes=metric_changes,
            confusion_changes=confusion_changes,
            task_type_before=task_type_before,
            task_type_after=task_type_after,
            misclassified_before=misclassified_before,
            misclassified_after=misclassified_after,
        )

        summary = EvaluationComparisonService._build_summary(
            metric_changes=metric_changes,
            confusion_changes=confusion_changes,
            changed=changed,
        )

        return {
            "available": True,
            "status": "both",
            "changed": changed,
            "summary": summary,
            "task_type": {
                "before": task_type_before,
                "after": task_type_after,
                "changed": task_type_before != task_type_after,
            },
            "before": evaluation_before,
            "after": evaluation_after,
            "metrics": metric_changes,
            "confusion_matrix": {
                "before": confusion_before,
                "after": confusion_after,
                "changes": confusion_changes,
            },
            "misclassified_samples": {
                "before": misclassified_before,
                "after": misclassified_after,
                "count_before": len(misclassified_before),
                "count_after": len(misclassified_after),
            },
            "changes": changes,
        }

    @staticmethod
    def _confusion_matrix(
        evaluation: dict[str, Any],
    ) -> dict[str, Any]:
        matrix = evaluation.get("confusion_matrix")

        if not isinstance(matrix, dict):
            return {}

        return {
            "tp": matrix.get("tp"),
            "tn": matrix.get("tn"),
            "fp": matrix.get("fp"),
            "fn": matrix.get("fn"),
        }

    @staticmethod
    def _build_changes(
        metric_changes: dict[str, Any],
        confusion_changes: dict[str, Any],
        task_type_before: str | None,
        task_type_after: str | None,
        misclassified_before: list[Any],
        misclassified_after: list[Any],
    ) -> list[str]:
        changes: list[str] = []

        if task_type_before != task_type_after:
            changes.append(
                f"task type changed from {task_type_before} "
                f"to {task_type_after}"
            )

        for metric, info in metric_changes.items():
            delta = info.get("delta")

            if delta not in (None, 0):
                changes.append(
                    f"{metric} changed by {delta}"
                )

        for key, info in confusion_changes.items():
            delta = info.get("delta")

            if delta not in (None, 0):
                changes.append(
                    f"{key.upper()} changed by {delta}"
                )

        before_count = len(misclassified_before)
        after_count = len(misclassified_after)

        if before_count != after_count:
            changes.append(
                f"misclassified sample count changed "
                f"from {before_count} to {after_count}"
            )

        return changes

    @staticmethod
    def _build_summary(
        metric_changes: dict[str, Any],
        confusion_changes: dict[str, Any],
        changed: bool,
    ) -> str:
        if not changed:
            return "Recorded evaluation evidence did not change."

        accuracy_related = []

        for metric in ["precision", "recall", "f1", "error_rate"]:
            info = metric_changes.get(metric)

            if not info:
                continue

            delta = info.get("delta")

            if delta not in (None, 0):
                accuracy_related.append(
                    f"{metric} changed by {delta}"
                )

        confusion_related = []

        for key in ["fp", "fn", "tp", "tn"]:
            info = confusion_changes.get(key)

            if not info:
                continue

            delta = info.get("delta")

            if delta not in (None, 0):
                confusion_related.append(
                    f"{key.upper()} changed by {delta}"
                )

        parts = accuracy_related + confusion_related

        if not parts:
            return "Evaluation evidence changed."

        return "Evaluation changed: " + "; ".join(parts) + "."