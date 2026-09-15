from pathlib import Path

import pandas as pd


class UndersamplingService:
    """
    Balance a classification dataset by randomly undersampling
    majority classes until every class reaches the minority-class
    sample count.

    The target column is required.
    """

    @staticmethod
    def validate(
        dataframe: pd.DataFrame,
        target_column: str,
        random_state: int = 42,
    ) -> list[str]:
        errors: list[str] = []

        if not target_column:
            errors.append(
                "target_column is required."
            )

        elif target_column not in dataframe.columns:
            errors.append(
                f"Target column '{target_column}' does not exist."
            )

        if not isinstance(random_state, int):
            errors.append(
                "random_state must be an integer."
            )

        if target_column in dataframe.columns:
            target_series = dataframe[target_column]

            if target_series.isna().any():
                errors.append(
                    "Target column must not contain missing values "
                    "for undersampling."
                )

            if target_series.nunique(dropna=True) < 2:
                errors.append(
                    "Undersampling requires at least two target classes."
                )

        return errors

    @staticmethod
    def undersample(
        dataframe: pd.DataFrame,
        target_column: str,
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, dict]:
        validation_errors = UndersamplingService.validate(
            dataframe=dataframe,
            target_column=target_column,
            random_state=random_state,
        )

        if validation_errors:
            raise ValueError(
                "; ".join(validation_errors)
            )

        original_class_counts = (
            dataframe[target_column]
            .value_counts()
            .sort_index()
        )

        target_count = int(
            original_class_counts.min()
        )

        class_frames = []
        class_results = {}

        for class_value, class_count in (
            original_class_counts.items()
        ):
            class_dataframe = dataframe[
                dataframe[target_column] == class_value
            ].copy()

            class_count = int(class_count)

            rows_removed = (
                class_count - target_count
            )

            if class_count > target_count:
                balanced_class_dataframe = (
                    class_dataframe.sample(
                        n=target_count,
                        replace=False,
                        random_state=random_state,
                    )
                )
            else:
                balanced_class_dataframe = (
                    class_dataframe.copy()
                )

            class_frames.append(
                balanced_class_dataframe
            )

            class_results[str(class_value)] = {
                "original_count": class_count,
                "target_count": target_count,
                "rows_removed": rows_removed,
                "final_count": len(
                    balanced_class_dataframe
                ),
            }

        result = pd.concat(
            class_frames,
            ignore_index=True,
        )

        result = result.sample(
            frac=1,
            random_state=random_state,
        ).reset_index(
            drop=True
        )

        final_class_counts = (
            result[target_column]
            .value_counts()
            .sort_index()
        )

        details = {
            "method": "undersampling",
            "target_column": target_column,
            "random_state": random_state,
            "original_class_distribution": {
                str(class_value): int(class_count)
                for class_value, class_count
                in original_class_counts.items()
            },
            "target_class_count": target_count,
            "final_class_distribution": {
                str(class_value): int(class_count)
                for class_value, class_count
                in final_class_counts.items()
            },
            "class_results": class_results,
            "rows_removed": (
                len(dataframe) - len(result)
            ),
            "data_modified": True,
        }

        return result, details

    @staticmethod
    def execute(
        file_path: str,
        target_column: str,
        random_state: int = 42,
    ) -> dict:
        input_path = Path(file_path)

        if not input_path.exists():
            raise ValueError(
                "Dataset file was not found."
            )

        if input_path.suffix.lower() == ".csv":
            dataframe = pd.read_csv(
                input_path
            )

        elif input_path.suffix.lower() == ".json":
            dataframe = pd.read_json(
                input_path
            )

        else:
            raise ValueError(
                "Only CSV and JSON datasets are supported."
            )

        rows_before = len(dataframe)
        columns_before = len(dataframe.columns)

        result_dataframe, details = (
            UndersamplingService.undersample(
                dataframe=dataframe,
                target_column=target_column,
                random_state=random_state,
            )
        )

        output_path = (
            input_path.parent
            / (
                f"{input_path.stem}"
                "_undersampled"
                f"{input_path.suffix}"
            )
        )

        if input_path.suffix.lower() == ".csv":
            result_dataframe.to_csv(
                output_path,
                index=False,
            )

        else:
            result_dataframe.to_json(
                output_path,
                orient="records",
            )

        return {
            "status": "success",
            "input_file": input_path.name,
            "output_file": output_path.name,
            "output_format": (
                input_path.suffix
                .lower()
                .lstrip(".")
            ),
            "rows_before": rows_before,
            "rows_after": len(result_dataframe),
            "columns_before": columns_before,
            "columns_after": len(result_dataframe.columns),
            "details": details,
            "output_path": str(output_path),
        }