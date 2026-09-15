from __future__ import annotations

from pathlib import Path

import pandas as pd


class ZScoreScalingService:
    """
    Scale selected numeric columns using Z-score standardization.

    Formula:

        z = (x - mean) / standard_deviation

    Default behavior uses sample standard deviation (ddof=1).

    Example:

    {
        "columns": [
            "age",
            "income"
        ]
    }
    """

    @staticmethod
    def validate(
        dataframe: pd.DataFrame,
        columns: list,
    ) -> list[str]:
        errors: list[str] = []

        if not isinstance(columns, list):
            return [
                "z_score_scaling 'columns' must be a list."
            ]

        if not columns:
            return [
                "z_score_scaling requires at least one column."
            ]

        missing_columns = [
            column
            for column in columns
            if column not in dataframe.columns
        ]

        if missing_columns:
            errors.append(
                "One or more scaling columns do not exist: "
                f"{missing_columns}"
            )

        non_numeric_columns = [
            column
            for column in columns
            if column in dataframe.columns
            and not pd.api.types.is_numeric_dtype(
                dataframe[column]
            )
        ]

        if non_numeric_columns:
            errors.append(
                "Z-score scaling requires numeric columns. "
                f"Non-numeric columns: {non_numeric_columns}"
            )

        return errors

    @staticmethod
    def scale(
        dataframe: pd.DataFrame,
        columns: list,
    ) -> tuple[pd.DataFrame, dict]:

        errors = ZScoreScalingService.validate(
            dataframe=dataframe,
            columns=columns,
        )

        if errors:
            raise ValueError(
                "; ".join(errors)
            )

        result = dataframe.copy()

        column_results = {}

        for column in columns:

            original = result[column].copy()

            mean_value = original.mean()

            standard_deviation = original.std(
                ddof=1
            )

            if pd.isna(mean_value):
                raise ValueError(
                    f"Column '{column}' does not contain "
                    "usable numeric values."
                )

            if pd.isna(standard_deviation):
                raise ValueError(
                    f"Column '{column}' requires at least "
                    "two valid numeric values for Z-score scaling."
                )

            if standard_deviation == 0:
                result[column] = 0.0

                constant_column = True

            else:
                result[column] = (
                    original - mean_value
                ) / standard_deviation

                constant_column = False

            column_results[column] = {
                "original_mean": float(mean_value),
                "original_standard_deviation": float(
                    standard_deviation
                ),
                "standardized_mean": 0.0,
                "standardized_standard_deviation": (
                    0.0
                    if constant_column
                    else 1.0
                ),
                "constant_column": constant_column,
            }

        details = {
            "method": "z_score_scaling",
            "columns": columns,
            "ddof": 1,
            "column_results": column_results,
            "data_modified": True,
        }

        return result, details

    @staticmethod
    def execute(
        file_path: str | Path,
        columns: list,
    ) -> dict:

        input_path = Path(file_path)

        if not input_path.exists():
            raise ValueError(
                "Dataset file was not found."
            )

        if input_path.suffix.lower() != ".csv":
            raise ValueError(
                "Z-score scaling currently supports "
                "CSV input files."
            )

        try:
            dataframe = pd.read_csv(
                input_path
            )

        except Exception as error:
            raise ValueError(
                f"Could not read CSV dataset: {error}"
            ) from error

        rows_before = len(dataframe)
        columns_before = len(dataframe.columns)

        dataframe, details = (
            ZScoreScalingService.scale(
                dataframe=dataframe,
                columns=columns,
            )
        )

        output_path = (
            input_path.parent
            / f"{input_path.stem}_z_score_scaled.csv"
        )

        dataframe.to_csv(
            output_path,
            index=False,
        )

        return {
            "status": "success",
            "input_file": input_path.name,
            "output_file": output_path.name,
            "output_path": str(output_path),
            "rows_before": rows_before,
            "rows_after": len(dataframe),
            "columns_before": columns_before,
            "columns_after": len(dataframe.columns),
            "details": details,
        }