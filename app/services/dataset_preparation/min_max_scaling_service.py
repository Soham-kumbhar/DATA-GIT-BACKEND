from __future__ import annotations

from pathlib import Path

import pandas as pd


class MinMaxScalingService:
    """
    Scale selected numeric columns using Min-Max normalization.

    Default range:
        0 to 1

    Example:

    {
        "columns": [
            "age",
            "income"
        ],
        "feature_range": [
            0,
            1
        ]
    }
    """

    @staticmethod
    def validate(
        dataframe: pd.DataFrame,
        columns: list,
        feature_range: list,
    ) -> list[str]:
        errors: list[str] = []

        if not isinstance(columns, list):
            errors.append(
                "min_max_scaling 'columns' must be a list."
            )
            return errors

        if not columns:
            errors.append(
                "min_max_scaling requires at least one column."
            )
            return errors

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
                "Min-Max scaling requires numeric columns. "
                f"Non-numeric columns: {non_numeric_columns}"
            )

        if not isinstance(feature_range, list):
            errors.append(
                "min_max_scaling 'feature_range' must be a list."
            )
            return errors

        if len(feature_range) != 2:
            errors.append(
                "min_max_scaling 'feature_range' must contain "
                "exactly two values: [minimum, maximum]."
            )
            return errors

        try:
            lower = float(feature_range[0])
            upper = float(feature_range[1])

            if lower >= upper:
                errors.append(
                    "min_max_scaling feature_range minimum "
                    "must be less than maximum."
                )

        except (TypeError, ValueError):
            errors.append(
                "min_max_scaling feature_range values "
                "must be numbers."
            )

        return errors

    @staticmethod
    def scale(
        dataframe: pd.DataFrame,
        columns: list,
        feature_range: list | None = None,
    ) -> tuple[pd.DataFrame, dict]:

        if feature_range is None:
            feature_range = [0, 1]

        errors = MinMaxScalingService.validate(
            dataframe=dataframe,
            columns=columns,
            feature_range=feature_range,
        )

        if errors:
            raise ValueError(
                "; ".join(errors)
            )

        lower = float(feature_range[0])
        upper = float(feature_range[1])

        result = dataframe.copy()

        column_results = {}

        for column in columns:
            original = result[column].copy()

            column_min = original.min()
            column_max = original.max()

            if pd.isna(column_min) or pd.isna(column_max):
                raise ValueError(
                    f"Column '{column}' does not contain "
                    "usable numeric values."
                )

            if column_max == column_min:
                result[column] = lower

                constant_column = True

            else:
                result[column] = (
                    (
                        original - column_min
                    )
                    /
                    (
                        column_max - column_min
                    )
                ) * (
                    upper - lower
                ) + lower

                constant_column = False

            column_results[column] = {
                "original_min": float(column_min),
                "original_max": float(column_max),
                "scaled_min": lower,
                "scaled_max": upper,
                "constant_column": constant_column,
            }

        details = {
            "method": "min_max_scaling",
            "columns": columns,
            "feature_range": [
                lower,
                upper,
            ],
            "column_results": column_results,
            "data_modified": True,
        }

        return result, details

    @staticmethod
    def execute(
        file_path: str | Path,
        columns: list,
        feature_range: list | None = None,
    ) -> dict:

        input_path = Path(file_path)

        if not input_path.exists():
            raise ValueError(
                "Dataset file was not found."
            )

        if input_path.suffix.lower() != ".csv":
            raise ValueError(
                "Min-Max scaling currently supports "
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
            MinMaxScalingService.scale(
                dataframe=dataframe,
                columns=columns,
                feature_range=feature_range,
            )
        )

        output_path = (
            input_path.parent
            / f"{input_path.stem}_min_max_scaled.csv"
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