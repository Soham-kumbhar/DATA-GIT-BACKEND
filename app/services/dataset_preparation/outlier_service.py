from math import log, log1p
from pathlib import Path

import pandas as pd


class OutlierHandlingService:

    @staticmethod
    def _validate_columns(
        dataframe: pd.DataFrame,
        columns: list[str],
    ) -> None:
        if not isinstance(columns, list):
            raise ValueError(
                "'columns' must be a list of column names."
            )

        if not columns:
            raise ValueError(
                "At least one column must be provided."
            )

        missing_columns = [
            column
            for column in columns
            if column not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                "One or more columns do not exist: "
                f"{missing_columns}"
            )

        non_numeric_columns = [
            column
            for column in columns
            if not pd.api.types.is_numeric_dtype(
                dataframe[column]
            )
        ]

        if non_numeric_columns:
            raise ValueError(
                "Outlier operations require numeric columns. "
                f"Non-numeric columns: {non_numeric_columns}"
            )

    @staticmethod
    def _validate_iqr_multiplier(
        iqr_multiplier: float,
    ) -> float:
        try:
            value = float(iqr_multiplier)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "'iqr_multiplier' must be a number."
            ) from error

        if value <= 0:
            raise ValueError(
                "'iqr_multiplier' must be greater than 0."
            )

        return value

    @staticmethod
    def _calculate_bounds(
        series: pd.Series,
        iqr_multiplier: float,
    ) -> tuple[float, float]:
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))

        iqr = q3 - q1

        lower_bound = q1 - (
            iqr_multiplier * iqr
        )

        upper_bound = q3 + (
            iqr_multiplier * iqr
        )

        return lower_bound, upper_bound

    @staticmethod
    def trim(
        dataframe: pd.DataFrame,
        columns: list[str],
        iqr_multiplier: float = 1.5,
    ) -> tuple[pd.DataFrame, dict]:
        OutlierHandlingService._validate_columns(
            dataframe,
            columns,
        )

        multiplier = (
            OutlierHandlingService._validate_iqr_multiplier(
                iqr_multiplier
            )
        )

        result = dataframe.copy()

        rows_before = len(result)

        keep_mask = pd.Series(
            True,
            index=result.index,
        )

        column_results = {}

        for column in columns:
            lower_bound, upper_bound = (
                OutlierHandlingService._calculate_bounds(
                    result[column],
                    multiplier,
                )
            )

            outlier_mask = (
                (result[column] < lower_bound)
                | (result[column] > upper_bound)
            )

            outliers_found = int(
                outlier_mask.sum()
            )

            keep_mask &= ~outlier_mask

            column_results[column] = {
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outliers_found": outliers_found,
            }

        result = result.loc[
            keep_mask
        ].copy()

        return result, {
            "method": "trim",
            "iqr_multiplier": multiplier,
            "columns": columns,
            "rows_before": rows_before,
            "rows_after": len(result),
            "rows_removed": (
                rows_before - len(result)
            ),
            "column_results": column_results,
        }

    @staticmethod
    def winsorize(
        dataframe: pd.DataFrame,
        columns: list[str],
        iqr_multiplier: float = 1.5,
    ) -> tuple[pd.DataFrame, dict]:
        OutlierHandlingService._validate_columns(
            dataframe,
            columns,
        )

        multiplier = (
            OutlierHandlingService._validate_iqr_multiplier(
                iqr_multiplier
            )
        )

        result = dataframe.copy()

        column_results = {}

        for column in columns:
            lower_bound, upper_bound = (
                OutlierHandlingService._calculate_bounds(
                    result[column],
                    multiplier,
                )
            )

            below_count = int(
                (
                    result[column] < lower_bound
                ).sum()
            )

            above_count = int(
                (
                    result[column] > upper_bound
                ).sum()
            )

            result[column] = (
                result[column].clip(
                    lower=lower_bound,
                    upper=upper_bound,
                )
            )

            column_results[column] = {
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "values_below_lower_bound": (
                    below_count
                ),
                "values_above_upper_bound": (
                    above_count
                ),
                "values_capped": (
                    below_count + above_count
                ),
            }

        return result, {
            "method": "winsorize",
            "iqr_multiplier": multiplier,
            "columns": columns,
            "rows_before": len(dataframe),
            "rows_after": len(result),
            "rows_removed": 0,
            "column_results": column_results,
        }

    @staticmethod
    def log_transform(
        dataframe: pd.DataFrame,
        columns: list[str],
    ) -> tuple[pd.DataFrame, dict]:
        OutlierHandlingService._validate_columns(
            dataframe,
            columns,
        )

        result = dataframe.copy()

        column_results = {}

        for column in columns:
            minimum_value = result[column].min()

            if pd.isna(minimum_value):
                raise ValueError(
                    f"Column '{column}' contains no usable values."
                )

            minimum_value = float(
                minimum_value
            )

            if minimum_value >= 0:
                result[column] = result[column].apply(
                    lambda value: (
                        pd.NA
                        if pd.isna(value)
                        else log1p(float(value))
                    )
                )

                transformation = "log1p"
                shift = 0.0

            else:
                shift = 1.0 - minimum_value

                result[column] = result[column].apply(
                    lambda value: (
                        pd.NA
                        if pd.isna(value)
                        else log(
                            float(value) + shift
                        )
                    )
                )

                transformation = "log_shifted"

            column_results[column] = {
                "minimum_before": minimum_value,
                "shift": shift,
                "transformation": transformation,
            }

        return result, {
            "method": "log_transform",
            "columns": columns,
            "rows_before": len(dataframe),
            "rows_after": len(result),
            "rows_removed": 0,
            "column_results": column_results,
        }

    @staticmethod
    def execute(
        file_path: str,
        method: str,
        columns: list[str],
        iqr_multiplier: float = 1.5,
    ) -> dict:
        path = Path(file_path)

        if not path.exists():
            raise ValueError(
                "Dataset file was not found."
            )

        if path.suffix.lower() != ".csv":
            raise ValueError(
                "Outlier handling currently supports CSV files."
            )

        try:
            dataframe = pd.read_csv(path)

        except Exception as error:
            raise ValueError(
                f"Could not read CSV dataset: {error}"
            ) from error

        if method == "trim":
            processed, details = (
                OutlierHandlingService.trim(
                    dataframe=dataframe,
                    columns=columns,
                    iqr_multiplier=iqr_multiplier,
                )
            )

        elif method == "winsorize":
            processed, details = (
                OutlierHandlingService.winsorize(
                    dataframe=dataframe,
                    columns=columns,
                    iqr_multiplier=iqr_multiplier,
                )
            )

        elif method == "log_transform":
            processed, details = (
                OutlierHandlingService.log_transform(
                    dataframe=dataframe,
                    columns=columns,
                )
            )

        else:
            raise ValueError(
                "Unsupported outlier method. "
                "Use 'trim', 'winsorize', or "
                "'log_transform'."
            )

        output_directory = (
            Path("dataset_preparations")
            / "prepared"
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            output_directory
            / f"{path.stem}_outliers_handled.csv"
        )

        processed.to_csv(
            output_path,
            index=False,
        )

        return {
            "input_file": path.name,
            "output_file": output_path.name,
            "method": method,
            **details,
            "status": "success",
            "output_path": str(output_path),
        }