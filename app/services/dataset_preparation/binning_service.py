from pathlib import Path

import pandas as pd


class BinningService:
    """
    Divide numeric columns into discrete bins.

    Supported methods:
    - equal_width
    - quantile

    The original column is replaced by the bin labels.
    """

    SUPPORTED_METHODS = {
        "equal_width",
        "quantile",
    }

    @staticmethod
    def validate(
        dataframe: pd.DataFrame,
        columns: list[str],
        bins: int,
        method: str = "equal_width",
        labels: list[str] | None = None,
    ) -> list[str]:
        errors: list[str] = []

        if not columns:
            errors.append(
                "At least one column must be selected for binning."
            )

        if not isinstance(bins, int):
            errors.append(
                "bins must be an integer."
            )
        elif bins < 2:
            errors.append(
                "bins must be at least 2."
            )

        if method not in BinningService.SUPPORTED_METHODS:
            errors.append(
                (
                    "Unsupported binning method: "
                    f"{method}. Supported methods: "
                    f"{sorted(BinningService.SUPPORTED_METHODS)}."
                )
            )

        missing_columns = [
            column
            for column in columns
            if column not in dataframe.columns
        ]

        if missing_columns:
            errors.append(
                (
                    "Selected columns were not found in the dataset: "
                    f"{missing_columns}"
                )
            )

        for column in columns:
            if column not in dataframe.columns:
                continue

            if not pd.api.types.is_numeric_dtype(
                dataframe[column]
            ):
                errors.append(
                    (
                        f"Column '{column}' must be numeric "
                        "for binning."
                    )
                )

        if labels is not None:
            if not isinstance(labels, list):
                errors.append(
                    "labels must be a list of strings."
                )
            else:
                if len(labels) != bins:
                    errors.append(
                        (
                            "The number of labels must match "
                            f"the number of bins. Expected {bins}, "
                            f"received {len(labels)}."
                        )
                    )

                if not all(
                    isinstance(label, str)
                    for label in labels
                ):
                    errors.append(
                        "All bin labels must be strings."
                    )

        if not errors:
            for column in columns:
                series = dataframe[column].dropna()

                if series.empty:
                    errors.append(
                        (
                            f"Column '{column}' contains no "
                            "valid numeric values."
                        )
                    )
                    continue

                if series.nunique() < 2:
                    errors.append(
                        (
                            f"Column '{column}' must contain at "
                            "least two distinct values for binning."
                        )
                    )

        return errors

    @staticmethod
    def apply_binning(
        dataframe: pd.DataFrame,
        columns: list[str],
        bins: int,
        method: str = "equal_width",
        labels: list[str] | None = None,
    ) -> tuple[pd.DataFrame, dict]:
        validation_errors = BinningService.validate(
            dataframe=dataframe,
            columns=columns,
            bins=bins,
            method=method,
            labels=labels,
        )

        if validation_errors:
            raise ValueError(
                "; ".join(validation_errors)
            )

        result = dataframe.copy()
        column_results = {}

        for column in columns:
            original_series = result[column]

            if method == "equal_width":
                binned = pd.cut(
                    original_series,
                    bins=bins,
                    labels=labels,
                    include_lowest=True,
                    duplicates="drop",
                )

            else:
                binned = pd.qcut(
                    original_series,
                    q=bins,
                    labels=labels,
                    duplicates="drop",
                )

            result[column] = binned

            non_null_before = int(
                original_series.notna().sum()
            )

            non_null_after = int(
                binned.notna().sum()
            )

            column_results[column] = {
                "method": method,
                "bins": bins,
                "labels": labels,
                "original_non_null_values": non_null_before,
                "binned_non_null_values": non_null_after,
                "unique_bins_created": int(
                    binned.dropna().nunique()
                ),
            }

        details = {
            "method": "binning",
            "binning_method": method,
            "columns": columns,
            "bins": bins,
            "labels": labels,
            "column_results": column_results,
            "data_modified": True,
        }

        return result, details

    @staticmethod
    def execute(
        file_path: str,
        columns: list[str],
        bins: int,
        method: str = "equal_width",
        labels: list[str] | None = None,
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
            BinningService.apply_binning(
                dataframe=dataframe,
                columns=columns,
                bins=bins,
                method=method,
                labels=labels,
            )
        )

        output_path = (
            input_path.parent
            / (
                f"{input_path.stem}"
                "_binned"
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
            "output_format": input_path.suffix.lower().lstrip("."),
            "rows_before": rows_before,
            "rows_after": len(result_dataframe),
            "columns_before": columns_before,
            "columns_after": len(result_dataframe.columns),
            "details": details,
            "output_path": str(output_path),
        }