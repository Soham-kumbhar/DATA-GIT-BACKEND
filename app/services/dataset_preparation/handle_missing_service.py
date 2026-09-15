from pathlib import Path

import pandas as pd


class HandleMissingService:

    SUPPORTED_STRATEGIES = {
        "drop_rows",
        "drop_column",
        "mean",
        "median",
        "mode",
        "constant",
        "forward_fill",
        "backward_fill",
    }

    @staticmethod
    def execute(
        file_path: str,
        columns: dict[str, str],
        constant_values: dict | None = None,
    ) -> dict:
        path = Path(file_path)

        if not path.exists():
            raise ValueError(
                "Dataset file was not found."
            )

        if path.suffix.lower() != ".csv":
            raise ValueError(
                "Handle missing values currently supports CSV files."
            )

        try:
            dataframe = pd.read_csv(path)

        except Exception as error:
            raise ValueError(
                f"Could not read CSV dataset: {error}"
            ) from error

        if not isinstance(columns, dict) or not columns:
            raise ValueError(
                "At least one column strategy must be provided."
            )

        constant_values = constant_values or {}

        rows_before = len(dataframe)
        columns_before = len(dataframe.columns)

        missing_before = int(
            dataframe.isna().sum().sum()
        )

        handled_columns = {}

        for column, strategy in columns.items():

            if column not in dataframe.columns:
                raise ValueError(
                    f"Column '{column}' does not exist."
                )

            if strategy not in (
                HandleMissingService.SUPPORTED_STRATEGIES
            ):
                raise ValueError(
                    f"Unsupported missing-value strategy "
                    f"'{strategy}' for column '{column}'."
                )

            missing_before_column = int(
                dataframe[column].isna().sum()
            )

            # ----------------------------------------------------
            # DROP COLUMN
            # ----------------------------------------------------

            if strategy == "drop_column":

                dataframe = dataframe.drop(
                    columns=[column]
                )

                handled_columns[column] = {
                    "strategy": strategy,
                    "missing_before": missing_before_column,
                    "missing_after": 0,
                    "values_handled": missing_before_column,
                    "column_removed": True,
                }

                continue

            # ----------------------------------------------------
            # NO MISSING VALUES
            # ----------------------------------------------------

            if missing_before_column == 0:

                handled_columns[column] = {
                    "strategy": strategy,
                    "missing_before": 0,
                    "missing_after": 0,
                    "values_handled": 0,
                    "column_removed": False,
                }

                continue

            # ----------------------------------------------------
            # DROP ROWS
            # ----------------------------------------------------

            if strategy == "drop_rows":

                dataframe = dataframe.dropna(
                    subset=[column]
                )

            # ----------------------------------------------------
            # MEAN
            # ----------------------------------------------------

            elif strategy == "mean":

                if not pd.api.types.is_numeric_dtype(
                    dataframe[column]
                ):
                    raise ValueError(
                        f"Mean strategy requires numeric "
                        f"column '{column}'."
                    )

                replacement = dataframe[column].mean()

                dataframe[column] = (
                    dataframe[column].fillna(
                        replacement
                    )
                )

            # ----------------------------------------------------
            # MEDIAN
            # ----------------------------------------------------

            elif strategy == "median":

                if not pd.api.types.is_numeric_dtype(
                    dataframe[column]
                ):
                    raise ValueError(
                        f"Median strategy requires numeric "
                        f"column '{column}'."
                    )

                replacement = dataframe[column].median()

                dataframe[column] = (
                    dataframe[column].fillna(
                        replacement
                    )
                )

            # ----------------------------------------------------
            # MODE
            # ----------------------------------------------------

            elif strategy == "mode":

                mode_values = dataframe[column].mode()

                if mode_values.empty:
                    raise ValueError(
                        f"Could not determine mode for "
                        f"column '{column}'."
                    )

                replacement = mode_values.iloc[0]

                dataframe[column] = (
                    dataframe[column].fillna(
                        replacement
                    )
                )

            # ----------------------------------------------------
            # CONSTANT
            # ----------------------------------------------------

            elif strategy == "constant":

                if column not in constant_values:
                    raise ValueError(
                        f"Constant strategy requires a value "
                        f"for column '{column}'."
                    )

                replacement = constant_values[column]

                dataframe[column] = (
                    dataframe[column].fillna(
                        replacement
                    )
                )

            # ----------------------------------------------------
            # FORWARD FILL
            # ----------------------------------------------------

            elif strategy == "forward_fill":

                dataframe[column] = (
                    dataframe[column].ffill()
                )

            # ----------------------------------------------------
            # BACKWARD FILL
            # ----------------------------------------------------

            elif strategy == "backward_fill":

                dataframe[column] = (
                    dataframe[column].bfill()
                )

            missing_after_column = int(
                dataframe[column].isna().sum()
            )

            handled_columns[column] = {
                "strategy": strategy,
                "missing_before": missing_before_column,
                "missing_after": missing_after_column,
                "values_handled": (
                    missing_before_column
                    - missing_after_column
                ),
                "column_removed": False,
            }

        rows_after = len(dataframe)
        columns_after = len(dataframe.columns)

        missing_after = int(
            dataframe.isna().sum().sum()
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
            / f"{path.stem}_missing_handled.csv"
        )

        dataframe.to_csv(
            output_path,
            index=False,
        )

        return {
            "input_file": path.name,
            "output_file": output_path.name,
            "strategies": handled_columns,
            "rows_before": rows_before,
            "rows_after": rows_after,
            "columns_before": columns_before,
            "columns_after": columns_after,
            "missing_before": missing_before,
            "missing_after": missing_after,
            "values_handled": (
                missing_before - missing_after
            ),
            "status": "success",
            "output_path": str(output_path),
        }