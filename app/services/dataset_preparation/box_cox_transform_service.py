from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import boxcox


class BoxCoxTransformService:

    @staticmethod
    def validate(
        dataframe: pd.DataFrame,
        columns: list,
    ) -> list[str]:

        errors = []

        if not columns:
            errors.append(
                "At least one column must be selected "
                "for Box-Cox transformation."
            )

            return errors

        if not isinstance(
            columns,
            list,
        ):
            errors.append(
                "Box-Cox 'columns' must be a list."
            )

            return errors

        missing_columns = [
            column
            for column in columns
            if column not in dataframe.columns
        ]

        if missing_columns:
            errors.append(
                (
                    "One or more selected columns do not exist: "
                    f"{missing_columns}"
                )
            )

            return errors

        for column in columns:

            if not pd.api.types.is_numeric_dtype(
                dataframe[column]
            ):
                errors.append(
                    (
                        f"Column '{column}' must be numeric "
                        "for Box-Cox transformation."
                    )
                )

                continue

            values = dataframe[column].dropna()

            if values.empty:
                errors.append(
                    (
                        f"Column '{column}' contains no "
                        "non-missing values."
                    )
                )

                continue

            if not np.isfinite(
                values.astype(float)
            ).all():
                errors.append(
                    (
                        f"Column '{column}' contains infinite "
                        "or non-finite values."
                    )
                )

                continue

            if (
                values.astype(float)
                <= 0
            ).any():

                errors.append(
                    (
                        f"Column '{column}' contains zero or "
                        "negative values. Box-Cox transformation "
                        "requires strictly positive values."
                    )
                )

        return errors

    @staticmethod
    def apply(
        dataframe: pd.DataFrame,
        columns: list,
    ):

        validation_errors = (
            BoxCoxTransformService.validate(
                dataframe=dataframe,
                columns=columns,
            )
        )

        if validation_errors:
            raise ValueError(
                "; ".join(
                    validation_errors
                )
            )

        dataframe = dataframe.copy()

        lambdas = {}
        transformed_values = 0

        for column in columns:

            # Convert the selected column to float before
            # assigning Box-Cox floating-point results.
            dataframe[column] = (
                dataframe[column]
                .astype(float)
            )

            series = dataframe[column]

            non_missing_mask = (
                series.notna()
            )

            non_missing_values = (
                series.loc[
                    non_missing_mask
                ]
                .to_numpy(
                    dtype=float
                )
            )

            transformed_values_array, fitted_lambda = (
                boxcox(
                    non_missing_values
                )
            )

            dataframe.loc[
                non_missing_mask,
                column,
            ] = transformed_values_array

            lambdas[column] = float(
                fitted_lambda
            )

            transformed_values += int(
                non_missing_mask.sum()
            )

        details = {
            "method": "box_cox",
            "columns": columns,
            "lambdas": lambdas,
            "values_transformed": transformed_values,
            "data_modified": True,
        }

        return dataframe, details

    @staticmethod
    def execute(
        file_path: str,
        columns: list,
    ) -> dict:

        input_path = Path(
            file_path
        )

        if not input_path.exists():
            raise ValueError(
                "Dataset file was not found."
            )

        if input_path.suffix.lower() != ".csv":
            raise ValueError(
                "Box-Cox transformation currently supports "
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

        if dataframe.empty:
            raise ValueError(
                "Dataset is empty."
            )

        original_rows = len(
            dataframe
        )

        original_columns = len(
            dataframe.columns
        )

        dataframe, details = (
            BoxCoxTransformService.apply(
                dataframe=dataframe,
                columns=columns,
            )
        )

        output_path = (
            input_path.parent
            / f"{input_path.stem}_box_cox_transformed.csv"
        )

        dataframe.to_csv(
            output_path,
            index=False,
        )

        return {
            "status": "success",
            "input_file": input_path.name,
            "output_file": output_path.name,
            "output_path": str(
                output_path
            ),
            "rows_before": original_rows,
            "rows_after": len(
                dataframe
            ),
            "columns_before": original_columns,
            "columns_after": len(
                dataframe.columns
            ),
            "details": details,
        }