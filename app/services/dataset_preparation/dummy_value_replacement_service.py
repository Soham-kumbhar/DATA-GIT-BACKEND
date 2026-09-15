from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class DummyValueReplacementService:
    """
    Replace explicitly configured dummy values in selected columns.

    Example:

    {
        "city": {
            "dummy_values": [
                "N/A",
                "NA",
                "Unknown",
                "-",
                "?"
            ],
            "replacement": "Unknown"
        },
        "age": {
            "dummy_values": [
                -1,
                999,
                9999
            ],
            "replacement": None
        }
    }

    Only the configured columns and configured dummy values are changed.
    """

    @staticmethod
    def validate(
        dataframe: pd.DataFrame,
        columns: dict,
    ) -> list[str]:
        errors: list[str] = []

        if not isinstance(columns, dict):
            return [
                "dummy_value_replacement 'columns' "
                "must be an object."
            ]

        if not columns:
            return [
                "dummy_value_replacement requires at least "
                "one column configuration."
            ]

        missing_columns = [
            column
            for column in columns
            if column not in dataframe.columns
        ]

        if missing_columns:
            errors.append(
                "One or more dummy-value replacement columns "
                "do not exist: "
                f"{missing_columns}"
            )

        for column, configuration in columns.items():

            if column not in dataframe.columns:
                continue

            if not isinstance(
                configuration,
                dict,
            ):
                errors.append(
                    f"Configuration for column '{column}' "
                    "must be an object."
                )

                continue

            dummy_values = configuration.get(
                "dummy_values",
                [],
            )

            if not isinstance(
                dummy_values,
                list,
            ):
                errors.append(
                    f"dummy_values for column '{column}' "
                    "must be a list."
                )

                continue

            if not dummy_values:
                errors.append(
                    f"dummy_values for column '{column}' "
                    "must contain at least one value."
                )

            if "replacement" not in configuration:
                errors.append(
                    f"Configuration for column '{column}' "
                    "must include a replacement value."
                )

        return errors

    @staticmethod
    def _values_equal(
        value: Any,
        dummy_value: Any,
    ) -> bool:
        """
        Compare values safely, including pandas/numpy scalar values.
        """
        if pd.isna(value) and pd.isna(dummy_value):
            return True

        try:
            result = value == dummy_value

            if hasattr(result, "item"):
                return bool(result.item())

            return bool(result)

        except (TypeError, ValueError):
            return False

    @classmethod
    def _is_dummy_value(
        cls,
        value: Any,
        dummy_values: list,
    ) -> bool:
        for dummy_value in dummy_values:
            if cls._values_equal(
                value=value,
                dummy_value=dummy_value,
            ):
                return True

        return False

    @classmethod
    def replace(
        cls,
        dataframe: pd.DataFrame,
        columns: dict,
    ) -> tuple[pd.DataFrame, dict]:

        errors = cls.validate(
            dataframe=dataframe,
            columns=columns,
        )

        if errors:
            raise ValueError(
                "; ".join(errors)
            )

        result = dataframe.copy()

        column_results = {}

        total_values_replaced = 0

        for column, configuration in columns.items():

            dummy_values = configuration.get(
                "dummy_values",
                [],
            )

            replacement = configuration.get(
                "replacement"
            )

            original = result[column].copy()

            dummy_mask = original.apply(
                lambda value: cls._is_dummy_value(
                    value=value,
                    dummy_values=dummy_values,
                )
            )

            values_replaced = int(
                dummy_mask.sum()
            )

            if values_replaced > 0:
                result.loc[
                    dummy_mask,
                    column,
                ] = replacement

            total_values_replaced += values_replaced

            column_results[column] = {
                "dummy_values": dummy_values,
                "replacement": replacement,
                "values_replaced": values_replaced,
                "rows_processed": len(result),
            }

        details = {
            "method": "dummy_value_replacement",
            "columns": list(columns.keys()),
            "column_results": column_results,
            "total_values_replaced": total_values_replaced,
            "data_modified": total_values_replaced > 0,
        }

        return result, details

    @classmethod
    def execute(
        cls,
        file_path: str | Path,
        columns: dict,
    ) -> dict:

        input_path = Path(file_path)

        if not input_path.exists():
            raise ValueError(
                "Dataset file was not found."
            )

        if input_path.suffix.lower() != ".csv":
            raise ValueError(
                "Dummy value replacement currently supports "
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

        dataframe, details = cls.replace(
            dataframe=dataframe,
            columns=columns,
        )

        output_path = (
            input_path.parent
            / f"{input_path.stem}_dummy_values_replaced.csv"
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