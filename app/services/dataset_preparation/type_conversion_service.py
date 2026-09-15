from datetime import date, datetime
from pathlib import Path

import pandas as pd


class TypeConversionService:

    SUPPORTED_TYPES = {
        "integer",
        "float",
        "string",
        "boolean",
        "date",
        "datetime",
    }

    @staticmethod
    def validate_columns(
        dataframe: pd.DataFrame,
        columns: dict[str, str],
    ) -> None:
        if not isinstance(columns, dict):
            raise ValueError(
                "'columns' must be an object mapping "
                "column names to target types."
            )

        if not columns:
            raise ValueError(
                "At least one column must be selected."
            )

        for column, target_type in columns.items():

            if column not in dataframe.columns:
                raise ValueError(
                    f"Column '{column}' does not exist."
                )

            if target_type not in (
                TypeConversionService.SUPPORTED_TYPES
            ):
                raise ValueError(
                    f"Unsupported target type "
                    f"'{target_type}' for column "
                    f"'{column}'. Supported types: "
                    f"{sorted(TypeConversionService.SUPPORTED_TYPES)}"
                )

    @staticmethod
    def convert(
        dataframe: pd.DataFrame,
        columns: dict[str, str],
    ) -> tuple[pd.DataFrame, dict]:
        TypeConversionService.validate_columns(
            dataframe,
            columns,
        )

        result = dataframe.copy()

        conversion_results = {}

        for column, target_type in columns.items():

            original_dtype = str(
                result[column].dtype
            )

            try:

                if target_type == "integer":
                    result[column] = pd.to_numeric(
                        result[column],
                        errors="raise",
                    ).astype("int64")

                elif target_type == "float":
                    result[column] = pd.to_numeric(
                        result[column],
                        errors="raise",
                    ).astype("float64")

                elif target_type == "string":
                    result[column] = (
                        result[column]
                        .astype("string")
                    )

                elif target_type == "boolean":
                    result[column] = (
                        TypeConversionService
                        ._convert_boolean_column(
                            result[column]
                        )
                    )

                elif target_type == "date":
                    result[column] = (
                        pd.to_datetime(
                            result[column],
                            errors="raise",
                        ).dt.date
                    )

                elif target_type == "datetime":
                    result[column] = pd.to_datetime(
                        result[column],
                        errors="raise",
                    )

            except Exception as error:
                raise ValueError(
                    f"Could not convert column "
                    f"'{column}' from "
                    f"'{original_dtype}' to "
                    f"'{target_type}': {error}"
                ) from error

            conversion_results[column] = {
                "original_type": original_dtype,
                "target_type": target_type,
                "final_type": str(
                    result[column].dtype
                ),
            }

        return result, {
            "method": "type_conversion",
            "columns": columns,
            "column_results": conversion_results,
        }

    @staticmethod
    def _convert_boolean_column(
        series: pd.Series,
    ) -> pd.Series:
        if pd.api.types.is_bool_dtype(series):
            return series

        true_values = {
            "true",
            "1",
            "yes",
            "y",
            "on",
        }

        false_values = {
            "false",
            "0",
            "no",
            "n",
            "off",
        }

        def convert_value(value):
            if pd.isna(value):
                return pd.NA

            if isinstance(value, bool):
                return value

            normalized = str(value).strip().lower()

            if normalized in true_values:
                return True

            if normalized in false_values:
                return False

            raise ValueError(
                f"Value '{value}' cannot be converted "
                "to boolean."
            )

        converted = series.apply(
            convert_value
        )

        return converted.astype("boolean")

    @staticmethod
    def execute(
        file_path: str,
        columns: dict[str, str],
    ) -> dict:
        path = Path(file_path)

        if not path.exists():
            raise ValueError(
                "Dataset file was not found."
            )

        if path.suffix.lower() != ".csv":
            raise ValueError(
                "Type conversion currently supports "
                "CSV files."
            )

        try:
            dataframe = pd.read_csv(path)

        except Exception as error:
            raise ValueError(
                f"Could not read CSV dataset: {error}"
            ) from error

        rows_before = len(dataframe)
        columns_before = len(dataframe.columns)

        processed, details = (
            TypeConversionService.convert(
                dataframe=dataframe,
                columns=columns,
            )
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
            / f"{path.stem}_type_converted.csv"
        )

        processed.to_csv(
            output_path,
            index=False,
        )

        return {
            "status": "success",
            "input_file": path.name,
            "output_file": output_path.name,
            "rows_before": rows_before,
            "rows_after": len(processed),
            "columns_before": columns_before,
            "columns_after": len(processed.columns),
            **details,
            "output_path": str(output_path),
        }