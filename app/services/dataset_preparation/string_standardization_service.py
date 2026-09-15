from pathlib import Path

import pandas as pd


class StringStandardizationService:

    SUPPORTED_OPTIONS = {
        "trim_whitespace",
        "lowercase",
        "uppercase",
        "remove_extra_spaces",
    }

    @staticmethod
    def validate(
        dataframe: pd.DataFrame,
        columns: list[str],
        options: list[str],
    ) -> None:
        if not isinstance(columns, list):
            raise ValueError(
                "string_standardization 'columns' "
                "must be a list."
            )

        if not columns:
            raise ValueError(
                "string_standardization requires "
                "at least one column."
            )

        if not isinstance(options, list):
            raise ValueError(
                "string_standardization 'options' "
                "must be a list."
            )

        if not options:
            raise ValueError(
                "string_standardization requires "
                "at least one option."
            )

        unsupported_options = [
            option
            for option in options
            if option not in (
                StringStandardizationService.SUPPORTED_OPTIONS
            )
        ]

        if unsupported_options:
            raise ValueError(
                "Unsupported string-standardization options: "
                f"{unsupported_options}"
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

        non_string_columns = [
            column
            for column in columns
            if not (
                pd.api.types.is_string_dtype(
                    dataframe[column]
                )
                or dataframe[column].dtype == object
            )
        ]

        if non_string_columns:
            raise ValueError(
                "String standardization requires "
                "string columns. Non-string columns: "
                f"{non_string_columns}"
            )

    @staticmethod
    def standardize(
        dataframe: pd.DataFrame,
        columns: list[str],
        options: list[str],
    ) -> tuple[pd.DataFrame, dict]:
        StringStandardizationService.validate(
            dataframe=dataframe,
            columns=columns,
            options=options,
        )

        result = dataframe.copy()

        column_results = {}

        for column in columns:
            original_values = (
                result[column]
                .astype("string")
                .copy()
            )

            series = original_values.copy()

            option_results = []

            for option in options:

                if option == "trim_whitespace":
                    series = series.str.strip()

                elif option == "lowercase":
                    series = series.str.lower()

                elif option == "uppercase":
                    series = series.str.upper()

                elif option == "remove_extra_spaces":
                    series = series.str.replace(
                        r"\s+",
                        " ",
                        regex=True,
                    )

                option_results.append(option)

            changed_values = int(
                (
                    original_values.fillna("")
                    != series.fillna("")
                ).sum()
            )

            result[column] = series

            column_results[column] = {
                "options_applied": option_results,
                "values_changed": changed_values,
            }

        return result, {
            "method": "string_standardization",
            "columns": columns,
            "options": options,
            "column_results": column_results,
        }

    @staticmethod
    def execute(
        file_path: str,
        columns: list[str],
        options: list[str],
    ) -> dict:
        path = Path(file_path)

        if not path.exists():
            raise ValueError(
                "Dataset file was not found."
            )

        if path.suffix.lower() != ".csv":
            raise ValueError(
                "String standardization currently supports "
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
            StringStandardizationService.standardize(
                dataframe=dataframe,
                columns=columns,
                options=options,
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
            / f"{path.stem}_string_standardized.csv"
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