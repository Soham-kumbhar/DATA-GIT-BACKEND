from pathlib import Path

import pandas as pd


class CategoricalUniformityService:
    """
    Standardize inconsistent categorical values using explicit mappings.

    Example:

    {
        "gender": {
            "M": "Male",
            "male": "Male",
            "MALE": "Male",
            "F": "Female",
            "female": "Female"
        }
    }
    """

    @staticmethod
    def validate(
        dataframe: pd.DataFrame,
        mappings: dict,
    ) -> list[str]:
        errors = []

        if not isinstance(mappings, dict):
            return [
                "categorical_uniformity 'mappings' must be an object."
            ]

        if not mappings:
            return [
                "categorical_uniformity requires at least one "
                "column mapping."
            ]

        missing_columns = [
            column
            for column in mappings
            if column not in dataframe.columns
        ]

        if missing_columns:
            errors.append(
                "One or more mapped columns do not exist: "
                f"{missing_columns}"
            )

        for column, mapping in mappings.items():

            if column not in dataframe.columns:
                continue

            if not isinstance(mapping, dict):
                errors.append(
                    f"Mapping for column '{column}' must be an object."
                )
                continue

            if not mapping:
                errors.append(
                    f"Mapping for column '{column}' "
                    "must contain at least one value mapping."
                )
                continue

            dtype = dataframe[column].dtype

            is_categorical = isinstance(
                dtype,
                pd.CategoricalDtype,
            )

            if not (
                pd.api.types.is_object_dtype(dtype)
                or pd.api.types.is_string_dtype(dtype)
                or is_categorical
            ):
                errors.append(
                    f"Column '{column}' must contain categorical/string "
                    "data for categorical uniformity."
                )

            for source_value, target_value in mapping.items():

                if not isinstance(source_value, str):
                    errors.append(
                        f"Source value '{source_value}' in column "
                        f"'{column}' must be a string."
                    )

                if not isinstance(target_value, str):
                    errors.append(
                        f"Target value '{target_value}' in column "
                        f"'{column}' must be a string."
                    )

        return errors

    @staticmethod
    def standardize(
        dataframe: pd.DataFrame,
        mappings: dict,
    ) -> tuple[pd.DataFrame, dict]:

        errors = CategoricalUniformityService.validate(
            dataframe=dataframe,
            mappings=mappings,
        )

        if errors:
            raise ValueError("; ".join(errors))

        result = dataframe.copy()

        column_results = {}

        for column, mapping in mappings.items():

            original = result[column].copy()

            result[column] = result[column].replace(
                mapping
            )

            changed_mask = (
                original.fillna(
                    "__DATAGIT_NULL__"
                ).astype(str)
                !=
                result[column].fillna(
                    "__DATAGIT_NULL__"
                ).astype(str)
            )

            column_results[column] = {
                "mapping_count": len(mapping),
                "values_changed": int(
                    changed_mask.sum()
                ),
                "mappings_applied": mapping,
            }

        details = {
            "method": "categorical_uniformity",
            "columns": list(mappings.keys()),
            "column_results": column_results,
        }

        return result, details

    @staticmethod
    def execute(
        file_path: str,
        mappings: dict,
    ) -> dict:

        input_path = Path(file_path)

        if not input_path.exists():
            raise ValueError(
                "Dataset file was not found."
            )

        if input_path.suffix.lower() != ".csv":
            raise ValueError(
                "Categorical uniformity currently supports "
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
            CategoricalUniformityService.standardize(
                dataframe=dataframe,
                mappings=mappings,
            )
        )

        output_path = (
            input_path.parent
            / f"{input_path.stem}_categorical_uniformity.csv"
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