from pathlib import Path

import pandas as pd


class DatasetProfileService:
    @staticmethod
    def profile(file_path: str) -> dict:
        path = Path(file_path)

        if not path.exists():
            raise ValueError("Dataset file was not found.")

        if path.suffix.lower() != ".csv":
            raise ValueError(
                "Dataset profiling currently supports CSV files."
            )

        try:
            dataframe = pd.read_csv(path)
        except Exception as error:
            raise ValueError(
                f"Could not read CSV dataset: {error}"
            ) from error

        numeric_columns = dataframe.select_dtypes(
            include="number"
        ).columns.tolist()

        categorical_columns = dataframe.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        missing_values = {
            column: int(count)
            for column, count in dataframe.isna().sum().items()
            if int(count) > 0
        }

        column_information = []

        for column in dataframe.columns:
            column_information.append(
                {
                    "name": str(column),
                    "data_type": str(dataframe[column].dtype),
                    "missing": int(
                        dataframe[column].isna().sum()
                    ),
                    "unique": int(
                        dataframe[column].nunique(
                            dropna=True
                        )
                    ),
                }
            )

        return {
            "filename": path.name,
            "file_type": "CSV",
            "rows": int(len(dataframe)),
            "columns": int(len(dataframe.columns)),
            "column_names": [
                str(column)
                for column in dataframe.columns
            ],
            "column_information": column_information,
            "missing_values": missing_values,
            "missing_value_count": int(
                dataframe.isna().sum().sum()
            ),
            "duplicate_rows": int(
                dataframe.duplicated().sum()
            ),
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
        }