from pathlib import Path

import pandas as pd


class DropColumnsService:

    @staticmethod
    def execute(
        file_path: str,
        columns: list[str],
    ) -> dict:
        path = Path(file_path)

        if not path.exists():
            raise ValueError(
                "Dataset file was not found."
            )

        if path.suffix.lower() != ".csv":
            raise ValueError(
                "Dropping columns currently supports CSV files."
            )

        if not columns:
            raise ValueError(
                "At least one column must be provided."
            )

        try:
            dataframe = pd.read_csv(path)

        except Exception as error:
            raise ValueError(
                f"Could not read CSV dataset: {error}"
            ) from error

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

        rows_before = len(dataframe)
        columns_before = len(dataframe.columns)

        dataframe = dataframe.drop(
            columns=columns
        )

        rows_after = len(dataframe)
        columns_after = len(dataframe.columns)

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
            / f"{path.stem}_columns_dropped.csv"
        )

        dataframe.to_csv(
            output_path,
            index=False,
        )

        return {
            "input_file": path.name,
            "output_file": output_path.name,
            "columns_removed": columns,
            "rows_before": rows_before,
            "rows_after": rows_after,
            "columns_before": columns_before,
            "columns_after": columns_after,
            "status": "success",
            "output_path": str(output_path),
        }