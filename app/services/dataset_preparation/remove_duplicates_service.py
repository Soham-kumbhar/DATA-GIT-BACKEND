from pathlib import Path

import pandas as pd


class RemoveDuplicatesService:

    @staticmethod
    def execute(
        file_path: str,
        strategy: str = "keep_first",
        subset: list[str] | None = None,
    ) -> dict:
        path = Path(file_path)

        if not path.exists():
            raise ValueError(
                "Dataset file was not found."
            )

        if path.suffix.lower() != ".csv":
            raise ValueError(
                "Remove duplicates currently supports CSV files."
            )

        if strategy not in {
            "keep_first",
            "keep_last",
        }:
            raise ValueError(
                "Strategy must be 'keep_first' or 'keep_last'."
            )

        try:
            dataframe = pd.read_csv(path)

        except Exception as error:
            raise ValueError(
                f"Could not read CSV dataset: {error}"
            ) from error

        if subset is not None:
            if not isinstance(subset, list):
                raise ValueError(
                    "'subset' must be a list of column names."
                )

            if not subset:
                raise ValueError(
                    "'subset' must contain at least one column."
                )

            missing_columns = [
                column
                for column in subset
                if column not in dataframe.columns
            ]

            if missing_columns:
                raise ValueError(
                    "Duplicate subset contains columns that "
                    f"do not exist: {missing_columns}"
                )

        rows_before = len(dataframe)

        duplicate_rows = int(
            dataframe.duplicated(
                subset=subset,
                keep=False,
            ).sum()
        )

        duplicate_rows_removed = int(
            dataframe.duplicated(
                subset=subset,
                keep=(
                    "first"
                    if strategy == "keep_first"
                    else "last"
                ),
            ).sum()
        )

        processed = dataframe.drop_duplicates(
            subset=subset,
            keep=(
                "first"
                if strategy == "keep_first"
                else "last"
            ),
        )

        rows_after = len(processed)

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
            / f"{path.stem}_duplicates_handled.csv"
        )

        processed.to_csv(
            output_path,
            index=False,
        )

        duplicate_type = (
            "partial"
            if subset
            else "exact"
        )

        return {
            "input_file": path.name,
            "output_file": output_path.name,
            "duplicate_type": duplicate_type,
            "strategy": strategy,
            "subset": subset,
            "rows_before": rows_before,
            "rows_after": rows_after,
            "duplicate_rows_found": duplicate_rows,
            "duplicate_rows_removed": duplicate_rows_removed,
            "rows_removed": (
                rows_before - rows_after
            ),
            "status": "success",
            "output_path": str(output_path),
        }