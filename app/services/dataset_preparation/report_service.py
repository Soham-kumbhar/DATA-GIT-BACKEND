from hashlib import sha256
from pathlib import Path

import pandas as pd


class DatasetPreparationReportService:
    @staticmethod
    def _fingerprint(file_path: Path) -> str:
        hasher = sha256()

        with file_path.open("rb") as file:
            for chunk in iter(
                lambda: file.read(1024 * 1024),
                b"",
            ):
                hasher.update(chunk)

        return hasher.hexdigest()

    @staticmethod
    def _column_analysis(
        dataframe: pd.DataFrame,
    ) -> list[dict]:
        analysis = []

        for column in dataframe.columns:
            series = dataframe[column]

            analysis.append(
                {
                    "column": str(column),
                    "type": str(series.dtype),
                    "missing": int(series.isna().sum()),
                    "unique": int(
                        series.nunique(dropna=True)
                    ),
                }
            )

        return analysis

    @staticmethod
    def _outlier_count(
        dataframe: pd.DataFrame,
    ) -> int:
        numeric_columns = dataframe.select_dtypes(
            include="number"
        ).columns

        total_outliers = 0

        for column in numeric_columns:
            series = dataframe[column].dropna()

            if series.empty:
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            if iqr == 0:
                continue

            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)

            total_outliers += int(
                (
                    (series < lower_bound)
                    | (series > upper_bound)
                ).sum()
            )

        return total_outliers

    @staticmethod
    def _data_quality(
        dataframe: pd.DataFrame,
    ) -> dict:
        return {
            "duplicates": int(
                dataframe.duplicated().sum()
            ),
            "missing_values": int(
                dataframe.isna().sum().sum()
            ),
            "constant_columns": [
                str(column)
                for column in dataframe.columns
                if dataframe[column].nunique(
                    dropna=False
                ) <= 1
            ],
            "outliers": DatasetPreparationReportService._outlier_count(
                dataframe
            ),
        }

    @staticmethod
    def _file_size(file_path: Path) -> int:
        return file_path.stat().st_size

    @staticmethod
    def generate(
        input_file: str,
        output_file: str,
        operations: list[dict],
    ) -> dict:
        input_path = (
            Path("dataset_preparations")
            / "uploads"
            / Path(input_file).name
        )

        output_path = (
            Path("dataset_preparations")
            / "prepared"
            / Path(output_file).name
        )

        if not input_path.exists():
            raise ValueError(
                "Input dataset was not found."
            )

        if not output_path.exists():
            raise ValueError(
                "Prepared dataset was not found."
            )

        if input_path.suffix.lower() != ".csv":
            raise ValueError(
                "Report generation currently supports "
                "CSV input datasets."
            )

        try:
            before_dataframe = pd.read_csv(
                input_path
            )
        except Exception as error:
            raise ValueError(
                f"Could not read input CSV dataset: {error}"
            ) from error

        output_suffix = output_path.suffix.lower()

        try:
            if output_suffix == ".csv":
                after_dataframe = pd.read_csv(
                    output_path
                )

            elif output_suffix == ".json":
                after_dataframe = pd.read_json(
                    output_path
                )

            else:
                raise ValueError(
                    "Report generation currently supports "
                    "CSV and JSON prepared datasets."
                )

        except ValueError:
            raise

        except Exception as error:
            raise ValueError(
                f"Could not read prepared dataset: {error}"
            ) from error

        before_quality = (
            DatasetPreparationReportService._data_quality(
                before_dataframe
            )
        )

        after_quality = (
            DatasetPreparationReportService._data_quality(
                after_dataframe
            )
        )

        report = {
            "report_type": "Dataset Preparation Report",
            "status": "COMPLETED",
            "dataset": {
                "input_file": input_path.name,
                "output_file": output_path.name,
                "input_format": (
                    input_path.suffix
                    .lower()
                    .lstrip(".")
                ),
                "output_format": (
                    output_path.suffix
                    .lower()
                    .lstrip(".")
                ),
            },
            "dataset_overview": {
                "rows_before": int(
                    len(before_dataframe)
                ),
                "rows_after": int(
                    len(after_dataframe)
                ),
                "columns_before": int(
                    len(before_dataframe.columns)
                ),
                "columns_after": int(
                    len(after_dataframe.columns)
                ),
                "file_size_before": (
                    DatasetPreparationReportService._file_size(
                        input_path
                    )
                ),
                "file_size_after": (
                    DatasetPreparationReportService._file_size(
                        output_path
                    )
                ),
            },
            "column_analysis": {
                "before": (
                    DatasetPreparationReportService._column_analysis(
                        before_dataframe
                    )
                ),
                "after": (
                    DatasetPreparationReportService._column_analysis(
                        after_dataframe
                    )
                ),
            },
            "data_quality": {
                "before": before_quality,
                "after": after_quality,
            },
            "preparation_operations": operations,
            "before_vs_after": {
                "rows": {
                    "before": int(
                        len(before_dataframe)
                    ),
                    "after": int(
                        len(after_dataframe)
                    ),
                },
                "columns": {
                    "before": int(
                        len(before_dataframe.columns)
                    ),
                    "after": int(
                        len(after_dataframe.columns)
                    ),
                },
                "missing_values": {
                    "before": before_quality[
                        "missing_values"
                    ],
                    "after": after_quality[
                        "missing_values"
                    ],
                },
                "duplicates": {
                    "before": before_quality[
                        "duplicates"
                    ],
                    "after": after_quality[
                        "duplicates"
                    ],
                },
                "outliers": {
                    "before": before_quality[
                        "outliers"
                    ],
                    "after": after_quality[
                        "outliers"
                    ],
                },
            },
            "reproducibility": {
                "input_file": input_path.name,
                "input_fingerprint": (
                    DatasetPreparationReportService._fingerprint(
                        input_path
                    )
                ),
                "preparation_steps": operations,
                "output_fingerprint": (
                    DatasetPreparationReportService._fingerprint(
                        output_path
                    )
                ),
            },
            "output": {
                "processed_dataset": output_path.name,
                "format": (
                    output_path.suffix
                    .lower()
                    .lstrip(".")
                ),
                "preparation_status": "COMPLETED",
            },
        }

        return report