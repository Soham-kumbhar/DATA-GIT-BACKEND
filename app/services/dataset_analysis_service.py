from pathlib import Path
import tempfile
from typing import Any

import pandas as pd

from app.services.dvc_service import DVCService


class DatasetAnalysisService:

    # ============================================================
    # PUBLIC DATASET PROFILE
    # ============================================================

    @staticmethod
    def analyze_dataset(
        project_path: str,
        dvc_hash: str,
        dataset_name: str,
    ) -> dict[str, Any]:

        with tempfile.TemporaryDirectory() as temp_dir:

            dataset_path = (
                Path(temp_dir)
                / "dataset.csv"
            )

            # ----------------------------------------------------
            # GET EXACT DATASET FROM DVC CACHE
            # ----------------------------------------------------

            DVCService._get_from_cache(
                project_path=project_path,
                dvc_hash=dvc_hash,
                output_path=dataset_path,
            )

            # ----------------------------------------------------
            # READ DATASET
            # ----------------------------------------------------

            try:

                df = pd.read_csv(
                    dataset_path
                )

            except Exception as exc:

                raise RuntimeError(
                    "Could not read dataset "
                    f"'{dataset_name}' as CSV: "
                    f"{exc}"
                )

            return (
                DatasetAnalysisService._build_profile(
                    df=df,
                    dataset_name=dataset_name,
                    dvc_hash=dvc_hash,
                )
            )

    # ============================================================
    # BUILD DATASET PROFILE
    # ============================================================

    @staticmethod
    def _build_profile(
        df: pd.DataFrame,
        dataset_name: str,
        dvc_hash: str,
    ) -> dict[str, Any]:

        # --------------------------------------------------------
        # BASIC INFORMATION
        # --------------------------------------------------------

        rows = int(len(df))
        columns = list(df.columns)

        # --------------------------------------------------------
        # DATA TYPES
        # --------------------------------------------------------

        dtypes = {
            str(column):
                str(dtype)
            for column, dtype
            in df.dtypes.items()
        }

        # --------------------------------------------------------
        # MISSING VALUES
        # --------------------------------------------------------

        missing_values = {
            str(column):
                int(count)
            for column, count
            in df.isna().sum().items()
        }

        total_missing = int(
            df.isna().sum().sum()
        )

        # --------------------------------------------------------
        # DUPLICATES
        # --------------------------------------------------------

        duplicate_rows = int(
            df.duplicated().sum()
        )

        # --------------------------------------------------------
        # TARGET COLUMN
        # --------------------------------------------------------

        target_column = (
            DatasetAnalysisService._find_target_column(
                columns
            )
        )

        # --------------------------------------------------------
        # TARGET DISTRIBUTION
        # --------------------------------------------------------

        target_distribution = {}

        if target_column is not None:

            counts = (
                df[target_column]
                .value_counts(dropna=False)
            )

            target_distribution = {
                DatasetAnalysisService._safe_value(
                    key
                ):
                    int(value)
                for key, value in counts.items()
            }

        # --------------------------------------------------------
        # NUMERIC STATISTICS
        # --------------------------------------------------------

        numeric_statistics = {}

        numeric_df = df.select_dtypes(
            include="number"
        )

        for column in numeric_df.columns:

            series = numeric_df[column]

            numeric_statistics[
                str(column)
            ] = {
                "min":
                    DatasetAnalysisService._safe_number(
                        series.min()
                    ),

                "max":
                    DatasetAnalysisService._safe_number(
                        series.max()
                    ),

                "mean":
                    DatasetAnalysisService._safe_number(
                        series.mean()
                    ),

                "median":
                    DatasetAnalysisService._safe_number(
                        series.median()
                    ),

                "std":
                    DatasetAnalysisService._safe_number(
                        series.std()
                    ),
            }

        # --------------------------------------------------------
        # CATEGORICAL STATISTICS
        # --------------------------------------------------------

        categorical_statistics = {}

        categorical_df = df.select_dtypes(
            exclude="number"
        )

        for column in categorical_df.columns:

            categorical_statistics[
                str(column)
            ] = {
                "unique_values":
                    int(
                        df[column]
                        .nunique(
                            dropna=False
                        )
                    ),

                "top_values":
                    DatasetAnalysisService._top_values(
                        df[column]
                    ),
            }

        # --------------------------------------------------------
        # FEATURE PROFILE
        # --------------------------------------------------------

        feature_columns = [
            column
            for column in columns
            if column != target_column
        ]

        feature_profile = {}

        for column in feature_columns:

            series = df[column]

            feature_profile[
                str(column)
            ] = (
                DatasetAnalysisService._analyze_feature(
                    series
                )
            )

        # --------------------------------------------------------
        # FINAL PROFILE
        # --------------------------------------------------------

        return {
            "dataset":
                dataset_name,

            "dvc_hash":
                dvc_hash,

            "rows":
                rows,

            "columns":
                columns,

            "column_count":
                len(columns),

            "dtypes":
                dtypes,

            "missing_values":
                missing_values,

            "total_missing_values":
                total_missing,

            "duplicate_rows":
                duplicate_rows,

            "target_column":
                target_column,

            "target_distribution":
                target_distribution,

            "numeric_statistics":
                numeric_statistics,

            "categorical_statistics":
                categorical_statistics,

            "feature_columns":
                feature_columns,

            "feature_profile":
                feature_profile,
        }

    # ============================================================
    # FIND TARGET COLUMN
    # ============================================================

    @staticmethod
    def _find_target_column(
        columns: list[str],
    ) -> str | None:

        preferred_names = [
            "target",
            "label",
            "y",
            "class",
        ]

        lowercase_map = {
            str(column).lower():
                str(column)
            for column in columns
        }

        for name in preferred_names:

            if name in lowercase_map:
                return lowercase_map[name]

        return None

    # ============================================================
    # FEATURE ANALYSIS
    # ============================================================

    @staticmethod
    def _analyze_feature(
        series: pd.Series,
    ) -> dict[str, Any]:

        result = {
            "dtype":
                str(series.dtype),

            "missing":
                int(series.isna().sum()),

            "unique":
                int(
                    series.nunique(
                        dropna=False
                    )
                ),
        }

        if pd.api.types.is_numeric_dtype(
            series
        ):

            result.update(
                {
                    "type": "numeric",

                    "min":
                        DatasetAnalysisService._safe_number(
                            series.min()
                        ),

                    "max":
                        DatasetAnalysisService._safe_number(
                            series.max()
                        ),

                    "mean":
                        DatasetAnalysisService._safe_number(
                            series.mean()
                        ),

                    "median":
                        DatasetAnalysisService._safe_number(
                            series.median()
                        ),

                    "std":
                        DatasetAnalysisService._safe_number(
                            series.std()
                        ),
                }
            )

        else:

            result.update(
                {
                    "type": "categorical",

                    "top_values":
                        DatasetAnalysisService._top_values(
                            series
                        ),
                }
            )

        return result

    # ============================================================
    # TOP VALUES
    # ============================================================

    @staticmethod
    def _top_values(
        series: pd.Series,
        limit: int = 10,
    ) -> list[dict[str, Any]]:

        counts = (
            series
            .value_counts(
                dropna=False
            )
            .head(limit)
        )

        result = []

        for value, count in counts.items():

            result.append(
                {
                    "value":
                        DatasetAnalysisService._safe_value(
                            value
                        ),

                    "count":
                        int(count),
                }
            )

        return result

    # ============================================================
    # SAFE VALUE
    # ============================================================

    @staticmethod
    def _safe_value(
        value: Any,
    ) -> Any:

        if pd.isna(value):
            return None

        if hasattr(value, "item"):

            try:
                return value.item()
            except Exception:
                pass

        return value

    # ============================================================
    # SAFE NUMBER
    # ============================================================

    @staticmethod
    def _safe_number(
        value: Any,
    ) -> int | float | None:

        if pd.isna(value):
            return None

        try:

            numeric = float(value)

            if numeric.is_integer():
                return int(numeric)

            return numeric

        except Exception:

            return None