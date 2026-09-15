from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors


class SMOTEService:
    """
    Balance a classification dataset using Synthetic Minority
    Over-sampling Technique (SMOTE).

    Current implementation supports numeric feature columns.
    The target column is excluded from the feature matrix.
    """

    @staticmethod
    def validate(
        dataframe: pd.DataFrame,
        target_column: str,
        k_neighbors: int = 5,
        random_state: int = 42,
    ) -> list[str]:
        errors: list[str] = []

        if not target_column:
            errors.append(
                "target_column is required."
            )

        elif target_column not in dataframe.columns:
            errors.append(
                f"Target column '{target_column}' does not exist."
            )

        if not isinstance(k_neighbors, int):
            errors.append(
                "k_neighbors must be an integer."
            )
        elif k_neighbors < 1:
            errors.append(
                "k_neighbors must be at least 1."
            )

        if not isinstance(random_state, int):
            errors.append(
                "random_state must be an integer."
            )

        if target_column in dataframe.columns:

            target_series = dataframe[target_column]

            if target_series.isna().any():
                errors.append(
                    "Target column must not contain missing values "
                    "for SMOTE."
                )

            if target_series.nunique(dropna=True) < 2:
                errors.append(
                    "SMOTE requires at least two target classes."
                )

            feature_columns = [
                column
                for column in dataframe.columns
                if column != target_column
            ]

            if not feature_columns:
                errors.append(
                    "SMOTE requires at least one feature column."
                )

            else:
                non_numeric_columns = [
                    column
                    for column in feature_columns
                    if not pd.api.types.is_numeric_dtype(
                        dataframe[column]
                    )
                ]

                if non_numeric_columns:
                    errors.append(
                        "SMOTE currently supports numeric feature "
                        "columns only. Non-numeric columns: "
                        f"{non_numeric_columns}"
                    )

                missing_feature_values = [
                    column
                    for column in feature_columns
                    if dataframe[column].isna().any()
                ]

                if missing_feature_values:
                    errors.append(
                        "Feature columns must not contain missing "
                        "values for SMOTE. Columns: "
                        f"{missing_feature_values}"
                    )

            class_counts = (
                target_series.value_counts()
            )

            for class_value, class_count in class_counts.items():

                if class_count > 1:
                    if k_neighbors >= class_count:
                        errors.append(
                            (
                                f"Class '{class_value}' contains "
                                f"{class_count} samples. "
                                f"k_neighbors must be less than "
                                f"the class sample count "
                                f"({class_count})."
                            )
                        )

        return errors

    @staticmethod
    def oversample(
        dataframe: pd.DataFrame,
        target_column: str,
        k_neighbors: int = 5,
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, dict]:
        validation_errors = SMOTEService.validate(
            dataframe=dataframe,
            target_column=target_column,
            k_neighbors=k_neighbors,
            random_state=random_state,
        )

        if validation_errors:
            raise ValueError(
                "; ".join(validation_errors)
            )

        rng = np.random.default_rng(
            random_state
        )

        feature_columns = [
            column
            for column in dataframe.columns
            if column != target_column
        ]

        original_class_counts = (
            dataframe[target_column]
            .value_counts()
            .sort_index()
        )

        majority_count = int(
            original_class_counts.max()
        )

        synthetic_frames = []
        class_results = {}

        for class_value, class_count in (
            original_class_counts.items()
        ):
            class_count = int(class_count)

            class_dataframe = dataframe[
                dataframe[target_column] == class_value
            ].copy()

            additional_rows = (
                majority_count - class_count
            )

            if additional_rows <= 0:
                class_results[str(class_value)] = {
                    "original_count": class_count,
                    "target_count": majority_count,
                    "synthetic_rows": 0,
                    "final_count": class_count,
                }

                continue

            feature_matrix = (
                class_dataframe[
                    feature_columns
                ]
                .to_numpy(
                    dtype=float
                )
            )

            if len(feature_matrix) < 2:
                raise ValueError(
                    (
                        f"Class '{class_value}' must contain "
                        "at least two samples for SMOTE."
                    )
                )

            effective_neighbors = min(
                k_neighbors,
                len(feature_matrix) - 1,
            )

            neighbor_model = (
                NearestNeighbors(
                    n_neighbors=effective_neighbors + 1
                )
            )

            neighbor_model.fit(
                feature_matrix
            )

            neighbor_indices = (
                neighbor_model.kneighbors(
                    feature_matrix,
                    return_distance=False,
                )
            )

            synthetic_features = []

            for _ in range(
                additional_rows
            ):
                base_index = int(
                    rng.integers(
                        0,
                        len(feature_matrix),
                    )
                )

                neighbor_candidates = (
                    neighbor_indices[
                        base_index
                    ][1:]
                )

                neighbor_index = int(
                    rng.choice(
                        neighbor_candidates
                    )
                )

                base_sample = (
                    feature_matrix[
                        base_index
                    ]
                )

                neighbor_sample = (
                    feature_matrix[
                        neighbor_index
                    ]
                )

                interpolation_factor = (
                    rng.random()
                )

                synthetic_sample = (
                    base_sample
                    + interpolation_factor
                    * (
                        neighbor_sample
                        - base_sample
                    )
                )

                synthetic_features.append(
                    synthetic_sample
                )

            synthetic_dataframe = pd.DataFrame(
                synthetic_features,
                columns=feature_columns,
            )

            synthetic_dataframe[
                target_column
            ] = class_value

            synthetic_frames.append(
                synthetic_dataframe
            )

            class_results[str(class_value)] = {
                "original_count": class_count,
                "target_count": majority_count,
                "synthetic_rows": additional_rows,
                "final_count": (
                    class_count
                    + additional_rows
                ),
            }

        if synthetic_frames:
            synthetic_rows_dataframe = (
                pd.concat(
                    synthetic_frames,
                    ignore_index=True,
                )
            )

            result = pd.concat(
                [
                    dataframe.copy(),
                    synthetic_rows_dataframe,
                ],
                ignore_index=True,
            )

        else:
            result = dataframe.copy()

        result = result.sample(
            frac=1,
            random_state=random_state,
        ).reset_index(
            drop=True
        )

        final_class_counts = (
            result[target_column]
            .value_counts()
            .sort_index()
        )

        details = {
            "method": "smote",
            "target_column": target_column,
            "k_neighbors": k_neighbors,
            "random_state": random_state,
            "feature_columns": feature_columns,
            "original_class_distribution": {
                str(class_value): int(class_count)
                for class_value, class_count
                in original_class_counts.items()
            },
            "target_class_count": majority_count,
            "final_class_distribution": {
                str(class_value): int(class_count)
                for class_value, class_count
                in final_class_counts.items()
            },
            "class_results": class_results,
            "synthetic_rows_added": (
                len(result) - len(dataframe)
            ),
            "data_modified": True,
        }

        return result, details

    @staticmethod
    def execute(
        file_path: str,
        target_column: str,
        k_neighbors: int = 5,
        random_state: int = 42,
    ) -> dict:
        input_path = Path(file_path)

        if not input_path.exists():
            raise ValueError(
                "Dataset file was not found."
            )

        if input_path.suffix.lower() == ".csv":
            dataframe = pd.read_csv(
                input_path
            )

        elif input_path.suffix.lower() == ".json":
            dataframe = pd.read_json(
                input_path
            )

        else:
            raise ValueError(
                "Only CSV and JSON datasets are supported."
            )

        rows_before = len(dataframe)
        columns_before = len(dataframe.columns)

        result_dataframe, details = (
            SMOTEService.oversample(
                dataframe=dataframe,
                target_column=target_column,
                k_neighbors=k_neighbors,
                random_state=random_state,
            )
        )

        output_path = (
            input_path.parent
            / (
                f"{input_path.stem}"
                "_smote"
                f"{input_path.suffix}"
            )
        )

        if input_path.suffix.lower() == ".csv":
            result_dataframe.to_csv(
                output_path,
                index=False,
            )

        else:
            result_dataframe.to_json(
                output_path,
                orient="records",
            )

        return {
            "status": "success",
            "input_file": input_path.name,
            "output_file": output_path.name,
            "output_format": (
                input_path.suffix
                .lower()
                .lstrip(".")
            ),
            "rows_before": rows_before,
            "rows_after": len(result_dataframe),
            "columns_before": columns_before,
            "columns_after": len(result_dataframe.columns),
            "details": details,
            "output_path": str(output_path),
        }