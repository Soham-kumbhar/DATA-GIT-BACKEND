from pathlib import Path

import pandas as pd
from sklearn.impute import KNNImputer


class KNNImputationService:

    @staticmethod
    def validate(
        dataframe: pd.DataFrame,
        columns: list,
        n_neighbors: int = 5,
        weights: str = "uniform",
    ) -> list:

        errors = []

        if not isinstance(
            columns,
            list,
        ):
            errors.append(
                "knn_imputation 'columns' must be a list."
            )
            return errors

        if not columns:
            errors.append(
                "knn_imputation requires at least one column."
            )
            return errors

        missing_columns = [
            column
            for column in columns
            if column not in dataframe.columns
        ]

        if missing_columns:
            errors.append(
                "One or more columns do not exist: "
                f"{missing_columns}"
            )

        non_numeric_columns = [
            column
            for column in columns
            if column in dataframe.columns
            and not pd.api.types.is_numeric_dtype(
                dataframe[column]
            )
        ]

        if non_numeric_columns:
            errors.append(
                "KNN imputation requires numeric columns. "
                f"Non-numeric columns: {non_numeric_columns}"
            )

        if not isinstance(
            n_neighbors,
            int,
        ):
            errors.append(
                "knn_imputation 'n_neighbors' must be an integer."
            )

        elif n_neighbors <= 0:
            errors.append(
                "knn_imputation 'n_neighbors' must be greater than 0."
            )

        elif n_neighbors >= len(dataframe):
            errors.append(
                "knn_imputation 'n_neighbors' must be smaller "
                "than the number of dataset rows."
            )

        if weights not in {
            "uniform",
            "distance",
        }:
            errors.append(
                "knn_imputation 'weights' must be "
                "'uniform' or 'distance'."
            )

        return errors

    @staticmethod
    def apply(
        dataframe: pd.DataFrame,
        columns: list,
        n_neighbors: int = 5,
        weights: str = "uniform",
    ):
        errors = KNNImputationService.validate(
            dataframe=dataframe,
            columns=columns,
            n_neighbors=n_neighbors,
            weights=weights,
        )

        if errors:
            raise ValueError(
                errors
            )

        dataframe = dataframe.copy()

        missing_before = {
            column: int(
                dataframe[column].isna().sum()
            )
            for column in columns
        }

        total_missing_before = sum(
            missing_before.values()
        )

        if total_missing_before == 0:
            return dataframe, {
                "method": "knn_imputation",
                "columns": columns,
                "n_neighbors": n_neighbors,
                "weights": weights,
                "missing_values_before": missing_before,
                "missing_values_imputed": 0,
                "data_modified": False,
            }

        imputer = KNNImputer(
            n_neighbors=n_neighbors,
            weights=weights,
        )

        dataframe[columns] = imputer.fit_transform(
            dataframe[columns]
        )

        missing_after = {
            column: int(
                dataframe[column].isna().sum()
            )
            for column in columns
        }

        total_missing_after = sum(
            missing_after.values()
        )

        return dataframe, {
            "method": "knn_imputation",
            "columns": columns,
            "n_neighbors": n_neighbors,
            "weights": weights,
            "missing_values_before": missing_before,
            "missing_values_after": missing_after,
            "missing_values_imputed": (
                total_missing_before
                - total_missing_after
            ),
            "data_modified": (
                total_missing_before
                != total_missing_after
            ),
        }

    @staticmethod
    def execute(
        file_path: str,
        columns: list,
        n_neighbors: int = 5,
        weights: str = "uniform",
    ) -> dict:

        input_path = Path(
            file_path
        )

        if not input_path.exists():
            raise ValueError(
                "Dataset file was not found."
            )

        if input_path.suffix.lower() != ".csv":
            raise ValueError(
                "KNN imputation currently supports CSV "
                "input files."
            )

        try:
            dataframe = pd.read_csv(
                input_path
            )

        except Exception as error:
            raise ValueError(
                f"Could not read CSV dataset: {error}"
            ) from error

        dataframe, details = (
            KNNImputationService.apply(
                dataframe=dataframe,
                columns=columns,
                n_neighbors=n_neighbors,
                weights=weights,
            )
        )

        output_path = (
            input_path.parent
            / f"{input_path.stem}_knn_imputed.csv"
        )

        dataframe.to_csv(
            output_path,
            index=False,
        )

        return {
            "status": "success",
            "input_file": input_path.name,
            "output_file": output_path.name,
            "output_path": str(
                output_path
            ),
            "rows_before": len(
                pd.read_csv(file_path)
            ),
            "rows_after": len(
                dataframe
            ),
            "columns_before": len(
                pd.read_csv(file_path).columns
            ),
            "columns_after": len(
                dataframe.columns
            ),
            "details": details,
        }