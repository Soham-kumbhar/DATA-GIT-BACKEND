from pathlib import Path

import pandas as pd

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer


class MICEImputationService:

    @staticmethod
    def validate(
        dataframe: pd.DataFrame,
        columns: list,
        max_iter: int = 10,
        random_state: int = 42,
    ) -> list:

        errors = []

        if not isinstance(
            columns,
            list,
        ):
            errors.append(
                "mice_imputation 'columns' must be a list."
            )
            return errors

        if not columns:
            errors.append(
                "mice_imputation requires at least one column."
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
                "MICE imputation currently requires numeric columns. "
                f"Non-numeric columns: {non_numeric_columns}"
            )

        if not isinstance(
            max_iter,
            int,
        ):
            errors.append(
                "mice_imputation 'max_iter' must be an integer."
            )

        elif max_iter <= 0:
            errors.append(
                "mice_imputation 'max_iter' must be greater than 0."
            )

        if not isinstance(
            random_state,
            int,
        ):
            errors.append(
                "mice_imputation 'random_state' must be an integer."
            )

        return errors

    @staticmethod
    def apply(
        dataframe: pd.DataFrame,
        columns: list,
        max_iter: int = 10,
        random_state: int = 42,
    ):

        errors = MICEImputationService.validate(
            dataframe=dataframe,
            columns=columns,
            max_iter=max_iter,
            random_state=random_state,
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
                "method": "mice_imputation",
                "columns": columns,
                "max_iter": max_iter,
                "random_state": random_state,
                "missing_values_before": missing_before,
                "missing_values_after": missing_before,
                "missing_values_imputed": 0,
                "data_modified": False,
            }

        imputer = IterativeImputer(
            max_iter=max_iter,
            random_state=random_state,
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
            "method": "mice_imputation",
            "columns": columns,
            "max_iter": max_iter,
            "random_state": random_state,
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
        max_iter: int = 10,
        random_state: int = 42,
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
                "MICE imputation currently supports "
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

        rows_before = len(
            dataframe
        )

        columns_before = len(
            dataframe.columns
        )

        dataframe, details = (
            MICEImputationService.apply(
                dataframe=dataframe,
                columns=columns,
                max_iter=max_iter,
                random_state=random_state,
            )
        )

        output_path = (
            input_path.parent
            / f"{input_path.stem}_mice_imputed.csv"
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
            "rows_before": rows_before,
            "rows_after": len(
                dataframe
            ),
            "columns_before": columns_before,
            "columns_after": len(
                dataframe.columns
            ),
            "details": details,
        }