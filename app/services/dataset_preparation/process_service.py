from pathlib import Path

import pandas as pd

from app.services.dataset_preparation.binning_service import (
    BinningService,
)

from app.services.dataset_preparation.box_cox_transform_service import (
    BoxCoxTransformService,
)

from app.services.dataset_preparation.categorical_uniformity_service import (
    CategoricalUniformityService,
)

from app.services.dataset_preparation.dummy_value_replacement_service import (
    DummyValueReplacementService,
)

from app.services.dataset_preparation.logical_rule_validation_service import (
    LogicalRuleValidationService,
)

from app.services.dataset_preparation.min_max_scaling_service import (
    MinMaxScalingService,
)

from app.services.dataset_preparation.oversampling_service import (
    OversamplingService,
)

from app.services.dataset_preparation.schema_alignment_service import (
    SchemaAlignmentService,
)

from app.services.dataset_preparation.smote_service import (
    SMOTEService,
)

from app.services.dataset_preparation.string_standardization_service import (
    StringStandardizationService,
)

from app.services.dataset_preparation.syntax_correction_service import (
    SyntaxCorrectionService,
)

from app.services.dataset_preparation.type_conversion_service import (
    TypeConversionService,
)

from app.services.dataset_preparation.undersampling_service import (
    UndersamplingService,
)

from app.services.dataset_preparation.knn_imputation_service import (
    KNNImputationService,
)

from app.services.dataset_preparation.mice_imputation_service import (
    MICEImputationService,
)

from app.services.dataset_preparation.z_score_scaling_service import (
    ZScoreScalingService,
)


class DatasetPreparationProcessService:

    @staticmethod
    def execute(
        file_path: str,
        operations: list[dict],
        output_format: str = "csv",
    ) -> dict:

        input_path = Path(file_path)

        if not input_path.exists():
            raise ValueError(
                "Dataset file was not found."
            )

        if input_path.suffix.lower() != ".csv":
            raise ValueError(
                "Dataset preparation currently supports "
                "CSV input files."
            )

        if not operations:
            raise ValueError(
                "At least one preparation operation "
                "must be selected."
            )

        try:
            dataframe = pd.read_csv(
                input_path
            )

        except Exception as error:
            raise ValueError(
                f"Could not read CSV dataset: {error}"
            ) from error

        if dataframe.empty:
            raise ValueError(
                "Dataset is empty."
            )

        original_rows = len(
            dataframe
        )

        original_columns = len(
            dataframe.columns
        )

        step_results = []

        for step_number, operation in enumerate(
            operations,
            start=1,
        ):

            if not isinstance(
                operation,
                dict,
            ):
                raise ValueError(
                    "Every operation must be an object."
                )

            name = operation.get(
                "operation"
            )

            parameters = operation.get(
                "parameters",
                {},
            )

            if not name:
                raise ValueError(
                    "Every operation must contain "
                    "an 'operation' field."
                )

            if not isinstance(
                parameters,
                dict,
            ):
                raise ValueError(
                    f"Parameters for operation '{name}' "
                    "must be an object."
                )

            rows_before = len(
                dataframe
            )

            columns_before = len(
                dataframe.columns
            )

            details = {}

            # ====================================================
            # REMOVE DUPLICATES
            # ====================================================

            if name == "remove_duplicates":

                strategy = parameters.get(
                    "strategy",
                    "keep_first",
                )

                subset = parameters.get(
                    "subset"
                )

                if strategy not in {
                    "keep_first",
                    "keep_last",
                }:
                    raise ValueError(
                        "remove_duplicates strategy must be "
                        "'keep_first' or 'keep_last'."
                    )

                if subset is not None:

                    if not isinstance(
                        subset,
                        list,
                    ):
                        raise ValueError(
                            "remove_duplicates 'subset' "
                            "must be a list."
                        )

                    if not subset:
                        raise ValueError(
                            "remove_duplicates 'subset' "
                            "must contain at least one column."
                        )

                    missing_columns = [
                        column
                        for column in subset
                        if column not in dataframe.columns
                    ]

                    if missing_columns:
                        raise ValueError(
                            "Duplicate subset contains columns "
                            f"that do not exist: {missing_columns}"
                        )

                duplicate_count = int(
                    dataframe.duplicated(
                        subset=subset
                    ).sum()
                )

                dataframe = dataframe.drop_duplicates(
                    subset=subset,
                    keep=(
                        "first"
                        if strategy == "keep_first"
                        else "last"
                    ),
                )

                dataframe = dataframe.reset_index(
                    drop=True
                )

                duplicate_type = (
                    "partial"
                    if subset
                    else "exact"
                )

                details = {
                    "duplicate_type": duplicate_type,
                    "duplicates_removed": duplicate_count,
                    "strategy": strategy,
                    "subset": subset,
                }

            # ====================================================
            # HANDLE MISSING
            # ====================================================

            elif name == "handle_missing":

                columns = parameters.get(
                    "columns",
                    {},
                )

                constant_values = parameters.get(
                    "constant_values",
                    {},
                )

                if not isinstance(
                    columns,
                    dict,
                ):
                    raise ValueError(
                        "handle_missing 'columns' "
                        "must be an object."
                    )

                if not isinstance(
                    constant_values,
                    dict,
                ):
                    raise ValueError(
                        "handle_missing 'constant_values' "
                        "must be an object."
                    )

                if not columns:
                    raise ValueError(
                        "handle_missing requires at least "
                        "one column and strategy."
                    )

                dataframe = dataframe.copy()

                affected_columns = []
                affected_values = 0

                for column, strategy in columns.items():

                    if column not in dataframe.columns:
                        raise ValueError(
                            f"Column '{column}' does not exist."
                        )

                    missing_before = int(
                        dataframe[column].isna().sum()
                    )

                    if strategy == "drop_rows":

                        dataframe = dataframe.dropna(
                            subset=[column]
                        )

                    elif strategy == "drop_column":

                        dataframe = dataframe.drop(
                            columns=[column]
                        )

                    elif strategy == "mean":

                        if not pd.api.types.is_numeric_dtype(
                            dataframe[column]
                        ):
                            raise ValueError(
                                f"Mean strategy requires numeric "
                                f"column '{column}'."
                            )

                        dataframe[column] = (
                            dataframe[column]
                            .fillna(
                                dataframe[column].mean()
                            )
                        )

                    elif strategy == "median":

                        if not pd.api.types.is_numeric_dtype(
                            dataframe[column]
                        ):
                            raise ValueError(
                                f"Median strategy requires numeric "
                                f"column '{column}'."
                            )

                        dataframe[column] = (
                            dataframe[column]
                            .fillna(
                                dataframe[column].median()
                            )
                        )

                    elif strategy == "mode":

                        mode_values = (
                            dataframe[column]
                            .mode()
                        )

                        if mode_values.empty:
                            raise ValueError(
                                (
                                    f"Cannot apply mode "
                                    f"strategy to column "
                                    f"'{column}'."
                                )
                            )

                        dataframe[column] = (
                            dataframe[column]
                            .fillna(
                                mode_values.iloc[0]
                            )
                        )

                    elif strategy == "constant":

                        if column not in constant_values:
                            raise ValueError(
                                (
                                    "Constant strategy "
                                    "requires a value for "
                                    f"column '{column}'."
                                )
                            )

                        dataframe[column] = (
                            dataframe[column]
                            .fillna(
                                constant_values[column]
                            )
                        )

                    elif strategy == "forward_fill":

                        dataframe[column] = (
                            dataframe[column]
                            .ffill()
                        )

                    elif strategy == "backward_fill":

                        dataframe[column] = (
                            dataframe[column]
                            .bfill()
                        )

                    else:
                        raise ValueError(
                            (
                                "Unsupported missing-value "
                                f"strategy '{strategy}'."
                            )
                        )

                    affected_columns.append(
                        column
                    )

                    affected_values += (
                        missing_before
                    )

                details = {
                    "columns_processed": (
                        affected_columns
                    ),
                    "missing_values_affected": (
                        affected_values
                    ),
                }

            # ====================================================
            # DROP COLUMNS
            # ====================================================

            elif name == "drop_columns":

                columns = parameters.get(
                    "columns",
                    [],
                )

                if not isinstance(
                    columns,
                    list,
                ):
                    raise ValueError(
                        "drop_columns 'columns' "
                        "must be a list."
                    )

                if not columns:
                    raise ValueError(
                        "drop_columns requires at least "
                        "one column."
                    )

                missing_columns = [
                    column
                    for column in columns
                    if column not in dataframe.columns
                ]

                if missing_columns:
                    raise ValueError(
                        (
                            "One or more columns do not "
                            "exist: "
                            f"{missing_columns}"
                        )
                    )

                dataframe = dataframe.drop(
                    columns=columns
                )

                details = {
                    "columns_dropped": columns,
                    "count": len(columns),
                }

            # ====================================================
            # TRIM OUTLIERS
            # ====================================================

            elif name == "trim_outliers":

                columns = parameters.get(
                    "columns",
                    [],
                )

                iqr_multiplier = parameters.get(
                    "iqr_multiplier",
                    1.5,
                )

                dataframe, details = (
                    DatasetPreparationProcessService._trim_outliers(
                        dataframe=dataframe,
                        columns=columns,
                        iqr_multiplier=iqr_multiplier,
                    )
                )

            # ====================================================
            # WINSORIZE OUTLIERS
            # ====================================================

            elif name == "winsorize_outliers":

                columns = parameters.get(
                    "columns",
                    [],
                )

                iqr_multiplier = parameters.get(
                    "iqr_multiplier",
                    1.5,
                )

                dataframe, details = (
                    DatasetPreparationProcessService._winsorize_outliers(
                        dataframe=dataframe,
                        columns=columns,
                        iqr_multiplier=iqr_multiplier,
                    )
                )

            # ====================================================
            # LOG TRANSFORM
            # ====================================================

            elif name == "log_transform":

                columns = parameters.get(
                    "columns",
                    [],
                )

                dataframe, details = (
                    DatasetPreparationProcessService._log_transform(
                        dataframe=dataframe,
                        columns=columns,
                    )
                )

            # ====================================================
            # BOX-COX TRANSFORM
            # ====================================================

            elif name == "box_cox_transform":

                columns = parameters.get(
                    "columns",
                    [],
                )

                dataframe, details = (
                    BoxCoxTransformService.apply(
                        dataframe=dataframe,
                        columns=columns,
                    )
                )

            # ====================================================
            # TYPE CONVERSION
            # ====================================================

            elif name == "type_conversion":

                dataframe, details = (
                    TypeConversionService.apply(
                        dataframe=dataframe,
                        **parameters,
                    )
                )

            # ====================================================
            # STRING STANDARDIZATION
            # ====================================================

            elif name == "string_standardization":

                dataframe, details = (
                    StringStandardizationService.apply(
                        dataframe=dataframe,
                        **parameters,
                    )
                )

            # ====================================================
            # CATEGORICAL UNIFORMITY
            # ====================================================

            elif name == "categorical_uniformity":

                dataframe, details = (
                    CategoricalUniformityService.apply(
                        dataframe=dataframe,
                        **parameters,
                    )
                )

            # ====================================================
            # SCHEMA ALIGNMENT
            # ====================================================

            elif name == "schema_alignment":

                dataframe, details = (
                    SchemaAlignmentService.apply(
                        dataframe=dataframe,
                        **parameters,
                    )
                )

            # ====================================================
            # LOGICAL RULE VALIDATION
            # ====================================================

            elif name == "logical_rule_validation":

                dataframe, details = (
                    LogicalRuleValidationService.apply(
                        dataframe=dataframe,
                        **parameters,
                    )
                )

            # ====================================================
            # SYNTAX CORRECTION
            # ====================================================

            elif name == "syntax_correction":

                dataframe, details = (
                    SyntaxCorrectionService.apply(
                        dataframe=dataframe,
                        **parameters,
                    )
                )

            # ====================================================
            # DUMMY VALUE REPLACEMENT
            # ====================================================

            elif name == "dummy_value_replacement":

                dataframe, details = (
                    DummyValueReplacementService.apply(
                        dataframe=dataframe,
                        **parameters,
                    )
                )

            # ====================================================
            # MIN-MAX SCALING
            # ====================================================

            elif name == "min_max_scaling":

                dataframe, details = (
                    MinMaxScalingService.apply(
                        dataframe=dataframe,
                        **parameters,
                    )
                )

            # ====================================================
            # Z-SCORE SCALING
            # ====================================================

            elif name == "z_score_scaling":

                dataframe, details = (
                    ZScoreScalingService.apply(
                        dataframe=dataframe,
                        **parameters,
                    )
                )

            # ====================================================
            # BINNING
            # ====================================================

            elif name == "binning":

                dataframe, details = (
                    BinningService.apply(
                        dataframe=dataframe,
                        **parameters,
                    )
                )

            # ====================================================
            # OVERSAMPLING
            # ====================================================

            elif name == "oversampling":

                target_column = parameters.get(
                    "target_column"
                )

                random_state = parameters.get(
                    "random_state",
                    42,
                )

                dataframe, details = (
                    OversamplingService.oversample(
                        dataframe=dataframe,
                        target_column=target_column,
                        random_state=random_state,
                    )
                )

            # ====================================================
            # SMOTE
            # ====================================================

            elif name == "smote":

                target_column = parameters.get(
                    "target_column"
                )

                k_neighbors = parameters.get(
                    "k_neighbors",
                    5,
                )

                random_state = parameters.get(
                    "random_state",
                    42,
                )

                dataframe, details = (
                    SMOTEService.smote(
                        dataframe=dataframe,
                        target_column=target_column,
                        k_neighbors=k_neighbors,
                        random_state=random_state,
                    )
                )

            # ====================================================
            # UNDERSAMPLING
            # ====================================================

            elif name == "undersampling":

                target_column = parameters.get(
                    "target_column"
                )

                random_state = parameters.get(
                    "random_state",
                    42,
                )

                dataframe, details = (
                    UndersamplingService.undersample(
                        dataframe=dataframe,
                        target_column=target_column,
                        random_state=random_state,
                    )
                )

            # ====================================================
            # KNN IMPUTATION
            # ====================================================

            elif name == "knn_imputation":

                columns = parameters.get(
                    "columns",
                    [],
                )

                n_neighbors = parameters.get(
                    "n_neighbors",
                    5,
                )

                weights = parameters.get(
                    "weights",
                    "uniform",
                )

                dataframe, details = (
                    KNNImputationService.apply(
                        dataframe=dataframe,
                        columns=columns,
                        n_neighbors=n_neighbors,
                        weights=weights,
                    )
                )

            # ====================================================
            # MICE IMPUTATION
            # ====================================================

            elif name == "mice_imputation":

                columns = parameters.get(
                    "columns",
                    [],
                )

                max_iter = parameters.get(
                    "max_iter",
                    10,
                )

                random_state = parameters.get(
                    "random_state",
                    42,
                )

                dataframe, details = (
                    MICEImputationService.apply(
                        dataframe=dataframe,
                        columns=columns,
                        max_iter=max_iter,
                        random_state=random_state,
                    )
                )

            else:
                raise ValueError(
                    f"Unsupported operation: {name}"
                )

            step_results.append(
                {
                    "step": step_number,
                    "operation": name,
                    "status": "success",
                    "parameters": parameters,
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
            )

        # ========================================================
        # SAVE PREPARED DATASET
        # ========================================================

        prepared_directory = (
            Path("dataset_preparations")
            / "prepared"
        )

        prepared_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_format = (
            output_format or "csv"
        ).lower()

        if output_format == "csv":

            output_path = (
                prepared_directory
                / f"{input_path.stem}_prepared.csv"
            )

            dataframe.to_csv(
                output_path,
                index=False,
            )

        elif output_format == "json":

            output_path = (
                prepared_directory
                / f"{input_path.stem}_prepared.json"
            )

            dataframe.to_json(
                output_path,
                orient="records",
                indent=2,
            )

        elif output_format == "parquet":

            output_path = (
                prepared_directory
                / f"{input_path.stem}_prepared.parquet"
            )

            dataframe.to_parquet(
                output_path,
                index=False,
            )

        else:
            raise ValueError(
                (
                    "Unsupported output format. "
                    "Use csv, json, or parquet."
                )
            )

        return {
            "status": "success",
            "input_file": input_path.name,
            "output_file": output_path.name,
            "output_path": str(
                output_path
            ),
            "output_format": output_format,
            "rows_before": original_rows,
            "rows_after": len(
                dataframe
            ),
            "columns_before": original_columns,
            "columns_after": len(
                dataframe.columns
            ),
            "operation_count": len(
                operations
            ),
            "step_results": step_results,
        }

    # ============================================================
    # TRIM OUTLIERS
    # ============================================================

    @staticmethod
    def _trim_outliers(
        dataframe: pd.DataFrame,
        columns: list,
        iqr_multiplier: float = 1.5,
    ):

        dataframe = dataframe.copy()

        if not columns:
            raise ValueError(
                "trim_outliers requires at least one column."
            )

        missing_columns = [
            column
            for column in columns
            if column not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                "trim_outliers contains columns that do not "
                f"exist: {missing_columns}"
            )

        rows_before = len(
            dataframe
        )

        for column in columns:

            q1 = dataframe[column].quantile(
                0.25
            )

            q3 = dataframe[column].quantile(
                0.75
            )

            iqr = q3 - q1

            lower_bound = (
                q1 - iqr_multiplier * iqr
            )

            upper_bound = (
                q3 + iqr_multiplier * iqr
            )

            dataframe = dataframe[
                dataframe[column].isna()
                | dataframe[column].between(
                    lower_bound,
                    upper_bound,
                )
            ]

        dataframe = dataframe.reset_index(
            drop=True
        )

        return dataframe, {
            "rows_removed": (
                rows_before
                - len(dataframe)
            ),
            "iqr_multiplier": (
                iqr_multiplier
            ),
            "columns": columns,
        }

    # ============================================================
    # WINSORIZE OUTLIERS
    # ============================================================

    @staticmethod
    def _winsorize_outliers(
        dataframe: pd.DataFrame,
        columns: list,
        iqr_multiplier: float = 1.5,
    ):

        dataframe = dataframe.copy()

        if not columns:
            raise ValueError(
                "winsorize_outliers requires at least one column."
            )

        missing_columns = [
            column
            for column in columns
            if column not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                "winsorize_outliers contains columns that do not "
                f"exist: {missing_columns}"
            )

        total_capped = 0

        bounds = {}

        for column in columns:

            q1 = dataframe[column].quantile(
                0.25
            )

            q3 = dataframe[column].quantile(
                0.75
            )

            iqr = q3 - q1

            lower_bound = (
                q1 - iqr_multiplier * iqr
            )

            upper_bound = (
                q3 + iqr_multiplier * iqr
            )

            before = dataframe[column].copy()

            dataframe[column] = (
                dataframe[column]
                .clip(
                    lower_bound,
                    upper_bound,
                )
            )

            total_capped += int(
                (
                    before
                    != dataframe[column]
                ).sum()
            )

            bounds[column] = {
                "lower_bound": float(
                    lower_bound
                ),
                "upper_bound": float(
                    upper_bound
                ),
            }

        return dataframe, {
            "values_capped": total_capped,
            "iqr_multiplier": (
                iqr_multiplier
            ),
            "columns": columns,
            "bounds": bounds,
        }

    # ============================================================
    # LOG TRANSFORM
    # ============================================================

    @staticmethod
    def _log_transform(
        dataframe: pd.DataFrame,
        columns: list,
    ):

        import numpy as np

        dataframe = dataframe.copy()

        if not columns:
            raise ValueError(
                "log_transform requires at least one column."
            )

        missing_columns = [
            column
            for column in columns
            if column not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                "log_transform contains columns that do not "
                f"exist: {missing_columns}"
            )

        for column in columns:

            if not pd.api.types.is_numeric_dtype(
                dataframe[column]
            ):
                raise ValueError(
                    f"Log transformation requires numeric "
                    f"column '{column}'."
                )

            non_null_values = dataframe[column].dropna()

            if (
                non_null_values <= 0
            ).any():

                raise ValueError(
                    (
                        f"Column '{column}' contains "
                        "zero or negative values. "
                        "Log transformation requires "
                        "positive values."
                    )
                )

            dataframe[column] = np.log1p(
                dataframe[column]
            )

        return dataframe, {
            "columns": columns,
            "transformation": "log1p",
        }