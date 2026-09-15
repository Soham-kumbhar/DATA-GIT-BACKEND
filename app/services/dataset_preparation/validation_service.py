from pathlib import Path

import pandas as pd

from app.services.dataset_preparation.binning_service import (
    BinningService,
)

from app.services.dataset_preparation.logical_rule_validation_service import (
    LogicalRuleValidationService,
)

from app.services.dataset_preparation.syntax_correction_service import (
    SyntaxCorrectionService,
)

from app.services.dataset_preparation.dummy_value_replacement_service import (
    DummyValueReplacementService,
)

from app.services.dataset_preparation.min_max_scaling_service import (
    MinMaxScalingService,
)

from app.services.dataset_preparation.z_score_scaling_service import (
    ZScoreScalingService,
)

from app.services.dataset_preparation.oversampling_service import (
    OversamplingService,
)

from app.services.dataset_preparation.smote_service import (
    SMOTEService,
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

from app.services.dataset_preparation.box_cox_transform_service import (
    BoxCoxTransformService,
)


class DatasetPreparationValidationService:

    @staticmethod
    def validate(
        file_path: str,
        operations: list[dict],
    ) -> dict:

        path = Path(file_path)

        errors = []
        warnings = []

        # ========================================================
        # FILE VALIDATION
        # ========================================================

        if not path.exists():
            raise ValueError(
                "Dataset file was not found."
            )

        if path.suffix.lower() != ".csv":
            raise ValueError(
                "Dataset preparation currently supports "
                "CSV input files."
            )

        try:
            dataframe = pd.read_csv(path)

        except Exception as error:
            raise ValueError(
                f"Could not read CSV dataset: {error}"
            ) from error

        if not operations:
            errors.append(
                "At least one preparation operation "
                "must be selected."
            )

            return {
                "valid": False,
                "errors": errors,
                "warnings": warnings,
                "operation_count": 0,
            }

        # ========================================================
        # OPERATION VALIDATION
        # ========================================================

        supported_operations = {
            "remove_duplicates",
            "handle_missing",
            "drop_columns",
            "trim_outliers",
            "winsorize_outliers",
            "log_transform",
            "type_conversion",
            "string_standardization",
            "categorical_uniformity",
            "schema_alignment",
            "logical_rule_validation",
            "syntax_correction",
            "dummy_value_replacement",
            "min_max_scaling",
            "z_score_scaling",
            "binning",
            "oversampling",
            "smote",
            "undersampling",
            "knn_imputation",
            "mice_imputation",
            "box_cox_transform",
        }

        for operation in operations:

            name = operation.get(
                "operation"
            )

            parameters = operation.get(
                "parameters",
                {},
            )

            if not name:
                errors.append(
                    "Every operation must contain "
                    "an 'operation' field."
                )

                continue

            if name not in supported_operations:
                errors.append(
                    f"Unsupported operation: {name}"
                )

                continue

            if not isinstance(
                parameters,
                dict,
            ):
                errors.append(
                    f"Operation '{name}' parameters "
                    "must be an object."
                )

                continue

            # ====================================================
            # REMOVE DUPLICATES
            # ====================================================

            if name == "remove_duplicates":

                strategy = parameters.get(
                    "strategy",
                    "keep_first",
                )

                if strategy not in {
                    "keep_first",
                    "keep_last",
                }:
                    errors.append(
                        "remove_duplicates strategy must be "
                        "'keep_first' or 'keep_last'."
                    )

                subset = parameters.get(
                    "subset"
                )

                if subset is not None:

                    if not isinstance(
                        subset,
                        list,
                    ):
                        errors.append(
                            "remove_duplicates 'subset' "
                            "must be a list."
                        )

                    elif not subset:
                        errors.append(
                            "remove_duplicates 'subset' "
                            "must contain at least one column."
                        )

                    else:
                        missing_columns = [
                            column
                            for column in subset
                            if column not in dataframe.columns
                        ]

                        if missing_columns:
                            errors.append(
                                "Duplicate subset contains "
                                "columns that do not exist: "
                                f"{missing_columns}"
                            )

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
                    errors.append(
                        "drop_columns 'columns' "
                        "must be a list."
                    )

                elif not columns:
                    errors.append(
                        "drop_columns requires at least "
                        "one column."
                    )

                else:
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

            # ====================================================
            # HANDLE MISSING
            # ====================================================

            elif name == "handle_missing":

                columns = parameters.get(
                    "columns",
                    {},
                )

                if not isinstance(
                    columns,
                    dict,
                ):
                    errors.append(
                        "handle_missing 'columns' "
                        "must be an object."
                    )

                    continue

                if not columns:
                    errors.append(
                        "handle_missing requires at least "
                        "one column and strategy."
                    )

                    continue

                constant_values = parameters.get(
                    "constant_values",
                    {},
                )

                if not isinstance(
                    constant_values,
                    dict,
                ):
                    errors.append(
                        "handle_missing 'constant_values' "
                        "must be an object."
                    )

                    continue

                valid_strategies = {
                    "drop_rows",
                    "drop_column",
                    "mean",
                    "median",
                    "mode",
                    "constant",
                    "forward_fill",
                    "backward_fill",
                }

                for column, strategy in columns.items():

                    if column not in dataframe.columns:
                        errors.append(
                            f"Column '{column}' does not exist."
                        )

                        continue

                    if strategy not in valid_strategies:
                        errors.append(
                            f"Unsupported missing-value "
                            f"strategy '{strategy}' "
                            f"for column '{column}'."
                        )

                        continue

                    if strategy in {
                        "mean",
                        "median",
                    }:
                        if not pd.api.types.is_numeric_dtype(
                            dataframe[column]
                        ):
                            errors.append(
                                f"{strategy.capitalize()} "
                                "strategy requires numeric "
                                f"column '{column}'."
                            )

                    if strategy == "constant":
                        if column not in constant_values:
                            errors.append(
                                "Constant strategy requires "
                                f"a value for column '{column}'."
                            )

            # ====================================================
            # OUTLIER OPERATIONS
            # ====================================================

            elif name in {
                "trim_outliers",
                "winsorize_outliers",
            }:

                columns = parameters.get(
                    "columns",
                    [],
                )

                if not isinstance(
                    columns,
                    list,
                ):
                    errors.append(
                        f"{name} 'columns' must be a list."
                    )

                    continue

                if not columns:
                    errors.append(
                        f"{name} requires at least "
                        "one numeric column."
                    )

                    continue

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

                    continue

                non_numeric_columns = [
                    column
                    for column in columns
                    if not pd.api.types.is_numeric_dtype(
                        dataframe[column]
                    )
                ]

                if non_numeric_columns:
                    errors.append(
                        f"{name} requires numeric columns. "
                        f"Non-numeric columns: "
                        f"{non_numeric_columns}"
                    )

                iqr_multiplier = parameters.get(
                    "iqr_multiplier",
                    1.5,
                )

                try:
                    iqr_multiplier = float(
                        iqr_multiplier
                    )

                    if iqr_multiplier <= 0:
                        errors.append(
                            f"{name} 'iqr_multiplier' "
                            "must be greater than 0."
                        )

                except (
                    TypeError,
                    ValueError,
                ):
                    errors.append(
                        f"{name} 'iqr_multiplier' "
                        "must be a number."
                    )

            # ====================================================
            # LOG TRANSFORM
            # ====================================================

            elif name == "log_transform":

                columns = parameters.get(
                    "columns",
                    [],
                )

                if not isinstance(
                    columns,
                    list,
                ):
                    errors.append(
                        "log_transform 'columns' "
                        "must be a list."
                    )

                    continue

                if not columns:
                    errors.append(
                        "log_transform requires at least "
                        "one numeric column."
                    )

                    continue

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

                    continue

                non_numeric_columns = [
                    column
                    for column in columns
                    if not pd.api.types.is_numeric_dtype(
                        dataframe[column]
                    )
                ]

                if non_numeric_columns:
                    errors.append(
                        "log_transform requires numeric columns. "
                        f"Non-numeric columns: "
                        f"{non_numeric_columns}"
                    )

            # ====================================================
            # BOX-COX TRANSFORM
            # ====================================================

            elif name == "box_cox_transform":

                columns = parameters.get(
                    "columns",
                    [],
                )

                box_cox_validation = (
                    BoxCoxTransformService.validate(
                        dataframe=dataframe,
                        columns=columns,
                    )
                )

                errors.extend(
                    box_cox_validation
                )

            # ====================================================
            # TYPE CONVERSION
            # ====================================================

            elif name == "type_conversion":

                columns = parameters.get(
                    "columns",
                    {},
                )

                if not isinstance(
                    columns,
                    dict,
                ):
                    errors.append(
                        "type_conversion 'columns' "
                        "must be an object mapping "
                        "column names to target types."
                    )

                    continue

                if not columns:
                    errors.append(
                        "type_conversion requires at least "
                        "one column."
                    )

                    continue

                supported_types = {
                    "integer",
                    "float",
                    "string",
                    "boolean",
                    "date",
                    "datetime",
                }

                for column, target_type in columns.items():

                    if column not in dataframe.columns:
                        errors.append(
                            f"Column '{column}' does not exist."
                        )

                        continue

                    if target_type not in supported_types:
                        errors.append(
                            f"Unsupported target type "
                            f"'{target_type}' for column "
                            f"'{column}'."
                        )

            # ====================================================
            # STRING STANDARDIZATION
            # ====================================================

            elif name == "string_standardization":

                columns = parameters.get(
                    "columns",
                    [],
                )

                options = parameters.get(
                    "options",
                    [],
                )

                if not isinstance(
                    columns,
                    list,
                ):
                    errors.append(
                        "string_standardization 'columns' "
                        "must be a list."
                    )

                    continue

                if not columns:
                    errors.append(
                        "string_standardization requires "
                        "at least one column."
                    )

                    continue

                if not isinstance(
                    options,
                    list,
                ):
                    errors.append(
                        "string_standardization 'options' "
                        "must be a list."
                    )

                    continue

                if not options:
                    errors.append(
                        "string_standardization requires "
                        "at least one option."
                    )

                    continue

                supported_options = {
                    "trim_whitespace",
                    "lowercase",
                    "uppercase",
                    "remove_extra_spaces",
                }

                unsupported_options = [
                    option
                    for option in options
                    if option not in supported_options
                ]

                if unsupported_options:
                    errors.append(
                        "Unsupported string-standardization "
                        f"options: {unsupported_options}"
                    )

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

                    continue

                non_string_columns = [
                    column
                    for column in columns
                    if not (
                        pd.api.types.is_string_dtype(
                            dataframe[column]
                        )
                        or dataframe[column].dtype == object
                    )
                ]

                if non_string_columns:
                    errors.append(
                        "String standardization requires "
                        "string columns. Non-string columns: "
                        f"{non_string_columns}"
                    )

            # ====================================================
            # CATEGORICAL UNIFORMITY
            # ====================================================

            elif name == "categorical_uniformity":

                mappings = parameters.get(
                    "mappings",
                    {},
                )

                if not isinstance(
                    mappings,
                    dict,
                ):
                    errors.append(
                        "categorical_uniformity 'mappings' "
                        "must be an object."
                    )

                    continue

                if not mappings:
                    errors.append(
                        "categorical_uniformity requires at "
                        "least one column mapping."
                    )

                    continue

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

                    if not isinstance(
                        mapping,
                        dict,
                    ):
                        errors.append(
                            f"Mapping for column '{column}' "
                            "must be an object."
                        )

                        continue

                    if not mapping:
                        errors.append(
                            f"Mapping for column '{column}' "
                            "must contain at least one "
                            "value mapping."
                        )

                        continue

                    dtype = dataframe[column].dtype

                    if not (
                        pd.api.types.is_object_dtype(dtype)
                        or pd.api.types.is_string_dtype(dtype)
                        or isinstance(
                            dtype,
                            pd.CategoricalDtype,
                        )
                    ):
                        errors.append(
                            f"Column '{column}' must contain "
                            "categorical/string data for "
                            "categorical uniformity."
                        )

                    for source_value, target_value in mapping.items():

                        if not isinstance(
                            source_value,
                            str,
                        ):
                            errors.append(
                                f"Source value '{source_value}' "
                                f"in column '{column}' "
                                "must be a string."
                            )

                        if not isinstance(
                            target_value,
                            str,
                        ):
                            errors.append(
                                f"Target value '{target_value}' "
                                f"in column '{column}' "
                                "must be a string."
                            )

            # ====================================================
            # SCHEMA ALIGNMENT
            # ====================================================

            elif name == "schema_alignment":

                target_columns = parameters.get(
                    "target_columns",
                    [],
                )

                rename_columns = parameters.get(
                    "rename_columns",
                    {},
                )

                add_missing_columns = parameters.get(
                    "add_missing_columns",
                    {},
                )

                drop_extra_columns = parameters.get(
                    "drop_extra_columns",
                    True,
                )

                if not isinstance(
                    target_columns,
                    list,
                ):
                    errors.append(
                        "schema_alignment 'target_columns' "
                        "must be a list."
                    )

                    continue

                if not target_columns:
                    errors.append(
                        "schema_alignment requires at least "
                        "one target column."
                    )

                    continue

                if not isinstance(
                    rename_columns,
                    dict,
                ):
                    errors.append(
                        "schema_alignment 'rename_columns' "
                        "must be an object."
                    )

                    continue

                if not isinstance(
                    add_missing_columns,
                    dict,
                ):
                    errors.append(
                        "schema_alignment 'add_missing_columns' "
                        "must be an object."
                    )

                    continue

                if not isinstance(
                    drop_extra_columns,
                    bool,
                ):
                    errors.append(
                        "schema_alignment 'drop_extra_columns' "
                        "must be a boolean."
                    )

                    continue

                non_string_target_columns = [
                    column
                    for column in target_columns
                    if not isinstance(
                        column,
                        str,
                    )
                ]

                if non_string_target_columns:
                    errors.append(
                        "schema_alignment target column names "
                        "must be strings."
                    )

                duplicate_target_columns = [
                    column
                    for column in set(target_columns)
                    if target_columns.count(column) > 1
                ]

                if duplicate_target_columns:
                    errors.append(
                        "schema_alignment target_columns "
                        "must not contain duplicates: "
                        f"{duplicate_target_columns}"
                    )

                missing_rename_columns = [
                    column
                    for column in rename_columns
                    if column not in dataframe.columns
                ]

                if missing_rename_columns:
                    errors.append(
                        "One or more columns to rename do not exist: "
                        f"{missing_rename_columns}"
                    )

                rename_targets = list(
                    rename_columns.values()
                )

                non_string_rename_targets = [
                    value
                    for value in rename_targets
                    if not isinstance(
                        value,
                        str,
                    )
                ]

                if non_string_rename_targets:
                    errors.append(
                        "schema_alignment rename target names "
                        "must be strings."
                    )

                duplicate_rename_targets = [
                    value
                    for value in set(rename_targets)
                    if rename_targets.count(value) > 1
                ]

                if duplicate_rename_targets:
                    errors.append(
                        "Multiple columns cannot be renamed "
                        "to the same target name: "
                        f"{duplicate_rename_targets}"
                    )

                renamed_columns = [
                    rename_columns.get(
                        column,
                        column,
                    )
                    for column in dataframe.columns
                ]

                duplicate_columns_after_rename = [
                    column
                    for column in set(renamed_columns)
                    if renamed_columns.count(column) > 1
                ]

                if duplicate_columns_after_rename:
                    errors.append(
                        "Column renaming creates duplicate "
                        "column names: "
                        f"{duplicate_columns_after_rename}"
                    )

                missing_target_columns = [
                    column
                    for column in target_columns
                    if column not in renamed_columns
                    and column not in add_missing_columns
                ]

                if missing_target_columns:
                    errors.append(
                        "Target schema contains columns that are "
                        "missing from the dataset and were not "
                        "included in add_missing_columns: "
                        f"{missing_target_columns}"
                    )

                invalid_added_columns = [
                    column
                    for column in add_missing_columns
                    if column not in target_columns
                ]

                if invalid_added_columns:
                    errors.append(
                        "add_missing_columns contains columns "
                        "that are not present in target_columns: "
                        f"{invalid_added_columns}"
                    )

                extra_columns = [
                    column
                    for column in renamed_columns
                    if column not in target_columns
                ]

                if extra_columns and not drop_extra_columns:
                    errors.append(
                        "Dataset contains columns outside the "
                        "target schema and drop_extra_columns "
                        "is false: "
                        f"{extra_columns}"
                    )

            # ====================================================
            # LOGICAL RULE VALIDATION
            # ====================================================

            elif name == "logical_rule_validation":

                rules = parameters.get(
                    "rules",
                    [],
                )

                if not isinstance(
                    rules,
                    list,
                ):
                    errors.append(
                        "logical_rule_validation 'rules' "
                        "must be a list."
                    )

                    continue

                if not rules:
                    errors.append(
                        "logical_rule_validation requires at "
                        "least one rule."
                    )

                    continue

                logical_validation = (
                    LogicalRuleValidationService.validate(
                        dataframe=dataframe,
                        rules=rules,
                    )
                )

                errors.extend(
                    logical_validation["errors"]
                )

            # ====================================================
            # SYNTAX CORRECTION
            # ====================================================

            elif name == "syntax_correction":

                columns = parameters.get(
                    "columns",
                    {},
                )

                syntax_validation = (
                    SyntaxCorrectionService.validate(
                        dataframe=dataframe,
                        columns=columns,
                    )
                )

                errors.extend(
                    syntax_validation
                )

            # ====================================================
            # DUMMY VALUE REPLACEMENT
            # ====================================================

            elif name == "dummy_value_replacement":

                columns = parameters.get(
                    "columns",
                    {},
                )

                dummy_validation = (
                    DummyValueReplacementService.validate(
                        dataframe=dataframe,
                        columns=columns,
                    )
                )

                errors.extend(
                    dummy_validation
                )

            # ====================================================
            # MIN-MAX SCALING
            # ====================================================

            elif name == "min_max_scaling":

                columns = parameters.get(
                    "columns",
                    [],
                )

                feature_range = parameters.get(
                    "feature_range",
                    [0, 1],
                )

                min_max_validation = (
                    MinMaxScalingService.validate(
                        dataframe=dataframe,
                        columns=columns,
                        feature_range=feature_range,
                    )
                )

                errors.extend(
                    min_max_validation
                )

            # ====================================================
            # Z-SCORE SCALING
            # ====================================================

            elif name == "z_score_scaling":

                columns = parameters.get(
                    "columns",
                    [],
                )

                z_score_validation = (
                    ZScoreScalingService.validate(
                        dataframe=dataframe,
                        columns=columns,
                    )
                )

                errors.extend(
                    z_score_validation
                )

            # ====================================================
            # BINNING
            # ====================================================

            elif name == "binning":

                columns = parameters.get(
                    "columns",
                    [],
                )

                bins = parameters.get(
                    "bins",
                    3,
                )

                method = parameters.get(
                    "method",
                    "equal_width",
                )

                labels = parameters.get(
                    "labels"
                )

                binning_validation = (
                    BinningService.validate(
                        dataframe=dataframe,
                        columns=columns,
                        bins=bins,
                        method=method,
                        labels=labels,
                    )
                )

                errors.extend(
                    binning_validation
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

                oversampling_validation = (
                    OversamplingService.validate(
                        dataframe=dataframe,
                        target_column=target_column,
                        random_state=random_state,
                    )
                )

                errors.extend(
                    oversampling_validation
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

                smote_validation = (
                    SMOTEService.validate(
                        dataframe=dataframe,
                        target_column=target_column,
                        k_neighbors=k_neighbors,
                        random_state=random_state,
                    )
                )

                errors.extend(
                    smote_validation
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

                undersampling_validation = (
                    UndersamplingService.validate(
                        dataframe=dataframe,
                        target_column=target_column,
                        random_state=random_state,
                    )
                )

                errors.extend(
                    undersampling_validation
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

                knn_validation = (
                    KNNImputationService.validate(
                        dataframe=dataframe,
                        columns=columns,
                        n_neighbors=n_neighbors,
                        weights=weights,
                    )
                )

                errors.extend(
                    knn_validation
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

                mice_validation = (
                    MICEImputationService.validate(
                        dataframe=dataframe,
                        columns=columns,
                        max_iter=max_iter,
                        random_state=random_state,
                    )
                )

                errors.extend(
                    mice_validation
                )

        # ========================================================
        # RESULT
        # ========================================================

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "operation_count": len(operations),
        }