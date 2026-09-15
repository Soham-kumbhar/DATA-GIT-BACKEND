from pathlib import Path

import pandas as pd


class SchemaAlignmentService:

    @staticmethod
    def validate(
        dataframe: pd.DataFrame,
        target_columns: list,
        rename_columns: dict,
        add_missing_columns: dict,
        drop_extra_columns: bool,
    ) -> list[str]:

        errors = []

        # ========================================================
        # TARGET COLUMNS
        # ========================================================

        if not isinstance(
            target_columns,
            list,
        ):
            errors.append(
                "schema_alignment 'target_columns' "
                "must be a list."
            )
            return errors

        if not target_columns:
            errors.append(
                "schema_alignment requires at least "
                "one target column."
            )
            return errors

        non_string_target_columns = [
            column
            for column in target_columns
            if not isinstance(column, str)
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

        # ========================================================
        # RENAME COLUMNS
        # ========================================================

        if not isinstance(
            rename_columns,
            dict,
        ):
            errors.append(
                "schema_alignment 'rename_columns' "
                "must be an object."
            )
            rename_columns = {}

        rename_source_columns = list(
            rename_columns.keys()
        )

        missing_rename_columns = [
            column
            for column in rename_source_columns
            if column not in dataframe.columns
        ]

        if missing_rename_columns:
            errors.append(
                "One or more columns to rename do not exist: "
                f"{missing_rename_columns}"
            )

        rename_target_values = list(
            rename_columns.values()
        )

        non_string_rename_targets = [
            value
            for value in rename_target_values
            if not isinstance(value, str)
        ]

        if non_string_rename_targets:
            errors.append(
                "schema_alignment rename target names "
                "must be strings."
            )

        duplicate_rename_targets = [
            value
            for value in set(rename_target_values)
            if rename_target_values.count(value) > 1
        ]

        if duplicate_rename_targets:
            errors.append(
                "Multiple columns cannot be renamed "
                "to the same target name: "
                f"{duplicate_rename_targets}"
            )

        # ========================================================
        # ADD MISSING COLUMNS
        # ========================================================

        if not isinstance(
            add_missing_columns,
            dict,
        ):
            errors.append(
                "schema_alignment 'add_missing_columns' "
                "must be an object."
            )
            add_missing_columns = {}

        # ========================================================
        # BOOLEAN OPTION
        # ========================================================

        if not isinstance(
            drop_extra_columns,
            bool,
        ):
            errors.append(
                "schema_alignment 'drop_extra_columns' "
                "must be a boolean."
            )

        # ========================================================
        # DETERMINE COLUMNS AFTER RENAME
        # ========================================================

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
                f"column names: {duplicate_columns_after_rename}"
            )

        # ========================================================
        # TARGET COLUMN VALIDATION
        # ========================================================

        missing_target_columns = [
            column
            for column in target_columns
            if column not in renamed_columns
            and column not in add_missing_columns
        ]

        if missing_target_columns:
            errors.append(
                "Target schema contains columns that are "
                "missing from the dataset and were not included "
                "in add_missing_columns: "
                f"{missing_target_columns}"
            )

        extra_target_columns = [
            column
            for column in add_missing_columns
            if column not in target_columns
        ]

        if extra_target_columns:
            errors.append(
                "add_missing_columns contains columns that "
                "are not present in target_columns: "
                f"{extra_target_columns}"
            )

        # ========================================================
        # EXTRA COLUMNS
        # ========================================================

        extra_columns = [
            column
            for column in renamed_columns
            if column not in target_columns
        ]

        if extra_columns and not drop_extra_columns:
            errors.append(
                "Dataset contains columns outside the target "
                f"schema and drop_extra_columns is false: "
                f"{extra_columns}"
            )

        return errors

    @staticmethod
    def align(
        dataframe: pd.DataFrame,
        target_columns: list,
        rename_columns: dict | None = None,
        add_missing_columns: dict | None = None,
        drop_extra_columns: bool = True,
    ) -> tuple[pd.DataFrame, dict]:

        rename_columns = (
            rename_columns
            if rename_columns is not None
            else {}
        )

        add_missing_columns = (
            add_missing_columns
            if add_missing_columns is not None
            else {}
        )

        errors = SchemaAlignmentService.validate(
            dataframe=dataframe,
            target_columns=target_columns,
            rename_columns=rename_columns,
            add_missing_columns=add_missing_columns,
            drop_extra_columns=drop_extra_columns,
        )

        if errors:
            raise ValueError(
                "; ".join(errors)
            )

        result = dataframe.copy()

        columns_before = list(
            result.columns
        )

        # ========================================================
        # STEP 1 — RENAME
        # ========================================================

        if rename_columns:
            result = result.rename(
                columns=rename_columns
            )

        columns_after_rename = list(
            result.columns
        )

        # ========================================================
        # STEP 2 — ADD MISSING COLUMNS
        # ========================================================

        added_columns = []

        for column in target_columns:

            if column not in result.columns:

                result[column] = (
                    add_missing_columns[column]
                )

                added_columns.append(
                    column
                )

        # ========================================================
        # STEP 3 — REMOVE EXTRA COLUMNS
        # ========================================================

        removed_columns = []

        extra_columns = [
            column
            for column in result.columns
            if column not in target_columns
        ]

        if drop_extra_columns:
            removed_columns = extra_columns

            result = result.drop(
                columns=extra_columns
            )

        # ========================================================
        # STEP 4 — REORDER
        # ========================================================

        result = result[
            target_columns
        ]

        details = {
            "method": "schema_alignment",
            "target_columns": target_columns,
            "rename_columns": rename_columns,
            "add_missing_columns": add_missing_columns,
            "drop_extra_columns": drop_extra_columns,
            "columns_before": columns_before,
            "columns_after_rename": columns_after_rename,
            "columns_added": added_columns,
            "columns_removed": removed_columns,
            "final_column_order": list(
                result.columns
            ),
        }

        return result, details

    @staticmethod
    def execute(
        file_path: str,
        target_columns: list,
        rename_columns: dict | None = None,
        add_missing_columns: dict | None = None,
        drop_extra_columns: bool = True,
    ) -> dict:

        input_path = Path(file_path)

        if not input_path.exists():
            raise ValueError(
                "Dataset file was not found."
            )

        if input_path.suffix.lower() != ".csv":
            raise ValueError(
                "Schema alignment currently supports "
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

        rows_before = len(dataframe)
        columns_before = len(dataframe.columns)

        dataframe, details = (
            SchemaAlignmentService.align(
                dataframe=dataframe,
                target_columns=target_columns,
                rename_columns=rename_columns,
                add_missing_columns=add_missing_columns,
                drop_extra_columns=drop_extra_columns,
            )
        )

        output_path = (
            input_path.parent
            / f"{input_path.stem}_schema_aligned.csv"
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
            "rows_after": len(dataframe),
            "columns_before": columns_before,
            "columns_after": len(dataframe.columns),
            "details": details,
        }