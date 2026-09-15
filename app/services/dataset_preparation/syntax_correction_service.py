from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import pandas as pd


class SyntaxCorrectionService:
    """
    Correct common syntax/format problems in selected dataset columns.

    Supported correction types:

        email
        phone
        numeric
        date
        datetime
        whitespace
        alphanumeric

    The operation modifies only the selected columns.

    Example:

        {
            "email": "email",
            "phone": "phone",
            "amount": "numeric",
            "join_date": "date",
            "created_at": "datetime",
            "name": "whitespace"
        }
    """

    SUPPORTED_CORRECTION_TYPES = {
        "email",
        "phone",
        "numeric",
        "date",
        "datetime",
        "whitespace",
        "alphanumeric",
    }

    @staticmethod
    def validate(
        dataframe: pd.DataFrame,
        columns: dict,
    ) -> list[str]:
        errors: list[str] = []

        if not isinstance(columns, dict):
            return [
                "syntax_correction 'columns' must be an object."
            ]

        if not columns:
            return [
                "syntax_correction requires at least one column "
                "and correction type."
            ]

        missing_columns = [
            column
            for column in columns
            if column not in dataframe.columns
        ]

        if missing_columns:
            errors.append(
                "One or more syntax-correction columns do not exist: "
                f"{missing_columns}"
            )

        for column, correction_type in columns.items():

            if column not in dataframe.columns:
                continue

            if correction_type not in (
                SyntaxCorrectionService.SUPPORTED_CORRECTION_TYPES
            ):
                errors.append(
                    f"Unsupported syntax correction type "
                    f"'{correction_type}' for column '{column}'. "
                    f"Supported types: "
                    f"{sorted(SyntaxCorrectionService.SUPPORTED_CORRECTION_TYPES)}"
                )

        return errors

    @staticmethod
    def _normalize_whitespace(value: Any) -> Any:
        if pd.isna(value):
            return value

        if not isinstance(value, str):
            return value

        return " ".join(value.split())

    @staticmethod
    def _normalize_email(value: Any) -> Any:
        if pd.isna(value):
            return value

        if not isinstance(value, str):
            return value

        value = value.strip()
        value = value.replace(" ", "")

        return value.lower()

    @staticmethod
    def _normalize_phone(value: Any) -> Any:
        if pd.isna(value):
            return value

        if not isinstance(value, str):
            value = str(value)

        value = value.strip()

        has_plus = value.startswith("+")
        digits = re.sub(r"\D", "", value)

        if not digits:
            return value

        if has_plus:
            return f"+{digits}"

        return digits

    @staticmethod
    def _normalize_numeric(value: Any) -> Any:
        if pd.isna(value):
            return value

        if isinstance(value, (int, float)):
            return value

        if not isinstance(value, str):
            return value

        cleaned = value.strip()

        if not cleaned:
            return value

        negative = False

        if cleaned.startswith("(") and cleaned.endswith(")"):
            negative = True
            cleaned = cleaned[1:-1]

        cleaned = cleaned.replace(",", "")
        cleaned = cleaned.replace("$", "")
        cleaned = cleaned.replace("₹", "")
        cleaned = cleaned.replace("€", "")
        cleaned = cleaned.replace("£", "")

        cleaned = cleaned.strip()

        if not cleaned:
            return value

        try:
            number = float(cleaned)

            if negative:
                number = -number

            if number.is_integer():
                return int(number)

            return number

        except ValueError:
            return value

    @staticmethod
    def _normalize_date(value: Any) -> Any:
        if pd.isna(value):
            return value

        if isinstance(value, (pd.Timestamp,)):
            return value.strftime("%Y-%m-%d")

        parsed = pd.to_datetime(
            value,
            errors="coerce",
        )

        if pd.isna(parsed):
            return value

        return parsed.strftime("%Y-%m-%d")

    @staticmethod
    def _normalize_datetime(value: Any) -> Any:
        if pd.isna(value):
            return value

        if isinstance(value, (pd.Timestamp,)):
            return value.strftime("%Y-%m-%d %H:%M:%S")

        parsed = pd.to_datetime(
            value,
            errors="coerce",
        )

        if pd.isna(parsed):
            return value

        return parsed.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    @staticmethod
    def _normalize_alphanumeric(value: Any) -> Any:
        if pd.isna(value):
            return value

        if not isinstance(value, str):
            return value

        return re.sub(
            r"[^A-Za-z0-9]+",
            "",
            value,
        )

    @classmethod
    def _correct_value(
        cls,
        value: Any,
        correction_type: str,
    ) -> Any:

        if correction_type == "email":
            return cls._normalize_email(value)

        if correction_type == "phone":
            return cls._normalize_phone(value)

        if correction_type == "numeric":
            return cls._normalize_numeric(value)

        if correction_type == "date":
            return cls._normalize_date(value)

        if correction_type == "datetime":
            return cls._normalize_datetime(value)

        if correction_type == "whitespace":
            return cls._normalize_whitespace(value)

        if correction_type == "alphanumeric":
            return cls._normalize_alphanumeric(value)

        raise ValueError(
            f"Unsupported syntax correction type: "
            f"{correction_type}"
        )

    @classmethod
    def correct(
        cls,
        dataframe: pd.DataFrame,
        columns: dict,
    ) -> tuple[pd.DataFrame, dict]:

        errors = cls.validate(
            dataframe=dataframe,
            columns=columns,
        )

        if errors:
            raise ValueError(
                "; ".join(errors)
            )

        result = dataframe.copy()

        column_results = {}

        total_values_changed = 0

        for column, correction_type in columns.items():

            original = result[column].copy()

            corrected = original.apply(
                lambda value: cls._correct_value(
                    value=value,
                    correction_type=correction_type,
                )
            )

            result[column] = corrected

            changed_mask = (
                original.fillna(
                    "__DATAGIT_NULL__"
                ).astype(str)
                !=
                corrected.fillna(
                    "__DATAGIT_NULL__"
                ).astype(str)
            )

            values_changed = int(
                changed_mask.sum()
            )

            total_values_changed += values_changed

            column_results[column] = {
                "correction_type": correction_type,
                "values_changed": values_changed,
                "rows_processed": len(result),
            }

        details = {
            "method": "syntax_correction",
            "columns": list(columns.keys()),
            "column_results": column_results,
            "total_values_changed": total_values_changed,
            "data_modified": total_values_changed > 0,
        }

        return result, details

    @classmethod
    def execute(
        cls,
        file_path: str | Path,
        columns: dict,
    ) -> dict:

        input_path = Path(file_path)

        if not input_path.exists():
            raise ValueError(
                "Dataset file was not found."
            )

        if input_path.suffix.lower() != ".csv":
            raise ValueError(
                "Syntax correction currently supports "
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

        dataframe, details = cls.correct(
            dataframe=dataframe,
            columns=columns,
        )

        output_path = (
            input_path.parent
            / f"{input_path.stem}_syntax_corrected.csv"
        )

        dataframe.to_csv(
            output_path,
            index=False,
        )

        return {
            "status": "success",
            "input_file": input_path.name,
            "output_file": output_path.name,
            "output_path": str(output_path),
            "rows_before": rows_before,
            "rows_after": len(dataframe),
            "columns_before": columns_before,
            "columns_after": len(dataframe.columns),
            "details": details,
        }