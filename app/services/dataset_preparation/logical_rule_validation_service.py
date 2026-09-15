from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pandas as pd


class LogicalRuleValidationService:
    """
    Validate dataset rows against user-defined logical rules.

    Rules are evaluated without modifying the dataframe.

    Supported examples:
        age >= 18
        income > 0
        city != "Pune"
        age >= 18 and income > 0
        age < 18 or city == "Pune"
        age + income > 50000
    """

    ALLOWED_NODES = (
        ast.Expression,
        ast.BoolOp,
        ast.Compare,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.And,
        ast.Or,
        ast.Not,
        ast.UnaryOp,
        ast.USub,
        ast.UAdd,
        ast.BinOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.Pow,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Is,
        ast.IsNot,
        ast.In,
        ast.NotIn,
    )

    @classmethod
    def validate_rule_expression(
        cls,
        rule: str,
        dataframe: pd.DataFrame,
    ) -> list[str]:
        """
        Validate the syntax and referenced columns of one rule.
        """
        errors: list[str] = []

        if not isinstance(rule, str):
            errors.append("Each logical rule must be a string.")
            return errors

        rule = rule.strip()

        if not rule:
            errors.append("Logical rule cannot be empty.")
            return errors

        try:
            tree = ast.parse(rule, mode="eval")
        except SyntaxError as exc:
            errors.append(
                f"Invalid logical rule syntax: {exc.msg}."
            )
            return errors

        for node in ast.walk(tree):
            if not isinstance(node, cls.ALLOWED_NODES):
                errors.append(
                    f"Unsupported expression element: "
                    f"{type(node).__name__}."
                )

        referenced_columns = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
        }

        missing_columns = sorted(
            column
            for column in referenced_columns
            if column not in dataframe.columns
        )

        if missing_columns:
            errors.append(
                "Logical rule references columns that do not exist: "
                f"{missing_columns}."
            )

        return errors

    @staticmethod
    def _normalize_value(value: Any) -> Any:
        """
        Convert pandas missing values into None and numpy scalar values
        into normal Python scalar values where possible.
        """
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass

        if hasattr(value, "item"):
            try:
                return value.item()
            except (ValueError, TypeError):
                pass

        return value

    @classmethod
    def _evaluate_node(
        cls,
        node: ast.AST,
        row_values: dict[str, Any],
    ) -> Any:
        """
        Safely evaluate a restricted AST against one dataframe row.
        """

        if isinstance(node, ast.Expression):
            return cls._evaluate_node(
                node.body,
                row_values,
            )

        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, ast.Name):
            return row_values.get(node.id)

        if isinstance(node, ast.BoolOp):
            values = [
                cls._evaluate_node(
                    value,
                    row_values,
                )
                for value in node.values
            ]

            if isinstance(node.op, ast.And):
                return all(values)

            if isinstance(node.op, ast.Or):
                return any(values)

            raise ValueError(
                "Unsupported boolean operator: "
                f"{type(node.op).__name__}"
            )

        if isinstance(node, ast.UnaryOp):
            operand = cls._evaluate_node(
                node.operand,
                row_values,
            )

            if isinstance(node.op, ast.Not):
                return not operand

            if isinstance(node.op, ast.USub):
                return -operand

            if isinstance(node.op, ast.UAdd):
                return +operand

            raise ValueError(
                "Unsupported unary operator: "
                f"{type(node.op).__name__}"
            )

        if isinstance(node, ast.BinOp):
            left = cls._evaluate_node(
                node.left,
                row_values,
            )

            right = cls._evaluate_node(
                node.right,
                row_values,
            )

            if left is None or right is None:
                return None

            if isinstance(node.op, ast.Add):
                return left + right

            if isinstance(node.op, ast.Sub):
                return left - right

            if isinstance(node.op, ast.Mult):
                return left * right

            if isinstance(node.op, ast.Div):
                return left / right

            if isinstance(node.op, ast.Mod):
                return left % right

            if isinstance(node.op, ast.Pow):
                return left**right

            raise ValueError(
                "Unsupported binary operator: "
                f"{type(node.op).__name__}"
            )

        if isinstance(node, ast.Compare):
            left = cls._evaluate_node(
                node.left,
                row_values,
            )

            for operator, comparator in zip(
                node.ops,
                node.comparators,
            ):
                right = cls._evaluate_node(
                    comparator,
                    row_values,
                )

                if isinstance(operator, ast.Eq):
                    result = left == right

                elif isinstance(operator, ast.NotEq):
                    result = left != right

                elif isinstance(operator, ast.Lt):
                    result = (
                        False
                        if left is None or right is None
                        else left < right
                    )

                elif isinstance(operator, ast.LtE):
                    result = (
                        False
                        if left is None or right is None
                        else left <= right
                    )

                elif isinstance(operator, ast.Gt):
                    result = (
                        False
                        if left is None or right is None
                        else left > right
                    )

                elif isinstance(operator, ast.GtE):
                    result = (
                        False
                        if left is None or right is None
                        else left >= right
                    )

                elif isinstance(operator, ast.Is):
                    result = left is right

                elif isinstance(operator, ast.IsNot):
                    result = left is not right

                elif isinstance(operator, ast.In):
                    result = (
                        False
                        if left is None or right is None
                        else left in right
                    )

                elif isinstance(operator, ast.NotIn):
                    result = (
                        True
                        if left is None or right is None
                        else left not in right
                    )

                else:
                    raise ValueError(
                        "Unsupported comparison operator: "
                        f"{type(operator).__name__}"
                    )

                if not result:
                    return False

                left = right

            return True

        raise ValueError(
            f"Unsupported AST node: {type(node).__name__}"
        )

    @classmethod
    def validate(
        cls,
        dataframe: pd.DataFrame,
        rules: list[str],
    ) -> dict[str, Any]:
        """
        Validate a set of logical rules before execution.
        """
        errors: list[str] = []

        if not isinstance(rules, list):
            errors.append(
                "logical_rule_validation 'rules' must be a list."
            )

            return {
                "valid": False,
                "errors": errors,
            }

        if not rules:
            errors.append(
                "logical_rule_validation requires at least one rule."
            )

            return {
                "valid": False,
                "errors": errors,
            }

        duplicate_rules = [
            rule
            for rule in set(rules)
            if rules.count(rule) > 1
        ]

        if duplicate_rules:
            errors.append(
                "logical_rule_validation rules must not contain "
                "duplicates: "
                f"{sorted(duplicate_rules)}."
            )

        for rule in rules:
            errors.extend(
                cls.validate_rule_expression(
                    rule=rule,
                    dataframe=dataframe,
                )
            )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
        }

    @classmethod
    def execute_rules(
        cls,
        dataframe: pd.DataFrame,
        rules: list[str],
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Execute logical validation rules.

        The dataframe is returned unchanged.
        """
        validation = cls.validate(
            dataframe=dataframe,
            rules=rules,
        )

        if not validation["valid"]:
            raise ValueError(
                "Logical rule validation configuration is invalid: "
                f"{validation['errors']}"
            )

        rule_results: list[dict[str, Any]] = []
        total_violations = 0

        for rule_number, rule in enumerate(
            rules,
            start=1,
        ):
            tree = ast.parse(
                rule.strip(),
                mode="eval",
            )

            violations: list[dict[str, Any]] = []

            for index, row in dataframe.iterrows():
                row_values = {
                    column: cls._normalize_value(value)
                    for column, value in row.to_dict().items()
                }

                try:
                    result = cls._evaluate_node(
                        tree,
                        row_values,
                    )

                    passed = bool(result)

                except Exception as exc:
                    raise ValueError(
                        f"Unable to evaluate logical rule "
                        f"'{rule}': {exc}"
                    ) from exc

                if not passed:
                    violations.append(
                        {
                            "row_number": int(index) + 1,
                            "index": cls._normalize_value(index),
                        }
                    )

            violation_count = len(violations)

            total_violations += violation_count

            rule_results.append(
                {
                    "rule_number": rule_number,
                    "rule": rule,
                    "status": (
                        "passed"
                        if violation_count == 0
                        else "violations_found"
                    ),
                    "rows_checked": len(dataframe),
                    "violations_count": violation_count,
                    "violations": violations,
                }
            )

        details = {
            "method": "logical_rule_validation",
            "rules_count": len(rules),
            "rules": rules,
            "rows_checked": len(dataframe),
            "total_violations": total_violations,
            "rules_with_violations": sum(
                1
                for result in rule_results
                if result["violations_count"] > 0
            ),
            "rule_results": rule_results,
            "data_modified": False,
        }

        return dataframe.copy(), details

    @classmethod
    def execute(
        cls,
        file_path: str | Path,
        rules: list[str],
    ) -> dict[str, Any]:
        """
        Direct service execution against a CSV file.

        The input file is not modified.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise ValueError(
                "Dataset file was not found."
            )

        if file_path.suffix.lower() != ".csv":
            raise ValueError(
                "Logical rule validation currently supports "
                "CSV input files."
            )

        try:
            dataframe = pd.read_csv(file_path)

        except Exception as error:
            raise ValueError(
                f"Could not read CSV dataset: {error}"
            ) from error

        prepared_dataframe, details = (
            cls.execute_rules(
                dataframe=dataframe,
                rules=rules,
            )
        )

        output_path = (
            file_path.parent
            / f"{file_path.stem}_logical_rule_validation.csv"
        )

        prepared_dataframe.to_csv(
            output_path,
            index=False,
        )

        return {
            "status": "success",
            "input_file": file_path.name,
            "output_file": output_path.name,
            "output_path": str(output_path),
            "rows_before": len(dataframe),
            "rows_after": len(prepared_dataframe),
            "columns_before": len(dataframe.columns),
            "columns_after": len(prepared_dataframe.columns),
            "details": details,
        }