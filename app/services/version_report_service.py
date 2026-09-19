from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import (
    Dataset,
    MLRun,
    Model,
    Project,
    Version,
    VersionResultEvidence,
)


class VersionReportService:
    """Build a deterministic, evidence-backed report for one DATAGIT version."""

    @staticmethod
    def _git_command(
        project_path: str,
        args: list[str],
    ) -> tuple[bool, str]:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=project_path,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=15,
            )
        except (
            FileNotFoundError,
            OSError,
            subprocess.TimeoutExpired,
        ) as exc:
            return False, str(exc)

        if result.returncode != 0:
            return False, (
                result.stderr.strip()
                or result.stdout.strip()
                or f"git exited with code {result.returncode}"
            )

        return True, result.stdout.strip()

    @staticmethod
    def _profile_rows(
        rows: list[dict[str, Any]],
        columns: list[str],
    ) -> dict[str, Any]:
        missing_by_column: dict[str, int] = {}
        seen: set[str] = set()
        duplicate_count = 0

        for row in rows:
            normalized = {
                column: row.get(column)
                for column in columns
            }

            for column in columns:
                value = normalized.get(column)

                if value is None or value == "":
                    missing_by_column[column] = (
                        missing_by_column.get(column, 0) + 1
                    )

            serialized = json.dumps(
                normalized,
                sort_keys=True,
                default=str,
                ensure_ascii=False,
            )

            if serialized in seen:
                duplicate_count += 1
            else:
                seen.add(serialized)

        return {
            "row_count": len(rows),
            "column_count": len(columns),
            "columns": columns,
            "missing_values": sum(
                missing_by_column.values()
            ),
            "missing_by_column": missing_by_column,
            "duplicate_rows": duplicate_count,
        }

    @staticmethod
    def _profile_csv(
        path: Path,
    ) -> dict[str, Any]:
        delimiter = (
            "\t"
            if path.suffix.lower() == ".tsv"
            else ","
        )

        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            reader = csv.DictReader(
                handle,
                delimiter=delimiter,
            )

            columns = [
                name.strip()
                for name in (reader.fieldnames or [])
                if name is not None
            ]

            rows: list[dict[str, Any]] = []

            for row in reader:
                rows.append(
                    {
                        str(key).strip(): value
                        for key, value in row.items()
                        if key is not None
                    }
                )

        profile = VersionReportService._profile_rows(
            rows,
            columns,
        )

        profile["status"] = "available"
        profile["format"] = (
            "tsv"
            if delimiter == "\t"
            else "csv"
        )

        return profile

    @staticmethod
    def _profile_json(
        path: Path,
    ) -> dict[str, Any]:
        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            payload = json.load(handle)

        if isinstance(payload, list):
            rows = [
                item
                for item in payload
                if isinstance(item, dict)
            ]

            columns: list[str] = []

            for row in rows:
                for key in row:
                    key_text = str(key)

                    if key_text not in columns:
                        columns.append(key_text)

            profile = VersionReportService._profile_rows(
                rows,
                columns,
            )

            profile["status"] = "available"
            profile["format"] = "json"
            profile["document_type"] = "records"

            return profile

        if isinstance(payload, dict):
            scalar_fields = {
                str(key): value
                for key, value in payload.items()
                if not isinstance(value, (dict, list))
            }

            if scalar_fields:
                profile = VersionReportService._profile_rows(
                    [scalar_fields],
                    list(scalar_fields.keys()),
                )

                profile["status"] = "available"
                profile["format"] = "json"
                profile["document_type"] = "object"

                return profile

            return {
                "status": "available",
                "format": "json",
                "document_type": "object",
                "row_count": None,
                "column_count": len(payload),
                "columns": [
                    str(key)
                    for key in payload.keys()
                ],
                "missing_values": None,
                "missing_by_column": {},
                "duplicate_rows": None,
            }

        return {
            "status": "available",
            "format": "json",
            "document_type": type(payload).__name__,
            "row_count": None,
            "column_count": None,
            "columns": [],
            "missing_values": None,
            "missing_by_column": {},
            "duplicate_rows": None,
        }

    @staticmethod
    def _profile_jsonl(
        path: Path,
    ) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []

        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            for line in handle:
                stripped = line.strip()

                if not stripped:
                    continue

                payload = json.loads(stripped)

                if isinstance(payload, dict):
                    rows.append(payload)

        columns: list[str] = []

        for row in rows:
            for key in row:
                key_text = str(key)

                if key_text not in columns:
                    columns.append(key_text)

        profile = VersionReportService._profile_rows(
            rows,
            columns,
        )

        profile["status"] = "available"
        profile["format"] = "jsonl"

        return profile

    @staticmethod
    def _profile_parquet(
        path: Path,
    ) -> dict[str, Any]:
        try:
            import pandas as pd
        except ImportError:
            return {
                "status": "unavailable",
                "format": "parquet",
                "reason": (
                    "pandas is not installed; "
                    "Parquet profiling is unavailable."
                ),
            }

        try:
            frame = pd.read_parquet(path)

            missing_by_column = {
                str(column): int(
                    frame[column].isna().sum()
                )
                for column in frame.columns
            }

            return {
                "status": "available",
                "format": "parquet",
                "row_count": int(len(frame)),
                "column_count": int(len(frame.columns)),
                "columns": [
                    str(column)
                    for column in frame.columns
                ],
                "missing_values": int(
                    frame.isna().sum().sum()
                ),
                "missing_by_column": missing_by_column,
                "duplicate_rows": int(
                    frame.duplicated().sum()
                ),
            }

        except Exception as exc:
            return {
                "status": "unavailable",
                "format": "parquet",
                "reason": str(exc),
            }

    @staticmethod
    def _profile_text(
        path: Path,
    ) -> dict[str, Any]:
        with path.open(
            "r",
            encoding="utf-8",
            errors="replace",
        ) as handle:
            lines = handle.readlines()

        return {
            "status": "available",
            "format": "text",
            "row_count": len(lines),
            "column_count": None,
            "columns": [],
            "missing_values": None,
            "duplicate_rows": None,
            "character_count": sum(
                len(line)
                for line in lines
            ),
        }

    @staticmethod
    def _profile_file(
        project_path: str,
        data_path: str,
        dvc_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        candidate = Path(data_path)

        path = (
            candidate
            if candidate.is_absolute()
            else Path(project_path) / candidate
        )

        result: dict[str, Any] = {
            "data_path": data_path,
            "absolute_path": str(path),
            "file_name": path.name,
            "exists": path.exists(),
            "size_bytes": (
                path.stat().st_size
                if path.exists() and path.is_file()
                else None
            ),
            "dvc": dvc_metadata or {},
        }

        if not path.exists():
            result["profile"] = {
                "status": "unavailable",
                "reason": "Dataset file was not found.",
            }
            return result

        if not path.is_file():
            result["profile"] = {
                "status": "unavailable",
                "reason": "Dataset path is not a file.",
            }
            return result

        try:
            suffix = path.suffix.lower()

            if suffix in {".csv", ".tsv"}:
                profile = VersionReportService._profile_csv(
                    path
                )

            elif suffix == ".json":
                profile = VersionReportService._profile_json(
                    path
                )

            elif suffix in {".jsonl", ".ndjson"}:
                profile = VersionReportService._profile_jsonl(
                    path
                )

            elif suffix == ".parquet":
                profile = VersionReportService._profile_parquet(
                    path
                )

            elif suffix in {".txt", ".log"}:
                profile = VersionReportService._profile_text(
                    path
                )

            else:
                profile = {
                    "status": "unavailable",
                    "format": (
                        suffix.lstrip(".")
                        or "unknown"
                    ),
                    "reason": (
                        "No deterministic profiler is "
                        "implemented for this file type."
                    ),
                }

            result["profile"] = profile

        except Exception as exc:
            result["profile"] = {
                "status": "unavailable",
                "reason": str(exc),
            }

        return result

    @staticmethod
    def _git_report(
        project_path: str,
        commit: str,
    ) -> dict[str, Any]:
        report: dict[str, Any] = {
            "status": (
                "recorded"
                if commit
                else "not_recorded"
            ),
            "commit": commit,
            "changed_files": [],
            "changed_file_count": 0,
        }

        if not commit:
            return report

        ok, output = VersionReportService._git_command(
            project_path,
            [
                "show",
                "-s",
                "--format=%H%n%an%n%ae%n%ad%n%s",
                "--date=iso-strict",
                commit,
            ],
        )

        if ok:
            lines = output.splitlines()

            if len(lines) >= 1:
                report["commit"] = lines[0]

            if len(lines) >= 2:
                report["author"] = lines[1]

            if len(lines) >= 3:
                report["author_email"] = lines[2]

            if len(lines) >= 4:
                report["committed_at"] = lines[3]

            if len(lines) >= 5:
                report["message"] = "\n".join(
                    lines[4:]
                )

        else:
            report["metadata_status"] = "unavailable"
            report["metadata_error"] = output

        ok, output = VersionReportService._git_command(
            project_path,
            [
                "show",
                "--format=",
                "--name-status",
                "--find-renames",
                commit,
            ],
        )

        if not ok:
            report["changed_files_status"] = (
                "unavailable"
            )
            report["changed_files_error"] = output
            return report

        changed_files: list[dict[str, str]] = []

        for line in output.splitlines():
            if not line.strip():
                continue

            parts = line.split("\t")

            if len(parts) >= 2:
                item = {
                    "status": parts[0],
                    "path": parts[-1],
                }

                if len(parts) >= 3:
                    item["previous_path"] = parts[1]

                changed_files.append(item)

        report["changed_files"] = changed_files
        report["changed_file_count"] = len(
            changed_files
        )

        return report

    @staticmethod
    def _dataset_report(
        project: Project,
        datasets: list[Dataset],
        dvc_state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        tracked_files: list[dict[str, Any]] = []

        if isinstance(dvc_state, dict):
            raw = dvc_state.get(
                "tracked_files",
                [],
            )

            if isinstance(raw, list):
                tracked_files = [
                    item
                    for item in raw
                    if isinstance(item, dict)
                    and item.get("data_path")
                ]

        files: list[dict[str, Any]] = []

        for item in tracked_files:
            data_path = str(
                item.get("data_path")
            ).strip()

            dvc_file = str(
                item.get("dvc_file")
                or ""
            ).strip()

            if not data_path:
                continue

            if dvc_file:
                dvc_parent = Path(
                    dvc_file
                ).parent

                resolved_relative = (
                    dvc_parent / Path(data_path)
                )

                resolved_data_path = (
                    resolved_relative.as_posix()
                )
            else:
                resolved_data_path = data_path

            files.append(
                VersionReportService._profile_file(
                    str(project.path),
                    resolved_data_path,
                    item,
                )
            )

        if not files:
            for dataset in datasets:
                files.append(
                    VersionReportService._profile_file(
                        str(project.path),
                        dataset.path,
                        {
                            "dataset_id": dataset.id,
                            "name": dataset.name,
                        },
                    )
                )

        available_files = [
            item
            for item in files
            if isinstance(
                item.get("profile"),
                dict,
            )
            and item["profile"].get("status")
            == "available"
        ]

        if not available_files:
            return {
                "status": "not_recorded",
                "name": None,
                "path": None,
                "row_count": None,
                "column_count": None,
                "columns": [],
                "missing_values": None,
                "missing_by_column": {},
                "duplicate_rows": None,
                "files": files,
                "registered_datasets": [
                    {
                        "id": dataset.id,
                        "name": dataset.name,
                        "path": dataset.path,
                    }
                    for dataset in datasets
                ],
                "reason": (
                    "A dataset was referenced by DVC or the "
                    "project registry, but its file could not "
                    "be profiled from the current project path."
                ),
            }

        primary_file = available_files[0]
        primary_profile = primary_file["profile"]

        return {
            "status": "available",
            "name": primary_file.get(
                "file_name"
            ),
            "path": primary_file.get(
                "data_path"
            ),
            "absolute_path": primary_file.get(
                "absolute_path"
            ),
            "exists": primary_file.get(
                "exists"
            ),
            "size_bytes": primary_file.get(
                "size_bytes"
            ),
            "row_count": primary_profile.get(
                "row_count"
            ),
            "column_count": primary_profile.get(
                "column_count"
            ),
            "columns": primary_profile.get(
                "columns",
                [],
            ),
            "missing_values": primary_profile.get(
                "missing_values"
            ),
            "missing_by_column": primary_profile.get(
                "missing_by_column",
                {},
            ),
            "duplicate_rows": primary_profile.get(
                "duplicate_rows"
            ),
            "format": primary_profile.get(
                "format"
            ),
            "files": files,
            "registered_datasets": [
                {
                    "id": dataset.id,
                    "name": dataset.name,
                    "path": dataset.path,
                    "created_at": (
                        dataset.created_at.isoformat()
                        if dataset.created_at
                        else None
                    ),
                }
                for dataset in datasets
            ],
        }

    @staticmethod
    def _preparation_report(
        version: Version,
    ) -> dict[str, Any]:
        evidence = version.preparation_evidence

        if evidence is None:
            return {
                "status": "not_recorded",
                "operations": [],
                "operation_count": 0,
                "reason": (
                    "No preparation evidence was "
                    "recorded for this version."
                ),
            }

        operations = (
            evidence.operations
            if isinstance(
                evidence.operations,
                list,
            )
            else []
        )

        return {
            "status": "recorded",
            "operations": operations,
            "operation_count": len(
                operations
            ),
            "reason": (
                "Preparation evidence exists, "
                "but no operations were recorded."
                if not operations
                else None
            ),
        }

    @staticmethod
    def _result_evidence_report(
        evidence: VersionResultEvidence | None,
    ) -> dict[str, Any]:
        if evidence is None:
            return {
                "status": "not_recorded",
                "model": None,
                "metrics": {},
                "evaluation": None,
                "notes": None,
                "reason": (
                    "No model, metrics, or evaluation "
                    "evidence was attached to this version."
                ),
            }

        model_recorded = any(
            value is not None
            for value in [
                evidence.model_name,
                evidence.model_path,
                evidence.model_sha256,
                evidence.framework,
                evidence.framework_version,
            ]
        )

        metrics = (
            evidence.metrics
            if isinstance(
                evidence.metrics,
                dict,
            )
            else {}
        )

        evaluation = (
            evidence.evaluation
            if isinstance(
                evidence.evaluation,
                dict,
            )
            else None
        )

        return {
            "status": "recorded",
            "model": (
                {
                    "name": evidence.model_name,
                    "path": evidence.model_path,
                    "sha256": evidence.model_sha256,
                    "framework": evidence.framework,
                    "framework_version": (
                        evidence.framework_version
                    ),
                }
                if model_recorded
                else None
            ),
            "metrics": metrics,
            "evaluation": evaluation,
            "notes": evidence.notes,
            "reason": None,
        }

    @staticmethod
    def _legacy_training_report(
        ml_run: MLRun | None,
    ) -> dict[str, Any]:
        if ml_run is None:
            return {
                "status": "not_recorded",
                "ml_run_id": None,
            }

        return {
            "status": "legacy",
            "ml_run_id": ml_run.id,
            "model_name": ml_run.model_name,
            "metrics": ml_run.metrics,
            "evaluation": ml_run.evaluation,
        }

    @staticmethod
    def _evidence_report(
        dataset_report: dict[str, Any],
        preparation_report: dict[str, Any],
        result_evidence_report: dict[str, Any],
        git_report: dict[str, Any],
        dvc_state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        checks = {
            "version_identity": "recorded",
            "dataset": dataset_report.get(
                "status",
                "unknown",
            ),
            "data_quality": (
                "recorded"
                if dataset_report.get(
                    "status"
                ) == "available"
                else "not_recorded"
            ),
            "preparation": preparation_report.get(
                "status",
                "unknown",
            ),
            "model": (
                "recorded"
                if result_evidence_report.get(
                    "model"
                )
                else "not_recorded"
            ),
            "metrics": (
                "recorded"
                if result_evidence_report.get(
                    "metrics"
                )
                else "not_recorded"
            ),
            "evaluation": (
                "recorded"
                if result_evidence_report.get(
                    "evaluation"
                )
                else "not_recorded"
            ),
            "git": git_report.get(
                "status",
                "unknown",
            ),
            "dvc": (
                "recorded"
                if dvc_state is not None
                else "not_recorded"
            ),
        }

        recorded_count = sum(
            status in {"recorded", "available"}
            for status in checks.values()
        )

        overall = (
            "complete"
            if recorded_count == len(checks)
            else "partial"
        )

        return {
            "status": overall,
            "recorded_sections": recorded_count,
            "total_sections": len(checks),
            "checks": checks,
            "notes": [
                (
                    "Missing evidence is reported as "
                    "not recorded."
                ),
                (
                    "Missing evidence is not treated "
                    "as proof that no change occurred."
                ),
            ],
        }

    @staticmethod
    def build_report(
        db: Session,
        project_id: int,
        version_id: int,
    ) -> dict[str, Any]:
        project = (
            db.query(Project)
            .filter(Project.id == project_id)
            .first()
        )

        if project is None:
            raise ValueError(
                "Project not found."
            )

        version = (
            db.query(Version)
            .filter(
                Version.id == version_id,
                Version.project_id == project_id,
            )
            .first()
        )

        if version is None:
            raise ValueError(
                "Version not found."
            )

        datasets = (
            db.query(Dataset)
            .filter(
                Dataset.project_id == project_id
            )
            .order_by(
                Dataset.created_at.asc()
            )
            .all()
        )

        result_evidence = (
            db.query(
                VersionResultEvidence
            )
            .filter(
                VersionResultEvidence.version_id
                == version.id
            )
            .first()
        )

        legacy_ml_run = None

        if version.ml_run_id is not None:
            legacy_ml_run = (
                db.query(MLRun)
                .filter(
                    MLRun.id == version.ml_run_id,
                    MLRun.project_id == project_id,
                )
                .first()
            )

        dataset_report = (
            VersionReportService._dataset_report(
                project,
                datasets,
                version.dvc_state,
            )
        )

        preparation_report = (
            VersionReportService._preparation_report(
                version
            )
        )

        result_evidence_report = (
            VersionReportService._result_evidence_report(
                result_evidence
            )
        )

        git_report = (
            VersionReportService._git_report(
                str(project.path),
                version.git_commit,
            )
        )

        evidence_report = (
            VersionReportService._evidence_report(
                dataset_report,
                preparation_report,
                result_evidence_report,
                git_report,
                version.dvc_state,
            )
        )

        return {
            "id": version.id,
            "project_id": version.project_id,
            "version_number": version.version_number,
            "git_commit": version.git_commit,
            "dvc_state": version.dvc_state,
            "description": version.description,
            "ml_run_id": version.ml_run_id,
            "created_at": version.created_at,

            "project": {
                "id": project.id,
                "name": project.name,
                "path": project.path,
                "description": project.description,
                "created_at": project.created_at,
                "updated_at": project.updated_at,
            },

            "dataset": dataset_report,

            "data_quality": {
                "status": (
                    "available"
                    if dataset_report.get(
                        "status"
                    ) == "available"
                    else "not_recorded"
                ),
                "missing_values": dataset_report.get(
                    "missing_values"
                ),
                "missing_by_column": dataset_report.get(
                    "missing_by_column",
                    {},
                ),
                "duplicate_rows": dataset_report.get(
                    "duplicate_rows"
                ),
                "validity": (
                    "assessable"
                    if dataset_report.get(
                        "status"
                    ) == "available"
                    else "not_assessable"
                ),
            },

            "preparation": preparation_report,

            # New version-centric evidence.
            "result_evidence": result_evidence_report,

            # Compatibility aliases for the current frontend.
            "model": result_evidence_report.get(
                "model"
            ),
            "performance": {
                "status": (
                    "recorded"
                    if result_evidence_report.get(
                        "metrics"
                    )
                    else "not_recorded"
                ),
                "metrics": result_evidence_report.get(
                    "metrics",
                    {},
                ),
            },
            "evaluation": {
                "status": (
                    "recorded"
                    if result_evidence_report.get(
                        "evaluation"
                    )
                    else "not_recorded"
                ),
                "evaluation": result_evidence_report.get(
                    "evaluation"
                ),
            },

            # Existing legacy field is kept only so older
            # clients do not crash. It is NOT used as
            # the new evidence source.
            "training": VersionReportService._legacy_training_report(
                legacy_ml_run
            ),

            "git": git_report,
            "dvc": version.dvc_state,

            "lineage": {
                "status": "partial",
                "project": project.name,
                "dataset": dataset_report.get(
                    "name"
                ),
                "preparation": preparation_report.get(
                    "status"
                ),
                "dvc": (
                    "recorded"
                    if version.dvc_state is not None
                    else "not_recorded"
                ),
                "model": (
                    "recorded"
                    if result_evidence_report.get(
                        "model"
                    )
                    else "not_recorded"
                ),
                "metrics": (
                    "recorded"
                    if result_evidence_report.get(
                        "metrics"
                    )
                    else "not_recorded"
                ),
                "evaluation": (
                    "recorded"
                    if result_evidence_report.get(
                        "evaluation"
                    )
                    else "not_recorded"
                ),
                "git": git_report.get(
                    "status"
                ),
                "version": (
                    f"V{version.version_number}"
                ),
            },

            "evidence_completeness": evidence_report,

            "report_metadata": {
                "type": "version_report",
                "source": (
                    "deterministic_datagit_evidence"
                ),
                "ai_generated": False,
            },

            "ai_report": None,
        }