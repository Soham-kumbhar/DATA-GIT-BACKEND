from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from dvc import api as dvc_api
except Exception:  # pragma: no cover - optional runtime dependency
    dvc_api = None

try:
    from git import Repo
except Exception:  # pragma: no cover - optional runtime dependency
    Repo = None

try:
    from groq import Groq
except Exception:  # pragma: no cover - optional runtime dependency
    Groq = None

from sqlalchemy.orm import Session

from app.db.models import MLRun, Project, Version


@dataclass
class CompareEvidence:
    status: str
    evidence_level: str
    summary: dict[str, Any]
    provenance: dict[str, Any]
    dataset: dict[str, Any]
    schema_diff: dict[str, Any]
    quality: dict[str, Any]
    drift: dict[str, Any]
    model: dict[str, Any]
    performance: dict[str, Any]
    code: dict[str, Any]
    preparation: dict[str, Any]
    validity: dict[str, Any]
    recommendations: list[str]
    limitations: list[str]


class DeepCompareService:
    """
    Deterministic V1 -> V2 comparison engine.

    Important rules:
    - Versions from the same project remain comparable even when evidence is partial.
    - This service never mutates Git, DVC, datasets, or DATAGIT version records.
    - Historical datasets are read through DVC using the Git revision stored on each version.
    - AI receives the deterministic evidence package, not arbitrary raw project state.
    """

    MAX_DATASET_ROWS = int(
        os.getenv("DATAGIT_COMPARE_MAX_ROWS", "100000")
    )

    NUMERIC_COLUMNS_LIMIT = int(
        os.getenv("DATAGIT_COMPARE_MAX_FEATURES", "250")
    )

    MODEL_FILE_LIMIT = int(
        os.getenv("DATAGIT_COMPARE_MODEL_FILE_LIMIT", "12")
    )

    MODEL_CODE_CHAR_LIMIT = int(
        os.getenv("DATAGIT_COMPARE_MODEL_CODE_LIMIT", "12000")
    )

    @classmethod
    def compare(
        cls,
        db: Session,
        project_id: int,
        version_a_id: int,
        version_b_id: int,
        generate_ai: bool = True,
    ) -> dict[str, Any]:
        project = db.query(Project).filter(Project.id == project_id).first()

        if project is None:
            raise ValueError(
                f"Project {project_id} was not found."
            )

        version_a = (
            db.query(Version)
            .filter(
                Version.id == version_a_id,
                Version.project_id == project_id,
            )
            .first()
        )

        if version_a is None:
            raise ValueError(
                f"Baseline version {version_a_id} was not found "
                f"for project {project_id}."
            )

        version_b = (
            db.query(Version)
            .filter(
                Version.id == version_b_id,
                Version.project_id == project_id,
            )
            .first()
        )

        if version_b is None:
            raise ValueError(
                f"Target version {version_b_id} was not found "
                f"for project {project_id}."
            )

        if version_a.id == version_b.id:
            raise ValueError(
                "Version A and Version B must be different."
            )

        ml_run_a = cls._load_ml_run(
            db,
            project_id=project_id,
            ml_run_id=getattr(version_a, "ml_run_id", None),
        )

        ml_run_b = cls._load_ml_run(
            db,
            project_id=project_id,
            ml_run_id=getattr(version_b, "ml_run_id", None),
        )

        project_path = cls._project_path(project)

        dataset_a = cls._load_dataset_evidence(
            project_path=project_path,
            version=version_a,
        )

        dataset_b = cls._load_dataset_evidence(
            project_path=project_path,
            version=version_b,
        )

        schema_diff = cls._compare_schema(
            dataset_a.get("profile"),
            dataset_b.get("profile"),
        )

        quality = cls._compare_quality(
            dataset_a.get("profile"),
            dataset_b.get("profile"),
        )

        drift = cls._compare_drift(
            dataset_a.get("dataframe"),
            dataset_b.get("dataframe"),
            ml_run_a,
            ml_run_b,
        )

        performance = cls._compare_performance(
            ml_run_a,
            ml_run_b,
        )

        model = cls._compare_model_evidence(
            ml_run_a,
            ml_run_b,
        )

        code = cls._compare_git_evidence(
            project_path=project_path,
            version_a=version_a,
            version_b=version_b,
        )

        preparation = cls._compare_preparation(
            version_a=version_a,
            version_b=version_b,
            ml_run_a=ml_run_a,
            ml_run_b=ml_run_b,
        )

        provenance = cls._build_provenance(
            version_a,
            version_b,
        )

        validity = cls._build_validity(
            version_a=version_a,
            version_b=version_b,
            dataset_a=dataset_a,
            dataset_b=dataset_b,
            ml_run_a=ml_run_a,
            ml_run_b=ml_run_b,
        )

        summary = cls._build_summary(
            version_a=version_a,
            version_b=version_b,
            dataset_a=dataset_a,
            dataset_b=dataset_b,
            schema_diff=schema_diff,
            quality=quality,
            drift=drift,
            model=model,
            performance=performance,
            code=code,
            validity=validity,
        )

        limitations = cls._collect_limitations(
            dataset_a=dataset_a,
            dataset_b=dataset_b,
            ml_run_a=ml_run_a,
            ml_run_b=ml_run_b,
            validity=validity,
            drift=drift,
            performance=performance,
        )

        recommendations = cls._build_recommendations(
            drift=drift,
            performance=performance,
            schema_diff=schema_diff,
            quality=quality,
            model=model,
            validity=validity,
        )

        evidence = CompareEvidence(
            status=summary["status"],
            evidence_level=summary["evidence_level"],
            summary=summary,
            provenance=provenance,
            dataset={
                "baseline": cls._without_dataframe(dataset_a),
                "target": cls._without_dataframe(dataset_b),
            },
            schema_diff=schema_diff,
            quality=quality,
            drift=drift,
            model=model,
            performance=performance,
            code=code,
            preparation=preparation,
            validity=validity,
            recommendations=recommendations,
            limitations=limitations,
        )

        deterministic_payload = asdict(evidence)

        ai_result: dict[str, Any] = {
            "status": "not_requested",
            "report": None,
            "provider": None,
        }

        if generate_ai:
            ai_result = cls._generate_ai_report(
                project=project,
                version_a=version_a,
                version_b=version_b,
                evidence=deterministic_payload,
            )

        return {
            "project": {
                "id": project.id,
                "name": getattr(project, "name", None),
                "path": getattr(project, "path", None),
            },
            "version_a": cls._version_payload(
                version_a,
                ml_run=ml_run_a,
            ),
            "version_b": cls._version_payload(
                version_b,
                ml_run=ml_run_b,
            ),
            "comparison": deterministic_payload,
            "ai_insights": ai_result,
        }

    # ------------------------------------------------------------------
    # Version / project
    # ------------------------------------------------------------------

    @classmethod
    def _project_path(
        cls,
        project: Project,
    ) -> str | None:
        value = getattr(project, "path", None)

        if not value:
            return None

        try:
            return str(Path(value).expanduser().resolve())
        except Exception:
            return str(value)

    @classmethod
    def _version_payload(
        cls,
        version: Version,
        ml_run: MLRun | None,
    ) -> dict[str, Any]:
        return {
            "id": version.id,
            "project_id": version.project_id,
            "version_number": version.version_number,
            "git_commit": version.git_commit,
            "dvc_state": cls._json_safe(
                getattr(version, "dvc_state", None)
            ),
            "description": version.description,
            "created_at": cls._json_safe(
                version.created_at
            ),
            "ml_run_id": version.ml_run_id,
            "ml_run": cls._ml_run_payload(ml_run),
        }

    @classmethod
    def _load_ml_run(
        cls,
        db: Session,
        project_id: int,
        ml_run_id: int | None,
    ) -> MLRun | None:
        if ml_run_id is None:
            return None

        return (
            db.query(MLRun)
            .filter(
                MLRun.id == ml_run_id,
                MLRun.project_id == project_id,
            )
            .first()
        )

    @classmethod
    def _ml_run_payload(
        cls,
        ml_run: MLRun | None,
    ) -> dict[str, Any] | None:
        if ml_run is None:
            return None

        return {
            "id": ml_run.id,
            "project_id": ml_run.project_id,
            "git_commit": cls._json_safe(
                getattr(ml_run, "git_commit", None)
            ),
            "dvc_state": cls._json_safe(
                getattr(ml_run, "dvc_state", None)
            ),
            "model_name": cls._json_safe(
                getattr(ml_run, "model_name", None)
            ),
            "features": cls._json_safe(
                getattr(ml_run, "features", None)
            ),
            "parameters": cls._json_safe(
                getattr(ml_run, "parameters", None)
            ),
            "metrics": cls._json_safe(
                getattr(ml_run, "metrics", None)
            ),
            "evaluation": cls._json_safe(
                getattr(ml_run, "evaluation", None)
            ),
            "created_at": cls._json_safe(
                getattr(ml_run, "created_at", None)
            ),
        }

    # ------------------------------------------------------------------
    # DVC / dataset inspection
    # ------------------------------------------------------------------

    @classmethod
    def _load_dataset_evidence(
        cls,
        project_path: str | None,
        version: Version,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "available": False,
            "source": "dvc",
            "git_revision": version.git_commit,
            "tracked_files": [],
            "selected_file": None,
            "profile": None,
            "dataframe": None,
            "limitations": [],
        }

        dvc_state = getattr(version, "dvc_state", None)

        if isinstance(dvc_state, dict):
            tracked = dvc_state.get("tracked_files") or []

            result["tracked_files"] = cls._json_safe(tracked)

            selected = cls._select_tracked_file(
                tracked
            )

            if selected:
                result["selected_file"] = selected

        if not project_path:
            result["limitations"].append(
                "Project path is unavailable, so historical DVC "
                "dataset content could not be loaded."
            )
            return result

        if dvc_api is None:
            result["limitations"].append(
                "The DVC Python API is not available in the backend "
                "environment."
            )
            return result

        selected_file = result.get("selected_file")

        if not selected_file:
            result["limitations"].append(
                "No tracked dataset artifact was recorded on this version."
            )
            return result

        data_path = cls._tracked_data_path(
            selected_file
        )

        if not data_path:
            result["limitations"].append(
                "The DVC tracked artifact does not contain a resolvable "
                "dataset path."
            )
            return result

        try:
            dataframe = cls._read_dvc_dataset(
                project_path=project_path,
                git_revision=version.git_commit,
                data_path=data_path,
            )

            if dataframe is None:
                result["limitations"].append(
                    "The tracked dataset format is not currently supported "
                    "by Compare."
                )
                return result

            original_rows = len(dataframe)

            sampled = False

            if original_rows > cls.MAX_DATASET_ROWS:
                dataframe = dataframe.head(
                    cls.MAX_DATASET_ROWS
                )
                sampled = True

            result["available"] = True
            result["dataframe"] = dataframe
            result["profile"] = cls._profile_dataframe(
                dataframe,
                sampled=sampled,
                original_rows=original_rows,
                data_path=data_path,
                tracked_file=selected_file,
            )

            return result

        except Exception as exc:
            result["limitations"].append(
                f"Historical DVC dataset could not be read: {exc}"
            )
            return result

    @classmethod
    def _select_tracked_file(
        cls,
        tracked_files: list[Any],
    ) -> dict[str, Any] | None:
        if not isinstance(tracked_files, list):
            return None

        candidates = []

        for item in tracked_files:
            if not isinstance(item, dict):
                continue

            data_path = item.get("data_path")
            dvc_file = item.get("dvc_file")

            candidate = {
                "dvc_file": dvc_file,
                "data_path": data_path,
                "md5": item.get("md5"),
                "size": item.get("size"),
            }

            if data_path:
                candidates.append(candidate)

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: str(
                item.get("data_path") or ""
            )
        )

        return candidates[0]

    @classmethod
    def _tracked_data_path(
        cls,
        tracked_file: dict[str, Any],
    ) -> str | None:
        data_path = tracked_file.get("data_path")

        if data_path:
            normalized = str(
                data_path
            ).replace("\\", "/").lstrip("/")

            if normalized:
                # Some stored DVC metadata has data_path=filename only
                # while dvc_file contains the parent directory.
                dvc_file = tracked_file.get("dvc_file")

                if dvc_file and "/" not in normalized:
                    dvc_parent = str(
                        dvc_file
                    ).replace("\\", "/")

                    if dvc_parent.endswith(".dvc"):
                        dvc_parent = dvc_parent[
                            : -len(".dvc")
                        ]

                    parent = str(
                        Path(dvc_parent).parent
                    ).replace("\\", "/")

                    if parent not in {"", "."}:
                        normalized = (
                            f"{parent}/{normalized}"
                        )

                return normalized

        dvc_file = tracked_file.get("dvc_file")

        if dvc_file:
            normalized = str(
                dvc_file
            ).replace("\\", "/")

            if normalized.endswith(".dvc"):
                normalized = normalized[
                    : -len(".dvc")
                ]

            return normalized.lstrip("/")

        return None

    @classmethod
    def _read_dvc_dataset(
        cls,
        project_path: str,
        git_revision: str,
        data_path: str,
    ) -> pd.DataFrame | None:
        if not git_revision:
            raise ValueError(
                "Version does not contain a Git revision."
            )

        suffix = Path(data_path).suffix.lower()

        with dvc_api.open(
            data_path,
            repo=project_path,
            rev=git_revision,
            mode="rb",
        ) as handle:
            if suffix == ".csv":
                return pd.read_csv(
                    handle,
                    nrows=cls.MAX_DATASET_ROWS,
                )

            if suffix == ".parquet":
                dataframe = pd.read_parquet(handle)
                return dataframe.head(
                    cls.MAX_DATASET_ROWS
                )

            if suffix in {".json", ".jsonl"}:
                raw = handle.read()

                if isinstance(raw, bytes):
                    raw = raw.decode(
                        "utf-8",
                        errors="replace",
                    )

                if suffix == ".jsonl":
                    return pd.read_json(
                        io.StringIO(raw),
                        lines=True,
                    ).head(
                        cls.MAX_DATASET_ROWS
                    )

                loaded = json.loads(raw)

                if isinstance(loaded, list):
                    return pd.DataFrame(
                        loaded
                    ).head(
                        cls.MAX_DATASET_ROWS
                    )

                if isinstance(loaded, dict):
                    for key in (
                        "data",
                        "rows",
                        "records",
                    ):
                        value = loaded.get(key)

                        if isinstance(
                            value,
                            list,
                        ):
                            return pd.DataFrame(
                                value
                            ).head(
                                cls.MAX_DATASET_ROWS
                            )

                    return pd.DataFrame(
                        [loaded]
                    )

        return None

    # ------------------------------------------------------------------
    # Dataset profile / schema / quality
    # ------------------------------------------------------------------

    @classmethod
    def _profile_dataframe(
        cls,
        dataframe: pd.DataFrame,
        sampled: bool,
        original_rows: int,
        data_path: str,
        tracked_file: dict[str, Any],
    ) -> dict[str, Any]:
        columns = []

        for column in dataframe.columns:
            series = dataframe[column]

            column_type = (
                "numeric"
                if pd.api.types.is_numeric_dtype(series)
                else "datetime"
                if pd.api.types.is_datetime64_any_dtype(
                    series
                )
                else "categorical"
            )

            missing_count = int(
                series.isna().sum()
            )

            item: dict[str, Any] = {
                "name": str(column),
                "dtype": str(series.dtype),
                "kind": column_type,
                "missing_count": missing_count,
                "missing_percent": round(
                    (
                        missing_count
                        / len(dataframe)
                        * 100
                    )
                    if len(dataframe)
                    else 0.0,
                    4,
                ),
                "unique_count": int(
                    series.nunique(
                        dropna=True
                    )
                ),
            }

            if column_type == "numeric":
                numeric = pd.to_numeric(
                    series,
                    errors="coerce",
                ).dropna()

                if not numeric.empty:
                    q1 = float(
                        numeric.quantile(0.25)
                    )
                    q3 = float(
                        numeric.quantile(0.75)
                    )
                    iqr = q3 - q1

                    lower = q1 - (
                        1.5 * iqr
                    )
                    upper = q3 + (
                        1.5 * iqr
                    )

                    outliers = int(
                        (
                            (numeric < lower)
                            | (numeric > upper)
                        ).sum()
                    )

                    item["statistics"] = {
                        "min": cls._number(
                            numeric.min()
                        ),
                        "max": cls._number(
                            numeric.max()
                        ),
                        "mean": cls._number(
                            numeric.mean()
                        ),
                        "median": cls._number(
                            numeric.median()
                        ),
                        "std": cls._number(
                            numeric.std()
                        ),
                        "q1": cls._number(
                            q1
                        ),
                        "q3": cls._number(
                            q3
                        ),
                    }

                    item[
                        "outlier_count"
                    ] = outliers

                    item[
                        "outlier_percent"
                    ] = round(
                        (
                            outliers
                            / len(numeric)
                            * 100
                        )
                        if len(numeric)
                        else 0.0,
                        4,
                    )

            else:
                counts = (
                    series
                    .astype("string")
                    .fillna("<MISSING>")
                    .value_counts(
                        normalize=True
                    )
                    .head(20)
                )

                item[
                    "top_categories"
                ] = {
                    str(key): round(
                        float(value) * 100,
                        4,
                    )
                    for key, value in counts.items()
                }

            columns.append(item)

        missing_total = int(
            dataframe.isna().sum().sum()
        )

        duplicate_rows = int(
            dataframe.duplicated().sum()
        )

        return {
            "available": True,
            "data_path": data_path,
            "tracked_file": cls._json_safe(
                tracked_file
            ),
            "rows": int(
                len(dataframe)
            ),
            "original_rows": int(
                original_rows
            ),
            "sampled": sampled,
            "columns": int(
                len(dataframe.columns)
            ),
            "missing_values": missing_total,
            "missing_percent": round(
                (
                    missing_total
                    / (
                        len(dataframe)
                        * len(dataframe.columns)
                    )
                    * 100
                )
                if (
                    len(dataframe)
                    and len(dataframe.columns)
                )
                else 0.0,
                4,
            ),
            "duplicate_rows": duplicate_rows,
            "duplicate_percent": round(
                (
                    duplicate_rows
                    / len(dataframe)
                    * 100
                )
                if len(dataframe)
                else 0.0,
                4,
            ),
            "columns_profile": columns,
        }

    @classmethod
    def _compare_schema(
        cls,
        profile_a: dict[str, Any] | None,
        profile_b: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not profile_a or not profile_b:
            return {
                "available": False,
                "status": "unavailable",
                "added": [],
                "removed": [],
                "modified": [],
            }

        a_map = {
            item["name"]: item
            for item in (
                profile_a.get(
                    "columns_profile"
                )
                or []
            )
            if isinstance(item, dict)
            and item.get("name")
        }

        b_map = {
            item["name"]: item
            for item in (
                profile_b.get(
                    "columns_profile"
                )
                or []
            )
            if isinstance(item, dict)
            and item.get("name")
        }

        added = []
        removed = []
        modified = []

        for name in sorted(
            set(b_map) - set(a_map)
        ):
            added.append(
                {
                    "column": name,
                    "target": b_map[name].get(
                        "dtype"
                    ),
                }
            )

        for name in sorted(
            set(a_map) - set(b_map)
        ):
            removed.append(
                {
                    "column": name,
                    "baseline": a_map[name].get(
                        "dtype"
                    ),
                }
            )

        for name in sorted(
            set(a_map) & set(b_map)
        ):
            a_type = a_map[name].get(
                "dtype"
            )
            b_type = b_map[name].get(
                "dtype"
            )

            if a_type != b_type:
                modified.append(
                    {
                        "column": name,
                        "baseline": a_type,
                        "target": b_type,
                        "change": "dtype",
                    }
                )

        return {
            "available": True,
            "status": (
                "changed"
                if (
                    added
                    or removed
                    or modified
                )
                else "unchanged"
            ),
            "added": added,
            "removed": removed,
            "modified": modified,
            "counts": {
                "added": len(added),
                "removed": len(removed),
                "modified": len(modified),
            },
        }

    @classmethod
    def _compare_quality(
        cls,
        profile_a: dict[str, Any] | None,
        profile_b: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not profile_a or not profile_b:
            return {
                "available": False,
                "status": "unavailable",
                "metrics": [],
            }

        metrics = []

        definitions = [
            (
                "rows",
                profile_a.get("rows"),
                profile_b.get("rows"),
                "higher_is_not_automatically_better",
            ),
            (
                "columns",
                profile_a.get("columns"),
                profile_b.get("columns"),
                "informational",
            ),
            (
                "missing_percent",
                profile_a.get(
                    "missing_percent"
                ),
                profile_b.get(
                    "missing_percent"
                ),
                "lower_is_better",
            ),
            (
                "duplicate_percent",
                profile_a.get(
                    "duplicate_percent"
                ),
                profile_b.get(
                    "duplicate_percent"
                ),
                "lower_is_better",
            ),
        ]

        for (
            name,
            baseline,
            target,
            interpretation,
        ) in definitions:
            if (
                baseline is None
                or target is None
            ):
                continue

            metrics.append(
                {
                    "name": name,
                    "baseline": cls._number(
                        baseline
                    ),
                    "target": cls._number(
                        target
                    ),
                    "delta": cls._number(
                        float(target)
                        - float(baseline)
                    ),
                    "interpretation": interpretation,
                }
            )

        return {
            "available": True,
            "status": (
                "changed"
                if any(
                    metric["delta"] != 0
                    for metric in metrics
                )
                else "unchanged"
            ),
            "metrics": metrics,
        }

    # ------------------------------------------------------------------
    # Drift
    # ------------------------------------------------------------------

    @classmethod
    def _compare_drift(
        cls,
        dataframe_a: pd.DataFrame | None,
        dataframe_b: pd.DataFrame | None,
        ml_run_a: MLRun | None,
        ml_run_b: MLRun | None,
    ) -> dict[str, Any]:
        if (
            dataframe_a is None
            or dataframe_b is None
        ):
            return {
                "available": False,
                "status": "unavailable",
                "method_summary": {},
                "features": [],
                "target_distribution": None,
                "prediction_distribution": cls._prediction_distribution(
                    ml_run_a,
                    ml_run_b,
                ),
            }

        shared = [
            column
            for column in dataframe_a.columns
            if column in dataframe_b.columns
        ]

        shared = shared[
            : cls.NUMERIC_COLUMNS_LIMIT
        ]

        findings = []

        for column in shared:
            a = dataframe_a[column]
            b = dataframe_b[column]

            if (
                pd.api.types.is_numeric_dtype(a)
                and pd.api.types.is_numeric_dtype(b)
            ):
                finding = cls._numeric_drift(
                    column,
                    a,
                    b,
                )
            else:
                finding = cls._categorical_drift(
                    column,
                    a,
                    b,
                )

            findings.append(finding)

        findings.sort(
            key=lambda item: (
                item.get(
                    "severity_rank",
                    0,
                ),
                item.get(
                    "primary_score",
                    0.0,
                ),
            ),
            reverse=True,
        )

        target_distribution = (
            cls._target_distribution_from_data(
                dataframe_a,
                dataframe_b,
                ml_run_a,
                ml_run_b,
            )
        )

        return {
            "available": True,
            "status": (
                "significant_drift_detected"
                if any(
                    item["status"] == "high"
                    for item in findings
                )
                else "drift_detected"
                if any(
                    item["status"] == "medium"
                    for item in findings
                )
                else "low_or_no_significant_drift"
            ),
            "method_summary": {
                "numeric": [
                    "Kolmogorov-Smirnov",
                    "Population Stability Index",
                    "Jensen-Shannon divergence",
                ],
                "categorical": [
                    "Chi-square",
                    "Jensen-Shannon divergence",
                ],
            },
            "features": findings,
            "target_distribution": target_distribution,
            "prediction_distribution": cls._prediction_distribution(
                ml_run_a,
                ml_run_b,
            ),
        }

    @classmethod
    def _numeric_drift(
        cls,
        column: str,
        baseline: pd.Series,
        target: pd.Series,
    ) -> dict[str, Any]:
        a = pd.to_numeric(
            baseline,
            errors="coerce",
        ).dropna().to_numpy(
            dtype=float
        )

        b = pd.to_numeric(
            target,
            errors="coerce",
        ).dropna().to_numpy(
            dtype=float
        )

        if len(a) == 0 or len(b) == 0:
            return {
                "feature": column,
                "type": "numeric",
                "available": False,
                "status": "unknown",
                "reason": "One side contains no usable numeric observations.",
                "severity_rank": 0,
                "primary_score": 0.0,
            }

        ks = cls._ks_statistic(
            a,
            b,
        )

        psi = cls._psi(
            a,
            b,
        )

        js = cls._js_divergence(
            a,
            b,
        )

        status = cls._numeric_status(
            ks,
            psi,
            js,
        )

        return {
            "feature": column,
            "type": "numeric",
            "available": True,
            "status": status,
            "severity_rank": cls._severity_rank(
                status
            ),
            "ks": round(ks, 6),
            "psi": round(psi, 6),
            "jensen_shannon": round(
                js,
                6,
            ),
            "primary_score": round(
                max(
                    ks,
                    psi,
                    js,
                ),
                6,
            ),
            "baseline_mean": cls._number(
                np.mean(a)
            ),
            "target_mean": cls._number(
                np.mean(b)
            ),
            "baseline_missing_percent": cls._missing_percent(
                baseline
            ),
            "target_missing_percent": cls._missing_percent(
                target
            ),
        }

    @classmethod
    def _categorical_drift(
        cls,
        column: str,
        baseline: pd.Series,
        target: pd.Series,
    ) -> dict[str, Any]:
        a = (
            baseline
            .astype("string")
            .fillna("<MISSING>")
        )

        b = (
            target
            .astype("string")
            .fillna("<MISSING>")
        )

        chi2 = cls._chi_square(
            a,
            b,
        )

        js = cls._js_categorical(
            a,
            b,
        )

        new_categories = sorted(
            set(
                b.unique()
            )
            - set(
                a.unique()
            )
        )[
            :50
        ]

        removed_categories = sorted(
            set(
                a.unique()
            )
            - set(
                b.unique()
            )
        )[
            :50
        ]

        status = cls._categorical_status(
            chi2,
            js,
            new_categories,
            removed_categories,
        )

        return {
            "feature": column,
            "type": "categorical",
            "available": True,
            "status": status,
            "severity_rank": cls._severity_rank(
                status
            ),
            "chi_square": round(
                chi2,
                6,
            ),
            "jensen_shannon": round(
                js,
                6,
            ),
            "primary_score": round(
                max(
                    chi2,
                    js,
                ),
                6,
            ),
            "new_categories": new_categories,
            "removed_categories": removed_categories,
            "baseline_missing_percent": cls._missing_percent(
                baseline
            ),
            "target_missing_percent": cls._missing_percent(
                target
            ),
        }

    # ------------------------------------------------------------------
    # Statistical methods
    # ------------------------------------------------------------------

    @staticmethod
    def _ks_statistic(
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:
        a = np.sort(
            np.asarray(
                a,
                dtype=float,
            )
        )

        b = np.sort(
            np.asarray(
                b,
                dtype=float,
            )
        )

        values = np.sort(
            np.unique(
                np.concatenate(
                    [a, b]
                )
            )
        )

        if values.size == 0:
            return 0.0

        cdf_a = (
            np.searchsorted(
                a,
                values,
                side="right",
            )
            / len(a)
        )

        cdf_b = (
            np.searchsorted(
                b,
                values,
                side="right",
            )
            / len(b)
        )

        return float(
            np.max(
                np.abs(
                    cdf_a - cdf_b
                )
            )
        )

    @staticmethod
    def _psi(
        baseline: np.ndarray,
        target: np.ndarray,
        bins: int = 10,
    ) -> float:
        baseline = np.asarray(
            baseline,
            dtype=float,
        )

        target = np.asarray(
            target,
            dtype=float,
        )

        if (
            len(baseline) == 0
            or len(target) == 0
        ):
            return 0.0

        quantiles = np.linspace(
            0,
            1,
            bins + 1,
        )

        edges = np.unique(
            np.quantile(
                baseline,
                quantiles,
            )
        )

        if len(edges) <= 2:
            lower = float(
                min(
                    baseline.min(),
                    target.min(),
                )
            )
            upper = float(
                max(
                    baseline.max(),
                    target.max(),
                )
            )

            if lower == upper:
                return 0.0

            edges = np.linspace(
                lower,
                upper,
                bins + 1,
            )

        baseline_counts, _ = np.histogram(
            baseline,
            bins=edges,
        )

        target_counts, _ = np.histogram(
            target,
            bins=edges,
        )

        epsilon = 1e-8

        baseline_pct = (
            baseline_counts
            / max(
                baseline_counts.sum(),
                1,
            )
        )

        target_pct = (
            target_counts
            / max(
                target_counts.sum(),
                1,
            )
        )

        baseline_pct = np.clip(
            baseline_pct,
            epsilon,
            None,
        )

        target_pct = np.clip(
            target_pct,
            epsilon,
            None,
        )

        return float(
            np.sum(
                (
                    baseline_pct
                    - target_pct
                )
                * np.log(
                    baseline_pct
                    / target_pct
                )
            )
        )

    @staticmethod
    def _js_divergence(
        baseline: np.ndarray,
        target: np.ndarray,
        bins: int = 20,
    ) -> float:
        baseline = np.asarray(
            baseline,
            dtype=float,
        )

        target = np.asarray(
            target,
            dtype=float,
        )

        if (
            len(baseline) == 0
            or len(target) == 0
        ):
            return 0.0

        lower = float(
            min(
                baseline.min(),
                target.min(),
            )
        )

        upper = float(
            max(
                baseline.max(),
                target.max(),
            )
        )

        if lower == upper:
            return 0.0

        edges = np.linspace(
            lower,
            upper,
            bins + 1,
        )

        a_counts, _ = np.histogram(
            baseline,
            bins=edges,
        )

        b_counts, _ = np.histogram(
            target,
            bins=edges,
        )

        a = (
            a_counts
            / max(
                a_counts.sum(),
                1,
            )
        )

        b = (
            b_counts
            / max(
                b_counts.sum(),
                1,
            )
        )

        epsilon = 1e-12

        a = np.clip(
            a,
            epsilon,
            None,
        )

        b = np.clip(
            b,
            epsilon,
            None,
        )

        a = a / a.sum()
        b = b / b.sum()

        m = 0.5 * (
            a + b
        )

        return float(
            0.5
            * (
                np.sum(
                    a * np.log(a / m)
                )
                + np.sum(
                    b * np.log(b / m)
                )
            )
        )

    @staticmethod
    def _chi_square(
        baseline: pd.Series,
        target: pd.Series,
    ) -> float:
        keys = sorted(
            set(
                baseline.tolist()
            )
            | set(
                target.tolist()
            )
        )

        if not keys:
            return 0.0

        observed_a = np.array(
            [
                int(
                    (
                        baseline == key
                    ).sum()
                )
                for key in keys
            ],
            dtype=float,
        )

        observed_b = np.array(
            [
                int(
                    (
                        target == key
                    ).sum()
                )
                for key in keys
            ],
            dtype=float,
        )

        total_a = observed_a.sum()
        total_b = observed_b.sum()

        total = (
            total_a
            + total_b
        )

        if total == 0:
            return 0.0

        row_totals = np.array(
            [
                total_a,
                total_b,
            ],
            dtype=float,
        )

        col_totals = (
            observed_a
            + observed_b
        )

        expected_a = (
            row_totals[0]
            * col_totals
            / total
        )

        expected_b = (
            row_totals[1]
            * col_totals
            / total
        )

        statistic = 0.0

        for observed, expected in zip(
            observed_a,
            expected_a,
        ):
            if expected > 0:
                statistic += (
                    (
                        observed
                        - expected
                    )
                    ** 2
                    / expected
                )

        for observed, expected in zip(
            observed_b,
            expected_b,
        ):
            if expected > 0:
                statistic += (
                    (
                        observed
                        - expected
                    )
                    ** 2
                    / expected
                )

        return float(
            statistic
        )

    @staticmethod
    def _js_categorical(
        baseline: pd.Series,
        target: pd.Series,
    ) -> float:
        keys = sorted(
            set(
                baseline.tolist()
            )
            | set(
                target.tolist()
            )
        )

        if not keys:
            return 0.0

        a_counts = np.array(
            [
                int(
                    (
                        baseline == key
                    ).sum()
                )
                for key in keys
            ],
            dtype=float,
        )

        b_counts = np.array(
            [
                int(
                    (
                        target == key
                    ).sum()
                )
                for key in keys
            ],
            dtype=float,
        )

        a = (
            a_counts
            / max(
                a_counts.sum(),
                1,
            )
        )

        b = (
            b_counts
            / max(
                b_counts.sum(),
                1,
            )
        )

        epsilon = 1e-12

        a = np.clip(
            a,
            epsilon,
            None,
        )

        b = np.clip(
            b,
            epsilon,
            None,
        )

        a = a / a.sum()
        b = b / b.sum()

        m = 0.5 * (
            a + b
        )

        return float(
            0.5
            * (
                np.sum(
                    a
                    * np.log(a / m)
                )
                + np.sum(
                    b
                    * np.log(b / m)
                )
            )
        )

    # ------------------------------------------------------------------
    # Drift status
    # ------------------------------------------------------------------

    @staticmethod
    def _numeric_status(
        ks: float,
        psi: float,
        js: float,
    ) -> str:
        if (
            ks >= 0.20
            or psi >= 0.25
            or js >= 0.20
        ):
            return "high"

        if (
            ks >= 0.10
            or psi >= 0.10
            or js >= 0.10
        ):
            return "medium"

        return "low"

    @staticmethod
    def _categorical_status(
        chi2: float,
        js: float,
        new_categories: list[str],
        removed_categories: list[str],
    ) -> str:
        category_change = (
            len(new_categories)
            + len(removed_categories)
        )

        if (
            js >= 0.20
            or chi2 >= 20
            or category_change >= 5
        ):
            return "high"

        if (
            js >= 0.10
            or chi2 >= 5
            or category_change > 0
        ):
            return "medium"

        return "low"

    @staticmethod
    def _severity_rank(
        status: str,
    ) -> int:
        return {
            "unknown": 0,
            "low": 1,
            "medium": 2,
            "high": 3,
        }.get(
            status,
            0,
        )

    # ------------------------------------------------------------------
    # Target / predictions
    # ------------------------------------------------------------------

    @classmethod
    def _target_distribution_from_data(
        cls,
        dataframe_a: pd.DataFrame,
        dataframe_b: pd.DataFrame,
        ml_run_a: MLRun | None,
        ml_run_b: MLRun | None,
    ) -> dict[str, Any] | None:
        target_name = cls._find_target_column(
            ml_run_a,
            ml_run_b,
        )

        if (
            not target_name
            or target_name
            not in dataframe_a.columns
            or target_name
            not in dataframe_b.columns
        ):
            return None

        a = (
            dataframe_a[target_name]
            .astype("string")
            .fillna("<MISSING>")
            .value_counts(
                normalize=True
            )
        )

        b = (
            dataframe_b[target_name]
            .astype("string")
            .fillna("<MISSING>")
            .value_counts(
                normalize=True
            )
        )

        categories = sorted(
            set(a.index)
            | set(b.index)
        )

        changes = []

        for category in categories:
            baseline = float(
                a.get(
                    category,
                    0.0,
                )
            )
            target = float(
                b.get(
                    category,
                    0.0,
                )
            )

            changes.append(
                {
                    "category": category,
                    "baseline_percent": round(
                        baseline * 100,
                        4,
                    ),
                    "target_percent": round(
                        target * 100,
                        4,
                    ),
                    "delta_percentage_points": round(
                        (
                            target
                            - baseline
                        )
                        * 100,
                        4,
                    ),
                }
            )

        return {
            "available": True,
            "target_column": target_name,
            "changes": changes,
        }

    @classmethod
    def _find_target_column(
        cls,
        ml_run_a: MLRun | None,
        ml_run_b: MLRun | None,
    ) -> str | None:
        for ml_run in (
            ml_run_a,
            ml_run_b,
        ):
            if ml_run is None:
                continue

            for payload_name in (
                "evaluation",
                "parameters",
            ):
                payload = getattr(
                    ml_run,
                    payload_name,
                    None,
                )

                if not isinstance(
                    payload,
                    dict,
                ):
                    continue

                for key in (
                    "target_column",
                    "target",
                    "label_column",
                    "label",
                    "y_column",
                ):
                    value = payload.get(
                        key
                    )

                    if (
                        isinstance(
                            value,
                            str,
                        )
                        and value.strip()
                    ):
                        return value.strip()

        return None

    @classmethod
    def _prediction_distribution(
        cls,
        ml_run_a: MLRun | None,
        ml_run_b: MLRun | None,
    ) -> dict[str, Any]:
        baseline = cls._extract_prediction_distribution(
            ml_run_a
        )

        target = cls._extract_prediction_distribution(
            ml_run_b
        )

        return {
            "available": (
                baseline is not None
                or target is not None
            ),
            "baseline": baseline,
            "target": target,
        }

    @classmethod
    def _extract_prediction_distribution(
        cls,
        ml_run: MLRun | None,
    ) -> dict[str, float] | None:
        if ml_run is None:
            return None

        evaluation = getattr(
            ml_run,
            "evaluation",
            None,
        )

        metrics = getattr(
            ml_run,
            "metrics",
            None,
        )

        candidates = [
            evaluation,
            metrics,
        ]

        for payload in candidates:
            if not isinstance(
                payload,
                dict,
            ):
                continue

            value = payload.get(
                "prediction_distribution"
            )

            if isinstance(
                value,
                dict,
            ):
                output: dict[
                    str,
                    float,
                ] = {}

                for key, item in value.items():
                    number = cls._safe_float(
                        item
                    )

                    if number is not None:
                        output[
                            str(key)
                        ] = number

                if output:
                    return output

        return None

    # ------------------------------------------------------------------
    # Model / metrics
    # ------------------------------------------------------------------

    @classmethod
    def _compare_model_evidence(
        cls,
        ml_run_a: MLRun | None,
        ml_run_b: MLRun | None,
    ) -> dict[str, Any]:
        if (
            ml_run_a is None
            and ml_run_b is None
        ):
            return {
                "available": False,
                "status": "unavailable",
                "baseline_ml_run_id": None,
                "target_ml_run_id": None,
                "model_name": None,
                "feature_changes": [],
                "parameter_changes": [],
            }

        name_a = (
            getattr(
                ml_run_a,
                "model_name",
                None,
            )
            if ml_run_a
            else None
        )

        name_b = (
            getattr(
                ml_run_b,
                "model_name",
                None,
            )
            if ml_run_b
            else None
        )

        feature_changes = cls._compare_json_set(
            getattr(
                ml_run_a,
                "features",
                None,
            )
            if ml_run_a
            else None,
            getattr(
                ml_run_b,
                "features",
                None,
            )
            if ml_run_b
            else None,
        )

        parameter_changes = cls._compare_json_dict(
            getattr(
                ml_run_a,
                "parameters",
                None,
            )
            if ml_run_a
            else None,
            getattr(
                ml_run_b,
                "parameters",
                None,
            )
            if ml_run_b
            else None,
        )

        return {
            "available": True,
            "status": (
                "both_present"
                if (
                    ml_run_a is not None
                    and ml_run_b is not None
                )
                else "partial"
            ),
            "baseline_ml_run_id": (
                ml_run_a.id
                if ml_run_a
                else None
            ),
            "target_ml_run_id": (
                ml_run_b.id
                if ml_run_b
                else None
            ),
            "model_name": {
                "baseline": name_a,
                "target": name_b,
                "changed": (
                    name_a != name_b
                    and (
                        name_a is not None
                        or name_b is not None
                    )
                ),
            },
            "feature_changes": feature_changes,
            "parameter_changes": parameter_changes,
        }

    @classmethod
    def _compare_performance(
        cls,
        ml_run_a: MLRun | None,
        ml_run_b: MLRun | None,
    ) -> dict[str, Any]:
        baseline_metrics = cls._merged_metrics(
            ml_run_a
        )

        target_metrics = cls._merged_metrics(
            ml_run_b
        )

        common = sorted(
            set(
                baseline_metrics
            )
            & set(
                target_metrics
            )
        )

        rows = []

        aliases = {
            "roc_auc": [
                "roc_auc",
                "roc-auc",
                "auc",
                "rocauc",
            ],
            "accuracy": [
                "accuracy",
                "acc",
            ],
            "precision": [
                "precision",
            ],
            "recall": [
                "recall",
                "sensitivity",
                "true_positive_rate",
            ],
            "f1": [
                "f1",
                "f1_score",
            ],
            "false_positives": [
                "false_positives",
                "fp",
            ],
            "false_negatives": [
                "false_negatives",
                "fn",
            ],
        }

        normalized: dict[
            str,
            tuple[str, float, float],
        ] = {}

        for display_name, keys in aliases.items():
            a_value = None
            b_value = None

            for key in keys:
                if key in baseline_metrics:
                    a_value = cls._safe_float(
                        baseline_metrics[key]
                    )
                    break

            for key in keys:
                if key in target_metrics:
                    b_value = cls._safe_float(
                        target_metrics[key]
                    )
                    break

            if (
                a_value is not None
                and b_value is not None
            ):
                normalized[
                    display_name
                ] = (
                    display_name,
                    a_value,
                    b_value,
                )

        for (
            display_name,
            (
                _,
                baseline,
                target,
            ),
        ) in normalized.items():
            rows.append(
                {
                    "metric": display_name,
                    "baseline": baseline,
                    "target": target,
                    "delta": (
                        target - baseline
                    ),
                    "delta_percentage_points": (
                        target - baseline
                    )
                    * 100
                    if 0
                    <= baseline
                    <= 1
                    and 0
                    <= target
                    <= 1
                    else None,
                }
            )

        degraded = [
            row
            for row in rows
            if row["delta"] < 0
            and row["metric"]
            in {
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
            }
        ]

        improved = [
            row
            for row in rows
            if row["delta"] > 0
            and row["metric"]
            in {
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
            }
        ]

        return {
            "available": bool(
                rows
                or baseline_metrics
                or target_metrics
            ),
            "status": (
                "degraded"
                if degraded
                else "improved"
                if improved
                else "no_common_performance_metrics"
            ),
            "baseline_metrics": cls._json_safe(
                baseline_metrics
            ),
            "target_metrics": cls._json_safe(
                target_metrics
            ),
            "rows": rows,
            "training_vs_production": cls._compare_training_production(
                ml_run_a,
                ml_run_b,
            ),
        }

    @classmethod
    def _merged_metrics(
        cls,
        ml_run: MLRun | None,
    ) -> dict[str, Any]:
        if ml_run is None:
            return {}

        result: dict[
            str,
            Any,
        ] = {}

        metrics = getattr(
            ml_run,
            "metrics",
            None,
        )

        evaluation = getattr(
            ml_run,
            "evaluation",
            None,
        )

        if isinstance(
            metrics,
            dict,
        ):
            result.update(
                cls._flatten_numeric_dict(
                    metrics
                )
            )

        if isinstance(
            evaluation,
            dict,
        ):
            for key, value in cls._flatten_numeric_dict(
                evaluation
            ).items():
                result.setdefault(
                    key,
                    value,
                )

        return result

    @classmethod
    def _compare_training_production(
        cls,
        ml_run_a: MLRun | None,
        ml_run_b: MLRun | None,
    ) -> dict[str, Any]:
        baseline = cls._extract_named_metric_group(
            ml_run_a,
            "training",
        )

        production_a = cls._extract_named_metric_group(
            ml_run_a,
            "production",
        )

        production_b = cls._extract_named_metric_group(
            ml_run_b,
            "production",
        )

        if not (
            baseline
            or production_a
            or production_b
        ):
            return {
                "available": False,
                "baseline_training": None,
                "baseline_production": None,
                "target_production": None,
                "rows": [],
            }

        target_production = (
            production_b
            or production_a
        )

        common = sorted(
            set(baseline)
            & set(target_production)
        )

        rows = []

        for key in common:
            a = cls._safe_float(
                baseline[key]
            )
            b = cls._safe_float(
                target_production[key]
            )

            if (
                a is None
                or b is None
            ):
                continue

            rows.append(
                {
                    "metric": key,
                    "training": a,
                    "production": b,
                    "delta": b - a,
                }
            )

        return {
            "available": bool(rows),
            "baseline_training": cls._json_safe(
                baseline
            ),
            "baseline_production": cls._json_safe(
                production_a
            ),
            "target_production": cls._json_safe(
                production_b
            ),
            "rows": rows,
        }

    @classmethod
    def _extract_named_metric_group(
        cls,
        ml_run: MLRun | None,
        group_name: str,
    ) -> dict[str, Any]:
        if ml_run is None:
            return {}

        for container_name in (
            "evaluation",
            "metrics",
        ):
            payload = getattr(
                ml_run,
                container_name,
                None,
            )

            if not isinstance(
                payload,
                dict,
            ):
                continue

            group = payload.get(
                group_name
            )

            if isinstance(
                group,
                dict,
            ):
                return cls._flatten_numeric_dict(
                    group
                )

        return {}

    # ------------------------------------------------------------------
    # Git / code analysis
    # ------------------------------------------------------------------

    @classmethod
    def _compare_git_evidence(
        cls,
        project_path: str | None,
        version_a: Version,
        version_b: Version,
    ) -> dict[str, Any]:
        base = {
            "available": False,
            "status": "unavailable",
            "baseline_commit": version_a.git_commit,
            "target_commit": version_b.git_commit,
            "changed_files": [],
            "model_candidate_files": [],
            "model_signals": [],
        }

        if not project_path:
            return base

        if Repo is None:
            base["reason"] = (
                "GitPython is not available."
            )
            return base

        try:
            repo = Repo(
                project_path
            )

            commit_a = repo.commit(
                version_a.git_commit
            )

            commit_b = repo.commit(
                version_b.git_commit
            )

            diffs = commit_a.diff(
                commit_b,
                create_patch=False,
            )

            changed_files = []

            for diff in diffs:
                path = (
                    diff.b_path
                    or diff.a_path
                )

                if not path:
                    continue

                changed_files.append(
                    {
                        "path": path,
                        "change_type": (
                            diff.change_type
                            or "M"
                        ),
                    }
                )

            model_candidate_files = []

            for item in changed_files:
                if cls._is_model_candidate(
                    item["path"]
                ):
                    model_candidate_files.append(
                        item
                    )

            model_signals = cls._extract_model_signals(
                repo=repo,
                version_a=version_a,
                version_b=version_b,
                candidate_files=model_candidate_files,
            )

            return {
                "available": True,
                "status": (
                    "changed"
                    if changed_files
                    else "unchanged"
                ),
                "baseline_commit": version_a.git_commit,
                "target_commit": version_b.git_commit,
                "commit_changed": (
                    version_a.git_commit
                    != version_b.git_commit
                ),
                "changed_files": changed_files[
                    :200
                ],
                "model_candidate_files": model_candidate_files[
                    : cls.MODEL_FILE_LIMIT
                ],
                "model_signals": model_signals,
            }

        except Exception as exc:
            base["reason"] = str(
                exc
            )
            return base

    @staticmethod
    def _is_model_candidate(
        path: str,
    ) -> bool:
        lowered = path.lower()

        keywords = (
            "model",
            "train",
            "training",
            "predict",
            "inference",
            "pipeline",
            "classifier",
            "regressor",
            "estimator",
            "xgboost",
            "lightgbm",
            "catboost",
            "torch",
            "keras",
            "tensorflow",
            "sklearn",
        )

        if not lowered.endswith(
            (
                ".py",
                ".ipynb",
                ".yaml",
                ".yml",
                ".json",
                ".toml",
                ".ini",
            )
        ):
            return False

        return any(
            keyword in lowered
            for keyword in keywords
        )

    @classmethod
    def _extract_model_signals(
        cls,
        repo: Any,
        version_a: Version,
        version_b: Version,
        candidate_files: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        signals = []

        for item in candidate_files[
            : cls.MODEL_FILE_LIMIT
        ]:
            path = item["path"]

            text_a = cls._git_file_text(
                repo,
                version_a.git_commit,
                path,
            )

            text_b = cls._git_file_text(
                repo,
                version_b.git_commit,
                path,
            )

            combined = (
                text_a
                or ""
            ) + (
                "\n"
            ) + (
                text_b
                or ""
            )

            matched = []

            for token in (
                "RandomForest",
                "XGBClassifier",
                "XGBRegressor",
                "LGBMClassifier",
                "CatBoost",
                "LogisticRegression",
                "LinearRegression",
                "RandomForestClassifier",
                "RandomForestRegressor",
                "GradientBoosting",
                "HistGradientBoosting",
                "SVC",
                "SVR",
                "KNeighbors",
                "DecisionTree",
                "torch",
                "tensorflow",
                "keras",
                "sklearn",
            ):
                if token.lower() in combined.lower():
                    matched.append(
                        token
                    )

            digest_a = cls._sha256_text(
                text_a or ""
            )

            digest_b = cls._sha256_text(
                text_b or ""
            )

            signals.append(
                {
                    "path": path,
                    "baseline_content_hash": digest_a,
                    "target_content_hash": digest_b,
                    "changed": digest_a != digest_b,
                    "model_framework_signals": sorted(
                        set(matched)
                    ),
                    "baseline_code_excerpt": cls._safe_code_excerpt(
                        text_a
                    ),
                    "target_code_excerpt": cls._safe_code_excerpt(
                        text_b
                    ),
                }
            )

        return signals

    @classmethod
    def _git_file_text(
        cls,
        repo: Any,
        commit: str,
        path: str,
    ) -> str | None:
        try:
            blob = repo.commit(
                commit
            ).tree / path

            data = blob.data_stream.read(
                cls.MODEL_CODE_CHAR_LIMIT
            )

            return data.decode(
                "utf-8",
                errors="replace",
            )
        except Exception:
            return None

    @classmethod
    def _safe_code_excerpt(
        cls,
        value: str | None,
    ) -> str | None:
        if not value:
            return None

        cleaned = re.sub(
            r"(?i)(api[_-]?key|secret|token|password)\\s*=\\s*['\"][^'\"]+['\"]",
            r"\1='[REDACTED]'",
            value,
        )

        cleaned = re.sub(
            r"(?i)Bearer\\s+[A-Za-z0-9._-]+",
            "Bearer [REDACTED]",
            cleaned,
        )

        return cleaned[
            : cls.MODEL_CODE_CHAR_LIMIT
        ]

    # ------------------------------------------------------------------
    # Preparation / provenance / validity
    # ------------------------------------------------------------------

    @classmethod
    def _compare_preparation(
        cls,
        version_a: Version,
        version_b: Version,
        ml_run_a: MLRun | None,
        ml_run_b: MLRun | None,
    ) -> dict[str, Any]:
        candidates = []

        for version_label, version, ml_run in (
            (
                "baseline",
                version_a,
                ml_run_a,
            ),
            (
                "target",
                version_b,
                ml_run_b,
            ),
        ):
            for source_name, payload in (
                (
                    "version",
                    getattr(
                        version,
                        "dvc_state",
                        None,
                    ),
                ),
                (
                    "ml_run.parameters",
                    getattr(
                        ml_run,
                        "parameters",
                        None,
                    )
                    if ml_run
                    else None,
                ),
            ):
                if not isinstance(
                    payload,
                    dict,
                ):
                    continue

                for key in (
                    "preparation",
                    "preprocessing",
                    "preprocessing_config",
                    "operations",
                    "steps",
                ):
                    value = payload.get(
                        key
                    )

                    if value is not None:
                        candidates.append(
                            {
                                "version": version_label,
                                "source": source_name,
                                "key": key,
                                "value": cls._json_safe(
                                    value
                                ),
                            }
                        )

        if not candidates:
            return {
                "available": False,
                "status": "unavailable",
                "changes": [],
                "message": (
                    "Preparation history is not persisted in "
                    "the selected comparison evidence."
                ),
            }

        return {
            "available": True,
            "status": "recorded",
            "changes": candidates,
        }

    @classmethod
    def _build_provenance(
        cls,
        version_a: Version,
        version_b: Version,
    ) -> dict[str, Any]:
        return {
            "git": {
                "baseline": version_a.git_commit,
                "target": version_b.git_commit,
                "changed": (
                    version_a.git_commit
                    != version_b.git_commit
                ),
            },
            "dvc": {
                "baseline": cls._json_safe(
                    getattr(
                        version_a,
                        "dvc_state",
                        None,
                    )
                ),
                "target": cls._json_safe(
                    getattr(
                        version_b,
                        "dvc_state",
                        None,
                    )
                ),
            },
            "created_at": {
                "baseline": cls._json_safe(
                    version_a.created_at
                ),
                "target": cls._json_safe(
                    version_b.created_at
                ),
            },
            "description": {
                "baseline": version_a.description,
                "target": version_b.description,
            },
        }

    @classmethod
    def _build_validity(
        cls,
        version_a: Version,
        version_b: Version,
        dataset_a: dict[str, Any],
        dataset_b: dict[str, Any],
        ml_run_a: MLRun | None,
        ml_run_b: MLRun | None,
    ) -> dict[str, Any]:
        controlled = []
        uncontrolled = []
        limitations = []

        if version_a.project_id == version_b.project_id:
            controlled.append(
                "Both versions belong to the same project."
            )
        else:
            uncontrolled.append(
                "Versions belong to different projects."
            )

        profile_a = dataset_a.get(
            "profile"
        )
        profile_b = dataset_b.get(
            "profile"
        )

        if profile_a and profile_b:
            if (
                profile_a.get(
                    "columns"
                )
                == profile_b.get(
                    "columns"
                )
                and cls._schema_names(
                    profile_a
                )
                == cls._schema_names(
                    profile_b
                )
            ):
                controlled.append(
                    "Dataset schema is structurally aligned."
                )
            else:
                limitations.append(
                    "Dataset schema changed between versions."
                )

        if (
            ml_run_a is not None
            and ml_run_b is not None
        ):
            controlled.append(
                "Both versions have recorded MLRun evidence."
            )

        elif (
            ml_run_a is not None
            or ml_run_b is not None
        ):
            limitations.append(
                "MLRun evidence exists for only one side."
            )

        if not dataset_a.get(
            "available"
        ) or not dataset_b.get(
            "available"
        ):
            limitations.append(
                "Historical dataset content was not available "
                "for one or both versions."
            )

        if uncontrolled:
            status = "invalid"
            confidence = "limited"
        elif limitations:
            status = "partial"
            confidence = "medium"
        else:
            status = "controlled"
            confidence = "high"

        return {
            "status": status,
            "confidence": confidence,
            "controlled_factors": controlled,
            "limitations": limitations,
            "uncontrolled_factors": uncontrolled,
        }

    # ------------------------------------------------------------------
    # Summary / recommendations
    # ------------------------------------------------------------------

    @classmethod
    def _build_summary(
        cls,
        version_a: Version,
        version_b: Version,
        dataset_a: dict[str, Any],
        dataset_b: dict[str, Any],
        schema_diff: dict[str, Any],
        quality: dict[str, Any],
        drift: dict[str, Any],
        model: dict[str, Any],
        performance: dict[str, Any],
        code: dict[str, Any],
        validity: dict[str, Any],
    ) -> dict[str, Any]:
        performance_status = performance.get(
            "status"
        )

        if performance_status == "degraded":
            status = "PERFORMANCE REGRESSION DETECTED"

        elif performance_status == "improved":
            status = "PERFORMANCE IMPROVEMENT DETECTED"

        elif (
            drift.get("status")
            == "significant_drift_detected"
        ):
            status = "SIGNIFICANT DATA DRIFT DETECTED"

        elif (
            schema_diff.get("status")
            == "changed"
        ):
            status = "DATASET / SCHEMA CHANGED"

        elif code.get("commit_changed"):
            status = "PROJECT STATE CHANGED"

        else:
            status = "COMPARISON AVAILABLE"

        evidence_points = 0

        if dataset_a.get("available") and dataset_b.get(
            "available"
        ):
            evidence_points += 1

        if schema_diff.get(
            "available"
        ):
            evidence_points += 1

        if quality.get(
            "available"
        ):
            evidence_points += 1

        if drift.get(
            "available"
        ):
            evidence_points += 1

        if model.get(
            "available"
        ):
            evidence_points += 1

        if performance.get(
            "available"
        ):
            evidence_points += 1

        if code.get(
            "available"
        ):
            evidence_points += 1

        if evidence_points >= 5:
            evidence_level = "RICH"

        elif evidence_points >= 3:
            evidence_level = "PARTIAL"

        else:
            evidence_level = "LIMITED"

        return {
            "status": status,
            "evidence_level": evidence_level,
            "baseline_version": version_a.version_number,
            "target_version": version_b.version_number,
            "direction": (
                f"V{version_a.version_number}"
                f" → "
                f"V{version_b.version_number}"
            ),
            "evidence_points": evidence_points,
            "dataset_changed": (
                dataset_a.get("selected_file")
                != dataset_b.get("selected_file")
                or (
                    dataset_a.get("profile")
                    and dataset_b.get("profile")
                    and (
                        dataset_a["profile"].get(
                            "rows"
                        )
                        != dataset_b["profile"].get(
                            "rows"
                        )
                    )
                )
            ),
            "schema_changed": (
                schema_diff.get(
                    "status"
                )
                == "changed"
            ),
            "quality_changed": (
                quality.get(
                    "status"
                )
                == "changed"
            ),
            "drift_status": drift.get(
                "status"
            ),
            "performance_status": performance_status,
            "model_status": model.get(
                "status"
            ),
            "validity": validity.get(
                "status"
            ),
        }

    @classmethod
    def _collect_limitations(
        cls,
        dataset_a: dict[str, Any],
        dataset_b: dict[str, Any],
        ml_run_a: MLRun | None,
        ml_run_b: MLRun | None,
        validity: dict[str, Any],
        drift: dict[str, Any],
        performance: dict[str, Any],
    ) -> list[str]:
        limitations = []

        limitations.extend(
            dataset_a.get(
                "limitations",
                [],
            )
        )

        limitations.extend(
            dataset_b.get(
                "limitations",
                [],
            )
        )

        limitations.extend(
            validity.get(
                "limitations",
                [],
            )
        )

        if (
            ml_run_a is None
            or ml_run_b is None
        ):
            limitations.append(
                "Model performance cannot be fully compared "
                "when MLRun evidence is missing from one side."
            )

        if not drift.get(
            "available"
        ):
            limitations.append(
                "Feature drift could not be calculated because "
                "both historical datasets were not available."
            )

        if not performance.get(
            "rows"
        ):
            limitations.append(
                "No comparable scalar model-performance metrics "
                "were recorded on both sides."
            )

        # Preserve order but remove duplicates.
        return list(
            dict.fromkeys(
                str(item)
                for item in limitations
                if item
            )
        )

    @classmethod
    def _build_recommendations(
        cls,
        drift: dict[str, Any],
        performance: dict[str, Any],
        schema_diff: dict[str, Any],
        quality: dict[str, Any],
        model: dict[str, Any],
        validity: dict[str, Any],
    ) -> list[str]:
        recommendations = []

        high_drift = [
            feature
            for feature in (
                drift.get(
                    "features"
                )
                or []
            )
            if feature.get(
                "status"
            )
            == "high"
        ]

        if high_drift:
            names = ", ".join(
                feature["feature"]
                for feature in high_drift[:5]
            )

            recommendations.append(
                "Review high-drift features before treating the "
                f"target version as a stable production state: {names}."
            )

        if performance.get(
            "status"
        ) == "degraded":
            recommendations.append(
                "Investigate the performance regression against the "
                "recorded dataset drift, schema, quality and prediction "
                "evidence before deciding whether to retrain."
            )

        if performance.get(
            "status"
        ) == "degraded" and high_drift:
            recommendations.append(
                "A retraining review is warranted if the observed drift "
                "represents the intended production population and the "
                "performance regression persists on a controlled evaluation set."
            )

        if schema_diff.get(
            "counts",
            {},
        ).get(
            "added",
            0,
        ) > 0 or schema_diff.get(
            "counts",
            {},
        ).get(
            "removed",
            0,
        ) > 0:
            recommendations.append(
                "Validate downstream feature contracts because the "
                "dataset schema changed between the selected versions."
            )

        if quality.get(
            "metrics"
        ):
            recommendations.append(
                "Review changed missing-value and duplicate rates "
                "before interpreting model metrics in isolation."
            )

        if model.get(
            "status"
        ) == "partial":
            recommendations.append(
                "Associate a recorded MLRun with the target version "
                "to enable a full model-performance comparison."
            )

        if validity.get(
            "status"
        ) == "partial":
            recommendations.append(
                "Treat the comparison as partial evidence and verify "
                "the missing evaluation or dataset context."
            )

        if not recommendations:
            recommendations.append(
                "Review the evidence sections and confirm the target "
                "version is appropriate for downstream use."
            )

        return recommendations[:8]

    # ------------------------------------------------------------------
    # AI
    # ------------------------------------------------------------------

    @classmethod
    def _generate_ai_report(
        cls,
        project: Project,
        version_a: Version,
        version_b: Version,
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        if Groq is None:
            return {
                "status": "unavailable",
                "report": None,
                "provider": None,
                "reason": (
                    "Groq SDK is not installed."
                ),
            }

        api_key = os.getenv(
            "GROQ_API_KEY"
        )

        if not api_key:
            return {
                "status": "unavailable",
                "report": None,
                "provider": "groq",
                "reason": (
                    "GROQ_API_KEY is not configured."
                ),
            }

        model = os.getenv(
            "GROQ_COMPARE_MODEL",
            os.getenv(
                "GROQ_MODEL",
                "openai/gpt-oss-safeguard-20b",
            ),
        )

        client = Groq(
            api_key=api_key
        )

        evidence_for_ai = cls._json_safe(
            evidence
        )

        # Keep the prompt bounded even when a Git diff contains many files.
        prompt = cls._compact_ai_payload(
            evidence_for_ai
        )

        system_prompt = """
You are the DATAGIT Compare Investigation Agent.

Your job is to explain what changed between two versions of the SAME
machine-learning project using ONLY the deterministic evidence supplied.

You MUST:
- distinguish facts from interpretations and hypotheses;
- mention V1/V2 style version identifiers exactly as supplied;
- analyze dataset/DVC evidence;
- analyze Git/code/model evidence;
- analyze MLRun metrics and evaluation evidence when present;
- discuss data drift, concept drift, prediction drift, and performance
  degradation only when the supplied evidence supports them;
- use the supplied KS / PSI / Chi-square / Jensen-Shannon results;
- rank likely contributors to performance changes;
- explicitly identify missing evidence;
- explain comparison validity/confidence;
- give a retraining recommendation only when the evidence supports review;
- avoid claiming causality unless the evidence establishes it;
- never invent metric values, datasets, files, parameters, feature names,
  predictions, or drift results.

Return a detailed readable report with these headings:

EXECUTIVE SUMMARY
DATASET CHANGES
SCHEMA CHANGES
FEATURE DRIFT
TARGET / CONCEPT DRIFT
PREDICTION DRIFT
MODEL / CODE CHANGES
MODEL PERFORMANCE
ROOT-CAUSE INVESTIGATION
COMPARISON VALIDITY
RECOMMENDATIONS
LIMITATIONS

Use concise engineering language. Where evidence is missing, say
"Not recorded" or "Unavailable in the selected evidence."
"""

        user_prompt = (
            "Project: "
            f"{getattr(project, 'name', project.id)}\n"
            f"Baseline: V{version_a.version_number}\n"
            f"Target: V{version_b.version_number}\n\n"
            "DETERMINISTIC EVIDENCE:\n"
            + json.dumps(
                prompt,
                indent=2,
                ensure_ascii=False,
            )
        )

        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.1,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt.strip(),
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
            )

            content = (
                response.choices[0]
                .message
                .content
            )

            return {
                "status": "generated",
                "report": content,
                "provider": "groq",
                "model": model,
            }

        except Exception as exc:
            return {
                "status": "error",
                "report": None,
                "provider": "groq",
                "model": model,
                "reason": str(exc),
            }

    @classmethod
    def _compact_ai_payload(
        cls,
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        result = json.loads(
            json.dumps(
                evidence,
                default=str,
            )
        )

        try:
            code = result.get(
                "code",
                {}
            )

            signals = code.get(
                "model_signals",
                []
            )

            for signal in signals:
                signal.pop(
                    "baseline_code_excerpt",
                    None,
                )
                signal.pop(
                    "target_code_excerpt",
                    None,
                )

        except Exception:
            pass

        try:
            dataset = result.get(
                "dataset",
                {}
            )

            for side in (
                "baseline",
                "target",
            ):
                entry = dataset.get(
                    side
                )

                if isinstance(
                    entry,
                    dict,
                ):
                    entry.pop(
                        "dataframe",
                        None,
                    )

        except Exception:
            pass

        return result

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compare_json_set(
        baseline: Any,
        target: Any,
    ) -> dict[str, Any]:
        def normalize(
            value: Any,
        ) -> set[str]:
            if isinstance(
                value,
                list,
            ):
                return {
                    str(item)
                    for item in value
                }

            if isinstance(
                value,
                dict,
            ):
                return {
                    str(key)
                    for key in value
                }

            return set()

        a = normalize(
            baseline
        )

        b = normalize(
            target
        )

        return {
            "added": sorted(
                b - a
            ),
            "removed": sorted(
                a - b
            ),
            "unchanged": sorted(
                a & b
            ),
        }

    @staticmethod
    def _compare_json_dict(
        baseline: Any,
        target: Any,
    ) -> list[dict[str, Any]]:
        if not isinstance(
            baseline,
            dict,
        ):
            baseline = {}

        if not isinstance(
            target,
            dict,
        ):
            target = {}

        rows = []

        for key in sorted(
            set(baseline)
            | set(target)
        ):
            a = baseline.get(
                key
            )

            b = target.get(
                key
            )

            if a != b:
                rows.append(
                    {
                        "parameter": str(key),
                        "baseline": DeepCompareService._json_safe(
                            a
                        ),
                        "target": DeepCompareService._json_safe(
                            b
                        ),
                    }
                )

        return rows

    @staticmethod
    def _schema_names(
        profile: dict[str, Any],
    ) -> set[str]:
        return {
            item["name"]
            for item in (
                profile.get(
                    "columns_profile"
                )
                or []
            )
            if isinstance(
                item,
                dict,
            )
            and item.get(
                "name"
            )
        }

    @staticmethod
    def _missing_percent(
        series: pd.Series,
    ) -> float:
        return round(
            (
                series.isna().mean()
                * 100
            ),
            4,
        )

    @staticmethod
    def _safe_float(
        value: Any,
    ) -> float | None:
        try:
            if isinstance(
                value,
                bool,
            ):
                return None

            number = float(
                value
            )

            if not math.isfinite(
                number
            ):
                return None

            return number

        except Exception:
            return None

    @staticmethod
    def _number(
        value: Any,
    ) -> float | int | None:
        number = DeepCompareService._safe_float(
            value
        )

        if number is None:
            return None

        if float(
            number
        ).is_integer():
            return int(
                number
            )

        return round(
            number,
            8,
        )

    @staticmethod
    def _flatten_numeric_dict(
        payload: dict[str, Any],
        prefix: str = "",
    ) -> dict[str, float]:
        result: dict[
            str,
            float,
        ] = {}

        for key, value in payload.items():
            name = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            numeric = DeepCompareService._safe_float(
                value
            )

            if numeric is not None:
                result[
                    name.lower()
                ] = numeric
                continue

            if isinstance(
                value,
                dict,
            ):
                result.update(
                    DeepCompareService._flatten_numeric_dict(
                        value,
                        prefix=name,
                    )
                )

        return result

    @staticmethod
    def _json_safe(
        value: Any,
    ) -> Any:
        if value is None:
            return None

        if isinstance(
            value,
            dict,
        ):
            return {
                str(key): DeepCompareService._json_safe(
                    item
                )
                for key, item in value.items()
            }

        if isinstance(
            value,
            (list, tuple),
        ):
            return [
                DeepCompareService._json_safe(
                    item
                )
                for item in value
            ]

        if isinstance(
            value,
            np.ndarray,
        ):
            return (
                value.tolist()
            )

        if isinstance(
            value,
            np.integer,
        ):
            return int(
                value
            )

        if isinstance(
            value,
            np.floating,
        ):
            return float(
                value
            )

        if isinstance(
            value,
            (pd.Timestamp,),
        ):
            return value.isoformat()

        if hasattr(
            value,
            "isoformat",
        ):
            try:
                return value.isoformat()
            except Exception:
                pass

        try:
            json.dumps(
                value
            )
            return value
        except Exception:
            return str(
                value
            )

    @staticmethod
    def _without_dataframe(
        dataset_evidence: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            key: value
            for key, value in dataset_evidence.items()
            if key != "dataframe"
        }

    @staticmethod
    def _sha256_text(
        value: str,
    ) -> str:
        return hashlib.sha256(
            value.encode(
                "utf-8",
                errors="replace",
            )
        ).hexdigest()