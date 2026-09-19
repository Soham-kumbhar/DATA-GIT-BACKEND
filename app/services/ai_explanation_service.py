import copy
import json
from typing import Any

from groq import Groq
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import (
    Dataset,
    Model,
    Project,
    Version,
    VersionResultEvidence,
)
from app.services.version_comparison_service import (
    VersionComparisonService,
)


class AIExplanationService:
    MODEL = "openai/gpt-oss-120b"

    @staticmethod
    def _client() -> Groq:
        if not settings.groq_api_key:
            raise RuntimeError(
                "Groq API key is not configured."
            )

        return Groq(
            api_key=settings.groq_api_key
        )

    @staticmethod
    def _parse_json(
        content: str,
    ) -> dict[str, Any]:
        text = (content or "").strip()

        if not text:
            raise ValueError(
                "AI provider returned an empty response."
            )

        try:
            value = json.loads(text)

        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")

            if start < 0 or end <= start:
                raise ValueError(
                    "AI provider returned invalid JSON."
                )

            value = json.loads(
                text[start:end + 1]
            )

        if not isinstance(value, dict):
            raise ValueError(
                "AI provider returned a non-object JSON response."
            )

        return value

    @staticmethod
    def normalize_comparison(
        comparison: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Normalize the comparison for the AI layer.

        Legacy ML-run fields are removed from the AI evidence boundary.
        The AI receives version-centric evidence instead.
        """

        result = copy.deepcopy(
            comparison
        )

        for key in [
            "ml_run_before",
            "ml_run_after",
            "ml_comparison",
            "training",
            "run",
        ]:
            result.pop(key, None)

        # Do not allow legacy run-derived performance
        # to become AI evidence.
        if "performance" in result:
            result["legacy_performance"] = (
                result.pop("performance")
            )

        return result

    @staticmethod
    def _result_evidence(
        db: Session,
        project_id: int,
        version_id: int | None,
    ) -> dict[str, Any] | None:
        if version_id is None:
            return None

        evidence = (
            db.query(
                VersionResultEvidence
            )
            .join(
                Version,
                Version.id
                == VersionResultEvidence.version_id,
            )
            .filter(
                VersionResultEvidence.version_id
                == version_id,
                Version.project_id
                == project_id,
            )
            .first()
        )

        if evidence is None:
            return None

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
                if any(
                    value is not None
                    for value in [
                        evidence.model_name,
                        evidence.model_path,
                        evidence.model_sha256,
                        evidence.framework,
                        evidence.framework_version,
                    ]
                )
                else None
            ),
            "metrics": (
                evidence.metrics
                if isinstance(
                    evidence.metrics,
                    dict,
                )
                else {}
            ),
            "evaluation": (
                evidence.evaluation
                if isinstance(
                    evidence.evaluation,
                    dict,
                )
                else None
            ),
            "notes": evidence.notes,
        }

    @staticmethod
    def _metric_comparison(
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> dict[str, Any]:
        changes: dict[str, Any] = {}

        names = sorted(
            set(before.keys())
            | set(after.keys())
        )

        for name in names:
            old = before.get(name)
            new = after.get(name)

            if (
                isinstance(old, (int, float))
                and isinstance(new, (int, float))
            ):
                changes[name] = {
                    "before": old,
                    "after": new,
                    "delta": new - old,
                }

            elif old != new:
                changes[name] = {
                    "before": old,
                    "after": new,
                }

        return changes

    @staticmethod
    def build_evidence(
        db: Session,
        project_id: int,
        comparison: dict[str, Any],
    ) -> dict[str, Any]:
        normalized = (
            AIExplanationService
            .normalize_comparison(
                comparison
            )
        )

        version_1 = normalized.get(
            "version_1"
        )

        version_2 = normalized.get(
            "version_2"
        )

        before = (
            AIExplanationService
            ._result_evidence(
                db,
                project_id,
                version_1,
            )
        )

        after = (
            AIExplanationService
            ._result_evidence(
                db,
                project_id,
                version_2,
            )
        )

        before_metrics = (
            before.get("metrics", {})
            if before
            else {}
        )

        after_metrics = (
            after.get("metrics", {})
            if after
            else {}
        )

        return {
            "project_id": project_id,
            "version_1": version_1,
            "version_2": version_2,

            "git_changed": normalized.get(
                "git_changed"
            ),
            "dvc_changed": normalized.get(
                "dvc_changed"
            ),
            "code_changed": normalized.get(
                "code_changed"
            ),

            "changed_files": normalized.get(
                "changed_files",
                [],
            ),

            "code_changed_files": normalized.get(
                "code_changed_files",
                [],
            ),

            "dataset_diff": normalized.get(
                "dataset_diff"
            ),

            "dataset_analysis": normalized.get(
                "dataset_analysis"
            ),

            "preparation": normalized.get(
                "preparation"
            ),

            "result_evidence_before": before,
            "result_evidence_after": after,

            "metric_changes": (
                AIExplanationService
                ._metric_comparison(
                    before_metrics,
                    after_metrics,
                )
                if before is not None
                and after is not None
                else {}
            ),

            "evidence_chain": normalized.get(
                "evidence_chain",
                [],
            ),
        }

    @staticmethod
    def analyze_comparison(
        comparison: dict[str, Any],
        db: Session,
    ) -> dict[str, Any]:
        evidence = (
            AIExplanationService
            .build_evidence(
                db=db,
                project_id=int(
                    comparison.get(
                        "project_id"
                    )
                    or 0
                ),
                comparison=comparison,
            )
        )

        try:
            system_prompt = """
You are DATAGIT AI, an evidence-grounded
version intelligence analyst.

Interpret ONLY the supplied DATAGIT evidence.

DATAGIT does not track the user's training
process. Do not discuss training runs,
run IDs, run-before/run-after, or infer a
training process.

Version result evidence may contain:

- model identity
- model artifact reference
- model SHA-256
- framework
- framework version
- metrics
- evaluation evidence
- notes

IMPORTANT RULES:

1. Never invent facts.
2. Never invent metrics.
3. Never invent models.
4. Never invent dataset changes.
5. Never invent preparation operations.
6. Never invent evaluation results.
7. Missing evidence means "not recorded".
8. Missing evidence is NOT proof that no change occurred.
9. Never claim causality unless the supplied evidence supports it.
10. Distinguish observed facts from interpretation.
11. State limitations clearly.
12. Use exact supplied metric values.
13. Do not turn missing data into a zero.
14. Do not claim a metric improved or decreased unless
    both comparable values actually exist.
15. Model identity changes may be reported only when
    both model records are available.

Return JSON:

{
  "summary": "",
  "confidence": "high|medium|low",

  "limitations": [],

  "impact_assessment": [
    {
      "area":
        "git|dvc|code|dataset|preparation|model|metrics|evaluation",

      "status":
        "observed|unchanged|unknown",

      "explanation": "",

      "evidence": []
    }
  ],

  "root_cause": {
    "summary": "",
    "contributors": [],
    "overall_confidence": "high|medium|low",
    "limitations": [],
    "alternative_explanations": []
  },

  "recommendations": [
    {
      "recommendation": "",
      "reason": "",
      "priority": "high|medium|low"
    }
  ]
}
"""

            response = (
                AIExplanationService
                ._client()
                .chat
                .completions
                .create(
                    model=AIExplanationService.MODEL,
                    temperature=0.15,
                    response_format={
                        "type": "json_object"
                    },
                    messages=[
                        {
                            "role": "system",
                            "content":
                                system_prompt,
                        },
                        {
                            "role": "user",
                            "content": json.dumps(
                                evidence,
                                indent=2,
                                default=str,
                            ),
                        },
                    ],
                )
            )

            result = (
                AIExplanationService
                ._parse_json(
                    response
                    .choices[0]
                    .message
                    .content
                )
            )

            return {
                "status": "success",
                "reason": None,
                "summary": result.get(
                    "summary",
                    "",
                ),
                "confidence": result.get(
                    "confidence",
                    "low",
                ),
                "limitations": result.get(
                    "limitations",
                    [],
                ),
                "impact_assessment": result.get(
                    "impact_assessment",
                    [],
                ),
                "root_cause": result.get(
                    "root_cause",
                    {},
                ),
                "recommendations": result.get(
                    "recommendations",
                    [],
                ),
            }

        except Exception as exc:
            return {
                "status": (
                    "unavailable"
                    if "not configured"
                    in str(exc)
                    else "error"
                ),
                "reason": str(exc),
                "summary": (
                    "DATAGIT could not generate "
                    "an AI interpretation."
                ),
                "confidence": "low",
                "limitations": [
                    "AI interpretation is "
                    "currently unavailable."
                ],
                "impact_assessment": [],
                "root_cause": {
                    "summary": "",
                    "contributors": [],
                    "overall_confidence": "low",
                    "limitations": [],
                    "alternative_explanations": [],
                },
                "recommendations": [],
            }

    @staticmethod
    def _project_context(
        db: Session,
        project_id: int,
        comparison: dict[str, Any] | None,
    ) -> dict[str, Any]:
        project = (
            db.query(Project)
            .filter(
                Project.id
                == project_id
            )
            .first()
        )

        if project is None:
            raise ValueError(
                "Project not found."
            )

        versions = (
            db.query(Version)
            .filter(
                Version.project_id
                == project_id
            )
            .order_by(
                Version.version_number.asc()
            )
            .all()
        )

        datasets = (
            db.query(Dataset)
            .filter(
                Dataset.project_id
                == project_id
            )
            .all()
        )

        models = (
            db.query(Model)
            .filter(
                Model.project_id
                == project_id
            )
            .all()
        )

        version_records = []

        for version in versions:
            version_records.append(
                {
                    "id": version.id,
                    "version_number": (
                        version.version_number
                    ),
                    "description": (
                        version.description
                    ),
                    "git_commit": (
                        version.git_commit
                    ),
                    "dvc_state": (
                        version.dvc_state
                    ),
                    "created_at": (
                        version.created_at
                    ),
                    "result_evidence": (
                        AIExplanationService
                        ._result_evidence(
                            db,
                            project_id,
                            version.id,
                        )
                    ),
                }
            )

        current_comparison = None

        if comparison is not None:
            current_comparison = (
                AIExplanationService
                .build_evidence(
                    db=db,
                    project_id=project_id,
                    comparison=comparison,
                )
            )

        return {
            "project": {
                "id": project.id,
                "name": project.name,
                "description": project.description,
                "path": project.path,
            },

            "versions": version_records,

            "datasets": [
                {
                    "id": item.id,
                    "name": item.name,
                    "path": item.path,
                }
                for item in datasets
            ],

            "models": [
                {
                    "id": item.id,
                    "name": item.name,
                    "path": item.path,
                }
                for item in models
            ],

            "current_comparison":
                current_comparison,
        }

    @staticmethod
    def answer_question(
        db: Session,
        project_id: int,
        question: str,
        comparison: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        question = " ".join(
            (question or "").split()
        )

        if not question:
            raise ValueError(
                "Question cannot be empty."
            )

        if len(question) > 2000:
            raise ValueError(
                "Question is too long. "
                "Keep it under 2000 characters."
            )

        context = (
            AIExplanationService
            ._project_context(
                db=db,
                project_id=project_id,
                comparison=comparison,
            )
        )

        try:
            system_prompt = """
You are DATAGIT AI, a project-aware
version intelligence copilot.

Use only the supplied DATAGIT project
context for project-specific claims.

DATAGIT does NOT track the user's training
process.

Do not discuss:

- training runs
- run IDs
- run-before/run-after
- training run history

You may answer questions about:

- projects
- versions
- datasets
- dataset changes
- Git
- DVC
- preparation
- model evidence
- model identity
- metrics
- accuracy
- precision
- recall
- F1
- loss
- evaluation evidence
- confusion matrices
- provenance
- evidence gaps
- version comparisons
- reproducibility

Missing evidence must be described as
"not recorded" or "not available".

Never invent:

- metrics
- model identities
- datasets
- evaluation results
- preparation operations
- causal relationships

When comparing metrics, use only values
actually supplied by the evidence.

General technical guidance is allowed,
but clearly distinguish it from
project-specific facts.

Return JSON:

{
  "answer": "",
  "evidence": [],
  "limitations": [],
  "confidence": "high|medium|low"
}
"""

            user_prompt = (
                "PROJECT CONTEXT:\n"
                + json.dumps(
                    context,
                    indent=2,
                    default=str,
                )
                + "\n\nUSER QUESTION:\n"
                + question
            )

            response = (
                AIExplanationService
                ._client()
                .chat
                .completions
                .create(
                    model=AIExplanationService.MODEL,
                    temperature=0.2,
                    response_format={
                        "type": "json_object"
                    },
                    messages=[
                        {
                            "role": "system",
                            "content":
                                system_prompt,
                        },
                        {
                            "role": "user",
                            "content":
                                user_prompt,
                        },
                    ],
                )
            )

            result = (
                AIExplanationService
                ._parse_json(
                    response
                    .choices[0]
                    .message
                    .content
                )
            )

            return {
                "status": "success",
                "question": question,
                "answer": str(
                    result.get(
                        "answer",
                        "DATAGIT AI could not "
                        "produce an answer.",
                    )
                ).strip(),
                "evidence": (
                    result.get(
                        "evidence",
                        [],
                    )
                    if isinstance(
                        result.get(
                            "evidence",
                            [],
                        ),
                        list,
                    )
                    else []
                ),
                "limitations": (
                    result.get(
                        "limitations",
                        [],
                    )
                    if isinstance(
                        result.get(
                            "limitations",
                            [],
                        ),
                        list,
                    )
                    else []
                ),
                "confidence": result.get(
                    "confidence",
                    "low",
                ),
            }

        except Exception as exc:
            return {
                "status": (
                    "unavailable"
                    if "not configured"
                    in str(exc)
                    else "error"
                ),
                "question": question,
                "answer": (
                    "DATAGIT AI is "
                    "currently unavailable."
                ),
                "evidence": [],
                "limitations": [
                    str(exc)
                ],
                "confidence": "low",
            }

    @staticmethod
    def compare_and_answer(
        db: Session,
        project_id: int,
        version_1: int,
        version_2: int,
        question: str,
    ) -> dict[str, Any]:
        comparison = (
            VersionComparisonService
            .compare_versions(
                db=db,
                project_id=project_id,
                version_1=version_1,
                version_2=version_2,
            )
        )

        return (
            AIExplanationService
            .answer_question(
                db=db,
                project_id=project_id,
                question=question,
                comparison=comparison,
            )
        )