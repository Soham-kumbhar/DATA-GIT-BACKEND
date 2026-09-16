import logging

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
)
from sqlalchemy.orm import Session

from app.db.database import get_db

from app.schemas.version_comparison import (
    VersionComparisonResponse,
)

from app.services.groq_report_service import (
    GroqReportService,
)

from app.services.version_comparison_service import (
    VersionComparisonService,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/projects/{project_id}/versions",
    tags=["Version Comparison"],
)


@router.get(
    "/compare",
    response_model=VersionComparisonResponse,
)
def compare_versions(
    project_id: int,
    version_1: int,
    version_2: int,
    generate_ai: bool = Query(
        True,
        description=(
            "Generate evidence-grounded AI interpretation "
            "after deterministic comparison."
        ),
    ),
    db: Session = Depends(get_db),
):
    if version_1 == version_2:
        raise HTTPException(
            status_code=422,
            detail="Version 1 and Version 2 must be different.",
        )

    try:
        result = VersionComparisonService.compare_versions(
            db=db,
            project_id=project_id,
            version_1=version_1,
            version_2=version_2,
        )

        if generate_ai:
            result["ai_insights"] = (
                GroqReportService.analyze_comparison(
                    comparison=result,
                )
            )
        else:
            result["ai_insights"] = {
                "status": "disabled",
                "reason": "AI comparison generation was disabled.",
                "root_cause": {
                    "summary": "",
                    "contributors": [],
                    "overall_confidence": "low",
                    "limitations": [],
                    "alternative_explanations": [],
                },
                "recommendations": [],
            }

        return result

    except ValueError as error:
        logger.exception(
            "Version comparison validation error: "
            "project_id=%s version_1=%s version_2=%s",
            project_id,
            version_1,
            version_2,
        )
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    except Exception as error:
        logger.exception(
            "Version comparison crashed: "
            "project_id=%s version_1=%s version_2=%s",
            project_id,
            version_1,
            version_2,
        )
        raise HTTPException(
            status_code=500,
            detail=(
                f"Version comparison failed: "
                f"{type(error).__name__}: {error}"
            ),
        ) from error


@router.get("/multi-report")
def multi_version_report(
    project_id: int,
    versions: str = Query(...),
    mode: str = Query("evolution"),
    db: Session = Depends(get_db),
):
    try:
        version_ids = [
            int(value.strip())
            for value in versions.split(",")
            if value.strip()
        ]
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid versions value. "
                "Use comma-separated integer IDs."
            ),
        ) from error

    if len(version_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least two versions are required.",
        )

    if mode not in {"evolution", "baseline"}:
        raise HTTPException(
            status_code=400,
            detail="Invalid mode. Use 'evolution' or 'baseline'.",
        )

    try:
        return VersionComparisonService.compare_multiple_versions(
            db=db,
            project_id=project_id,
            version_ids=version_ids,
            mode=mode,
        )

    except ValueError as error:
        logger.exception(
            "Multi-version comparison validation error: "
            "project_id=%s",
            project_id,
        )
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    except Exception as error:
        logger.exception(
            "Multi-version comparison crashed: project_id=%s",
            project_id,
        )
        raise HTTPException(
            status_code=500,
            detail=(
                f"Multi-version comparison failed: "
                f"{type(error).__name__}: {error}"
            ),
        ) from error
